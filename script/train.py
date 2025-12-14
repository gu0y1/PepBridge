import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import autocast, GradScaler  
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

import os
import math
from typing import Dict, Optional, Iterable, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

from .loss import contact_losses, bce_loss, loss_pair_pseudolikelihood
from .model.lora import build_model_with_lora

task_every = {
    "mp_contact": 50,  
    "pt_contact": 50,  
}

def rampup_weight(step, warmup=2000, ramp=3000):
    if step < warmup: return 0.0
    t = min(1., (step - warmup)/ramp)
    return 0.5 - 0.5 * math.cos(math.pi * t)

def linear_decay(step: int, total_steps: int, start: float = 1.0, end: float = 0.0):
    if total_steps <= 0:
        return end
    progress = min(max(step, 0), total_steps) / total_steps
    return start + (end - start) * progress

def _to_set(x):
    if isinstance(x, str):
        return {x}
    return set(x)

def freeze_module(model, module_names, logger=None):
    targets = _to_set(module_names)
    for name, module in model.named_modules():
        if name in targets:
            msg = (f"Freezing module: {name}")
            logger.info(msg) if logger is not None else print(msg)
            for param in module.parameters():
                param.requires_grad = False

def set_eval_mode(model, module_names):
    targets = _to_set(module_names)
    for name, module in model.named_modules():
        if name in targets: 
            module.eval()

def _infinite_iter(dl):
    it = iter(dl)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(dl)

def train_three_phases_multi_loaders(
    model,
    loaders: Dict[str, Optional[Iterable]],  
    *,
    device="cuda",
    save_dir="./checkpoints_multi",
    epochs_A=5, epochs_B=10, epochs_C=10,
    steps_per_epoch=1000, 
    optimizer_ctor=lambda params: torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01),
    grad_accum_steps=1,
    amp=True,
    new_optimizer_each_phase=True,
    log_interval=50,
    task_every = None,   #
    val_loaders= None,
    eval_every_epochs=1,
    pep_align=True,
    all_align=True,
    use_lora=True,
    last_n=2,
    cfg_seq_pair=((8,16),(4,8)),
    logger=None,
    use_logits=True
):
    """
    loaders key: value
      align         -> (mhc, peptide, cdr3, esm_mhc)
      mp            -> (mhc, peptide, esm_mhc, y_mp)
      pt            -> (peptide, cdr3, y_pt)
      mp_contact    -> (mhc, peptide, esm_mhc, contact_bin, contact_dist)
      pt_contact    -> (peptide, cdr3, contact_bin, contact_dist)
      imm           -> (mhc, peptide, esm_mhc, y_imm)
      mpt           -> (mhc, peptide, cdr3, esm_mhc, trbv, y_mpt)
    """
    if task_every is None:
        task_every = {}

    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    scaler = GradScaler(enabled=amp)

    # lora
    if use_lora:
        model = build_model_with_lora(model, last_n, cfg_seq_pair, 
                                      dropout=0.1, freeze_base=True,
                                      print_trainabel=False)
    else:
        freeze_module(model, ['peptide_encoder','cdr3_encoder'],logger)

    # λ
    lambdas_A = dict(align=1.00, MP=1.50, PT=1.00, IMM=0.0, mp_contact=0.0, pt_contact=0.0, MPT=0.0)
    lambdas_B = dict(align=1.00, MP=1.20, PT=0.80, IMM=0.0, mp_contact=0.40, pt_contact=0.04, MPT=0.0)
    lambdas_C = dict(align=0.00, MP=0.00, PT=0.00, IMM=1.00, mp_contact=0.0, pt_contact=0.0, MPT=1.00)

    phases = {
        "A": dict(lmb=lambdas_A, tasks=("align", "mp", "pt")),
        "B": dict(lmb=lambdas_B, tasks=("align", "mp", "pt", "mp_contact", "pt_contact")),
        "C": dict(lmb=lambdas_C, tasks=( "imm", "mpt")),
    }
    recorder = MetricsRecorder()

    def make_optimizer():
        params = (p for p in model.parameters() if p.requires_grad)
        return optimizer_ctor(params)

    # phase #
    optimizer = None
    for ph, cfg in phases.items():
        if new_optimizer_each_phase or optimizer is None:
            optimizer = make_optimizer()
        if ph == 'C':
            if pep_align:
                ckpt = torch.load(os.path.join(save_dir, "phase_B.pt"), map_location=device)
                model.load_state_dict(ckpt['model_state'], strict=True)
            freeze_module(model,['peptide_encoder', 'mhc_encoder','chain_id_embedder', 
                                 'mhc_ln','pep_ln','cdr3_ln', 
                                 'mp_joint_embedder', 'pt_joint_embedder',
                                 'mp_pair_aware_trunk','pt_pair_aware_trunk',
                                 'mp_pred_head','pt_pred_head','mp_contact_pred_head','pt_contact_pred_head',
                                 'pep_seq_proj','pep_pair_proj'], 
                                 logger=None)

            optimizer = make_optimizer()

        active = {k: loaders.get(k) for k in cfg["tasks"] if loaders.get(k) is not None}
        assert len(active) > 0, f"Phase {ph} without dataloader"

        # iter
        iters = {k: _infinite_iter(dl) for k, dl in active.items()}
        if steps_per_epoch is None:
            steps_per_epoch = max(len(dl) for dl in active.values())

        _train_one_phase_multi(
            model=model,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            iters=iters,
            steps_per_epoch=steps_per_epoch,
            epochs= (epochs_A if ph=="A" else epochs_B if ph=="B" else epochs_C),
            lambdas=cfg["lmb"],
            phase_name=ph,
            save_path=os.path.join(save_dir, f"phase_{ph}.pt"),
            grad_accum_steps=grad_accum_steps,
            amp=amp,
            log_interval=log_interval,
            task_every=task_every,            #
            val_loaders=val_loaders,
            eval_every_epochs=eval_every_epochs,
            pep_align=pep_align,
            all_align=all_align,
            use_lora=use_lora,
            recorder=recorder,
            logger=logger,
            use_logits=use_logits
        )

    plot_losses(recorder, out_dir=save_dir, phase=None)

def _train_one_phase_multi(
    *,
    model,
    device,
    optimizer,
    scaler,
    iters: Dict[str, Iterable],
    steps_per_epoch: int,
    epochs: int,
    lambdas: Dict[str, float],
    phase_name: str,
    save_path: str,
    grad_accum_steps: int,
    amp: bool,
    log_interval: int,
    task_every: Dict[str, int],            #
    val_loaders=None,
    eval_every_epochs: int = 1,
    pep_align=True,
    all_align=True,
    use_lora=True,
    recorder=None, 
    logger=None,
    use_logits=True
):
    def every_of(task: str) -> int:
        v = task_every.get(task, 1)
        return max(int(v), 1)
    
    if every_of("mp") != 1 or every_of("pt") != 1:
        print("[Warning] task_every['mp'] and task_every['pt'] setting as 1")

    best_val = float("inf") 
    for epoch in range(1, epochs + 1):
        model.train()
        if use_lora is False:
            model.peptide_encoder.eval()
        model.cdr3_encoder.eval()
        if phase_name =='C':
            set_eval_mode(model,['peptide_encoder','mhc_encoder', 'chain_id_embedder', 
                                 'mhc_ln','pep_ln','cdr3_ln', 
                                 'mp_joint_embedder', 'pt_joint_embedder',
                                 'mp_pair_aware_trunk','pt_pair_aware_trunk',
                                 'mp_pred_head','pt_pred_head','mp_contact_pred_head','pt_contact_pred_head',
                                 'pep_seq_proj','pep_pair_proj'])

        running_sum = defaultdict(float)
        running_cnt = defaultdict(int) 

        for step in range(1, steps_per_epoch + 1):
            due_tasks = [name for name in iters.keys() if (step % every_of(name) == 0)]
            assert len(due_tasks) > 0

            batches = {name: next(iters[name]) for name in due_tasks}

            with autocast(enabled=amp):
                if phase_name == 'A':
                    w_align = rampup_weight(step=(epoch-1)*steps_per_epoch + step, 
                                            warmup=0,ramp=5000)
                    lambdas["align"] = w_align * 1.00
                    
                    w_pt = rampup_weight(step=(epoch-1)*steps_per_epoch + step, 
                                            warmup=5000,ramp=5000)
                    lambdas["PT"] = w_pt * 1.00
                    
                elif phase_name == 'B':
                    lambdas["align"] = linear_decay(step=(epoch-1)*steps_per_epoch + step,
                                                    total_steps=epochs*steps_per_epoch, 
                                                    start=1.00, end=0.00)
                                                    
                    w_contact = rampup_weight(step=(epoch-1)*steps_per_epoch + step, 
                                            warmup=0,ramp=6000)
                    lambdas["pt_contact"] = w_contact * 0.04
                    lambdas["mp_contact"] = w_contact * 0.40

                # multi-dataloader loss
                loss_parts = compute_losses_multi_batches(
                    model=model,
                    device=device,
                    phase=phase_name,
                    batches=batches,
                    pep_align=pep_align,
                    all_align=all_align,
                    use_logits=use_logits,
                    l_pt=lambdas["PT"]
                )

                total = (
                    lambdas["align"]   * loss_parts["align"] +
                    lambdas["MP"]      * loss_parts["MP"] +
                    lambdas["PT"]      * loss_parts["PT"] +
                    lambdas["mp_contact"] * loss_parts["mp_contact"] +
                    lambdas["pt_contact"] * loss_parts["pt_contact"] +
                    lambdas["IMM"]     * loss_parts["IMM"] +
                    lambdas["MPT"]     * loss_parts["MPT"] 
                )
                loss = total / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # log
            for k, v in loss_parts.items():
                if v.requires_grad or (v.detach().item() != 0.0):
                    running_sum[k] += float(v.detach().item())
                    running_cnt[k] += 1

            running_sum["total"] += float(total.detach().item())
            running_cnt["total"] += 1

            if (step % log_interval) == 0:
                def avg(key):
                    return (running_sum[key] / max(running_cnt[key], 1))
                msg = (f"[Phase {phase_name}] epoch {epoch}/{epochs} step {step}/{steps_per_epoch} "
                       f"total={avg('total'):.4f} | "
                       f"align={avg('align'):.4f} MP={avg('MP'):.4f} "
                       f"PT={avg('PT'):.4f} mp_contact={avg('mp_contact'):.4f} pt_contact={avg('pt_contact'):.4f} "
                       f"IMM={avg('IMM'):.4f} MPT={avg('MPT'):.4f}")
                logger.info(msg) if logger is not None else print(msg)

                if recorder is not None:
                    rec_point = {
                        "total": avg("total"),
                        "align": avg("align"),
                        "MP": avg("MP"),
                        "PT": avg("PT"),
                        "mp_contact": avg("mp_contact"),
                        "pt_contact": avg("pt_contact"),
                        "IMM": avg("IMM"),
                        "MPT": avg("MPT"),
                    }
                    recorder.log_train(
                        phase=phase_name, epoch=epoch, step=step,
                        steps_per_epoch=steps_per_epoch, loss_parts=rec_point
                    )
                running_sum.clear();running_cnt.clear()

        # val        
        if (val_loaders is not None) and (epoch % eval_every_epochs == 0):
            val_parts, val_total = evaluate_phase_multi(
                model=model,
                val_loaders=val_loaders,
                device=device,
                phase_name=phase_name,
                lambdas=lambdas,
                max_steps=200,
                pep_align=pep_align,
                all_align=all_align,
                use_logits=use_logits
            )
            parts = " ".join(f"{k}={v:.4f}" for k, v in val_parts.items())
            msg = f"[Phase {phase_name}] epoch {epoch} VAL: {parts} | total={val_total:.4f}"
            logger.info(msg) if logger is not None else print(msg)
            if recorder is not None:
                recorder.log_val(phase=phase_name, epoch=epoch, val_parts=val_parts, val_total=val_total)

            if val_total < best_val - 1e-12:
                best_val = val_total
                torch.save({"model_state": model.state_dict(),
                            "phase": phase_name,
                            "lambdas": lambdas,
                            "val_total": best_val,
                            "epoch": epoch}, save_path)
                msg = f"[Phase {phase_name}] ☆ improved, saved best full ckpt to {save_path}"
                logger.info(msg) if logger is not None else print(msg)
    # save
    if not os.path.isfile(save_path):
        torch.save({"model_state": model.state_dict(),
                    "phase": phase_name,
                    "lambdas": lambdas}, save_path)
        msg = f"[Phase {phase_name}] saved to {save_path}"
        logger.info(msg) if logger is not None else print(msg)

@torch.no_grad()
def evaluate_phase_multi(
    model,
    val_loaders,
    device: str,
    phase_name: str,
    lambdas: Dict[str, float],
    max_steps=None, 
    pep_align=True,
    all_align=True,
    use_logits=True
) -> Tuple[Dict[str, float], float]:
    if not val_loaders:
        zeros = {k: 0.0 for k in ["align","MP","PT","mp_contact",'pt_contact',"IMM","MPT"]}
        return zeros, 0.0
    model.eval()

    phases_task = {
        "A": ("align", "mp", "pt"), 
        "B": ("align", "mp", "pt", "mp_contact", "pt_contact"),
        "C": ("imm", "mpt")
    }
    active = {}
    for k in phases_task[phase_name]:
        dl = val_loaders.get(k, None)
        if dl is None:
            continue
        try:
            if hasattr(dl, "__len__") and len(dl) == 0:
                continue
        except Exception:
            pass
        active[k] = dl

    if len(active) == 0:
        zeros = {k: 0.0 for k in ["align","MP","PT","mp_contact",'pt_contact',"IMM","MPT"]}
        return zeros, 0.0
    
    if max_steps is None:
        try:
            max_steps = min(len(dl) for dl in active.values())
            if max_steps is None or max_steps <= 0:
                raise ValueError
        except Exception:
            max_steps = 200 
            
    iters = {k: _infinite_iter(dl) for k, dl in active.items()}

    sums = defaultdict(float)
    cnts = defaultdict(int)
    for _ in range(max_steps):
        batches = {name: next(iters[name]) for name in phases_task[phase_name]}

        parts = compute_losses_multi_batches(
            model=model,
            device=device,
            phase=phase_name,
            batches=batches,
            pep_align=pep_align,
            all_align=all_align,
            use_logits=use_logits,
            l_pt=lambdas["PT"] 
        )
        for k, v in parts.items():
            val = float(v.detach().item())
            sums[k] += val
            cnts[k] += 1
    # mean
    means = {}
    for k in ["align","MP","PT","mp_contact",'pt_contact',"IMM","MPT"]:
        means[k] = (sums[k] / max(cnts[k], 1)) if k in sums else 0.0

    # λ
    val_total = (
        lambdas["align"]   * means["align"]   +
        lambdas["MP"]      * means["MP"]      +
        lambdas["PT"]      * means["PT"]      +
        lambdas["mp_contact"] * means["mp_contact"] +
        lambdas["pt_contact"] * means["pt_contact"] +
        lambdas["IMM"]     * means["IMM"]     +
        lambdas["MPT"]     * means["MPT"]    

    )

    return means, float(val_total)

class MetricsRecorder:
    TRAIN_KEYS = ["total","align","MP","PT","mp_contact",'pt_contact',"IMM","MPT"]

    def __init__(self):
        self.train = []  # list of dict({'phase','epoch','step','epoch_t', each loss...})
        self.val   = []  # list of dict({'phase','epoch', each loss...})

    def log_train(self, *, phase, epoch, step, steps_per_epoch, loss_parts: dict):
    
        epoch_t = (epoch - 1) + (step / max(steps_per_epoch, 1))
        row = {"phase": phase, "epoch": epoch, "step": step, "epoch_t": epoch_t}
        for k in self.TRAIN_KEYS:
            if k in loss_parts:
                row[k] = float(loss_parts[k])
        self.train.append(row)

    def log_val(self, *, phase, epoch, val_parts: dict, val_total: float):
        row = {"phase": phase, "epoch": epoch, "total": float(val_total)}
        for k, v in val_parts.items():
            row[k] = float(v)
        self.val.append(row)

def plot_losses(
    recorder: MetricsRecorder,
    out_dir: str,
    phase: str,
    keys = ("total","align","MP","PT","mp_contact","pt_contact","IMM","MPT"),
    dpi: int = 200
):

    os.makedirs(out_dir, exist_ok=True)

    if not recorder.train and not recorder.val:
        print("[plot] no data to plot.")
        return

    phases = sorted(set(r["phase"] for r in (recorder.train + recorder.val))) if phase is None else [phase]

    for key in keys:
        plt.figure(figsize=(7, 4.5))
        has_any = False
        for ph in phases:
            # train
            tr_x = [r["epoch_t"] for r in recorder.train if r["phase"] == ph and (key in r)]
            tr_y = [r[key]        for r in recorder.train if r["phase"] == ph and (key in r)]
            if tr_x:
                plt.plot(tr_x, tr_y, label=f"train-{ph}", linewidth=1.8)
                has_any = True
            # val
            va_x = [r["epoch"] for r in recorder.val if r["phase"] == ph and (key in r)]
            va_y = [r[key]     for r in recorder.val if r["phase"] == ph and (key in r)]
            if va_x:
                plt.plot(va_x, va_y, linestyle="--", marker="o", markersize=3.5, label=f"val-{ph}")
                has_any = True

        if not has_any:
            plt.close(); continue

        plt.title(f"{key} loss (train vs val)")
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fname = f"loss_{key}{'' if phase is None else '_'+phase}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=dpi)
        plt.close()

def compute_losses_multi_batches(
    *,
    model,
    device,
    phase: str,
    batches: Dict[str, dict], 
    pep_align=True,
    all_align=True,
    use_logits=True,
    l_pt=0,
) -> Dict[str, torch.Tensor]:
    zeros = torch.tensor(0.0, device=device)
    out = dict(align=zeros, MP=zeros, PT=zeros, 
               mp_contact=zeros, pt_contact=zeros,
               IMM=zeros, MPT=zeros)

    # -------- Align --------
    if (pep_align is True) and (phase in ('A', 'B')):
        if "align" in batches:
            b = batches["align"][0]
            mhc      = b["mhc"].to(device)
            peptide  = b["peptide"].to(device)
            cdr3     = b["cdr3"].to(device)
            esm_mhc = b.get("esm_mhc", None)
            if esm_mhc is not None: esm_mhc = esm_mhc.to(device)

            out["align"] = model.pep_align(mhc, peptide, cdr3, esm_mhc,all_align=all_align)
    else:
        out["align"] = torch.tensor(0.0, device=device)

    # -------- MP binding --------
    if ("mp" in batches) and (phase in ('A', 'B')):
        b = batches["mp"][0]
        mhc      = b["mhc"].to(device)
        peptide  = b["peptide"].to(device)
        esm_mhc = b.get("esm_mhc", None)
        if esm_mhc is not None: esm_mhc = esm_mhc.to(device)
        y_mp     = b["y_mp"].to(device).view(-1, 1)

        mp_out = model.mp_pred(mhc, peptide, esm_mhc, contact=False, immunogenicity=(phase=="C"))
        out["MP"] = bce_loss(mp_out["binding_prob"], y_mp, use_logits=use_logits)

        # # L_logic_IMM
        # if phase=="C":
        #     out["logic_imm"] = out["logic_imm"] + \
        #         F.relu(mp_out["immunogenicity_prob"] - mp_out["binding_prob"]).mean() * 0.5

    # -------- PT binding --------
    if ("pt" in batches) and (phase in ('A', 'B')):
        if l_pt !=0:
            b = batches["pt"][0]
            peptide = b["peptide"].to(device)
            cdr3_pos = b["cdr3_pos"].to(device)
            cdr3_neg = b["cdr3_negs"].to(device)
            B, K, Lc = cdr3_neg.shape[0], cdr3_neg.shape[1], cdr3_neg.shape[2]
            Lp = peptide.size(1)
            # y_pt    = b["y_pt"].to(device).view(-1, 1)

            pos_out = model.pt_pred(peptide, cdr3_pos, contact=False)

            pep_rep = peptide.unsqueeze(1).expand(-1, K, -1).reshape(B*K, Lp)
            neg_out = model.pt_pred(pep_rep, cdr3_neg.reshape(B*K, Lc), contact=False)

            pos_logit, neg_logit = pos_out["binding_prob"], neg_out["binding_prob"].view(B, K)
            pos_loss = bce_loss(pos_logit, torch.ones_like(pos_logit), use_logits=use_logits) 
            
            neg_top_vals, _ = torch.topk(neg_logit, k=min(2, K), dim=1)
            neg_loss = bce_loss(neg_top_vals, torch.zeros_like(neg_top_vals), use_logits=use_logits)
            loss_bce = 0.5 * (pos_loss + neg_loss)

            logits_concat = torch.cat([pos_logit, neg_logit], dim=1)
            loss_pair = F.cross_entropy(logits_concat / 0.25, 
                                        torch.zeros(B, dtype=torch.long, device=device))
            # loss_pair = -F.logsigmoid(pos_logit - neg_logit).mean()

            out["PT"] = loss_bce + 0.1 * loss_pair
        else:
            out["PT"] = torch.tensor(0.0, device=device)

    # -------- MP contact --------
    if 'mp_contact' in batches:
        if phase == "B":
            b = batches["mp_contact"][0]
            mhc      = b["mhc"].to(device)
            peptide  = b["peptide"].to(device)
            esm_mhc = b.get("esm_mhc", None)
            if esm_mhc is not None: esm_mhc = esm_mhc.to(device)

            mp_out = model.mp_pred(mhc, peptide, esm_mhc, contact=True, immunogenicity=False)

            mask_m = mp_out["mask_dict"]["mhc"]
            mask_p = mp_out["mask_dict"]["pep"]
            pair_mask = mask_m.unsqueeze(2) & mask_p.unsqueeze(1)
            p_bin = b.get("contact_mp_bin", None)
            p_dis = b.get("contact_mp_dist", None)
            if p_bin is not None: p_bin = p_bin.to(device)
            if p_dis is not None: p_dis = p_dis.to(device)
            lp, ld = contact_losses(
                prob_pred=mp_out.get("contact_prob", None),
                dist_pred=mp_out.get("contact_dist", None),
                prob_tgt=p_bin, dist_tgt=p_dis, pair_mask=pair_mask,
                use_logits=use_logits, distogram=True
            )
            out["mp_contact"] = out["mp_contact"] + lp + ld
            
    # -------- PT contact --------
    if 'pt_contact' in batches:
        if phase == "B":
            b = batches["pt_contact"][0]
            peptide = b["peptide"].to(device)
            cdr3    = b["cdr3"].to(device)

            pt_out = model.pt_pred(peptide, cdr3, contact=True)

            mask_p = pt_out["mask_dict"]["pep"]
            mask_t = pt_out["mask_dict"]["cdr3"]
            pair_mask = mask_p.unsqueeze(2) & mask_t.unsqueeze(1)
            p_bin = b.get("contact_pt_bin", None)
            p_dis = b.get("contact_pt_dist", None)
            if p_bin is not None: p_bin = p_bin.to(device)
            if p_dis is not None: p_dis = p_dis.to(device)
            lp, ld = contact_losses(
                prob_pred=pt_out.get("contact_prob", None),
                dist_pred=pt_out.get("contact_dist", None),
                prob_tgt=p_bin, dist_tgt=p_dis, pair_mask=pair_mask,
                use_logits=use_logits, distogram=False
            )
            out["pt_contact"] = out["pt_contact"] + lp + ld

    # -------- IMM --------
    if "imm" in batches:
        if phase == "C":
            b = batches["imm"][0]
            mhc      = b["mhc"].to(device)
            peptide  = b["peptide"].to(device)
            esm_mhc = b.get("esm_mhc", None)
            if esm_mhc is not None: esm_mhc = esm_mhc.to(device)
            y_imm    = b["y_imm"].to(device).view(-1, 1)

            mp_out = model.mp_pred(mhc, peptide, esm_mhc, contact=False, immunogenicity=True)
            out["IMM"] = bce_loss(mp_out["immunogenicity_prob"], y_imm, use_logits=use_logits)

            # L_logic_IMM
            # neg = (y_imm.expand_as(mp_out["immunogenicity_prob"]) == 0).float()
            # out["logic_imm"] = out["logic_imm"] + \
            #     (F.relu(mp_out["immunogenicity_prob"] - mp_out["binding_prob"] + 0.3) * neg).sum() / neg.sum().clamp_min(1e-8)

    # -------- MPT --------
    if "mpt" in batches:
        if phase == "C":
            b = batches["mpt"][0]
            mhc      = b["mhc"].to(device)
            peptide  = b["peptide"].to(device)
            cdr3     = b["cdr3"].to(device)
            esm_mhc = b.get("esm_mhc", None)
            if esm_mhc is not None: esm_mhc = esm_mhc.to(device)
            trbv     = b.get("trbv", None)
            if trbv is not None: trbv = trbv.to(device)
            y_mpt    = b["y_mpt"].to(device).view(-1, 1)
    
            mpt_out = model.mpt_pred(mhc, peptide, cdr3, esm_mhc, trbv)
            out["MPT"] = bce_loss(mpt_out["mpt_prob"], y_mpt, use_logits=use_logits)
    
            # L_logic_MPT = ReLU(P_MPT - min(P_MP, P_PT))
            # min_mp_pt = torch.minimum(mpt_out["mp_prob"], mpt_out["pt_prob"])
            # neg = (y_mpt.expand_as(mpt_out["mpt_prob"]) == 0).float()
            # out["logic_mpt"] = out["logic_mpt"] + \
            #     (F.relu(mpt_out["mpt_prob"] - min_mp_pt + 0.3) * neg).sum() / neg.sum().clamp_min(1e-8)
    return out

def train_fintune(
    model,
    device,
    optimizer_ctor,
    scaler,
    loader,
    steps_per_epoch: int,
    epochs: int,
    save_path: str,
    grad_accum_steps: int,
    amp: bool,
    log_interval: int,
    logger=None):

    os.makedirs(save_path, exist_ok=True)

    model.to(device)
    opt = optimizer_ctor(p for p in model.parameters() if p.requires_grad)
    scaler = GradScaler(enabled=amp)
    global_step = 0

    for n, p in model.named_parameters():
        if n in _to_set(['trav_emb', 'traj_emb', 'mpt_pred_head']):
            p.requires_grad = True
        else:
            p.requires_grad = False

    model.train()
    for epoch in range(1, epochs + 1):
        num_steps = 0
        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            global_step += 1
            num_steps += 1
            batch    =  batch[0]
            mhc      = batch["mhc"].to(device)
            esm_mhc  = batch.get("esm_mhc", None)
            if esm_mhc is not None:
                esm_mhc = esm_mhc.to(device)
            peptide  = batch["peptide"].to(device)
            cdr3     = batch["cdr3"].to(device)
            trbv     = batch["trbv"].to(device)
            score    = batch["score"].to(device)
            trav     = batch.get("trav", None)
            if trav is not None:
                trav = trav.to(device)
            traj     = batch.get("traj", None)
            if traj is not None:
                traj = traj.to(device)

            with autocast(enabled=amp):
                logits = model.mpt_pred_finetune(
                    mhc, peptide, cdr3, esm_mhc, trbv, 
                    trav, traj
                )

                idx = torch.randperm(logits.size(0), device=logits.device)
                logits1, logits2 = logits, logits[idx]
                s1, s2 = score, score[idx]
                y_pair = (s1 > s2).float() * 2.0 - 1.0  # +1 / -1
                Lmargin = F.softplus(-(logits2 - logits1) * y_pair).mean()

            scaler.scale(Lmargin / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True) 

            if (global_step % log_interval) == 0:
                msg = (
                    f"[Epoch {epoch} Step {step+1}/{steps_per_epoch}] "
                    f"Lmargin={Lmargin.item():.4f}"
                )
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg)
                    
    ckpt_path = os.path.join(save_path, "pepbridge_finetune.pt")
    torch.save(model.state_dict(), ckpt_path)

def _mrf_energy(
    seq_logits: torch.Tensor,      # [B, L, A]
    pair_logits: torch.Tensor,     # [B, L, L, A, A] 或 [B, L, L, A*A]
    labels: torch.Tensor,          # [B, L] (long)
    mask=None,  # [B, L] (float/bool)
):
    device = seq_logits.device
    dtype = seq_logits.dtype
    B, L, A = seq_logits.shape

    if pair_logits.dim() == 5 and pair_logits.shape[-2:] == (A, A):
        pair_aa = pair_logits
    elif pair_logits.dim() == 4 and pair_logits.shape[-1] == A * A:
        pair_aa = pair_logits.view(B, L, L, A, A)
    else:
        raise ValueError("pair_logits must be [B,L,L,A,A] or [B,L,L,A*A].")

    if mask is None:
        mask = torch.ones(B, L, device=device, dtype=dtype)
    else:
        mask = mask.to(dtype)

    labels = labels.clamp_min(0).clamp_max(A - 1)
    y_onehot = F.one_hot(labels, num_classes=A).to(dtype)    # [B, L, A]

    # sum_i h_i(y_i)
    h_y = torch.gather(seq_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, L]
    h_y = h_y * mask
    term_h = h_y.sum(dim=1)  # [B]

    # 0.5 * sum_{i,j} e_ij(y_i, y_j)
    pair_mask = (mask[:, :, None] * mask[:, None, :])  # [B, L, L]
    pair_mask = pair_mask.unsqueeze(-1).unsqueeze(-1)  # [B, L, L, 1, 1]

    # [B, 1, L, 1, A] * [B, L, 1, A, 1] -> [B, L, L, A, A]
    y_i = y_onehot[:, None, :, None, :]    # [B, 1, L, 1, A]
    y_j = y_onehot[:, :, None, :, None]    # [B, L, 1, A, 1]
    y_outer = y_i * y_j                    # [B, L, L, A, A]

    pair_contrib = (y_outer * pair_aa * pair_mask).sum(dim=(1, 2, 3, 4))  # [B]
    term_pair = 0.5 * pair_contrib

    E = -(term_h + term_pair)  # [B]
    return E

def generator_train(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    loader,  
    device: str = "cuda",
    save_dir: str = "./checkpoints_pepbridge_gen",
    epochs: int = 100,
    steps_per_epoch: int = 200, 
    optimizer_ctor=lambda params: torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01),
    grad_accum_steps: int = 1,
    amp: bool = True,
    log_interval: int = 50,
    val_loader = None,
    eval_every_epochs: int = 1,
    val_max_steps: int = 200,
    logger = None,
    distillation: bool = False,
    lambda_bv: float = 0.2,
    lambda_adv: float = 0.1,
    lambda_margin: float = 1.0,
    lambda_base: float = 0.5,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    PepBridgeGenerator + Discriminator

    Loss(base):
        L = Lseq + Lpair + λ1 * LBV + λ2 * Ladv

    Loss(distillation):
        L = Lseq + Lpair + Lmargin + λ1 * LBV + λ2 * Ladv
    """
    os.makedirs(save_dir, exist_ok=True)

    generator.to(device)
    discriminator.to(device)
    use_D = not distillation
    # 两个优化器
    opt_G = optimizer_ctor(filter(lambda p: p.requires_grad, generator.parameters()))
    opt_D = optimizer_ctor(filter(lambda p: p.requires_grad, discriminator.parameters())) if use_D else None

    scaler = GradScaler(enabled=amp)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train() if use_D else discriminator.eval()

        running = {
            "G_total": 0.0,
            "Lseq": 0.0,
            "Lpair": 0.0,
            "LBV": 0.0,
            "Ladv": 0.0,
            "Lmargin": 0.0,
            "D_total": 0.0,
        }
        num_steps = 0

        opt_G.zero_grad(set_to_none=True)
        if use_D:
            opt_D.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            global_step += 1
            num_steps += 1
            batch    =  batch[0]
            mhc      = batch["mhc"].to(device)
            esm_mhc  = batch.get("esm_mhc", None)
            if esm_mhc is not None:
                esm_mhc = esm_mhc.to(device)
            peptide  = batch["peptide"].to(device)
            neg_cdr3 = batch["neg_cdr3"].to(device)
            pos_bv   = batch["pos_bv"].to(device)
            pos_cdr3 = batch["pos_cdr3"].to(device)
            cdr3_mask = (neg_cdr3 != 0)

            if use_D:
                real_bv   = batch["real_bv"].to(device)
                real_cdr3 = batch["real_cdr3"].to(device)
            else:
                score = batch.get("score", None)
                if score is not None:
                    score = score.to(device).view(-1)

            # =============== Step 1: D ===============
            if use_D:
                for p in discriminator.parameters():
                    p.requires_grad_(True)

                with autocast(enabled=amp):
                    with torch.no_grad():
                        seq_fake_logits, _, bv_logits = generator(mhc, peptide, 
                                                                neg_cdr3, esm_mhc)

                    D_real1 = discriminator(
                        pos_cdr3,
                        cdr3_mask,
                        pos_bv,
                        is_logits=False
                    )
                    D_real2 = discriminator(
                        real_cdr3,
                        cdr3_mask,
                        real_bv,
                        is_logits=False
                    )
                    D_fake_gen = discriminator(
                        seq_fake_logits, 
                        cdr3_mask,
                        bv_logits,
                        is_logits=True
                    )

                    loss_D = (
                        (F.softplus(-D_real1).mean() + F.softplus(-D_real2).mean()) *0.5 +
                        F.softplus(D_fake_gen).mean()
                    )

                scaler.scale(loss_D / grad_accum_steps).backward()
            else:
                loss_D = torch.tensor(0.0, device=device)

            # =============== Step 2: G ===============
            for p in discriminator.parameters():
                p.requires_grad_(False)

            with autocast(enabled=amp):
                seq_logits, pair_logits, bv_logits = generator(
                    mhc, peptide, neg_cdr3, esm_mhc
                )

                Lseq = F.cross_entropy(seq_logits[cdr3_mask], 
                                       pos_cdr3[cdr3_mask])

                # pair pseudo-likelihood Lpair
                Lpair = loss_pair_pseudolikelihood(
                    seq_logits=seq_logits,
                    pair_logits=pair_logits,
                    aatype=pos_cdr3,
                    mask=cdr3_mask,
                    pair_mask=None,
                    lambda_reg=1e-2,
                    temperature=1.0,
                    reduction="mean",
                )
                LBV = F.cross_entropy(bv_logits, pos_bv)

                # Ladv
                D_fake_for_G = discriminator(
                    seq_logits,
                    cdr3_mask,
                    bv_logits,
                    is_logits=True
                )
                Ladv = F.softplus(-D_fake_for_G).mean()

                # distillation:
                if distillation:
                    if score is None:
                        Lmargin = seq_logits.new_tensor(0.0)
                    else:
                        E = _mrf_energy(
                            seq_logits=seq_logits,
                            pair_logits=pair_logits,
                            labels=pos_cdr3,
                            mask=cdr3_mask,
                        )  # [B]

                        # s = (score - score.mean()) / (score.std() + 1e-6)

                        idx = torch.randperm(E.size(0), device=E.device)
                        E1, E2 = E, E[idx]
                        s1, s2 = score, score[idx]
                        y_pair = (s1 > s2).float() * 2.0 - 1.0  # +1 / -1

                        Lmargin = F.softplus(-(E2 - E1) * y_pair).mean()
                else:
                    Lmargin = seq_logits.new_tensor(0.0)

                # total loss
                G_base = Lseq + Lpair + lambda_bv * LBV + lambda_adv * Ladv
                if distillation:
                    loss_G = lambda_base * G_base + lambda_margin * Lmargin
                else:
                    loss_G = G_base

            scaler.scale(loss_G / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                if use_D:
                    scaler.step(opt_D)
                scaler.step(opt_G)
                scaler.update()

                if use_D:
                    opt_D.zero_grad(set_to_none=True)
                opt_G.zero_grad(set_to_none=True)

            # 
            running["G_total"] += loss_G.detach().item()
            running["D_total"] += loss_D.detach().item()
            running["Lseq"]    += Lseq.detach().item()
            running["Lpair"]   += Lpair.detach().item()
            running["LBV"]     += LBV.detach().item()
            running["Ladv"]    += Ladv.detach().item()
            running["Lmargin"] += Lmargin.detach().item()

            if (global_step % log_interval) == 0:
                msg = (
                    f"[Epoch {epoch} Step {step+1}/{steps_per_epoch}] "
                    f"G={loss_G.item():.4f} D={loss_D.item():.4f} | "
                    f"Lseq={Lseq.item():.4f} Lpair={Lpair.item():.4f} "
                    f"LBV={LBV.item():.4f} Ladv={Ladv.item():.4f} "
                    f"Lmargin={Lmargin.item():.4f}"
                )
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg)

        for k in running:
            running[k] /= max(1, num_steps)

        msg = (
            f"[Epoch {epoch}] Train: "
            f"G={running['G_total']:.4f} D={running['D_total']:.4f} | "
            f"Lseq={running['Lseq']:.4f} Lpair={running['Lpair']:.4f} "
            f"LBV={running['LBV']:.4f} Ladv={running['Ladv']:.4f} "
            f"Lmargin={running['Lmargin']:.4f}"
        )
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

        # ================ validation & save ================
        if (val_loader is not None) and (epoch % eval_every_epochs == 0):
            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                val_loss,val_Lseq,val_Lpair,val_LBV = 0.0, 0.0, 0.0, 0.0
                val_steps = 0
                for v_batch in val_loader:
                    val_steps += 1
                    if val_steps > val_max_steps:
                        break

                    v_batch    =  v_batch[0]
                    mhc_v      = v_batch["mhc"].to(device)
                    esm_mhc_v  = v_batch.get("esm_mhc", None)
                    if esm_mhc_v is not None:
                        esm_mhc_v = esm_mhc_v.to(device)
                    peptide_v  = v_batch["peptide"].to(device)
                    neg_cdr3_v = v_batch["neg_cdr3"].to(device)
                    pos_bv_v   = v_batch["pos_bv"].to(device)
                    pos_cdr3_v = v_batch["pos_cdr3"].to(device)
                    cdr3_mask_v = (neg_cdr3_v != 0)

                    with autocast(enabled=amp):
                        seq_logits_v, pair_logits_v, bv_logits_v = generator(
                            mhc_v, peptide_v, neg_cdr3_v, esm_mhc_v
                        )
                        Lseq_v = F.cross_entropy(seq_logits_v[cdr3_mask_v], 
                                                 pos_cdr3_v[cdr3_mask_v])
                        Lpair_v = loss_pair_pseudolikelihood(
                            seq_logits_v, pair_logits_v, pos_cdr3_v,
                            mask=cdr3_mask_v, pair_mask=None,
                            lambda_reg=1e-2, temperature=1.0, reduction="mean"
                        )
                        LBV_v = F.cross_entropy(bv_logits_v, pos_bv_v)
                        loss_v = Lseq_v + Lpair_v + lambda_bv * LBV_v

                    val_loss += loss_v.item()
                    val_Lseq  += Lseq_v.item()
                    val_Lpair += Lpair_v.item()
                    val_LBV   += LBV_v.item()
                denom = max(1, val_steps)
                val_Lseq  /= denom
                val_Lpair /= denom
                val_LBV   /= denom
                val_loss  /= denom

            msg = (f"[Epoch {epoch}] Val: "
                   f"Total={val_loss:.4f}        | "
                   f"Lseq={val_Lseq:.4f} "
                   f"Lpair={val_Lpair:.4f} "
                   f"LBV={val_LBV:.4f}"
                   )
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(save_dir, "generator.pt")
                torch.save(generator.state_dict(), ckpt_path)
                if logger is not None:
                    logger.info(f"** Save best generator to {ckpt_path}")
                else:
                    print(f"** Save best generator to {ckpt_path}")
    
    if val_loader is None:
        ckpt_path = os.path.join(save_dir, "generator.pt")
        torch.save(generator.state_dict(), ckpt_path)

    ckpt_discriminator = os.path.join(save_dir, "discriminator.pt")
    torch.save(discriminator.state_dict(), ckpt_discriminator)
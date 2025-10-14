import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import os
import math
from typing import Dict, Optional, Iterable, Tuple
from collections import defaultdict

from loss import contact_losses, bce_loss
from model.lora import build_model_with_lora

task_every = {
    "mp_contact": 10,  
    "pt_contact": 10,  
}

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
    optimizer_ctor=lambda params: torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01),
    grad_accum_steps=1,
    amp=True,
    use_logits=False,
    new_optimizer_each_phase=True,
    log_interval=50,
    task_every: Dict[str, int] = None,   #
    val_loaders: Optional[Dict[str, Iterable]] = None,
    eval_every_epochs: int = 1,
    pep_align=True,
    use_lora=True,
    last_n: int = 2,
    cfg_seq_pair: Tuple[Tuple[int,int], Tuple[int,int]] = ((8,16),(4,8))
):
    """
    loaders key: value
      align         -> (mhc, peptide, cdr3, esm2_mhc)
      mp            -> (mhc, peptide, esm2_mhc, y_mp)
      pt            -> (peptide, cdr3, y_pt)
      mp_contact    -> (mhc, peptide, esm2_mhc, contact_bin, contact_dist)
      pt_contact    -> (peptide, cdr3, contact_bin, contact_dist)
      imm           -> (mhc, peptide, esm2_mhc, y_imm)
      mpt           -> (mhc, peptide, cdr3, esm2_mhc, trbv, y_mpt)
    """
    if task_every is None:
        task_every = {}

    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    scaler = GradScaler(enabled=amp)

    # lora
    if use_lora:
        model = build_model_with_lora(model, last_n, cfg_seq_pair, dropout=0.1, freeze_base=True)

    # λ
    lambdas_A = dict(align=0.10, MP=1.00, PT=1.00, IMM=0.0, contact=0.0, MPT=0.0, lg=0.0)
    lambdas_B = dict(align=0.10, MP=0.80, PT=0.80, IMM=0.0, contact=0.10, MPT=0.0, lg=0.0)
    lambdas_C = dict(align=0.10, MP=0.50, PT=0.50, IMM=0.60, contact=0.10, MPT=0.60, lg=0.30)

    phases = {
        "A": dict(lmb=lambdas_A, tasks=("align", "mp", "pt")),
        "B": dict(lmb=lambdas_B, tasks=("align", "mp", "pt", "mp_contact", "pt_contact")),
        "C": dict(lmb=lambdas_C, tasks=("align", "mp", "pt", "mp_contact", "pt_contact", "imm", "mpt")),
    }

    def make_optimizer():
        params = (p for p in model.parameters() if p.requires_grad)
        return optimizer_ctor(params)

    # phase #
    optimizer = None
    for ph, cfg in phases.items():
        if new_optimizer_each_phase or optimizer is None:
            optimizer = make_optimizer()

        active = {k: loaders.get(k) for k in cfg["tasks"] if loaders.get(k) is not None}
        assert len(active) > 0, f"Phase {ph} without dataloader"

        # iter
        iters = {k: _infinite_iter(dl) for k, dl in active.items()}
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
            pep_align=pep_align
        )

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
    val_loaders: Optional[Dict[str, Iterable]] = None,
    eval_every_epochs: int = 1,
    pep_align=True,
):
    def every_of(task: str) -> int:
        v = task_every.get(task, 1)
        return max(int(v), 1)
    
    if every_of("mp") != 1 or every_of("pt") != 1:
        print("[Warning] task_every['mp'] and task_every['pt'] setting as 1")
    
    global_step = 0

    running_sum = defaultdict(float)
    running_cnt = defaultdict(int) 

    best_val = float("inf") 
    for epoch in range(1, epochs + 1):
        model.train()
        # running = defaultdict(float)
        for step in range(1, steps_per_epoch + 1):
            due_tasks = [name for name in iters.keys() if (step % every_of(name) == 0)]
            assert len(due_tasks) > 0

            batches = {name: next(iters[name]) for name in due_tasks}

            with autocast(enabled=amp):
                # multi-dataloader loss
                loss_parts = compute_losses_multi_batches(
                    model=model,
                    device=device,
                    phase=phase_name,
                    batches=batches,
                    pep_align=pep_align
                )

                total = (
                    lambdas["align"]   * loss_parts["align"] +
                    lambdas["MP"]      * loss_parts["MP"] +
                    lambdas["PT"]      * loss_parts["PT"] +
                    lambdas["contact"] * loss_parts["contact"] +
                    lambdas["IMM"]     * loss_parts["IMM"] +
                    lambdas["MPT"]     * loss_parts["MPT"] +
                    lambdas["lg"]      * loss_parts["logic_imm"] +
                    lambdas["lg"]      * loss_parts["logic_mpt"]
                )
                loss = total / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

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
                       f"PT={avg('PT'):.4f} contact={avg('contact'):.4f} "
                       f"IMM={avg('IMM'):.4f} MPT={avg('MPT'):.4f} "
                       f"logic_imm={avg('logic_imm'):.4f} logic_mpt={avg('logic_mpt'):.4f}")
                print(msg)
                running_sum.clear()
                running_cnt.clear()

        # val        
        if (val_loaders is not None) and (epoch % eval_every_epochs == 0):
            val_parts, val_total = evaluate_phase_multi(
                model=model,
                val_loaders=val_loaders,
                device=device,
                phase_name=phase_name,
                lambdas=lambdas,
                pep_align=pep_align
            )
            print(f"[Phase {phase_name}] epoch {epoch} VAL:", {k: f"{v:.4f}" for k, v in val_parts.items()}, f"total={val_total:.4f}")

            if val_total < best_val - 1e-12:
                best_val = val_total
                torch.save({"model_state": model.state_dict(),
                            "phase": phase_name,
                            "lambdas": lambdas,
                            "val_total": best_val,
                            "epoch": epoch}, save_path)
                print(f"[Phase {phase_name}] ☆ improved, saved best full ckpt to {save_path}")
    # save
    if not os.path.isfile(save_path):
        torch.save({"model_state": model.state_dict(),
                    "phase": phase_name,
                    "lambdas": lambdas}, save_path)
    print(f"[Phase {phase_name}] saved to {save_path}")

@torch.no_grad()
def evaluate_phase_multi(
    model,
    val_loaders: Optional[Dict[str, Iterable]],
    device: str,
    phase_name: str,
    lambdas: Dict[str, float],
    max_batches: Optional[int] = None, 
    pep_align=True
) -> Tuple[Dict[str, float], float]:
    if not val_loaders:
        zeros = {k: 0.0 for k in ["align","MP","PT","contact","IMM","MPT","logic_imm","logic_mpt"]}
        return zeros, 0.0

    model.eval()
    sums = defaultdict(float)
    cnts = defaultdict(int)

    for task, dl in val_loaders.items():
        if dl is None:
            continue
        seen = 0
        for batch in dl:
            batches = {task: batch}
            parts = compute_losses_multi_batches(
                model=model,
                device=device,
                phase=phase_name,
                batches=batches,
                pep_align=pep_align
            )
            for k, v in parts.items():
                val = float(v.detach().item())
                sums[k] += val
                cnts[k] += 1
            seen += 1
            if (max_batches is not None) and (seen >= max_batches):
                break

    # mean
    means = {}
    for k in ["align","MP","PT","contact","IMM","MPT","logic_imm","logic_mpt"]:
        means[k] = (sums[k] / max(cnts[k], 1)) if k in sums else 0.0

    # λ
    val_total = (
        lambdas["align"]   * means["align"]   +
        lambdas["MP"]      * means["MP"]      +
        lambdas["PT"]      * means["PT"]      +
        lambdas["contact"] * means["contact"] +
        lambdas["IMM"]     * means["IMM"]     +
        lambdas["MPT"]     * means["MPT"]     +
        lambdas["lg"]      * means["logic_imm"] +
        lambdas["lg"]      * means["logic_mpt"]
    )

    return means, float(val_total)

def compute_losses_multi_batches(
    *,
    model,
    device,
    phase: str,
    batches: Dict[str, dict], 
    pep_align=True 
) -> Dict[str, torch.Tensor]:
    zeros = torch.tensor(0.0, device=device)
    out = dict(align=zeros, MP=zeros, PT=zeros, contact=zeros, IMM=zeros, MPT=zeros, 
               logic_imm=zeros, logic_mpt=zeros)

    # -------- Align --------
    if pep_align:
        if "align" in batches:
            b = batches["align"]
            mhc      = b["mhc"].to(device)
            peptide  = b["peptide"].to(device)
            cdr3     = b["cdr3"].to(device)
            esm2_mhc = b.get("esm2_mhc", None)
            if esm2_mhc is not None: esm2_mhc = esm2_mhc.to(device)

            out["align"] = model.pep_align(mhc, peptide, cdr3, esm2_mhc)
    else:
        out["align"] = torch.tensor(0.0, device=device)

    # -------- MP binding --------
    if "mp" in batches:
        b = batches["mp"]
        mhc      = b["mhc"].to(device)
        peptide  = b["peptide"].to(device)
        esm2_mhc = b.get("esm2_mhc", None)
        if esm2_mhc is not None: esm2_mhc = esm2_mhc.to(device)
        y_mp     = b["y_mp"].to(device).view(-1, 1)

        mp_out = model.mp_pred(mhc, peptide, esm2_mhc, contact=False, immunogenicity=(phase=="C"))
        out["MP"] = bce_loss(mp_out["binding_prob"], y_mp)

        # L_logic_IMM
        if phase=="C":
            out["logic_imm"] = out["logic_imm"] + \
                F.relu(mp_out["immunogenicity_prob"] - mp_out["binding_prob"]).mean() * 0.5

    # -------- PT binding --------
    if "pt" in batches:
        b = batches["pt"]
        peptide = b["peptide"].to(device)
        cdr3    = b["cdr3"].to(device)
        y_pt    = b["y_pt"].to(device).view(-1, 1)

        pt_out = model.pt_pred(peptide, cdr3, contact=False)
        out["PT"] = bce_loss(pt_out["binding_prob"], y_pt)

    # -------- MP contact --------
    if 'mp_contact' in batches:
        if phase in ("B", "C"):
            b = batches["mp_contact"]
            mhc      = b["mhc"].to(device)
            peptide  = b["peptide"].to(device)
            esm2_mhc = b.get("esm2_mhc", None)
            if esm2_mhc is not None: esm2_mhc = esm2_mhc.to(device)

            mp_out = model.mp_pred(mhc, peptide, esm2_mhc, contact=True, immunogenicity=False)

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
                prob_tgt=p_bin, dist_tgt=p_dis, pair_mask=pair_mask
            )
            out["contact"] = out["contact"] + lp + ld
            
    # -------- PT contact --------
    if 'pt_contact' in batches:
        if phase in ("B", "C"):
            b = batches["pt_contact"]
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
                prob_tgt=p_bin, dist_tgt=p_dis, pair_mask=pair_mask
            )
            out["contact"] = out["contact"] + lp + ld

    # -------- IMM --------
    if "imm" in batches:
        if phase == "C":
            b = batches["imm"]
            mhc      = b["mhc"].to(device)
            peptide  = b["peptide"].to(device)
            esm2_mhc = b.get("esm2_mhc", None)
            if esm2_mhc is not None: esm2_mhc = esm2_mhc.to(device)
            y_imm    = b["y_imm"].to(device).view(-1, 1)

            mp_out = model.mp_pred(mhc, peptide, esm2_mhc, contact=False, immunogenicity=True)
            out["IMM"] = bce_loss(mp_out["immunogenicity_prob"], y_imm)

            # L_logic_IMM
            out["logic_imm"] = out["logic_imm"] + \
                F.relu(mp_out["immunogenicity_prob"] - mp_out["binding_prob"]).mean() * 0.5

    # -------- MPT（若提供，Phase C 才计入 λ）--------
    if "mpt" in batches:
        b = batches["mpt"]
        mhc      = b["mhc"].to(device)
        peptide  = b["peptide"].to(device)
        cdr3     = b["cdr3"].to(device)
        esm2_mhc = b.get("esm2_mhc", None)
        if esm2_mhc is not None: esm2_mhc = esm2_mhc.to(device)
        trbv     = b.get("trbv", None)
        if trbv is not None: trbv = trbv.to(device)
        y_mpt    = b["y_mpt"].to(device).view(-1, 1)

        mpt_out = model.mpt_pred(mhc, peptide, cdr3, esm2_mhc, trbv)
        out["MPT"] = bce_loss(mpt_out["mpt_prob"], y_mpt)

        # L_logic_MPT = ReLU(P_MPT - min(P_MP, P_PT))
        min_mp_pt = torch.minimum(mpt_out["mp_prob"], mpt_out["pt_prob"])
        out["logic_mpt"] = out["logic_mpt"] + F.relu(mpt_out["mpt_prob"] - min_mp_pt).mean()
    return out
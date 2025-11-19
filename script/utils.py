import os, json
from typing import Dict, Optional, Callable, Any, Tuple
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .train import train_three_phases_multi_loaders, evaluate_phase_multi
from .model.pepbridge import PepBridge

import csv
import pandas as pd

import logging
import sys
import random

def setup_logger(logfile=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if logfile:
        file_handler = logging.FileHandler(logfile, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger

def unwrap_sd(obj):
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    out = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."): k = k[7:]
        if k.startswith("model."):  k = k[6:]
        out[k] = v
    return out

def take_embed_trunk(sd):
    return {k: v for k, v in sd.items()
            if k.startswith("embedder.") or k.startswith("pair_aware_trunk.")}

def encoder_load_state_dict(model, peptide_pt_path, cdr3_pt_path, device):
    pep_mlm_sd  = take_embed_trunk(unwrap_sd(torch.load(peptide_pt_path, 
                                                        map_location=device)))
    cdr3_mlm_sd = take_embed_trunk(unwrap_sd(torch.load(cdr3_pt_path, 
                                                        map_location=device)))
    
    model.peptide_encoder.load_state_dict(pep_mlm_sd,  strict=True)
    model.cdr3_encoder.load_state_dict(cdr3_mlm_sd,   strict=True)

    assert all(torch.equal(getattr(model,enc).state_dict()[k], 
                           sd[k].to(getattr(model,enc).state_dict()[k].device,
                           dtype=getattr(model,enc).state_dict()[k].dtype)) for enc, sd in [('peptide_encoder', pep_mlm_sd), ('cdr3_encoder', cdr3_mlm_sd)] for k in sd)
    
    return model

def df_train_test_split(df, val_split, seed):
    df_val = df.sample(frac=val_split, random_state=seed)
    df_train = df.drop(df_val.index)

    df_val = df_val.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    return df_train, df_val

def read_csv_with_index_allow_duplicate_names(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)      
        data = list(reader)

    df = pd.DataFrame(data, columns=header)
    df.set_index(df.columns[0], inplace=True)
    return df

def model_fn(aa_vocab_size=26, trbv_vocab_size=78):
    model = PepBridge(
        aa_size=aa_vocab_size,
        max_len_dict={"mhc":34, "peptide":15, "cdr3":20},
        d_seq=128, d_head_seq=32,
        d_pair=64, d_head_pair=32,
        dropout=0.1,
        n_layers_dict={"mhc":3, "peptide":6, "cdr3":6,
                        "mp":3, "pt":3, "mpt":1},
        trbv_size=trbv_vocab_size,
    )
    return model

def _make_folds(n_samples: int, n_splits: int, shuffle: bool, seed: int,
                y: Optional[np.ndarray]):
    """返回 [(train_idx, val_idx), ...]；若给 y 则分层。"""
    idx = np.arange(n_samples)
    try:
        if y is not None:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
            return [(tr, va) for tr, va in skf.split(idx, y)]
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
            return [(tr, va) for tr, va in kf.split(idx)]
    except Exception:
        # 无 sklearn 时的简易实现
        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(idx)
        chunks = np.array_split(idx, n_splits)
        folds = []
        for i in range(n_splits):
            va = np.array(chunks[i], dtype=int)
            tr = np.setdiff1d(idx, va, assume_unique=False)
            folds.append((tr, va))
        return folds

def _mk_loader(ds, indices, batch_size, shuffle, num_workers, pin_memory, drop_last, collate_fn):
    sub = Subset(ds, indices)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, drop_last=drop_last, collate_fn=collate_fn)

def run_five_fold_cv_multi_loaders(
    model_fn: Callable[[], torch.nn.Module],
    task_datasets: Dict[str, torch.utils.data.Dataset],
    *,
    save_root: str = "./cv_ckpts_multi",
    n_splits: int = 5,
    seed: int = 42,
    # 不同任务可不同 batch_size / collate
    batch_sizes: Dict[str, int] = None,
    collate_fns: Dict[str, Optional[Callable]] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last_train: bool = False,
    drop_last_val: bool = False,
    # 可选分层标签：每个任务各自的 y（1D numpy array），没有就传 None
    stratify_y: Dict[str, Optional[np.ndarray]] = None,
    # 训练函数参数（原样透传）
    device: str = "cuda",
    epochs_A: int = 5, epochs_B: int = 10, epochs_C: int = 10,
    grad_accum_steps: int = 1,
    amp: bool = True,
    use_logits: bool = False,
    new_optimizer_each_phase: bool = True,
    log_interval: int = 50,
    task_every: Optional[Dict[str, int]] = None,
    pep_align: bool = True,
    use_lora: bool = True,
    last_n: int = 2,
    cfg_seq_pair: Tuple[Tuple[int, int], Tuple[int, int]] = ((8, 16), (4, 8)),
) -> Dict[str, Any]:
    """
    task_datasets: 任务名 -> Dataset；键需与训练函数的 loaders 键一致，例如：
      {"align": ds_align, "mp": ds_mp, "pt": ds_pt, "mp_contact": ds_mpc, "pt_contact": ds_ptc, "imm": ds_imm, "mpt": ds_mpt}
    stratify_y: 任务名 -> 1D numpy 数组（与该任务 dataset 等长），可为 None
    batch_sizes/collate_fns: 任务名 -> 值；未提供的任务使用默认 batch_size=32、collate_fn=None
    """
    # seed_everything(seed)
    os.makedirs(save_root, exist_ok=True)
    batch_sizes = batch_sizes or {}
    collate_fns = collate_fns or {}
    stratify_y = stratify_y or {}

    # 只对存在的任务做 CV；其余缺失任务会在每折自动跳过
    task_names = list(task_datasets.keys())

    # 准备每个任务自己的 K 折索引
    task_folds: Dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for task in task_names:
        ds = task_datasets[task]
        y = stratify_y.get(task, None)
        task_folds[task] = _make_folds(len(ds), n_splits=n_splits, shuffle=True, seed=seed, y=y)

    all_results = []
    for fold_id in range(1, n_splits + 1):
        print(f"\n========== Fold {fold_id}/{n_splits} ==========")
        fold_dir = os.path.join(save_root, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        # 为每个任务取该折的索引并构建 train/val loader
        train_loaders, val_loaders = {}, {}
        for task in task_names:
            ds = task_datasets[task]
            folds = task_folds[task]
            tr_idx, va_idx = folds[fold_id - 1]   # 每任务各自第 fold_id 折
            bs = batch_sizes.get(task, 32)
            collate = collate_fns.get(task, None)

            # 训练集通常 shuffle=True
            train_loaders[task] = _mk_loader(ds, tr_idx, bs, True,  num_workers, pin_memory, drop_last_train, collate)
            # 验证集 shuffle=False
            val_loaders[task]   = _mk_loader(ds, va_idx, bs, False, num_workers, pin_memory, drop_last_val,   collate)

        # 构建全新模型
        model = model_fn().to(device)

        # —— 开始三阶段训练（内部会保存 phase_A/B/C.pt）——
        train_three_phases_multi_loaders(
            model,
            loaders=train_loaders,
            device=device,
            save_dir=fold_dir,
            epochs_A=epochs_A, epochs_B=epochs_B, epochs_C=epochs_C,
            grad_accum_steps=grad_accum_steps,
            amp=amp,
            use_logits=use_logits,
            new_optimizer_each_phase=new_optimizer_each_phase,
            log_interval=log_interval,
            task_every=task_every or {},
            val_loaders=val_loaders,            # 同折验证集
            eval_every_epochs=1,
            pep_align=pep_align,
            use_lora=use_lora,
            last_n=last_n,
            cfg_seq_pair=cfg_seq_pair,
        )

        # —— 评估每阶段 ckpt（各自使用 ckpt 中的 lambdas）——
        phase_ckpts = {
            "A": os.path.join(fold_dir, "phase_A.pt"),
            "B": os.path.join(fold_dir, "phase_B.pt"),
            "C": os.path.join(fold_dir, "phase_C.pt"),
        }
        fold_metrics = {}
        for ph, ckpt in phase_ckpts.items():
            if not os.path.isfile(ckpt):
                print(f"[WARN] missing checkpoint: {ckpt}")
                continue
            state = torch.load(ckpt, map_location=device)
            lambdas = state.get("lambdas", dict(align=0.0, MP=1.0, PT=1.0, IMM=0.0, contact=0.0, MPT=0.0, lg=0.0))

            # 重新 build 一个模型实例再加载（避免遗留状态）
            m_eval = model_fn().to(device)
            m_eval.load_state_dict(state["model_state"], strict=True)
            m_eval.eval()

            mets, total = evaluate_phase_multi(
                model=m_eval,
                val_loaders=val_loaders,
                device=device,
                phase_name=ph,
                lambdas=lambdas,
                pep_align=pep_align,
            )
            fold_metrics[f"phase_{ph}"] = {"parts": mets, "total": total}

            # 保存该阶段验证指标
            with open(os.path.join(fold_dir, f"val_metrics_phase_{ph}.json"), "w") as f:
                json.dump({"parts": mets, "total": total}, f, indent=2)

        all_results.append(fold_metrics)

    # —— 汇总均值 —— #
    mean_metrics = {
        "phase_A": {"parts": defaultdict(float), "total": 0.0},
        "phase_B": {"parts": defaultdict(float), "total": 0.0},
        "phase_C": {"parts": defaultdict(float), "total": 0.0},
    }
    count_folds = 0
    for fm in all_results:
        if not fm:
            continue
        count_folds += 1
        for ph in ("phase_A", "phase_B", "phase_C"):
            if ph not in fm:
                continue
            for k, v in fm[ph]["parts"].items():
                mean_metrics[ph]["parts"][k] += float(v)
            mean_metrics[ph]["total"] += float(fm[ph]["total"])
    if count_folds > 0:
        for ph in mean_metrics:
            mean_metrics[ph]["parts"] = {k: v / count_folds for k, v in mean_metrics[ph]["parts"].items()}
            mean_metrics[ph]["total"] = mean_metrics[ph]["total"] / count_folds

    summary = {"fold_metrics": all_results, "mean_metrics": mean_metrics, "n_folds": count_folds}
    with open(os.path.join(save_root, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== 5-fold summary (mean over folds) ====")
    for ph in ("phase_A", "phase_B", "phase_C"):
        pm = mean_metrics[ph]
        friendly = {k: round(v, 4) for k, v in pm["parts"].items()}
        print(f"{ph} total={pm['total']:.4f} parts={friendly}")
    return summary

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# def seed_everything(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

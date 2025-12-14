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

def df_train_test_split(df, val_split=None, seed=42,
                        n_splits=None, fold=None,
                        meta=None):
    df = df.reset_index(drop=True)
    n = len(df)

    rng = np.random.RandomState(seed)

    # ============ K-fold ============
    if n_splits is not None and fold is not None:
        assert 0 <= fold < n_splits, "fold must in [0, n_splits-1]"

        perm = rng.permutation(n)

        fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
        fold_sizes[: n % n_splits] += 1

        start = fold_sizes[:fold].sum()
        stop = start + fold_sizes[fold]

        val_idx = perm[start:stop]
        train_idx = np.concatenate([perm[:start], perm[stop:]])

    else:
        # ============ random splite ============
        assert val_split is not None

        perm = rng.permutation(n)
        val_size = int(round(n * val_split))

        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)

    def split_meta(sub_df: pd.DataFrame):
        if meta is None:
            return None
        res = {}
        for _, row in sub_df.reset_index(drop=True).iterrows():
            pep = str(row["peptide"])
            cdr3 = str(row["cdr3"])
            key = pep + "||" + cdr3
            if key in meta:
                res[key] = meta[key]
        return res

    meta_train = split_meta(df_train)
    meta_val   = split_meta(df_val)

    if meta is None:
        return df_train, df_val
    else:
        return df_train, df_val, meta_train, meta_val

def read_csv_with_index_allow_duplicate_names(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)      
        data = list(reader)

    df = pd.DataFrame(data, columns=header)
    df.set_index(df.columns[0], inplace=True)
    return df

def model_fn(aa_vocab_size=26, trbv_vocab_size=78, 
             trav_vocab_size=None, traj_vocab_size=None):
    model = PepBridge(
        aa_size=aa_vocab_size,
        max_len_dict={"mhc":34, "peptide":15, "cdr3":20},
        d_seq=128, d_head_seq=32,
        d_pair=64, d_head_pair=32,
        dropout=0.1,
        n_layers_dict={"mhc":3, "peptide":6, "cdr3":6,
                        "mp":3, "pt":3, "mpt":1},
        trbv_size=trbv_vocab_size,
        trav_size=trav_vocab_size,
        traj_size=traj_vocab_size
    )
    return model

def build_hard_neg_map_from_df(df_train: pd.DataFrame,
                               neg_df: pd.DataFrame):
    df_train_idx = df_train.reset_index(drop=True).copy()
    df_train_idx["__key__"] = df_train_idx["peptide"].astype(str) + "||" + df_train_idx["cdr3"].astype(str)

    neg_df = neg_df.copy()
    neg_df["__key__"] = neg_df["peptide"].astype(str) + "||" + neg_df["cdr3_pos"].astype(str)

    neg_by_key = defaultdict(list)
    for _, row in neg_df.iterrows():
        key = row["__key__"]
        neg_by_key[key].append(row["cdr3_neg"])

    hard_neg_map = {}
    for _, row in df_train_idx.iterrows():
        key = row["__key__"]
        hard_list = neg_by_key.get(key, [])
        hard_neg_map[key] = hard_list

    return hard_neg_map

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


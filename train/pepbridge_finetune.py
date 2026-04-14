from script.model.pepbridge import PepBridge
from script.model.lora import build_model_with_lora
from script.dataset import  MPTFineTuneDataSet
from script.dataprocess import mk_aa_dict, mk_bv_dict, mk_aj_dict, mk_av_dict
from script.utils import df_train_test_split, setup_logger, model_fn
from script.train import train_fintune

import pandas as pd 
import numpy as np
import os
import sys

import torch
from torch.utils.data import DataLoader

def parse_kv_args(argv):
    out = {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            out[k] = v
    return out

def str2bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y", "t")

if __name__ == "__main__":
    os.chdir('/data5/tem/laiwp131/pMHC_TCR_20251103/')

    aa_dict = mk_aa_dict()
    bv_dict = mk_bv_dict()
    av_dict = mk_av_dict()
    aj_dict = mk_aj_dict()

    logger = setup_logger()
    device = 'cuda'

    cli_args = parse_kv_args(sys.argv)

    score_column = cli_args.get("score_column", "clone_id_size")
    csv_dir = cli_args.get("csv_path", "A0201_GILGFVFTL_Influenza.csv")

    save_dir = cli_args.get("save_path", "./checkpoints_finetune")
    pepbridge_dir = cli_args.get("load_path", './checkpoints_multi_lora_align3_ln/fold_1/phase_C.pt')
    
    finetune = str2bool(cli_args.get("finetune", "True"))
    
    trav = str2bool(cli_args.get("trav", "False"))
    traj = str2bool(cli_args.get("traj", "False"))
    trav_vocab_size = len(av_dict) if trav else None
    traj_vocab_size = len(aj_dict) if traj else None

    ####
    fintune_csv = pd.read_csv(csv_dir,index_col=0,header=0)
    df_train, df_val = df_train_test_split(fintune_csv, val_split=0.1, seed=42)

    train_loader = DataLoader(
        MPTFineTuneDataSet(
            mpt_df=df_train, mhc_type='HLAI', 
                mhc_max_len=34, pep_max_len=15, cdr3_max_len=20,
                bv=True, av=trav, aj=traj, score=score_column),
                batch_size=64, shuffle=True, drop_last=True)

    val_loader = DataLoader(   
            MPTFineTuneDataSet(            
                mpt_df=df_val, mhc_type='HLAI', 
                mhc_max_len=34, pep_max_len=15, cdr3_max_len=20,
                bv=True, av=trav, aj=traj, score=score_column),
                batch_size=64, shuffle=False, drop_last=False)
                
    ###
    model = model_fn(aa_vocab_size=len(aa_dict),
                    trbv_vocab_size=len(bv_dict),
                    trav_vocab_size=trav_vocab_size,
                    traj_vocab_size=traj_vocab_size)
    
    model = build_model_with_lora(model, last_n=2, 
                                cfg_seq_pair=((8,16),(4,8)), 
                                dropout=0.1, freeze_base=True,
                                print_trainabel=False)
    
    model = model.to(device)
    ckpt = torch.load(pepbridge_dir, map_location=device)
    state_dict = ckpt['model_state']
    
    if finetune is False:
        model.load_state_dict(state_dict, strict=True)
        arr_prob = np.zeros((val_loader.dataset.shape[0], 1))
        model.eval()
        with torch.no_grad():
            for batch, idx in val_loader:
                mhc      = batch["mhc"].to(device)
                esm_mhc  = batch.get("esm_mhc", None)
                if esm_mhc is not None:
                    esm_mhc = esm_mhc.to(device)
                peptide  = batch["peptide"].to(device)
                cdr3     = batch["cdr3"].to(device)
                trbv     = batch["trbv"].to(device)
                score    = batch["score"].to(device)
                trav     = batch.get("trav", None)
                traj     = batch.get("traj", None)
                if trav is not None:
                    trav = trav.to(device)
                if traj is not None:
                    traj = traj.to(device)
                logits = model.mpt_pred_finetune(
                        mhc, peptide, cdr3, esm_mhc, trbv, 
                        trav, traj
                    )
                arr_prob[idx] = logits.detach().cpu().numpy()
        df_val['pred'] = arr_prob
        x = df_val[score_column]
        y = df_val["pred"]

        m = x.notna() & y.notna()
        r_s = x[m].corr(y[m], method="spearman")

        logger.info(f"spearman r: {score_column} vs pred = {r_s:.4f} ")

    else:    
        def is_mpt_head_key(k: str) -> bool:
            parts = k.split(".")
            return ("mpt_pred_head" in parts)
    
        to_drop = [k for k in list(state_dict.keys()) if is_mpt_head_key(k)]
        for k in to_drop:
            state_dict.pop(k)
    
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        train_fintune(
            model,
            device,
            optimizer_ctor=lambda params: torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01),
            loader = train_loader,
            steps_per_epoch = 200,
            epochs = 10,
            save_path = save_dir,
            grad_accum_steps = 1,
            amp = True,
            log_interval = 50,
            logger=logger
        )
    
        arr_prob = np.zeros((val_loader.dataset.shape[0], 1))
        model.eval()
        with torch.no_grad():
            for batch, idx in val_loader:
                mhc      = batch["mhc"].to(device)
                esm_mhc  = batch.get("esm_mhc", None)
                if esm_mhc is not None:
                    esm_mhc = esm_mhc.to(device)
                peptide  = batch["peptide"].to(device)
                cdr3     = batch["cdr3"].to(device)
                trbv     = batch["trbv"].to(device)
                score    = batch["score"].to(device)
                trav     = batch.get("trav", None)
                traj     = batch.get("traj", None)
                if trav is not None:
                    trav = trav.to(device)
                if traj is not None:
                    traj = traj.to(device)
                logits = model.mpt_pred_finetune(
                        mhc, peptide, cdr3, esm_mhc, trbv, 
                        trav, traj
                    )
                arr_prob[idx] = logits.detach().cpu().numpy()
        df_val['pred'] = arr_prob
        x = df_val[score_column]
        y = df_val["pred"]
    
        m = x.notna() & y.notna()
        r_s = x[m].corr(y[m], method="spearman")
    
        logger.info(f"spearman r: {score_column} vs pred = {r_s:.4f} ")


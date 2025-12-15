from script.model.pepbridge import PepBridge
from script.model.lora import build_model_with_lora
from script.dataset import PTDataSet, MPDataSet, MPTDataSet, MultiNegPairPTDataset, \
    build_loader_for_long_tail, build_loader_uniform_by_peptide
from script.dataprocess import mk_aa_dict, mk_bv_dict
from script.utils import model_fn, encoder_load_state_dict, \
    df_train_test_split, setup_logger, build_hard_neg_map_from_df
from script.train import train_three_phases_multi_loaders

import pandas as pd 
import numpy as np
import os
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

def weighted_sampler(df, column, replacement, alpha=1):
    lengths = np.array(df[column])
    group_ids = lengths.copy()
    group_ids[group_ids >= 12] = 12

    unique_groups, counts = np.unique(group_ids, return_counts=True)
    inv = 1.0 / np.power(counts.astype(np.float32), alpha)
    group_weights_arr = inv / inv.sum()
    group_weights = {g: w for g, w in zip(unique_groups, group_weights_arr)}
    # group_weights = {g: 1.0 / c for g, c in zip(unique_groups, counts)}
    
    sample_weights = np.array([group_weights[g] for g in group_ids], dtype=np.float32)
    sample_weights_t = torch.from_numpy(sample_weights) 
    sampler = WeightedRandomSampler(
        weights=sample_weights_t,
        num_samples=len(sample_weights_t), 
        replacement=replacement                   
    )
    return sampler

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
    logger = setup_logger()
    
    cli_args = parse_kv_args(sys.argv)
    default_ckpt_path = './checkpoints_multi_lora_align_all/'
    save_root = cli_args.get("path", default_ckpt_path)

    pep_align = str2bool(cli_args.get("pep_align", "true"))
    all_align =  cli_args.get("all_align", -3)
    ln = str2bool(cli_args.get("ln", "False"))
    msg = f"pep_align: {pep_align}, all_align: {all_align}, ln: {ln}"
    logger.info(msg)
    
    aa_dict = mk_aa_dict()
    bv_dcit = mk_bv_dict()

    device = 'cuda'

    pt_df = pd.read_csv('pt_pos_train.csv', header=0, index_col=0)
    pt_neg_df = pd.read_csv('pt_pos_train_hard20.csv', header=0, index_col=0)
    pt_neg_dict = build_hard_neg_map_from_df(df_train=pt_df, neg_df=pt_neg_df)

    mp_df = pd.read_csv('mp_train2.csv', header=0, index_col=0)
    mpt_df = pd.read_csv('mpt_train.csv', header=0, index_col=0)
    align_df = pd.read_csv('random_pMHC_cdr3_train.csv', header=0, index_col=0)
    imm_df = pd.read_csv('immunogenicity_train.csv', header=0, index_col=0)
    mp_contact_df = pd.read_csv('./PepBridge-main/data/mp_pdb_train.csv',  header=0, index_col=0)
    pt_contact_df = pd.read_csv('./PepBridge-main/data/pt_pdb_train.csv', header=0, index_col=0)

    
    seed=42
    mhc_max_len=34
    pep_max_len=15
    cdr3_max_len=20
    n_folds=5

    for i in range(n_folds): 
        logger.info(f"\n========== Fold {i+1} / {n_folds} ==========\n")
        save_dir = save_root + f"fold_{i+1}"
        pt_train, pt_val, meta_train, meta_val = df_train_test_split(pt_df, seed=seed, n_splits=n_folds, 
                                                                    fold=i, meta=pt_neg_dict)
        mp_train, mp_val = df_train_test_split(mp_df, seed=seed, n_splits=n_folds, fold=i)
        mpt_train, mpt_val = df_train_test_split(mpt_df, seed=seed, n_splits=n_folds, fold=i)
        align_train, align_val = df_train_test_split(align_df, seed=seed, n_splits=n_folds, fold=i)

        imm_train, imm_val = df_train_test_split(imm_df, seed=seed, n_splits=n_folds, fold=i)
        mp_contact_train, mp_contact_val = df_train_test_split(mp_contact_df, seed=seed, n_splits=n_folds, fold=i)
        pt_contact_train, pt_contact_val = df_train_test_split(pt_contact_df, seed=seed, n_splits=n_folds, fold=i)

        mp_train_sampler = weighted_sampler(df=mp_train, column='len', replacement=True)
        mp_val_sampler = weighted_sampler(df=mp_val, column='len', replacement=False)

        mp_train_loader = DataLoader(MPDataSet(mp_df=mp_train, mhc_type='HLAI', 
                            mhc_max_len=mhc_max_len, pep_max_len=pep_max_len,
                                binding=True, 
                                immunogenicity=False, contact=False, mask=None),
                                batch_size=128, shuffle=False,sampler=mp_train_sampler)

        mp_val_loader = DataLoader(MPDataSet(mp_df=mp_val, mhc_type='HLAI', 
                                mhc_max_len=mhc_max_len, pep_max_len=pep_max_len,
                                    binding=True, 
                                    immunogenicity=False, contact=False, mask=None),
                                    batch_size=128, shuffle=False, sampler=mp_val_sampler)

        imm_train_loader = DataLoader(MPDataSet(mp_df=imm_train, mhc_type='HLAI', 
                            mhc_max_len=mhc_max_len, pep_max_len=pep_max_len,
                                binding=False, 
                                immunogenicity=True, contact=False, mask=None),
                                batch_size=32, shuffle=True)

        imm_val_loader = DataLoader(MPDataSet(mp_df=imm_val, mhc_type='HLAI', 
                            mhc_max_len=mhc_max_len, pep_max_len=pep_max_len,
                                binding=False, 
                                immunogenicity=True, contact=False, mask=None),
                                batch_size=32, shuffle=True)

        mp_contact_train_loader = DataLoader(MPDataSet(mp_df=mp_contact_train, mhc_type='HLAI', 
                            mhc_max_len=mhc_max_len, pep_max_len=pep_max_len,
                                binding=False, 
                                immunogenicity=False, contact=True, mask=None),
                                batch_size=8, shuffle=True)

        mp_contact_val_loader = DataLoader(MPDataSet(mp_df=mp_contact_val, mhc_type='HLAI', 
                            mhc_max_len=mhc_max_len, pep_max_len=pep_max_len,
                                binding=False, 
                                immunogenicity=False, contact=True, mask=None),
                                batch_size=4, shuffle=True)

        pt_train_loader =  DataLoader(
            MultiNegPairPTDataset(pt_train, 
                    pep_max_len=pep_max_len, cdr3_max_len=cdr3_max_len,
                    hard_neg_map=meta_train,
                    k_cross=8, k_hard=2,
                    pep_mask=0.5, cdr3_mask=0.5,
                    avoid_duplicates=True),
            batch_size=64, shuffle=True)

        pt_val_loader = build_loader_uniform_by_peptide(
            dataset=MultiNegPairPTDataset(pt_val, 
                            pep_max_len=pep_max_len, cdr3_max_len=cdr3_max_len,
                            hard_neg_map=meta_val,
                            k_cross=8, k_hard=2,
                            pep_mask=None, cdr3_mask=None,
                            avoid_duplicates=True),
            peptide_ids = pt_val.peptide, 
            peptides_per_step=32, 
            samples_per_peptide=1, 
            seed=seed,
            num_workers=0, pin_memory=False,
            ensure_full_batch=False
        )

        mpt_train_loader= DataLoader(
            MPTDataSet(mpt_train, mhc_type='HLAI', 
                    mhc_max_len=mhc_max_len, pep_max_len=pep_max_len, cdr3_max_len=cdr3_max_len,
                    bv=True, binding=True, 
                    pep_mask=0.5, cdr3_mask=0.5),
        batch_size=128, shuffle=True)

        mpt_val_loader = build_loader_uniform_by_peptide(
            dataset=MPTDataSet(mpt_val, mhc_type='HLAI', 
                    mhc_max_len=mhc_max_len, pep_max_len=pep_max_len, cdr3_max_len=cdr3_max_len,
                    bv=True, binding=True, 
                    pep_mask=None, cdr3_mask=None),
            peptide_ids = mpt_val.peptide, 
            peptides_per_step=64, 
            samples_per_peptide=2, 
            seed=seed,
            num_workers=0, pin_memory=False,
            ensure_full_batch=False
        )

        pt_contact_train_loader = DataLoader(PTDataSet(pt_contact_train, 
                                        pep_max_len=pep_max_len, cdr3_max_len=cdr3_max_len,
                                        binding=False, contact=True, 
                                        pep_mask=None, cdr3_mask=None),
                                        batch_size=4, shuffle=True)

        pt_contact_val_loader = DataLoader(PTDataSet(pt_contact_val, 
                                        pep_max_len=pep_max_len, cdr3_max_len=cdr3_max_len,
                                        binding=False, contact=True, 
                                        pep_mask=None, cdr3_mask=None),
                                        batch_size=2, shuffle=True)

        align_train_loader = DataLoader(MPTDataSet(align_train, mhc_type='HLAI', 
                    mhc_max_len=mhc_max_len, pep_max_len=pep_max_len, cdr3_max_len=cdr3_max_len,
                    bv=False, binding=False, 
                    pep_mask=None, cdr3_mask=None),
                    batch_size=64, shuffle=True)

        align_val_loader = DataLoader(MPTDataSet(align_val, mhc_type='HLAI', 
                    mhc_max_len=mhc_max_len, pep_max_len=pep_max_len, cdr3_max_len=cdr3_max_len,
                    bv=False, binding=False, 
                    pep_mask=None, cdr3_mask=None),
                    batch_size=128, shuffle=True)

        train_loaders =dict(
            align=align_train_loader, mp=mp_train_loader,pt=pt_train_loader,
            mp_contact=mp_contact_train_loader, pt_contact=pt_contact_train_loader,
            imm=imm_train_loader, mpt=mpt_train_loader
        )

        val_loaders =dict(
            align=align_val_loader, mp=mp_val_loader,pt=pt_val_loader,
            mp_contact=mp_contact_val_loader, pt_contact=pt_contact_val_loader,
            imm=imm_val_loader, mpt=mpt_val_loader
        )
        
        model = model_fn(aa_vocab_size=len(aa_dict),
                        trbv_vocab_size=len(bv_dcit)).to(device)
        model = encoder_load_state_dict(model, peptide_pt_path='peptide_mlm.pt',
                                        cdr3_pt_path='cdr3_mlm.pt', device=device)

        train_three_phases_multi_loaders(  
            model=model,
            loaders=train_loaders,
            device="cuda",
            save_dir=save_dir,
            epochs_A=12, epochs_B=9, epochs_C=6,
            steps_per_epoch=1000,
            optimizer_ctor=lambda params: torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01),
            grad_accum_steps=1,
            amp=True,
            new_optimizer_each_phase=False,
            log_interval=200,
            task_every = {"mp_contact": 50, "pt_contact": 50},   #
            val_loaders= val_loaders,
            eval_every_epochs=1,
            pep_align=pep_align,
            all_align=all_align,
            ln=ln,
            use_lora=True,
            last_n=2,
            cfg_seq_pair=((8,16),(4,8)),
            logger=logger)

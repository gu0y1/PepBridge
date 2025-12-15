from script.model.generator import PepBridgeGenerator,Discriminator
from script.model.lora import inject_lora_into_trunk_last_n,freeze_module_except_lora
from script.dataset import  MPTGenDataSet,build_loader_uniform_by_peptide
from script.dataprocess import mk_aa_dict, mk_bv_dict
from script.utils import df_train_test_split, setup_logger
from script.train import generator_train

import pandas as pd 
import numpy as np
import os
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

def weighted_sampler(df, column, replacement, alpha=1):
    lengths = np.array(df[column])
    group_ids = lengths.copy()

    unique_groups, counts = np.unique(group_ids, return_counts=True)
    inv = 1.0 / np.power(counts.astype(np.float32), alpha)
    group_weights_arr = inv / inv.sum()
    group_weights = {g: w for g, w in zip(unique_groups, group_weights_arr)}
    
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

    aa_dict = mk_aa_dict()
    bv_dict = mk_bv_dict()
    logger = setup_logger()
    device = 'cuda'

    cli_args = parse_kv_args(sys.argv)
    save_dir = cli_args.get("save_path", "./checkpoints_generator")
    pepbridge_dir = cli_args.get("load_path", './checkpoints_multi_lora_align_all/phase_C.pt')

    use_lora = str2bool(cli_args.get("use_lora", "true"))
    finetune_trunk = str2bool(cli_args.get("finetune_trunk", "False"))

    ####
    gen_train = pd.read_csv('gen_train.csv',index_col=0,header=0)
    gen_train, gen_val = df_train_test_split(gen_train,val_split=0.01,seed=42)

    train_sampler = weighted_sampler(df=gen_train, column='peptide', replacement=True, alpha=0.5)
    val_sampler = weighted_sampler(df=gen_val, column='peptide', replacement=False, alpha=0.5)

    gen_train_loader = DataLoader(
        MPTGenDataSet(mpt_df=gen_train, mhc_type='HLAI', 
                mhc_max_len=34, pep_max_len=15, cdr3_max_len=20,
                bv=True, pos=True, real=True,
                distillation=False,
                pep_mask=None, cdr3_mask=0.5),
                batch_size=128, shuffle=False, sampler=train_sampler)

    gen_val_loader = build_loader_uniform_by_peptide(   
            MPTGenDataSet(mpt_df=gen_val, mhc_type='HLAI', 
                mhc_max_len=34, pep_max_len=15, cdr3_max_len=20,
                bv=True, pos=True, real=True,
                distillation=False,
                pep_mask=None, cdr3_mask=None),
            peptide_ids = gen_val.peptide, 
            peptides_per_step=64, 
            samples_per_peptide=1, 
            seed=42,
            num_workers=0, pin_memory=False,
            ensure_full_batch=False)

    generator = PepBridgeGenerator( aa_size=len(aa_dict),
            max_len_dict={"mhc":34, "peptide":15, "cdr3":20},
            d_seq=128, d_head_seq=32,
            d_pair=64, d_head_pair=32,
            dropout=0.1,
            n_layers_dict={"mhc":3, "peptide":6, "cdr3":6,
                            "mp":3, "pt":3, "mpt":1, 
                            'gen':12},
            trbv_size=len(bv_dict), film=True, lora=True).to(device)

    discriminator = Discriminator(aa_size=len(aa_dict),bv_size=len(bv_dict),
                                d_emb=64, d_hidden=128, dropout=0.1, tau=0.3).to(device)

    pepbridge_pama = torch.load(pepbridge_dir, map_location=device)
    state_dict = pepbridge_pama['model_state']
    keys_to_drop = [
        "mp_contact_pred_head.mlp.3.weight",
        "mp_contact_pred_head.mlp.3.bias",
    ]

    for k in keys_to_drop:
        if k in state_dict:
            # print(f"Drop key from checkpoint: {k}")
            state_dict.pop(k)

    msg = generator.pepbridge.load_state_dict(state_dict, strict=False)
    logger.info(msg)

    for p in generator.pepbridge.parameters():
        p.requires_grad = False

    if use_lora:
        inject_lora_into_trunk_last_n(
            generator.pepbridge.mpt_pair_aware_trunk,
            last_n=1,
            cfg_seq_pair=((8, 16), (4, 8)),
            dropout=0.1,
            freeze_base=False, 
        )
        freeze_module_except_lora(generator.pepbridge.mpt_pair_aware_trunk)
        logger.info("[configure_pepbridge] mode = LoRA, only LoRA in mpt_pair_aware_trunk is trainable")
    else:
        if finetune_trunk:
            for p in generator.pepbridge.mpt_pair_aware_trunk.parameters():
                p.requires_grad = True
            logger.info("[configure_pepbridge] mode = mpt_pair_aware_trunk are trainable (no LoRA)")
        else:
            logger.info("[configure_pepbridge] mode = all frozen (pepbridge as frozen encoder)")


    generator_train(
        generator=generator,
        discriminator=discriminator,
        loader=gen_train_loader,  
        device = device,
        save_dir=save_dir,
        epochs=20,
        steps_per_epoch=1000, 
        optimizer_ctor=lambda params: torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01),
        grad_accum_steps=1,
        amp=True,
        log_interval = 50,
        val_loader = gen_val_loader,
        eval_every_epochs= 1,
        val_max_steps=200,
        logger=logger,
        distillation=False,
        lambda_bv=0.2,
        lambda_adv=0.1,
        lambda_margin=1.0,
        lambda_base=0.5)

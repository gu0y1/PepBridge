from script.model.pepbridge import PepBridge
from script.model.lora import build_model_with_lora
from script.dataset import PTDataSet, MPDataSet, MPTDataSet, \
    build_loader_for_long_tail, build_loader_uniform_by_peptide
from script.dataprocess import mk_aa_dict, mk_bv_dict
from script.utils import model_fn, encoder_load_state_dict, df_train_test_split, setup_logger
from script.train import train_three_phases_multi_loaders

import pandas as pd 
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

os.chdir('/data5/tem/laiwp131/pMHC_TCR_20251103/')

aa_dict = mk_aa_dict()
bv_dcit = mk_bv_dict()

device = 'cuda'

model = model_fn(aa_vocab_size=len(aa_dict),
                trbv_vocab_size=len(bv_dcit))
model.to(device)
model = encoder_load_state_dict(model, peptide_pt_path='peptide_mlm.pt',
                                cdr3_pt_path='cdr3_mlm.pt', device=device)

pt_df = pd.read_csv('pt_train.csv', header=0, index_col=0)
pt_train, pt_val = df_train_test_split(pt_df, val_split=0.2, seed=42)

mp_df = pd.read_csv('mp_train.csv', header=0, index_col=0)
mp_train, mp_val = df_train_test_split(mp_df, val_split=0.2, seed=42)

mpt_df = pd.read_csv('mpt_train.csv', header=0, index_col=0)
mpt_train, mpt_val = df_train_test_split(mpt_df, val_split=0.2, seed=42)

align_df = pd.read_csv('random_pMHC_cdr3_train.csv', header=0, index_col=0)
align_train, align_val = df_train_test_split(align_df, val_split=0.2, seed=42)
# align_val = align_val.sample(n=100000, random_state=42)
del align_df, pt_df, mp_df, mpt_df

imm_df = pd.read_csv('immunogenicity_train.csv', header=0, index_col=0)
imm_train, imm_val = df_train_test_split(imm_df, val_split=0.2, seed=42)

mp_contact_df = pd.read_csv('./PepBridge-main/data/mp_pdb_train.csv',  header=0, index_col=0)
mp_contact_train, mp_contact_val = df_train_test_split(mp_contact_df, val_split=0.2, seed=42)

pt_contact_df = pd.read_csv('./PepBridge-main/data/pt_pdb_train.csv', header=0, index_col=0)
pt_contact_train, pt_contact_val = df_train_test_split(pt_contact_df, val_split=0.2, seed=42)

def weighted_sampler(df, column, replacement):
    lengths = np.array(df[column])
    group_ids = lengths.copy()
    group_ids[group_ids >= 14] = 14
    unique_groups, counts = np.unique(group_ids, return_counts=True)
    group_weights = {g: 1.0 / c for g, c in zip(unique_groups, counts)}
    sample_weights = np.array([group_weights[g] for g in group_ids], dtype=np.float32)
    sample_weights_t = torch.from_numpy(sample_weights) 
    sampler = WeightedRandomSampler(
        weights=sample_weights_t,
        num_samples=len(sample_weights_t), 
        replacement=replacement                   
    )
    return sampler
  
mp_train_sampler = weighted_sampler(df=mp_train, column='len', replacement=True)
mp_val_sampler = weighted_sampler(df=mp_val, column='len', replacement=False)

mp_train_loader = DataLoader(MPDataSet(mp_df=mp_train, mhc_type='HLAI', 
                       mhc_max_len=34, pep_max_len=15,
                        binding=True, 
                        immunogenicity=False, contact=False, mask=None),
                        batch_size=128, shuffle=False,sampler=mp_train_sampler)

mp_val_loader = DataLoader(MPDataSet(mp_df=mp_val, mhc_type='HLAI', 
                        mhc_max_len=34, pep_max_len=15,
                            binding=True, 
                            immunogenicity=False, contact=False, mask=None),
                            batch_size=128, shuffle=False, sampler=mp_val_sampler)

imm_train_loader = DataLoader(MPDataSet(mp_df=imm_train, mhc_type='HLAI', 
                       mhc_max_len=34, pep_max_len=15,
                        binding=False, 
                        immunogenicity=True, contact=False, mask=None),
                        batch_size=32, shuffle=True)

imm_val_loader = DataLoader(MPDataSet(mp_df=imm_val, mhc_type='HLAI', 
                       mhc_max_len=34, pep_max_len=15,
                        binding=False, 
                        immunogenicity=True, contact=False, mask=None),
                        batch_size=32, shuffle=True)

mp_contact_train_loader = DataLoader(MPDataSet(mp_df=mp_contact_train, mhc_type='HLAI', 
                       mhc_max_len=34, pep_max_len=15,
                        binding=False, 
                        immunogenicity=False, contact=True, mask=None),
                        batch_size=8, shuffle=True)

mp_contact_val_loader = DataLoader(MPDataSet(mp_df=mp_contact_val, mhc_type='HLAI', 
                       mhc_max_len=34, pep_max_len=15,
                        binding=False, 
                        immunogenicity=False, contact=True, mask=None),
                        batch_size=4, shuffle=True)

pt_train_loader =  DataLoader(
    PTDataSet(pt_train, pep_max_len=15, 
                      cdr3_max_len=20,
                    binding=True, contact=False, 
                    pep_mask=0.5, cdr3_mask=0.5),
    batch_size=128, shuffle=True)

pt_val_loader = build_loader_uniform_by_peptide(
    dataset=PTDataSet(pt_val, pep_max_len=15, 
                      cdr3_max_len=20,
                    binding=True, contact=False, 
                    pep_mask=None, cdr3_mask=None),
    peptide_ids = pt_val.peptide, 
    peptides_per_step=64, 
    samples_per_peptide=2, 
    seed=42,
    num_workers=0, pin_memory=False,
    ensure_full_batch=False
)

mpt_train_loader= DataLoader(
    MPTDataSet(mpt_train, mhc_type='HLAI', 
               mhc_max_len=34, pep_max_len=15, cdr3_max_len=20,
               bv=True, binding=True, 
               pep_mask=0.5, cdr3_mask=0.5),
 batch_size=128, shuffle=True)

mpt_val_loader = build_loader_uniform_by_peptide(
    dataset=MPTDataSet(mpt_val, mhc_type='HLAI', 
               mhc_max_len=34, pep_max_len=15, cdr3_max_len=20,
               bv=True, binding=True, 
               pep_mask=None, cdr3_mask=None),
    peptide_ids = mpt_val.peptide, 
    peptides_per_step=64, 
    samples_per_peptide=2, 
    seed=42,
    num_workers=0, pin_memory=False,
    ensure_full_batch=False
)

pt_contact_train_loader = DataLoader(PTDataSet(pt_contact_train, 
                                pep_max_len=15, cdr3_max_len=20,
                                binding=False, contact=True, 
                                pep_mask=None, cdr3_mask=None),
                                batch_size=4, shuffle=True)

pt_contact_val_loader = DataLoader(PTDataSet(pt_contact_val, 
                                pep_max_len=15, cdr3_max_len=20,
                                binding=False, contact=True, 
                                pep_mask=None, cdr3_mask=None),
                                batch_size=2, shuffle=True)

align_train_loader = DataLoader(MPTDataSet(align_train, mhc_type='HLAI', 
               mhc_max_len=34, pep_max_len=15, cdr3_max_len=20,
               bv=False, binding=False, 
               pep_mask=None, cdr3_mask=None),
               batch_size=64, shuffle=True)

align_val_loader = DataLoader(MPTDataSet(align_val, mhc_type='HLAI', 
               mhc_max_len=34, pep_max_len=15, cdr3_max_len=20,
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
logger = setup_logger()
train_three_phases_multi_loaders(  
    model=model,
    loaders=train_loaders,
    device="cuda",
    save_dir="./checkpoints_multi_lora_align_all",
    epochs_A=10, epochs_B=6, epochs_C=3,
    steps_per_epoch=1200,
    optimizer_ctor=lambda params: torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01),
    grad_accum_steps=1,
    amp=True,
    new_optimizer_each_phase=False,
    log_interval=50,
    task_every = {"mp_contact": 50, "pt_contact": 50},   #
    val_loaders= val_loaders,
    eval_every_epochs=1,
    pep_align=True,
    all_align=-3,
    use_lora=True,
    last_n=2,
    cfg_seq_pair=((8,16),(4,8)),
    logger=logger)
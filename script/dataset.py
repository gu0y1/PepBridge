import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import os

from .dataprocess import mk_aa_dict, mk_bv_dict, load_mhc_dict,\
    aa_to_vec, pad_1d, pad_2d, get_masked_sample, mhc_to_aa, mhc_to_esm

class MaskedLMDataSet(Dataset):
  def __init__(self, seq_list, max_len, 
               masked_rate=0.15, contiguous_prob=0.50):
    self.list = seq_list
    self.max_len = max_len
    self.aa_dict = mk_aa_dict()
    self.masked_rate = masked_rate
    self.contiguous_prob = contiguous_prob

  def __len__(self):
    return len(self.list)
  
  def __getitem__(self, idx):
    aa_seq_vec = aa_to_vec(self.list[idx], self.aa_dict)
    mlm_input_tokens_id, mlm_label = get_masked_sample(aa_seq_vec, self.aa_dict, 
                                          self.masked_rate, self.contiguous_prob)
    mlm_input_tokens_id = pad_1d(mlm_input_tokens_id, self.max_len)
    mlm_label = pad_1d(mlm_label, self.max_len)
    out = {}
    out['mlm_input'] = torch.as_tensor(mlm_input_tokens_id, dtype=torch.long)
    out['mlm_label'] = torch.as_tensor(mlm_label, dtype=torch.long)
    return out, idx

class MPDataSet(Dataset):
  def __init__(self, mp_df, mhc_type, mhc_max_len, pep_max_len,
               binding=False, immunogenicity=False, contact=False,
               mask=None):
    self.df = mp_df
    self.mhc_type = mhc_type
    self.mhc_max_len = mhc_max_len
    self.pep_max_len = pep_max_len
    self.binding = binding
    self.imm = immunogenicity
    self.contact = contact
    self.mask =  mask
    self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    self.aa_dict = mk_aa_dict()
    self.pseudo_mhc_dict = load_mhc_dict(mhc_type, pseudo=True)
    self.esm_mhc_dict = load_mhc_dict(mhc_type, pseudo=False)
    
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    mhc_name = row['MHC']
    pep_seq  = row['peptide']

    mhc_seq = mhc_to_aa(mhc_name, self.pseudo_mhc_dict)        # str
    mhc_ids = aa_to_vec(mhc_seq, self.aa_dict)                 # 1D ids
    mhc_ids = pad_1d(mhc_ids, self.mhc_max_len, pad_value=0, dtype=int)

    esm_mhc = mhc_to_esm(mhc_name, self.esm_mhc_dict) 

    pep_ids = aa_to_vec(pep_seq, self.aa_dict)
    if self.mask is not None and random.random() < self.mask:
      pep_ids,_ = get_masked_sample(pep_ids, self.aa_dict, 0.15, 0)
    pep_ids = pad_1d(pep_ids, self.pep_max_len, pad_value=0, dtype=int)

    out = {
        'mhc': torch.as_tensor(mhc_ids, dtype=torch.long),
        'peptide': torch.as_tensor(pep_ids, dtype=torch.long),
        'esm_mhc': torch.as_tensor(esm_mhc, dtype=torch.float32)
    }

    if self.binding:
        out['y_mp'] = torch.tensor(float(row['binding']), dtype=torch.float32)
    if self.imm:
        out['y_imm'] = torch.tensor(float(row['immunogenicity']), dtype=torch.float32)

    if self.contact:
      pdb_chains  = row['pdb_chains']
      csv_path = os.path.join(self.project_root, 'data', 'mp', f'{pdb_chains}_pesudo.csv')

      contact_df = pd.read_csv(csv_path, header=0, index_col=0)
      contact_mhc_seq = "".join(contact_df.index.astype(str))
      contact_pep_seq = "".join(contact_df.columns.astype(str))

      contact_dist = contact_df.to_numpy(dtype=float)

      if not (mhc_seq == contact_mhc_seq and pep_seq == contact_pep_seq):
          if (mhc_seq == contact_pep_seq and pep_seq == contact_mhc_seq):
              contact_dist = contact_dist.T
              contact_mhc_seq, contact_pep_seq = contact_pep_seq, contact_mhc_seq
          else:
              raise ValueError(
                  f"Contact labels mismatch: "
                  f"mhc_csv='{contact_mhc_seq}', pep_csv='{contact_pep_seq}', "
                  f"mhc='{mhc_seq}', pep='{pep_seq}'"
              )
      
      contact_prob = (contact_dist < 4.0).astype(np.float32)

      if contact_dist.ndim != 2 or contact_prob.ndim != 2:
          raise ValueError(f'contact matrix must be 2D')
      
      contact_dist = pad_2d(contact_dist, self.mhc_max_len, self.pep_max_len,
                            pad_value=0.0, dtype=float)
      contact_prob = pad_2d(contact_prob, self.mhc_max_len, self.pep_max_len,
                          pad_value=0.0, dtype=float)
      out['contact_mp_dist'] = torch.as_tensor(contact_dist, dtype=torch.float32)
      out['contact_mp_bin'] = torch.as_tensor(contact_prob, dtype=torch.float32)
    return out, idx
  
class PTDataSet(Dataset):
  def __init__(self, pt_df, pep_max_len, cdr3_max_len,
               binding=False, contact=False, mask=None):
    self.df = pt_df
    self.cdr3_max_len = cdr3_max_len
    self.pep_max_len = pep_max_len
    self.binding = binding
    self.contact = contact
    self.mask = mask
    self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    self.aa_dict = mk_aa_dict()

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    pep_seq  = row['peptide']
    cdr3_seq = row['cdr3']

    pep_ids = aa_to_vec(pep_seq, self.aa_dict)
    if self.mask is not None and random.random() < self.mask:
      pep_ids,_ = get_masked_sample(pep_ids, self.aa_dict, 0.15, 0)
    pep_ids = pad_1d(pep_ids, self.pep_max_len, pad_value=0, dtype=int)

    cdr3_ids = aa_to_vec(cdr3_seq, self.aa_dict)
    if self.mask is not None and random.random() < self.mask:
      cdr3_ids,_ = get_masked_sample(cdr3_ids, self.aa_dict, 0.15, 0)
    cdr3_ids = pad_1d(cdr3_ids, self.cdr3_max_len, pad_value=0, dtype=int)
    
    out = {
        'peptide': torch.as_tensor(pep_ids, dtype=torch.long),
        'cdr3': torch.as_tensor(cdr3_ids, dtype=torch.long),
    }

    if self.binding:
      out['y_pt'] = torch.tensor(float(row['binding']), dtype=torch.float32)

    if self.contact:
      pdb_chains  = row['pdb_chains']
      csv_path = os.path.join(self.project_root, 'data', 'pt', f'{pdb_chains}.csv')

      contact_df = pd.read_csv(csv_path, header=0, index_col=0)
      contact_pep_seq = "".join(contact_df.index.astype(str))
      contact_cdr3_seq = "".join(contact_df.columns.astype(str))

      contact_dist = contact_df.to_numpy(dtype=float)

      if not (cdr3_seq == contact_cdr3_seq and pep_seq == contact_pep_seq):
          if (cdr3_seq == contact_pep_seq and pep_seq == contact_cdr3_seq):
              contact_dist = contact_dist.T
              contact_cdr3_seq, contact_pep_seq = contact_pep_seq, contact_cdr3_seq
          else:
              raise ValueError(
                  f"Contact labels mismatch: "
                  f"cdr3_csv='{contact_cdr3_seq}', pep_csv='{contact_pep_seq}', "
                  f"cdr3='{cdr3_seq}', pep='{pep_seq}'"
              )

      contact_prob = (contact_dist < 5.0).astype(np.float32)

      if contact_dist.ndim != 2 or contact_prob.ndim != 2:
        raise ValueError(f'contact matrix must be 2D')
      
      contact_dist = pad_2d(contact_dist, self.pep_max_len, self.cdr3_max_len, 
                            pad_value=0.0, dtype=float)
      contact_prob = pad_2d(contact_prob,  self.pep_max_len, self.cdr3_max_len,
                          pad_value=0.0, dtype=float)
      out['contact_pt_dist'] = torch.as_tensor(contact_dist, dtype=torch.float32)
      out['contact_pt_bin'] = torch.as_tensor(contact_prob, dtype=torch.float32)
    return out, idx  

class MPTDataSet(Dataset):
  def __init__(self, mpt_df, mhc_type, 
               mhc_max_len, pep_max_len, cdr3_max_len,
               bv=False, binding=False, mask=None):
    self.df = mpt_df
    self.mhc_type = mhc_type

    self.mhc_max_len = mhc_max_len
    self.cdr3_max_len = cdr3_max_len
    self.pep_max_len = pep_max_len

    self.binding = binding
    self.bv = bv
    self.mask = mask

    self.aa_dict = mk_aa_dict()
    self.bv_dict = mk_bv_dict()
    self.pseudo_mhc_dict = load_mhc_dict(mhc_type, pseudo=True)
    self.esm_mhc_dict = load_mhc_dict(mhc_type, pseudo=False)

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    mhc_name = row['MHC']
    pep_seq  = row['peptide']
    cdr3_seq = row['cdr3']

    mhc_seq = mhc_to_aa(mhc_name, self.pseudo_mhc_dict)        # str
    mhc_ids = aa_to_vec(mhc_seq, self.aa_dict)                 # 1D ids
    mhc_ids = pad_1d(mhc_ids, self.mhc_max_len, pad_value=0, dtype=int)

    esm_mhc = mhc_to_esm(mhc_name, self.esm_mhc_dict) 

    pep_ids = aa_to_vec(pep_seq, self.aa_dict)
    if self.mask is not None and random.random() < self.mask:
      pep_ids,_ = get_masked_sample(pep_ids, self.aa_dict, 0.15, 0)
    pep_ids = pad_1d(pep_ids, self.pep_max_len, pad_value=0, dtype=int)

    cdr3_ids = aa_to_vec(cdr3_seq, self.aa_dict)
    if self.mask is not None and random.random() < self.mask:
      cdr3_ids,_ = get_masked_sample(cdr3_ids, self.aa_dict, 0.15, 0)
    cdr3_ids = pad_1d(cdr3_ids, self.cdr3_max_len, pad_value=0, dtype=int)

    out = {
       'mhc': torch.as_tensor(mhc_ids, dtype=torch.long),
        'peptide': torch.as_tensor(pep_ids, dtype=torch.long),
        'cdr3': torch.as_tensor(cdr3_ids, dtype=torch.long),
        'esm_mhc': torch.as_tensor(esm_mhc, dtype=torch.float32)
    }

    if self.bv:
        bv_name = row['v_gene']
        out['trbv'] = torch.tensor(self.bv_dict.get(bv_name, 0), dtype=torch.long)
        
    if self.binding:
        out['y_mpt'] = torch.tensor(float(row['binding']), dtype=torch.float32)

    return out, idx  
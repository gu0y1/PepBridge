import torch
from torch.utils.data import Dataset, Sampler, DataLoader

import numpy as np

import random
import os
import math
from collections import defaultdict, Counter, deque
from typing import List, Sequence, Hashable, Optional

from .dataprocess import mk_aa_dict, mk_bv_dict, load_mhc_dict,\
    aa_to_vec, pad_1d, pad_2d, get_masked_sample, mhc_to_aa, mhc_to_esm
from .utils import read_csv_with_index_allow_duplicate_names

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
    self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    self.aa_dict = mk_aa_dict()
    self.pseudo_mhc_dict = load_mhc_dict(mhc_type, pseudo=True)
    self.esm_mhc_dict = load_mhc_dict(mhc_type, pseudo=False)
    
    self.shape = mp_df.shape
    
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
      pep_ids,_ = get_masked_sample(pep_ids, self.aa_dict, 0.10, 0)
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
      csv_path = os.path.join(self.project_root, 'data', 'mp', f'{pdb_chains}.csv')

      contact_df = read_csv_with_index_allow_duplicate_names(csv_path)
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
               binding=False, contact=False, 
               pep_mask=None, cdr3_mask=None):
    self.df = pt_df
    self.cdr3_max_len = cdr3_max_len
    self.pep_max_len = pep_max_len
    self.binding = binding
    self.contact = contact

    self.pep_mask = pep_mask
    self.cdr3_mask = cdr3_mask

    self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    self.aa_dict = mk_aa_dict()
    
    self.shape = pt_df.shape
    
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    pep_seq  = row['peptide']
    cdr3_seq = row['cdr3']

    pep_ids = aa_to_vec(pep_seq, self.aa_dict)
    if self.pep_mask is not None and random.random() < self.pep_mask:
      pep_ids,_ = get_masked_sample(pep_ids, self.aa_dict, 0.10, 0)
    pep_ids = pad_1d(pep_ids, self.pep_max_len, pad_value=0, dtype=int)

    cdr3_ids = aa_to_vec(cdr3_seq, self.aa_dict)
    if self.cdr3_mask is not None and random.random() < self.cdr3_mask:
      cdr3_ids,_ = get_masked_sample(cdr3_ids, self.aa_dict, 0.10, 0)
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

      contact_df = read_csv_with_index_allow_duplicate_names(csv_path)
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

class MultiNegPairPTDataset(Dataset):
    def __init__(self, pt_df, pep_max_len, cdr3_max_len,
                 hard_neg_map,
                 k_cross=8, k_hard=2,
                 pep_mask=None, cdr3_mask=None,
                avoid_duplicates=True):
        self.df = pt_df.reset_index(drop=True)
        self.pep_max_len = int(pep_max_len)
        self.cdr3_max_len = int(cdr3_max_len)
        self.k_cross = int(k_cross)
        self.k_hard = int(k_hard)
        self.k_total = self.k_cross + self.k_hard
        self.pep_mask = pep_mask
        self.cdr3_mask = cdr3_mask
        self.avoid_duplicates = bool(avoid_duplicates)

        self.hard_neg_map = {str(k): list(v) for k, v in dict(hard_neg_map).items()}

        self.aa_dict = mk_aa_dict()

        pos_by_pep = defaultdict(set)
        for _, row in self.df.iterrows():
            pep = row['peptide']
            cpos = row['cdr3']
            if isinstance(pep, str) and isinstance(cpos, str) and pep and cpos:
                pos_by_pep[pep].add(cpos)
        all_pos = set()
        for s in pos_by_pep.values(): all_pos |= s

        cross_pool_by_pep = {}
        for pep, s in pos_by_pep.items():
            cross_pool_by_pep[pep] = list(all_pos - s)

        self.pos_by_pep = {k: list(v) for k, v in pos_by_pep.items()}
        self.cross_pool_by_pep = cross_pool_by_pep
        self.global_pos_pool = list(all_pos)

    def __len__(self):
        return len(self.df)

    # --- utils ---
    def _encode_seq(self, seq, max_len, mask_prob=None):
        ids = aa_to_vec(seq, self.aa_dict)
        if (mask_prob is not None) and (random.random() < mask_prob):
            ids, _ = get_masked_sample(ids, self.aa_dict, 0.10, 0)
        ids = pad_1d(ids, max_len, pad_value=0, dtype=int)
        return torch.as_tensor(ids, dtype=torch.long)

    def _sample_unique_or_choices(self, pool_list, k, ban_set=None):
        if ban_set:
            pool = [x for x in pool_list if x not in ban_set]
        else:
            pool = list(pool_list)

        picked = []
        if self.avoid_duplicates and len(pool) >= k:
            picked = random.sample(pool, k)
        else:
            if len(pool) == 0:
                return picked
            while len(picked) < k:
                picked.append(random.choice(pool))
        return picked[:k]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pep = row['peptide']
        cdr3_pos_seq = row['cdr3']

        cross_pool = self.cross_pool_by_pep.get(pep, self.global_pos_pool)
        ban = set(self.pos_by_pep.get(pep, [])) | {cdr3_pos_seq}
        neg_cross = self._sample_unique_or_choices(cross_pool, self.k_cross, ban_set=ban)

        key = pep + "||" + cdr3_pos_seq
        hard_list = self.hard_neg_map.get(key, [])
        neg_hard = self._sample_unique_or_choices(hard_list, self.k_hard, ban_set=ban)

        neg_list = list(neg_cross) + list(neg_hard)
        if len(neg_list) < self.k_total:
            fill = self._sample_unique_or_choices(cross_pool, self.k_total - len(neg_list), ban_set=set(neg_list) | ban)
            neg_list.extend(fill)

        pep_ids = self._encode_seq(pep, self.pep_max_len, mask_prob=self.pep_mask)
        pos_ids = self._encode_seq(cdr3_pos_seq, self.cdr3_max_len, mask_prob=self.cdr3_mask)
        neg_ids = [self._encode_seq(s, self.cdr3_max_len, mask_prob=self.cdr3_mask) for s in neg_list]
        neg_ids = torch.stack(neg_ids, dim=0)  # [K_total, Lc]

        out = {
            'peptide': pep_ids,           # [Lp]
            'cdr3_pos': pos_ids,          # [Lc]
            'cdr3_negs': neg_ids,         # [K_total, Lc]， K_total = k_cross + k_hard
        }
        return out, idx

class MPTDataSet(Dataset):
  def __init__(self, mpt_df, mhc_type, 
               mhc_max_len, pep_max_len, cdr3_max_len,
               bv=False, binding=False, 
               pep_mask=0.1, cdr3_mask=0.1):
    self.df = mpt_df
    self.mhc_type = mhc_type

    self.mhc_max_len = mhc_max_len
    self.cdr3_max_len = cdr3_max_len
    self.pep_max_len = pep_max_len

    self.binding = binding
    self.bv = bv

    self.pep_mask = pep_mask
    self.cdr3_mask = cdr3_mask

    self.aa_dict = mk_aa_dict()
    self.bv_dict = mk_bv_dict()
    self.pseudo_mhc_dict = load_mhc_dict(mhc_type, pseudo=True)
    self.esm_mhc_dict = load_mhc_dict(mhc_type, pseudo=False)

    self.shape = mpt_df.shape

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
    if self.pep_mask is not None and random.random() < self.pep_mask:
      pep_ids,_ = get_masked_sample(pep_ids, self.aa_dict, 0.10, 0)
    pep_ids = pad_1d(pep_ids, self.pep_max_len, pad_value=0, dtype=int)

    cdr3_ids = aa_to_vec(cdr3_seq, self.aa_dict)
    if self.cdr3_mask is not None and random.random() < self.cdr3_mask:
      cdr3_ids,_ = get_masked_sample(cdr3_ids, self.aa_dict, 0.10, 0)
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

class MPTGenDataSet(Dataset):
  def __init__(self, mpt_df, mhc_type, 
               mhc_max_len, pep_max_len, cdr3_max_len,
               bv=False, pos=False, real=False,
               distillation=False,
               pep_mask=0.1, cdr3_mask=0.1):
    self.df = mpt_df
    self.mhc_type = mhc_type

    self.mhc_max_len = mhc_max_len
    self.cdr3_max_len = cdr3_max_len
    self.pep_max_len = pep_max_len

    self.pos = pos
    self.real = real
    self.bv = bv
    self.distillation = distillation

    self.pep_mask = pep_mask
    self.cdr3_mask = cdr3_mask

    self.aa_dict = mk_aa_dict()
    self.bv_dict = mk_bv_dict()
    self.pseudo_mhc_dict = load_mhc_dict(mhc_type, pseudo=True)
    self.esm_mhc_dict = load_mhc_dict(mhc_type, pseudo=False)

    self.shape = mpt_df.shape

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    mhc_name = row['MHC']
    pep_seq  = row['peptide']
    cdr3_seq = row['cdr3_neg']

    mhc_seq = mhc_to_aa(mhc_name, self.pseudo_mhc_dict)        # str
    mhc_ids = aa_to_vec(mhc_seq, self.aa_dict)                 # 1D ids
    mhc_ids = pad_1d(mhc_ids, self.mhc_max_len, pad_value=0, dtype=int)

    esm_mhc = mhc_to_esm(mhc_name, self.esm_mhc_dict) 

    pep_ids = aa_to_vec(pep_seq, self.aa_dict)
    if self.pep_mask is not None and random.random() < self.pep_mask:
      pep_ids,_ = get_masked_sample(pep_ids, self.aa_dict, 0.10, 0)
    pep_ids = pad_1d(pep_ids, self.pep_max_len, pad_value=0, dtype=int)

    cdr3_ids = aa_to_vec(cdr3_seq, self.aa_dict)
    if self.cdr3_mask is not None and random.random() < self.cdr3_mask:
      cdr3_ids,_ = get_masked_sample(cdr3_ids, self.aa_dict, 0.10, 0)
    cdr3_ids = pad_1d(cdr3_ids, self.cdr3_max_len, pad_value=0, dtype=int)

    out = {
       'mhc': torch.as_tensor(mhc_ids, dtype=torch.long),
        'peptide': torch.as_tensor(pep_ids, dtype=torch.long),
        'neg_cdr3': torch.as_tensor(cdr3_ids, dtype=torch.long),
        'esm_mhc': torch.as_tensor(esm_mhc, dtype=torch.float32)
    }

    if self.pos:
        pos_cdr3 = row['cdr3_pos']
        pos_cdr3 = aa_to_vec(pos_cdr3, self.aa_dict)
        pos_cdr3 = pad_1d(pos_cdr3, self.cdr3_max_len, pad_value=0, dtype=int)
        out['pos_cdr3'] = torch.tensor(pos_cdr3, dtype=torch.long)
        if self.bv:
            bv_name = row['bv_pos']
            out['pos_bv'] = torch.tensor(self.bv_dict.get(bv_name, 0), 
                                         dtype=torch.long)

    if self.real:
        real_cdr3 = row['cdr3_real']
        real_cdr3 = aa_to_vec(real_cdr3, self.aa_dict)
        real_cdr3 = pad_1d(real_cdr3, self.cdr3_max_len, pad_value=0, dtype=int)
        out['real_cdr3'] = torch.tensor(real_cdr3, dtype=torch.long)
        if self.bv:
            bv_name = row['bv_real']
            out['real_bv'] = torch.tensor(self.bv_dict.get(bv_name, 0), 
                                         dtype=torch.long)
            
    if self.distillation:
        out['score'] = torch.tensor(float(row['score']), 
                                    dtype=torch.float32)

    return out, idx  

def rfs_repeat_factors(counts, t=1e-3):
  freqs = counts / counts.sum()
  return np.maximum(1.0, np.sqrt(t / np.clip(freqs, 1e-12, None)))

def mixed_group_probs_from_rfs(rfs, alpha=1, mix=0.1):
  p = rfs**alpha
  p = p / p.sum()
  mix = 0.10                         
  p = (1 - mix) * p + mix * (1.0 / len(p))
  return p 

class QuotaGroupedBatchSampler(Sampler[List[int]]):
    """
    - epoch 级：按给定组概率 p_g -> 整数配额 q_g（随机四舍五入），生成无放回的“组日程表”
    - batch 级：每批选多个不同组；组内不放回最多取 max_per_group 个样本
    - 防记忆化：sample_repeat_cap 限制每样本在一个 epoch 内的出现次数
    - 健壮性：绝不产出空 batch；当某组被 cap 用尽时自动跳过并回填
    """
    def __init__(self,
                 peptide_ids: Sequence[Hashable],
                 probs: np.ndarray,
                 batch_size: int,
                 max_per_group: int = 2,
                 num_batches: Optional[int] = None,
                 sample_repeat_cap: Optional[int] = 3,
                 seed: Optional[int] = None):
        self.batch_size = int(batch_size)
        self.max_per_group = int(max_per_group)
        self.sample_repeat_cap = sample_repeat_cap
        self.rng = random.Random(seed)

        # 分桶：gid -> indices（与 dataset 索引对齐）
        groups = defaultdict(list)
        for idx, gid in enumerate(peptide_ids):
            groups[int(gid)].append(idx)   # 强制 int，避免 np.int64 做 dict key 时混淆
        self.group_keys = list(groups.keys())               # [0..P-1]
        self.groups = {g: list(v) for g, v in groups.items()}
        self.P = len(self.group_keys)

        N = len(peptide_ids)
        if num_batches is None:
            num_batches = math.ceil(2 * N / self.batch_size)
        self.num_batches = int(num_batches)
        # 每批最多可容纳的“不同组”的个数
        self.groups_per_batch = max(1, self.batch_size // self.max_per_group)

        # 归一化组概率
        p = np.asarray(probs, dtype=np.float64)
        self.p = p / p.sum()

    def __len__(self):
        return self.num_batches

    def _make_epoch_schedule(self,
                            u_share: float = 0.20,     # 20% 均匀地板
                            cap_mult: float = 2.0):    # 最高不超过 2× 平均
        P = len(self.group_keys)
        T = self.num_batches * self.groups_per_batch
        p = self.p / self.p.sum()

        # --- (a) 均匀“地板” ---
        Tu = int(round(u_share * T))
        q = np.zeros(P, dtype=int)
        q_per = Tu // P
        q[:] = q_per
        rem = Tu - q_per * P
        if rem > 0:                               # 把剩余的 +1 随机分到 rem 个组
            add_idx = np.random.choice(P, size=rem, replace=False)
            q[add_idx] += 1

        # --- (b) 概率配额（剩余部分） + 无偏随机四舍五入 ---
        Tb = T - q.sum()
        if Tb > 0:
            exp_q = Tb * p
            base = np.floor(exp_q).astype(int)
            need = Tb - base.sum()
            if need > 0:
                probs = exp_q - base
                if probs.sum() > 0:
                    take = np.random.choice(P, size=need, replace=False, p=probs/probs.sum())
                else:
                    take = np.random.choice(P, size=need, replace=False)
                base[take] += 1
            q += base

        # --- (c) 天花板 + 水位填充再分配 ---
        cap = int(np.ceil(cap_mult * T / P))
        overflow = (q - cap).clip(min=0)
        q[q > cap] = cap
        leftover = int(overflow.sum())
        if leftover > 0:
            # 仅在未达 cap 的组中按 p 再分配
            mask = (q < cap)
            while leftover > 0 and mask.any():
                alloc = np.minimum(leftover, mask.sum())
                probs = np.where(mask, p, 0.0)
                probs_sum = probs.sum()
                if probs_sum == 0:
                    idx = np.random.choice(np.where(mask)[0], size=alloc, replace=False)
                else:
                    idx = np.random.choice(P, size=alloc, replace=False, p=probs/probs_sum)
                q[idx] += 1
                leftover -= alloc
                mask = (q < cap)

        # --- (d) 展开为 schedule 并打乱 ---
        schedule = []
        for i, qi in enumerate(q):
            if qi > 0:
                schedule.extend([i] * int(qi))
        self.rng.shuffle(schedule)

        # 可选：存调试信息
        self.last_quota_vec = q.copy()
        self.last_T = int(T)
        return schedule
    

    def _build_group_queues(self):
        q = {}
        for gi, g in enumerate(self.group_keys):
            pool = list(self.groups[g])
            self.rng.shuffle(pool)
            q[gi] = deque(pool)
        return q

    def __iter__(self):
        schedule = self._make_epoch_schedule()
        used = Counter()

        P = len(self.group_keys)
        # 每组剩余“样本预算”（cap * 组内样本数）；None 表示无限
        if self.sample_repeat_cap is None:
            group_remaining = np.full(P, np.inf, dtype=float)
        else:
            group_remaining = np.array([
                int(self.sample_repeat_cap) * len(self.groups[self.group_keys[gi]])
                for gi in range(P)
            ], dtype=float)

        # 每组还可被“命中”的配额（来自 schedule 的整数配额，防止某组过多命中）
        if hasattr(self, "last_quota_vec"):
            hit_quota = self.last_quota_vec.astype(int).copy()
        else:
            # 没存的话，就当无限配额（不推荐，但兼容）
            hit_quota = np.full(P, np.iinfo(np.int32).max, dtype=int)

        hit_count = np.zeros(P, dtype=int)  # 已命中次数（用于不超过配额）
        queues = self._build_group_queues()
        ptr = 0
        batches = 0

        # 便捷函数：按“还活着 & 本批未选”的组重抽一个 gi
        def draw_replacement(exclude_set):
            alive = np.where((group_remaining > 0) & (hit_count < hit_quota))[0]
            if exclude_set:
                alive = np.array([x for x in alive if x not in exclude_set], dtype=int)
            if alive.size == 0:
                return None
            probs = self.p[alive].astype(float)
            probs /= probs.sum()
            return int(np.random.choice(alive, p=probs))

        while batches < self.num_batches:
            batch = []
            chosen_gset = set()

            while len(batch) < self.batch_size:
                if len(chosen_gset) >= self.groups_per_batch:
                    break

                # 先按 schedule 取一个候选 gi
                gi = None
                while ptr < len(schedule):
                    cand = schedule[ptr]; ptr += 1
                    if cand in chosen_gset:
                        continue
                    gi = cand
                    break

                # schedule 用尽，尝试在线重抽
                if gi is None:
                    gi = draw_replacement(chosen_gset)
                    if gi is None:
                        break  # 没有可用组了

                # 若该组已无预算或配额用尽，在线重抽
                if group_remaining[gi] <= 0 or hit_count[gi] >= hit_quota[gi]:
                    gi = draw_replacement(chosen_gset)
                    if gi is None:
                        break

                # —— 组内取样（队列轮转 + 小组单条/批）——
                gkey = self.group_keys[gi]
                q = queues[gi]

                # 小组单条/批：n_g ≤ 2 则本批最多取 1 条
                n_g = len(self.groups[gkey])
                per_batch_take = 2 if n_g <= 2 else self.max_per_group

                take = min(per_batch_take, self.batch_size - len(batch))
                if not np.isinf(group_remaining[gi]):
                    take = min(take, int(group_remaining[gi]))

                got = []
                tries = 0
                # 轮转：从队头拿，达 cap 的样本丢弃，不再放回
                while len(got) < take and len(q) > 0 and tries < 2 * len(q):
                    idx = q.popleft(); tries += 1
                    if self.sample_repeat_cap is not None and used[idx] >= self.sample_repeat_cap:
                        # 达 cap：永久移出队列
                        continue
                    got.append(idx)
                    used[idx] += 1
                    q.append(idx)  # 放回队尾，保证先覆盖再复用

                if len(got) == 0:
                    # 该组当前拿不到样本；换一个组
                    gi2 = draw_replacement(chosen_gset)
                    if gi2 is None:
                        break
                    gi = gi2
                    continue

                # 记录命中与预算扣减
                hit_count[gi] += 1
                group_remaining[gi] -= len(got)
                batch.extend(got)
                chosen_gset.add(gi)

            if len(batch) == 0:
                break  # 绝不 yield 空批

            batches += 1
            yield batch

def build_loader_for_long_tail(dataset, peptide_ids, t=1e-3,
                               batch_size=128, max_per_group=2,
                               alpha=1, mix=0.1, repeat_cap=4, seed=42,
                               num_workers=4, pin_memory=True):
    peps = np.asarray(peptide_ids)
    uniq, inv = np.unique(peps, return_inverse=True)
    counts = np.bincount(inv)
    rfs = rfs_repeat_factors(counts,t)
    probs  = mixed_group_probs_from_rfs(rfs, alpha=alpha, mix=mix)

    sampler = QuotaGroupedBatchSampler(
        peptide_ids=inv,     
        probs=probs,
        batch_size=batch_size,
        max_per_group=max_per_group,
        sample_repeat_cap=repeat_cap,
        seed=seed
    )

    return DataLoader(dataset, batch_sampler=sampler,
                      num_workers=num_workers, pin_memory=pin_memory)

class UniformPeptideBatchSampler(Sampler[List[int]]):
    """
    每个 epoch：将所有 peptide(组)打乱 -> 按 peptides_per_step 分块。
    每 step：从块内每个 peptide 抽 samples_per_peptide 条样本，组成一个 batch。
    - ensure_full_batch=True: 组样本不足时循环取，保证批尺寸稳定
    - set_epoch(epoch) 可用于每轮不同随机序
    """
    def __init__(self,
                 peptide_ids: Sequence[Hashable],
                 *,
                 peptides_per_step: int,        # 每步抽多少个不同 peptide
                 samples_per_peptide: int = 2,   # 每个 peptide 取几条
                 seed: Optional[int] = None,
                 ensure_full_batch: bool = True):
        self.peptides_per_step = int(peptides_per_step)
        self.samples_per_peptide = int(samples_per_peptide)
        self.ensure_full_batch = bool(ensure_full_batch)
        self.base_seed = 0 if seed is None else int(seed)
        self._epoch = 0

        # 分组：gid -> 样本索引列表
        groups = defaultdict(list)
        for idx, gid in enumerate(peptide_ids):
            groups[gid].append(idx)          # 不做 int() 强转
        self.group_keys = list(groups.keys())           # 组ID列表（可能是字符串）
        self.groups = {g: list(v) for g, v in groups.items()}  # 仍用原始ID做key
        self.P = len(self.group_keys)

        if self.peptides_per_step <= 0:
            raise ValueError("peptides_per_step must be > 0")
        if self.samples_per_peptide <= 0:
            raise ValueError("samples_per_peptide must be > 0")

        # 预计算 epoch 内 step 数
        self.num_steps = math.ceil(self.P / self.peptides_per_step)

    def __len__(self):
        return self.num_steps

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _rng(self):
        return random.Random(self.base_seed + 1000003 * self._epoch)

    def __iter__(self):
        rng = self._rng()

        # 1) 打乱所有 peptide
        order = list(range(self.P))
        rng.shuffle(order)

        # 2) 为每个 peptide 建立一个打乱的循环队列（组内轮转，先覆盖再复用）
        queues: dict[int, deque] = {}
        for gi, gkey in enumerate(self.group_keys):
            pool = list(self.groups[gkey])
            rng.shuffle(pool)
            queues[gi] = deque(pool)

        # 3) 按块产出 batch
        ptr = 0
        for _ in range(self.num_steps):
            # 本步选出的 peptide 索引（在 group_keys 内的下标）
            chunk = order[ptr: ptr + self.peptides_per_step]
            ptr += self.peptides_per_step

            batch = []
            for gi in chunk:
                q = queues[gi]
                m = len(q)

                need = self.samples_per_peptide
                taken = []

                if m == 0:
                    # 该组没有样本（极端情况），跳过
                    continue

                # 组内轮转：尽量不重复；若样本不足且 ensure_full_batch=True 则循环补齐
                # 先取最多不重复的
                k = min(need, m)
                for _ in range(k):
                    idx = q.popleft()
                    taken.append(idx)
                    q.append(idx)   # 放回队尾，保证轮转

                # 若不足且需要稳定批大小，则循环补齐（允许同批重复）
                if self.ensure_full_batch and len(taken) < need:
                    # 再次从队头循环拿到 need
                    for _ in range(need - len(taken)):
                        idx = q[0]         # 看队头
                        taken.append(idx)
                        q.rotate(-1)       # 轮转

                # 若不要求稳定批大小，则保留 len(taken) < need 的情况
                batch.extend(taken)

            # 允许最后一个 batch 不满（当 P 不是 peptides_per_step 的整数倍时）
            if len(batch) == 0:
                break
            yield batch

def build_loader_uniform_by_peptide(dataset, peptide_ids,
                                    *, peptides_per_step: int,
                                    samples_per_peptide: int = 2,
                                    seed: int = 42,
                                    num_workers: int = 0,
                                    pin_memory: bool = False,
                                    ensure_full_batch: bool = True):
    sampler = UniformPeptideBatchSampler(
        peptide_ids=peptide_ids,
        peptides_per_step=peptides_per_step,
        samples_per_peptide=samples_per_peptide,
        seed=seed,
        ensure_full_batch=ensure_full_batch
    )
    return DataLoader(dataset, batch_sampler=sampler,
                      num_workers=num_workers, pin_memory=pin_memory)

# def validate_sampler_epoch(pt_loader, inv, *, batch_size, max_per_group, sample_repeat_cap=None):
#     """
#     pt_loader: DataLoader 返回 (out, idx)；idx 是样本索引 1D tensor
#     inv: 长度 N 的数组，把每个样本索引映射到 peptide 组ID（np.unique(..., return_inverse=True) 得到的 inv）
#     """
#     N = len(inv)
#     P = int(inv.max()) + 1

#     sample_repeats = np.zeros(N, dtype=np.int64)   # 每个样本本epoch被抽次数
#     group_hits = np.zeros(P, dtype=np.int64)       # 每个组在多少个batch里出现过（命中次数）
#     empty_batches = 0
#     bad_batches = []

#     for step, (out, idx_tensor) in enumerate(pt_loader):
#         idx = idx_tensor.detach().cpu().numpy().astype(int)
#         if idx.size == 0:
#             empty_batches += 1
#             continue

#         # 统计这个 batch 内各组的样本数
#         g = inv[idx]
#         uniq_g, cnt_g = np.unique(g, return_counts=True)
#         # (2) 每个 batch 内每组样本数 ≤ max_per_group
#         if (cnt_g > max_per_group).any():
#             bad_batches.append((step, uniq_g[cnt_g > max_per_group].tolist(),
#                                       cnt_g[cnt_g > max_per_group].tolist()))

#         # (3) 样本重复上限（跨 batch 汇总）
#         sample_repeats[idx] += 1

#         # (4) 组命中（这个 batch 里出现过就算 1 次）
#         group_hits[uniq_g] += 1

#     # 汇总检查
#     report = {}

#     # (1) 空 batch
#     report['empty_batches'] = int(empty_batches)

#     # (2) 违反 per-group-per-batch 上限
#     report['violated_batches'] = bad_batches[:5]  # 只显示前几条
#     report['violated_batches_count'] = len(bad_batches)

#     # (3) 样本重复上限
#     if sample_repeat_cap is not None:
#         over_cap = np.where(sample_repeats > sample_repeat_cap)[0]
#         report['over_cap_count'] = int(over_cap.size)
#         report['max_sample_repeats'] = int(sample_repeats.max())
#     else:
#         report['max_sample_repeats'] = int(sample_repeats.max())

#     # (5) EPS 分布（真实值）
#     # 先把每个样本的重复次数按组聚合，得到每组的“每样本重复次数”的均值/分位
#     eps_by_group = defaultdict(list)
#     for s_idx, rep in enumerate(sample_repeats):
#         eps_by_group[int(inv[s_idx])].append(int(rep))
#     # 全体样本 EPS 分布
#     eps_all = sample_repeats[sample_repeats > 0]  # 没出现过的样本不计入
#     if eps_all.size > 0:
#         q = np.percentile(eps_all, [50, 90, 95, 99])
#         report['EPS_all'] = {'P50': float(q[0]), 'P90': float(q[1]),
#                              'P95': float(q[2]), 'P99': float(q[3]),
#                              'max': int(eps_all.max())}
#     else:
#         report['EPS_all'] = None

#     # (4) 组出场次数的统计
#     qg = np.percentile(group_hits[group_hits > 0], [50, 90, 95]) if (group_hits > 0).any() else [0, 0, 0]
#     report['group_hits_summary'] = {
#         'nonzero_groups': int((group_hits > 0).sum()),
#         'P50': float(qg[0]),
#         'P90': float(qg[1]),
#         'P95': float(qg[2]),
#         'max': int(group_hits.max())
#     }

#     return report, sample_repeats, group_hits
# peps = np.asarray(pt_df.peptide)
# uniq, inv = np.unique(peps, return_inverse=True)

# # 跑一个 epoch 后（或只迭代一次完整 loader）
# report, sample_repeats, group_hits = validate_sampler_epoch(
#     pt_loader, inv,
#     batch_size=128,
#     max_per_group=2,
#     sample_repeat_cap=4
# )

# print(report)

# peptide_repeat_counts = np.bincount(inv, weights=sample_repeats)
# df = pd.DataFrame({
#     "peptide": uniq,
#     "repeats": peptide_repeat_counts
# })

# x = np.asarray(peptide_repeat_counts)
# x_pos = x[x > 0]
# if x_pos.size > 0:
#     bins = np.logspace(np.log10(100), np.log10(x_pos.max()), 80)
#     plt.figure()
#     plt.hist(x_pos, bins=bins, density=True)
#     plt.xscale("log")
#     plt.xlabel("Total repeats per peptide (log scale)")
#     plt.ylabel("Density")
#     plt.title("Distribution (log-x)")
#     plt.tight_layout()
#     plt.show()

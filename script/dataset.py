import torch
from torch.utils.data import Dataset, Sampler, DataLoader

import numpy as np

import random
import os
import math
from collections import defaultdict, Counter, deque
from typing import List, Sequence, Hashable, Optional

from .dataprocess import mk_aa_dict, mk_bv_dict, load_mhc_dict,\
    aa_to_vec, pad_1d, pad_2d, get_masked_sample, mhc_to_aa, mhc_to_esm, mk_aj_dict, mk_av_dict
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


class MPTFineTuneDataSet(Dataset):
  def __init__(self, mpt_df, mhc_type, 
               mhc_max_len, pep_max_len, cdr3_max_len,
               bv=False, av=False, aj=False, score=None):
    self.df = mpt_df
    self.mhc_type = mhc_type

    self.mhc_max_len = mhc_max_len
    self.cdr3_max_len = cdr3_max_len
    self.pep_max_len = pep_max_len

    self.bv = bv
    self.av = av
    self.aj = aj
    self.score = score

    self.aa_dict = mk_aa_dict()
    self.bv_dict = mk_bv_dict()
    self.av_dict = mk_av_dict()
    self.aj_dict = mk_aj_dict()
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
    pep_ids = pad_1d(pep_ids, self.pep_max_len, pad_value=0, dtype=int)

    cdr3_ids = aa_to_vec(cdr3_seq, self.aa_dict)
    cdr3_ids = pad_1d(cdr3_ids, self.cdr3_max_len, pad_value=0, dtype=int)

    out = {
       'mhc': torch.as_tensor(mhc_ids, dtype=torch.long),
        'peptide': torch.as_tensor(pep_ids, dtype=torch.long),
        'cdr3': torch.as_tensor(cdr3_ids, dtype=torch.long),
        'esm_mhc': torch.as_tensor(esm_mhc, dtype=torch.float32)
    }

    if self.bv:
        bv_name = row['trbv']
        out['trbv'] = torch.tensor(self.bv_dict.get(bv_name, 0), dtype=torch.long)

    if self.av:
        av_name = row['trav']
        out['trav'] = torch.tensor(self.av_dict.get(av_name, 0), dtype=torch.long)

    if self.aj:
        aj_name = row['traj']
        out['traj'] = torch.tensor(self.aj_dict.get(aj_name, 0), dtype=torch.long)

    if self.score is not None:
        out['score'] = torch.tensor(float(row[self.score]), dtype=torch.float32)

    return out, idx  

class UniformPeptideBatchSampler(Sampler[List[int]]):
    def __init__(self,
                 peptide_ids: Sequence[Hashable],
                 *,
                 peptides_per_step: int,        
                 samples_per_peptide: int = 2, 
                 seed: Optional[int] = None,
                 ensure_full_batch: bool = True):
        self.peptides_per_step = int(peptides_per_step)
        self.samples_per_peptide = int(samples_per_peptide)
        self.ensure_full_batch = bool(ensure_full_batch)
        self.base_seed = 0 if seed is None else int(seed)
        self._epoch = 0

        groups = defaultdict(list)
        for idx, gid in enumerate(peptide_ids):
            groups[gid].append(idx)         
        self.group_keys = list(groups.keys())          
        self.groups = {g: list(v) for g, v in groups.items()} 
        self.P = len(self.group_keys)

        if self.peptides_per_step <= 0:
            raise ValueError("peptides_per_step must be > 0")
        if self.samples_per_peptide <= 0:
            raise ValueError("samples_per_peptide must be > 0")

        self.num_steps = math.ceil(self.P / self.peptides_per_step)

    def __len__(self):
        return self.num_steps

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _rng(self):
        return random.Random(self.base_seed + 1000003 * self._epoch)

    def __iter__(self):
        rng = self._rng()

        order = list(range(self.P))
        rng.shuffle(order)

        queues: dict[int, deque] = {}
        for gi, gkey in enumerate(self.group_keys):
            pool = list(self.groups[gkey])
            rng.shuffle(pool)
            queues[gi] = deque(pool)

        ptr = 0
        for _ in range(self.num_steps):
            chunk = order[ptr: ptr + self.peptides_per_step]
            ptr += self.peptides_per_step

            batch = []
            for gi in chunk:
                q = queues[gi]
                m = len(q)

                need = self.samples_per_peptide
                taken = []

                if m == 0:
                    continue

                k = min(need, m)
                for _ in range(k):
                    idx = q.popleft()
                    taken.append(idx)
                    q.append(idx)  

                if self.ensure_full_batch and len(taken) < need:
                    for _ in range(need - len(taken)):
                        idx = q[0]      
                        taken.append(idx)
                        q.rotate(-1)      

                batch.extend(taken)

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
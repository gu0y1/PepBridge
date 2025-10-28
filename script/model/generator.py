import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import random
import math
import numpy as np

from pepbridge import PepBridge
from pair_aware_block import PairAwareTrunk

class SeqHead(nn.Module):
    def __init__(self, d_seq, aa_size):
        super().__init__()
        self.ln = nn.LayerNorm(d_seq)
        self.proj = nn.Linear(d_seq, aa_size)
        nn.init.zeros_(self.proj.weight)

    def forward(self, s):
        logits = self.proj(self.ln(s))

        return logits
    
class PairHead(nn.Module):
    def __init__(self, d_pair, aa_size):
        super().__init__()
        self.aa_size = aa_size

        self.ln = nn.LayerNorm(d_pair)
        self.proj = nn.Linear(d_pair, aa_size * aa_size)
        nn.init.zeros_(self.proj.weight)

    def forward(self, z):
        logits = self.proj(self.ln(z))
        logits = rearrange(logits, 'b i j (a c) -> b i j a c', a = self.aa_size)
        logits = 0.5 * (logits + rearrange(logits, 'b i j a c -> b j i c a'))

        return logits
    
class ConvBNAct2d(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        pad = k // 2   # "same" padding for stride=1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class Discriminator(nn.Module):
    def __init__(self, aa_size, d_seq, 
                 d_pair, dropout, tau, bv_size):
        super().__init__()
        self.aa_size = aa_size
        self.bv_size = bv_size
        self.tau = tau

        self.cnn1 = ConvBNAct2d(1, d_pair, 5)
        self.cnn2 = ConvBNAct2d(d_pair, d_seq, 3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        if bv_size is not None:
            trbv_dim = 48
            self.trbv_emb = nn.Linear(bv_size, trbv_dim)
        else:
            trbv_dim = 0

        in_dim = d_seq + trbv_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _smooth_then_log(self, probs, eps=0.05):
        if eps > 0:
            V = probs.size(-1)
            probs = probs * (1 - eps) + eps / V
        return torch.log(probs.clamp_min(1e-6))

    def _ids_to_onehot(self, aa_seq, bv):
        seq = F.one_hot(aa_seq.long().clamp_min(0), num_classes=self.aa_size).to(torch.float)
        if bv is not None:
            if bv.dim() == 2 and bv.size(-1) == 1:
                bv = bv.squeeze(-1)
            bv = F.one_hot(bv.long().clamp_min(0), num_classes=self.bv_size).to(torch.float)
            return self._smooth_then_log(seq), self._smooth_then_log(bv)
        else:
            return self._smooth_then_log(seq), None
        
    def forward(self, seq, mask, bv, is_logits):
        if is_logits:
            seq = F.log_softmax(seq / self.tau, dim=-1)
            if bv is not None:
                bv = F.log_softmax(bv / self.tau, dim=-1)
        else:
            seq, bv = self._ids_to_onehot(seq, bv)

        if mask is not None:
            fill = -math.log(self.aa_size)        
            seq = seq.masked_fill(~mask.unsqueeze(-1).bool(), fill)

        s = seq.unsqueeze(1) 
        s = self.cnn1(s)                       
        s = self.cnn2(s) 
        s = self.gap(s).flatten(1)

        if (bv is not None) and (self.bv_size is not None):
            bv_emb = self.trbv_emb(bv) 
            feat = torch.cat([s, bv_emb], dim=-1)
        else:
            feat = s

        logits = self.mlp(feat)           
        return logits

class BVPredHead(nn.Module):
    def __init__(self, d_seq, bv_size, dropout):
        super().__init__()
        self.ln_mp = nn.LayerNorm(d_seq)
        self.ln_cdr3 = nn.LayerNorm(d_seq)

        self.linear_mp = nn.Linear(d_seq, d_seq, bias=False)
        self.linear_cdr3 = nn.Linear(d_seq, d_seq, bias=False)

        self.g_mp_proj = nn.Linear(d_seq * 2, d_seq)
        self.g_cdr3_proj  = nn.Linear(d_seq * 2, d_seq)
        nn.init.zeros_(self.g_mp_proj.weight); torch.nn.init.ones_(self.g_mp_proj.bias)
        nn.init.zeros_(self.g_cdr3_proj.weight); torch.nn.init.ones_(self.g_cdr3_proj.bias)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_seq * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, bv_size)
        )

    @staticmethod
    def _masked_mean(x, mask, dim=1):
        if mask.dim() == x.dim() - 1:
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=x.dtype)
        num = (x * mask).sum(dim=dim)
        den = mask.sum(dim=dim).clamp_min(1e-6)
        return num / den

    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor, dim=1):
        if mask.dim() == x.dim() - 1:
            mask = mask.unsqueeze(-1)
        x_masked = x.masked_fill(mask == 0, float('-inf'))
        v = x_masked.amax(dim=dim)
        return torch.where(torch.isinf(v), torch.zeros_like(v), v)
    
    def forward(self, mp, cdr3, mask_mp, mask_cdr3):
        mp = self.linear_mp(self.ln_mp(mp))
        cdr3 = self.linear_cdr3(self.ln_cdr3(cdr3))

        if mask_mp is not None:  
            mp_mean = self._masked_mean(mp, mask_mp, dim=1)    # [B, d_seq]
            mp_max  = self._masked_max(mp, mask_mp, dim=1) # [B, d_seq]
        else:
            mp_mean = mp.mean(dim=1)
            mp_max  = mp.amax(dim=1)
        gate_mp = torch.sigmoid(self.g_mp_proj(torch.cat([mp_mean, mp_max],dim=-1)))
        mp_feat = mp_mean + gate_mp * (mp_max - mp_mean)   # [B, d_seq]

        if mask_cdr3 is not None:
            cdr3_mean = self._masked_mean(cdr3, mask_cdr3, dim=1)    # [B, d_seq]
            cdr3_max  = self._masked_max(cdr3, mask_cdr3, dim=1) # [B, d_seq]
        else:
            cdr3_mean = cdr3.mean(dim=1)
            cdr3_max  = cdr3.amax(dim=1)
        gate_cdr3 = torch.sigmoid(self.g_cdr3_proj(torch.cat([cdr3_mean, cdr3_max],dim=-1)))
        cdr3_feat = cdr3_mean + gate_cdr3 * (cdr3_max - cdr3_mean)   # [B, d_seq]

        logits = self.mlp(torch.cat([mp_feat, cdr3_feat],dim=-1))

        return logits
    
class PepBridgeGenerator(nn.Module):
    def __init__(self, aa_size, max_len_dict, d_seq, d_head_seq, 
                 d_pair, d_head_pair, dropout, n_layers_dict, 
                 trbv_size, film=False):
        super().__init__()        
        self.mhc_len = max_len_dict['mhc']
        self.pep_len = max_len_dict['peptide']
        self.cdr3_len = max_len_dict['cdr3']
        self.mp_len = self.mhc_len + self.pep_len

        self.FiLM = film
        if self.FiLM:
            self.ctx_to_film_seq  = nn.Sequential(nn.Linear(d_seq, 2*d_seq), nn.ReLU(),
                                      nn.Linear(2*d_seq, 2*d_seq))
            self.ctx_to_film_pair = nn.Sequential(nn.Linear(d_pair, 2*d_pair), nn.ReLU(),
                                      nn.Linear(2*d_pair, 2*d_pair))
            nn.init.zeros_(self.ctx_to_film_seq[-1].weight)
            nn.init.zeros_(self.ctx_to_film_seq[-1].bias)
            nn.init.zeros_(self.ctx_to_film_pair[-1].weight)
            nn.init.zeros_(self.ctx_to_film_pair[-1].bias)

        self.pepbridge = PepBridge(
            aa_size, max_len_dict, d_seq, d_head_seq, 
            d_pair, d_head_pair, dropout, n_layers_dict, trbv_size)
        
        self.pair_aware_trunk = PairAwareTrunk(d_seq, d_head_seq, d_pair, d_head_pair, 
                                                dropout, n_layers=12)
        self.prev_seq_norm = nn.LayerNorm(d_seq)
        self.prev_pair_norm = nn.LayerNorm(d_pair)
        
        self.seq_head = SeqHead(d_seq, aa_size)
        self.pair_head = PairHead(d_pair, aa_size)
        self.bv_head = BVPredHead(d_seq, trbv_size, dropout)

    def _mpt_encode(self, mhc, peptide, cdr3, esm_mhc):
        repr_dict, mask_dict, _ = self.pepbridge.aa_seq_encode(
            mhc=mhc,  peptide=peptide, cdr3=cdr3, esm_mhc=esm_mhc)

        mp_seq, mp_pair = self.pepbridge.mp_repr_encode(
            repr_dict['mhc_seq'], repr_dict['mhc_pair'],
            repr_dict['pep_seq'], repr_dict['pep_pair'],
            mask_dict['mhc'], mask_dict['pep']
        )

        pt_seq, pt_pair = self.pepbridge.pt_repr_encode(
            repr_dict['pep_seq'], repr_dict['pep_pair'],
            repr_dict['cdr3_seq'], repr_dict['cdr3_pair'],
            mask_dict['pep'], mask_dict['cdr3']
        )
        
        mpt_seq, mpt_pair = self.pepbridge.mpt_repr_encode(     
            mp_seq, mp_pair, pt_seq, pt_pair,
            mask_dict['mhc'], mask_dict['pep'], mask_dict['cdr3']
        )

        return mpt_seq, mpt_pair, mask_dict

    def _film(self, seq, pair, mask):
        w = mask.unsqueeze(-1).float()
        seq_ctx = (seq * w).sum(1) / (w.sum(1) + 1e-6)  

        pair_w = mask.unsqueeze(2).float() * mask.unsqueeze(1).float()                  # [B,D_seq]
        pair_ctx = (pair * pair_w.unsqueeze(-1)).sum(1,2) / pair_w.sum(dim=(1, 2)).clamp_min(1e-6).unsqueeze(-1)

        gamma_seq, beta_seq   = self.ctx_to_film_seq(seq_ctx).chunk(2, dim=-1)   # [B,D_seq]×2
        gamma_pair, beta_pair = self.ctx_to_film_pair(pair_ctx).chunk(2, dim=-1)  # [B,D_pair]×2
        gamma_seq  = 0.1 * torch.tanh(gamma_seq)
        gamma_pair = 0.1 * torch.tanh(gamma_pair)

        return gamma_seq, beta_seq, gamma_pair, beta_pair 

    def forward(self, mhc, peptide, cdr3, esm_mhc):
        mpt_seq, mpt_pair, mask_dict = self._mpt_encode(mhc, peptide, cdr3, esm_mhc)
        mp_seq, mp_pair = mpt_seq[:,:self.mp_len,:], mpt_pair[:,:self.mp_len,:self.mp_len,:]
        mp_mask = torch.cat([mask_dict['mhc'], mask_dict['pep']], dim=1)

        if self.FiLM:
            gamma_seq, beta_seq, gamma_pair, beta_pair = self._film(mp_seq, mp_pair, mp_mask)

        init_cdr3_seq = mpt_seq[:, self.mp_len:, :]
        init_cdr3_pair = mpt_pair[:, self.mp_len:, self.mp_len:, :]
        cdr3_mask = mask_dict['cdr3']

        num_recycle = random.randint(0, 3) if self.training else 3

        prev_cdr3_seq  = torch.zeros_like(init_cdr3_seq)
        prev_cdr3_pair = torch.zeros_like(init_cdr3_pair)

        if num_recycle>0:
            for _ in range(num_recycle):
                cdr3_seq_act  = init_cdr3_seq  + prev_cdr3_seq
                cdr3_pair_act = init_cdr3_pair + prev_cdr3_pair

                with torch.no_grad():
                    prev_cdr3_seq, prev_cdr3_pair = self.pair_aware_trunk(cdr3_seq_act, 
                                                                          cdr3_pair_act, 
                                                                          cdr3_mask)
                prev_cdr3_seq = self.prev_seq_norm(prev_cdr3_seq)
                prev_cdr3_pair = self.prev_pair_norm(prev_cdr3_pair)

                if self.FiLM:
                    prev_cdr3_seq = prev_cdr3_seq  * (1 + gamma_seq.unsqueeze(1)) \
                                    + beta_seq.unsqueeze(1)                
                    prev_cdr3_pair = prev_cdr3_pair * (1 + gamma_pair.unsqueeze(1).unsqueeze(1)) \
                                    + beta_pair.unsqueeze(1).unsqueeze(1)    

        cdr3_seq_act  = init_cdr3_seq  + prev_cdr3_seq
        cdr3_pair_act = init_cdr3_pair + prev_cdr3_pair
        cdr3_seq_act, cdr3_pair_act = self.pair_aware_trunk(cdr3_seq_act, 
                                                            cdr3_pair_act, 
                                                            cdr3_mask)

        seq_logits = self.seq_head(cdr3_seq_act)
        pair_logits = self.pair_head(cdr3_pair_act)

        bv_logits = self.bv_head(
            mp_seq.detach(), cdr3_seq_act, 
            mp_mask, cdr3_mask
            )

        return seq_logits, pair_logits, bv_logits



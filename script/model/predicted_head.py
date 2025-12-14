import torch
import torch.nn as nn

class PredHead(nn.Module):
    def __init__(self, d_seq, d_pair, dropout, near_band=4, 
                 trbv_size=None, use_ln=True, extra_dim=0):
        super().__init__()
        assert d_seq == d_pair * 2
        self.ln_s = nn.LayerNorm(d_seq)  if use_ln else nn.Identity()
        self.ln_z = nn.LayerNorm(d_seq) if use_ln else nn.Identity()
        self.linear_seq = nn.Linear(d_seq, d_seq, bias=False)
        self.linear_pair = nn.Linear(d_seq, d_seq, bias=False)

        self.near_band = int(near_band)

        self.trbv_emb = None
        trbv_dim = 0
        if trbv_size is not None:
            trbv_dim = 48
            self.trbv_emb = nn.Embedding(num_embeddings=trbv_size, 
                                         embedding_dim=trbv_dim, 
                                         padding_idx=0)
            
        in_dim = d_seq * 2 + trbv_dim + extra_dim
        self.mlp0 = nn.Sequential(nn.Linear(in_dim, 64),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        self.mlp1 = nn.Sequential(nn.Linear(in_dim, 64),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        self.mlp2 = nn.Sequential(nn.Linear(in_dim, 64),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
    @staticmethod
    def _masked_mean(x, mask, dim):
        if not isinstance(dim, (tuple, list)):
            dim = (dim,)
        num = x * mask
        den = mask
        for d in sorted(dim, reverse=True):
            num = num.sum(dim=d)
            den = den.sum(dim=d)
        den = den.clamp_min(1e-6)
        if den.dim() == num.dim() - 1:
            den = den.unsqueeze(-1)
        return num / den
    
    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor, dim):
        x_masked = x.masked_fill(mask == 0, float('-inf'))
        v = x_masked.amax(dim=dim)
        return torch.where(torch.isinf(v), torch.zeros_like(v), v)
    
    def forward(self, seq_repr, pair_repr, chain_id, mask=None, trbv=None, extra_emb=None):
        assert chain_id is not None, "chain_id can't be None"
        B, L, _ = seq_repr.shape
        device = seq_repr.device

        # seq_repr
        seq_repr = self.linear_seq(self.ln_s(seq_repr))
        if mask is not None:
            m = mask.float().unsqueeze(-1)  
            s_mean = self._masked_mean(seq_repr, m, dim=1)    # [B, d_seq]
            s_max  = self._masked_max(seq_repr, m, dim=1) # [B, d_seq]
        else:
            s_mean = seq_repr.mean(dim=1)
            s_max  = seq_repr.amax(dim=1)
        s_feat = torch.cat([s_mean, s_max],-1)

        # pair_repr
        pair_repr = torch.cat([pair_repr, 
                               pair_repr.transpose(1, 2)],
                               dim=-1)
        eye = torch.eye(L, device=device, dtype=pair_repr.dtype).view(1, L, L, 1)
        pair_repr = pair_repr * (1.0 - eye)
        pair_repr = self.linear_pair(self.ln_z(pair_repr))                     # [B, L, L, d_seq]

        if mask is not None:
            pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).float()  # [B,L,L,1]
        else:
            pair_mask = torch.ones(B, L, L, 1, device=device)

        same_chain = (chain_id.unsqueeze(2) == chain_id.unsqueeze(1)).unsqueeze(-1)
        intra_mask = pair_mask * same_chain.float()
        inter_mask = pair_mask * (~same_chain).float()

        # inter
        inter_mean = self._masked_mean(pair_repr, inter_mask, dim=(1, 2))   # [B, d_seq]

        row_den_raw = inter_mask.sum(dim=2)                                  # [B,L,1]
        row_valid   = (row_den_raw.squeeze(-1) > 0)                          # [B,L]
        row_den     = row_den_raw.clamp_min(1e-6)
        row_mean_i  = (pair_repr * inter_mask).sum(dim=2) / row_den          # [B,L,d_seq]

        neg_big = -1e4 if row_mean_i.dtype in (torch.float16, torch.bfloat16) else -1e9
        score = row_mean_i.norm(p=2, dim=-1).masked_fill(~row_valid, neg_big)  # [B,L]
        k_base = 3 if self.trbv_emb is None else 5
        valid_counts = row_valid.sum(dim=1)
        k_use = int(torch.clamp(valid_counts.min(), min=1).item())
        k = min(k_base, k_use)

        topk_idx = score.topk(k=k, dim=1).indices
        selected = row_mean_i.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, row_mean_i.size(-1)))
        inter_peak = selected.mean(dim=1)                                   

        inter_feat =  torch.cat([inter_mean, inter_peak],dim=-1) # [B, d_seq*2]

        # intra
        idx  = torch.arange(L, device=device)
        diff = (idx[None, :, None] - idx[None, None, :]).abs()  # [L, L]
        band = (diff <= self.near_band) & (diff > 0)       
        band = band.view(1, L, L, 1).float()
        intra_near_mask = intra_mask * band
        intra_compact = self._masked_mean(pair_repr, intra_near_mask, dim=(1, 2))  # [B, d_seq]

        intra_max = self._masked_max(pair_repr, intra_near_mask, dim=(1, 2)) # [B, d_seq]
        intra_feat =  torch.cat([intra_compact, intra_max], dim=-1) # [B, d_seq *2]

        feats = torch.cat([s_feat.unsqueeze(1), 
                           inter_feat.unsqueeze(1), 
                           intra_feat.unsqueeze(1)],
                          dim=1)  # [B, 3, d_seq]
        if self.trbv_emb is not None and trbv is not None:
            trbv = trbv.squeeze(-1) if trbv.dim() > 1 else trbv     # [B]
            bv_emb = self.trbv_emb(trbv).unsqueeze(1).repeat(1,3,1)                      # [B, 3, 48]
            feats = torch.cat([feats, bv_emb], dim=-1)  # [B, 3, in_dim]
        
        if extra_emb is not None:
            extra_emb = extra_emb.unsqueeze(1).repeat(1,3,1) 
            feats = torch.cat([feats, extra_emb], dim=-1)

        logits = torch.cat([self.mlp0(feats[:,0,:]),
                    self.mlp1(feats[:,1,:]),
                    self.mlp2(feats[:,2,:])], dim=1)   # [B,3]    
          
        return self.fusion(logits)
      
class ContactPredHead(nn.Module):
    def __init__(self, d_pair, dropout, use_ln=True, gate=True,
                 dist_bin=None):
        super().__init__()
        d_in = d_pair * 2
        self.ln = nn.LayerNorm(d_in)  if use_ln else nn.Identity()
        self.linear = nn.Linear(d_in, d_in)
        self.act_drop = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))

        self.dist_bin = dist_bin
        if dist_bin is not None:
            d_out = dist_bin + 1
        else:
            d_out = 2

        self.gate = gate
        if gate:
            self.g_proj = nn.Linear(d_in, d_in)
            nn.init.zeros_(self.g_proj.weight)
            nn.init.ones_(self.g_proj.bias)

        self.mlp = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_out)
            )
    
    def forward(self, pair_repr_ab, pair_repr_ba):
        pair_repr = torch.cat([pair_repr_ab, 
                               pair_repr_ba.transpose(1, 2)],
                               dim=-1)
        feat = self.act_drop(self.linear(self.ln(pair_repr)))
        if self.gate:
            feat = feat + torch.sigmoid(self.g_proj(feat)) * pair_repr
        else:
            feat = feat + pair_repr

        out = self.mlp(feat)    
        bind_logits = out[:,:,:,0]
        if self.dist_bin is None:
            dist_logits = torch.relu(out[:,:,:,1])
        else:
            dist_logits = out[:,:,:,1:]

        return bind_logits, dist_logits
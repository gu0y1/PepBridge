import torch
import torch.nn as nn

class PredHead(nn.Module):
    def __init__(self, d_seq, d_pair, dropout, near_band=4, 
                 trbv_size=None, use_ln=True, gate=True):
        super().__init__()

        self.ln_s = nn.LayerNorm(d_seq)  if use_ln else nn.Identity()
        self.ln_z = nn.LayerNorm(d_pair) if use_ln else nn.Identity()
        self.linear_seq = nn.Linear(d_seq, d_seq, bias=False)
        self.linear_pair = nn.Linear(d_pair, d_pair, bias=False)

        self.gate = gate
        if gate:
            self.g_s_proj = nn.Linear(d_seq * 2, d_seq)
            self.g_z_proj  = nn.Linear(d_pair * 2, d_pair)
            nn.init.zeros_(self.g_s_proj.weight); torch.nn.init.ones_(self.g_s_proj.bias)
            nn.init.zeros_(self.g_z_proj.weight); torch.nn.init.ones_(self.g_z_proj.bias)
            
        self.near_band = int(near_band)

        self.trbv_emb = None
        trbv_dim = 0
        if trbv_size is not None:
            trbv_dim = 48
            self.trbv_emb = nn.Embedding(num_embeddings=trbv_size, 
                                         embedding_dim=trbv_dim, 
                                         padding_idx=0)
            
        in_dim = d_seq + d_pair * 2 + trbv_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
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
    
    def forward(self, seq_repr, pair_repr, chain_id, mask=None, trbv=None):
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
        if self.gate:
            gate_s = torch.sigmoid(self.g_s_proj(torch.cat([s_max, s_mean],dim=-1)))
            s_feat = s_mean + gate_s * (s_max - s_mean)   # [B, d_seq]
        else:
            s_feat = 0.5 * (s_mean+s_max)

        # pair_repr
        pair_repr = 0.5 * (pair_repr + pair_repr.transpose(1, 2))
        eye = torch.eye(L, device=device, dtype=pair_repr.dtype).view(1, L, L, 1)
        pair_repr = pair_repr * (1.0 - eye)
        pair_repr = self.linear_pair(self.ln_z(pair_repr))                     # [B, L, L, d_pair]

        if mask is not None:
            pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).float()  # [B,L,L,1]
        else:
            pair_mask = torch.ones(B, L, L, 1, device=device)

        same_chain = (chain_id.unsqueeze(2) == chain_id.unsqueeze(1)).unsqueeze(-1)
        intra_mask = pair_mask * same_chain.float()
        inter_mask = pair_mask * (~same_chain).float()

        # inter
        inter_mean = self._masked_mean(pair_repr, inter_mask, dim=(1, 2))   # [B, d_pair]
        row_den = inter_mask.sum(dim=2).clamp_min(1e-6)         # [B, L, 1]
        row_mean_i = (pair_repr * inter_mask).sum(dim=2) / row_den     # [B, L, d_pair]
        inter_peak = row_mean_i.max(dim=1).values                  # [B, d_pair]B, d_pair]

        if self.gate:
            gate_z = torch.sigmoid(self.g_z_proj(torch.cat([inter_peak, inter_mean],dim=-1)))
            inter_feat = inter_mean + gate_z * (inter_peak - inter_mean)
        else:
            inter_feat = 0.5 * (inter_mean+inter_peak)

        # intra
        idx  = torch.arange(L, device=device)
        band = (idx[None, :, None] - idx[None, None, :]).abs() <= self.near_band
        band = band.view(1, L, L, 1).float()
        intra_near_mask = intra_mask * band
        intra_compact = self._masked_mean(pair_repr, intra_near_mask, dim=(1, 2))  # [B, d_pair]

        feats = [s_feat, inter_feat, intra_compact]
        if self.trbv_emb is not None and trbv is not None:
            trbv = trbv.squeeze(-1) if trbv.dim() > 1 else trbv     # [B]
            feats.append(self.trbv_emb(trbv))                       # [B, 48]

        feat = torch.cat(feats, dim=-1)  # [B, in_dim]
        prob = self.mlp(feat).squeeze(-1)           # [B]
        return prob
    
class ContactPredHead(nn.Module):
    def __init__(self, d_pair, dropout, use_ln=True, gate=True):
        super().__init__()
        d_in = d_pair * 2
        self.ln = nn.LayerNorm(d_in)  if use_ln else nn.Identity()
        self.linear = nn.Linear(d_in, d_in)
        self.act_drop = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))

        self.gate = gate
        if gate:
            self.g_proj = nn.Linear(d_in, d_in)
            nn.init.zeros_(self.g_proj.weight)
            nn.init.ones_(self.g_proj.bias)

        self.mlp = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
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
        prob = torch.sigmoid(out[:,:,:,0])
        dist = torch.relu(out[:,:,:,1])
        return prob, dist


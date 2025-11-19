import torch
import torch.nn as nn

class RelativePositionEmbedding(nn.Module):
    def __init__(self, d_pair: int, K: int = 32, add_break_bucket: bool = True):
        super().__init__()
        self.K = int(K)
        self.add_break = bool(add_break_bucket)
        self.num_buckets = 2 * self.K + 1 + (1 if self.add_break else 0)
        self.break_id = 2 * self.K + 1 if self.add_break else None

        self.embed = nn.Embedding(self.num_buckets, d_pair)

    @staticmethod
    def _pairwise_diff(res_idx: torch.Tensor) -> torch.Tensor:
        if res_idx.dim() == 1:
            res_idx = res_idx.unsqueeze(0)  # -> [1, L]
        # [B, L, L]，d_ij = i_index - j_index
        return res_idx.unsqueeze(2) - res_idx.unsqueeze(1)

    def forward(
        self,
        residue_index: torch.Tensor,      # [B, L]
        chain_id=None,  # [B, L]
    ) -> torch.Tensor:
        """
        return: relpos_embed  [B, L, L, d_pair]
        """
        device = residue_index.device
        d = self._pairwise_diff(residue_index).to(torch.int64)  # [B, L, L]

        d = d.clamp(min=-self.K, max=self.K)

        bucket_ids = d + self.K  # [B, L, L], int64

        if self.add_break and chain_id is not None:
            if chain_id.dim() == 1:
                chain_id = chain_id.unsqueeze(0)               # [1, L]
            same_chain = (chain_id.unsqueeze(2) == chain_id.unsqueeze(1))  # [B, L, L]
            bucket_ids = torch.where(same_chain, bucket_ids, torch.full_like(bucket_ids, self.break_id))

        relpos_embed = self.embed(bucket_ids) 
        return relpos_embed

class Embedder(nn.Module):
    def __init__(self, aa_size, max_len, d_seq, d_pair, dropout, 
                 mhc=False):
        super(Embedder, self).__init__()
        self.seq_aa_emb =  nn.Embedding(aa_size, d_seq, padding_idx=0)
        self.abs_pos_emb = nn.Embedding(max_len, d_seq)
        self.seq_ln = nn.LayerNorm(d_seq)

        self.pair_aa_emb_i =  nn.Embedding(aa_size, d_pair, padding_idx=0)
        self.pair_aa_emb_j=  nn.Embedding(aa_size, d_pair, padding_idx=0)
        self.rel_pos_emb = RelativePositionEmbedding(d_pair, K=max_len-1, 
                                                     add_break_bucket=False)
        self.pair_ln = nn.LayerNorm(d_pair)

        self.drop = nn.Dropout(dropout)

        if mhc:
            self.mhc_poj = nn.Linear(96, d_seq, bias=False)
            self.mhc_cross_attn = nn.MultiheadAttention(embed_dim=d_seq, num_heads=4,
                                                        dropout=dropout, batch_first=True)

    def forward(self, x, esm_mhc=None):
        B, L = x.shape

        seq_repr = self.seq_aa_emb(x)
        pos_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1) 
        seq_repr = seq_repr + self.abs_pos_emb(pos_idx)

        if esm_mhc is not None:
            if esm_mhc.dim() == 2:
                esm_mhc = esm_mhc.unsqueeze(1)
            esm_mhc = self.mhc_poj(esm_mhc)
            attn_out,_ = self.mhc_cross_attn(seq_repr, esm_mhc, esm_mhc, 
                                           need_weights=False)
            seq_repr = seq_repr + self.drop(attn_out)

        seq_repr = self.drop(self.seq_ln(seq_repr))

        pair_repr = self.pair_aa_emb_i(x).unsqueeze(2) + \
            self.pair_aa_emb_j(x).unsqueeze(1)
        relpos = self.rel_pos_emb(pos_idx) 
        pair_repr = pair_repr + relpos
        pair_repr = self.drop(self.pair_ln(pair_repr))
        
        return seq_repr, pair_repr
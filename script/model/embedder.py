import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionEmbedding(nn.Module):
    """
    AlphaFold2 风格的相对位置编码：
      - 对每对残基 (i,j) 计算差值 d = i - j
      - 裁剪到 [-K, K]，映射到桶 ID ∈ [0 .. 2K]
      - （可选）跨链对使用 break 桶 ID = 2K+1
      - 用 nn.Embedding(num_buckets, d_pair) 投影，得到 [B, L, L, d_pair]
    """
    def __init__(self, d_pair: int, K: int = 32, add_break_bucket: bool = True):
        super().__init__()
        self.K = int(K)
        self.add_break = bool(add_break_bucket)
        self.num_buckets = 2 * self.K + 1 + (1 if self.add_break else 0)
        self.break_id = 2 * self.K + 1 if self.add_break else None

        # 与 "one-hot + Linear" 等价，但更省显存和快
        self.embed = nn.Embedding(self.num_buckets, d_pair)

    @staticmethod
    def _pairwise_diff(res_idx: torch.Tensor) -> torch.Tensor:
        # res_idx: [B, L] 或 [L]
        if res_idx.dim() == 1:
            res_idx = res_idx.unsqueeze(0)  # -> [1, L]
        # [B, L, L]，d_ij = i_index - j_index
        return res_idx.unsqueeze(2) - res_idx.unsqueeze(1)

    def forward(
        self,
        residue_index: torch.Tensor,      # [B, L] 或 [L]，同链上递增的残基序号
        chain_id: torch.Tensor | None = None,  # [B, L]，可选；不同链的残基编号不同
    ) -> torch.Tensor:
        """
        return: relpos_embed  [B, L, L, d_pair]
        """
        device = residue_index.device
        d = self._pairwise_diff(residue_index).to(torch.int64)  # [B, L, L]

        # 1) 裁剪到 [-K, K]
        d = d.clamp(min=-self.K, max=self.K)

        # 2) shift 到 [0 .. 2K] 作为桶 ID
        bucket_ids = d + self.K  # [B, L, L], int64

        # 3) 跨链位置：若提供 chain_id，把跨链对覆写成 break 桶
        if self.add_break and chain_id is not None:
            if chain_id.dim() == 1:
                chain_id = chain_id.unsqueeze(0)               # [1, L]
            same_chain = (chain_id.unsqueeze(2) == chain_id.unsqueeze(1))  # [B, L, L]
            bucket_ids = torch.where(same_chain, bucket_ids, torch.full_like(bucket_ids, self.break_id))

        # 4) 查表得到 [B, L, L, d_pair]
        relpos_embed = self.embed(bucket_ids)  # float32
        return relpos_embed

class Embedder(nn.Module):
    def __init__(self, d_seq, d_pair, max_len, dropout, mhc, num_heads):
        super(Embedder, self).__init__()
        self.seq_aa_emb =  nn.Embedding(22, d_seq, padding_idx=0)
        self.abs_pos_emb = nn.Embedding(max_len, d_seq)
        self.seq_ln = nn.LayerNorm(d_seq)

        self.pair_aa_emb_i =  nn.Embedding(22, d_pair, padding_idx=0)
        self.pair_aa_emb_j=  nn.Embedding(22, d_pair, padding_idx=0)
        self.rel_pos_emb = RelativePositionEmbedding(d_pair, 32)
        self.pair_ln = nn.LayerNorm(d_pair)

        self.drop = nn.Dropout(dropout)

        if mhc:
            self.mhc_poj = nn.Linear(1024, d_seq, bias=False)
            self.mhc_cross_attn = nn.MultiheadAttention(embed_dim=d_seq, num_heads=num_heads,
                                                        dropout=dropout, batch_first=True)

    def forward(self, x, mask, esm2_mhc=None):
        B, L = x.shape

        seq_repr = self.seq_aa_emb(x)
        pos_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1) 
        seq_repr = seq_repr + self.abs_pos_emb(pos_idx)

        if esm2_mhc is not None:
            if esm2_mhc.dim() == 2:
                esm2_mhc = esm2_mhc.unsqueeze(1)
            esm2_mhc = self.mhc_poj(esm2_mhc)
            attn_out = self.mhc_cross_attn(seq_repr, esm2_mhc, esm2_mhc, 
                                           need_weights=False)
            seq_repr = seq_repr + self.drop(attn_out)

        seq_repr = self.drop(self.seq_ln(seq_repr))
        seq_repr = seq_repr * mask.unsqueeze(-1)

        pair_repr = self.pair_aa_emb_i(x).unsqueeze(1) + \
            self.pair_aa_emb_j(x).unsqueeze(2)
        relpos = self.rel_pos_emb(pos_idx) 
        pair_repr = pair_repr + relpos
        pair_repr = self.drop(self.pair_ln(seq_repr))
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)      # [B, L, L]
        pair_repr = pair_repr * pair_mask.unsqueeze(-1)

        return seq_repr, pair_repr
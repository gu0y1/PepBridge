###ESM folding block and openfold###
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

def is_fp16_enabled():
    # Autocast world
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()

    return fp16_enabled

class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.
    """

    def __init__(self, r, batch_dim):
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        return x * self.dropout(x.new_ones(shape))

class PairToSeq(nn.Module):
    def __init__(self, d_pair, num_heads):
        super().__init__()
        self.ln = nn.LayerNorm(d_pair)
        self.linear = nn.Linear(d_pair, num_heads, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, pair_repr):
        assert len(pair_repr.shape) == 4
        pair_repr = self.ln(pair_repr)
        pairwise_bias = self.linear(pair_repr)
        return pairwise_bias

class SelfAttention(nn.Module):
    def __init__(self, d_seq, num_heads, d_head, inf=1e9, gate=True):
        super().__init__()
        assert d_seq == num_heads * d_head

        self.d_seq = d_seq
        self.num_heads = num_heads
        self.d_head = d_head
        self.inf = inf

        self.q_proj = nn.Linear(d_seq, d_seq, bias=False)
        self.k_proj = nn.Linear(d_seq, d_seq, bias=False)
        self.v_proj = nn.Linear(d_seq, d_seq, bias=False)
        self.o_proj = nn.Linear(d_seq, d_seq, bias=False)

        self.rescale_factor = self.d_head**-0.5
        nn.init.zeros_(self.o_proj.weight)

        self.gate = gate
        if gate:
            self.g_proj = nn.Linear(d_seq, d_seq)
            nn.init.zeros_(self.g_proj.weight)
            nn.init.ones_(self.g_proj.bias)

    def forward(self, seq_repr, mask=None, bias=None):
        """
        Inputs:
          x: batch of input sequneces (.. x L x C)
          mask: batch of boolean masks where 1=valid, 0=padding position (.. x L_k). optional.
          bias: batch of scalar pairwise attention biases (.. x Lq x Lk x num_heads). optional.

        Outputs:
          sequence projection (B x L x embed_dim), attention maps (B x L x L x num_heads)
        """

        q = rearrange(self.q_proj(seq_repr), "... l (h c) -> ... h l c", h=self.num_heads)
        k = rearrange(self.k_proj(seq_repr), "... l (h c) -> ... h l c", h=self.num_heads)
        v = rearrange(self.v_proj(seq_repr), "... l (h c) -> ... h l c", h=self.num_heads)

        q = self.rescale_factor * q
        a = torch.einsum("...qc,...kc->...qk", q, k)

        # Add external attention bias.
        if bias is not None:
            a = a + rearrange(bias, "... lq lk h -> ... h lq lk")

        # Do not attend to padding tokens.
        if mask is not None:
            mask = repeat(
                mask, "... lk -> ... h lq lk", h=self.num_heads, lq=q.shape[-2]
            )
            a = a.masked_fill(mask == False, -self.inf)

        a = F.softmax(a, dim=-1)

        y = torch.einsum("...hqk,...hkc->...qhc", a, v)
        y = rearrange(y, "... h c -> ... (h c)", h=self.num_heads)

        if self.gate:
            y = self.g_proj(seq_repr).sigmoid() * y
        y = self.o_proj(y)

        return y, a

class Transition(nn.Module):
    def __init__(self, dim, expansion, dropout):
        super().__init__()
        inner_dim = dim * expansion
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.zeros_(self.mlp[-2].bias)

    def forward(self, x):
        return x + self.mlp(x)
    
class SeqToPair(nn.Module):
    def __init__(self, d_seq, d_pair):
        super().__init__()
        assert d_pair % 2 == 0

        self.ln = nn.LayerNorm(d_seq)
        self.proj = nn.Linear(d_seq, d_pair, bias=True)
        self.o_proj = nn.Linear(d_pair, d_pair, bias=True)
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    def forward(self, seq_repr):
        assert len(seq_repr.shape) == 3

        s = self.ln(seq_repr)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)

        return x
    
class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_z, c_hidden, _outgoing=True):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = nn.Linear(self.c_z, self.c_z)
        self.linear_z = nn.Linear(self.c_hidden, self.c_z)

        self.layer_norm_in = nn.LayerNorm(self.c_z)
        self.layer_norm_out = nn.LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

        self.linear_a_p = nn.Linear(self.c_z, self.c_hidden)
        self.linear_a_g = nn.Linear(self.c_z, self.c_hidden)
        self.linear_b_p = nn.Linear(self.c_z, self.c_hidden)
        self.linear_b_g = nn.Linear(self.c_z, self.c_hidden)

        nn.init.zeros_(self.linear_g.weight); torch.nn.init.ones_(self.linear_g.bias)
        nn.init.zeros_(self.linear_a_g.weight); torch.nn.init.ones_(self.linear_a_g.bias)
        nn.init.zeros_(self.linear_b_g.weight); torch.nn.init.ones_(self.linear_b_g.bias)

        nn.init.zeros_(self.linear_z.weight)
        nn.init.zeros_(self.linear_z.bias)
    
    def _combine_projections(self, a, b, outgoing):
        if outgoing:
            # out[i,j,d] = sum_k a[i,k,d] * b[j,k,d]
            b_jkd = b.transpose(1, 2).contiguous()    # [B, j, k, D]
            out = torch.einsum('bikd, bjkd -> bijd', a, b_jkd)
        else:
            # incoming: out[i,j,d] = sum_k a[k,i,d] * b[k,j,d]
            a_kid = a.transpose(1, 2).contiguous()      # [B, k, i, D]
            out = torch.einsum('bkid, bkjd -> bijd', a_kid, b)
        return out

    def forward(self, z, mask):
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask 1=valid, 0=padding
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)
        
        z = self.layer_norm_in(z)
        a = mask * self.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = mask * self.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)

        if(is_fp16_enabled()):
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float(), self._outgoing)
        else:
            x = self._combine_projections(a, b, self._outgoing)
        
        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x

class TriangleAttention(nn.Module):
    def __init__(self, d_pair, num_heads, d_head, 
                 starting=True, inf=1e9, gate=True):
        super().__init__()
        assert d_pair == num_heads * d_head

        self.d_pair = d_pair
        self.num_heads = num_heads
        self.d_head = d_head
        self.inf = inf
        self.starting = starting

        self.ln = nn.LayerNorm(d_pair)
        self.w_bias = nn.Linear(d_pair, num_heads)

        self.q_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.k_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.v_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.o_proj = nn.Linear(d_pair, d_pair, bias=False)

        nn.init.zeros_(self.o_proj.weight)
        self.rescale_factor = self.d_head**-0.5

        self.gate = gate
        if gate:
            self.g_proj = nn.Linear(d_pair, d_pair)
            nn.init.zeros_(self.g_proj.weight)
            nn.init.ones_(self.g_proj.bias)

    def forward(self, pair_repr, mask=None):
        if mask is None:
            mask = pair_repr.new_ones(pair_repr.shape[:-1]) 
        z = self.ln(pair_repr)

        if(not self.starting):
            z = z.transpose(1, 2)
            mask = mask.transpose(1, 2)

        # bias terms
        mask_bias = (self.inf * (mask.to(z.dtype) - 1))[..., :, None, None, :] # [*, I, 1, 1, J]
        tri_bias = rearrange(self.w_bias(z), "... i j h -> ... 1 h i j")
        
        q = rearrange(self.q_proj(z), "... i j (h c) -> ... h i j c", h=self.num_heads)
        k = rearrange(self.k_proj(z), "... i j (h c) -> ... h i j c", h=self.num_heads)
        v = rearrange(self.v_proj(z), "... i j (h c) -> ... h i j c", h=self.num_heads)

        q = self.rescale_factor * q

        a = torch.einsum("bhijc,bhikc->bhijk", q, k) #[*, H, I, J, J]
        a = a + mask_bias + tri_bias
        a = F.softmax(a, dim=-1)

        y = torch.einsum("bhijk,bhikc->bhijc", a, v)
        y = rearrange(y, "... h i j c -> ... i j (h c)", h=self.num_heads)

        if self.gate:
            y = self.g_proj(z).sigmoid() * y
        y = self.o_proj(y)

        if(not self.starting):
            y = y.transpose(1, 2)

        return y
    
class PairAwareBlock(nn.Module):
    def __init__(self, d_seq, d_head_seq, d_pair, d_head_pair, dropout):
        super().__init__()

        assert d_seq % d_head_seq == 0
        assert d_pair % d_head_pair == 0
        num_heads_seq = d_seq // d_head_seq
        num_heads_pair = d_pair // d_head_pair
        assert d_seq == num_heads_seq * d_head_seq
        assert d_pair == num_heads_pair * d_head_pair
        assert d_pair % 2 == 0

        self.d_seq = d_seq
        self.d_pair = d_pair

        self.ln = nn.LayerNorm(d_seq) 

        self.pair_to_seq = PairToSeq(d_pair, num_heads_seq)
        self.seq_to_pair = SeqToPair(d_seq, d_pair)

        self.seq_attn = SelfAttention(d_seq, num_heads_seq, d_head_seq, 
                                      inf=1e9, gate=True)
        
        self.tri_mul_out = TriangleMultiplicativeUpdate(d_pair, d_pair, 
                                                        _outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(d_pair, d_pair, 
                                                        _outgoing=False)
        self.tri_attn_start = TriangleAttention(d_pair, num_heads_pair, d_head_pair, 
                                                starting=True, inf=1e9, gate=True)  
        self.tri_attn_end = TriangleAttention(d_pair, num_heads_pair, d_head_pair, 
                                              starting=False, inf=1e9, gate=True)
        self.trans_seq = Transition(d_seq, 2, dropout)
        self.trans_pair = Transition(d_pair, 2, dropout)

        assert 0 <= dropout < 0.5   
        self.drop = nn.Dropout(dropout)
        self.row_drop = Dropout(dropout * 2, 2)
        self.col_drop = Dropout(dropout * 2, 1)

    def forward(self, seq_repr, pair_repr, mask=None):
        """
        Inputs:
          seq_repr: B x L x d_seq
          pair_repr: B x L x L x d_pair
          mask: B x L boolean tensor of valid positions

        Output:
          seq_repr: B x L x d_seq
          pair_repr: B x L x L x d_pair
        """
        assert len(seq_repr.shape) == 3
        assert len(pair_repr.shape) == 4

        bs, seq_len, d_seq = seq_repr.shape
        d_pair = pair_repr.shape[3]
        assert d_seq == self.d_seq
        assert d_pair == self.d_pair
        assert bs == pair_repr.shape[0]
        assert seq_len == pair_repr.shape[1]
        assert seq_len == pair_repr.shape[2]

        if mask is not None:
            assert len(mask.shape) == 2

        # Sequence representation update
        # Self attention with bias and 
        bias = self.pair_to_seq(pair_repr)
        y = self.ln(seq_repr)
        y, _ = self.seq_attn(y, mask=mask, bias=bias)
        seq_repr = seq_repr + self.drop(y)

        # Transition 
        seq_repr = self.trans_seq(seq_repr)

        # Pair representation update
        # Pairwise produce & difference
        pair_repr = pair_repr + self.seq_to_pair(seq_repr)

        # Triangular
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) if mask is not None else None
        pair_repr = pair_repr + self.row_drop(
            self.tri_mul_out(pair_repr, mask=pair_mask)
        )
        pair_repr = pair_repr + self.col_drop(
            self.tri_mul_in(pair_repr, mask=pair_mask)
        )
        pair_repr = pair_repr + self.row_drop(
            self.tri_attn_start(pair_repr, mask=pair_mask)
        )
        pair_repr = pair_repr + self.col_drop(
            self.tri_attn_end(pair_repr, mask=pair_mask)
        )

        # Transition
        pair_repr = self.trans_pair(pair_repr)
        return seq_repr, pair_repr

class PairAwareTrunk(nn.Module):
    def __init__(self, d_seq, d_head_seq, d_pair, d_head_pair, dropout, n_layers):
        super().__init__()
        self.pair_aware_trunk = nn.ModuleList([
            PairAwareBlock(d_seq, d_head_seq, d_pair, d_head_pair, dropout) for _ in range(n_layers)
        ])

    def forward(self, seq_repr, pair_repr, mask):
        for _, pair_aware_layer in enumerate(self.pair_aware_trunk):
            seq_repr, pair_repr = pair_aware_layer(seq_repr, pair_repr, mask)
        return seq_repr, pair_repr
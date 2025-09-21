import torch
import torch.nn as nn
import torch.nn.functional as F

def assemble_full_pair2(zAA, zAB, zBA, zBB):
    """
    zAA: [B, LA, LA, C], zAB: [B, LA, LB, C]
    zBA: [B, LB, LA, C], zBB: [B, LB, LB, C]
    -> z_full: [B, LA+LB, LA+LB, C]
    """
    upper = torch.cat([zAA, zAB], dim=2)   # [B, LA, LA+LB, C]
    lower = torch.cat([zBA, zBB], dim=2)   # [B, LB, LA+LB, C]
    z_full = torch.cat([upper, lower], dim=1).contiguous()
    return z_full

def assemble_full_pair3(zAA, zAB, zAC,
                        zBA, zBB, zBC,
                        zCA, zCB, zCC):
    """
      zAA: [B, LA, LA, C]   zAB: [B, LA, LB, C]   zAC: [B, LA, LC, C]
      zBA: [B, LB, LA, C]   zBB: [B, LB, LB, C]   zBC: [B, LB, LC, C]
      zCA: [B, LC, LA, C]   zCB: [B, LC, LB, C]   zCC: [B, LC, LC, C]
    """
    upper = torch.cat([zAA, zAB, zAC], dim=2)  # [B, LA, LA+LB+LC, C]
    middle = torch.cat([zBA, zBB, zBC], dim=2) # [B, LB, LA+LB+LC, C]
    lower = torch.cat([zCA, zCB, zCC], dim=2)  # [B, LC, LA+LB+LC, C]
    z_full = torch.cat([upper, middle, lower], dim=1).contiguous()
    return z_full

class JointPair(nn.Module):
    def __init__(self, d_seq, d_pair):
        super().__init__()
        self.Wa = nn.Linear(d_seq, d_pair, bias=False)
        self.Wb = nn.Linear(d_seq, d_pair, bias=False)

        self.break_embed = nn.Parameter(torch.zeros(d_pair))
        self.ln = nn.LayerNorm(d_pair)

    def forward(self, seq_repr_a, seq_repr_b):
        pair_repr_ab = self.Wa(seq_repr_a).unsqueeze(2) + \
            self.Wb(seq_repr_b).unsqueeze(1)

        pair_repr_ab = pair_repr_ab + self.break_embed.view(1, 1, 1, -1)
        pair_repr_ab = self.ln(pair_repr_ab)
        pair_repr_ba = pair_repr_ab.transpose(1,2).contiguous()

        return pair_repr_ab, pair_repr_ba

class JointEmbedder(nn.Module):
    def __init__(self, d_seq, d_pair, n_chains=2, use_cross_linear =True):
        super().__init__()
        assert n_chains in (2,3), "Only 2 or 3 chains supported in this class."
        self.n_chains = n_chains
        self.use_cross_linear = use_cross_linear

        self.linear_seq_a = nn.Linear(d_seq, d_seq)
        self.linear_seq_b = nn.Linear(d_seq, d_seq)

        self.linear_pair_a = nn.Linear(d_pair, d_pair)
        self.linear_pair_b = nn.Linear(d_pair, d_pair)

        self.joint_pair = JointPair(d_seq, d_pair)

        if n_chains == 3 and use_cross_linear:
            self.linear_seq_c = nn.Linear(d_seq, d_seq)
            self.linear_pair_c = nn.Linear(d_pair, d_pair)

            self.linear_pair_ab = nn.Linear(d_pair, d_pair)
            self.linear_pair_ba = nn.Linear(d_pair, d_pair)

            self.linear_pair_bc = nn.Linear(d_pair, d_pair)
            self.linear_pair_cb = nn.Linear(d_pair, d_pair)

    def forward(self, seq_repr_a, pair_repr_a, seq_repr_b,  pair_repr_b,
                seq_repr_c=None,  pair_repr_c=None, 
                pair_repr_ab=None,  pair_repr_ba=None,
                pair_repr_bc=None,  pair_repr_cb=None):
        
        seq_repr_a = self.linear_seq_a(seq_repr_a)
        seq_repr_b = self.linear_seq_b(seq_repr_b)
        
        pair_repr_a = self.linear_pair_a(pair_repr_a)
        pair_repr_b = self.linear_pair_b(pair_repr_b)

        if self.n_chains==2:
            if pair_repr_ab is None or pair_repr_ba is None:
                pair_repr_ab, pair_repr_ba = self.joint_pair(seq_repr_a, seq_repr_b)
            pair_repr = assemble_full_pair2(pair_repr_a, pair_repr_ab,
                                        pair_repr_ba, pair_repr_b)
            seq_repr = torch.cat([seq_repr_a,seq_repr_b],dim=1).contiguous()
        
        elif self.n_chains==3:
            assert seq_repr_c is not None and pair_repr_c is not None, \
            "For n_chains=3, seq_repr_c and pair_repr_c are required."
            seq_repr_c = self.linear_seq_c(seq_repr_c)
            pair_repr_c = self.linear_pair_c(pair_repr_c)

            if pair_repr_ab is None or pair_repr_ba is None:
                pair_repr_ab, pair_repr_ba = self.joint_pair(seq_repr_a, seq_repr_b)
            if pair_repr_bc is None or pair_repr_cb is None:
                pair_repr_bc, pair_repr_cb = self.joint_pair(seq_repr_b, seq_repr_c)
            pair_repr_ac, pair_repr_ca = self.joint_pair(seq_repr_a, seq_repr_c)
        
            if self.use_cross_linear:
                pair_repr_ab = self.linear_pair_ab(pair_repr_ab)
                pair_repr_ba = self.linear_pair_ba(pair_repr_ba)
                pair_repr_bc = self.linear_pair_bc(pair_repr_bc)
                pair_repr_cb = self.linear_pair_cb(pair_repr_cb)

            pair_repr = assemble_full_pair3(pair_repr_a, pair_repr_ab, pair_repr_ac,
                                pair_repr_ba, pair_repr_b, pair_repr_bc,
                                pair_repr_ca, pair_repr_cb, pair_repr_c)
            seq_repr = torch.cat([seq_repr_a,seq_repr_b, seq_repr_c],dim=1).contiguous()

        return seq_repr, pair_repr

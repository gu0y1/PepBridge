import torch
import torch.nn as nn

from .embedder import Embedder
from .pair_aware_block import PairAwareTrunk

class DecoderHead(nn.Module):
    def __init__(self, d_seq, aa_size, embedding_weights=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_seq, d_seq),
            nn.ReLU(),
            nn.LayerNorm(d_seq)
        )
        self.linear_decoder = nn.Linear(d_seq, aa_size)
        if embedding_weights is not None:
            self.linear_decoder.weight = nn.Parameter(embedding_weights)
        self.linear_decoder.bias = nn.Parameter(torch.zeros(aa_size))
    
    def forward(self, seq_repr):
        logits = self.linear_decoder(self.mlp(seq_repr))
        return logits

class MaskedLM(nn.Module):
    def __init__(self, aa_size, max_len, d_seq, d_head_seq, 
                 d_pair, d_head_pair, dropout, n_layers):
        super().__init__()
        self.aa_size = aa_size
        self.embedder = Embedder(aa_size, max_len, d_seq, d_pair, dropout, mhc=False)

        self.pair_aware_trunk = PairAwareTrunk(d_seq, d_head_seq, 
                                               d_pair, d_head_pair, 
                                               dropout, n_layers)

        self.decoder = DecoderHead(d_seq, aa_size, self.embedder.seq_aa_emb.weight)
    
    def forward(self, aa_seq, masked_label=None):
        mask = (aa_seq != 0).clone().detach()

        seq_repr, pair_repr = self.embedder(aa_seq)
        seq_repr, pair_repr = self.pair_aware_trunk(seq_repr, pair_repr, mask)

        logits = self.decoder(seq_repr)
        if masked_label is None:
            return logits
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            mlm_loss = loss_fct(logits.reshape(-1, self.aa_size), masked_label.reshape(-1))
            return logits, mlm_loss
    
    def _encode(self, aa_seq, mask):
        if mask is None:
            mask = (aa_seq != 0).clone().detach()
        seq_repr, pair_repr = self.embedder(aa_seq)
        seq_repr, pair_repr = self.pair_aware_trunk(seq_repr, pair_repr, mask)
        return seq_repr, pair_repr

class Encoder(nn.Module):
    def __init__(self, aa_size, max_len, d_seq, d_head_seq, 
                 d_pair, d_head_pair, dropout, 
                 n_layers, mhc=False):
        super().__init__()
        self.embedder = Embedder(aa_size, max_len, d_seq, d_pair, dropout, mhc)
        self.pair_aware_trunk = PairAwareTrunk(d_seq, d_head_seq, 
                                               d_pair, d_head_pair, 
                                               dropout, n_layers)
    def forward(self, aa_seq, mask, esm_mhc=None):
        seq_repr, pair_repr = self.embedder(aa_seq, esm_mhc)
        seq_repr, pair_repr = self.pair_aware_trunk(seq_repr, pair_repr, mask)
        return pair_repr, pair_repr




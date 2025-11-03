import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union

from .pair_aware_block import PairAwareTrunk
from .encoder import Encoder
from .joint_embedder import JointEmbedder
from .predicted_head import PredHead, ContactPredHead

HeadType = Union[torch.nn.Module, Callable[..., torch.Tensor]]

## max_len{mhc:34, peptide:15, cdr3:20}
## pair_aware_trunk n_layers 3 6 6 3 3 1

class PepBridge(nn.Module):
    def __init__(self, aa_size, max_len_dict, d_seq, d_head_seq, 
                 d_pair, d_head_pair, dropout, n_layers_dict, 
                 trbv_size):
        super().__init__()
        self.mhc_len = max_len_dict['mhc']
        self.pep_len = max_len_dict['peptide']
        self.cdr3_len = max_len_dict['cdr3']

        # Encoders 
        self.mhc_encoder = Encoder(
            aa_size, self.mhc_len, d_seq, d_head_seq, d_pair, d_head_pair, dropout,
            n_layers_dict['mhc'], mhc=True
        )
        self.peptide_encoder = Encoder(
            aa_size, self.pep_len, d_seq, d_head_seq, d_pair, d_head_pair, dropout,
            n_layers_dict['peptide']
        )
        self.cdr3_encoder = Encoder(
            aa_size, self.cdr3_len, d_seq, d_head_seq, d_pair, d_head_pair, dropout,
            n_layers_dict['cdr3']
        )

        # chain id & LayerNorm
        self.chain_id_embedder = nn.Embedding(3, d_seq)  # 0:MHC 1:PEP 2:CDR3
        self.mhc_ln  = nn.LayerNorm(d_seq)
        self.pep_ln  = nn.LayerNorm(d_seq)
        self.cdr3_ln = nn.LayerNorm(d_seq)

        # Joint embedders
        self.mp_joint_embedder = JointEmbedder(d_seq, d_pair, n_chains=2)
        self.pt_joint_embedder = JointEmbedder(d_seq, d_pair, n_chains=2)
        self.mpt_joint_embedder = JointEmbedder(d_seq, d_pair, n_chains=3)
        
        # Pair-aware trunks
        self.mp_pair_aware_trunk = PairAwareTrunk(d_seq, d_head_seq, d_pair, d_head_pair, dropout, n_layers_dict['mp'])
        self.pt_pair_aware_trunk = PairAwareTrunk(d_seq, d_head_seq, d_pair, d_head_pair, dropout, n_layers_dict['pt'])
        self.mpt_pair_aware_trunk = PairAwareTrunk(d_seq, d_head_seq, d_pair, d_head_pair, dropout, n_layers_dict['mpt'])
       
        # Heads
        self.mp_pred_head = PredHead(d_seq, d_pair, dropout)
        self.pt_pred_head = PredHead(d_seq, d_pair, dropout)
        self.mpt_pred_head = PredHead(d_seq, d_pair, dropout,trbv_size=trbv_size)
        self.imm_pred_head = PredHead(d_seq, d_pair, dropout)

        self.mp_contact_pred_head = ContactPredHead(d_pair, dropout)
        self.pt_contact_pred_head = ContactPredHead(d_pair, dropout)
    
    @staticmethod
    def _tokens_to_mask(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        # 0 为 pad
        return (x != 0).clone().detach()

    def aa_seq_encode(
        self,
        mhc: Optional[torch.Tensor] = None,
        peptide: Optional[torch.Tensor] = None,
        cdr3: Optional[torch.Tensor] = None,
        esm_mhc: Optional[torch.Tensor] = None,
    ):
        repr_dict, mask_dict, chain_id_dict = {}, {}, {}

        if mhc is not None:
            mhc_mask = self._tokens_to_mask(mhc).bool()
            mhc_chain_id = torch.zeros_like(mhc)
            mhc_seq, mhc_pair = self.mhc_encoder(mhc, mhc_mask, esm_mhc)
            mhc_seq = self.mhc_ln(mhc_seq + self.chain_id_embedder(mhc_chain_id))
            repr_dict['mhc_seq'], repr_dict['mhc_pair'] = mhc_seq, mhc_pair
            mask_dict['mhc'], chain_id_dict['mhc'] = mhc_mask, mhc_chain_id
        else:
            repr_dict['mhc_seq'] = repr_dict['mhc_pair'] = None
            mask_dict['mhc'] = chain_id_dict['mhc'] = None

        if peptide is not None:
            pep_mask = self._tokens_to_mask(peptide).bool()
            pep_chain_id = torch.ones_like(peptide)
            pep_seq, pep_pair = self.peptide_encoder(peptide, pep_mask)
            pep_seq = self.pep_ln(pep_seq + self.chain_id_embedder(pep_chain_id))
            repr_dict['pep_seq'], repr_dict['pep_pair'] = pep_seq, pep_pair
            mask_dict['pep'], chain_id_dict['pep'] = pep_mask, pep_chain_id
        else:
            repr_dict['pep_seq'] = repr_dict['pep_pair'] = None
            mask_dict['pep'] = chain_id_dict['pep'] = None

        if cdr3 is not None:
            cdr3_mask = self._tokens_to_mask(cdr3).bool()
            cdr3_chain_id = torch.full_like(cdr3, 2)  # 2
            cdr3_seq, cdr3_pair = self.cdr3_encoder(cdr3, cdr3_mask)
            cdr3_seq = self.cdr3_ln(cdr3_seq + self.chain_id_embedder(cdr3_chain_id))
            repr_dict['cdr3_seq'], repr_dict['cdr3_pair'] = cdr3_seq, cdr3_pair
            mask_dict['cdr3'], chain_id_dict['cdr3'] = cdr3_mask, cdr3_chain_id
        else:
            repr_dict['cdr3_seq'] = repr_dict['cdr3_pair'] = None
            mask_dict['cdr3'] = chain_id_dict['cdr3'] = None

        return repr_dict, mask_dict, chain_id_dict
    
    def mp_repr_encode(self, mhc_seq, mhc_pair, pep_seq, pep_pair, mhc_mask, pep_mask):
        seq, pair = self.mp_joint_embedder(mhc_seq, mhc_pair, pep_seq, pep_pair)
        mask = torch.cat([mhc_mask, pep_mask], dim=1).bool()
        seq, pair = self.mp_pair_aware_trunk(seq, pair, mask)
        return seq, pair

    def pt_repr_encode(self, pep_seq, pep_pair, cdr3_seq, cdr3_pair, pep_mask, cdr3_mask):
        seq, pair = self.pt_joint_embedder(pep_seq, pep_pair, cdr3_seq, cdr3_pair)
        mask = torch.cat([pep_mask, cdr3_mask], dim=1).bool()
        seq, pair = self.pt_pair_aware_trunk(seq, pair, mask)
        return seq, pair

    def mpt_repr_encode(
            self,         
            mp_seq, mp_pair,
            pt_seq, pt_pair,
            mhc_mask, pep_mask, cdr3_mask
        ):
        Lm, Lp, Lt = self.mhc_len, self.pep_len, self.cdr3_len

        # A=MHC
        mhc_seq = mp_seq[:, :Lm, :]
        mhc_pair = mp_pair[:, :Lm, :Lm, :]

        # B=PEP
        pep_seq_from_mp  = mp_seq[:, Lm:Lm+Lp, :]
        pep_pair_from_mp = mp_pair[:, Lm:Lm+Lp, Lm:Lm+Lp, :]

        pep_seq_from_pt  = pt_seq[:, :Lp, :]
        pep_pair_from_pt = pt_pair[:, :Lp, :Lp, :]

        pep_seq  = 0.5 * (pep_seq_from_mp  + pep_seq_from_pt)
        pep_pair = 0.5 * (pep_pair_from_mp + pep_pair_from_pt)

        # C=TCR
        cdr3_seq  = pt_seq[:, Lp:Lp+Lt, :]
        cdr3_pair = pt_pair[:, Lp:Lp+Lt, Lp:Lp+Lt, :]

        # pair_cross
        pair_ab = mp_pair[:, :Lm, Lm:Lm+Lp, :]  # M->P
        pair_ba = mp_pair[:, Lm:Lm+Lp, :Lm, :]  # P->M

        pair_bc = pt_pair[:, :Lp, Lp:Lp+Lt, :]  # P->T  
        pair_cb = pt_pair[:, Lp:Lp+Lt, :Lp, :]  # T->P  

        # MPT
        seq, pair = self.mpt_joint_embedder(
            seq_repr_a=mhc_seq, pair_repr_a=mhc_pair,
            seq_repr_b=pep_seq, pair_repr_b=pep_pair,
            seq_repr_c=cdr3_seq, pair_repr_c=cdr3_pair,
            pair_repr_ab=pair_ab, pair_repr_ba=pair_ba,
            pair_repr_bc=pair_bc, pair_repr_cb=pair_cb,
        )

        mask = torch.cat([mhc_mask, pep_mask, cdr3_mask], dim=1).bool()
        seq, pair = self.mpt_pair_aware_trunk(seq, pair, mask)
        return seq, pair 

    @staticmethod
    def _pred(
        seq_repr: torch.Tensor,          
        pair_repr: torch.Tensor,        
        chain_id_a: Optional[torch.Tensor] = None,   
        chain_id_b: Optional[torch.Tensor] = None, 
        mask_a: Optional[torch.Tensor] = None,     
        mask_b: Optional[torch.Tensor] = None,    
        pred_head: Optional[HeadType] = None,
        task: str = "binding",
        pair_repr_ab: Optional[torch.Tensor] = None,
        pair_repr_ba: Optional[torch.Tensor] = None, 
    ):
        if task not in ("binding", "contact"):
            raise ValueError('task must be binding or contact')
    
        if pred_head is None or not callable(pred_head):
            raise ValueError("`pred_head` must be a callable nn.Module or function.")

        if task == 'binding':
            if chain_id_a is None or chain_id_b is None:
                raise ValueError("binding task requires `chain_id_a` and `chain_id_b`.")
            
            if mask_a is None or mask_b is None:
                mask = torch.ones_like(seq_repr[:,:,0], dtype=torch.bool)
            else:
                mask = torch.cat([mask_a, mask_b],dim=-1).bool()
            chain_id = torch.cat([chain_id_a, chain_id_b],dim=-1)
            prob = pred_head(seq_repr, pair_repr, chain_id, mask)
            return prob
        else:
            prob, dist = pred_head(pair_repr_ab, pair_repr_ba)
            return prob, dist
        
    # -----------------------------
    # High-level tasks
    # -----------------------------
    def mp_pred(self, mhc, peptide, esm_mhc, contact: bool = False, immunogenicity: bool = False, repr_out: bool = False):
        repr_dict, mask_dict, chain_id_dict = self.aa_seq_encode(mhc=mhc, peptide=peptide, esm_mhc=esm_mhc)

        mp_seq, mp_pair = self.mp_repr_encode(
            repr_dict['mhc_seq'], repr_dict['mhc_pair'],
            repr_dict['pep_seq'], repr_dict['pep_pair'],
            mask_dict['mhc'], mask_dict['pep']
        )

        output={}
        binding_prob = self._pred(
            seq_repr=mp_seq, pair_repr=mp_pair,
            chain_id_a=chain_id_dict['mhc'], chain_id_b=chain_id_dict['pep'],
            mask_a=mask_dict['mhc'], mask_b=mask_dict['pep'],
            pred_head=self.mp_pred_head, task='binding'
        )
        output['binding_prob'] = binding_prob
        output['mask_dict'] = mask_dict
        output['chain_id_dict'] = chain_id_dict

        if contact:
            pair_ab = mp_pair[:, :self.mhc_len, self.mhc_len:self.mhc_len+self.pep_len, :]
            pair_ba = mp_pair[:, self.mhc_len:self.mhc_len+self.pep_len, :self.mhc_len, :]
            contact_prob, contact_dist = self._pred(
                seq_repr=None, pair_repr=None, pred_head=self.mp_contact_pred_head, task='contact',
                pair_repr_ab=pair_ab, pair_repr_ba=pair_ba
            )
            output['contact_prob']=contact_prob
            output['contact_dist']=contact_dist
            
        if immunogenicity:
            imm_prob = self._pred(
                seq_repr=mp_seq.detach(), pair_repr=mp_pair.detach(),
                chain_id_a=chain_id_dict['mhc'], chain_id_b=chain_id_dict['pep'],
                mask_a=mask_dict['mhc'], mask_b=mask_dict['pep'],
                pred_head=self.imm_pred_head, task='binding'
            )
            output['immunogenicity_prob'] = imm_prob

        if repr_out:
            output['seq_repr'], output['pair_repr'] = mp_seq, mp_pair
        return output
    
    def pt_pred(self, peptide, cdr3, contact: bool = False, repr_out: bool = False):
        repr_dict, mask_dict, chain_id_dict = self.aa_seq_encode(peptide=peptide, cdr3=cdr3)

        pt_seq, pt_pair = self.pt_repr_encode(
            repr_dict['pep_seq'], repr_dict['pep_pair'],
            repr_dict['cdr3_seq'], repr_dict['cdr3_pair'],
            mask_dict['pep'], mask_dict['cdr3']
        )

        output = {}
        binding_prob = self._pred(
            seq_repr=pt_seq, pair_repr=pt_pair,
            chain_id_a=chain_id_dict['pep'], chain_id_b=chain_id_dict['cdr3'],
            mask_a=mask_dict['pep'], mask_b=mask_dict['cdr3'],
            pred_head=self.pt_pred_head, task='binding'
        )
        output['binding_prob'] = binding_prob
        output['mask_dict'] = mask_dict    
        output['chain_id_dict'] = chain_id_dict

        if contact:
            pair_ab = pt_pair[:, :self.pep_len, self.pep_len:self.pep_len+self.cdr3_len, :]
            pair_ba = pt_pair[:, self.pep_len:self.pep_len+self.cdr3_len, :self.pep_len, :]
            contact_prob, contact_dist = self._pred(
                seq_repr=None, pair_repr=None, pred_head=self.pt_contact_pred_head, task='contact',
                pair_repr_ab=pair_ab, pair_repr_ba=pair_ba
            )
            output['contact_prob'] = contact_prob
            output['contact_dist'] = contact_dist

        if repr_out:
            output['seq_repr'], output['pair_repr'] = pt_seq, pt_pair
        return output

    def mpt_pred(self, mhc, peptide, cdr3, esm_mhc, trbv):
        mp_out = self.mp_pred(mhc, peptide, esm_mhc, contact=False, immunogenicity=False, repr_out=True)
        pt_out = self.pt_pred(peptide, cdr3, contact=False, repr_out=True)

        seq, pair = self.mpt_repr_encode(
            mp_out['seq_repr'], mp_out['pair_repr'],
            pt_out['seq_repr'], pt_out['pair_repr'],
            mp_out['mask_dict']['mhc'], mp_out['mask_dict']['pep'], pt_out['mask_dict']['cdr3']
        )

        chain_id = torch.cat([mp_out['chain_id_dict']['mhc'], mp_out['chain_id_dict']['pep'], pt_out['chain_id_dict']['cdr3']], dim=-1)
        mask = torch.cat([mp_out['mask_dict']['mhc'], mp_out['mask_dict']['pep'], pt_out['mask_dict']['cdr3']], dim=-1).bool()

        mpt_prob = self.mpt_pred_head(seq, pair, chain_id, mask, trbv)
        return {
            'mpt_prob': mpt_prob,
            'mp_prob': mp_out['binding_prob'],
            'pt_prob': pt_out['binding_prob'],
        }
    
    def pep_align(self, mhc, peptide, cdr3, esm_mhc):
        """Alignment of peptide representation"""
        repr_dict, mask_dict, _ = self.aa_seq_encode(mhc=mhc, peptide=peptide, cdr3=cdr3, esm_mhc=esm_mhc)

        mp_seq, mp_pair = self.mp_repr_encode(
            repr_dict['mhc_seq'], repr_dict['mhc_pair'],
            repr_dict['pep_seq'], repr_dict['pep_pair'],
            mask_dict['mhc'], mask_dict['pep']
        )
        pt_seq, pt_pair = self.pt_repr_encode(
            repr_dict['pep_seq'], repr_dict['pep_pair'],
            repr_dict['cdr3_seq'], repr_dict['cdr3_pair'],
            mask_dict['pep'], mask_dict['cdr3']
        )

        pep_seq_mp  = mp_seq[:, self.mhc_len:self.mhc_len+self.pep_len, :]
        pep_pair_mp = mp_pair[:, self.mhc_len:self.mhc_len+self.pep_len, self.mhc_len:self.mhc_len+self.pep_len, :]

        pep_seq_pt  = pt_seq[:, :self.pep_len, :]
        pep_pair_pt = pt_pair[:, :self.pep_len, :self.pep_len, :]

        pep_mask = mask_dict['pep'].bool()                     # [B, Lp]
        seq_align_loss = F.mse_loss(pep_seq_mp[pep_mask], pep_seq_pt[pep_mask], reduction='mean')

        pair_mask = (pep_mask.unsqueeze(2) & pep_mask.unsqueeze(1))  # [B, Lp, Lp]
        pair_align_loss = F.mse_loss(pep_pair_mp[pair_mask], pep_pair_pt[pair_mask], reduction='mean')

        return seq_align_loss + pair_align_loss
    
    def forward(self, mhc, peptide, cdr3, esm_mhc, trbv):
        mp_out = self.mp_pred(mhc, peptide, esm_mhc, contact=True, immunogenicity=True, repr_out=True)
        pt_out = self.pt_pred(peptide, cdr3, contact=True, repr_out=True)

        seq, pair = self.mpt_repr_encode(
            mp_out['seq_repr'], mp_out['pair_repr'],
            pt_out['seq_repr'], pt_out['pair_repr'],
            mp_out['mask_dict']['mhc'], mp_out['mask_dict']['pep'], pt_out['mask_dict']['cdr3']
        )

        chain_id = torch.cat([mp_out['chain_id_dict']['mhc'], mp_out['chain_id_dict']['pep'], pt_out['chain_id_dict']['cdr3']], dim=-1)
        mask = torch.cat([mp_out['mask_dict']['mhc'], mp_out['mask_dict']['pep'], pt_out['mask_dict']['cdr3']], dim=-1).bool()

        mpt_prob = self.mpt_pred_head(seq, pair, chain_id, mask, trbv)

        return {
            'mpt_prob': mpt_prob,
            'mp_prob': mp_out['binding_prob'],
            'pt_prob': pt_out['binding_prob'],
            'mp_contact_dist': mp_out.get('contact_dist', None),
            'mp_contact_prob': mp_out.get('contact_prob', None),
            'pt_contact_dist': pt_out.get('contact_dist', None),
            'pt_contact_prob': pt_out.get('contact_prob', None),
            'immunogenicity_prob': mp_out.get('immunogenicity_prob', None),
        }



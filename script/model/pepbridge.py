import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from .pair_aware_block import PairAwareTrunk
from .encoder import Encoder
from .joint_embedder import JointEmbedder
from .predicted_head import PredHead, ContactPredHead
from ..loss import vicreg

HeadType = Union[torch.nn.Module, Callable[..., torch.Tensor]] 

class PepBridge(nn.Module):
    def __init__(self, aa_size, max_len_dict, d_seq, d_head_seq, 
                 d_pair, d_head_pair, dropout, n_layers_dict, 
                 trbv_size, trav_size=None, traj_size=None):
        super().__init__()
        self.mhc_len = max_len_dict['mhc']
        self.pep_len = max_len_dict['peptide']
        self.cdr3_len = max_len_dict['cdr3']

        #finetune
        if trav_size is not None:
            self.trav_emb = nn.Embedding(num_embeddings=trav_size, 
                                         embedding_dim=48, 
                                         padding_idx=0)
        if traj_size is not None:
            self.traj_emb = nn.Embedding(num_embeddings=traj_size, 
                                         embedding_dim=48, 
                                         padding_idx=0)
        n_extra = int(trav_size is not None) + int(traj_size is not None)
        extra_dim = 48 * n_extra

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
        self.mpt_pred_head = PredHead(d_seq, d_pair, dropout, trbv_size=trbv_size, extra_dim=extra_dim)
        self.imm_pred_head = PredHead(d_seq, d_pair, dropout)

        self.mp_contact_pred_head = ContactPredHead(d_pair, dropout, dist_bin=9)
        self.pt_contact_pred_head = ContactPredHead(d_pair, dropout)
        
        #projector
        # self.pep_seq_proj = SharedProj(d_seq)
        # self.pep_pair_proj = SharedProj(d_pair)
    
    @staticmethod
    def _tokens_to_mask(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        # padding token id :0
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
            output['immunogenicity_prob'] = binding_prob - F.softplus(imm_prob)

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

        mpt_logit = self.mpt_pred_head(seq, pair, chain_id, mask, trbv)
        min_logit = torch.minimum(mp_out['binding_prob'], pt_out['binding_prob'])
        mpt_prob = min_logit - F.softplus(mpt_logit) 
        return {
            'mpt_prob': mpt_prob,
            'mp_prob': mp_out['binding_prob'],
            'pt_prob': pt_out['binding_prob'],
        }

    def mpt_pred_finetune(self, mhc, peptide, cdr3, esm_mhc, trbv, 
                          trav=None, traj=None):
        mp_out = self.mp_pred(mhc, peptide, esm_mhc, contact=False, immunogenicity=False, repr_out=True)
        pt_out = self.pt_pred(peptide, cdr3, contact=False, repr_out=True)

        seq, pair = self.mpt_repr_encode(
            mp_out['seq_repr'], mp_out['pair_repr'],
            pt_out['seq_repr'], pt_out['pair_repr'],
            mp_out['mask_dict']['mhc'], mp_out['mask_dict']['pep'], pt_out['mask_dict']['cdr3']
        )

        chain_id = torch.cat([mp_out['chain_id_dict']['mhc'], mp_out['chain_id_dict']['pep'], pt_out['chain_id_dict']['cdr3']], dim=-1)
        mask = torch.cat([mp_out['mask_dict']['mhc'], mp_out['mask_dict']['pep'], pt_out['mask_dict']['cdr3']], dim=-1).bool()
        
        av_emb =  self.trav_emb(trav) if trav is not None else None
        aj_emb =  self.traj_emb(traj) if traj is not None else None

        if av_emb is not None and aj_emb is not None:
            extra_emb = torch.cat([av_emb, aj_emb], dim=-1) 
        elif av_emb is not None:
            extra_emb = av_emb
        elif aj_emb is not None:
            extra_emb = aj_emb
        else:
            extra_emb = None

        mpt_logit = self.mpt_pred_head(seq, pair, chain_id, mask, trbv, extra_emb)
        return mpt_logit
    
    def pep_align(self, mhc, peptide, cdr3, esm_mhc, all_align=-1, ln=False):
        """Alignment of peptide representation"""
        repr_dict, mask_dict, _ = self.aa_seq_encode(
            mhc=mhc, peptide=peptide, cdr3=cdr3, esm_mhc=esm_mhc
        )

        # MP trunk
        mp_seq, mp_pair = self.mp_joint_embedder(
            repr_dict['mhc_seq'], repr_dict['mhc_pair'], 
            repr_dict['pep_seq'], repr_dict['pep_pair'],
            mask_dict['mhc'], mask_dict['pep']
        )
        mp_mask = torch.cat([mask_dict['mhc'], mask_dict['pep']], dim=1).bool()
        _, _, mp_seq_list, mp_pair_list = self.mp_pair_aware_trunk(
            mp_seq, mp_pair, mp_mask, return_hidden=True
        )

        # PT trunk
        pt_seq, pt_pair = self.pt_joint_embedder(
            repr_dict['pep_seq'], repr_dict['pep_pair'],
            repr_dict['cdr3_seq'], repr_dict['cdr3_pair'],
            mask_dict['pep'], mask_dict['cdr3']
        )
        pt_mask = torch.cat([mask_dict['pep'], mask_dict['cdr3']], dim=1).bool()
        _, _, pt_seq_list, pt_pair_list = self.pt_pair_aware_trunk(
            pt_seq, pt_pair, pt_mask, return_hidden=True
        )

        pep_mask = mask_dict['pep'].bool()              # [B, Lp]
        pair_mask = pep_mask[:, :, None] & pep_mask[:, None, :]  # [B, Lp, Lp]

        token_mask_flat = pep_mask.view(-1)             # [B*Lp]
        pair_mask_flat  = pair_mask.view(-1)            # [B*Lp*Lp]

        align_loss = None
        count = 0.0
        
        n_layers = len(mp_seq_list)
        k = abs(int(all_align)) if all_align is not None else 1
        if k == 0:
            k = 1
        if k >= n_layers:
            layer_indices = range(n_layers)              
        else:
            layer_indices = range(n_layers - k, n_layers)

        for i in layer_indices:
            pep_seq_mp  = mp_seq_list[i][:, self.mhc_len:self.mhc_len+self.pep_len, :]
            pep_pair_mp = mp_pair_list[i][:, self.mhc_len:self.mhc_len+self.pep_len,
                                            self.mhc_len:self.mhc_len+self.pep_len, :]
            pep_seq_mp  = pep_seq_mp.detach()
            pep_pair_mp = pep_pair_mp.detach()

            pep_seq_pt  = pt_seq_list[i][:, :self.pep_len, :]
            pep_pair_pt = pt_pair_list[i][:, :self.pep_len, :self.pep_len, :]

            x_mp = pep_seq_mp.reshape(-1, pep_seq_mp.size(-1))[token_mask_flat]  # [N_tok, D]
            x_pt = pep_seq_pt.reshape(-1, pep_seq_pt.size(-1))[token_mask_flat]  # [N_tok, D]

            if x_mp.size(0) < 2:
                continue

            with autocast(enabled=False,device_type='cuda'):
                if ln:
                    x_mp = F.layer_norm(x_mp, (x_mp.size(-1),))
                    x_pt = F.layer_norm(x_pt, (x_pt.size(-1),))
                seq_align_loss = vicreg(x_mp.float(), x_pt.float())

                if pair_mask_flat.any():
                    y_mp = pep_pair_mp.reshape(-1, pep_pair_mp.size(-1))[pair_mask_flat]  # [N_pair, Dp]
                    y_pt = pep_pair_pt.reshape(-1, pep_pair_pt.size(-1))[pair_mask_flat]
                    if ln:
                        y_mp = F.layer_norm(y_mp, (y_mp.size(-1),))
                        y_pt = F.layer_norm(y_pt, (y_pt.size(-1),))
                    pair_align_loss = F.mse_loss(y_mp.float(), y_pt.float())
                else:
                    pair_align_loss = x_mp.new_zeros([])

            cur_loss = 0.5 * (seq_align_loss + pair_align_loss)
            if align_loss is None:
                align_loss = cur_loss
            else:
                align_loss = align_loss + cur_loss
            count += 1.0
        if align_loss is None or count == 0:
            return torch.zeros([], device=pep_seq_pt.device, dtype=pep_seq_pt.dtype)

        align_loss = align_loss / count
        return align_loss
      
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



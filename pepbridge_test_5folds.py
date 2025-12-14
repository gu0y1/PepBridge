from script.model.pepbridge import PepBridge
from script.model.lora import build_model_with_lora
from script.dataset import PTDataSet, MPDataSet, MPTDataSet
from script.dataprocess import mk_aa_dict, mk_bv_dict
from script.utils import model_fn, setup_logger
from script.metric import binary_evaluate_metrics, distance_evaluate_metrics

import pandas as pd 
import numpy as np
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def model_inference(models, dataloader, task, device):
    if isinstance(models, (list, tuple)):
        model_list = list(models)
    else:
        model_list = [models]

    if task in ('mp_contact', 'pt_contact'):
        h = 34 if task == 'mp_contact' else 15
        w = 15 if task == 'mp_contact' else 20
        pred_dist = np.zeros((dataloader.dataset.shape[0], h, w))
        pred_prob = np.zeros((dataloader.dataset.shape[0], h, w))
        gt_dist = np.zeros((dataloader.dataset.shape[0], h, w))
        gt_prob = np.zeros((dataloader.dataset.shape[0], h, w))
        arr_mask = np.zeros((dataloader.dataset.shape[0], h, w))
    else:
        arr_prob = np.zeros((dataloader.dataset.shape[0], 1))

    for m in model_list:
        m.eval()
        
    with torch.no_grad():
        for b, idx in dataloader:
            if task in ('mp', 'mp_contact', 'imm'):
                mhc = b["mhc"].to(device)
                peptide = b["peptide"].to(device)
                esm_mhc = b.get("esm_mhc", None)
                if esm_mhc is not None:
                    esm_mhc = esm_mhc.to(device)

                out0 = model_list[0].mp_pred(
                    mhc, peptide, esm_mhc,
                    contact=(task == 'mp_contact'),
                    immunogenicity=(task == 'imm')
                )
                if task in ('mp_contact', 'pt_contact', 'mp'):  
                    mask_1 = out0["mask_dict"]["mhc"]
                    mask_2 = out0["mask_dict"]["pep"]

            elif task in ('pt', 'pt_contact'):
                peptide = b["peptide"].to(device)
                cdr3 = b["cdr3"].to(device)

                out0 = model_list[0].pt_pred(
                    peptide, cdr3,
                    contact=(task == 'pt_contact')
                )
                if task in ('pt_contact', 'pt'):
                    mask_1 = out0["mask_dict"]["pep"]
                    mask_2 = out0["mask_dict"]["cdr3"]

            elif task == 'mpt':
                mhc = b["mhc"].to(device)
                peptide = b["peptide"].to(device)
                cdr3 = b["cdr3"].to(device)
                esm_mhc = b.get("esm_mhc", None)
                if esm_mhc is not None:
                    esm_mhc = esm_mhc.to(device)
                trbv = b.get("trbv", None)
                if trbv is not None:
                    trbv = trbv.to(device)

            else:
                raise KeyError('task must in [mp, mp_contact, imm, pt, pt_contact, mpt]')

            if task in ('mp_contact', 'pt_contact'):
                pair_mask = mask_1.unsqueeze(2) & mask_2.unsqueeze(1)

                if task == 'pt_contact':
                    p_bin = b.get("contact_pt_bin", None).to(device)      # [B, H, W]
                    p_dis = b.get("contact_pt_dist", None).to(device)     # [B, H, W]
                else:
                    p_bin = b.get("contact_mp_bin", None).to(device)
                    p_dis = b.get("contact_mp_dist", None).to(device)

                gt_dist[idx] = p_dis.detach().cpu().numpy()
                gt_prob[idx] = p_bin.detach().cpu().numpy()
                arr_mask[idx] = pair_mask.detach().cpu().numpy()

                contact_logits_list = []
                contact_dist_list = []

                for i, m in enumerate(model_list):
                    if task == 'mp_contact':
                        out = m.mp_pred(
                            mhc, peptide, esm_mhc,
                            contact=True,
                            immunogenicity=False
                        )
                    else:
                        out = m.pt_pred(
                            peptide, cdr3,
                            contact=True
                        )

                    contact_logits_list.append(out['contact_prob'])   # [B,H,W] logits
                    contact_dist_list.append(out['contact_dist'])     # [B,H,W]

                # [M,B,H,W] -> [B,H,W]
                contact_logits = torch.stack(contact_logits_list, dim=0).mean(dim=0)
                contact_dist = torch.stack(contact_dist_list, dim=0).mean(dim=0)

                pred_dist[idx] = contact_dist.detach().cpu().numpy()
                pred_prob[idx] = torch.sigmoid(contact_logits).detach().cpu().numpy()

            elif task in ('mp', 'pt', 'imm', 'mpt'):
                if task in ('mp', 'pt'):
                    logit_key = 'binding_prob'
                elif task == 'imm':
                    logit_key = 'immunogenicity_prob'
                elif task == 'mpt':
                    logit_key = 'mpt_prob'
                else:
                    raise RuntimeError("Unexpected task")

                logits_list = []

                for m in model_list:
                    if task in ('mp', 'imm'):
                        out = m.mp_pred(
                            mhc, peptide, esm_mhc,
                            contact=False,
                            immunogenicity=(task == 'imm')
                        )
                    elif task == 'pt':
                        out = m.pt_pred(
                            peptide, cdr3,
                            contact=False
                        )
                    elif task == 'mpt':
                        out = m.mpt_pred(
                            mhc, peptide, cdr3, esm_mhc, trbv
                        )
                    else:
                        raise RuntimeError("Unexpected task")

                    # out[logit_key]: [B, something]
                    logit = out[logit_key].mean(dim=1, keepdim=True)   # => [B,1]
                    logits_list.append(logit)

                # [M,B,1] -> [B,1]
                ensemble_logit = torch.stack(logits_list, dim=0).mean(dim=0)
                pred = torch.sigmoid(ensemble_logit).detach().cpu().numpy()
                arr_prob[idx] = pred

    if task in ('mp_contact', 'pt_contact'):
        out_dict = {
            'mask': arr_mask,
            'gt_prob': gt_prob,
            'gt_dist': gt_dist,
            'pred_dist': pred_dist,
            'pred_prob': pred_prob,
        }
        return out_dict
    else:
        return arr_prob

def parse_kv_args(argv):
    out = {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            out[k] = v
    return out


def str2bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y", "t")


if __name__ == "__main__":
    os.chdir('/data5/tem/laiwp131/pMHC_TCR_20251103/')
    logger = setup_logger()
    
    cli_args = parse_kv_args(sys.argv)

    default_ckpt_path = './checkpoints_multi_lora_align_all/phase_C.pt'
    paths_arg = cli_args.get("paths", None)
    if paths_arg is not None:
        ckpt_paths = [p for p in paths_arg.split(",") if p.strip()]
    else:
        single_path = cli_args.get("path", default_ckpt_path)
        ckpt_paths = [single_path]

    use_lora = str2bool(cli_args.get("use_lora", "true"))

    aa_dict = mk_aa_dict()
    bv_dcit = mk_bv_dict()

    device = 'cuda'
    models = []
    for p in ckpt_paths:
        logger.info(f"Loading checkpoint: {p}")
        base_model = model_fn(
            aa_vocab_size=len(aa_dict),
            trbv_vocab_size=len(bv_dcit)
        )

        if use_lora:
            base_model = build_model_with_lora(
                base_model,
                last_n=2,
                cfg_seq_pair=((8, 16), (4, 8)),
                dropout=0.1,
                freeze_base=True,
                print_trainabel=False
            )

        ckpt = torch.load(p, map_location=device)
        base_model.load_state_dict(ckpt['model_state'], strict=True)
        base_model.to(device)
        base_model.eval()
        models.append(base_model)

    pt_df = pd.read_csv('pt_test.csv', header=0, index_col=0)
    mp_df = pd.read_csv('External_mp_test.csv', header=0, index_col=0)
    mp_df_NetMHCpan = pd.read_csv('NetMHCpan_mp_test.csv', header=0, index_col=0)
    mpt_df = pd.read_csv('mpt_test.csv', header=0, index_col=0)
    imm_df = pd.read_csv('immunogenicity_test.csv', header=0, index_col=0)
    mp_contact_df = pd.read_csv('./PepBridge-main/data/mp_pdb_test.csv',  header=0, index_col=0)
    pt_contact_df = pd.read_csv('./PepBridge-main/data/pt_pdb_test.csv', header=0, index_col=0)

    mp_loader = DataLoader(MPDataSet(mp_df=mp_df, mhc_type='HLAI', 
                                mhc_max_len=34, pep_max_len=15,
                                binding=True, 
                                immunogenicity=False, contact=False, mask=None),
                                batch_size=128, shuffle=False, drop_last=False)
    
    mp_loader2 = DataLoader(MPDataSet(mp_df=mp_df_NetMHCpan, mhc_type='HLAI', 
                                mhc_max_len=34, pep_max_len=15,
                                binding=True, 
                                immunogenicity=False, contact=False, mask=None),
                                batch_size=128, shuffle=False, drop_last=False)
    
    pt_loader = DataLoader(
        dataset=PTDataSet(pt_df, pep_max_len=15, cdr3_max_len=20,
                                binding=True, contact=False, 
                                pep_mask=None, cdr3_mask=None),
                batch_size=128, shuffle=False, drop_last=False
    )

    mpt_loader=DataLoader(
        dataset=MPTDataSet(mpt_df, mhc_type='HLAI', 
                mhc_max_len=34, pep_max_len=15, cdr3_max_len=20,
                bv=True, binding=True, 
                pep_mask=None, cdr3_mask=None),
        batch_size=128, shuffle=False, drop_last=False
    )

    imm_loader = DataLoader(
        MPDataSet(mp_df=imm_df, mhc_type='HLAI', 
                mhc_max_len=34, pep_max_len=15,
                binding=False, 
                immunogenicity=True, contact=False, mask=None),
            batch_size=128, shuffle=False, drop_last=False)

    mp_contact_loader = DataLoader(
        MPDataSet(mp_df=mp_contact_df, mhc_type='HLAI', 
                mhc_max_len=34, pep_max_len=15,
                binding=False, 
                immunogenicity=False, contact=True, mask=None),
                batch_size=8, shuffle=False, drop_last=False)

    pt_contact_loader = DataLoader(
        PTDataSet(pt_contact_df, 
                pep_max_len=15, cdr3_max_len=20,
                binding=False, contact=True, 
                pep_mask=None, cdr3_mask=None),
                batch_size=8, shuffle=False, drop_last=False)

    

    logger.info('NetMHCpan_mp_binding:')
    mp_df_NetMHCpan['pred'] = model_inference(models, mp_loader2, 'mp', device)
    mp_df_NetMHCpan.loc[mp_df_NetMHCpan['len'] >= 12, 'len'] = '12-15'
    logger.info(binary_evaluate_metrics(mp_df_NetMHCpan['binding'],mp_df_NetMHCpan['pred']))
    logger.info(binary_evaluate_metrics(mp_df_NetMHCpan['binding'],mp_df_NetMHCpan['pred'],group=mp_df_NetMHCpan['len']))
    logger.info(binary_evaluate_metrics(mp_df_NetMHCpan['binding'],mp_df_NetMHCpan['pred'],group=mp_df_NetMHCpan['MHC']))
    
    logger.info('mp_binding:')
    mp_df['pred'] = model_inference(models, mp_loader, 'mp', device)
    mp_df.loc[mp_df['len'] >= 12, 'len'] = '12-15'
    logger.info(binary_evaluate_metrics(mp_df['binding'],mp_df['pred']))
    logger.info(binary_evaluate_metrics(mp_df['binding'],mp_df['pred'],group=mp_df['len']))
    logger.info(binary_evaluate_metrics(mp_df['binding'],mp_df['pred'],group=mp_df['MHC']))

    logger.info('pt_binding:')
    pt_df['pred'] = model_inference(models, pt_loader, 'pt', device)
    logger.info(binary_evaluate_metrics(pt_df['binding'],pt_df['pred']))
    logger.info(binary_evaluate_metrics(pt_df['binding'],pt_df['pred'],group=pt_df['peptide_category']))

    logger.info('mpt_binding:')
    mpt_df['pred'] = model_inference(models, mpt_loader, 'mpt', device)
    logger.info(binary_evaluate_metrics(mpt_df['binding'],mpt_df['pred']))
    logger.info(binary_evaluate_metrics(mpt_df['binding'],mpt_df['pred'],group=mpt_df['peptide_category']))

    logger.info('imm:')
    imm_df['pred'] = model_inference(models, imm_loader, 'imm', device)
    logger.info(binary_evaluate_metrics(imm_df['immunogenicity'], imm_df['pred']))

    logger.info('mp_contact:')
    mp_contact_out = model_inference(models, mp_contact_loader, 'mp_contact', device)
    logger.info(binary_evaluate_metrics(mp_contact_out['gt_prob'], 
                                        mp_contact_out['pred_prob'], 
                                        mp_contact_out['mask']))
    logger.info(distance_evaluate_metrics(mp_contact_out['gt_dist'], 
                                        mp_contact_out['pred_dist'], 
                                        mp_contact_out['mask'],
                                        distogram=True))
                                        
    logger.info('pt_contact:')
    pt_contact_out = model_inference(models, pt_contact_loader, 'pt_contact', device)
    logger.info(binary_evaluate_metrics(pt_contact_out['gt_prob'], 
                                        pt_contact_out['pred_prob'], 
                                        pt_contact_out['mask']))
    logger.info(distance_evaluate_metrics(pt_contact_out['gt_dist'], 
                                        pt_contact_out['pred_dist'], 
                                        pt_contact_out['mask']))

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
import glob

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
        if task == 'mp_contact':
            pred_dist = np.zeros((dataloader.dataset.shape[0], h, w, 9))
        else:
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
        return arr_prob.squeeze(1)

def parse_kv_args(argv):
    out = {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            out[k] = v
    return out

def str2bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y", "t")

def expand_ckpt_paths_from_arg(p: str, phase_name: str = "phase_C.pt", folds=range(1, 6)):
    p = os.path.expanduser(p.strip())
    if not p:
        return []

    if os.path.isdir(p):
        out = []
        for i in folds:
            cand = os.path.join(p, f"fold_{i}", phase_name)
            if os.path.exists(cand):
                out.append(cand)
        if not out:
            out = sorted(glob.glob(os.path.join(p, "fold*", phase_name)))
        return out
    
    return [p]

if __name__ == "__main__":
    BASE = os.path.expanduser("~/pMHC_TCR")
    CODE = os.path.join(BASE, "code")
    DATA = os.path.join(BASE, "data")

    logger = setup_logger()
    
    cli_args = parse_kv_args(sys.argv)
    
    default_ckpt_path = os.path.join(BASE, "checkpoints_multi_lora_align3_ln/")
    paths_arg = cli_args.get("paths", None)
    if paths_arg:
        raw_paths = [p.strip() for p in paths_arg.split(",") if p.strip()]
    else:
        raw_paths = [cli_args.get("path", None) or default_ckpt_path]
    ckpt_paths = []
    for rp in raw_paths:
        ckpt_paths.extend(expand_ckpt_paths_from_arg(rp, phase_name="phase_C.pt", folds=range(1, 6)))

    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No checkpoints found from: {raw_paths}")

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
        base_model.to(device)
        ckpt = torch.load(p, map_location=device)
        base_model.load_state_dict(ckpt['model_state'], strict=True)
        base_model.eval()
        models.append(base_model)
    #    
    pt_hard_df = pd.read_csv(os.path.join(DATA,'pt_test_hard.csv'), header=0, index_col=0).reset_index(drop=True)
    pt_random_df = pd.read_csv(os.path.join(DATA,'pt_test_random.csv'), header=0, index_col=0).reset_index(drop=True)
    Benchmark_pt_df = pd.read_csv(os.path.join(DATA,'TCREpitopeBenchmark_pt_test.csv'), header=0, index_col=0).reset_index(drop=True)
    TRAIT_pt_pep_mute = pd.read_csv(os.path.join(DATA,'TRAIT_pt_pep_mute.csv'), header=0, index_col=0).reset_index(drop=True)
    #
    External_mp_df = pd.read_csv(os.path.join(DATA,'External_mp_test.csv'), header=0, index_col=0).reset_index(drop=True)
    mp_df = pd.read_csv(os.path.join(DATA,'mp_test2.csv'), header=0, index_col=0).reset_index(drop=True)
    #
    mpt_df = pd.read_csv(os.path.join(DATA,'mpt_test.csv'), header=0, index_col=0).reset_index(drop=True)
    Benchmark_mpt_df = pd.read_csv(os.path.join(DATA,'TCREpitopeBenchmark_mpt_test.csv'), header=0, index_col=0).reset_index(drop=True)
    #
    imm_df = pd.read_csv(os.path.join(DATA,'immunogenicity_test.csv'), header=0, index_col=0).reset_index(drop=True)
    mp_contact_df = pd.read_csv(os.path.join(DATA,'mp_pdb_test1.csv'),  header=0, index_col=0).reset_index(drop=True)
    pt_contact_df = pd.read_csv(os.path.join(DATA,'pt_pdb_test.csv'), header=0, index_col=0).reset_index(drop=True)
####
    mp_loader = DataLoader(MPDataSet(mp_df=mp_df, mhc_type='HLAI', 
                                mhc_max_len=34, pep_max_len=15,
                                binding=True, 
                                immunogenicity=False, contact=False, mask=None),
                                batch_size=128, shuffle=False, drop_last=False)
    
    External_mp_loader = DataLoader(MPDataSet(mp_df=External_mp_df, mhc_type='HLAI', 
                                mhc_max_len=34, pep_max_len=15,
                                binding=True, 
                                immunogenicity=False, contact=False, mask=None),
                                batch_size=128, shuffle=False, drop_last=False)
    
    pt_hard_loader = DataLoader(
        dataset=PTDataSet(pt_hard_df, pep_max_len=15, cdr3_max_len=20,
                                binding=True, contact=False, 
                                pep_mask=None, cdr3_mask=None),
                batch_size=128, shuffle=False, drop_last=False
    )

    pt_random_loader = DataLoader(
        dataset=PTDataSet(pt_random_df, pep_max_len=15, cdr3_max_len=20,
                                binding=True, contact=False, 
                                pep_mask=None, cdr3_mask=None),
                batch_size=128, shuffle=False, drop_last=False
    )
    Benchmark_pt_loader = DataLoader(
        dataset=PTDataSet(Benchmark_pt_df, pep_max_len=15, cdr3_max_len=20,
                                binding=True, contact=False, 
                                pep_mask=None, cdr3_mask=None),
                batch_size=128, shuffle=False, drop_last=False
    )
    TRAIT_pt_pep_mute_loader = DataLoader(
        dataset=PTDataSet(TRAIT_pt_pep_mute, pep_max_len=15, cdr3_max_len=20,
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
    Benchmark_mpt_loader=DataLoader(
        dataset=MPTDataSet(Benchmark_mpt_df, mhc_type='HLAI', 
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

    
    #########
    logger.info('Independent_mp_binding:')
    mp_df['pred'] = model_inference(models, mp_loader, 'mp', device)
    logger.info(binary_evaluate_metrics(mp_df['binding'],mp_df['pred']))
    
    logger.info('External_mp_binding:')
    External_mp_df['pred'] = model_inference(models, External_mp_loader, 'mp', device)
    logger.info(binary_evaluate_metrics(External_mp_df['binding'],External_mp_df['pred']))
    ########
    logger.info('pt_random_binding:')
    pt_random_df['pred'] = model_inference(models, pt_random_loader, 'pt', device)
    logger.info(binary_evaluate_metrics(pt_random_df['binding'],pt_random_df['pred']))
    logger.info(binary_evaluate_metrics(pt_random_df['binding'],pt_random_df['pred'],
                                        group=pt_random_df['peptide_category']))
    
    logger.info('pt_hard_binding:')
    pt_hard_df['pred'] = model_inference(models, pt_hard_loader, 'pt', device)
    logger.info(binary_evaluate_metrics(pt_hard_df['binding'],pt_hard_df['pred']))
    logger.info(binary_evaluate_metrics(pt_hard_df['binding'],pt_hard_df['pred'],
                                        group=pt_hard_df['peptide_category']))
    
    logger.info('Benchmark_pt:')
    Benchmark_pt_df['pred'] = model_inference(models, Benchmark_pt_loader, 'pt', device)
    logger.info(binary_evaluate_metrics(Benchmark_pt_df['binding'],Benchmark_pt_df['pred']))


    logger.info('TRAIT_pt_pep_mute:')
    TRAIT_pt_pep_mute['pred'] = model_inference(models, TRAIT_pt_pep_mute_loader, 'pt', device)
    logger.info(binary_evaluate_metrics(TRAIT_pt_pep_mute['binding'],TRAIT_pt_pep_mute['pred']))

    
    #####
    logger.info('mpt_binding:')
    mpt_df['pred'] = model_inference(models, mpt_loader, 'mpt', device)
    logger.info(binary_evaluate_metrics(mpt_df['binding'],mpt_df['pred']))
    logger.info(binary_evaluate_metrics(mpt_df['binding'],mpt_df['pred'],
                                        group=mpt_df['peptide_category']))
    logger.info('avg pre-peptide:')
    logger.info(binary_evaluate_metrics(mpt_df['binding'],mpt_df['pred'],
                                        group=mpt_df['peptide'],mean=True))
    logger.info('Benchmark_mpt:')
    Benchmark_mpt_df['pred'] = model_inference(models, Benchmark_mpt_loader, 'mpt', device)
    logger.info(binary_evaluate_metrics(Benchmark_mpt_df['binding'],Benchmark_mpt_df['pred']))

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

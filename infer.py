from script.model.pepbridge import PepBridge
from script.model.lora import build_model_with_lora
from script.dataset import PTDataSet, MPDataSet, MPTDataSet
from script.dataprocess import mk_aa_dict, mk_bv_dict
from script.utils import model_fn, setup_logger
from script.metric import dist_pred_from_logits_np

import pandas as pd
import numpy as np
import os
import sys
import glob
import re

import torch
from torch.utils.data import DataLoader


# -----------------------------
# basic utils
# -----------------------------
def parse_kv_args(argv):
    out = {}
    for arg in argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            out[k] = v
    return out


def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def parse_tasks(task_str: str):
    if not task_str:
        raise ValueError(
            "Please provide task, e.g. task=mp or task=mp,pt,mpt,mp_contact,pt_contact,imm"
        )

    raw = task_str.replace(",", " ").split()
    tasks = [x.strip() for x in raw if x.strip()]

    valid = {"mp", "pt", "imm", "mpt", "mp_contact", "pt_contact"}
    bad = [x for x in tasks if x not in valid]
    if bad:
        raise ValueError(f"Unsupported task(s): {bad}. Valid tasks: {sorted(valid)}")

    seen = set()
    out = []
    for t in tasks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


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


def safe_name(s: str, max_len: int = 120) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s if s else "NA"


def ensure_out_dir(input_csv: str, out_dir: str = None):
    if out_dir is None or str(out_dir).strip() == "":
        out_dir = os.path.splitext(os.path.abspath(input_csv))[0] + "_infer_out"
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# -----------------------------
# pseudo HLA map
# -----------------------------
def load_pseudo_hla_map(base_dir: str):
    pseudo_path = os.path.join(base_dir, "doc", "pseudo_HLAI.csv")
    if not os.path.exists(pseudo_path):
        raise FileNotFoundError(f"pseudo HLA file not found: {pseudo_path}")

    pseudo_df = pd.read_csv(pseudo_path)

    required_cols = {"MHC", "pseudo_seq"}
    miss = required_cols - set(pseudo_df.columns)
    if miss:
        raise ValueError(f"{pseudo_path} missing required columns: {miss}")

    pseudo_df = pseudo_df.copy()
    pseudo_df["MHC"] = pseudo_df["MHC"].astype(str).str.strip().str.upper()
    pseudo_df["pseudo_seq"] = pseudo_df["pseudo_seq"].astype(str).str.strip().str.upper()

    pseudo_map = dict(zip(pseudo_df["MHC"], pseudo_df["pseudo_seq"]))
    return pseudo_map


# -----------------------------
# input normalization / validation
# -----------------------------
def normalize_input_df(df: pd.DataFrame, pseudo_map: dict = None) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}

    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("mhc", "hla"):
            rename_map[c] = "MHC"
        elif cl in ("peptide", "epitope"):
            rename_map[c] = "peptide"
        elif cl in ("cdr3", "cdr3b"):
            rename_map[c] = "cdr3"
        elif cl in ("v_gene", "trbv", "bv", "tcrbv"):
            rename_map[c] = "v_gene"
        elif cl in ("pseudo_mhc", "pseudo_seq", "pseudo_hla", "pseudo_hlai"):
            rename_map[c] = "pseudo_MHC"

    df = df.rename(columns=rename_map)

    if "MHC" in df.columns:
        df["MHC"] = df["MHC"].fillna("").astype(str).str.strip().str.upper()

    if "peptide" in df.columns:
        df["peptide"] = df["peptide"].fillna("").astype(str).str.strip().str.upper()

    if "cdr3" in df.columns:
        df["cdr3"] = df["cdr3"].fillna("").astype(str).str.strip().str.upper()

    if "v_gene" in df.columns:
        df["v_gene"] = df["v_gene"].fillna("").astype(str).str.strip()

    if "pseudo_MHC" in df.columns:
        df["pseudo_MHC"] = df["pseudo_MHC"].fillna("").astype(str).str.strip().str.upper()

    if "MHC" in df.columns and "pseudo_MHC" not in df.columns and pseudo_map is not None:
        df["pseudo_MHC"] = df["MHC"].map(pseudo_map).fillna("")
        df["pseudo_MHC"] = df["pseudo_MHC"].astype(str).str.strip().str.upper()

    return df.reset_index(drop=True)


def validate_df_for_task(df: pd.DataFrame, task: str, logger):
    required = {
        "mp": ["MHC", "peptide", "pseudo_MHC"],
        "imm": ["MHC", "peptide", "pseudo_MHC"],
        "pt": ["peptide", "cdr3"],
        "mpt": ["MHC", "peptide", "cdr3", "v_gene", "pseudo_MHC"],
        "mp_contact": ["MHC", "peptide", "pseudo_MHC"],
        "pt_contact": ["peptide", "cdr3"],
    }

    miss = [x for x in required[task] if x not in df.columns]
    if miss:
        logger.error(f"Input csv missing required columns for task={task}: {miss}")
        return pd.Series(False, index=df.index)

    valid_mask = pd.Series(True, index=df.index)
    for col in required[task]:
        bad_col = df[col].isna() | (df[col] == "") | (df[col] == "NAN")
        valid_mask &= ~bad_col

    if not valid_mask.all():
        bad_count = (~valid_mask).sum()
        logger.warning(f"Task {task}: Dropping {bad_count} invalid/empty rows to prevent crash.")
        
    return valid_mask


# -----------------------------
# dataloader
# NOTE:
# all dataset flags are False for inference-only mode
# -----------------------------
def build_infer_loader(df: pd.DataFrame, task: str, batch_size: int):
    if task in ("mp", "imm", "mp_contact"):
        ds = MPDataSet(
            mp_df=df,
            mhc_type="HLAI",
            mhc_max_len=34,
            pep_max_len=15,
            binding=False,
            immunogenicity=False,
            contact=False,
        )
    elif task in ("pt", "pt_contact"):
        ds = PTDataSet(
            df,
            pep_max_len=15,
            cdr3_max_len=20,
            binding=False,
            contact=False,
        )
    elif task == "mpt":
        ds = MPTDataSet(
            df,
            mhc_type="HLAI",
            mhc_max_len=34,
            pep_max_len=15,
            cdr3_max_len=20,
            bv=True,
            binding=False,
        )
    else:
        raise KeyError(f"Unsupported task: {task}")

    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


# -----------------------------
# model loading
# -----------------------------
def load_models(ckpt_paths, use_lora, device, logger):
    aa_dict = mk_aa_dict()
    bv_dict = mk_bv_dict()

    models = []
    for p in ckpt_paths:
        logger.info(f"Loading checkpoint: {p}")
        base_model = model_fn(
            aa_vocab_size=len(aa_dict),
            trbv_vocab_size=len(bv_dict),
        )
        if use_lora:
            base_model = build_model_with_lora(
                base_model,
                last_n=2,
                cfg_seq_pair=((8, 16), (4, 8)),
                dropout=0.1,
                freeze_base=True,
                print_trainabel=False,
            )

        base_model.to(device)
        ckpt = torch.load(p, map_location=device, weights_only=True)
        base_model.load_state_dict(ckpt["model_state"], strict=True)
        base_model.eval()
        models.append(base_model)

    return models


# -----------------------------
# inference
# -----------------------------
def infer_binding(models, dataloader, task, device):
    if not isinstance(models, (list, tuple)):
        models = [models]

    arr_prob = np.zeros((len(dataloader.dataset),), dtype=np.float32)

    for m in models:
        m.eval()

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                b, idx = batch
            else:
                raise RuntimeError("Dataset should return (batch_dict, idx)")

            if task in ("mp", "imm"):
                mhc = b["mhc"].to(device)
                peptide = b["peptide"].to(device)
                esm_mhc = b.get("esm_mhc", None)
                if esm_mhc is not None:
                    esm_mhc = esm_mhc.to(device)

            elif task == "pt":
                peptide = b["peptide"].to(device)
                cdr3 = b["cdr3"].to(device)

            elif task == "mpt":
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
                raise KeyError(f"Unsupported binding task: {task}")

            logits_list = []
            for m in models:
                if task == "mp":
                    out = m.mp_pred(mhc, peptide, esm_mhc, contact=False, immunogenicity=False)
                    logit = out["binding_prob"].mean(dim=1, keepdim=True)
                elif task == "imm":
                    out = m.mp_pred(mhc, peptide, esm_mhc, contact=False, immunogenicity=True)
                    logit = out["immunogenicity_prob"].mean(dim=1, keepdim=True)
                elif task == "pt":
                    out = m.pt_pred(peptide, cdr3, contact=False)
                    logit = out["binding_prob"].mean(dim=1, keepdim=True)
                elif task == "mpt":
                    out = m.mpt_pred(mhc, peptide, cdr3, esm_mhc, trbv)
                    logit = out["mpt_prob"].mean(dim=1, keepdim=True)
                else:
                    raise RuntimeError("Unexpected task")

                logits_list.append(logit)

            ensemble_logit = torch.stack(logits_list, dim=0).mean(dim=0)
            pred = torch.sigmoid(ensemble_logit).squeeze(1).detach().cpu().numpy()
            arr_prob[idx.numpy()] = pred.astype(np.float32)

    return arr_prob


def infer_contact(models, dataloader, task, device):
    if not isinstance(models, (list, tuple)):
        models = [models]

    if task == "mp_contact":
        h, w = 34, 15
    elif task == "pt_contact":
        h, w = 15, 20
    else:
        raise KeyError(f"Unsupported contact task: {task}")

    pred_prob = np.zeros((len(dataloader.dataset), h, w), dtype=np.float32)
    pred_dist = np.zeros((len(dataloader.dataset), h, w), dtype=np.float32)
    arr_mask = np.zeros((len(dataloader.dataset), h, w), dtype=np.float32)

    for m in models:
        m.eval()

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                b, idx = batch
            else:
                raise RuntimeError("Dataset should return (batch_dict, idx)")

            if task == "mp_contact":
                mhc = b["mhc"].to(device)
                peptide = b["peptide"].to(device)
                esm_mhc = b.get("esm_mhc", None)
                if esm_mhc is not None:
                    esm_mhc = esm_mhc.to(device)

                out0 = models[0].mp_pred(mhc, peptide, esm_mhc, contact=True, immunogenicity=False)
                mask_1 = out0["mask_dict"]["mhc"]
                mask_2 = out0["mask_dict"]["pep"]

            else:
                peptide = b["peptide"].to(device)
                cdr3 = b["cdr3"].to(device)

                out0 = models[0].pt_pred(peptide, cdr3, contact=True)
                mask_1 = out0["mask_dict"]["pep"]
                mask_2 = out0["mask_dict"]["cdr3"]

            pair_mask = mask_1.unsqueeze(2) & mask_2.unsqueeze(1)
            arr_mask[idx.numpy()] = pair_mask.detach().cpu().numpy().astype(np.float32)

            contact_logits_list = []
            contact_dist_list = []

            for m in models:
                if task == "mp_contact":
                    out = m.mp_pred(mhc, peptide, esm_mhc, contact=True, immunogenicity=False)
                else:
                    out = m.pt_pred(peptide, cdr3, contact=True)

                contact_logits_list.append(out["contact_prob"])
                contact_dist_list.append(out["contact_dist"])

            contact_logits = torch.stack(contact_logits_list, dim=0).mean(dim=0)
            contact_dist = torch.stack(contact_dist_list, dim=0).mean(dim=0)

            pred_prob[idx.numpy()] = torch.sigmoid(contact_logits).detach().cpu().numpy().astype(np.float32)

            cd_np = contact_dist.detach().cpu().numpy().astype(np.float32)
            if task == "mp_contact":
                cd_np = dist_pred_from_logits_np(
                    cd_np,
                    np.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 18.0, 25.0], dtype=np.float32),
                )

            pred_dist[idx.numpy()] = cd_np

    return {
        "pred_prob": pred_prob,
        "pred_dist": pred_dist,
        "mask": arr_mask,
    }


# -----------------------------
# save outputs
# -----------------------------
def merge_binding_predictions(df_out, pred, task):
    col_name_map = {
        "mp": "pred_mp_binding",
        "pt": "pred_pt_binding",
        "imm": "pred_immunogenicity",
        "mpt": "pred_mpt_binding",
    }
    out_col = col_name_map[task]
    df_out[out_col] = pred
    return df_out


def save_binding_result(df_out, out_dir, input_csv, logger):
    input_name = os.path.splitext(os.path.basename(input_csv))[0]
    out_csv = os.path.join(out_dir, f"{input_name}_pred.csv")
    df_out.to_csv(out_csv, index=False)
    logger.info(f"Saved merged binding prediction csv to: {out_csv}")


def save_contact_result(df, out_dict, out_dir, task, logger, save_dist=True, keep_index_prefix=True, original_indices=None):
    task_dir = os.path.join(out_dir, task)
    os.makedirs(task_dir, exist_ok=True)

    pred_prob = out_dict["pred_prob"]
    pred_dist = out_dict["pred_dist"]
    mask = out_dict["mask"]

    for i in range(len(df)):
        orig_i = i if original_indices is None else original_indices[i]
        
        if task == "mp_contact":
            pseudo_seq = str(df.loc[i, "pseudo_MHC"]).strip().upper()
            pep_seq = str(df.loc[i, "peptide"]).strip().upper()

            pseudo_name = safe_name(pseudo_seq)
            pep_name = safe_name(pep_seq)

            if keep_index_prefix:
                prefix = f"{orig_i:06d}_{pseudo_name}_{pep_name}"
            else:
                prefix = f"{pseudo_name}_{pep_name}"

            h = min(34, len(pseudo_seq))
            w = min(15, len(pep_seq))

            row_labels = list(pseudo_seq)[:h]
            col_labels = list(pep_seq)[:w]

        else:
            pep_seq = str(df.loc[i, "peptide"]).strip().upper()
            cdr3_seq = str(df.loc[i, "cdr3"]).strip().upper()

            pep_name = safe_name(pep_seq)
            cdr3_name = safe_name(cdr3_seq)

            if keep_index_prefix:
                prefix = f"{orig_i:06d}_{pep_name}_{cdr3_name}"
            else:
                prefix = f"{pep_name}_{cdr3_name}"

            h = min(15, len(pep_seq))
            w = min(20, len(cdr3_seq))

            row_labels = list(pep_seq)[:h]
            col_labels = list(cdr3_seq)[:w]

        prob_mat = pred_prob[i][:h, :w].copy()
        prob_mask = mask[i][:h, :w]
        prob_mat[prob_mask < 0.5] = np.nan

        prob_csv = os.path.join(task_dir, prefix + "_site.csv")
        pd.DataFrame(prob_mat, index=row_labels, columns=col_labels).to_csv(prob_csv)

        if save_dist and pred_dist is not None:
            dist_mat = pred_dist[i][:h, :w].copy()
            dist_mat[prob_mask < 0.5] = np.nan

            dist_csv = os.path.join(task_dir, prefix + "_dist.csv")
            pd.DataFrame(dist_mat, index=row_labels, columns=col_labels).to_csv(dist_csv)

    logger.info(f"[{task}] Saved contact matrices to directory: {task_dir}")


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    BASE = os.path.abspath(os.path.dirname(__file__))
    logger = setup_logger()

    cli_args = parse_kv_args(sys.argv)

    tasks = parse_tasks(cli_args.get("task", "").strip())

    input_csv = cli_args.get("input_csv", None)
    if input_csv is None:
        raise ValueError("Please provide input_csv=XXX.csv")
    input_csv = os.path.expanduser(input_csv)

    out_dir = ensure_out_dir(input_csv, cli_args.get("out_dir", None))

    default_ckpt_path = os.path.join(BASE, "doc", "checkpoints_multi_lora_align3_ln")
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
    save_dist = str2bool(cli_args.get("save_dist", "true"))
    keep_index_prefix = str2bool(cli_args.get("keep_index_prefix", "true"))

    batch_size = int(cli_args.get("batch_size", "16"))
    contact_batch_size = int(cli_args.get("contact_batch_size", cli_args.get("batch_size", "8")))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Output directory: {out_dir}")

    df_raw = pd.read_csv(input_csv)

    need_pseudo = any(t in ("mp", "imm", "mpt", "mp_contact") for t in tasks)
    pseudo_map = load_pseudo_hla_map(BASE) if need_pseudo else None

    df_norm = normalize_input_df(df_raw, pseudo_map=pseudo_map)
    logger.info(f"Loaded input csv: {input_csv}, n={len(df_norm)}")

    models = load_models(ckpt_paths, use_lora=use_lora, device=device, logger=logger)

    # binding outputs are merged into ONE csv
    df_binding_out = df_raw.copy()
    has_binding_task = False

    for task in tasks:
        logger.info(f"===== Running task: {task} =====")
        valid_mask = validate_df_for_task(df_norm, task, logger=logger)
        
        if not valid_mask.any():
            logger.warning(f"No valid rows for task {task}. Skipping.")
            continue
            
        df_task = df_norm[valid_mask].reset_index(drop=False)

        bs = contact_batch_size if task in ("mp_contact", "pt_contact") else batch_size
        dataloader = build_infer_loader(df_task, task=task, batch_size=bs)

        if task in ("mp", "pt", "imm", "mpt"):
            pred = infer_binding(models, dataloader, task=task, device=device)
            arr_prob_full = np.full((len(df_norm),), np.nan, dtype=np.float32)
            arr_prob_full[df_task["index"].to_numpy()] = pred
            df_binding_out = merge_binding_predictions(df_binding_out, arr_prob_full, task)
            has_binding_task = True

        elif task in ("mp_contact", "pt_contact"):
            out_dict = infer_contact(models, dataloader, task=task, device=device)
            save_contact_result(
                df_task,
                out_dict,
                out_dir=out_dir,
                task=task,
                logger=logger,
                save_dist=save_dist,
                keep_index_prefix=keep_index_prefix,
                original_indices=df_task["index"].to_numpy()
            )

        else:
            raise RuntimeError(f"Unexpected task: {task}")

    if has_binding_task:
        save_binding_result(df_binding_out, out_dir=out_dir, input_csv=input_csv, logger=logger)

    logger.info("Inference done.")
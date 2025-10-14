import os
import random
import pickle

import numpy as np
import pandas as pd

def mk_aa_dict():
    """Returns the amino acid dictionary mapping."""
    amino_acid_dict = {
        'X': 0,
        'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
        'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
        'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
        'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
        '[MASK]': 21
    }
    return amino_acid_dict

def mk_bv_dict():
    bv_dict = {
        'X': 0,'TRBV1': 1,'TRBV10-1': 2,'TRBV10-2': 3,'TRBV10-3': 4,'TRBV11-1': 5,'TRBV11-2': 6,'TRBV11-3': 7,'TRBV12-1': 8,'TRBV12-2': 9,
        'TRBV12-3': 10,'TRBV12-4': 11,'TRBV12-5': 12,'TRBV13': 13,'TRBV14': 14,'TRBV15': 15,'TRBV16': 16,'TRBV17': 17,'TRBV18': 18,'TRBV19': 19,
        'TRBV2': 20,'TRBV20-1': 21,'TRBV20/OR9-2': 22,'TRBV21-1': 23,'TRBV21/OR9-2': 24,'TRBV22-1': 25,'TRBV22/OR9-2': 26,'TRBV23-1': 27,'TRBV23/OR9-2': 28,
        'TRBV24-1': 29,'TRBV24/OR9-2': 30,'TRBV25-1': 31,'TRBV26': 32,'TRBV26/OR9-2': 33,'TRBV27': 34,'TRBV28': 35,'TRBV29-1': 36,'TRBV29/OR9-2': 37,'TRBV3-1': 38,
        'TRBV30': 39,'TRBV4-1': 40,'TRBV4-2': 41,'TRBV4-3': 42,'TRBV5-1': 43,'TRBV5-2': 44,'TRBV5-3': 45,'TRBV5-4': 46,'TRBV5-5': 47,'TRBV5-6': 48,'TRBV5-7': 49,
        'TRBV5-8': 50,'TRBV6-1': 51,'TRBV6-2': 52,'TRBV6-4': 53,'TRBV6-5': 54,'TRBV6-6': 55,'TRBV6-7': 56,'TRBV6-8': 57,'TRBV6-9': 58,'TRBV7-1': 59,'TRBV7-2': 60,
        'TRBV7-3': 61,'TRBV7-4': 62,'TRBV7-5': 63,'TRBV7-6': 64,'TRBV7-7': 65,'TRBV7-8': 66,'TRBV7-9': 67,'TRBV8-1': 68,'TRBV8-2': 69,'TRBV9': 70,'TRBVA': 71,
        'TRBVB': 72, 'TRBV6-3': 73, 'TRBV3-2':74, 'TRBVA/OR9-2':75, 'TRBV25/OR9-2':76
    }
    return bv_dict

def load_mhc_dict(mhc_type, pseudo=True):
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir))
    
    assert mhc_type in ('MHCI', 'MHCII'),'mhc_type must be MHCI or MHCII'
    if pseudo:
        file_name = 'pseudo_MHCI.csv' if mhc_type == 'MHCI' else 'pseudo_MHCII.csv'
    else:
        file_name = 'esm2_emb_MHCI.pkl' if mhc_type == 'MHCI' else 'esm2_emb_MHCII.pkl'
    file_path = os.path.join(project_root, 'doc', file_name)
    if pseudo:
        mhc_data = pd.read_csv(file_path, index_col=0)
        mhc_dict = dict(zip(mhc_data['MHC'], mhc_data['pseudo_seq']))
    else:
        with open(file_path, 'rb') as f:
            mhc_dict = pickle.load(f)
    return mhc_dict

def aa_to_vec(aa_seq, aa_dict):
    """Converts an amino acid sequence to a vector representation."""
    sequence = sequence.replace(u'\xa0', u'').upper()
    return np.array([aa_dict.get(aa, aa_dict['X']) for aa in aa_seq], dtype=int)

def pad_1d(arr, max_len, pad_value=0, dtype=int):
    arr = np.asarray(arr, dtype=dtype)
    if arr.shape[0] >= max_len:
        return arr[:max_len]
    if arr.shape[0] < max_len:
        return np.pad(arr, (0, max_len - arr.shape[0]),
                      mode='constant', constant_values=pad_value)
    return arr

def pad_2d(mat, target_h, target_w, pad_value=0, dtype=float):
    mat = np.asarray(mat, dtype=dtype)
    if mat.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {mat.shape}")

    h, w = mat.shape

    h_trunc = min(h, target_h)
    w_trunc = min(w, target_w)
    mat_cropped = mat[:h_trunc, :w_trunc]

    pad_h = max(target_h - h_trunc, 0)
    pad_w = max(target_w - w_trunc, 0)

    # --- pad ---
    if pad_h > 0 or pad_w > 0:
        mat_padded = np.pad(
            mat_cropped,
            pad_width=((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=pad_value,
        )
    else:
        mat_padded = mat_cropped

    if mat_padded.shape != (target_h, target_w):
        raise RuntimeError(
            f"pad_2d failed: got {mat_padded.shape}, expected ({target_h}, {target_w})"
        )
    return mat_padded
    
def mhc_to_aa(mhc_name, mhc_dict):
    mhc_aa = mhc_dict.get(mhc_name)
    if mhc_aa is None:
        raise KeyError(f"MHC name '{mhc_name}' not found in mhc_dict.")
    return mhc_aa

def mhc_to_esm2(mhc_name, esm2_dict):
    mhc_esm2_emb = esm2_dict.get(mhc_name)
    if mhc_esm2_emb is None:
        raise KeyError(f"MHC name '{mhc_name}' not found in esm2_dict.")
    return mhc_esm2_emb

def labelMap(df, label):
    return df[label].to_numpy()

###mask###
def _sample_contiguous_segment(candidate_positions, num_mlm_preds):
    if not candidate_positions or num_mlm_preds <= 0:
        return set()
    
    cand = sorted(candidate_positions)
    num_mlm_preds = min(num_mlm_preds, len(cand))
    max_start = len(candidate_positions) - num_mlm_preds

    if max_start <= 0:
        return set(cand)
    
    start_idx = random.randint(0, max_start)
    return set(cand[start_idx:start_idx + num_mlm_preds])

def replace_masked_tokens(token_ids, candidate_positions, num_mlm_preds, vocab_dict, contiguous_prob=0.5):
    """Replaces masked tokens in the sequence according to MLM rules."""
    mlm_input_tokens_id = token_ids.copy()

    if num_mlm_preds <= 0 or not candidate_positions:
        mlm_label = [vocab_dict['X']] * len(token_ids)
        return mlm_input_tokens_id, mlm_label
    
    if random.random() < contiguous_prob:
        pred_positions = _sample_contiguous_segment(candidate_positions, num_mlm_preds)
    else:
        cand = list(candidate_positions)    # 避免就地修改
        random.shuffle(cand)
        pred_positions = set(cand[:num_mlm_preds])

    ban = {vocab_dict.get('[CLS]'), vocab_dict.get('[SEP]'),
        vocab_dict.get('[MASK]'), vocab_dict.get('X')}
    valid_ids = [vid for vid in set(vocab_dict.values()) if vid not in ban]
    if not valid_ids:
        valid_ids = [vocab_dict['[MASK]']]

    for pos in pred_positions:
        rand_val = random.random()
        if rand_val < 0.8:
            masked_token_id = vocab_dict['[MASK]']
        elif rand_val < 0.9:
            masked_token_id = token_ids[pos]
        else:
            masked_token_id = random.choice(valid_ids)

        mlm_input_tokens_id[pos] = masked_token_id

    mlm_label = [
        vocab_dict['X'] if idx not in pred_positions else token_ids[idx] 
        for idx in range(len(token_ids))
    ]
    return mlm_input_tokens_id, mlm_label

def get_masked_sample(aa_seq_vec, aa_dict, masked_rate, contiguous_prob=0.5):
    candidate_positions = [i for i, token in enumerate(aa_seq_vec)]
    num_mlm_preds = round(len(aa_seq_vec) * masked_rate) 
    
    if num_mlm_preds == 0: 
        return aa_seq_vec.copy(), np.full_like(aa_seq_vec, aa_dict['X'])  
    return replace_masked_tokens(aa_seq_vec, candidate_positions, num_mlm_preds, aa_dict, contiguous_prob)
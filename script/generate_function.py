import torch
import numpy as np

def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)

def temp_softmax(z, T):
    exp_z = np.exp(z/T)
    sum_exp_z = np.sum(exp_z)
    return exp_z / sum_exp_z

def one_hot(label, num_classes=22):
  return np.squeeze(np.eye(num_classes)[label.reshape(-1)])

def mrf_score(label, seq_logits, pair_logits, seq_mask, pair_mask):
    """
    能量评分可以用于两方面:
    1.模型蒸馏的时候评价ground truth和logits之间的能量;
    2.sampling后的生成序列和logits之间的能量。
    """
    N, C = seq_logits.shape
    pair_logits = np.reshape(pair_logits, [N, N, C, C])
    label = one_hot(label)
    
    score = np.sum(seq_logits * label * seq_mask[:,None]) + \
        np.sum(label[None,:,None,:] * label[:,None,:,None] * pair_logits * pair_mask[:,:,None,None]) / 2.0
    return score

def band_topk_strength_pair_mask(pair_mask, pair_logits, W=3, topk=3):
    pm_in = pair_mask.astype(bool)
    L = pm_in.shape[0]
    seq_mask = (pm_in.sum(axis=1) > 0) | (pm_in.sum(axis=0) > 0)

    # ===== band =====   
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    band = (np.abs(i - j) <= W)
    pm_band = band & pm_in
    np.fill_diagonal(pm_band, False)

    # ===== Top-k =====
    L2, L3, A1, A2 = pair_logits.shape
    assert L2 == L and L3 == L, "pair_logits L unmatch pair_mask"
    assert A1 == A2, "pair_logits last dims should be A,A"

    E = pair_logits.astype(np.float32, copy=False)
    Ea  = E.mean(axis=2, keepdims=True)         
    Eb  = E.mean(axis=3, keepdims=True)         
    Eab = E.mean(axis=(2, 3), keepdims=True)   
    E_c = E - Ea - Eb + Eab

    s = np.sqrt((E_c ** 2).mean(axis=(2, 3)))   
    s = s * pm_in.astype(s.dtype)

    pm_topk = np.zeros((L, L), dtype=bool)
    for ii in range(L):
        if not seq_mask[ii]:
            continue
        cand = s[ii].copy()
        cand[~seq_mask] = -np.inf
        cand[ii] = -np.inf
        js = np.argsort(-cand)[:max(0, topk)]
        if js.size > 0:
            pm_topk[ii, js] = True

    pm_topk = np.logical_or(pm_topk, pm_topk.T)
    np.fill_diagonal(pm_topk, False)

    pm_out = pm_band | pm_topk

    return pm_out.astype(pair_mask.dtype)

def infer_mrf(seq_logits, pair_logits, seq_mask, pair_mask, T, steps=5, 
              forbid_indices=(0,21,22,23,24,25)):
    """
    单点Gibbs/ICM推理：
      scores_i(a) = h_i(a) + sum_{k: pm[i,k]=1} e_{i,k}(a, x_k)
    假设：0 为 padding/X，21 为 [MASK]；有效类别为 1..20
    """
    N, C = seq_logits.shape
    if pair_logits.ndim != 4:
        pair_logits = np.reshape(pair_logits, [N, N, C, C])

    pair_mask = band_topk_strength_pair_mask(pair_mask, pair_logits)
    deg = np.sum(pair_mask, axis=-1)

    H = seq_logits.copy()
    if forbid_indices:
        H[:, np.array(forbid_indices, dtype=int)] = -np.inf
    site_prob = softmax(H)
    init_label = np.argmax(site_prob, axis=-1)

    prev_label = np.array(init_label)
    pos = np.argsort(deg)
    VALID_IDX = [i for i in range(C) if i not in forbid_indices]

    for _ in range(steps):
        #shuffle(pos)
        updated_count = 0
        ###单点
        for i in pos:
            if not seq_mask[i]:
                continue
            t_lis = np.zeros(len(VALID_IDX), dtype=np.float32)
            ###单点各类别打分
            for c in VALID_IDX:
                t = seq_logits[i, c]
                for k in range(N):
                    if pair_mask[i, k]:
                        t += pair_logits[i, k, c, prev_label[k]]
                # 约定 0 为 padding/X，所以用 c-1 对齐到 0..19
                t_lis[c-1] = t
            ###单点采样/贪心
            if T >= 0.1:
                probs = temp_softmax(t_lis, T)
                obj_c = np.argmax(np.random.multinomial(1, probs))
            else:
                obj_c = np.argmax(t_lis)
            # 从位置映射（0..19）回真实类别索引（1..20）
            obj_c +=1

            if prev_label[i] != obj_c:
                prev_label[i] = obj_c
                updated_count += 1
        if updated_count == 0:
            break
    return prev_label
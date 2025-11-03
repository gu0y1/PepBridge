from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, \
    accuracy_score, f1_score, precision_score, recall_score, mean_squared_error

import torch.nn.functional as F
import torch

def evaluate_metrics(y_true, y_prob, task):
    y_pred = [int(p > 0.5) for p in y_prob] 
    if task == 'BCE':
        return {
            'AUC': roc_auc_score(y_true, y_prob),
            'PR-AUC': average_precision_score(y_true, y_prob),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred)
        }
    elif task == 'MSE':
        return {
            'RMSE': mean_squared_error(y_true, y_prob, squared=False),
            'Pearson': pearsonr(y_true, y_prob)[0],
            'Spearman': spearmanr(y_true, y_prob)[0]
        }

def bertscore(seq_emb, cand_emb, ref_mask, cand_mask):
    """
    seq_emb:  [B, Lr, D]  # reference
    cand_emb: [B, Lc, D]  # candidate
    ref_mask:  [B, Lr]    # 1=valid, 0=pad
    cand_mask: [B, Lc]    # 1=valid, 0=pad
    """
    cand_norm = F.normalize(cand_emb, dim=-1)   # [B, Lc, D]
    ref_norm  = F.normalize(seq_emb, dim=-1)    # [B, Lr, D]

    sim = torch.matmul(cand_norm, ref_norm.transpose(1, 2))   # [B, Lc, Lr]

    # 把pad位置的相似度干掉
    # cand pad 的行全是无效
    sim = sim.masked_fill(cand_mask.unsqueeze(-1) == 0, -1e4)
    # ref pad 的列全是无效
    sim = sim.masked_fill(ref_mask.unsqueeze(1) == 0, -1e4)

    # precision: 对每个 cand token 在 ref 里取最大
    prec_tok = sim.max(dim=2).values                # [B, Lc]
    # 只对有效的cand取平均
    prec = (prec_tok * cand_mask).sum(dim=1) / cand_mask.sum(dim=1).clamp_min(1.0)

    # recall: 对每个 ref token 在 cand 里取最大
    rec_tok = sim.max(dim=1).values                 # [B, Lr]
    rec = (rec_tok * ref_mask).sum(dim=1) / ref_mask.sum(dim=1).clamp_min(1.0)

    f1 = 2 * prec * rec / (prec + rec + 1e-8)       # [B]
    return f1

def perplexity_per_sample(logits, labels, pad_id):
    """
    logits: [B, L, V]
    labels: [B, L]
    return: [B] 的每条样本 perplexity
    """
    B, L, V = logits.shape

    logits_flat = logits.view(B * L, V)      # [B*L, V]
    labels_flat = labels.view(B * L)         # [B*L]

    loss_tok = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=pad_id,
        reduction="none"                     # [B*L]
    ).view(B, L)                            

    mask = (labels != pad_id).float()        # [B, L]
    tok_per_seq = mask.sum(1).clamp_min(1.0) 

    loss_per_sample = (loss_tok * mask).sum(1) / tok_per_seq   # [B]
    ppl_per_sample = torch.exp(loss_per_sample)                # [B]
    return ppl_per_sample
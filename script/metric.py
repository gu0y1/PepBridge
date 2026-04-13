from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, \
    accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

import numpy as np

def _round_metrics_dict(d, ndigits=4):
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            out[k] = float(np.round(v, ndigits))
        else:
            out[k] = v
    return out

def _binary_metrics_on_flat(y_true_flat, y_pred_flat, threshold=0.5):
    if y_true_flat.size == 0:
        raise ValueError("none samples")

    y_pred_label = (y_pred_flat > threshold).astype(int)

    unique_labels = np.unique(y_true_flat)
    if unique_labels.size < 2:
        auc_roc = np.nan
        aupr_val = np.nan
        precision_curve = np.array([1.0])
        recall_curve = np.array([0.0])
    else:
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true_flat, y_pred_flat
        )
        auc_roc = roc_auc_score(y_true_flat, y_pred_flat)
        aupr_val = auc(recall_curve, precision_curve)

    if np.unique(y_pred_label).size < 2:
        mcc_val = np.nan
    else:
        mcc_val = matthews_corrcoef(y_true_flat, y_pred_label)
    metrics = {
        'AUC': auc_roc,
        'AUPR': aupr_val,
        'MCC': mcc_val,
        "Accuracy": accuracy_score(y_true_flat, y_pred_label),
        "Precision": precision_score(y_true_flat, y_pred_label, zero_division=0),
        "Recall": recall_score(y_true_flat, y_pred_label, zero_division=0),
        "F1": f1_score(y_true_flat, y_pred_label, zero_division=0)
    }
    return _round_metrics_dict(metrics, ndigits=4)

def binary_evaluate_metrics(y_true, y_pred, mask=None, threshold=0.5, group=None, mean=False):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        if (y_pred.ndim == y_true.ndim + 1 and
            y_pred.shape[-1] == 1 and
            y_pred.shape[:-1] == y_true.shape):
            y_pred = np.squeeze(y_pred, axis=-1)
        elif (y_true.ndim == y_pred.ndim + 1 and
              y_true.shape[-1] == 1 and
              y_true.shape[:-1] == y_pred.shape):
            y_true = np.squeeze(y_true, axis=-1)
        else:
            raise ValueError(
                f"y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}"
            )

    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != y_true.shape:
            raise ValueError(
                f"mask.shape={mask.shape} y_true.shape={y_true.shape}"
            )
        y_true_flat = y_true[mask]
        y_pred_flat = y_pred[mask]
    else:
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

    if y_true_flat.size == 0:
        raise ValueError("none samples")

    if group is None:
        return _binary_metrics_on_flat(y_true_flat, y_pred_flat, threshold)

    group = np.asarray(group).astype(str)
    if group.shape != y_true.shape:
        raise ValueError(
            f"group.shape={group.shape} y_true.shape={y_true.shape}"
        )
    group_flat = group[mask] if mask is not None else group.ravel()

    unique_groups = np.unique(group_flat)

    group_to_metrics = {}
    for g in unique_groups:
        idx = (group_flat == g)
        y_true_g = y_true_flat[idx]
        y_pred_g = y_pred_flat[idx]

        if y_true_g.size == 0:
            continue

        group_to_metrics[g] = _binary_metrics_on_flat(
            y_true_g, y_pred_g, threshold
        )

    all_metric_names = next(iter(group_to_metrics.values())).keys()
    metrics_by_group = {m: {} for m in all_metric_names}

    for g, m_dict in group_to_metrics.items():
        for m_name, m_val in m_dict.items():
            metrics_by_group[m_name][g] = m_val
    if not mean:
        return metrics_by_group

    mean_metrics = {}
    for m_name, g2v in metrics_by_group.items():
        vals = np.array(list(g2v.values()), dtype=float)
        mean_metrics[m_name] = float(np.nanmean(vals))

    return _round_metrics_dict(mean_metrics, ndigits=4)


def dist_pred_from_logits_np(logits: np.ndarray, bin_centers: np.ndarray):
    """
    logits:      [B, mhc_len, pep_len, K] 
    bin_centers: [K]
    return:      [B, mhc_len, pep_len]
    """
    # softmax(logits): [B, Mh, P, K]
    max_logits = np.max(logits, axis=-1, keepdims=True)          # [B, Mh, P, 1]
    exp_logits = np.exp(logits - max_logits)                     # [B, Mh, P, K]
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # [B, Mh, P, K]

    # bin_centers: [K] -> [1, 1, 1, K]
    centers = bin_centers.reshape(1, 1, 1, -1)                    # [1, 1, 1, K]

    pred_dist = np.sum(probs * centers, axis=-1)                  # [B, Mh, P]
    return pred_dist

def distance_evaluate_metrics(y_true, y_pred, mask, eps=1e-8, distogram=False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if distogram:
        y_pred = dist_pred_from_logits_np(y_pred, 
        np.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 18.0, 25.0], 
                 dtype=np.float32))
        y_true = np.clip(y_true, None, 30)

    mask   = np.asarray(mask).astype(bool)

    assert y_true.shape == y_pred.shape == mask.shape
    N = y_true.shape[0]

    mse_list = np.zeros(N)
    mape_list = np.zeros(N)
    pear_list = np.zeros(N)
    spea_list = np.zeros(N)

    for i in range(N):
        m = mask[i]
        if not np.any(m):
            mse_list[i]  = np.nan
            mape_list[i] = np.nan
            pear_list[i] = np.nan
            spea_list[i] = np.nan
            continue

        yt = y_true[i][m]
        yp = y_pred[i][m]

        diff = yp - yt
        mse  = np.mean(diff ** 2)
        mape = np.mean(np.abs(diff) / (np.abs(yt) + eps))

        mse_list[i]  = mse
        mape_list[i] = mape

        if yt.size < 2:
            pear_list[i] = np.nan
            spea_list[i] = np.nan
        else:
            pear_list[i] = pearsonr(yt, yp)[0]
            spea_list[i] = spearmanr(yt, yp)[0]

    metrics = {
        "MSE":      np.nanmean(mse_list),
        "MAPE":     np.nanmean(mape_list),
        "Pearson":  np.nanmean(pear_list),
        "Spearman": np.nanmean(spea_list),
    }
    return _round_metrics_dict(metrics, ndigits=4)
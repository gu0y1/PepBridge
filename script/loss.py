import torch
import torch.nn.functional as F

from typing import Optional

def mask_mean(x: torch.Tensor, mask: Optional[torch.Tensor]):
    if mask is None:
        return x.mean()
    valid = mask.to(dtype=x.dtype)
    s = (x * valid).sum()
    n = valid.sum()
    if n.item() == 0:
        return x.new_zeros(())
    return s / n

def contact_losses(prob_pred: torch.Tensor,
                   dist_pred: Optional[torch.Tensor],
                   prob_tgt: Optional[torch.Tensor],
                   dist_tgt: Optional[torch.Tensor],
                   pair_mask: Optional[torch.Tensor],
                   *, use_logits: bool = False):
    # squeeze（if [..., 1]）
    if prob_pred.dim() == 4 and prob_pred.size(-1) == 1:
        prob_pred = prob_pred.squeeze(-1)
    if dist_pred is not None and dist_pred.dim() == 4 and dist_pred.size(-1) == 1:
        dist_pred = dist_pred.squeeze(-1)
    if prob_tgt is not None and prob_tgt.dim() == 4 and prob_tgt.size(-1) == 1:
        prob_tgt = prob_tgt.squeeze(-1)
    if dist_tgt is not None and dist_tgt.dim() == 4 and dist_tgt.size(-1) == 1:
        dist_tgt = dist_tgt.squeeze(-1)

    loss_prob = prob_pred.new_zeros(())
    loss_dist = prob_pred.new_zeros(())

    if prob_tgt is not None:
        ce = F.binary_cross_entropy_with_logits(prob_pred, prob_tgt, reduction="none") \
             if use_logits else F.binary_cross_entropy(prob_pred, prob_tgt, reduction="none")
        loss_prob = mask_mean(ce, pair_mask)

    if (dist_pred is not None) and (dist_tgt is not None):
        mse = F.mse_loss(dist_pred, dist_tgt, reduction="none")
        loss_dist = mask_mean(mse, pair_mask)

    return loss_prob, loss_dist

def bce_loss(pred, target, *, use_logits: bool = False, reduction: str = "mean"):
    if use_logits:
        return F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)
    else:
        return F.binary_cross_entropy(pred, target, reduction=reduction)

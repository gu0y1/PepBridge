import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import Optional

def vicreg(z_s, z_t, lmb=20.0, mu=25.0, nu=1.0, gamma=1.0, eps=1e-4):
    # z_*: [N, d]
    # 1) invariance
    inv = F.mse_loss(z_s, z_t)

    # 2) variance (per feature)
    def _var(z):
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(gamma - std))
    var = _var(z_s) + _var(z_t)

    # 3) covariance (off-diagonal)
    def _cov(z):
        z = z - z.mean(dim=0, keepdim=True)
        N = z.size(0)
        c = (z.T @ z) / (N - 1)
        off = c - torch.diag(torch.diag(c))
        return (off ** 2).sum() / z.size(1)
    cov = _cov(z_s) + _cov(z_t)

    return (lmb * inv + mu * var + nu * cov) / (lmb + mu + nu)

def _batch_masked_mean(x, mask, w=None, dims=(1,2), eps=1e-8):
    # x, mask: [B, L1, L2]
    if w is None:
        w = 1.0
    valid = mask.to(dtype=x.dtype)
    num = (w * x * valid).sum(dim=dims)
    den = (w * valid).sum(dim=dims).clamp_min(eps)
    per_ex = num / den                # [B]
    return per_ex.mean()              # scalar

# @torch.no_grad()
# def _pos_weight_from_mask(y, mask, eps=1e-8):
#     yb = (y > 0.5) & mask
#     pos = yb.sum(dim=(1,2)).float()            # [B]
#     neg = (mask.sum(dim=(1,2)) - yb.sum(dim=(1,2))).float() # [B]
#     return (neg + eps) / (pos + eps)  # [B]

class MPDistogramLoss(nn.Module):
    def __init__(self, sigmas=None):
        super().__init__()

        centers = torch.tensor(
            [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 18.0, 25.0],
            dtype=torch.float32,
        )
        self.register_buffer("bin_centers", centers)

        if sigmas is None:
            gaps = torch.diff(centers)                         
            left_gap  = torch.cat([gaps[:1], gaps])              # [K]
            right_gap = torch.cat([gaps, gaps[-1:]])             # [K]
            widths = 0.5 * (left_gap + right_gap)                # [K]
            sigmas = torch.clamp(widths * 0.5, min=0.3)
            sigmas[-1] = sigmas[-1] * 1.5
        else:
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
            assert sigmas.numel() == centers.numel()

        self.register_buffer("bin_sigmas", sigmas)

    def forward(
        self,
        logits: torch.Tensor,      # [B, mhc_len, pep_len, K]
        dist_true: torch.Tensor,   # [B, mhc_len, pep_len]
        pair_mask: torch.Tensor,   # [B, mhc_len, pep_len]
        eps: float = 1e-8,
    ) -> torch.Tensor:
        B, Mh, P, K = logits.shape
        assert K == self.bin_centers.numel(), \
            f"num_bins mismatch: logits last dim = {K}, bin_centers = {self.bin_centers.numel()}"

        logits_flat = logits.reshape(-1, K)          # [B * Mh * P, K]
        dist_flat   = dist_true.clamp(min=2.0, max=30.0).reshape(-1)          # [B * Mh * P]
        mask_flat   = pair_mask.reshape(-1)          # [B * Mh * P]

        if mask_flat.dtype != torch.bool:
            valid = mask_flat > 0.5
        else:
            valid = mask_flat

        if valid.sum() == 0:
            return logits_flat.sum() * 0.0

        logits_valid = logits_flat[valid]   # [N_valid, K]
        dist_valid   = dist_flat[valid]     # [N_valid]

        # ===== construct soft Gassin label：y_probs [N_valid, K] =====
        # bin_centers: [K] -> [1, K]
        # dist_valid:  [N_valid] -> [N_valid, 1]
        centers = self.bin_centers.to(logits_valid.device)  # [K]
        sigmas = self.bin_sigmas.to(logits_valid.device)

        diff = (centers.unsqueeze(0) - dist_valid.unsqueeze(1)) / sigmas.unsqueeze(0)   # [N_valid, K]
        y_probs = torch.exp(-0.5 * diff**2)                              
        y_probs = y_probs / (y_probs.sum(dim=1, keepdim=True) + eps)     

        # ===== log_prob =====
        log_probs = F.log_softmax(logits_valid, dim=-1)  # [N_valid, K]
        loss = -(y_probs * log_probs).sum(dim=-1).mean()

        d_hat = (F.softmax(logits_valid, dim=-1) * centers.unsqueeze(0)).sum(-1) 
        mse = (d_hat - dist_valid).pow(2).mean()

        return loss + 0.1 * mse

def contact_losses(prob_pred, dist_pred, prob_tgt,
                   dist_tgt, pair_mask, use_logits=True,
                   distogram=False):
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
        # pw = _pos_weight_from_mask(prob_tgt, pair_mask).clamp_(1.0, 3.0).to(ce.dtype).view(-1, 1, 1)
        # posmask = ((prob_tgt > 0.5) & pair_mask).to(ce.dtype) 
        # w = 1.0 + (pw - 1.0) * posmask
        loss_prob = _batch_masked_mean(ce, pair_mask)

    if (dist_pred is not None) and (dist_tgt is not None):
        if distogram:
            criterion_mp = MPDistogramLoss()
            loss_dist = criterion_mp(dist_pred, dist_tgt, pair_mask)
        else:
            mse = F.mse_loss(dist_pred, dist_tgt, reduction="none")
            loss_dist = _batch_masked_mean(mse, pair_mask)

    return loss_prob, loss_dist
  
def bce_loss(pred, target, use_logits=True, reduction="mean", pos_weight=None):
    if pos_weight is not None:
        pos_weight = torch.tensor([pos_weight], device=pred.device, dtype=pred.dtype)
    if use_logits:
        loss_heads = F.binary_cross_entropy_with_logits(pred, target.expand_as(pred), 
                                                        reduction=reduction, pos_weight=pos_weight)
    else:
        assert pos_weight is None, "if use_logits is Fasle, pos_weight is ignored"
        loss_heads = F.binary_cross_entropy(pred, target.expand_as(pred), reduction=reduction)
    return loss_heads
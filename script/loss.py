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
        # ===== soft ce loss / KL=====
        loss = -(y_probs * log_probs).sum(dim=-1).mean()

        return loss

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
        # pw = _pos_weight_from_mask(prob_tgt, pair_mask).clamp_(1.0, 30.0).to(ce.dtype).view(-1, 1, 1)
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

def loss_pair_pseudolikelihood(
    seq_logits,              # [B, L, A]  -> h_i(a)
    pair_logits,             # [B, L, L, A, A] or [B, L, L, A*A] -> e_ij(a,b)
    aatype,                  # [B, L] (long)
    mask=None,               # [B, L] (float/bool), 1=有效
    pair_mask=None,          # [B, L, L] (float/bool), 1=有效
    lambda_reg: float = 1e-2,# 
    temperature: float = 1.0,# 
    reduction: str = "mean"  # "mean" | "sum" | "none"
):
    """
    \log P(X_i = x_i, X_j = x_j \mid X_{\setminus \{i,j\}} = x_{\setminus \{i,j\}}; s,z)
    \propto
    h_i(x_i) + h_j(x_j) + e_{ij}(x_i,x_j)
    + \sum_{k\neq i,j} \big[ e_{ik}(x_i,x_k) + e_{jk}(x_j,x_k) \big]
    """
    device = seq_logits.device
    dtype  = seq_logits.dtype
    B, L, A = seq_logits.shape

    # --- 形状标准化 -> [B, L, L, A, A]
    if pair_logits.dim() == 5 and pair_logits.shape[-2:] == (A, A):
        pair_aa = pair_logits
    elif pair_logits.dim() == 4 and pair_logits.shape[-1] == A * A:
        pair_aa = pair_logits.view(B, L, L, A, A).contiguous()
    else:
        raise ValueError("pair_logits must be [B,L,L,A,A] or [B,L,L,A*A].")

    # --- 掩码
    if mask is None:
        mask = torch.ones(B, L, device=device, dtype=dtype)
    else:
        mask = mask.to(dtype)
    if pair_mask is None:
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
    else:
        pair_mask = pair_mask.to(dtype)

    # 去掉对角
    eye = torch.eye(L, device=device, dtype=dtype).unsqueeze(0)    # [1,L,L]
    pair_mask = pair_mask * (1.0 - eye)

    # --- one-hot(x_k) 用于一次性选择 b = x_k
    aatype_clamped = aatype.clamp_min(0).clamp_max(A - 1)
    x_onehot = F.one_hot(aatype_clamped, num_classes=A).to(dtype)  # [B,L,A]
    x_onehot = x_onehot * mask.unsqueeze(-1)                       # 无效位点不贡献

    # not_self: k != i
    not_self = (1.0 - eye)                                         # [1,L,L]
    pair_aa_masked = pair_aa * not_self[:, :, :, None, None]       # [B,L,L,A,A]

    # S_all[i,a] = Σ_{k!=i} e_{i,k}(a, x_k)
    # 广播: x_onehot[:, None, k, None, b] 与 pair_aa_masked 逐元素相乘后，sum_{k,b}
    S_all = (pair_aa_masked * x_onehot[:, None, :, None, :]).sum(dim=(2, 4))  # [B,L,A]

    # e_{i,j}(a, x_j)
    idx_xj = aatype_clamped[:, None, :, None].expand(B, L, L, A).unsqueeze(-1)  # [B,L,L,A,1]
    eij_axj = torch.gather(pair_aa, dim=4, index=idx_xj).squeeze(-1)               # [B,L,L,A]

    # e_{j,i}(b, x_i)
    idx_xi = aatype_clamped[:, :, None, None].expand(B, L, L, A).unsqueeze(-2)  # [B,L,L,1,A]
    eji_bxi = torch.gather(pair_aa, dim=3, index=idx_xi).squeeze(-2)            # [B,L,L,A]

    # --- 构造 logits_ij(a,b) = h_i(a)+h_j(b)+e_ij(a,b)+S_i^{¬j}(a)+S_j^{¬i}(b)
    h = seq_logits
    if temperature != 1.0:
        invT = 1.0 / temperature
        h        = h        * invT
        pair_aa  = pair_aa  * invT
        S_all    = S_all    * invT
        eij_axj  = eij_axj  * invT
        eji_bxi  = eji_bxi  * invT

    Hi = h[:, :, None,   :, None]                                  # [B,L,1,A,1]
    Hj = h[:, None, :,   None, :]                                  # [B,1,L,1,A]
    S_i_excl = S_all[:, :, None, :] - eij_axj                      # [B,L,L,A]
    S_j_excl = S_all[:, None, :, :] - eji_bxi                      # [B,L,L,A]

    logits_ij = Hi + Hj + pair_aa \
                + S_i_excl[..., :, None] \
                + S_j_excl[..., None, :]                           # [B,L,L,A,A]

    # --- 归一化 + 取标签 NLL
    logits_flat = logits_ij.view(B, L, L, A * A)                    # [B,L,L,A*A]
    log_probs = F.log_softmax(logits_flat.float(), dim=-1)          # 数值更稳
    tgt = (aatype_clamped[:, :, None] * A + aatype_clamped[:, None, :])  # [B,L,L]
    nll = -torch.gather(log_probs, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)  # [B,L,L]
    nll = nll.to(dtype)

    valid_pairs = (mask[:, :, None] * mask[:, None, :]) * pair_mask # [B,L,L]
    nll = nll * valid_pairs

    if reduction == "mean":
        loss_ce = nll.sum() / valid_pairs.sum().clamp_min(1.0)
    elif reduction == "sum":
        loss_ce = nll.sum()
    elif reduction == "none":
        loss_ce = nll  # [B,L,L]
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")

    # --- 正则（并入 L_pair）
    valid = pair_mask.bool().unsqueeze(-1).unsqueeze(-1)
    reg_l1 = pair_aa.abs().masked_select(valid).mean()
    reg_l2 = pair_aa.pow(2).masked_select(valid).mean()
    loss_reg = lambda_reg * (reg_l1 + reg_l2)

    loss = loss_ce + loss_reg
    return loss

# def margin_loss(p_pos, p_neg, m=1, rho=0.03):
#     margin = F.relu(m+p_neg-p_pos)
#     reg = rho * torch.sqrt(p_neg ** 2 + p_pos ** 2)
#     return (margin + reg).mean()
    

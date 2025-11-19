import torch
import torch.nn.functional as F

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

@torch.no_grad()
def _pos_weight_from_mask(y, mask, eps=1e-8):
    yb = (y > 0.5) & mask
    pos = yb.sum(dim=(1,2)).float()            # [B]
    neg = (mask.sum(dim=(1,2)) - yb.sum(dim=(1,2))).float() # [B]
    return (neg + eps) / (pos + eps)  # [B]

def contact_losses(prob_pred, dist_pred, prob_tgt,
                   dist_tgt, pair_mask, use_logits=True):
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
        pw = _pos_weight_from_mask(prob_tgt, pair_mask).clamp_(1.0, 30.0).to(ce.dtype).view(-1, 1, 1)
        posmask = ((prob_tgt > 0.5) & pair_mask).to(ce.dtype) 
        w = 1.0 + (pw - 1.0) * posmask
        loss_prob = _batch_masked_mean(ce, pair_mask, w)

    if (dist_pred is not None) and (dist_tgt is not None):
        mse = F.mse_loss(dist_pred, dist_tgt, reduction="none")
        # hub = F.smooth_l1_loss(dist_pred, dist_tgt, reduction='none', beta=1.0)
        loss_dist = _batch_masked_mean(mse, pair_mask)

    return loss_prob, loss_dist
  
def bce_loss(pred, target, use_logits=True, reduction="mean"):
    if use_logits:
        loss_heads = F.binary_cross_entropy_with_logits(pred, target.expand_as(pred), reduction=reduction)
    else:
        loss_heads = F.binary_cross_entropy(pred, target.expand_as(pred), reduction=reduction)
    return loss_heads

def loss_pair_pseudolikelihood(
    seq_logits,              # [B, L, A]  -> h_i(a)
    pair_logits,             # [B, L, L, A, A] 或 [B, L, L, A*A] -> e_ij(a,b)
    aatype,                  # [B, L] (long)
    mask=None,               # [B, L] (float/bool), 1=有效
    pair_mask=None,          # [B, L, L] (float/bool), 1=有效
    lambda_reg: float = 1e-2,# L1+L2 系数（论文 0.01）
    temperature: float = 1.0,# 训练一般 1.0；与推理对齐可调
    reduction: str = "mean"  # "mean" | "sum" | "none"（返回 [B,L,L]）
):
    """
    Pairwise pseudo-likelihood (式(5)) 的无分块实现。
    返回: (loss, {"pair_ce":..., "pair_reg":...})
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
        pair_mask = torch.ones(B, L, L, device=device, dtype=dtype)
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
    idx_xj = aatype_clamped[:, None, None, None].expand(B, L, L, A).unsqueeze(-1)  # [B,L,L,A,1]
    eij_axj = torch.gather(pair_aa, dim=4, index=idx_xj).squeeze(-1)               # [B,L,L,A]

    # e_{j,i}(b, x_i)：交换 i/j 轴后按 x_i gather，再换回
    pair_aa_ji = pair_aa.transpose(1, 2).contiguous()
    idx_xi = aatype_clamped[:, None, None, None].expand(B, L, L, A).unsqueeze(-1)  # [B,L,L,A,1]
    eji_bxi = torch.gather(pair_aa_ji, dim=4, index=idx_xi).squeeze(-1)            # [B,L,L,A]
    eji_bxi = eji_bxi.transpose(1, 2).contiguous()                                  # [B,L,L,A]

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
    return loss, {"pair_ce": loss_ce.detach(), "pair_reg": loss_reg.detach()}

def margin_loss(p_pos, p_neg, m=1, rho=0.03):
    margin = F.relu(m+p_neg-p_pos)
    reg = rho * torch.sqrt(p_neg ** 2 + p_pos ** 2)
    return (margin + reg).mean()
    

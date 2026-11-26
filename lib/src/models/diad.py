"""DIAD model: shared encoder, fraud head, leakage head, HSIC disentanglement (stub + loss)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def hsic(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Biased HSIC estimator with RBF kernels (Gretton et al. 2005)."""
    def k(a):
        d = torch.cdist(a, a) ** 2
        return torch.exp(-d / (2 * sigma**2))
    n = x.size(0)
    H = torch.eye(n, device=x.device) - 1.0 / n
    return torch.trace(H @ k(x) @ H @ k(y)) / (n - 1) ** 2


def focal_bce(logits, y, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = torch.where(y == 1, p, 1 - p)
    return ((1 - pt) ** gamma * ce).mean()


class DIAD(nn.Module):
    def __init__(self, d_in: int, d_model: int = 64, d_head: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU())
        self.fraud_repr = nn.Sequential(nn.Linear(d_model, d_head), nn.GELU())
        self.leak_repr = nn.Sequential(nn.Linear(d_model + 1, d_head), nn.GELU())  # +1 = AE recon error
        self.fraud_out = nn.Linear(d_head, 1)
        self.leak_out = nn.Linear(d_head, 1)

    def forward(self, x, recon_err):
        z = self.encoder(x)
        zf = self.fraud_repr(z)
        zl = self.leak_repr(torch.cat([z, recon_err.unsqueeze(1)], 1))
        return self.fraud_out(zf).squeeze(1), self.leak_out(zl).squeeze(1), zf, zl


def diad_loss(model, x, recon_err, y_f, y_l, beta=0.1, gamma=2.0, eps=0.05):
    lf, ll, zf, zl = model(x, recon_err)
    loss_f = focal_bce(lf, y_f, gamma)
    loss_l = F.binary_cross_entropy_with_logits(ll, y_l)
    # adversarial augmentation on legit samples: push fraud head to be robust to small mimicry moves
    x_adv = x + eps * torch.sign(torch.randn_like(x)) * (y_f == 0).unsqueeze(1)
    lf_adv, _, _, _ = model(x_adv, recon_err)
    loss_adv = F.binary_cross_entropy_with_logits(lf_adv, y_f)
    loss_dis = hsic(zf, zl)
    return loss_f + loss_l + 0.5 * loss_adv + beta * loss_dis, {"f": loss_f.item(), "l": loss_l.item(), "dis": loss_dis.item()}

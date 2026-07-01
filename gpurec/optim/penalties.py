# gpurec/optim/penalties.py
"""Pure-torch regularization primitives for the first-order optimizer.

All quantities are in log2 (bits) units. Every function is a byte-identical
no-op at its disabled default (lam=0 / None). No Triton/CUDA dependency: this
module is CPU/fp64 gradcheck-testable in isolation.
"""
from __future__ import annotations

import torch


def _root_mask(sp_parent: torch.Tensor):
    """Return (nonroot_mask[S], parent_safe[S]). Root = (parent < 0) or self-parent."""
    S = sp_parent.shape[0]
    parent = sp_parent.reshape(-1).long()
    arange = torch.arange(S, device=parent.device)
    is_root = (parent < 0) | (parent == arange)
    nonroot = ~is_root
    parent_safe = torch.where(nonroot, parent, arange)  # root rows self-reference (masked out)
    return nonroot, parent_safe


def tv_prior_and_grad(theta: torch.Tensor, sp_parent: torch.Tensor, lam, eps: float = 1e-3):
    """Pseudo-Huber total-variation (fused-lasso L1) prior on adjacent parent-child log-rates.

    rho(d) = sqrt(d^2 + eps^2) - eps  (~ |d| for |d| >> eps)
    P      = sum_k lam_k * sum_{v != root} rho(theta[v,k] - theta[pa(v),k])
    rho'(d)= d / sqrt(d^2 + eps^2);  grad[v] += g, grad[pa(v)] -= g
    """
    if theta.ndim != 2:
        raise ValueError(f"theta must be [S, K], got {tuple(theta.shape)}")
    S, K = theta.shape
    device, dtype = theta.device, theta.dtype
    lam_v = torch.as_tensor(lam, device=device, dtype=dtype)
    if lam_v.ndim == 0:
        lam_v = lam_v.expand(K)
    elif lam_v.shape != (K,):
        raise ValueError(f"lam must be scalar or [{K}], got {tuple(lam_v.shape)}")

    nonroot, parent_safe = _root_mask(sp_parent.to(device))
    delta = theta - theta.index_select(0, parent_safe)                 # [S, K]
    delta = torch.where(nonroot.unsqueeze(1), delta, torch.zeros_like(delta))
    r = torch.sqrt(delta * delta + eps * eps)                          # [S, K]
    penalty = (lam_v * (r - eps).sum(dim=0)).sum()
    g = lam_v.unsqueeze(0) * delta / r                                 # rho'(d)
    g = torch.where(nonroot.unsqueeze(1), g, torch.zeros_like(g))
    grad = torch.zeros_like(theta) + g                                 # grad[v] += g
    grad.index_add_(0, parent_safe, -g)                               # grad[pa(v)] -= g
    return penalty, grad

# gpurec/optim/penalties.py
"""Pure-torch regularization primitives for the first-order optimizer.

All quantities are in log2 (bits) units. Every function is a byte-identical
no-op at its disabled default (lam=0 / None). No Triton/CUDA dependency: this
module is CPU/fp64 gradcheck-testable in isolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

DEFAULT_TV_EPS = 1e-3


def _root_mask(sp_parent: torch.Tensor):
    """Return (nonroot_mask[S], parent_safe[S]). Root = (parent < 0) or self-parent."""
    S = sp_parent.shape[0]
    parent = sp_parent.reshape(-1).long()
    arange = torch.arange(S, device=parent.device)
    is_root = (parent < 0) | (parent == arange)
    nonroot = ~is_root
    parent_safe = torch.where(nonroot, parent, arange)  # root rows self-reference (masked out)
    return nonroot, parent_safe


def tv_prior_and_grad(theta: torch.Tensor, sp_parent: torch.Tensor, lam, eps: float = DEFAULT_TV_EPS):
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


def _logsumexp2(x, dim=-1, keepdim=False):
    return torch.logsumexp(x * math.log(2.0), dim=dim, keepdim=keepdim) / math.log(2.0)


def origination_log_pO(omega: torch.Tensor) -> torch.Tensor:
    """log2 p^O = omega - logsumexp2(omega)."""
    return omega - _logsumexp2(omega, dim=-1, keepdim=True)


def origination_log_pO_floored(omega: torch.Tensor, floor: float) -> torch.Tensor:
    """log2 of p^O = (1-floor)*softmax(omega) + floor/S (differentiable anti-collapse mix)."""
    lp = origination_log_pO(omega)
    if floor <= 0.0:
        return lp
    S = omega.shape[-1]
    p = (1.0 - floor) * torch.exp2(lp) + floor / S
    return torch.log2(p)


def species_node_depths(sp_parent: torch.Tensor) -> torch.Tensor:
    """Integer hop-count from each species node to the root (root depth 0)."""
    parent = sp_parent.reshape(-1).long().cpu().tolist()
    S = len(parent)
    depth = torch.zeros(S, dtype=torch.float64)
    for v in range(S):
        d, cur = 0, v
        while cur >= 0 and parent[cur] >= 0 and parent[cur] != cur:
            cur = parent[cur]; d += 1
        depth[v] = d
    return depth


def species_root_index(sp_parent: torch.Tensor) -> int:
    parent = sp_parent.reshape(-1).long()
    arange = torch.arange(parent.shape[0], device=parent.device)
    is_root = (parent < 0) | (parent == arange)
    return int(torch.nonzero(is_root, as_tuple=False)[0].item())


@dataclass
class OriginationPenalty:
    l2: float = 0.0
    depth_lambda: float = 0.0
    depth: "torch.Tensor | None" = None
    root_lambda: float = 0.0
    root_index: "int | None" = None
    dirichlet_c: float = 0.0
    barrier_kind: str = "meanlog"   # meanlog | simpson | renyi2
    dirichlet_pi: "torch.Tensor | None" = None
    floor: float = 0.0

    def any_active(self) -> bool:
        return (self.l2 > 0 or self.depth_lambda > 0 or self.root_lambda > 0
                or self.dirichlet_c > 0)


def origination_penalty_value(omega: torch.Tensor, cfg: OriginationPenalty, *, sp_parent=None):
    """Forward-only sum of the enabled origination anti-collapse penalties (log2 units).

    Mirrors gergely_version wave_optimizer penalty VALUES: l2 acts on raw omega;
    depth/root/dirichlet act on p^O (optionally floored via origination_floor).
    barrier_kind selects meanlog | simpson | renyi2. The gradient is obtained by
    autograd in origination_penalty_and_grad (below), so it is exact for every term
    including the floored mixture, whose chain rule is easy to get wrong by hand.
    """
    dtype, device = omega.dtype, omega.device
    pen = omega.new_zeros(())
    # (1) L2 ridge on the raw logits: pen = c*Σ omega^2
    if cfg.l2 > 0:
        pen = pen + cfg.l2 * (omega * omega).sum()
    if cfg.depth_lambda > 0 or cfg.root_lambda > 0 or cfg.dirichlet_c > 0:
        log_pO = origination_log_pO_floored(omega, cfg.floor)
        p = torch.exp2(log_pO)                                        # [S] p^O
        # (2) expected-depth prior: pen = lam*E_{p^O}[depth]
        if cfg.depth_lambda > 0:
            depth = cfg.depth
            if depth is None:
                if sp_parent is None:
                    raise ValueError("depth_lambda>0 requires cfg.depth or sp_parent")
                depth = species_node_depths(sp_parent)
            depth = depth.to(device=device, dtype=dtype)
            pen = pen + cfg.depth_lambda * (p * depth).sum()
        # (3) root-mass prior: pen = lam*(1 - p^O_root)
        if cfg.root_lambda > 0:
            ridx = cfg.root_index if cfg.root_index is not None else species_root_index(sp_parent)
            pen = pen + cfg.root_lambda * (1.0 - p[ridx])
        # (4) anti-concentration barrier
        if cfg.dirichlet_c > 0:
            c = cfg.dirichlet_c
            if cfg.barrier_kind == "meanlog":
                pi = cfg.dirichlet_pi.to(device=device, dtype=dtype)
                pi = pi / pi.sum()
                pen = pen - c * (pi * log_pO).sum()
            elif cfg.barrier_kind == "simpson":
                pen = pen + c * (p * p).sum()
            elif cfg.barrier_kind == "renyi2":
                pen = pen + c * torch.log2((p * p).sum())
            else:
                raise ValueError(f"unknown barrier_kind {cfg.barrier_kind!r}")
    return pen


def origination_penalty_and_grad(omega: torch.Tensor, cfg: OriginationPenalty, *, sp_parent=None):
    """Penalty value + EXACT d(penalty)/d(omega) (log2 units).

    The gradient is taken with autograd on a detached copy of omega (wrapped in
    torch.enable_grad so it also works when called inside a no_grad forward). This
    is correct for the floored mixture and every barrier_kind, and it equals the
    hand-derived softmax-Jacobian gradients exactly when floor == 0.
    """
    grad = torch.zeros_like(omega)
    if not cfg.any_active():
        return omega.new_zeros(()), grad
    with torch.enable_grad():
        w = omega.detach().requires_grad_(True)
        pen = origination_penalty_value(w, cfg, sp_parent=sp_parent)
        (g,) = torch.autograd.grad(pen, w)
    return pen.detach(), g


def group_expand(theta_param: torch.Tensor, group_index) -> torch.Tensor:
    """[G,K] group rows -> [S,K] per-species (identity when group_index is None)."""
    if group_index is None:
        return theta_param
    return theta_param.index_select(0, group_index.to(theta_param.device).long())


def group_reduce(grad_full: torch.Tensor, group_index, n_groups) -> torch.Tensor:
    """[S,K] per-species grad -> [G,K] per-group grad (identity when group_index is None)."""
    if group_index is None:
        return grad_full
    gidx = group_index.to(grad_full.device).long()
    out = torch.zeros((int(n_groups), grad_full.shape[1]),
                      dtype=grad_full.dtype, device=grad_full.device)
    out.index_add_(0, gidx, grad_full)
    return out

"""Exact box-constrained multinomial M-step for the Coleman EM experiment.

The three-dimensional active set has only 3**3 = 27 possibilities.  Enumerating
them avoids the one-way pinning bug in the recovered implementation: coupling
through the softmax denominator can make an earlier pin invalid after another
coordinate is pinned.
"""
from __future__ import annotations

import itertools
import math

import torch


LN2 = math.log(2.0)
LO = math.log2(1e-6)
HI = math.log2(2.0)


def probabilities(theta: torch.Tensor) -> torch.Tensor:
    logits = torch.cat((torch.zeros_like(theta[:, :1]), theta), dim=1) * LN2
    return torch.softmax(logits, dim=1)


def surrogate(theta: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Rate-dependent part of Q(theta), in natural-log units."""
    z = 1.0 + torch.exp2(theta).sum(dim=1)
    return LN2 * (counts[:, 1:] * theta).sum(dim=1) - counts.sum(dim=1) * torch.log(z)


def m_step(
    counts: torch.Tensor,
    lo: float = LO,
    hi: float = HI,
    kkt_rtol: float = 2e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact boxed maximizers, active status, and max scaled KKT residual.

    ``status`` is -1 at the lower bound, 0 when free, and +1 at the upper
    bound.  Counts must be finite and nonnegative with a positive total.
    """
    if counts.ndim != 2 or counts.shape[1] != 4:
        raise ValueError(f"counts must have shape [F,4], got {tuple(counts.shape)}")
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
        raise ValueError("counts must be finite and nonnegative")
    totals = counts.sum(dim=1)
    if bool((totals <= 0).any()):
        raise ValueError("each family must have a positive total count")

    dtype, device = counts.dtype, counts.device
    ns, nk = counts[:, 0], counts[:, 1:]
    n_fam = counts.shape[0]
    best_q = torch.full((n_fam,), -torch.inf, dtype=dtype, device=device)
    best_theta = torch.zeros((n_fam, 3), dtype=dtype, device=device)
    best_status = torch.zeros((n_fam, 3), dtype=torch.int8, device=device)

    for state_tuple in itertools.product((-1, 0, 1), repeat=3):
        state = torch.tensor(state_tuple, dtype=torch.int8, device=device)
        lower = state == -1
        free = state == 0
        upper = state == 1
        pinned = ~free
        bound = torch.where(lower, torch.full((3,), lo, dtype=dtype, device=device),
                            torch.full((3,), hi, dtype=dtype, device=device))
        c = 1.0 + torch.where(pinned, torch.exp2(bound), torch.zeros_like(bound)).sum()
        denom = ns + (nk * pinned.to(dtype)).sum(dim=1)
        ratio = nk * c / denom[:, None]
        theta_free = torch.log2(ratio)
        theta = torch.where(pinned[None, :], bound[None, :], theta_free)

        finite_free = torch.where(free[None, :], torch.isfinite(theta), torch.ones_like(theta, dtype=torch.bool)).all(dim=1)
        feasible = finite_free & torch.where(
            free[None, :], (theta >= lo - 1e-12) & (theta <= hi + 1e-12),
            torch.ones_like(theta, dtype=torch.bool),
        ).all(dim=1)
        p = probabilities(theta)[:, 1:]
        dq = nk - totals[:, None] * p  # Q derivative / ln(2)
        tol = kkt_rtol * totals[:, None]
        kkt = torch.where(
            lower[None, :], dq <= tol,
            torch.where(upper[None, :], dq >= -tol, dq.abs() <= tol),
        ).all(dim=1)
        valid = feasible & kkt
        q = surrogate(theta, counts)
        take = valid & (q > best_q)
        best_q = torch.where(take, q, best_q)
        best_theta = torch.where(take[:, None], theta, best_theta)
        best_status = torch.where(take[:, None], state[None, :], best_status)

    if bool(~torch.isfinite(best_q).all()):
        bad = (~torch.isfinite(best_q)).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(f"no feasible KKT active set for families {bad[:20]}")
    p = probabilities(best_theta)[:, 1:]
    dq = nk - totals[:, None] * p
    lower, free, upper = best_status == -1, best_status == 0, best_status == 1
    signed_violation = torch.where(lower, torch.relu(dq), torch.where(upper, torch.relu(-dq), dq.abs()))
    residual = (signed_violation / totals[:, None]).amax(dim=1)
    return best_theta, best_status, residual


def complete_information(theta: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Expected complete-data information in bits per log2-rate squared."""
    p = probabilities(theta)[:, 1:]
    return LN2 * counts.sum(dim=1)[:, None, None] * (
        torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    )

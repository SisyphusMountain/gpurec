"""Small complete-history calculations for the optional genewise EM warm-up."""
from __future__ import annotations

import itertools
import math

import torch


def event_probabilities(theta: torch.Tensor) -> torch.Tensor:
    logits = torch.cat((torch.zeros_like(theta[:, :1]), theta), dim=1)
    return torch.softmax(logits * math.log(2.0), dim=1)


def boxed_em_m_step(counts: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Maximize the multinomial surrogate over the original three log-rate bounds.

    Counts are ordered S,D,L,T. All 27 active sets are checked, because pinning
    one rate can change whether another rate belongs on its bound. This small
    calculation runs in float64 on CPU during the warm-up.
    """
    if counts.ndim != 2 or counts.shape[1] != 4:
        raise ValueError("event counts must have shape [families, 4]")
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
        raise ValueError("event counts must be finite and nonnegative")
    total = counts.sum(dim=1)
    if bool((total <= 0).any()):
        raise ValueError("each family must have positive total event counts")
    ns, nk = counts[:, 0], counts[:, 1:]
    best_q = torch.full_like(total, -torch.inf)
    best_theta = torch.zeros_like(nk)
    for state_tuple in itertools.product((-1, 0, 1), repeat=3):
        state = torch.tensor(state_tuple, device=counts.device)
        lower, free, pinned = state == -1, state == 0, state != 0
        bound = torch.where(lower, counts.new_full((3,), lo), counts.new_full((3,), hi))
        c = 1.0 + torch.where(pinned, torch.exp2(bound), torch.zeros_like(bound)).sum()
        denominator = ns + (nk * pinned).sum(dim=1)
        theta = torch.where(pinned, bound, torch.log2(nk * c / denominator[:, None]))
        feasible = torch.where(free, torch.isfinite(theta) & (theta >= lo - 1e-12)
                               & (theta <= hi + 1e-12), True).all(dim=1)
        score = nk - total[:, None] * event_probabilities(theta)[:, 1:]
        tolerance = 2e-12 * total[:, None]
        kkt = torch.where(lower, score <= tolerance,
                          torch.where(free, score.abs() <= tolerance,
                                      score >= -tolerance)).all(dim=1)
        q = math.log(2.0) * (nk * theta).sum(dim=1) - total * torch.log1p(torch.exp2(theta).sum(dim=1))
        take = feasible & kkt & (q > best_q)
        best_q = torch.where(take, q, best_q)
        best_theta = torch.where(take[:, None], theta, best_theta)
    if not bool(torch.isfinite(best_q).all()):
        raise RuntimeError("no feasible EM M-step satisfies the rate-bound optimality conditions")
    return best_theta


def complete_information(theta: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Fixed-count surrogate Hessian in bits per squared log2-rate unit."""
    p = event_probabilities(theta)[:, 1:]
    return math.log(2.0) * counts.sum(dim=1)[:, None, None] * (
        torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    )

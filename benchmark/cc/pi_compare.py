"""Shared helpers for comparing two forward Pi solves entry by entry.

Path-neutral: they take whatever ``SolverOptions.forward_self_loop`` the caller configured and say
nothing about which one it is. Used by ``test_exact_forward.py`` and ``diagnose_exact_forward.py``;
they lived in the fused linear self-loop's own script until that mode was removed.
"""
from __future__ import annotations

import torch


def _abs_pi(pi_residual: torch.Tensor, pi_offset: torch.Tensor) -> torch.Tensor:
    """Represented absolute log2 Pi, in the offset (accumulator) dtype."""
    return pi_residual.to(pi_offset.dtype) + pi_offset.unsqueeze(1)


def _forward_abs_pi(static, theta_static, receiver_weights):
    from gpurec.core.inference.solver import solve_resident_e_pi

    with torch.no_grad():
        result = solve_resident_e_pi(
            static, theta_static, receiver_weights, warm_start_E=None, pi_iters=None,
            pi_residual_out=None,
        )
    pi_wave = result[5]
    state = static.pi_forward_state
    out = _abs_pi(pi_wave, state.pi_offset)
    del result, pi_wave
    return out


def _compare_pi(reference: torch.Tensor, candidate: torch.Tensor, window: float, eps: float) -> dict:
    """Row-max-windowed comparison of two absolute log2 Pi matrices."""
    finite_reference = torch.isfinite(reference)
    row_max = torch.where(finite_reference, reference, torch.full_like(reference, -float("inf")))
    row_max = row_max.amax(dim=1, keepdim=True)
    in_window = finite_reference & torch.isfinite(row_max) & (reference >= row_max - window)
    both_finite = finite_reference & torch.isfinite(candidate)
    difference = (candidate - reference).abs()
    windowed = in_window & both_finite
    total = reference.numel()
    largest_magnitude = float(reference[finite_reference].abs().max().item()) if bool(finite_reference.any()) else 0.0
    return {
        "entries": total,
        # Both paths carry the row's log2 values through fp32 frame shifts of this magnitude,
        # so this times fp32's 2**-24 is the floor on any disagreement between them.
        "max_abs_log2_value": largest_magnitude,
        "frame_resolution": largest_magnitude * eps,
        "finite_frac": float(finite_reference.sum().item()) / total,
        "in_window_frac": float(in_window.sum().item()) / total,
        "below_window_frac": float((finite_reference & ~in_window).sum().item()) / total,
        "max_abs_diff_in_window": float(difference[windowed].max().item()) if bool(windowed.any()) else 0.0,
        "max_abs_diff_all_finite": float(difference[both_finite].max().item()) if bool(both_finite.any()) else 0.0,
        "finite_mismatch": int((finite_reference ^ torch.isfinite(candidate)).sum().item()),
    }

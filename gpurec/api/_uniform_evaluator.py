"""Internal uniform-transfer evaluator helpers.

This module is support code for ``gpurec.api`` model internals, not a public
import surface.  It centralizes resident loss-only evaluation so no-gradient
model calls share the same E/Pi/root-likelihood boundary.
"""
from __future__ import annotations

import torch

from gpurec.core.forward import pi_root_row_loss_request
from gpurec.core.likelihood import compute_nll_root_rows

from .autograd import (
    ResidentESolveResult,
    ReconStaticState,
    ResidentSolveResult,
    _origination_probs_for_static,
    _record_forward_solver_stats,
    solve_resident_e,
    solve_resident_pi_given_e,
    solve_resident_e_pi,
)
from ._validation import require_default_objective


@torch.no_grad()
def evaluate_resident_no_grad(
    static: ReconStaticState,
    theta: torch.Tensor,
    *,
    per_family: bool = False,
    scratch_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return resident no-grad NLL for the active static state."""
    require_default_objective("GeneReconModel")

    solve = solve_resident_e_pi(
        static,
        theta,
        pi_request=pi_root_row_loss_request(),
        scratch_tensors=scratch_tensors,
    )
    _record_forward_solver_stats(static, solve.e_out, solve.pi_out)

    loss_vec = compute_nll_root_rows(
        solve.pi_out["Pi_root_rows"],
        solve.e_out["E"],
        _origination_probs_for_static(static),
        origination_probs_prepared=True,
    )
    static.warm_E = None
    return loss_vec.detach() if per_family else loss_vec.sum().detach()


@torch.no_grad()
def evaluate_resident_no_grad_with_solved_e(
    static: ReconStaticState,
    e_solve: ResidentESolveResult,
    *,
    per_family: bool = False,
    scratch_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
    origination_denominator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return resident no-grad NLL using a shared E solution."""
    require_default_objective("GeneReconModel")

    solve = solve_resident_pi_given_e(
        static,
        e_solve,
        pi_request=pi_root_row_loss_request(),
        scratch_tensors=scratch_tensors,
    )
    _record_forward_solver_stats(static, solve.e_out, solve.pi_out)

    loss_vec = compute_nll_root_rows(
        solve.pi_out["Pi_root_rows"],
        solve.e_out["E"],
        _origination_probs_for_static(static),
        origination_probs_prepared=True,
        denominator=origination_denominator,
    )
    static.warm_E = None
    return loss_vec.detach() if per_family else loss_vec.sum().detach()

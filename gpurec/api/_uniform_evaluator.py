"""Internal uniform-transfer evaluator helpers.

This module is support code for ``gpurec.api`` model internals, not a public
import surface.  It centralizes resident loss-only evaluation so no-gradient
model calls share the same E/Pi/root-likelihood boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import E_fixed_point, compute_nll_root_rows

from .autograd import ReconStaticState, _extract_parameters
from ._validation import require_default_objective


@dataclass(frozen=True)
class ResidentSolveResult:
    theta: torch.Tensor
    e_out: dict[str, Any]
    pi_out: dict[str, Any]
    log_p_s: torch.Tensor
    log_p_d: torch.Tensor
    log_p_l: torch.Tensor
    max_transfer: torch.Tensor


def _record_forward_solver_stats(
    static: ReconStaticState,
    e_out: dict[str, Any],
    pi_out: dict[str, Any],
) -> None:
    static.last_solver_stats = {
        "E_iterations": int(e_out["iterations"]),
        "E_convergence_delta": e_out.get("E_convergence_delta"),
        "Pi_max_iterations": int(
            pi_out.get("Pi_max_iterations", static.fixed_iters_Pi)
        ),
        "Pi_wave_iterations": [
            int(value) for value in pi_out.get("Pi_wave_iterations", [])
        ],
        "Pi_converged_waves": int(sum(pi_out.get("Pi_wave_converged", []))),
        "Pi_wave_count": int(len(pi_out.get("Pi_wave_converged", []))),
    }


def solve_resident_e_pi(
    static: ReconStaticState,
    theta: torch.Tensor,
    *,
    return_original: bool,
    return_root_rows: bool,
) -> ResidentSolveResult:
    """Solve resident E and Pi tensors without owning caller side effects."""
    theta_eval = theta.detach().to(device=static.device, dtype=static.dtype)
    log_pS, log_pD, log_pL, max_transfer_vec = _extract_parameters(theta_eval, static)
    e_max_iters = (
        static.fixed_iters_E
        if static.fixed_iters_E is not None
        else static.max_iters_E
    )
    e_tolerance = (
        static.e_logsumexp_tol
        if static.adaptive_iters
        else (-1.0 if static.fixed_iters_E is not None else static.tol_E)
    )
    e_out = E_fixed_point(
        species_helpers=static.species_helpers,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        max_transfer_mat=max_transfer_vec,
        max_iters=e_max_iters,
        tolerance=e_tolerance,
        warm_start_E=None,
        dtype=static.dtype,
        device=static.device,
        ancestors_T=static.ancestors_T,
        check_interval=static.convergence_check_interval,
        convergence_metric="logsumexp" if static.adaptive_iters else "max_diff",
    )

    pi_out = Pi_wave_forward(
        wave_layout=static.wave_layout,
        species_helpers=static.species_helpers,
        E=e_out["E"],
        Ebar=e_out["E_bar"],
        E_s1=e_out["E_s1"],
        E_s2=e_out["E_s2"],
        log_pS=log_pS,
        log_pD=log_pD,
        max_transfer_mat=max_transfer_vec,
        device=static.device,
        dtype=static.dtype,
        fixed_iters=static.fixed_iters_Pi,
        return_original=return_original,
        return_root_rows=return_root_rows,
        family_idx=static.wave_layout.get("family_idx") if static.genewise else None,
        convergence_tolerance=(
            static.pi_max_diff_tol if static.adaptive_iters else -1.0
        ),
        convergence_check_interval=static.convergence_check_interval,
    )
    return ResidentSolveResult(
        theta=theta_eval,
        e_out=e_out,
        pi_out=pi_out,
        log_p_s=log_pS,
        log_p_d=log_pD,
        log_p_l=log_pL,
        max_transfer=max_transfer_vec,
    )


@torch.no_grad()
def evaluate_resident_no_grad(
    static: ReconStaticState,
    theta: torch.Tensor,
    *,
    per_family: bool = False,
) -> torch.Tensor:
    """Return resident no-grad NLL for the active static state."""
    require_default_objective("GeneReconModel")

    solve = solve_resident_e_pi(
        static,
        theta,
        return_original=False,
        return_root_rows=True,
    )
    _record_forward_solver_stats(static, solve.e_out, solve.pi_out)

    loss_vec = compute_nll_root_rows(
        solve.pi_out["Pi_root_rows"],
        solve.e_out["E"],
        static.origination_probs,
        origination_probs_prepared=True,
    )
    static.warm_E = None
    return loss_vec.detach() if per_family else loss_vec.sum().detach()

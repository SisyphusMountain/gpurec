"""Autograd bridge: wraps the existing implicit-gradient pipeline as a
``torch.autograd.Function`` so a notebook user can call standard
``loss.backward()`` and use any ``torch.optim`` optimizer.

The forward pass mirrors the retained wave-ordered inference path and the
backward pass delegates to the existing
:func:`gpurec.optimization.implicit_grad.implicit_grad_loglik_vjp_wave`.
No new gradient math is written here.

Sign convention: the core likelihood helper is ``compute_nll``.  The bridge
keeps the NLL convention and returns NLL from ``forward()``, so users write
``loss = model(); loss.backward()`` directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from gpurec.core.likelihood import E_fixed_point, compute_nll
from gpurec.core.forward import Pi_wave_forward
from gpurec.core._helpers import _nvtx_range
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.optimization.implicit_grad import (
    implicit_grad_loglik_vjp_wave,
)
from ._validation import require_default_objective


@dataclass
class ReconStaticState:
    """Container for non-differentiable state shared across ``forward()`` calls.

    Built once by :class:`gpurec.api.model.GeneReconModel` from a
    :class:`GeneDataset`. Mutated only via ``warm_E`` (warm start cache).
    """

    device: torch.device
    dtype: torch.dtype

    # Wave layout + likelihood inputs (precomputed once)
    wave_layout: dict[str, Any]
    species_helpers: dict[str, Any]
    unnorm_row_max: torch.Tensor                              # [S]
    ancestors_T: Optional[torch.Tensor]                       # sparse COO (uniform only)

    # Mode flags (mapped from "global" / "specieswise" / "genewise")
    genewise: bool
    specieswise: bool
    origination_probs: Optional[torch.Tensor] = None           # [S] or [G, S]

    # Solver knobs
    fixed_iters_E: Optional[int] = None
    max_iters_E: int = 2000
    tol_E: float = 1e-8
    fixed_iters_Pi: int = 6
    neumann_terms: int = 3
    adaptive_iters: bool = False
    convergence_check_interval: int = 4
    e_logsumexp_tol: float = 1e-5
    pi_max_diff_tol: float = 1e-5
    gradient_change_tol: float = 1e-4
    gradient_change_rtol: float = 1e-4
    use_pruning: bool = True
    pruning_threshold: float = 1e-6

    # Warm start cache, mutated across calls
    warm_E: Optional[torch.Tensor] = None
    clear_runtime_after_backward: bool = False
    last_solver_stats: Optional[dict[str, Any]] = None


def _extract_parameters(theta: torch.Tensor, static: ReconStaticState):
    """Extract parameters for the retained uniform-transfer path."""
    return (
        extract_parameters_uniform(
            theta,
            static.unnorm_row_max,
            specieswise=static.specieswise,
            genewise=static.genewise,
        )
    )


def _record_backward_solver_stats(static: ReconStaticState, stats: Any) -> None:
    """Attach backward solver telemetry to the last forward solver record.

    E-adjoint solver failure is recorded as diagnostic telemetry here; this
    helper does not convert ``success=False`` into a failed optimization step.
    """
    if static.last_solver_stats is None:
        static.last_solver_stats = {}
    neumann_terms = getattr(stats, "neumann_terms", None)
    if neumann_terms is None:
        neumann_terms = static.neumann_terms
    static.last_solver_stats.update(
        {
            "Neumann_terms": int(neumann_terms),
            "E_adjoint_iterations": int(getattr(stats, "iters", 0)),
            "E_adjoint_rel_res": float(getattr(stats, "rel_res", float("nan"))),
            "E_adjoint_success": bool(getattr(stats, "success", True)),
        }
    )
    for source, target in (
        ("gradient_convergence_delta", "Gradient_convergence_delta"),
        ("gradient_convergence_threshold", "Gradient_convergence_threshold"),
        ("gradient_converged", "Gradient_converged"),
    ):
        value = getattr(stats, source, None)
        if value is not None:
            static.last_solver_stats[target] = value


class _GeneReconFunction(torch.autograd.Function):
    """``forward`` runs the existing E + Pi pipeline; ``backward`` calls the
    existing implicit gradient. Inputs other than ``theta`` are treated as
    constants by autograd (passed via the static dataclass)."""

    @staticmethod
    def forward(ctx, theta: torch.Tensor, static: ReconStaticState, reduce: str):
        require_default_objective("GeneReconModel")
        if reduce not in ("sum", "per_family"):
            raise ValueError(f"reduce must be 'sum' or 'per_family', got {reduce!r}")
        if reduce == "per_family" and not static.genewise:
            raise ValueError(
                "reduce='per_family' is only valid in genewise mode."
            )

        device = static.device
        dtype = static.dtype

        with torch.no_grad():
            # 1. Extract parameters
            with _nvtx_range("forward extract parameters"):
                log_pS, log_pD, log_pL, max_transfer_vec = (
                    _extract_parameters(theta, static)
                )

            # 2. E fixed-point with warm start
            with _nvtx_range("forward E fixed point"):
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
                E_out = E_fixed_point(
                    species_helpers=static.species_helpers,
                    log_pS=log_pS,
                    log_pD=log_pD,
                    log_pL=log_pL,
                    max_transfer_mat=max_transfer_vec,
                    max_iters=e_max_iters,
                    tolerance=e_tolerance,
                    warm_start_E=(
                        None if static.fixed_iters_E is not None else static.warm_E
                    ),
                    dtype=dtype,
                    device=device,
                    ancestors_T=static.ancestors_T,
                    check_interval=static.convergence_check_interval,
                    convergence_metric=(
                        "logsumexp" if static.adaptive_iters else "max_diff"
                    ),
                )
                E = E_out["E"]
                E_s1 = E_out["E_s1"]
                E_s2 = E_out["E_s2"]
                Ebar = E_out["E_bar"]

            # 3. Pi wave forward
            with _nvtx_range("forward Pi waves"):
                Pi_out = Pi_wave_forward(
                    wave_layout=static.wave_layout,
                    species_helpers=static.species_helpers,
                    E=E,
                    Ebar=Ebar,
                    E_s1=E_s1,
                    E_s2=E_s2,
                    log_pS=log_pS,
                    log_pD=log_pD,
                    max_transfer_mat=max_transfer_vec,
                    device=device,
                    dtype=dtype,
                    fixed_iters=static.fixed_iters_Pi,
                    return_original=False,
                    family_idx=(
                        static.wave_layout.get("family_idx") if static.genewise else None
                    ),
                    convergence_tolerance=(
                        static.pi_max_diff_tol if static.adaptive_iters else -1.0
                    ),
                    convergence_check_interval=static.convergence_check_interval,
                )
                static.last_solver_stats = {
                    "E_iterations": int(E_out["iterations"]),
                    "E_convergence_delta": E_out.get("E_convergence_delta"),
                    "Pi_max_iterations": int(
                        Pi_out.get("Pi_max_iterations", static.fixed_iters_Pi)
                    ),
                    "Pi_wave_iterations": [
                        int(value) for value in Pi_out.get("Pi_wave_iterations", [])
                    ],
                    "Pi_converged_waves": int(
                        sum(Pi_out.get("Pi_wave_converged", []))
                    ),
                    "Pi_wave_count": int(len(Pi_out.get("Pi_wave_converged", []))),
                }

            # 4. NLL: nll_vec is per-family.
            with _nvtx_range("forward root likelihood"):
                nll_vec = compute_nll(
                    Pi_out["Pi_wave_ordered"],
                    E,
                    static.wave_layout["root_clade_ids"],
                    static.origination_probs,
                    origination_probs_prepared=True,
                )

        # 5. Save state for backward.
        with _nvtx_range("forward save outputs"):
            ctx.save_for_backward(
                theta,
                Pi_out["Pi_wave_ordered"],
                Pi_out["Pibar_wave_ordered"],
                E,
                E_s1,
                E_s2,
                Ebar,
                log_pS,
                log_pD,
                log_pL,
                max_transfer_vec,
                (
                    Pi_out["uniform_pibar_row_max"]
                    if Pi_out.get("uniform_pibar_row_max") is not None
                    else theta.new_empty(0)
                ),
            )
            ctx.static = static
            ctx.reduce = reduce

        # 6. Update warm-start cache (in-place mutation of the shared static).
        with _nvtx_range("forward reduce"):
            static.warm_E = (
                None
                if static.fixed_iters_E is not None or static.clear_runtime_after_backward
                else E.detach()
            )

            # 7. Reduce.
            return nll_vec.sum() if reduce == "sum" else nll_vec

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (
            theta,
            Pi_star_wave,
            Pibar_star_wave,
            E_star,
            E_s1,
            E_s2,
            Ebar,
            log_pS,
            log_pD,
            log_pL,
            max_transfer_vec,
            uniform_pibar_row_max,
        ) = ctx.saved_tensors
        static: ReconStaticState = ctx.static

        wave_layout = static.wave_layout

        if static.genewise:
            # Cross-family batched genewise path. This is the retained
            # per-family gradient path used by PyTorch optimizers and
            # BatchedLBFGS.
            grad_theta, _stats = implicit_grad_loglik_vjp_wave(
                wave_layout,
                static.species_helpers,
                Pi_star_wave=Pi_star_wave,
                Pibar_star_wave=Pibar_star_wave,
                E_star=E_star,
                Ebar=Ebar,
                E_s1=E_s1,
                E_s2=E_s2,
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                max_transfer_mat=max_transfer_vec,
                root_clade_ids_perm=wave_layout["root_clade_ids"],
                theta=theta,
                unnorm_row_max=static.unnorm_row_max,
                specieswise=static.specieswise,
                device=static.device,
                dtype=static.dtype,
                neumann_terms=static.neumann_terms,
                use_pruning=static.use_pruning,
                pruning_threshold=static.pruning_threshold,
                ancestors_T=static.ancestors_T,
                uniform_pibar_row_max=(
                    uniform_pibar_row_max
                    if uniform_pibar_row_max.numel() > 0
                    else None
                ),
                origination_probs=static.origination_probs,
                origination_probs_prepared=True,
                genewise=True,
                family_idx=wave_layout["family_idx"],
                gradient_convergence_tol=(
                    static.gradient_change_tol if static.adaptive_iters else -1.0
                ),
                gradient_convergence_rtol=static.gradient_change_rtol,
                gradient_convergence_check_interval=static.convergence_check_interval,
            )
        else:
            # Shared theta path: delegate to the public wrapper.
            grad_theta, _stats = implicit_grad_loglik_vjp_wave(
                wave_layout,
                static.species_helpers,
                Pi_star_wave=Pi_star_wave,
                Pibar_star_wave=Pibar_star_wave,
                E_star=E_star,
                E_s1=E_s1,
                E_s2=E_s2,
                Ebar=Ebar,
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                max_transfer_mat=max_transfer_vec,
                root_clade_ids_perm=wave_layout["root_clade_ids"],
                theta=theta,
                unnorm_row_max=static.unnorm_row_max,
                specieswise=static.specieswise,
                device=static.device,
                dtype=static.dtype,
                neumann_terms=static.neumann_terms,
                use_pruning=static.use_pruning,
                pruning_threshold=static.pruning_threshold,
                ancestors_T=static.ancestors_T,
                uniform_pibar_row_max=(
                    uniform_pibar_row_max
                    if uniform_pibar_row_max.numel() > 0
                    else None
                ),
                origination_probs=static.origination_probs,
                origination_probs_prepared=True,
                gradient_convergence_tol=(
                    static.gradient_change_tol if static.adaptive_iters else -1.0
                ),
                gradient_convergence_rtol=static.gradient_change_rtol,
                gradient_convergence_check_interval=static.convergence_check_interval,
            )

        _record_backward_solver_stats(static, _stats)

        # grad_theta is d(NLL_total)/d(theta). The forward returned NLL_total
        # (or NLL_per_family). No sign flip required.
        if ctx.reduce == "sum":
            # grad_output is a scalar.
            grad_theta = grad_theta * grad_output
        else:
            # per_family (genewise): grad_output is [G]; broadcast across the
            # remaining theta dims.
            gvec = grad_output.view((-1,) + (1,) * (theta.ndim - 1))
            grad_theta = grad_theta * gvec

        if static.clear_runtime_after_backward:
            static.warm_E = None

        return grad_theta, None, None

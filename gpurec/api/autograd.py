"""Autograd bridge: wraps the existing implicit-gradient pipeline as a
``torch.autograd.Function`` so a notebook user can call standard
``loss.backward()`` and use any ``torch.optim`` optimizer.

The forward pass mirrors the retained wave-ordered inference path and the
backward pass delegates to the existing
:func:`gpurec.optimization.implicit_grad.implicit_grad_loglik_vjp_wave`.
No new gradient math is written here.

Sign convention: the core likelihood helper is ``compute_nll_root_rows``.  The
bridge keeps the NLL convention and returns NLL from ``forward()``, so users
write ``loss = model(); loss.backward()`` directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from gpurec.core.likelihood import E_fixed_point, compute_nll_root_rows
from gpurec.core.forward import (
    _PiForwardRequest,
    pi_training_state_request,
)
from gpurec.core._helpers import _nvtx_range
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.origination import PreparedOriginationPrior
from gpurec.core.parameter_layout import ParameterLayout, RateMode
from gpurec.optimization.implicit_grad import (
    implicit_grad_loglik_vjp_wave,
)
from ._validation import require_default_objective


@dataclass
class ReconStaticState:
    """Container for non-differentiable state shared across ``forward()`` calls.

    Built once by :class:`gpurec.api.model.GeneReconModel` from a
    :class:`GeneDataset`. Runtime warm-start caches are mutable and must be
    cleared when the owning model discards solver state.
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
    origination_prior: Optional[PreparedOriginationPrior] = None
    origination_probs: Optional[torch.Tensor] = None           # [S] or [G, S]

    # Solver knobs
    fixed_iters_E: Optional[int] = None
    max_iters_E: int = 2000
    tol_E: float = 1e-8
    fixed_iters_Pi: int = 6
    neumann_terms: int = 3
    adaptive_iters: bool = False
    adaptive_neumann_terms: bool = False
    convergence_check_interval: int = 4
    e_logsumexp_tol: float = 1e-5
    pi_max_diff_tol: float = 1e-5
    gradient_change_tol: float = 1e-4
    gradient_change_rtol: float = 1e-4
    use_pruning: bool = True
    pruning_threshold: float = 1e-6

    # Warm start cache, mutated across calls
    warm_E: Optional[torch.Tensor] = None
    pi_adjoint_warmstart: bool = False
    pi_adjoint_cache: Optional[torch.Tensor] = None
    pi_adjoint_pending_cache: Optional[torch.Tensor] = None
    pi_adjoint_cache_update_mode: str = "immediate"
    pi_fixed_point_relaxation: float = 1.0
    clear_runtime_after_backward: bool = False
    last_solver_stats: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ResidentESolveResult:
    theta: torch.Tensor
    e_out: dict[str, Any]
    log_p_s: torch.Tensor
    log_p_d: torch.Tensor
    log_p_l: torch.Tensor
    max_transfer: torch.Tensor


@dataclass(frozen=True)
class ResidentSolveResult:
    theta: torch.Tensor
    e_out: dict[str, Any]
    pi_out: dict[str, Any]
    log_p_s: torch.Tensor
    log_p_d: torch.Tensor
    log_p_l: torch.Tensor
    max_transfer: torch.Tensor


@dataclass(frozen=True)
class ResidentGradientForwardResult:
    solve: ResidentSolveResult
    loss_vec: torch.Tensor


def _origination_probs_for_static(static: ReconStaticState) -> torch.Tensor | None:
    prior = getattr(static, "origination_prior", None)
    if prior is None:
        return static.origination_probs
    return prior.probs


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


def _static_family_count(static: ReconStaticState) -> int:
    roots = static.wave_layout.get("root_clade_ids")
    if torch.is_tensor(roots):
        return int(roots.numel())
    if roots is None:
        raise ValueError("static wave layout is missing root_clade_ids")
    return len(roots)


def _e_shape_for_static(static: ReconStaticState) -> tuple[int, ...]:
    layout = ParameterLayout.for_mode(
        RateMode.from_flags(
            genewise=bool(static.genewise),
            specieswise=bool(static.specieswise),
        ),
        species_count=int(static.species_helpers["S"]),
        family_count=_static_family_count(static),
    )
    return layout.e_shape


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
        ("pi_adjoint_residual_absmax", "Pi_adjoint_residual_absmax"),
        ("pi_adjoint_residual_relmax", "Pi_adjoint_residual_relmax"),
        ("pi_adjoint_residual_wave_count", "Pi_adjoint_residual_wave_count"),
    ):
        value = getattr(stats, source, None)
        if value is not None:
            static.last_solver_stats[target] = value


def _pi_adjoint_initial_guess(
    static: ReconStaticState,
    pi_wave_ordered: torch.Tensor,
) -> torch.Tensor | None:
    """Return a valid cached Pi adjoint for the current resident layout."""
    if not bool(getattr(static, "pi_adjoint_warmstart", False)):
        return None
    cache = getattr(static, "pi_adjoint_cache", None)
    if cache is None:
        return None
    if not torch.is_tensor(cache) or tuple(cache.shape) != tuple(pi_wave_ordered.shape):
        static.pi_adjoint_cache = None
        return None
    return cache.detach().to(
        device=pi_wave_ordered.device,
        dtype=pi_wave_ordered.dtype,
    )


def _pi_adjoint_cache_update_mode(static: ReconStaticState) -> str:
    mode = getattr(static, "pi_adjoint_cache_update_mode", "immediate")
    if mode not in {"immediate", "stage"}:
        raise ValueError(
            "pi_adjoint_cache_update_mode must be 'immediate' or 'stage'"
        )
    return str(mode)


def _clear_pi_adjoint_runtime_cache(static: ReconStaticState) -> None:
    if hasattr(static, "pi_adjoint_cache"):
        static.pi_adjoint_cache = None
    if hasattr(static, "pi_adjoint_pending_cache"):
        static.pi_adjoint_pending_cache = None


def _commit_pi_adjoint_pending_cache(static: ReconStaticState) -> bool:
    pending = getattr(static, "pi_adjoint_pending_cache", None)
    if not torch.is_tensor(pending):
        static.pi_adjoint_pending_cache = None
        return False
    static.pi_adjoint_cache = pending.detach()
    static.pi_adjoint_pending_cache = None
    return True


def _discard_pi_adjoint_pending_cache(static: ReconStaticState) -> bool:
    had_pending = torch.is_tensor(getattr(static, "pi_adjoint_pending_cache", None))
    static.pi_adjoint_pending_cache = None
    return bool(had_pending)


def _clear_post_gradient_runtime_cache(static: ReconStaticState) -> None:
    """Clear transient gradient state while preserving staged Pi commits."""
    if hasattr(static, "warm_E"):
        static.warm_E = None
    if (
        bool(getattr(static, "pi_adjoint_warmstart", False))
        and _pi_adjoint_cache_update_mode(static) == "stage"
    ):
        return
    _clear_pi_adjoint_runtime_cache(static)


def _store_pi_adjoint_cache(
    static: ReconStaticState,
    cache: torch.Tensor | None,
) -> tuple[str, torch.Tensor | None, torch.Tensor | None]:
    mode = _pi_adjoint_cache_update_mode(static)
    if not torch.is_tensor(cache):
        if mode == "stage":
            static.pi_adjoint_pending_cache = None
        else:
            static.pi_adjoint_cache = None
        return mode, getattr(static, "pi_adjoint_cache", None), None
    stored = cache.detach()
    if mode == "stage":
        static.pi_adjoint_pending_cache = stored
    else:
        static.pi_adjoint_cache = stored
        static.pi_adjoint_pending_cache = None
    return (
        mode,
        getattr(static, "pi_adjoint_cache", None),
        getattr(static, "pi_adjoint_pending_cache", None),
    )


def _record_pi_adjoint_cache_stats(
    static: ReconStaticState,
    *,
    used_initial_guess: bool,
    cache: torch.Tensor | None,
    pending_cache: torch.Tensor | None = None,
) -> None:
    if static.last_solver_stats is None:
        static.last_solver_stats = {}
    static.last_solver_stats.update(
        {
            "Pi_adjoint_warmstart_enabled": bool(
                getattr(static, "pi_adjoint_warmstart", False)
            ),
            "Pi_adjoint_warmstart_used": bool(used_initial_guess),
            "Pi_adjoint_cache_update_mode": _pi_adjoint_cache_update_mode(static),
            "Pi_adjoint_cache_pending": torch.is_tensor(pending_cache),
        }
    )
    if cache is not None:
        static.last_solver_stats["Pi_adjoint_cache_rows"] = int(cache.shape[0])
        static.last_solver_stats["Pi_adjoint_cache_species"] = int(cache.shape[1])


def solve_resident_e(
    static: ReconStaticState,
    theta: torch.Tensor,
    *,
    warm_start_E: torch.Tensor | None = None,
) -> ResidentESolveResult:
    """Solve resident E tensors and extract uniform-transfer parameters."""
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
        warm_start_E=warm_start_E,
        dtype=static.dtype,
        device=static.device,
        ancestors_T=static.ancestors_T,
        check_interval=static.convergence_check_interval,
        convergence_metric="logsumexp" if static.adaptive_iters else "max_diff",
        e_shape=_e_shape_for_static(static),
    )
    return ResidentESolveResult(
        theta=theta_eval,
        e_out=e_out,
        log_p_s=log_pS,
        log_p_d=log_pD,
        log_p_l=log_pL,
        max_transfer=max_transfer_vec,
    )


def solve_resident_pi_given_e(
    static: ReconStaticState,
    e_solve: ResidentESolveResult,
    *,
    pi_request: _PiForwardRequest,
    scratch_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> ResidentSolveResult:
    """Solve resident Pi tensors from a precomputed E solution."""
    e_out = e_solve.e_out
    pi_out = pi_request.run(
        wave_layout=static.wave_layout,
        species_helpers=static.species_helpers,
        E=e_out["E"],
        Ebar=e_out["E_bar"],
        E_s1=e_out["E_s1"],
        E_s2=e_out["E_s2"],
        log_pS=e_solve.log_p_s,
        log_pD=e_solve.log_p_d,
        max_transfer_mat=e_solve.max_transfer,
        device=static.device,
        dtype=static.dtype,
        fixed_iters=static.fixed_iters_Pi,
        family_idx=static.wave_layout.get("family_idx") if static.genewise else None,
        convergence_tolerance=(
            static.pi_max_diff_tol if static.adaptive_iters else -1.0
        ),
        convergence_check_interval=static.convergence_check_interval,
        scratch_tensors=scratch_tensors,
    )
    return ResidentSolveResult(
        theta=e_solve.theta,
        e_out=e_out,
        pi_out=pi_out,
        log_p_s=e_solve.log_p_s,
        log_p_d=e_solve.log_p_d,
        log_p_l=e_solve.log_p_l,
        max_transfer=e_solve.max_transfer,
    )


def solve_resident_e_pi(
    static: ReconStaticState,
    theta: torch.Tensor,
    *,
    pi_request: _PiForwardRequest,
    warm_start_E: torch.Tensor | None = None,
    scratch_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> ResidentSolveResult:
    """Solve resident E and Pi tensors without owning caller side effects."""
    e_solve = solve_resident_e(static, theta, warm_start_E=warm_start_E)
    return solve_resident_pi_given_e(
        static,
        e_solve,
        pi_request=pi_request,
        scratch_tensors=scratch_tensors,
    )


def evaluate_resident_gradient_forward(
    static: ReconStaticState,
    theta: torch.Tensor,
    *,
    warm_start_E: torch.Tensor | None = None,
) -> ResidentGradientForwardResult:
    """Return the shared resident forward solve used by gradient paths."""
    require_default_objective("GeneReconModel")
    with _nvtx_range("resident E/Pi solve"):
        solve = solve_resident_e_pi(
            static,
            theta,
            pi_request=pi_training_state_request(),
            warm_start_E=warm_start_E,
        )
        _record_forward_solver_stats(static, solve.e_out, solve.pi_out)

    with _nvtx_range("resident root likelihood"):
        root_clade_ids = static.wave_layout["root_clade_ids"]
        root_rows = solve.pi_out["Pi_wave_ordered"][root_clade_ids, :]
        loss_vec = compute_nll_root_rows(
            root_rows,
            solve.e_out["E"],
            _origination_probs_for_static(static),
            origination_probs_prepared=True,
        )
    return ResidentGradientForwardResult(solve=solve, loss_vec=loss_vec)


def compute_resident_implicit_gradient(
    static: ReconStaticState,
    *,
    theta: torch.Tensor,
    pi_wave_ordered: torch.Tensor,
    pibar_wave_ordered: torch.Tensor,
    e: torch.Tensor,
    ebar: torch.Tensor,
    e_s1: torch.Tensor,
    e_s2: torch.Tensor,
    log_p_s: torch.Tensor,
    log_p_d: torch.Tensor,
    log_p_l: torch.Tensor,
    max_transfer: torch.Tensor,
    uniform_pibar_row_max: torch.Tensor | None,
) -> torch.Tensor:
    """Compute and record the resident implicit gradient for a forward solve."""
    if uniform_pibar_row_max is not None and uniform_pibar_row_max.numel() == 0:
        uniform_pibar_row_max = None
    use_pi_adjoint_warmstart = bool(
        getattr(static, "pi_adjoint_warmstart", False)
    )
    pi_adjoint_initial_guess = _pi_adjoint_initial_guess(static, pi_wave_ordered)
    result = implicit_grad_loglik_vjp_wave(
        static.wave_layout,
        static.species_helpers,
        Pi_star_wave=pi_wave_ordered,
        Pibar_star_wave=pibar_wave_ordered,
        E_star=e,
        Ebar=ebar,
        E_s1=e_s1,
        E_s2=e_s2,
        log_pS=log_p_s,
        log_pD=log_p_d,
        log_pL=log_p_l,
        max_transfer_mat=max_transfer,
        root_clade_ids_perm=static.wave_layout["root_clade_ids"],
        theta=theta,
        unnorm_row_max=static.unnorm_row_max,
        specieswise=static.specieswise,
        device=static.device,
        dtype=static.dtype,
        neumann_terms=static.neumann_terms,
        use_pruning=static.use_pruning,
        pruning_threshold=static.pruning_threshold,
        ancestors_T=static.ancestors_T,
        family_idx=static.wave_layout["family_idx"] if static.genewise else None,
        uniform_pibar_row_max=uniform_pibar_row_max,
        origination_probs=_origination_probs_for_static(static),
        origination_probs_prepared=True,
        genewise=static.genewise,
        gradient_convergence_tol=(
            static.gradient_change_tol if static.adaptive_neumann_terms else -1.0
        ),
        gradient_convergence_rtol=static.gradient_change_rtol,
        gradient_convergence_check_interval=static.convergence_check_interval,
        pi_adjoint_initial_guess=pi_adjoint_initial_guess,
        return_aux=use_pi_adjoint_warmstart,
        record_pi_adjoint_residual=use_pi_adjoint_warmstart,
        pi_fixed_point_relaxation=getattr(static, "pi_fixed_point_relaxation", 1.0),
    )
    if use_pi_adjoint_warmstart:
        grad_theta, stats, aux = result
        _cache_mode, pi_adjoint_cache, pi_adjoint_pending_cache = (
            _store_pi_adjoint_cache(static, aux.get("pi_adjoint"))
        )
        _record_backward_solver_stats(static, stats)
        _record_pi_adjoint_cache_stats(
            static,
            used_initial_guess=bool(aux.get("used_pi_initial_guess", False)),
            cache=pi_adjoint_cache,
            pending_cache=pi_adjoint_pending_cache,
        )
        return grad_theta
    grad_theta, stats = result
    _record_backward_solver_stats(static, stats)
    return grad_theta


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

        with torch.no_grad():
            # 1. Resident E/Pi solve with the autograd warm-start policy.
            with _nvtx_range("forward resident gradient evaluation"):
                gradient_forward = evaluate_resident_gradient_forward(
                    static,
                    theta,
                    warm_start_E=(
                        None if static.fixed_iters_E is not None else static.warm_E
                    ),
                )
                solve = gradient_forward.solve
                E_out = solve.e_out
                Pi_out = solve.pi_out
                E = E_out["E"]
                E_s1 = E_out["E_s1"]
                E_s2 = E_out["E_s2"]
                Ebar = E_out["E_bar"]
                log_pS = solve.log_p_s
                log_pD = solve.log_p_d
                log_pL = solve.log_p_l
                max_transfer_vec = solve.max_transfer
                nll_vec = gradient_forward.loss_vec

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
        grad_theta = compute_resident_implicit_gradient(
            static,
            theta=theta,
            pi_wave_ordered=Pi_star_wave,
            pibar_wave_ordered=Pibar_star_wave,
            e=E_star,
            ebar=Ebar,
            e_s1=E_s1,
            e_s2=E_s2,
            log_p_s=log_pS,
            log_p_d=log_pD,
            log_p_l=log_pL,
            max_transfer=max_transfer_vec,
            uniform_pibar_row_max=uniform_pibar_row_max,
        )

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
            _clear_post_gradient_runtime_cache(static)

        return grad_theta, None, None

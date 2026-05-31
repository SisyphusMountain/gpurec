from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from gpurec.api._theta_constraints import (
    finite_theta_rate_bounds_log2,
    projected_theta_gradient_and_free,
    tensor_inf_norm,
)
from gpurec.api.model import GeneReconModel

from .diagnostics import parameter_stats, tensor_stats
from .config import RunConfig
from ._runtime_helpers import (
    _commit_pi_adjoint_pending_caches,
    _discard_pi_adjoint_pending_caches,
)

_FD_NEWTON_CURVATURE_EPS = 1e-12
_FD_NEWTON_EXTENDED_LINE_SEARCH_MAX_FAMILIES = 256
_FD_NEWTON_FALLBACK_LINE_SEARCH_MAX_STEPS = 2
_FD_NEWTON_LARGE_BATCH_MAX_LS = 8


@dataclass
class _FDNewtonRuntime:
    config: RunConfig
    active_batch_indices: Callable[[GeneReconModel], torch.Tensor]
    set_model_theta: Callable[[GeneReconModel, torch.Tensor], None]
    evaluate_active_genewise_vector_grad_at_current_theta: Callable[
        [GeneReconModel, str],
        tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
    ]
    evaluate_genewise_loss_vector_probe: Callable[[GeneReconModel, bool], torch.Tensor]
    projected_grad_inf: Callable[[GeneReconModel, float, float], tuple[torch.Tensor, float]]


def _set_model_theta(
    model: GeneReconModel,
    theta: torch.Tensor,
) -> None:
    with torch.no_grad():
        model.theta.copy_(theta)


def _fd_newton_runtime_for_runner(runner: Any) -> _FDNewtonRuntime:
    return _FDNewtonRuntime(
        config=runner.config,
        active_batch_indices=runner.evaluation._active_batch_indices,
        set_model_theta=_set_model_theta,
        evaluate_active_genewise_vector_grad_at_current_theta=(
            runner._evaluate_active_genewise_vector_grad_at_current_theta
        ),
        evaluate_genewise_loss_vector_probe=(
            runner._evaluate_genewise_loss_vector_probe
        ),
        projected_grad_inf=runner.evaluation.projected_grad_inf,
    )


@dataclass
class _FDNewtonHessianState:
    batch_index: int
    solver_stage: str
    family_indices: tuple[int, ...]
    hessian: torch.Tensor
    active_theta: torch.Tensor
    active_grad: torch.Tensor
    active_loss: torch.Tensor
    updates_since_refresh: int = 0


def active_fd_newton_step_for_runner(
    runner: Any,
    model: GeneReconModel,
    *,
    solver_stage: str,
    hessian_state: _FDNewtonHessianState | None = None,
    update_hessian_with_bfgs: bool = True,
    step_scale: float = 1.0,
    use_line_search: bool = True,
    reject_loss_increases_after_step: bool = False,
    hessian_refresh_steps: int | None = None,
    line_search_max_steps: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any], int, _FDNewtonHessianState]:
    return active_fd_newton_step(
        _fd_newton_runtime_for_runner(runner),
        model,
        solver_stage=solver_stage,
        hessian_state=hessian_state,
        update_hessian_with_bfgs=update_hessian_with_bfgs,
        step_scale=step_scale,
        use_line_search=use_line_search,
        reject_loss_increases_after_step=reject_loss_increases_after_step,
        hessian_refresh_steps=hessian_refresh_steps,
        line_search_max_steps=line_search_max_steps,
    )


def _fd_newton_state_matches(
    runtime: _FDNewtonRuntime,
    model: GeneReconModel,
    state: _FDNewtonHessianState | None,
    *,
    solver_stage: str,
) -> bool:
    if state is None:
        return False
    if state.batch_index != int(model.current_batch_index):
        return False
    if state.solver_stage != solver_stage:
        return False
    family_indices = tuple(int(index) for index in model.current_batch_metadata.family_indices)
    if state.family_indices != family_indices:
        return False
    idx = runtime.active_batch_indices(model)  # noqa: SLF001
    active_theta = model.theta.detach().index_select(0, idx)
    return torch.equal(active_theta, state.active_theta)


def _bfgs_update_fd_newton_hessian(
    *,
    state: _FDNewtonHessianState,
    active_theta: torch.Tensor,
    active_grad: torch.Tensor,
    active_loss: torch.Tensor,
    accepted: torch.Tensor,
    free_before: torch.Tensor,
    free_after: torch.Tensor,
) -> tuple[_FDNewtonHessianState, torch.Tensor]:
    old_hessian = state.hessian.detach()
    s = active_theta - state.active_theta
    y = active_grad - state.active_grad
    bs = torch.bmm(old_hessian, s.unsqueeze(-1)).squeeze(-1)
    sbs = (s * bs).sum(dim=1)
    ys = (y * s).sum(dim=1)
    finite = (
        torch.isfinite(s).all(dim=1)
        & torch.isfinite(y).all(dim=1)
        & torch.isfinite(bs).all(dim=1)
        & torch.isfinite(sbs)
        & torch.isfinite(ys)
    )
    active_set_same = (free_before == free_after).all(dim=1)
    moved = s.abs().amax(dim=1) > _FD_NEWTON_CURVATURE_EPS
    valid_update = (
        accepted
        & moved
        & finite
        & active_set_same
        & (ys > _FD_NEWTON_CURVATURE_EPS)
        & (sbs > _FD_NEWTON_CURVATURE_EPS)
    )
    safe_sbs = sbs.abs().clamp_min(_FD_NEWTON_CURVATURE_EPS)
    safe_ys = ys.abs().clamp_min(_FD_NEWTON_CURVATURE_EPS)
    bfgs_hessian = (
        old_hessian
        - torch.einsum("bi,bj->bij", bs, bs) / safe_sbs[:, None, None]
        + torch.einsum("bi,bj->bij", y, y) / safe_ys[:, None, None]
    )
    bfgs_hessian = 0.5 * (bfgs_hessian + bfgs_hessian.transpose(1, 2))
    hessian = torch.where(
        valid_update[:, None, None],
        bfgs_hessian,
        old_hessian,
    )
    new_state = _FDNewtonHessianState(
        batch_index=state.batch_index,
        solver_stage=state.solver_stage,
        family_indices=state.family_indices,
        hessian=hessian.detach().clone(),
        active_theta=active_theta.detach().clone(),
        active_grad=active_grad.detach().clone(),
        active_loss=active_loss.detach().clone(),
        updates_since_refresh=state.updates_since_refresh + 1,
    )
    return new_state, valid_update


def _refresh_fd_newton_hessian_state(
    runtime: _FDNewtonRuntime,
    model: GeneReconModel,
    *,
    solver_stage: str,
    baseline_state: _FDNewtonHessianState | None = None,
) -> tuple[_FDNewtonHessianState, dict[str, Any], int]:
    config = runtime.config
    idx = runtime.active_batch_indices(model)  # noqa: SLF001
    theta0 = model.theta.detach().clone()
    lower_bound, upper_bound = finite_theta_rate_bounds_log2(
        config.min_rate,
        config.max_rate,
    )
    eps = float(config.fd_hessian_epsilon)

    if baseline_state is None:
        loss0, grad0, metrics0 = (
            runtime.evaluate_active_genewise_vector_grad_at_current_theta(
                model,
                solver_stage=solver_stage,
            )
        )
        grad_evals = 1
        active_grad0 = grad0.index_select(0, idx)
        active_loss0 = loss0.index_select(0, idx)
    else:
        active_grad0 = baseline_state.active_grad.detach().clone()
        active_loss0 = baseline_state.active_loss.detach().clone()
        metrics0 = {
            "grad/inf": (
                float(active_grad0.detach().abs().amax().cpu())
                if active_grad0.numel()
                else 0.0
            ),
        }
        grad_evals = 0

    active_theta0 = theta0.index_select(0, idx)
    rows, cols = active_grad0.shape
    if cols != 3:
        raise RuntimeError(
            "Hessian-conditioned genewise optimization expects three D/T/L "
            "parameters per family; "
            f"got {cols}"
        )

    projected_grad, free = projected_theta_gradient_and_free(
        active_theta0,
        active_grad0,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    hessian = torch.zeros(
        (rows, cols, cols),
        device=model.theta.device,
        dtype=model.theta.dtype,
    )
    for col in range(cols):
        plus = theta0.clone()
        minus = theta0.clone()
        plus_active = torch.clamp(
            active_theta0[:, col] + eps,
            min=lower_bound,
            max=upper_bound,
        )
        minus_active = torch.clamp(
            active_theta0[:, col] - eps,
            min=lower_bound,
            max=upper_bound,
        )
        plus_rows = active_theta0.clone()
        minus_rows = active_theta0.clone()
        plus_rows[:, col] = plus_active
        minus_rows[:, col] = minus_active
        plus.index_copy_(0, idx, plus_rows)
        minus.index_copy_(0, idx, minus_rows)
        runtime.set_model_theta(model, plus)
        _plus_loss, plus_grad, _plus_metrics = (
            runtime.evaluate_active_genewise_vector_grad_at_current_theta(
                model,
                solver_stage=solver_stage,
            )
        )
        runtime.set_model_theta(model, minus)
        _minus_loss, minus_grad, _minus_metrics = (
            runtime.evaluate_active_genewise_vector_grad_at_current_theta(
                model,
                solver_stage=solver_stage,
            )
        )
        grad_evals += 2
        denom = (plus_active - minus_active).abs().clamp_min(1e-12)
        hessian[:, :, col] = (
            plus_grad.index_select(0, idx) - minus_grad.index_select(0, idx)
        ) / denom[:, None]

    runtime.set_model_theta(model, theta0)
    hessian = 0.5 * (hessian + hessian.transpose(1, 2))
    state = _FDNewtonHessianState(
        batch_index=int(model.current_batch_index),
        solver_stage=solver_stage,
        family_indices=tuple(
            int(index) for index in model.current_batch_metadata.family_indices
        ),
        hessian=hessian.detach().clone(),
        active_theta=active_theta0.detach().clone(),
        active_grad=active_grad0.detach().clone(),
        active_loss=active_loss0.detach().clone(),
        updates_since_refresh=0,
    )
    metrics0 = dict(metrics0)
    metrics0["grad/projected_inf"] = tensor_inf_norm(projected_grad)
    return state, metrics0, grad_evals


def active_fd_newton_step(
    runtime: _FDNewtonRuntime,
    model: GeneReconModel,
    *,
    solver_stage: str,
    hessian_state: _FDNewtonHessianState | None = None,
    update_hessian_with_bfgs: bool = True,
    step_scale: float = 1.0,
    use_line_search: bool = True,
    reject_loss_increases_after_step: bool = False,
    hessian_refresh_steps: int | None = None,
    line_search_max_steps: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any], int, _FDNewtonHessianState]:
    config = runtime.config
    hessian_refresh_steps = (
        config.fd_hessian_refresh_steps
        if hessian_refresh_steps is None
        else int(hessian_refresh_steps)
    )
    if hessian_refresh_steps < 1:
        raise ValueError("hessian_refresh_steps must be positive")
    idx = runtime.active_batch_indices(model)  # noqa: SLF001
    pi_adjoint_pending_discards = _discard_pi_adjoint_pending_caches(model)
    lower_bound, upper_bound = finite_theta_rate_bounds_log2(
        config.min_rate,
        config.max_rate,
    )
    damping = float(config.fd_newton_damping)
    grad_evals = 0
    loss_evals = 0
    hessian_state_matches = _fd_newton_state_matches(
        runtime,
        model,
        hessian_state,
        solver_stage=solver_stage,
    )
    refreshed_hessian = (
        not hessian_state_matches
        or hessian_state is None
        or hessian_state.updates_since_refresh >= hessian_refresh_steps
    )
    if refreshed_hessian:
        hessian_state, metrics0, refresh_grad_evals = _refresh_fd_newton_hessian_state(
            runtime,
            model,
            solver_stage=solver_stage,
            baseline_state=(hessian_state if hessian_state_matches else None),
        )
        grad_evals += refresh_grad_evals
    else:
        metrics0 = {}

    theta0 = model.theta.detach().clone()
    active_theta0 = hessian_state.active_theta.detach()
    active_grad0 = hessian_state.active_grad.detach()
    active_loss0 = hessian_state.active_loss.detach()
    hessian = hessian_state.hessian.detach()
    rows, cols = active_grad0.shape
    if cols != 3:
        raise RuntimeError(
            "Hessian-conditioned genewise optimization expects three D/T/L "
            "parameters per family; "
            f"got {cols}"
        )

    projected_grad, free = projected_theta_gradient_and_free(
        active_theta0,
        active_grad0,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    projected_grad_inf = tensor_inf_norm(projected_grad)
    row_active = free.any(dim=1)
    eye = torch.eye(cols, device=hessian.device, dtype=hessian.dtype).expand(rows, cols, cols)
    free_matrix = free[:, :, None] & free[:, None, :]
    hessian_solve = torch.where(free_matrix, hessian, torch.zeros_like(hessian))
    hessian_solve = hessian_solve + damping * eye
    diag = torch.diagonal(hessian_solve, dim1=1, dim2=2)
    diag.copy_(torch.where(free, diag, torch.ones_like(diag)))
    rhs = -torch.where(free, projected_grad, torch.zeros_like(projected_grad))
    solution, solve_info = torch.linalg.solve_ex(hessian_solve, rhs.unsqueeze(-1))
    step = solution.squeeze(-1)
    solve_ok = (solve_info == 0) & torch.isfinite(step).all(dim=1)
    descent = (projected_grad * step).sum(dim=1) < -1e-12
    fallback = row_active & (~solve_ok | ~descent)
    step = torch.where(fallback[:, None], -projected_grad, step)
    step = torch.where(row_active[:, None], step, torch.zeros_like(step))
    step = step * float(step_scale)
    raw_step_inf = float(step.detach().abs().amax().cpu()) if step.numel() else 0.0
    bounded_step = (
        torch.clamp(active_theta0 + step, min=lower_bound, max=upper_bound)
        - active_theta0
    )
    bounded_step_inf = (
        float(bounded_step.detach().abs().amax().cpu())
        if bounded_step.numel()
        else 0.0
    )
    gtd = (projected_grad * bounded_step).sum(dim=1)
    valid_projected_step = row_active & torch.isfinite(gtd) & (gtd < -1e-12)
    searching = valid_projected_step
    accepted = torch.zeros(rows, device=model.theta.device, dtype=torch.bool)
    accepted_active = active_theta0.clone()
    alpha = torch.ones(rows, device=model.theta.device, dtype=model.theta.dtype)
    line_search_fallback_attempted = torch.zeros_like(accepted)
    line_search_fallback_accepted = torch.zeros_like(accepted)
    max_line_search_steps = 0
    if use_line_search:
        max_line_search_steps = (
            int(config.lbfgs_max_ls)
            if line_search_max_steps is None
            else int(line_search_max_steps)
        )
        if max_line_search_steps < 1:
            raise ValueError("line_search_max_steps must be positive")
        if rows > _FD_NEWTON_EXTENDED_LINE_SEARCH_MAX_FAMILIES:
            max_line_search_steps = min(
                max_line_search_steps,
                _FD_NEWTON_LARGE_BATCH_MAX_LS,
            )

        for _ in range(max_line_search_steps):
            if not bool(searching.any()):
                break
            trial_active = torch.clamp(
                active_theta0 + alpha[:, None] * step,
                min=lower_bound,
                max=upper_bound,
            )
            candidate_active = torch.where(
                searching[:, None],
                trial_active,
                accepted_active,
            )
            candidate = theta0.clone()
            candidate.index_copy_(0, idx, candidate_active)
            runtime.set_model_theta(model, candidate)
            trial_loss_vec = runtime.evaluate_genewise_loss_vector_probe(model, active_batch=True)
            loss_evals += 1
            trial_active_loss = trial_loss_vec.index_select(0, idx)
            trial_delta = trial_active - active_theta0
            trial_gtd = (projected_grad * trial_delta).sum(dim=1)
            trial_searching = searching & torch.isfinite(trial_gtd) & (
                trial_gtd < -1e-12
            )
            armijo_rhs = active_loss0 + 1e-4 * trial_gtd
            ok = trial_searching & torch.isfinite(trial_active_loss) & (
                trial_active_loss <= armijo_rhs
            )
            if bool(ok.any()):
                accepted = accepted | ok
                accepted_active = torch.where(
                    ok[:, None],
                    trial_active,
                    accepted_active,
                )
            searching = trial_searching & ~accepted
            alpha = torch.where(searching, alpha * 0.5, alpha)

        fallback_searching = row_active & ~accepted & torch.isfinite(
            projected_grad,
        ).all(dim=1)
        fallback_step = -torch.where(
            free,
            projected_grad,
            torch.zeros_like(projected_grad),
        ) * float(step_scale)
        fallback_bounded_step = (
            torch.clamp(
                active_theta0 + fallback_step,
                min=lower_bound,
                max=upper_bound,
            )
            - active_theta0
        )
        fallback_gtd = (projected_grad * fallback_bounded_step).sum(dim=1)
        fallback_searching = fallback_searching & torch.isfinite(
            fallback_gtd,
        ) & (fallback_gtd < -1e-12)
        line_search_fallback_attempted = fallback_searching.clone()
        fallback_alpha = torch.ones(
            rows,
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        fallback_line_search_steps = min(
            max_line_search_steps,
            _FD_NEWTON_FALLBACK_LINE_SEARCH_MAX_STEPS,
        )
        for _ in range(fallback_line_search_steps):
            if not bool(fallback_searching.any()):
                break
            trial_active = torch.clamp(
                active_theta0 + fallback_alpha[:, None] * fallback_step,
                min=lower_bound,
                max=upper_bound,
            )
            candidate_active = torch.where(
                fallback_searching[:, None],
                trial_active,
                accepted_active,
            )
            candidate = theta0.clone()
            candidate.index_copy_(0, idx, candidate_active)
            runtime.set_model_theta(model, candidate)
            trial_loss_vec = runtime.evaluate_genewise_loss_vector_probe(model, active_batch=True)
            loss_evals += 1
            trial_active_loss = trial_loss_vec.index_select(0, idx)
            trial_delta = trial_active - active_theta0
            trial_gtd = (projected_grad * trial_delta).sum(dim=1)
            trial_searching = fallback_searching & torch.isfinite(trial_gtd) & (
                trial_gtd < -1e-12
            )
            armijo_rhs = active_loss0 + 1e-4 * trial_gtd
            ok = trial_searching & torch.isfinite(trial_active_loss) & (
                trial_active_loss <= armijo_rhs
            )
            if bool(ok.any()):
                accepted = accepted | ok
                line_search_fallback_accepted = line_search_fallback_accepted | ok
                accepted_active = torch.where(
                    ok[:, None],
                    trial_active,
                    accepted_active,
                )
            fallback_searching = trial_searching & ~accepted
            fallback_alpha = torch.where(
                fallback_searching,
                fallback_alpha * 0.5,
                fallback_alpha,
            )
    else:
        trial_active = active_theta0 + bounded_step
        accepted = valid_projected_step
        accepted_active = torch.where(
            accepted[:, None],
            trial_active,
            accepted_active,
        )

    final_theta = theta0.clone()
    final_theta.index_copy_(0, idx, accepted_active)
    runtime.set_model_theta(model, final_theta)
    loss_vec, _grad, metrics = (
        runtime.evaluate_active_genewise_vector_grad_at_current_theta(model, solver_stage=solver_stage)
    )
    grad_evals += 1
    active_theta1 = final_theta.index_select(0, idx).detach()
    active_loss1 = loss_vec.detach().index_select(0, idx)
    active_grad1 = model.theta.grad.detach().index_select(0, idx)
    loss_rejected = torch.zeros_like(accepted)
    if reject_loss_increases_after_step:
        finite_loss = torch.isfinite(active_loss1)
        accepted_after_loss = accepted & finite_loss & (active_loss1 <= active_loss0)
        loss_rejected = accepted & ~accepted_after_loss
        if bool(loss_rejected.any().detach().cpu()):
            accepted = accepted_after_loss
            active_theta1 = torch.where(
                accepted[:, None],
                active_theta1,
                active_theta0,
            )
            active_loss1 = torch.where(accepted, active_loss1, active_loss0)
            active_grad1 = torch.where(
                accepted[:, None],
                active_grad1,
                active_grad0,
            )
            final_theta = final_theta.clone()
            final_theta.index_copy_(0, idx, active_theta1)
            runtime.set_model_theta(model, final_theta)
            loss_vec = loss_vec.detach().clone()
            loss_vec.index_copy_(0, idx, active_loss1)
            grad = model.theta.grad.detach().clone()
            grad.index_copy_(0, idx, active_grad1)
            model.theta.grad = grad
            corrected_loss = loss_vec.sum()
            metrics = dict(metrics)
            metrics["likelihood/data_nll_bits"] = float(
                corrected_loss.detach().cpu()
            )
            metrics["likelihood/log_likelihood_bits"] = float(
                -corrected_loss.detach().cpu()
            )
            metrics.update(tensor_stats("grad", model.theta.grad))
            metrics.update(parameter_stats(model.theta))

    lower_bound, upper_bound = finite_theta_rate_bounds_log2(
        config.min_rate,
        config.max_rate,
    )
    final_projected_grad, final_projected_grad_inf = runtime.projected_grad_inf(
        model,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    active_projected_grad1 = final_projected_grad.index_select(0, idx)
    free_after = active_projected_grad1.abs() > 0
    if update_hessian_with_bfgs:
        next_state, bfgs_updated = _bfgs_update_fd_newton_hessian(
            state=hessian_state,
            active_theta=active_theta1,
            active_grad=active_grad1,
            active_loss=active_loss1,
            accepted=accepted,
            free_before=free,
            free_after=free_after,
        )
        hessian_update = "bfgs"
    else:
        next_state = _FDNewtonHessianState(
            batch_index=hessian_state.batch_index,
            solver_stage=hessian_state.solver_stage,
            family_indices=hessian_state.family_indices,
            hessian=hessian_state.hessian.detach().clone(),
            active_theta=active_theta1.detach().clone(),
            active_grad=active_grad1.detach().clone(),
            active_loss=active_loss1.detach().clone(),
            updates_since_refresh=hessian_state.updates_since_refresh + 1,
        )
        bfgs_updated = torch.zeros_like(accepted)
        hessian_update = "fixed"
    bfgs_skipped = accepted & ~bfgs_updated
    if refreshed_hessian:
        refresh_grad_inf = metrics0.get("grad/inf")
        refresh_projected_inf = metrics0.get("grad/projected_inf")
        if refresh_grad_inf is not None:
            metrics["optimizer/fd_newton_refresh_grad_inf"] = refresh_grad_inf
        if refresh_projected_inf is not None:
            metrics["optimizer/fd_newton_refresh_projected_inf"] = (
                refresh_projected_inf
            )
    metrics["grad/projected_inf"] = final_projected_grad_inf
    metrics["optimizer/fd_newton_grad_evals"] = float(grad_evals)
    metrics["optimizer/fd_newton_loss_evals"] = float(loss_evals)
    metrics["optimizer/fd_newton_line_search"] = bool(use_line_search)
    metrics["optimizer/fd_newton_post_step_loss_filter"] = bool(
        reject_loss_increases_after_step
    )
    metrics["optimizer/fd_newton_loss_rejected_rows"] = float(
        loss_rejected.sum().detach().cpu()
    )
    metrics["optimizer/fd_newton_max_ls"] = float(max_line_search_steps)
    metrics["optimizer/fd_newton_fallback_max_ls"] = float(
        min(max_line_search_steps, _FD_NEWTON_FALLBACK_LINE_SEARCH_MAX_STEPS)
        if use_line_search
        else 0
    )
    metrics["optimizer/fd_newton_accepted_rows"] = float(accepted.sum().detach().cpu())
    metrics["optimizer/fd_newton_accepted_fraction"] = float(
        accepted.to(dtype=torch.float32).mean().detach().cpu()
    )
    metrics["optimizer/fd_newton_fallback_rows"] = float(fallback.sum().detach().cpu())
    metrics["optimizer/fd_newton_line_search_fallback_attempted_rows"] = float(
        line_search_fallback_attempted.sum().detach().cpu()
    )
    metrics["optimizer/fd_newton_line_search_fallback_rows"] = float(
        line_search_fallback_accepted.sum().detach().cpu()
    )
    metrics["optimizer/fd_newton_hessian_source"] = (
        "finite_difference"
        if refreshed_hessian
        else ("bfgs_update" if update_hessian_with_bfgs else "fixed_hessian")
    )
    metrics["optimizer/fd_newton_hessian_update"] = hessian_update
    metrics["optimizer/fd_newton_hessian_refreshed"] = bool(refreshed_hessian)
    metrics["optimizer/fd_newton_hessian_updates_since_refresh"] = float(
        next_state.updates_since_refresh
    )
    metrics["optimizer/fd_newton_hessian_refresh_steps"] = float(
        hessian_refresh_steps
    )
    metrics["optimizer/fd_newton_bfgs_updated_rows"] = float(
        bfgs_updated.sum().detach().cpu()
    )
    metrics["optimizer/fd_newton_bfgs_skipped_rows"] = float(
        bfgs_skipped.sum().detach().cpu()
    )
    metrics["optimizer/fd_newton_baseline_projected_inf"] = projected_grad_inf
    metrics["optimizer/fd_newton_step_scale"] = float(step_scale)
    metrics["optimizer/fd_newton_raw_step_inf"] = raw_step_inf
    metrics["optimizer/fd_newton_bound_projected_step_inf"] = bounded_step_inf
    if bool(loss_rejected.any().detach().cpu()):
        pi_adjoint_pending_commits = 0
        pi_adjoint_pending_discards += _discard_pi_adjoint_pending_caches(model)
    else:
        pi_adjoint_pending_commits = _commit_pi_adjoint_pending_caches(model)
    metrics["solver/pi_adjoint_pending_cache_commits"] = float(
        pi_adjoint_pending_commits
    )
    metrics["solver/pi_adjoint_pending_cache_discards"] = float(
        pi_adjoint_pending_discards
    )
    return loss_vec, metrics, grad_evals + loss_evals, next_state

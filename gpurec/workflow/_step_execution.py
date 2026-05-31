from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from gpurec.api._theta_constraints import finite_theta_rate_bounds_log2
from gpurec.api.model import GeneReconModel
from gpurec.api.autograd import _clear_pi_adjoint_runtime_cache

from ._fd_newton import _FDNewtonHessianState
from ._runtime_helpers import (
    _commit_pi_adjoint_pending_caches,
    _discard_pi_adjoint_pending_caches,
    _is_finite_tensor,
)
from ._phase import _ActiveAdagradRestartPhase
from ._evaluation import EvaluationOps
from ._step_metrics import (
    _adagrad_restart_step_metrics,
    _batched_lbfgs_step_metrics,
    _cached_genewise_loss_metrics,
    _cached_scalar_loss_metrics,
    _hessian_sgd_budget_metrics,
    _projected_grad_inf_from_optimizer_state,
    _projected_optimizer_step_metrics,
)
from ._solver_stage import SolverStageController
from .config import RunConfig


class _NonfiniteParameterUpdate(RuntimeError):
    """Internal sentinel for optimizer updates that corrupt theta."""


def _set_model_theta(model: GeneReconModel, theta: torch.Tensor) -> None:
    with torch.no_grad():
        model.theta.copy_(theta)


def _restore_theta_if_nonfinite_update(
    model: GeneReconModel,
    theta_before: torch.Tensor,
) -> bool:
    if _is_finite_tensor(model.theta):
        return False
    _set_model_theta(model, theta_before)
    model.theta.grad = None
    return True


def _active_adam_step(
    model: GeneReconModel,
    optimizer: torch.optim.Optimizer,
    *,
    evaluation: EvaluationOps,
    config: Any,
    solver_stage: str,
    theta_before: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any], int]:
    pi_adjoint_pending_discards = _discard_pi_adjoint_pending_caches(model)
    optimizer.zero_grad(set_to_none=True)
    _ = evaluation.evaluate_active_genewise_vector_grad_at_current_theta(
        model,
        solver_stage=solver_stage,
    )
    optimizer.step()
    with torch.no_grad():
        model.clamp_theta_(config.min_rate, config.max_rate)
    if _restore_theta_if_nonfinite_update(model, theta_before):
        _discard_pi_adjoint_pending_caches(model)
        raise _NonfiniteParameterUpdate()
    loss_vec, _grad, metrics = (
        evaluation.evaluate_active_genewise_vector_grad_at_current_theta(
            model,
            solver_stage=solver_stage,
        )
    )
    lower_bound, upper_bound = finite_theta_rate_bounds_log2(
        config.min_rate,
        config.max_rate,
    )
    _, projected_grad_inf = evaluation.projected_grad_inf(
        model,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    metrics["grad/projected_inf"] = projected_grad_inf
    metrics["solver/pi_adjoint_pending_cache_commits"] = float(
        _commit_pi_adjoint_pending_caches(model)
    )
    metrics["solver/pi_adjoint_pending_cache_discards"] = float(
        pi_adjoint_pending_discards
    )
    return loss_vec, metrics, 2


def _clear_solver_runtime_state_preserving_pi_cache(model: GeneReconModel) -> None:
    statics = getattr(model, "cached_static_states", None)
    if statics is None:
        model.clear()
        return
    for static in list(statics):
        if hasattr(static, "warm_E"):
            static.warm_E = None
        _clear_pi_adjoint_runtime_cache(static)
        if hasattr(static, "last_solver_stats"):
            static.last_solver_stats = None


@dataclass
class _StepExecutionResult:
    status: dict[str, str] | None
    metrics: dict[str, Any]
    closure_evals: int
    theta_step: float
    loss_vec_current: torch.Tensor | None
    first_order_pending_step: bool
    next_fd_newton_hessian_state: _FDNewtonHessianState | None
    active_batch_local_step: int
    hessian_sgd_validation_step: bool
    cacheable_active_batch_final_result: bool


@dataclass
class _StepExecutionContext:
    config: RunConfig
    evaluation: EvaluationOps
    solver: SolverStageController
    batchwise_active_optimizer: bool
    fd_adam_warmup_steps: int
    hessian_sgd_no_line_refresh_min_clades: int
    hessian_sgd_no_line_refresh_steps: int
    hessian_sgd_line_search_max_steps: int
    active_fd_newton_step: Callable[
        ...,
        tuple[torch.Tensor, dict[str, Any], int, _FDNewtonHessianState],
    ]


@dataclass
class _StepExecutionState:
    active_solver_stage: str
    active_batch_local_step: int = 0
    fd_newton_hessian_state: _FDNewtonHessianState | None = None
    hessian_sgd_line_search_active: bool = False


def execute_optimization_step(
    context: _StepExecutionContext,
    state: _StepExecutionState,
    model: GeneReconModel,
    optimizer: torch.optim.Optimizer,
    *,
    phase: str,
    step: int,
    adagrad_restart_active_phase: _ActiveAdagradRestartPhase | None,
) -> _StepExecutionResult:
    evaluation = context.evaluation
    solver = context.solver
    config = context.config
    batchwise_active_optimizer = context.batchwise_active_optimizer
    fd_adam_warmup_steps = context.fd_adam_warmup_steps
    fd_newton_hessian_state = state.fd_newton_hessian_state
    hessian_sgd_line_search_active = state.hessian_sgd_line_search_active
    hessian_sgd_no_line_refresh_min_clades = (
        context.hessian_sgd_no_line_refresh_min_clades
    )
    hessian_sgd_no_line_refresh_steps = (
        context.hessian_sgd_no_line_refresh_steps
    )
    hessian_sgd_line_search_max_steps = (
        context.hessian_sgd_line_search_max_steps
    )
    active_solver_stage = state.active_solver_stage
    active_batch_local_step = state.active_batch_local_step

    status: dict[str, str] | None = None
    closure_evals = 0
    batched_grad_evals = 0
    batched_loss_evals = 0
    projected_loss_evals = 0
    metrics: dict[str, Any] = {}
    loss_vec_current: torch.Tensor | None = None
    first_order_pending_step = False
    next_fd_newton_hessian_state = fd_newton_hessian_state
    hessian_sgd_validation_step = False
    active_batch_local_step_next = active_batch_local_step
    theta_step = 0.0
    cacheable_active_batch_final_result = False
    theta_before = model.theta.detach().clone()
    lower_bound, upper_bound = finite_theta_rate_bounds_log2(
        config.min_rate,
        config.max_rate,
    )

    def closure() -> torch.Tensor:
        nonlocal closure_evals, metrics
        with torch.no_grad():
            model.clamp_theta_(config.min_rate, config.max_rate)
        if optimizer is None:
            raise RuntimeError("missing optimizer")
        optimizer.zero_grad(set_to_none=True)
        loss_i, metrics_i = evaluation.evaluate_and_backward(model)
        metrics = metrics_i
        closure_evals += 1
        return loss_i

    def batched_closure() -> torch.Tensor:
        nonlocal batched_grad_evals, metrics
        batched_grad_evals += 1
        with torch.no_grad():
            model.clamp_theta_(config.min_rate, config.max_rate)
        if optimizer is None:
            raise RuntimeError("missing optimizer")
        optimizer.zero_grad(set_to_none=True)
        if batchwise_active_optimizer:
                loss_vec_i, metrics_i = evaluation.evaluate_active_genewise_vector_and_grad(
                    model,
                    solver_stage=active_solver_stage,
                )
        else:
            loss_vec_i, metrics_i = evaluation.evaluate_genewise_vector_and_grad(model)
        metrics = metrics_i
        return loss_vec_i

    def batched_loss_closure() -> torch.Tensor:
        nonlocal batched_loss_evals
        batched_loss_evals += 1
        with torch.no_grad():
            model.clamp_theta_(config.min_rate, config.max_rate)
        if batchwise_active_optimizer:
            return evaluation.evaluate_genewise_loss_vector_probe(
                model,
                active_batch=True,
            )
        return evaluation.evaluate_genewise_loss_vector_probe(
            model,
            active_batch=False,
        )

    def projected_loss_closure() -> torch.Tensor:
        nonlocal projected_loss_evals
        projected_loss_evals += 1
        with torch.no_grad():
            model.clamp_theta_(config.min_rate, config.max_rate)
        return evaluation.evaluate_loss_only_probe(model)

    def fail(step_status: dict[str, str]) -> _StepExecutionResult:
        return _StepExecutionResult(
            status=step_status,
            metrics=metrics,
            closure_evals=closure_evals,
            theta_step=theta_step,
            loss_vec_current=loss_vec_current,
            first_order_pending_step=first_order_pending_step,
            next_fd_newton_hessian_state=next_fd_newton_hessian_state,
            active_batch_local_step=active_batch_local_step_next,
            hessian_sgd_validation_step=hessian_sgd_validation_step,
            cacheable_active_batch_final_result=False,
        )

    if phase == "lbfgs":
        try:
            optimizer.step(closure)
        except RuntimeError:
            status = {"status": "failed", "reason": "lbfgs_runtime_error"}
        else:
            with torch.no_grad():
                model.clamp_theta_(config.min_rate, config.max_rate)
            if _restore_theta_if_nonfinite_update(model, theta_before):
                status = {
                    "status": "failed",
                    "reason": "nonfinite_parameter_update",
                }
                model.clear()
            else:
                theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
                model.clear()
                model.theta.grad = None
                loss_current, metrics = evaluation.evaluate_and_backward(model)
                closure_evals += 1
                if (
                    not torch.isfinite(loss_current).item()
                    or not _is_finite_tensor(model.theta.grad)
):
                    model.clear()
                    status = {
                        "status": "failed",
                        "reason": "nonfinite_objective_or_gradient",
                    }
                else:
                    model.clear()

    elif phase in {"projected-lbfgs", "lbfgsb"}:
        metric_prefix = (
            "projected_lbfgs" if phase == "projected-lbfgs" else "lbfgsb"
        )
        try:
            optimizer.step(
                closure,
                loss_closure=projected_loss_closure,
            )
        except RuntimeError:
            status = {
                "status": "failed",
                "reason": f"{metric_prefix}_runtime_error",
            }
        else:
            opt_state = optimizer.state.get(model.theta, {})
            closure_evals = int(opt_state.get("last_grad_evals", closure_evals)) + int(
                opt_state.get("last_loss_evals", projected_loss_evals)
            )
            with torch.no_grad():
                model.clamp_theta_(config.min_rate, config.max_rate)
            if _restore_theta_if_nonfinite_update(model, theta_before):
                status = {
                    "status": "failed",
                    "reason": "nonfinite_parameter_update",
                }
                model.clear()
            else:
                theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
                grad_current = opt_state.get("last_grad")
                loss_current = opt_state.get("last_loss")
                if torch.is_tensor(grad_current):
                    model.theta.grad = grad_current.detach().reshape_as(model.theta).to(
                        device=model.theta.device,
                        dtype=model.theta.dtype,
                    )
                if torch.is_tensor(loss_current):
                    loss_current = loss_current.detach().to(
                        device=model.theta.device,
                        dtype=model.theta.dtype,
                    ).reshape(())
                    metrics = _cached_scalar_loss_metrics(
                        metrics,
                        model,
                        loss_current,
                    )
                else:
                    model.theta.grad = None
                    loss_current, metrics = evaluation.evaluate_and_backward(model)
                    closure_evals += 1

                projected_inf = _projected_grad_inf_from_optimizer_state(
                    opt_state=opt_state,
                    evaluation=evaluation,
                    model=model,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                metrics.update(
                    _projected_optimizer_step_metrics(
                        metric_prefix=metric_prefix,
                        opt_state=opt_state,
                        projected_loss_evals=projected_loss_evals,
                        theta_step=theta_step,
                        projected_grad_inf=projected_inf,
                    )
                )
                if (
                    (torch.is_tensor(loss_current) and not torch.isfinite(loss_current).item())
                    or not _is_finite_tensor(model.theta.grad)
                ):
                    model.clear()
                    status = {
                        "status": "failed",
                        "reason": "nonfinite_objective_or_gradient",
                    }
                else:
                    model.clear()

    elif phase == "batched-lbfgs":
        try:
            optimizer.step(
                batched_closure,
                loss_closure=batched_loss_closure,
            )
        except RuntimeError:
            status = {
                "status": "failed",
                "reason": "batched_lbfgs_runtime_error",
            }
        else:
            opt_state = optimizer.state.get(model.theta, {})
            closure_evals = batched_grad_evals + batched_loss_evals
            with torch.no_grad():
                model.clamp_theta_(config.min_rate, config.max_rate)
            if _restore_theta_if_nonfinite_update(model, theta_before):
                status = {
                    "status": "failed",
                    "reason": "nonfinite_parameter_update",
                }
                model.clear()
            else:
                theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
                model.clear()
                reused_optimizer_gradient = False
                loss_vec_current = opt_state.get("last_loss")
                grad_current = opt_state.get("last_grad")
                if (
                    config.lbfgs_line_search == "none"
                    and torch.is_tensor(loss_vec_current)
                    and torch.is_tensor(grad_current)
                    and loss_vec_current.numel() == int(model.n_families)
                    and grad_current.numel() == model.theta.numel()
                ):
                    model.theta.grad = grad_current.detach().reshape_as(model.theta).to(
                        device=model.theta.device,
                        dtype=model.theta.dtype,
                    )
                    loss_vec_current = loss_vec_current.detach().to(
                        device=model.theta.device,
                        dtype=model.theta.dtype,
                    ).reshape(int(model.n_families))
                    metrics = _cached_genewise_loss_metrics(
                        metrics,
                        model,
                        loss_vec_current,
                    )
                    reused_optimizer_gradient = True
                else:
                    model.theta.grad = None
                    if batchwise_active_optimizer:
                        loss_vec_current, metrics = (
                            evaluation.evaluate_active_genewise_vector_and_grad(
                                model,
                                solver_stage=active_solver_stage,
                            )
                        )
                    else:
                        loss_vec_current, metrics = (
                            evaluation.evaluate_genewise_vector_and_grad(model)
                        )
                    closure_evals += 1
                metrics.update(
                    _batched_lbfgs_step_metrics(
                        opt_state=opt_state,
                        batched_grad_evals=batched_grad_evals,
                        batched_loss_evals=batched_loss_evals,
                        reused_optimizer_gradient=reused_optimizer_gradient,
                    )
                )
                if (
                    not bool(torch.isfinite(loss_vec_current).all().item())
                    or not _is_finite_tensor(model.theta.grad)
                ):
                    model.clear()
                    status = {
                        "status": "failed",
                        "reason": "nonfinite_objective_or_gradient",
                    }
                else:
                    model.clear()

    elif phase in {"adam-fd-newton", "hessian-sgd"}:
        if phase == "hessian-sgd":
            hessian_sgd_validation_step = solver.is_hessian_sgd_validation_step(
                phase=phase,
                solver_stage=active_solver_stage,
                active_batch_local_step=active_batch_local_step,
                line_search_active=hessian_sgd_line_search_active,
            )
            if hessian_sgd_validation_step:
                solver.configure_hessian_sgd_validation_solver_stage(model)
            elif (
                active_solver_stage == "full"
                and config.hessian_sgd_validation_interval > 0
            ):
                solver.configure_active_stage(model, active_solver_stage)
        if phase == "adam-fd-newton" and active_batch_local_step < fd_adam_warmup_steps:
            next_fd_newton_hessian_state = None
            try:
                loss_vec_current, metrics, closure_evals = _active_adam_step(
                    model,
                    optimizer,
                    evaluation=evaluation,
                    config=config,
                    solver_stage=active_solver_stage,
                    theta_before=theta_before,
                )
            except Exception as exc:
                if isinstance(exc, _NonfiniteParameterUpdate):
                    status = {
                        "status": "failed",
                        "reason": "nonfinite_parameter_update",
                    }
                    model.clear()
                else:
                    raise
            else:
                metrics["optimizer/fd_newton_subphase"] = "adam_warmup"
        else:
            hessian_refresh_steps = int(config.fd_hessian_refresh_steps)
            active_clade_count = int(
                getattr(
                    model.current_batch_metadata,
                    "clade_count",
                    0,
                )
                or 0
            )
            if (
                phase == "hessian-sgd"
                and not hessian_sgd_line_search_active
                and active_clade_count >= hessian_sgd_no_line_refresh_min_clades
            ):
                hessian_refresh_steps = max(
                    hessian_refresh_steps,
                    hessian_sgd_no_line_refresh_steps,
                )
            loss_vec_current, metrics, closure_evals, next_fd_state = (
                context.active_fd_newton_step(
                    model,
                    solver_stage=active_solver_stage,
                    hessian_state=(
                        None
                        if hessian_sgd_validation_step
                        else fd_newton_hessian_state
                    ),
                    update_hessian_with_bfgs=phase in {"adam-fd-newton", "hessian-sgd"},
                    step_scale=(1.0 if phase == "adam-fd-newton" else config.lr),
                    use_line_search=(
                        phase == "adam-fd-newton"
                        or (phase == "hessian-sgd" and hessian_sgd_line_search_active)
                    ),
                    reject_loss_increases_after_step=(
                        phase == "hessian-sgd"
                        and not hessian_sgd_line_search_active
                    ),
                    hessian_refresh_steps=hessian_refresh_steps,
                    line_search_max_steps=(
                        hessian_sgd_line_search_max_steps
                        if (phase == "hessian-sgd" and hessian_sgd_line_search_active)
                        else None
                    ),
                )
            )
            next_fd_newton_hessian_state = next_fd_state
            if hessian_sgd_validation_step:
                metrics["optimizer/fd_newton_subphase"] = "hessian_sgd_validation"
            else:
                metrics["optimizer/fd_newton_subphase"] = (
                    "fd_newton" if phase == "adam-fd-newton" else "hessian_sgd"
                )
            if phase == "hessian-sgd" and config.hessian_sgd_validation_interval > 0:
                metrics.update(
                    _hessian_sgd_budget_metrics(
                        config=config,
                        solver=solver,
                        model=model,
                        active_solver_stage=active_solver_stage,
                        hessian_sgd_validation_step=hessian_sgd_validation_step,
                    )
                )

            if _restore_theta_if_nonfinite_update(model, theta_before):
                status = {
                    "status": "failed",
                    "reason": "nonfinite_parameter_update",
                }
                model.clear()
            else:
                theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
                if (
                    not bool(torch.isfinite(loss_vec_current).all().item())
                    or not _is_finite_tensor(model.theta.grad)
                ):
                    model.clear()
                    status = {
                        "status": "failed",
                        "reason": "nonfinite_objective_or_gradient",
                    }
                else:
                    if batchwise_active_optimizer:
                        cacheable_active_batch_final_result = (
                            solver.hessian_sgd_validation_result_is_canonical_full_solver()
                            if hessian_sgd_validation_step
                            else solver.active_batch_result_is_canonical_full_solver(
                                phase=phase,
                                solver_stage=active_solver_stage,
                            )
                        )
                    active_batch_local_step_next = active_batch_local_step + 1
                    _clear_solver_runtime_state_preserving_pi_cache(model)

    else:
        loss = closure()
        if not torch.isfinite(loss).item() or not _is_finite_tensor(model.theta.grad):
            status = {
                "status": "failed",
                "reason": "nonfinite_objective_or_gradient",
            }
        else:
            if phase == "projected-sgd" or (
                adagrad_restart_active_phase is not None and phase.startswith("adagrad-restarts:")
            ):
                _, projected_grad_inf = evaluation.projected_grad_inf(
                    model,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                metrics["grad/projected_inf"] = projected_grad_inf
            if phase.startswith("adagrad-restarts:"):
                if adagrad_restart_active_phase is None:
                    raise RuntimeError("missing adagrad-restarts active phase")
                metrics.update(
                    _adagrad_restart_step_metrics(
                        active_phase=adagrad_restart_active_phase,
                        step=step,
                    )
                )
            theta_step = 0.0
            first_order_pending_step = True

    if status is not None:
        return _StepExecutionResult(
            status=status,
            metrics=metrics,
            closure_evals=closure_evals,
            theta_step=theta_step,
            loss_vec_current=loss_vec_current,
            first_order_pending_step=False,
            next_fd_newton_hessian_state=next_fd_newton_hessian_state,
            active_batch_local_step=active_batch_local_step_next,
            hessian_sgd_validation_step=hessian_sgd_validation_step,
            cacheable_active_batch_final_result=False,
        )

    return _StepExecutionResult(
        status=None,
        metrics=metrics,
        closure_evals=closure_evals,
        theta_step=theta_step,
        loss_vec_current=loss_vec_current,
        first_order_pending_step=first_order_pending_step,
        next_fd_newton_hessian_state=next_fd_newton_hessian_state,
        active_batch_local_step=active_batch_local_step_next,
        hessian_sgd_validation_step=hessian_sgd_validation_step,
        cacheable_active_batch_final_result=cacheable_active_batch_final_result,
    )

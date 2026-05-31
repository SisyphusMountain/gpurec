from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from gpurec.api.model import GeneReconModel

from ._artifacts import (
    _build_run_manifest,
    _final_check_summary_metrics,
    _final_solver_summary_metrics,
    _write_final_artifacts,
)
from ._result import OptimizationResult, optimization_result_from_summary
from .config import LossStopPhase, RunConfig, effective_route_metadata
from ._evaluation import (
    EvaluationOps,
    _is_memory_retryable_runtime_error,
)
from ._runtime_helpers import (
    _clear_cached_solver_runtime_state,
    _clear_cuda_allocator_cache_if_needed,
    _drop_cached_static_states_if_needed,
    _is_finite_tensor,
    _is_single_value_tensor,
    _tensor_shape,
)
from .model_factory import build_alerax_workflow_model
from .diagnostics import parameter_stats, solver_stats, tensor_stats
from ._solver_stage import SolverStageController


@dataclass(frozen=True)
class _FinalizationInputs:
    model: GeneReconModel
    optimizer: torch.optim.Optimizer | None
    history: list[dict[str, Any]]
    history_jsonl: Path
    best_checkpoint: Path
    latest_checkpoint: Path
    status: dict[str, Any]
    resume_info: dict[str, Any]
    final_row: dict[str, Any]
    start_step: int
    stable_loss_steps: int
    best_nll: float | None
    best_step: int | None
    previous_objective: float | None
    lbfgsb_fallback_used_count: int
    lbfgsb_best_retry_count: int
    lbfgsb_loss_schedule: tuple[LossStopPhase, ...]
    lbfgsb_loss_schedule_index: int
    batchwise_active_optimizer: bool
    batch_final_loss_cache: torch.Tensor | None
    batch_final_grad_cache: torch.Tensor | None
    batch_final_cache_ready: torch.Tensor | None
    runtime_seed_context: dict[str, Any]
    started: float
    current_phase: str
    command_argv: tuple[str, ...] | list[str] | None


def finalize_optimization(
    config: RunConfig,
    inputs: _FinalizationInputs,
    *,
    evaluation: EvaluationOps,
    solver: SolverStageController,
    save_status: Callable[[Path, Any], None],
    adaptive_checkpoint_status: Any | None = None,
) -> OptimizationResult:
    if adaptive_checkpoint_status is None:
        adaptive_checkpoint_status = lambda payload: payload

    model = inputs.model
    optimizer = inputs.optimizer
    final_row = dict(inputs.final_row)
    status = dict(inputs.status)
    resume_info = dict(inputs.resume_info)

    final_eval_started = time.perf_counter()
    final_eval_at_check_iters = solver.configure_specieswise_final_eval_solver_stage(model)
    if not final_eval_at_check_iters and inputs.batchwise_active_optimizer:
        solver.configure_stage(model, "full")

    model.theta.grad = None
    final_per_family_nll: torch.Tensor | None = None
    final_closure_evals = 1

    if (
        inputs.batchwise_active_optimizer
        and inputs.batch_final_loss_cache is not None
        and inputs.batch_final_grad_cache is not None
        and inputs.batch_final_cache_ready is not None
        and bool(inputs.batch_final_cache_ready.all().item())
    ):
        final_per_family_nll = inputs.batch_final_loss_cache.detach().clone()
        model.theta.grad = inputs.batch_final_grad_cache.detach().clone()
        final_loss = final_per_family_nll.sum()
        final_metrics = {
            "likelihood/data_nll_bits": float(final_loss.detach().cpu()),
            "likelihood/log_likelihood_bits": float(-final_loss.detach().cpu()),
            "optimizer/final_eval_source": "cached_active_batches",
        }
        final_metrics.update(tensor_stats("grad", model.theta.grad))
        final_metrics.update(parameter_stats(model.theta))
        final_metrics.update(solver_stats(model))
        final_closure_evals = 0
    elif config.mode == "genewise" and callable(
        getattr(model, "full_genewise_nll_and_grad", None)
    ):
        final_loss_vec, final_metrics = (
            evaluation.evaluate_genewise_vector_and_grad_with_memory_fallback(model)
        )
        final_loss = final_loss_vec.sum()
        final_per_family_nll = final_loss_vec.detach()
    else:
        final_loss, final_metrics = evaluation.evaluate_and_backward(model)

    final_step = max(
        inputs.start_step,
        min(config.steps, int(final_row.get("step", -1)) + 1),
    )
    final_eval_failed = (
        not torch.isfinite(final_loss).item()
        or not torch.is_tensor(model.theta.grad)
        or not torch.isfinite(model.theta.grad).all().item()
    )

    best_nll = inputs.best_nll
    best_step = inputs.best_step

    if final_eval_failed:
        final_eval_s = time.perf_counter() - final_eval_started
        final_improved = False
        status["status"] = "failed"
        status["reason"] = "nonfinite_objective_or_gradient"
        final_nll_bits = (
            math.nan if inputs.previous_objective is None else float(inputs.previous_objective)
        )
        final_grad_inf = math.inf
        final_row = {
            "step": final_step,
            "optimizer/phase": "final_eval",
            "optimizer/eval_position": "final",
            "optimizer/step_applied": False,
            "optimizer/final_eval_status": "failed",
            "optimizer/final_eval_reason": "nonfinite_objective_or_gradient",
            "closure_evals": final_closure_evals,
            "theta_step_inf": 0.0,
            "delta_likelihood_bits": None,
            "stable_loss_steps": inputs.stable_loss_steps,
            "optimizer/lbfgsb_fallback_used_count": float(
                inputs.lbfgsb_fallback_used_count
            ),
            "best_nll_bits": best_nll,
            "best_step": best_step,
            **resume_info,
            "step_s": final_eval_s,
        }
        model.theta.grad = None
        model.clear()
        final_metrics = final_metrics if "final_metrics" in locals() else {}
    else:
        final_metrics.update(
            _evaluate_final_iteration_check(
                config,
                solver=solver,
                evaluation=evaluation,
                model=model,
                baseline_loss=final_loss,
                baseline_grad=model.theta.grad,
                baseline_at_check_iters=final_eval_at_check_iters,
            )
        )
        _, final_projected_grad_inf = evaluation.projected_grad_inf(
            model,
            lower_bound=math.log2(config.min_rate),
            upper_bound=math.log2(config.max_rate),
        )
        final_metrics["grad/projected_inf"] = final_projected_grad_inf
        final_eval_s = time.perf_counter() - final_eval_started
        final_nll_bits = float(final_loss.detach().cpu())
        final_grad_inf = float(final_metrics.get("grad/inf", math.inf))
        final_improved = (
            best_nll is None or final_nll_bits < best_nll - config.best_likelihood_min_delta
        )
        if final_improved:
            best_nll = final_nll_bits
            best_step = final_step
        final_row = {
            "step": final_step,
            "optimizer/phase": "final_eval",
            "optimizer/eval_position": "final",
            "optimizer/step_applied": False,
            "closure_evals": final_closure_evals,
            "theta_step_inf": 0.0,
            "delta_likelihood_bits": None,
            "stable_loss_steps": inputs.stable_loss_steps,
            "best_nll_bits": best_nll,
            "best_step": best_step,
            **resume_info,
            "step_s": final_eval_s,
            **final_metrics,
        }

    inputs.history.append(final_row)

    final_status = {
        **status,
        **resume_info,
        "elapsed_s": time.perf_counter() - inputs.started,
        "best_nll_bits": best_nll,
        "best_step": best_step,
        "previous_objective": (None if final_eval_failed else final_nll_bits),
        "stable_loss_steps": inputs.stable_loss_steps,
        "lbfgsb_fallback_used_count": inputs.lbfgsb_fallback_used_count,
        "lbfgsb_best_retry_count": inputs.lbfgsb_best_retry_count,
    }
    if inputs.lbfgsb_loss_schedule:
        final_status["lbfgsb_loss_schedule_index"] = inputs.lbfgsb_loss_schedule_index

    if final_improved:
        save_status(
            inputs.best_checkpoint,
            model=model,
            optimizer=optimizer,
            step=int(final_row["step"]),
            next_step=final_step,
            status=adaptive_checkpoint_status(final_status),
            row=final_row,
            optimizer_phase=inputs.current_phase,
        )
        sampling_checkpoint = inputs.best_checkpoint
    else:
        sampling_checkpoint = inputs.latest_checkpoint

    save_status(
        inputs.latest_checkpoint,
        model=model,
        optimizer=optimizer,
        step=int(final_row["step"]),
        next_step=final_step,
        status=adaptive_checkpoint_status(final_status),
        row=final_row,
        optimizer_phase=inputs.current_phase,
    )

    if final_status["status"] == "failed":
        sampling_checkpoint = None
    elif sampling_checkpoint is None:
        sampling_checkpoint = inputs.latest_checkpoint

    final_log_likelihood_bits = None if final_eval_failed else -final_nll_bits
    best_log_likelihood_bits = None if best_nll is None else -float(best_nll)
    final_check_summary = _final_check_summary_metrics(final_metrics)
    final_solver_summary = _final_solver_summary_metrics(final_metrics)
    final_projected_grad_inf = (
        None if final_eval_failed else float(final_metrics.get("grad/projected_inf", math.inf))
    )
    route_metadata = effective_route_metadata(config)
    summary = {
        **final_status,
        **route_metadata,
        "families": model.n_families,
        "species": int(model.n_species),
        "batches": len(model.batch_metadata),
        "steps_completed": int(final_row["step"]),
        "sampling_checkpoint": (
            None if sampling_checkpoint is None else str(sampling_checkpoint)
        ),
        "final_nll_bits": final_nll_bits,
        "final_log_likelihood_bits": final_log_likelihood_bits,
        "best_log_likelihood_bits": best_log_likelihood_bits,
        "final_grad_inf": final_grad_inf,
        "final_projected_grad_inf": final_projected_grad_inf,
        **final_check_summary,
        **final_solver_summary,
    }
    run_manifest = _build_run_manifest(
        config,
        command=(
            " ".join(str(item) for item in inputs.command_argv)
            if inputs.command_argv is not None
            else None
        ),
        command_argv=inputs.command_argv,
        route_metadata=route_metadata,
        summary=summary,
        started_wall_time=inputs.started,
        elapsed_wall_s=final_status["elapsed_s"],
        runtime_seed_context=inputs.runtime_seed_context,
    )
    _write_final_artifacts(
        config,
        model=model,
        history=inputs.history,
        final_row=final_row,
        summary=summary,
        run_manifest=run_manifest,
        history_jsonl=inputs.history_jsonl,
        per_family_nll=final_per_family_nll,
        include_per_family_likelihoods=not final_eval_failed,
    )

    return optimization_result_from_summary(config.out_dir, summary)


def _evaluate_final_check_genewise_with_memory_fallback(
    config: RunConfig,
    solver: SolverStageController,
    model: GeneReconModel,
    *,
    check_iters: int,
    evaluation: EvaluationOps,
    original_exc: RuntimeError,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    budgets = evaluation._final_eval_fallback_clade_budgets()
    if not budgets:
        raise original_exc
    _clear_cuda_allocator_cache_if_needed(model)
    fallback_errors: list[str] = []
    for budget in budgets:
        fallback_model: GeneReconModel | None = None
        try:
            fallback_data = config.to_dict()
            fallback_data["clade_budget"] = budget
            fallback_config = RunConfig.from_dict(fallback_data)
            fallback_model = build_alerax_workflow_model(
                fallback_config,
                prefetch_batches=1,
            )
            with torch.no_grad():
                fallback_model.theta.copy_(
                    model.theta.detach().to(
                        device=fallback_model.theta.device,
                        dtype=fallback_model.theta.dtype,
                    )
                )
            configure_solver = getattr(fallback_model, "configure_solver_iterations", None)
            if callable(configure_solver):
                configure_solver(
                    fixed_iters_E=solver.final_check_fixed_iters_E(check_iters),
                    fixed_iters_Pi=check_iters,
                    neumann_terms=check_iters,
                )
            loss_vec, metrics = evaluation.evaluate_genewise_vector_and_grad(
                fallback_model
            )
            if fallback_model.theta.grad is None:
                raise RuntimeError("fallback final check did not produce gradients")
            grad = fallback_model.theta.grad.detach().to(
                device=model.theta.device,
                dtype=model.theta.dtype,
            )
            metrics = dict(metrics)
            metrics["optimizer/final_check_source"] = "fallback_clade_budget"
            metrics["optimizer/final_check_fallback_clade_budget"] = float(budget)
            metrics["optimizer/final_check_reason"] = (
                f"{type(original_exc).__name__}: {original_exc}"
            )
            metrics["optimizer/final_check_fallback_reason"] = (
                f"{type(original_exc).__name__}: {original_exc}"
            )
            metrics.update(tensor_stats("grad", grad))
            return (
                loss_vec.detach().to(
                    device=model.theta.device,
                    dtype=model.theta.dtype,
                ),
                grad,
                metrics,
            )
        except RuntimeError as fallback_exc:
            if not _is_memory_retryable_runtime_error(fallback_exc):
                raise
            fallback_errors.append(
                f"clade_budget={budget}: "
                f"{type(fallback_exc).__name__}: {fallback_exc}"
            )
        finally:
            if fallback_model is not None:
                fallback_model.close()
            _clear_cuda_allocator_cache_if_needed(model)
    raise RuntimeError(
        "final iteration check failed in the resident layout and all "
        "smaller-clade fallbacks failed; original error: "
        f"{type(original_exc).__name__}: {original_exc}; fallbacks: "
        + "; ".join(fallback_errors)
    ) from original_exc


def _evaluate_final_iteration_check(
    config: RunConfig,
    *,
    solver: SolverStageController,
    evaluation: EvaluationOps,
    model: GeneReconModel,
    baseline_loss: torch.Tensor,
    baseline_grad: torch.Tensor,
    baseline_at_check_iters: bool = False,
) -> dict[str, Any]:
    check_iters = solver.final_iteration_check_iters()
    check_iters_E = solver.final_check_fixed_iters_E(check_iters)
    configure_solver = getattr(model, "configure_solver_iterations", None)
    if not callable(configure_solver):
        return {
            "optimizer/final_check_status": "skipped",
            "optimizer/final_check_source": "not_evaluated",
            "optimizer/final_check_reason": (
                "model_has_no_solver_iteration_controls"
            ),
            "optimizer/final_check_iters": check_iters,
            "optimizer/final_check_iters_E": check_iters_E,
        }
    if check_iters <= 0:
        return {
            "optimizer/final_check_status": "disabled",
            "optimizer/final_check_source": "not_evaluated",
            "optimizer/final_check_reason": "final_check_iters_disabled",
            "optimizer/final_check_iters": 0,
            "optimizer/final_check_iters_E": 0,
        }

    metrics: dict[str, Any] = {
        "optimizer/final_check_status": "failed",
        "optimizer/final_check_source": "configured_solver_budget",
        "optimizer/final_check_iters": check_iters,
        "optimizer/final_check_iters_E": check_iters_E,
        "optimizer/final_check_evals": 1,
    }
    if not _is_single_value_tensor(baseline_loss):
        metrics["optimizer/final_check_reason"] = "baseline_loss_not_scalar"
        return metrics
    if not torch.is_tensor(baseline_grad):
        metrics["optimizer/final_check_reason"] = "baseline_gradient_not_tensor"
        return metrics

    baseline_grad = baseline_grad.detach().clone()
    baseline_grad_shape = _tensor_shape(baseline_grad)
    theta_shape = _tensor_shape(model.theta)
    if baseline_grad_shape != theta_shape:
        metrics["optimizer/final_check_reason"] = (
            "baseline_gradient_shape_mismatch: baseline gradient shape "
            f"{baseline_grad_shape} does not match theta shape {theta_shape}"
        )
        return metrics

    baseline_loss_bits = float(baseline_loss.detach().reshape(()).cpu())
    baseline_grad_inf = (
        float(baseline_grad.detach().abs().amax().cpu())
        if baseline_grad.numel()
        else 0.0
    )
    if baseline_at_check_iters:
        return {
            "optimizer/final_check_status": "baseline",
            "optimizer/final_check_iters": check_iters,
            "optimizer/final_check_iters_E": check_iters_E,
            "optimizer/final_check_evals": 0,
            "optimizer/final_check_loss_abs_delta_bits": 0.0,
            "optimizer/final_check_grad_max_abs_delta": 0.0,
            "optimizer/final_check_baseline_grad_inf": baseline_grad_inf,
            "optimizer/final_check_grad_inf": baseline_grad_inf,
        }

    try:
        _clear_cuda_allocator_cache_if_needed(model)
        _clear_cached_solver_runtime_state(model)
        configure_solver(
            fixed_iters_E=check_iters_E,
            fixed_iters_Pi=check_iters,
            neumann_terms=check_iters,
        )
        if config.mode == "genewise" and callable(
            getattr(model, "full_genewise_nll_and_grad", None)
        ):
            try:
                check_loss_vec, _check_metrics = (
                    evaluation.evaluate_genewise_vector_and_grad(model)
                )
                check_grad = model.theta.grad
            except RuntimeError as check_exc:
                if not _is_memory_retryable_runtime_error(check_exc):
                    raise
                _drop_cached_static_states_if_needed(model)
                configure_solver(
                    fixed_iters_E=check_iters_E,
                    fixed_iters_Pi=check_iters,
                    neumann_terms=check_iters,
                )
                try:
                    check_loss_vec, _check_metrics = (
                        evaluation.evaluate_genewise_vector_and_grad(model)
                    )
                    check_grad = model.theta.grad
                    metrics.update(
                        {
                            "optimizer/final_check_source": (
                                "recomputed_after_cache_drop"
                            ),
                            "optimizer/final_check_reason": (
                                f"{type(check_exc).__name__}: {check_exc}"
                            ),
                            "optimizer/final_check_fallback_reason": (
                                f"{type(check_exc).__name__}: {check_exc}"
                            ),
                        }
                    )
                except RuntimeError as retry_exc:
                    if not _is_memory_retryable_runtime_error(retry_exc):
                        raise
                    check_loss_vec, check_grad, fallback_metrics = (
                        _evaluate_final_check_genewise_with_memory_fallback(
                            config,
                            solver=solver,
                            model=model,
                            check_iters=check_iters,
                            evaluation=evaluation,
                            original_exc=check_exc,
                        )
                    )
                    metrics.update(fallback_metrics)
        else:
            check_loss_vec, _check_metrics = evaluation.evaluate_and_backward(model)
            check_grad = model.theta.grad

        if not _is_single_value_tensor(check_loss_vec.sum()):
            metrics["optimizer/final_check_reason"] = "check_loss_not_scalar"
            return metrics
        check_loss = check_loss_vec.sum().detach().reshape(())
        if check_grad is None:
            metrics["optimizer/final_check_reason"] = "missing_check_gradient"
            return metrics
        if not torch.is_tensor(check_grad):
            metrics["optimizer/final_check_reason"] = "check_gradient_not_tensor"
            return metrics
        check_grad = check_grad.detach()
        if _tensor_shape(check_grad) != baseline_grad_shape:
            metrics["optimizer/final_check_reason"] = (
                "gradient_shape_mismatch: check gradient shape "
                f"{_tensor_shape(check_grad)} does not match baseline gradient "
                f"shape {baseline_grad_shape}"
            )
            return metrics
        check_failed = (
            not torch.isfinite(check_loss).item()
            or not _is_finite_tensor(check_grad)
        )
        if check_failed:
            metrics["optimizer/final_check_reason"] = (
                "nonfinite_objective_or_gradient"
            )
            return metrics

        check_loss_bits = float(check_loss.detach().cpu())
        grad_delta = (check_grad - baseline_grad).detach()
        grad_delta_inf = (
            float(grad_delta.abs().amax().cpu()) if grad_delta.numel() else 0.0
        )
        check_grad_inf = (
            float(check_grad.abs().amax().cpu()) if check_grad.numel() else 0.0
        )
        grad_scale = max(baseline_grad_inf, check_grad_inf, 1.0)
        metrics.update(
            {
                "optimizer/final_check_status": "ok",
                "optimizer/final_check_loss_bits": check_loss_bits,
                "optimizer/final_check_loss_delta_bits": (
                    check_loss_bits - baseline_loss_bits
                ),
                "optimizer/final_check_loss_abs_delta_bits": abs(
                    check_loss_bits - baseline_loss_bits
                ),
                "optimizer/final_check_grad_inf": check_grad_inf,
                "optimizer/final_check_grad_baseline_inf": baseline_grad_inf,
                "optimizer/final_check_grad_max_abs_delta": grad_delta_inf,
                "optimizer/final_check_grad_rel_inf_delta": (
                    grad_delta_inf / grad_scale
                ),
            }
        )
        return metrics
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        metrics["optimizer/final_check_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return metrics
    finally:
        solver.configure_stage(model, "full")
        model.theta.grad = baseline_grad
        _clear_cached_solver_runtime_state(model)

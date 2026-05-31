from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from gpurec.api._theta_constraints import finite_theta_rate_bounds_log2
from gpurec.api.model import GeneReconModel

from . import _finalization_checks as _finalization_check_module
from ._artifacts import (
    _build_run_manifest,
    _final_check_summary_metrics,
    _final_solver_summary_metrics,
    _write_final_artifacts,
)
from ._batch_final_cache import BatchFinalCache
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
    batch_final_cache: BatchFinalCache | None
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
    cached_batch_final_result = (
        inputs.batch_final_cache.cached_final_result()
        if inputs.batchwise_active_optimizer and inputs.batch_final_cache is not None
        else None
    )

    if cached_batch_final_result is not None:
        final_per_family_nll, model.theta.grad = cached_batch_final_result
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
        lower_bound, upper_bound = finite_theta_rate_bounds_log2(
            config.min_rate,
            config.max_rate,
        )
        _, final_projected_grad_inf = evaluation.projected_grad_inf(
            model,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
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


def _sync_finalization_check_hooks() -> None:
    _finalization_check_module.build_alerax_workflow_model = (
        build_alerax_workflow_model
    )
    _finalization_check_module._clear_cuda_allocator_cache_if_needed = (
        _clear_cuda_allocator_cache_if_needed
    )
    _finalization_check_module._clear_cached_solver_runtime_state = (
        _clear_cached_solver_runtime_state
    )
    _finalization_check_module._drop_cached_static_states_if_needed = (
        _drop_cached_static_states_if_needed
    )
    _finalization_check_module._is_memory_retryable_runtime_error = (
        _is_memory_retryable_runtime_error
    )
    _finalization_check_module._is_finite_tensor = _is_finite_tensor
    _finalization_check_module._is_single_value_tensor = _is_single_value_tensor
    _finalization_check_module._tensor_shape = _tensor_shape
    _finalization_check_module.tensor_stats = tensor_stats


def _evaluate_final_check_genewise_with_memory_fallback(
    config: RunConfig,
    solver: SolverStageController,
    model: GeneReconModel,
    *,
    check_iters: int,
    evaluation: EvaluationOps,
    original_exc: RuntimeError,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    _sync_finalization_check_hooks()
    helper = (
        _finalization_check_module._evaluate_final_check_genewise_with_memory_fallback
    )
    return helper(
        config,
        solver,
        model,
        check_iters=check_iters,
        evaluation=evaluation,
        original_exc=original_exc,
    )


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
    _sync_finalization_check_hooks()
    return _finalization_check_module._evaluate_final_iteration_check(
        config,
        solver=solver,
        evaluation=evaluation,
        model=model,
        baseline_loss=baseline_loss,
        baseline_grad=baseline_grad,
        baseline_at_check_iters=baseline_at_check_iters,
    )

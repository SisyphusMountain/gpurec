from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _IterationArtifacts:
    row: dict[str, Any]
    checkpoint_status: dict[str, Any]


@dataclass(frozen=True)
class _IterationArtifactsContext:
    active_objective_scope: bool
    global_solver_warmup: bool
    adagrad_restart_dynamic_enabled: bool
    lbfgsb_loss_schedule: tuple[Any, ...]


@dataclass(frozen=True)
class _IterationArtifactsState:
    adagrad_restart_dynamic_phase_index: int
    adagrad_restart_dynamic_phase_start_step: int
    lbfgsb_loss_schedule_index: int


@dataclass(frozen=True)
class _IterationArtifactsInputs:
    step: int
    phase: str
    eval_position: str
    closure_evals: int
    theta_step: float
    delta: float | None
    loss_change_tol_bits: float
    best_likelihood_min_delta_bits: float
    stable_loss_steps: int
    row_best_nll: float | None
    row_best_step: int | None
    resume_info: dict[str, Any]
    step_s: float
    metrics: dict[str, Any]
    previous_objective: float | None
    lbfgsb_fallback_used_count: int
    lbfgsb_best_retry_count: int
    lbfgsb_loss_schedule_next_index: int | None
    active_batch_index: int
    active_solver_stage: str
    active_batch_local_step: int
    adagrad_restart_phase_next_index: int | None
    adagrad_restart_phase_next_start_step: int | None


def build_iteration_artifacts(
    context: _IterationArtifactsContext,
    state: _IterationArtifactsState,
    inputs: _IterationArtifactsInputs,
) -> _IterationArtifacts:
    row = {
        "step": inputs.step,
        "optimizer/phase": inputs.phase,
        "optimizer/eval_position": inputs.eval_position,
        "closure_evals": inputs.closure_evals,
        "theta_step_inf": inputs.theta_step,
        "delta_likelihood_bits": inputs.delta,
        "loss_change_tol_bits": inputs.loss_change_tol_bits,
        "best_likelihood_min_delta_bits": inputs.best_likelihood_min_delta_bits,
        "stable_loss_steps": inputs.stable_loss_steps,
        "best_nll_bits": inputs.row_best_nll,
        "best_step": inputs.row_best_step,
        **inputs.resume_info,
        "step_s": inputs.step_s,
        **inputs.metrics,
    }

    checkpoint_status: dict[str, Any] = {
        "status": "running",
        "reason": "running",
        **inputs.resume_info,
        "best_nll_bits": inputs.row_best_nll,
        "best_step": inputs.row_best_step,
        "previous_objective": inputs.previous_objective,
        "stable_loss_steps": inputs.stable_loss_steps,
        "lbfgsb_fallback_used_count": inputs.lbfgsb_fallback_used_count,
        "lbfgsb_best_retry_count": inputs.lbfgsb_best_retry_count,
    }
    if context.lbfgsb_loss_schedule:
        checkpoint_status["lbfgsb_loss_schedule_index"] = (
            state.lbfgsb_loss_schedule_index
            if inputs.lbfgsb_loss_schedule_next_index is None
            else inputs.lbfgsb_loss_schedule_next_index
        )
        if inputs.lbfgsb_loss_schedule_next_index is not None:
            checkpoint_status["stable_loss_steps"] = 0
    if context.active_objective_scope:
        checkpoint_status["active_batch_index"] = inputs.active_batch_index
        checkpoint_status["active_solver_stage"] = inputs.active_solver_stage
        checkpoint_status["active_batch_local_step"] = inputs.active_batch_local_step
    elif context.global_solver_warmup:
        checkpoint_status["active_solver_stage"] = inputs.active_solver_stage
    if context.adagrad_restart_dynamic_enabled:
        checkpoint_status["adagrad_restart_dynamic_phase_index"] = (
            state.adagrad_restart_dynamic_phase_index
            if inputs.adagrad_restart_phase_next_index is None
            else inputs.adagrad_restart_phase_next_index
        )
        checkpoint_status["adagrad_restart_dynamic_phase_start_step"] = (
            state.adagrad_restart_dynamic_phase_start_step
            if inputs.adagrad_restart_phase_next_start_step is None
            else inputs.adagrad_restart_phase_next_start_step
        )
        if inputs.adagrad_restart_phase_next_index is not None:
            checkpoint_status["previous_objective"] = None
            checkpoint_status["stable_loss_steps"] = 0
    return _IterationArtifacts(row=row, checkpoint_status=checkpoint_status)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gpurec.api.model import GeneReconModel

from ._phase import (
    _ActiveAdagradRestartPhase,
    _adagrad_restart_phase_name,
    _continues_after_adagrad_restart_prefix,
)
from .config import AdagradRestartPhase, LossStopPhase, RunConfig


@dataclass(frozen=True)
class _LoopPolicyContext:
    config: RunConfig
    batchwise_active_optimizer: bool
    batchwise_active_optimizer_phases: frozenset[str]
    global_solver_warmup: bool
    adagrad_restart_dynamic_enabled: bool
    adagrad_restart_specs: tuple[AdagradRestartPhase, ...]
    lbfgsb_loss_schedule: tuple[LossStopPhase, ...]


@dataclass(frozen=True)
class _LoopPolicyState:
    objective_state: Any
    batch_state: Any
    lbfgsb_state: Any


@dataclass(frozen=True)
class _LoopPolicyInputs:
    step: int
    phase: str
    metrics: dict[str, Any]
    model: GeneReconModel
    optimizer: torch.optim.Optimizer | None
    adagrad_restart_active_phase: _ActiveAdagradRestartPhase | None
    adagrad_restart_phase_step: int | None


@dataclass(frozen=True)
class _ProjectedLoopPolicy:
    backoff: bool = False
    min_lr_reached: bool = False
    bounded_high_projected_plateau: bool = False


@dataclass(frozen=True)
class _AdagradRestartLoopPolicy:
    next_index: int | None = None
    next_start_step: int | None = None
    terminal_status: dict[str, str] | None = None


@dataclass(frozen=True)
class _BestLoopPolicy:
    row_best_nll: float | None
    row_best_step: int | None
    save_best_after_row: bool


@dataclass(frozen=True)
class _LBFGSBHighKKTLoopPolicy:
    status: dict[str, str] | None = None


@dataclass(frozen=True)
class _LBFGSBLossScheduleLoopPolicy:
    next_index: int | None = None


@dataclass(frozen=True)
class _LoopPolicyResult:
    active_objective_scope: bool
    solver_stage_scope: bool
    effective_loss_patience: int
    loss_change_tol_bits: float
    best_likelihood_min_delta_bits: float
    objective: float
    delta: float | None
    projected_lbfgs_backoff: bool
    projected_lbfgs_min_lr_reached: bool
    bounded_high_projected_plateau: bool
    row_best_nll: float | None
    row_best_step: int | None
    save_best_after_row: bool
    adagrad_restart_phase_next_index: int | None
    adagrad_restart_phase_next_start_step: int | None
    adagrad_restart_terminal_status: dict[str, str] | None
    lbfgsb_high_kkt_status: dict[str, str] | None
    lbfgsb_loss_schedule_next_index: int | None


def _apply_objective_scope(
    context: _LoopPolicyContext,
    state: _LoopPolicyState,
    inputs: _LoopPolicyInputs,
) -> tuple[bool, bool, int]:
    active_objective_scope = (
        context.batchwise_active_optimizer
        and inputs.phase in context.batchwise_active_optimizer_phases
    )
    solver_stage_scope = active_objective_scope or context.global_solver_warmup
    if solver_stage_scope:
        inputs.metrics.setdefault(
            "optimizer/solver_stage",
            state.batch_state.solver_stage,
        )
    active_family_count = (
        max(1, int(inputs.metrics.get("optimizer/batch_family_count", 1)))
        if active_objective_scope
        else 1
    )
    return active_objective_scope, solver_stage_scope, active_family_count


def _effective_loss_policy(
    context: _LoopPolicyContext,
    state: _LoopPolicyState,
    inputs: _LoopPolicyInputs,
) -> tuple[float, int]:
    config = context.config
    metrics = inputs.metrics
    effective_loss_change_tol = float(config.loss_change_tol)
    effective_loss_patience = int(config.loss_patience)
    if inputs.phase == "lbfgsb" and context.lbfgsb_loss_schedule:
        state.lbfgsb_state.loss_schedule_index = min(
            state.lbfgsb_state.loss_schedule_index,
            len(context.lbfgsb_loss_schedule) - 1,
        )
        loss_phase = context.lbfgsb_loss_schedule[
            state.lbfgsb_state.loss_schedule_index
        ]
        effective_loss_change_tol = float(loss_phase.loss_change_tol)
        effective_loss_patience = int(loss_phase.loss_patience)
        metrics["optimizer/lbfgsb_loss_schedule_index"] = float(
            state.lbfgsb_state.loss_schedule_index
        )
        metrics["optimizer/lbfgsb_loss_schedule_phases"] = float(
            len(context.lbfgsb_loss_schedule)
        )
        metrics["optimizer/lbfgsb_loss_schedule_active_tol"] = (
            effective_loss_change_tol
        )
        metrics["optimizer/lbfgsb_loss_schedule_active_patience"] = float(
            effective_loss_patience
        )
    return effective_loss_change_tol, effective_loss_patience


def _apply_projected_loop_policy(
    context: _LoopPolicyContext,
    inputs: _LoopPolicyInputs,
    *,
    delta: float | None,
    loss_change_tol_bits: float,
) -> _ProjectedLoopPolicy:
    config = context.config
    metrics = inputs.metrics
    projected_lbfgs_backoff = False
    projected_lbfgs_min_lr_reached = False
    bounded_high_projected_plateau = False
    if inputs.phase in {"projected-lbfgs", "lbfgsb"} and inputs.optimizer is not None:
        metric_prefix = (
            "projected_lbfgs" if inputs.phase == "projected-lbfgs" else "lbfgsb"
        )
        projected_inf_raw = metrics.get("grad/projected_inf")
        projected_inf_value = (
            float(projected_inf_raw)
            if projected_inf_raw is not None
            else float("inf")
        )
        accepted = bool(metrics.get(f"optimizer/{metric_prefix}_accepted", True))
        plateau = delta is not None and delta <= loss_change_tol_bits
        high_projected_grad = projected_inf_value > config.projected_grad_tol
        bounded_high_projected_plateau = (
            config.loss_stop_projected_grad_gate
            and high_projected_grad
            and (plateau or not accepted)
        )
        if (
            inputs.phase == "projected-lbfgs"
            and high_projected_grad
            and (plateau or not accepted)
        ):
            group = inputs.optimizer.param_groups[0]
            old_lr = float(group["lr"])
            min_lr = float(config.projected_lbfgs_min_lr)
            shrink = float(group.get("shrink", 0.5))
            accepted_alpha = float(
                metrics.get("optimizer/projected_lbfgs_alpha", 0.0)
            )
            if 0.0 < accepted_alpha < old_lr:
                candidate_lr = accepted_alpha * shrink
            else:
                candidate_lr = old_lr * shrink
            new_lr = max(min_lr, candidate_lr)
            if new_lr < old_lr:
                group["lr"] = new_lr
                projected_lbfgs_backoff = True
            else:
                projected_lbfgs_min_lr_reached = True
            metrics["optimizer/projected_lbfgs_projected_grad_tol"] = float(
                config.projected_grad_tol
            )
            metrics["optimizer/projected_lbfgs_loss_stop_projected_grad_gate"] = bool(
                config.loss_stop_projected_grad_gate
            )
            metrics["optimizer/projected_lbfgs_lr_before"] = old_lr
            metrics["optimizer/projected_lbfgs_lr_after"] = new_lr
            metrics["optimizer/projected_lbfgs_lr_reduced"] = (
                projected_lbfgs_backoff
            )
            metrics["optimizer/projected_lbfgs_min_lr_reached"] = (
                projected_lbfgs_min_lr_reached
            )
            metrics["optimizer/projected_lbfgs_high_projected_grad"] = True
        else:
            metrics[f"optimizer/{metric_prefix}_projected_grad_tol"] = float(
                config.projected_grad_tol
            )
            metrics[f"optimizer/{metric_prefix}_loss_stop_projected_grad_gate"] = bool(
                config.loss_stop_projected_grad_gate
            )
            if inputs.phase == "projected-lbfgs":
                metrics["optimizer/projected_lbfgs_lr_reduced"] = False
                metrics["optimizer/projected_lbfgs_min_lr_reached"] = False
            metrics[f"optimizer/{metric_prefix}_high_projected_grad"] = (
                high_projected_grad
            )
            metrics[f"optimizer/{metric_prefix}_blocked_loss_stop"] = (
                bounded_high_projected_plateau
            )
    return _ProjectedLoopPolicy(
        backoff=projected_lbfgs_backoff,
        min_lr_reached=projected_lbfgs_min_lr_reached,
        bounded_high_projected_plateau=bounded_high_projected_plateau,
    )


def _update_objective_plateau(
    state: _LoopPolicyState,
    *,
    objective: float,
    delta: float | None,
    loss_change_tol_bits: float,
    projected_policy: _ProjectedLoopPolicy,
) -> bool:
    objective_plateau_this_row = (
        delta is not None
        and delta <= loss_change_tol_bits
        and not projected_policy.backoff
        and not projected_policy.min_lr_reached
    )
    if (
        objective_plateau_this_row
        and not projected_policy.bounded_high_projected_plateau
    ):
        state.objective_state.stable_loss_steps += 1
    else:
        state.objective_state.stable_loss_steps = 0
    state.objective_state.previous_objective = objective
    return objective_plateau_this_row


def _apply_adagrad_restart_policy(
    context: _LoopPolicyContext,
    state: _LoopPolicyState,
    inputs: _LoopPolicyInputs,
) -> _AdagradRestartLoopPolicy:
    config = context.config
    metrics = inputs.metrics
    adagrad_restart_phase_next_index: int | None = None
    adagrad_restart_phase_next_start_step: int | None = None
    adagrad_restart_terminal_status: dict[str, str] | None = None
    if (
        context.adagrad_restart_dynamic_enabled
        and inputs.adagrad_restart_active_phase is not None
        and inputs.adagrad_restart_phase_step is not None
    ):
        phase_done_by_loss = state.objective_state.stable_loss_steps >= int(
            config.adagrad_restart_phase_loss_patience
        )
        phase_done_by_cap = (
            inputs.adagrad_restart_phase_step + 1
            >= int(inputs.adagrad_restart_active_phase.phase.steps)
        )
        phase_done_reason = None
        if phase_done_by_loss:
            phase_done_reason = "loss_change_patience"
        elif phase_done_by_cap:
            phase_done_reason = "phase_step_cap"
        if phase_done_reason is not None:
            last_adagrad_phase = (
                inputs.adagrad_restart_active_phase.index + 1
                >= len(context.adagrad_restart_specs)
            )
            metrics["optimizer/adagrad_restart_dynamic_phase"] = True
            metrics["optimizer/adagrad_restart_phase_complete"] = True
            metrics["optimizer/adagrad_restart_phase_complete_reason"] = (
                phase_done_reason
            )
            metrics["optimizer/adagrad_restart_phase_loss_patience"] = float(
                config.adagrad_restart_phase_loss_patience
            )
            if last_adagrad_phase:
                if _continues_after_adagrad_restart_prefix(config.optimizer):
                    metrics["optimizer/adagrad_restart_next_phase"] = "lbfgsb"
                    adagrad_restart_phase_next_index = len(
                        context.adagrad_restart_specs
                    )
                    adagrad_restart_phase_next_start_step = inputs.step + 1
                else:
                    adagrad_restart_terminal_status = {
                        "status": "converged",
                        "reason": (
                            "adagrad_restart_phase_loss_patience"
                            if phase_done_by_loss
                            else "adagrad_restart_schedule_complete"
                        ),
                    }
            else:
                adagrad_restart_phase_next_index = (
                    inputs.adagrad_restart_active_phase.index + 1
                )
                adagrad_restart_phase_next_start_step = inputs.step + 1
                metrics["optimizer/adagrad_restart_next_phase"] = (
                    _adagrad_restart_phase_name(
                        context.adagrad_restart_specs,
                        adagrad_restart_phase_next_index,
                    )
                )
        else:
            metrics["optimizer/adagrad_restart_dynamic_phase"] = True
            metrics["optimizer/adagrad_restart_phase_complete"] = False
    return _AdagradRestartLoopPolicy(
        next_index=adagrad_restart_phase_next_index,
        next_start_step=adagrad_restart_phase_next_start_step,
        terminal_status=adagrad_restart_terminal_status,
    )


def _update_best_policy(
    state: _LoopPolicyState,
    inputs: _LoopPolicyInputs,
    *,
    active_objective_scope: bool,
    objective: float,
    best_likelihood_min_delta_bits: float,
) -> _BestLoopPolicy:
    if active_objective_scope:
        (
            row_best_nll,
            row_best_step,
            _,
        ) = state.batch_state.update_best(
            objective=objective,
            step=inputs.step,
            best_likelihood_min_delta_bits=best_likelihood_min_delta_bits,
        )
        save_best_after_row = False
    else:
        (
            row_best_nll,
            row_best_step,
            save_best_after_row,
        ) = state.objective_state.update_best(
            objective=objective,
            step=inputs.step,
            best_likelihood_min_delta_bits=best_likelihood_min_delta_bits,
        )
    return _BestLoopPolicy(
        row_best_nll=row_best_nll,
        row_best_step=row_best_step,
        save_best_after_row=save_best_after_row,
    )


def _apply_lbfgsb_high_kkt_policy(
    context: _LoopPolicyContext,
    state: _LoopPolicyState,
    inputs: _LoopPolicyInputs,
    *,
    objective_plateau_this_row: bool,
) -> _LBFGSBHighKKTLoopPolicy:
    config = context.config
    metrics = inputs.metrics
    lbfgsb_high_kkt_status: dict[str, str] | None = None
    high_kkt_stop_patience = 0
    high_kkt_stop_signal = False
    high_kkt_objective_stalled = False
    if inputs.phase == "lbfgsb":
        if bool(metrics.get("optimizer/lbfgsb_fallback_used", False)):
            state.lbfgsb_state.fallback_used_count += 1
        high_kkt_stall_count = int(
            metrics.get("optimizer/lbfgsb_high_kkt_stall_count", 0)
        )
        high_kkt_stop_patience = int(config.lbfgsb_high_kkt_stop_patience)
        fallback_used_this_row = bool(
            metrics.get("optimizer/lbfgsb_fallback_used", False)
        )
        fallback_budget_exhausted_this_row = bool(
            metrics.get(
                "optimizer/lbfgsb_fallback_budget_exhausted",
                False,
            )
        )
        high_kkt_stop_signal = high_kkt_stall_count >= (
            2 if high_kkt_stop_patience <= 1 else high_kkt_stop_patience
        ) or (
            high_kkt_stall_count >= high_kkt_stop_patience
            and (fallback_used_this_row or fallback_budget_exhausted_this_row)
        )
        high_kkt_objective_stalled = objective_plateau_this_row
    high_kkt_final_loss_phase = (
        not context.lbfgsb_loss_schedule
        or state.lbfgsb_state.loss_schedule_index
        >= len(context.lbfgsb_loss_schedule) - 1
    )
    high_kkt_stop_ready = (
        high_kkt_stop_patience > 0
        and high_kkt_stop_signal
        and high_kkt_objective_stalled
        and high_kkt_final_loss_phase
        and state.lbfgsb_state.fallback_used_count
        >= int(config.lbfgsb_high_kkt_stop_min_fallbacks)
    )
    metrics["optimizer/lbfgsb_fallback_used_count"] = float(
        state.lbfgsb_state.fallback_used_count
    )
    metrics["optimizer/lbfgsb_high_kkt_stop_patience"] = float(
        high_kkt_stop_patience
    )
    metrics["optimizer/lbfgsb_high_kkt_stop_min_fallbacks"] = float(
        int(config.lbfgsb_high_kkt_stop_min_fallbacks)
    )
    metrics["optimizer/lbfgsb_high_kkt_objective_stalled"] = (
        high_kkt_objective_stalled
    )
    metrics["optimizer/lbfgsb_high_kkt_final_loss_phase"] = (
        high_kkt_final_loss_phase
    )
    metrics["optimizer/lbfgsb_high_kkt_stop_ready"] = high_kkt_stop_ready
    if high_kkt_stop_ready:
        lbfgsb_high_kkt_status = {
            "status": "converged",
            "reason": "lbfgsb_high_kkt_tiny_progress_patience",
        }
    return _LBFGSBHighKKTLoopPolicy(status=lbfgsb_high_kkt_status)


def _apply_lbfgsb_loss_schedule_policy(
    context: _LoopPolicyContext,
    state: _LoopPolicyState,
    inputs: _LoopPolicyInputs,
    *,
    effective_loss_patience: int,
    lbfgsb_high_kkt_status: dict[str, str] | None,
) -> _LBFGSBLossScheduleLoopPolicy:
    config = context.config
    metrics = inputs.metrics
    lbfgsb_loss_schedule_next_index: int | None = None
    if (
        inputs.phase == "lbfgsb"
        and context.lbfgsb_loss_schedule
        and lbfgsb_high_kkt_status is None
        and effective_loss_patience
        and state.objective_state.stable_loss_steps >= effective_loss_patience
        and state.lbfgsb_state.loss_schedule_index + 1
        < len(context.lbfgsb_loss_schedule)
    ):
        lbfgsb_loss_schedule_next_index = state.lbfgsb_state.loss_schedule_index + 1
        next_loss_phase = context.lbfgsb_loss_schedule[
            lbfgsb_loss_schedule_next_index
        ]
        metrics["optimizer/lbfgsb_loss_schedule_advance"] = True
        metrics["optimizer/lbfgsb_loss_schedule_next_index"] = float(
            lbfgsb_loss_schedule_next_index
        )
        metrics["optimizer/lbfgsb_loss_schedule_next_tol"] = float(
            next_loss_phase.loss_change_tol
        )
        metrics["optimizer/lbfgsb_loss_schedule_next_patience"] = float(
            next_loss_phase.loss_patience
        )
        if config.lbfgsb_loss_schedule_force_fallback and inputs.optimizer is not None:
            opt_state = inputs.optimizer.state.get(inputs.model.theta)
            if isinstance(opt_state, dict):
                previous_stalls = int(
                    opt_state.get("consecutive_high_kkt_stalls", 0)
                )
                opt_state["consecutive_high_kkt_stalls"] = max(previous_stalls, 2)
                metrics["optimizer/lbfgsb_loss_schedule_force_fallback_next"] = True
                metrics[
                    "optimizer/lbfgsb_loss_schedule_force_fallback_previous_stalls"
                ] = float(previous_stalls)
    elif inputs.phase == "lbfgsb" and context.lbfgsb_loss_schedule:
        metrics["optimizer/lbfgsb_loss_schedule_advance"] = False
        metrics["optimizer/lbfgsb_loss_schedule_force_fallback_next"] = False
    return _LBFGSBLossScheduleLoopPolicy(next_index=lbfgsb_loss_schedule_next_index)


def apply_post_step_loop_policies(
    context: _LoopPolicyContext,
    state: _LoopPolicyState,
    inputs: _LoopPolicyInputs,
) -> _LoopPolicyResult:
    active_objective_scope, solver_stage_scope, active_family_count = (
        _apply_objective_scope(context, state, inputs)
    )
    effective_loss_change_tol, effective_loss_patience = _effective_loss_policy(
        context,
        state,
        inputs,
    )
    loss_change_tol_bits = effective_loss_change_tol * active_family_count
    best_likelihood_min_delta_bits = (
        context.config.best_likelihood_min_delta * active_family_count
    )
    objective = float(inputs.metrics["likelihood/data_nll_bits"])
    delta = (
        None
        if state.objective_state.previous_objective is None
        else state.objective_state.previous_objective - objective
    )
    projected_policy = _apply_projected_loop_policy(
        context,
        inputs,
        delta=delta,
        loss_change_tol_bits=loss_change_tol_bits,
    )
    objective_plateau_this_row = _update_objective_plateau(
        state,
        objective=objective,
        delta=delta,
        loss_change_tol_bits=loss_change_tol_bits,
        projected_policy=projected_policy,
    )
    adagrad_restart_policy = _apply_adagrad_restart_policy(context, state, inputs)
    best_policy = _update_best_policy(
        state,
        inputs,
        active_objective_scope=active_objective_scope,
        objective=objective,
        best_likelihood_min_delta_bits=best_likelihood_min_delta_bits,
    )
    lbfgsb_high_kkt_policy = _apply_lbfgsb_high_kkt_policy(
        context,
        state,
        inputs,
        objective_plateau_this_row=objective_plateau_this_row,
    )
    lbfgsb_loss_schedule_policy = _apply_lbfgsb_loss_schedule_policy(
        context,
        state,
        inputs,
        effective_loss_patience=effective_loss_patience,
        lbfgsb_high_kkt_status=lbfgsb_high_kkt_policy.status,
    )
    return _LoopPolicyResult(
        active_objective_scope=active_objective_scope,
        solver_stage_scope=solver_stage_scope,
        effective_loss_patience=effective_loss_patience,
        loss_change_tol_bits=loss_change_tol_bits,
        best_likelihood_min_delta_bits=best_likelihood_min_delta_bits,
        objective=objective,
        delta=delta,
        projected_lbfgs_backoff=projected_policy.backoff,
        projected_lbfgs_min_lr_reached=projected_policy.min_lr_reached,
        bounded_high_projected_plateau=(
            projected_policy.bounded_high_projected_plateau
        ),
        row_best_nll=best_policy.row_best_nll,
        row_best_step=best_policy.row_best_step,
        save_best_after_row=best_policy.save_best_after_row,
        adagrad_restart_phase_next_index=adagrad_restart_policy.next_index,
        adagrad_restart_phase_next_start_step=adagrad_restart_policy.next_start_step,
        adagrad_restart_terminal_status=adagrad_restart_policy.terminal_status,
        lbfgsb_high_kkt_status=lbfgsb_high_kkt_policy.status,
        lbfgsb_loss_schedule_next_index=lbfgsb_loss_schedule_policy.next_index,
    )

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch

from gpurec.api.model import GeneReconModel

from ._adaptive_rebatch import _AdaptiveRebatchState
from ._batch_final_cache import BatchFinalCache
from ._fd_newton import _FDNewtonHessianState
from ._runtime_helpers import _is_finite_tensor
from ._solver_stage import SolverStageController
from ._step_plan import _StepPlanningState
from .config import RunConfig

_HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES = 400_000
_HESSIAN_SGD_SKIP_FULL_AFTER_WARMUP_MIN_CLADES = (
    _HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
)


@dataclass
class IterationTransition:
    status: dict[str, str] | None = None
    continue_loop: bool = False
    break_loop: bool = False
    reset_optimizer: bool = False
    save_latest: bool = False
    save_best: bool = False
    action: str | None = None
    adagrad_restart_next: tuple[int, int] | None = None
    next_adagrad_phase: tuple[int, int] | None = None
    adaptive_rebatch_indices: list[int] | None = None
    lbfgsb_loss_schedule_next_index: int | None = None
    next_batch_active_index: int | None = None


@dataclass
class IterationTransitionExecution:
    status: dict[str, str] | None
    continue_loop: bool
    break_loop: bool
    optimizer: torch.optim.Optimizer | None
    fd_newton_hessian_state: _FDNewtonHessianState | None
    hessian_sgd_line_search_active: bool
    hessian_sgd_low_accept_steps: int
    resume_info: dict[str, Any]
    planning_state: _StepPlanningState


@dataclass(frozen=True)
class IterationTransitionOps:
    active_batch_indices: Callable[[GeneReconModel], torch.Tensor]
    clear_cached_static_states_if_needed: Callable[[GeneReconModel], None]
    clear_cached_solver_runtime_state: Callable[[GeneReconModel], None]
    load_checkpoint: Callable[[Path], dict[str, Any]]
    validate_checkpoint_model_compatibility: Callable[..., None]
    restore_model_theta: Callable[[GeneReconModel, dict[str, Any]], None]
    make_optimizer: Callable[[RunConfig, GeneReconModel, str], torch.optim.Optimizer]
    restore_optimizer_state: Callable[..., dict[str, Any]]
    resume_state_from_payload: Callable[[Path, dict[str, Any]], Any]
    save_status: Callable[..., None]
    adaptive_checkpoint_status: Callable[[dict[str, Any]], dict[str, Any]]
    print_progress_row: Callable[..., None]
    fd_adam_warmup_steps: int


def build_batch_transition_checkpoint_status(
    checkpoint_status: dict[str, Any],
    batch_state: Any,
    *,
    previous_objective: float | None,
    stable_loss_steps: int = 0,
    best_nll_bits: float | None,
    best_step: int | None,
) -> dict[str, Any]:
    return {
        **checkpoint_status,
        "active_batch_index": batch_state.active_index,
        "active_solver_stage": batch_state.solver_stage,
        "active_batch_local_step": batch_state.local_step,
        "previous_objective": previous_objective,
        "stable_loss_steps": stable_loss_steps,
        "best_nll_bits": best_nll_bits,
        "best_step": best_step,
    }


def _classify_iteration_transition(
    *,
    adaptive_rebatch_stop: bool,
    rejected_nonfinite_parameter_update: bool,
    adaptive_rebatch_pending_indices: list[int] | None,
    adagrad_restart_terminal_status: dict[str, str] | None,
    adagrad_restart_phase_next_index: int | None,
    adagrad_restart_phase_next_start_step: int | None,
    lbfgsb_loss_schedule_next_index: int | None,
    lbfgsb_high_kkt_status: dict[str, str] | None,
    projected_lbfgs_min_lr_reached: bool,
    hessian_sgd_activate_line_search: bool,
    step_status: dict[str, str] | None,
    active_objective_scope: bool,
    active_batch_count: int,
    batch_state_active_index: int,
    can_lbfgsb_retry: bool,
) -> IterationTransition:
    if rejected_nonfinite_parameter_update:
        return IterationTransition(
            status={"status": "failed", "reason": "nonfinite_parameter_update"},
            break_loop=True,
            action="nonfinite_parameter_update",
            reset_optimizer=False,
            save_latest=False,
            save_best=False,
        )

    if adaptive_rebatch_stop:
        return IterationTransition(
            status={"status": "converged", "reason": "best_likelihood_patience"},
            break_loop=True,
            action="adaptive_rebatch_stop",
            reset_optimizer=False,
            save_latest=False,
            save_best=False,
        )

    if adagrad_restart_terminal_status is not None:
        return IterationTransition(
            status=adagrad_restart_terminal_status,
            break_loop=True,
            action="adagrad_restart_terminal",
            reset_optimizer=False,
            save_latest=True,
            save_best=False,
        )

    if adagrad_restart_phase_next_index is not None:
        return IterationTransition(
            continue_loop=True,
            action="adagrad_restart_advance",
            reset_optimizer=True,
            save_latest=False,
            save_best=False,
            adagrad_restart_next=(
                adagrad_restart_phase_next_index,
                adagrad_restart_phase_next_start_step,
            ),
            next_adagrad_phase=(
                adagrad_restart_phase_next_index,
                adagrad_restart_phase_next_start_step,
            ),
        )

    if adaptive_rebatch_pending_indices is not None:
        return IterationTransition(
            continue_loop=True,
            action="adaptive_rebatch",
            reset_optimizer=True,
            save_latest=True,
            save_best=False,
            adaptive_rebatch_indices=adaptive_rebatch_pending_indices,
        )

    if lbfgsb_loss_schedule_next_index is not None:
        return IterationTransition(
            continue_loop=True,
            action="lbfgsb_loss_schedule",
            reset_optimizer=False,
            save_latest=False,
            save_best=False,
            lbfgsb_loss_schedule_next_index=lbfgsb_loss_schedule_next_index,
        )

    if projected_lbfgs_min_lr_reached:
        return IterationTransition(
            status={"status": "not_converged", "reason": "projected_lbfgs_min_lr_reached"},
            break_loop=True,
            action="projected_lbfgs_min_lr_reached",
            reset_optimizer=False,
            save_latest=False,
            save_best=False,
        )

    if hessian_sgd_activate_line_search:
        return IterationTransition(
            continue_loop=True,
            reset_optimizer=True,
            save_latest=False,
            save_best=False,
            action="hessian_sgd_line_search",
        )

    if lbfgsb_high_kkt_status is not None:
        if can_lbfgsb_retry:
            return IterationTransition(
                continue_loop=True,
                reset_optimizer=False,
                save_latest=False,
                save_best=False,
                action="lbfgsb_retry",
            )
        return IterationTransition(
            status=lbfgsb_high_kkt_status,
            break_loop=True,
            action="step_stopping",
            reset_optimizer=False,
            save_latest=False,
            save_best=False,
        )

    if step_status is not None and active_objective_scope:
        if batch_state_active_index + 1 < active_batch_count:
            return IterationTransition(
                continue_loop=True,
                reset_optimizer=True,
                save_latest=False,
                save_best=False,
                action="next_batch",
                next_batch_active_index=batch_state_active_index + 1,
            )
        return IterationTransition(
            status=step_status,
            break_loop=True,
            action="step_stopping",
            reset_optimizer=False,
            save_latest=False,
            save_best=False,
        )

    if step_status is not None and not active_objective_scope:
        if can_lbfgsb_retry:
            return IterationTransition(
                continue_loop=True,
                reset_optimizer=False,
                save_latest=False,
                save_best=False,
                action="lbfgsb_retry",
            )
        return IterationTransition(
            status=step_status,
            break_loop=True,
            action="step_stopping",
            reset_optimizer=False,
            save_latest=False,
            save_best=False,
        )

    return IterationTransition(
        continue_loop=False,
        break_loop=False,
        action=None,
        reset_optimizer=False,
        save_latest=False,
        save_best=False,
    )


def execute_iteration_transition(
    *,
    config: RunConfig,
    transition: IterationTransition,
    status: dict[str, str] | None,
    model: GeneReconModel,
    objective_state: Any,
    batch_state: Any,
    restart_state: Any,
    lbfgsb_state: Any,
    adaptive_state: _AdaptiveRebatchState,
    planning_state: _StepPlanningState,
    optimizer: torch.optim.Optimizer | None,
    fd_newton_hessian_state: _FDNewtonHessianState | None,
    hessian_sgd_line_search_active: bool,
    hessian_sgd_low_accept_steps: int,
    resume_info: dict[str, Any],
    step: int,
    phase: str,
    objective: float,
    row_best_nll: float | None,
    row_best_step: int | None,
    row: dict[str, Any],
    checkpoint_status: dict[str, Any],
    solver: SolverStageController,
    batch_final_cache: BatchFinalCache | None,
    batchwise_hessian_sgd: bool,
    best_checkpoint: Path,
    latest_checkpoint: Path,
    log_every: int,
    checkpoint_every: int | None,
    ops: IterationTransitionOps,
) -> IterationTransitionExecution:
    status_out = dict(status) if status is not None else None
    resume_info_out = dict(resume_info)
    planning_state_out = planning_state
    optimizer_out = optimizer
    fd_newton_hessian_state_out = fd_newton_hessian_state
    hessian_sgd_line_search_active_out = hessian_sgd_line_search_active
    hessian_sgd_low_accept_steps_out = hessian_sgd_low_accept_steps

    if transition.status is not None:
        status_out = transition.status

    if transition.action == "adagrad_restart_terminal":
        if step % max(1, log_every) == 0:
            ops.print_progress_row(
                step=step,
                phase=phase,
                row=row,
                objective=objective,
                delta=row.get("delta_likelihood_bits"),
                row_best_nll=row_best_nll,
            )
        if checkpoint_every:
            ops.save_status(
                latest_checkpoint,
                model=model,
                optimizer=optimizer_out,
                step=step,
                next_step=step + 1,
                status=ops.adaptive_checkpoint_status(
                    {**checkpoint_status, **(status_out or {})}
                ),
                row=row,
                optimizer_phase=phase,
            )

    elif transition.action == "adagrad_restart_advance":
        if step % max(1, log_every) == 0:
            ops.print_progress_row(
                step=step,
                phase=phase,
                row=row,
                objective=objective,
                delta=row.get("delta_likelihood_bits"),
                row_best_nll=row_best_nll,
            )
        if checkpoint_every:
            ops.save_status(
                latest_checkpoint,
                model=model,
                optimizer=None,
                step=step,
                next_step=step + 1,
                status=ops.adaptive_checkpoint_status(checkpoint_status),
                row=row,
                optimizer_phase=phase,
            )
        if transition.next_adagrad_phase is not None:
            restart_state.advance(
                index=transition.next_adagrad_phase[0],
                start_step=int(transition.next_adagrad_phase[1]),
            )
        objective_state.reset_tracking()
        optimizer_out = None
        restart_state.active_phase_index = None
        batch_state.optimizer_batch_index = None
        objective_state.reset_tracking()
        resume_info_out = {}
        planning_state_out = replace(
            planning_state_out,
            restart_dynamic_phase_index=restart_state.phase_index,
            restart_dynamic_phase_start_step=restart_state.phase_start_step,
            active_adagrad_restart_phase_index=restart_state.active_phase_index,
            active_optimizer_batch_index=batch_state.optimizer_batch_index,
            previous_objective=objective_state.previous_objective,
            stable_loss_steps=objective_state.stable_loss_steps,
            lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
        )

    elif transition.action == "adaptive_rebatch":
        adaptive_state.batch_plan_generation += 1
        ops.clear_cached_static_states_if_needed(model)
        if transition.adaptive_rebatch_indices is not None:
            model.replan_resident_batches(
                transition.adaptive_rebatch_indices,
            )
        batch_state.active_index = 0
        batch_state.reset_for_batch(warmup=False)
        batch_state.local_step = ops.fd_adam_warmup_steps
        fd_newton_hessian_state_out = None
        hessian_sgd_line_search_active_out = False
        hessian_sgd_low_accept_steps_out = 0
        objective_state.reset_tracking()
        optimizer_out = None
        adaptive_state.last_checked_converged_count = 0
        if batch_final_cache is not None and (
            transition.adaptive_rebatch_indices is not None
        ):
            batch_final_cache.invalidate(transition.adaptive_rebatch_indices)
        solver.configure_active_stage(
            model,
            batch_state.solver_stage,
        )
        if checkpoint_every:
            transition_status = build_batch_transition_checkpoint_status(
                checkpoint_status,
                batch_state,
                previous_objective=None,
                stable_loss_steps=0,
                best_nll_bits=None,
                best_step=None,
            )
            ops.save_status(
                latest_checkpoint,
                model=model,
                optimizer=None,
                step=step,
                next_step=step + 1,
                status=ops.adaptive_checkpoint_status(transition_status),
                row=row,
                optimizer_phase=phase,
            )
        resume_info_out = {}
        planning_state_out = replace(
            planning_state_out,
            active_batch_index=batch_state.active_index,
            active_optimizer_batch_index=None,
            previous_objective=None,
            stable_loss_steps=0,
            lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
        )

    elif transition.action == "lbfgsb_loss_schedule":
        if transition.lbfgsb_loss_schedule_next_index is not None:
            lbfgsb_state.loss_schedule_index = transition.lbfgsb_loss_schedule_next_index
        objective_state.stable_loss_steps = 0
        planning_state_out = replace(
            planning_state_out,
            stable_loss_steps=objective_state.stable_loss_steps,
            lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
            previous_objective=objective_state.previous_objective,
        )
        resume_info_out = {}

    elif transition.action is None:
        pass

    elif transition.action in {
        "nonfinite_parameter_update",
        "adaptive_rebatch_stop",
        "projected_lbfgs_min_lr_reached",
    }:
        pass
    else:
        raise RuntimeError(
            f"Unexpected first-step transition action {transition.action}"
        )

    if status_out is not None and "status" not in status_out:
        status_out = None

    return IterationTransitionExecution(
        status=status_out,
        continue_loop=bool(transition.continue_loop),
        break_loop=bool(transition.break_loop),
        optimizer=optimizer_out,
        fd_newton_hessian_state=fd_newton_hessian_state_out,
        hessian_sgd_line_search_active=hessian_sgd_line_search_active_out,
        hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps_out,
        resume_info=resume_info_out,
        planning_state=planning_state_out,
    )


@dataclass
class IterationStatusTransitionExecution:
    status: dict[str, str] | None
    continue_loop: bool
    break_loop: bool
    optimizer: torch.optim.Optimizer | None
    fd_newton_hessian_state: _FDNewtonHessianState | None
    hessian_sgd_line_search_active: bool
    hessian_sgd_low_accept_steps: int
    resume_info: dict[str, Any]
    planning_state: _StepPlanningState
    current_phase: str


@dataclass
class IterationTransitionContext:
    config: RunConfig
    model: GeneReconModel
    evaluation: Any
    solver: SolverStageController
    objective_state: Any
    batch_state: Any
    restart_state: Any
    lbfgsb_state: Any
    adaptive_state: _AdaptiveRebatchState
    planning_state: _StepPlanningState
    optimizer: torch.optim.Optimizer | None
    fd_newton_hessian_state: _FDNewtonHessianState | None
    hessian_sgd_line_search_active: bool
    hessian_sgd_low_accept_steps: int
    resume_info: dict[str, Any]
    batch_final_cache: BatchFinalCache | None
    solver_stage_scope: bool
    batchwise_hessian_sgd: bool
    global_solver_warmup: bool
    lbfgsb_loss_schedule: tuple[Any, ...]
    current_phase: str
    best_checkpoint: Path
    latest_checkpoint: Path
    checkpoint_every: int | None
    log_every: int
    ops: IterationTransitionOps


@dataclass
class IterationTransitionInputs:
    status: dict[str, str] | None
    step: int
    phase: str
    row: dict[str, Any]
    checkpoint_status: dict[str, Any]
    step_status: dict[str, str] | None
    objective: float
    row_best_nll: float | None
    row_best_step: int | None
    active_objective_scope: bool
    active_batch_count: int
    can_lbfgsb_retry: bool
    lbfgsb_high_kkt_status: dict[str, str] | None
    hessian_sgd_activate_line_search: bool
    projected_lbfgs_min_lr_reached: bool
    adaptive_rebatch_stop: bool
    rejected_nonfinite_parameter_update: bool
    adaptive_rebatch_pending_indices: list[int] | None
    adagrad_restart_terminal_status: dict[str, str] | None
    adagrad_restart_phase_next_index: int | None
    adagrad_restart_phase_next_start_step: int | None
    lbfgsb_loss_schedule_next_index: int | None


def execute_iteration_full_transition(
    *,
    context: IterationTransitionContext,
    inputs: IterationTransitionInputs,
) -> IterationStatusTransitionExecution:
    return _execute_iteration_full_transition(
        config=context.config,
        model=context.model,
        evaluation=context.evaluation,
        solver=context.solver,
        status=inputs.status,
        objective_state=context.objective_state,
        batch_state=context.batch_state,
        restart_state=context.restart_state,
        lbfgsb_state=context.lbfgsb_state,
        adaptive_state=context.adaptive_state,
        planning_state=context.planning_state,
        optimizer=context.optimizer,
        fd_newton_hessian_state=context.fd_newton_hessian_state,
        hessian_sgd_line_search_active=context.hessian_sgd_line_search_active,
        hessian_sgd_low_accept_steps=context.hessian_sgd_low_accept_steps,
        resume_info=context.resume_info,
        batch_final_cache=context.batch_final_cache,
        step=inputs.step,
        phase=inputs.phase,
        row=inputs.row,
        checkpoint_status=inputs.checkpoint_status,
        solver_stage_scope=context.solver_stage_scope,
        batchwise_hessian_sgd=context.batchwise_hessian_sgd,
        global_solver_warmup=context.global_solver_warmup,
        lbfgsb_loss_schedule=context.lbfgsb_loss_schedule,
        current_phase=context.current_phase,
        best_checkpoint=context.best_checkpoint,
        latest_checkpoint=context.latest_checkpoint,
        checkpoint_every=context.checkpoint_every,
        log_every=context.log_every,
        hessian_sgd_activate_line_search=inputs.hessian_sgd_activate_line_search,
        step_status=inputs.step_status,
        can_lbfgsb_retry=inputs.can_lbfgsb_retry,
        active_objective_scope=inputs.active_objective_scope,
        active_batch_count=inputs.active_batch_count,
        objective=inputs.objective,
        row_best_nll=inputs.row_best_nll,
        row_best_step=inputs.row_best_step,
        lbfgsb_high_kkt_status=inputs.lbfgsb_high_kkt_status,
        ops=context.ops,
        adaptive_rebatch_stop=inputs.adaptive_rebatch_stop,
        rejected_nonfinite_parameter_update=inputs.rejected_nonfinite_parameter_update,
        adaptive_rebatch_pending_indices=inputs.adaptive_rebatch_pending_indices,
        adagrad_restart_terminal_status=inputs.adagrad_restart_terminal_status,
        adagrad_restart_phase_next_index=inputs.adagrad_restart_phase_next_index,
        adagrad_restart_phase_next_start_step=inputs.adagrad_restart_phase_next_start_step,
        lbfgsb_loss_schedule_next_index=inputs.lbfgsb_loss_schedule_next_index,
        projected_lbfgs_min_lr_reached=inputs.projected_lbfgs_min_lr_reached,
    )


def apply_iteration_transition(
    *,
    context: IterationTransitionContext,
    inputs: IterationTransitionInputs,
) -> IterationStatusTransitionExecution:
    """Apply a prepared per-iteration transition."""
    return execute_iteration_full_transition(context=context, inputs=inputs)


def execute_step_status_transition(
    *,
    config: RunConfig,
    transition: IterationTransition,
    status: dict[str, str] | None,
    model: GeneReconModel,
    objective_state: Any,
    batch_state: Any,
    lbfgsb_state: Any,
    planning_state: _StepPlanningState,
    optimizer: torch.optim.Optimizer | None,
    fd_newton_hessian_state: _FDNewtonHessianState | None,
    hessian_sgd_line_search_active: bool,
    hessian_sgd_low_accept_steps: int,
    resume_info: dict[str, Any],
    step: int,
    phase: str,
    row: dict[str, Any],
    checkpoint_status: dict[str, Any],
    solver: SolverStageController,
    active_batch_count: int,
    solver_warmup_enabled: bool,
    lbfgsb_loss_schedule: tuple[Any, ...],
    best_checkpoint: Path,
    latest_checkpoint: Path,
    current_phase: str,
    checkpoint_every: int | None,
    ops: IterationTransitionOps,
    adaptive_state: _AdaptiveRebatchState,
) -> IterationStatusTransitionExecution:
    status_out = dict(status) if status is not None else None

    if transition.status is not None:
        status_out = transition.status

    if transition.action is None:
        return IterationStatusTransitionExecution(
            status=status_out,
            continue_loop=False,
            break_loop=False,
            optimizer=optimizer,
            fd_newton_hessian_state=fd_newton_hessian_state,
            hessian_sgd_line_search_active=hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
            resume_info=resume_info,
            planning_state=planning_state,
            current_phase=current_phase,
        )

    if transition.action == "next_batch":
        batch_state.active_index = min(
            transition.next_batch_active_index
            if transition.next_batch_active_index is not None
            else batch_state.active_index + 1,
            active_batch_count - 1,
        )
        batch_state.reset_for_batch(warmup=solver_warmup_enabled)
        optimizer = None
        fd_newton_hessian_state = None
        hessian_sgd_line_search_active = False
        hessian_sgd_low_accept_steps = 0
        objective_state.reset_tracking()
        adaptive_state.last_checked_converged_count = 0

        if checkpoint_every:
            transition_status = build_batch_transition_checkpoint_status(
                checkpoint_status,
                batch_state,
                previous_objective=None,
                stable_loss_steps=0,
                best_nll_bits=None,
                best_step=None,
            )
            ops.save_status(
                latest_checkpoint,
                model=model,
                optimizer=None,
                step=step,
                next_step=step + 1,
                status=ops.adaptive_checkpoint_status(transition_status),
                row=row,
                optimizer_phase=phase,
            )
            ops.clear_cached_solver_runtime_state(model)
            model.select_batch(batch_state.active_index)
            solver.configure_active_stage(
                model,
                batch_state.solver_stage,
            )
            return IterationStatusTransitionExecution(
                status=status_out,
                continue_loop=True,
                break_loop=False,
                optimizer=optimizer,
                fd_newton_hessian_state=fd_newton_hessian_state,
                hessian_sgd_line_search_active=hessian_sgd_line_search_active,
                hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
                resume_info={},
                planning_state=replace(
                    planning_state,
                    active_batch_index=batch_state.active_index,
                    active_optimizer_batch_index=None,
                    current_phase="lbfgsb" if phase == "lbfgsb" else current_phase,
                    previous_objective=objective_state.previous_objective,
                    stable_loss_steps=0,
                ),
                current_phase=current_phase,
            )

        return IterationStatusTransitionExecution(
            status=status_out,
            continue_loop=False,
            break_loop=True,
            optimizer=optimizer,
            fd_newton_hessian_state=fd_newton_hessian_state,
            hessian_sgd_line_search_active=hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
            resume_info={},
            planning_state=replace(
                planning_state,
                active_batch_index=batch_state.active_index,
                active_optimizer_batch_index=None,
                current_phase="lbfgsb" if phase == "lbfgsb" else current_phase,
                previous_objective=objective_state.previous_objective,
                stable_loss_steps=0,
            ),
            current_phase=current_phase,
        )

    if transition.action == "lbfgsb_retry":
        retry_payload = ops.load_checkpoint(best_checkpoint)
        ops.validate_checkpoint_model_compatibility(
            path=best_checkpoint,
            config=config,
            model=model,
            payload=retry_payload,
        )
        retry_phase = retry_payload.get("optimizer_phase")
        if retry_phase == "lbfgsb":
            retry_state = ops.resume_state_from_payload(
                best_checkpoint,
                retry_payload,
            )
            ops.restore_model_theta(model, retry_payload)
            current_phase = "lbfgsb"
            optimizer = ops.make_optimizer(
                config,
                model,
                current_phase,
            )
            batch_state.optimizer_batch_index = None
            restore_info = ops.restore_optimizer_state(
                optimizer,
                retry_payload.get("optimizer_state"),
                current_phase=current_phase,
                checkpoint_phase=retry_phase,
            )
            objective_state.best_nll = retry_state.best_nll
            objective_state.best_step = retry_state.best_step
            objective_state.previous_objective = retry_state.previous_objective
            objective_state.stable_loss_steps = retry_state.stable_loss_steps
            lbfgsb_state.fallback_used_count = retry_state.lbfgsb_fallback_used_count
            lbfgsb_state.loss_schedule_index = min(
                int(retry_state.lbfgsb_loss_schedule_index),
                max(0, len(lbfgsb_loss_schedule) - 1),
            )
            lbfgsb_state.best_retry_count += 1
            model.clear()
            return IterationStatusTransitionExecution(
                status=status_out,
                continue_loop=True,
                break_loop=False,
                optimizer=optimizer,
                fd_newton_hessian_state=None,
                hessian_sgd_line_search_active=False,
                hessian_sgd_low_accept_steps=0,
                resume_info={
                    **restore_info,
                    "optimizer/lbfgsb_best_retry_count": float(
                        lbfgsb_state.best_retry_count
                    ),
                    "optimizer/lbfgsb_best_retry_source_step": float(
                        -1 if retry_state.best_step is None else retry_state.best_step
                    ),
                },
                planning_state=replace(
                    planning_state,
                    current_phase=current_phase,
                    active_optimizer_batch_index=batch_state.optimizer_batch_index,
                    previous_objective=objective_state.previous_objective,
                    stable_loss_steps=objective_state.stable_loss_steps,
                    lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
                ),
                current_phase=current_phase,
            )
        return IterationStatusTransitionExecution(
            status=status_out,
            continue_loop=False,
            break_loop=True,
            optimizer=optimizer,
            fd_newton_hessian_state=fd_newton_hessian_state,
            hessian_sgd_line_search_active=hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
            resume_info=resume_info,
            planning_state=planning_state,
            current_phase=current_phase,
        )

    if transition.action == "step_stopping":
        if checkpoint_every:
            transition_status = {
                **checkpoint_status,
                "active_batch_index": batch_state.active_index,
                "active_solver_stage": batch_state.solver_stage,
                "active_batch_local_step": batch_state.local_step,
            }
            ops.save_status(
                latest_checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                next_step=step + 1,
                status=ops.adaptive_checkpoint_status(transition_status),
                row=row,
                optimizer_phase=phase,
            )
        return IterationStatusTransitionExecution(
            status=status_out,
            continue_loop=False,
            break_loop=True,
            optimizer=optimizer,
            fd_newton_hessian_state=fd_newton_hessian_state,
            hessian_sgd_line_search_active=hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
            resume_info=resume_info,
            planning_state=replace(
                planning_state,
                previous_objective=objective_state.previous_objective,
                stable_loss_steps=objective_state.stable_loss_steps,
                lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
            ),
            current_phase=current_phase,
        )

    raise RuntimeError(
        f"Unexpected step-status transition action {transition.action}"
    )


def execute_iteration_post_step_transition(
    *,
    config: RunConfig,
    model: GeneReconModel,
    evaluation: Any,
    solver: SolverStageController,
    status: dict[str, str] | None,
    objective_state: Any,
    batch_state: Any,
    restart_state: Any,
    lbfgsb_state: Any,
    adaptive_state: _AdaptiveRebatchState,
    planning_state: _StepPlanningState,
    optimizer: torch.optim.Optimizer | None,
    fd_newton_hessian_state: _FDNewtonHessianState | None,
    hessian_sgd_line_search_active: bool,
    hessian_sgd_low_accept_steps: int,
    resume_info: dict[str, Any],
    batch_final_cache: BatchFinalCache | None,
    step: int,
    phase: str,
    row: dict[str, Any],
    checkpoint_status: dict[str, Any],
    solver_stage_scope: bool,
    batchwise_hessian_sgd: bool,
    global_solver_warmup: bool,
    lbfgsb_loss_schedule: tuple[Any, ...],
    current_phase: str,
    best_checkpoint: Path,
    latest_checkpoint: Path,
    checkpoint_every: int | None,
    hessian_sgd_activate_line_search: bool,
    step_status: dict[str, str] | None,
    can_lbfgsb_retry: bool,
    active_objective_scope: bool,
    active_batch_count: int,
    lbfgsb_high_kkt_status: dict[str, str] | None,
    ops: IterationTransitionOps,
) -> IterationStatusTransitionExecution:
    status_out = dict(status) if status is not None else None

    if hessian_sgd_activate_line_search:
        objective_state.reset_tracking()
        batch_state.reset_for_batch(warmup=False)
        optimizer = None
        fd_newton_hessian_state = None
        hessian_sgd_line_search_active = True
        hessian_sgd_low_accept_steps = 0
        resume_info = {}
        planning_state = replace(
            planning_state,
            current_phase=current_phase,
            active_batch_index=batch_state.active_index,
            active_optimizer_batch_index=batch_state.optimizer_batch_index,
            active_adagrad_restart_phase_index=restart_state.active_phase_index,
            previous_objective=objective_state.previous_objective,
            stable_loss_steps=objective_state.stable_loss_steps,
            lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
        )
        return IterationStatusTransitionExecution(
            status=status_out,
            continue_loop=True,
            break_loop=False,
            optimizer=optimizer,
            fd_newton_hessian_state=fd_newton_hessian_state,
            hessian_sgd_line_search_active=hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
            resume_info=resume_info,
            planning_state=planning_state,
            current_phase=current_phase,
        )

    warmup_switch = (
        solver_stage_scope
        and batch_state.solver_stage == "warmup"
        and solver.should_switch_solver_warmup(
            stable_loss_steps=objective_state.stable_loss_steps,
        )
    )
    if (
        step_status is not None
        and solver_stage_scope
        and batch_state.solver_stage == "warmup"
    ):
        active_clade_count = int(
            getattr(
                model.current_batch_metadata,
                "clade_count",
                0,
            )
            or 0
        )
        skip_full_after_warmup = (
            batchwise_hessian_sgd
            and phase == "hessian-sgd"
            and not hessian_sgd_line_search_active
            and active_clade_count
            >= _HESSIAN_SGD_SKIP_FULL_AFTER_WARMUP_MIN_CLADES
        )
        if skip_full_after_warmup:
            cache_skipped_full = solver.active_batch_result_is_canonical_full_solver(
                phase=phase,
                solver_stage="full",
            )
            if cache_skipped_full:
                solver.configure_active_stage(
                    model,
                    "full",
                )
                loss_vec_current, _grad, _metrics = (
                    evaluation.evaluate_active_genewise_vector_grad_at_current_theta(
                        model,
                        solver_stage="full",
                    )
                )
                if (
                    not bool(torch.isfinite(loss_vec_current).all().item())
                    or not _is_finite_tensor(model.theta.grad)
                ):
                    status_out = {
                        "status": "failed",
                        "reason": "nonfinite_objective_or_gradient",
                    }
                    model.clear()
                    return IterationStatusTransitionExecution(
                        status=status_out,
                        continue_loop=False,
                        break_loop=True,
                        optimizer=optimizer,
                        fd_newton_hessian_state=fd_newton_hessian_state,
                        hessian_sgd_line_search_active=hessian_sgd_line_search_active,
                        hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
                        resume_info=resume_info,
                        planning_state=replace(
                            planning_state,
                            previous_objective=objective_state.previous_objective,
                            stable_loss_steps=objective_state.stable_loss_steps,
                            lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
                        ),
                        current_phase=current_phase,
                    )
                if batch_final_cache is not None:
                    batch_final_cache.cache(
                        model=model,
                        loss_vec=loss_vec_current,
                        active_indices=ops.active_batch_indices(model),
                    )
                model.clear()
            warmup_switch = False
        else:
            warmup_switch = True
            step_status = None
    if warmup_switch:
        active_clade_count = int(
            getattr(
                model.current_batch_metadata,
                "clade_count",
                0,
            )
            or 0
        )
        carry_warmup_hessian = (
            batchwise_hessian_sgd
            and phase == "hessian-sgd"
            and not hessian_sgd_line_search_active
            and active_clade_count
            >= _HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
            and fd_newton_hessian_state is not None
        )
        warmup_hessian_state = fd_newton_hessian_state
        batch_state.reset_for_batch(warmup=False)
        fd_newton_hessian_state = None
        hessian_sgd_line_search_active = False
        hessian_sgd_low_accept_steps = 0
        objective_state.reset_tracking()
        optimizer = None
        adaptive_state.last_checked_converged_count = 0
        if global_solver_warmup:
            objective_state.reset_best()
        solver.configure_active_stage(model, batch_state.solver_stage)
        if carry_warmup_hessian and warmup_hessian_state is not None:
            loss_vec_current, _grad, _metrics = (
                evaluation.evaluate_active_genewise_vector_grad_at_current_theta(
                    model,
                    solver_stage=batch_state.solver_stage,
                )
            )
            idx = ops.active_batch_indices(model)
            fd_newton_hessian_state = _FDNewtonHessianState(
                batch_index=int(model.current_batch_index),
                solver_stage=batch_state.solver_stage,
                family_indices=tuple(
                    int(index) for index in model.current_batch_metadata.family_indices
                ),
                hessian=warmup_hessian_state.hessian.detach().clone(),
                active_theta=model.theta.detach().index_select(0, idx).clone(),
                active_grad=model.theta.grad.detach().index_select(0, idx).clone(),
                active_loss=loss_vec_current.detach().index_select(0, idx).clone(),
                updates_since_refresh=warmup_hessian_state.updates_since_refresh,
            )
            model.clear()
        if checkpoint_every:
            transition_status = build_batch_transition_checkpoint_status(
                checkpoint_status,
                batch_state,
                previous_objective=None,
                stable_loss_steps=0,
                best_nll_bits=None,
                best_step=None,
            )
            ops.save_status(
                latest_checkpoint,
                model=model,
                optimizer=None,
                step=step,
                next_step=step + 1,
                status=ops.adaptive_checkpoint_status(transition_status),
                row=row,
                optimizer_phase=phase,
            )
        resume_info = {}
        planning_state = replace(
            planning_state,
            current_phase=current_phase,
            active_batch_index=batch_state.active_index,
            active_optimizer_batch_index=batch_state.optimizer_batch_index,
            active_adagrad_restart_phase_index=restart_state.active_phase_index,
            previous_objective=objective_state.previous_objective,
            stable_loss_steps=objective_state.stable_loss_steps,
            lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
        )
        return IterationStatusTransitionExecution(
            status=status_out,
            continue_loop=True,
            break_loop=False,
            optimizer=optimizer,
            fd_newton_hessian_state=fd_newton_hessian_state,
            hessian_sgd_line_search_active=hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
            resume_info=resume_info,
            planning_state=planning_state,
            current_phase=current_phase,
        )

    if step_status is None:
        return IterationStatusTransitionExecution(
            status=status_out,
            continue_loop=False,
            break_loop=False,
            optimizer=optimizer,
            fd_newton_hessian_state=fd_newton_hessian_state,
            hessian_sgd_line_search_active=hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
            resume_info=resume_info,
            planning_state=planning_state,
            current_phase=current_phase,
        )

    pre_transition = _classify_iteration_transition(
        adaptive_rebatch_stop=False,
        rejected_nonfinite_parameter_update=False,
        adaptive_rebatch_pending_indices=None,
        adagrad_restart_terminal_status=None,
        adagrad_restart_phase_next_index=None,
        adagrad_restart_phase_next_start_step=None,
        lbfgsb_loss_schedule_next_index=None,
        lbfgsb_high_kkt_status=lbfgsb_high_kkt_status,
        projected_lbfgs_min_lr_reached=False,
        hessian_sgd_activate_line_search=False,
        step_status=step_status,
        active_objective_scope=active_objective_scope,
        active_batch_count=active_batch_count,
        batch_state_active_index=batch_state.active_index,
        can_lbfgsb_retry=can_lbfgsb_retry,
    )
    transition_result = execute_step_status_transition(
        config=config,
        transition=pre_transition,
        status=status_out,
        model=model,
        objective_state=objective_state,
        batch_state=batch_state,
        lbfgsb_state=lbfgsb_state,
        planning_state=planning_state,
        optimizer=optimizer,
        fd_newton_hessian_state=fd_newton_hessian_state,
        hessian_sgd_line_search_active=hessian_sgd_line_search_active,
        hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
        resume_info=resume_info,
        step=step,
        phase=phase,
        row=row,
        checkpoint_status=checkpoint_status,
        solver=solver,
        active_batch_count=active_batch_count,
        solver_warmup_enabled=(
            global_solver_warmup
            or (active_objective_scope and solver.uses_warmup())
        ),
        lbfgsb_loss_schedule=lbfgsb_loss_schedule,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        current_phase=current_phase,
        checkpoint_every=checkpoint_every,
        ops=ops,
        adaptive_state=adaptive_state,
    )
    return IterationStatusTransitionExecution(
        status=transition_result.status,
        continue_loop=transition_result.continue_loop,
        break_loop=transition_result.break_loop,
        optimizer=transition_result.optimizer,
        fd_newton_hessian_state=transition_result.fd_newton_hessian_state,
        hessian_sgd_line_search_active=transition_result.hessian_sgd_line_search_active,
        hessian_sgd_low_accept_steps=transition_result.hessian_sgd_low_accept_steps,
        resume_info=transition_result.resume_info,
        planning_state=transition_result.planning_state,
        current_phase=transition_result.current_phase,
    )


def _execute_iteration_full_transition(
    *,
    config: RunConfig,
    model: GeneReconModel,
    evaluation: Any,
    solver: SolverStageController,
    status: dict[str, str] | None,
    objective_state: Any,
    batch_state: Any,
    restart_state: Any,
    lbfgsb_state: Any,
    adaptive_state: _AdaptiveRebatchState,
    planning_state: _StepPlanningState,
    optimizer: torch.optim.Optimizer | None,
    fd_newton_hessian_state: _FDNewtonHessianState | None,
    hessian_sgd_line_search_active: bool,
    hessian_sgd_low_accept_steps: int,
    resume_info: dict[str, Any],
    batch_final_cache: BatchFinalCache | None,
    step: int,
    phase: str,
    row: dict[str, Any],
    checkpoint_status: dict[str, Any],
    solver_stage_scope: bool,
    batchwise_hessian_sgd: bool,
    global_solver_warmup: bool,
    lbfgsb_loss_schedule: tuple[Any, ...],
    current_phase: str,
    best_checkpoint: Path,
    latest_checkpoint: Path,
    checkpoint_every: int | None,
    log_every: int,
    hessian_sgd_activate_line_search: bool,
    step_status: dict[str, str] | None,
    can_lbfgsb_retry: bool,
    active_objective_scope: bool,
    active_batch_count: int,
    objective: float,
    row_best_nll: float | None,
    row_best_step: int | None,
    lbfgsb_high_kkt_status: dict[str, str] | None,
    ops: IterationTransitionOps,
    # first-step transition inputs
    adaptive_rebatch_stop: bool,
    rejected_nonfinite_parameter_update: bool,
    adaptive_rebatch_pending_indices: list[int] | None,
    adagrad_restart_terminal_status: dict[str, str] | None,
    adagrad_restart_phase_next_index: int | None,
    adagrad_restart_phase_next_start_step: int | None,
    lbfgsb_loss_schedule_next_index: int | None,
    projected_lbfgs_min_lr_reached: bool,
) -> IterationStatusTransitionExecution:
    pre_transition = _classify_iteration_transition(
        adaptive_rebatch_stop=adaptive_rebatch_stop,
        rejected_nonfinite_parameter_update=rejected_nonfinite_parameter_update,
        adaptive_rebatch_pending_indices=adaptive_rebatch_pending_indices,
        adagrad_restart_terminal_status=adagrad_restart_terminal_status,
        adagrad_restart_phase_next_index=adagrad_restart_phase_next_index,
        adagrad_restart_phase_next_start_step=adagrad_restart_phase_next_start_step,
        lbfgsb_loss_schedule_next_index=lbfgsb_loss_schedule_next_index,
        lbfgsb_high_kkt_status=None,
        projected_lbfgs_min_lr_reached=projected_lbfgs_min_lr_reached,
        hessian_sgd_activate_line_search=False,
        step_status=None,
        active_objective_scope=active_objective_scope,
        active_batch_count=active_batch_count,
        batch_state_active_index=batch_state.active_index,
        can_lbfgsb_retry=can_lbfgsb_retry,
    )

    transition_result = execute_iteration_transition(
        config=config,
        transition=pre_transition,
        status=status,
        model=model,
        objective_state=objective_state,
        batch_state=batch_state,
        restart_state=restart_state,
        lbfgsb_state=lbfgsb_state,
        adaptive_state=adaptive_state,
        planning_state=planning_state,
        optimizer=optimizer,
        fd_newton_hessian_state=fd_newton_hessian_state,
        hessian_sgd_line_search_active=hessian_sgd_line_search_active,
        hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
        resume_info=resume_info,
        step=step,
        phase=phase,
        objective=objective,
        row_best_nll=row_best_nll,
        row_best_step=row_best_step,
        row=row,
        checkpoint_status=checkpoint_status,
        solver=solver,
        batch_final_cache=batch_final_cache,
        batchwise_hessian_sgd=batchwise_hessian_sgd,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        ops=ops,
    )

    # Run post-step transition logic only when no early decision was reached
    # by the pre-step transition handler.
    if transition_result.break_loop or transition_result.continue_loop:
        return IterationStatusTransitionExecution(
            status=transition_result.status,
            continue_loop=transition_result.continue_loop,
            break_loop=transition_result.break_loop,
            optimizer=transition_result.optimizer,
            fd_newton_hessian_state=transition_result.fd_newton_hessian_state,
            hessian_sgd_line_search_active=transition_result.hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=transition_result.hessian_sgd_low_accept_steps,
            resume_info=transition_result.resume_info,
            planning_state=transition_result.planning_state,
            current_phase=current_phase,
        )

    return execute_iteration_post_step_transition(
        config=config,
        model=model,
        evaluation=evaluation,
        solver=solver,
        status=transition_result.status,
        objective_state=objective_state,
        batch_state=batch_state,
        restart_state=restart_state,
        lbfgsb_state=lbfgsb_state,
        adaptive_state=adaptive_state,
        planning_state=transition_result.planning_state,
        optimizer=transition_result.optimizer,
        fd_newton_hessian_state=transition_result.fd_newton_hessian_state,
        hessian_sgd_line_search_active=transition_result.hessian_sgd_line_search_active,
        hessian_sgd_low_accept_steps=transition_result.hessian_sgd_low_accept_steps,
        resume_info=transition_result.resume_info,
        batch_final_cache=batch_final_cache,
        step=step,
        phase=phase,
        row=row,
        checkpoint_status=checkpoint_status,
        solver_stage_scope=solver_stage_scope,
        batchwise_hessian_sgd=batchwise_hessian_sgd,
        global_solver_warmup=global_solver_warmup,
        lbfgsb_loss_schedule=lbfgsb_loss_schedule,
        current_phase=current_phase,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        checkpoint_every=checkpoint_every,
        hessian_sgd_activate_line_search=hessian_sgd_activate_line_search,
        step_status=step_status,
        can_lbfgsb_retry=can_lbfgsb_retry,
        active_objective_scope=active_objective_scope,
        active_batch_count=active_batch_count,
        lbfgsb_high_kkt_status=lbfgsb_high_kkt_status,
        ops=ops,
    )

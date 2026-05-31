from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from gpurec.api.model import GeneReconModel

from ._adaptive_rebatch import _AdaptiveRebatchState
from ._batch_final_cache import BatchFinalCache
from ._fd_newton import _FDNewtonHessianState
from ._solver_stage import SolverStageController
from ._step_plan import _StepPlanningState
from ._transition_types import (
    IterationTransition,
    IterationTransitionExecution,
    IterationTransitionOps,
)


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


def execute_iteration_transition(
    *,
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
    row: dict[str, Any],
    checkpoint_status: dict[str, Any],
    solver: SolverStageController,
    batch_final_cache: BatchFinalCache | None,
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


__all__ = [
    "build_batch_transition_checkpoint_status",
    "execute_iteration_transition",
]

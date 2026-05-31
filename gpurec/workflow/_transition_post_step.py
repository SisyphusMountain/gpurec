from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from gpurec.api.model import GeneReconModel

from ._adaptive_rebatch import _AdaptiveRebatchState
from ._batch_final_cache import BatchFinalCache
from ._fd_newton import _FDNewtonHessianState
from ._hessian_sgd_policy import (
    hessian_sgd_active_clade_count,
    hessian_sgd_should_carry_warmup_hessian,
    hessian_sgd_should_skip_full_after_warmup,
)
from ._runtime_helpers import _is_finite_tensor
from ._solver_stage import SolverStageController
from ._step_plan import _StepPlanningState
from ._transition_policy import _classify_iteration_transition
from ._transition_pre_step import build_batch_transition_checkpoint_status
from ._transition_types import (
    IterationStatusTransitionExecution,
    IterationTransition,
    IterationTransitionOps,
)
from .config import RunConfig


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

    def execution(
        *,
        continue_loop: bool,
        break_loop: bool,
    ) -> IterationStatusTransitionExecution:
        return IterationStatusTransitionExecution(
            status=status_out,
            continue_loop=continue_loop,
            break_loop=break_loop,
            optimizer=optimizer,
            fd_newton_hessian_state=fd_newton_hessian_state,
            hessian_sgd_line_search_active=hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
            resume_info=resume_info,
            planning_state=planning_state,
            current_phase=current_phase,
        )

    if transition.status is not None:
        status_out = transition.status

    if transition.action is None:
        return execution(continue_loop=False, break_loop=False)

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

        continue_loop = False
        break_loop = True
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
            continue_loop = True
            break_loop = False

        resume_info = {}
        planning_state = replace(
            planning_state,
            active_batch_index=batch_state.active_index,
            active_optimizer_batch_index=None,
            current_phase="lbfgsb" if phase == "lbfgsb" else current_phase,
            previous_objective=objective_state.previous_objective,
            stable_loss_steps=0,
        )
        return execution(continue_loop=continue_loop, break_loop=break_loop)

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
            fd_newton_hessian_state = None
            hessian_sgd_line_search_active = False
            hessian_sgd_low_accept_steps = 0
            resume_info = {
                **restore_info,
                "optimizer/lbfgsb_best_retry_count": float(
                    lbfgsb_state.best_retry_count
                ),
                "optimizer/lbfgsb_best_retry_source_step": float(
                    -1 if retry_state.best_step is None else retry_state.best_step
                ),
            }
            planning_state = replace(
                planning_state,
                current_phase=current_phase,
                active_optimizer_batch_index=batch_state.optimizer_batch_index,
                previous_objective=objective_state.previous_objective,
                stable_loss_steps=objective_state.stable_loss_steps,
                lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
            )
            return execution(continue_loop=True, break_loop=False)
        return execution(continue_loop=False, break_loop=True)

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
        planning_state = replace(
            planning_state,
            previous_objective=objective_state.previous_objective,
            stable_loss_steps=objective_state.stable_loss_steps,
            lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
        )
        return execution(continue_loop=False, break_loop=True)

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

    def execution(
        *,
        continue_loop: bool,
        break_loop: bool,
    ) -> IterationStatusTransitionExecution:
        return IterationStatusTransitionExecution(
            status=status_out,
            continue_loop=continue_loop,
            break_loop=break_loop,
            optimizer=optimizer,
            fd_newton_hessian_state=fd_newton_hessian_state,
            hessian_sgd_line_search_active=hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=hessian_sgd_low_accept_steps,
            resume_info=resume_info,
            planning_state=planning_state,
            current_phase=current_phase,
        )

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
        return execution(continue_loop=True, break_loop=False)

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
        skip_full_after_warmup = hessian_sgd_should_skip_full_after_warmup(
            batchwise_hessian_sgd=batchwise_hessian_sgd,
            phase=phase,
            line_search_active=hessian_sgd_line_search_active,
            active_clade_count=hessian_sgd_active_clade_count(
                model.current_batch_metadata
            ),
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
                    planning_state = replace(
                        planning_state,
                        previous_objective=objective_state.previous_objective,
                        stable_loss_steps=objective_state.stable_loss_steps,
                        lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
                    )
                    return execution(continue_loop=False, break_loop=True)
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
        carry_warmup_hessian = hessian_sgd_should_carry_warmup_hessian(
            batchwise_hessian_sgd=batchwise_hessian_sgd,
            phase=phase,
            line_search_active=hessian_sgd_line_search_active,
            active_clade_count=hessian_sgd_active_clade_count(
                model.current_batch_metadata
            ),
            has_hessian_state=fd_newton_hessian_state is not None,
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
        return execution(continue_loop=True, break_loop=False)

    if step_status is None:
        return execution(continue_loop=False, break_loop=False)

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
    return execute_step_status_transition(
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


__all__ = [
    "execute_iteration_post_step_transition",
    "execute_step_status_transition",
]

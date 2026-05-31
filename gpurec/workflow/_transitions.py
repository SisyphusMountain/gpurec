from __future__ import annotations

from ._transition_policy import _classify_iteration_transition
from ._transition_post_step import (
    execute_iteration_post_step_transition,
    execute_step_status_transition,
)
from ._transition_pre_step import (
    build_batch_transition_checkpoint_status,
    execute_iteration_transition,
)
from ._transition_types import (
    IterationStatusTransitionExecution,
    IterationTransition,
    IterationTransitionContext,
    IterationTransitionExecution,
    IterationTransitionInputs,
    IterationTransitionOps,
)


def execute_iteration_full_transition(
    *,
    context: IterationTransitionContext,
    inputs: IterationTransitionInputs,
) -> IterationStatusTransitionExecution:
    pre_transition = _classify_iteration_transition(
        adaptive_rebatch_stop=inputs.adaptive_rebatch_stop,
        rejected_nonfinite_parameter_update=(
            inputs.rejected_nonfinite_parameter_update
        ),
        adaptive_rebatch_pending_indices=inputs.adaptive_rebatch_pending_indices,
        adagrad_restart_terminal_status=inputs.adagrad_restart_terminal_status,
        adagrad_restart_phase_next_index=inputs.adagrad_restart_phase_next_index,
        adagrad_restart_phase_next_start_step=(
            inputs.adagrad_restart_phase_next_start_step
        ),
        lbfgsb_loss_schedule_next_index=inputs.lbfgsb_loss_schedule_next_index,
        lbfgsb_high_kkt_status=None,
        projected_lbfgs_min_lr_reached=inputs.projected_lbfgs_min_lr_reached,
        hessian_sgd_activate_line_search=False,
        step_status=None,
        active_objective_scope=inputs.active_objective_scope,
        active_batch_count=inputs.active_batch_count,
        batch_state_active_index=context.batch_state.active_index,
        can_lbfgsb_retry=inputs.can_lbfgsb_retry,
    )

    transition_result = execute_iteration_transition(
        transition=pre_transition,
        status=inputs.status,
        model=context.model,
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
        step=inputs.step,
        phase=inputs.phase,
        objective=inputs.objective,
        row_best_nll=inputs.row_best_nll,
        row=inputs.row,
        checkpoint_status=inputs.checkpoint_status,
        solver=context.solver,
        batch_final_cache=context.batch_final_cache,
        latest_checkpoint=context.latest_checkpoint,
        log_every=context.log_every,
        checkpoint_every=context.checkpoint_every,
        ops=context.ops,
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
            hessian_sgd_low_accept_steps=(
                transition_result.hessian_sgd_low_accept_steps
            ),
            resume_info=transition_result.resume_info,
            planning_state=transition_result.planning_state,
            current_phase=context.current_phase,
        )

    return execute_iteration_post_step_transition(
        config=context.config,
        model=context.model,
        evaluation=context.evaluation,
        solver=context.solver,
        status=transition_result.status,
        objective_state=context.objective_state,
        batch_state=context.batch_state,
        restart_state=context.restart_state,
        lbfgsb_state=context.lbfgsb_state,
        adaptive_state=context.adaptive_state,
        planning_state=transition_result.planning_state,
        optimizer=transition_result.optimizer,
        fd_newton_hessian_state=transition_result.fd_newton_hessian_state,
        hessian_sgd_line_search_active=transition_result.hessian_sgd_line_search_active,
        hessian_sgd_low_accept_steps=transition_result.hessian_sgd_low_accept_steps,
        resume_info=transition_result.resume_info,
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
        hessian_sgd_activate_line_search=inputs.hessian_sgd_activate_line_search,
        step_status=inputs.step_status,
        can_lbfgsb_retry=inputs.can_lbfgsb_retry,
        active_objective_scope=inputs.active_objective_scope,
        active_batch_count=inputs.active_batch_count,
        lbfgsb_high_kkt_status=inputs.lbfgsb_high_kkt_status,
        ops=context.ops,
    )


def apply_iteration_transition(
    *,
    context: IterationTransitionContext,
    inputs: IterationTransitionInputs,
) -> IterationStatusTransitionExecution:
    """Apply a prepared per-iteration transition."""
    return execute_iteration_full_transition(context=context, inputs=inputs)


__all__ = [
    "IterationStatusTransitionExecution",
    "IterationTransition",
    "IterationTransitionContext",
    "IterationTransitionExecution",
    "IterationTransitionInputs",
    "IterationTransitionOps",
    "_classify_iteration_transition",
    "apply_iteration_transition",
    "build_batch_transition_checkpoint_status",
    "execute_iteration_full_transition",
    "execute_iteration_post_step_transition",
    "execute_iteration_transition",
    "execute_step_status_transition",
]

"""Private workflow transition classification policy.

This module holds pure per-iteration transition classification helpers for
workflow orchestration. It is not a public workflow API surface, and execution
side effects stay in ``gpurec.workflow._transitions``.
"""

from __future__ import annotations

from ._transition_types import IterationTransition


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
            status={
                "status": "not_converged",
                "reason": "projected_lbfgs_min_lr_reached",
            },
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


__all__ = ["_classify_iteration_transition"]

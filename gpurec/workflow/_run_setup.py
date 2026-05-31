"""Pure setup values derived before a workflow optimization loop."""

from __future__ import annotations

from dataclasses import dataclass

from ._phase import (
    _continues_after_adagrad_restart_prefix,
    _uses_adagrad_restart_prefix,
)
from .config import (
    AdagradRestartPhase,
    LossStopPhase,
    RunConfig,
    adagrad_restart_schedule_specs,
    adagrad_restart_schedule_total_steps,
    loss_stop_schedule_specs,
)


@dataclass(frozen=True)
class _WorkflowRunSetup:
    adagrad_restart_specs: tuple[AdagradRestartPhase, ...]
    adagrad_restart_step_limit: int | None
    lbfgsb_loss_schedule: tuple[LossStopPhase, ...]
    adagrad_restart_dynamic_enabled: bool
    batchwise_active_optimizer: bool
    batchwise_batched_lbfgs: bool
    batchwise_fd_newton: bool
    batchwise_hessian_sgd: bool

    def optimization_stop_step(self, config: RunConfig) -> int:
        if (
            self.adagrad_restart_step_limit is not None
            and not _continues_after_adagrad_restart_prefix(config.optimizer)
        ):
            return min(config.steps, self.adagrad_restart_step_limit)
        return config.steps


def _derive_workflow_run_setup(
    config: RunConfig,
    *,
    batchwise_active_optimizer_phases: frozenset[str],
) -> _WorkflowRunSetup:
    adagrad_restart_specs: tuple[AdagradRestartPhase, ...] = ()
    adagrad_restart_step_limit: int | None = None
    if _uses_adagrad_restart_prefix(config.optimizer):
        adagrad_restart_specs = adagrad_restart_schedule_specs(
            config.adagrad_restart_schedule,
        )
        adagrad_restart_step_limit = adagrad_restart_schedule_total_steps(
            config.adagrad_restart_schedule,
        )

    lbfgsb_loss_schedule: tuple[LossStopPhase, ...] = (
        loss_stop_schedule_specs(config.lbfgsb_loss_change_tol_schedule)
        if config.lbfgsb_loss_change_tol_schedule is not None
        else ()
    )
    adagrad_restart_dynamic_enabled = (
        _uses_adagrad_restart_prefix(config.optimizer)
        and config.adagrad_restart_phase_loss_patience > 0
    )
    batchwise_active_optimizer = (
        config.mode == "genewise"
        and config.optimizer in batchwise_active_optimizer_phases
    )
    batchwise_batched_lbfgs = (
        config.mode == "genewise" and config.optimizer == "batched-lbfgs"
    )
    batchwise_fd_newton = (
        config.mode == "genewise" and config.optimizer == "adam-fd-newton"
    )
    batchwise_hessian_sgd = (
        config.mode == "genewise" and config.optimizer == "hessian-sgd"
    )

    return _WorkflowRunSetup(
        adagrad_restart_specs=adagrad_restart_specs,
        adagrad_restart_step_limit=adagrad_restart_step_limit,
        lbfgsb_loss_schedule=lbfgsb_loss_schedule,
        adagrad_restart_dynamic_enabled=adagrad_restart_dynamic_enabled,
        batchwise_active_optimizer=batchwise_active_optimizer,
        batchwise_batched_lbfgs=batchwise_batched_lbfgs,
        batchwise_fd_newton=batchwise_fd_newton,
        batchwise_hessian_sgd=batchwise_hessian_sgd,
    )

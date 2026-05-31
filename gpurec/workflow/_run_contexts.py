"""Immutable workflow contexts assembled before the optimization loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from gpurec.api.model import GeneReconModel

from ._hessian_sgd_policy import (
    HESSIAN_SGD_LINE_SEARCH_MAX_STEPS,
    HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES,
    HESSIAN_SGD_NO_LINE_REFRESH_STEPS,
)
from ._loop_policies import _LoopPolicyContext
from ._rows import _IterationArtifactsContext
from ._run_setup import _WorkflowRunSetup
from ._runtime_helpers import _clear_cached_solver_runtime_state
from ._step_execution import _StepExecutionContext
from ._step_plan import _StepPlanningContext
from .config import RunConfig


@dataclass(frozen=True)
class _WorkflowRunContexts:
    planning_context: _StepPlanningContext
    step_execution_context: _StepExecutionContext
    iteration_artifacts_context: _IterationArtifactsContext
    loop_policy_context: _LoopPolicyContext
    solver_warmup_enabled: bool
    global_solver_warmup: bool


def _build_workflow_run_contexts(
    config: RunConfig,
    *,
    run_setup: _WorkflowRunSetup,
    solver: Any,
    evaluation: Any,
    batchwise_active_optimizer_phases: frozenset[str],
    make_optimizer: Callable[[GeneReconModel, str], torch.optim.Optimizer],
    active_fd_newton_step: Callable[..., Any],
) -> _WorkflowRunContexts:
    solver_warmup_enabled = bool(solver.uses_warmup())
    global_solver_warmup = (
        solver_warmup_enabled and not run_setup.batchwise_active_optimizer
    )
    planning_context = _StepPlanningContext(
        solver=solver,
        config=config,
        adagrad_restart_specs=run_setup.adagrad_restart_specs,
        adagrad_restart_step_limit=run_setup.adagrad_restart_step_limit,
        adagrad_restart_dynamic_enabled=run_setup.adagrad_restart_dynamic_enabled,
        adagrad_restart_dynamic_state_loaded=False,
        batchwise_active_optimizer=run_setup.batchwise_active_optimizer,
        batchwise_active_optimizer_phases=batchwise_active_optimizer_phases,
        batchwise_batched_lbfgs=run_setup.batchwise_batched_lbfgs,
        batchwise_fd_newton=run_setup.batchwise_fd_newton,
        batchwise_hessian_sgd=run_setup.batchwise_hessian_sgd,
        clear_cached_solver_runtime_state=_clear_cached_solver_runtime_state,
        make_optimizer=make_optimizer,
    )
    step_execution_context = _StepExecutionContext(
        config=config,
        evaluation=evaluation,
        solver=solver,
        batchwise_active_optimizer=run_setup.batchwise_active_optimizer,
        fd_adam_warmup_steps=config.fd_adam_warmup_steps,
        hessian_sgd_no_line_refresh_min_clades=(
            HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
        ),
        hessian_sgd_no_line_refresh_steps=HESSIAN_SGD_NO_LINE_REFRESH_STEPS,
        hessian_sgd_line_search_max_steps=HESSIAN_SGD_LINE_SEARCH_MAX_STEPS,
        active_fd_newton_step=active_fd_newton_step,
    )
    iteration_artifacts_context = _IterationArtifactsContext(
        active_objective_scope=run_setup.batchwise_active_optimizer,
        global_solver_warmup=global_solver_warmup,
        adagrad_restart_dynamic_enabled=run_setup.adagrad_restart_dynamic_enabled,
        lbfgsb_loss_schedule=run_setup.lbfgsb_loss_schedule,
    )
    loop_policy_context = _LoopPolicyContext(
        config=config,
        batchwise_active_optimizer=run_setup.batchwise_active_optimizer,
        batchwise_active_optimizer_phases=batchwise_active_optimizer_phases,
        global_solver_warmup=global_solver_warmup,
        adagrad_restart_dynamic_enabled=run_setup.adagrad_restart_dynamic_enabled,
        adagrad_restart_specs=run_setup.adagrad_restart_specs,
        lbfgsb_loss_schedule=run_setup.lbfgsb_loss_schedule,
    )
    return _WorkflowRunContexts(
        planning_context=planning_context,
        step_execution_context=step_execution_context,
        iteration_artifacts_context=iteration_artifacts_context,
        loop_policy_context=loop_policy_context,
        solver_warmup_enabled=solver_warmup_enabled,
        global_solver_warmup=global_solver_warmup,
    )

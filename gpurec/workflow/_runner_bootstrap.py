"""Private workflow runner bootstrap assembly.

This module gathers the pre-loop optimization state and callback wiring used by
``OptimizationRunner``. It is not a public workflow API surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from gpurec.api.model import GeneReconModel

from ._adaptive_rebatch import _AdaptiveRebatchState
from ._loop_policies import _LoopPolicyContext, _LoopPolicyState
from ._rows import _IterationArtifactsContext
from ._run_contexts import _build_workflow_run_contexts
from ._run_setup import _WorkflowRunSetup, _derive_workflow_run_setup
from ._run_state import (
    BatchRunState,
    LBFGSBRunState,
    ObjectiveState,
    RestartRunState,
    _OptimizationRunState,
)
from ._step_execution import _StepExecutionContext
from ._step_plan import (
    _StepPlanningContext,
    _StepPlanningState,
    prepare_initial_optimization_plan,
)
from ._transition_types import (
    IterationTransitionContext,
    IterationTransitionOps,
)
from .config import RunConfig


@dataclass
class _WorkflowRunnerBootstrap:
    run_setup: _WorkflowRunSetup
    planning_context: _StepPlanningContext
    step_execution_context: _StepExecutionContext
    iteration_artifacts_context: _IterationArtifactsContext
    loop_policy_context: _LoopPolicyContext
    loop_policy_state: _LoopPolicyState
    run_state: _OptimizationRunState
    adaptive_state: _AdaptiveRebatchState
    best_checkpoint: Path
    latest_checkpoint: Path
    optimization_stop_step: int
    transition_context: IterationTransitionContext
    current_phase: str
    solver_stage_scope: bool
    solver_warmup_enabled: bool
    global_solver_warmup: bool
    adaptive_rebatch_enabled: bool
    batchwise_active_optimizer: bool
    batchwise_hessian_sgd: bool
    print_progress_row: Callable[..., None]


def _print_workflow_progress_row(
    *,
    step: int,
    phase: str,
    row: dict[str, Any],
    objective: float,
    delta: float | None,
    row_best_nll: float | None,
) -> None:
    print(
        f"step={step} phase={phase} "
        f"solver={row.get('optimizer/solver_stage', 'full')} "
        f"nll_bits={objective:.6f} "
        f"grad_inf={row.get('grad/inf', float('nan')):.6g} "
        f"delta={float('nan') if delta is None else delta:.6g} "
        f"best={float('nan') if row_best_nll is None else row_best_nll:.6f} "
        f"step_s={row['step_s']:.3f}",
        flush=True,
    )


def _build_initial_run_state(
    *,
    solver_warmup_enabled: bool,
    adagrad_restart_dynamic_enabled: bool,
) -> _OptimizationRunState:
    batch_state = BatchRunState(
        solver_stage=("warmup" if solver_warmup_enabled else "full"),
    )
    objective_state = ObjectiveState()
    lbfgsb_state = LBFGSBRunState()
    restart_state = RestartRunState(
        dynamic_enabled=adagrad_restart_dynamic_enabled,
        phase_index=0,
        phase_start_step=0,
        active_phase_index=None,
    )
    planning_state = _StepPlanningState(
        restart_dynamic_phase_index=restart_state.phase_index,
        restart_dynamic_phase_start_step=restart_state.phase_start_step,
        current_phase="",
        active_batch_index=batch_state.active_index,
        active_optimizer_batch_index=batch_state.optimizer_batch_index,
        active_adagrad_restart_phase_index=restart_state.active_phase_index,
        previous_objective=objective_state.previous_objective,
        stable_loss_steps=objective_state.stable_loss_steps,
        lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
        optimizer=None,
    )
    return _OptimizationRunState(
        objective_state=objective_state,
        batch_state=batch_state,
        restart_state=restart_state,
        lbfgsb_state=lbfgsb_state,
        planning_state=planning_state,
        current_phase="",
        batch_final_cache=None,
    )


def _configure_initial_batch_scope(
    *,
    model: GeneReconModel,
    solver: Any,
    run_state: _OptimizationRunState,
    batchwise_active_optimizer: bool,
    global_solver_warmup: bool,
    create_batch_final_cache: Callable[[GeneReconModel], Any],
    clear_cached_solver_runtime_state: Callable[[GeneReconModel], None],
) -> bool:
    batch_state = run_state.batch_state
    if batchwise_active_optimizer:
        if batch_state.active_index >= len(model.batch_metadata):
            raise RuntimeError(
                f"checkpoint active batch {batch_state.active_index} exceeds "
                f"{len(model.batch_metadata)} model batches"
            )
        run_state.batch_final_cache = create_batch_final_cache(model)
        if model.current_batch_index != batch_state.active_index:
            clear_cached_solver_runtime_state(model)
        model.select_batch(batch_state.active_index)
        solver.configure_active_stage(
            model,
            batch_state.solver_stage,
        )
    elif global_solver_warmup:
        solver.configure_active_stage(model, batch_state.solver_stage)
    return batchwise_active_optimizer or global_solver_warmup


def _build_transition_ops(
    *,
    config: RunConfig,
    evaluation: Any,
    adaptive_state: _AdaptiveRebatchState,
    make_optimizer: Callable[[GeneReconModel, str], torch.optim.Optimizer],
    restore_optimizer_state: Callable[..., dict[str, Any]],
    save_status: Callable[..., None],
    load_checkpoint: Callable[[Path], dict[str, Any]],
    validate_checkpoint_model_compatibility: Callable[..., None],
    restore_model_theta: Callable[[GeneReconModel, dict[str, Any]], None],
    resume_state_from_payload: Callable[[Path, dict[str, Any]], Any],
    clear_cached_static_states_if_needed: Callable[[GeneReconModel], None],
    clear_cached_solver_runtime_state: Callable[[GeneReconModel], None],
    print_progress_row: Callable[..., None],
) -> IterationTransitionOps:
    return IterationTransitionOps(
        active_batch_indices=evaluation._active_batch_indices,
        clear_cached_static_states_if_needed=clear_cached_static_states_if_needed,
        clear_cached_solver_runtime_state=clear_cached_solver_runtime_state,
        load_checkpoint=load_checkpoint,
        validate_checkpoint_model_compatibility=(
            validate_checkpoint_model_compatibility
        ),
        restore_model_theta=restore_model_theta,
        make_optimizer=lambda _config, model_arg, phase: make_optimizer(
            model_arg,
            phase,
        ),
        restore_optimizer_state=restore_optimizer_state,
        resume_state_from_payload=resume_state_from_payload,
        save_status=save_status,
        adaptive_checkpoint_status=adaptive_state.checkpoint_status,
        print_progress_row=print_progress_row,
        fd_adam_warmup_steps=config.fd_adam_warmup_steps,
    )


def build_workflow_runner_bootstrap(
    config: RunConfig,
    *,
    model: GeneReconModel,
    solver: Any,
    evaluation: Any,
    batchwise_active_optimizer_phases: frozenset[str],
    adaptive_rebatch_min_active_families: int,
    make_optimizer: Callable[[GeneReconModel, str], torch.optim.Optimizer],
    active_fd_newton_step: Callable[..., Any],
    restore_optimizer_state: Callable[..., dict[str, Any]],
    save_status: Callable[..., None],
    load_checkpoint: Callable[[Path], dict[str, Any]],
    validate_checkpoint_model_compatibility: Callable[..., None],
    restore_model_theta: Callable[[GeneReconModel, dict[str, Any]], None],
    apply_resume_checkpoint_state: Callable[..., Any],
    resume_state_from_payload: Callable[[Path, dict[str, Any]], Any],
    create_batch_final_cache: Callable[[GeneReconModel], Any],
    clear_cached_static_states_if_needed: Callable[[GeneReconModel], None],
    clear_cached_solver_runtime_state: Callable[[GeneReconModel], None],
) -> _WorkflowRunnerBootstrap:
    run_setup = _derive_workflow_run_setup(
        config,
        batchwise_active_optimizer_phases=batchwise_active_optimizer_phases,
    )
    run_contexts = _build_workflow_run_contexts(
        config=config,
        run_setup=run_setup,
        solver=solver,
        evaluation=evaluation,
        batchwise_active_optimizer_phases=batchwise_active_optimizer_phases,
        make_optimizer=make_optimizer,
        active_fd_newton_step=active_fd_newton_step,
    )
    planning_context = run_contexts.planning_context
    solver_warmup_enabled = run_contexts.solver_warmup_enabled
    global_solver_warmup = run_contexts.global_solver_warmup
    adaptive_rebatch_enabled = bool(
        config.adaptive_rebatch and run_setup.batchwise_active_optimizer
    )
    run_state = _build_initial_run_state(
        solver_warmup_enabled=solver_warmup_enabled,
        adagrad_restart_dynamic_enabled=run_setup.adagrad_restart_dynamic_enabled,
    )
    adaptive_state = _AdaptiveRebatchState.create(
        enabled=adaptive_rebatch_enabled,
        model=model,
        min_active_families=adaptive_rebatch_min_active_families,
    )
    best_checkpoint = config.out_dir / "checkpoints" / "best.pt"
    latest_checkpoint = config.out_dir / "checkpoints" / "latest.pt"
    loop_policy_state = _LoopPolicyState(
        objective_state=run_state.objective_state,
        batch_state=run_state.batch_state,
        lbfgsb_state=run_state.lbfgsb_state,
    )

    if config.resume_from is not None:
        resume_application = apply_resume_checkpoint_state(
            config=config,
            model=model,
            run_state=run_state,
            planning_context=planning_context,
            lbfgsb_loss_schedule=run_setup.lbfgsb_loss_schedule,
            solver_warmup_enabled=solver_warmup_enabled,
            batchwise_active_optimizer=run_setup.batchwise_active_optimizer,
            adagrad_restart_dynamic_enabled=(
                run_setup.adagrad_restart_dynamic_enabled
            ),
            adaptive_rebatch_enabled=adaptive_rebatch_enabled,
            adaptive_state=adaptive_state,
            load_checkpoint=load_checkpoint,
            validate_checkpoint_model_compatibility=(
                validate_checkpoint_model_compatibility
            ),
            restore_model_theta=restore_model_theta,
        )
        planning_context = resume_application.planning_context
        planning_state = resume_application.planning_state
    else:
        planning_state = run_state.planning_state

    solver_stage_scope = _configure_initial_batch_scope(
        model=model,
        solver=solver,
        run_state=run_state,
        batchwise_active_optimizer=run_setup.batchwise_active_optimizer,
        global_solver_warmup=global_solver_warmup,
        create_batch_final_cache=create_batch_final_cache,
        clear_cached_solver_runtime_state=clear_cached_solver_runtime_state,
    )
    optimization_stop_step = run_setup.optimization_stop_step(config)
    initial_plan = prepare_initial_optimization_plan(
        planning_context,
        planning_state,
        model,
        start_step=run_state.start_step,
        optimization_stop_step=optimization_stop_step,
        resume_payload=run_state.resume_payload,
        restore_optimizer_state=restore_optimizer_state,
    )
    run_state.apply_initial_plan(initial_plan)
    planning_state = run_state.planning_state
    current_phase = run_state.current_phase
    print_progress_row = _print_workflow_progress_row
    transition_ops = _build_transition_ops(
        config=config,
        evaluation=evaluation,
        adaptive_state=adaptive_state,
        make_optimizer=make_optimizer,
        restore_optimizer_state=restore_optimizer_state,
        save_status=save_status,
        load_checkpoint=load_checkpoint,
        validate_checkpoint_model_compatibility=validate_checkpoint_model_compatibility,
        restore_model_theta=restore_model_theta,
        resume_state_from_payload=resume_state_from_payload,
        clear_cached_static_states_if_needed=clear_cached_static_states_if_needed,
        clear_cached_solver_runtime_state=clear_cached_solver_runtime_state,
        print_progress_row=print_progress_row,
    )
    transition_context = run_state.make_transition_context(
        config=config,
        model=model,
        evaluation=evaluation,
        solver=solver,
        adaptive_state=adaptive_state,
        solver_stage_scope=solver_stage_scope,
        batchwise_hessian_sgd=run_setup.batchwise_hessian_sgd,
        global_solver_warmup=global_solver_warmup,
        lbfgsb_loss_schedule=run_setup.lbfgsb_loss_schedule,
        planning_state=planning_state,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        checkpoint_every=config.checkpoint_every,
        log_every=config.log_every,
        ops=transition_ops,
        current_phase=current_phase,
    )
    return _WorkflowRunnerBootstrap(
        run_setup=run_setup,
        planning_context=planning_context,
        step_execution_context=run_contexts.step_execution_context,
        iteration_artifacts_context=run_contexts.iteration_artifacts_context,
        loop_policy_context=run_contexts.loop_policy_context,
        loop_policy_state=loop_policy_state,
        run_state=run_state,
        adaptive_state=adaptive_state,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        optimization_stop_step=optimization_stop_step,
        transition_context=transition_context,
        current_phase=current_phase,
        solver_stage_scope=solver_stage_scope,
        solver_warmup_enabled=solver_warmup_enabled,
        global_solver_warmup=global_solver_warmup,
        adaptive_rebatch_enabled=adaptive_rebatch_enabled,
        batchwise_active_optimizer=run_setup.batchwise_active_optimizer,
        batchwise_hessian_sgd=run_setup.batchwise_hessian_sgd,
        print_progress_row=print_progress_row,
    )

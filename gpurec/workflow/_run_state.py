"""Internal optimizer run-state plumbing for workflow optimization.

This module holds private workflow state containers used by
``OptimizationRunner`` and is not a public workflow API surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ._rows import _IterationArtifactsInputs, _IterationArtifactsState
from ._transitions import IterationTransitionContext

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from gpurec.api.model import GeneReconModel

    from ._adaptive_rebatch import _AdaptiveRebatchState
    from ._batch_final_cache import BatchFinalCache
    from ._fd_newton import _FDNewtonHessianState
    from ._solver_stage import SolverStageController
    from ._step_plan import (
        _InitialOptimizationPlan,
        _StepIterationPlan,
        _StepPlanningState,
    )
    from ._transitions import IterationTransitionOps
    from .config import RunConfig


@dataclass
class ObjectiveState:
    best_nll: float | None = None
    best_step: int | None = None
    previous_objective: float | None = None
    stable_loss_steps: int = 0

    def reset_tracking(self) -> None:
        self.previous_objective = None
        self.stable_loss_steps = 0

    def reset_best(self) -> None:
        self.best_nll = None
        self.best_step = None

    def update_best(
        self,
        objective: float,
        step: int,
        best_likelihood_min_delta_bits: float,
    ) -> tuple[float | None, int | None, bool]:
        improved = (
            self.best_nll is None
            or objective < self.best_nll - best_likelihood_min_delta_bits
        )
        if improved:
            self.best_nll = objective
            self.best_step = step
        return self.best_nll, self.best_step, improved


@dataclass
class BatchRunState:
    active_index: int = 0
    local_step: int = 0
    solver_stage: str = "full"
    best_nll: float | None = None
    best_step: int | None = None
    optimizer_batch_index: int | None = None

    def reset_for_batch(self, *, warmup: bool) -> None:
        self.local_step = 0
        self.solver_stage = "warmup" if warmup else "full"
        self.best_nll = None
        self.best_step = None
        self.optimizer_batch_index = None

    def update_best(
        self,
        objective: float,
        step: int,
        best_likelihood_min_delta_bits: float,
    ) -> tuple[float | None, int | None, bool]:
        improved = (
            self.best_nll is None
            or objective < self.best_nll - best_likelihood_min_delta_bits
        )
        if improved:
            self.best_nll = objective
            self.best_step = step
        return self.best_nll, self.best_step, improved


@dataclass
class RestartRunState:
    dynamic_enabled: bool = False
    phase_index: int = 0
    phase_start_step: int = 0
    active_phase_index: int | None = None

    def advance(self, index: int, start_step: int) -> None:
        self.phase_index = index
        self.phase_start_step = start_step
        self.active_phase_index = None


@dataclass
class LBFGSBRunState:
    fallback_used_count: int = 0
    loss_schedule_index: int = 0
    best_retry_count: int = 0


@dataclass
class _OptimizationRunState:
    objective_state: ObjectiveState
    batch_state: BatchRunState
    restart_state: RestartRunState
    lbfgsb_state: LBFGSBRunState
    planning_state: _StepPlanningState
    current_phase: str
    optimizer: torch.optim.Optimizer | None = None
    resume_payload: dict[str, Any] | None = None
    resume_info: dict[str, Any] = field(default_factory=dict)
    fd_newton_hessian_state: _FDNewtonHessianState | None = None
    hessian_sgd_line_search_active: bool = False
    hessian_sgd_low_accept_steps: int = 0
    batch_final_cache: BatchFinalCache | None = None
    status: dict[str, str] = field(
        default_factory=lambda: {"status": "running", "reason": "running"}
    )
    final_row: dict[str, Any] = field(default_factory=dict)
    start_step: int = 0

    def update_planning_state(
        self,
        *,
        current_phase: str,
        optimizer: torch.optim.Optimizer | None = None,
        active_optimizer_batch_index: int | None = None,
        active_adagrad_restart_phase_index: int | None = None,
    ) -> _StepPlanningState:
        self.planning_state = replace(
            self.planning_state,
            optimizer=optimizer,
            restart_dynamic_phase_index=self.restart_state.phase_index,
            restart_dynamic_phase_start_step=self.restart_state.phase_start_step,
            current_phase=current_phase,
            active_batch_index=self.batch_state.active_index,
            active_optimizer_batch_index=active_optimizer_batch_index,
            active_adagrad_restart_phase_index=active_adagrad_restart_phase_index,
            previous_objective=self.objective_state.previous_objective,
            stable_loss_steps=self.objective_state.stable_loss_steps,
            lbfgsb_fallback_used_count=self.lbfgsb_state.fallback_used_count,
        )
        return self.planning_state

    def apply_transition_result(self, transition_result: Any) -> None:
        if transition_result.status is not None:
            self.status = transition_result.status
        self.optimizer = transition_result.optimizer
        self.fd_newton_hessian_state = transition_result.fd_newton_hessian_state
        self.hessian_sgd_line_search_active = (
            transition_result.hessian_sgd_line_search_active
        )
        self.hessian_sgd_low_accept_steps = transition_result.hessian_sgd_low_accept_steps
        self.resume_info = transition_result.resume_info
        self.planning_state = transition_result.planning_state
        if hasattr(transition_result, "current_phase"):
            self.current_phase = transition_result.current_phase

    def apply_initial_plan(
        self,
        plan: _InitialOptimizationPlan,
    ) -> None:
        self.current_phase = plan.current_phase
        self.optimizer = plan.optimizer
        self.batch_state.optimizer_batch_index = plan.active_optimizer_batch_index
        self.restart_state.active_phase_index = plan.active_adagrad_restart_phase_index
        self.resume_info = plan.resume_info
        self.restart_state.phase_index = plan.adagrad_restart_dynamic_phase_index
        self.restart_state.phase_start_step = plan.adagrad_restart_dynamic_phase_start_step
        self.planning_state = self.update_planning_state(
            current_phase=self.current_phase,
            optimizer=plan.optimizer,
            active_optimizer_batch_index=self.batch_state.optimizer_batch_index,
            active_adagrad_restart_phase_index=self.restart_state.active_phase_index,
        )

    def apply_step_plan(
        self,
        plan: _StepIterationPlan,
    ) -> None:
        self.current_phase = plan.current_phase
        self.optimizer = plan.optimizer
        self.batch_state.optimizer_batch_index = plan.active_optimizer_batch_index
        self.restart_state.active_phase_index = plan.active_adagrad_restart_phase_index
        self.objective_state.previous_objective = plan.previous_objective
        self.objective_state.stable_loss_steps = plan.stable_loss_steps
        self.lbfgsb_state.fallback_used_count = plan.lbfgsb_fallback_used_count
        self.planning_state = self.update_planning_state(
            current_phase=self.current_phase,
            optimizer=plan.optimizer,
            active_optimizer_batch_index=self.batch_state.optimizer_batch_index,
            active_adagrad_restart_phase_index=self.restart_state.active_phase_index,
        )

    def make_transition_context(
        self,
        *,
        config: RunConfig,
        model: GeneReconModel,
        evaluation: Any,
        solver: SolverStageController,
        adaptive_state: _AdaptiveRebatchState,
        solver_stage_scope: bool,
        batchwise_hessian_sgd: bool,
        global_solver_warmup: bool,
        lbfgsb_loss_schedule: tuple[Any, ...],
        planning_state: _StepPlanningState,
        best_checkpoint: Path,
        latest_checkpoint: Path,
        checkpoint_every: int | None,
        log_every: int,
        ops: IterationTransitionOps,
        current_phase: str,
    ) -> IterationTransitionContext:
        return IterationTransitionContext(
            config=config,
            model=model,
            evaluation=evaluation,
            solver=solver,
            objective_state=self.objective_state,
            batch_state=self.batch_state,
            restart_state=self.restart_state,
            lbfgsb_state=self.lbfgsb_state,
            adaptive_state=adaptive_state,
            planning_state=planning_state,
            optimizer=self.optimizer,
            fd_newton_hessian_state=self.fd_newton_hessian_state,
            hessian_sgd_line_search_active=self.hessian_sgd_line_search_active,
            hessian_sgd_low_accept_steps=self.hessian_sgd_low_accept_steps,
            resume_info=self.resume_info,
            batch_final_cache=self.batch_final_cache,
            solver_stage_scope=solver_stage_scope,
            batchwise_hessian_sgd=batchwise_hessian_sgd,
            global_solver_warmup=global_solver_warmup,
            lbfgsb_loss_schedule=lbfgsb_loss_schedule,
            current_phase=current_phase,
            best_checkpoint=best_checkpoint,
            latest_checkpoint=latest_checkpoint,
            checkpoint_every=checkpoint_every,
            log_every=log_every,
            ops=ops,
        )

    def sync_transition_context(
        self,
        context: IterationTransitionContext,
        *,
        planning_state: _StepPlanningState,
        solver_stage_scope: bool,
        current_phase: str,
    ) -> IterationTransitionContext:
        context.planning_state = planning_state
        context.optimizer = self.optimizer
        context.fd_newton_hessian_state = self.fd_newton_hessian_state
        context.hessian_sgd_line_search_active = (
            self.hessian_sgd_line_search_active
        )
        context.hessian_sgd_low_accept_steps = self.hessian_sgd_low_accept_steps
        context.resume_info = self.resume_info
        context.batch_final_cache = self.batch_final_cache
        context.solver_stage_scope = solver_stage_scope
        context.current_phase = current_phase
        return context

    def make_iteration_artifacts_state(self) -> _IterationArtifactsState:
        return _IterationArtifactsState(
            adagrad_restart_dynamic_phase_index=self.restart_state.phase_index,
            adagrad_restart_dynamic_phase_start_step=self.restart_state.phase_start_step,
            lbfgsb_loss_schedule_index=self.lbfgsb_state.loss_schedule_index,
        )

    def make_iteration_artifacts_inputs(
        self,
        *,
        step: int,
        phase: str,
        eval_position: str,
        closure_evals: int,
        theta_step: float,
        delta: float | None,
        loss_change_tol_bits: float,
        best_likelihood_min_delta_bits: float,
        row_best_nll: float | None,
        row_best_step: int | None,
        step_s: float,
        metrics: dict[str, Any],
        lbfgsb_loss_schedule_next_index: int | None,
        adagrad_restart_phase_next_index: int | None,
        adagrad_restart_phase_next_start_step: int | None,
    ) -> _IterationArtifactsInputs:
        return _IterationArtifactsInputs(
            step=step,
            phase=phase,
            eval_position=eval_position,
            closure_evals=closure_evals,
            theta_step=theta_step,
            delta=delta,
            loss_change_tol_bits=loss_change_tol_bits,
            best_likelihood_min_delta_bits=best_likelihood_min_delta_bits,
            stable_loss_steps=self.objective_state.stable_loss_steps,
            row_best_nll=row_best_nll,
            row_best_step=row_best_step,
            resume_info=self.resume_info,
            step_s=step_s,
            metrics=metrics,
            previous_objective=self.objective_state.previous_objective,
            lbfgsb_fallback_used_count=self.lbfgsb_state.fallback_used_count,
            lbfgsb_best_retry_count=self.lbfgsb_state.best_retry_count,
            lbfgsb_loss_schedule_next_index=lbfgsb_loss_schedule_next_index,
            active_batch_index=self.batch_state.active_index,
            active_solver_stage=self.batch_state.solver_stage,
            active_batch_local_step=self.batch_state.local_step,
            adagrad_restart_phase_next_index=adagrad_restart_phase_next_index,
            adagrad_restart_phase_next_start_step=adagrad_restart_phase_next_start_step,
        )

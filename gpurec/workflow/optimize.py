from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import torch

from gpurec.api.model import GeneReconModel

from . import _artifacts as _artifact_module
from ._cleanup import close_model_after_error
from ._metadata import (
    checkpoint_status_dict,
)
from .checkpoint import (
    load_checkpoint,
    restore_model_theta,
    save_checkpoint,
    validate_checkpoint_model_compatibility,
)
from ._artifacts import (
    _FINAL_ARTIFACT_FILES,
    _RUN_CONFIG_ARTIFACT_FILE,
    _RUN_MANIFEST_ARTIFACT_FILE,
    _final_check_summary_metrics,
    _run_manifest_hash,
    _runtime_seed_context_from_environment,
    _write_per_family_likelihoods,
    _write_rate_table,
)
from ._evaluation import (
    EvaluationOps,
    _clear_solver_runtime_state_preserving_pi_cache,
    _is_memory_retryable_runtime_error,
)
from ._finalization import (
    _FinalizationInputs,
    finalize_optimization,
)
from ._step_execution import (
    _StepExecutionContext,
    _StepExecutionState,
    execute_optimization_step,
)
from ._step_execution import _restore_theta_if_nonfinite_update
from ._phase import (
    _adagrad_restart_phase_name,
    _continues_after_adagrad_restart_prefix,
    _uses_adagrad_restart_prefix,
)
from ._runtime_state import (
    _resume_state_from_payload,
    _validate_resume_progress,
)
from ._optimizer_factory import (
    _make_optimizer,
    _refresh_optimizer_runtime_options,
)
from ._step_plan import (
    _InitialOptimizationPlan,
    _StepPlanningContext,
    _StepPlanningState,
    _StepIterationPlan,
    prepare_initial_optimization_plan,
    select_step_optimization_plan,
)
from ._solver_stage import SolverStageController
from ._transitions import (
    IterationTransitionContext,
    IterationTransitionInputs,
    apply_iteration_transition,
)
from .config import (
    AdagradRestartPhase,
    LossStopPhase,
    RunConfig,
    adagrad_restart_schedule_specs,
    adagrad_restart_schedule_total_steps,
    loss_stop_schedule_specs,
)
from .diagnostics import (
    append_jsonl,
)
from .model_factory import build_alerax_workflow_model
from ._result import (
    OptimizationResult,
    optimization_result_from_summary as _optimization_result_from_summary,
)
from ._rows import build_iteration_artifacts
from ._rows import (
    _IterationArtifactsContext,
    _IterationArtifactsInputs,
    _IterationArtifactsState,
)
from ._fd_newton import (
    _FDNewtonHessianState,
    _FDNewtonRuntime,
    active_fd_newton_step as _active_fd_newton_step_impl,
)
from ._adaptive_rebatch import _AdaptiveRebatchState
from ._runtime_helpers import (
    _clear_cached_solver_runtime_state,
    _clear_cuda_allocator_cache_if_needed,
    _commit_pi_adjoint_pending_caches,
    _discard_pi_adjoint_pending_caches,
    _drop_cached_static_states_if_needed,
)

_ACTIVE_BATCH_LBFGS_STALL_PATIENCE = 3
_ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES = 64
_HESSIAN_SGD_LINE_SEARCH_MAX_STEPS = 8
_HESSIAN_SGD_LINE_SEARCH_ACCEPT_FRACTION = 0.6
_HESSIAN_SGD_LINE_SEARCH_LOW_ACCEPT_PATIENCE = 2
_HESSIAN_SGD_NO_LINE_REFRESH_STEPS = 64
_HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES = 400_000
_HESSIAN_SGD_SKIP_FULL_AFTER_WARMUP_MIN_CLADES = (
    _HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
)
_BATCHWISE_ACTIVE_OPTIMIZERS = frozenset(
    {"batched-lbfgs", "adam-fd-newton", "hessian-sgd"}
)
_POST_STEP_OPTIMIZERS = frozenset(
    {
        "lbfgs",
        "projected-lbfgs",
        "lbfgsb",
        "batched-lbfgs",
        "adam-fd-newton",
        "hessian-sgd",
    }
)


def _sync_artifact_hooks() -> None:
    _artifact_module._write_rate_table = _write_rate_table
    _artifact_module._write_per_family_likelihoods = _write_per_family_likelihoods


def _step_stopping_status(
    config: RunConfig,
    *,
    step: int,
    stable_loss_steps: int,
    best_step: int | None,
    loss_patience: int | None = None,
    best_likelihood_patience: int | None = None,
) -> dict[str, str] | None:
    loss_patience = config.loss_patience if loss_patience is None else loss_patience
    best_likelihood_patience = (
        config.best_likelihood_patience
        if best_likelihood_patience is None
        else best_likelihood_patience
    )
    if loss_patience and stable_loss_steps >= loss_patience:
        return {"status": "converged", "reason": "loss_change_patience"}
    if (
        best_likelihood_patience
        and best_step is not None
        and step - int(best_step) >= best_likelihood_patience
    ):
        return {"status": "converged", "reason": "best_likelihood_patience"}
    return None


def _active_batch_patience(configured_patience: int) -> int:
    if configured_patience <= 0:
        return configured_patience
    return min(configured_patience, _ACTIVE_BATCH_LBFGS_STALL_PATIENCE)


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
    batch_final_loss_cache: torch.Tensor | None = None
    batch_final_grad_cache: torch.Tensor | None = None
    batch_final_cache_ready: torch.Tensor | None = None
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
        cache_active_batch_final_result: Callable[
            [GeneReconModel, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
            None,
        ],
        active_batch_indices: Callable[[GeneReconModel], torch.Tensor],
        clear_cached_static_states_if_needed: Callable[[GeneReconModel], None],
        clear_cached_solver_runtime_state: Callable[[GeneReconModel], None],
        load_checkpoint_fn: Callable[[Path], dict[str, Any]],
        validate_checkpoint_model_compatibility: Callable[..., None],
        restore_model_theta_fn: Callable[[GeneReconModel, dict[str, Any]], None],
        make_optimizer_fn: Callable[
            [RunConfig, GeneReconModel, str],
            torch.optim.Optimizer,
        ],
        restore_optimizer_state_fn: Callable[
            [torch.optim.Optimizer, Any, str | None, Any | None],
            dict[str, Any],
        ],
        resume_state_from_payload_fn: Callable[[Path, dict[str, Any]], Any],
        save_status: Callable[[Path, Any], None],
        adaptive_checkpoint_status: Callable[[dict[str, Any]], dict[str, Any]],
        print_progress_row: Callable[..., None],
        fd_adam_warmup_steps: int,
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
            batch_final_loss_cache=self.batch_final_loss_cache,
            batch_final_grad_cache=self.batch_final_grad_cache,
            batch_final_cache_ready=self.batch_final_cache_ready,
            solver_stage_scope=solver_stage_scope,
            batchwise_hessian_sgd=batchwise_hessian_sgd,
            global_solver_warmup=global_solver_warmup,
            lbfgsb_loss_schedule=lbfgsb_loss_schedule,
            current_phase=current_phase,
            best_checkpoint=best_checkpoint,
            latest_checkpoint=latest_checkpoint,
            checkpoint_every=checkpoint_every,
            log_every=log_every,
            cache_active_batch_final_result=cache_active_batch_final_result,
            active_batch_indices=active_batch_indices,
            clear_cached_static_states_if_needed=clear_cached_static_states_if_needed,
            clear_cached_solver_runtime_state=clear_cached_solver_runtime_state,
            load_checkpoint_fn=load_checkpoint_fn,
            validate_checkpoint_model_compatibility=validate_checkpoint_model_compatibility,
            restore_model_theta_fn=restore_model_theta_fn,
            make_optimizer_fn=make_optimizer_fn,
            restore_optimizer_state_fn=restore_optimizer_state_fn,
            resume_state_from_payload_fn=resume_state_from_payload_fn,
            save_status=save_status,
            adaptive_checkpoint_status=adaptive_checkpoint_status,
            print_progress_row=print_progress_row,
            fd_adam_warmup_steps=fd_adam_warmup_steps,
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
        context.batch_final_loss_cache = self.batch_final_loss_cache
        context.batch_final_grad_cache = self.batch_final_grad_cache
        context.batch_final_cache_ready = self.batch_final_cache_ready
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


class OptimizationRunner:
    def __init__(
        self,
        config: RunConfig,
        *,
        command_argv: tuple[str, ...] | list[str] | None = None,
    ):
        self.config = config
        self.evaluation = EvaluationOps(config)
        self.solver_stage = SolverStageController(config)
        self.history: list[dict[str, Any]] = []
        self.history_jsonl = config.out_dir / "history.jsonl"
        self.command_argv = (
            tuple(command_argv)
            if command_argv is not None
            else None
        )

    def build_model(self) -> GeneReconModel:
        config = self.config
        build_config = config
        if _uses_adagrad_restart_prefix(config.optimizer):
            first_phase = adagrad_restart_schedule_specs(
                config.adagrad_restart_schedule,
            )[0]
            build_config = replace(
                config,
                fixed_iters_e=first_phase.fixed_iters_e,
                fixed_iters_pi=first_phase.fixed_iters_pi,
                neumann_terms=first_phase.neumann_terms,
            )
        prefetch_batches: int | str = (
            1
            if config.mode == "genewise"
            and config.optimizer in _BATCHWISE_ACTIVE_OPTIMIZERS
            else "all"
        )
        return build_alerax_workflow_model(
            build_config,
            prefetch_batches=prefetch_batches,
        )

    def _uses_solver_warmup(self) -> bool:
        return self.solver_stage.uses_warmup()

    def _configure_solver_stage(self, model: GeneReconModel, stage: str) -> None:
        self.solver_stage.configure_stage(model, stage)

    def _configure_active_solver_stage(
        self,
        model: GeneReconModel,
        stage: str,
    ) -> None:
        self.solver_stage.configure_active_stage(model, stage)

    def _make_optimizer(
        self,
        model: GeneReconModel,
        phase: str,
    ) -> torch.optim.Optimizer:
        return _make_optimizer(self.config, model, phase)

    def _evaluate_loss_only_probe(self, model: GeneReconModel) -> torch.Tensor:
        return self.evaluation.evaluate_loss_only_probe(model)

    def _evaluate_genewise_loss_vector_probe(
        self,
        model: GeneReconModel,
        *,
        active_batch: bool,
    ) -> torch.Tensor:
        return self.evaluation.evaluate_genewise_loss_vector_probe(
            model,
            active_batch=active_batch,
        )

    def _evaluate_active_genewise_vector_and_grad(
        self,
        model: GeneReconModel,
        *,
        solver_stage: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.evaluation.evaluate_active_genewise_vector_and_grad(
            model,
            solver_stage=solver_stage,
        )

    def _evaluate_active_genewise_vector_grad_at_current_theta(
        self,
        model: GeneReconModel,
        *,
        solver_stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self.evaluation.evaluate_active_genewise_vector_grad_at_current_theta(
            model,
            solver_stage=solver_stage,
        )

    def _active_fd_newton_step(
        self,
        model: GeneReconModel,
        *,
        solver_stage: str,
        hessian_state: _FDNewtonHessianState | None = None,
        update_hessian_with_bfgs: bool = True,
        step_scale: float = 1.0,
        use_line_search: bool = True,
        reject_loss_increases_after_step: bool = False,
        hessian_refresh_steps: int | None = None,
        line_search_max_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any], int, _FDNewtonHessianState]:
        def set_model_theta(
            model_arg: GeneReconModel,
            theta: torch.Tensor,
        ) -> None:
            with torch.no_grad():
                model_arg.theta.copy_(theta)

        runtime = _FDNewtonRuntime(
            config=self.config,
            active_batch_indices=self.evaluation._active_batch_indices,
            set_model_theta=set_model_theta,
            evaluate_active_genewise_vector_grad_at_current_theta=(
                self._evaluate_active_genewise_vector_grad_at_current_theta
            ),
            evaluate_genewise_loss_vector_probe=(
                self._evaluate_genewise_loss_vector_probe
            ),
            projected_grad_inf=self.evaluation.projected_grad_inf,
        )
        return _active_fd_newton_step_impl(
            runtime,
            model,
            solver_stage=solver_stage,
            hessian_state=hessian_state,
            update_hessian_with_bfgs=update_hessian_with_bfgs,
            step_scale=step_scale,
            use_line_search=use_line_search,
            reject_loss_increases_after_step=reject_loss_increases_after_step,
            hessian_refresh_steps=hessian_refresh_steps,
            line_search_max_steps=line_search_max_steps,
        )

    def _cache_active_batch_final_result(
        self,
        model: GeneReconModel,
        *,
        loss_vec: torch.Tensor,
        batch_final_loss_cache: torch.Tensor | None,
        batch_final_grad_cache: torch.Tensor | None,
        batch_final_cache_ready: torch.Tensor | None,
    ) -> None:
        if (
            batch_final_loss_cache is None
            or batch_final_grad_cache is None
            or batch_final_cache_ready is None
        ):
            return
        idx = self.evaluation._active_batch_indices(model)
        if idx.numel() == 0:
            return
        batch_final_loss_cache.index_copy_(0, idx, loss_vec.detach().index_select(0, idx))
        if model.theta.grad is not None:
            batch_final_grad_cache.index_copy_(
                0,
                idx,
                model.theta.grad.detach().index_select(0, idx),
            )
        batch_final_cache_ready.index_fill_(0, idx, True)

    def _record(self, row: dict[str, Any]) -> None:
        self.history.append(row)
        append_jsonl(self.history_jsonl, row)

    def _restore_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
        state: Any,
        *,
        current_phase: str | None = None,
        checkpoint_phase: Any = None,
    ) -> dict[str, Any]:
        if state is None:
            return {"resume_optimizer_state": "missing"}
        if checkpoint_phase is not None and not isinstance(checkpoint_phase, str):
            return {
                "resume_optimizer_state": "discarded",
                "resume_optimizer_reason": "invalid_phase",
            }
        if (
            current_phase is not None
            and checkpoint_phase is not None
            and checkpoint_phase != current_phase
        ):
            return {
                "resume_optimizer_state": "discarded",
                "resume_optimizer_reason": "phase_mismatch",
                "resume_optimizer_checkpoint_phase": checkpoint_phase,
                "resume_optimizer_current_phase": current_phase,
            }
        try:
            optimizer.load_state_dict(state)
        except (RuntimeError, TypeError, ValueError) as exc:
            return {
                "resume_optimizer_state": "discarded",
                "resume_optimizer_error": str(exc),
            }
        _refresh_optimizer_runtime_options(optimizer, current_phase, self.config)
        return {"resume_optimizer_state": "restored"}

    def _save_status(
        self,
        path: Path,
        *,
        model: GeneReconModel,
        optimizer: torch.optim.Optimizer | None,
        step: int,
        status: dict[str, Any],
        row: dict[str, Any] | None,
        next_step: int | None = None,
        optimizer_phase: str | None = None,
    ) -> None:
        save_checkpoint(
            path,
            config=self.config,
            model=model,
            optimizer=optimizer,
            optimizer_phase=optimizer_phase,
            step=step,
            next_step=next_step,
            status=status,
            row=row,
        )

    def run(self) -> OptimizationResult:
        config = self.config
        config.out_dir.mkdir(parents=True, exist_ok=True)
        config.write_json(config.out_dir / _RUN_CONFIG_ARTIFACT_FILE)
        if self.history_jsonl.exists() and config.resume_from is None:
            self.history_jsonl.unlink()

        runtime_seed_context = _runtime_seed_context_from_environment()
        model = self.build_model()
        adagrad_restart_specs: tuple[AdagradRestartPhase, ...] = ()
        adagrad_restart_step_limit: int | None = None
        if _uses_adagrad_restart_prefix(config.optimizer):
            adagrad_restart_specs = adagrad_restart_schedule_specs(
                config.adagrad_restart_schedule,
            )
            adagrad_restart_step_limit = adagrad_restart_schedule_total_steps(
                config.adagrad_restart_schedule,
            )
        started = time.perf_counter()
        lbfgsb_loss_schedule: tuple[LossStopPhase, ...] = (
            loss_stop_schedule_specs(config.lbfgsb_loss_change_tol_schedule)
            if config.lbfgsb_loss_change_tol_schedule is not None
            else ()
        )
        adagrad_restart_dynamic_enabled = (
            _uses_adagrad_restart_prefix(config.optimizer)
            and config.adagrad_restart_phase_loss_patience > 0
        )
        adagrad_restart_dynamic_state_loaded = False
        batchwise_active_optimizer = (
            config.mode == "genewise"
            and config.optimizer in _BATCHWISE_ACTIVE_OPTIMIZERS
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
        planning_context = _StepPlanningContext(
            solver=self.solver_stage,
            config=config,
            adagrad_restart_specs=adagrad_restart_specs,
            adagrad_restart_step_limit=adagrad_restart_step_limit,
            adagrad_restart_dynamic_enabled=adagrad_restart_dynamic_enabled,
            adagrad_restart_dynamic_state_loaded=adagrad_restart_dynamic_state_loaded,
            batchwise_active_optimizer=batchwise_active_optimizer,
            batchwise_active_optimizer_phases=frozenset(_BATCHWISE_ACTIVE_OPTIMIZERS),
            batchwise_batched_lbfgs=batchwise_batched_lbfgs,
            batchwise_fd_newton=batchwise_fd_newton,
            batchwise_hessian_sgd=batchwise_hessian_sgd,
            clear_cached_solver_runtime_state=_clear_cached_solver_runtime_state,
            make_optimizer=self._make_optimizer,
        )
        step_execution_context = _StepExecutionContext(
            config=config,
            evaluation=self.evaluation,
            solver=self.solver_stage,
            batchwise_active_optimizer=batchwise_active_optimizer,
            fd_adam_warmup_steps=config.fd_adam_warmup_steps,
            hessian_sgd_no_line_refresh_min_clades=_HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES,
            hessian_sgd_no_line_refresh_steps=_HESSIAN_SGD_NO_LINE_REFRESH_STEPS,
            hessian_sgd_line_search_max_steps=_HESSIAN_SGD_LINE_SEARCH_MAX_STEPS,
        )
        adaptive_rebatch_enabled = bool(
            config.adaptive_rebatch
            and batchwise_active_optimizer
        )
        solver_warmup_enabled = self.solver_stage.uses_warmup()
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
        run_state = _OptimizationRunState(
            objective_state=objective_state,
            batch_state=batch_state,
            restart_state=restart_state,
            lbfgsb_state=lbfgsb_state,
            planning_state=planning_state,
            current_phase="",
            batch_final_loss_cache=None,
            batch_final_grad_cache=None,
            batch_final_cache_ready=None,
        )
        global_solver_warmup = solver_warmup_enabled and not batchwise_active_optimizer
        adaptive_state = _AdaptiveRebatchState.create(
            enabled=adaptive_rebatch_enabled,
            model=model,
            min_active_families=_ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES,
        )
        best_checkpoint = config.out_dir / "checkpoints" / "best.pt"
        latest_checkpoint = config.out_dir / "checkpoints" / "latest.pt"
        iteration_artifacts_context = _IterationArtifactsContext(
            active_objective_scope=batchwise_active_optimizer,
            global_solver_warmup=global_solver_warmup,
            adagrad_restart_dynamic_enabled=adagrad_restart_dynamic_enabled,
            lbfgsb_loss_schedule=lbfgsb_loss_schedule,
        )

        def _print_progress_row(
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

        try:
            if config.resume_from is not None:
                run_state.resume_payload = load_checkpoint(
                    config.resume_from,
                    map_location="cpu",
                )
                validate_checkpoint_model_compatibility(
                    path=config.resume_from,
                    config=config,
                    model=model,
                    payload=run_state.resume_payload,
                )
                resume_state = _resume_state_from_payload(
                    config.resume_from,
                    run_state.resume_payload,
                )
                _validate_resume_progress(
                    config.resume_from,
                    resume_state,
                    configured_steps=config.steps,
                )
                restore_model_theta(model, run_state.resume_payload)
                run_state.start_step = resume_state.start_step
                objective_state.best_nll = resume_state.best_nll
                objective_state.best_step = resume_state.best_step
                objective_state.previous_objective = resume_state.previous_objective
                objective_state.stable_loss_steps = resume_state.stable_loss_steps
                lbfgsb_state.fallback_used_count = resume_state.lbfgsb_fallback_used_count
                lbfgsb_state.loss_schedule_index = min(
                    int(resume_state.lbfgsb_loss_schedule_index),
                    max(0, len(lbfgsb_loss_schedule) - 1),
                )
                lbfgsb_state.best_retry_count = resume_state.lbfgsb_best_retry_count
                resume_status = checkpoint_status_dict(
                    config.resume_from,
                    run_state.resume_payload,
                )
                if (
                    run_state.start_step < config.steps
                    and resume_status.get("status") != "running"
                ):
                    objective_state.reset_tracking()
                batch_state.active_index = resume_state.active_batch_index
                batch_state.solver_stage = resume_state.active_solver_stage
                batch_state.local_step = resume_state.active_batch_local_step
                if adagrad_restart_dynamic_enabled:
                    if resume_state.adagrad_restart_dynamic_phase_index is not None:
                        restart_state.phase_index = int(
                            resume_state.adagrad_restart_dynamic_phase_index
                        )
                    if resume_state.adagrad_restart_dynamic_phase_start_step is not None:
                        restart_state.phase_start_step = int(
                            resume_state.adagrad_restart_dynamic_phase_start_step
                        )
                    adagrad_restart_dynamic_state_loaded = (
                        resume_state.adagrad_restart_dynamic_phase_index is not None
                        and resume_state.adagrad_restart_dynamic_phase_start_step
                        is not None
                    )
                    planning_context = replace(
                        planning_context,
                        adagrad_restart_dynamic_state_loaded=(
                            adagrad_restart_dynamic_state_loaded
                        ),
                    )
                if adaptive_rebatch_enabled:
                    resume_replan_indices = adaptive_state.restore_from_resume(
                        model=model,
                        resume_state=resume_state,
                        active_batch_index=batch_state.active_index,
                        checkpoint_path=str(config.resume_from),
                    )
                    if (
                        resume_replan_indices is not None
                        and adaptive_rebatch_enabled
                        and resume_replan_indices
                    ):
                        model.replan_resident_batches(resume_replan_indices)
                if batch_state.solver_stage not in {"warmup", "full"}:
                    raise RuntimeError(
                        f"checkpoint {config.resume_from} has invalid active_solver_stage"
                    )
                if batch_state.solver_stage == "warmup" and not solver_warmup_enabled:
                    batch_state.solver_stage = "full"
                if batchwise_active_optimizer:
                    batch_state.best_nll = objective_state.best_nll
                    batch_state.best_step = objective_state.best_step
                    objective_state.reset_best()
                planning_state = run_state.update_planning_state(
                    current_phase="warmup" if solver_warmup_enabled else "full",
                    optimizer=run_state.optimizer,
                    active_optimizer_batch_index=batch_state.optimizer_batch_index,
                    active_adagrad_restart_phase_index=restart_state.active_phase_index,
                )

            if batchwise_active_optimizer:
                if batch_state.active_index >= len(model.batch_metadata):
                    raise RuntimeError(
                        f"checkpoint active batch {batch_state.active_index} exceeds "
                        f"{len(model.batch_metadata)} model batches"
                    )
                batch_final_loss_cache = torch.empty(
                    (int(model.n_families),),
                    device=model.theta.device,
                    dtype=model.theta.dtype,
                )
                run_state.batch_final_loss_cache = batch_final_loss_cache
                batch_final_grad_cache = torch.empty_like(model.theta)
                run_state.batch_final_grad_cache = batch_final_grad_cache
                batch_final_cache_ready = torch.zeros(
                    (int(model.n_families),),
                    device=model.theta.device,
                    dtype=torch.bool,
                )
                run_state.batch_final_cache_ready = batch_final_cache_ready
                if model.current_batch_index != batch_state.active_index:
                    _clear_cached_solver_runtime_state(model)
                model.select_batch(batch_state.active_index)
                self.solver_stage.configure_active_stage(
                    model,
                    batch_state.solver_stage,
                )
            elif global_solver_warmup:
                self.solver_stage.configure_active_stage(model, batch_state.solver_stage)
            solver_stage_scope = batchwise_active_optimizer or global_solver_warmup

            optimization_stop_step = config.steps
            if (
                adagrad_restart_step_limit is not None
                and not _continues_after_adagrad_restart_prefix(config.optimizer)
            ):
                optimization_stop_step = min(
                    optimization_stop_step,
                    adagrad_restart_step_limit,
                )

            initial_plan = prepare_initial_optimization_plan(
                planning_context,
                planning_state,
                model,
                start_step=run_state.start_step,
                optimization_stop_step=optimization_stop_step,
                resume_payload=run_state.resume_payload,
                restore_optimizer_state=self._restore_optimizer_state,
            )
            run_state.apply_initial_plan(initial_plan)
            current_phase = run_state.current_phase
            planning_state = run_state.planning_state
            transition_context = run_state.make_transition_context(
                config=config,
                model=model,
                evaluation=self.evaluation,
                solver=self.solver_stage,
                adaptive_state=adaptive_state,
                solver_stage_scope=solver_stage_scope,
                batchwise_hessian_sgd=batchwise_hessian_sgd,
                global_solver_warmup=global_solver_warmup,
                lbfgsb_loss_schedule=lbfgsb_loss_schedule,
                planning_state=planning_state,
                best_checkpoint=best_checkpoint,
                latest_checkpoint=latest_checkpoint,
                checkpoint_every=config.checkpoint_every,
                log_every=config.log_every,
                cache_active_batch_final_result=self._cache_active_batch_final_result,
                active_batch_indices=self.evaluation._active_batch_indices,
                clear_cached_static_states_if_needed=_drop_cached_static_states_if_needed,
                clear_cached_solver_runtime_state=_clear_cached_solver_runtime_state,
                load_checkpoint_fn=lambda path: load_checkpoint(
                    path,
                    map_location="cpu",
                ),
                validate_checkpoint_model_compatibility=validate_checkpoint_model_compatibility,
                restore_model_theta_fn=restore_model_theta,
                make_optimizer_fn=lambda config, model_arg, phase: self._make_optimizer(
                    model_arg,
                    phase,
                ),
                restore_optimizer_state_fn=self._restore_optimizer_state,
                resume_state_from_payload_fn=_resume_state_from_payload,
                save_status=self._save_status,
                adaptive_checkpoint_status=adaptive_state.checkpoint_status,
                print_progress_row=_print_progress_row,
                fd_adam_warmup_steps=config.fd_adam_warmup_steps,
                current_phase=current_phase,
            )

            for step in range(run_state.start_step, optimization_stop_step):
                step_plan = select_step_optimization_plan(
                    planning_context,
                    planning_state,
                    model,
                    step=step,
                )
                phase = step_plan.phase
                adagrad_restart_active_phase = (
                    step_plan.adagrad_restart_active_phase
                )
                adagrad_restart_phase_step = (
                    step_plan.adagrad_restart_phase_step
                )
                run_state.apply_step_plan(step_plan)
                current_phase = run_state.current_phase
                planning_state = run_state.planning_state

                step_s: float
                step_s = 0.0
                step_start = time.perf_counter()
                theta_before = model.theta.detach().clone()
                save_best_after_row = False
                first_order_pending_step = False
                adaptive_rebatch_pending_indices: list[int] | None = None
                adaptive_rebatch_stop = False
                eval_position = (
                    "post_step"
                    if phase in _POST_STEP_OPTIMIZERS
                    else "pre_step"
                )
                step_result = execute_optimization_step(
                    step_execution_context,
                    _StepExecutionState(
                        active_solver_stage=batch_state.solver_stage,
                        active_batch_local_step=batch_state.local_step,
                        fd_newton_hessian_state=run_state.fd_newton_hessian_state,
                        hessian_sgd_line_search_active=run_state.hessian_sgd_line_search_active,
                    ),
                    model,
                    run_state.optimizer,
                    phase=phase,
                    step=step,
                    adagrad_restart_active_phase=adagrad_restart_active_phase,
                )
                step_s = time.perf_counter() - step_start
                if step_result.status is not None:
                    run_state.status = step_result.status
                    break
                metrics = step_result.metrics
                closure_evals = step_result.closure_evals
                theta_step = step_result.theta_step
                loss_vec_current = step_result.loss_vec_current
                first_order_pending_step = step_result.first_order_pending_step
                batch_state.local_step = step_result.active_batch_local_step
                if phase in {"adam-fd-newton", "hessian-sgd"}:
                    run_state.fd_newton_hessian_state = (
                        None
                        if phase == "hessian-sgd"
                        and step_result.hessian_sgd_validation_step
                        else step_result.next_fd_newton_hessian_state
                    )
                if (
                    phase == "batched-lbfgs"
                    and batch_state.solver_stage == "full"
                    and run_state.batch_final_loss_cache is not None
                    and run_state.batch_final_grad_cache is not None
                    and run_state.batch_final_cache_ready is not None
                ):
                    self._cache_active_batch_final_result(
                        model,
                        loss_vec=loss_vec_current,
                        batch_final_loss_cache=run_state.batch_final_loss_cache,
                        batch_final_grad_cache=run_state.batch_final_grad_cache,
                        batch_final_cache_ready=run_state.batch_final_cache_ready,
                    )
                elif (
                    phase in {"adam-fd-newton", "hessian-sgd"}
                    and step_result.cacheable_active_batch_final_result
                    and run_state.batch_final_loss_cache is not None
                    and run_state.batch_final_grad_cache is not None
                    and run_state.batch_final_cache_ready is not None
                ):
                    self._cache_active_batch_final_result(
                        model,
                        loss_vec=loss_vec_current,
                        batch_final_loss_cache=run_state.batch_final_loss_cache,
                        batch_final_grad_cache=run_state.batch_final_grad_cache,
                        batch_final_cache_ready=run_state.batch_final_cache_ready,
                    )
                if phase.startswith("adagrad-restarts:") and adagrad_restart_active_phase is not None:
                    adagrad_restart_phase_step = (
                        step - adagrad_restart_active_phase.start_step
                    )

                if adaptive_rebatch_enabled and phase in _BATCHWISE_ACTIVE_OPTIMIZERS:
                    adaptive_rebatch_decision = adaptive_state.evaluate(
                        config=config,
                        model=model,
                        active_solver_stage=batch_state.solver_stage,
                        step=step,
                        loss_vec_current=loss_vec_current,
                    )
                    metrics.update(adaptive_rebatch_decision.metrics)
                    adaptive_rebatch_pending_indices = (
                        adaptive_rebatch_decision.pending_indices
                    )
                    adaptive_rebatch_stop = adaptive_rebatch_decision.stop

                active_objective_scope = (
                    batchwise_active_optimizer
                    and phase in _BATCHWISE_ACTIVE_OPTIMIZERS
                )
                solver_stage_scope = active_objective_scope or global_solver_warmup
                if solver_stage_scope:
                    metrics.setdefault(
                        "optimizer/solver_stage",
                        batch_state.solver_stage,
                    )
                active_family_count = (
                    max(1, int(metrics.get("optimizer/batch_family_count", 1)))
                    if active_objective_scope
                    else 1
                )
                effective_loss_change_tol = float(config.loss_change_tol)
                effective_loss_patience = int(config.loss_patience)
                if phase == "lbfgsb" and lbfgsb_loss_schedule:
                    lbfgsb_state.loss_schedule_index = min(
                        lbfgsb_state.loss_schedule_index,
                        len(lbfgsb_loss_schedule) - 1,
                    )
                    loss_phase = lbfgsb_loss_schedule[lbfgsb_state.loss_schedule_index]
                    effective_loss_change_tol = float(loss_phase.loss_change_tol)
                    effective_loss_patience = int(loss_phase.loss_patience)
                    metrics["optimizer/lbfgsb_loss_schedule_index"] = float(
                        lbfgsb_state.loss_schedule_index
                    )
                    metrics["optimizer/lbfgsb_loss_schedule_phases"] = float(
                        len(lbfgsb_loss_schedule)
                    )
                    metrics["optimizer/lbfgsb_loss_schedule_active_tol"] = (
                        effective_loss_change_tol
                    )
                    metrics["optimizer/lbfgsb_loss_schedule_active_patience"] = (
                        float(effective_loss_patience)
                    )
                loss_change_tol_bits = effective_loss_change_tol * active_family_count
                best_likelihood_min_delta_bits = (
                    config.best_likelihood_min_delta * active_family_count
                )
                objective = float(metrics["likelihood/data_nll_bits"])
                delta = (
                    None
                    if objective_state.previous_objective is None
                    else objective_state.previous_objective - objective
                )
                projected_lbfgs_backoff = False
                projected_lbfgs_min_lr_reached = False
                bounded_high_projected_plateau = False
                if phase in {"projected-lbfgs", "lbfgsb"} and run_state.optimizer is not None:
                    metric_prefix = (
                        "projected_lbfgs" if phase == "projected-lbfgs" else "lbfgsb"
                    )
                    projected_inf_raw = metrics.get("grad/projected_inf")
                    projected_inf_value = (
                        float(projected_inf_raw)
                        if projected_inf_raw is not None
                        else float("inf")
                    )
                    accepted = bool(
                        metrics.get(f"optimizer/{metric_prefix}_accepted", True)
                    )
                    plateau = delta is not None and delta <= loss_change_tol_bits
                    high_projected_grad = projected_inf_value > config.projected_grad_tol
                    bounded_high_projected_plateau = (
                        config.loss_stop_projected_grad_gate
                        and high_projected_grad
                        and (plateau or not accepted)
                    )
                    if (
                        phase == "projected-lbfgs"
                        and high_projected_grad
                        and (plateau or not accepted)
                    ):
                        group = run_state.optimizer.param_groups[0]
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
                        metrics["optimizer/projected_lbfgs_projected_grad_tol"] = (
                            float(config.projected_grad_tol)
                        )
                        metrics[
                            "optimizer/projected_lbfgs_loss_stop_projected_grad_gate"
                        ] = bool(config.loss_stop_projected_grad_gate)
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
                        metrics[f"optimizer/{metric_prefix}_projected_grad_tol"] = (
                            float(config.projected_grad_tol)
                        )
                        metrics[
                            f"optimizer/{metric_prefix}_loss_stop_projected_grad_gate"
                        ] = bool(config.loss_stop_projected_grad_gate)
                        if phase == "projected-lbfgs":
                            metrics["optimizer/projected_lbfgs_lr_reduced"] = False
                            metrics["optimizer/projected_lbfgs_min_lr_reached"] = False
                        metrics[f"optimizer/{metric_prefix}_high_projected_grad"] = (
                            high_projected_grad
                        )
                        metrics[f"optimizer/{metric_prefix}_blocked_loss_stop"] = (
                            bounded_high_projected_plateau
                        )
                objective_plateau_this_row = (
                    delta is not None
                    and delta <= loss_change_tol_bits
                    and not projected_lbfgs_backoff
                    and not projected_lbfgs_min_lr_reached
                )
                if objective_plateau_this_row and not bounded_high_projected_plateau:
                    objective_state.stable_loss_steps += 1
                else:
                    objective_state.stable_loss_steps = 0
                objective_state.previous_objective = objective
                adagrad_restart_phase_next_index: int | None = None
                adagrad_restart_phase_next_start_step: int | None = None
                adagrad_restart_terminal_status: dict[str, str] | None = None
                if (
                    adagrad_restart_dynamic_enabled
                    and adagrad_restart_active_phase is not None
                    and adagrad_restart_phase_step is not None
                ):
                    phase_done_by_loss = (
                        objective_state.stable_loss_steps
                        >= int(config.adagrad_restart_phase_loss_patience)
                    )
                    phase_done_by_cap = (
                        adagrad_restart_phase_step + 1
                        >= int(adagrad_restart_active_phase.phase.steps)
                    )
                    phase_done_reason = None
                    if phase_done_by_loss:
                        phase_done_reason = "loss_change_patience"
                    elif phase_done_by_cap:
                        phase_done_reason = "phase_step_cap"
                    if phase_done_reason is not None:
                        last_adagrad_phase = (
                            adagrad_restart_active_phase.index + 1
                            >= len(adagrad_restart_specs)
                        )
                        metrics["optimizer/adagrad_restart_dynamic_phase"] = True
                        metrics["optimizer/adagrad_restart_phase_complete"] = True
                        metrics["optimizer/adagrad_restart_phase_complete_reason"] = (
                            phase_done_reason
                        )
                        metrics["optimizer/adagrad_restart_phase_loss_patience"] = (
                            float(config.adagrad_restart_phase_loss_patience)
                        )
                        if last_adagrad_phase:
                            if _continues_after_adagrad_restart_prefix(
                                config.optimizer
                            ):
                                metrics["optimizer/adagrad_restart_next_phase"] = (
                                    "lbfgsb"
                                )
                                adagrad_restart_phase_next_index = (
                                    len(adagrad_restart_specs)
                                )
                                adagrad_restart_phase_next_start_step = step + 1
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
                                adagrad_restart_active_phase.index + 1
                            )
                            adagrad_restart_phase_next_start_step = step + 1
                            metrics["optimizer/adagrad_restart_next_phase"] = (
                                _adagrad_restart_phase_name(
                                    adagrad_restart_specs,
                                    adagrad_restart_phase_next_index,
                                )
                            )
                    else:
                        metrics["optimizer/adagrad_restart_dynamic_phase"] = True
                        metrics["optimizer/adagrad_restart_phase_complete"] = False

                if active_objective_scope:
                    (
                        row_best_nll,
                        row_best_step,
                        _,
                    ) = batch_state.update_best(
                        objective=objective,
                        step=step,
                        best_likelihood_min_delta_bits=best_likelihood_min_delta_bits,
                    )
                    save_best_after_row = False
                else:
                    (
                        row_best_nll,
                        row_best_step,
                        save_best_after_row,
                    ) = objective_state.update_best(
                        objective=objective,
                        step=step,
                        best_likelihood_min_delta_bits=best_likelihood_min_delta_bits,
                    )

                lbfgsb_high_kkt_status: dict[str, str] | None = None
                high_kkt_stop_patience = 0
                high_kkt_stop_signal = False
                high_kkt_objective_stalled = False
                if phase == "lbfgsb":
                    if bool(metrics.get("optimizer/lbfgsb_fallback_used", False)):
                        lbfgsb_state.fallback_used_count += 1
                    high_kkt_stall_count = int(
                        metrics.get("optimizer/lbfgsb_high_kkt_stall_count", 0)
                    )
                    high_kkt_stop_patience = int(
                        config.lbfgsb_high_kkt_stop_patience
                    )
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
                        and (
                            fallback_used_this_row
                            or fallback_budget_exhausted_this_row
                        )
                    )
                    high_kkt_objective_stalled = objective_plateau_this_row
                high_kkt_final_loss_phase = (
                    not lbfgsb_loss_schedule
                    or lbfgsb_state.loss_schedule_index >= len(lbfgsb_loss_schedule) - 1
                )
                high_kkt_stop_ready = (
                    high_kkt_stop_patience > 0
                    and high_kkt_stop_signal
                    and high_kkt_objective_stalled
                    and high_kkt_final_loss_phase
                    and lbfgsb_state.fallback_used_count
                    >= int(config.lbfgsb_high_kkt_stop_min_fallbacks)
                )
                metrics["optimizer/lbfgsb_fallback_used_count"] = float(
                    lbfgsb_state.fallback_used_count
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
                metrics["optimizer/lbfgsb_high_kkt_stop_ready"] = (
                    high_kkt_stop_ready
                )
                if high_kkt_stop_ready:
                    lbfgsb_high_kkt_status = {
                        "status": "converged",
                        "reason": "lbfgsb_high_kkt_tiny_progress_patience",
                    }

                lbfgsb_loss_schedule_next_index: int | None = None
                if (
                    phase == "lbfgsb"
                    and lbfgsb_loss_schedule
                    and lbfgsb_high_kkt_status is None
                    and effective_loss_patience
                    and objective_state.stable_loss_steps >= effective_loss_patience
                    and lbfgsb_state.loss_schedule_index + 1
                    < len(lbfgsb_loss_schedule)
                ):
                    lbfgsb_loss_schedule_next_index = (
                        lbfgsb_state.loss_schedule_index + 1
                    )
                    next_loss_phase = lbfgsb_loss_schedule[
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
                    if (
                        config.lbfgsb_loss_schedule_force_fallback
                        and run_state.optimizer is not None
                    ):
                        opt_state = run_state.optimizer.state.get(model.theta)
                        if isinstance(opt_state, dict):
                            previous_stalls = int(
                                opt_state.get("consecutive_high_kkt_stalls", 0)
                            )
                            opt_state["consecutive_high_kkt_stalls"] = max(
                                previous_stalls,
                                2,
                            )
                            metrics[
                                "optimizer/lbfgsb_loss_schedule_force_fallback_next"
                            ] = True
                            metrics[
                                "optimizer/lbfgsb_loss_schedule_force_fallback_previous_stalls"
                            ] = float(previous_stalls)
                elif phase == "lbfgsb" and lbfgsb_loss_schedule:
                    metrics["optimizer/lbfgsb_loss_schedule_advance"] = False
                    metrics[
                        "optimizer/lbfgsb_loss_schedule_force_fallback_next"
                    ] = False

                artifacts = build_iteration_artifacts(
                    iteration_artifacts_context,
                    run_state.make_iteration_artifacts_state(),
                    run_state.make_iteration_artifacts_inputs(
                        step=step,
                        phase=phase,
                        eval_position=eval_position,
                        closure_evals=closure_evals,
                        theta_step=theta_step,
                        delta=delta,
                        loss_change_tol_bits=loss_change_tol_bits,
                        best_likelihood_min_delta_bits=best_likelihood_min_delta_bits,
                        row_best_nll=row_best_nll,
                        row_best_step=row_best_step,
                        step_s=step_s,
                        metrics=metrics,
                        lbfgsb_loss_schedule_next_index=lbfgsb_loss_schedule_next_index,
                        adagrad_restart_phase_next_index=adagrad_restart_phase_next_index,
                        adagrad_restart_phase_next_start_step=adagrad_restart_phase_next_start_step,
                    ),
                )
                row = artifacts.row
                checkpoint_status = artifacts.checkpoint_status
                if save_best_after_row and phase not in _POST_STEP_OPTIMIZERS:
                    best_row = dict(row)
                    best_row["optimizer/step_applied"] = False
                    best_row["step_s"] = step_s
                    self._save_status(
                        best_checkpoint,
                        model=model,
                        optimizer=run_state.optimizer,
                        step=step,
                        next_step=step,
                        status=adaptive_state.checkpoint_status(checkpoint_status),
                        row=best_row,
                        optimizer_phase=phase,
                    )
                    save_best_after_row = False

                if first_order_pending_step:
                    if run_state.optimizer is not None:
                        run_state.optimizer.step()
                    with torch.no_grad():
                        model.clamp_theta_(config.min_rate, config.max_rate)
                    theta_step = float(
                        (model.theta.detach() - theta_before).abs().amax().cpu()
                    )
                rejected_nonfinite_parameter_update = False
                if first_order_pending_step and _restore_theta_if_nonfinite_update(
                    model,
                    theta_before,
                ):
                    theta_step = 0.0
                    rejected_nonfinite_parameter_update = True
                    row["optimizer/step_rejected_reason"] = (
                        "nonfinite_parameter_update"
                    )
                    run_state.status = {
                        "status": "failed",
                        "reason": "nonfinite_parameter_update",
                    }
                model.clear()
                row["theta_step_inf"] = theta_step
                row["optimizer/step_applied"] = bool(
                    (
                        first_order_pending_step
                        and not rejected_nonfinite_parameter_update
                    )
                    or phase in _POST_STEP_OPTIMIZERS
                )
                row["step_s"] = step_s

                run_state.final_row = row
                self._record(row)

                if (
                    projected_lbfgs_backoff
                    or projected_lbfgs_min_lr_reached
                    or bounded_high_projected_plateau
                ):
                    step_status = None
                else:
                    step_status = _step_stopping_status(
                        config,
                        step=step,
                        stable_loss_steps=objective_state.stable_loss_steps,
                        best_step=row_best_step,
                        loss_patience=(
                            _active_batch_patience(config.loss_patience)
                            if active_objective_scope
                            else effective_loss_patience
                        ),
                        best_likelihood_patience=(
                            _active_batch_patience(config.best_likelihood_patience)
                            if active_objective_scope
                            else None
                        ),
                    )
                if lbfgsb_high_kkt_status is not None:
                    step_status = lbfgsb_high_kkt_status
                full_stage_plateau = (
                    step_status is not None
                    and active_objective_scope
                    and batch_state.solver_stage == "full"
                )
                hessian_sgd_activate_line_search = False
                if (
                    batchwise_hessian_sgd
                    and phase == "hessian-sgd"
                    and active_objective_scope
                    and not run_state.hessian_sgd_line_search_active
                    and not full_stage_plateau
                ):
                    accepted_fraction = metrics.get(
                        "optimizer/fd_newton_accepted_fraction"
                    )
                    loss_rejected_rows = metrics.get(
                        "optimizer/fd_newton_loss_rejected_rows",
                        0.0,
                    )
                    low_acceptance = (
                        accepted_fraction is not None
                        and float(accepted_fraction)
                        < _HESSIAN_SGD_LINE_SEARCH_ACCEPT_FRACTION
                        and float(loss_rejected_rows) > 0.0
                    )
                    if low_acceptance:
                        run_state.hessian_sgd_low_accept_steps += 1
                    else:
                        run_state.hessian_sgd_low_accept_steps = 0
                    hessian_sgd_activate_line_search = (
                        run_state.hessian_sgd_low_accept_steps
                        >= _HESSIAN_SGD_LINE_SEARCH_LOW_ACCEPT_PATIENCE
                    )
                    active_clade_count = int(
                        getattr(
                            model.current_batch_metadata,
                            "clade_count",
                            0,
                        )
                        or 0
                    )
                    if (
                        hessian_sgd_activate_line_search
                        and batch_state.solver_stage == "full"
                        and objective_state.stable_loss_steps > 0
                        and active_clade_count
                        >= _HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
                    ):
                        hessian_sgd_activate_line_search = False

                can_lbfgsb_retry = (
                    phase == "lbfgsb"
                    and not active_objective_scope
                    and lbfgsb_state.best_retry_count
                    < int(config.lbfgsb_best_retry_attempts)
                    and row_best_step is not None
                    and best_checkpoint.exists()
                )
                planning_state = run_state.update_planning_state(
                    current_phase=current_phase,
                    optimizer=run_state.optimizer,
                    active_optimizer_batch_index=batch_state.optimizer_batch_index,
                    active_adagrad_restart_phase_index=(
                        restart_state.active_phase_index
                    ),
                )
                transition_context = run_state.sync_transition_context(
                    transition_context,
                    planning_state=planning_state,
                    solver_stage_scope=solver_stage_scope,
                    current_phase=current_phase,
                )
                transition = apply_iteration_transition(
                    context=transition_context,
                    inputs=IterationTransitionInputs(
                        status=run_state.status,
                        step=step,
                        phase=phase,
                        row=row,
                        checkpoint_status=checkpoint_status,
                        step_status=step_status,
                        objective=objective,
                        row_best_nll=row_best_nll,
                        row_best_step=row_best_step,
                        active_objective_scope=active_objective_scope,
                        active_batch_count=len(model.batch_metadata),
                        can_lbfgsb_retry=can_lbfgsb_retry,
                        lbfgsb_high_kkt_status=lbfgsb_high_kkt_status,
                        hessian_sgd_activate_line_search=hessian_sgd_activate_line_search,
                        projected_lbfgs_min_lr_reached=projected_lbfgs_min_lr_reached,
                        adaptive_rebatch_stop=adaptive_rebatch_stop,
                        rejected_nonfinite_parameter_update=rejected_nonfinite_parameter_update,
                        adaptive_rebatch_pending_indices=adaptive_rebatch_pending_indices,
                        adagrad_restart_terminal_status=adagrad_restart_terminal_status,
                        adagrad_restart_phase_next_index=adagrad_restart_phase_next_index,
                        adagrad_restart_phase_next_start_step=adagrad_restart_phase_next_start_step,
                        lbfgsb_loss_schedule_next_index=lbfgsb_loss_schedule_next_index,
                    ),
                )
                run_state.apply_transition_result(transition)
                planning_state = run_state.planning_state
                current_phase = run_state.current_phase

                if transition.status is not None:
                    run_state.status = transition.status
                if transition.break_loop:
                    break
                if transition.continue_loop:
                    continue

                if save_best_after_row and not transition.continue_loop:
                    self._save_status(
                        best_checkpoint,
                        model=model,
                        optimizer=run_state.optimizer,
                        step=step,
                        status=adaptive_state.checkpoint_status(checkpoint_status),
                        row=row,
                        optimizer_phase=phase,
                    )
                if config.checkpoint_every and (step + 1) % config.checkpoint_every == 0:
                    self._save_status(
                        latest_checkpoint,
                        model=model,
                        optimizer=run_state.optimizer,
                        step=step,
                        status=adaptive_state.checkpoint_status(checkpoint_status),
                        row=row,
                        optimizer_phase=phase,
                    )

                if step % config.log_every == 0:
                    _print_progress_row(
                        step=step,
                        phase=phase,
                        row=row,
                        objective=objective,
                        delta=delta,
                        row_best_nll=row_best_nll,
                    )
            else:
                if (
                    config.optimizer == "adagrad-restarts"
                    and adagrad_restart_step_limit is not None
                    and optimization_stop_step >= adagrad_restart_step_limit
                    and config.steps >= adagrad_restart_step_limit
                ):
                    run_state.status = {
                        "status": "converged",
                        "reason": "adagrad_restart_schedule_complete",
                    }
                else:
                    run_state.status = {"status": "not_converged", "reason": "max_steps"}

            # keep state mirror coherent for callers expecting state-owned status
            _sync_artifact_hooks()
            result = finalize_optimization(
                config,
                _FinalizationInputs(
                    model=model,
                    optimizer=run_state.optimizer,
                    history=self.history,
                    history_jsonl=self.history_jsonl,
                    best_checkpoint=best_checkpoint,
                    latest_checkpoint=latest_checkpoint,
                    status=run_state.status,
                    resume_info=run_state.resume_info,
                    final_row=run_state.final_row,
                    start_step=run_state.start_step,
                    stable_loss_steps=objective_state.stable_loss_steps,
                    best_nll=objective_state.best_nll,
                    best_step=objective_state.best_step,
                    previous_objective=objective_state.previous_objective,
                    lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count,
                    lbfgsb_best_retry_count=lbfgsb_state.best_retry_count,
                    lbfgsb_loss_schedule=lbfgsb_loss_schedule,
                    lbfgsb_loss_schedule_index=lbfgsb_state.loss_schedule_index,
                    batchwise_active_optimizer=batchwise_active_optimizer,
                    batch_final_loss_cache=run_state.batch_final_loss_cache,
                    batch_final_grad_cache=run_state.batch_final_grad_cache,
                    batch_final_cache_ready=run_state.batch_final_cache_ready,
                    runtime_seed_context=runtime_seed_context,
                    started=started,
                    current_phase=current_phase,
                    command_argv=self.command_argv,
                ),
                evaluation=self.evaluation,
                solver=self.solver_stage,
                save_status=self._save_status,
                adaptive_checkpoint_status=adaptive_state.checkpoint_status,
            )
        except BaseException as exc:
            close_model_after_error(model, exc)
            raise
        else:
            model.close()
            return result


def optimize(
    config: RunConfig,
    command_argv: tuple[str, ...] | list[str] | None = None,
) -> OptimizationResult:
    return OptimizationRunner(
        config,
        command_argv=command_argv,
    ).run()

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from gpurec.api.model import GeneReconModel

from . import _artifacts as _artifact_module
from . import _evaluation as _evaluation_module
from . import _finalization as _finalization_module
from ._cleanup import close_model_after_error
from .checkpoint import (
    load_checkpoint,
    restore_model_theta,
    save_checkpoint,
    validate_checkpoint_model_compatibility,
)
from ._artifacts import (
    _FINAL_ARTIFACT_FILES,  # noqa: F401
    _RUN_CONFIG_ARTIFACT_FILE,
    _RUN_MANIFEST_ARTIFACT_FILE,  # noqa: F401
    _final_check_summary_metrics,  # noqa: F401
    _run_manifest_hash,  # noqa: F401
    _runtime_seed_context_from_environment,
    _write_per_family_likelihoods,
    _write_rate_table,
)
from ._batch_final_cache import BatchFinalCache
from ._evaluation import (
    EvaluationOps,
    _clear_solver_runtime_state_preserving_pi_cache,  # noqa: F401
    _is_memory_retryable_runtime_error,  # noqa: F401
)
from ._finalization import (
    _FinalizationInputs,
    _evaluate_final_iteration_check as _evaluate_final_iteration_check_impl,
    finalize_optimization,
)
from ._step_execution import (
    _StepExecutionContext,  # noqa: F401
    _StepExecutionState,
    execute_optimization_step,
)
from ._step_execution import _restore_theta_if_nonfinite_update
from ._phase import _uses_adagrad_restart_prefix
from ._runtime_state import (
    _apply_resume_checkpoint_state,
    _resume_state_from_payload,
)
from ._run_state import (
    BatchRunState,  # noqa: F401
    LBFGSBRunState,  # noqa: F401
    ObjectiveState,  # noqa: F401
    RestartRunState,  # noqa: F401
    _OptimizationRunState,  # noqa: F401
)
from ._run_contexts import _build_workflow_run_contexts  # noqa: F401
from ._run_setup import _derive_workflow_run_setup  # noqa: F401
from ._runner_bootstrap import build_workflow_runner_bootstrap
from ._optimizer_factory import (
    _make_optimizer,
    _refresh_optimizer_runtime_options,
)
from ._step_plan import (
    _StepPlanningContext,  # noqa: F401
    _StepPlanningState,  # noqa: F401
    prepare_initial_optimization_plan,  # noqa: F401
    select_step_optimization_plan,
)
from ._solver_stage import SolverStageController
from ._stopping_policy import (
    _active_batch_patience,
    _step_stopping_status,
)
from ._transitions import (
    apply_iteration_transition,
)
from ._transition_types import (
    IterationTransitionInputs,
    IterationTransitionOps,  # noqa: F401
)
from .config import (
    RunConfig,
    adagrad_restart_schedule_specs,
)
from .diagnostics import (
    append_jsonl,
)
from .model_factory import build_alerax_workflow_model
from ._result import (
    OptimizationResult,
    optimization_result_from_summary as _optimization_result_from_summary,  # noqa: F401
)
from ._rows import build_iteration_artifacts
from ._rows import (
    _IterationArtifactsContext,  # noqa: F401
)
from ._fd_newton import (
    _FDNewtonHessianState,  # noqa: F401
    active_fd_newton_step_for_runner as _run_active_fd_newton_step,
)
from ._adaptive_rebatch import _AdaptiveRebatchState  # noqa: F401
from ._loop_policies import (
    _LoopPolicyContext,  # noqa: F401
    _LoopPolicyInputs,
    _LoopPolicyState,  # noqa: F401
    apply_post_step_loop_policies,
)
from ._hessian_sgd_policy import (
    HESSIAN_SGD_LINE_SEARCH_MAX_STEPS as _HESSIAN_SGD_LINE_SEARCH_MAX_STEPS,  # noqa: F401
    HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES as _HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES,  # noqa: F401
    HESSIAN_SGD_NO_LINE_REFRESH_STEPS as _HESSIAN_SGD_NO_LINE_REFRESH_STEPS,  # noqa: F401
    hessian_sgd_active_clade_count,
    hessian_sgd_line_search_decision,
)
from ._runtime_helpers import (
    _clear_cached_solver_runtime_state,
    _clear_cuda_allocator_cache_if_needed,
    _commit_pi_adjoint_pending_caches,  # noqa: F401
    _discard_pi_adjoint_pending_caches,  # noqa: F401
    _drop_cached_static_states_if_needed,
)

_ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES = 64
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
    _evaluation_module.build_alerax_workflow_model = build_alerax_workflow_model
    _finalization_module.build_alerax_workflow_model = build_alerax_workflow_model
    _finalization_module._clear_cuda_allocator_cache_if_needed = (
        _clear_cuda_allocator_cache_if_needed
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

    def _final_iteration_check_iters(self) -> int:
        return self.solver_stage.final_iteration_check_iters()

    def _evaluate_and_backward(
        self,
        model: GeneReconModel,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.evaluation.evaluate_and_backward(model)

    def _evaluate_loss_only_probe(self, model: GeneReconModel) -> torch.Tensor:
        return self.evaluation.evaluate_loss_only_probe(model)

    def _evaluate_genewise_vector_and_grad(
        self,
        model: GeneReconModel,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.evaluation.evaluate_genewise_vector_and_grad(model)

    def _evaluate_genewise_vector_and_grad_with_memory_fallback(
        self,
        model: GeneReconModel,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _sync_artifact_hooks()
        return self.evaluation.evaluate_genewise_vector_and_grad_with_memory_fallback(
            model,
        )

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

    def _active_batch_metrics(
        self,
        model: GeneReconModel,
        *,
        loss_vec: torch.Tensor,
        solver_stage: str,
    ) -> dict[str, Any]:
        return self.evaluation.active_batch_metrics(
            model,
            loss_vec=loss_vec,
            solver_stage=solver_stage,
        )

    def _projected_grad_inf(
        self,
        model: GeneReconModel,
        *,
        lower_bound: float,
        upper_bound: float,
    ) -> tuple[torch.Tensor, float]:
        return self.evaluation.projected_grad_inf(
            model,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def _evaluate_final_iteration_check(
        self,
        model: GeneReconModel,
        *,
        baseline_loss: torch.Tensor,
        baseline_grad: torch.Tensor,
        baseline_at_check_iters: bool = False,
    ) -> dict[str, Any]:
        _sync_artifact_hooks()
        runner = self

        class _EvaluationFacade:
            def evaluate_and_backward(
                self,
                model_arg: GeneReconModel,
            ) -> tuple[torch.Tensor, dict[str, Any]]:
                return runner._evaluate_and_backward(model_arg)

            def evaluate_genewise_vector_and_grad(
                self,
                model_arg: GeneReconModel,
            ) -> tuple[torch.Tensor, dict[str, Any]]:
                return runner._evaluate_genewise_vector_and_grad(model_arg)

            def _final_eval_fallback_clade_budgets(self) -> list[int]:
                return runner.evaluation._final_eval_fallback_clade_budgets()

        return _evaluate_final_iteration_check_impl(
            self.config,
            solver=self.solver_stage,
            evaluation=_EvaluationFacade(),
            model=model,
            baseline_loss=baseline_loss,
            baseline_grad=baseline_grad,
            baseline_at_check_iters=baseline_at_check_iters,
        )

    _active_fd_newton_step = _run_active_fd_newton_step

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
        started = time.perf_counter()
        try:
            bootstrap = build_workflow_runner_bootstrap(
                config,
                model=model,
                solver=self.solver_stage,
                evaluation=self.evaluation,
                batchwise_active_optimizer_phases=_BATCHWISE_ACTIVE_OPTIMIZERS,
                adaptive_rebatch_min_active_families=(
                    _ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES
                ),
                make_optimizer=self._make_optimizer,
                active_fd_newton_step=self._active_fd_newton_step,
                restore_optimizer_state=self._restore_optimizer_state,
                save_status=self._save_status,
                load_checkpoint=lambda path: load_checkpoint(path, map_location="cpu"),
                validate_checkpoint_model_compatibility=(
                    validate_checkpoint_model_compatibility
                ),
                restore_model_theta=restore_model_theta,
                apply_resume_checkpoint_state=_apply_resume_checkpoint_state,
                resume_state_from_payload=_resume_state_from_payload,
                create_batch_final_cache=BatchFinalCache.create,
                clear_cached_static_states_if_needed=(
                    _drop_cached_static_states_if_needed
                ),
                clear_cached_solver_runtime_state=_clear_cached_solver_runtime_state,
            )
            run_setup = bootstrap.run_setup
            adagrad_restart_step_limit = run_setup.adagrad_restart_step_limit
            lbfgsb_loss_schedule = run_setup.lbfgsb_loss_schedule
            batchwise_active_optimizer = bootstrap.batchwise_active_optimizer
            batchwise_hessian_sgd = bootstrap.batchwise_hessian_sgd
            adaptive_rebatch_enabled = bootstrap.adaptive_rebatch_enabled
            planning_context = bootstrap.planning_context
            step_execution_context = bootstrap.step_execution_context
            iteration_artifacts_context = bootstrap.iteration_artifacts_context
            loop_policy_context = bootstrap.loop_policy_context
            loop_policy_state = bootstrap.loop_policy_state
            run_state = bootstrap.run_state
            objective_state = run_state.objective_state
            batch_state = run_state.batch_state
            restart_state = run_state.restart_state
            lbfgsb_state = run_state.lbfgsb_state
            adaptive_state = bootstrap.adaptive_state
            best_checkpoint = bootstrap.best_checkpoint
            latest_checkpoint = bootstrap.latest_checkpoint
            optimization_stop_step = bootstrap.optimization_stop_step
            transition_context = bootstrap.transition_context
            current_phase = bootstrap.current_phase
            solver_stage_scope = bootstrap.solver_stage_scope
            planning_state = run_state.planning_state
            print_progress_row = bootstrap.print_progress_row

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
                    and run_state.batch_final_cache is not None
                ):
                    run_state.batch_final_cache.cache(
                        model=model,
                        loss_vec=loss_vec_current,
                        active_indices=self.evaluation._active_batch_indices(model),
                    )
                elif (
                    phase in {"adam-fd-newton", "hessian-sgd"}
                    and step_result.cacheable_active_batch_final_result
                    and run_state.batch_final_cache is not None
                ):
                    run_state.batch_final_cache.cache(
                        model=model,
                        loss_vec=loss_vec_current,
                        active_indices=self.evaluation._active_batch_indices(model),
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

                loop_policy = apply_post_step_loop_policies(
                    loop_policy_context,
                    loop_policy_state,
                    _LoopPolicyInputs(
                        step=step,
                        phase=phase,
                        metrics=metrics,
                        model=model,
                        optimizer=run_state.optimizer,
                        adagrad_restart_active_phase=adagrad_restart_active_phase,
                        adagrad_restart_phase_step=adagrad_restart_phase_step,
                    ),
                )
                active_objective_scope = loop_policy.active_objective_scope
                solver_stage_scope = loop_policy.solver_stage_scope
                effective_loss_patience = loop_policy.effective_loss_patience
                loss_change_tol_bits = loop_policy.loss_change_tol_bits
                best_likelihood_min_delta_bits = (
                    loop_policy.best_likelihood_min_delta_bits
                )
                objective = loop_policy.objective
                delta = loop_policy.delta
                projected_lbfgs_backoff = loop_policy.projected_lbfgs_backoff
                projected_lbfgs_min_lr_reached = (
                    loop_policy.projected_lbfgs_min_lr_reached
                )
                bounded_high_projected_plateau = (
                    loop_policy.bounded_high_projected_plateau
                )
                row_best_nll = loop_policy.row_best_nll
                row_best_step = loop_policy.row_best_step
                save_best_after_row = loop_policy.save_best_after_row
                adagrad_restart_phase_next_index = (
                    loop_policy.adagrad_restart_phase_next_index
                )
                adagrad_restart_phase_next_start_step = (
                    loop_policy.adagrad_restart_phase_next_start_step
                )
                adagrad_restart_terminal_status = (
                    loop_policy.adagrad_restart_terminal_status
                )
                lbfgsb_high_kkt_status = loop_policy.lbfgsb_high_kkt_status
                lbfgsb_loss_schedule_next_index = (
                    loop_policy.lbfgsb_loss_schedule_next_index
                )

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
                line_search_decision = hessian_sgd_line_search_decision(
                    batchwise_hessian_sgd=batchwise_hessian_sgd,
                    phase=phase,
                    active_objective_scope=active_objective_scope,
                    line_search_active=run_state.hessian_sgd_line_search_active,
                    full_stage_plateau=full_stage_plateau,
                    accepted_fraction=metrics.get(
                        "optimizer/fd_newton_accepted_fraction"
                    ),
                    loss_rejected_rows=metrics.get(
                        "optimizer/fd_newton_loss_rejected_rows",
                        0.0,
                    ),
                    current_low_accept_steps=run_state.hessian_sgd_low_accept_steps,
                    solver_stage=batch_state.solver_stage,
                    stable_loss_steps=objective_state.stable_loss_steps,
                    active_clade_count=hessian_sgd_active_clade_count(
                        getattr(model, "current_batch_metadata", None)
                    ),
                )
                run_state.hessian_sgd_low_accept_steps = (
                    line_search_decision.low_accept_steps
                )
                hessian_sgd_activate_line_search = line_search_decision.activate

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
                    print_progress_row(
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
                    batch_final_cache=run_state.batch_final_cache,
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

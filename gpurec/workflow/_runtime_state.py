from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from ._metadata import MISSING, checkpoint_finite_float, checkpoint_nonnegative_int
from ._metadata import checkpoint_progress, checkpoint_status_dict


@dataclass(frozen=True)
class _ResumeState:
    start_step: int = 0
    best_nll: float | None = None
    best_step: int | None = None
    previous_objective: float | None = None
    stable_loss_steps: int = 0
    active_batch_index: int = 0
    active_solver_stage: str = "full"
    active_batch_local_step: int = 0
    adagrad_restart_dynamic_phase_index: int | None = None
    adagrad_restart_dynamic_phase_start_step: int | None = None
    converged_family_indices: tuple[int, ...] = ()
    batch_plan_generation: int = 0
    lbfgsb_fallback_used_count: int = 0
    lbfgsb_loss_schedule_index: int = 0
    lbfgsb_best_retry_count: int = 0


@dataclass(frozen=True)
class _ResumeApplicationResult:
    planning_context: Any
    planning_state: Any


def _checkpoint_index_tuple(
    path: Path,
    name: str,
    value: Any,
) -> tuple[int, ...]:
    if value is MISSING or value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"checkpoint {path} has invalid status.{name}")
    out: list[int] = []
    seen: set[int] = set()
    for position, item in enumerate(value):
        index = int(
            checkpoint_nonnegative_int(
                path,
                f"status.{name}[{position}]",
                item,
            )
        )
        if index in seen:
            raise RuntimeError(
                f"checkpoint {path} has duplicate family index {index} in status.{name}"
            )
        seen.add(index)
        out.append(index)
    return tuple(out)


def _resume_state_from_payload(path: Path, payload: dict[str, Any]) -> _ResumeState:
    _, start_step = checkpoint_progress(path, payload)
    ckpt_status = checkpoint_status_dict(path, payload)

    return _ResumeState(
        start_step=start_step,
        best_nll=checkpoint_finite_float(
            path,
            "status.best_nll_bits",
            ckpt_status.get("best_nll_bits"),
            allow_none=True,
        ),
        best_step=checkpoint_nonnegative_int(
            path,
            "status.best_step",
            ckpt_status.get("best_step"),
            allow_none=True,
        ),
        previous_objective=checkpoint_finite_float(
            path,
            "status.previous_objective",
            ckpt_status.get("previous_objective"),
            allow_none=True,
        ),
        stable_loss_steps=int(
            checkpoint_nonnegative_int(
                path,
                "status.stable_loss_steps",
                ckpt_status.get("stable_loss_steps", MISSING),
                default=0,
            )
        ),
        active_batch_index=int(
            checkpoint_nonnegative_int(
                path,
                "status.active_batch_index",
                ckpt_status.get("active_batch_index", MISSING),
                default=0,
            )
        ),
        active_solver_stage=str(ckpt_status.get("active_solver_stage", "full")),
        active_batch_local_step=int(
            checkpoint_nonnegative_int(
                path,
                "status.active_batch_local_step",
                ckpt_status.get("active_batch_local_step", MISSING),
                default=0,
            )
        ),
        adagrad_restart_dynamic_phase_index=checkpoint_nonnegative_int(
            path,
            "status.adagrad_restart_dynamic_phase_index",
            ckpt_status.get("adagrad_restart_dynamic_phase_index"),
            allow_none=True,
        ),
        adagrad_restart_dynamic_phase_start_step=checkpoint_nonnegative_int(
            path,
            "status.adagrad_restart_dynamic_phase_start_step",
            ckpt_status.get("adagrad_restart_dynamic_phase_start_step"),
            allow_none=True,
        ),
        converged_family_indices=_checkpoint_index_tuple(
            path,
            "converged_family_indices",
            ckpt_status.get("converged_family_indices", MISSING),
        ),
        batch_plan_generation=int(
            checkpoint_nonnegative_int(
                path,
                "status.batch_plan_generation",
                ckpt_status.get("batch_plan_generation", MISSING),
                default=0,
            )
        ),
        lbfgsb_fallback_used_count=int(
            checkpoint_nonnegative_int(
                path,
                "status.lbfgsb_fallback_used_count",
                ckpt_status.get("lbfgsb_fallback_used_count", MISSING),
                default=0,
            )
        ),
        lbfgsb_loss_schedule_index=int(
            checkpoint_nonnegative_int(
                path,
                "status.lbfgsb_loss_schedule_index",
                ckpt_status.get("lbfgsb_loss_schedule_index", MISSING),
                default=0,
            )
        ),
        lbfgsb_best_retry_count=int(
            checkpoint_nonnegative_int(
                path,
                "status.lbfgsb_best_retry_count",
                ckpt_status.get("lbfgsb_best_retry_count", MISSING),
                default=0,
            )
        ),
    )


def _validate_resume_progress(
    path: Path,
    state: _ResumeState,
    *,
    configured_steps: int,
) -> None:
    if state.start_step > configured_steps:
        raise RuntimeError(
            f"checkpoint {path} has next_step {state.start_step}, which exceeds "
            f"configured steps {configured_steps}"
        )


def _apply_resume_checkpoint_state(
    *,
    config: Any,
    model: Any,
    run_state: Any,
    planning_context: Any,
    lbfgsb_loss_schedule: tuple[Any, ...],
    solver_warmup_enabled: bool,
    batchwise_active_optimizer: bool,
    adagrad_restart_dynamic_enabled: bool,
    adaptive_rebatch_enabled: bool,
    adaptive_state: Any,
    load_checkpoint: Callable[[Path], dict[str, Any]],
    validate_checkpoint_model_compatibility: Callable[..., None],
    restore_model_theta: Callable[[Any, dict[str, Any]], None],
) -> _ResumeApplicationResult:
    resume_path = config.resume_from
    if resume_path is None:
        raise RuntimeError("missing resume checkpoint path")

    run_state.resume_payload = load_checkpoint(resume_path)
    validate_checkpoint_model_compatibility(
        path=resume_path,
        config=config,
        model=model,
        payload=run_state.resume_payload,
    )
    resume_state = _resume_state_from_payload(
        resume_path,
        run_state.resume_payload,
    )
    _validate_resume_progress(
        resume_path,
        resume_state,
        configured_steps=config.steps,
    )
    restore_model_theta(model, run_state.resume_payload)
    run_state.start_step = resume_state.start_step
    objective_state = run_state.objective_state
    batch_state = run_state.batch_state
    restart_state = run_state.restart_state
    lbfgsb_state = run_state.lbfgsb_state

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
        resume_path,
        run_state.resume_payload,
    )
    if run_state.start_step < config.steps and resume_status.get("status") != "running":
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
            and resume_state.adagrad_restart_dynamic_phase_start_step is not None
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
            checkpoint_path=str(resume_path),
        )
        if resume_replan_indices is not None and resume_replan_indices:
            model.replan_resident_batches(resume_replan_indices)
    if batch_state.solver_stage not in {"warmup", "full"}:
        raise RuntimeError(
            f"checkpoint {resume_path} has invalid active_solver_stage"
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
    return _ResumeApplicationResult(
        planning_context=planning_context,
        planning_state=planning_state,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

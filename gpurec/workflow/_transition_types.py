"""Internal workflow transition DTOs.

This module holds private transition data containers shared by workflow
orchestration helpers and is not a public workflow API surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Callable

    import torch

    from gpurec.api.model import GeneReconModel

    from ._adaptive_rebatch import _AdaptiveRebatchState
    from ._batch_final_cache import BatchFinalCache
    from ._fd_newton import _FDNewtonHessianState
    from ._solver_stage import SolverStageController
    from ._step_plan import _StepPlanningState
    from .config import RunConfig


@dataclass
class IterationTransition:
    status: dict[str, str] | None = None
    continue_loop: bool = False
    break_loop: bool = False
    reset_optimizer: bool = False
    save_latest: bool = False
    save_best: bool = False
    action: str | None = None
    adagrad_restart_next: tuple[int, int] | None = None
    next_adagrad_phase: tuple[int, int] | None = None
    adaptive_rebatch_indices: list[int] | None = None
    lbfgsb_loss_schedule_next_index: int | None = None
    next_batch_active_index: int | None = None


@dataclass
class IterationTransitionExecution:
    status: dict[str, str] | None
    continue_loop: bool
    break_loop: bool
    optimizer: torch.optim.Optimizer | None
    fd_newton_hessian_state: _FDNewtonHessianState | None
    hessian_sgd_line_search_active: bool
    hessian_sgd_low_accept_steps: int
    resume_info: dict[str, Any]
    planning_state: _StepPlanningState


@dataclass(frozen=True)
class IterationTransitionOps:
    active_batch_indices: Callable[[GeneReconModel], torch.Tensor]
    clear_cached_static_states_if_needed: Callable[[GeneReconModel], None]
    clear_cached_solver_runtime_state: Callable[[GeneReconModel], None]
    load_checkpoint: Callable[[Path], dict[str, Any]]
    validate_checkpoint_model_compatibility: Callable[..., None]
    restore_model_theta: Callable[[GeneReconModel, dict[str, Any]], None]
    make_optimizer: Callable[[RunConfig, GeneReconModel, str], torch.optim.Optimizer]
    restore_optimizer_state: Callable[..., dict[str, Any]]
    resume_state_from_payload: Callable[[Path, dict[str, Any]], Any]
    save_status: Callable[..., None]
    adaptive_checkpoint_status: Callable[[dict[str, Any]], dict[str, Any]]
    print_progress_row: Callable[..., None]
    fd_adam_warmup_steps: int


@dataclass
class IterationStatusTransitionExecution:
    status: dict[str, str] | None
    continue_loop: bool
    break_loop: bool
    optimizer: torch.optim.Optimizer | None
    fd_newton_hessian_state: _FDNewtonHessianState | None
    hessian_sgd_line_search_active: bool
    hessian_sgd_low_accept_steps: int
    resume_info: dict[str, Any]
    planning_state: _StepPlanningState
    current_phase: str


@dataclass
class IterationTransitionContext:
    config: RunConfig
    model: GeneReconModel
    evaluation: Any
    solver: SolverStageController
    objective_state: Any
    batch_state: Any
    restart_state: Any
    lbfgsb_state: Any
    adaptive_state: _AdaptiveRebatchState
    planning_state: _StepPlanningState
    optimizer: torch.optim.Optimizer | None
    fd_newton_hessian_state: _FDNewtonHessianState | None
    hessian_sgd_line_search_active: bool
    hessian_sgd_low_accept_steps: int
    resume_info: dict[str, Any]
    batch_final_cache: BatchFinalCache | None
    solver_stage_scope: bool
    batchwise_hessian_sgd: bool
    global_solver_warmup: bool
    lbfgsb_loss_schedule: tuple[Any, ...]
    current_phase: str
    best_checkpoint: Path
    latest_checkpoint: Path
    checkpoint_every: int | None
    log_every: int
    ops: IterationTransitionOps


@dataclass
class IterationTransitionInputs:
    status: dict[str, str] | None
    step: int
    phase: str
    row: dict[str, Any]
    checkpoint_status: dict[str, Any]
    step_status: dict[str, str] | None
    objective: float
    row_best_nll: float | None
    row_best_step: int | None
    active_objective_scope: bool
    active_batch_count: int
    can_lbfgsb_retry: bool
    lbfgsb_high_kkt_status: dict[str, str] | None
    hessian_sgd_activate_line_search: bool
    projected_lbfgs_min_lr_reached: bool
    adaptive_rebatch_stop: bool
    rejected_nonfinite_parameter_update: bool
    adaptive_rebatch_pending_indices: list[int] | None
    adagrad_restart_terminal_status: dict[str, str] | None
    adagrad_restart_phase_next_index: int | None
    adagrad_restart_phase_next_start_step: int | None
    lbfgsb_loss_schedule_next_index: int | None


__all__ = [
    "IterationStatusTransitionExecution",
    "IterationTransition",
    "IterationTransitionContext",
    "IterationTransitionExecution",
    "IterationTransitionInputs",
    "IterationTransitionOps",
]

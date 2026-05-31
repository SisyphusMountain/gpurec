from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from gpurec.api.model import GeneReconModel

from ._phase import (
    _ActiveAdagradRestartPhase,
    _active_adagrad_restart_phase,
    _adagrad_restart_phase_by_index,
    _continues_after_adagrad_restart_prefix,
    _is_adagrad_restart_phase,
    _uses_adagrad_restart_prefix,
)
from ._phase_controller import _build_phase_for_step
from ._solver_stage import SolverStageController
from .config import AdagradRestartPhase, RunConfig


_HESSIAN_CONDITIONED_OPTIMIZERS = frozenset({"adam-fd-newton", "hessian-sgd"})


@dataclass
class _StepPlanningContext:
    solver: SolverStageController
    config: RunConfig
    adagrad_restart_specs: tuple[AdagradRestartPhase, ...]
    adagrad_restart_step_limit: int | None
    adagrad_restart_dynamic_enabled: bool
    adagrad_restart_dynamic_state_loaded: bool
    batchwise_active_optimizer: bool
    batchwise_active_optimizer_phases: frozenset[str]
    batchwise_batched_lbfgs: bool
    batchwise_fd_newton: bool
    batchwise_hessian_sgd: bool
    clear_cached_solver_runtime_state: Callable[[GeneReconModel], None]
    make_optimizer: Callable[[GeneReconModel, str], torch.optim.Optimizer]


@dataclass
class _StepPlanningState:
    restart_dynamic_phase_index: int
    restart_dynamic_phase_start_step: int
    current_phase: str
    active_batch_index: int
    active_optimizer_batch_index: int | None
    active_adagrad_restart_phase_index: int | None
    previous_objective: float | None
    stable_loss_steps: int
    lbfgsb_fallback_used_count: int
    optimizer: torch.optim.Optimizer | None = None


@dataclass(frozen=True)
class _InitialOptimizationPlan:
    initial_adagrad_restart_phase: _ActiveAdagradRestartPhase | None
    current_phase: str
    optimizer: torch.optim.Optimizer
    active_adagrad_restart_phase_index: int | None
    active_optimizer_batch_index: int | None
    resume_info: dict[str, Any]
    adagrad_restart_dynamic_phase_index: int
    adagrad_restart_dynamic_phase_start_step: int


@dataclass(frozen=True)
class _StepIterationPlan:
    phase: str
    adagrad_restart_active_phase: _ActiveAdagradRestartPhase | None
    adagrad_restart_phase_step: int | None
    current_phase: str
    optimizer: torch.optim.Optimizer
    active_optimizer_batch_index: int | None
    active_adagrad_restart_phase_index: int | None
    previous_objective: float | None
    stable_loss_steps: int
    lbfgsb_fallback_used_count: int


def prepare_initial_optimization_plan(
    context: _StepPlanningContext,
    state: _StepPlanningState,
    model: GeneReconModel,
    *,
    start_step: int,
    optimization_stop_step: int,
    resume_payload: dict[str, Any] | None,
    restore_optimizer_state: Callable[..., dict[str, Any]],
) -> _InitialOptimizationPlan:
    solver = context.solver
    config = context.config
    adagrad_restart_specs = context.adagrad_restart_specs
    adagrad_restart_step_limit = context.adagrad_restart_step_limit
    adagrad_restart_dynamic_enabled = context.adagrad_restart_dynamic_enabled
    adagrad_restart_dynamic_phase_index = state.restart_dynamic_phase_index
    adagrad_restart_dynamic_phase_start_step = state.restart_dynamic_phase_start_step
    adagrad_restart_dynamic_state_loaded = context.adagrad_restart_dynamic_state_loaded
    batchwise_active_optimizer = context.batchwise_active_optimizer
    batchwise_active_optimizer_phases = context.batchwise_active_optimizer_phases
    active_batch_index = state.active_batch_index

    initial_adagrad_restart_phase: _ActiveAdagradRestartPhase | None = None

    if (
        _uses_adagrad_restart_prefix(config.optimizer)
        and not (
            _continues_after_adagrad_restart_prefix(config.optimizer)
            and adagrad_restart_dynamic_enabled
            and adagrad_restart_dynamic_phase_index >= len(adagrad_restart_specs)
        )
        and (
            adagrad_restart_step_limit is None
            or start_step < adagrad_restart_step_limit
        )
    ):
        phase_lookup_step = (
            start_step
            if start_step < optimization_stop_step
            else max(0, optimization_stop_step - 1)
        )
        if (
            adagrad_restart_dynamic_enabled
            and not adagrad_restart_dynamic_state_loaded
            and resume_payload is not None
        ):
            fallback_phase = _active_adagrad_restart_phase(
                adagrad_restart_specs,
                phase_lookup_step,
            )
            if fallback_phase is None:
                raise RuntimeError(
                    "adagrad-restarts schedule did not contain the start step"
                )
            adagrad_restart_dynamic_phase_index = fallback_phase.index
            adagrad_restart_dynamic_phase_start_step = fallback_phase.start_step

        if adagrad_restart_dynamic_enabled:
            initial_adagrad_restart_phase = _adagrad_restart_phase_by_index(
                adagrad_restart_specs,
                index=adagrad_restart_dynamic_phase_index,
                start_step=adagrad_restart_dynamic_phase_start_step,
            )
        else:
            initial_adagrad_restart_phase = _active_adagrad_restart_phase(
                adagrad_restart_specs,
                phase_lookup_step,
            )
        if initial_adagrad_restart_phase is None:
            raise RuntimeError(
                "adagrad-restarts schedule did not contain the start step"
            )

        current_phase = (
            f"adagrad-restarts:{initial_adagrad_restart_phase.name}"
        )
        if start_step < optimization_stop_step:
            solver.configure_specieswise_adagrad_restart_phase(
                model,
                initial_adagrad_restart_phase,
            )
        active_adagrad_restart_phase_index = int(
            initial_adagrad_restart_phase.index,
        )
    elif (
        _continues_after_adagrad_restart_prefix(config.optimizer)
        and (
            (adagrad_restart_step_limit is not None and start_step >= adagrad_restart_step_limit)
            or (
                adagrad_restart_dynamic_enabled
                and adagrad_restart_dynamic_phase_index >= len(adagrad_restart_specs)
            )
        )
    ):
        solver.configure_specieswise_adagrad_lbfgsb_tail_solver(model)
        current_phase = "lbfgsb"
        active_adagrad_restart_phase_index = None
    elif start_step >= optimization_stop_step:
        current_phase = _build_phase_for_step(config, start_step)
        active_adagrad_restart_phase_index = None
    else:
        current_phase = _build_phase_for_step(config, start_step)
        active_adagrad_restart_phase_index = None

    optimizer = context.make_optimizer(model, current_phase)
    if initial_adagrad_restart_phase is not None:
        optimizer.param_groups[0]["lr"] = float(
            initial_adagrad_restart_phase.phase.lr,
        )

    active_optimizer_batch_index: int | None = None
    if current_phase in batchwise_active_optimizer_phases and batchwise_active_optimizer:
        active_optimizer_batch_index = active_batch_index

    if resume_payload is not None:
        resume_info = restore_optimizer_state(
            optimizer,
            resume_payload.get("optimizer_state"),
            current_phase=current_phase,
            checkpoint_phase=resume_payload.get("optimizer_phase"),
        )
    else:
        resume_info = {"resume_optimizer_state": "missing"}

    return (
        _InitialOptimizationPlan(
            initial_adagrad_restart_phase=initial_adagrad_restart_phase,
            current_phase=current_phase,
            optimizer=optimizer,
            active_adagrad_restart_phase_index=active_adagrad_restart_phase_index,
            active_optimizer_batch_index=active_optimizer_batch_index,
            resume_info=resume_info,
            adagrad_restart_dynamic_phase_index=adagrad_restart_dynamic_phase_index,
            adagrad_restart_dynamic_phase_start_step=(
                adagrad_restart_dynamic_phase_start_step
            ),
        )
    )


def select_step_optimization_plan(
    context: _StepPlanningContext,
    state: _StepPlanningState,
    model: GeneReconModel,
    *,
    step: int,
) -> _StepIterationPlan:
    solver = context.solver
    config = context.config
    adagrad_restart_specs = context.adagrad_restart_specs
    adagrad_restart_step_limit = context.adagrad_restart_step_limit
    adagrad_restart_dynamic_enabled = context.adagrad_restart_dynamic_enabled
    adagrad_restart_dynamic_phase_index = state.restart_dynamic_phase_index
    adagrad_restart_dynamic_phase_start_step = state.restart_dynamic_phase_start_step
    batchwise_batched_lbfgs = context.batchwise_batched_lbfgs
    batchwise_fd_newton = context.batchwise_fd_newton
    batchwise_hessian_sgd = context.batchwise_hessian_sgd
    active_batch_index = state.active_batch_index
    clear_cached_solver_runtime_state = context.clear_cached_solver_runtime_state
    active_optimizer_batch_index = state.active_optimizer_batch_index
    active_adagrad_restart_phase_index = state.active_adagrad_restart_phase_index
    previous_objective = state.previous_objective
    stable_loss_steps = state.stable_loss_steps
    lbfgsb_fallback_used_count = state.lbfgsb_fallback_used_count
    current_phase = state.current_phase
    optimizer = state.optimizer
    adagrad_restart_active_phase: _ActiveAdagradRestartPhase | None = None
    adagrad_restart_phase_step: int | None = None

    if (
        _uses_adagrad_restart_prefix(config.optimizer)
        and not (
            _continues_after_adagrad_restart_prefix(config.optimizer)
            and adagrad_restart_dynamic_enabled
            and adagrad_restart_dynamic_phase_index >= len(adagrad_restart_specs)
        )
        and (
            not _continues_after_adagrad_restart_prefix(config.optimizer)
            or adagrad_restart_step_limit is None
            or step < adagrad_restart_step_limit
        )
    ):
        if adagrad_restart_dynamic_enabled:
            adagrad_restart_active_phase = _adagrad_restart_phase_by_index(
                adagrad_restart_specs,
                index=adagrad_restart_dynamic_phase_index,
                start_step=adagrad_restart_dynamic_phase_start_step,
            )
        else:
            adagrad_restart_active_phase = _active_adagrad_restart_phase(
                adagrad_restart_specs,
                step,
            )
        if adagrad_restart_active_phase is None:
            raise RuntimeError(
                "adagrad-restarts schedule ended before optimization stop"
            )
        phase = f"adagrad-restarts:{adagrad_restart_active_phase.name}"
        adagrad_restart_phase_step = (
            step - adagrad_restart_active_phase.start_step
        )
    else:
        if (
            _continues_after_adagrad_restart_prefix(config.optimizer)
            and adagrad_restart_dynamic_enabled
            and adagrad_restart_dynamic_phase_index >= len(adagrad_restart_specs)
        ):
            phase = "lbfgsb"
        else:
            phase = _build_phase_for_step(config, step)

    if batchwise_batched_lbfgs and phase == "batched-lbfgs":
        if model.current_batch_index != active_batch_index:
            clear_cached_solver_runtime_state(model)
            model.select_batch(active_batch_index)
        if (
            optimizer is None
            or phase != current_phase
            or active_optimizer_batch_index != active_batch_index
        ):
            current_phase = phase
            optimizer = context.make_optimizer(model, phase)
            active_optimizer_batch_index = active_batch_index
    elif _is_adagrad_restart_phase(phase):
        if adagrad_restart_active_phase is None:
            raise RuntimeError("missing adagrad-restarts active phase")
        if (
            optimizer is None
            or phase != current_phase
            or active_adagrad_restart_phase_index != adagrad_restart_active_phase.index
        ):
            clear_cached_solver_runtime_state(model)
            solver.configure_specieswise_adagrad_restart_phase(
                model,
                adagrad_restart_active_phase,
            )
            current_phase = phase
            optimizer = context.make_optimizer(model, phase)
            optimizer.param_groups[0]["lr"] = float(
                adagrad_restart_active_phase.phase.lr,
            )
            active_adagrad_restart_phase_index = (
                adagrad_restart_active_phase.index
            )
            active_optimizer_batch_index = None
    elif (
        (batchwise_fd_newton or batchwise_hessian_sgd)
        and phase in _HESSIAN_CONDITIONED_OPTIMIZERS
    ):
        if model.current_batch_index != active_batch_index:
            clear_cached_solver_runtime_state(model)
            model.select_batch(active_batch_index)
        if (
            optimizer is None
            or phase != current_phase
            or active_optimizer_batch_index != active_batch_index
        ):
            current_phase = phase
            optimizer = context.make_optimizer(model, phase)
            active_optimizer_batch_index = active_batch_index
    elif optimizer is None or phase != current_phase:
        if (
            _continues_after_adagrad_restart_prefix(config.optimizer)
            and phase == "lbfgsb"
            and _is_adagrad_restart_phase(current_phase)
        ):
            solver.configure_specieswise_adagrad_lbfgsb_tail_solver(model)
            previous_objective = None
            stable_loss_steps = 0
            lbfgsb_fallback_used_count = 0
        current_phase = phase
        optimizer = context.make_optimizer(model, phase)
        active_optimizer_batch_index = None

    return _StepIterationPlan(
        phase=phase,
        adagrad_restart_active_phase=adagrad_restart_active_phase,
        adagrad_restart_phase_step=adagrad_restart_phase_step,
        current_phase=current_phase,
        optimizer=optimizer,
        active_optimizer_batch_index=active_optimizer_batch_index,
        active_adagrad_restart_phase_index=active_adagrad_restart_phase_index,
        previous_objective=previous_objective,
        stable_loss_steps=stable_loss_steps,
        lbfgsb_fallback_used_count=lbfgsb_fallback_used_count,
    )

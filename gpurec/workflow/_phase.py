from __future__ import annotations

from dataclasses import dataclass

from .config import AdagradRestartPhase


@dataclass(frozen=True)
class _ActiveAdagradRestartPhase:
    index: int
    name: str
    start_step: int
    phase: AdagradRestartPhase


def _is_adagrad_restart_phase(phase: str) -> bool:
    return phase == "adagrad-restarts" or phase.startswith("adagrad-restarts:")


def _uses_adagrad_restart_prefix(optimizer: str) -> bool:
    return optimizer in {"adagrad-restarts", "adagrad-restarts-lbfgsb"}


def _continues_after_adagrad_restart_prefix(optimizer: str) -> bool:
    return optimizer == "adagrad-restarts-lbfgsb"


def _adagrad_restart_phase_name(
    specs: tuple[AdagradRestartPhase, ...],
    index: int,
) -> str:
    phase = specs[index]
    if len(specs) == 3:
        label = ("warmup", "bridge", "repair")[index]
    else:
        label = f"phase{index + 1}"
    return f"{phase.budget_label()}_{label}"


def _active_adagrad_restart_phase(
    specs: tuple[AdagradRestartPhase, ...],
    step: int,
) -> _ActiveAdagradRestartPhase | None:
    start_step = 0
    for index, phase in enumerate(specs):
        stop_step = start_step + phase.steps
        if step < stop_step:
            return _ActiveAdagradRestartPhase(
                index=index,
                name=_adagrad_restart_phase_name(specs, index),
                start_step=start_step,
                phase=phase,
            )
        start_step = stop_step
    return None


def _adagrad_restart_phase_by_index(
    specs: tuple[AdagradRestartPhase, ...],
    *,
    index: int,
    start_step: int,
) -> _ActiveAdagradRestartPhase:
    if index < 0 or index >= len(specs):
        raise RuntimeError("adagrad-restarts dynamic phase index is out of range")
    return _ActiveAdagradRestartPhase(
        index=index,
        name=_adagrad_restart_phase_name(specs, index),
        start_step=start_step,
        phase=specs[index],
    )

from __future__ import annotations

from ._phase import (
    _active_adagrad_restart_phase,
    _continues_after_adagrad_restart_prefix,
    _uses_adagrad_restart_prefix,
)
from .config import adagrad_restart_schedule_specs, RunConfig


def _build_phase_for_step(
    config: RunConfig,
    step: int,
) -> str:
    if _uses_adagrad_restart_prefix(config.optimizer):
        specs = adagrad_restart_schedule_specs(
            config.adagrad_restart_schedule,
        )
        active_phase = _active_adagrad_restart_phase(specs, step)
        if active_phase is None:
            if _continues_after_adagrad_restart_prefix(config.optimizer):
                return "lbfgsb"
            return "adagrad-restarts:complete"
        return f"adagrad-restarts:{active_phase.name}"
    if config.optimizer == "adam-lbfgs":
        return "adam" if step < config.adam_warmup_steps else "lbfgs"
    return config.optimizer

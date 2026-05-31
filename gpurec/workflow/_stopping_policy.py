"""Internal optimizer stopping and patience policy helpers.

This module is private support for ``gpurec.workflow.optimize`` orchestration,
not a public workflow API. It owns small pure decisions around loop stop status
and active-batch patience caps; it does not own optimizer stepping,
checkpointing, or artifact finalization.
"""
from __future__ import annotations

from .config import RunConfig

_ACTIVE_BATCH_LBFGS_STALL_PATIENCE = 3


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

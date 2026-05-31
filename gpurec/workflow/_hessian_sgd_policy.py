"""Private Hessian-SGD workflow threshold policy helpers.

This module holds pure decisions shared by optimization orchestration helpers.
It imports no model, tensor, or config objects and is not a public workflow API
surface.
"""

from __future__ import annotations

from dataclasses import dataclass


HESSIAN_SGD_LINE_SEARCH_MAX_STEPS = 8
HESSIAN_SGD_LINE_SEARCH_ACCEPT_FRACTION = 0.6
HESSIAN_SGD_LINE_SEARCH_LOW_ACCEPT_PATIENCE = 2
HESSIAN_SGD_NO_LINE_REFRESH_STEPS = 64
HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES = 400_000
HESSIAN_SGD_SKIP_FULL_AFTER_WARMUP_MIN_CLADES = (
    HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
)


@dataclass(frozen=True)
class HessianSGDLineSearchDecision:
    low_accept_steps: int
    activate: bool


def hessian_sgd_active_clade_count(metadata: object) -> int:
    return int(getattr(metadata, "clade_count", 0) or 0)


def hessian_sgd_line_search_decision(
    *,
    batchwise_hessian_sgd: bool,
    phase: str,
    active_objective_scope: bool,
    line_search_active: bool,
    full_stage_plateau: bool,
    accepted_fraction: float | None,
    loss_rejected_rows: float,
    current_low_accept_steps: int,
    solver_stage: str,
    stable_loss_steps: int,
    active_clade_count: int,
) -> HessianSGDLineSearchDecision:
    if (
        not batchwise_hessian_sgd
        or phase != "hessian-sgd"
        or not active_objective_scope
        or line_search_active
        or full_stage_plateau
    ):
        return HessianSGDLineSearchDecision(
            low_accept_steps=current_low_accept_steps,
            activate=False,
        )

    low_acceptance = (
        accepted_fraction is not None
        and float(accepted_fraction) < HESSIAN_SGD_LINE_SEARCH_ACCEPT_FRACTION
        and float(loss_rejected_rows) > 0.0
    )
    low_accept_steps = current_low_accept_steps + 1 if low_acceptance else 0
    activate = low_accept_steps >= HESSIAN_SGD_LINE_SEARCH_LOW_ACCEPT_PATIENCE
    if (
        activate
        and solver_stage == "full"
        and stable_loss_steps > 0
        and active_clade_count >= HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
    ):
        activate = False
    return HessianSGDLineSearchDecision(
        low_accept_steps=low_accept_steps,
        activate=activate,
    )


def hessian_sgd_should_skip_full_after_warmup(
    *,
    batchwise_hessian_sgd: bool,
    phase: str,
    line_search_active: bool,
    active_clade_count: int,
) -> bool:
    return (
        batchwise_hessian_sgd
        and phase == "hessian-sgd"
        and not line_search_active
        and active_clade_count >= HESSIAN_SGD_SKIP_FULL_AFTER_WARMUP_MIN_CLADES
    )


def hessian_sgd_should_carry_warmup_hessian(
    *,
    batchwise_hessian_sgd: bool,
    phase: str,
    line_search_active: bool,
    active_clade_count: int,
    has_hessian_state: bool,
) -> bool:
    return (
        batchwise_hessian_sgd
        and phase == "hessian-sgd"
        and not line_search_active
        and active_clade_count >= HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
        and has_hessian_state
    )

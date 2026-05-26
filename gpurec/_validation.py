"""Small validation helpers shared across package layers."""

from __future__ import annotations

import math

import torch


ADAPTIVE_NEUMANN_TERMS_DISABLED_MESSAGE = (
    "adaptive_neumann_terms mode is disabled: the current behaviour is "
    "absolutely terrible and MUST be fixed before proceeding; it recomputes "
    "full gradients at each adaptive check."
)


def finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or (
        torch.is_tensor(value) and value.dtype == torch.bool
    ):
        raise ValueError(f"{name} must be a number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def bool_value(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def disabled_adaptive_neumann_terms_value(value: bool) -> bool:
    enabled = bool_value("adaptive_neumann_terms", value)
    if enabled:
        raise ValueError(ADAPTIVE_NEUMANN_TERMS_DISABLED_MESSAGE)
    return enabled

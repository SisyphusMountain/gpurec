"""Internal validation helpers for solver controls."""

from __future__ import annotations

from typing import Any

from gpurec._validation import positive_float


_FIXED_POINT_RELAXATION_MESSAGE = (
    "fixed_point_relaxation must be a positive finite number"
)


def fixed_point_relaxation_value(value: Any) -> float:
    """Normalize the Pi-adjoint fixed-point relaxation control."""
    try:
        return positive_float("fixed_point_relaxation", value)
    except ValueError as exc:
        raise ValueError(_FIXED_POINT_RELAXATION_MESSAGE) from exc

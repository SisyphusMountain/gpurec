"""Small validation helpers shared across package layers."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch


ADAPTIVE_NEUMANN_TERMS_DISABLED_MESSAGE = (
    "adaptive_neumann_terms mode is disabled because it recomputes full "
    "gradients at each adaptive check and is not part of the supported "
    "production optimization route. Leave adaptive_neumann_terms=false and "
    "use fixed neumann_terms or the documented hessian-sgd/adagrad-restarts "
    "defaults."
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


def integer_value(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number_float = finite_float(name, float(value))
        if not number_float.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(number_float)
    raise ValueError(f"{name} must be an integer")


def positive_int(name: str, value: object) -> int:
    number = integer_value(name, value)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


def nonnegative_int(name: str, value: object) -> int:
    number = integer_value(name, value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def positive_even_int(name: str, value: object) -> int:
    number = positive_int(name, value)
    if number % 2 != 0:
        raise ValueError(f"{name} must be a positive even integer")
    return number


def disabled_adaptive_neumann_terms_value(value: bool) -> bool:
    enabled = bool_value("adaptive_neumann_terms", value)
    if enabled:
        raise ValueError(ADAPTIVE_NEUMANN_TERMS_DISABLED_MESSAGE)
    return enabled

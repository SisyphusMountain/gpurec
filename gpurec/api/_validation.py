from __future__ import annotations

import math
import os
from numbers import Integral, Real
from typing import Any, Optional, Sequence

import torch


def require_cuda_device(device: Any, *, owner: str) -> torch.device:
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"{owner} received invalid CUDA device {device!r}") from exc
    if resolved.type != "cuda":
        raise ValueError(f"{owner} currently requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resolved.index is not None:
        device_count = torch.cuda.device_count()
        if resolved.index >= device_count:
            raise ValueError(
                f"{owner} requested CUDA device {resolved}, but only "
                f"{device_count} CUDA device(s) are available"
            )
    return resolved


def require_default_objective(owner: str) -> None:
    value = os.environ.get("GPUREC_ALERAX_COMPAT", "0")
    if value != "0":
        raise RuntimeError(
            f"{owner} does not support GPUREC_ALERAX_COMPAT={value!r}; "
            "unset GPUREC_ALERAX_COMPAT or set it to '0'. The AleRax "
            "fixed-pass compatibility objective is not implemented for "
            "differentiable GPUREC model optimization."
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


def nonnegative_float(name: str, value: float) -> float:
    number = finite_float(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def positive_float(name: str, value: float) -> float:
    number = finite_float(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def bool_value(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def integer_value(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    elif isinstance(value, Real):
        number_float = finite_float(name, float(value))
        if not number_float.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(number_float)
    raise ValueError(f"{name} must be an integer")


def positive_int(name: str, value: int) -> int:
    number = integer_value(name, value)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


def nonnegative_int(name: str, value: int) -> int:
    number = integer_value(name, value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def positive_even_int(name: str, value: int) -> int:
    number = positive_int(name, value)
    if number % 2 != 0:
        raise ValueError(f"{name} must be a positive even integer")
    return number


def optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return positive_int(name, value)


def _contains_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if torch.is_tensor(value):
        return value.dtype == torch.bool
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence):
        return any(_contains_bool(item) for item in value)
    return False


def theta_init_base_from_rates(
    theta_init_rates: Optional[Sequence[float]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if theta_init_rates is None:
        return None
    if _contains_bool(theta_init_rates):
        raise ValueError("theta_init_rates must contain numeric rates, not booleans")
    try:
        rates = torch.as_tensor(theta_init_rates, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "theta_init_rates must contain exactly three numeric D/L/T rates"
        ) from exc
    if rates.numel() != 3:
        raise ValueError("theta_init_rates must contain exactly three D/L/T rates")
    rates = rates.reshape(3)
    if not torch.isfinite(rates).all().item():
        raise ValueError("theta_init_rates must be finite")
    if torch.any(rates <= 0):
        raise ValueError("theta_init_rates must be strictly positive")
    return torch.log2(rates).to(device=device, dtype=dtype)


def validate_theta_shape(
    name: str,
    theta: torch.Tensor,
    *,
    mode: str,
    species_count: int,
    family_count: int,
) -> torch.Tensor:
    if not torch.is_tensor(theta):
        raise ValueError(f"{name} must be a torch.Tensor")
    if mode == "global":
        expected_shape = (3,)
    elif mode == "specieswise":
        expected_shape = (int(species_count), 3)
    elif mode == "genewise":
        expected_shape = (int(family_count), 3)
    else:
        raise ValueError(f"Unknown mode {mode!r} for {name} shape validation")
    actual_shape = tuple(int(dim) for dim in theta.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{name} shape for {mode} mode must be {expected_shape}, "
            f"got {actual_shape}"
        )
    return theta

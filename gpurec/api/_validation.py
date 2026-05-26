"""Internal validation helpers shared by direct API and workflow adapters.

The helpers in this module are support code for ``gpurec.api`` and
``gpurec.workflow``.  They keep device, numeric-control, integer-control, and
theta-shape validation consistent, but they are not standalone public API.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import torch

from gpurec._validation import (
    bool_value,
    finite_float,
    integer_value,
    nonnegative_float,
    nonnegative_int,
    optional_positive_int,
    positive_even_int,
    positive_float,
    positive_int,
)


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


def auto_int(name: str, value: int | float | str | None) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "auto", "default"):
            return "auto"
        if text in ("0", "none", "null"):
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, 'auto', or none") from exc
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, 'auto', or none")
    try:
        return integer_value(name, value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, 'auto', or none") from exc


def auto_nonnegative_int(
    name: str,
    value: int | float | str | None,
) -> int | str | None:
    normalized = auto_int(name, value)
    if isinstance(normalized, int) and normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def auto_positive_int(
    name: str,
    value: int | float | str | None,
) -> int | str | None:
    normalized = auto_int(name, value)
    if isinstance(normalized, int) and normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def nonnegative_int_sequence(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integers")
    try:
        return tuple(nonnegative_int(f"{name} entries", value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc


def positive_int_sequence(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integers")
    try:
        return tuple(positive_int(f"{name} entries", value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc


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
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Validate raw theta shape, dtype, and device for the active sharing mode."""
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
    if not torch.is_floating_point(theta):
        raise ValueError(f"{name} must be a floating-point theta tensor")
    if dtype is not None and theta.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}, got {theta.dtype}")
    if device is not None:
        expected_device = torch.device(device)
        device_matches = (
            theta.device == expected_device
            if expected_device.index is not None
            else theta.device.type == expected_device.type
        )
        if not device_matches:
            raise ValueError(
                f"{name} device must be {expected_device}, got {theta.device}"
            )
    if not bool(torch.isfinite(theta.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return theta

"""Tensor shape validators for model-level loss and gradient outputs."""
from __future__ import annotations

import torch


def _validate_tensor_shape(
    name: str,
    value: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    actual_shape = tuple(int(dim) for dim in value.shape)
    if actual_shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {actual_shape}")
    return value


def _validate_scalar_loss(name: str, value: torch.Tensor) -> torch.Tensor:
    return _validate_tensor_shape(name, value, ())


def _validate_genewise_loss_vector(
    name: str,
    value: torch.Tensor,
    *,
    family_count: int,
) -> torch.Tensor:
    return _validate_tensor_shape(name, value, (int(family_count),))


def _validate_genewise_gradient_matrix(
    name: str,
    value: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    return _validate_tensor_shape(name, value, expected_shape)


def _validate_gradient_shape(
    name: str,
    value: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    return _validate_tensor_shape(name, value, expected_shape)

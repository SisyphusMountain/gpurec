"""Shared box-bound tensor helpers for internal optimizers."""

from __future__ import annotations

import torch
from torch import Tensor


Bound = float | Tensor | None


def bound_for_flat(
    bound: Bound,
    flat: Tensor,
    parameter_shape: torch.Size | tuple[int, ...],
    *,
    broadcast_to_flat: bool = False,
) -> Tensor | None:
    """Return ``bound`` on ``flat``'s device/dtype with optimizer shape semantics."""
    if bound is None:
        return None
    if torch.is_tensor(bound):
        bound_tensor = bound.detach().to(device=flat.device, dtype=flat.dtype)
    else:
        bound_tensor = torch.as_tensor(bound, device=flat.device, dtype=flat.dtype)
    if bound_tensor.ndim == 0:
        return bound_tensor
    if tuple(bound_tensor.shape) == tuple(flat.shape):
        return bound_tensor
    parameter_shape = tuple(parameter_shape)
    if tuple(bound_tensor.shape) == parameter_shape:
        return bound_tensor.reshape_as(flat)
    if broadcast_to_flat:
        try:
            return torch.broadcast_to(bound_tensor, parameter_shape).reshape_as(flat)
        except RuntimeError:
            return torch.broadcast_to(bound_tensor, flat.shape)
    return torch.broadcast_to(bound_tensor, parameter_shape).reshape_as(flat)


def bounds_for_flat(
    flat: Tensor,
    lower_bound: Bound,
    upper_bound: Bound,
    parameter_shape: torch.Size | tuple[int, ...],
    *,
    broadcast_to_flat: bool = False,
) -> tuple[Tensor | None, Tensor | None]:
    """Return compatible lower/upper bounds and reject invalid intervals."""
    lower = bound_for_flat(
        lower_bound,
        flat,
        parameter_shape,
        broadcast_to_flat=broadcast_to_flat,
    )
    upper = bound_for_flat(
        upper_bound,
        flat,
        parameter_shape,
        broadcast_to_flat=broadcast_to_flat,
    )
    if lower is not None and upper is not None and bool((lower > upper).any()):
        raise ValueError("lower_bound must be <= upper_bound")
    return lower, upper


def project_flat(
    flat: Tensor,
    lower_bound: Bound,
    upper_bound: Bound,
    parameter_shape: torch.Size | tuple[int, ...],
    *,
    broadcast_to_flat: bool = False,
) -> Tensor:
    """Clamp a flattened parameter tensor to box bounds."""
    lower, upper = bounds_for_flat(
        flat,
        lower_bound,
        upper_bound,
        parameter_shape,
        broadcast_to_flat=broadcast_to_flat,
    )
    projected = flat
    if lower is not None:
        projected = torch.maximum(projected, lower)
    if upper is not None:
        projected = torch.minimum(projected, upper)
    return projected


def projected_gradient(
    flat: Tensor,
    grad: Tensor,
    lower_bound: Bound,
    upper_bound: Bound,
    parameter_shape: torch.Size | tuple[int, ...],
    *,
    broadcast_to_flat: bool = False,
) -> Tensor:
    """Return the box-projected gradient used for first-order convergence."""
    return flat - project_flat(
        flat - grad,
        lower_bound,
        upper_bound,
        parameter_shape,
        broadcast_to_flat=broadcast_to_flat,
    )


def feasible_direction(
    flat: Tensor,
    direction: Tensor,
    lower_bound: Bound,
    upper_bound: Bound,
    parameter_shape: torch.Size | tuple[int, ...],
    *,
    broadcast_to_flat: bool = False,
) -> Tensor:
    """Zero direction entries that point outside active box bounds."""
    lower, upper = bounds_for_flat(
        flat,
        lower_bound,
        upper_bound,
        parameter_shape,
        broadcast_to_flat=broadcast_to_flat,
    )
    feasible = torch.ones_like(direction, dtype=torch.bool)
    if lower is not None:
        feasible = feasible & ((flat > lower) | (direction >= 0))
    if upper is not None:
        feasible = feasible & ((flat < upper) | (direction <= 0))
    return torch.where(feasible, direction, torch.zeros_like(direction))


class BoxBoundsMixin:
    """Private optimizer methods backed by the shared bound helpers."""

    _bounds_broadcast_to_flat = False
    _bounds_scalar_to_flat = False

    def _bounds_parameter_shape(self) -> torch.Size:
        return self._param.shape

    def _bound_for_flat(self, bound: Bound, flat: Tensor) -> Tensor | None:
        normalized = bound_for_flat(
            bound,
            flat,
            self._bounds_parameter_shape(),
            broadcast_to_flat=self._bounds_broadcast_to_flat,
        )
        if normalized is not None and self._bounds_scalar_to_flat and normalized.ndim == 0:
            return torch.full_like(flat, normalized)
        return normalized

    def _bounds_for_flat(
        self,
        flat: Tensor,
        lower_bound: Bound,
        upper_bound: Bound,
    ) -> tuple[Tensor | None, Tensor | None]:
        lower, upper = bounds_for_flat(
            flat,
            lower_bound,
            upper_bound,
            self._bounds_parameter_shape(),
            broadcast_to_flat=self._bounds_broadcast_to_flat,
        )
        if lower is not None and self._bounds_scalar_to_flat and lower.ndim == 0:
            lower = torch.full_like(flat, lower)
        if upper is not None and self._bounds_scalar_to_flat and upper.ndim == 0:
            upper = torch.full_like(flat, upper)
        return lower, upper

    def _project_flat(
        self,
        flat: Tensor,
        lower_bound: Bound,
        upper_bound: Bound,
    ) -> Tensor:
        return project_flat(
            flat,
            lower_bound,
            upper_bound,
            self._bounds_parameter_shape(),
            broadcast_to_flat=self._bounds_broadcast_to_flat,
        )

    def _projected_gradient(
        self,
        flat: Tensor,
        grad: Tensor,
        lower_bound: Bound,
        upper_bound: Bound,
    ) -> Tensor:
        return projected_gradient(
            flat,
            grad,
            lower_bound,
            upper_bound,
            self._bounds_parameter_shape(),
            broadcast_to_flat=self._bounds_broadcast_to_flat,
        )

    def _feasible_direction(
        self,
        flat: Tensor,
        direction: Tensor,
        lower_bound: Bound,
        upper_bound: Bound,
    ) -> Tensor:
        return feasible_direction(
            flat,
            direction,
            lower_bound,
            upper_bound,
            self._bounds_parameter_shape(),
            broadcast_to_flat=self._bounds_broadcast_to_flat,
        )

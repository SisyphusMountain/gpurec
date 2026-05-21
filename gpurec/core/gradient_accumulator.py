"""Internal theta-gradient accumulation helpers.

This module owns model-boundary accumulation for public theta-shaped
gradients.  Kernel-local adjoint scatter/add paths stay in their CUDA-facing
modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import torch

from .parameter_layout import ParameterLayout, RateMode


class GradientAccumulator:
    """Accumulate theta gradients according to a validated parameter layout."""

    def __init__(self, layout: ParameterLayout, gradient: torch.Tensor) -> None:
        if not torch.is_tensor(gradient):
            raise TypeError("gradient accumulator must be a torch.Tensor")
        layout.validate_theta_shape(gradient, name="gradient accumulator")
        self.layout = layout
        self._gradient = gradient

    @classmethod
    def zeros(
        cls,
        layout: ParameterLayout,
        *,
        device: Any | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GradientAccumulator":
        """Create a zero accumulator with the layout's public theta shape."""
        return cls(
            layout,
            torch.zeros(layout.theta_shape, device=device, dtype=dtype),
        )

    @classmethod
    def zeros_like(
        cls,
        layout: ParameterLayout,
        theta: torch.Tensor,
    ) -> "GradientAccumulator":
        """Create a zero accumulator preserving ``theta`` dtype and device."""
        if not torch.is_tensor(theta):
            raise TypeError("theta must be a torch.Tensor")
        layout.validate_theta_shape(theta, name="theta")
        return cls(layout, torch.zeros_like(theta.detach()))

    @property
    def tensor(self) -> torch.Tensor:
        """The mutable accumulated gradient tensor."""
        return self._gradient

    def result(self) -> torch.Tensor:
        """Return the accumulated gradient tensor."""
        return self._gradient

    def add(
        self,
        contribution: torch.Tensor,
        *,
        family_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> "GradientAccumulator":
        """Add one gradient contribution.

        Shared global/specieswise layouts require a full theta-shaped
        contribution.  Genewise layouts accept batch-local rows and accumulate
        them into the full family axis with ``index_add_``.
        """
        if not torch.is_tensor(contribution):
            raise TypeError("gradient contribution must be a torch.Tensor")

        if self.layout.mode is RateMode.GENEWISE:
            indices = _family_indices_for_contribution(
                self.layout,
                family_indices,
            )
            expected_shape = (len(indices), self.layout.theta_shape[1])
            _validate_contribution_shape(contribution, expected_shape)
            index = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=self._gradient.device,
            )
            with torch.no_grad():
                self._gradient.index_add_(
                    0,
                    index,
                    contribution.to(
                        device=self._gradient.device,
                        dtype=self._gradient.dtype,
                    ),
            )
            return self

        if family_indices is not None:
            raise ValueError("family_indices are only valid for genewise gradients")
        _validate_contribution_shape(contribution, self.layout.theta_shape)
        with torch.no_grad():
            self._gradient.add_(
                contribution.to(
                    device=self._gradient.device,
                    dtype=self._gradient.dtype,
                )
            )
        return self


def _family_indices_for_contribution(
    layout: ParameterLayout,
    family_indices: Sequence[int] | torch.Tensor | None,
) -> tuple[int, ...]:
    if family_indices is None:
        return layout.family_indices
    return _normalize_family_indices(
        family_indices,
        family_count=layout.family_count,
    )


def _normalize_family_indices(
    family_indices: Sequence[int] | torch.Tensor,
    *,
    family_count: int,
) -> tuple[int, ...]:
    if isinstance(family_indices, (str, bytes)):
        raise ValueError("family_indices must be a sequence of integers")
    if torch.is_tensor(family_indices):
        if family_indices.ndim != 1:
            raise ValueError("family_indices must be one-dimensional")
        raw_values = tuple(family_indices.detach().cpu().tolist())
    else:
        try:
            raw_values = tuple(family_indices)
        except TypeError as exc:
            raise ValueError("family_indices must be a sequence of integers") from exc
    if not raw_values:
        raise ValueError("family_indices must not be empty")

    values: list[int] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("family_indices entries must be integers")
        index = int(value)
        if index < 0 or index >= family_count:
            raise IndexError(
                f"family index {index} out of range for {family_count} families"
            )
        values.append(index)
    return tuple(values)


def _validate_contribution_shape(
    contribution: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    shape = tuple(int(dim) for dim in contribution.shape)
    if shape != expected_shape:
        raise ValueError(
            f"gradient contribution shape must be {expected_shape}, got {shape}"
        )


__all__ = ["GradientAccumulator"]

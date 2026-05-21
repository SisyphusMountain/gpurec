"""Internal gradient accumulation helpers.

This module owns model-boundary accumulation for public theta-shaped gradients
and same-shape structured reductions of already-computed gradient dictionaries.
Kernel-local adjoint scatter/add paths stay in their CUDA-facing modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


class StructuredGradientAccumulator:
    """Accumulate fixed-schema gradient dictionaries.

    Tensor fields are detached and cloned on first use, then later
    contributions must match the original tensor shape, dtype, and device
    exactly.  Counter fields are summed as Python integers.  This helper is for
    reductions of already-computed gradient tensors; it does not own any
    kernel-local scatter semantics.
    """

    def __init__(
        self,
        *,
        tensor_keys: Sequence[str],
        counter_keys: Sequence[str] = (),
    ) -> None:
        self.tensor_keys = _normalize_keys(
            "tensor_keys",
            tensor_keys,
            allow_empty=False,
        )
        self.counter_keys = _normalize_keys(
            "counter_keys",
            counter_keys,
            allow_empty=True,
        )
        self._gradient: dict[str, Any] | None = None

    @property
    def is_empty(self) -> bool:
        return self._gradient is None

    def add(
        self,
        contribution: Mapping[str, Any],
    ) -> "StructuredGradientAccumulator":
        if not isinstance(contribution, Mapping):
            raise TypeError("structured gradient contribution must be a mapping")

        if self._gradient is None:
            self._gradient = {
                key: _require_tensor_field(contribution, key).detach().clone()
                for key in self.tensor_keys
            }
            for key in self.counter_keys:
                self._gradient[key] = int(contribution.get(key, 0))
            return self

        for key in self.tensor_keys:
            value = _require_tensor_field(contribution, key)
            target = self._gradient[key]
            _validate_matching_tensor_field(key, value, target)
            target.add_(value)
        for key in self.counter_keys:
            self._gradient[key] = int(self._gradient.get(key, 0)) + int(
                contribution.get(key, 0)
            )
        return self

    def result(self) -> dict[str, Any]:
        if self._gradient is None:
            raise RuntimeError("structured gradient accumulator is empty")
        return self._gradient


def _normalize_keys(
    name: str,
    keys: Sequence[str],
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(keys, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    values = tuple(keys)
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    seen: set[str] = set()
    for key in values:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} entries must be non-empty strings")
        if key in seen:
            raise ValueError(f"{name} contains duplicate key {key!r}")
        seen.add(key)
    return values


def _require_tensor_field(
    contribution: Mapping[str, Any],
    key: str,
) -> torch.Tensor:
    if key not in contribution:
        raise KeyError(f"gradient contribution missing tensor field {key!r}")
    value = contribution[key]
    if not torch.is_tensor(value):
        raise TypeError(f"gradient contribution field {key!r} must be a torch.Tensor")
    return value


def _validate_matching_tensor_field(
    key: str,
    value: torch.Tensor,
    target: torch.Tensor,
) -> None:
    value_shape = tuple(int(dim) for dim in value.shape)
    target_shape = tuple(int(dim) for dim in target.shape)
    if value_shape != target_shape:
        raise ValueError(
            f"gradient contribution field {key!r} shape must be "
            f"{target_shape}, got {value_shape}"
        )
    if value.dtype != target.dtype:
        raise ValueError(
            f"gradient contribution field {key!r} dtype must be "
            f"{target.dtype}, got {value.dtype}"
        )
    if value.device != target.device:
        raise ValueError(
            f"gradient contribution field {key!r} device must be "
            f"{target.device}, got {value.device}"
        )


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


__all__ = ["GradientAccumulator", "StructuredGradientAccumulator"]

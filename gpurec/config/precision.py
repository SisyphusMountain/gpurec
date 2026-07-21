"""Floating-point precision policy for model state and small accumulations.

Configuration files store dtype names rather than :class:`torch.dtype` objects
so precision settings round-trip through TOML and ``GpurecConfig.to_dict()``.
The conversion to PyTorch dtypes happens once at construction boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


_TORCH_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float64": torch.float64,
}
_NAME_BY_TORCH_DTYPE = {dtype: name for name, dtype in _TORCH_DTYPE_BY_NAME.items()}


def normalize_dtype_name(name: str) -> str:
    """Return the canonical TOML dtype name, rejecting unsupported values."""

    if not isinstance(name, str):
        raise TypeError(f"dtype name must be a string, got {type(name).__name__}")
    normalized = name.strip().lower()
    if normalized not in _TORCH_DTYPE_BY_NAME:
        choices = ", ".join(sorted(_TORCH_DTYPE_BY_NAME))
        raise ValueError(f"unsupported floating-point dtype {name!r}; expected one of: {choices}")
    return normalized


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Resolve a TOML-safe dtype name to its :class:`torch.dtype`."""

    return _TORCH_DTYPE_BY_NAME[normalize_dtype_name(name)]


def torch_dtype_name(dtype: torch.dtype | str) -> str:
    """Return the canonical config name for a supported dtype or dtype name."""

    if isinstance(dtype, str):
        return normalize_dtype_name(dtype)
    try:
        return _NAME_BY_TORCH_DTYPE[dtype]
    except (KeyError, TypeError) as exc:
        choices = ", ".join(sorted(_TORCH_DTYPE_BY_NAME))
        raise ValueError(f"unsupported floating-point dtype {dtype!r}; expected one of: {choices}") from exc


@dataclass
class PrecisionOptions:
    """Precision of dense model state and numerically sensitive accumulations.

    ``model_dtype`` controls parameters and dense E/Pi residual state.
    ``accumulator_dtype`` controls row offsets and small reductions. An
    accumulator may be wider than the model state, but never narrower.
    """

    model_dtype: str = "float32"
    accumulator_dtype: str = "float64"

    def validate(self, *, model_dtype: torch.dtype | str | None = None) -> None:
        """Normalize names and ensure the effective accumulator is not narrower.

        ``model_dtype`` is an optional explicit runtime override, such as the
        public ``GeneReconModel(dtype=...)`` argument. It participates in the
        width check without making the stored configuration non-serializable.
        """

        self.model_dtype = normalize_dtype_name(self.model_dtype)
        self.accumulator_dtype = normalize_dtype_name(self.accumulator_dtype)
        effective_model = resolve_torch_dtype(
            self.model_dtype if model_dtype is None else torch_dtype_name(model_dtype)
        )
        accumulator = resolve_torch_dtype(self.accumulator_dtype)
        if torch.finfo(accumulator).bits < torch.finfo(effective_model).bits:
            raise ValueError(
                "accumulator_dtype must be at least as wide as the resolved "
                f"model dtype ({self.accumulator_dtype} < {torch_dtype_name(effective_model)})"
            )

    @property
    def model_torch_dtype(self) -> torch.dtype:
        return resolve_torch_dtype(self.model_dtype)

    @property
    def accumulator_torch_dtype(self) -> torch.dtype:
        return resolve_torch_dtype(self.accumulator_dtype)

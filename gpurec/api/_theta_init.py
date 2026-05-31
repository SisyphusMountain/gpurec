"""Mode and theta normalization helpers for :class:`GeneReconModel`."""
from __future__ import annotations

from typing import Any

import torch

from ._model_types import _MODE_MAP


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _MODE_MAP:
        raise ValueError(f"Unknown mode {mode!r}. Valid: {sorted(_MODE_MAP)}")
    return normalized


def _mode_to_flags(mode: str) -> tuple[bool, bool]:
    return _MODE_MAP[_normalize_mode(mode)]


def _validate_gene_dtype(dtype: Any) -> torch.dtype:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"dtype must be torch.float32 or torch.float64, got {dtype!r}"
        )
    return dtype


def _default_theta_init(dataset: Any, mode: str) -> torch.Tensor:
    base = torch.log2(torch.tensor(1e-10, dtype=dataset.dtype, device=dataset.device))
    genewise, specieswise = _mode_to_flags(mode)
    if genewise:
        shape = (len(dataset.families), 3)
    elif specieswise:
        shape = (int(dataset.S), 3)
    else:
        shape = (3,)
    return torch.full(shape, base.item(), dtype=dataset.dtype, device=dataset.device)


def _expand_theta_base(
    theta_base: torch.Tensor | None,
    *,
    mode: str,
    species_count: int,
    family_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    if theta_base is None:
        return None

    theta_base = theta_base.to(device=device)
    if mode == "specieswise":
        return theta_base.unsqueeze(0).expand(species_count, -1).clone()
    if mode == "genewise":
        return theta_base.unsqueeze(0).expand(family_count, -1).clone()
    return theta_base

"""Private chunk layout helpers for :mod:`gpurec.api.uniform_chunked`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from gpurec.core.model import GeneDataset
from gpurec.core.origination import PreparedOriginationPrior


@dataclass(frozen=True)
class _UniformChunkSpec:
    indices: list[int]
    clades: int
    splits: int


@dataclass(frozen=True)
class _UniformBuiltChunk:
    spec: _UniformChunkSpec
    wave_layout: dict[str, Any]
    waves: int
    max_wave: int
    split_rows: int
    max_wave_split_rows: int


@dataclass
class _UniformChunkedState:
    dataset: GeneDataset
    species_helpers: dict[str, Any]
    ancestors_T: torch.Tensor | None
    unnorm_row_max: torch.Tensor
    built_chunks: list[_UniformBuiltChunk]
    device: torch.device
    dtype: torch.dtype
    origination_prior: PreparedOriginationPrior
    origination_probs: torch.Tensor | None = None
    fixed_iters_Pi: int = 6
    fixed_iters_E: int | None = None
    max_iters_E: int = 2000
    tol_E: float = 1e-8
    neumann_terms: int = 3
    use_pruning: bool = True
    pruning_threshold: float = 1e-6
    warm_start_E: bool = True
    profile: bool = False
    warm_E: torch.Tensor | None = None


def _dtype_name_for_rust(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise ValueError(f"unsupported fused Rust layout dtype: {dtype}")


def _move_wave_layout_to_device(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    if torch.is_tensor(value):
        if value.dtype.is_floating_point:
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, list):
        return [
            _move_wave_layout_to_device(item, device=device, dtype=dtype)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _move_wave_layout_to_device(item, device=device, dtype=dtype)
            for key, item in value.items()
        }
    return value


def _built_chunks_from_rust(
    chunk_payloads: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[_UniformBuiltChunk]:
    built_chunks: list[_UniformBuiltChunk] = []
    for payload in chunk_payloads:
        spec = _UniformChunkSpec(
            indices=[int(idx) for idx in payload["indices"]],
            clades=int(payload["clades"]),
            splits=int(payload["splits"]),
        )
        built_chunks.append(
            _UniformBuiltChunk(
                spec=spec,
                wave_layout=_move_wave_layout_to_device(
                    payload["wave_layout"],
                    device=device,
                    dtype=dtype,
                ),
                waves=int(payload["waves"]),
                max_wave=int(payload["max_wave"]),
                split_rows=int(payload["split_rows"]),
                max_wave_split_rows=int(payload["max_wave_split_rows"]),
            )
        )
    return built_chunks

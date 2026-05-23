"""Species-tree helper utilities shared by uniform forward/backward paths."""

from __future__ import annotations

from typing import Any

import torch

from .preprocess_rust import (
    rust_species_parent_from_helpers,
    rust_species_wave_topology,
    rust_uniform_ancestors_t_from_topology,
)


def _canonical_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def species_parent_from_helpers(
    species_helpers: dict,
    *,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Build CPU parent pointers from compact species child helper arrays."""
    return rust_species_parent_from_helpers(species_helpers, dtype=dtype)


def species_wave_topology(
    species_helpers: dict[str, Any],
    *,
    device: torch.device | str,
    S: int | None = None,
) -> dict[str, Any]:
    """Return cached species topology tensors used by wave kernels."""
    if S is None:
        S = int(species_helpers["S"])
    target_device = _canonical_device(device)
    cache = species_helpers.setdefault("_wave_species_topology_cache", {})
    key = (str(target_device), int(S))
    cached = cache.get(key)
    if isinstance(cached, dict):
        required = (
            "sp_child1",
            "sp_child2",
            "sp_parent",
            "sp_child1_cpu",
            "sp_child2_cpu",
            "sp_parent_cpu",
            "compact_level_ptr",
            "compact_level_parents",
            "compact_level_child1",
            "compact_level_child2",
        )
        if all(k in cached for k in required):
            device_tensors_ok = all(
                torch.is_tensor(cached[name]) and cached[name].device == target_device
                for name in (
                    "sp_child1",
                    "sp_child2",
                    "sp_parent",
                    "compact_level_ptr",
                    "compact_level_parents",
                    "compact_level_child1",
                    "compact_level_child2",
                )
            )
            cpu_tensors_ok = all(
                torch.is_tensor(cached[name]) and cached[name].device.type == "cpu"
                for name in ("sp_child1_cpu", "sp_child2_cpu", "sp_parent_cpu")
            )
            if device_tensors_ok and cpu_tensors_ok:
                return cached

    topology = rust_species_wave_topology(
        species_helpers,
        device=target_device,
        S=int(S),
    )
    cache[key] = topology
    return topology


def uniform_ancestors_t_from_topology(
    species_helpers: dict,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build sparse ``ancestors_dense.T`` from compact species parent pointers."""
    S = int(species_helpers["S"])
    target_device = _canonical_device(device)
    cache = species_helpers.setdefault("_uniform_ancestors_t_cache", {})
    key = (str(target_device), str(dtype), S)
    cached = cache.get(key)
    if torch.is_tensor(cached) and cached.device == target_device and cached.dtype == dtype:
        return cached

    ancestors_t = rust_uniform_ancestors_t_from_topology(
        species_helpers,
        device=target_device,
        dtype=dtype,
    )
    cache[key] = ancestors_t
    return ancestors_t

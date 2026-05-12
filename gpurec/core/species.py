"""Species-tree helper utilities shared by uniform forward/backward paths."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch


def _canonical_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def species_parent_from_helpers(species_helpers: dict, *, dtype: torch.dtype = torch.long) -> torch.Tensor:
    """Build CPU parent pointers from compact species child helper arrays."""
    S = int(species_helpers["S"])
    sp_p = species_helpers["s_P_indexes"].detach().cpu().long()
    sp_c = species_helpers["s_C12_indexes"].detach().cpu().long()
    mask_c1 = sp_p < S

    parent = torch.full((S,), -1, dtype=torch.long)
    parent[sp_c[mask_c1]] = sp_p[mask_c1]
    parent[sp_c[~mask_c1]] = sp_p[~mask_c1] - S
    return parent.to(dtype=dtype)


def _to_long_list(values: Any, name: str) -> list[int]:
    if torch.is_tensor(values):
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1D tensor")
        return [int(v) for v in values.detach().cpu().long().tolist()]
    return [int(v) for v in values]


def species_child_arrays_from_helpers(
    species_helpers: Mapping[str, Any],
    *,
    S: int | None = None,
) -> tuple[list[int], list[int]]:
    """Build child arrays from compact species helper indices."""
    if S is None:
        S = int(species_helpers["S"])
    if S < 0:
        raise ValueError("S must be non-negative")

    p_values = _to_long_list(species_helpers["s_P_indexes"], "s_P_indexes")
    c_values = _to_long_list(species_helpers["s_C12_indexes"], "s_C12_indexes")
    if len(p_values) != len(c_values):
        raise ValueError("s_P_indexes and s_C12_indexes must have the same length")

    child1 = [S] * S
    child2 = [S] * S
    for p, c in zip(p_values, c_values):
        if c < 0 or c >= S:
            raise ValueError(f"child index out of range: {c}")
        if 0 <= p < S:
            child1[p] = c
        elif S <= p < 2 * S:
            child2[p - S] = c
        else:
            raise ValueError(f"parent index out of range: {p}")
    return child1, child2


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

    sp_p = species_helpers["s_P_indexes"].detach().cpu().long()
    sp_c = species_helpers["s_C12_indexes"].detach().cpu().long()
    if sp_p.numel() != sp_c.numel():
        raise ValueError("s_P_indexes and s_C12_indexes must have the same length")
    mask_c1 = sp_p < S

    sp_child1_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child2_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child1_cpu[sp_p[mask_c1]] = sp_c[mask_c1]
    sp_child2_cpu[sp_p[~mask_c1] - S] = sp_c[~mask_c1]

    sp_parent_cpu = torch.full((S,), -1, dtype=torch.long)
    sp_parent_cpu[sp_c[mask_c1]] = sp_p[mask_c1]
    sp_parent_cpu[sp_c[~mask_c1]] = sp_p[~mask_c1] - S

    parent_values = sp_parent_cpu.tolist()
    max_ancestor_depth = 0
    for s_idx in range(S):
        depth = 0
        cur = s_idx
        while cur >= 0:
            depth += 1
            if depth > S:
                raise RuntimeError("Cycle detected in species parent pointers")
            cur = parent_values[cur]
        max_ancestor_depth = max(max_ancestor_depth, depth)

    child1_values = sp_child1_cpu.tolist()
    child2_values = sp_child2_cpu.tolist()
    levels = [-1] * S
    for s_idx in range(S):
        if levels[s_idx] >= 0:
            continue
        stack = [(s_idx, False)]
        while stack:
            node, expanded = stack.pop()
            if levels[node] >= 0:
                continue
            c1 = child1_values[node]
            c2 = child2_values[node]
            if not expanded:
                stack.append((node, True))
                if c2 < S and levels[c2] < 0:
                    stack.append((c2, False))
                if c1 < S and levels[c1] < 0:
                    stack.append((c1, False))
                continue
            child_levels = []
            if c1 < S:
                child_levels.append(levels[c1])
            if c2 < S:
                child_levels.append(levels[c2])
            levels[node] = (max(child_levels) + 1) if child_levels else 0

    max_level = max(levels) if levels else 0
    ptr_values = [0]
    flat_parents: list[int] = []
    flat_child1: list[int] = []
    flat_child2: list[int] = []
    for level in range(1, max_level + 1):
        parents = [
            s_idx for s_idx, node_level in enumerate(levels)
            if node_level == level
            and (child1_values[s_idx] < S or child2_values[s_idx] < S)
        ]
        flat_parents.extend(parents)
        flat_child1.extend(child1_values[parent] for parent in parents)
        flat_child2.extend(child2_values[parent] for parent in parents)
        ptr_values.append(len(flat_parents))
    if len(ptr_values) == 1:
        ptr_values.append(0)

    topology = {
        "S": int(S),
        "sp_child1_cpu": sp_child1_cpu,
        "sp_child2_cpu": sp_child2_cpu,
        "sp_parent_cpu": sp_parent_cpu,
        "sp_child1": sp_child1_cpu.to(device=target_device, dtype=torch.int32),
        "sp_child2": sp_child2_cpu.to(device=target_device, dtype=torch.int32),
        "sp_parent": sp_parent_cpu.to(device=target_device, dtype=torch.int32),
        "max_ancestor_depth": int(max_ancestor_depth),
        "compact_level_ptr": torch.tensor(
            ptr_values,
            dtype=torch.long,
            device=target_device,
        ).contiguous(),
        "compact_level_parents": torch.tensor(
            flat_parents,
            dtype=torch.int32,
            device=target_device,
        ).contiguous(),
        "compact_level_child1": torch.tensor(
            flat_child1,
            dtype=torch.int32,
            device=target_device,
        ).contiguous(),
        "compact_level_child2": torch.tensor(
            flat_child2,
            dtype=torch.int32,
            device=target_device,
        ).contiguous(),
    }
    cache[key] = topology
    return topology


def uniform_unnorm_row_max_from_topology(
    species_helpers: dict,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Compute ``log2(max_j Recipients_mat[i,j])`` without materializing ``[S,S]``.

    In uniform transfer mode, every non-ancestor recipient of species ``i`` has
    weight ``1 / (S - |ancestors(i)|)`` and ancestor recipients have weight 0.
    """
    S = int(species_helpers["S"])
    parent = species_parent_from_helpers(species_helpers)
    values = torch.empty((S,), dtype=torch.float64)
    for s_idx in range(S):
        depth = 0
        cur = s_idx
        while cur >= 0:
            depth += 1
            cur = int(parent[cur])
            if depth > S:
                raise RuntimeError("Cycle detected in species parent pointers")
        n_recipients = S - depth
        if n_recipients <= 0:
            values[s_idx] = float("-inf")
        else:
            values[s_idx] = -math.log2(float(n_recipients))
    return values.to(dtype=dtype)


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

    parent = species_parent_from_helpers(species_helpers)
    pairs: list[tuple[int, int]] = []
    for desc in range(S):
        depth = 0
        cur = desc
        while cur >= 0:
            pairs.append((cur, desc))
            depth += 1
            if depth > S:
                raise RuntimeError("Cycle detected in species parent pointers")
            cur = int(parent[cur])

    pairs.sort()
    rows, cols = zip(*pairs) if pairs else ([], [])
    indices = torch.tensor([rows, cols], dtype=torch.long, device=target_device)
    values = torch.ones((len(rows),), dtype=dtype, device=target_device)
    with torch.sparse.check_sparse_tensor_invariants(False):
        ancestors_t = torch.sparse_coo_tensor(
            indices,
            values,
            (S, S),
            device=target_device,
            dtype=dtype,
            is_coalesced=True,
        )
    cache[key] = ancestors_t
    return ancestors_t

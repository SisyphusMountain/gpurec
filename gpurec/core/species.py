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

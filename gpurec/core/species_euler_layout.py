"""CPU diagnostics for species-tree Euler layouts.

The uniform-Pibar VJP tree kernels need subtree sums over the species
dimension.  These helpers measure whether the current species id order already
turns every subtree into a contiguous interval and provide a small exact
Euler-prefix prototype for parity tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class SpeciesEulerLayout:
    """DFS layout and current-order subtree fragmentation statistics."""

    S: int
    dfs_order: tuple[int, ...]
    subtree_start: tuple[int, ...]
    subtree_end: tuple[int, ...]
    current_interval_start: tuple[int, ...]
    current_interval_end: tuple[int, ...]
    current_interval_length: tuple[int, ...]
    subtree_size: tuple[int, ...]
    fragment_count: tuple[int, ...]
    excess_span: tuple[int, ...]
    max_fragment_count: int
    mean_fragment_count: float
    mean_excess_span: float
    contiguous_subtrees: int
    all_subtrees_contiguous: bool


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
    """Build `[S]` child arrays from `species_helpers`.

    Missing children use `S` as the sentinel, matching the backward kernels.
    The input layout is the existing pair:
    `s_P_indexes = [parent..., parent + S...]`,
    `s_C12_indexes = [child1..., child2...]`.
    """

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


def species_euler_layout_report(
    species_helpers: Mapping[str, Any] | None = None,
    *,
    sp_child1: Sequence[int] | torch.Tensor | None = None,
    sp_child2: Sequence[int] | torch.Tensor | None = None,
    S: int | None = None,
) -> SpeciesEulerLayout:
    """Compute DFS order and current-order subtree interval statistics.

    The current order is the existing species id order `0..S-1`; fragment
    counts are measured against that order.  A fragment count of one means the
    node's complete descendant set is already a contiguous current-order
    interval.
    """

    if species_helpers is not None:
        if sp_child1 is not None or sp_child2 is not None:
            raise ValueError("pass either species_helpers or child arrays, not both")
        child1, child2 = species_child_arrays_from_helpers(species_helpers, S=S)
        S = len(child1)
    else:
        if sp_child1 is None or sp_child2 is None:
            raise ValueError("species_helpers or both child arrays are required")
        child1 = _to_long_list(sp_child1, "sp_child1")
        child2 = _to_long_list(sp_child2, "sp_child2")
        if len(child1) != len(child2):
            raise ValueError("sp_child1 and sp_child2 must have the same length")
        if S is None:
            S = len(child1)
        if int(S) != len(child1):
            raise ValueError("S must match child array length")

    S = int(S)
    _validate_child_arrays(child1, child2, S)
    parent = _parent_array(child1, child2, S)
    roots = [s for s, p in enumerate(parent) if p < 0]
    if not roots and S:
        raise ValueError("species tree has no root")

    subtree_sets: list[set[int] | None] = [None] * S
    dfs_order: list[int] = []

    # Build deterministic postorder. Children retain the current left/right
    # topology, matching the layout produced by preprocessing.
    for root in roots:
        _postorder(root, child1, child2, S, dfs_order, subtree_sets)

    position = [0] * S
    for idx, species in enumerate(dfs_order):
        position[species] = idx

    subtree_start = [0] * S
    subtree_end = [0] * S
    current_start = [0] * S
    current_end = [0] * S
    interval_length = [0] * S
    subtree_size = [0] * S
    fragment_count = [0] * S
    excess_span = [0] * S

    for s in range(S):
        descendants = sorted(subtree_sets[s] or ())
        if not descendants:
            raise ValueError(f"species {s} was not reached from a root")

        dfs_positions = [position[d] for d in descendants]
        subtree_start[s] = min(dfs_positions)
        subtree_end[s] = max(dfs_positions) + 1

        current_start[s] = descendants[0]
        current_end[s] = descendants[-1] + 1
        interval_length[s] = current_end[s] - current_start[s]
        subtree_size[s] = len(descendants)
        fragment_count[s] = 1 + sum(
            1 for left, right in zip(descendants, descendants[1:]) if right != left + 1
        )
        excess_span[s] = interval_length[s] - subtree_size[s]

    max_frag = max(fragment_count, default=0)
    mean_frag = float(sum(fragment_count) / S) if S else 0.0
    mean_excess = float(sum(excess_span) / S) if S else 0.0
    contiguous = sum(1 for count in fragment_count if count == 1)

    return SpeciesEulerLayout(
        S=S,
        dfs_order=tuple(dfs_order),
        subtree_start=tuple(subtree_start),
        subtree_end=tuple(subtree_end),
        current_interval_start=tuple(current_start),
        current_interval_end=tuple(current_end),
        current_interval_length=tuple(interval_length),
        subtree_size=tuple(subtree_size),
        fragment_count=tuple(fragment_count),
        excess_span=tuple(excess_span),
        max_fragment_count=max_frag,
        mean_fragment_count=mean_frag,
        mean_excess_span=mean_excess,
        contiguous_subtrees=contiguous,
        all_subtrees_contiguous=(contiguous == S),
    )


def euler_subtree_sums(values: torch.Tensor, layout: SpeciesEulerLayout) -> torch.Tensor:
    """Exact subtree sums using the report's DFS intervals.

    `values` may be shaped `[S]` or `[..., S]`.  The returned tensor has the
    same shape and species order as `values`.
    """

    if values.shape[-1] != layout.S:
        raise ValueError(
            f"last values dimension must equal layout.S={layout.S}, got {values.shape[-1]}"
        )
    order = torch.tensor(layout.dfs_order, device=values.device, dtype=torch.long)
    start = torch.tensor(layout.subtree_start, device=values.device, dtype=torch.long)
    end = torch.tensor(layout.subtree_end, device=values.device, dtype=torch.long)

    ordered = values.index_select(-1, order)
    prefix = torch.cat(
        [torch.zeros_like(ordered[..., :1]), ordered.cumsum(dim=-1)],
        dim=-1,
    )
    return prefix.index_select(-1, end) - prefix.index_select(-1, start)


def reference_subtree_sums(
    values: torch.Tensor,
    sp_child1: Sequence[int] | torch.Tensor,
    sp_child2: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Tree-walk subtree sums in current species order for parity tests."""

    child1 = _to_long_list(sp_child1, "sp_child1")
    child2 = _to_long_list(sp_child2, "sp_child2")
    S = len(child1)
    if values.shape[-1] != S:
        raise ValueError(f"last values dimension must equal S={S}, got {values.shape[-1]}")
    _validate_child_arrays(child1, child2, S)

    result = values.clone()
    for node in _bottom_up_nodes(child1, child2, S):
        for child in (child1[node], child2[node]):
            if child < S:
                result[..., node] = result[..., node] + result[..., child]
    return result


def _validate_child_arrays(child1: Sequence[int], child2: Sequence[int], S: int) -> None:
    if len(child1) != S or len(child2) != S:
        raise ValueError("child arrays must have length S")
    seen_children: set[int] = set()
    for parent, children in enumerate(zip(child1, child2)):
        for child in children:
            if child == S:
                continue
            if child < 0 or child >= S:
                raise ValueError(f"child index out of range: {child}")
            if child == parent:
                raise ValueError(f"species {parent} cannot be its own child")
            if child in seen_children:
                raise ValueError(f"species {child} has multiple parents")
            seen_children.add(child)


def _parent_array(child1: Sequence[int], child2: Sequence[int], S: int) -> list[int]:
    parent = [-1] * S
    for p, children in enumerate(zip(child1, child2)):
        for child in children:
            if child < S:
                parent[child] = p
    return parent


def _postorder(
    root: int,
    child1: Sequence[int],
    child2: Sequence[int],
    S: int,
    order: list[int],
    subtree_sets: list[set[int] | None],
) -> set[int]:
    descendants = {root}
    for child in (child1[root], child2[root]):
        if child < S:
            descendants.update(_postorder(child, child1, child2, S, order, subtree_sets))
    subtree_sets[root] = descendants
    order.append(root)
    return descendants


def _bottom_up_nodes(
    child1: Sequence[int],
    child2: Sequence[int],
    S: int,
) -> list[int]:
    parent = _parent_array(child1, child2, S)
    roots = [s for s, p in enumerate(parent) if p < 0]
    order: list[int] = []
    subtree_sets: list[set[int] | None] = [None] * S
    for root in roots:
        _postorder(root, child1, child2, S, order, subtree_sets)
    return order

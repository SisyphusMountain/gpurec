"""Shared family batch planning for resident and chunked model APIs."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class FamilyBatchPlan:
    """One planned family batch before tensors and wave layouts are built."""

    indices: list[int]
    clades: int
    splits: int


def normalize_batch_packing(value: str | None) -> str:
    """Normalize user-facing batch-packing aliases."""
    if value is None:
        return "sequential"
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "sequential",
        "sequential": "sequential",
        "contiguous": "sequential",
        "input_order": "sequential",
        "clade_first_fit": "clade_first_fit",
        "first_fit_decreasing": "clade_first_fit",
        "ffd": "clade_first_fit",
        "clade_ffd": "clade_first_fit",
        "depth_first_fit": "depth_first_fit",
        "depth_ffd": "depth_first_fit",
        "critical_path_first_fit": "depth_first_fit",
        "critical_first_fit": "depth_first_fit",
        "wave_first_fit": "depth_first_fit",
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            "batch_packing must be 'sequential', 'clade_first_fit', or "
            f"'depth_first_fit', got {value!r}"
        ) from exc


def normalize_clade_budget(value: int | None) -> int | None:
    """Normalize an optional positive clade budget."""
    if value is None:
        return None
    budget = int(value)
    if budget <= 0:
        raise ValueError("clade_budget must be positive when provided")
    return budget


def _selected_indices(
    *,
    indices: Sequence[int] | None,
    total: int | None,
    clade_counts: Sequence[int],
) -> list[int]:
    if indices is not None:
        return [int(idx) for idx in indices]
    if total is None:
        total = len(clade_counts)
    return list(range(int(total)))


def _require_indexed_stats(
    name: str,
    values: Sequence[int] | None,
    selected: Sequence[int],
) -> Sequence[int]:
    if values is None:
        raise ValueError(f"batch_packing='depth_first_fit' requires {name}")
    if selected:
        required = max(selected) + 1
        if len(values) < required:
            raise ValueError(f"{name} must cover selected family indices")
    return values


def _plan_from_chunks(
    chunks: Sequence[Sequence[int]],
    clade_counts: Sequence[int],
    split_counts: Sequence[int] | None,
) -> list[FamilyBatchPlan]:
    plans: list[FamilyBatchPlan] = []
    for chunk in chunks:
        indices = [int(idx) for idx in chunk]
        plans.append(
            FamilyBatchPlan(
                indices=indices,
                clades=sum(int(clade_counts[idx]) for idx in indices),
                splits=(
                    0
                    if split_counts is None
                    else sum(int(split_counts[idx]) for idx in indices)
                ),
            )
        )
    return plans


def plan_family_batches(
    *,
    clade_counts: Sequence[int],
    family_chunk_size: int,
    clade_budget: int | None,
    batch_packing: str = "sequential",
    indices: Sequence[int] | None = None,
    total: int | None = None,
    split_counts: Sequence[int] | None = None,
    leaf_counts: Sequence[int] | None = None,
    nonleaf_counts: Sequence[int] | None = None,
    schedule_depths: Sequence[int] | None = None,
    max_wave_size: int | None = None,
) -> list[FamilyBatchPlan]:
    """Plan family batches using a shared packing implementation.

    ``indices`` may select a subset of families, but all count arrays are
    indexed by the original family index.
    """
    selected = _selected_indices(
        indices=indices,
        total=total,
        clade_counts=clade_counts,
    )
    family_limit = int(family_chunk_size)
    if family_limit < 0:
        raise ValueError("family_chunk_size must be non-negative")

    packing = normalize_batch_packing(batch_packing)
    budget = normalize_clade_budget(clade_budget)

    if packing == "clade_first_fit":
        if budget is None:
            raise ValueError("batch_packing='clade_first_fit' requires clade_budget")
        chunks: list[list[int]] = []
        chunk_clades: list[int] = []
        order = sorted(selected, key=lambda idx: int(clade_counts[idx]), reverse=True)
        for idx in order:
            n_clades = int(clade_counts[idx])
            best_j: int | None = None
            best_remaining: int | None = None
            for j, current_clades in enumerate(chunk_clades):
                if family_limit > 0 and len(chunks[j]) >= family_limit:
                    continue
                remaining = budget - current_clades - n_clades
                if remaining < 0:
                    continue
                if best_remaining is None or remaining < best_remaining:
                    best_j = j
                    best_remaining = remaining
            if best_j is None:
                chunks.append([int(idx)])
                chunk_clades.append(n_clades)
            else:
                chunks[best_j].append(int(idx))
                chunk_clades[best_j] += n_clades
        return _plan_from_chunks(chunks, clade_counts, split_counts)

    if packing == "depth_first_fit":
        if budget is None:
            raise ValueError("batch_packing='depth_first_fit' requires clade_budget")
        leaves = _require_indexed_stats("leaf_counts", leaf_counts, selected)
        nonleaves = _require_indexed_stats("nonleaf_counts", nonleaf_counts, selected)
        depths = _require_indexed_stats("schedule_depths", schedule_depths, selected)

        wave_cap = (
            sum(int(clade_counts[idx]) for idx in selected)
            if max_wave_size is None
            else int(max_wave_size)
        )
        if wave_cap <= 0:
            raise ValueError("max_wave_size must be positive")

        def lower_bound(leaves_count: int, nonleaves_count: int, depth: int) -> int:
            leaf_waves = math.ceil(int(leaves_count) / wave_cap)
            work_waves = math.ceil(int(nonleaves_count) / wave_cap)
            return leaf_waves + max(int(depth), work_waves)

        chunks: list[list[int]] = []
        chunk_clades: list[int] = []
        chunk_leaves: list[int] = []
        chunk_nonleaves: list[int] = []
        chunk_depths: list[int] = []
        order = sorted(
            selected,
            key=lambda idx: (int(depths[idx]), int(clade_counts[idx])),
            reverse=True,
        )
        for idx in order:
            n_clades = int(clade_counts[idx])
            n_leaves = int(leaves[idx])
            n_nonleaves = int(nonleaves[idx])
            depth = int(depths[idx])
            best_j: int | None = None
            best_key: tuple[int, int, int] | None = None
            for j, current_clades in enumerate(chunk_clades):
                if family_limit > 0 and len(chunks[j]) >= family_limit:
                    continue
                new_clades = current_clades + n_clades
                if new_clades > budget:
                    continue
                before = lower_bound(
                    chunk_leaves[j],
                    chunk_nonleaves[j],
                    chunk_depths[j],
                )
                after = lower_bound(
                    chunk_leaves[j] + n_leaves,
                    chunk_nonleaves[j] + n_nonleaves,
                    max(chunk_depths[j], depth),
                )
                remaining = budget - new_clades
                key = (after - before, after, remaining)
                if best_key is None or key < best_key:
                    best_j = j
                    best_key = key
            if best_j is None:
                chunks.append([int(idx)])
                chunk_clades.append(n_clades)
                chunk_leaves.append(n_leaves)
                chunk_nonleaves.append(n_nonleaves)
                chunk_depths.append(depth)
            else:
                chunks[best_j].append(int(idx))
                chunk_clades[best_j] += n_clades
                chunk_leaves[best_j] += n_leaves
                chunk_nonleaves[best_j] += n_nonleaves
                chunk_depths[best_j] = max(chunk_depths[best_j], depth)
        return _plan_from_chunks(chunks, clade_counts, split_counts)

    chunks: list[list[int]] = []
    current: list[int] = []
    current_clades = 0

    def flush() -> None:
        nonlocal current, current_clades
        if current:
            chunks.append(list(current))
            current = []
            current_clades = 0

    for idx in selected:
        n_clades = int(clade_counts[idx])
        family_cap_hit = family_limit > 0 and len(current) >= family_limit
        clade_cap_hit = (
            budget is not None
            and current
            and current_clades + n_clades > budget
        )
        if family_cap_hit or clade_cap_hit:
            flush()
        current.append(int(idx))
        current_clades += n_clades
    flush()
    return _plan_from_chunks(chunks, clade_counts, split_counts)


__all__ = [
    "FamilyBatchPlan",
    "normalize_batch_packing",
    "normalize_clade_budget",
    "plan_family_batches",
]

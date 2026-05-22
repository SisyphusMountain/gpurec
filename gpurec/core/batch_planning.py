"""Shared family batch planning for resident and chunked model APIs.

This module is a narrow low-level support boundary for in-repo API, workflow,
CLI, memory-policy, and white-box test callers.  The exported names are shared
planning helpers, not a promise that the rest of ``gpurec.core`` is stable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import os
from typing import Sequence


_SCHEDULER_BACKEND_ENV = "GPUREC_SCHEDULER_BACKEND"


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


def _normalize_int_control(name: str, value: int | float | str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(number)
    raise ValueError(f"{name} must be an integer")


def normalize_clade_budget(value: int | float | str | None) -> int | None:
    """Normalize an optional positive clade budget."""
    if value is None:
        return None
    budget = _normalize_int_control("clade_budget", value)
    if budget <= 0:
        raise ValueError("clade_budget must be positive when provided")
    return budget


def normalize_family_chunk_size(
    value: int | float | str | None,
    *,
    allow_auto: bool = False,
) -> int | str:
    """Normalize family chunk-size controls shared by APIs and workflows."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "all", "none", "null"}:
            return 0
        if text == "auto":
            if allow_auto:
                return "auto"
            raise ValueError(
                "family chunk size 'auto' is not supported; use 0 for one "
                "resident batch or a positive integer"
            )
    size = _normalize_int_control("family_chunk_size", value)
    if size < 0:
        raise ValueError("family_chunk_size must be non-negative")
    return size


def _selected_indices(
    *,
    indices: Sequence[int] | None,
    total: int | None,
    clade_counts: Sequence[int],
) -> list[int]:
    if indices is not None:
        return [
            _normalize_int_control("family index", idx)
            for idx in indices
        ]
    if total is None:
        total = len(clade_counts)
    total_int = _normalize_int_control("total", total)
    if total_int < 0:
        raise ValueError("total must be non-negative")
    return list(range(total_int))


def _validate_selected_indices(selected: Sequence[int], clade_counts: Sequence[int]) -> None:
    limit = len(clade_counts)
    seen: set[int] = set()
    for position, idx in enumerate(selected):
        if idx < 0 or idx >= limit:
            raise ValueError(
                f"family index {idx} at selected position {position} is outside "
                f"valid range [0, {limit})"
            )
        if idx in seen:
            raise ValueError(f"duplicate family index {idx} at selected position {position}")
        seen.add(idx)


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


def _plan_family_batches_python(
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
    _validate_selected_indices(selected, clade_counts)
    family_limit = _normalize_int_control("family_chunk_size", family_chunk_size)
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
            else _normalize_int_control("max_wave_size", max_wave_size)
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
    """Plan family batches using the configured scheduling backend."""
    backend = os.environ.get(_SCHEDULER_BACKEND_ENV, "rust").strip().lower()
    if backend in {"python", "py"}:
        return _plan_family_batches_python(
            clade_counts=clade_counts,
            family_chunk_size=family_chunk_size,
            clade_budget=clade_budget,
            batch_packing=batch_packing,
            indices=indices,
            total=total,
            split_counts=split_counts,
            leaf_counts=leaf_counts,
            nonleaf_counts=nonleaf_counts,
            schedule_depths=schedule_depths,
            max_wave_size=max_wave_size,
        )
    if backend in {"", "rust"}:
        from gpurec.core.schedule_rust import plan_family_batches as rust_plan

        return [
            FamilyBatchPlan(
                indices=list(plan["indices"]),
                clades=int(plan["clades"]),
                splits=int(plan["splits"]),
            )
            for plan in rust_plan(
                clade_counts=clade_counts,
                family_chunk_size=family_chunk_size,
                clade_budget=clade_budget,
                batch_packing=batch_packing,
                indices=indices,
                total=total,
                split_counts=split_counts,
                leaf_counts=leaf_counts,
                nonleaf_counts=nonleaf_counts,
                schedule_depths=schedule_depths,
                max_wave_size=max_wave_size,
            )
        ]
    raise ValueError(
        f"{_SCHEDULER_BACKEND_ENV} must be 'python' or 'rust', got {backend!r}"
    )


__all__ = [
    "FamilyBatchPlan",
    "normalize_batch_packing",
    "normalize_clade_budget",
    "normalize_family_chunk_size",
    "plan_family_batches",
]

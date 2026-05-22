"""Shared family batch planning for resident and chunked model APIs.

This module is a narrow low-level support boundary for in-repo API, workflow,
CLI, memory-policy, and white-box test callers.  The exported names are shared
planning helpers, not a promise that the rest of ``gpurec.core`` is stable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
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
    """Plan family batches using the Rust scheduler backend."""
    from gpurec.core.schedule_rust import plan_family_batches as rust_plan

    rust_plans = rust_plan(
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
    return [
        FamilyBatchPlan(
            indices=list(plan["indices"]),
            clades=int(plan["clades"]),
            splits=int(plan["splits"]),
        )
        for plan in rust_plans
    ]


__all__ = [
    "FamilyBatchPlan",
    "normalize_batch_packing",
    "normalize_clade_budget",
    "normalize_family_chunk_size",
    "plan_family_batches",
]

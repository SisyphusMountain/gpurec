"""Python bridge for Rust wave scheduling."""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from typing import Any, Sequence

import torch

from .preprocess_rust import _load_native_module


def _long_list(value: Any) -> list[int]:
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]


def _integer_value(name: str, value: Any) -> int:
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
        if math.isfinite(number) and number.is_integer():
            return int(number)
    raise ValueError(f"{name} must be an integer")


def _optional_integer_value(name: str, value: Any | None) -> int | None:
    if value is None:
        return None
    return _integer_value(name, value)


def _schedule_item(item: dict[str, Any]) -> dict[str, Any]:
    ccp = item["ccp"]
    request_ccp = {
        "C": int(ccp["C"]),
        "N_splits": int(ccp["N_splits"]),
        "split_parents_sorted": _long_list(ccp["split_parents_sorted"]),
        "split_leftrights_sorted": _long_list(ccp["split_leftrights_sorted"]),
        "root_clade_id": int(ccp["root_clade_id"]),
    }
    if "split_counts" in ccp:
        request_ccp["split_counts"] = _long_list(ccp["split_counts"])
    return {"ccp": request_ccp}


def family_schedule_summary(ccp: dict[str, Any]) -> dict[str, int]:
    """Return Rust-computed per-family scheduling stats."""
    module = _load_native_module()
    request_ccp = _schedule_item({"ccp": ccp})["ccp"]
    output = json.loads(module.family_schedule_summary_json(json.dumps(request_ccp)))
    return {key: int(value) for key, value in output.items()}


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
) -> list[dict[str, Any]]:
    """Return Rust-computed family batch plans with the Python planner contract."""
    request = {
        "clade_counts": [
            _integer_value("clade_counts entries", value)
            for value in clade_counts
        ],
        "family_chunk_size": _integer_value("family_chunk_size", family_chunk_size),
        "clade_budget": _optional_integer_value("clade_budget", clade_budget),
        "batch_packing": "sequential" if batch_packing is None else str(batch_packing),
        "indices": (
            None
            if indices is None
            else [_integer_value("family index", index) for index in indices]
        ),
        "total": _optional_integer_value("total", total),
        "split_counts": (
            None
            if split_counts is None
            else [
                _integer_value("split_counts entries", value)
                for value in split_counts
            ]
        ),
        "leaf_counts": (
            None
            if leaf_counts is None
            else [
                _integer_value("leaf_counts entries", value)
                for value in leaf_counts
            ]
        ),
        "nonleaf_counts": (
            None
            if nonleaf_counts is None
            else [
                _integer_value("nonleaf_counts entries", value)
                for value in nonleaf_counts
            ]
        ),
        "schedule_depths": (
            None
            if schedule_depths is None
            else [
                _integer_value("schedule_depths entries", value)
                for value in schedule_depths
            ]
        ),
        "max_wave_size": _optional_integer_value("max_wave_size", max_wave_size),
    }
    module = _load_native_module()
    output = json.loads(module.plan_family_batches_json(json.dumps(request)))
    return [
        {
            "indices": [int(index) for index in plan["indices"]],
            "clades": int(plan["clades"]),
            "splits": int(plan["splits"]),
        }
        for plan in output
    ]


def schedule_global_phased_waves(
    items: Sequence[dict[str, Any]],
    family_clade_offsets: Sequence[int],
    *,
    max_wave_size: int | None,
    max_root_wave_size: int | None = None,
    max_dts_partial_rows: int | None = None,
    dts_partial_tile_splits: int = 64,
) -> tuple[list[list[int]], list[int]]:
    """Return Rust-computed phased waves with the Python scheduler contract."""
    request = {
        "items": [_schedule_item(item) for item in items],
        "family_clade_offsets": [int(offset) for offset in family_clade_offsets],
        "max_wave_size": None if max_wave_size is None else int(max_wave_size),
        "max_root_wave_size": (
            None if max_root_wave_size is None else int(max_root_wave_size)
        ),
        "max_dts_partial_rows": (
            None if max_dts_partial_rows is None else int(max_dts_partial_rows)
        ),
        "dts_partial_tile_splits": int(dts_partial_tile_splits),
    }
    module = _load_native_module()
    output = json.loads(module.schedule_global_phased_waves_json(json.dumps(request)))
    return (
        [[int(clade) for clade in wave] for wave in output["waves"]],
        [int(phase) for phase in output["phases"]],
    )

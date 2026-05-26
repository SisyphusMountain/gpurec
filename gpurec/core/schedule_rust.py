"""Python bridge for Rust wave scheduling."""

from __future__ import annotations

import json
from typing import Any, Sequence

import torch

from gpurec._validation import integer_value

from .preprocess_rust import _load_native_module


class RustSchedulerBackendUnavailable(RuntimeError):
    """Raised when the Rust scheduler native extension cannot be loaded."""


def _load_scheduler_native_module():
    try:
        return _load_native_module()
    except Exception as exc:
        raise RustSchedulerBackendUnavailable(str(exc)) from exc


def _native_json(call, payload: str) -> Any:
    try:
        return json.loads(call(payload))
    except RuntimeError as exc:
        message = str(exc)
        prefix = "invalid input: "
        if message.startswith(prefix):
            raise ValueError(message[len(prefix):]) from exc
        raise


def _long_list(value: Any) -> list[int]:
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]


def _integer_value(name: str, value: Any) -> int:
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    return integer_value(name, value)


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
    module = _load_scheduler_native_module()
    request_ccp = _schedule_item({"ccp": ccp})["ccp"]
    output = _native_json(
        module.family_schedule_summary_json,
        json.dumps(request_ccp),
    )
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
    small_family_max_leaves: int | None = None,
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
        "small_family_max_leaves": _optional_integer_value(
            "small_family_max_leaves",
            small_family_max_leaves,
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
    module = _load_scheduler_native_module()
    output = _native_json(module.plan_family_batches_json, json.dumps(request))
    return [
        {
            "indices": [int(index) for index in plan["indices"]],
            "clades": int(plan["clades"]),
            "splits": int(plan["splits"]),
        }
        for plan in output
    ]


def build_wave_layout_plan(
    *,
    waves: Sequence[Sequence[int]],
    phases: Sequence[int],
    c: int,
    n_splits: int,
    split_leftrights_sorted: Any,
    split_parents_sorted: Any,
    leaf_row_index: Any,
    leaf_col_index: Any,
    root_clade_ids: Any,
    family_clade_counts: Sequence[int] | None = None,
    family_clade_offsets: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return Rust-computed wave-layout index metadata."""
    request = {
        "waves": [[int(clade) for clade in wave] for wave in waves],
        "phases": [int(phase) for phase in phases],
        "c": int(c),
        "n_splits": int(n_splits),
        "split_leftrights_sorted": _long_list(split_leftrights_sorted),
        "split_parents_sorted": _long_list(split_parents_sorted),
        "leaf_row_index": _long_list(leaf_row_index),
        "leaf_col_index": _long_list(leaf_col_index),
        "root_clade_ids": _long_list(root_clade_ids),
        "family_clade_counts": (
            None
            if family_clade_counts is None
            else [int(count) for count in family_clade_counts]
        ),
        "family_clade_offsets": (
            None
            if family_clade_offsets is None
            else [int(offset) for offset in family_clade_offsets]
        ),
    }
    module = _load_scheduler_native_module()
    return _native_json(module.build_wave_layout_plan_json, json.dumps(request))


def schedule_global_phased_waves(
    items: Sequence[dict[str, Any]],
    family_clade_offsets: Sequence[int],
    *,
    max_wave_size: int | None,
    max_root_wave_size: int | None = None,
    max_dts_partial_rows: int | None = None,
    dts_partial_tile_splits: int = 64,
    nonleaf_schedule_policy: str = "auto",
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
        "nonleaf_schedule_policy": str(nonleaf_schedule_policy),
    }
    module = _load_scheduler_native_module()
    output = _native_json(module.schedule_global_phased_waves_json, json.dumps(request))
    return (
        [[int(clade) for clade in wave] for wave in output["waves"]],
        [int(phase) for phase in output["phases"]],
    )

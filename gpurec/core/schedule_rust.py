"""Python bridge for Rust wave scheduling."""

from __future__ import annotations

import json
from typing import Any, Sequence

import torch

from .preprocess_rust import _load_native_module


def _long_list(value: Any) -> list[int]:
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]


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

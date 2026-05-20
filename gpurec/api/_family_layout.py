from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from gpurec.api._validation import integer_value
from gpurec.core.batching import (
    build_wave_layout,
    collate_gene_families,
    schedule_global_phased_waves,
)
from gpurec.core.model import GeneDataset


@dataclass(frozen=True)
class FamilyWaveInputs:
    family_indices: list[int]
    items: tuple[dict[str, Any], ...]
    family_clade_counts: list[int]
    family_clade_offsets: list[int]
    root_clade_rows: list[int]
    clade_count: int
    split_count: int


@dataclass(frozen=True)
class FamilyWaveLayout:
    inputs: FamilyWaveInputs
    batched: dict[str, Any]
    waves: list[list[int]]
    phases: list[int]
    wave_layout: dict[str, Any]


def origination_probs_for_family_indices(
    origination_probs: torch.Tensor | None,
    family_indices: Sequence[int],
) -> torch.Tensor | None:
    if origination_probs is None or origination_probs.ndim == 1:
        return origination_probs
    idx = torch.as_tensor(
        family_indices,
        dtype=torch.long,
        device=origination_probs.device,
    )
    return origination_probs.index_select(0, idx)


def family_wave_inputs(
    dataset: GeneDataset,
    family_indices: Sequence[int],
) -> FamilyWaveInputs:
    indices = [
        integer_value("family_indices entries", index)
        for index in family_indices
    ]
    items: list[dict[str, Any]] = []
    family_clade_counts: list[int] = []
    family_clade_offsets: list[int] = []
    root_clade_rows: list[int] = []
    clade_offset = 0
    split_count = 0

    for family_idx in indices:
        family = dataset.families[family_idx]
        items.append(
            {
                "ccp": family["ccp_helpers"],
                "leaf_row_index": family["leaf_row_index"],
                "leaf_col_index": family["leaf_col_index"],
                "root_clade_id": int(family["root_clade_id"]),
            }
        )
        C_i = int(family["C"])
        family_clade_counts.append(C_i)
        family_clade_offsets.append(clade_offset)
        root_clade_rows.append(int(family["root_clade_id"]) + clade_offset)
        clade_offset += C_i
        split_count += int(family["N_splits"])

    return FamilyWaveInputs(
        family_indices=indices,
        items=tuple(items),
        family_clade_counts=family_clade_counts,
        family_clade_offsets=family_clade_offsets,
        root_clade_rows=root_clade_rows,
        clade_count=clade_offset,
        split_count=split_count,
    )


def schedule_family_waves(
    inputs: FamilyWaveInputs,
    *,
    max_wave_size: int | None,
    max_root_wave_size: int | None,
    max_dts_partial_rows: int | None = None,
) -> tuple[list[list[int]], list[int]]:
    return schedule_global_phased_waves(
        list(inputs.items),
        inputs.family_clade_offsets,
        max_wave_size=max_wave_size,
        max_root_wave_size=max_root_wave_size,
        max_dts_partial_rows=max_dts_partial_rows,
    )


def build_family_wave_layout(
    inputs: FamilyWaveInputs,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    max_wave_size: int | None = None,
    max_root_wave_size: int | None = None,
    max_dts_partial_rows: int | None = None,
    waves: list[list[int]] | None = None,
    phases: list[int] | None = None,
) -> FamilyWaveLayout:
    if (waves is None) != (phases is None):
        raise ValueError("waves and phases must be provided together")

    batched = collate_gene_families(list(inputs.items), dtype=dtype, device=device)
    if waves is None or phases is None:
        waves, phases = schedule_family_waves(
            inputs,
            max_wave_size=max_wave_size,
            max_root_wave_size=max_root_wave_size,
            max_dts_partial_rows=max_dts_partial_rows,
        )

    wave_layout = build_wave_layout(
        waves=waves,
        phases=phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device,
        dtype=dtype,
        family_clade_counts=inputs.family_clade_counts,
        family_clade_offsets=inputs.family_clade_offsets,
    )
    return FamilyWaveLayout(
        inputs=inputs,
        batched=batched,
        waves=waves,
        phases=phases,
        wave_layout=wave_layout,
    )

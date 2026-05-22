from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gpurec.api._family_layout import (
    build_family_wave_layout,
    family_wave_inputs,
    origination_probs_for_family_indices,
    schedule_family_waves,
)


def _ccp(C: int, parents: list[int], lefts: list[int], rights: list[int], root: int):
    counts = [0] * C
    for parent in parents:
        counts[parent] += 1
    return {
        "C": C,
        "N_splits": len(parents),
        "num_segs_eq1": len(parents),
        "end_rows_ge2": 0,
        "split_counts": torch.tensor(counts, dtype=torch.long),
        "split_parents_sorted": torch.tensor(parents, dtype=torch.long),
        "split_leftrights_sorted": torch.tensor(lefts + rights, dtype=torch.long),
        "log_split_probs_sorted": torch.zeros(len(parents), dtype=torch.float64),
        "root_clade_id": root,
    }


def _dataset():
    families = [
        {
            "C": 3,
            "N_splits": 1,
            "root_clade_id": 0,
            "ccp_helpers": _ccp(3, [0], [1], [2], root=0),
            "leaf_row_index": torch.tensor([1, 2], dtype=torch.long),
            "leaf_col_index": torch.tensor([0, 1], dtype=torch.long),
        },
        {
            "C": 2,
            "N_splits": 1,
            "root_clade_id": 1,
            "ccp_helpers": _ccp(2, [1], [0], [0], root=1),
            "leaf_row_index": torch.tensor([0], dtype=torch.long),
            "leaf_col_index": torch.tensor([1], dtype=torch.long),
        },
    ]
    return SimpleNamespace(families=families, device=torch.device("cpu"), dtype=torch.float64)


def test_origination_probs_for_family_indices_preserves_shared_distribution():
    shared = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)

    assert origination_probs_for_family_indices(None, [2, 0]) is None
    assert origination_probs_for_family_indices(shared, [2, 0]) is shared


def test_origination_probs_for_family_indices_selects_family_rows_on_input_device():
    family_probs = torch.tensor(
        [
            [0.10, 0.20, 0.70],
            [0.25, 0.25, 0.50],
            [0.30, 0.30, 0.40],
        ],
        dtype=torch.float64,
    )

    selected = origination_probs_for_family_indices(family_probs, [2, 0])

    torch.testing.assert_close(selected, family_probs[[2, 0]])
    assert selected.device == family_probs.device
    assert selected.dtype == family_probs.dtype


def test_family_wave_inputs_uses_caller_family_order_and_local_offsets():
    inputs = family_wave_inputs(_dataset(), [1, 0])

    assert inputs.family_indices == [1, 0]
    assert [item["root_clade_id"] for item in inputs.items] == [1, 0]
    assert inputs.family_clade_counts == [2, 3]
    assert inputs.family_clade_offsets == [0, 2]
    assert inputs.root_clade_rows == [1, 2]
    assert inputs.clade_count == 5
    assert inputs.split_count == 2


@pytest.mark.parametrize("family_index", [True, 1.5])
def test_family_wave_inputs_rejects_bool_and_nonintegral_indices(
    family_index: object,
):
    with pytest.raises(ValueError, match="family_indices entries"):
        family_wave_inputs(_dataset(), [family_index])  # type: ignore[list-item]


@pytest.mark.parametrize("family_index", [-1, 2])
def test_family_wave_inputs_rejects_out_of_range_indices(family_index: int):
    with pytest.raises(IndexError, match="family index"):
        family_wave_inputs(_dataset(), [family_index])


def test_family_wave_inputs_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="duplicate family index 0"):
        family_wave_inputs(_dataset(), [0, 0])


def test_build_family_wave_layout_matches_inputs_and_wave_order():
    inputs = family_wave_inputs(_dataset(), [1, 0])
    layout = build_family_wave_layout(
        inputs,
        device="cpu",
        dtype=torch.float64,
        max_wave_size=2,
        max_root_wave_size=None,
    )

    assert layout.inputs is inputs
    assert int(layout.batched["ccp"]["C"]) == inputs.clade_count
    assert int(layout.batched["ccp"]["N_splits"]) == inputs.split_count
    assert [
        int(value)
        for value in layout.batched["root_clade_ids"].detach().cpu().tolist()
    ] == inputs.root_clade_rows
    assert len(layout.waves) == len(layout.phases)
    assert layout.wave_layout["C"] == inputs.clade_count
    assert layout.wave_layout["family_idx"].shape == (inputs.clade_count,)
    root_ids = [
        int(value)
        for value in layout.wave_layout["root_clade_ids"].detach().cpu().tolist()
    ]
    assert len(root_ids) == len(inputs.family_indices)
    assert all(0 <= root_id < inputs.clade_count for root_id in root_ids)


def test_schedule_family_waves_can_use_rust_backend(monkeypatch):
    inputs = family_wave_inputs(_dataset(), [1, 0])
    monkeypatch.setenv("GPUREC_SCHEDULER_BACKEND", "rust")

    waves, phases = schedule_family_waves(
        inputs,
        max_wave_size=2,
        max_root_wave_size=None,
    )

    assert waves == [[0, 3], [4], [1, 2]]
    assert phases == [1, 1, 3]


def test_schedule_family_waves_rejects_unknown_backend(monkeypatch):
    inputs = family_wave_inputs(_dataset(), [0])
    monkeypatch.setenv("GPUREC_SCHEDULER_BACKEND", "nope")

    with pytest.raises(ValueError, match="GPUREC_SCHEDULER_BACKEND"):
        schedule_family_waves(
            inputs,
            max_wave_size=2,
            max_root_wave_size=None,
        )


def test_build_family_wave_layout_requires_waves_and_phases_together():
    inputs = family_wave_inputs(_dataset(), [0])

    with pytest.raises(ValueError, match="waves and phases"):
        build_family_wave_layout(
            inputs,
            device="cpu",
            dtype=torch.float64,
            waves=[[1], [2], [0]],
        )

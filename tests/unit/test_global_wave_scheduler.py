from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

from gpurec.api.uniform_chunked import _make_chunks
from gpurec.api.model import _family_index_chunks
from gpurec.core.batch_planning import plan_family_batches
from gpurec.core.batching import (
    _family_schedule_data,
    _schedule_deadline_nonleaf_waves,
    _select_nonleaf_schedule_candidate,
    build_wave_layout,
    collate_gene_families,
    schedule_global_phased_waves,
)

ROOT = Path(__file__).resolve().parents[2]
SUBPROCESS_TIMEOUT = 30


def _ccp(C, parents, lefts, rights, root):
    counts = [0] * C
    for parent in parents:
        counts[parent] += 1
    return {
        "C": C,
        "N_splits": len(parents),
        "split_counts": torch.tensor(counts, dtype=torch.long),
        "split_parents_sorted": torch.tensor(parents, dtype=torch.long),
        "split_leftrights_sorted": torch.tensor(lefts + rights, dtype=torch.long),
        "root_clade_id": root,
    }


def test_core_runtime_contracts_do_not_use_asserts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    sources = [
        ROOT / "gpurec/core/batching.py",
        ROOT / "gpurec/core/forward.py",
    ]

    offenders = [
        str(path)
        for path in sources
        if "assert " in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_collate_gene_families_validates_split_lengths_under_optimized_python():
    code = textwrap.dedent(
        """
        import torch
        from gpurec.core.batching import collate_gene_families

        item = {
            "ccp": {
                "C": 3,
                "N_splits": 1,
                "num_segs_eq1": 1,
                "end_rows_ge2": 0,
                "split_leftrights_sorted": torch.tensor([1], dtype=torch.long),
                "log_split_probs_sorted": torch.zeros(1),
                "split_parents_sorted": torch.tensor([0], dtype=torch.long),
            },
            "leaf_row_index": torch.tensor([1, 2], dtype=torch.long),
            "leaf_col_index": torch.tensor([0, 1], dtype=torch.long),
            "root_clade_id": 0,
        }

        try:
            collate_gene_families([item])
        except ValueError as exc:
            if "split_leftrights_sorted" not in str(exc):
                raise
        else:
            raise SystemExit("expected explicit ValueError")
        """
    )

    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_collate_gene_families_rejects_empty_batch():
    with pytest.raises(ValueError, match="at least one family"):
        collate_gene_families([])


def test_build_wave_layout_rejects_duplicate_clade_coverage():
    ccp = {
        "C": 2,
        "N_splits": 0,
        "split_parents_sorted": torch.empty(0, dtype=torch.long),
        "split_leftrights_sorted": torch.empty(0, dtype=torch.long),
        "log_split_probs_sorted": torch.empty(0, dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="duplicate clade 0"):
        build_wave_layout(
            [[0, 0]],
            [1],
            ccp,
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            device="cpu",
            dtype=torch.float32,
        )


@pytest.mark.parametrize("phases", [[], [1, 2]])
def test_build_wave_layout_rejects_phase_count_mismatch(phases):
    ccp = {
        "C": 1,
        "N_splits": 0,
        "split_parents_sorted": torch.empty(0, dtype=torch.long),
        "split_leftrights_sorted": torch.empty(0, dtype=torch.long),
        "log_split_probs_sorted": torch.empty(0, dtype=torch.float32),
    }

    with pytest.raises(
        ValueError,
        match="waves and phases must have matching lengths",
    ):
        build_wave_layout(
            [[0]],
            phases,
            ccp,
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            device="cpu",
            dtype=torch.float32,
        )


def _valid_two_clade_layout_ccp():
    return {
        "C": 2,
        "N_splits": 1,
        "split_parents_sorted": torch.tensor([0], dtype=torch.long),
        "split_leftrights_sorted": torch.tensor([1, 1], dtype=torch.long),
        "log_split_probs_sorted": torch.zeros(1, dtype=torch.float32),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "split_leftrights_sorted",
            torch.tensor([1, 2], dtype=torch.long),
            "split_leftrights_sorted contains clade 2 at position 1",
        ),
        (
            "split_parents_sorted",
            torch.tensor([-1], dtype=torch.long),
            "split_parents_sorted contains clade -1 at position 0",
        ),
    ],
)
def test_build_wave_layout_rejects_invalid_split_clade_ids(field, value, message):
    ccp = _valid_two_clade_layout_ccp()
    ccp[field] = value

    with pytest.raises(ValueError, match=message):
        build_wave_layout(
            [[1], [0]],
            [1, 3],
            ccp,
            torch.tensor([1], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            device="cpu",
            dtype=torch.float32,
        )


@pytest.mark.parametrize(
    ("leaf_row_index", "root_clade_ids", "message"),
    [
        (
            torch.tensor([-1], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            "leaf_row_index contains clade -1 at position 0",
        ),
        (
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            "root_clade_ids contains clade 2 at position 0",
        ),
    ],
)
def test_build_wave_layout_rejects_invalid_leaf_or_root_clade_ids(
    leaf_row_index,
    root_clade_ids,
    message,
):
    with pytest.raises(ValueError, match=message):
        build_wave_layout(
            [[1], [0]],
            [1, 3],
            _valid_two_clade_layout_ccp(),
            leaf_row_index,
            torch.tensor([0], dtype=torch.long),
            root_clade_ids,
            device="cpu",
            dtype=torch.float32,
        )


def test_build_wave_layout_rejects_leaf_row_col_length_mismatch():
    with pytest.raises(
        ValueError,
        match="leaf_col_index has length 0 but expected 1",
    ):
        build_wave_layout(
            [[1], [0]],
            [1, 3],
            _valid_two_clade_layout_ccp(),
            torch.tensor([1], dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            device="cpu",
            dtype=torch.float32,
        )


@pytest.mark.parametrize(
    ("family_clade_counts", "family_clade_offsets", "message"),
    [
        ([2, 2], [0], "must have matching lengths"),
        ([2, 2], None, "must be provided together"),
        ([2, -1], [0, 2], r"family_clade_counts\[1\] must be non-negative"),
        ([2, 2], [0, -1], r"family_clade_offsets\[1\] must be non-negative"),
        ([2, 2], [0, 3], r"family 1 clade range \[3, 5\) is outside C=4"),
        ([2, 2], [0, 1], "overlaps clade 1"),
        ([2, 1], [0, 2], "does not cover clade 3"),
    ],
)
def test_build_wave_layout_rejects_invalid_family_clade_metadata(
    family_clade_counts,
    family_clade_offsets,
    message,
):
    ccp = {
        "C": 4,
        "N_splits": 0,
        "split_parents_sorted": torch.empty(0, dtype=torch.long),
        "split_leftrights_sorted": torch.empty(0, dtype=torch.long),
        "log_split_probs_sorted": torch.empty(0, dtype=torch.float32),
    }

    with pytest.raises(ValueError, match=message):
        build_wave_layout(
            [[0, 1, 2, 3]],
            [1],
            ccp,
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            device="cpu",
            dtype=torch.float32,
            family_clade_counts=family_clade_counts,
            family_clade_offsets=family_clade_offsets,
        )


def _assert_topological(waves, offsets, items):
    wave_of = {clade: wave_idx for wave_idx, wave in enumerate(waves) for clade in wave}
    for offset, item in zip(offsets, items):
        ccp = item["ccp"]
        parents = ccp["split_parents_sorted"].tolist()
        lefts = ccp["split_leftrights_sorted"][: len(parents)].tolist()
        rights = ccp["split_leftrights_sorted"][len(parents) :].tolist()
        for parent, left, right in zip(parents, lefts, rights):
            parent_wave = wave_of[offset + parent]
            assert wave_of[offset + left] < parent_wave
            assert wave_of[offset + right] < parent_wave


def _winning_nonleaf_policy(
    items,
    *,
    max_wave_size,
    max_root_wave_size=None,
    max_dts_partial_rows=None,
    dts_partial_tile_splits=64,
):
    families = [_family_schedule_data(item["ccp"]) for item in items]
    policy, batches = _select_nonleaf_schedule_candidate(
        families,
        wave_cap=max_wave_size,
        root_cap=max_root_wave_size,
        max_dts_partial_rows=max_dts_partial_rows,
        dts_partial_tile_splits=dts_partial_tile_splits,
    )
    return policy, batches


def _assert_nonleaf_batches_cover_once(families, batches, *, max_wave_size):
    scheduled = [node for batch in batches for node in batch]
    expected = [
        (fi, c)
        for fi, family in enumerate(families)
        for c, split_count in enumerate(family["split_counts"])
        if int(split_count) != 0
    ]

    assert sorted(scheduled) == sorted(expected)
    assert all(len(batch) <= max_wave_size for batch in batches)

    wave_of = {
        (fi, c): wave_idx
        for wave_idx, batch in enumerate(batches)
        for fi, c in batch
    }
    for fi, family in enumerate(families):
        for parent, children in enumerate(family["children"]):
            if int(family["split_counts"][parent]) == 0:
                continue
            parent_wave = wave_of[(fi, parent)]
            for child in children:
                if int(family["split_counts"][child]) != 0:
                    assert wave_of[(fi, child)] < parent_wave


def test_global_scheduler_candidate_policy_keeps_forward_when_it_is_optimal():
    items = [
        {"ccp": _ccp(4, [0, 1], [1, 3], [2, 3], root=0)},
        {"ccp": _ccp(4, [0, 1], [1, 3], [2, 3], root=0)},
    ]

    policy, batches = _winning_nonleaf_policy(items, max_wave_size=4)

    assert policy == "forward"
    assert batches == [[(0, 1), (1, 1)], [(0, 0), (1, 0)]]
    _assert_nonleaf_batches_cover_once(
        [_family_schedule_data(item["ccp"]) for item in items],
        batches,
        max_wave_size=4,
    )


def test_global_scheduler_candidate_policy_uses_deadline_when_it_wins():
    parents = [0, 0, 1, 0, 2, 1, 4, 1, 5, 8, 1, 4, 7, 5, 1, 2, 7, 4, 5, 2, 3, 0, 1, 2]
    lefts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 6, 5, 9, 8, 9, 7, 8, 5, 8, 3, 8, 7, 8, 6]
    rights = [3, 2, 5, 5, 7, 7, 7, 8, 9, 9, 9, 7, 8, 6, 8, 8, 8, 7, 9, 6, 9, 3, 8, 8]
    items = [{"ccp": _ccp(10, parents, lefts, rights, root=0)}]

    policy, batches = _winning_nonleaf_policy(items, max_wave_size=2)

    assert policy == "deadline"
    assert len(batches) == 5
    _assert_nonleaf_batches_cover_once(
        [_family_schedule_data(item["ccp"]) for item in items],
        batches,
        max_wave_size=2,
    )


def test_global_scheduler_candidate_policy_uses_coffman_graham_when_it_wins():
    parents = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 6]
    lefts = [7, 3, 3, 6, 7, 7, 6, 6, 4, 5, 5, 7, 7]
    rights = [7, 6, 3, 4, 5, 6, 3, 4, 4, 5, 7, 7, 7]
    items = [{"ccp": _ccp(8, parents, lefts, rights, root=0)}]

    policy, batches = _winning_nonleaf_policy(items, max_wave_size=2)

    assert policy == "coffman_graham"
    assert batches == [[(0, 6), (0, 4)], [(0, 3), (0, 1)], [(0, 2), (0, 0)]]
    _assert_nonleaf_batches_cover_once(
        [_family_schedule_data(item["ccp"]) for item in items],
        batches,
        max_wave_size=2,
    )


def test_global_scheduler_packs_ready_clades_after_leaf_phase():
    # Two identical tiny DAGs:
    # root 0 depends on internal 1 and leaf 2; internal 1 depends on leaf 3.
    items = [
        {"ccp": _ccp(4, [0, 1], [1, 3], [2, 3], root=0)},
        {"ccp": _ccp(4, [0, 1], [1, 3], [2, 3], root=0)},
    ]
    waves, phases = schedule_global_phased_waves(
        items,
        [0, 4],
        max_wave_size=4,
    )

    assert phases == [1, 2, 3]
    assert waves[0] == [2, 3, 6, 7]
    assert waves[1] == [1, 5]
    assert waves[2] == [0, 4]
    _assert_topological(waves, [0, 4], items)


def test_global_scheduler_respects_cap_and_topological_order():
    items = [
        {"ccp": _ccp(4, [0, 1], [1, 3], [2, 3], root=0)},
        {"ccp": _ccp(4, [0, 1], [1, 3], [2, 3], root=0)},
    ]
    offsets = [0, 4]
    waves, _phases = schedule_global_phased_waves(
        items,
        offsets,
        max_wave_size=2,
    )

    assert all(len(wave) <= 2 for wave in waves)
    wave_of = {clade: wave_idx for wave_idx, wave in enumerate(waves) for clade in wave}
    assert sorted(wave_of) == list(range(8))
    _assert_topological(waves, offsets, items)


def test_global_scheduler_rejects_split_counts_that_disagree_with_parents():
    ccp = _ccp(3, [0], [1], [2], root=0)
    ccp["split_counts"] = torch.tensor([0, 1, 0], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="split_counts does not match split_parents_sorted",
    ):
        schedule_global_phased_waves([{"ccp": ccp}], [0], max_wave_size=2)


@pytest.mark.parametrize("parent_id", [-1, 3])
def test_global_scheduler_rejects_invalid_parent_clade_ids(parent_id):
    ccp = {
        "C": 3,
        "N_splits": 1,
        "split_counts": torch.tensor([1, 0, 0], dtype=torch.long),
        "split_parents_sorted": torch.tensor([parent_id], dtype=torch.long),
        "split_leftrights_sorted": torch.tensor([1, 2], dtype=torch.long),
        "root_clade_id": 0,
    }

    with pytest.raises(
        ValueError,
        match=f"split_parents_sorted contains parent {parent_id}",
    ):
        schedule_global_phased_waves([{"ccp": ccp}], [0], max_wave_size=2)


@pytest.mark.parametrize("root_id", [-1, 3])
def test_global_scheduler_rejects_invalid_root_clade_id(root_id):
    ccp = _ccp(3, [0], [1], [2], root=root_id)

    with pytest.raises(ValueError, match="root_clade_id"):
        schedule_global_phased_waves([{"ccp": ccp}], [0], max_wave_size=2)


@pytest.mark.parametrize("child_id", [-1, 3])
def test_global_scheduler_rejects_invalid_child_clade_ids(child_id):
    ccp = _ccp(3, [0], [child_id], [2], root=0)

    with pytest.raises(
        ValueError,
        match="split_leftrights_sorted contains child .* outside valid range",
    ):
        schedule_global_phased_waves([{"ccp": ccp}], [0], max_wave_size=2)


def test_global_scheduler_splits_root_waves_by_root_cap():
    items = [
        {"ccp": _ccp(2, [0], [1], [1], root=0)},
        {"ccp": _ccp(2, [0], [1], [1], root=0)},
        {"ccp": _ccp(2, [0], [1], [1], root=0)},
    ]
    offsets = [0, 2, 4]

    waves, phases = schedule_global_phased_waves(
        items,
        offsets,
        max_wave_size=6,
        max_root_wave_size=1,
    )

    assert waves == [[1, 3, 5], [0], [2], [4]]
    assert phases == [1, 3, 3, 3]
    _assert_topological(waves, offsets, items)


def test_global_scheduler_preserves_phase_barriers_and_root_only_cap():
    items = [
        {"ccp": _ccp(5, [0, 1, 1], [1, 2, 3], [4, 3, 4], root=0)},
        {"ccp": _ccp(3, [0], [1], [2], root=0)},
        {"ccp": _ccp(2, [0], [1], [1], root=0)},
    ]
    offsets = [0, 5, 8]

    waves, phases = schedule_global_phased_waves(
        items,
        offsets,
        max_wave_size=3,
        max_root_wave_size=1,
    )

    assert waves == [[2, 3, 4], [6, 7, 9], [1, 5, 8], [0]]
    assert phases == [1, 1, 2, 3]
    assert phases == sorted(phases)
    _assert_topological(waves, offsets, items)

    clades = {}
    for item, offset in zip(items, offsets):
        ccp = item["ccp"]
        root_id = int(ccp["root_clade_id"])
        for local_id, split_count in enumerate(ccp["split_counts"].tolist()):
            clades[offset + local_id] = {
                "is_leaf": int(split_count) == 0,
                "is_root": local_id == root_id,
            }

    for wave, phase in zip(waves, phases):
        if phase == 1:
            assert all(clades[clade]["is_leaf"] for clade in wave)
        else:
            assert all(not clades[clade]["is_leaf"] for clade in wave)
        if phase == 3:
            assert len(wave) <= 1
            assert all(clades[clade]["is_root"] for clade in wave)

    assert clades[8]["is_root"]
    assert 8 in waves[2]
    assert any(not clades[clade]["is_root"] for clade in waves[2])


def test_global_scheduler_uses_deadline_compaction_when_forward_greedy_wastes_wave():
    # This CCP-like DAG has two leaves and eight non-leaf clades.  A pure
    # bottom-up ready queue takes six non-leaf waves at cap=2, but the
    # deadline-search candidate packs the same DAG in five non-leaf waves.
    parents = [0, 0, 1, 0, 2, 1, 4, 1, 5, 8, 1, 4, 7, 5, 1, 2, 7, 4, 5, 2, 3, 0, 1, 2]
    lefts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 6, 5, 9, 8, 9, 7, 8, 5, 8, 3, 8, 7, 8, 6]
    rights = [3, 2, 5, 5, 7, 7, 7, 8, 9, 9, 9, 7, 8, 6, 8, 8, 8, 7, 9, 6, 9, 3, 8, 8]
    items = [{"ccp": _ccp(10, parents, lefts, rights, root=0)}]

    waves, phases = schedule_global_phased_waves(items, [0], max_wave_size=2)

    assert len(waves) == 6
    assert phases == [1, 2, 2, 2, 2, 3]
    assert all(len(wave) <= 2 for wave in waves)
    assert sorted(clade for wave in waves for clade in wave) == list(range(10))
    _assert_topological(waves, [0], items)


def test_global_scheduler_uses_layered_compaction_when_ready_order_wastes_wave():
    # The post-leaf graph has six non-leaf clades and cap=2, so three non-leaf
    # waves are possible.  Forward and reverse ready-queue schedules both need
    # four non-leaf waves on this DAG unless we assign clades by layered
    # first-fit compaction.
    parents = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 6]
    lefts = [7, 3, 3, 6, 7, 7, 6, 6, 4, 5, 5, 7, 7]
    rights = [7, 6, 3, 4, 5, 6, 3, 4, 4, 5, 7, 7, 7]
    items = [{"ccp": _ccp(8, parents, lefts, rights, root=0)}]

    waves, phases = schedule_global_phased_waves(items, [0], max_wave_size=2)

    assert waves == [[5, 7], [6, 4], [3, 1], [2, 0]]
    assert phases == [1, 2, 2, 2]
    _assert_topological(waves, [0], items)


def test_deadline_nonleaf_scheduler_respects_target_horizon():
    # Leaves 4 and 5 are handled first.  The remaining DAG has two bottom
    # clades and two top clades, so cap=2 fits exactly two non-leaf waves.
    item = {"ccp": _ccp(6, [0, 1, 2, 3], [2, 3, 4, 4], [4, 5, 5, 5], root=0)}
    families = [_family_schedule_data(item["ccp"])]

    candidate = _schedule_deadline_nonleaf_waves(
        families,
        wave_cap=2,
        target_waves=2,
        max_dts_partial_rows=None,
        dts_partial_tile_splits=64,
    )

    assert candidate == [[(0, 2), (0, 3)], [(0, 0), (0, 1)]]
    assert (
        _schedule_deadline_nonleaf_waves(
            families,
            wave_cap=2,
            target_waves=1,
            max_dts_partial_rows=None,
            dts_partial_tile_splits=64,
        )
        is None
    )


def test_global_scheduler_can_cap_dts_partial_rows_per_wave():
    parents = [0, 0, 0]
    lefts = [1, 3, 5]
    rights = [2, 4, 5]
    for i in range(10):
        parents.append(1)
        lefts.append(6 + i)
        rights.append(6 + ((i + 1) % 10))
    for parent, pairs in {
        2: [(6, 7), (8, 9)],
        3: [(10, 11), (12, 13)],
        4: [(14, 15), (6, 10)],
        5: [(7, 11), (8, 12)],
    }.items():
        for left, right in pairs:
            parents.append(parent)
            lefts.append(left)
            rights.append(right)
    items = [{"ccp": _ccp(16, parents, lefts, rights, root=0)}]

    uncapped, _ = schedule_global_phased_waves(items, [0], max_wave_size=16)
    capped, phases = schedule_global_phased_waves(
        items,
        [0],
        max_wave_size=16,
        max_dts_partial_rows=8,
        dts_partial_tile_splits=4,
    )

    assert len(uncapped) == 3
    assert len(capped) == 4
    assert phases == [1, 2, 2, 3]
    assert capped[1] == [1, 2]
    assert all(len(wave) <= 16 for wave in capped)
    assert sorted(clade for wave in capped for clade in wave) == list(range(16))
    _assert_topological(capped, [0], items)


def test_clade_first_fit_packs_non_contiguous_families():
    chunks = [
        plan.indices
        for plan in plan_family_batches(
            total=5,
            clade_counts=[8, 7, 6, 5, 4],
            family_chunk_size=0,
            clade_budget=12,
            batch_packing="clade_first_fit",
        )
    ]

    assert chunks == [[0, 4], [1, 3], [2]]
    assert all(
        sum([8, 7, 6, 5, 4][idx] for idx in chunk) <= 12
        for chunk in chunks
    )


def test_model_family_index_chunks_delegates_to_shared_planner():
    chunks = _family_index_chunks(
        total=5,
        clade_counts=[8, 7, 6, 5, 4],
        family_chunk_size=0,
        clade_budget=12,
        batch_packing="clade_first_fit",
    )

    assert chunks == [[0, 4], [1, 3], [2]]


@pytest.mark.parametrize("batch_packing", ["clade_first_fit", "depth_first_fit"])
def test_plan_family_batches_treats_clade_budget_as_soft_packing_target(
    batch_packing,
):
    kwargs = {
        "clade_counts": [20, 6, 5],
        "family_chunk_size": 0,
        "clade_budget": 12,
        "batch_packing": batch_packing,
    }
    if batch_packing == "depth_first_fit":
        kwargs.update(
            {
                "leaf_counts": [10, 3, 2],
                "nonleaf_counts": [10, 3, 3],
                "schedule_depths": [4, 2, 2],
                "max_wave_size": 4,
            }
        )

    plans = plan_family_batches(**kwargs)

    assert plans[0].indices == [0]
    assert plans[0].clades == 20
    assert plans[0].clades > 12
    assert [plan.indices for plan in plans[1:]] == [[1, 2]]
    assert plans[1].clades <= 12


@pytest.mark.parametrize("indices", [[-1], [2]])
def test_plan_family_batches_rejects_out_of_range_indices(indices):
    with pytest.raises(ValueError, match="family index"):
        plan_family_batches(
            indices=indices,
            clade_counts=[5, 7],
            family_chunk_size=0,
            clade_budget=None,
        )


@pytest.mark.parametrize("indices", [[True], [1.5]])
def test_plan_family_batches_rejects_nonintegral_indices(indices):
    with pytest.raises(ValueError, match="family index"):
        plan_family_batches(
            indices=indices,
            clade_counts=[5, 7],
            family_chunk_size=0,
            clade_budget=None,
        )


def test_plan_family_batches_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="duplicate family index"):
        plan_family_batches(
            indices=[0, 0],
            clade_counts=[5, 7],
            family_chunk_size=0,
            clade_budget=None,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total": True}, "total"),
        ({"family_chunk_size": True}, "family_chunk_size"),
        ({"family_chunk_size": 1.5}, "family_chunk_size"),
        (
            {
                "batch_packing": "depth_first_fit",
                "clade_budget": 12,
                "leaf_counts": [1, 1],
                "nonleaf_counts": [4, 6],
                "schedule_depths": [2, 3],
                "max_wave_size": 2.5,
            },
            "max_wave_size",
        ),
    ],
)
def test_plan_family_batches_rejects_nonintegral_controls(kwargs, message):
    arguments = {
        "clade_counts": [5, 7],
        "family_chunk_size": 0,
        "clade_budget": None,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        plan_family_batches(**arguments)


def test_depth_first_fit_groups_deep_families_under_clade_budget():
    chunks = [
        plan.indices
        for plan in plan_family_batches(
            total=5,
            clade_counts=[6, 6, 6, 6, 6],
            family_chunk_size=0,
            clade_budget=12,
            batch_packing="depth_first_fit",
            leaf_counts=[1, 1, 1, 1, 1],
            nonleaf_counts=[5, 5, 5, 5, 5],
            schedule_depths=[10, 9, 2, 1, 1],
            max_wave_size=8,
        )
    ]

    assert chunks == [[0, 1], [2, 3], [4]]


def test_plan_family_batches_can_use_rust_backend(monkeypatch):
    monkeypatch.setenv("GPUREC_SCHEDULER_BACKEND", "rust")

    plans = plan_family_batches(
        total=5,
        clade_counts=[6, 6, 6, 6, 6],
        split_counts=[10, 20, 30, 40, 50],
        family_chunk_size=0,
        clade_budget=12,
        batch_packing="depth_first_fit",
        leaf_counts=[1, 1, 1, 1, 1],
        nonleaf_counts=[5, 5, 5, 5, 5],
        schedule_depths=[10, 9, 2, 1, 1],
        max_wave_size=8,
    )

    assert [plan.indices for plan in plans] == [[0, 1], [2, 3], [4]]
    assert [plan.clades for plan in plans] == [12, 12, 6]
    assert [plan.splits for plan in plans] == [30, 70, 50]


def test_uniform_chunk_specs_use_shared_depth_first_fit_planner():
    specs = _make_chunks(
        [0, 1, 2, 3, 4],
        clade_counts=[6, 6, 6, 6, 6],
        split_counts=[10, 20, 30, 40, 50],
        family_chunk_size=0,
        clade_budget=12,
        batch_packing="depth_first_fit",
        leaf_counts=[1, 1, 1, 1, 1],
        nonleaf_counts=[5, 5, 5, 5, 5],
        schedule_depths=[10, 9, 2, 1, 1],
        max_wave_size=8,
    )

    assert [spec.indices for spec in specs] == [[0, 1], [2, 3], [4]]
    assert [spec.clades for spec in specs] == [12, 12, 6]
    assert [spec.splits for spec in specs] == [30, 70, 50]

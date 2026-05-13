import torch

from gpurec.api.model import _family_index_chunks
from gpurec.core.batching import schedule_global_phased_waves


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

    for offset, item in zip(offsets, items):
        ccp = item["ccp"]
        parents = ccp["split_parents_sorted"].tolist()
        lefts = ccp["split_leftrights_sorted"][: len(parents)].tolist()
        rights = ccp["split_leftrights_sorted"][len(parents) :].tolist()
        for parent, left, right in zip(parents, lefts, rights):
            parent_wave = wave_of[offset + parent]
            assert wave_of[offset + left] < parent_wave
            assert wave_of[offset + right] < parent_wave


def test_clade_first_fit_packs_non_contiguous_families():
    chunks = _family_index_chunks(
        total=5,
        clade_counts=[8, 7, 6, 5, 4],
        family_chunk_size=0,
        clade_budget=12,
        batch_packing="clade_first_fit",
    )

    assert chunks == [[0, 4], [1, 3], [2]]
    assert all(
        sum([8, 7, 6, 5, 4][idx] for idx in chunk) <= 12
        for chunk in chunks
    )

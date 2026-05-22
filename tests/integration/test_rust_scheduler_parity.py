from pathlib import Path

import torch

from gpurec.core.batching import schedule_global_phased_waves as py_schedule
from gpurec.core.model import parse_alerax_family_file
from gpurec.core.preprocess_rust import RustPreprocessExtension
from gpurec.core.schedule_rust import schedule_global_phased_waves as rust_schedule


ROOT = Path(__file__).resolve().parents[2]
HOGENOM_BENCH = ROOT / "tests" / "data" / "hogenom_bench"


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


def _assert_scheduler_parity(items, offsets, **kwargs):
    assert rust_schedule(items, offsets, **kwargs) == py_schedule(items, offsets, **kwargs)


def test_rust_scheduler_matches_python_ready_packing_and_root_caps():
    items = [
        {"ccp": _ccp(4, [0, 1], [1, 3], [2, 3], root=0)},
        {"ccp": _ccp(4, [0, 1], [1, 3], [2, 3], root=0)},
    ]
    _assert_scheduler_parity(items, [0, 4], max_wave_size=4)
    _assert_scheduler_parity(items, [0, 4], max_wave_size=2)

    root_items = [
        {"ccp": _ccp(2, [0], [1], [1], root=0)},
        {"ccp": _ccp(2, [0], [1], [1], root=0)},
        {"ccp": _ccp(2, [0], [1], [1], root=0)},
    ]
    _assert_scheduler_parity(
        root_items,
        [0, 2, 4],
        max_wave_size=6,
        max_root_wave_size=1,
    )


def test_rust_scheduler_matches_python_compaction_policies():
    deadline_parents = [
        0, 0, 1, 0, 2, 1, 4, 1, 5, 8, 1, 4,
        7, 5, 1, 2, 7, 4, 5, 2, 3, 0, 1, 2,
    ]
    deadline_lefts = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 6, 5,
        9, 8, 9, 7, 8, 5, 8, 3, 8, 7, 8, 6,
    ]
    deadline_rights = [
        3, 2, 5, 5, 7, 7, 7, 8, 9, 9, 9, 7,
        8, 6, 8, 8, 8, 7, 9, 6, 9, 3, 8, 8,
    ]
    _assert_scheduler_parity(
        [{"ccp": _ccp(10, deadline_parents, deadline_lefts, deadline_rights, root=0)}],
        [0],
        max_wave_size=2,
    )

    layered_parents = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 6]
    layered_lefts = [7, 3, 3, 6, 7, 7, 6, 6, 4, 5, 5, 7, 7]
    layered_rights = [7, 6, 3, 4, 5, 6, 3, 4, 4, 5, 7, 7, 7]
    _assert_scheduler_parity(
        [{"ccp": _ccp(8, layered_parents, layered_lefts, layered_rights, root=0)}],
        [0],
        max_wave_size=2,
    )


def test_rust_scheduler_matches_python_dts_partial_cap():
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

    _assert_scheduler_parity(
        [{"ccp": _ccp(16, parents, lefts, rights, root=0)}],
        [0],
        max_wave_size=16,
        max_dts_partial_rows=8,
        dts_partial_tile_splits=4,
    )


def test_rust_scheduler_matches_python_on_hogenom_bench_fixture():
    names, tree_paths, leaf_maps = parse_alerax_family_file(HOGENOM_BENCH / "families.txt")
    families = {name: paths for name, paths in zip(names, tree_paths)}
    leaf_species_maps = {
        name: mapping
        for name, mapping in zip(names, leaf_maps)
        if mapping
    }
    raw = RustPreprocessExtension().preprocess_multiple_families(
        str(HOGENOM_BENCH / "sp.nwk"),
        families,
        leaf_species_maps=leaf_species_maps,
        include_details=True,
        include_species_matrices=False,
        include_debug_details=False,
        include_scheduler_details=False,
        include_legacy_ccp_details=False,
        num_threads=8,
    )

    items = []
    offsets = []
    offset = 0
    for name in names:
        ccp = raw["families"][name]["ccp"]
        items.append({"ccp": ccp})
        offsets.append(offset)
        offset += int(ccp["C"])

    _assert_scheduler_parity(items, offsets, max_wave_size=8192)

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

from gpurec import (
    GeneReconModel,
    export_backtracking_input,
    recphyloxml_event_counts,
    sample_recphyloxml,
    sample_recphyloxmls,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "test_trees_3"
DATA_DIR_100 = Path(__file__).resolve().parents[1] / "data" / "test_trees_100"
TEST_TREES_3_EXPECTED_CLADES = 35
TEST_TREES_3_EXPECTED_SPECIES = 15
TEST_TREES_3_EXPECTED_LEAVES = 10


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_rust_stochastic_backtracking_exports_recphyloxml():
    model = GeneReconModel.from_trees(
        str(DATA_DIR / "sp.nwk"),
        [str(DATA_DIR / "g.nwk")],
        mode="global",
        device="cuda",
        dtype=torch.float64,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_E=8,
        fixed_iters_Pi=6,
    )

    payload = export_backtracking_input(model, family_index=0, seed=7, max_events=10_000)
    assert payload["pi"]["rows"] == TEST_TREES_3_EXPECTED_CLADES
    assert payload["pi"]["cols"] == TEST_TREES_3_EXPECTED_SPECIES
    assert payload["root_clade"] < payload["pi"]["rows"]

    xml = sample_recphyloxml(model, family_index=0, seed=7, max_events=10_000)
    root = ET.fromstring(xml)
    assert root.tag.endswith("recPhylo")
    assert xml.count("<recGeneTree>") == 1
    counts = recphyloxml_event_counts(xml)
    assert counts["Leaf"] == TEST_TREES_3_EXPECTED_LEAVES
    assert sum(counts[k] for k in ("S", "SL", "D", "DL", "T", "TL")) > 0


def _parse_alerax_event_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        counts[key.strip()] = int(float(value.strip()))
    return counts


@pytest.mark.skipif(
    not (DATA_DIR_100 / "output_global" / "reconciliations" / "all").exists(),
    reason="test_trees_100 AleRax reconciliation fixture not present",
)
def test_rust_backtracking_family_0000_matches_alerax_event_ranges():
    params_path = DATA_DIR_100 / "output_global" / "model_parameters" / "model_parameters.txt"
    first_rate_row = params_path.read_text().splitlines()[1].split()
    duplication, loss, transfer = map(float, first_rate_row[1:4])
    model = GeneReconModel.from_trees(
        str(DATA_DIR_100 / "sp.nwk"),
        [str(DATA_DIR_100 / "g_0000.nwk")],
        mode="global",
        device="cuda",
        dtype=torch.float64,
        theta_init_rates=(duplication, loss, transfer),
        fixed_iters_Pi=6,
        max_iters_E=4000,
        tol_E=1e-10,
    )

    alerax_counts = [
        _parse_alerax_event_counts(path)
        for path in sorted(
            (DATA_DIR_100 / "output_global" / "reconciliations" / "all").glob(
                "family_0000_eventCounts_*.txt"
            )
        )
    ]
    assert len(alerax_counts) == 100
    ranges = {
        key: (
            min(counts.get(key, 0) for counts in alerax_counts),
            max(counts.get(key, 0) for counts in alerax_counts),
        )
        for key in ("S", "SL", "D", "DL", "T", "TL", "L", "Leaf")
    }

    for seed, xml in enumerate(
        sample_recphyloxmls(model, family_index=0, num_samples=2, seed=0, max_events=50_000)
    ):
        counts = recphyloxml_event_counts(xml)
        for key, (lower, upper) in ranges.items():
            assert lower <= counts[key] <= upper, (seed, key, counts[key], lower, upper)

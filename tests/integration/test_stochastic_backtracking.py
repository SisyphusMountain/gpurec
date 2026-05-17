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
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "test_trees_3"
RUSTREE_DIR = Path(__file__).resolve().parents[2] / "rustree"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not RUSTREE_DIR.exists(), reason="local rustree checkout required")
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
    assert payload["pi"]["rows"] == 35
    assert payload["pi"]["cols"] == 15
    assert payload["root_clade"] < payload["pi"]["rows"]

    xml = sample_recphyloxml(model, family_index=0, seed=7, max_events=10_000)
    root = ET.fromstring(xml)
    assert root.tag.endswith("recPhylo")
    assert xml.count("<recGeneTree>") == 1
    counts = recphyloxml_event_counts(xml)
    assert counts["Leaf"] == 10
    assert sum(counts[k] for k in ("S", "SL", "D", "DL", "T", "TL")) > 0

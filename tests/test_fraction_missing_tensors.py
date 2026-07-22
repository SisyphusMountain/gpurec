# tests/test_fraction_missing_tensors.py
import math
import torch
import pytest
from gpurec.core.parameters.fraction_missing import (
    species_leaf_mask, build_fraction_missing_tensors,
)

NEG_INF = float("-inf")

def _tiny_species_helpers():
    # 3 species nodes in postorder: leaves 0,1 ; internal root 2. S=3.
    # sentinel for "no child" is S (==3). Root's children are leaves 0 and 1.
    S = 3
    return {
        "S": S,
        "sp_child1": torch.tensor([S, S, 0], dtype=torch.int32),
        "sp_child2": torch.tensor([S, S, 1], dtype=torch.int32),
        "sp_parent": torch.tensor([2, 2, -1], dtype=torch.int32),
    }

def test_leaf_mask_picks_leaves_via_sentinel():
    sh = _tiny_species_helpers()
    m = species_leaf_mask(sh)
    assert m.tolist() == [True, True, False]

def test_none_returns_none():
    sh = _tiny_species_helpers()
    assert build_fraction_missing_tensors(
        sh, fraction_missing=None, device="cpu", dtype=torch.float64) is None

def test_all_zero_returns_none():
    sh = _tiny_species_helpers()
    assert build_fraction_missing_tensors(
        sh, fraction_missing=0.0, device="cpu", dtype=torch.float64) is None

def test_scalar_applies_to_every_leaf_only():
    sh = _tiny_species_helpers()
    out = build_fraction_missing_tensors(
        sh, fraction_missing=0.3, device="cpu", dtype=torch.float64)
    assert out.shape == (3,)
    assert out[0].item() == pytest.approx(math.log2(0.3))
    assert out[1].item() == pytest.approx(math.log2(0.3))
    assert out[2].item() == NEG_INF  # internal species: never missing

def test_dict_by_name_maps_to_index():
    sh = _tiny_species_helpers()
    out = build_fraction_missing_tensors(
        sh, fraction_missing={"A": 0.5}, device="cpu", dtype=torch.float64,
        species_name_to_index={"A": 0, "B": 1})
    assert out[0].item() == pytest.approx(math.log2(0.5))
    assert out[1].item() == NEG_INF  # B absent from dict -> observed
    assert out[2].item() == NEG_INF

def test_tensor_per_species_ignores_internal():
    sh = _tiny_species_helpers()
    out = build_fraction_missing_tensors(
        sh, fraction_missing=[0.1, 0.2, 0.9], device="cpu", dtype=torch.float64)
    assert out[0].item() == pytest.approx(math.log2(0.1))
    assert out[1].item() == pytest.approx(math.log2(0.2))
    assert out[2].item() == NEG_INF  # index 2 is internal -> forced off

def test_out_of_range_raises():
    sh = _tiny_species_helpers()
    with pytest.raises(ValueError):
        build_fraction_missing_tensors(sh, fraction_missing=1.0, device="cpu", dtype=torch.float64)
    with pytest.raises(ValueError):
        build_fraction_missing_tensors(sh, fraction_missing=-0.1, device="cpu", dtype=torch.float64)

def test_no_clamp_used():
    import inspect, gpurec.core.parameters.fraction_missing as m
    assert "clamp" not in inspect.getsource(m)

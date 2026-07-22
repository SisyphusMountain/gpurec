# tests/test_model_fraction_missing_wiring.py
import math, torch, pytest
from gpurec.api.model import GeneReconModel

def _tiny(tmp_path):
    sp = tmp_path / "species.nwk"; sp.write_text("(A:1,B:1)Root;")
    g = tmp_path / "g.nwk"; g.write_text("(A_1:1,B_1:1)GeneRoot;")
    return str(sp), [str(g)]

def test_default_leaf_fm_log_is_none(tmp_path):
    sp, g = _tiny(tmp_path)
    m = GeneReconModel(sp, g, mode="global", device="cpu")
    assert m.leaf_fm_log is None
    assert all(s.leaf_fm_log is None for s in m.batch_statics)

def test_scalar_fraction_missing_populates_statics(tmp_path):
    sp, g = _tiny(tmp_path)
    m = GeneReconModel(sp, g, mode="global", device="cpu", fraction_missing=0.3)
    assert m.leaf_fm_log is not None
    assert m.leaf_fm_log.shape == (int(m.species_helpers["S"]),)
    # leaves carry log2(0.3); internal carries -inf
    leaves = m.leaf_fm_log[torch.isfinite(m.leaf_fm_log)]
    assert torch.allclose(leaves, torch.full_like(leaves, math.log2(0.3)))
    assert all(s.leaf_fm_log is m.leaf_fm_log for s in m.batch_statics)

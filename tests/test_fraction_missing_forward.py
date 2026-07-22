import torch, pytest
pytestmark = pytest.mark.gpu
from gpurec.api.model import GeneReconModel


def _tiny(tmp_path):
    sp = tmp_path / "species.nwk"; sp.write_text("((A:1,B:1)AB:1,C:1)Root;")
    g = tmp_path / "g.nwk"; g.write_text("((A_1:1,B_1:1)x:1,C_1:1)GeneRoot;")
    return str(sp), [str(g)]


def _nll(model):
    with torch.no_grad():
        return float(model().item())


def test_forward_fm0_equals_no_fm(tmp_path):
    sp, g = _tiny(tmp_path)
    a = _nll(GeneReconModel(sp, g, mode="global", device="cuda", dtype=torch.float64))
    b = _nll(GeneReconModel(sp, g, mode="global", device="cuda", dtype=torch.float64, fraction_missing=0.0))
    assert a == b


def test_forward_fm_changes_nll(tmp_path):
    sp, g = _tiny(tmp_path)
    base = _nll(GeneReconModel(sp, g, mode="global", device="cuda", dtype=torch.float64))
    fm = _nll(GeneReconModel(sp, g, mode="global", device="cuda", dtype=torch.float64, fraction_missing=0.3))
    assert abs(fm - base) > 1e-6

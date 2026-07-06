import pytest
from gpurax.recon.adapter import ReconBatch
from gpurax.recon.parity import recon_loglk

def test_batch_of_duplicates_is_uniform(fixture_dir):
    rb = ReconBatch(str(fixture_dir / "species.nwk"), device="cuda")
    paths = [str(fixture_dir / "gene.nwk")] * 3
    rates = [(0.2, 0.3, 0.1)] * 3
    lls = rb.score(paths, rates)
    assert len(lls) == 3
    assert lls[0] == pytest.approx(lls[1]) == pytest.approx(lls[2])

def test_batch_matches_single(fixture_dir):
    rb = ReconBatch(str(fixture_dir / "species.nwk"), device="cuda")
    ll_batch = rb.score([str(fixture_dir / "gene.nwk")], [(0.2, 0.3, 0.1)])[0]
    ll_single = recon_loglk(str(fixture_dir / "species.nwk"),
                            str(fixture_dir / "gene.nwk"), 0.2, 0.3, 0.1, device="cuda")
    assert ll_batch == pytest.approx(ll_single, rel=1e-5)

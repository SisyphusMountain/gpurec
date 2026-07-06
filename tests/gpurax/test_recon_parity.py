import pytest
from gpurax.recon.parity import recon_loglk

# gpurec's known value on the 4-taxon fixture at D=0.2,L=0.3,T=0.1 (measured; AleRax convention).
GPUREC_REF_NATS = -5.910620282602933

def test_recon_regression_and_finite(fixture_dir):
    sp, gt = str(fixture_dir / "species.nwk"), str(fixture_dir / "gene.nwk")
    ll = recon_loglk(sp, gt, 0.2, 0.3, 0.1, device="cuda")
    assert ll == pytest.approx(GPUREC_REF_NATS, rel=1e-5)   # pins gpurec behavior
    assert recon_loglk(sp, gt, 0.2, 0.3, 0.1, device="cuda") == pytest.approx(ll, rel=1e-9)  # deterministic

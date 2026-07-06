import pathlib

from gpurax.rates.optimize import optimize_global_rates
from gpurax.recon.parity import recon_loglk

FX = pathlib.Path(__file__).parent / "fixtures"


def test_fit_improves_recon():
    sp, gt = str(FX / "species.nwk"), str(FX / "gene.nwk")
    bad = recon_loglk(sp, gt, 0.9, 0.9, 0.9, device="cuda")
    D, L, T = optimize_global_rates(sp, [gt], init=(0.9, 0.9, 0.9), device="cuda")
    fit = recon_loglk(sp, gt, D, L, T, device="cuda")
    assert fit >= bad - 1e-6
    assert all(x > 0 for x in (D, L, T))

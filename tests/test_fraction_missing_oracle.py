import torch, pytest

pytestmark = pytest.mark.gpu

from gpurec.api.model import GeneReconModel
from tests._fraction_missing_oracle import oracle_nll


def test_wave_matches_cpu_oracle(tmp_path):
    sp = tmp_path / "s.nwk"
    sp.write_text("((A:1,B:1)AB:1,C:1)Root;")
    g = tmp_path / "g.nwk"
    g.write_text("((A_1:1,B_1:1)x:1,C_1:1)GeneRoot;")
    D = L = T = 0.1
    fm = 0.3

    # Independent CPU fixed-point oracle (pure torch, no Triton kernels).
    ref = oracle_nll(str(sp), str(g), D, L, T, fm)

    # Production wave path on CUDA.
    m = GeneReconModel(
        str(sp),
        [str(g)],
        mode="global",
        device="cuda",
        dtype=torch.float64,
        fraction_missing=fm,
    )
    # This constructor has no theta_init_rates knob: set theta directly.
    # theta = log2([D, L, T]); the softmax over [1, D, L, T] recovers pS/pD/pL/pT.
    with torch.no_grad():
        m.theta.copy_(
            torch.log2(torch.tensor([D, L, T], dtype=torch.float64, device=m.theta.device))
        )
    # Tighten the wave solver so it is fully converged for a converged-vs-converged check.
    m.configure_solver(e_tol=1e-13, e_max_iter=4000, pi_iters=256)

    with torch.no_grad():
        got = float(m().item())

    assert abs(got - ref) < 1e-6, f"wave={got!r} oracle={ref!r} delta={got - ref:.3e}"

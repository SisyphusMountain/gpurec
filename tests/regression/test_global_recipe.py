"""fit_global (the genewise 3x3 TR-Newton specialized to the single shared global block) must reach
the SAME optimum as the generic optimize() global path -- that equivalence is what justifies routing
global -> fit_global in fit_dtl. Also guards the dispatcher's mode validation."""
import pytest

pytest.importorskip("rustree")
torch = pytest.importorskip("torch")

from gpurec.fit.dtl_fit import fit_dtl


def test_fit_dtl_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        fit_dtl("sp.nwk", ["g.nwk"], "bogus")


@pytest.mark.gpu
def test_fit_global_matches_optimize(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from gpurec.bench.simulate import simulate_dataset
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.optimize import optimize, final_eval
    from gpurec.fit.global_fit import fit_global

    sp, genes = simulate_dataset("global", tmp_path, n_species=80, n_families=120, dtl=0.05, seed=11)

    # baseline: the generic optimize() global recipe
    m = GeneReconModel(sp, genes, mode="global", device="cuda", dtype=torch.float32,
                       solver_options=SolverOptions(e_adjoint_solver="neumann"))
    th, _ = optimize(m.batch_statics, m.theta.detach(), m.receiver_weights.detach(), verbose=False)
    nll_opt, _g = final_eval(m.batch_statics, th, m.receiver_weights.detach())
    rates_opt = (2.0 ** th.detach().float().cpu()).numpy().reshape(3)
    del m; torch.cuda.empty_cache()

    res = fit_global(sp, genes, device="cuda", dtype=torch.float32, verbose=False)
    rates_new = res["rates"].numpy().reshape(3)

    # same converged optimum: NLL to a few bits (both hit the fp32 solver floor), rates to <1% rel.
    assert abs(res["nll_bits"] - float(nll_opt)) < max(1e-4 * abs(float(nll_opt)), 1.0), \
        f"fit_global NLL {res['nll_bits']} vs optimize {float(nll_opt)}"
    import numpy as np
    rel = np.abs(rates_new - rates_opt).max() / (np.abs(rates_opt).mean() + 1e-30)
    assert rel < 1e-2, f"rates differ by {rel:.2e}: {rates_new} vs {rates_opt}"

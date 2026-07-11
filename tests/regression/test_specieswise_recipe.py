import pytest

pytest.importorskip("rustree")
torch = pytest.importorskip("torch")


def test_fit_specieswise_requires_lam():
    from gpurec.fit.specieswise_fit import fit_specieswise
    with pytest.raises(ValueError, match="requires an explicit prior"):
        # lam is keyword-only with no default -> omitting it is a TypeError; passing None -> ValueError
        fit_specieswise("bs", "th", "rw", lam=None)


@pytest.mark.gpu
def test_fit_specieswise_fits_at_given_lam(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from gpurec.bench.simulate import simulate_dataset
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.specieswise_fit import fit_specieswise
    import numpy as np

    sp, genes = simulate_dataset("specieswise", tmp_path, n_species=60, n_families=80,
                                 dtl=0.05, seed=5)
    m = GeneReconModel(sp, genes, mode="specieswise", device="cuda", dtype=torch.float32,
                       solver_options=SolverOptions(e_adjoint_solver="neumann"))
    S = m.theta.shape[0]
    res = fit_specieswise(m.batch_statics, m.theta.detach(), m.receiver_weights.detach(),
                          lam=10.0, verbose=False)
    rates = np.asarray(res["rates"])
    assert res["mode"] == "specieswise" and res["lam"] == 10.0
    assert rates.shape == (S, 3) and np.isfinite(res["nll_bits"]) and np.isfinite(rates).all()
    assert res["gnorm"] < 1.0, f"MAP not converged: |gF|={res['gnorm']}"

    # ridge sanity: a huge lam pins the optimum at theta_ref (a constant reference).
    theta_ref = torch.full((S, 3), float(np.log2(0.1)), device="cuda")
    res_big = fit_specieswise(m.batch_statics, m.theta.detach(), m.receiver_weights.detach(),
                              lam=1e6, theta_ref=theta_ref, verbose=False)
    drift = (res_big["theta"] - theta_ref.cpu()).abs().max().item()
    assert drift < 1e-1, f"large lam should hold theta near theta_ref, drift={drift}"

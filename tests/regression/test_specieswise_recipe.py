import pytest

pytest.importorskip("rustree")
torch = pytest.importorskip("torch")


def test_fit_dtl_raises_for_specieswise():
    from gpurec.fit.dtl_fit import fit_dtl
    with pytest.raises(NotImplementedError, match="fit_specieswise"):
        # must raise on mode alone -- no model build, no CUDA, dummy paths are never touched.
        fit_dtl("sp.nwk", ["g.nwk"], "specieswise")


def test_cli_fit_specieswise_exits_cleanly(capsys):
    import types
    from gpurec.cli import fit as cli_fit
    args = types.SimpleNamespace(mode="specieswise", species="sp.nwk", gene=["g.nwk"],
                                 device="cuda", dtype="float32", config=None, pi_iters=None,
                                 neumann_terms=None, e_max_iter=None, steps=300, init_rate=None, out=None)
    rc = cli_fit.run_fit(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "fit_specieswise" in err or "map_cv" in err   # the fit_dtl guidance was surfaced cleanly


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


@pytest.mark.gpu
def test_map_cv_smoke_uses_fit_specieswise(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import inspect
    from gpurec.fit import map_cv as mod
    # the rewire: map_cv's body calls fit_specieswise (not the removed L-BFGS fit_map path).
    assert "fit_specieswise" in inspect.getsource(mod.map_cv)

    from gpurec.bench.simulate import simulate_dataset
    sp, genes = simulate_dataset("specieswise", tmp_path, n_species=40, n_families=60,
                                 dtl=0.05, seed=3)
    out = mod.map_cv(sp, genes, k=2, lambdas=(1.0, 100.0), adam_steps=5, max_newton=4)
    import math
    assert out["lam_star"] in (1.0, 100.0)
    assert all(math.isfinite(v) for v in out["cv"].values())

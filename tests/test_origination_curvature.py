"""Analytic second-order for origination: joint (theta, alpha, omega) HVP correctness + consumers.

The omega (origination) logits enter only the NLL aggregation head, so their Hessian blocks are the
head's second derivatives composed with the existing forward-tangent/adjoint (no new kernels). These
tests verify, in fp64 on the tiny fixture: the joint HVP is symmetric, both softmax gauge nulls are
exact, the gauge-projected PD certificate matches the dense gauge-projected eigenvalue, and the
origination Fisher CG solve converges to finite standard errors.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.core.scheduling import batching

DT = torch.float64


def test_origination_information_mixed_accumulator_and_curvature_dtypes():
    from gpurec.solver.curvature.origination import origination_information

    static = SimpleNamespace(accumulator_dtype=torch.float32)
    info = origination_information(
        static,
        torch.zeros(1, dtype=torch.float32),
        torch.tensor([-0.2, 0.2], dtype=torch.float32),
        torch.tensor([-0.3, 0.3], dtype=torch.float32),
        hvp=lambda v: v,
        theta_numel=1,
        S=2,
        ridge=1.0,
        cg_tol=1e-12,
        cg_max=20,
        verbose=False,
    )

    assert info["p"].dtype == torch.float32
    assert info["Sigma_oo"].dtype == torch.float64
    assert info["se_p"].dtype == torch.float64
    assert torch.isfinite(info["se_p"]).all()


def _require_cuda_triton() -> torch.device:
    pytest.importorskip("triton")
    try:
        batching._load_native_module()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        torch.empty(1, device="cuda")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return torch.device("cuda")


def _tiny(tmp_path, device, forward_self_loop, adjoint_self_loop):
    (tmp_path / "s.nwk").write_text("((A:1,B:1)AB:1,C:1)Root:1;\n", encoding="utf-8")
    (tmp_path / "f.nwk").write_text("((A_1:1,B_1:1)g:1,C_1:1)GeneRoot:1;\n", encoding="utf-8")
    so = SolverOptions(pi_iters=200, neumann_terms=128, e_max_iter=500, e_tol=1e-15,
                       forward_self_loop=forward_self_loop,
                       adjoint_self_loop=adjoint_self_loop)
    return GeneReconModel(tmp_path / "s.nwk", [tmp_path / "f.nwk"], mode="specieswise",
                          device=str(device), family_chunk_size=1, clade_budget=None,
                          batch_packing="sequential", max_wave_size=4, solver_options=so)


def _joint_hessian(model, th, al, om, device):
    """The dense joint (theta, alpha, omega) Hessian, one HVP per basis direction."""
    from gpurec.solver.curvature.origination import build_joint_hvp

    S = int(model.species_helpers["S"])
    p = 3 * S + 2 * S
    hvp, _l, _sv, _c = build_joint_hvp(model.batch_statics[0], th, al, om, tangent_self_iters=200)
    return torch.stack([hvp(torch.eye(p, dtype=DT, device=device)[i]) for i in range(p)], dim=1)


def test_joint_hvp_symmetric_and_gauge_null(tmp_path: Path):
    device = _require_cuda_triton()
    from gpurec.solver.curvature.origination import build_joint_hvp
    torch.manual_seed(0)
    m = _tiny(tmp_path, device, "linear", "series")
    S = int(m.species_helpers["S"]); tn = 3 * S; p = tn + 2 * S
    th = 0.3 * torch.randn(S, 3, dtype=DT, device=device)
    al = (lambda a: a - a.mean())(0.6 * torch.randn(S, dtype=DT, device=device))
    om = (lambda a: a - a.mean())(0.7 * torch.randn(S, dtype=DT, device=device))
    hvp, _l, _sv, _c = build_joint_hvp(m.batch_statics[0], th, al, om, tangent_self_iters=200)

    H = torch.stack([hvp(torch.eye(p, dtype=DT, device=device)[i]) for i in range(p)], dim=1)
    assert float((H - H.T).abs().max()) < 1e-10                     # symmetric Hessian
    ea = torch.zeros(p, dtype=DT, device=device); ea[tn:tn + S] = 1.0
    eo = torch.zeros(p, dtype=DT, device=device); eo[tn + S:] = 1.0
    assert float(hvp(ea).norm()) < 1e-10 and float(hvp(eo).norm()) < 1e-10   # both gauge nulls


def test_exact_solves_reproduce_the_joint_hessian_under_weighted_receivers(tmp_path: Path):
    """The three exact tree solves must give the iterated solves' joint Hessian, weights and all.

    They are not the default, so every other test here runs the linear/series/sweeps paths and
    none of them would notice. They are also the only paths whose receiver-weight handling is
    written out by hand instead of falling out of the primal recurrence, which is exactly what a
    non-uniform alpha probes: the exact tangent used to drop the half of a node's receiver-weight
    tangent that its children see through their own remaining donor tangent, which left this
    Hessian asymmetric by 2.8e-3 in float64 while every uniform-weight result stayed bit-identical
    and every fit stayed green.
    """
    device = _require_cuda_triton()
    torch.manual_seed(0)
    reference = _tiny(tmp_path, device, "linear", "series")
    S = int(reference.species_helpers["S"])
    th = 0.3 * torch.randn(S, 3, dtype=DT, device=device)
    al = (lambda a: a - a.mean())(0.6 * torch.randn(S, dtype=DT, device=device))
    om = (lambda a: a - a.mean())(0.7 * torch.randn(S, dtype=DT, device=device))
    assert float(al.std()) > 0.1, "the receiver weights must be non-uniform for this to test anything"

    reference_hessian = _joint_hessian(reference, th, al, om, device)
    for forward_self_loop, adjoint_self_loop in (("exact", "series"), ("linear", "exact"),
                                                 ("exact", "exact")):
        model = _tiny(tmp_path, device, forward_self_loop, adjoint_self_loop)
        hessian = _joint_hessian(model, th, al, om, device)
        label = f"{forward_self_loop}/{adjoint_self_loop}"
        assert float((hessian - hessian.T).abs().max()) < 1e-10, f"{label} Hessian is asymmetric"
        assert float((hessian - reference_hessian).abs().max()) < 1e-10, f"{label} Hessian moved"


def test_certify_matches_dense_and_fisher_converges(tmp_path: Path):
    device = _require_cuda_triton()
    from gpurec.solver.curvature.origination import (
        build_joint_hvp, certify_joint_min, origination_information)
    torch.manual_seed(0)
    m = _tiny(tmp_path, device, "linear", "series")
    S = int(m.species_helpers["S"]); tn = 3 * S; p = tn + 2 * S
    th = 0.3 * torch.randn(S, 3, dtype=DT, device=device)
    al = (lambda a: a - a.mean())(0.6 * torch.randn(S, dtype=DT, device=device))
    om = (lambda a: a - a.mean())(0.7 * torch.randn(S, dtype=DT, device=device))
    hvp, _l, _sv, _c = build_joint_hvp(m.batch_statics[0], th, al, om, tangent_self_iters=200)

    # dense gauge-projected reduced minimum
    H = torch.stack([hvp(torch.eye(p, dtype=DT, device=device)[i]) for i in range(p)], dim=1)
    H = 0.5 * (H + H.T)
    P = torch.eye(p, dtype=DT, device=device)
    P[tn:tn + S, tn:tn + S] -= torch.ones(S, S, dtype=DT, device=device) / S
    P[tn + S:, tn + S:] -= torch.ones(S, S, dtype=DT, device=device) / S
    evals = torch.linalg.eigvalsh(P @ H @ P).tolist()
    dense_min = min(x for x in evals if abs(x) > 1e-6)

    cert = certify_joint_min(m.batch_statics[0], th, al, om, m=min(24, p), hvp=hvp,
                             theta_numel=tn, S=S, verbose=False)
    assert abs(cert["lam_min_gauge"] - dense_min) < 1e-6
    assert cert["ritz_resid"] < 1e-6

    info = origination_information(m.batch_statics[0], th, al, om, hvp=hvp, theta_numel=tn, S=S,
                                   ridge=abs(dense_min) + 1.0, cg_tol=1e-9, verbose=False)
    assert max(info["cg_resid"]) < 1e-6
    assert torch.isfinite(info["se_omega"]).all() and torch.isfinite(info["se_p"]).all()
    assert float((info["Sigma_oo"] - info["Sigma_oo"].T).abs().max()) < 1e-9


def test_newton_joint_descends_and_holds_gauge(tmp_path: Path):
    device = _require_cuda_triton()
    from gpurec.solver.curvature.origination import newton_joint
    torch.manual_seed(0)
    m = _tiny(tmp_path, device, "linear", "series")
    S = int(m.species_helpers["S"])
    th = 0.2 * torch.randn(S, 3, dtype=DT, device=device)
    al = (lambda a: a - a.mean())(0.3 * torch.randn(S, dtype=DT, device=device))
    om = (lambda a: a - a.mean())(0.3 * torch.randn(S, dtype=DT, device=device))
    th_o, al_o, om_o, hist = newton_joint(m.batch_statics[0], th, al, om, max_newton=4, max_cg=20,
                                          tangent_self_iters=200, verbose=False)
    assert hist[-1]["F"] <= hist[0]["F"] + 1e-9          # Armijo => loss non-increasing
    assert torch.isfinite(th_o).all() and torch.isfinite(al_o).all() and torch.isfinite(om_o).all()
    assert abs(float(al_o.mean())) < 1e-9 and abs(float(om_o.mean())) < 1e-9   # gauge-fixed outputs


def test_newton_joint_wires_in_origination_penalty(tmp_path: Path):
    """``origination_penalty`` must actually move the fit: an l2 ridge pulling omega toward 0 makes
    the fitted omega's norm smaller than an unpenalized fit from the same start, and F (the quantity
    Armijo enforces non-increasing) must include the penalty term, not just the NLL."""
    device = _require_cuda_triton()
    from gpurec.solver.curvature.origination import newton_joint
    from gpurec.solver.penalties import OriginationPenalty, origination_penalty_and_grad
    torch.manual_seed(0)
    m = _tiny(tmp_path, device, "linear", "series")
    S = int(m.species_helpers["S"])
    th = 0.2 * torch.randn(S, 3, dtype=DT, device=device)
    al = (lambda a: a - a.mean())(0.3 * torch.randn(S, dtype=DT, device=device))
    om = (lambda a: a - a.mean())(0.5 * torch.randn(S, dtype=DT, device=device))

    cfg = OriginationPenalty(l2=5.0)
    th_p, al_p, om_p, hist_p = newton_joint(m.batch_statics[0], th, al, om, max_newton=4, max_cg=20,
                                            tangent_self_iters=200, origination_penalty=cfg,
                                            verbose=False)
    th_u, al_u, om_u, hist_u = newton_joint(m.batch_statics[0], th, al, om, max_newton=4, max_cg=20,
                                            tangent_self_iters=200, verbose=False)

    assert hist_p[-1]["F"] <= hist_p[0]["F"] + 1e-9      # Armijo still holds with the penalty active
    assert torch.isfinite(om_p).all()
    assert abs(float(om_p.mean())) < 1e-9                # still gauge-fixed
    assert float(om_p.norm()) < float(om_u.norm())        # the l2 ridge actually pulled omega in

    # F at the (penalized) start must equal NLL + penalty, not just NLL.
    pen0, _ = origination_penalty_and_grad(om - om.mean(), cfg)
    assert abs(hist_p[0]["F"] - (hist_u[0]["F"] + float(pen0))) < 1e-6

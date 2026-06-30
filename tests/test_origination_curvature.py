"""Analytic second-order for origination: joint (theta, alpha, omega) HVP correctness + consumers.

The omega (origination) logits enter only the NLL aggregation head, so their Hessian blocks are the
head's second derivatives composed with the existing forward-tangent/adjoint (no new kernels). These
tests verify, in fp64 on the tiny fixture: the joint HVP is symmetric, both softmax gauge nulls are
exact, the gauge-projected PD certificate matches the dense gauge-projected eigenvalue, and the
origination Fisher CG solve converges to finite standard errors.
"""
from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.core.scheduling import batching

DT = torch.float64


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


def _tiny(tmp_path, device):
    (tmp_path / "s.nwk").write_text("((A:1,B:1)AB:1,C:1)Root:1;\n", encoding="utf-8")
    (tmp_path / "f.nwk").write_text("((A_1:1,B_1:1)g:1,C_1:1)GeneRoot:1;\n", encoding="utf-8")
    so = SolverOptions(pi_iters=200, neumann_terms=128, e_max_iter=500, e_tol=1e-15)
    return GeneReconModel(tmp_path / "s.nwk", [tmp_path / "f.nwk"], mode="specieswise",
                          device=str(device), family_chunk_size=1, clade_budget=None,
                          batch_packing="sequential", max_wave_size=4, solver_options=so)


def test_joint_hvp_symmetric_and_gauge_null(tmp_path: Path):
    device = _require_cuda_triton()
    from gpurec.optim.origination_curvature import build_joint_hvp
    torch.manual_seed(0)
    m = _tiny(tmp_path, device)
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


def test_certify_matches_dense_and_fisher_converges(tmp_path: Path):
    device = _require_cuda_triton()
    from gpurec.optim.origination_curvature import (
        build_joint_hvp, certify_joint_min, origination_information)
    torch.manual_seed(0)
    m = _tiny(tmp_path, device)
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

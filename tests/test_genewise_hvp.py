"""Genewise analytic HVP gates: analytic forward-over-reverse == fp64 finite-difference (+ symmetry).

The genewise Hessian is block-diagonal per family; broadcasting a unit tangent e_j across all families
recovers column j of every family's block at once. This gate first pins the theta-only block (P0); the
joint (theta, omega) block is added in P2. Mirrors gpurec.optim._verify_hvp but for genewise (F,3) theta.
"""
import math

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.optim.hvp_exact import make_exact_hvp
from gpurec.optim.newton_cg import _fd_hessian_hvp
from gpurec.optim.value_and_grad import forward_solve, make_value_and_grad

_D = "tests/data/alerax/test_trees_200"
_SO = dict(e_max_iter=2000, e_tol=1e-10, pi_iters=128, neumann_terms=64, self_loop_solver="neumann",
           bicgstab_max_iter=500, bicgstab_tol=1e-10, bicgstab_breakdown_tol=1e-30,
           adjoint_pruning_threshold=1e-6, use_adjoint_pruning=True, pibar_side_threshold=0.0)


def build_genewise_model(n_fam=2, dtype=torch.float64, device="cuda"):
    """n_fam identical genewise families in one batch -> block-diagonal [n_fam,3,3]; if the machinery
    wrongly summed across families, each block would come out ~n_fam x and fail vs per-family FD."""
    so = SolverOptions(**_SO)
    so.validate()
    m = GeneReconModel(f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * n_fam, mode="genewise",
                       device=device, dtype=dtype, solver_options=so)
    assert len(m.batch_statics) == 1, f"expected 1 batch, got {len(m.batch_statics)}"
    return m


@pytest.mark.gpu
def test_genewise_theta_hvp_matches_fd():
    m = build_genewise_model()
    static = m.batch_statics[0]
    F = len(m.families)
    S = int(m.species_helpers["S"])
    theta = torch.full((F, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)

    _l, sv = forward_solve([static], theta, rw)
    hvp = make_exact_hvp([static], theta, rw, sv, tangent_self_iters=128)
    fd = _fd_hessian_hvp(make_value_and_grad([static], rw, theta_shape=(F, 3)),
                         theta.reshape(-1).contiguous(), None, eps=1e-5)

    probes = []
    for j in range(3):  # broadcast e_j across all families
        u = torch.zeros(F, 3, device="cuda", dtype=torch.float64)
        u[:, j] = 1.0
        u = u.reshape(-1)
        Ha, Hf = hvp(u).double(), fd(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"broadcast e_{j}: rel={rel:.2e}"
        probes.append((u, Ha))

    (u, Hu), (w, Hw) = probes[0], probes[1]
    sym = abs(float(torch.dot(u, Hw)) - float(torch.dot(w, Hu)))
    scale = max((abs(float(torch.dot(u, Hu))) * abs(float(torch.dot(w, Hw)))) ** 0.5, 1e-30)
    assert sym / scale < 5e-3, f"non-symmetric: rel_asym={sym / scale:.2e}"


@pytest.mark.gpu
def test_theta_blocks_fp32_vs_fp64():
    from gpurec.optim.genewise_curvature import genewise_hessian_blocks
    def blocks(dtype):
        m = build_genewise_model(dtype=dtype); st = m.batch_statics[0]
        G = len(m.families); S = int(m.species_helpers["S"])
        th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=dtype)
        rw = torch.zeros(S, device="cuda", dtype=dtype)
        _l, sv = forward_solve([st], th, rw)
        return genewise_hessian_blocks(st, th, rw, sv)["H_tt"]
    H32, H64 = blocks(torch.float32).double(), blocks(torch.float64)
    assert torch.isfinite(H32).all()
    torch.testing.assert_close(H32, H64, rtol=1e-3, atol=1e-2)
    assert float((H64 - H64.transpose(1, 2)).abs().max()) < 1e-6  # each 3x3 block symmetric

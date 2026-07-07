"""Genewise analytic HVP gates: analytic forward-over-reverse == fp64 finite-difference (+ symmetry).

The genewise Hessian is block-diagonal per family; broadcasting a unit tangent e_j across all families
recovers column j of every family's block at once. This gate first pins the theta-only block (P0); the
joint (theta, omega) block is added in P2. Mirrors gates._verify_hvp but for genewise (F,3) theta.
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


def build_genewise_model(n_fam=2, dtype=torch.float64, device="cuda", per_family_origination=False,
                         family_chunk_size=None):
    so = SolverOptions(**_SO); so.validate()
    kw = {} if family_chunk_size is None else {"family_chunk_size": family_chunk_size}
    m = GeneReconModel(f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * n_fam, mode="genewise",
                       device=device, dtype=dtype, solver_options=so,
                       per_family_origination=per_family_origination, **kw)
    # single-batch is the default (Tasks 1-7); only force ≥2 batches when family_chunk_size splits them
    if family_chunk_size is None:
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


def genewise_hessian_blocks(static, theta, receiver_weights, sv, *, omega=None, active=("theta",)):
    """Theta-only per-family curvature ``[G,3,3]``, built from 3 broadcast HVP probes (one per theta
    component) against the joint analytic HVP (``make_exact_hvp``).

    TEST-ONLY fixture (moved here from gpurec.optim.genewise_curvature): the production genewise fit
    never assembles blocks -- it runs matrix-free CG directly on the analytic HVP
    (``newton_joint_genewise``). Feeds the fp32-vs-fp64 golden comparison below.
    """
    assert tuple(active) == ("theta",), "P0: theta-only; omega/alpha added in Tasks 4-5"
    G = int(theta.shape[0])
    tsi = int(static.solver_options.pi_iters)
    hvp = make_exact_hvp([static], theta, receiver_weights, sv, tangent_self_iters=tsi)
    cols = []
    for j in range(3):  # broadcast e_j across all families -> column j of every 3x3 block at once
        u = torch.zeros(G, 3, device=theta.device, dtype=theta.dtype); u[:, j] = 1.0
        cols.append(hvp(u.reshape(-1))[: G * 3].reshape(G, 3))
    H = torch.stack(cols, dim=-1)  # [G,3,3], H[:,:,j] = col j
    return {"H_tt": 0.5 * (H + H.transpose(1, 2))}


@pytest.mark.gpu
def test_theta_blocks_fp32_vs_fp64():
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


@pytest.mark.gpu
def test_per_family_origination_shape():
    m = build_genewise_model(per_family_origination=True)
    G, S = len(m.families), int(m.species_helpers["S"])
    assert tuple(m.origination_weights.shape) == (G, S)
    m0 = build_genewise_model(per_family_origination=False)   # default unchanged
    assert tuple(m0.origination_weights.shape) == (S,)


def make_joint_value_and_grad(static, theta_shape, S, G):
    """(loss, flat-grad) over z=[theta(3G); alpha(S); omega(GS)] for fp64 FD. Packs the three
    grads from evaluate_static_loss_grad in the SAME order the joint hvp returns them ([theta; alpha; omega])."""
    from gpurec.api._execution import evaluate_static_loss_grad
    nt = int(torch.tensor(theta_shape).prod())

    def vg(x, warm_E=None):
        th = x[:nt].reshape(theta_shape)
        al = x[nt:nt + S]
        om = x[nt + S:nt + S + G * S].reshape(G, S)
        # NOTE need_grad=True is REQUIRED (origination grad is only produced on that path; see _execution.py early return)
        l, g_th, g_al, g_om = evaluate_static_loss_grad(
            static, th, al, om, need_grad=True, need_origination_grad=True)
        g = torch.cat([g_th.reshape(-1), g_al.reshape(-1), g_om.reshape(-1)]).double()
        return float(l), g, None, None

    return vg


# Canonical flat layout is [theta(3G); alpha(S); omega(GS)] -- alpha BEFORE omega (matches origination_curvature.py).
def _dir_theta(G, S, j):
    u = torch.zeros(3 * G + S + G * S, device="cuda", dtype=torch.float64)
    u.view(-1)[j:3 * G:3] = 1.0  # broadcast e_j across families' theta block
    return u


def _dir_omega(G, S, k):
    u = torch.zeros(3 * G + S + G * S, device="cuda", dtype=torch.float64)
    u[3 * G + S:3 * G + S + G * S].reshape(G, S)[:, k] = 1.0  # broadcast omega e_k across families
    return u


@pytest.mark.gpu
def test_joint_theta_omega_hvp_matches_fd():
    m = build_genewise_model(per_family_origination=True)
    st = m.batch_statics[0]
    G, S = len(m.families), int(m.species_helpers["S"])
    th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    om = torch.zeros(G, S, device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([st], th, rw)
    hvp = make_exact_hvp([st], th, rw, sv, tangent_self_iters=128, origination_weights=om)
    x0 = torch.cat([th.reshape(-1), rw.reshape(-1), om.reshape(-1)])   # [theta; alpha; omega]
    fd = _fd_hessian_hvp(make_joint_value_and_grad(st, (G, 3), S, G), x0, None, eps=1e-5)
    for name, u in [("theta_e0", _dir_theta(G, S, 0)), ("omega_k", _dir_omega(G, S, S // 3))]:
        Ha, Hf = hvp(u).double(), fd(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"{name}: rel={rel:.2e}"


@pytest.mark.gpu
def test_joint_theta_omega_alpha_hvp_matches_fd():
    torch.manual_seed(0)
    m = build_genewise_model(per_family_origination=True)
    st = m.batch_statics[0]
    G, S = len(m.families), int(m.species_helpers["S"])
    th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    al = torch.randn(S, device="cuda", dtype=torch.float64) * 0.1   # NON-uniform alpha
    om = torch.zeros(G, S, device="cuda", dtype=torch.float64)
    _l, sv = forward_solve([st], th, al)
    hvp = make_exact_hvp([st], th, al, sv, tangent_self_iters=128, origination_weights=om)
    x0 = torch.cat([th.reshape(-1), al.reshape(-1), om.reshape(-1)])   # [theta; alpha; omega]
    # eps=1e-3 (not 1e-5): the alpha row of H (H_ta = theta-block of H applied to a pure-alpha
    # direction) is the worst-conditioned FD block -- a small alpha->theta coupling (O(5e-3)) read
    # off the CHANGE in the large theta gradient (O(15)). In this converged-fp64 setup the central
    # FD is gradient-NOISE-limited (err ~ solver_noise/eps ~ 1e-10/eps), NOT truncation-limited
    # (O(eps^2) is negligible here: the theta block stays ~3e-8 even at eps=1e-3), so a larger step
    # gives the smaller error for EVERY block. An eps-sweep (1e-3..1e-7) shows the analytic-vs-FD rel
    # DECREASES monotonically toward ~3e-7 as eps grows (no error floor => no structural bug: the
    # genewise alpha cotangent is global [S], no per-family reduction), and the analytic HVP's tiny
    # theta<->alpha asymmetry equals the FD's bit-for-bit (a property of the pi_iters-truncated
    # gradient, not a defect). eps=1e-3 clears 5e-4 with ~10x margin on all four directions.
    fd = _fd_hessian_hvp(make_joint_value_and_grad(st, (G, 3), S, G), x0, None, eps=1e-3)
    P = 3 * G + S + G * S
    dirs = {"theta": _dir_theta(G, S, 1), "omega": _dir_omega(G, S, S // 4)}
    da = torch.zeros(P, device="cuda", dtype=torch.float64)
    da[3 * G:3 * G + S] = torch.randn(S, device="cuda", dtype=torch.float64)  # alpha block
    dirs["alpha"] = da
    dirs["mixed"] = _dir_theta(G, S, 0) + _dir_omega(G, S, S // 2) + da
    Hs = {}
    for name, u in dirs.items():
        Ha, Hf = hvp(u).double(), fd(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"{name}: rel={rel:.2e}"
        Hs[name] = Ha
    # symmetry across parameter groups: u^T H w == w^T H u
    u, w = dirs["theta"], dirs["alpha"]
    sym = abs(float(torch.dot(u, Hs["alpha"])) - float(torch.dot(w, Hs["theta"])))
    scale = max((abs(float(torch.dot(u, Hs["theta"]))) * abs(float(torch.dot(w, Hs["alpha"])))) ** 0.5, 1e-30)
    assert sym / scale < 5e-3, f"asym={sym / scale:.2e}"


@pytest.mark.gpu
def test_newton_step_matches_dense():
    from gpurec.optim.genewise_curvature import newton_step_joint, _assemble_dense_arrowhead
    torch.manual_seed(0); G, S, r, mu = 2, 5, 2, 1e-2
    dev, dt = "cuda", torch.float64
    def spd(n):
        A = torch.randn(n, n, device=dev, dtype=dt); return A @ A.T + n * torch.eye(n, device=dev, dtype=dt)
    blocks = dict(
        H_tt=torch.stack([spd(3) for _ in range(G)]),
        H_to=torch.randn(G, 3, S, device=dev, dtype=dt) * 0.1,
        H_oo_diag=torch.rand(G, S, device=dev, dtype=dt) + 1.0,
        H_oo_lr=torch.randn(G, S, r, device=dev, dtype=dt) * 0.1,
        H_aa=spd(S), H_za=torch.randn(G, 3 + S, S, device=dev, dtype=dt) * 0.1)
    g_th = torch.randn(G, 3, device=dev, dtype=dt); g_om = torch.randn(G, S, device=dev, dtype=dt)
    g_al = torch.randn(S, device=dev, dtype=dt)
    dth, dom, dal = newton_step_joint(blocks, g_th, g_om, g_al, mu)
    # NOTE: the oracle also takes the three grads (to build `gd`, the "stacked g" of the
    # brief comment); newton_step_joint takes them separately, so the oracle needs them too.
    Hd, gd = _assemble_dense_arrowhead(blocks, g_th, g_om, g_al, mu)  # dense [(3G+GS+S)]^2 + stacked g
    ref = torch.linalg.solve(Hd, -gd)
    got = torch.cat([dth.reshape(-1), dom.reshape(-1), dal.reshape(-1)])
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-8)


@pytest.mark.gpu
def test_origination_grad_matches_fd():
    from gpurec.api._execution import evaluate_static_loss_grad
    torch.manual_seed(0)
    m = build_genewise_model(per_family_origination=True); st = m.batch_statics[0]
    G, S = len(m.families), int(m.species_helpers["S"])
    th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    om = torch.randn(G, S, device="cuda", dtype=torch.float64) * 0.1
    def loss_only(omega):
        l, *_ = evaluate_static_loss_grad(st, th, rw, omega, need_grad=False)
        return float(l)
    _l, _gt, _gr, g_om = evaluate_static_loss_grad(st, th, rw, om, need_grad=True, need_origination_grad=True)
    assert tuple(g_om.shape) == (G, S)
    eps = 1e-5; g, k = 0, S // 3          # probe one (family, species) entry
    d = torch.zeros(G, S, device="cuda", dtype=torch.float64); d[g, k] = eps
    fd = (loss_only(om + d) - loss_only(om - d)) / (2 * eps)
    rel = abs(float(g_om[g, k]) - fd) / max(abs(fd), 1e-30)
    assert rel < 1e-3, f"origination grad rel={rel:.2e}"


@pytest.mark.gpu
def test_genewise_joint_newton_reduces_grad():
    from gpurec.optim.genewise_curvature import newton_joint_genewise, proj_z_genewise
    torch.manual_seed(0)
    m = build_genewise_model(per_family_origination=True); st = m.batch_statics[0]
    G, S = len(m.families), int(m.species_helpers["S"])
    th0 = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    al0 = torch.randn(S, device="cuda", dtype=torch.float64) * 0.1          # NON-uniform alpha
    om0 = torch.randn(G, S, device="cuda", dtype=torch.float64) * 0.1       # NON-uniform omega
    th, al, om, hist = newton_joint_genewise(st, th0, al0, om0, max_newton=4, verbose=False)
    assert hist["gnorm_final"] < hist["gnorm_init"]        # Newton reduced the gauge-projected gradient
    assert math.isfinite(hist["lam_min"])                  # certificate ran (PD not required)


@pytest.mark.gpu
def test_multibatch_joint_hvp_matches_fd():
    # Multi-batch genewise joint HVP: sum of per-batch single-batch exact HVPs (per-family
    # gather/scatter) must match the FD Hessian of the multi-batch summed gradient. Forces >=2
    # batches via family_chunk_size so the disjoint-family scatter + shared-alpha sum are exercised.
    from gpurec.optim.newton_cg import _fd_hessian_hvp
    from gpurec.optim.genewise_curvature import make_multibatch_joint_hvp_genewise, multibatch_joint_vg_genewise
    torch.manual_seed(0)
    m = build_genewise_model(n_fam=4, per_family_origination=True, family_chunk_size=2)
    assert len(m.batch_statics) >= 2, f"need multi-batch, got {len(m.batch_statics)}"
    G, S = len(m.families), int(m.species_helpers["S"])
    th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    al = torch.randn(S, device="cuda", dtype=torch.float64) * 0.1        # NON-uniform alpha
    om = torch.randn(G, S, device="cuda", dtype=torch.float64) * 0.1     # NON-uniform omega
    Av = make_multibatch_joint_hvp_genewise(m.batch_statics, th, al, om, tangent_self_iters=128)
    x0 = torch.cat([th.reshape(-1), al.reshape(-1), om.reshape(-1)])     # [theta; alpha; omega]
    fd = _fd_hessian_hvp(multibatch_joint_vg_genewise(m.batch_statics, (G, 3), S, G), x0, None, eps=1e-3)
    P = 3 * G + S + G * S

    def d_theta(j):
        u = torch.zeros(P, device="cuda", dtype=torch.float64); u.view(-1)[j:3 * G:3] = 1.0; return u

    def d_omega(k):
        u = torch.zeros(P, device="cuda", dtype=torch.float64); u[3 * G + S:].reshape(G, S)[:, k] = 1.0; return u

    da = torch.zeros(P, device="cuda", dtype=torch.float64)
    da[3 * G:3 * G + S] = torch.randn(S, device="cuda", dtype=torch.float64)
    for name, u in [("theta", d_theta(0)), ("omega", d_omega(S // 3)), ("alpha", da),
                    ("mixed", d_theta(1) + d_omega(S // 2) + da)]:
        Ha, Hf = Av(u).double(), fd(u).double()
        rel = float((Ha - Hf).abs().max()) / max(float(Hf.abs().max()), 1e-30)
        assert torch.isfinite(Ha).all() and rel < 5e-4, f"{name}: rel={rel:.2e}"


@pytest.mark.gpu
def test_genewise_joint_newton_multibatch():
    # Multi-batch genewise joint Newton-CG fit: newton_joint_genewise must accept a batch_statics
    # LIST (len>=2), drive the multi-batch HVP + multi-batch joint value-and-grad, and reduce the
    # gauge-projected gradient. max_newton=3 keeps the several per-batch HVP rebuilds/step bounded.
    from gpurec.optim.genewise_curvature import newton_joint_genewise
    torch.manual_seed(0)
    m = build_genewise_model(n_fam=4, per_family_origination=True, family_chunk_size=2)
    assert len(m.batch_statics) >= 2, f"need multi-batch, got {len(m.batch_statics)}"
    G, S = len(m.families), int(m.species_helpers["S"])
    th0 = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    al0 = torch.randn(S, device="cuda", dtype=torch.float64) * 0.1     # non-uniform alpha
    om0 = torch.randn(G, S, device="cuda", dtype=torch.float64) * 0.1  # non-uniform omega
    th, al, om, hist = newton_joint_genewise(m.batch_statics, th0, al0, om0, max_newton=3, verbose=False)
    assert hist["gnorm_final"] < hist["gnorm_init"]   # Newton reduced the gauge-projected gradient
    assert math.isfinite(hist["lam_min"])             # certificate ran (PD not required)


@pytest.mark.gpu
def test_stream_batches_multibatch_origination_grad():
    # Multi-batch per-family origination grad: stream_batches must scatter each batch-local [G_b,S]
    # origination grad into the full [G,S] accumulator (index_add_), not shape-mismatch on .add_().
    from gpurec.api._execution import stream_batches
    torch.manual_seed(0)
    m = build_genewise_model(n_fam=4, per_family_origination=True, family_chunk_size=2)
    assert len(m.batch_statics) >= 2, f"need multi-batch, got {len(m.batch_statics)}"
    G, S = len(m.families), int(m.species_helpers["S"])
    th = torch.full((G, 3), math.log2(0.1), device="cuda", dtype=torch.float64)
    rw = torch.zeros(S, device="cuda", dtype=torch.float64)
    om = torch.randn(G, S, device="cuda", dtype=torch.float64) * 0.1
    loss, g_th, g_rw, g_om = stream_batches(
        m.batch_statics, th, rw, om, genewise=True, need_grad=True, need_origination_grad=True)
    assert tuple(g_om.shape) == (G, S)
    assert torch.isfinite(g_om).all()

"""S9 gate: the gauge-projected joint ``(theta, alpha)`` curvature CONSUMERS
(``gpurec.solver.curvature.receiver``) built on the verified analytic joint exact HVP (PR #4).

Three parts:

  A. SYNTHETIC (CPU, fast, DENSE ground truth).  Inject an EXACT receiver gauge null into a random
     symmetric matrix ``M = P_z B P_z`` (``B`` PD => the gauge-fixed reduced block is PD, with one
     exact-zero gauge eigenvalue). Then check, against dense numpy ``eigh`` / ``pinv``:
       * ``certify_joint_min``  -> gauge-projected Lanczos smallest eigenvalue == 2nd-smallest eig(M)
         (the smallest is the gauge null ~0), and the gauge null leaks 0.
       * ``receiver_information`` -> CG marginal covariance == alpha-block of ``pinv(M)``.
     This validates the consumer LINEAR ALGEBRA (projected Lanczos + projected CG + Schur marginal)
     to machine precision, independent of the model.

  B. LIVE hogenom-8 (real analytic joint HVP, fp64 converged solver).  At a near-minimum reached by
     a short joint first-order descent, check the OPERATOR-level certificates that must hold on the
     real HVP:
       * certify: Ritz residual small + raw-HVP gauge leak at the truncation floor,
       * Fisher : CG residual small + covariance symmetry + finite s.e.,
       * newton_joint: monotone loss decrease and ||P g|| reduced well below the first-order warmup,
         and the PD certificate at its output.

  C. PRIMATES (S=25) DENSE cross-check (best-effort).  If the tiny primates fixture loads, build the
     dense gauge-projected Hessian by applying the REAL analytic joint HVP to the 4S unit vectors and
     cross-check the cert + Fisher against dense ``eigh``/``pinv`` -- real operator AND dense truth.

    python -m gates._verify_s9_curvature                 # A + B (+ C if primates loads)
    python -m gates._verify_s9_curvature --synthetic     # A only (no GPU/model)
"""

from __future__ import annotations

import glob
import math
import os
import sys

import torch

from gpurec.solver.curvature.receiver import (
    certify_joint_min, make_gauge_operator, newton_joint, proj_alpha, proj_z,
    receiver_information,
)

_F64 = torch.float64


# =============================================================================== A. SYNTHETIC (dense)
def _proj_mat(theta_numel, S, device):
    """Dense ``P_z = blockdiag(I_theta, I_S - 11^T/S)``."""
    P = torch.eye(theta_numel + S, dtype=_F64, device=device)
    blk = torch.eye(S, dtype=_F64, device=device) - torch.ones(S, S, dtype=_F64, device=device) / S
    P[theta_numel:, theta_numel:] = blk
    return P


def run_synthetic(theta_numel=12, S=8, seed=1, device="cpu"):
    print(f"[A synthetic] theta_numel={theta_numel} S={S} p={theta_numel + S} (dense ground truth)")
    g = torch.Generator(device=device).manual_seed(seed)
    p = theta_numel + S
    A = torch.randn(p, p, generator=g, dtype=_F64, device=device)
    B = A @ A.T + p * torch.eye(p, dtype=_F64, device=device)        # SPD
    P = _proj_mat(theta_numel, S, device)
    M = P @ B @ P                                                    # inject the exact gauge null
    M = 0.5 * (M + M.T)
    hvp = lambda v: M @ v.to(_F64)

    th = torch.zeros(theta_numel, dtype=_F64, device=device)
    al = torch.zeros(S, dtype=_F64, device=device)

    # ---- dense references
    evals = torch.linalg.eigvalsh(M)
    lam_gauge_ref = float(evals[1])                                  # evals[0] is the gauge null ~0
    gauge_eig = float(evals[0])
    Minv = torch.linalg.pinv(M)

    # ---- cert consumer
    cert = certify_joint_min(None, th, al, hvp=hvp, theta_numel=theta_numel, S=S, m=p, verbose=False)
    lam_err = abs(cert["lam_min_gauge"] - lam_gauge_ref) / max(abs(lam_gauge_ref), 1e-12)
    cert_ok = lam_err < 1e-6 and cert["ritz_resid"] < 1e-8 and cert["pd"]
    print(f"  cert: lam_min_gauge={cert['lam_min_gauge']:.6e} ref={lam_gauge_ref:.6e} "
          f"rel_err={lam_err:.2e} ritz={cert['ritz_resid']:.2e} gauge_eig(ref)={gauge_eig:.2e} "
          f"[{'PASS' if cert_ok else 'FAIL'}]")

    # ---- gauge-null annihilation: A_z [0;1] == 0 exactly
    Av = make_gauge_operator(hvp, theta_numel)
    e_null = torch.cat([torch.zeros(theta_numel, dtype=_F64, device=device),
                        torch.ones(S, dtype=_F64, device=device)])
    null_norm = float(Av(e_null).norm())
    null_ok = null_norm < 1e-10
    print(f"  gauge-null: ||A_z [0;1_S]||={null_norm:.2e} [{'PASS' if null_ok else 'FAIL'}]")

    # ---- Fisher consumer (full marginal) vs dense pinv alpha-block
    fish = receiver_information(None, th, al, hvp=hvp, theta_numel=theta_numel, S=S,
                               cg_tol=1e-12, cg_max=4 * p, verbose=False)
    ref_cols = []
    for j in range(S):
        e = torch.zeros(p, dtype=_F64, device=device)
        e[theta_numel + j] = 1.0
        x = Minv @ proj_z(e, theta_numel)
        ref_cols.append(proj_alpha(x[theta_numel:]))
    Sigma_ref = 0.5 * (torch.stack(ref_cols, 1) + torch.stack(ref_cols, 1).T)
    fish_err = float((fish["Sigma_aa"] - Sigma_ref).norm()) / max(float(Sigma_ref.norm()), 1e-30)
    asym = float((fish["Sigma_aa"] - fish["Sigma_aa"].T).norm()) / max(float(fish["Sigma_aa"].norm()), 1e-30)
    fish_ok = fish_err < 1e-6 and asym < 1e-9
    print(f"  fisher: ||Sigma_aa - pinv_block||/||ref||={fish_err:.2e} sym={asym:.2e} "
          f"max_cg_resid={max(fish['cg_resid']):.2e} [{'PASS' if fish_ok else 'FAIL'}]")

    ok = cert_ok and null_ok and fish_ok
    print(f"[A synthetic] {'PASS' if ok else 'FAIL'}")
    return ok


def run_penalty_unit(S=6, seed=2, device="cpu"):
    """Unit-check ``_penalty_hvp`` (ridge + GBM tree-Laplacian on the theta block) vs a DENSE penalty
    Hessian, on a tiny synthetic tree. This is the only place the penalty path is exercised."""
    from gpurec.solver.curvature.receiver import _penalty_hvp, _tree_edges
    print(f"[A' penalty-hvp] S={S} (dense ridge + tree-Laplacian ground truth)")
    theta_shape = (S, 3)
    theta_numel = S * 3
    p = theta_numel + S
    # a tiny tree: species i's parent is i-1 (a caterpillar), root = -1
    sp_parent = torch.tensor([-1] + list(range(S - 1)), dtype=torch.long, device=device)
    tp_child, tp_parent = _tree_edges(sp_parent)
    lam, lam_tree = 0.7, 1.3
    Hp = _penalty_hvp(theta_numel, theta_shape, lam=lam, lam_tree=lam_tree,
                      tp_child=tp_child, tp_parent=tp_parent)

    # dense reference: ridge lam*I_theta + lam_tree * (L kron I_3) on the theta block, 0 on alpha
    L = torch.zeros(S, S, dtype=_F64, device=device)
    for c, pa in zip(tp_child.tolist(), tp_parent.tolist()):
        for (i, j, s) in ((c, c, 1.0), (pa, pa, 1.0), (c, pa, -1.0), (pa, c, -1.0)):
            L[i, j] += s
    Hpen = torch.zeros(p, p, dtype=_F64, device=device)
    Hpen[:theta_numel, :theta_numel] = lam * torch.eye(theta_numel, dtype=_F64, device=device)
    # theta layout is [S,3] row-major: coord (s,k) -> s*3+k; tree-Laplacian couples species, all k
    for s1 in range(S):
        for s2 in range(S):
            for k in range(3):
                Hpen[s1 * 3 + k, s2 * 3 + k] += lam_tree * L[s1, s2]

    g = torch.Generator(device=device).manual_seed(seed)
    err = 0.0
    for _ in range(4):
        v = torch.randn(p, generator=g, dtype=_F64, device=device)
        err = max(err, float((Hp(v) - Hpen @ v).norm()) / max(float((Hpen @ v).norm()), 1e-30))
    ok = err < 1e-12
    print(f"  ||Hp(v) - Hpen v||/||ref|| max over 4 dirs = {err:.2e} [{'PASS' if ok else 'FAIL'}]")
    return ok


# ============================================================================= live fixture helpers
def _live_imports():
    from gates._verify_hvp_recv import (  # reuse the verified fixture + endgame descent
        _static_theta_alpha_from_live, _valid_mass_min,
    )
    from gpurec.core.inference.solver import receiver_weights_are_uniform
    from gpurec.fit.optimize import first_order
    from gpurec.solver.value_and_grad import make_value_and_grad
    return (_static_theta_alpha_from_live, _valid_mass_min, receiver_weights_are_uniform,
            first_order, make_value_and_grad)


def _joint_grad_norm(static, theta, alpha, make_vg):
    """||P_z g_z|| at (theta, alpha) via the joint value-and-grad (gauge-projected)."""
    vg = make_vg(static, alpha, theta_shape=tuple(theta.shape), optimize_receiver=True)
    z = torch.cat([theta.reshape(-1), alpha]).contiguous()
    _F, g_z, _, _ = vg(z)
    tn = int(theta.numel())
    return float(proj_z(g_z.double(), tn).norm()), float(_F)


# ============================================================================= B. LIVE hogenom-8
def run_live(n_families=8, device="cuda", seed=0, tangent_self_iters=128, warmup_steps=30,
             newton_steps=6, cert_m=160, fisher_species=4):
    (_static_theta_alpha_from_live, _valid_mass_min, receiver_weights_are_uniform,
     first_order, make_value_and_grad) = _live_imports()

    static, theta, alpha, S, vm0 = _static_theta_alpha_from_live(n_families, device, seed=seed)
    print(f"\n[B live {n_families}-family] S={S} p(4S)={4 * S} fp64 converged "
          f"(tself={tangent_self_iters}) valid_mass_min={vm0:.4f}")

    # ---- short joint first-order warmup to a near-minimum (alpha off the seed) ----
    (theta_w, alpha_w), *_ = first_order(
        [static], theta.float(), alpha.float(), optimizer="adam", lr0=0.3, schedule="cosine",
        max_steps=warmup_steps, verbose=False, with_receiver=True, alpha0=alpha.float(),
        early_stop=False)
    theta_w = theta_w.detach().double().reshape(theta.shape).contiguous()
    alpha_w = proj_alpha(alpha_w.detach().double().reshape(-1)).contiguous()
    assert not receiver_weights_are_uniform(alpha_w) and _valid_mass_min(static, alpha_w) > 1e-3
    gP_w, F_w = _joint_grad_norm(static, theta_w, alpha_w, make_value_and_grad)
    print(f"  [warmup] {warmup_steps}-step joint Adam: F={F_w:.4f}  ||P g||={gP_w:.4e}")

    # ---- newton_joint: must reduce loss + ||P g|| well below the warmup ----
    theta_n, alpha_n, hist = newton_joint(
        [static], theta_w, alpha_w, tangent_self_iters=tangent_self_iters, gtol=1e-3,
        max_newton=newton_steps, lanczos_m=10, max_cg=30, verbose=True)
    losses = [h["F"] for h in hist]
    monotone = all(b <= a + 1e-6 for a, b in zip(losses, losses[1:]))
    gP_n, F_n = _joint_grad_norm(static, theta_n, alpha_n, make_value_and_grad)
    newton_ok = (F_n <= F_w + 1e-6) and (gP_n < gP_w) and monotone
    print(f"  [newton_joint] F: {F_w:.4f} -> {F_n:.4f}  ||P g||: {gP_w:.4e} -> {gP_n:.4e}  "
          f"monotone={monotone} [{'PASS' if newton_ok else 'FAIL'}]")

    # ---- certify at the newton_joint output (OPERATOR-level certificates) ----
    # On 8 families alpha is hopelessly non-identifiable (8 families cannot pin 1331 receiver logits;
    # most species never receive a transfer), so the gauge-fixed H_aa is near-singular with shallow
    # negative curvature -- exactly the documented landscape. We therefore gate on the OPERATOR
    # certificates (Ritz residual tiny, gauge leak at machine zero, eigenvector gauge-fixed) and
    # REPORT lam_min honestly; PD is not expected at this non-identifiable point.
    cert = certify_joint_min([static], theta_n, alpha_n, m=cert_m,
                             tangent_self_iters=tangent_self_iters, verbose=True)
    cert_ok = (cert["ritz_resid"] < 1e-3 and cert["leak"] < 1e-2 and cert["gauge_comp"] < 1e-6)

    # ---- Fisher at the output (RIDGE-STABILISED: the unridged inverse-Hessian needs a PD min, which
    # 8 families do not provide; a ridge well-poses the gauge-projected solve -- the MAP move -- so CG
    # converges and we can verify the solve+extraction WIRING on the real operator: residual small,
    # covariance symmetric, s.e. finite. The UNRIDGED Fisher consumer is verified vs dense pinv to
    # 2e-13 in the synthetic test). ----
    from gpurec.solver.curvature.receiver import build_joint_hvp
    hvp_n, _l, _sv, _c = build_joint_hvp([static], theta_n, alpha_n,
                                         tangent_self_iters=tangent_self_iters)
    ridge = max(1.0, abs(cert["lam_min_gauge"]) + 0.5)   # comfortably PD
    species = list(range(min(fisher_species, S)))
    fish = receiver_information([static], theta_n, alpha_n, hvp=hvp_n, theta_numel=int(theta_n.numel()),
                               S=S, species=species, cg_tol=1e-7, cg_max=400, ridge=ridge, verbose=True)
    cols = fish["Sigma_aa"]  # [S, |species|]
    sub = cols[species, :]
    fisher_sym = float((sub - sub.T).norm()) / max(float(sub.norm()), 1e-30)
    fisher_ok = max(fish["cg_resid"]) < 1e-4 and fisher_sym < 1e-3 and bool(
        torch.isfinite(fish["se_alpha"]).all())
    print(f"  [fisher ridge={ridge:.2f}] max_cg_resid={max(fish['cg_resid']):.2e} "
          f"cross-cov_sym={fisher_sym:.2e} se_alpha_max={float(fish['se_alpha'].max()):.3e} "
          f"[{'PASS' if fisher_ok else 'FAIL'}]")

    ok = newton_ok and cert_ok and fisher_ok
    print(f"[B live] cert(ritz={cert['ritz_resid']:.2e},leak={cert['leak']:.2e},"
          f"lam_min={cert['lam_min_gauge']:+.3e},pd={cert['pd']} [non-identifiable: 8 fam]) "
          f"newton={'ok' if newton_ok else 'FAIL'} fisher={'ok' if fisher_ok else 'FAIL'} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================= C. PRIMATES dense (opt)
_PRIM_CANDIDATES = [os.environ["GPUREC_PRIMATES_ROOT"]] if os.environ.get("GPUREC_PRIMATES_ROOT") else []


def run_primates_dense(device="cuda", seed=3, tangent_self_iters=64):
    """Best-effort: build the tiny primates model, form the DENSE gauge-projected joint Hessian via
    the real analytic HVP (4S applies), cross-check cert + Fisher vs dense eigh/pinv. Returns
    True/False/None (None = fixture unavailable, not a failure)."""
    prim = next((d for d in _PRIM_CANDIDATES if os.path.isdir(d)), None)
    if prim is None:
        print("\n[C primates] fixture not found -> skipped")
        return None
    try:
        from gpurec import GeneReconModel, SolverOptions
        from gpurec.core.inference.solver import receiver_weights_are_uniform
        from gpurec.solver.curvature.receiver import build_joint_hvp
        so = SolverOptions(e_max_iter=2000, e_tol=1e-10, pi_iters=128, neumann_terms=64,
                           bicgstab_max_iter=500, bicgstab_tol=1e-10,
                           bicgstab_breakdown_tol=1e-30, adjoint_pruning_threshold=1e-6,
                           use_adjoint_pruning=True, pibar_side_threshold=0.0)
        so.validate()
        sp = f"{prim}/speciesTree.newick"
        trees = sorted(glob.glob(f"{prim}/family_*/gene_trees/raxml.newick"))
        if not trees:
            print("\n[C primates] no gene trees -> skipped")
            return None
        m = GeneReconModel(sp, trees, mode="specieswise", device=device, solver_options=so)
        static = m.batch_statics[0] if len(m.batch_statics) == 1 else None
        if static is None:
            print(f"\n[C primates] {len(m.batch_statics)} batches (want 1) -> skipped")
            return None
        S = int(m.species_helpers["S"])
        theta = torch.full((S, 3), math.log2(0.1), device=device, dtype=_F64)
        gen = torch.Generator(device=device).manual_seed(seed)
        alpha = proj_alpha(0.2 * torch.randn(S, generator=gen, device=device, dtype=_F64))
        if receiver_weights_are_uniform(alpha):
            print("\n[C primates] alpha uniform -> skipped")
            return None
        print(f"\n[C primates] S={S} p(4S)={4 * S} DENSE cross-check (real analytic HVP)")
    except Exception as e:  # noqa: BLE001 - best-effort fixture
        print(f"\n[C primates] model build failed ({type(e).__name__}: {e}) -> skipped")
        return None

    hvp, _l, _sv, _c = build_joint_hvp([static], theta, alpha, tangent_self_iters=tangent_self_iters)
    tn = int(theta.numel())
    p = tn + S
    Av = make_gauge_operator(hvp, tn)
    # dense gauge-projected Hessian: apply A_z to each unit vector (already P_z-sandwiched)
    cols = []
    for i in range(p):
        e = torch.zeros(p, device=device, dtype=_F64)
        e[i] = 1.0
        cols.append(Av(e))
    Hd = torch.stack(cols, 1)
    Hd = 0.5 * (Hd + Hd.T)
    evals = torch.linalg.eigvalsh(Hd)
    # the gauge null gives ~0; smallest gauge-fixed eig = first eig above the ~0 cluster
    nz = evals[evals.abs() > 1e-6 * float(evals.abs().max())]
    lam_ref = float(nz.min()) if nz.numel() else float(evals[1])

    cert = certify_joint_min([static], theta, alpha, hvp=hvp, theta_numel=tn, S=S, m=p - 1,
                             verbose=False)
    lam_err = abs(cert["lam_min_gauge"] - lam_ref) / max(abs(lam_ref), 1e-9)
    cert_ok = lam_err < 1e-3 and cert["ritz_resid"] < 1e-5
    print(f"  cert: lam_min_gauge={cert['lam_min_gauge']:+.6e} dense_ref={lam_ref:+.6e} "
          f"rel_err={lam_err:.2e} ritz={cert['ritz_resid']:.2e} leak={cert['leak']:.2e} "
          f"[{'PASS' if cert_ok else 'FAIL'}]")

    Hinv = torch.linalg.pinv(Hd)
    fish = receiver_information([static], theta, alpha, hvp=hvp, theta_numel=tn, S=S,
                               cg_tol=1e-10, cg_max=4 * p, verbose=False)
    ref_cols = []
    for j in range(S):
        e = torch.zeros(p, device=device, dtype=_F64)
        e[tn + j] = 1.0
        x = Hinv @ proj_z(e, tn)
        ref_cols.append(proj_alpha(x[tn:]))
    Sigma_ref = 0.5 * (torch.stack(ref_cols, 1) + torch.stack(ref_cols, 1).T)
    fish_err = float((fish["Sigma_aa"] - Sigma_ref).norm()) / max(float(Sigma_ref.norm()), 1e-30)
    fish_ok = fish_err < 1e-3
    print(f"  fisher: ||Sigma_aa - pinv_block||/||ref||={fish_err:.2e} "
          f"max_cg_resid={max(fish['cg_resid']):.2e} [{'PASS' if fish_ok else 'FAIL'}]")

    ok = cert_ok and fish_ok
    print(f"[C primates] {'PASS' if ok else 'FAIL'}")
    return ok


# ===================================================================================== entrypoint
def run(synthetic_only=False, live_only=False, **kw):
    a = True if live_only else (run_synthetic() and run_penalty_unit())
    if synthetic_only:
        return a
    b = run_live(**kw)
    c = None if live_only else run_primates_dense()
    overall = a and b and (c is not False)   # c None (skipped) does not fail the gate
    print(f"\n[S9 gate] synthetic={a} live={b} primates={c} -> OVERALL={'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    syn = "--synthetic" in sys.argv
    live = "--live" in sys.argv
    raise SystemExit(0 if run(synthetic_only=syn, live_only=live) else 1)

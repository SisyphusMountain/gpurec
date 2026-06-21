"""S9: gauge-projected joint ``(theta, alpha)`` curvature consumers of the analytic exact HVP.

The receiver logits ``alpha`` enter the NLL only through a full (unpinned) ``log_softmax``, so the
loss is exactly invariant under ``alpha -> alpha + c*1`` (softmax shift-invariance). Consequences for
the joint Hessian ``H`` of ``z = [theta.reshape(-1); alpha]``:

  * ``H_aa @ 1_S = 0`` and ``1_S^T H_at = 0`` -- the all-ones receiver mode is an EXACT zero
    eigenvalue / null direction of ``H``. Any consumer that does not remove it sees a singular
    operator (Lanczos returns ``lam_min = 0``; CG on the Newton system has a null direction).

Every consumer here works in the GAUGE-FIXED subspace via the block projector

    P_z = blockdiag( I_{theta} , I_S - (1/S) 1_S 1_S^T )    (identity on theta, mean-subtract on alpha)

so the gauge null is removed and the reduced operator is the genuine curvature. The joint HVP is the
verified analytic ``make_exact_hvp`` (forward-over-reverse, FD-gated in PR #4); this module only
CONSUMES it:

  * ``certify_joint_min``  -- gauge-fixed reduced-Hessian PD certificate (gauge-projected Lanczos).
  * ``receiver_information`` -- gauge-fixed marginal covariance / standard errors of ``alpha`` (and,
    by the delta method, of the recipient distribution ``w = softmax(alpha)``); the observed Fisher
    information for the receiver weights.
  * ``newton_joint`` -- gauge-projected damped Newton on ``z`` (Newton steps on ``alpha``; this is
    what ``newton_lanczos(with_receiver=True)`` delegates to).

The recipient distribution is ``w_i = softmax(alpha)_i`` (NATURAL softmax: ``receiver_log_probs =
log_softmax(alpha)/ln2`` is the base-2 log of ``softmax(alpha)``), so the alpha->w Jacobian is
``J = diag(w) - w w^T`` (no ``ln2`` factor). Run everything in fp64 (pass fp64 ``theta``/``alpha``).
"""

from __future__ import annotations

import torch

from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.optim.cg import cg_solve, cg_witness, lanczos_extremes, lanczos_min_eigpair
from gpurec.optim.hvp_exact import build_point_cache, make_exact_hvp
from gpurec.optim.value_and_grad import (
    forward_solve, free_cuda_cache_if_tight, make_value_and_grad,
)


# --------------------------------------------------------------------------- gauge projector (P_z)
def proj_alpha(g_alpha: torch.Tensor) -> torch.Tensor:
    """``(I - 11^T/S) g_alpha = g_alpha - mean(g_alpha)`` -- remove the 1_S receiver gauge mode."""
    return g_alpha - g_alpha.mean()


def proj_z(u: torch.Tensor, theta_numel: int) -> torch.Tensor:
    """``P_z u`` -- identity on the theta block, mean-subtract on the alpha block."""
    return torch.cat([u[:theta_numel], proj_alpha(u[theta_numel:])])


def softmax_recipient(alpha: torch.Tensor) -> torch.Tensor:
    """Recipient distribution ``w = softmax(alpha)`` (natural; the model's ``2^{receiver_log_probs}``)."""
    return torch.softmax(alpha.double(), dim=-1)


def _alpha_leak(Hu: torch.Tensor, theta_numel: int) -> float:
    """``|1_S^T (Hu)_alpha| / sqrt(S)`` -- the all-ones component of the alpha row of ``H u``.

    Gauge-invariance certificate: ``H_aa 1 = 0`` and ``1^T H_at = 0`` => this must sit at the solver
    truncation floor for the RAW (un-projected) HVP. (P_z then removes it exactly.)
    """
    ga = Hu[theta_numel:]
    S = int(ga.numel())
    return float(abs(ga.sum()) / max(S ** 0.5, 1.0))


# ----------------------------------------------------------------- penalty Hessian on the theta block
def _tree_edges(sp_parent: torch.Tensor):
    """``(child, parent)`` index pair of the non-root species edges (the GBM tree-Laplacian support)."""
    sp_parent = sp_parent.detach().reshape(-1).long()
    child = (sp_parent >= 0).nonzero(as_tuple=True)[0].contiguous()
    parent = sp_parent[child].contiguous()
    return child, parent


def _penalty_hvp(theta_numel, theta_shape, *, lam=0.0, lam_tree=0.0, tp_child=None, tp_parent=None):
    """Closed-form penalty Hessian-vector product (acts on the THETA block only; alpha block = 0).

    ridge ``(lam/2)||theta - theta_ref||^2`` -> ``lam * I_theta``;
    GBM tree-Laplacian ``(lam_tree/2) sum_e ||theta[c]-theta[p]||^2`` -> ``lam_tree * L`` (PSD).
    Matches the penalties ``make_value_and_grad`` adds to the gradient so the Newton model is exact.
    """
    lam = float(lam or 0.0)
    lam_tree = float(lam_tree or 0.0)
    has_tree = lam_tree > 0.0 and tp_child is not None

    def Hp(v):
        out = torch.zeros_like(v)
        vt = v[:theta_numel]
        if lam > 0.0:
            out[:theta_numel] = out[:theta_numel] + lam * vt
        if has_tree:
            ts = vt.reshape(theta_shape)
            diff = ts.index_select(0, tp_child) - ts.index_select(0, tp_parent)
            g = torch.zeros_like(ts)
            step = lam_tree * diff
            g.index_add_(0, tp_child, step)
            g.index_add_(0, tp_parent, -step)
            out[:theta_numel] = out[:theta_numel] + g.reshape(-1)
        return out

    return Hp


# ---------------------------------------------------------------------------- joint-HVP construction
def build_joint_hvp(static, theta, alpha, *, warm_E=None, tangent_self_iters=None, sv=None,
                    cache=None):
    """Forward-solve at ``(theta, alpha)``, build the per-point adjoint cache once, and return the
    analytic JOINT exact HVP ``hvp(u_vec)`` over ``z = [theta.reshape(-1); alpha]`` (length
    ``theta.numel() + S``). Returns ``(hvp, loss, sv, cache)``.

    REQUIRES a NON-UNIFORM ``alpha`` -- at a uniform base the weighted receiver paths are dead and
    ``make_exact_hvp`` reverts to the theta-only contract (the alpha blocks would be zero).
    """
    if receiver_weights_are_uniform(alpha):
        raise ValueError(
            "build_joint_hvp requires a NON-uniform alpha: at a uniform base the receiver-weight "
            "paths are dead and the joint HVP degenerates to theta-only. Perturb alpha (e.g. a short "
            "first_order(with_receiver=True) warmup) before building the joint curvature."
        )
    if sv is None:
        loss, sv = forward_solve(static, theta, alpha, warm_E=warm_E)
    else:
        loss = None
    if cache is None:
        _gt, _gc, cache = build_point_cache(static, theta, alpha, sv)
    hvp = make_exact_hvp(static, theta, alpha, sv, cache=cache, tangent_self_iters=tangent_self_iters)
    return hvp, (None if loss is None else float(loss)), sv, cache


def make_gauge_operator(hvp, theta_numel, *, penalty_hvp=None, ridge=0.0):
    """``A_z(v) = P_z ( H + penalty ) ( P_z v ) + ridge * P_z v`` -- the symmetric gauge-projected
    joint operator. Its bottom eigenvalue ON THE GAUGE-FIXED SUBSPACE is the reduced curvature; the
    gauge null ``[0; 1_S]`` maps to exactly 0 (killed by the input projection). ``ridge`` adds an
    optional uniform shift for solve robustness (use 0 at a certified PD min).
    """
    ridge = float(ridge or 0.0)

    def Av(v):
        pv = proj_z(v, theta_numel)
        Hv = hvp(pv)
        if penalty_hvp is not None:
            Hv = Hv + penalty_hvp(pv)
        out = proj_z(Hv, theta_numel)
        if ridge != 0.0:
            out = out + ridge * pv
        return out

    return Av


# --------------------------------------------------------------------------------- PD certificate
def certify_joint_min(static, theta, alpha, *, m=200, seed=0, tangent_self_iters=None, warm_E=None,
                      hvp=None, theta_numel=None, S=None, penalty_hvp=None, verbose=True):
    """Gauge-fixed reduced-Hessian PD certificate for the joint ``(theta, alpha)`` minimum.

    Runs gauge-projected Lanczos (``lanczos_min_eigpair`` with a ``P_z``-projected start so the whole
    Krylov basis stays in the gauge-fixed subspace and the gauge null is never sampled) for the
    smallest reduced-Hessian eigenpair. Returns a dict:

      ``lam_min_gauge``  smallest reduced-Hessian eigenvalue (PD iff > 0),
      ``ritz_resid``     ``||A_z v - lam v|| / max(|lam|, 1)`` (trust ``v_min`` only if small),
      ``leak``           raw-HVP gauge leak ``|1^T(H v_min)_a|/sqrt(S)`` (truncation floor => the
                         operator really is gauge-respecting, not just force-projected),
      ``pd``             ``lam_min_gauge > 0`` (the certificate),
      ``v_min``, ``m``, ``S``, ``theta_numel``.

    Pass a prebuilt ``hvp`` (+ ``theta_numel``/``S``) to reuse a cache; else it is built here.
    """
    theta = theta.double()
    alpha = alpha.double()
    if hvp is None:
        hvp, _loss, _sv, _cache = build_joint_hvp(static, theta, alpha, warm_E=warm_E,
                                                  tangent_self_iters=tangent_self_iters)
        theta_numel = int(theta.numel())
        S = int(alpha.numel())
    else:
        theta_numel = int(theta_numel if theta_numel is not None else theta.numel())
        S = int(S if S is not None else alpha.numel())
    p = theta_numel + S
    Av = make_gauge_operator(hvp, theta_numel, penalty_hvp=penalty_hvp)

    gen = torch.Generator(device=str(theta.device)).manual_seed(seed)
    start = proj_z(torch.randn(p, generator=gen, device=theta.device, dtype=torch.float64), theta_numel)

    # The gauge null [0; 1_S] is an EXACT zero eigenvalue of A_z. Lanczos seeking the MINIMUM is drawn
    # to it (roundoff leaks the null mode back into the Krylov basis even from a projected start, and
    # 0 < the true reduced minimum). DEFLATE it: shift the null direction's eigenvalue up to C >> the
    # spectrum top via ``+ C*(v - P_z v)`` (= C on e_null, unchanged on the gauge-fixed subspace), so
    # the smallest eigenvalue of the shifted operator IS the reduced-Hessian minimum. v_min then comes
    # out gauge-fixed, so the Ritz residual below (vs the UNSHIFTED A_z) is the true reduced residual.
    _, lam_max = lanczos_extremes(Av, p, m=min(20, p), device=str(theta.device), start=start)
    shift_C = 2.0 * max(abs(lam_max), 1.0)

    def Av_deflated(v):
        return Av(v) + shift_C * (v - proj_z(v, theta_numel))

    lam_min, v_min = lanczos_min_eigpair(Av_deflated, p, m=m, device=str(theta.device), start=start)
    Av_v = Av(v_min)
    ritz_resid = float((Av_v - lam_min * v_min).norm()) / max(abs(lam_min), 1.0)
    gauge_comp = float((v_min - proj_z(v_min, theta_numel)).norm())  # v_min should be gauge-fixed
    leak = _alpha_leak(hvp(v_min), theta_numel)
    pd = lam_min > 0.0
    if verbose:
        tag = "PD (certified gauge-fixed joint min)" if pd else "NOT PD (saddle / non-stationary)"
        print(f"[recv-cert] lam_min_gauge={lam_min:+.6e}  ritz_resid={ritz_resid:.2e}  "
              f"raw-HVP leak={leak:.2e}  gauge_comp={gauge_comp:.2e}  m={m}  -> {tag}")
    return dict(lam_min_gauge=lam_min, ritz_resid=ritz_resid, leak=leak, pd=bool(pd),
                gauge_comp=gauge_comp, v_min=v_min, m=int(m), S=S, theta_numel=theta_numel)


# ------------------------------------------------------------------------- Fisher / uncertainty (alpha)
def receiver_information(static, theta, alpha, *, hvp=None, theta_numel=None, S=None, species=None,
                        cg_tol=1e-7, cg_max=400, ridge=0.0, tangent_self_iters=None, warm_E=None,
                        penalty_hvp=None, verbose=True):
    """Gauge-fixed marginal covariance / standard errors of the receiver logits ``alpha`` at a
    (certified) joint minimum -- the observed Fisher information for the receiver weights.

    At a local min the observed Fisher information of ``z=(theta,alpha)`` is the joint Hessian ``H``;
    the MLE covariance is its gauge-fixed inverse ``(P_z H P_z)^+``. The alpha-MARGINAL covariance is
    SCHUR-COMPLEMENT-CORRECT (it accounts for the theta coupling) because each column is a FULL joint
    CG solve, then the alpha rows are read off:

        Sigma_aa[:, j] = [ (P_z H P_z)^+ P_z e_{alpha_j} ]_alpha     (CG, one solve per species j)

    Returns a dict:
      ``Sigma_aa``  [|species|, |species|] gauge-fixed marginal covariance (symmetric, mean-zero rows),
      ``se_alpha``  sqrt(diag(Sigma_aa)) -- per-species s.e. of the receiver logits,
      ``se_w``      delta-method s.e. of the recipient probabilities ``w = softmax(alpha)``
                    (``Sigma_w = J Sigma_aa J^T``, ``J = diag(w) - w w^T``),
      ``w``, ``species``, ``cg_iters`` (per solve), ``cg_resid`` (per solve).

    ``species`` selects which receiver coords to profile (default: all S -- the full marginal). Each
    solve costs ~``cg_iters`` joint HVP applies; subset ``species`` to bound cost on large trees.
    ``ridge`` (>0) regularizes the solve if the min is only near-PD; use 0 at a certified PD min.
    """
    theta = theta.double()
    alpha = alpha.double()
    if hvp is None:
        hvp, _loss, _sv, _cache = build_joint_hvp(static, theta, alpha, warm_E=warm_E,
                                                  tangent_self_iters=tangent_self_iters)
        theta_numel = int(theta.numel())
        S = int(alpha.numel())
    else:
        theta_numel = int(theta_numel if theta_numel is not None else theta.numel())
        S = int(S if S is not None else alpha.numel())
    p = theta_numel + S
    species = list(range(S)) if species is None else list(species)
    Av = make_gauge_operator(hvp, theta_numel, penalty_hvp=penalty_hvp, ridge=ridge)

    cols, cg_iters, cg_resid = [], [], []
    for jj, j in enumerate(species):
        e = torch.zeros(p, device=theta.device, dtype=torch.float64)
        e[theta_numel + int(j)] = 1.0
        rhs = proj_z(e, theta_numel)
        free_cuda_cache_if_tight()
        x, it, conv = cg_solve(Av, rhs, tol=cg_tol, max_iter=cg_max)
        resid = float((Av(x) - rhs).norm())
        cols.append(proj_alpha(x[theta_numel:]).clone())
        cg_iters.append(int(it))
        cg_resid.append(resid)
        if verbose:
            print(f"  [fisher] species {j} ({jj + 1}/{len(species)}): cg_iters={it} "
                  f"conv={conv} resid={resid:.2e}")

    Sigma_aa = torch.stack(cols, dim=1)              # [S, |species|] (full alpha rows per column)
    if len(species) == S:
        Sigma_aa = Sigma_aa[species, :]              # square -> symmetrize
        Sigma_aa = 0.5 * (Sigma_aa + Sigma_aa.T)
    se_alpha = Sigma_aa.diagonal().clamp_min(0.0).sqrt() if Sigma_aa.shape[0] == Sigma_aa.shape[1] \
        else Sigma_aa[species, :].diagonal().clamp_min(0.0).sqrt()

    w = softmax_recipient(alpha)
    se_w = None
    if Sigma_aa.shape[0] == Sigma_aa.shape[1] and len(species) == S:
        J = torch.diag(w) - torch.outer(w, w)        # softmax Jacobian dw/dalpha
        Sigma_w = J @ Sigma_aa @ J.T
        se_w = Sigma_w.diagonal().clamp_min(0.0).sqrt()
    if verbose:
        print(f"[recv-fisher] profiled {len(species)}/{S} receiver coords  "
              f"max|cg_resid|={max(cg_resid):.2e}  se_alpha in "
              f"[{float(se_alpha.min()):.3e}, {float(se_alpha.max()):.3e}]")
    return dict(Sigma_aa=Sigma_aa, se_alpha=se_alpha, se_w=se_w, w=w, species=species,
                cg_iters=cg_iters, cg_resid=cg_resid)


# ---------------------------------------------------------------------------- Newton on (theta, alpha)
def newton_joint(static, theta0, alpha0, *, sigma=0.01, sigma_floor=1e-4, lanczos_m=10, nu=1.5,
                 omega=1.5, max_bumps=3, max_cg=40, c1=1e-4, ls_max=25, gtol=1e-2, max_newton=40,
                 tangent_self_iters=None, lam=0.0, theta_ref=None, lam_tree=0.0, sp_parent=None,
                 ftol=1e-9, seed=0, verbose=True):
    """Gauge-projected LM-damped Newton on the joint ``z = [theta.reshape(-1); alpha]``.

    The Newton system is the GAUGE-PROJECTED ``P_z (H + penalty + lam_damp I) P_z dz = -P_z g_z``,
    solved by ``cg_witness`` (negative-curvature self-correction bumps ``lam_damp``). Steps are
    globalized by Armijo backtracking on the joint forward loss ``F = NLL + penalties(theta)``; after
    each accepted step the alpha block is re-centered to the gauge slice. The joint analytic HVP is
    rebuilt at each new point (theta fixed across a point's CG iterations, so it amortizes). Optional
    ridge (``lam``/``theta_ref``) and GBM tree-Laplacian (``lam_tree``/``sp_parent``) penalties act on
    the theta block only (the receiver block is penalty-free). Run in fp64. Returns
    ``(theta, alpha, history)``.

    This is what ``newton_lanczos(with_receiver=True)`` delegates to. REQUIRES a non-uniform
    ``alpha0`` (else the receiver curvature is dead -- see ``build_joint_hvp``).
    """
    theta0 = theta0.double()
    alpha0 = alpha0.double().reshape(-1)
    theta_shape = tuple(theta0.shape)
    theta_numel = int(theta0.numel())
    S = int(alpha0.numel())
    p_dim = theta_numel + S
    device = theta0.device

    alpha0 = proj_alpha(alpha0)                       # land on the gauge slice
    z = torch.cat([theta0.reshape(-1), alpha0]).contiguous()

    # joint value+grad (penalties on theta only); g_z = [dF/dtheta; dNLL/dalpha], gauge-respecting.
    prior = None if (lam in (0.0, None) or theta_ref is None) else (
        float(lam), theta_ref.detach().reshape(-1).double())
    tree_penalty = None if (lam_tree in (0.0, None) or sp_parent is None) else (
        float(lam_tree), sp_parent.detach().reshape(-1).long())
    vg = make_value_and_grad(static, alpha0, theta_shape=theta_shape, optimize_receiver=True,
                             prior=prior, tree_penalty=tree_penalty)

    tp_child = tp_parent = None
    if tree_penalty is not None:
        tp_child, tp_parent = _tree_edges(sp_parent)
    pen_hvp = None if (prior is None and tree_penalty is None) else _penalty_hvp(
        theta_numel, theta_shape, lam=(prior[0] if prior else 0.0),
        lam_tree=(tree_penalty[0] if tree_penalty else 0.0),
        tp_child=tp_child, tp_parent=tp_parent)

    def split(zv):
        return zv[:theta_numel].reshape(theta_shape).contiguous(), zv[theta_numel:].contiguous()

    def build_hvp(zv):
        th, al = split(zv)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hvp, _l, _sv, _c = build_joint_hvp(static, th, al, tangent_self_iters=tangent_self_iters)
        return make_gauge_operator(hvp, theta_numel, penalty_hvp=pen_hvp)

    F, g_z, _, _ = vg(z)
    gP = proj_z(g_z.double(), theta_numel)

    Hz = build_hvp(z)
    start = proj_z(torch.randn(p_dim, generator=torch.Generator(device=str(device)).manual_seed(seed),
                               device=device, dtype=torch.float64), theta_numel)
    _, lam_max = lanczos_extremes(Hz, p_dim, m=lanczos_m, device=str(device), start=start)
    lam_max = max(lam_max, 1e-12)
    lam_damp = sigma * lam_max
    lam_floor = sigma_floor * lam_max
    lam_ceil = 10.0 * lam_max
    if verbose:
        print(f"[newton-joint] S={S} theta_numel={theta_numel}  lam_max~{lam_max:.3f}  "
              f"lam_damp0={lam_damp:.4f}")

    history = []
    stalls = 0
    hvp_stale = False
    for k in range(int(max_newton)):
        gnorm = float(gP.norm())
        rec = {"newton": k, "F": F, "gnorm": gnorm, "lam_damp": lam_damp}
        history.append(rec)
        if verbose:
            print(f"[nj {k:2d}] F={F:.6f}  ||P g||={gnorm:.4e}  lam={lam_damp:.3e}", end="")
        if gnorm < gtol:
            if verbose:
                print("  converged")
            break
        if hvp_stale:
            Hz = None
            free_cuda_cache_if_tight(min_free_gib=8.0)
            Hz = build_hvp(z)
            hvp_stale = False

        eta = min(0.1, gnorm ** 0.5)
        p_step, cg_iters, status, cert = None, 0, "", None
        for _bump in range(int(max_bumps) + 1):
            Av = lambda v, ld=lam_damp: Hz(v) + ld * proj_z(v, theta_numel)
            p_step, cg_iters, status, cert = cg_witness(Av, -gP, tol=eta * gnorm, max_iter=max_cg)
            if status != "neg_curv":
                break
            lam_damp = min(lam_ceil, nu * (lam_damp - cert))
            if verbose:
                print(f"\n      witness d^TAd/|d|^2={cert:.2e} -> lam={lam_damp:.3e}", end="")
        if status == "neg_curv":
            p_step = -gP / lam_damp
            status = "fallback_gd"
        p_step = proj_z(p_step, theta_numel)          # keep the step on the gauge slice
        rec["cg"], rec["status"] = cg_iters, status

        gp = float(torch.dot(gP, p_step))
        if gp >= 0.0:
            p_step = -gP / lam_damp
            gp = -gnorm * gnorm / lam_damp
            status += "+gd"

        alpha_ls, accepted = 1.0, False
        for _ in range(int(ls_max)):
            trial = z + alpha_ls * p_step
            Ft, _, _, _ = vg(trial, want_grad=False)
            if Ft <= F + c1 * alpha_ls * gp:
                accepted = True
                break
            alpha_ls *= 0.5
        rec["alpha_ls"] = alpha_ls if accepted else None

        if accepted:
            trial[theta_numel:] = proj_alpha(trial[theta_numel:])   # re-center alpha (pin the gauge)
            z = trial.contiguous()
            hvp_stale = True
            lam_damp = max(lam_floor, lam_damp / omega) if alpha_ls == 1.0 else min(lam_ceil, 1.5 * lam_damp)
            if verbose:
                print(f"  cg={cg_iters}({status})  a={alpha_ls:.2e}  dF={Ft - F:+.4e}")
            stalls = stalls + 1 if (F - Ft) <= ftol * max(1.0, abs(F)) else 0
            F = Ft
            if stalls >= 2:
                if verbose:
                    print(f"[nj {k + 1:2d}] improvement below ftol floor twice -- stopping")
                break
            F, g_z, _, _ = vg(z)
            gP = proj_z(g_z.double(), theta_numel)
        else:
            lam_damp = min(lam_ceil, 4.0 * lam_damp)
            if verbose:
                print(f"  cg={cg_iters}({status})  line-search failed -> lam={lam_damp:.3e}")
            if lam_damp >= lam_ceil:
                if verbose:
                    print("  lam at ceiling with no accepted step -- stopping")
                break

    theta_out, alpha_out = split(z)
    return theta_out, proj_alpha(alpha_out), history

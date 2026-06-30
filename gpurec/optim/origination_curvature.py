"""Gauge-projected joint ``(theta, alpha, omega)`` curvature consumers of the analytic exact HVP.

Both the recipient logits ``alpha`` and the origination logits ``omega`` enter the NLL only through a
full (unpinned) ``log_softmax``, so the loss is exactly invariant under ``alpha -> alpha + c*1`` and
``omega -> omega + c*1``. The joint Hessian ``H`` of ``z = [theta.reshape(-1); alpha; omega]`` therefore
has TWO exact zero eigenvalues / null directions: ``[0; 1_S; 0]`` and ``[0; 0; 1_S]``. Every consumer
here works in the gauge-fixed subspace via the block projector

    P_z = blockdiag( I_theta , I_S - 11^T/S , I_S - 11^T/S )   (identity on theta, mean-subtract on alpha and omega)

so both gauge nulls are removed and the reduced operator is the genuine curvature. The joint HVP is the
analytic ``make_exact_hvp`` (forward-over-reverse; FD-verified to ~1e-9 over all three blocks,
including the weighted theta/alpha blocks and the omega cross-blocks). This module CONSUMES it:

  * ``build_joint_hvp``       -- forward-solve + per-point cache + the joint exact HVP over z.
  * ``certify_joint_min``     -- gauge-fixed reduced-Hessian PD certificate (gauge-projected Lanczos).
  * ``origination_information`` -- gauge-fixed marginal covariance / standard errors of ``omega`` (and,
    by the delta method, of the origination distribution ``p = softmax(omega)``); the observed Fisher
    information for the origination weights, Schur-correct over (theta, alpha).

REQUIRES a NON-UNIFORM ``alpha`` AND ``omega`` (at a uniform base the corresponding curvature is dead).
Run everything in fp64 (pass fp64 theta/alpha/omega).
"""

from __future__ import annotations

import torch

from gpurec.core.inference.solver import origination_weights_are_uniform, receiver_weights_are_uniform
from gpurec.core.parameters.extract_parameters import origination_log_probs_from_weights
from gpurec.optim.cg import cg_solve, cg_witness, lanczos_extremes, lanczos_min_eigpair
from gpurec.optim.hvp_exact import build_point_cache, make_exact_hvp
from gpurec.optim.receiver_curvature import _penalty_hvp, _tree_edges
from gpurec.optim.value_and_grad import forward_solve, free_cuda_cache_if_tight, make_value_and_grad


# --------------------------------------------------------------------------- gauge projector (P_z)
def proj_z(u: torch.Tensor, theta_numel: int, S: int) -> torch.Tensor:
    """``P_z u`` -- identity on theta, mean-subtract on the alpha block and the omega block."""
    th = u[:theta_numel]
    a = u[theta_numel:theta_numel + S]
    o = u[theta_numel + S:theta_numel + 2 * S]
    return torch.cat([th, a - a.mean(), o - o.mean()])


def softmax_origination(omega: torch.Tensor) -> torch.Tensor:
    """Origination distribution ``p = softmax(omega)`` (the model's ``2^{origination_log_probs}``)."""
    return torch.softmax(omega.double(), dim=-1)


# ---------------------------------------------------------------------------- joint-HVP construction
def build_joint_hvp(static, theta, alpha, omega, *, warm_E=None, tangent_self_iters=None, sv=None,
                    cache=None):
    """Forward-solve at ``(theta, alpha, omega)``, build the per-point adjoint cache once, and return
    the analytic JOINT exact HVP ``hvp(u_vec)`` over ``z = [theta.reshape(-1); alpha; omega]`` (length
    ``theta.numel() + 2S``). Returns ``(hvp, loss, sv, cache)``.

    REQUIRES non-uniform ``alpha`` AND ``omega`` -- at a uniform base the corresponding weighted paths
    are dead and that block of the joint HVP degenerates.
    """
    if receiver_weights_are_uniform(alpha):
        raise ValueError("build_joint_hvp requires a NON-uniform alpha (perturb it first).")
    if origination_weights_are_uniform(omega):
        raise ValueError("build_joint_hvp requires a NON-uniform omega (perturb it first).")
    lw = origination_log_probs_from_weights(omega)
    w = torch.exp2(lw)
    if sv is None:
        loss, sv = forward_solve(static, theta, alpha, warm_E=warm_E)
    else:
        loss = None
    if cache is None:
        _gt, _gc, cache = build_point_cache(static, theta, alpha, sv,
                                            origination_log_probs=lw, origination_probs=w)
    hvp = make_exact_hvp(static, theta, alpha, sv, cache=cache, tangent_self_iters=tangent_self_iters,
                         origination_log_probs=lw, origination_probs=w, origination_weights=omega)
    return hvp, (None if loss is None else float(loss)), sv, cache


def make_gauge_operator(hvp, theta_numel, S, *, penalty_hvp=None, ridge=0.0):
    """``A_z(v) = P_z ( H + penalty ) ( P_z v ) + ridge * P_z v`` -- the symmetric gauge-projected joint
    operator. Both gauge nulls ``[0;1_S;0]`` and ``[0;0;1_S]`` map to 0 (killed by the input projection).
    ``penalty_hvp`` (theta block only) makes the Newton model exact under ridge/tree penalties."""
    ridge = float(ridge or 0.0)

    def Av(v):
        pv = proj_z(v, theta_numel, S)
        Hv = hvp(pv)
        if penalty_hvp is not None:
            Hv = Hv + penalty_hvp(pv)
        out = proj_z(Hv, theta_numel, S)
        if ridge != 0.0:
            out = out + ridge * pv
        return out

    return Av


# --------------------------------------------------------------------------------- PD certificate
def certify_joint_min(static, theta, alpha, omega, *, m=200, seed=0, tangent_self_iters=None,
                      warm_E=None, hvp=None, theta_numel=None, S=None, verbose=True):
    """Gauge-fixed reduced-Hessian PD certificate for the joint ``(theta, alpha, omega)`` minimum.

    Gauge-projected Lanczos for the smallest reduced-Hessian eigenpair, deflating BOTH gauge nulls.
    Returns a dict with ``lam_min_gauge`` (PD iff > 0), ``ritz_resid``, ``pd``, ``v_min``.
    """
    theta, alpha, omega = theta.double(), alpha.double(), omega.double()
    if hvp is None:
        hvp, _loss, _sv, _cache = build_joint_hvp(static, theta, alpha, omega, warm_E=warm_E,
                                                  tangent_self_iters=tangent_self_iters)
        theta_numel = int(theta.numel())
        S = int(alpha.numel())
    else:
        theta_numel = int(theta_numel if theta_numel is not None else theta.numel())
        S = int(S if S is not None else alpha.numel())
    p = theta_numel + 2 * S
    Av = make_gauge_operator(hvp, theta_numel, S)
    gen = torch.Generator(device=str(theta.device)).manual_seed(seed)
    start = proj_z(torch.randn(p, generator=gen, device=theta.device, dtype=torch.float64), theta_numel, S)

    # Deflate the 2-D gauge null (spanned by [0;1;0] and [0;0;1]): shift v - P_z v (the null
    # component) up by C >> spectrum top, so the smallest eigenvalue of the shifted operator is the
    # genuine reduced-Hessian minimum on the gauge-fixed subspace.
    _, lam_max = lanczos_extremes(Av, p, m=min(20, p), device=str(theta.device), start=start)
    shift_C = 2.0 * max(abs(lam_max), 1.0)

    def Av_deflated(v):
        return Av(v) + shift_C * (v - proj_z(v, theta_numel, S))

    lam_min, v_min = lanczos_min_eigpair(Av_deflated, p, m=m, device=str(theta.device), start=start)
    Av_v = Av(v_min)
    ritz_resid = float((Av_v - lam_min * v_min).norm()) / max(abs(lam_min), 1.0)
    gauge_comp = float((v_min - proj_z(v_min, theta_numel, S)).norm())
    pd = lam_min > 0.0
    if verbose:
        tag = "PD (certified gauge-fixed joint min)" if pd else "NOT PD (saddle / non-stationary)"
        print(f"[orig-cert] lam_min_gauge={lam_min:+.6e}  ritz_resid={ritz_resid:.2e}  "
              f"gauge_comp={gauge_comp:.2e}  m={m}  -> {tag}")
    return dict(lam_min_gauge=lam_min, ritz_resid=ritz_resid, pd=bool(pd), gauge_comp=gauge_comp,
                v_min=v_min, m=int(m), S=S, theta_numel=theta_numel)


# --------------------------------------------------------------------- Fisher / uncertainty (omega)
def origination_information(static, theta, alpha, omega, *, hvp=None, theta_numel=None, S=None,
                            species=None, cg_tol=1e-7, cg_max=400, ridge=0.0, tangent_self_iters=None,
                            warm_E=None, verbose=True):
    """Gauge-fixed marginal covariance / standard errors of the origination logits ``omega`` at a
    (certified) joint minimum -- the observed Fisher information for the origination weights,
    Schur-complement-correct over (theta, alpha) (each column is a full joint CG solve).

    Returns a dict: ``Sigma_oo`` [|species|,|species|] gauge-fixed marginal covariance, ``se_omega``
    per-species s.e. of the origination logits, ``se_p`` delta-method s.e. of the origination
    probabilities ``p = softmax(omega)`` (``Sigma_p = J Sigma_oo J^T``, ``J = diag(p) - p p^T``),
    ``p``, ``species``, ``cg_iters``, ``cg_resid``.
    """
    theta, alpha, omega = theta.double(), alpha.double(), omega.double()
    if hvp is None:
        hvp, _loss, _sv, _cache = build_joint_hvp(static, theta, alpha, omega, warm_E=warm_E,
                                                  tangent_self_iters=tangent_self_iters)
        theta_numel = int(theta.numel())
        S = int(alpha.numel())
    else:
        theta_numel = int(theta_numel if theta_numel is not None else theta.numel())
        S = int(S if S is not None else alpha.numel())
    p_dim = theta_numel + 2 * S
    o0 = theta_numel + S  # offset of the omega block
    species = list(range(S)) if species is None else list(species)
    Av = make_gauge_operator(hvp, theta_numel, S, ridge=ridge)

    cols, cg_iters, cg_resid = [], [], []
    for jj, j in enumerate(species):
        e = torch.zeros(p_dim, device=theta.device, dtype=torch.float64)
        e[o0 + int(j)] = 1.0
        rhs = proj_z(e, theta_numel, S)
        free_cuda_cache_if_tight()
        x, it, conv = cg_solve(Av, rhs, tol=cg_tol, max_iter=cg_max)
        og = x[o0:o0 + S]
        cols.append((og - og.mean()).clone())  # gauge-fixed omega rows
        cg_iters.append(int(it))
        cg_resid.append(float((Av(x) - rhs).norm()))
        if verbose:
            print(f"  [orig-fisher] species {j} ({jj + 1}/{len(species)}): cg_iters={it} "
                  f"conv={conv} resid={cg_resid[-1]:.2e}")

    Sigma_oo = torch.stack(cols, dim=1)
    if len(species) == S:
        Sigma_oo = Sigma_oo[species, :]
        Sigma_oo = 0.5 * (Sigma_oo + Sigma_oo.T)
    se_omega = Sigma_oo.diagonal().clamp_min(0.0).sqrt()

    pdist = softmax_origination(omega)
    se_p = None
    if len(species) == S:
        J = torch.diag(pdist) - torch.outer(pdist, pdist)  # softmax Jacobian dp/domega
        Sigma_p = J @ Sigma_oo @ J.T
        se_p = Sigma_p.diagonal().clamp_min(0.0).sqrt()
    if verbose:
        print(f"[orig-fisher] profiled {len(species)}/{S} origination coords  "
              f"max|cg_resid|={max(cg_resid):.2e}  se_omega in "
              f"[{float(se_omega.min()):.3e}, {float(se_omega.max()):.3e}]")
    return dict(Sigma_oo=Sigma_oo, se_omega=se_omega, se_p=se_p, p=pdist, species=species,
                cg_iters=cg_iters, cg_resid=cg_resid)


# --------------------------------------------------------------- Newton on (theta, alpha, omega)
def newton_joint(static, theta0, alpha0, omega0, *, sigma=0.01, sigma_floor=1e-4, lanczos_m=10,
                 nu=1.5, decrease=1.5, max_bumps=3, max_cg=40, c1=1e-4, ls_max=25, gtol=1e-2,
                 max_newton=40, tangent_self_iters=None, lam=0.0, theta_ref=None, lam_tree=0.0,
                 sp_parent=None, ftol=1e-9, seed=0, verbose=True):
    """Gauge-projected LM-damped Newton on the joint ``z = [theta.reshape(-1); alpha; omega]``.

    The Newton system is the gauge-projected ``P_z (H + penalty + lam_damp I) P_z dz = -P_z g_z``,
    solved by ``cg_witness`` (negative-curvature self-correction bumps ``lam_damp``). Steps are
    globalized by Armijo backtracking on the joint forward loss; after each accepted step BOTH the
    alpha and omega blocks are re-centered to the gauge slice (``P_z``). The joint analytic HVP is
    rebuilt at each new point. Optional ridge (``lam``/``theta_ref``) and GBM tree-Laplacian
    (``lam_tree``/``sp_parent``) penalties act on the theta block only. Run in fp64. Returns
    ``(theta, alpha, omega, history)``. REQUIRES non-uniform ``alpha0`` and ``omega0``.
    """
    theta0 = theta0.double()
    alpha0 = alpha0.double().reshape(-1)
    omega0 = omega0.double().reshape(-1)
    theta_shape = tuple(theta0.shape)
    theta_numel = int(theta0.numel())
    S = int(alpha0.numel())
    p_dim = theta_numel + 2 * S
    device = theta0.device

    alpha0 = alpha0 - alpha0.mean()           # land on the gauge slice
    omega0 = omega0 - omega0.mean()
    z = torch.cat([theta0.reshape(-1), alpha0, omega0]).contiguous()

    prior = None if (lam in (0.0, None) or theta_ref is None) else (
        float(lam), theta_ref.detach().reshape(-1).double())
    tree_penalty = None if (lam_tree in (0.0, None) or sp_parent is None) else (
        float(lam_tree), sp_parent.detach().reshape(-1).long())
    vg = make_value_and_grad(static, alpha0, theta_shape=theta_shape, optimize_receiver=True,
                             optimize_origination=True, origination_weights=omega0,
                             prior=prior, tree_penalty=tree_penalty)

    tp_child = tp_parent = None
    if tree_penalty is not None:
        tp_child, tp_parent = _tree_edges(sp_parent)
    pen_hvp = None if (prior is None and tree_penalty is None) else _penalty_hvp(
        theta_numel, theta_shape, lam=(prior[0] if prior else 0.0),
        lam_tree=(tree_penalty[0] if tree_penalty else 0.0), tp_child=tp_child, tp_parent=tp_parent)

    def split(zv):
        return (zv[:theta_numel].reshape(theta_shape).contiguous(),
                zv[theta_numel:theta_numel + S].contiguous(),
                zv[theta_numel + S:].contiguous())

    def build_hvp(zv):
        th, al, om = split(zv)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hvp, _l, _sv, _c = build_joint_hvp(static, th, al, om, tangent_self_iters=tangent_self_iters)
        return make_gauge_operator(hvp, theta_numel, S, penalty_hvp=pen_hvp)

    F, g_z, _, _ = vg(z)
    gP = proj_z(g_z.double(), theta_numel, S)
    Hz = build_hvp(z)
    start = proj_z(torch.randn(p_dim, generator=torch.Generator(device=str(device)).manual_seed(seed),
                               device=device, dtype=torch.float64), theta_numel, S)
    _, lam_max = lanczos_extremes(Hz, p_dim, m=lanczos_m, device=str(device), start=start)
    lam_max = max(lam_max, 1e-12)
    lam_damp = sigma * lam_max
    lam_floor = sigma_floor * lam_max
    lam_ceil = 10.0 * lam_max
    if verbose:
        print(f"[newton-joint(orig)] S={S} theta_numel={theta_numel}  lam_max~{lam_max:.3f}  "
              f"lam_damp0={lam_damp:.4f}")

    history, stalls, hvp_stale = [], 0, False
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
            Av = lambda v, ld=lam_damp: Hz(v) + ld * proj_z(v, theta_numel, S)
            p_step, cg_iters, status, cert = cg_witness(Av, -gP, tol=eta * gnorm, max_iter=max_cg)
            if status != "neg_curv":
                break
            lam_damp = min(lam_ceil, nu * (lam_damp - cert))
            if verbose:
                print(f"\n      witness d^TAd/|d|^2={cert:.2e} -> lam={lam_damp:.3e}", end="")
        if status == "neg_curv":
            p_step, status = -gP / lam_damp, "fallback_gd"
        p_step = proj_z(p_step, theta_numel, S)
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
            z = proj_z(trial, theta_numel, S).contiguous()   # re-center alpha AND omega (pin gauge)
            hvp_stale = True
            lam_damp = max(lam_floor, lam_damp / decrease) if alpha_ls == 1.0 else min(lam_ceil, 1.5 * lam_damp)
            if verbose:
                print(f"  cg={cg_iters}({status})  a={alpha_ls:.2e}  dF={Ft - F:+.4e}")
            stalls = stalls + 1 if (F - Ft) <= ftol * max(1.0, abs(F)) else 0
            F = Ft
            if stalls >= 2:
                if verbose:
                    print(f"[nj {k + 1:2d}] improvement below ftol floor twice -- stopping")
                break
            F, g_z, _, _ = vg(z)
            gP = proj_z(g_z.double(), theta_numel, S)
        else:
            lam_damp = min(lam_ceil, 4.0 * lam_damp)
            if verbose:
                print(f"  cg={cg_iters}({status})  line-search failed -> lam={lam_damp:.3e}")
            if lam_damp >= lam_ceil:
                if verbose:
                    print("  lam at ceiling with no accepted step -- stopping")
                break

    theta_out, alpha_out, omega_out = split(z)
    return theta_out, alpha_out - alpha_out.mean(), omega_out - omega_out.mean(), history

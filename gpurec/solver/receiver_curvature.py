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

from gpurec.config.newton import NewtonOptions
from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.solver import curvature as _curv
from gpurec.solver.cg import cg_solve
from gpurec.solver.hvp_exact import build_point_cache, make_exact_hvp
from gpurec.solver.value_and_grad import (
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
    return _curv.gauge_operator(hvp, lambda v: proj_z(v, theta_numel),
                                penalty_hvp=penalty_hvp, ridge=ridge)


# --------------------------------------------------------------------------------- PD certificate
def certify_joint_min(static, theta, alpha, *, m=None, seed=None, tangent_self_iters=None, warm_E=None,
                      hvp=None, theta_numel=None, S=None, penalty_hvp=None, verbose=True,
                      newton: NewtonOptions | None = None):
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
    ``m``/``seed`` default (``None``) to ``NewtonOptions.certify_m``/``seed`` (pass ``newton=`` to
    override the whole block, or ``m=``/``seed=`` directly -- back-compat kwargs).
    """
    theta = theta.double()
    alpha = alpha.double()
    opts = _curv.resolve_newton(newton, certify_m=m, seed=seed)
    if hvp is None:
        hvp, _loss, _sv, _cache = build_joint_hvp(static, theta, alpha, warm_E=warm_E,
                                                  tangent_self_iters=tangent_self_iters)
        theta_numel = int(theta.numel())
        S = int(alpha.numel())
    else:
        theta_numel = int(theta_numel if theta_numel is not None else theta.numel())
        S = int(S if S is not None else alpha.numel())
    p = theta_numel + S
    # Deflated gauge-projected Lanczos (the gauge null [0; 1_S] is an EXACT zero eigenvalue of A_z;
    # certify_min shifts it up so the smallest eigenvalue of the shifted operator is the genuine
    # reduced-Hessian minimum, and reports the Ritz residual against the UNSHIFTED A_z).
    lam_min, ritz_resid, v_min, gauge_comp, leak = _curv.certify_min(
        hvp, lambda v: proj_z(v, theta_numel), p, m=opts.certify_m, seed=opts.seed,
        penalty_hvp=penalty_hvp, device=theta.device, leak_fn=lambda Hv: _alpha_leak(Hv, theta_numel))
    pd = lam_min > 0.0
    if verbose:
        tag = "PD (certified gauge-fixed joint min)" if pd else "NOT PD (saddle / non-stationary)"
        print(f"[recv-cert] lam_min_gauge={lam_min:+.6e}  ritz_resid={ritz_resid:.2e}  "
              f"raw-HVP leak={leak:.2e}  gauge_comp={gauge_comp:.2e}  m={opts.certify_m}  -> {tag}")
    return dict(lam_min_gauge=lam_min, ritz_resid=ritz_resid, leak=leak, pd=bool(pd),
                gauge_comp=gauge_comp, v_min=v_min, m=int(opts.certify_m), S=S, theta_numel=theta_numel)


# ------------------------------------------------------------------------- Fisher / uncertainty (alpha)
def receiver_information(static, theta, alpha, *, hvp=None, theta_numel=None, S=None, species=None,
                        cg_tol=None, cg_max=None, ridge=0.0, tangent_self_iters=None, warm_E=None,
                        penalty_hvp=None, verbose=True, newton: NewtonOptions | None = None):
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
    ``cg_tol``/``cg_max`` default (``None``) to ``NewtonOptions.cg_tol``/``cg_max``.
    """
    theta = theta.double()
    alpha = alpha.double()
    opts = _curv.resolve_newton(newton, cg_tol=cg_tol, cg_max=cg_max)
    cg_tol, cg_max = opts.cg_tol, opts.cg_max
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
def newton_joint(static, theta0, alpha0, *, sigma=None, sigma_floor=None, lanczos_m=None, nu=None,
                 omega=None, max_bumps=None, max_cg=None, c1=None, ls_max=None, gtol=None,
                 max_newton=None, tangent_self_iters=None, lam=0.0, theta_ref=None, lam_tree=0.0,
                 sp_parent=None, ftol=None, seed=None, verbose=True, newton: NewtonOptions | None = None):
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

    All the LM/Lanczos/CG/line-search knobs (``sigma``, ``sigma_floor``, ``lanczos_m``, ``nu``,
    ``omega``, ``max_bumps``, ``max_cg``, ``c1``, ``ls_max``, ``gtol``, ``max_newton``, ``ftol``,
    ``seed``) default (``None``) to the matching ``NewtonOptions`` field -- pass ``newton=`` to
    override the whole block at once, or any of these kwargs directly (back-compat).
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

    def build_hvp(zv):
        th = zv[:theta_numel].reshape(theta_shape).contiguous()
        al = zv[theta_numel:].contiguous()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hvp, _l, _sv, _c = build_joint_hvp(static, th, al, tangent_self_iters=tangent_self_iters)
        return make_gauge_operator(hvp, theta_numel, penalty_hvp=pen_hvp)

    # ``omega`` is the accepted-step damping-decrease factor (kept as the public kwarg name for
    # backward compatibility with newton_cg); it maps to NewtonOptions.decrease / the shared core's
    # ``decrease`` field.
    opts = _curv.resolve_newton(
        newton, sigma=sigma, sigma_floor=sigma_floor, lanczos_m=lanczos_m, nu=nu, decrease=omega,
        max_bumps=max_bumps, max_cg=max_cg, c1=c1, ls_max=ls_max, gtol=gtol, max_newton=max_newton,
        ftol=ftol, seed=seed)
    z, history = _curv.newton_min(
        z, p_dim, lambda v: proj_z(v, theta_numel), vg, build_hvp, theta_numel=theta_numel, S=S,
        newton=opts, device=device, tag="newton-joint", verbose=verbose)

    theta_out = z[:theta_numel].reshape(theta_shape).contiguous()
    alpha_out = z[theta_numel:].contiguous()
    return theta_out, proj_alpha(alpha_out), history

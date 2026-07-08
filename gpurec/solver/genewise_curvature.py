import torch
from gpurec.solver.hvp_exact import make_exact_hvp

from gpurec.api._execution import evaluate_static_loss_grad, stream_batches
from gpurec.config.newton import NewtonOptions
from gpurec.solver import curvature as _curv
from gpurec.solver.origination_curvature import build_joint_hvp
from gpurec.solver.value_and_grad import forward_solve, free_cuda_cache_if_tight


def _assemble_dense_arrowhead(blocks, g_theta, g_omega, g_alpha, mu):
    """TEST-ONLY oracle: materialize the full arrowhead Hessian + mu*I and the stacked gradient.

    NOT wired into the fit -- see the TEST-ONLY / block-order / PSD-assumption banner on
    ``newton_step_joint`` below; this oracle assembles the dense counterpart of that same
    non-production structured solver.

    Block/flat order matches ``newton_step_joint``'s output packing and the test's
    ``cat([dth.reshape(-1), dom.reshape(-1), dal.reshape(-1)])``:
        [ theta_0..theta_{G-1} (3 each) | omega_0..omega_{G-1} (S each) | alpha (S) ].
    ``H_oo_g = diag(H_oo_diag_g) + H_oo_lr_g @ H_oo_lr_g^T`` is materialized densely here (only here).
    Returns ``(Hd, gd)`` so the test can compare against ``torch.linalg.solve(Hd, -gd)``.
    """
    H_tt = blocks["H_tt"]            # [G,3,3]
    H_to = blocks["H_to"]            # [G,3,S]
    H_oo_diag = blocks["H_oo_diag"]  # [G,S]
    H_oo_lr = blocks["H_oo_lr"]      # [G,S,r]
    H_aa = blocks["H_aa"]            # [S,S]
    H_za = blocks["H_za"]            # [G,3+S,S]
    G = int(H_tt.shape[0])
    S = int(H_aa.shape[0])
    dev, dt = H_aa.device, H_aa.dtype
    N = 3 * G + G * S + S
    Hd = torch.zeros(N, N, device=dev, dtype=dt)

    al = slice(3 * G + G * S, 3 * G + G * S + S)
    for g in range(G):
        ts = slice(3 * g, 3 * g + 3)
        os_ = slice(3 * G + g * S, 3 * G + g * S + S)
        H_oo = torch.diag(H_oo_diag[g]) + H_oo_lr[g] @ H_oo_lr[g].T  # [S,S]
        # per-family core B_g = [[H_tt, H_to],[H_to^T, H_oo]]
        Hd[ts, ts] = H_tt[g]
        Hd[ts, os_] = H_to[g]
        Hd[os_, ts] = H_to[g].T
        Hd[os_, os_] = H_oo
        # couplings z_g = [theta_g; omega_g] <-> alpha
        H_ta = H_za[g][:3, :]   # [3,S]  theta_g <-> alpha
        H_oa = H_za[g][3:, :]   # [S,S]  omega_g <-> alpha
        Hd[ts, al] = H_ta
        Hd[al, ts] = H_ta.T
        Hd[os_, al] = H_oa
        Hd[al, os_] = H_oa.T
    Hd[al, al] = H_aa
    Hd = Hd + mu * torch.eye(N, device=dev, dtype=dt)

    gd = torch.cat([g_theta.reshape(-1), g_omega.reshape(-1), g_alpha.reshape(-1)])
    return Hd, gd


def newton_step_joint(blocks, g_theta, g_omega, g_alpha, mu):
    """TEST-ONLY / NOT wired into the fit. This structured solver uses the internal flat block order
    ``[theta; omega; alpha]`` and assumes a POSITIVE-SEMIDEFINITE low-rank ``H_oo = D + UU^T``. The
    real per-family ``H_omegaomega`` is INDEFINITE, and the live/canonical flat layout used by the fit
    is ``[theta; alpha; omega]``. The production genewise Newton fit runs matrix-free CG directly on
    the analytic HVP (``newton_joint_genewise``) and never calls this solver -- do NOT wire it into
    the fit without first deriving a signed low-rank ``H_oo`` and reconciling the block order.

    Arrowhead Newton solve of ``(H + mu*I) delta = -g`` via Schur complement + Woodbury.

    ``H`` is block-diagonal per-family cores ``B_g = [[H_tt_g, H_to_g],[H_to_g^T, H_oo_g]]``
    (each ``(3+S)``) plus a global ``alpha`` arrow (dense ``H_aa`` ``S x S`` + couplings
    ``H_za_g`` ``(3+S) x S``). ``H_oo_g = diag(H_oo_diag_g) + H_oo_lr_g @ H_oo_lr_g^T`` is applied
    matrix-free via Woodbury -- the dense ``S x S`` ``H_oo`` is NEVER formed.

    Args:
        blocks: dict with H_tt[G,3,3], H_to[G,3,S], H_oo_diag[G,S], H_oo_lr[G,S,r],
            H_aa[S,S], H_za[G,3+S,S].
        g_theta[G,3], g_omega[G,S], g_alpha[S]: gradient pieces.
        mu: Levenberg damping (added to the whole diagonal).
    Returns:
        (dtheta[G,3], domega[G,S], dalpha[S]).
    """
    H_tt = blocks["H_tt"]            # [G,3,3]
    H_to = blocks["H_to"]            # [G,3,S]
    H_oo_diag = blocks["H_oo_diag"]  # [G,S]
    H_oo_lr = blocks["H_oo_lr"]      # [G,S,r]
    H_aa = blocks["H_aa"]            # [S,S]
    H_za = blocks["H_za"]            # [G,3+S,S]
    G = int(H_tt.shape[0])
    S = int(H_aa.shape[0])
    r = int(H_oo_lr.shape[-1])
    dev, dt = H_aa.device, H_aa.dtype

    A = H_tt + mu * torch.eye(3, device=dev, dtype=dt)   # [G,3,3]  H_tt + mu I_3
    Bmat = H_to                                          # [G,3,S]
    Bt = H_to.transpose(1, 2)                            # [G,S,3]
    d = H_oo_diag + mu                                   # [G,S]    diag of H_oo + mu
    U = H_oo_lr                                          # [G,S,r]
    dinv = (1.0 / d).unsqueeze(-1)                       # [G,S,1]

    # Woodbury factor for D = diag(d) + U U^T:  M = I_r + U^T D_d^{-1} U  (SPD).
    Ir = torch.eye(r, device=dev, dtype=dt)
    M = Ir + U.transpose(1, 2) @ (dinv * U)              # [G,r,r]
    Lm = torch.linalg.cholesky(M)

    def Dinv(X):  # X:[G,S,k] -> (diag(d)+UU^T)^{-1} X  via Woodbury
        DiX = dinv * X
        t = U.transpose(1, 2) @ DiX                      # [G,r,k]
        w = torch.cholesky_solve(t, Lm)                  # [G,r,k]
        return DiX - dinv * (U @ w)

    # Schur complement of D inside B_g:  S_A = A - Bmat D^{-1} Bmat^T  (3x3).
    DinvBt = Dinv(Bt)                                    # [G,S,3]  D^{-1} Bmat^T
    S_A = A - Bmat @ DinvBt                              # [G,3,3]
    lu_A, piv_A = torch.linalg.lu_factor(S_A)

    def Binv(RHS):  # RHS:[G,3+S,k] -> B_g^{-1} RHS  (2x2 block solve, 3 vs S)
        r1 = RHS[:, :3, :]                               # [G,3,k]
        r2 = RHS[:, 3:, :]                               # [G,S,k]
        Dinv_r2 = Dinv(r2)
        w1 = r1 - Bmat @ Dinv_r2                         # [G,3,k]
        x1 = torch.linalg.lu_solve(lu_A, piv_A, w1)      # [G,3,k]  S_A^{-1} w1
        x2 = Dinv(r2 - Bt @ x1)                          # [G,S,k]
        return torch.cat([x1, x2], dim=1)               # [G,3+S,k]

    g_z = torch.cat([g_theta, g_omega], dim=1)          # [G,3+S]
    H_az = H_za.transpose(1, 2)                          # [G,S,3+S]

    BinvHza = Binv(H_za)                                 # [G,3+S,S]  B^{-1} H_za
    Binv_gz = Binv(g_z.unsqueeze(-1))                    # [G,3+S,1]  B^{-1} g_z

    # Dense S x S alpha system: (H_aa + mu I - sum_g H_az B^{-1} H_za) d_alpha = -(g_a - sum_g H_az B^{-1} g_z)
    S_alpha = H_aa + mu * torch.eye(S, device=dev, dtype=dt) - (H_az @ BinvHza).sum(0)
    rhs_alpha = -g_alpha + (H_az @ Binv_gz).squeeze(-1).sum(0)
    d_alpha = torch.linalg.solve(S_alpha, rhs_alpha)     # [S]

    # Back-substitute: d_z_g = B_g^{-1}(-g_z_g - H_za_g d_alpha)
    rhs_z = -g_z - (H_za @ d_alpha)                      # [G,3+S]
    d_z = Binv(rhs_z.unsqueeze(-1)).squeeze(-1)          # [G,3+S]
    return d_z[:, :3].contiguous(), d_z[:, 3:].contiguous(), d_alpha


# ==================================================================================================
# Matrix-free gauge-projected joint Newton-CG over the GENEWISE parameter
#   z = [theta.reshape(-1) (3G); alpha (S); omega.reshape(-1) (G*S)],   p_dim = 3G + S + G*S.
#
# Genewise analog of gpurec.solver.origination_curvature (which handles the specieswise / global-omega
# case). It reuses the validated joint analytic HVP (build_joint_hvp -> make_exact_hvp, FD-verified
# genewise in Tasks 4/5) and the operator-based CG/Lanczos primitives (gpurec.solver.cg) verbatim.
#
# The ONLY genewise deltas vs origination_curvature are (i) the gauge projector: genewise omega has a
# softmax gauge PER FAMILY, so there are G omega-nulls plus 1 alpha-null (proj_z_genewise below kills
# all G+1); (ii) the joint value-and-grad is built directly from evaluate_static_loss_grad because
# make_value_and_grad's joint layout assumes a GLOBAL omega [S]; (iii) sizes/layout carry G. The
# HVP itself is genewise-correct as-is (theta [G,3], omega [G,S]).
# ==================================================================================================


def proj_z_genewise(u, theta_numel, S, G):
    """Gauge projector ``P_z u`` for genewise ``z = [theta (3G); alpha (S); omega (G*S)]``.

    Identity on the theta block; mean-subtract the single alpha block (1 gauge null); mean-subtract
    EACH of the G omega rows independently (G gauge nulls -- one softmax gauge per family). Kills all
    G+1 exact-zero directions of the joint Hessian on input, so the reduced operator built from it is
    the genuine curvature on the gauge-fixed subspace.
    """
    th = u[:theta_numel]
    a = u[theta_numel:theta_numel + S]
    o = u[theta_numel + S:theta_numel + S + G * S].reshape(G, S)
    return torch.cat([th, a - a.mean(), (o - o.mean(dim=1, keepdim=True)).reshape(-1)])


def make_gauge_operator_genewise(hvp, theta_numel, S, G, *, ridge=0.0):
    """``A_z(v) = P_z ( H (P_z v) ) + ridge * P_z v`` -- symmetric gauge-projected joint operator.

    All G+1 gauge nulls map to 0 (killed by the input projection). Mirrors
    ``curvature.gauge_operator`` with the genewise projector.
    """
    return _curv.gauge_operator(hvp, lambda v: proj_z_genewise(v, theta_numel, S, G), ridge=ridge)


def certify_joint_min_genewise(static, theta, alpha, omega, *, m=120, seed=0,
                               tangent_self_iters=None, warm_E=None, hvp=None, theta_numel=None,
                               S=None, G=None, verbose=True):
    """Gauge-fixed reduced-Hessian PD certificate for the genewise ``(theta, alpha, omega)`` point.

    Gauge-projected Lanczos for the smallest reduced-Hessian eigenpair, deflating ALL G+1 gauge nulls:
    ``proj_z_genewise`` kills them on input, and the ``shift_C * (v - P_z v)`` term lifts any residual
    null component above the spectrum top so the bottom Ritz value is the genuine minimum on the
    gauge-fixed subspace. PD is NOT guaranteed -- the point may be a saddle or the omega block may be
    near-singular; ``lam_min_gauge`` is reported honestly (PD iff > 0). Returns a dict with
    ``lam_min_gauge``, ``ritz_resid``, ``pd``, ``gauge_comp``, ``v_min``. Run in fp64.
    """
    theta, alpha, omega = theta.double(), alpha.double(), omega.double()
    S = int(S if S is not None else alpha.numel())
    theta_numel = int(theta_numel if theta_numel is not None else theta.numel())
    G = int(G if G is not None else omega.reshape(-1).numel() // S)
    # `static` is EITHER a single static (single-batch, bit-for-bit) OR a batch_statics list (len>1).
    statics = static if isinstance(static, (list, tuple)) else [static]
    multibatch = len(statics) > 1
    if hvp is None:
        if multibatch:
            hvp = make_multibatch_joint_hvp_genewise(statics, theta, alpha, omega,
                                                     tangent_self_iters=tangent_self_iters)
        else:
            hvp, _loss, _sv, _cache = build_joint_hvp(statics[0], theta, alpha, omega, warm_E=warm_E,
                                                      tangent_self_iters=tangent_self_iters)
    p = theta_numel + S + G * S
    # Deflated gauge-projected Lanczos over all G+1 gauge nulls (shared core; genewise projector).
    lam_min, ritz_resid, v_min, gauge_comp, _leak = _curv.certify_min(
        hvp, lambda v: proj_z_genewise(v, theta_numel, S, G), p, m=m, seed=seed, device=theta.device)
    pd = lam_min > 0.0
    if verbose:
        tag = "PD (certified gauge-fixed genewise min)" if pd else "NOT PD (saddle / near-singular)"
        print(f"[gw-cert] lam_min_gauge={lam_min:+.6e}  ritz_resid={ritz_resid:.2e}  "
              f"gauge_comp={gauge_comp:.2e}  m={m}  -> {tag}")
    return dict(lam_min_gauge=lam_min, ritz_resid=ritz_resid, pd=bool(pd), gauge_comp=gauge_comp,
                v_min=v_min, m=int(m), S=S, G=G, theta_numel=theta_numel)


def newton_joint_genewise(static, theta0, alpha0, omega0, *, sigma=None, sigma_floor=None,
                          lanczos_m=None, nu=None, decrease=None, max_bumps=None, max_cg=None, c1=None,
                          ls_max=None, gtol=None, max_newton=None, tangent_self_iters=None, ftol=None,
                          seed=None, cert_m=120, verbose=True, newton: NewtonOptions | None = None):
    """Gauge-projected LM-damped Newton-CG on the genewise joint ``z = [theta (3G); alpha (S); omega (G*S)]``.

    Mirrors ``origination_curvature.newton_joint`` with three genewise deltas: (i) ``proj_z_genewise``
    (G+1 gauge nulls -- one alpha null plus one softmax null per family); (ii) loss/grad come from
    ``evaluate_static_loss_grad`` (genewise theta ``[G,3]``, omega ``[G,S]``; g_alpha is the receiver
    grad); (iii) the joint analytic HVP is ``build_joint_hvp`` (genewise-correct as-is). Each Newton
    system ``P (H + lam*I) P dz = -P g`` is solved by ``cg_witness`` (negative-curvature self-correction
    bumps ``lam``), globalized by Armijo backtracking on the joint forward loss; after each accepted step
    alpha is re-centered and EACH omega row is re-centered to the gauge slice, and the HVP is rebuilt at
    the new point. REQUIRES non-uniform ``alpha0`` AND ``omega0`` (build_joint_hvp raises otherwise). Run
    in fp64. Returns ``(theta, alpha, omega, history)`` where ``history`` is a dict holding
    ``gnorm_init``/``gnorm_final`` (``||P g||`` at the start / final iterate), the final ``lam_min`` and
    ``pd`` (from ``certify_joint_min_genewise``, PD not forced), the per-iteration ``iters`` trace, and
    the full ``cert`` dict.

    All the LM/Lanczos/CG/line-search knobs (``sigma``, ``sigma_floor``, ``lanczos_m``, ``nu``,
    ``decrease``, ``max_bumps``, ``max_cg``, ``c1``, ``ls_max``, ``gtol``, ``max_newton``, ``ftol``,
    ``seed``) default (``None``) to the matching ``NewtonOptions`` field -- pass ``newton=`` to
    override the whole block at once, or any of these kwargs directly (back-compat). ``cert_m`` is
    the genewise certificate's OWN default (120, distinct from ``NewtonOptions.certify_m``'s 200 --
    the two are not the same knob) and stays a plain literal.
    """
    theta0 = theta0.double()
    alpha0 = alpha0.double().reshape(-1)
    omega0 = omega0.double()
    theta_shape = tuple(theta0.shape)          # (G, 3)
    G = int(theta0.shape[0])
    theta_numel = int(theta0.numel())          # 3G
    S = int(alpha0.numel())
    p_dim = theta_numel + S + G * S
    device = theta0.device
    # `static` is EITHER a single static / [static] (single-batch, bit-for-bit) OR a batch_statics
    # list (len>1). p_dim uses the FULL G (sum of per-batch families), independent of batch count;
    # everything below (proj/LM/CG/line-search/certificate) operates on the flat [theta;alpha;omega].
    statics = static if isinstance(static, (list, tuple)) else [static]
    multibatch = len(statics) > 1
    single = statics[0]

    alpha0 = alpha0 - alpha0.mean()                                              # land on the gauge slice
    omega0 = omega0.reshape(G, S)
    omega0 = omega0 - omega0.mean(dim=1, keepdim=True)                           # per-family row-center
    z = torch.cat([theta0.reshape(-1), alpha0, omega0.reshape(-1)]).contiguous()

    def split(zv):
        th = zv[:theta_numel].reshape(theta_shape).contiguous()
        al = zv[theta_numel:theta_numel + S].contiguous()
        om = zv[theta_numel + S:].reshape(G, S).contiguous()
        return th, al, om

    # True multi-batch joint value-and-grad (9a): sums per-batch losses, index_add/scatter of the
    # disjoint per-family theta/omega grads + plain sum of the shared alpha grad. Built once.
    _mb_vg = multibatch_joint_vg_genewise(statics, theta_shape, S, G) if multibatch else None

    def vg(zv, *, want_grad=True):
        th, al, om = split(zv)
        if multibatch:
            if want_grad:
                free_cuda_cache_if_tight()
                loss, g_z, _, _ = _mb_vg(zv)
                return float(loss), g_z.double(), None, None
            loss, _, _, _ = stream_batches(statics, th, al, om, genewise=True, need_grad=False)
            return float(loss), None, None, None
        if want_grad:
            free_cuda_cache_if_tight()
            loss, g_th, g_al, g_om = evaluate_static_loss_grad(
                single, th, al, om, need_grad=True, need_origination_grad=True)
            g_z = torch.cat([g_th.reshape(-1), g_al.reshape(-1), g_om.reshape(-1)]).double()
            return float(loss), g_z, None, None
        loss, _, _, _ = evaluate_static_loss_grad(single, th, al, om, need_grad=False)
        return float(loss), None, None, None

    def build_hvp(zv):
        th, al, om = split(zv)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if multibatch:
            hvp = make_multibatch_joint_hvp_genewise(statics, th, al, om,
                                                     tangent_self_iters=tangent_self_iters)
        else:
            hvp, _l, _sv, _c = build_joint_hvp(single, th, al, om, tangent_self_iters=tangent_self_iters)
        return make_gauge_operator_genewise(hvp, theta_numel, S, G)

    # Gauge-projected LM-Newton loop (shared core; genewise projector + the per-family vg/build_hvp
    # closures above). ``decrease`` is the accepted-step damping-decrease factor.
    proj = lambda v: proj_z_genewise(v, theta_numel, S, G)
    opts = _curv.resolve_newton(
        newton, sigma=sigma, sigma_floor=sigma_floor, lanczos_m=lanczos_m, nu=nu, decrease=decrease,
        max_bumps=max_bumps, max_cg=max_cg, c1=c1, ls_max=ls_max, gtol=gtol, max_newton=max_newton,
        ftol=ftol, seed=seed)
    z, trace = _curv.newton_min(
        z, p_dim, proj, vg, build_hvp, theta_numel=theta_numel, S=S,
        newton=opts, device=device, tag="newton-joint(gw)", verbose=verbose)

    theta_out, alpha_out, omega_out = split(z)
    alpha_out = alpha_out - alpha_out.mean()
    omega_out = omega_out - omega_out.mean(dim=1, keepdim=True)
    gnorm_init = float(trace[0]["gnorm"]) if trace else 0.0        # ||P g|| at the start (rec 0)
    _F, g_final, _, _ = vg(z)                                     # final gradient at the polished iterate
    gnorm_final = float(proj(g_final.double()).norm())

    # Final gauge-fixed PD certificate at the polished point (fresh HVP; PD is NOT forced).
    if multibatch:
        hvp_f = make_multibatch_joint_hvp_genewise(statics, theta_out, alpha_out, omega_out,
                                                   tangent_self_iters=tangent_self_iters)
    else:
        hvp_f, _l, _sv, _c = build_joint_hvp(single, theta_out, alpha_out, omega_out,
                                             tangent_self_iters=tangent_self_iters)
    cert = certify_joint_min_genewise(static, theta_out, alpha_out, omega_out, m=cert_m, hvp=hvp_f,
                                      theta_numel=theta_numel, S=S, G=G, verbose=verbose)
    history = dict(gnorm_init=gnorm_init, gnorm_final=gnorm_final, lam_min=cert["lam_min_gauge"],
                   pd=cert["pd"], iters=trace, cert=cert)
    return theta_out, alpha_out, omega_out, history


# ==================================================================================================
# Multi-batch genewise joint HVP + value-and-grad.
#
# Genewise theta [G,3] and omega [G,S] are PER-FAMILY -- each family lives in exactly ONE batch,
# identified by static.family_index_tensor (GLOBAL indices into [0,G)). alpha [S] is GLOBAL/shared.
# The joint Hessian over z=[theta(3G); alpha(S); omega(G*S)] is ADDITIVE over families/batches
# (H = sum_families d^2 NLL); batches hold DISJOINT family sets. So the multi-batch HVP is the SUM
# of per-batch single-batch exact HVPs, each acting on its own families, with per-family
# gather/scatter on the theta/omega blocks and a plain sum on the shared alpha block. This is the
# same summation principle as experiments/sanderson_cv/certify.make_exact_multibatch_hvp (the
# specieswise template), extended with the genewise per-family index bookkeeping.
#
# API asymmetry consumed below (verified against gpurec.api._execution / value_and_grad):
#   * forward_solve([static_b], theta, alpha) RE-selects the batch families from a FULL [G,3] theta
#     via static_b.family_index_tensor (genewise) -> it is passed the FULL theta.
#   * make_exact_hvp([static_b], theta_b, alpha, sv_b, origination_weights=omega_b) consumes the
#     PRE-selected theta_b / omega_b directly (no internal select).
#   * evaluate_static_loss_grad(static_b, theta_b, alpha, omega, ...) uses theta DIRECTLY but
#     RE-selects origination_weights internally (origination_weights_for_static) -> it is passed the
#     PRE-selected theta_b but the FULL omega (passing omega_b would double-select, out of bounds).
# ==================================================================================================


def make_multibatch_joint_hvp_genewise(batch_statics, theta, alpha, omega, *, tangent_self_iters=None):
    """Multi-batch genewise joint HVP over ``z = [theta (3G); alpha (S); omega (G*S)]``.

    Builds one single-batch exact joint HVP per batch (over that batch's OWN families, in the
    batch-local layout ``[theta_b (3 G_b); alpha (S); omega_b (S G_b)]``) and returns ``Av(u)`` that
    gathers each family's rows into the per-batch operator, applies it, and scatters the outputs
    back: ``index_add`` on the DISJOINT per-family theta/omega blocks (pure placement) and a plain
    sum on the SHARED global alpha block (the arrowhead spine). Requires non-uniform ``alpha`` and,
    for a live omega block, non-uniform ``omega``. Run in fp64. ``tangent_self_iters`` is forwarded
    to each ``make_exact_hvp`` (fixed per-wave tangent self-loop count).
    """
    theta = theta.double()
    alpha = alpha.double()
    omega = omega.double()
    G = int(theta.shape[0])
    S = int(alpha.numel())
    dev, dt = theta.device, theta.dtype

    batch_hvps = []  # [(fam_b, hvp_b)] one per batch
    for static_b in batch_statics:
        fam_b = static_b.family_index_tensor.to(device=dev)
        theta_b = theta.index_select(0, fam_b).contiguous()   # [G_b, 3]
        omega_b = omega.index_select(0, fam_b).contiguous()   # [G_b, S]
        # FULL theta to forward_solve (it re-selects the batch families internally); theta_b/omega_b
        # to make_exact_hvp (it consumes them directly). Both see the identical batch-local theta_b.
        _loss, sv_b = forward_solve([static_b], theta, alpha)
        hvp_b = make_exact_hvp([static_b], theta_b, alpha, sv_b,
                               tangent_self_iters=tangent_self_iters, origination_weights=omega_b)
        batch_hvps.append((fam_b, hvp_b))

    def Av(u):
        u = u.to(device=dev, dtype=dt)
        u_theta = u[:3 * G].reshape(G, 3)
        u_alpha = u[3 * G:3 * G + S].contiguous()
        u_omega = u[3 * G + S:3 * G + S + G * S].reshape(G, S)
        out_theta = torch.zeros(G, 3, device=dev, dtype=dt)
        out_alpha = torch.zeros(S, device=dev, dtype=dt)
        out_omega = torch.zeros(G, S, device=dev, dtype=dt)
        for fam_b, hvp_b in batch_hvps:
            G_b = int(fam_b.numel())
            u_theta_b = u_theta.index_select(0, fam_b)   # [G_b, 3]
            u_omega_b = u_omega.index_select(0, fam_b)   # [G_b, S]
            u_b = torch.cat([u_theta_b.reshape(-1), u_alpha, u_omega_b.reshape(-1)])
            o_b = hvp_b(u_b).to(dtype=dt)
            out_theta.index_add_(0, fam_b, o_b[:3 * G_b].reshape(G_b, 3))
            out_alpha = out_alpha + o_b[3 * G_b:3 * G_b + S]
            out_omega.index_add_(0, fam_b, o_b[3 * G_b + S:3 * G_b + S + G_b * S].reshape(G_b, S))
        return torch.cat([out_theta.reshape(-1), out_alpha, out_omega.reshape(-1)])

    return Av


def multibatch_joint_vg_genewise(batch_statics, theta_shape, S, G):
    """Multi-batch joint value-and-grad ``vg(x, warm_E=None) -> (loss, g_z, None, None)`` over
    ``z = [theta (3G); alpha (S); omega (G*S)]`` (matches ``_fd_hessian_hvp``'s vg contract) -- this is
    the joint ``[theta; alpha; omega]`` gradient consumed by the FD gate and by ``newton_joint_genewise``.

    Loops the batches; per batch evaluates the genewise loss + grad over that batch's OWN families
    (``theta_b`` gathered by ``static.family_index_tensor``; the FULL per-family ``omega`` [G,S] is
    passed because ``evaluate_static_loss_grad`` re-selects the batch rows internally). Aggregates the
    DISJOINT theta/omega grads by ``index_add`` and the SHARED alpha grad by sum -- the per-family
    origination (omega) grad aggregation here is consistent with the now-fixed ``stream_batches``,
    which likewise ``index_add_``s the batch-local ``[G_b,S]`` origination grad into the full ``[G,S]``
    accumulator when ``origination_weights.ndim == 2``.
    """
    theta_shape = tuple(theta_shape)
    nt = 1
    for _d in theta_shape:
        nt *= int(_d)

    def vg(x, warm_E=None):
        x = x.double()
        th = x[:nt].reshape(G, 3)            # FULL theta [G,3]
        al = x[nt:nt + S].contiguous()       # global alpha [S]
        om = x[nt + S:nt + S + G * S].reshape(G, S)   # FULL omega [G,S]
        dev, dt = x.device, x.dtype
        loss = 0.0
        g_theta = torch.zeros(G, 3, device=dev, dtype=dt)
        g_alpha = torch.zeros(S, device=dev, dtype=dt)
        g_omega = torch.zeros(G, S, device=dev, dtype=dt)
        for static_b in batch_statics:
            fam_b = static_b.family_index_tensor.to(device=dev)
            theta_b = th.index_select(0, fam_b).contiguous()   # [G_b,3]; omega stays FULL (re-selected inside)
            loss_b, gth_b, gal_b, gom_b = evaluate_static_loss_grad(
                static_b, theta_b, al, om, need_grad=True, need_origination_grad=True)
            loss = loss + float(loss_b)
            g_theta.index_add_(0, fam_b, gth_b.to(dtype=dt))
            g_omega.index_add_(0, fam_b, gom_b.to(dtype=dt))
            g_alpha = g_alpha + gal_b.to(dtype=dt)
        g_z = torch.cat([g_theta.reshape(-1), g_alpha.reshape(-1), g_omega.reshape(-1)])
        return loss, g_z, None, None

    return vg

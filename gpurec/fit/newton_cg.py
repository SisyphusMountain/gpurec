"""Levenberg-Marquardt Newton-CG outer loop using the analytic exact HVP.

Each step: forward+backward at theta -> (loss, g, sv); CG solves (M + lambda I) p = -g with the
matrix-free Hessian HVP; an Armijo backtracking line search on the deterministic forward loss
accepts a step and adapts lambda.
"""

from __future__ import annotations

import math

import torch

from gpurec.solver.value_and_grad import make_value_and_grad, forward_solve
from gpurec.solver.krylov import cg_witness, lanczos_extremes


def newton_lanczos(static, theta0, receiver_weights, *, sigma=0.01, sigma_floor=1e-4, lanczos_m=10,
                   nu=1.5, omega=1.5, max_bumps=3, eta_max=0.1, max_cg=40, c1=1e-4, ls_max=25,
                   gtol=1e-2, max_newton=40, lam=0.0, theta_ref=None,
                   lanczos_refresh=0, ftol=1e-9, verbose=True,
                   with_receiver=False, alpha0=None):
    """Lanczos-initialized, witness-corrected damped Newton descent ("Newton-gradient descent").

    ``with_receiver=True`` runs the JOINT ``(theta, alpha)`` receiver-weight Newton: it delegates to
    :func:`gpurec.solver.curvature.receiver.newton_joint`, which drives the gauge-projected
    ``P_z (H + lam I) P_z`` system with the analytic joint exact HVP (Newton steps on ``alpha``, not
    just ``theta``). ``alpha0`` is the starting receiver logits (defaults to ``receiver_weights``;
    must be NON-uniform or the receiver curvature is dead). The return is then ``(theta, alpha,
    history)``. The theta-only path (``with_receiver=False``) is unchanged.

    lam_damp interpolates between Newton (small) and scaled gradient descent (large). It is
    initialized by the cheap spectral rule ``lam_damp = sigma * lam_max`` (m~10 Lanczos: only
    lam_max is needed, which converges almost immediately; no lam_min estimation). At runtime the
    CG negative-curvature witness self-corrects: if CG on ``H_eff + lam_damp*I`` encounters
    ``d^T A d <= 0``, the direction certifies the damping needed and lam_damp is bumped to
    ``nu *`` that magnitude and the solve restarted. Steps are globalized by Armijo backtracking
    on the (deterministic) forward loss; lam_damp decays toward ``sigma_floor*lam_max`` on full
    steps and grows on backtracked/failed ones.

    ``lam``/``theta_ref`` optionally add the ridge/MAP objective term (as in ``newton_tr``):
    F = L + lam/2 ||x - theta_ref||^2 with H_eff = H + lam*I — required for a quadratic endgame
    on this problem's flat-at-the-optimum spectrum. Run in fp64 (pass an fp64 theta0).

    Returns (theta, history); history rows carry loss/F, ||gF||, lam_damp, cg iters/status,
    witness certificates, alpha, and cumulative gradient-eval count.
    """
    if with_receiver:
        # JOINT (theta, alpha) Newton: delegate to the gauge-projected solver built on the analytic
        # joint exact HVP. Returns (theta, alpha, history).
        from gpurec.solver.curvature.receiver import newton_joint

        return newton_joint(
            static, theta0, receiver_weights if alpha0 is None else alpha0,
            sigma=sigma, sigma_floor=sigma_floor, lanczos_m=lanczos_m, nu=nu, omega=omega,
            max_bumps=max_bumps, max_cg=max_cg, c1=c1, ls_max=ls_max, gtol=gtol,
            max_newton=max_newton, lam=lam, theta_ref=theta_ref, ftol=ftol, verbose=verbose,
        )
    theta_shape = tuple(theta0.shape)
    theta_vec = theta0.reshape(-1).clone()
    p_dim = theta_vec.numel()
    vg = make_value_and_grad(static, receiver_weights, theta_shape=theta_shape)
    lam_obj = float(lam)
    x_ref = (theta_vec if theta_ref is None else theta_ref.reshape(-1).to(theta_vec)).double().clone()
    evals = [0]  # cumulative forward+backward evaluations, tracked via wrapper

    def vg_counted(x, **kw):
        evals[0] += 1
        return vg(x, **kw)

    def penalty(x):
        return 0.5 * lam_obj * float((x.double() - x_ref).norm() ** 2) if lam_obj > 0 else 0.0

    loss, g, sv, warm_E = vg_counted(theta_vec)
    sv = None

    def make_hvp_eff(x_vec, warm):
        # One forward+backward builds the per-point adjoint cache; each CG iteration then costs
        # one tangent-forward and one tangent-adjoint sweep.
        from gpurec.solver.hvp.exact import make_exact_hvp

        theta_m = x_vec.reshape(theta_shape)
        _, sv_pt = forward_solve(static, theta_m, receiver_weights, warm_E=warm)
        evals[0] += 2
        h = make_exact_hvp(static, theta_m, receiver_weights, sv_pt)
        if lam_obj > 0:
            return lambda v: h(v).double() + lam_obj * v.double()
        return lambda v: h(v).double()

    # The initial gradient eval (vg_counted above) runs a full forward+backward whose freed scratch
    # FRAGMENTS the caching-allocator pool. A single exact HVP fits from a clean pool, but the
    # fragmented pool can't find a contiguous [C,S] (~2.36 GiB on 1007x64) -> the descent OOMs in
    # step-0 CG even though Lanczos and isolated HVPs pass. empty_cache() unconditionally here
    # returns the gradient's blocks to the driver so the cache build + CG run on a clean pool.
    # Cheap once per descent; on fixtures that already fit it is a no-op-cost defrag.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    hvp_eff = make_hvp_eff(theta_vec, warm_E)
    _, lam_max = lanczos_extremes(hvp_eff, p_dim, m=lanczos_m, device=str(theta_vec.device))
    lam_damp = sigma * lam_max
    lam_floor = sigma_floor * lam_max
    lam_ceil = 10.0 * lam_max
    if verbose:
        print(f"[lanczos-newton] m={lanczos_m}  lam_max~{lam_max:.2f}  "
              f"lam_damp0={lam_damp:.4f}  floor={lam_floor:.2e}  ceil={lam_ceil:.1f}")

    history = []
    accepted_steps = 0
    stalls = 0
    # hvp_eff was just built for the current theta_vec (the Lanczos build above), so it is NOT
    # stale entering the loop -- the k=0 rebuild is redundant (theta unchanged) and its
    # free-old+build-new churn spikes the memory high-water mark, which OOMs the descent on the
    # big fixtures even though a single HVP fits. Only rebuild after theta_vec actually moves.
    hvp_stale = False
    for k in range(int(max_newton)):
        gF = g.double() + (lam_obj * (theta_vec.double() - x_ref) if lam_obj > 0 else 0.0)
        F = loss + penalty(theta_vec)
        gnorm = float(torch.linalg.vector_norm(gF))
        rec = {"newton": k, "loss": loss, "F": F, "gnorm": gnorm, "lam_damp": lam_damp,
               "witness_certs": [], "evals": evals[0]}
        history.append(rec)
        if verbose:
            print(f"[ln {k:2d}] F={F:.6f}  ||gF||={gnorm:.4e}  lam={lam_damp:.3e}", end="")
        if gnorm < gtol:
            if verbose:
                print("  converged")
            break

        # rebuild the HVP cache only when theta_vec has moved since it was built. Dropping the
        # previous closure (which pins its GB-sized cached forward sv) and returning those blocks to
        # the driver BEFORE building the next point's cache keeps only one point's forward
        # intermediates live at once (else the backward's driver-free scratch gate trips on the big
        # fixtures). A periodic lanczos_refresh also forces a rebuild to re-estimate lam_max.
        from gpurec.solver.value_and_grad import free_cuda_cache_if_tight
        do_refresh = bool(lanczos_refresh and accepted_steps
                          and accepted_steps % int(lanczos_refresh) == 0)
        if hvp_stale or do_refresh:
            hvp_eff = None
            # higher threshold than the default 4 GiB: the inter-step line searches + gradient eval
            # re-fragment the pool, so on the tight big fixtures (1007x64) defrag before the rebuild
            # to keep a contiguous [C,S] available. On roomy fixtures (666x80, >8 GiB free) this is a
            # no-op (skips empty_cache), so it does not slow them.
            free_cuda_cache_if_tight(min_free_gib=8.0)
            hvp_eff = make_hvp_eff(theta_vec, warm_E)
            hvp_stale = False
            if do_refresh:
                _, lam_max = lanczos_extremes(hvp_eff, p_dim, m=lanczos_m, device=str(theta_vec.device))
                lam_floor = sigma_floor * lam_max
                lam_ceil = 10.0 * lam_max

        # damped solve with witness self-correction
        eta = min(eta_max, gnorm ** 0.5)
        p, cg_iters, status = None, 0, ""
        for bump in range(int(max_bumps) + 1):
            Av = lambda v: hvp_eff(v) + lam_damp * v
            p, cg_iters, status, cert = cg_witness(Av, -gF, tol=eta * gnorm, max_iter=max_cg)
            if status != "neg_curv":
                break
            new_lam = nu * (lam_damp - cert)  # cert <= 0 is the damped Rayleigh quotient
            rec["witness_certs"].append(lam_damp - cert)
            if verbose:
                print(f"\n      witness: d^T(H+lam)d/|d|^2={cert:.3e} -> lam {lam_damp:.3e} -> {new_lam:.3e}", end="")
            lam_damp = min(lam_ceil, new_lam)
        if status == "neg_curv":  # bumps exhausted
            p = -gF / lam_damp
            status = "fallback_gd"
        rec["cg"] = cg_iters
        rec["status"] = status

        gp = float(torch.dot(gF, p))
        if gp >= 0.0:
            p = -gF / lam_damp
            gp = -gnorm * gnorm / lam_damp
            status += "+gd"

        # Armijo backtracking on the deterministic forward loss
        alpha, accepted, sv_t = 1.0, False, None
        for _ in range(int(ls_max)):
            trial = (theta_vec.double() + alpha * p).to(theta_vec.dtype)
            lt, st = forward_solve(static, trial.reshape(theta_shape), receiver_weights, warm_E=warm_E)
            Ft = float(lt) + penalty(trial)
            # A step large enough to leave the E-step's contractive region makes forward_solve
            # return a non-finite loss (survival normalization diverges -> NLL -> -inf). Armijo
            # `Ft <= F + ...` would then *accept* it, since -inf <= (finite) is True, cascading the
            # whole polish to F=-inf and breaking the adjoint solve. Require a finite Ft so such a
            # step is rejected and the search backtracks (or fails cleanly and bumps lambda).
            if math.isfinite(Ft) and Ft <= F + c1 * alpha * gp:
                accepted, sv_t = True, st
                break
            alpha *= 0.5
        rec["alpha"] = alpha if accepted else None

        if accepted:
            accepted_steps += 1
            theta_vec = trial
            hvp_stale = True  # theta moved -> the cached HVP must be rebuilt next iteration
            # multi-batch forward_solve returns saved=None (streams+frees the ~GB intermediates);
            # fall back to a cold warm-start (correct, just no warm-start speedup). Single-batch
            # sv_t is a real dict, so this guard is a no-op there.
            warm_E = sv_t["E"] if sv_t is not None else None
            sv_t = None
            lam_damp = max(lam_floor, lam_damp / omega) if alpha == 1.0 else min(lam_ceil, 1.5 * lam_damp)
            if verbose:
                print(f"  cg={cg_iters}({status})  a={alpha:.2e}  dF={Ft - F:+.4e}")
            # accepted improvements at the forward solver's truncation floor are noise; two in a
            # row means further polishing cannot be validated -- stop instead of micro-stepping
            stalls = stalls + 1 if (F - Ft) <= ftol * max(1.0, abs(F)) else 0
            if stalls >= 2:
                if verbose:
                    print(f"[ln {k + 1:2d}] improvement below ftol floor twice -- stopping")
                break
            loss, g, sv, warm_E = vg_counted(theta_vec, warm_E=warm_E)
            sv = None
        else:
            lam_damp = min(lam_ceil, 4.0 * lam_damp)
            if verbose:
                print(f"  cg={cg_iters}({status})  line-search failed -> lam={lam_damp:.3e}")
            if lam_damp >= lam_ceil:
                if verbose:
                    print("  lam at ceiling with no accepted step -- stopping")
                break

    return theta_vec.reshape(theta_shape), history

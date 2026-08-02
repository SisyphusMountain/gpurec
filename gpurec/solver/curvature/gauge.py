"""Shared gauge-projected joint-curvature core for the receiver ``(theta, alpha)`` and
origination ``(theta, alpha, omega)`` consumers of the analytic exact HVP.

Each softmax weight block (``alpha``; and, for origination, ``omega``) enters the NLL only
through an unpinned ``log_softmax``, so it contributes an exact all-ones zero eigenvalue to the
joint Hessian. Consumers work in the gauge-fixed subspace via a block projector ``P_z`` that
mean-subtracts each softmax block. The pieces that are IDENTICAL across the 1-gauge-block
(receiver) and 2-gauge-block (origination) cases live here, parametrized by the caller's
projector ``proj = P_z``:

  * ``gauge_operator``  -- ``A_z(v) = P_z (H + penalty)(P_z v) + ridge P_z v``.
  * ``certify_min``     -- deflated gauge-projected Lanczos smallest reduced-Hessian eigenpair.
  * ``newton_min``      -- gauge-projected LM-damped Newton loop on ``z``.

``receiver.py`` and ``origination.py`` are thin wrappers that supply ``proj`` +
the HVP builder + the value/grad closure and keep their own public signatures. Run in fp64.
"""

from __future__ import annotations

import dataclasses

import torch

from gpurec.config.memory import MemoryOptions
from gpurec.config.newton import NewtonOptions
from gpurec.solver.krylov import cg_witness, lanczos_extremes, lanczos_min_eigpair
from gpurec.solver.value_and_grad import free_cuda_cache_if_tight


def resolve_newton(newton: "NewtonOptions | None", **overrides) -> NewtonOptions:
    """Resolve a ``NewtonOptions`` instance from an optional base ``newton`` plus explicit-kwarg
    overrides (the deprecation shim for the old copy-pasted individual kwargs): any override value
    that is ``None`` falls back to ``newton`` (or ``NewtonOptions()`` if ``newton`` is also ``None``);
    a non-``None`` override replaces that field. Used by the ``newton_*``/``certify_*`` wrappers so
    existing callers passing e.g. ``max_newton=4`` explicitly keep working unchanged."""
    opts = newton if newton is not None else NewtonOptions()
    active = {k: v for k, v in overrides.items() if v is not None}
    return dataclasses.replace(opts, **active) if active else opts


def gauge_operator(hvp, proj, *, penalty_hvp=None, ridge=0.0):
    """``A_z(v) = P_z ( H + penalty ) ( P_z v ) + ridge * P_z v`` -- symmetric gauge-projected
    joint operator. ``proj`` is the caller's block projector ``P_z``; each gauge null maps to 0
    (killed by the input projection). ``penalty_hvp`` (theta block only) makes the Newton model
    exact under ridge/tree penalties; ``ridge`` adds a uniform shift for solve robustness."""
    ridge = float(ridge or 0.0)

    def Av(v):
        pv = proj(v)
        Hv = hvp(pv)
        if penalty_hvp is not None:
            Hv = Hv + penalty_hvp(pv)
        out = proj(Hv)
        if ridge != 0.0:
            out = out + ridge * pv
        return out

    return Av


def certify_min(hvp, proj, p, *, m=None, seed=None, penalty_hvp=None, device=None, leak_fn=None,
                newton: NewtonOptions | None = None):
    """Gauge-fixed reduced-Hessian PD certificate via deflated gauge-projected Lanczos.

    ``proj`` = ``P_z``; ``p`` = ``len(z)``. Lanczos is started from a ``P_z``-projected random
    vector so the Krylov basis stays gauge-fixed, and the gauge null(s) are deflated by shifting
    ``v - P_z v`` up by ``C >> spectrum-top`` so the smallest eigenvalue of the shifted operator
    is the genuine reduced-Hessian minimum. Returns
    ``(lam_min, ritz_resid, v_min, gauge_comp, leak)`` (``leak`` is ``None`` unless ``leak_fn``
    is given). The Ritz residual is measured against the UNSHIFTED ``A_z``.

    ``m``/``seed`` default (``None``) to ``NewtonOptions.certify_m``/``seed``; pass ``newton=`` to
    override the whole block, or ``m=``/``seed=`` directly (back-compat kwargs)."""
    opts = resolve_newton(newton, certify_m=m, seed=seed)
    m, seed = opts.certify_m, opts.seed
    Av = gauge_operator(hvp, proj, penalty_hvp=penalty_hvp)
    gen = torch.Generator(device=str(device)).manual_seed(seed)
    start = proj(torch.randn(p, generator=gen, device=device, dtype=torch.float64))

    _, lam_max = lanczos_extremes(Av, p, m=min(20, p), device=str(device), start=start)
    shift_C = 2.0 * max(abs(lam_max), 1.0)

    def Av_deflated(v):
        return Av(v) + shift_C * (v - proj(v))

    lam_min, v_min = lanczos_min_eigpair(Av_deflated, p, m=m, device=str(device), start=start)
    Av_v = Av(v_min)
    ritz_resid = float((Av_v - lam_min * v_min).norm()) / max(abs(lam_min), 1.0)
    gauge_comp = float((v_min - proj(v_min)).norm())  # v_min should be gauge-fixed
    leak = leak_fn(hvp(v_min)) if leak_fn is not None else None
    return lam_min, ritz_resid, v_min, gauge_comp, leak


def newton_min(z, p_dim, proj, vg, build_hvp, *, theta_numel, S, newton: NewtonOptions | None = None,
               device=None, tag="newton-joint", verbose=True):
    """Gauge-projected LM-damped Newton on the joint ``z`` (already on the gauge slice).

    The Newton system is the gauge-projected ``P_z (H + penalty + lam_damp I) P_z dz = -P_z g_z``,
    solved by ``cg_witness`` (negative-curvature self-correction bumps ``lam_damp``), globalized by
    Armijo backtracking on the joint forward loss; after each accepted step ``z`` is re-projected
    to the gauge slice. ``vg(z, want_grad=?)`` returns ``(F, g_z, ...)``; ``build_hvp(z)`` returns
    the gauge operator at ``z``. ``newton`` (a ``NewtonOptions``, default ``NewtonOptions()``)
    supplies every LM/Lanczos/CG/line-search knob. Returns ``(z, history)``."""
    opts = newton if newton is not None else NewtonOptions()
    sigma, sigma_floor = opts.sigma, opts.sigma_floor
    lanczos_m, nu, decrease = opts.lanczos_m, opts.nu, opts.decrease
    max_bumps, max_cg, c1, ls_max = opts.max_bumps, opts.max_cg, opts.c1, opts.ls_max
    gtol, max_newton, ftol, seed = opts.gtol, opts.max_newton, opts.ftol, opts.seed

    F, g_z, _, _ = vg(z)
    gP = proj(g_z.double())

    Hz = build_hvp(z)
    start = proj(torch.randn(p_dim, generator=torch.Generator(device=str(device)).manual_seed(seed),
                             device=device, dtype=torch.float64))
    _, lam_max = lanczos_extremes(Hz, p_dim, m=lanczos_m, device=str(device), start=start)
    lam_max = max(lam_max, 1e-12)
    lam_damp = sigma * lam_max
    lam_floor = sigma_floor * lam_max
    lam_ceil = opts.lam_ceil_factor * lam_max
    if verbose:
        print(f"[{tag}] S={S} theta_numel={theta_numel}  lam_max~{lam_max:.3f}  "
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
            free_cuda_cache_if_tight(min_free_gib=MemoryOptions().min_free_gib_hvp)
            Hz = build_hvp(z)
            hvp_stale = False

        eta = min(opts.forcing_eta, gnorm ** 0.5)
        p_step, cg_iters, status, cert = None, 0, "", None
        for _bump in range(int(max_bumps) + 1):
            Av = lambda v, ld=lam_damp: Hz(v) + ld * proj(v)
            p_step, cg_iters, status, cert = cg_witness(Av, -gP, tol=eta * gnorm, max_iter=max_cg)
            if status != "neg_curv":
                break
            lam_damp = min(lam_ceil, nu * (lam_damp - cert))
            if verbose:
                print(f"\n      witness d^TAd/|d|^2={cert:.2e} -> lam={lam_damp:.3e}", end="")
        if status == "neg_curv":
            p_step = -gP / lam_damp
            status = "fallback_gd"
        p_step = proj(p_step)          # keep the step on the gauge slice
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
            z = proj(trial).contiguous()          # re-center each gauge block (pin the gauge)
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
            gP = proj(g_z.double())
        else:
            lam_damp = min(lam_ceil, 4.0 * lam_damp)
            if verbose:
                print(f"  cg={cg_iters}({status})  line-search failed -> lam={lam_damp:.3e}")
            if lam_damp >= lam_ceil:
                if verbose:
                    print("  lam at ceiling with no accepted step -- stopping")
                break

    return z, history

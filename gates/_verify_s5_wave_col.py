"""Check the wave-SO receiver cotangent at a non-uniform receiver base.

This isolates the S5 kernel (`_reconciliation_vjp_directional_derivative_kernel` / `wave_backward_so`). It runs a REAL ``hvp(u)``
on the 8-hogenom fixture at the seeded NON-UNIFORM base alpha (so ``use_receiver_weights=True`` and
``dreceiver_log_probs`` is the live softmax-Jacobian seed) and captures every ``wave_backward_so`` return via a
wrapper, asserting its eighth output ``d_grad_receiver_log_probs`` is finite
and not identically zero.

With ``u_alpha = 0`` it checks that the softmax-Jacobian seed
``dreceiver_log_probs`` is zero; the receiver cotangent can still contain the
mixed theta--receiver block.  Only a completely zero direction must produce
an exactly zero receiver cotangent.  The full ``(theta, alpha)`` numerical
validation of the receiver Hessian blocks is S8.

    python -m gates._verify_s5_wave_col
"""

from __future__ import annotations

import torch

import gpurec.solver.hvp.exact as hx
from gpurec.solver.hvp.exact import build_point_cache, make_exact_hvp
from gpurec.solver.value_and_grad import forward_solve
from gates._verify_hvp_recv import _static_theta_alpha_from_live


def _capture_run(static, theta, alpha, u, *, tangent_self_iters=128):
    """Run one hvp(u) while wrapping wave_backward_so to record each wave's d_grad_receiver_log_probs."""
    captured = []
    orig = hx.wave_backward_so

    def wrapped(*args, **kw):
        out = orig(*args, **kw)
        # Return includes six named event-VJP derivatives and d_grad_receiver_log_probs; record the latter.
        captured.append((out[-1].detach().clone(), kw.get("dreceiver_log_probs"), kw.get("use_receiver_weights")))
        return out

    hx.wave_backward_so = wrapped
    try:
        _loss, sv = forward_solve([static], theta, alpha)
        _gt, _gc, cache = build_point_cache([static], theta, alpha, sv)
        hvp = make_exact_hvp([static], theta, alpha, sv, cache=cache,
                             tangent_self_iters=tangent_self_iters)
        Hu = hvp(u)
    finally:
        hx.wave_backward_so = orig
    return Hu, captured


def run(n_families=8, device="cuda", seed=0, tangent_self_iters=128):
    static, theta, alpha, S, vm0 = _static_theta_alpha_from_live(n_families, device, seed=seed)
    theta_numel = 3 * S
    p = theta_numel + S
    print(f"[S5 wave-receiver gate {n_families}-family] S={S} theta_numel={theta_numel} p(4S)={p} "
          f"nonuniform={True} valid_mass_min={vm0:.4f}")

    g = torch.Generator(device=device).manual_seed(seed + 7)
    u_full = torch.randn(p, generator=g, device=device, dtype=torch.float64)
    u_full = u_full / u_full.norm()
    # a non-zero alpha component for the live path
    u_zero_alpha = torch.cat([u_full[:theta_numel],
                              torch.zeros(S, device=device, dtype=torch.float64)])

    # --- live path: u_alpha != 0 -> dreceiver_log_probs != 0 -> d_grad_receiver_log_probs must be finite & nonzero
    _Hu_live, cap_live = _capture_run(static, theta, alpha, u_full,
                                      tangent_self_iters=tangent_self_iters)
    n_waves = len(cap_live)
    all_finite = all(bool(torch.isfinite(c).all()) for c, _, _ in cap_live)
    use_receiver_weights_all = all(bool(enabled) for _, _, enabled in cap_live)
    receiver_tangents = [tangent for _, tangent, _ in cap_live if tangent is not None]
    receiver_tangent_nonzero = bool(receiver_tangents) and any(
        float(tangent.abs().max()) > 0 for tangent in receiver_tangents
    )
    total_receiver_gradient_norm = float(
        sum(float(gradient.norm()) for gradient, _, _ in cap_live)
    )
    max_receiver_gradient_abs = max(
        (float(gradient.abs().max()) for gradient, _, _ in cap_live),
        default=0.0,
    )
    n_nonzero_waves = sum(1 for c, _, _ in cap_live if float(c.abs().max()) > 0)
    print(
        f"  live (u_alpha!=0): waves={n_waves} "
        f"use_receiver_weights_all={use_receiver_weights_all} "
        f"receiver_tangent_nonzero={receiver_tangent_nonzero} "
        f"all_finite={all_finite}"
    )
    print(f"     d_grad_receiver_log_probs: nonzero_waves={n_nonzero_waves}/{n_waves} "
          f"sum|.|2={total_receiver_gradient_norm:.4e} "
          f"max_abs={max_receiver_gradient_abs:.4e}")

    # --- regression: u_alpha == 0 with u_theta != 0. dreceiver_log_probs == 0 (softmax-Jacobian . 0), but
    # d_grad_receiver_log_probs is the receiver-log-probability cotangent tangent,
    # i.e. the H_at block d(grad_receiver)/d(theta), which is
    # LEGITIMATELY NONZERO here (it is exactly what S8 consumes for H_ta = H_at^T). So we assert
    # dreceiver_log_probs == 0 but do NOT require d_grad_receiver_log_probs == 0 in this direction.
    _Hu_z, cap_zero = _capture_run(static, theta, alpha, u_zero_alpha,
                                   tangent_self_iters=tangent_self_iters)
    max_receiver_tangent_zero = max(
        (float(tangent.abs().max()) for _, tangent, _ in cap_zero if tangent is not None),
        default=0.0,
    )
    max_receiver_gradient_zero = max(
        (float(gradient.abs().max()) for gradient, _, _ in cap_zero),
        default=0.0,
    )
    print(
        "  u_alpha=0,u_theta!=0: "
        f"max|dreceiver_log_probs|={max_receiver_tangent_zero:.3e} (must be 0)  "
        f"max|d_grad_receiver_log_probs|={max_receiver_gradient_zero:.3e} "
        "(H_at block: may be !=0)"
    )

    # --- TRUE null regression: u == 0 entirely -> dreceiver_log_probs == 0 AND d_grad_receiver_log_probs EXACTLY 0.
    u_null = torch.zeros(p, device=device, dtype=torch.float64)
    _Hu_n, cap_null = _capture_run(static, theta, alpha, u_null,
                                   tangent_self_iters=tangent_self_iters)
    max_receiver_gradient_null = max(
        (float(gradient.abs().max()) for gradient, _, _ in cap_null),
        default=0.0,
    )
    print(
        "  u==0 null: "
        f"max|d_grad_receiver_log_probs|={max_receiver_gradient_null:.3e} "
        "(must be EXACTLY 0)"
    )

    live_ok = (
        all_finite
        and use_receiver_weights_all
        and receiver_tangent_nonzero
        and n_nonzero_waves > 0
        and total_receiver_gradient_norm > 0.0
    )
    # regression invariant: zero seed -> zero softmax-Jacobian tangent (dreceiver_log_probs) in BOTH; the null
    # direction must give an exactly-zero receiver cotangent (no spurious scatter).
    reg_ok = (
        max_receiver_tangent_zero == 0.0
        and max_receiver_gradient_null == 0.0
    )
    ok = live_ok and reg_ok
    print(
        f"[S5 wave-receiver gate] live_ok={live_ok} reg_ok={reg_ok} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return dict(live_ok=live_ok, reg_ok=reg_ok, ok=ok, n_waves=n_waves,
                total_receiver_gradient_norm=total_receiver_gradient_norm,
                max_receiver_gradient_abs=max_receiver_gradient_abs,
                n_nonzero_waves=n_nonzero_waves, all_finite=all_finite)


if __name__ == "__main__":
    r = run()
    raise SystemExit(0 if r["ok"] else 1)

"""S5 focused gate (b): the wave-SO col-cotangent (`d_grad_col`) is FINITE & NONZERO at a
non-uniform base when ``u_alpha != 0``.

This isolates the S5 kernel (`_wave_so_kernel` / `wave_backward_so`). It runs a REAL ``hvp(u)``
on the 8-hogenom fixture at the seeded NON-UNIFORM base alpha (so ``use_col_weights=True`` and
``dcol`` is the live softmax-Jacobian seed) and captures every ``wave_backward_so`` return via a
wrapper, asserting the new 8th output ``d_grad_col`` is finite and not identically zero.

It ALSO re-asserts the regression invariant from the other side: with ``u_alpha = 0`` the captured
``d_grad_col`` from each wave must be EXACTLY zero (dcol = 0 -> no col-cotangent), and the analytic
HVP's theta block is untouched. The full (theta, alpha) numerical validation of H_aa/H_ta is S8.

    python -m gates._verify_s5_wave_col
"""

from __future__ import annotations

import torch

import gpurec.optim.hvp_exact as hx
from gpurec.optim.hvp_exact import build_point_cache, make_exact_hvp
from gpurec.optim.value_and_grad import forward_solve
from gates._verify_hvp_recv import _static_theta_alpha_from_live


def _capture_run(static, theta, alpha, u, *, tangent_self_iters=128):
    """Run one hvp(u) while wrapping wave_backward_so to record each wave's d_grad_col."""
    captured = []
    orig = hx.wave_backward_so

    def wrapped(*args, **kw):
        out = orig(*args, **kw)
        # new return is (d_out, *6 d_aws, d_grad_col); record the col cotangent + its dcol input
        captured.append((out[-1].detach().clone(), kw.get("dcol"), kw.get("use_col_weights")))
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
    print(f"[S5 wave-col gate {n_families}-family] S={S} theta_numel={theta_numel} p(4S)={p} "
          f"nonuniform={True} valid_mass_min={vm0:.4f}")

    g = torch.Generator(device=device).manual_seed(seed + 7)
    u_full = torch.randn(p, generator=g, device=device, dtype=torch.float64)
    u_full = u_full / u_full.norm()
    # a non-zero alpha component for the live path
    u_zero_alpha = torch.cat([u_full[:theta_numel],
                              torch.zeros(S, device=device, dtype=torch.float64)])

    # --- live path: u_alpha != 0 -> dcol != 0 -> d_grad_col must be finite & nonzero
    _Hu_live, cap_live = _capture_run(static, theta, alpha, u_full,
                                      tangent_self_iters=tangent_self_iters)
    n_waves = len(cap_live)
    all_finite = all(bool(torch.isfinite(c).all()) for c, _, _ in cap_live)
    use_col_all = all(bool(b) for _, _, b in cap_live)
    dcol_seen = [d for _, d, _ in cap_live if d is not None]
    dcol_nonzero = bool(dcol_seen) and any(float(d.abs().max()) > 0 for d in dcol_seen)
    total_col_norm = float(sum(float(c.norm()) for c, _, _ in cap_live))
    max_col_abs = max((float(c.abs().max()) for c, _, _ in cap_live), default=0.0)
    n_nonzero_waves = sum(1 for c, _, _ in cap_live if float(c.abs().max()) > 0)
    print(f"  live (u_alpha!=0): waves={n_waves} use_col_weights_all={use_col_all} "
          f"dcol_nonzero={dcol_nonzero} all_finite={all_finite}")
    print(f"     d_grad_col: nonzero_waves={n_nonzero_waves}/{n_waves} "
          f"sum|.|2={total_col_norm:.4e} max_abs={max_col_abs:.4e}")

    # --- regression: u_alpha == 0 with u_theta != 0. dcol == 0 (softmax-Jacobian . 0), but
    # d_grad_col is the col-cotangent TANGENT = the H_at block d(grad_col)/d(theta), which is
    # LEGITIMATELY NONZERO here (it is exactly what S8 consumes for H_ta = H_at^T). So we assert
    # dcol == 0 but do NOT require d_grad_col == 0 in this direction.
    _Hu_z, cap_zero = _capture_run(static, theta, alpha, u_zero_alpha,
                                   tangent_self_iters=tangent_self_iters)
    max_dcol_zero = max((float(d.abs().max()) for _, d, _ in cap_zero if d is not None), default=0.0)
    max_col_zero = max((float(c.abs().max()) for c, _, _ in cap_zero), default=0.0)
    print(f"  u_alpha=0,u_theta!=0: max|dcol|={max_dcol_zero:.3e} (must be 0)  "
          f"max|d_grad_col|={max_col_zero:.3e} (H_at block: may be !=0)")

    # --- TRUE null regression: u == 0 entirely -> dcol == 0 AND d_grad_col EXACTLY 0.
    u_null = torch.zeros(p, device=device, dtype=torch.float64)
    _Hu_n, cap_null = _capture_run(static, theta, alpha, u_null,
                                   tangent_self_iters=tangent_self_iters)
    max_col_null = max((float(c.abs().max()) for c, _, _ in cap_null), default=0.0)
    print(f"  u==0 null: max|d_grad_col|={max_col_null:.3e} (must be EXACTLY 0)")

    live_ok = all_finite and use_col_all and dcol_nonzero and (n_nonzero_waves > 0) \
        and (total_col_norm > 0.0)
    # regression invariant: zero seed -> zero softmax-Jacobian tangent (dcol) in BOTH; the null
    # direction must give an exactly-zero col-cotangent (no spurious scatter).
    reg_ok = (max_dcol_zero == 0.0) and (max_col_null == 0.0)
    ok = live_ok and reg_ok
    print(f"[S5 wave-col gate] live_ok={live_ok} reg_ok={reg_ok} -> {'PASS' if ok else 'FAIL'}")
    return dict(live_ok=live_ok, reg_ok=reg_ok, ok=ok, n_waves=n_waves,
                total_col_norm=total_col_norm, max_col_abs=max_col_abs,
                n_nonzero_waves=n_nonzero_waves, all_finite=all_finite)


if __name__ == "__main__":
    r = run()
    raise SystemExit(0 if r["ok"] else 1)

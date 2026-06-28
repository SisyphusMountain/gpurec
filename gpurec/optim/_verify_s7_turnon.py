"""S7 turn-on gate: the BACKWARD/cache is FINITE + grad_col is CORRECT at a NON-UNIFORM base.

S7 derives ``use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)`` and
threads it through ``vjp_root_to_theta`` (ggn.py, killing the False hardcode at :58),
``build_point_cache`` (hvp_exact.py), and the HVP-loop kernel calls + uniform_fast hardcode in
``make_exact_hvp``. Before S7 the entire HVP machinery was hardcoded to the UNIFORM path, so at a
non-uniform base the E-adjoint blew up to ~1e18 (use_receiver_weights=False in ggn.py:58).

This gate (per the plan, the S7 FD gate) checks ONLY the backward/cache + grad_col -- it does NOT
run the HVP tangent sweep (that is S3: the tangent forward is still uniform and would NaN here).

  1. build_point_cache runs with the E-adjoint wE FINITE (~O(1), NOT 1e18) and grad_theta ~O(1).
  2. the cached-backward grad_col (= dNLL/dalpha from vjp_root_to_theta) equals the PRODUCTION
     stream_batches grad_receiver (make_value_and_grad(optimize_receiver=True)), P-projected.
  3. grad_col equals the central FD of the loss w.r.t. alpha, P-projected.  PASS: rel <= 5e-4.

Reuses the _verify_hvp_recv / _verify_recv_grad fixture (8 hogenom families, S=1331),
seeded NON-UNIFORM base alpha = 0.2*randn(S), fp64, converged solver (pi>=128, neu>=64).

    python -m gpurec.optim._verify_s7_turnon
"""

from __future__ import annotations

import torch

from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.optim.hvp_exact import build_point_cache
from gpurec.optim.value_and_grad import forward_solve, make_value_and_grad
from gpurec.optim._verify_hvp_recv import (
    _static_theta_alpha_from_live, _valid_mass_min, proj_alpha,
)


def run(n_families=8, device="cuda", seed=0, eps=1e-6):
    static, theta, alpha, S, vm0 = _static_theta_alpha_from_live(n_families, device, seed=seed)
    theta_numel = 3 * S
    nonuniform = not receiver_weights_are_uniform(alpha)
    assert nonuniform, "base alpha must be non-uniform (alpha paths must be LIVE)"
    print(f"[S7 turn-on gate {n_families}-family] S={S} theta_numel={theta_numel} "
          f"fp64 converged(pi=128,neu=64) nonuniform={nonuniform} valid_mass_min={vm0:.4f}")

    # ---- (1) build_point_cache at the NON-UNIFORM base: must be FINITE -------------------------
    _loss, sv = forward_solve([static], theta, alpha)
    grad_theta, grad_col, cache = build_point_cache([static], theta, alpha, sv)
    wE = cache["e_side"]["wE"]
    wE_max = float(wE.abs().max())
    gt_norm = float(grad_theta.norm())
    gc_norm = float(grad_col.norm())
    finite = bool(torch.isfinite(wE).all() and torch.isfinite(grad_theta).all()
                  and torch.isfinite(grad_col).all())
    print(f"  build_point_cache: finite={finite}  |wE|_max={wE_max:.3e}  "
          f"|grad_theta|={gt_norm:.3e}  |grad_col|={gc_norm:.3e}")
    cache_ok = finite and wE_max < 1e6 and gt_norm < 1e6  # NOT the 1e18 uniform-path blowup

    # ---- gauge: grad_col (dNLL/dalpha) lives on the gauge slice (1^T g_alpha == 0) -------------
    one_dot = float(grad_col.sum())
    gauge_rel = abs(one_dot) / max(gc_norm, 1e-30)
    print(f"  gauge: 1^T grad_col = {one_dot:.3e}  |.|/|grad_col| = {gauge_rel:.3e}")

    # ---- (2) production stream_batches grad_receiver (optimize_receiver=True) -------------------
    f = make_value_and_grad([static], alpha, theta_shape=(S, 3), optimize_receiver=True)
    z = torch.cat([theta.reshape(-1), alpha]).contiguous()
    _Lz, g_z, _, _ = f(z)
    g_recv = g_z[theta_numel:].double()
    # P-project both (the 1_S gauge null mode is uncontrolled; compare on the gauge slice)
    Pgc = proj_alpha(grad_col.double())
    Pgr = proj_alpha(g_recv)
    rel_prod = float((Pgc - Pgr).norm()) / max(float(Pgr.norm()), 1e-30)
    print(f"  grad_col vs production grad_receiver (P-proj): "
          f"|Pgc|={float(Pgc.norm()):.4e} |Pgr|={float(Pgr.norm()):.4e} rel={rel_prod:.3e}")

    # ---- (3) central FD of the loss w.r.t. alpha (P-projected directional derivative) ----------
    def loss_at(al):
        L, _ = forward_solve([static], theta, al)
        return float(L)

    # directional-derivative FD along seeded pure-alpha unit dirs, compared to <grad_col, dir>
    fd_rels = []
    for s in range(4):
        gen = torch.Generator(device=device).manual_seed(400 + s)
        ua = torch.randn(S, generator=gen, device=device, dtype=torch.float64)
        ua = proj_alpha(ua)            # restrict to the gauge slice (the meaningful directions)
        ua = ua / ua.norm()
        vmp = _valid_mass_min(static, alpha + eps * ua)
        vmm = _valid_mass_min(static, alpha - eps * ua)
        assert min(vmp, vmm) > 1e-3, f"valid_mass collapses along alpha dir {s}: {min(vmp, vmm)}"
        Lp = loss_at(alpha + eps * ua)
        Lm = loss_at(alpha - eps * ua)
        fd = (Lp - Lm) / (2 * eps)
        ana = float(torch.dot(grad_col.double(), ua))
        rel = abs(fd - ana) / max(1.0, abs(ana))
        fd_rels.append(rel)
        print(f"  FD alpha dir {s}: fd={fd:.6f} ana={ana:.6f} rel={rel:.3e}")
    max_fd_rel = max(fd_rels)

    grad_col_vs_fd_ok = max_fd_rel <= 5e-4
    prod_match_ok = rel_prod <= 5e-4
    gauge_ok = gauge_rel <= 1e-6
    ok = cache_ok and grad_col_vs_fd_ok and prod_match_ok and gauge_ok
    print(f"  GATE: cache_finite={cache_ok}  grad_col_vs_FD_rel={max_fd_rel:.3e} "
          f"[{'PASS' if grad_col_vs_fd_ok else 'FAIL'}]  "
          f"grad_col_vs_prod_rel={rel_prod:.3e} [{'PASS' if prod_match_ok else 'FAIL'}]  "
          f"gauge_rel={gauge_rel:.3e} [{'PASS' if gauge_ok else 'FAIL'}]")
    print(f"[S7 turn-on gate] {'ALL PASS' if ok else 'FAIL'}")
    return dict(ok=ok, cache_ok=cache_ok, wE_max=wE_max, gt_norm=gt_norm, gc_norm=gc_norm,
                grad_col_vs_fd_rel=max_fd_rel, grad_col_vs_prod_rel=rel_prod, gauge_rel=gauge_rel)


if __name__ == "__main__":
    r = run()
    raise SystemExit(0 if r["ok"] else 1)

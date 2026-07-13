"""S3 gate: the WEIGHTED forward tangent (jvp of the root scores) at a NON-UNIFORM base alpha.

S3 (``docs/optim/receiver_weights_hvp_plan.md``) replaces the uniform forward tangent with the
JVP of ``extract_parameters_weighted_receivers`` so the parameter tangent ``dMC`` carries the
``alpha -> receiver_log_probs -> receiver_valid_log_normalizer (receiver_norm) -> max_transfer``
coupling, and the softmax-Jacobian seed ``dcol = dreceiver_log_probs`` is threaded IDENTICALLY into
the E-step tangent fixed point AND the wave-step tangent (use_col_weights=True). This is the
prerequisite that makes a pure-theta tangent FINITE at a non-uniform base (the legacy uniform
tangent NaNs there) and adds the real alpha->rate forward sensitivity.

Two gates, both at the seeded NON-UNIFORM base alpha = 0.2*randn(S) (so the alpha paths are LIVE),
fp64, converged solver (pi>=128, neumann>=64, tangent_self>=128):

  (A) SEED unit-check (no solve): ``torch.func.jvp`` on
      ``extract_parameters_weighted_receivers`` must give
        * ``dcol == (I - 1 w^T) u_alpha / ln2``  (the log_softmax Jacobian in log2-space; w =
          softmax(alpha) -- NOT diag(w)-ww^T, which is d(softmax) not d(log_softmax)), and
        * ``dmax_transfer`` NONZERO (the receiver_norm coupling the uniform extractor dropped).

  (B) FORWARD-TANGENT FD: ``jvp_root_scores(..., alpha=alpha, u_alpha=u_alpha)`` vs the central FD
      of ``forward_solve``'s root Pi rows (``pi_wave[root_clade_ids]``) at ``z +/- eps*u``, for
        * ``u_alpha != 0`` (the new weighted path), and
        * ``u_alpha = 0``  (a pure-theta direction must be FINITE + match FD at the non-uniform
          base -- the regression guard for the NaN the harness pass found).
      Pass: rel <= 5e-4.  u_alpha is mean-subtracted (on the gauge slice; the 1_S null mode gives an
      exactly-zero tangent -> a 0/0 ratio, so it is tested separately as the gauge certificate).

Reuses the _verify_hvp_recv / _verify_recv_grad fixture (8 hogenom families, S=1331).

    python -m gates._verify_s3_fwd_tangent
"""

from __future__ import annotations

import math

import torch

from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.core.parameters.extract_parameters import (
    extract_parameters_weighted_receivers, receiver_log_probs_from_weights,
)
from gpurec.solver.forward_tangent import jvp_root_scores
from gpurec.solver.value_and_grad import forward_solve
from gates._verify_hvp_recv import (
    _static_theta_alpha_from_live, _valid_mass_min, proj_alpha,
)

_LN2 = 0.6931471805599453


def _logsoftmax_jac_dcol(alpha: torch.Tensor, u_alpha: torch.Tensor) -> torch.Tensor:
    """d(receiver_log_probs) . u_alpha = (I - 1 w^T) u_alpha / ln2 in log2-space.

    receiver_log_probs = log_softmax(alpha)/ln2, so its Jacobian is (I - 1 w^T)/ln2 where
    w = softmax(alpha) (NOT the softmax Jacobian diag(w)-ww^T, which is d(softmax)). Concretely
    dcol_i = (u_alpha_i - <w, u_alpha>) / ln2. Computed by hand here as the gate's oracle; the
    production path gets it from autograd, never hand-rolled."""
    w = torch.exp2(receiver_log_probs_from_weights(alpha))            # softmax(alpha)
    return (u_alpha - float(torch.dot(w, u_alpha))) / _LN2


def run(n_families=8, device="cuda", seed=0, tangent_self_iters=128, eps=1e-6, n_each=3):
    static, theta, alpha, S, vm0 = _static_theta_alpha_from_live(n_families, device, seed=seed)
    theta_numel = 3 * S
    nonuniform = not receiver_weights_are_uniform(alpha)
    assert nonuniform, "base alpha must be non-uniform (alpha paths must be LIVE)"
    sh = static.species_helpers
    print(f"[S3 fwd-tangent gate {n_families}-family] S={S} theta_numel={theta_numel} fp64 "
          f"converged(pi=128,neu=64,tself={tangent_self_iters}) nonuniform={nonuniform} "
          f"valid_mass_min={vm0:.4f}")

    # ============================ (A) SEED unit-check (no solve) ===============================
    gen = torch.Generator(device=device).manual_seed(seed + 11)
    u_theta_seed = torch.randn(S, 3, generator=gen, device=device, dtype=torch.float64)
    u_alpha_seed = proj_alpha(torch.randn(S, generator=gen, device=device, dtype=torch.float64))

    def f_extract(th, al):
        return extract_parameters_weighted_receivers(
            th, al, sh, specieswise=static.specieswise, genewise=static.genewise,
            uniform_fast=True,
        )

    _, tang = torch.func.jvp(f_extract, (theta, alpha), (u_theta_seed, u_alpha_seed))
    dpS_s, dpD_s, dpL_s, dmt_s, dcol_s = tang
    dcol_oracle = _logsoftmax_jac_dcol(alpha, u_alpha_seed)
    seed_rel = float((dcol_s - dcol_oracle).norm()) / max(float(dcol_oracle.norm()), 1e-30)
    dmt_norm = float(dmt_s.norm())
    # u_alpha=0 -> dmax_transfer must be the pure-theta dlog_pT (receiver_norm theta-independent);
    # u_theta=0 -> dmax_transfer must be the pure receiver_norm coupling (NONZERO, the new term).
    _, tang_a = torch.func.jvp(f_extract, (theta, alpha),
                               (torch.zeros_like(u_theta_seed), u_alpha_seed))
    dmt_alpha_only = float(tang_a[3].norm())
    seed_ok = seed_rel <= 1e-10 and dmt_alpha_only > 1e-6
    print(f"  (A) seed: dcol vs (I-1w^T)u/ln2 rel={seed_rel:.3e}  "
          f"|dmax_transfer|(mixed)={dmt_norm:.3e}  |dmax_transfer|(alpha-only)={dmt_alpha_only:.3e} "
          f"[{'PASS' if seed_ok else 'FAIL'}]")

    # ============================ (B) forward-tangent FD ======================================
    _loss0, sv = forward_solve([static], theta, alpha)
    root_ids = static.wave_layout["root_clade_ids"]

    def root_rows_at(th, al):
        _l, s = forward_solve([static], th, al)
        return s["pi_wave"].index_select(0, root_ids).clone()

    def fd_tangent(u_theta, u_alpha):
        """Central FD of the root Pi rows along (u_theta, u_alpha)."""
        rp = root_rows_at(theta + eps * u_theta, alpha + eps * u_alpha)
        rm = root_rows_at(theta - eps * u_theta, alpha - eps * u_alpha)
        return (rp - rm) / (2 * eps)

    def assert_valid(u_alpha):
        na = float(u_alpha.norm())
        if na == 0.0:
            return
        ua = u_alpha / na
        for s in (+1.0, -1.0):
            vm = _valid_mass_min(static, alpha + s * eps * ua)
            assert vm > 1e-3, f"valid_mass collapses along alpha dir: {vm}"

    def one_dir(u_theta, u_alpha, tag):
        assert_valid(u_alpha)
        t_ana = jvp_root_scores(
            static, theta, u_theta, sv, alpha=alpha, u_alpha=u_alpha,
            self_iters=tangent_self_iters, e_tol=1e-12,
        )
        t_fd = fd_tangent(u_theta, u_alpha)
        finite = bool(torch.isfinite(t_ana).all())
        num = float((t_ana - t_fd).norm())
        den = max(float(t_fd.norm()), 1e-30)
        rel = num / den if finite else float("inf")
        print(f"    {tag}: finite={finite} |t_ana|={float(t_ana.norm()):.4f} "
              f"|t_fd|={float(t_fd.norm()):.4f} rel={rel:.3e}")
        return rel, finite

    # (B1) u_alpha != 0 (mixed): the NEW weighted path
    print("  (B) forward tangent vs central FD of root Pi rows:")
    mixed_rels = []
    for k in range(n_each):
        g = torch.Generator(device=device).manual_seed(seed + 300 + k)
        ut = torch.randn(S, 3, generator=g, device=device, dtype=torch.float64)
        ua = proj_alpha(torch.randn(S, generator=g, device=device, dtype=torch.float64))
        # unit-normalize the joint direction so eps*u is a consistent step
        nrm = math.sqrt(float(ut.pow(2).sum() + ua.pow(2).sum()))
        ut, ua = ut / nrm, ua / nrm
        rel, fin = one_dir(ut, ua, f"mixed (u_alpha!=0) dir {k}")
        mixed_rels.append(rel if fin else float("inf"))

    # (B2) u_alpha = 0 (pure theta): regression guard -- MUST be finite at the non-uniform base
    theta_rels = []
    for k in range(n_each):
        g = torch.Generator(device=device).manual_seed(seed + 400 + k)
        ut = torch.randn(S, 3, generator=g, device=device, dtype=torch.float64)
        ut = ut / ut.norm()
        ua = torch.zeros(S, device=device, dtype=torch.float64)
        rel, fin = one_dir(ut, ua, f"theta  (u_alpha=0 ) dir {k}")
        theta_rels.append(rel if fin else float("inf"))

    # (B3) pure-alpha (u_theta=0): isolates the alpha->rate forward coupling
    alpha_rels = []
    for k in range(n_each):
        g = torch.Generator(device=device).manual_seed(seed + 500 + k)
        ua = proj_alpha(torch.randn(S, generator=g, device=device, dtype=torch.float64))
        ua = ua / ua.norm()
        ut = torch.zeros(S, 3, device=device, dtype=torch.float64)
        rel, fin = one_dir(ut, ua, f"alpha  (u_theta=0 ) dir {k}")
        alpha_rels.append(rel if fin else float("inf"))

    # (B4) gauge certificate: the 1_S null direction gives an exactly-zero tangent (NLL-invariant)
    ones = torch.ones(S, device=device, dtype=torch.float64) / math.sqrt(S)
    t_gauge = jvp_root_scores(static, theta, torch.zeros(S, 3, device=device, dtype=torch.float64),
                              sv, alpha=alpha, u_alpha=ones, self_iters=tangent_self_iters,
                              e_tol=1e-12)
    gauge_leak = float(t_gauge.abs().max())
    print(f"  (B4) gauge: |jvp along 1_S/sqrt(S)|_max = {gauge_leak:.3e} "
          f"(must be ~truncation floor; the alpha 1_S mode is NLL-invariant)")

    mixed_max = max(mixed_rels)
    theta_max = max(theta_rels)
    alpha_max = max(alpha_rels)
    mixed_ok = mixed_max <= 5e-4
    theta_ok = theta_max <= 5e-4
    alpha_ok = alpha_max <= 5e-4
    gauge_ok = gauge_leak <= 5e-4
    print(f"  BLOCK rel:  theta(u_a=0)={theta_max:.3e} [{'PASS' if theta_ok else 'FAIL'}]  "
          f"alpha(u_t=0)={alpha_max:.3e} [{'PASS' if alpha_ok else 'FAIL'}]  "
          f"mixed={mixed_max:.3e} [{'PASS' if mixed_ok else 'FAIL'}]")
    ok = seed_ok and mixed_ok and theta_ok and alpha_ok and gauge_ok
    print(f"[S3 fwd-tangent gate] {'ALL PASS' if ok else 'FAIL'}  "
          f"(seed={seed_ok} theta={theta_ok} alpha={alpha_ok} mixed={mixed_ok} gauge={gauge_ok})")
    return dict(ok=ok, seed_rel=seed_rel, dmt_alpha_only=dmt_alpha_only,
                theta=theta_max, alpha=alpha_max, mixed=mixed_max, gauge_leak=gauge_leak,
                seed_ok=seed_ok, theta_ok=theta_ok, alpha_ok=alpha_ok, mixed_ok=mixed_ok,
                gauge_ok=gauge_ok)


if __name__ == "__main__":
    r = run()
    raise SystemExit(0 if r["ok"] else 1)

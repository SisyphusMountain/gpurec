"""Canonical specieswise DTL fit: a SINGLE MAP fit at a GIVEN prior precision ``lam``.

Specieswise raw MLE over ``theta[S,3]`` is ill-posed -- non-identifiable, with ~half the per-state
rates saturating the 0/1 boundary (flat near-zero Hessian eigenvalues) and index-1 saddles
(``lam_min < 0``). See docs/superpowers/specs/2026-07-11-specieswise-recipe-organization-design.md and
kernel-bench/newton/_specieswise_basin_findings.md. The well-posed target is a MAP fit with a
Gaussian/ridge prior ``(lam/2)||theta - theta_ref||^2`` whose precision ``lam`` is chosen by
cross-validation (:func:`gpurec.fit.map_cv.map_cv`).

This module is the per-fit TOOL: it fits ONE prior. It does NOT choose ``lam`` and does NOT
cross-validate -- ``lam`` is a required argument, the explicit "you chose this" signal; ``lam=0.0`` is
the raw MLE and is intentionally never a default. Recipe: ~10 Adam warm-up steps (basin entry) on the
MAP objective, then the saddle-aware ``newton_lanczos`` (CG negative-curvature witness) with the ridge
term. At ``lam > |lam_min|`` the MAP Hessian ``H + lam*I`` is PD, restoring a quadratic endgame.
"""
from __future__ import annotations

import time

import torch

from gpurec.fit.newton_cg import newton_lanczos
from gpurec.fit.optimize import Schedule, final_eval
from gpurec.solver.value_and_grad import make_value_and_grad

_LN2 = 0.6931471805599453


def fit_specieswise(batch_statics, theta0, receiver_weights, *, lam, theta_ref=None,
                    adam_steps=10, adam_lr=1.0, max_newton=8, gtol=1e-2, lanczos_m=10,
                    sigma=0.01, verbose=False) -> dict:
    """Single MAP fit of specieswise theta[S,3] at prior precision ``lam``. See module docstring."""
    if lam is None:
        raise ValueError(
            "fit_specieswise requires an explicit prior precision `lam` -- the specieswise raw MLE "
            "is ill-posed. Choose lam by cross-validation (gpurec.fit.map_cv.map_cv) and pass it "
            "here; lam=0.0 is the raw MLE and is intentionally not a default."
        )
    theta_shape = tuple(theta0.shape)
    theta = theta0.detach().reshape(theta_shape).float().contiguous().clone()
    if theta_ref is None:
        theta_ref = theta.clone()
    theta_ref = theta_ref.detach().reshape(theta_shape).float().to(theta.device)
    t0 = time.perf_counter()

    # 1. Adam warm-up on the MAP objective (prior-enabled value-and-grad), mirroring map_cv.fit_map.
    f = make_value_and_grad(batch_statics, receiver_weights, theta_shape=theta_shape,
                            prior=(float(lam), theta_ref))
    if adam_steps > 0:
        leaf = theta.clone().requires_grad_(True)
        opt = torch.optim.Adam([leaf], lr=adam_lr)
        sched = Schedule("adaptive", adam_lr, t_max=adam_steps)
        warm = None
        for _ in range(int(adam_steps)):
            loss, g, _sv, warm = f(leaf.detach().reshape(-1), warm_E=warm)
            opt.param_groups[0]["lr"] = sched.update(loss, g)
            leaf.grad = g.reshape(theta_shape)
            opt.step()
        theta = leaf.detach().reshape(theta_shape).contiguous()

    # 2. saddle-aware Newton with the ridge/MAP term (exact HVP for specieswise theta[S,3]).
    theta_hat, hist = newton_lanczos(
        batch_statics, theta, receiver_weights, hvp_mode="exact", lam=float(lam),
        theta_ref=theta_ref, lanczos_m=lanczos_m, sigma=sigma, max_newton=max_newton,
        gtol=gtol, verbose=verbose)
    gnorm = float(hist[-1]["gnorm"]) if hist else float("nan")  # MAP projected-gradient norm

    # data NLL (excludes the ridge) via the fair fp64 eval -> comparable across modes and lam.
    nll_bits, _g = final_eval(batch_statics, theta_hat, receiver_weights)
    nll_bits = float(nll_bits)
    wall_s = time.perf_counter() - t0
    return {"mode": "specieswise", "theta": theta_hat.detach().cpu(),
            "rates": (2.0 ** theta_hat.detach().float().cpu()),
            "nll_bits": nll_bits, "nll_nats": nll_bits * _LN2, "gnorm": gnorm,
            "lam": float(lam), "wall_s": wall_s}

"""Global (shared-rate) DTL fitting via the genewise recipe with per-family curvature ACCUMULATED.

Global mode has one shared rate vector ``theta = [log2 D, log2 L, log2 T]`` (shape ``[3]``) for all
families. The objective is ``sum_f NLL_f(theta)`` with ``theta`` SHARED, so the gradient is
``sum_f grad_f`` and the Hessian is ``sum_f H_f``. Global therefore runs the SAME recipe as
``fit_genewise`` -- driving the genewise per-family forward + batched 3x3 analytic-Hessian
machinery -- but ACCUMULATES the per-family gradients/Hessians into a single shared 3x3 block:

  1. Build a genewise-mode model at the cheap ``fit_pi`` tier (like fit_genewise's forward).
  2. Adam warm-up (clipped, box-projected) on the aggregate gradient.
  3. Box-constrained trust-region Newton on the aggregate 3x3 Hessian (the SUM of the per-family
     3x3 blocks that ``genewise_fit._analytic_hessian`` builds from three analytic
     Hessian-vector products), eigenvalue-floored to ``mu`` -> PD, with a loss-plateau stop.

Step 3 used to build that 3x3 by finite differences of the aggregate gradient with a step of 1e-2.
Measured against the analytic sum on the toy fixtures in float64, the finite-difference matrix was
off by 3.2e-3 relative at that step and by 3.2e-5 at a step of 1e-4 -- the error shrinking exactly
with the step, i.e. the ordinary forward-difference truncation error of the finite-difference
matrix itself, not a disagreement. In float32 shrinking the step made it worse (5.8e-3 at 1e-4) as
cancellation took over. The analytic sum is the curvature the finite difference was approximating,
and it costs three Hessian-vector products instead of three extra gradients, so it simply replaces
it and ``fd_eps`` is gone from the signature.

There is NO family rebatching: genewise drops each family once ITS 3 rates converge, but here every
family constrains the single shared ``theta`` and none can be dropped -- all G families are
accumulated on every step. The fit runs at ``fit_pi=16``; the final fair NLL is evaluated at
``eval_pi=64`` (mirroring genewise's certify) -- EXCEPT under the exact self-loop solves, where
the accurate tier would recompute an identical forward and is collapsed into the fit tier.
"""
from __future__ import annotations

import math
import time
from dataclasses import fields

import torch

from gpurec.api.model import GeneReconModel
from gpurec.api.solver_options import SolverOptions
from gpurec.config import GpurecConfig
from gpurec.config.rates import RateBounds
from gpurec.fit.genewise_fit import _analytic_hessian, _resolve_gene_trees
from gpurec.optimization import clamp_log_rate_, log2_rate_bounds, project_rate_gradient_

_LN2 = 0.6931471805599453
# Same rate box as GENEWISE_REFERENCE; [1e-6, 2.0], non-binding at the DTL optimum (rates ~1e-2).
_GLOBAL_RATE_BOUNDS = RateBounds.genewise()


def _tier_solver_options(base: dict, *, pi_iters: int, neumann_terms: int) -> SolverOptions:
    """This recipe's Pi/Neumann tier applied on top of the resolved base solver settings.

    ``base`` carries every ``SolverOptions`` field the caller resolved (defaults, then
    ``config.solver``, then an explicit ``solver_options``); only the two tier counts are
    this recipe's own choice, so only they are overridden here. The E-adjoint linear solve
    is a Neumann series -- it is the only implementation, so there is nothing to select.
    """
    return SolverOptions(**{**base, "pi_iters": pi_iters, "neumann_terms": neumann_terms})


def fit_global(species_tree, gene_trees, *, device="cuda", dtype: torch.dtype | str | None = None,
               adam_steps=5, adam_lr=1.0, grad_clip=10.0, tol=1e-3, max_iter=120,
               trust=2.0, mu=1e-2, hess_every=5, ftol=1e-6, patience=3,
               fit_pi=16, fit_neu=16, eval_pi=64, eval_neu=64, init_rate=None,
               solver_options=None, config: GpurecConfig | None = None,
               verbose=False) -> dict:
    """Fit the shared 3-vector theta via the accumulated genewise recipe. Returns
    ``{mode, theta[cpu,3], rates[cpu,3], nll_bits, nll_nats, gnorm, n_families, wall_s, n_steps}``.

    This recipe fixes its own forward tiers (``fit_pi``/``fit_neu`` for the fit,
    ``eval_pi``/``eval_neu`` for the final NLL). Every other solver field comes from
    ``solver_options`` (a ``SolverOptions`` or a dict of overrides) when given, else from
    ``config.solver``, else from the ``SolverOptions`` defaults -- so the self-loop kernel
    exact solver safeguards and tolerances reach the fit.
    """
    bounds = _GLOBAL_RATE_BOUNDS
    lo, hi = log2_rate_bounds(bounds=bounds)          # hi finite (2.0), so bound-active logic is well defined
    hi_eps = hi - bounds.bound_active_eps
    lo_eps = lo + bounds.bound_active_eps
    genes = _resolve_gene_trees(gene_trees)
    t0 = time.perf_counter()

    # SOLVER: start from the SolverOptions defaults, let config.solver replace them, and let an
    # explicit solver_options win over both -- the same precedence fit_genewise uses. `pi_iters`
    # and `neumann_terms` are this recipe's per-tier choice and are overridden below either way.
    base = {f.name: getattr(SolverOptions(), f.name) for f in fields(SolverOptions)}
    if config is not None:
        base.update({f.name: getattr(config.solver, f.name) for f in fields(SolverOptions)})
    if isinstance(solver_options, SolverOptions):
        base.update({f.name: getattr(solver_options, f.name) for f in fields(SolverOptions)})
    elif isinstance(solver_options, dict):
        base.update(solver_options)

    # Exact elimination makes the accurate tier identical to the fit tier.
    eval_pi = fit_pi
    eval_neu = fit_neu

    # genewise-mode model at the cheap fit tier: per-family loss+grad that we ACCUMULATE (sum over
    # families) into the shared 3x3. sum_f NLL_f(theta) with theta shared -> grad = sum_f grad_f.
    so_fit = _tier_solver_options(base, pi_iters=fit_pi, neumann_terms=fit_neu)
    model = GeneReconModel(species_tree, genes, mode="genewise", device=device, dtype=dtype,
                           config=config, solver_options=so_fit)
    dtype = model.dtype
    G = model.theta.shape[0]

    def lg(theta3):
        """(3,) shared theta -> (loss:float, g:(3,)) accumulated over all G families (no dropping)."""
        tG = theta3.detach().reshape(1, 3).expand(G, 3).contiguous()
        lv, g_fam, _gr = model.genewise_loss_vector_and_grad(theta=tG, need_grad=True)
        return float(lv.sum()), g_fam.sum(0).reshape(3).to(dtype)

    init = 0.1 if init_rate is None else float(init_rate)
    theta = torch.full((3,), math.log2(init), device=device, dtype=dtype)
    clamp_log_rate_(theta, bounds=bounds)

    if adam_steps > 0:  # Adam warm-up (basin entry) on the aggregate gradient
        lf = theta.clone().requires_grad_(True)
        ad = torch.optim.Adam([lf], lr=adam_lr)
        for _ in range(adam_steps):
            _, g = lg(lf.detach()); lf.grad = g.clone()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(lf, grad_clip)
            project_rate_gradient_(lf.detach(), lf.grad, bounds=bounds)
            ad.step()
            with torch.no_grad():
                clamp_log_rate_(lf, bounds=bounds)
        theta = lf.detach().clone()

    sub = theta.reshape(1, 3)   # [1,3] so the 3x3 TR-Newton ops match genewise's batched form
    Hd = None
    n_steps = 0
    prev_loss = float("inf")
    stall = 0
    last_pg = float("nan")
    mu_t = sub.new_tensor(mu)
    trust_t = sub.new_tensor(trust)
    for it in range(int(max_iter)):
        loss, g3 = lg(sub.reshape(3)); g = g3.reshape(1, 3)
        pg = project_rate_gradient_(sub, g.clone(), bounds=bounds).abs().amax()
        last_pg = float(pg)
        if verbose:
            print(f"[fit_global] it={it:3d} loss={loss:.6f} |Pg|={float(pg):.3e}", flush=True)
        # Loss-plateau stop: tol on the ABSOLUTE projected gradient is fp32-noise-floored (|Pg|
        # oscillates ~1e-3..1e-2 for an aggregate-over-families objective), so |Pg|<tol alone wastes
        # tens of steps oscillating after the loss is flat. Stop when the loss stops improving.
        stall = 0 if (prev_loss - loss) > ftol * max(1.0, abs(loss)) else stall + 1
        prev_loss = loss
        if float(pg) < tol or stall >= patience:
            break
        if it % hess_every == 0 or Hd is None:
            # theta is SHARED, so NLL(theta) = sum_f NLL_f(theta) and the second derivative is the
            # plain sum of the per-family 3x3 blocks -- families are independent, so there are no
            # cross-family terms. `_analytic_hessian` returns those blocks [G,3,3] from three
            # analytic Hessian-vector products; summing them gives the global 3x3 exactly.
            tG = sub.reshape(1, 3).expand(G, 3).contiguous()
            H = _analytic_hessian(model, tG, fit_pi, species_tree, genes).sum(0).reshape(1, 3, 3)
            e, V = torch.linalg.eigh(H)
            e = torch.maximum(e, mu_t)                            # Levenberg floor -> PD
            Hd = V @ torch.diag_embed(e) @ V.transpose(1, 2)
        # freeze coords pinned at a bound with the gradient pushing further out (box-active set)
        fixed = ((sub >= hi_eps) & (g < 0)) | ((sub <= lo_eps) & (g > 0))
        free = (~fixed).float()
        Hred = Hd * free.unsqueeze(1) * free.unsqueeze(2) + torch.diag_embed(1.0 - free)
        delta = -torch.linalg.solve(Hred, (g * free).unsqueeze(-1)).squeeze(-1)
        dn = delta.norm(dim=1, keepdim=True)
        sub = sub + delta * (trust_t / torch.maximum(dn, trust_t))   # trust-region cap
        clamp_log_rate_(sub, bounds=bounds)
        n_steps += 1

    theta_hat = sub.reshape(3)

    # Final fair NLL at the accurate eval tier (mirrors genewise's certify): total data NLL in bits.
    if (eval_pi, eval_neu) != (fit_pi, fit_neu):
        del model
        torch.cuda.empty_cache()
        so_eval = _tier_solver_options(base, pi_iters=eval_pi, neumann_terms=eval_neu)
        eval_model = GeneReconModel(species_tree, genes, mode="genewise", device=device, dtype=dtype,
                                    config=config, solver_options=so_eval)
    else:
        eval_model = model
    Gv = eval_model.theta.shape[0]
    tG = theta_hat.reshape(1, 3).expand(Gv, 3).contiguous()
    nll_bits = float(eval_model.genewise_loss_vector(theta=tG).sum())
    wall_s = time.perf_counter() - t0
    return {"mode": "global", "theta": theta_hat.detach().cpu(),
            "rates": (2.0 ** theta_hat.detach().float().cpu()),
            "nll_bits": nll_bits, "nll_nats": nll_bits * _LN2, "gnorm": last_pg,
            "n_families": len(genes), "wall_s": wall_s, "n_steps": n_steps}

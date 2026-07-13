"""Global (shared-rate) DTL fitting via the genewise recipe with per-family curvature ACCUMULATED.

Global mode has one shared rate vector ``theta = [log2 D, log2 L, log2 T]`` (shape ``[3]``) for all
families. The objective is ``sum_f NLL_f(theta)`` with ``theta`` SHARED, so the gradient is
``sum_f grad_f`` and the Hessian is ``sum_f H_f``. Global therefore runs the SAME recipe as
``fit_genewise`` -- driving the genewise per-family forward + batched 3x3 FD-Hessian machinery -- but
ACCUMULATES the per-family gradients/Hessians into a single shared 3x3 block:

  1. Build a genewise-mode model at the cheap ``fit_pi`` tier (like fit_genewise's forward).
  2. Adam warm-up (clipped, box-projected) on the aggregate gradient.
  3. Box-constrained trust-region Newton on the aggregate 3x3 FD Hessian (the SUM of the per-family
     3x3 FD Hessians), eigenvalue-floored to ``mu`` -> PD, with a loss-plateau stop.

There is NO family rebatching: genewise drops each family once ITS 3 rates converge, but here every
family constrains the single shared ``theta`` and none can be dropped -- all G families are
accumulated on every step. The fit runs at ``fit_pi=16`` (the previous global recipe ran the whole
fit at pi=64 on a full-batch global-mode forward -- ~10x more work); the final fair NLL is evaluated
at ``eval_pi=64`` (mirroring genewise's certify). Same optimum as the old recipe, ~10x faster.
"""
from __future__ import annotations

import math
import time

import torch

from gpurec.api.model import GeneReconModel
from gpurec.api.solver_options import SolverOptions
from gpurec.config.rates import RateBounds
from gpurec.fit.genewise_fit import _resolve_gene_trees
from gpurec.optimization import clamp_log_rate_, log2_rate_bounds, project_rate_gradient_

_LN2 = 0.6931471805599453
# Same rate box as GENEWISE_REFERENCE; [1e-6, 2.0], non-binding at the DTL optimum (rates ~1e-2).
_GLOBAL_RATE_BOUNDS = RateBounds.genewise()


def _tier_solver_options(solver_options, *, pi_iters: int, neumann_terms: int) -> SolverOptions:
    """Keep representation selection while applying this recipe's fixed tiers."""
    if isinstance(solver_options, dict):
        pi_representation = solver_options.get("pi_representation", "absolute")
    else:
        pi_representation = getattr(solver_options, "pi_representation", "absolute")
    return SolverOptions(
        pi_iters=pi_iters,
        pi_representation=pi_representation,
        neumann_terms=neumann_terms,
        e_adjoint_solver="neumann",
    )


def fit_global(species_tree, gene_trees, *, device="cuda", dtype=torch.float32,
               adam_steps=5, adam_lr=1.0, grad_clip=10.0, tol=1e-3, max_iter=120,
               trust=2.0, fd_eps=1e-2, mu=1e-2, hess_every=5, ftol=1e-6, patience=3,
               fit_pi=16, fit_neu=16, eval_pi=64, eval_neu=64, init_rate=None,
               solver_options=None, verbose=False) -> dict:
    """Fit the shared 3-vector theta via the accumulated genewise recipe. Returns
    ``{mode, theta[cpu,3], rates[cpu,3], nll_bits, nll_nats, gnorm, n_families, wall_s, n_steps}``.

    This recipe fixes its own forward tiers (``fit_pi``/``fit_neu`` for the fit,
    ``eval_pi``/``eval_neu`` for the final NLL), always with the Neumann
    E-adjoint. It preserves ``solver_options.pi_representation`` across those
    internally constructed tiers.
    """
    bounds = _GLOBAL_RATE_BOUNDS
    lo, hi = log2_rate_bounds(bounds=bounds)          # hi finite (2.0), so bound-active logic is well defined
    hi_eps = hi - bounds.bound_active_eps
    lo_eps = lo + bounds.bound_active_eps
    genes = _resolve_gene_trees(gene_trees)
    t0 = time.perf_counter()

    # genewise-mode model at the cheap fit tier: per-family loss+grad that we ACCUMULATE (sum over
    # families) into the shared 3x3. sum_f NLL_f(theta) with theta shared -> grad = sum_f grad_f.
    so_fit = _tier_solver_options(
        solver_options, pi_iters=fit_pi, neumann_terms=fit_neu
    )
    model = GeneReconModel(species_tree, genes, mode="genewise", device=device, dtype=dtype,
                           solver_options=so_fit)
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
            H = torch.zeros(1, 3, 3, device=device, dtype=dtype)
            for j in range(3):
                tp = sub.clone(); tp[:, j] += fd_eps
                _, gp = lg(tp.reshape(3))
                H[:, :, j] = (gp.reshape(1, 3) - g) / fd_eps     # forward difference, reuse base g
            H = 0.5 * (H + H.transpose(1, 2))
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
        so_eval = _tier_solver_options(
            solver_options, pi_iters=eval_pi, neumann_terms=eval_neu
        )
        eval_model = GeneReconModel(species_tree, genes, mode="genewise", device=device, dtype=dtype,
                                    solver_options=so_eval)
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

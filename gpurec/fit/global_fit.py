"""Global (shared-rate) DTL fitting via the genewise recipe specialized to a single block.

Global mode has one shared rate vector ``theta = [log2 D, log2 L, log2 T]`` (shape ``[3]``) for all
families, so it is a single 3-parameter box-bounded MLE -- exactly the sub-problem ``fit_genewise``
solves per family, but with the family gradients/curvature SUMMED into one aggregate 3x3 instead of
kept per-family. So global uses the same recipe as genewise:

  1. **Adam warm-up** -- a few clipped, box-projected steps for basin entry.
  2. **Box-constrained trust-region Newton** on the 3x3 forward-difference Hessian (3 gradient evals
     reusing the base gradient; eigenvalue-floored to ``mu`` -> PD), converging on the projected
     gradient ``|Pg| < tol``.

The per-family *rebatching* step of ``fit_genewise`` has no analog here (there is one block, nothing
to drop) and is unnecessary. This replaces the generic ``optimize`` path (300 Adam steps + a CG
Newton polish) for global, which is far more work than a 3-parameter problem needs.
"""
from __future__ import annotations

import time

import torch

from gpurec.api.model import GeneReconModel
from gpurec.api.solver_options import SolverOptions
from gpurec.config.rates import RateBounds
from gpurec.fit.genewise_fit import _resolve_gene_trees
from gpurec.fit.optimize import final_eval
from gpurec.optimization import clamp_log_rate_, log2_rate_bounds, project_rate_gradient_
from gpurec.solver.value_and_grad import make_value_and_grad

_LN2 = 0.6931471805599453
# Same recipe knobs as GENEWISE_REFERENCE (the single-block specialization drops the rebatch knobs).
_GLOBAL_RATE_BOUNDS = RateBounds.genewise()  # [1e-6, 2.0]; non-binding at the DTL optimum (rates ~1e-2)


def fit_global(species_tree, gene_trees, *, device="cuda", dtype=torch.float32,
               adam_steps=5, adam_lr=1.0, grad_clip=10.0, tol=1e-3, max_iter=120,
               trust=2.0, fd_eps=1e-2, mu=1e-2, hess_every=5, ftol=1e-6, patience=3,
               init_rate=None, solver_options=None, verbose=False) -> dict:
    """Fit the shared 3-vector theta. Returns
    ``{mode, theta[cpu,3], rates[cpu,3], nll_bits, nll_nats, gnorm, n_families, wall_s, n_steps}``."""
    bounds = _GLOBAL_RATE_BOUNDS
    lo, hi = log2_rate_bounds(bounds=bounds)          # hi finite (2.0), so bound-active logic matches genewise
    hi_eps = hi - bounds.bound_active_eps
    lo_eps = lo + bounds.bound_active_eps
    if solver_options is None:
        solver_options = SolverOptions(e_adjoint_solver="neumann")
    genes = _resolve_gene_trees(gene_trees)
    t0 = time.perf_counter()

    model = GeneReconModel(species_tree, genes, mode="global", device=device, dtype=dtype,
                           solver_options=solver_options)
    rw = model.receiver_weights.detach()
    vg = make_value_and_grad(model.batch_statics, rw, theta_shape=(3,))

    def lg(theta3):
        """(3,) -> (loss:float, g:(3,)) aggregate over all families."""
        loss, g, _saved, _w = vg(theta3.detach().reshape(-1), want_grad=True)
        return float(loss), g.reshape(3).to(dtype)

    theta = model.theta.detach().reshape(3).to(dtype).clone()
    if init_rate is not None:
        theta.fill_(float(torch.log2(torch.tensor(float(init_rate)))))
    clamp_log_rate_(theta, bounds=bounds)

    if adam_steps > 0:  # Adam warm-up (basin entry)
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
    mu_t = sub.new_tensor(mu)
    trust_t = sub.new_tensor(trust)
    for it in range(int(max_iter)):
        loss, g3 = lg(sub.reshape(3)); g = g3.reshape(1, 3)
        pg = project_rate_gradient_(sub, g.clone(), bounds=bounds).abs().amax()
        if verbose:
            print(f"[fit_global] it={it:3d} loss={loss:.6f} |Pg|={float(pg):.3e}", flush=True)
        # Loss-plateau stop: the absolute projected-gradient tol below is a fp32-noise-floored target
        # (|Pg| oscillates ~1e-3..1e-2 for an aggregate-over-families objective), so relying on it
        # alone wastes tens of steps oscillating after the loss is already flat. Stop when the loss
        # stops improving (relative to |loss|) for `patience` consecutive steps -- the objective, not
        # its noisy gradient, is the ground truth for "converged".
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
            e = torch.maximum(e, mu_t)                            # Levenberg floor -> PD (== clamp(min=mu))
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
    nll_bits, gnorm = final_eval(model.batch_statics, theta_hat, rw)   # fair fp64 eval (same unit as coupled path)
    nll_bits = float(nll_bits)
    wall_s = time.perf_counter() - t0
    return {"mode": "global", "theta": theta_hat.detach().cpu(),
            "rates": (2.0 ** theta_hat.detach().float().cpu()),
            "nll_bits": nll_bits, "nll_nats": nll_bits * _LN2, "gnorm": float(gnorm),
            "n_families": len(genes), "wall_s": wall_s, "n_steps": n_steps}

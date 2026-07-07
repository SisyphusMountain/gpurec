"""Default genewise (per-family) DTL-rate fitting recipe.

``fit_genewise`` is the standard way to fit independent per-family Duplication/Loss/Transfer rates to a
converged, box-bounded maximum-likelihood optimum -- the same model AleRax fits with ``--rec-model
UndatedDTL --model-parametrization PER-FAMILY``. The recipe (validated on the archaea / Hogenom genewise
benchmarks; see ``kernel-bench/experiments/alerax_speed/OPTIMIZATION_PLAN.md``) is:

  1. **Adam warm-up** -- a few large clipped steps (lr=1, grad-clip-norm 10) projected against the rate
     box, to enter the basin fast.
  2. **Box-constrained trust-region Newton** on the per-family 3x3 **forward-difference** Hessian
     (3 evals, reusing the base gradient; eigenvalue-clamped to ``mu`` -> PD), converging on the
     per-family projected gradient ``|Pg| < tol``.
  3. **Convergence-based rebatching** -- once a fraction of the active batch has converged (verified at
     the high pi/Neumann tier), those families are frozen and dropped and the model is rebuilt over the
     survivors, so the long tail of hard families runs on a small batch.
  4. **pi-tier escalation** -- the bulk runs at a cheap forward pi (16) with the adjoint **warm-start**
     (``GPUREC_WARM_ADJOINT``, memory-gated automatically -- see ``gpurec.core.memory_policy``); only the
     forward-stiff families escalate to the accurate pi (64).

Returns a dict with the fitted log2-rates ``theta`` ([F,3]), the AleRax-comparable ``rates`` (= 2**theta,
relative to speciation), the rebatch/defer history, and -- when ``certify=True`` -- a final cold PD
certificate (per-family |Pg|, smallest Hessian eigenvalue, interior-PD / bound-active counts, total NLL).
"""
from __future__ import annotations

import glob
import math
import os
import time

import torch

from gpurec.api.model import GeneReconModel
from gpurec.api.solver_options import SolverOptions
from gpurec.core.inference.solver import solve_forward_residual
from gpurec.optimization import clamp_log_rate_, log2_rate_bounds, project_rate_gradient_

# Proven base solver settings (pi_iters / neumann_terms are overridden per tier below).
_BASE_SOLVER = dict(
    e_max_iter=2000, e_tol=1e-8, self_loop_solver="neumann",
    bicgstab_max_iter=500, bicgstab_tol=1e-7, bicgstab_breakdown_tol=1e-30,
    adjoint_pruning_threshold=1e-6, use_adjoint_pruning=True, pibar_side_threshold=0.0,
)


def _resolve_gene_trees(spec) -> list[str]:
    """A list of paths, a glob ('dir/*.ale'), a directory (-> *.ale then *.newick), or a listfile."""
    if not isinstance(spec, str):
        return [str(p) for p in spec]
    if os.path.isdir(spec):
        paths = sorted(glob.glob(os.path.join(spec, "*.ale"))) or sorted(glob.glob(os.path.join(spec, "*.newick")))
    elif any(c in spec for c in "*?[") or os.path.splitext(spec)[1] in (".ale", ".newick", ".nwk"):
        paths = sorted(glob.glob(spec))
    else:  # a families listfile: AleRax `[FAMILIES]` (starting_gene_tree = ...) or one path per line
        lines = [ln.strip() for ln in open(spec) if ln.strip() and not ln.startswith("#")]
        paths = [ln.split("=", 1)[1].strip() for ln in lines if ln.startswith("starting_gene_tree")] \
            or [ln for ln in lines if not ln.startswith(("[", "-"))]
    if not paths:
        raise FileNotFoundError(f"no gene trees found for: {spec}")
    return paths


def fit_genewise(
    species_tree,
    gene_trees,
    *,
    device="cuda",
    dtype: torch.dtype = torch.float32,
    min_rate: float = 1e-6,
    max_rate: float = 2.0,
    # --- the recipe (defaults = the accepted optimized recipe) ---
    adam_steps: int = 5,
    adam_lr: float = 1.0,
    grad_clip: float = 10.0,
    pi_tiers=(16, 64),
    neu_opt: int = 16,
    neu_cert: int = 64,
    clade_budget: int | None = None,
    tol: float = 1e-3,
    max_iter: int = 120,
    check_every: int = 4,
    drop_frac: float = 0.30,
    trust: float = 2.0,
    fd_eps: float = 1e-2,
    mu: float = 1e-2,
    fwd_tol: float = 1e-3,
    improve_frac: float = 0.8,
    verify_drop: bool = True,
    eager_defer: bool = True,
    warm_adjoint: bool = True,
    certify: bool = False,
    solver_options: SolverOptions | dict | None = None,
    verbose: bool = False,
) -> dict:
    """Fit per-family DTL rates to a converged, box-bounded optimum. See module docstring for the recipe."""
    dev = torch.device(device)
    pis = [int(p) for p in pi_tiers]
    cert_pi = max(pis)
    lo, hi = log2_rate_bounds(min_rate, max_rate)
    base = dict(_BASE_SOLVER)
    if isinstance(solver_options, SolverOptions):
        base = {k: getattr(solver_options, k) for k in _BASE_SOLVER}
    elif isinstance(solver_options, dict):
        base.update(solver_options)

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    def sopts(pi, neu):
        return SolverOptions(**{**base, "pi_iters": pi, "neumann_terms": neu})

    def build(paths, pi, neu):
        m = GeneReconModel(str(species_tree), [str(p) for p in paths], mode="genewise",
                           device=dev, solver_options=sopts(pi, neu),
                           **({} if clade_budget is None else {"clade_budget": clade_budget}))
        m.receiver_weights.requires_grad_(False)   # uniform transfer recipients (UndatedDTL default)
        return m

    def lg(m, th):
        lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
        return lv.to(dtype), g.to(dtype)

    def pgmax(th, g):
        return project_rate_gradient_(th, g.clone(), min_rate=min_rate, max_rate=max_rate).abs().amax(dim=1)

    def clamp_(th):
        clamp_log_rate_(th, min_rate=min_rate, max_rate=max_rate)
        return th

    def forward_resid(m, th, pi):
        out = torch.zeros(len(m.families), device=dev, dtype=torch.float32)
        rw = m.receiver_weights.detach()
        with torch.no_grad():
            for static in m.batch_statics:
                r = solve_forward_residual(static, m._theta_for_static(static, th), rw, pi_iters=pi)
                out[static.family_index_tensor.to(dev)] = r.to(dev)
        return out

    fam_paths = _resolve_gene_trees(gene_trees)
    F_all = len(fam_paths)
    theta = clamp_(torch.zeros(F_all, 3, device=dev, dtype=dtype))
    active = torch.arange(F_all, device=dev)
    was_dropped = torch.zeros(F_all, dtype=torch.bool, device=dev)
    pg_last = torch.full((F_all,), float("inf"), device=dev, dtype=dtype)
    rebatch_log, defer_log = [], []
    n_steps = n_builds = 0

    _warm_saved = os.environ.get("GPUREC_WARM_ADJOINT")
    if warm_adjoint:
        os.environ["GPUREC_WARM_ADJOINT"] = "1"   # request warm; the library memory-gate disables it if it won't fit
    else:
        os.environ.pop("GPUREC_WARM_ADJOINT", None)
    t0 = time.perf_counter()
    try:
        carry = None
        for pi_idx, pi_cur in enumerate(pis):
            if carry is not None:
                active = carry
            if active.numel() == 0:
                break
            last_tier = pi_idx == len(pis) - 1
            carry = active[:0].clone()
            m = build([fam_paths[j] for j in active.tolist()], pi_cur, neu_opt); n_builds += 1
            sub = theta.index_select(0, active).clone()
            _log(f"[fit_genewise] tier pi={pi_cur}: {active.numel()} families")

            if pi_idx == 0 and adam_steps > 0:   # Adam warm-up (basin entry), once
                lf = sub.clone().requires_grad_(True)
                ad = torch.optim.Adam([lf], lr=adam_lr)
                for _ in range(adam_steps):
                    _, g = lg(m, lf.detach()); lf.grad = g
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(lf, grad_clip)
                    project_rate_gradient_(lf.detach(), lf.grad, min_rate=min_rate, max_rate=max_rate)
                    ad.step()
                    with torch.no_grad():
                        clamp_(lf)
                sub = lf.detach().clone()

            Hd = None
            for it in range(max_iter):
                if active.numel() == 0:
                    break
                lv, g = lg(m, sub)
                if it % check_every == 0:
                    pgm = pgmax(sub, g)
                    plateau = pgm >= improve_frac * pg_last.index_select(0, active)
                    pg_last.index_copy_(0, active, pgm)
                    conv = pgm < tol
                    frac = float(conv.float().mean())
                    _log(f"  [pi{pi_cur} it{it}] active={active.numel()} conv={frac*100:.0f}% "
                         f"|Pg|max={float(pgm.max()):.2e}")
                    if frac > drop_frac and bool(conv.any()):
                        cert_ok = conv.clone()
                        if verify_drop:   # re-check the converged subset cold at the high tier before freezing
                            _w = os.environ.pop("GPUREC_WARM_ADJOINT", None)
                            m.solver_options = sopts(cert_pi, neu_cert)
                            cert_ok = conv & (pgmax(sub, lg(m, sub)[1]) < tol)
                            m.solver_options = sopts(pi_cur, neu_opt)
                            if _w:
                                os.environ["GPUREC_WARM_ADJOINT"] = _w
                        drop = cert_ok
                        defer = torch.zeros_like(conv)
                        reject = conv & ~cert_ok
                        resid_max = 0.0
                        if not last_tier and bool(reject.any()):   # stiff: escalate to the next pi tier
                            resid = forward_resid(m, sub, pi_cur)
                            resid_max = float(resid.max())
                            defer = reject & (resid > fwd_tol)
                            if eager_defer:
                                defer = defer | (reject & plateau)
                        if bool(drop.any()) or bool(defer.any()):
                            if bool(drop.any()):
                                theta.index_copy_(0, active[drop], sub[drop]); was_dropped[active[drop]] = True
                                rebatch_log.append(dict(pi=pi_cur, it=it, dropped=int(drop.sum()),
                                                        remain=int((~drop & ~defer).sum())))
                            if bool(defer.any()):
                                theta.index_copy_(0, active[defer], sub[defer]); carry = torch.cat([carry, active[defer]])
                                defer_log.append(dict(pi=pi_cur, it=it, deferred=int(defer.sum()),
                                                      to=pis[pi_idx + 1], resid_max=resid_max))
                            active = active[~(drop | defer)]; sub = sub[~(drop | defer)].clone()
                            if active.numel() == 0:
                                break
                            del m; torch.cuda.empty_cache()
                            m = build([fam_paths[j] for j in active.tolist()], pi_cur, neu_opt); n_builds += 1
                            Hd = None
                            continue
                if it % 5 == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
                    H = torch.zeros(sub.shape[0], 3, 3, device=dev, dtype=dtype)
                    for j in range(3):
                        tp = sub.clone(); tp[:, j] += fd_eps; _, gp = lg(m, tp)
                        H[:, :, j] = (gp - g) / fd_eps            # forward difference (reuse base g) -> 3 evals
                    H = 0.5 * (H + H.transpose(1, 2))
                    e, V = torch.linalg.eigh(H)
                    Hd = V @ torch.diag_embed(e.clamp(min=mu)) @ V.transpose(1, 2)   # convexify -> PD
                fixed = ((sub >= hi - 1e-6) & (g < 0)) | ((sub <= lo + 1e-6) & (g > 0))
                free = (~fixed).float()
                Hred = Hd * free.unsqueeze(1) * free.unsqueeze(2) + torch.diag_embed(1.0 - free)
                delta = -torch.linalg.solve(Hred, (g * free).unsqueeze(-1)).squeeze(-1)
                dn = delta.norm(dim=1, keepdim=True)
                sub = clamp_(sub + delta * (trust / dn.clamp(min=trust)))   # trust-region cap
                n_steps += 1
            if active.numel() > 0:
                theta.index_copy_(0, active, sub)
                carry = torch.cat([carry, active])
            del m; torch.cuda.empty_cache()
    finally:
        if _warm_saved is None:
            os.environ.pop("GPUREC_WARM_ADJOINT", None)
        else:
            os.environ["GPUREC_WARM_ADJOINT"] = _warm_saved

    result = dict(
        theta=theta, rates=torch.exp2(theta), n_families=F_all,
        opt_seconds=time.perf_counter() - t0, n_steps=n_steps, n_builds=n_builds,
        history=dict(rebatch=rebatch_log, defer=defer_log),
    )
    if certify:   # final cold PD certificate over ALL families at the high pi/Neumann tier
        _w = os.environ.pop("GPUREC_WARM_ADJOINT", None)
        try:
            mfull = build(fam_paths, cert_pi, neu_cert)
            _, g = lg(mfull, theta); pg = pgmax(theta, g)
            H = torch.zeros(F_all, 3, 3, device=dev, dtype=dtype)
            for j in range(3):
                tp = theta.clone(); tp[:, j] += fd_eps; _, gp = lg(mfull, tp)
                tm = theta.clone(); tm[:, j] -= fd_eps; _, gm = lg(mfull, tm)
                H[:, :, j] = (gp - gm) / (2 * fd_eps)
            H = 0.5 * (H + H.transpose(1, 2))
            lam_min = torch.linalg.eigvalsh(H)[:, 0]
            bound_active = ((theta <= lo + 1e-6) | (theta >= hi - 1e-6)).any(dim=1)
            conv = pg < tol
            nll_bits = float(mfull.genewise_loss_vector(theta=theta).sum())
            result.update(
                converged=int(conv.sum()),
                interior_pd=int((conv & (lam_min > tol) & ~bound_active).sum()),
                bound_active=int(bound_active.sum()),
                unconverged=int((~conv).sum()),
                premature_drops=int((was_dropped & ~conv).sum()),
                pg_max=float(pg.max()),
                loss_bits=nll_bits, loss_nats=nll_bits * math.log(2),
            )
            del mfull; torch.cuda.empty_cache()
        finally:
            if _w:
                os.environ["GPUREC_WARM_ADJOINT"] = _w
    return result

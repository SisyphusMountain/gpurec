"""Default genewise (per-family) DTL-rate fitting recipe.

``fit_genewise`` is the standard way to fit independent per-family Duplication/Loss/Transfer rates to a
converged, box-bounded maximum-likelihood optimum -- the same model AleRax fits with ``--rec-model
UndatedDTL --model-parametrization PER-FAMILY``. The recipe (validated on the archaea / Hogenom genewise
benchmarks; see ``kernel-bench/experiments/alerax_speed/OPTIMIZATION_PLAN.md``) is:

  1. **Adam warm-up** -- a few large clipped steps (lr=1, grad-clip-norm 10) projected against the rate
     box, to enter the basin fast.
  2. **Box-constrained trust-region Newton** on the per-family 3x3 **analytic-HVP** Hessian (3
     broadcast unit-theta-component probes, warm-started across repeated rebuilds; eigenvalue-clamped
     to ``mu`` -> PD), converging on the per-family projected gradient ``|Pg| < tol``.
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
from gpurec.config import GpurecConfig, PrecisionOptions, resolve_torch_dtype
from gpurec.config.rates import RateBounds
from gpurec.core.inference.solver import solve_forward_residual
from gpurec.optimization import clamp_log_rate_, log2_rate_bounds, project_rate_gradient_
from gpurec.solver.value_and_grad import forward_solve
from gpurec.solver.hvp_exact import make_exact_hvp

# The genewise rate-bounds preset (floor 1e-6, cap 2.0) -- tighter than the global (1e-10, None)
# floor in gpurec.optimization / GeneReconModel's theta init. Single source for the fit_genewise
# min_rate/max_rate signature defaults below (Global Constraint 2, task-5 brief).
_GENEWISE_RATE_BOUNDS = RateBounds.genewise()

# Proven base solver settings (pi_iters / neumann_terms are overridden per tier below). Single-sourced
# from ``GpurecConfig.genewise_reference().solver`` (task-10 brief) -- edit the values there, not here.
_BASE_SOLVER = {
    k: getattr(GpurecConfig.genewise_reference().solver, k)
    for k in (
        "e_max_iter", "e_tol",
        "bicgstab_max_iter", "bicgstab_tol", "bicgstab_breakdown_tol",
        "adjoint_pruning_threshold", "use_adjoint_pruning", "pibar_side_threshold",
    )
}

# Reference recipe tuned for the standard genewise problem. Import and clone-override per dataset:
#   fit_genewise(sp, genes, **{**GENEWISE_REFERENCE, "tol": 5e-4})
# Per-dataset values belong in your experiment script, NOT edited here (see docs/config_convention.md).
GENEWISE_REFERENCE = dict(
    adam_steps=5, adam_lr=1.0, grad_clip=10.0, pi_tiers=(16, 64), neu_opt=16, neu_cert=64,
    clade_budget=None, tol=1e-3, max_iter=120, check_every=4, drop_frac=0.30, trust=2.0,
    mu=1e-2, fwd_tol=1e-3, improve_frac=0.8, verify_drop=True, eager_defer=True,
    warm_adjoint=True, certify=False,
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


def _analytic_hessian(m, theta, pi_cur):
    """Per-family [G,3,3] curvature via 3 broadcast analytic-HVP probes, warm-started via
    probe_id. Mirrors hvp_exact.py's _make_exact_hvp_streaming genewise gather/scatter for
    single-batch (which the top-level make_exact_hvp does NOT do for you), and lets it handle
    multi-batch itself."""
    G = theta.shape[0]
    dev, dtype = theta.device, theta.dtype
    rw = m.receiver_weights.detach()
    if len(m.batch_statics) > 1:
        hvp = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u = torch.zeros(G, 3, device=dev, dtype=dtype); u[:, j] = 1.0
            cols.append(hvp(u.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3))
        H = torch.stack(cols, dim=-1)
    else:
        static = m.batch_statics[0]
        fam = static.family_index_tensor.to(dev)
        theta_b = theta.index_select(0, fam).contiguous()
        _l, sv = forward_solve(m.batch_statics, theta, rw)
        hvp = make_exact_hvp(m.batch_statics, theta_b, rw, sv, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u_b = torch.zeros(G, 3, device=dev, dtype=dtype); u_b[:, j] = 1.0
            out_b = hvp(u_b.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3)
            col = torch.zeros(G, 3, device=dev, dtype=dtype)
            col.index_add_(0, fam, out_b)
            cols.append(col)
        H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2))


def fit_genewise(
    species_tree,
    gene_trees,
    *,
    device="cuda",
    dtype: torch.dtype | str | None = None,
    min_rate: float = _GENEWISE_RATE_BOUNDS.min_rate,
    max_rate: float = _GENEWISE_RATE_BOUNDS.max_rate,
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
    mu: float = 1e-2,
    fwd_tol: float = 1e-3,
    improve_frac: float = 0.8,
    verify_drop: bool = True,
    eager_defer: bool = True,
    warm_adjoint: bool = True,
    certify: bool = False,
    solver_options: SolverOptions | dict | None = None,
    config: GpurecConfig | None = None,
    verbose: bool = False,
) -> dict:
    """Fit per-family DTL rates to a converged, box-bounded optimum. See module docstring for the recipe.

    ``config`` (a top-level :class:`GpurecConfig`) threads ``config.solver`` (the same key subset as
    ``_BASE_SOLVER``) and ``config.rates`` (``min_rate``/``max_rate``) when the corresponding explicit
    kwarg is left at its signature default; an explicit kwarg always wins. ``config=None`` (the
    default) reproduces today's behavior exactly.

    IMPORTANT -- ``config`` is AUTHORITATIVE, not a partial overlay. Because ``config.solver`` is
    taken wholesale and ``config.rates`` substitutes each field, passing ANY non-default ``config``
    (even one that only tweaks ``e_max_iter``) replaces this recipe's genewise-tuned defaults
    (``bicgstab_tol=1e-7``, ``bicgstab_breakdown_tol=1e-30``, rate box ``1e-6``/``2.0``) with
    ``config``'s values -- which default to the GLOBAL ``SolverOptions()``/``RateBounds()`` defaults.
    To keep the genewise tuning and change only a few knobs, START FROM THE RECIPE FACTORY and modify
    it: ``cfg = GpurecConfig.genewise_reference(); cfg.solver.e_max_iter = 999; fit_genewise(..., config=cfg)``.

    NOT threaded: ``config.newton`` (this recipe's Newton step is a bespoke box-constrained
    trust-region FD 3x3 Hessian solve, not a ``NewtonOptions`` consumer); ``config.regularizer``
    (unused -- this recipe has no regularization term); ``config.memory`` (the adjoint warm-start
    is controlled by the ``GPUREC_WARM_ADJOINT`` env var + the library's own memory gate, not a
    config field).
    """
    precision = config.precision if config is not None else PrecisionOptions()
    if dtype is None:
        dtype = precision.model_torch_dtype
    elif isinstance(dtype, str):
        dtype = resolve_torch_dtype(dtype)
    precision.validate(model_dtype=dtype)

    # RATES: reference-defaults invariant (test_fit_genewise_signature_defaults_come_from_genewise_preset)
    # pins min_rate/max_rate's signature defaults to RateBounds.genewise(). Only substitute
    # config.rates when the kwarg is still at that preset default, so an explicit min_rate/max_rate
    # always wins over config. Documented edge case: a caller who explicitly repasses the preset
    # value AND supplies a differing config gets the config value.
    if config is not None and min_rate == _GENEWISE_RATE_BOUNDS.min_rate:
        min_rate = config.rates.min_rate
    if config is not None and max_rate == _GENEWISE_RATE_BOUNDS.max_rate:
        max_rate = config.rates.max_rate
    dev = torch.device(device)
    pis = [int(p) for p in pi_tiers]
    cert_pi = max(pis)
    bounds = RateBounds(min_rate=min_rate, max_rate=max_rate)
    lo, hi = log2_rate_bounds(bounds=bounds)
    # SOLVER: config.solver supplies `base` (same key subset as _BASE_SOLVER) only when no explicit
    # solver_options is given; an explicit solver_options (SolverOptions or dict) always wins.
    base = dict(_BASE_SOLVER)
    if config is not None and solver_options is None:
        base = {k: getattr(config.solver, k) for k in _BASE_SOLVER}
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
                           device=dev, dtype=dtype, config=config, solver_options=sopts(pi, neu),
                           **({} if clade_budget is None else {"clade_budget": clade_budget}))
        m.receiver_weights.requires_grad_(False)   # uniform transfer recipients (UndatedDTL default)
        return m

    def lg(m, th):
        lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
        return lv.to(dtype), g.to(dtype)

    def pgmax(th, g):
        return project_rate_gradient_(th, g.clone(), bounds=bounds).abs().amax(dim=1)

    def clamp_(th):
        clamp_log_rate_(th, bounds=bounds)
        return th

    def forward_resid(m, th, pi):
        out = torch.zeros(len(m.families), device=dev, dtype=dtype)
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
                    project_rate_gradient_(lf.detach(), lf.grad, bounds=bounds)
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
                    H = _analytic_hessian(m, sub, pi_cur)
                    e, V = torch.linalg.eigh(H)
                    Hd = V @ torch.diag_embed(e.clamp(min=mu)) @ V.transpose(1, 2)   # convexify -> PD
                fixed = ((sub >= hi - bounds.bound_active_eps) & (g < 0)) | \
                    ((sub <= lo + bounds.bound_active_eps) & (g > 0))
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
            H = _analytic_hessian(mfull, theta, cert_pi)
            lam_min = torch.linalg.eigvalsh(H)[:, 0]
            bound_active = ((theta <= lo + bounds.bound_active_eps) | (theta >= hi - bounds.bound_active_eps)).any(dim=1)
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

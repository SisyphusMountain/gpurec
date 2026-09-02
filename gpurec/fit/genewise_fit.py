"""Default genewise (per-family) DTL-rate fitting recipe.

``fit_genewise`` is the standard way to fit independent per-family Duplication/Loss/Transfer rates to a
converged, box-bounded maximum-likelihood optimum -- the same model AleRax fits with ``--rec-model
UndatedDTL --model-parametrization PER-FAMILY``. The recipe (validated on the archaea / Hogenom genewise
benchmarks; see ``kernel-bench/experiments/alerax_speed/OPTIMIZATION_PLAN.md``) is:

  1. **Adam warm-up** -- a few large clipped steps (lr=1, grad-clip-norm 10) projected against the rate
     box, to enter the basin fast.
  2. **Box-constrained trust-region Newton** on the per-family 3x3 curvature matrix. The matrix is
     the exact **analytic-HVP** Hessian (3 broadcast unit-theta-component probes) at the first Newton
     iteration of each pi tier and again every ``hessian_refresh`` iterations; in between it is carried
     forward per family by a **BFGS** update built from the step and the gradient change the family
     actually took (see ``_bfgs_update``). Either way it is eigenvalue-clamped to ``mu`` -> PD before the
     step. Converges on the per-family projected gradient ``|Pg| < tol``.
  3. **Convergence-based freezing, then rebatching** -- as soon as ``min_drop`` families (or
     ``drop_frac`` of the active batch) look converged at the cheap tier, ONLY those candidates are
     re-checked cold at the high pi/Neumann tier -- on a temporary model built over just them -- and
     the ones that pass are FROZEN: their fitted rates are final and they stop being stepped, but they
     stay in the current model, so no rebuild happens yet. The model is re-planned over the survivors
     only once the frozen families own ``rebuild_frac`` of its clades. That split matters because the
     two costs are very different: at 5123 families a rebuild is ~30-40 s of re-planning while
     carrying a frozen family costs only its clade share of one ~55 s gradient per iteration, so
     rebuilding for a 5% shrink loses time and rebuilding at 25% pays for itself in a couple of
     iterations. The accurate-tier gradient is never paid for a non-candidate.
  4. **pi-tier escalation** -- the bulk runs at a cheap forward pi (16) with the adjoint **warm-start**
     (``GPUREC_WARM_ADJOINT``, memory-gated automatically -- see ``gpurec.core.memory_policy``); only the
     forward-stiff families escalate to the accurate pi (64).

Returns a dict with the fitted log2-rates ``theta`` ([F,3]), the AleRax-comparable ``rates`` (= 2**theta,
relative to speciation), the rebatch/defer history, per-phase timings, and -- when ``certify=True`` --
a final cold certificate (per-family |Pg|, bound-active counts, total NLL, and the smallest Hessian
eigenvalue / interior-PD count when ``certify_curvature=True``).

**Convergence is certified at freeze time.** A family's accurate-tier ``|Pg|`` is measured when it is
frozen, at exactly the theta it keeps for the rest of the fit, so the certificate reuses that number
instead of re-deriving it: it only computes an accurate-tier gradient for the families that were never
frozen (the unconverged / still-live survivors), on a small model over just them. The total NLL is a
forward-only accurate-tier pass over ALL families, so the headline likelihood is still one consistent
measurement on one model.
"""
from __future__ import annotations

import glob
import math
import os
import time

import torch

from gpurec.api import _failure_dump
from gpurec.api.model import GeneReconModel
from gpurec.api.solver_options import SolverOptions
from gpurec.config import GpurecConfig, PrecisionOptions, resolve_torch_dtype
from gpurec.config.rates import RateBounds
from gpurec.core.inference.solver import solve_forward_residual
from gpurec.core.scheduling.batching import parse_families
from gpurec.optimization import clamp_log_rate_, log2_rate_bounds, project_rate_gradient_
from gpurec.solver.value_and_grad import forward_solve, free_cuda_cache_if_tight
from gpurec.solver.hvp.exact import make_exact_hvp_single

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
        "e_adjoint_max_iter", "e_adjoint_tol",
        "adjoint_pruning_threshold", "use_adjoint_pruning", "pibar_side_threshold",
        # kernel-path knobs (fused self-loops): threaded from config so a run can select the
        # log-space forward or disable the early exits without editing the recipe.
        "forward_self_loop", "adjoint_self_loop", "pi_linear_tol", "neumann_term_tol",
    )
}

# Reference recipe tuned for the standard genewise problem. Import and clone-override per dataset:
#   fit_genewise(sp, genes, **{**GENEWISE_REFERENCE, "tol": 5e-4}, min_drop=32,
#                rebuild_frac=0.25, hessian_refresh=15, init_curvature="adam_bfgs",
#                certify_curvature=False)
# ``min_drop`` / ``rebuild_frac`` / ``hessian_refresh`` / ``init_curvature`` / ``certify_curvature``
# have NO signature default (every caller states them), so they are not in this dict -- pass them
# alongside it, as above.
# Per-dataset values belong in your experiment script, NOT edited here (see docs/config_convention.md).
GENEWISE_REFERENCE = dict(
    adam_steps=5, adam_lr=1.0, grad_clip=10.0, pi_tiers=(16, 64), neu_opt=16, neu_cert=64,
    clade_budget=None, tol=1e-3, max_iter=120, check_every=2, drop_frac=0.05, trust=2.0,
    mu=1e-2, fwd_tol=1e-3, improve_frac=0.8, verify_drop=True, eager_defer=True,
    warm_adjoint=True, certify=False,
)

# BFGS curvature-condition floor: a family's 3x3 matrix is only updated when the step and the
# gradient change point the same way, i.e. s.y > 1e-10 * |s| * |y|. Not a setting -- it is the
# standard "skip the update rather than destroy positive-definiteness" guard, and the fit is
# insensitive to its exact value (it only has to exclude s.y <= 0 and 0/0).
_BFGS_CURVATURE_FLOOR = 1e-10


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


def _report_batch_failure(static, theta_batch, receiver_weights, *, species_tree, family_paths,
                          reason, extra):
    """Print where this batch's forward turns non-finite and write it out, when dumps are on."""
    if not _failure_dump.is_enabled():
        return
    print(f"[gpurec] {reason} failed on a {len(static.family_indices)}-family batch; "
          f"theta min/max per column = "
          f"{[round(float(v), 3) for v in theta_batch.amin(dim=0)]} / "
          f"{[round(float(v), 3) for v in theta_batch.amax(dim=0)]}")
    print(_failure_dump.describe_forward_state(static, theta_batch, receiver_weights))
    path = _failure_dump.save_batch(
        static, theta_batch, receiver_weights, species_tree=species_tree,
        family_paths=family_paths, reason=reason, extra=extra,
    )
    print(f"[gpurec] wrote the failing batch to {path}")


def _analytic_hessian(m, theta, pi_cur, species_tree, model_family_paths):
    """Per-family [G,3,3] curvature via 3 analytic-HVP probes (unit theta-component directions).

    Batches are streamed one at a time. For each batch the forward solve and the adjoint point
    cache are built ONCE and shared by the 3 probes; the library's multi-batch streaming HVP
    (``make_exact_hvp`` on >1 batch) instead rebuilds both per probe, i.e. 3x the forward+backward
    work for an identical result. Probe ``j`` of batch ``b`` is scattered into the full ``[G,3]``
    column ``j`` on the batch's (disjoint) family rows, so the per-family 3x3 blocks are exact.
    Single-batch models take the same path (``forward_solve`` on a length-1 list and
    ``make_exact_hvp_single`` are what ``make_exact_hvp`` dispatched to before).
    """
    G = theta.shape[0]
    dev, dtype = theta.device, theta.dtype
    rw = m.receiver_weights.detach()
    cols = [torch.zeros(G, 3, device=dev, dtype=dtype) for _ in range(3)]
    for static in m.batch_statics:
        fam = static.family_index_tensor.to(dev)
        G_b = int(fam.numel())
        theta_b = theta.index_select(0, fam).contiguous()
        _l, sv = forward_solve([static], theta, rw)
        hvp = make_exact_hvp_single(static, theta_b, rw, sv, tangent_self_iters=pi_cur)
        for j in range(3):
            u_b = torch.zeros(G_b, 3, device=dev, dtype=dtype); u_b[:, j] = 1.0
            try:
                out_b = hvp(u_b.reshape(-1), probe_id=j)[: G_b * 3].reshape(G_b, 3)
            except RuntimeError:
                # A solve failing here names no families and no rates, and the fit cannot be
                # replayed to this iterate. When a driver has asked for dumps, say where the
                # first non-finite value is and write the batch out before re-raising.
                _report_batch_failure(
                    static, theta_b, rw, species_tree=species_tree,
                    family_paths=model_family_paths,
                    reason="analytic_hessian", extra={"pi_iters": int(pi_cur), "probe": j},
                )
                raise
            cols[j].index_add_(0, fam, out_b)
        del hvp, sv
        free_cuda_cache_if_tight()
    H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2))


def _bfgs_update(B, s, y, free_both):
    """Per-family BFGS refresh of the [G,3,3] curvature matrices ``B``. Returns the updated copy.

    For one family, ``s`` is the step it just took (theta_new - theta_old) and ``y`` is the change in
    its gradient over that same step (g_new - g_old). The secant condition says the curvature matrix
    should map the step onto the gradient change (``B_new @ s = y``); the BFGS formula is the rank-2
    correction that enforces it while staying symmetric and as close to ``B`` as possible:

        B_new = B - (B s)(B s)^T / (s . B s)  +  y y^T / (y . s)

    ``free_both`` is 1.0 on the coordinates that were strictly inside the rate box at BOTH iterates
    and 0.0 elsewhere; ``s`` and ``y`` are zeroed there before the update, so a coordinate pinned at a
    bound (whose step was cut by the projection, not by curvature) cannot inject bogus curvature. The
    two correction terms then live entirely in the free block, leaving the rest of ``B`` untouched.

    A family is left alone (its old ``B`` kept) when ``s . y <= 1e-10 * |s| * |y|`` -- the curvature
    condition, which also covers ``s = 0`` (no step, e.g. the iteration right after a rebuild) -- or
    when ``s . B s <= 0`` (the old matrix has no positive curvature along the step, so the first
    correction term would blow up), or when the result is not finite.
    """
    s = s * free_both
    y = y * free_both
    Bs = torch.einsum("gij,gj->gi", B, s) * free_both
    sBs = (s * Bs).sum(dim=1)
    sy = (s * y).sum(dim=1)
    ok = (sy > _BFGS_CURVATURE_FLOOR * s.norm(dim=1) * y.norm(dim=1)) & (sBs > 0)
    safe_sy = torch.where(ok, sy, torch.ones_like(sy))[:, None, None]
    safe_sBs = torch.where(ok, sBs, torch.ones_like(sBs))[:, None, None]
    upd = (y.unsqueeze(2) * y.unsqueeze(1)) / safe_sy - (Bs.unsqueeze(2) * Bs.unsqueeze(1)) / safe_sBs
    ok = ok & torch.isfinite(upd).flatten(1).all(dim=1)
    return B + torch.where(ok[:, None, None], upd, torch.zeros_like(upd))


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
    check_every: int = 2,
    drop_frac: float = 0.05,
    min_drop: int,
    rebuild_frac: float,
    hessian_refresh: int,
    init_curvature: str,
    trust: float = 2.0,
    mu: float = 1e-2,
    fwd_tol: float = 1e-3,
    improve_frac: float = 0.8,
    verify_drop: bool = True,
    eager_defer: bool = True,
    warm_adjoint: bool = True,
    certify: bool = False,
    certify_curvature: bool,
    init_log2_rates: tuple[float, float, float],
    solver_options: SolverOptions | dict | None = None,
    config: GpurecConfig | None = None,
    verbose: bool = False,
) -> dict:
    """Fit per-family DTL rates to a converged, box-bounded optimum. See module docstring for the recipe.

    Five keywords have no default and must be stated by every caller:

    ``min_drop`` -- how many families must look converged at the cheap tier before the fit pays for a
    verification round (a temporary accurate-tier model over just those candidates). A round also
    triggers on ``drop_frac`` of the still-live families, whichever comes first. Families that pass
    are frozen where they are; the model is NOT re-planned at that moment.

    ``rebuild_frac`` -- the share of the current model's clades that must belong to frozen (or
    deferred) families before the model is actually re-planned over the survivors. Frozen families
    still cost their clade share of every gradient, so this trades that against the fixed cost of a
    re-plan; 0.25 is the production value.

    ``hessian_refresh`` -- how many Newton iterations between exact analytic-HVP Hessians. The exact
    Hessian costs about 7 gradients; in between, each family's 3x3 is carried by the BFGS update in
    ``_bfgs_update``. An exact Hessian is always computed at the first iteration of each pi tier.

    ``init_curvature`` -- where the FIRST tier's starting 3x3 curvature comes from. ``"exact"``
    computes the 3-probe analytic Hessian (about 7 gradients). ``"adam_bfgs"`` instead builds it out
    of the Adam warm-up's own consecutive (step, gradient-change) pairs, which are already paid for:
    start from a Barzilai-Borwein scaled identity (``(y.y)/(s.y)`` per family, the scalar curvature
    that best fits the last pair) and fold in each pair with the same BFGS update the Newton loop
    uses. Later tiers and every ``hessian_refresh`` refresh still use the exact Hessian, so this only
    changes how the first few Newton steps are aimed.

    ``certify_curvature`` -- whether the final ``certify=True`` certificate also computes the 3-probe
    Hessian and its smallest eigenvalue. That is what fills in ``interior_pd``; with False the
    certificate still reports ``converged`` / ``unconverged`` / ``bound_active`` / ``pg_max`` and the
    total NLL, and ``interior_pd`` is simply absent from the result. NOTE: because convergence is
    certified at freeze time, ``premature_drops`` is 0 by construction -- a frozen family's reported
    ``|Pg|`` IS the one that justified freezing it. The key stays for result-shape compatibility.

    ``config`` (a top-level :class:`GpurecConfig`) threads ``config.solver`` (the same key subset as
    ``_BASE_SOLVER``) and ``config.rates`` (``min_rate``/``max_rate``) when the corresponding explicit
    kwarg is left at its signature default; an explicit kwarg always wins. ``config=None`` (the
    default) reproduces today's behavior exactly.

    IMPORTANT -- ``config`` is AUTHORITATIVE, not a partial overlay. Because ``config.solver`` is
    taken wholesale and ``config.rates`` substitutes each field, passing ANY non-default ``config``
    (even one that only tweaks ``e_max_iter``) replaces this recipe's genewise-tuned defaults
    (``e_adjoint_tol=1e-7``, rate box ``1e-6``/``2.0``) with
    ``config``'s values -- which default to the GLOBAL ``SolverOptions()``/``RateBounds()`` defaults.
    To keep the genewise tuning and change only a few knobs, START FROM THE RECIPE FACTORY and modify
    it: ``cfg = GpurecConfig.genewise_reference(); cfg.solver.e_max_iter = 999; fit_genewise(..., config=cfg)``.

    NOT threaded: ``config.newton`` (this recipe's Newton step is a bespoke box-constrained
    trust-region analytic-HVP 3x3 Hessian solve, not a ``NewtonOptions`` consumer); ``config.regularizer``
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
    if init_curvature not in ("exact", "adam_bfgs"):
        raise ValueError(f'init_curvature must be "exact" or "adam_bfgs", got {init_curvature!r}')
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

    if base.get("forward_self_loop") == "exact":
        # The exact tree solve returns the converged fixed point whatever `pi_iters` is, so the
        # second (pi=64) tier would recompute an identical forward: run a single tier and let
        # families that do not converge simply stay unconverged instead of being deferred
        # (deferral keyed on `pi_residual_out`, which in exact mode is |prologue - solution|, not
        # a stiffness signal). Backward accuracy still follows `neu_opt` / `neu_cert` as before.
        pis = pis[:1]
        cert_pi = pis[0]

    def sopts(pi, neu):
        return SolverOptions(**{**base, "pi_iters": pi, "neumann_terms": neu})

    def _sync():
        """Make the wall-clock timings below honest: GPU work is queued asynchronously."""
        if dev.type == "cuda":
            torch.cuda.synchronize()

    def clade_counts(model):
        """Per-family clade counts of ``model``, in its own family order (= the ``active`` order).

        The clade count is what a family costs in a gradient (the solver walks its clades), so it is
        also the right weight for deciding when the frozen families have grown expensive enough to
        justify re-planning the model without them.
        """
        return torch.tensor([int(f["C"]) for f in model.families], device=dev, dtype=torch.float64)

    def build(indices, pi, neu):
        """Rebuild the model over ``indices`` into ``fam_paths``, reusing the parsed families.

        Every rebuild (rebatch / tier escalation / certificate) re-plans batches and rebuilds
        tensors from the SAME resident parse -- no .ale file is read more than once per fit.
        """
        idx = [int(i) for i in indices]
        m = GeneReconModel(str(species_tree), [str(fam_paths[i]) for i in idx], mode="genewise",
                           device=dev, dtype=dtype, config=config, solver_options=sopts(pi, neu),
                           parsed_families=parsed, family_indices=idx,
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
    # Parse every family ONCE for the whole fit; build() re-plans subsets off this handle.
    parsed = parse_families(species_tree, fam_paths)
    # Starting point for every family's [log2 D, log2 L, log2 T]. The historical start was all
    # zeros (every rate = 1.0 x speciation), which is both far from typical optima and in the
    # slow, stiff high-rate regime for the wave/E fixed points; callers pass the start explicitly.
    theta = clamp_(torch.tensor(init_log2_rates, device=dev, dtype=dtype).reshape(1, 3).repeat(F_all, 1).contiguous())
    active = torch.arange(F_all, device=dev)
    was_dropped = torch.zeros(F_all, dtype=torch.bool, device=dev)
    pg_last = torch.full((F_all,), float("inf"), device=dev, dtype=dtype)
    rebatch_log, defer_log = [], []
    n_steps = n_builds = n_verify_builds = n_rebuilds = n_hessians = 0
    verify_seconds = rebuild_seconds = adam_seconds = 0.0
    hessian_seconds = newton_grad_seconds = certify_seconds = 0.0
    # Accurate-tier |Pg| measured at the moment a family was frozen, at the theta it keeps for the
    # rest of the fit. The certificate reuses these instead of re-running a gradient over everything.
    cert_pg = torch.full((F_all,), float("inf"), device=dev, dtype=dtype)
    cert_known = torch.zeros(F_all, dtype=torch.bool, device=dev)
    # Curvature state, kept per GLOBAL family index so it survives every rebuild (a rebuild changes
    # which families are in the batch, never their theta): B_fam is the raw (un-convexified) 3x3
    # curvature matrix, and prev_* is the (theta, gradient, free-coordinate) triple of the last
    # iterate, which the BFGS update differences against the current one.
    B_fam = torch.zeros(F_all, 3, 3, device=dev, dtype=dtype)
    prev_theta = torch.zeros(F_all, 3, device=dev, dtype=dtype)
    prev_g = torch.zeros(F_all, 3, device=dev, dtype=dtype)
    prev_free = torch.zeros(F_all, 3, device=dev, dtype=dtype)

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
            m = build(active.tolist(), pi_cur, neu_opt); n_builds += 1
            sub = theta.index_select(0, active).clone()
            # ``settled`` marks the rows of the current model that are finished for this tier --
            # frozen (verified converged, theta final) or deferred to the next pi tier. They are
            # still IN the model (and so still cost gradient time) until a re-plan removes them.
            clades = clade_counts(m); clade_total = float(clades.sum())
            settled = torch.zeros(active.numel(), dtype=torch.bool, device=dev)
            _log(f"[fit_genewise] tier pi={pi_cur}: {active.numel()} families")

            # An exact Hessian is due on the tier's first Newton iteration (nothing has been
            # measured for these families at this tier yet); it stays due until one is actually
            # computed, so a rebatch landing on a refresh iteration only postpones it by one step.
            # ``since_exact`` counts the Newton steps taken since the last exact Hessian.
            refresh_due, since_exact = True, 0
            if pi_idx == 0 and adam_steps > 0:   # Adam warm-up (basin entry), once
                _sync(); _t = time.perf_counter()
                lf = sub.clone().requires_grad_(True)
                ad = torch.optim.Adam([lf], lr=adam_lr)
                pairs, seen = [], None
                for _ in range(adam_steps):
                    _, g = lg(m, lf.detach())
                    th_a, g_a = lf.detach().clone(), g.clone()   # BEFORE clipping mutates lf.grad
                    fx_a = ((th_a >= hi - bounds.bound_active_eps) & (g_a < 0)) | \
                        ((th_a <= lo + bounds.bound_active_eps) & (g_a > 0))
                    free_a = (~fx_a).to(dtype)
                    if seen is not None:   # one (step, gradient-change) pair per Adam transition
                        pairs.append((th_a - seen[0], g_a - seen[1], free_a * seen[2]))
                    seen = (th_a, g_a, free_a)
                    lf.grad = g
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(lf, grad_clip)
                    project_rate_gradient_(lf.detach(), lf.grad, bounds=bounds)
                    ad.step()
                    with torch.no_grad():
                        clamp_(lf)
                sub = lf.detach().clone()
                if init_curvature == "adam_bfgs" and pairs:
                    # Barzilai-Borwein scaled identity from the LAST pair, then fold in every pair.
                    s_l, y_l, f_l = pairs[-1]
                    s_l, y_l = s_l * f_l, y_l * f_l
                    sy = (s_l * y_l).sum(dim=1)
                    good = sy > _BFGS_CURVATURE_FLOOR * s_l.norm(dim=1) * y_l.norm(dim=1)
                    scale = torch.where(good, (y_l * y_l).sum(dim=1) / torch.where(good, sy, torch.ones_like(sy)),
                                        torch.ones_like(sy))
                    B = scale[:, None, None] * torch.eye(3, device=dev, dtype=dtype)
                    for s_i, y_i, f_i in pairs:
                        B = _bfgs_update(B, s_i, y_i, f_i)
                    B_fam.index_copy_(0, active, B)
                    refresh_due = False   # the warm-up already paid for this curvature
                _sync(); adam_seconds += time.perf_counter() - _t
            for it in range(max_iter):
                live = ~settled
                if not bool(live.any()):
                    break
                _sync(); _t = time.perf_counter()
                lv, g = lg(m, sub)
                _sync(); newton_grad_seconds += time.perf_counter() - _t
                fixed = ((sub >= hi - bounds.bound_active_eps) & (g < 0)) | \
                    ((sub <= lo + bounds.bound_active_eps) & (g > 0))
                free = (~fixed).to(dtype)

                refresh_due = refresh_due or since_exact >= hessian_refresh
                if not refresh_due:   # carry the curvature forward from the step just taken
                    both = free * prev_free.index_select(0, active)
                    B_fam.index_copy_(0, active, _bfgs_update(
                        B_fam.index_select(0, active),
                        sub - prev_theta.index_select(0, active),
                        g - prev_g.index_select(0, active), both))
                prev_theta.index_copy_(0, active, sub)
                prev_g.index_copy_(0, active, g)
                prev_free.index_copy_(0, active, free)

                if it % check_every == 0:
                    pgm = pgmax(sub, g)
                    plateau = pgm >= improve_frac * pg_last.index_select(0, active)
                    pg_last.index_copy_(0, active, torch.where(live, pgm, pg_last.index_select(0, active)))
                    conv = live & (pgm < tol)
                    n_conv, n_live = int(conv.sum()), int(live.sum())
                    _log(f"  [pi{pi_cur} it{it}] live={n_live} (+{int(settled.sum())} settled in batch) "
                         f"conv={n_conv} |Pg|max={float(pgm[live].max()):.2e}")
                    if n_conv > 0 and (n_conv >= min_drop or n_conv >= drop_frac * n_live):
                        cert_ok = conv.clone()
                        if verify_drop:   # re-check the CANDIDATES ONLY, cold at the high tier
                            _sync(); _t = time.perf_counter()
                            cand = conv.nonzero(as_tuple=True)[0]
                            sub_c = sub.index_select(0, cand)
                            _w = os.environ.pop("GPUREC_WARM_ADJOINT", None)
                            mv = build(active.index_select(0, cand).tolist(), cert_pi, neu_cert)
                            n_verify_builds += 1; n_builds += 1
                            pg_c = pgmax(sub_c, lg(mv, sub_c)[1])
                            ok_c = pg_c < tol
                            cert_pg.index_copy_(0, active.index_select(0, cand), pg_c)
                            del mv; torch.cuda.empty_cache()
                            if _w:
                                os.environ["GPUREC_WARM_ADJOINT"] = _w
                            cert_ok = torch.zeros_like(conv)
                            cert_ok.index_copy_(0, cand, ok_c)
                            _sync(); verify_seconds += time.perf_counter() - _t
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
                        if bool(drop.any()):   # FREEZE in place: theta is final, no re-plan yet
                            theta.index_copy_(0, active[drop], sub[drop]); was_dropped[active[drop]] = True
                            if verify_drop:   # its |Pg| was just measured at exactly this theta
                                cert_known[active[drop]] = True
                            rebatch_log.append(dict(pi=pi_cur, it=it, dropped=int(drop.sum()),
                                                    remain=int((live & ~drop & ~defer).sum())))
                        if bool(defer.any()):
                            theta.index_copy_(0, active[defer], sub[defer]); carry = torch.cat([carry, active[defer]])
                            defer_log.append(dict(pi=pi_cur, it=it, deferred=int(defer.sum()),
                                                  to=pis[pi_idx + 1], resid_max=resid_max))
                        settled = settled | drop | defer
                        live = ~settled
                        if not bool(live.any()):
                            break
                        # Re-plan only once the settled families own enough of this model's clades.
                        if float(clades[settled].sum()) >= rebuild_frac * clade_total:
                            _sync(); _t = time.perf_counter()
                            active = active[live]; sub = sub[live].clone()
                            del m; torch.cuda.empty_cache()
                            m = build(active.tolist(), pi_cur, neu_opt); n_builds += 1; n_rebuilds += 1
                            clades = clade_counts(m); clade_total = float(clades.sum())
                            settled = torch.zeros(active.numel(), dtype=torch.bool, device=dev)
                            _sync(); rebuild_seconds += time.perf_counter() - _t
                            _log(f"  [pi{pi_cur} it{it}] re-planned over {active.numel()} live families")
                            continue   # the gradient above belongs to the old batch; re-measure
                if refresh_due:
                    _sync(); _t = time.perf_counter()
                    B_fam.index_copy_(
                        0, active,
                        _analytic_hessian(
                            m, sub, pi_cur, species_tree,
                            # ``static.family_indices`` are positions in THIS model's family
                            # list, so hand over the paths already in that order.
                            [fam_paths[i] for i in active.tolist()],
                        ),
                    )
                    _sync(); hessian_seconds += time.perf_counter() - _t
                    refresh_due = False; since_exact = 0; n_hessians += 1
                e, V = torch.linalg.eigh(B_fam.index_select(0, active))
                Hd = V @ torch.diag_embed(e.clamp(min=mu)) @ V.transpose(1, 2)   # convexify -> PD
                Hred = Hd * free.unsqueeze(1) * free.unsqueeze(2) + torch.diag_embed(1.0 - free)
                delta = -torch.linalg.solve(Hred, (g * free).unsqueeze(-1)).squeeze(-1)
                delta = delta * live.unsqueeze(1).to(dtype)   # settled rows keep their frozen theta
                dn = delta.norm(dim=1, keepdim=True)
                sub = clamp_(sub + delta * (trust / dn.clamp(min=trust)))   # trust-region cap
                n_steps += 1; since_exact += 1
            live = ~settled
            if bool(live.any()):   # ran out of iterations: carry the unfinished families forward
                theta.index_copy_(0, active[live], sub[live])
                carry = torch.cat([carry, active[live]])
            del m; torch.cuda.empty_cache()
    finally:
        if _warm_saved is None:
            os.environ.pop("GPUREC_WARM_ADJOINT", None)
        else:
            os.environ["GPUREC_WARM_ADJOINT"] = _warm_saved

    result = dict(
        theta=theta, rates=torch.exp2(theta), n_families=F_all,
        opt_seconds=time.perf_counter() - t0, n_steps=n_steps, n_builds=n_builds,
        # Where the time went, phase by phase (each bracketed by a CUDA sync, so they are real
        # wall-clock seconds and they sum to a little less than ``opt_seconds`` -- the remainder is
        # the tier builds, the parse and the small tensor bookkeeping).
        #   adam      -- the warm-up gradients
        #   hessian   -- the exact 3-probe analytic Hessians (n_hessians of them)
        #   newton_grad -- the Newton loop's own gradients
        #   verify    -- candidate-only models + their accurate-tier gradient
        #   rebuild   -- re-planning the model over the survivors
        #   certify   -- the whole final certificate
        adam_seconds=adam_seconds, hessian_seconds=hessian_seconds, n_hessians=n_hessians,
        newton_grad_seconds=newton_grad_seconds,
        n_verify_builds=n_verify_builds, verify_seconds=verify_seconds,
        n_rebuilds=n_rebuilds, rebuild_seconds=rebuild_seconds,
        certify_seconds=certify_seconds,   # overwritten below when certify=True
        history=dict(rebatch=rebatch_log, defer=defer_log),
    )
    if certify:   # final cold certificate at the high pi/Neumann tier
        _w = os.environ.pop("GPUREC_WARM_ADJOINT", None)
        try:
            _sync(); _t = time.perf_counter()
            # 1. |Pg|: reuse the freeze-time measurement (taken at this exact theta, at this exact
            #    tier) and pay a gradient ONLY for the families that were never frozen.
            pg = cert_pg.clone()
            need = (~cert_known).nonzero(as_tuple=True)[0]
            if 0 < need.numel() < F_all:
                mneed = build(need.tolist(), cert_pi, neu_cert)
                th_n = theta.index_select(0, need)
                pg.index_copy_(0, need, pgmax(th_n, lg(mneed, th_n)[1]))
                del mneed; torch.cuda.empty_cache()
            # 2. the headline likelihood: ONE forward-only pass over every family, so the total is a
            #    single consistent measurement on a single model (no backward, no Hessian).
            mfull = build(range(F_all), cert_pi, neu_cert)
            if need.numel() == F_all:   # nothing was ever frozen (verify_drop=False): one model does both
                pg = pgmax(theta, lg(mfull, theta)[1])
            with torch.no_grad():
                nll_bits = float(mfull.genewise_loss_vector(theta=theta).sum())
            bound_active = ((theta <= lo + bounds.bound_active_eps) | (theta >= hi - bounds.bound_active_eps)).any(dim=1)
            conv = pg < tol
            result.update(
                converged=int(conv.sum()),
                bound_active=int(bound_active.sum()),
                unconverged=int((~conv).sum()),
                # 0 by construction: a frozen family's |Pg| IS the one that justified freezing it.
                premature_drops=int((was_dropped & ~conv).sum()),
                pg_max=float(pg.max()),
                loss_bits=nll_bits, loss_nats=nll_bits * math.log(2),
            )
            if certify_curvature:   # the 3-probe Hessian is ~7 gradients; only pay it when asked
                lam_min = torch.linalg.eigvalsh(
                    _analytic_hessian(mfull, theta, cert_pi, species_tree, list(fam_paths))
                )[:, 0]
                result["interior_pd"] = int((conv & (lam_min > tol) & ~bound_active).sum())
            del mfull; torch.cuda.empty_cache()
            _sync(); result["certify_seconds"] = time.perf_counter() - _t
        finally:
            if _w:
                os.environ["GPUREC_WARM_ADJOINT"] = _w
    return result

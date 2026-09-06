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
     forward per family by the caller's ``curvature_update`` -- a **BFGS**, an **SR1** or a
     **multi-secant** fit to the step(s) and gradient change(s) the family actually took (see
     ``_bfgs_update`` / ``_sr1_update`` / ``_multisecant_solve``). Either way it is eigenvalue-clamped to ``mu`` -> PD before the
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
  4. **range fallback** -- the bulk uses the exact tree solve; only numerically wide rows take the
     iterative log-space fallback.

Returns a dict with the fitted log2-rates ``theta`` ([F,3]), the AleRax-comparable ``rates`` (= 2**theta,
relative to speciation), the rebatch/defer history, per-phase timings, and -- when ``certify=True`` --
a final cold certificate (per-family |Pg|, bound-active counts, total NLL, and the smallest Hessian
eigenvalue / interior-PD count when ``certify_curvature=True``).

**Convergence is certified at freeze time.** A family's accurate-tier ``|Pg|`` is measured when it is
frozen, at exactly the theta it keeps for the rest of the fit, so the certificate reuses that number
instead of re-deriving it (and when the live model already runs at the certificate tier with the
exact adjoint solve, the live gradient is that measurement and no candidate model is built at all): it only computes an accurate-tier gradient for the families that were never
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
from gpurec.config.memory import MemoryOptions
from gpurec.config.rates import RateBounds
from gpurec.core.inference.solver import solve_forward_residual
from gpurec.core.memory_policy import clade_budget_for_device
from gpurec.core.scheduling.batching import DEFAULT_CLADE_BUDGET, parse_families
from gpurec.optimization import clamp_log_rate_, log2_rate_bounds, project_rate_gradient_
from gpurec.solver.value_and_grad import forward_solve, free_cuda_cache_if_tight
from gpurec.solver.hvp.exact import make_exact_hvp_single

# The genewise rate-bounds preset (floor 1e-6, cap 2.0) -- tighter than the global (1e-10, None)
# floor in gpurec.optimization / GeneReconModel's theta init. Single source for the fit_genewise
# min_rate/max_rate signature defaults below (Global Constraint 2, task-5 brief).
_GENEWISE_RATE_BOUNDS = RateBounds.genewise()

# The library-default rate box (the GLOBAL floor 1e-10 with NO cap), used below as the "the config's
# [rates] table was never set" sentinel: a GpurecConfig built without a [rates] table holds exactly
# this object's values, and substituting it over the genewise preset would hand a None cap to
# log2_rate_bounds. Built once here at import rather than inline at the comparison site, because
# tests/test_config_wiring.py's fit_genewise probe monkeypatches this module's ``RateBounds`` name to
# record every box the fit constructs; an inline ``RateBounds()`` would be recorded as a spurious
# first box and hide the one the fit actually uses.
_GLOBAL_RATE_BOUNDS = RateBounds()

# Proven base solver settings (pi_iters / neumann_terms are overridden per tier below). Single-sourced
# from ``GpurecConfig.genewise_reference().solver`` (task-10 brief) -- edit the values there, not here.
_BASE_SOLVER = {
    k: getattr(GpurecConfig.genewise_reference().solver, k)
    for k in (
        "e_max_iter", "e_tol",
        "e_adjoint_max_iter", "e_adjoint_tol",
        "adjoint_pruning_threshold", "use_adjoint_pruning", "pibar_side_threshold",
        "neumann_term_tol",
    )
}

# Reference recipe tuned for the standard genewise problem. Import and clone-override per dataset:
#   fit_genewise(sp, genes, **{**GENEWISE_REFERENCE, "tol": 5e-4}, min_drop=32,
#                rebuild_frac=0.25, hessian_refresh=15, init_curvature="adam_bfgs",
#                curvature_update="bfgs", certify_curvature=False)
# ``min_drop`` / ``rebuild_frac`` / ``hessian_refresh`` / ``init_curvature`` / ``curvature_update``
# / ``certify_curvature``
# have NO signature default (every caller states them), so they are not in this dict -- pass them
# alongside it, as above.
# Per-dataset values belong in your experiment script, NOT edited here (see docs/config_convention.md).
GENEWISE_REFERENCE = dict(
    adam_steps=5, adam_lr=1.0, grad_clip=10.0, pi_tiers=(16, 64), neu_opt=16, neu_cert=64,
    clade_budget=None, tol=1e-3, max_iter=120, check_every=2, drop_frac=0.05, trust=2.0,
    mu=1e-4, fwd_tol=1e-3, improve_frac=0.8, verify_drop=True, eager_defer=True,
    certify=False,
)

# BFGS curvature-condition floor: a family's 3x3 matrix is only updated when the step and the
# gradient change point the same way, i.e. s.y > 1e-10 * |s| * |y|. Not a setting -- it is the
# standard "skip the update rather than destroy positive-definiteness" guard, and the fit is
# insensitive to its exact value (it only has to exclude s.y <= 0 and 0/0).
_BFGS_CURVATURE_FLOOR = 1e-10

# SR1 safeguard. The symmetric-rank-one update divides by (y - B s) . s, which can be arbitrarily
# close to zero even when nothing is wrong (it vanishes whenever the secant residual y - B s happens
# to be orthogonal to the step). The standard guard skips the update when
# |(y - B s) . s| < 1e-8 * |y - B s| * |s|. Not a setting -- it only has to exclude the 0/0 case.
_SR1_DENOMINATOR_FLOOR = 1e-8

# Multi-secant curvature fit. Not settings -- they are the shape of the small least-squares problem
# described in ``_multisecant_solve``:
#   _MULTISECANT_PAIRS      how many past (step, gradient change) pairs one family remembers. 4 is
#                           enough to over-determine a symmetric 3x3 (4 pairs = 12 equations for
#                           6 unknowns) without reaching back to steps taken far from the optimum.
#   _MULTISECANT_STEP_FLOOR steps shorter than this many log2 units are down-weighted, because
#                           their gradient change is dominated by the ~1e-3 float32 gradient noise.
#   _MULTISECANT_PRIOR      how hard the fit is pulled back towards the previous matrix, in the same
#                           units as one fully-trusted secant pair. 1e-2 = one hundredth of a pair,
#                           so it decides nothing in the directions the steps actually explored and
#                           everything in the directions they did not.
_MULTISECANT_PAIRS = 4
_MULTISECANT_STEP_FLOOR = 1e-3
_MULTISECANT_PRIOR = 1e-2

# Optional memory probe. ``None`` (the state a normal run is in) makes every ``_mem`` call below a
# single ``is None`` test, so an uninstrumented fit is unchanged. A benchmark driver installs a
# callable with ``set_memory_probe`` and is handed the name of every phase boundary the recipe
# already brackets with ``_sync()``; the callable reads the CUDA allocator counters itself. This is
# a measurement facility, never a setting: nothing the recipe computes depends on it.
_MEMORY_PROBE = None


def set_memory_probe(probe) -> None:
    """Install (or, with ``None``, remove) the phase-boundary memory probe. See ``_MEMORY_PROBE``."""
    global _MEMORY_PROBE
    _MEMORY_PROBE = probe


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


def _analytic_hessian_blocks(m, theta, pi_cur, species_tree, model_family_paths, skip_batches_that_do_not_fit):
    """Per-family [G,3,3] curvature via 3 analytic-HVP probes (unit theta-component directions).

    Batches are streamed one at a time. For each batch the forward solve and the adjoint point
    cache are built ONCE and shared by the 3 probes; the library's multi-batch streaming HVP
    (``make_exact_hvp`` on >1 batch) instead rebuilds both per probe, i.e. 3x the forward+backward
    work for an identical result. Probe ``j`` of batch ``b`` is scattered into the full ``[G,3]``
    column ``j`` on the batch's (disjoint) family rows, so the per-family 3x3 blocks are exact.
    Single-batch models take the same path (``forward_solve`` on a length-1 list and
    ``make_exact_hvp_single`` are what ``make_exact_hvp`` dispatched to before).

    Returns ``(H, refreshed)``: ``refreshed`` is a ``[G]`` bool mask of the families whose block was
    actually measured. A batch whose three probes do not fit in device memory -- the probes keep
    the forward state plus per-probe tangent buffers over the whole batch, several full
    ``[clades, species]`` tables more than a gradient needs, and a family larger than the batch
    budget is a batch of its own (400,918 clades on the Coleman COG3676_X family is 3 GiB per
    table, which a gradient survives on a 24 GB card and the probes do not) -- is skipped with a
    message when ``skip_batches_that_do_not_fit`` is True (the genewise fit's choice: those rows of
    ``H`` are zero and unmasked, and the fit keeps the curvature it was carrying for them) and
    re-raised when it is False (``_analytic_hessian`` below, for callers that need every block).
    The CUDA out-of-memory error is the gate itself: it is exact, where a predicted footprint for
    the probe chain would have to be kept in step with the HVP code.
    """
    G = theta.shape[0]
    dev, dtype = theta.device, theta.dtype
    rw = m.receiver_weights.detach()
    cols = [torch.zeros(G, 3, device=dev, dtype=dtype) for _ in range(3)]
    refreshed = torch.zeros(G, dtype=torch.bool, device=dev)
    for static in m.batch_statics:
        fam = static.family_index_tensor.to(dev)
        G_b = int(fam.numel())
        theta_b = theta.index_select(0, fam).contiguous()
        hvp = sv = None
        try:
            _l, sv = forward_solve([static], theta, rw)
            hvp = make_exact_hvp_single(static, theta_b, rw, sv, tangent_self_iters=pi_cur)
            for j in range(3):
                u_b = torch.zeros(G_b, 3, device=dev, dtype=dtype); u_b[:, j] = 1.0
                try:
                    out_b = hvp(u_b.reshape(-1))[: G_b * 3].reshape(G_b, 3)
                except torch.OutOfMemoryError:
                    raise
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
        except torch.OutOfMemoryError:
            if not skip_batches_that_do_not_fit:
                raise
            del hvp, sv
            hvp = sv = None
            torch.cuda.empty_cache()
            for c in cols:   # a probe may have landed before the one that ran out
                c.index_fill_(0, fam, 0.0)
            n_clades = sum(int(meta["W"]) for meta in static.wave_layout["wave_metas"])
            print(f"[fit_genewise] exact Hessian skipped for a {G_b}-family batch of {n_clades:,} clades: "
                  f"its three probes do not fit in device memory; these families keep their carried "
                  f"curvature", flush=True)
            continue
        refreshed[fam] = True
        del hvp, sv
        free_cuda_cache_if_tight()
    H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2)), refreshed


def _analytic_hessian(m, theta, pi_cur, species_tree, model_family_paths):
    """Every family's [3,3] block, or an error: ``_analytic_hessian_blocks`` with no batch skipped."""
    H, _all_measured = _analytic_hessian_blocks(m, theta, pi_cur, species_tree, model_family_paths,
                                                skip_batches_that_do_not_fit=False)
    return H


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


def _sr1_update(B, s, y, free_both):
    """Per-family symmetric-rank-one refresh of the [G,3,3] curvature matrices ``B``.

    Same inputs as ``_bfgs_update``: ``s`` is the step the family just took, ``y`` the change in its
    gradient over that step, ``free_both`` is 1.0 on the coordinates strictly inside the rate box at
    BOTH iterates and 0.0 elsewhere. Same masking too: ``s`` and ``y`` are zeroed on the pinned
    coordinates, so the correction lives entirely in the free block.

    SR1 enforces the same secant condition as BFGS (``B_new @ s = y``) with the SMALLEST possible
    correction -- a single rank-one term instead of BFGS's rank two:

        r = y - B s                      (the secant residual: how far the current B misses by)
        B_new = B + r r^T / (r . s)

    The reason to try it here is that, unlike BFGS, SR1 does NOT force the result to stay positive
    definite. Where the real curvature has a small or negative eigenvalue -- which happens on this
    surface, it is not quadratic until the last log2 unit -- BFGS can only approach it from above,
    while SR1 can reproduce it. The fit does not need a positive-definite ``B``: the Newton step
    convexifies it first (``lam = max(e, mu, |g_v|/radius)`` after an eigendecomposition), and the
    trust-region ratio test then judges the step by the NLL it actually delivered. The exact
    analytic Hessian taken at every refresh is already indefinite on some families and goes through
    that same path.

    A family is left alone (its old ``B`` kept) when ``|r . s| < 1e-8 * |r| * |s|`` -- the standard
    SR1 safeguard, which also covers ``s = 0`` (no step) and ``r = 0`` (the secant equation already
    holds, so the update would be zero anyway) -- or when the result is not finite.
    """
    s = s * free_both
    y = y * free_both
    Bs = torch.einsum("gij,gj->gi", B, s) * free_both
    r = y - Bs
    rs = (r * s).sum(dim=1)
    ok = rs.abs() > _SR1_DENOMINATOR_FLOOR * r.norm(dim=1) * s.norm(dim=1)
    safe_rs = torch.where(ok, rs, torch.ones_like(rs))[:, None, None]
    upd = (r.unsqueeze(2) * r.unsqueeze(1)) / safe_rs
    ok = ok & torch.isfinite(upd).flatten(1).all(dim=1)
    return B + torch.where(ok[:, None, None], upd, torch.zeros_like(upd))


# The six independent entries of a symmetric 3x3 matrix, in the order the multi-secant least-squares
# problem below solves for them: b = (B00, B11, B22, B01, B02, B12). The Frobenius norm of such a
# matrix is b^T diag(_MULTISECANT_FROBENIUS_WEIGHTS) b, because each off-diagonal entry appears
# twice in the matrix. Not a setting -- it is the parametrisation itself.
_MULTISECANT_FROBENIUS_WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)


def _multisecant_solve(B_prev, s_pairs, y_pairs, free_pairs, valid, prior_weight, step_floor):
    """Fit each family's symmetric 3x3 curvature to SEVERAL remembered secant pairs at once.

    BFGS and SR1 both rewrite ``B`` from the single most recent (step, gradient change) pair, and
    both then have to trust that pair completely. Near the optimum that is the weak point: the steps
    are nearly collinear, so pair after pair says the same thing about one direction and nothing
    about the other two, and the gradient change ``y`` is only a few times the ~1e-3 float32
    gradient noise. This update instead keeps the last few pairs per family and asks for the one
    symmetric matrix that fits them all in a least-squares sense, so a direction that was probed
    twice gets averaged rather than overwritten, and a direction never probed keeps its old value.

    Inputs, all indexed by family row ``g``:
      ``B_prev``     [G,3,3]  the matrix being replaced (the fit is pulled back towards it)
      ``s_pairs``    [G,K,3]  the remembered steps
      ``y_pairs``    [G,K,3]  the gradient change over each of those steps
      ``free_pairs`` [G,K,3]  1.0 on the coordinates that were free at both ends of that step
      ``valid``      [G,K]    which ring-buffer slots hold a pair at all (bool)

    What is minimised, over symmetric ``B``, for one family:

        sum_k  w_k * | free_k * (B s_k - y_k) |^2     +     prior_weight * | B - B_prev |_F^2

    with the per-pair weight

        w_k = 1 / ( |s_k| * (|s_k| + step_floor) ) .

    That weight is two things at once. The ``1/|s_k|^2`` part divides the secant equation by the
    step length, so every pair states a CURVATURE (gradient change per log2 unit of step) rather
    than a gradient change: the fit then means the same thing whether the family is taking one-log2
    strides at the start or 0.01-log2 nudges at the end, and the ``prior_weight`` below keeps a
    fixed, understandable size relative to it. The remaining factor ``|s_k| / (|s_k| + step_floor)``
    runs from 0 for a zero-length step to 1 for a step much longer than ``step_floor``, which is
    what down-weights the tiny steps whose gradient change is mostly float32 noise.

    ``free_k`` masks the RESIDUAL ROWS as well as (through the stored ``s_k``, ``y_k``, already
    zeroed on pinned coordinates) the columns, so a coordinate pinned at a rate bound contributes no
    equation and its row and column of ``B`` are decided by the prior term alone -- the same
    invariant ``_bfgs_update``'s masking gives.

    The prior term is what makes the problem always solvable. With collinear steps the secant
    equations only pin down ``B`` along one direction (3 of the 6 unknowns); ``prior_weight |B -
    B_prev|_F^2`` leaves the other three where they were instead of letting the solve pick anything.
    With no pairs at all the fit returns ``B_prev`` exactly.

    Parametrisation: a symmetric 3x3 is the 6-vector ``b = (B00, B11, B22, B01, B02, B12)``, and
    ``B s`` is linear in ``b``:

        (B s)_0 = s0*B00 + s1*B01 + s2*B02        ->  row 0 of A(s) = [s0,  0,  0, s1, s2,  0]
        (B s)_1 = s1*B11 + s0*B01 + s2*B12        ->  row 1 of A(s) = [ 0, s1,  0, s0,  0, s2]
        (B s)_2 = s2*B22 + s0*B02 + s1*B12        ->  row 2 of A(s) = [ 0,  0, s2,  0, s0, s1]

    so the whole thing is the 6x6 normal-equation system, built explicitly and solved per family:

        [ sum_k w_k A_k^T D_k A_k  +  prior_weight * W ] b  =  sum_k w_k A_k^T D_k y_k
                                                               + prior_weight * W b_prev

    where ``D_k = diag(free_k)`` (idempotent, so it appears once) and ``W =
    diag(1,1,1,2,2,2)`` turns ``b^T W b`` into the Frobenius norm. The solve is done in float64:
    it is 6x6 per family (a few hundred families, microseconds) and the data weight and the prior
    weight differ by two orders of magnitude, which float32 would resolve poorly.

    Returns the new [G,3,3] symmetric matrices, falling back to ``B_prev`` on any family whose
    solve came out non-finite.
    """
    dev = B_prev.device
    s = (s_pairs * free_pairs).to(torch.float64)
    y = (y_pairs * free_pairs).to(torch.float64)
    fr = free_pairs.to(torch.float64)
    Bp = B_prev.to(torch.float64)

    s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]
    z = torch.zeros_like(s0)
    A = torch.stack([                                          # [G,K,3,6]
        torch.stack([s0, z, z, s1, s2, z], dim=-1),
        torch.stack([z, s1, z, s0, z, s2], dim=-1),
        torch.stack([z, z, s2, z, s0, s1], dim=-1),
    ], dim=-2)

    sn = s.norm(dim=-1)                                        # [G,K] step length
    have = valid & (sn > 0)
    w = torch.where(have, 1.0 / (sn * (sn + step_floor)), torch.zeros_like(sn))
    row_w = fr * w[..., None]                                  # [G,K,3] = diag(free_k) * w_k
    At_D = A.transpose(-1, -2) * row_w[..., None, :]           # [G,K,6,3] = A_k^T D_k w_k
    M = (At_D @ A).sum(dim=1)                                  # [G,6,6]
    rhs = (At_D @ y.unsqueeze(-1)).squeeze(-1).sum(dim=1)      # [G,6]

    W = torch.tensor(_MULTISECANT_FROBENIUS_WEIGHTS, device=dev, dtype=torch.float64)
    b_prev = torch.stack([
        Bp[:, 0, 0], Bp[:, 1, 1], Bp[:, 2, 2],
        0.5 * (Bp[:, 0, 1] + Bp[:, 1, 0]),
        0.5 * (Bp[:, 0, 2] + Bp[:, 2, 0]),
        0.5 * (Bp[:, 1, 2] + Bp[:, 2, 1]),
    ], dim=-1)
    M = M + prior_weight * torch.diag(W)
    rhs = rhs + prior_weight * (W * b_prev)

    b = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)   # [G,6]
    B_new = torch.stack([
        torch.stack([b[:, 0], b[:, 3], b[:, 4]], dim=-1),
        torch.stack([b[:, 3], b[:, 1], b[:, 5]], dim=-1),
        torch.stack([b[:, 4], b[:, 5], b[:, 2]], dim=-1),
    ], dim=-2).to(B_prev.dtype)
    ok = torch.isfinite(b).all(dim=1)
    return torch.where(ok[:, None, None], B_new, B_prev)


# The four numbers of the trust-region ratio test, in the order the ``trust_test`` keyword takes
# them. This tuple IS that keyword's off value: passing it reproduces the measured recipe bit for
# bit, so every caller that is not sweeping the ratio test passes ``TRUST_TEST_OFF``.
#   shrink_factor       what the radius is multiplied by when a step delivered less than a quarter
#                       of the decrease its quadratic model predicted (0.25 = quarter the radius).
#   grow_ratio          the ratio (actual decrease / predicted decrease) above which a step that
#                       was cut to the radius doubles it.
#   radius_min          the radius never shrinks below this many log2 units: the fixed 2.0 cap of
#                       earlier rounds converged every family, so a far smaller radius only slows a
#                       family down.
#   min_predicted_bits  a family's float32 NLL carries a few 1e-4 bits of evaluation noise (atomics
#                       over ~3000-bit totals), so a step must predict at least this much gain
#                       before its actual gain is compared with the prediction, and it is undone
#                       only when the NLL rose by more than the same amount.
TRUST_TEST_OFF = (0.25, 0.75, 0.5, 0.05)

# ``targeted_hessian``'s definition of "stuck". Not settings -- they spell out the plain sentence
# "this family stopped converging": over the last ``_STUCK_LOOKBACK_PASSES`` gradient passes (two
# convergence checks at the production ``check_every`` of 2) its projected gradient did not even
# halve, which a superlinear method next to its optimum would do in one step.
_STUCK_LOOKBACK_PASSES = 4
_STUCK_CONTRACTION = 0.5
# How many past convergence checks a family remembers its |Pg| at, so that a check can always find
# one taken at least ``_STUCK_LOOKBACK_PASSES`` passes ago: with ``check_every=1`` that is four
# checks back, so five slots is the smallest ring that always holds one.
_STUCK_RING_SLOTS = _STUCK_LOOKBACK_PASSES + 1

# ln 2, the conversion between a log2-rate step and a natural-log one. Not a setting: this whole
# module works in log2 units and the rate-affine step model of ``step_model="rate_affine"`` needs
# the natural log to write "the rate multiplies by 2**d".
_LN2 = math.log(2.0)

# ``stop_nll_bits`` safety guard: a family may only stop on its model's predicted remaining NLL
# decrease while its projected gradient is already below this. Not a setting -- it exists so that a
# family whose curvature model is simply wrong far from the optimum (which can predict an
# arbitrarily small remaining gain) cannot stop there. 1e-2 is ten times the production ``tol``.
_STOP_NLL_PG_GUARD = 1e-2

# ``approach_pruning_threshold``: the Newton iteration at which the coarse approach phase ends even
# if no family has converged yet. Not a setting -- it is the point past which the traces show the
# median family is within ~1 log2 unit of its optimum, where a 20 %-wrong gradient stops helping.
_APPROACH_MAX_NEWTON_IT = 6


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
    warmup_method: str = "adam",
    em_steps: int = 2,
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
    init_curvature: str | torch.Tensor,
    curvature_update: str,
    trust: float = 2.0,
    trust_max: float,
    mu: float = 1e-4,
    fwd_tol: float = 1e-3,
    improve_frac: float = 0.8,
    verify_drop: bool = True,
    eager_defer: bool = True,
    certify: bool = False,
    certify_curvature: bool,
    init_log2_rates: tuple[float, float, float] | torch.Tensor,
    stall_patience: int,
    step_extrapolation: float,
    step_model: str,
    stop_nll_bits: float,
    approach_pruning_threshold: float,
    targeted_hessian: tuple[int, float],
    coordinate_staging: tuple[int, int],
    trust_test: tuple[float, float, float, float],
    solver_options: SolverOptions | dict | None = None,
    config: GpurecConfig | None = None,
    verbose: bool = False,
) -> dict:
    """Fit per-family DTL rates to a converged, box-bounded optimum. See module docstring for the recipe.

    Several keywords have no default and must be stated by every caller:

    ``trust_max`` -- the largest per-family trust radius (log2 units). ``trust`` is the STARTING
    radius; after every Newton step the actual decrease of the family's NLL is compared with the
    decrease its quadratic model predicted (the standard trust-region ratio): a ratio below 0.25
    quarters the radius and, if the NLL actually rose, the step is undone (the family's next step is
    recomputed from the previous point with the smaller radius; that costs nothing extra, since the
    batch evaluates every live family anyway); a ratio above 0.75 on a step that hit the radius
    doubles it, up to ``trust_max``. Measured on 200 Coleman families with a fixed 2.0 cap, only 4 %
    of steps hit the cap while the median step was 0.29 log2 units against a median distance to the
    optimum of 6.5, and paths were 1.75 times longer than the straight line: Newton under-stepped
    and zig-zagged. The adaptive radius is what lets a well-modelled family cover that distance in
    a few steps and stops a badly-modelled one from wandering.

    ``mu`` is the sign guard on the 3x3 curvature: eigenvalues below it are raised to it before the
    step. It is NOT a step-length control any more -- each eigen-direction's step is bounded by the
    family's trust radius through ``lam = max(e, mu, |g_v| / radius)`` -- so it only needs to keep
    negative or zero curvature from producing an uphill or infinite step. With the earlier value
    1e-2 a rate heading towards zero (gradient and curvature both shrinking with the rate) moved
    only gradient / 1e-2, about 0.1 to 0.2 log2 units per iteration, for 20 iterations while its
    NLL changed by 0.01 bits.

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

    ``curvature_update`` -- HOW the 3x3 is carried between exact Hessians (the refresh itself is
    unchanged: it always installs the exact analytic-HVP Hessian). Three values:

    * ``"bfgs"``  -- the production update, ``_bfgs_update``: the rank-two correction that enforces
      the secant condition ``B s = y`` from the single most recent (step, gradient change) pair
      while keeping the matrix positive definite.
    * ``"sr1"``   -- ``_sr1_update``: the rank-one correction that enforces the same condition from
      the same single pair with the smallest possible change, and does NOT keep the matrix positive
      definite (nothing downstream needs it to be -- the Newton step convexifies by eigenvalue).
    * ``"multisecant"`` -- ``_multisecant_solve``: each family remembers its last four (step,
      gradient change, free-mask) triples in a ring buffer and the matrix is re-fitted by weighted
      least squares to all of them at once, pulled back towards its previous value so that
      directions the steps never explored keep the value they had. The ring buffer is per GLOBAL
      family index, so it survives a re-plan, and it is emptied whenever the family's exact Hessian
      is taken or the family is settled (frozen, deferred or stalled).

    The same update also folds the Adam warm-up's own pairs into the starting curvature when
    ``init_curvature`` is ``"adam_bfgs"`` or a caller-supplied tensor, so one keyword selects the
    curvature machinery of the whole fit.

    ``init_curvature`` -- where the FIRST tier's starting 3x3 curvature comes from. ``"exact"``
    computes the 3-probe analytic Hessian (about 7 gradients). ``"adam_bfgs"`` instead builds it out
    of the Adam warm-up's own consecutive (step, gradient-change) pairs, which are already paid for:
    start from a Barzilai-Borwein scaled identity (``(y.y)/(s.y)`` per family, the scalar curvature
    that best fits the last pair) and fold in each pair with the same BFGS update the Newton loop
    uses. A ``[F,3,3]`` tensor is taken as the starting curvature itself (one raw, un-convexified
    matrix per family in ``gene_trees`` order -- typically the ``curvature`` a previous fit of the
    same families returned); the Adam pairs, if any, are folded into it the same way. Later tiers
    and every ``hessian_refresh`` refresh still use the exact Hessian, so this only changes how the
    first few Newton steps are aimed.

    ``init_log2_rates`` -- the starting point: a ``(log2 D, log2 L, log2 T)`` triple applied to every
    family, or a ``[F,3]`` tensor with one row per family in ``gene_trees`` order (typically the
    ``theta`` a previous fit of the same families returned). Either is clamped into the rate box.

    ``certify_curvature`` -- whether the final ``certify=True`` certificate also computes the 3-probe
    Hessian and its smallest eigenvalue. That is what fills in ``interior_pd``; with False the
    certificate still reports ``converged`` / ``unconverged`` / ``bound_active`` / ``pg_max`` and the
    total NLL, and ``interior_pd`` is simply absent from the result. NOTE: because convergence is
    certified at freeze time, ``premature_drops`` is 0 by construction -- a frozen family's reported
    ``|Pg|`` IS the one that justified freezing it. The key stays for result-shape compatibility.

    ``step_extrapolation`` -- a factor applied to the Newton step of a family whose LAST step was
    accepted well by the ratio test (actual decrease / predicted decrease > 0.75) and whose new step
    points the same way as the accepted one (cosine of the two directions > 0.9). The lengthened
    step is then cut to the trust radius and to the rate box exactly as any other step, and the
    ratio test judges it on the quadratic model's prediction AT THE APPLIED length, so a bad guess
    is undone by the usual machinery. ``1.0`` is off (today's behavior, bit for bit). After a
    rejected step the family loses the factor until it is accepted well again.

    ``step_model`` -- how the curvature and gradient are turned into a step.

    * ``"quadratic"`` -- today's behavior: the step is the (convexified, box-reduced) Newton step
      ``-Hred^-1 g``, and the predicted decrease is the quadratic model's.
    * ``"rate_affine"`` -- each coordinate's quadratic step ``delta`` is reshaped to
      ``log2(1 + ln2 * delta)``. That is the exact minimizer of a model whose GRADIENT is affine in
      the RATE (``2**theta``) rather than in the log-rate, which is what a Poisson-like count
      likelihood actually looks like: it lengthens downhill moves towards zero (``delta = -1.26``
      becomes ``-3``) and shortens runaway uphill ones (``delta = +10`` becomes ``+3``). A
      coordinate whose ``1 + ln2 * delta`` is not positive (the model wants the rate below zero)
      instead moves down by the trust radius, towards the box floor. The predicted decrease is the
      matching one, ``k ln2 d - a (2**d - 1)`` per coordinate with ``a = Hred_jj / ln2**2`` and
      ``k = a - g_j / ln2``, plus the quadratic model's off-diagonal cross terms. It is used ONLY on
      iterations whose curvature is the CARRIED one, never on the step that an exact-Hessian refresh
      just produced -- measured on 200 Coleman families, reshaping an exact-Hessian step sent
      families to the rate floor and cost 0.7 bits.

    ``stop_nll_bits`` -- an extra convergence rule, in bits, alongside ``|Pg| < tol``: a family also
    counts as converged when the quadratic model's predicted REMAINING decrease,
    ``0.5 * g_free^T Hred^-1 g_free`` with the same convexified ``Hred`` the step uses, falls below
    this many bits AND its ``|Pg|`` is already below ``_STOP_NLL_PG_GUARD``. A family's float32 NLL
    carries a few 1e-4 bits of evaluation noise, so a predicted remaining gain below that is a gain
    the run could not measure anyway. ``0.0`` is off (today's behavior, bit for bit). A family that
    stops this way is frozen at that theta and its measured ``|Pg|`` is reported as usual, so it is
    NOT counted in the certificate's ``converged`` unless its gradient also passed ``tol``; the
    result's ``n_nll_stopped`` says how many families took this exit.

    ``approach_pruning_threshold`` -- the ``adjoint_pruning_threshold`` (SolverOptions) to run the
    model at while the families are still far from their optima, i.e. until the first convergence
    check finds any family converged or Newton iteration ``_APPROACH_MAX_NEWTON_IT``, whichever
    comes first. At that point the model is re-planned at the run's real
    ``adjoint_pruning_threshold`` and the same point is re-measured. A coarser threshold drops more
    of the adjoint's small contributions, which is cheaper per gradient and inexact by a few tens of
    a percent -- fine while the step is a long way from the optimum, useless next to it. ``0.0`` is
    off, and so is any value equal to the run's own ``adjoint_pruning_threshold`` (both reproduce
    today's behavior bit for bit).

    ``targeted_hessian`` -- ``(stuck_from, stuck_max_frac)``. ``(0, 0.0)`` is off (today's behavior,
    bit for bit). The regular ``hessian_refresh`` clock spends an exact Hessian on EVERY live
    family whether or not it needs one; this rule instead spends one only on the families that
    stopped converging, and only while they are cheap. From Newton step ``stuck_from`` on, at every
    convergence check, a live family is called STUCK when its ``|Pg|`` is still above ``tol`` and
    has not fallen below ``_STUCK_CONTRACTION`` (half) of its value ``_STUCK_LOOKBACK_PASSES``
    (four) gradient passes earlier. When the stuck families own at most ``stuck_max_frac`` of the
    live model's clades -- which is what caps the price, since an exact Hessian costs about 7.7
    gradients over the families it is computed for -- and there are at least ``min_drop`` of them
    (or they are all that is still live), a temporary model is built over just them, their exact
    3-probe analytic Hessian replaces their carried 3x3, their trust radius is reset to ``trust``,
    and the temporary model is deleted. A family that was just targeted is not targeted again for
    ``hessian_refresh`` steps, and the whole rule stands down on an iteration where the regular
    refresh is already due. The result reports ``n_targeted_hessians``,
    ``targeted_hessian_families`` and ``targeted_hessian_seconds``.

    ``coordinate_staging`` -- ``(stage_freeze_T, stage_D_only)``, both counted in Newton steps after
    the Adam warm-up. ``(0, 0)`` is off (today's behavior, bit for bit). The distance to the optimum
    is dominated by the duplication rate, so it can pay to move the three rates in stages instead of
    all at once: for the first ``stage_D_only`` steps ONLY duplication moves, for the
    ``stage_freeze_T`` steps after that duplication and loss move while transfer is held, and from
    then on all three move. A held coordinate is masked exactly the way a coordinate pinned at a
    rate bound is: its gradient is out of the step AND out of the curvature update's (step, gradient
    change) pair, so the held rows and columns of the family's 3x3 keep whatever the warm-up left
    there and are simply picked up again when the stage ends -- no re-seeding.

    ``trust_test`` -- ``(shrink_factor, grow_ratio, radius_min, min_predicted_bits)``, the four
    numbers of the trust-region ratio test; see ``TRUST_TEST_OFF``, which is their measured
    production value and the off value of this keyword.

    ``warmup_method`` -- ``"adam"`` retains the existing ``adam_steps`` warm-up.
    ``"em"`` instead takes ``em_steps`` (two or three) exact box-constrained complete-history M-steps,
    including survival-conditioning counts, and seeds BFGS from the endpoint
    complete information calibrated by the measured EM gradient pair. These
    passes reuse the initial model. This method requires finite positive lower
    and upper rate bounds, ``init_curvature="adam_bfgs"`` (the EM endpoint seed
    replaces the Adam-derived seed), and ``curvature_update="bfgs"``.
    ``em_seconds`` reports its total cost.

    ``clade_budget`` -- how many clades one batch may hold, which is what sizes the transient
    [clades x species] forward / adjoint / curvature buffers and therefore the fit's peak GPU
    memory. ``None`` (the default) DERIVES it from the card: never above the tuned
    ``DEFAULT_CLADE_BUDGET`` of 315,000, and lower only when 315,000's predicted peak does not fit
    this device's memory budget. So a card with room to spare runs exactly the fit it always ran,
    and the same fit still runs on a small card with smaller batches. An explicit int uses that
    value as given, fitting or not.

    ``config`` (a top-level :class:`GpurecConfig`) threads ``config.solver`` (the same key subset as
    ``_BASE_SOLVER``) and ``config.rates`` (``min_rate``/``max_rate``) when the corresponding explicit
    kwarg is left at its signature default; an explicit kwarg always wins. ``config=None`` (the
    default) reproduces today's behavior exactly.

    IMPORTANT -- for the SOLVER, ``config`` is AUTHORITATIVE, not a partial overlay: ``config.solver``
    is taken wholesale, so passing ANY config (even one that only tweaks ``e_max_iter``) replaces this
    recipe's genewise-tuned solver defaults (such as ``e_adjoint_tol=1e-7``) with that config's
    values, which are the GLOBAL ``SolverOptions()`` defaults wherever its ``[solver]`` table is
    silent. To keep the genewise tuning and change only a few knobs, START FROM THE RECIPE FACTORY and
    modify it: ``cfg = GpurecConfig.genewise_reference(); cfg.solver.e_max_iter = 999;
    fit_genewise(..., config=cfg)``.

    The RATE BOX behaves differently, because ``RateBounds()``'s own default has no cap and this
    recipe needs one: ``config.rates`` is substituted only when the config's ``[rates]`` table was
    actually set (i.e. it differs from ``RateBounds()``). A config that leaves ``[rates]`` unset keeps
    this recipe's box (``1e-6``/``2.0``); a config that sets it wins over the preset for BOTH fields,
    so a ``[rates]`` table that names only ``min_rate`` also takes that table's ``max_rate``; and an
    explicit ``min_rate``/``max_rate`` kwarg beats both.

    NOT threaded: ``config.newton`` (this recipe's Newton step is a bespoke box-constrained
    trust-region analytic-HVP 3x3 Hessian solve, not a ``NewtonOptions`` consumer); ``config.regularizer``
    (unused -- this recipe has no regularization term). Of ``config.memory`` only
    ``scratch_tensors`` is read, as the clades x species multiplier that sizes a batch's working set
    when ``clade_budget`` is derived from the card.
    """
    if warmup_method not in ("adam", "em"):
        raise ValueError("warmup_method must be 'adam' or 'em'")
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
    # A config whose [rates] table was never set is NOT a request for the global box: it holds the
    # library defaults (floor 1e-10, no cap) only because that is what RateBounds() is, and taking
    # them here replaced the genewise cap 2.0 with None. `log2_rate_bounds` passes that None straight
    # through as `hi`, and the Newton bound test below (`th >= hi - bounds.bound_active_eps`) then
    # died with `TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'` -- which is how
    # `gpurec fit --mode genewise --config run.toml` used to crash. So only a
    # [rates] table that actually differs from the library default counts as set; when it does, BOTH
    # fields come from it (each still guarded by its own kwarg-is-still-the-preset test), so a config
    # that sets only min_rate also takes that table's max_rate.
    config_rates_set = config is not None and config.rates != _GLOBAL_RATE_BOUNDS
    if config_rates_set and min_rate == _GENEWISE_RATE_BOUNDS.min_rate:
        min_rate = config.rates.min_rate
    if config_rates_set and max_rate == _GENEWISE_RATE_BOUNDS.max_rate:
        max_rate = config.rates.max_rate
    given_curvature = init_curvature if isinstance(init_curvature, torch.Tensor) else None
    if warmup_method == "em":
        if isinstance(em_steps, bool) or not isinstance(em_steps, int) or em_steps not in (2, 3):
            raise ValueError('warmup_method="em" requires em_steps=2 or em_steps=3')
        if given_curvature is not None or init_curvature != "adam_bfgs":
            raise ValueError('warmup_method="em" requires init_curvature="adam_bfgs"; '
                             "caller-supplied and exact curvature seeds are incompatible")
        if curvature_update != "bfgs":
            raise ValueError('warmup_method="em" requires curvature_update="bfgs"')
        try:
            em_min_rate = float(min_rate)
            em_max_rate = float(max_rate)
        except (TypeError, ValueError):
            raise ValueError('warmup_method="em" requires finite positive min_rate and max_rate '
                             f"bounds, got min_rate={min_rate!r}, max_rate={max_rate!r}") from None
        if (isinstance(min_rate, bool) or isinstance(max_rate, bool)
                or not math.isfinite(em_min_rate) or em_min_rate <= 0.0
                or not math.isfinite(em_max_rate) or em_max_rate <= 0.0):
            raise ValueError('warmup_method="em" requires finite positive min_rate and max_rate '
                             f"bounds, got min_rate={min_rate!r}, max_rate={max_rate!r}")
        if em_max_rate < em_min_rate:
            raise ValueError('warmup_method="em" requires max_rate >= min_rate, '
                             f"got min_rate={min_rate!r}, max_rate={max_rate!r}")
    if given_curvature is None and init_curvature not in ("exact", "adam_bfgs"):
        raise ValueError(f'init_curvature must be "exact", "adam_bfgs" or a [F,3,3] tensor, got {init_curvature!r}')
    if curvature_update not in ("bfgs", "sr1", "multisecant"):
        raise ValueError(f'curvature_update must be "bfgs", "sr1" or "multisecant", got {curvature_update!r}')
    if step_model not in ("quadratic", "rate_affine"):
        raise ValueError(f'step_model must be "quadratic" or "rate_affine", got {step_model!r}')
    if not step_extrapolation >= 1.0:
        raise ValueError(f"step_extrapolation must be >= 1.0 (1.0 = off), got {step_extrapolation!r}")
    if not stop_nll_bits >= 0.0:
        raise ValueError(f"stop_nll_bits must be >= 0 (0.0 = off), got {stop_nll_bits!r}")
    if not approach_pruning_threshold >= 0.0:
        raise ValueError(f"approach_pruning_threshold must be >= 0 (0.0 = off), got {approach_pruning_threshold!r}")
    if len(targeted_hessian) != 2:
        raise ValueError(f"targeted_hessian must be (stuck_from, stuck_max_frac), got {targeted_hessian!r}")
    stuck_from, stuck_max_frac = int(targeted_hessian[0]), float(targeted_hessian[1])
    if stuck_from < 0 or not 0.0 <= stuck_max_frac <= 1.0:
        raise ValueError("targeted_hessian must be (stuck_from >= 0, 0 <= stuck_max_frac <= 1); "
                         f"(0, 0.0) is off, got {targeted_hessian!r}")
    if len(coordinate_staging) != 2:
        raise ValueError(f"coordinate_staging must be (stage_freeze_T, stage_D_only), got {coordinate_staging!r}")
    stage_freeze_T, stage_D_only = int(coordinate_staging[0]), int(coordinate_staging[1])
    if stage_freeze_T < 0 or stage_D_only < 0:
        raise ValueError(f"coordinate_staging counts must be >= 0 ((0, 0) is off), got {coordinate_staging!r}")
    if len(trust_test) != 4:
        raise ValueError("trust_test must be (shrink_factor, grow_ratio, radius_min, "
                         f"min_predicted_bits); {TRUST_TEST_OFF} is off, got {trust_test!r}")
    trust_shrink, trust_grow_ratio, trust_radius_min, trust_min_predicted_bits = (float(v) for v in trust_test)
    if not (0.0 < trust_shrink < 1.0 and 0.0 < trust_grow_ratio and trust_radius_min > 0.0
            and trust_min_predicted_bits > 0.0):
        raise ValueError("trust_test needs 0 < shrink_factor < 1 and positive grow_ratio, "
                         f"radius_min and min_predicted_bits; got {trust_test!r}")
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

    # Exact elimination returns the converged forward and adjoint independent of the fallback
    # iteration budgets, so additional accuracy tiers would only recompute the same result.
    pis = pis[:1]
    cert_pi = pis[0]
    neu_cert = neu_opt

    def live_is_certificate_tier(pi):
        """True when a gradient of the live model already is a certificate-tier measurement.

        The verification round exists to re-measure candidates under the certificate's solver
        fallback budgets. When the live model already uses those budgets, a candidate's live
        |Pg| is the certificate measurement and a temporary model over the candidates would only
        recompute it (up to float32 atomics order).
        """
        return pi == cert_pi and neu_opt == neu_cert

    # The adjoint pruning threshold every model is built at. It starts coarse when the caller asked
    # for an approach phase and is put back to the run's own value the moment that phase ends; when
    # no approach phase was asked for the two are the same number and every build is today's build.
    coarse_phase = (approach_pruning_threshold > 0.0
                    and approach_pruning_threshold != base["adjoint_pruning_threshold"])
    prune_now = approach_pruning_threshold if coarse_phase else base["adjoint_pruning_threshold"]

    def sopts(pi, neu):
        return SolverOptions(**{**base, "adjoint_pruning_threshold": prune_now,
                                "pi_iters": pi, "neumann_terms": neu})

    def _sync():
        """Make the wall-clock timings below honest: GPU work is queued asynchronously."""
        if dev.type == "cuda":
            torch.cuda.synchronize()

    def _mem(label):
        """Hand a phase-boundary name to the installed memory probe (a no-op when none is)."""
        if _MEMORY_PROBE is not None:
            _MEMORY_PROBE(label)

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
                           clade_budget=clade_budget)
        m.receiver_weights.requires_grad_(False)   # uniform transfer recipients (UndatedDTL default)
        return m

    gradient_work = []

    def record_gradient(m, phase):
        # Charge every resident family, including settled rows awaiting replan.
        # Hessian probes are separate work, reported by their own timing ledger.
        gradient_work.append(dict(phase=phase, families=len(m.families),
                                  clades=sum(int(f["C"]) for f in m.families)))

    def lg(m, th, phase):
        record_gradient(m, phase)
        lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
        return lv.to(dtype), g.to(dtype)

    def pgmax(th, g):
        return project_rate_gradient_(th, g.clone(), bounds=bounds).abs().amax(dim=1)

    def clamp_(th):
        clamp_log_rate_(th, bounds=bounds)
        return th

    # ``coordinate_staging``: which of (duplication, loss, transfer) this Newton step may move.
    # The schedule is applied by multiplying ``free`` -- the 1.0/0.0 mask of the coordinates not
    # pinned at a rate bound -- so a held coordinate is out of the step and out of the curvature
    # update's (step, gradient change) pair in exactly the same way a pinned one is.
    staging_on = stage_freeze_T > 0 or stage_D_only > 0
    stage_D_until = stage_D_only                    # steps 0 .. this-1: duplication only
    stage_T_until = stage_D_only + stage_freeze_T   # then up to this: transfer held
    stage_masks = None
    if staging_on:
        stage_masks = (torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=dtype),
                       torch.tensor([1.0, 1.0, 0.0], device=dev, dtype=dtype))

    def staged_free(free_rows, step_count):
        if not staging_on or step_count >= stage_T_until:
            return free_rows
        return free_rows * (stage_masks[0] if step_count < stage_D_until else stage_masks[1])

    def convexified_step(B_rows, g_rows, free_rows, r_rows):
        """The box-reduced, radius-bounded Newton step and the matrix it came from.

        Convexify AND bound each eigen-direction's step by the family's radius. A direction whose
        curvature is tiny (a rate heading towards zero: gradient and curvature both shrink with the
        rate) used to have its curvature floored at ``mu``, so its Newton step was gradient / mu --
        0.1 to 0.2 log2 units per iteration for 20 iterations on a family whose NLL moved by 0.01
        bits over all of them. Raising the curvature to |gradient component| / radius instead lets
        such a direction move the whole radius per step while every well-curved direction keeps its
        exact Newton step; ``mu`` now only guards the sign of negative curvature.
        """
        e, V = torch.linalg.eigh(B_rows)
        gv = (V.transpose(1, 2) @ (g_rows * free_rows).unsqueeze(-1)).squeeze(-1)
        r_dir = r_rows.unsqueeze(1)
        lam = torch.maximum(torch.maximum(e, torch.full_like(e, mu)), gv.abs() / r_dir)
        Hd = V @ torch.diag_embed(lam) @ V.transpose(1, 2)
        Hred = Hd * free_rows.unsqueeze(1) * free_rows.unsqueeze(2) + torch.diag_embed(1.0 - free_rows)
        delta = -torch.linalg.solve(Hred, (g_rows * free_rows).unsqueeze(-1)).squeeze(-1)
        return Hred, delta

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
    _mem("parse_families")
    if clade_budget is None and dev.type != "cuda":
        # No card to size against: the tuned batch size stands, and the per-family counts below are
        # not even read.
        clade_budget = DEFAULT_CLADE_BUDGET
    if clade_budget is None:
        # Size the batches to the card. The per-family clade and split counts come straight off the
        # parse handle (~2 s at 5123 families, 0.3 % of the fit) and are the two totals the static
        # part of the footprint scales with; the batch clade budget sizes the transient part.
        _fam_meta = parsed.families(list(range(F_all)))
        clade_budget, _budget_detail = clade_budget_for_device(
            total_clades=sum(int(f["C"]) for f in _fam_meta),
            total_splits=sum(int(f["N_splits"]) for f in _fam_meta),
            S=int(parsed.species()["S"]),
            dtype=dtype,
            device=dev,
            fixed_clade_budget=DEFAULT_CLADE_BUDGET,
            scratch_tensors=(config.memory if config is not None else MemoryOptions()).scratch_tensors,
        )
        del _fam_meta
        _gib = 1024 ** 3
        _log(f"[fit_genewise] clade_budget={clade_budget:,} "
             f"({'derived from the device' if _budget_detail['automatic'] else 'the tuned default; it fits'}): "
             f"device budget {(_budget_detail['device_budget_bytes'] or 0) / _gib:.1f} GiB, "
             f"statics {_budget_detail['static_bytes'] / _gib:.2f} GiB, "
             f"one batch {_budget_detail['working_set_bytes'] / _gib:.2f} GiB, "
             f"predicted peak {_budget_detail['predicted_peak_bytes'] / _gib:.2f} GiB")
    # Starting point for every family's [log2 D, log2 L, log2 T]. The historical start was all
    # zeros (every rate = 1.0 x speciation), which is both far from typical optima and in the
    # slow, stiff high-rate regime for the wave/E fixed points; callers pass the start explicitly.
    if isinstance(init_log2_rates, torch.Tensor):
        if tuple(init_log2_rates.shape) != (F_all, 3):
            raise ValueError(f"init_log2_rates must be [{F_all}, 3] for these families, got {tuple(init_log2_rates.shape)}")
        theta = clamp_(init_log2_rates.to(device=dev, dtype=dtype).clone().contiguous())
    else:
        theta = clamp_(torch.tensor(init_log2_rates, device=dev, dtype=dtype).reshape(1, 3).repeat(F_all, 1).contiguous())
    if given_curvature is not None and tuple(given_curvature.shape) != (F_all, 3, 3):
        raise ValueError(f"init_curvature must be [{F_all}, 3, 3] for these families, got {tuple(given_curvature.shape)}")
    active = torch.arange(F_all, device=dev)
    was_dropped = torch.zeros(F_all, dtype=torch.bool, device=dev)
    pg_last = torch.full((F_all,), float("inf"), device=dev, dtype=dtype)
    rebatch_log, defer_log = [], []
    n_steps = n_builds = n_verify_builds = n_rebuilds = n_hessians = 0
    # ``n_passes`` counts Newton gradient passes over the WHOLE fit (every ``for it`` iteration runs
    # exactly one), so it keeps counting across a pi tier boundary where ``it`` restarts at 0.
    # ``targeted_hessian``'s look-back is written in these units.
    n_passes = n_targeted = targeted_families = 0
    targeted_seconds = 0.0
    verify_seconds = rebuild_seconds = adam_seconds = 0.0
    hessian_seconds = newton_grad_seconds = certify_seconds = 0.0
    # Accurate-tier |Pg| measured at the moment a family was frozen, at the theta it keeps for the
    # rest of the fit. The certificate reuses these instead of re-running a gradient over everything.
    cert_pg = torch.full((F_all,), float("inf"), device=dev, dtype=dtype)
    cert_known = torch.zeros(F_all, dtype=torch.bool, device=dev)
    # The NLL measured in the same live pass, at the same theta: when the live tier is the
    # certificate tier (see ``live_is_certificate_tier``) it is the certificate's NLL too, and the
    # final forward over every family only re-measures it (float32 atomics order apart).
    # Kept in float64: the per-family values are float32 (~1e3-1e4 bits each, resolution ~1e-3), but
    # their sum over thousands of families (~9e6 bits on Coleman) would lose the decimals in float32.
    cert_nll = torch.full((F_all,), float("nan"), device=dev, dtype=torch.float64)
    # Best per-family NLL seen at any evaluated iterate, and the theta that produced it. A family
    # that never certifies (runs out of iterations on a knife-edge trajectory) is returned at this
    # best iterate rather than at its last one, so an unconverged tail can only lower the total NLL
    # relative to any point it visited. Certified families are untouched (their theta is final).
    best_nll = torch.full((F_all,), float("inf"), device=dev, dtype=dtype)
    best_theta = theta.clone()
    # Newton step count at which each family last improved its best NLL. A live family that has
    # not improved for ``stall_patience`` steps is going nowhere under this Newton (trust region,
    # convexified curvature): it is settled at its best iterate as unconverged instead of burning
    # iterations up to ``max_iter`` in a tail of one or two families.
    best_step = torch.zeros(F_all, dtype=torch.long, device=dev)
    # Same idea for the projected gradient: near a float32-flat optimum the NLL cannot register an
    # improvement while |Pg| still shrinks, so a family counts as stalled only when NEITHER its best
    # NLL nor its best |Pg| (by at least 10 %) improved during the last ``stall_patience`` steps.
    best_pg = torch.full((F_all,), float("inf"), device=dev, dtype=dtype)
    best_pg_step = torch.zeros(F_all, dtype=torch.long, device=dev)

    def _track_best(rows, nll_rows, theta_rows, mask, step):
        cur = best_nll.index_select(0, rows)
        better = mask & (nll_rows < cur)
        best_nll.index_copy_(0, rows, torch.where(better, nll_rows, cur))
        best_theta.index_copy_(0, rows, torch.where(
            better.unsqueeze(1), theta_rows, best_theta.index_select(0, rows)))
        best_step.index_copy_(0, rows, torch.where(better, torch.full_like(cur, step, dtype=torch.long),
                                                    best_step.index_select(0, rows)))
    # Curvature state, kept per GLOBAL family index so it survives every rebuild (a rebuild changes
    # which families are in the batch, never their theta): B_fam is the raw (un-convexified) 3x3
    # curvature matrix, and prev_* is the (theta, gradient, free-coordinate) triple of the last
    # iterate, which the BFGS update differences against the current one.
    B_fam = torch.zeros(F_all, 3, 3, device=dev, dtype=dtype)
    prev_theta = torch.zeros(F_all, 3, device=dev, dtype=dtype)
    prev_g = torch.zeros(F_all, 3, device=dev, dtype=dtype)
    prev_free = torch.zeros(F_all, 3, device=dev, dtype=dtype)
    # Multi-secant memory: the last ``_MULTISECANT_PAIRS`` (step, gradient change, free mask)
    # triples per GLOBAL family index, so they survive every re-plan exactly like ``B_fam``.
    # ``ms_pushed`` counts how many triples a family has contributed since its buffer was last
    # emptied; the slot a new triple goes into is ``ms_pushed % K``, so the oldest is overwritten.
    # Only allocated for the multisecant update -- BFGS and SR1 remember nothing.
    ms_K = _MULTISECANT_PAIRS
    ms_s = ms_y = ms_free = ms_pushed = None
    if curvature_update == "multisecant":
        ms_s = torch.zeros(F_all, ms_K, 3, device=dev, dtype=dtype)
        ms_y = torch.zeros(F_all, ms_K, 3, device=dev, dtype=dtype)
        ms_free = torch.zeros(F_all, ms_K, 3, device=dev, dtype=dtype)
        ms_pushed = torch.zeros(F_all, dtype=torch.long, device=dev)

    def _forget_secant_pairs(rows):
        """Empty these global family rows' ring buffers (exact Hessian taken, or family settled)."""
        if ms_pushed is not None and rows.numel() > 0:
            ms_pushed.index_fill_(0, rows, 0)

    def _carry_curvature(B, s, y, free_both, rows):
        """Carry the [n,3,3] curvature ``B`` of the families at global indices ``rows`` one step
        forward, from the step ``s`` they just took, the gradient change ``y`` over it and the
        coordinates ``free_both`` that were unpinned at both ends. Which formula is used is the
        caller's ``curvature_update``; see that keyword's documentation above."""
        if curvature_update == "bfgs":
            return _bfgs_update(B, s, y, free_both)
        if curvature_update == "sr1":
            return _sr1_update(B, s, y, free_both)
        # multisecant: remember this triple, then re-fit B to every triple still remembered.
        # A family that did not move contributes nothing (its "pair" carries no curvature and
        # would only push a real one out of the ring buffer).
        take = (s.norm(dim=1) > 0) & torch.isfinite(s).all(dim=1) & torch.isfinite(y).all(dim=1)
        if bool(take.any()):
            sel = take.nonzero(as_tuple=True)[0]
            g_rows = rows.index_select(0, sel)
            slot = ms_pushed.index_select(0, g_rows) % ms_K
            ms_s[g_rows, slot] = s.index_select(0, sel)
            ms_y[g_rows, slot] = y.index_select(0, sel)
            ms_free[g_rows, slot] = free_both.index_select(0, sel)
            ms_pushed.index_copy_(0, g_rows, ms_pushed.index_select(0, g_rows) + 1)
        n_pairs = ms_pushed.index_select(0, rows)
        # slots 0..K-1 fill in order, so the first min(n_pushed, K) of them hold a real triple
        filled = torch.minimum(n_pairs, torch.full_like(n_pairs, ms_K))
        valid = torch.arange(ms_K, device=dev)[None, :] < filled[:, None]
        return _multisecant_solve(
            B, ms_s.index_select(0, rows), ms_y.index_select(0, rows),
            ms_free.index_select(0, rows), valid,
            prior_weight=_MULTISECANT_PRIOR, step_floor=_MULTISECANT_STEP_FLOOR,
        )
    # Adaptive trust region, per family: the current radius, the NLL at the point the last step
    # left from, the decrease that step's quadratic model predicted (0 = no step pending a test),
    # and whether that step was cut to the radius (only a capped step earns a larger radius).
    radius = torch.full((F_all,), float(trust), device=dev, dtype=dtype)
    prev_nll = torch.zeros(F_all, device=dev, dtype=dtype)
    pred_dec = torch.zeros(F_all, device=dev, dtype=dtype)
    last_capped = torch.zeros(F_all, dtype=torch.bool, device=dev)
    # ``step_extrapolation`` state: the step each family actually applied last time, and whether the
    # ratio test then judged it well (actual / predicted > 0.75). A rejected -- or merely mediocre --
    # step clears the flag, so the factor is only ever applied on a run of confirmed-good steps.
    prev_step = torch.zeros(F_all, 3, device=dev, dtype=dtype)
    extrapolate_ok = torch.zeros(F_all, dtype=torch.bool, device=dev)
    # ``stop_nll_bits`` bookkeeping: which families left by the predicted-remaining-NLL rule rather
    # than by |Pg| < tol. Reported, never read back by the fit.
    nll_stopped = torch.zeros(F_all, dtype=torch.bool, device=dev)
    approach_end_it = None
    # ``targeted_hessian`` state, per GLOBAL family index so it survives every re-plan:
    #   stuck_pg / stuck_pg_it  a ring of the |Pg| the family measured at its last few convergence
    #                           checks and the gradient pass each was measured at, so a check can
    #                           look back exactly ``_STUCK_LOOKBACK_PASSES`` passes whatever
    #                           ``check_every`` is;
    #   stuck_pushed            how many checks the family has recorded (the slot a new one goes
    #                           into is this modulo the ring length);
    #   last_targeted           the Newton step at which the family last got a targeted exact
    #                           Hessian, so it is left alone for ``hessian_refresh`` steps after.
    targeted_on = stuck_max_frac > 0.0
    stuck_pg = stuck_pg_it = stuck_pushed = last_targeted = None
    if targeted_on:
        stuck_pg = torch.full((F_all, _STUCK_RING_SLOTS), float("inf"), device=dev, dtype=dtype)
        stuck_pg_it = torch.full((F_all, _STUCK_RING_SLOTS), -1, device=dev, dtype=torch.long)
        stuck_pushed = torch.zeros(F_all, dtype=torch.long, device=dev)
        last_targeted = torch.full((F_all,), -max_iter - hessian_refresh, device=dev, dtype=torch.long)

    em_seconds = 0.0
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
            _mem("tier_build")
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
            if pi_idx == 0 and given_curvature is not None:   # the caller already paid for this curvature
                B_fam.copy_(given_curvature.to(device=dev, dtype=dtype))
                _forget_secant_pairs(torch.arange(F_all, device=dev))
                refresh_due = False
            if pi_idx == 0 and warmup_method == "em":
                # Complete-history M-steps replace the Adam warm-up. Counts
                # come from the same reverse pass, including survival ghosts.
                # Reuse this model and its resident parse; no warm-start rebuild.
                from gpurec.fit.em_warmup import boxed_em_m_step, complete_information

                _sync(); _t = time.perf_counter()
                counts_out = torch.empty((sub.shape[0], 4), device=dev, dtype=torch.float64)
                theta_w = sub.detach().double().cpu()
                previous_theta = previous_gradient = None
                for em_step in range(em_steps):
                    record_gradient(m, "em")
                    lv_w, g_w, _ = m.genewise_loss_vector_and_grad(
                        theta=sub, need_grad=True, event_counts_out=counts_out,
                    )
                    _track_best(active, lv_w.to(dtype), sub,
                                torch.ones_like(lv_w, dtype=torch.bool), n_steps)
                    counts_w = counts_out.cpu()
                    gradient_w = g_w.detach().double().cpu()
                    next_theta = boxed_em_m_step(counts_w, lo, hi)
                    B_w = complete_information(next_theta, counts_w)
                    if previous_theta is not None:
                        s_w = theta_w - previous_theta
                        y_w = gradient_w - previous_gradient
                        Bs_w = torch.einsum("gij,gj->gi", B_w, s_w)
                        sy_w = (s_w * y_w).sum(dim=1)
                        sBs_w = (s_w * Bs_w).sum(dim=1)
                        good_w = ((sy_w > 0) & (sBs_w > 0)
                                  & torch.isfinite(sy_w) & torch.isfinite(sBs_w))
                        scale_w = torch.where(good_w, sy_w / torch.where(
                            good_w, sBs_w, torch.ones_like(sBs_w)), torch.ones_like(sy_w))
                        B_w = B_w * scale_w[:, None, None]
                        eps_w = bounds.bound_active_eps
                        free_w = (((previous_theta > lo + eps_w) & (previous_theta < hi - eps_w))
                                  & ((theta_w > lo + eps_w) & (theta_w < hi - eps_w))).to(B_w.dtype)
                        B_w = _bfgs_update(B_w, s_w, y_w, free_w)
                    previous_theta, previous_gradient = theta_w, gradient_w
                    theta_w = next_theta
                    sub = next_theta.to(device=dev, dtype=dtype).contiguous()
                    _log(f"[fit_genewise] EM warm-up {em_step + 1}/{em_steps}: "
                         f"nll={float(lv_w.double().sum()):.6f} bits")
                B_fam.index_copy_(0, active, B_w.to(device=dev, dtype=dtype))
                _forget_secant_pairs(active)
                refresh_due = False
                _sync(); em_seconds += time.perf_counter() - _t
                _mem("em_warmup")
            elif pi_idx == 0 and warmup_method == "adam" and adam_steps > 0:   # Adam warm-up (basin entry), once
                _sync(); _t = time.perf_counter()
                lf = sub.clone().requires_grad_(True)
                ad = torch.optim.Adam([lf], lr=adam_lr)
                pairs, seen = [], None
                for _ in range(adam_steps):
                    lv_a, g = lg(m, lf.detach(), "adam")
                    _track_best(active, lv_a, lf.detach(), torch.ones_like(lv_a, dtype=torch.bool), n_steps)
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
                if (given_curvature is not None or init_curvature == "adam_bfgs") and pairs:
                    if given_curvature is not None:   # fold the pairs into the caller's curvature
                        B = B_fam.index_select(0, active)
                    else:
                        # Barzilai-Borwein scaled identity from the LAST pair, then fold in every pair.
                        s_l, y_l, f_l = pairs[-1]
                        s_l, y_l = s_l * f_l, y_l * f_l
                        sy = (s_l * y_l).sum(dim=1)
                        good = sy > _BFGS_CURVATURE_FLOOR * s_l.norm(dim=1) * y_l.norm(dim=1)
                        scale = torch.where(good, (y_l * y_l).sum(dim=1) / torch.where(good, sy, torch.ones_like(sy)),
                                            torch.ones_like(sy))
                        B = scale[:, None, None] * torch.eye(3, device=dev, dtype=dtype)
                    for s_i, y_i, f_i in pairs:
                        B = _carry_curvature(B, s_i, y_i, f_i, active)
                    B_fam.index_copy_(0, active, B)
                    refresh_due = False   # the warm-up already paid for this curvature
                _sync(); adam_seconds += time.perf_counter() - _t
                _mem("adam_warmup")
            for it in range(max_iter):
                live = ~settled
                if not bool(live.any()):
                    break
                _sync(); _t = time.perf_counter()
                lv, g = lg(m, sub, "newton")
                _sync(); newton_grad_seconds += time.perf_counter() - _t
                _mem("newton_grad")
                n_passes += 1
                _track_best(active, lv, sub, live, n_steps)
                # Trust-region ratio test on the step that led here (see ``trust_max`` above).
                # Decided now, applied after the BFGS update below so the update still sees the
                # evaluated point: (step, gradient change) is valid curvature information whether or
                # not the step is kept.
                # Only a step whose predicted gain is well above the float32 noise of a family's NLL
                # (a few 1e-4 bits at ~3000 bits) can be judged; smaller steps are near convergence
                # and are left alone. A pending test is consumed here, so a re-plan that re-measures
                # the same point does not judge the same step twice.
                pred_a = pred_dec.index_select(0, active)
                pending = live & (pred_a > trust_min_predicted_bits)
                pred_dec.index_copy_(0, active, torch.zeros_like(pred_a))
                actual = prev_nll.index_select(0, active) - lv
                ratio = torch.where(pending, actual / torch.where(pending, pred_a, torch.ones_like(pred_a)),
                                    torch.ones_like(pred_a))
                r_act = radius.index_select(0, active)
                r_act = torch.where(pending & (ratio < 0.25),
                                    torch.maximum(trust_shrink * r_act,
                                                  torch.full_like(r_act, trust_radius_min)), r_act)
                r_act = torch.where(pending & (ratio > trust_grow_ratio) & last_capped.index_select(0, active),
                                    torch.minimum(2.0 * r_act, torch.full_like(r_act, trust_max)), r_act)
                radius.index_copy_(0, active, r_act)
                if step_extrapolation != 1.0:
                    # Only a step the ratio test judged WELL (the quadratic model under-promised)
                    # earns the next step a lengthening. Everything else -- a mediocre ratio, a
                    # rejected step, a step too small to judge -- clears the flag.
                    extrapolate_ok.index_copy_(0, active, pending & (ratio > trust_grow_ratio))
                reject = pending & (actual < -trust_min_predicted_bits)
                fixed = ((sub >= hi - bounds.bound_active_eps) & (g < 0)) | \
                    ((sub <= lo + bounds.bound_active_eps) & (g > 0))
                free = staged_free((~fixed).to(dtype), n_steps)

                refresh_due = refresh_due or since_exact >= hessian_refresh
                if not refresh_due:   # carry the curvature forward from the step just taken
                    both = free * prev_free.index_select(0, active)
                    B_fam.index_copy_(0, active, _carry_curvature(
                        B_fam.index_select(0, active),
                        sub - prev_theta.index_select(0, active),
                        g - prev_g.index_select(0, active), both, active))
                if bool(reject.any()):   # undo the step: back to the previous point, its gradient and NLL
                    keep = reject.unsqueeze(1)
                    sub = torch.where(keep, prev_theta.index_select(0, active), sub)
                    g = torch.where(keep, prev_g.index_select(0, active), g)
                    lv = torch.where(reject, prev_nll.index_select(0, active), lv)
                    fixed = ((sub >= hi - bounds.bound_active_eps) & (g < 0)) | \
                        ((sub <= lo + bounds.bound_active_eps) & (g > 0))
                    free = staged_free((~fixed).to(dtype), n_steps)
                prev_theta.index_copy_(0, active, sub)
                prev_g.index_copy_(0, active, g)
                prev_free.index_copy_(0, active, free)

                if it % check_every == 0:
                    pgm = pgmax(sub, g)
                    plateau = pgm >= improve_frac * pg_last.index_select(0, active)
                    pg_last.index_copy_(0, active, torch.where(live, pgm, pg_last.index_select(0, active)))
                    conv = live & (pgm < tol)
                    n_nll_stop = 0
                    if stop_nll_bits > 0.0:
                        # What the family's own model says is still on the table: the decrease the
                        # full (uncapped) Newton step would buy, 0.5 g_free^T Hred^-1 g_free, in
                        # bits. Below the float32 NLL noise there is nothing left to measure.
                        _Hq, d_q = convexified_step(B_fam.index_select(0, active), g, free,
                                                    radius.index_select(0, active))
                        remaining_bits = -0.5 * ((g * free) * d_q).sum(dim=1)
                        nll_stop = live & ~conv & (remaining_bits < stop_nll_bits) \
                            & (pgm < _STOP_NLL_PG_GUARD)
                        n_nll_stop = int(nll_stop.sum())
                        if n_nll_stop > 0:
                            nll_stopped[active[nll_stop]] = True
                        conv = conv | nll_stop
                    if coarse_phase and (bool(conv.any()) or it >= _APPROACH_MAX_NEWTON_IT):
                        # The approach is over: nothing may be frozen on a coarse gradient, so this
                        # iteration's convergence decisions are dropped, the model is re-planned at
                        # the run's real adjoint pruning threshold and the same point re-measured.
                        prune_now = base["adjoint_pruning_threshold"]
                        coarse_phase = False
                        approach_end_it = it
                        _sync(); _t = time.perf_counter()
                        del m; torch.cuda.empty_cache()
                        m = build(active.tolist(), pi_cur, neu_opt); n_builds += 1
                        clades = clade_counts(m); clade_total = float(clades.sum())
                        _sync(); rebuild_seconds += time.perf_counter() - _t
                        _mem("approach_end")
                        _log(f"  [pi{pi_cur} it{it}] coarse approach phase over "
                             f"(adjoint_pruning_threshold {approach_pruning_threshold:g} -> "
                             f"{prune_now:g}); re-measuring on the rebuilt model")
                        continue
                    cur_pg = best_pg.index_select(0, active)
                    pg_better = live & (pgm < 0.9 * cur_pg)
                    best_pg.index_copy_(0, active, torch.where(pg_better, pgm, cur_pg))
                    best_pg_step.index_copy_(0, active, torch.where(
                        pg_better, torch.full_like(cur_pg, n_steps, dtype=torch.long),
                        best_pg_step.index_select(0, active)))
                    stalled = live & ~conv \
                        & ((n_steps - best_step.index_select(0, active)) > stall_patience) \
                        & ((n_steps - best_pg_step.index_select(0, active)) > stall_patience)
                    if bool(stalled.any()):   # settle at the best iterate, reported as unconverged
                        theta.index_copy_(0, active[stalled], best_theta.index_select(0, active[stalled]))
                        _forget_secant_pairs(active[stalled])
                        settled = settled | stalled
                        live = ~settled
                        _log(f"  [pi{pi_cur} it{it}] {int(stalled.sum())} stalled families settled at their best iterate")
                        if not bool(live.any()):
                            break
                    n_conv, n_live = int(conv.sum()), int(live.sum())
                    _log(f"  [pi{pi_cur} it{it}] live={n_live} (+{int(settled.sum())} settled in batch) "
                         f"conv={n_conv} (nll_stop={n_nll_stop}) |Pg|max={float(pgm[live].max()):.2e} "
                         f"clades: live {float(clades[live].sum()) / 1e6:.2f}M of {clade_total / 1e6:.2f}M in model")
                    if targeted_on:
                        # ``targeted_hessian``. Record what every family's |Pg| is at THIS check --
                        # every check, because a check that ends in a re-plan skips the rest of this
                        # block and the look-back below has to find an entry whatever happened.
                        slot = stuck_pushed.index_select(0, active) % _STUCK_RING_SLOTS
                        stuck_pg[active, slot] = pgm
                        stuck_pg_it[active, slot] = n_passes
                        stuck_pushed.index_copy_(0, active, stuck_pushed.index_select(0, active) + 1)
                    if targeted_on and not refresh_due and n_steps >= stuck_from:
                        # A live family is STUCK when its |Pg| is still above tol and did not even
                        # halve over the last _STUCK_LOOKBACK_PASSES gradient passes. Look the
                        # comparison value up in the family's own ring of past checks: take the most
                        # recent entry that is at least that many passes old.
                        old_it = stuck_pg_it.index_select(0, active)
                        usable = (old_it >= 0) & (old_it <= n_passes - _STUCK_LOOKBACK_PASSES)
                        pick = torch.where(usable, old_it, torch.full_like(old_it, -1)).argmax(dim=1)
                        ref_pg = stuck_pg.index_select(0, active).gather(1, pick.unsqueeze(1)).squeeze(1)
                        stuck = live & usable.any(dim=1) & (pgm > tol) \
                            & (pgm >= _STUCK_CONTRACTION * ref_pg) \
                            & ((n_steps - last_targeted.index_select(0, active)) >= hessian_refresh)
                        n_stuck = int(stuck.sum())
                        # The price of the three probes is proportional to the clades they run over,
                        # so the gate is a clade share of the live model, not a family count.
                        share = float(clades[stuck].sum()) / clade_total if n_stuck else 0.0
                        _log(f"  [pi{pi_cur} it{it}] stuck={n_stuck} of {n_live} live "
                             f"({100 * share:.1f}% of the model's clades)")
                        if n_stuck > 0 and share <= stuck_max_frac \
                                and (n_stuck >= min_drop or n_stuck == n_live):
                            _sync(); _t = time.perf_counter()
                            rows_s = active[stuck]
                            mt = build(rows_s.tolist(), pi_cur, neu_opt); n_builds += 1
                            _mem("targeted_hessian_build")
                            H_s, ref_s = _analytic_hessian_blocks(
                                mt, sub[stuck], pi_cur, species_tree,
                                [fam_paths[i] for i in rows_s.tolist()],
                                skip_batches_that_do_not_fit=True,
                            )
                            got = rows_s[ref_s]
                            B_fam.index_copy_(0, got, H_s[ref_s])
                            _forget_secant_pairs(got)   # they describe a matrix that no longer exists
                            radius.index_fill_(0, got, float(trust))
                            last_targeted.index_fill_(0, got, n_steps)
                            del mt; torch.cuda.empty_cache()
                            _sync(); targeted_seconds += time.perf_counter() - _t
                            _mem("targeted_hessian")
                            n_targeted += 1; targeted_families += int(got.numel())
                            _log(f"[fit_genewise] targeted exact Hessian at pi{pi_cur} it{it}: "
                                 f"{int(got.numel())} of {n_stuck} stuck families "
                                 f"({100 * share:.1f}% of the model's clades), "
                                 f"{time.perf_counter() - _t:.1f}s")
                    if n_conv > 0 and (n_conv >= min_drop or n_conv >= drop_frac * n_live):
                        cert_ok = conv.clone()
                        if verify_drop and live_is_certificate_tier(pi_cur):
                            # This gradient is the certificate measurement (see the helper): keep
                            # the candidates' |Pg| and NLL at exactly the theta they will be frozen at.
                            cert_pg.index_copy_(0, active[conv], pgm[conv])
                            cert_nll.index_copy_(0, active[conv], lv[conv].to(torch.float64))
                        elif verify_drop:   # re-check the CANDIDATES ONLY, cold at the high tier
                            _sync(); _t = time.perf_counter()
                            cand = conv.nonzero(as_tuple=True)[0]
                            sub_c = sub.index_select(0, cand)
                            mv = build(active.index_select(0, cand).tolist(), cert_pi, neu_cert)
                            n_verify_builds += 1; n_builds += 1
                            _mem("verify_build")            # candidate model alongside the live one
                            pg_c = pgmax(sub_c, lg(mv, sub_c, "verify")[1])
                            ok_c = pg_c < tol
                            cert_pg.index_copy_(0, active.index_select(0, cand), pg_c)
                            del mv; torch.cuda.empty_cache()
                            cert_ok = torch.zeros_like(conv)
                            cert_ok.index_copy_(0, cand, ok_c)
                            _sync(); verify_seconds += time.perf_counter() - _t
                            _mem("verify_grad")
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
                            _forget_secant_pairs(active[drop])
                            if verify_drop:   # its |Pg| was just measured at exactly this theta
                                cert_known[active[drop]] = True
                            rebatch_log.append(dict(pi=pi_cur, it=it, dropped=int(drop.sum()),
                                                    remain=int((live & ~drop & ~defer).sum())))
                        if bool(defer.any()):
                            theta.index_copy_(0, active[defer], sub[defer]); carry = torch.cat([carry, active[defer]])
                            _forget_secant_pairs(active[defer])
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
                            _mem("replan")
                            _log(f"  [pi{pi_cur} it{it}] re-planned over {active.numel()} live families")
                            continue   # the gradient above belongs to the old batch; re-measure
                exact_curvature_now = refresh_due   # this step will be aimed by an exact Hessian
                if refresh_due:
                    _sync(); _t = time.perf_counter()
                    H_new, refreshed = _analytic_hessian_blocks(
                        m, sub, pi_cur, species_tree,
                        # ``static.family_indices`` are positions in THIS model's family
                        # list, so hand over the paths already in that order.
                        [fam_paths[i] for i in active.tolist()],
                        skip_batches_that_do_not_fit=True,
                    )
                    B_fam.index_copy_(0, active[refreshed], H_new[refreshed])
                    _forget_secant_pairs(active[refreshed])   # the carried pairs describe a matrix that no longer exists
                    _sync(); hessian_seconds += time.perf_counter() - _t
                    _mem("hessian")
                    refresh_due = False; since_exact = 0; n_hessians += 1
                r_step = radius.index_select(0, active).unsqueeze(1)
                Hred, delta = convexified_step(B_fam.index_select(0, active), g, free,
                                               radius.index_select(0, active))
                delta = delta * live.unsqueeze(1).to(dtype)   # settled rows keep their frozen theta
                reshaped = step_model == "rate_affine" and not exact_curvature_now
                if reshaped:
                    # The quadratic step assumes the gradient is affine in the LOG rate; a count
                    # likelihood's gradient is affine in the RATE. log2(1 + ln2 * delta) is the
                    # exact minimizer of that model, coordinate by coordinate. A coordinate whose
                    # 1 + ln2 * delta is not positive is asking for a negative rate: send it down by
                    # the trust radius instead (the box clamp below stops it at the rate floor).
                    inner = 1.0 + _LN2 * delta
                    positive = inner > 0
                    delta = torch.where(
                        positive,
                        torch.log2(torch.where(positive, inner, torch.ones_like(inner))),
                        -r_step.expand_as(inner))
                    delta = delta * live.unsqueeze(1).to(dtype)
                if step_extrapolation != 1.0:
                    # Lengthen a step that continues a well-judged one in the same direction.
                    prev_s = prev_step.index_select(0, active)
                    denom = delta.norm(dim=1) * prev_s.norm(dim=1)
                    turning = denom > 0
                    cosine = torch.where(turning, (delta * prev_s).sum(dim=1)
                                         / torch.where(turning, denom, torch.ones_like(denom)),
                                         torch.zeros_like(denom))
                    lengthen = extrapolate_ok.index_select(0, active) & (cosine > 0.9) & live
                    delta = torch.where(lengthen.unsqueeze(1), delta * step_extrapolation, delta)
                dn = delta.norm(dim=1, keepdim=True)
                capped = dn > r_step
                step = delta * torch.where(capped, r_step / torch.where(capped, dn, torch.ones_like(dn)),
                                           torch.ones_like(dn))
                new_sub = clamp_(sub + step)
                applied = new_sub - sub   # what the box left of the step
                if reshaped:
                    # The decrease the rate-affine model predicts at the step actually applied:
                    # k ln2 d - a (2**d - 1) per coordinate, with a = Hred_jj / ln2**2 the rate
                    # curvature and k = a - g_j / ln2 the linear-in-rate coefficient, plus the
                    # quadratic model's off-diagonal cross terms (the rate model is separable).
                    h_diag = torch.diagonal(Hred, dim1=1, dim2=2)
                    a_rate = h_diag / (_LN2 * _LN2)
                    k_rate = a_rate - g / _LN2
                    diagonal_gain = (k_rate * _LN2 * applied
                                     - a_rate * (torch.exp2(applied) - 1.0)).sum(dim=1)
                    quad_full = (applied.unsqueeze(1) @ Hred @ applied.unsqueeze(2)).reshape(-1)
                    quad_diag = (h_diag * applied * applied).sum(dim=1)
                    predicted = diagonal_gain - 0.5 * (quad_full - quad_diag)
                else:
                    predicted = -((g * applied).sum(dim=1)
                                  + 0.5 * (applied.unsqueeze(1) @ Hred @ applied.unsqueeze(2)).reshape(-1))
                pred_dec.index_copy_(0, active, torch.where(live, predicted, torch.zeros_like(predicted)))
                last_capped.index_copy_(0, active, capped.squeeze(1) & live)
                if step_extrapolation != 1.0:
                    prev_step.index_copy_(0, active, applied)
                prev_nll.index_copy_(0, active, lv)
                sub = new_sub
                n_steps += 1; since_exact += 1
            live = ~settled
            if bool(live.any()):   # ran out of iterations: keep each unfinished family's best iterate
                theta.index_copy_(0, active[live], best_theta.index_select(0, active[live]))
                carry = torch.cat([carry, active[live]])
            del m; torch.cuda.empty_cache()
            _mem("tier_end")
    finally:
        torch.cuda.empty_cache()

    result = dict(
        theta=theta, rates=torch.exp2(theta), n_families=F_all,
        # The last raw per-family 3x3 curvature (exact or BFGS-carried, never convexified), in
        # ``gene_trees`` order: what a later fit of the same families can start from.
        curvature=B_fam,
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
        adam_seconds=adam_seconds, em_seconds=em_seconds, warmup_method=warmup_method,
        em_steps=(em_steps if warmup_method == "em" else 0),
        gradient_work=gradient_work,
        hessian_seconds=hessian_seconds, n_hessians=n_hessians,
        newton_grad_seconds=newton_grad_seconds,
        n_verify_builds=n_verify_builds, verify_seconds=verify_seconds,
        # ``targeted_hessian``: how many targeted rounds ran, how many family blocks they actually
        # measured in total, and the seconds they cost (the temporary model's build included).
        n_targeted_hessians=n_targeted, targeted_hessian_families=targeted_families,
        targeted_hessian_seconds=targeted_seconds,
        n_rebuilds=n_rebuilds, rebuild_seconds=rebuild_seconds,
        # How many families left by the ``stop_nll_bits`` rule instead of |Pg| < tol, and the Newton
        # iteration at which the ``approach_pruning_threshold`` coarse phase ended (None = never on).
        n_nll_stopped=int(nll_stopped.sum()), approach_end_it=approach_end_it,
        certify_seconds=certify_seconds,   # overwritten below when certify=True
        history=dict(rebatch=rebatch_log, defer=defer_log),
    )
    if certify:   # final cold certificate at the high pi/Neumann tier
        try:
            _sync(); _t = time.perf_counter()
            # 1. |Pg|: reuse the freeze-time measurement (taken at this exact theta, at this exact
            #    tier) and pay a gradient ONLY for the families that were never frozen.
            pg = cert_pg.clone()
            nll_fam = cert_nll.clone()
            need = (~cert_known).nonzero(as_tuple=True)[0]
            if 0 < need.numel() < F_all:
                mneed = build(need.tolist(), cert_pi, neu_cert)
                th_n = theta.index_select(0, need)
                lv_n, g_n = lg(mneed, th_n, "certificate")
                pg.index_copy_(0, need, pgmax(th_n, g_n))
                nll_fam.index_copy_(0, need, lv_n.to(torch.float64))
                _mem("cert_unfrozen_grad")
                del mneed; torch.cuda.empty_cache()
            # 2. the headline likelihood. When every family's NLL was measured at the certificate
            #    tier already -- at freeze time by the live pass, or just above -- the total is their
            #    sum and no model over every family is built (the final forward over 23 M clades
            #    was 16 s of the Coleman fit). Otherwise ONE forward-only pass over every family
            #    gives a single consistent measurement on a single model.
            all_measured = bool(torch.isfinite(nll_fam).all()) and need.numel() < F_all
            if all_measured and not certify_curvature:
                nll_bits = float(nll_fam.sum())
                mfull = None
            else:
                mfull = build(range(F_all), cert_pi, neu_cert)
                _mem("cert_full_build")
                if need.numel() == F_all:   # nothing was ever frozen (verify_drop=False): one model does both
                    pg = pgmax(theta, lg(mfull, theta, "certificate")[1])
                with torch.no_grad():
                    nll_bits = float(mfull.genewise_loss_vector(theta=theta).sum())
                _mem("cert_nll_forward")
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
                H_cert, measured = _analytic_hessian_blocks(mfull, theta, cert_pi, species_tree, list(fam_paths),
                                                            skip_batches_that_do_not_fit=True)
                lam_min = torch.linalg.eigvalsh(H_cert)[:, 0]
                # A family whose batch could not afford the probes has no measured curvature and
                # is not counted as an interior PD optimum.
                result["interior_pd"] = int((conv & (lam_min > tol) & ~bound_active & measured).sum())
            if mfull is not None:
                del mfull; torch.cuda.empty_cache()
            _sync(); result["certify_seconds"] = time.perf_counter() - _t
            _mem("certify_end")
        finally:
            torch.cuda.empty_cache()
    return result

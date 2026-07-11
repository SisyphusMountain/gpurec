"""The single mode-aware entry point for DTL rate fitting.

There is one right recipe per parameterization, dictated by the Hessian structure of the mode --
this module is the ONLY place that choice is made, so no caller can drift onto the wrong recipe:

  - ``genewise`` (theta ``[G,3]``): the per-family Hessian is BLOCK-DIAGONAL (family f's rates affect
    only family f), so the fit decomposes into G independent 3x3 problems. ``fit_genewise`` exploits
    that -- 5 Adam warm-up steps, a per-family 3x3 trust-region Newton, convergence-based rebatching
    (drop families as they converge), and pi-tier escalation. Far faster than joint descent.
  - ``global`` (theta ``[3]``): a single shared 3-parameter box-bounded MLE -- the same 3x3 sub-problem
    as one genewise family, with the family gradients/curvature summed into one aggregate 3x3. So it
    uses the SAME recipe (``fit_global``): Adam warm-up + a 3x3 trust-region Newton on the FD Hessian.
    (Validated to reach optimize()'s optimum to <1e-4 rel, ~5x faster.)
  - ``specieswise`` (theta ``[S,3]``): the parameters are COUPLED (a species rate affects every family;
    families couple species through the transfer matrix), so the raw MLE is non-identifiable and
    boundary-saturated -- there is no well-posed one-shot fit. It is instead fit by MAP+CV: a single
    MAP prior fit via ``gpurec.fit.specieswise_fit.fit_specieswise``, with the prior strength
    cross-validated by ``gpurec.fit.map_cv.map_cv``. ``fit_dtl`` raises ``NotImplementedError`` for
    this mode and points callers at those two entry points directly.

``fit_genewise`` / ``fit_global`` / ``optimize`` are the internal engines this selects between; they
are not user-facing fit entry points. Everything that fits DTL rates (the ``gpurec fit`` CLI, the
non-regression benchmark) goes through ``fit_dtl`` so the mode->recipe mapping lives in exactly one
place.
"""
from __future__ import annotations

import time

import torch

from gpurec.fit.genewise_fit import fit_genewise
from gpurec.fit.global_fit import fit_global

_LN2 = 0.6931471805599453
_MODES = ("global", "specieswise", "genewise")


def fit_dtl(species_tree, gene_trees, mode, *, device="cuda", dtype=torch.float32,
            max_steps=300, init_rate=None, solver_options=None, verbose=False) -> dict:
    """Fit DTL rates with the best recipe for ``mode``. Returns a normalized result dict:
    ``{mode, theta[cpu], rates[cpu], nll_bits, nll_nats, n_families, wall_s, ...}`` (``gnorm`` for the
    coupled modes; ``genewise_result`` -- the full ``fit_genewise`` dict -- for genewise).

    ``init_rate`` (a rate, not log2) seeds theta for the coupled modes; ignored for genewise (which has
    its own box-projected warm start). ``solver_options`` overrides the E/adjoint solver; the default
    uses the Neumann E-adjoint, which -- unlike fp32 GMRES, whose orthogonalization residual floors
    ~1e-6 and fails mid-fit at large S -- converges to the fp32 floor in a handful of terms.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {_MODES}")
    t0 = time.perf_counter()

    if mode == "genewise":
        # fit_genewise resolves its own gene-tree spec and rebuilds tiered models internally.
        res = fit_genewise(species_tree, gene_trees, device=device, dtype=dtype,
                           certify=True, verbose=verbose)
        wall_s = time.perf_counter() - t0
        nll_bits = float(res["loss_bits"])  # cold PD-certified total NLL in bits (log2)
        return {"mode": mode, "theta": res["theta"].detach().cpu(),
                "rates": res["rates"].detach().float().cpu(),  # [G,3] order D,L,T
                "nll_bits": nll_bits, "nll_nats": nll_bits * _LN2,
                "n_families": int(res["n_families"]), "wall_s": wall_s, "genewise_result": res}

    if mode == "global":
        # single shared 3x3 block -> genewise's 3x3 TR-Newton (fit_global). Returns the normalized dict.
        return fit_global(species_tree, gene_trees, device=device, dtype=dtype,
                          init_rate=init_rate, solver_options=solver_options, verbose=verbose)

    if mode == "specieswise":
        raise NotImplementedError(
            "specieswise has no well-posed one-shot fit: the raw MLE over theta[S,3] is "
            "non-identifiable and boundary-saturated. Fit a single MAP prior with "
            "gpurec.fit.specieswise_fit.fit_specieswise(model.batch_statics, theta0, rw, lam=<chosen>), "
            "or cross-validate the prior with gpurec.fit.map_cv.map_cv(species, genes)."
        )

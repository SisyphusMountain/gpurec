# Specieswise fit recipe organization — design

**Date:** 2026-07-11
**Status:** approved (design phase)

## Motivation

`gpurec.fit.dtl_fit.fit_dtl` is the single mode-aware dispatcher: `global -> fit_global`,
`genewise -> fit_genewise`, and (currently) `specieswise -> optimize` (raw MLE). Global and genewise
have well-posed one-shot recipes. **Specieswise does not.**

The specieswise raw MLE over `theta[S,3]` is intrinsically ill-posed (documented in
`kernel-bench/newton/_specieswise_basin_findings.md`): the landscape is non-identifiable with ~half
the per-state rates saturating the 0/1 boundary (the flat near-zero Hessian eigenvalues), the "best"
checkpoints are index-1 **saddles** (`lam_min < 0`), and run-to-run NLL varies ~3 bits as the
unidentifiable rates wander. The statistically correct target is a **MAP fit with a cross-validated
Gaussian/ridge prior** (Sanderson-2002-style penalized likelihood) — which inherently requires *work*
(choosing the prior precision `lam` by cross-validation). It is therefore NOT plug-and-play.

The saddle-aware Newton machinery this needs already exists and is productionized in gpurec:
`gpurec/fit/newton_cg.py::newton_lanczos` / `newton_tr` (negative-curvature-following) and
`gpurec/solver/cg.py::cg_witness`/`steihaug_cg`/`lanczos_extremes`. The MAP+CV orchestration exists in
`gpurec/fit/map_cv.py` (parity-verified train/test data-subsetting), and a heavier auto-`lam` +
PD-certificate recipe exists in `gpurec/fit/map_fit.py`.

**Goal:** organize specieswise fitting into a clean canonical *tool* + *experiment* split, and make
`fit_dtl` stop pretending specieswise is plug-and-play.

## Design (Approach A — thin canonical layer)

Three layers, each with one responsibility:

### 1. `fit_specieswise` — the canonical tool (NEW)

New module `gpurec/fit/specieswise_fit.py`:

```python
def fit_specieswise(batch_statics, theta0, receiver_weights, *, lam, theta_ref=None,
                    adam_steps=10, adam_lr=1.0, max_newton=8, gtol=1e-2, lanczos_m=10,
                    sigma=0.01, verbose=False) -> dict
```

- A **single MAP fit at a GIVEN prior precision `lam`**. It does NOT choose `lam` and does NOT
  cross-validate — that is the experiment layer's job.
- `lam` is **required** (keyword-only, no default): the explicit "you must have chosen this" signal.
  `lam=0.0` is legal but is exactly the raw MLE the module warns against.
- `theta_ref` (prior mean for the ridge term `lam/2 * ||theta - theta_ref||^2`) defaults to `theta0`.
- **Recipe** (the recipe recovered from `kernel-bench/newton`): ~10 Adam warm-up steps (basin entry)
  → `newton_lanczos(hvp_mode="exact", lam=lam, theta_ref=theta_ref, ...)`, i.e. the saddle-aware
  Newton (CG negative-curvature witness / damping) with the MAP ridge term. At `lam > |lam_min|` the
  MAP Hessian `H + lam*I` is PD, giving the quadratic endgame the raw saddle denies.
- Operates on `batch_statics` (a built model's statics), matching the per-fold worker contract in
  `map_cv` so CV can call it fold-by-fold without a bespoke interface. One-off / benchmark callers
  build a `GeneReconModel` first and pass `model.batch_statics`.
- Returns a normalized dict `{mode: "specieswise", theta[cpu], rates[cpu], nll_bits, nll_nats,
  gnorm, lam, wall_s}`. `nll_bits` is the **data** NLL (via `final_eval`, excluding the ridge
  penalty) so it is comparable across modes and across `lam`; `gnorm` is the projected gradient of
  the **MAP** objective (data NLL + ridge) — i.e. the convergence measure of what was actually
  minimized.

Explicit negative-curvature **deflation** (`_deflate_step`) and the PD **certificate** are NOT baked
into `fit_specieswise` — at `lam > 0` the ridge + witness handle indefiniteness. They remain
experiment-side tools in `map_fit.py` for raw-landscape work.

### 2. `map_cv` — the experiment (rewire)

`gpurec/fit/map_cv.py` stays. Its k-fold-over-families CV with the lambda-homotopy and the
parity-verified train/test data-subsetting is UNCHANGED. Only its per-fold fit is rewired to call
`fit_specieswise` (saddle-aware Newton) instead of the current inline `fit_map` (L-BFGS). `map_cv`
remains where `lam*` is chosen and the final all-families refit happens.

### 3. `fit_dtl` — plug-and-play dispatcher (raise on specieswise)

`fit_dtl(mode="specieswise", ...)` **raises** (e.g. `NotImplementedError`) with a message:
> specieswise has no well-posed one-shot fit (non-identifiable raw MLE). Use
> `gpurec.fit.specieswise_fit.fit_specieswise(..., lam=<chosen>)` for a single MAP fit at a prior you
> chose, or `gpurec.fit.map_cv.map_cv(...)` to cross-validate the prior.

Only `global` and `genewise` remain plug-and-play through `fit_dtl`. `_MODES` validation still accepts
the string "specieswise" (so the error is the informative raise above, not "unknown mode").

## Benchmark consequence

The perf-golden benchmark currently fits specieswise through `fit_dtl(specieswise)` -> raw-MLE
`optimize()`, which will now raise. Replacement (decided): the specieswise perf-golden fits
**`fit_specieswise` at a fixed, committed `lam`** (deterministic — a legitimate recipe test at a
pinned prior), keeping "one golden per mode". A full `map_cv` is too expensive/noisy for a golden.

- `tests/regression/mint_goldens.py::fit_mode`: for `specieswise`, build a `GeneReconModel` and call
  `fit_specieswise(model.batch_statics, model.theta.detach(), model.receiver_weights.detach(),
  lam=SPECIESWISE_GOLDEN_LAM, ...)`; global/genewise still go through `fit_dtl`.
- Pin `SPECIESWISE_GOLDEN_LAM` (a single constant, e.g. `10.0`) next to `SIM_PARAMS`; the golden
  records it in provenance. The mint re-mints `specieswise.json` under this recipe.

## Testing

- `tests/regression/test_specieswise_recipe.py` (new):
  - fast/CPU: `fit_dtl(..., "specieswise")` raises with the pointer message.
  - `@gpu`: `fit_specieswise` on a small dataset at `lam>0` reduces `||g_MAP||` and returns finite
    `theta[S,3]`; at `lam` large the fit stays near `theta_ref` (ridge dominates) — a sanity monotone.
  - `@gpu` (optional, small): one `map_cv` run on a tiny dataset produces a finite CV curve and a
    `lam*` in the grid (guards the rewire).
- The specieswise `@slow` perf-golden (`test_perf_regression.py`) exercises `fit_specieswise` at the
  fixed `lam` end-to-end at 500 leaves.

## Non-goals

- Not rewriting `map_cv`'s CV masking / data-subsetting (parity-verified; leave it).
- Not removing `map_fit.py` or its `spectrum_min`/`_deflate_step` (experiment-side landscape tools).
- Not adding a paths-based convenience wrapper for `fit_specieswise` yet (YAGNI; callers build the
  model). Revisit only if a real one-off caller needs it.
- Not auto-choosing `lam` inside `fit_specieswise` (that would recreate the plug-and-play illusion).

## Files touched

- NEW `gpurec/fit/specieswise_fit.py` (`fit_specieswise`).
- `gpurec/fit/dtl_fit.py` (specieswise -> informative raise; docstring).
- `gpurec/fit/map_cv.py` (per-fold worker -> `fit_specieswise`).
- `tests/regression/mint_goldens.py` (specieswise fit_mode -> `fit_specieswise` at fixed `lam`;
  `SPECIESWISE_GOLDEN_LAM`).
- NEW `tests/regression/test_specieswise_recipe.py`.
- Re-mint `tests/regression/goldens/specieswise.json`.

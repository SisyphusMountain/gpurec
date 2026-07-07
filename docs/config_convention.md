# gpurec config convention

**Rule:** per-dataset configuration lives in your experiment script, never edited into a library
function body. The library ships only reference/neutral defaults.

- **Solver settings** → construct a `SolverOptions` in your script; pass it in.
- **Regularizers / priors** → build `OriginationPenalty(...)`, `tv_penalty=...`, `ridge`/`lam` in
  your script; pass them to the driver.
- **Recipe hyperparameters** → start from the driver's reference constant and override:
  `fit_genewise(sp, genes, **{**GENEWISE_REFERENCE, "tol": 5e-4})`. The reference constants
  (`GENEWISE_REFERENCE`, `OPTIMIZE_REFERENCE`, `MAP_CV_REFERENCE`) are tuned for reference problems
  (the genewise recipe / the 666x80 characterization) — they are starting points, not universal.

If you catch yourself editing a default inside `gpurec/fit/` or `gpurec/solver/` to make a dataset
work, that value belongs in your script instead.

# gpurec: `solver`/`fit` split + config convention — Design

_2026-07-07. Lean reorganization to separate the stable computation library from the volatile
per-dataset configuration._

## Goal

Adapting gpurec to a new dataset should mean **editing config in an experiment script**, not editing
library internals. Two behavior-preserving moves:

1. **Split the flat `optim/` folder** into a `solver/` (stable toolkit) and a `fit/` (drivers) folder,
   so the computation-vs-optimization layering that already holds in the import graph is visible in
   the tree.
2. **Surface the dataset-tuned defaults** (currently baked into function signatures) as labeled,
   importable **reference constants**, and document the **convention** that per-dataset overrides live
   in experiment scripts.

Nothing about the numerics changes.

## Non-goals (explicitly rejected as over-engineering)

- **No `FitRecipe` / config-object framework**, no unified `fit()` dispatcher over the drivers, no
  named-preset infrastructure beyond the reference constants below.
- **No changes to driver numerics** or to the `SolverOptions` / `OriginationPenalty` dataclasses.
- `fit_genewise`'s `pi_tiers` / `neu_opt` / `neu_cert` are a genuine **solver-precision escalation
  schedule** (coarse fit → escalate to certify), **not** duplication of `SolverOptions` — they stay.
- The per-driver `adam_steps` / `adam_lr` defaults (5 / 60 / 300) are the same name used consistently
  with workflow-appropriate values — not tangled, they stay.

## Layering (already true in imports; now reflected in folders)

Strictly downward dependencies, verified: `core` imports nothing above it; `api` does not import the
fitting layer; the toolkit does not import the drivers.

| Layer | Folder | Stability |
|---|---|---|
| Engine (kernels, solver, scheduling, params, backtracking) | `core/` (unchanged) | stable |
| Model + implicit gradient (`GeneReconModel`, `SolverOptions`) | `api/` (unchanged) | stable |
| Solver toolkit (value/grad, cg, curvature, HVP, tangents) | **`solver/`** (new) | stable |
| Fitting drivers (optimizers / fit recipes) | **`fit/`** (new) | stable algorithms |
| Per-dataset config (regularizers, priors, init, hyperparameters) | experiment scripts | **volatile** |

## Folder move mapping

`gpurec/optim/` is removed; its 16 modules distribute (mechanical `git mv`):

- → **`gpurec/solver/`** (10 — the stable toolkit): `value_and_grad`, `cg`, `curvature`,
  `hvp_exact`, `ggn`, `forward_tangent`, `receiver_curvature`, `origination_curvature`,
  `genewise_curvature`, `penalties`.
- → **`gpurec/fit/`** (6 — the drivers): `optimize`, `newton_cg`, `map_fit`, `map_cv`,
  `genewise_fit`, `baselines`.
- `optim/__init__.py`'s exports are re-homed into `solver/__init__.py` and `fit/__init__.py`.

**Out of scope (unchanged this pass):** `core/`, `api/`, and the root files `batched_lbfgs.py`,
`distributed.py`, `optimization.py` (the public façade). `batched_lbfgs.py` is a driver and *could*
move to `fit/` later, but it is reached via the `optimization` façade today, so relocating it is
extra churn we skip. No back-compat `optim/` shim — clean break (re-vendor accepted).

## Config: reference constants + convention

**Reference constants.** Each driver whose defaults encode dataset tuning gets a labeled,
module-level constant holding those tuned values; the signature defaults are sourced from it and the
docstring points at it. Behavior identical (same numbers), but the tuning is now explicit and
importable so a script can start from it and override:

- `fit/genewise_fit.py` → `GENEWISE_REFERENCE` (the "accepted optimized recipe": `adam_steps`,
  `pi_tiers`, `neu_opt`/`neu_cert`, `tol`, `trust`, `fd_eps`, `mu`, …).
- `fit/optimize.py` → `OPTIMIZE_REFERENCE` (the 666×80 characterization: `optimizer`, `lr0`,
  `schedule`, `max_steps`, `polish_mode`, `max_newton`).
- `fit/map_cv.py` → `MAP_CV_REFERENCE` (`lambdas`, `init_rate`, `adam_steps`, `lbfgs_iters`, …).

Mechanism: keep the individual keyword arguments (so call sites are unchanged), but make their
default values read from the constant (single source of truth) and add one docstring line —
"Defaults = `<CONSTANT>` (tuned for the reference problem); override per dataset." No signature
restructuring.

**Convention doc** (`docs/config_convention.md`, ~1 page): the library ships only reference/neutral
defaults; **per-dataset regularizers / priors / init / hyperparameters live in the experiment
script** (clone-and-override the reference constant, pass into the driver); nothing dataset-specific
is committed into a library function body. This is the rule that answers "which knobs live where."

## Blast radius / migration

- **Internal imports:** every `gpurec.optim.<mod>` reference rewrites to `gpurec.solver.<mod>` or
  `gpurec.fit.<mod>` per the move mapping (mechanical sweep). Includes `gpurec/__init__.py`
  (`from gpurec.optim.genewise_fit import fit_genewise` → `gpurec.fit.genewise_fit`) and
  `optimization.py` if it reaches into `optim`.
- **`tests/` and `gates/`:** update their `gpurec.optim.` imports the same way.
- **gpurax:** its `from gpurec.optim.optimize import optimize` (and any other `gpurec.optim.*`)
  rewrites; then re-vendor gpurec into `~/Documents/git/gpurax` and run its suite.

## Behavior preservation / testing

- The move + import sweep is **behavior-identical** (same code, new import paths). Gate: full suite
  green (baseline **175 passed / 3 skipped / 0 failures**, plus the known pre-existing `test_cli`
  collection clash); `pytest --collect-only` resolves; **zero dangling `gpurec.optim`** references
  (`grep -rn "gpurec\.optim" gpurec tests gates` empty).
- The reference-constant extraction keeps defaults **identical**. Gate: full suite green + a spot
  check that `GENEWISE_REFERENCE` etc. equal the prior literal defaults.
- Re-vendor gpurec into gpurax; **gpurax suite green** (53 tests).

## Branch

New branch off the current cleanup line (`refactor/genewise-dedup`, or `feat/cli-and-fidelity` once
merged). Suggested: `refactor/solver-fit-split`. Commits: (1) folder move + import sweep;
(2) reference constants + convention doc; (3) gpurax re-vendor.

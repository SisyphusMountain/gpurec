# gpurec config convention

**Rule:** per-dataset configuration lives in your experiment script (or a TOML file you pass in),
never edited into a library function body. The library ships only reference/neutral defaults.

## The hierarchy: `GpurecConfig`

`gpurec.config.GpurecConfig` is the single composition root. It is a plain dataclass of five
per-area option dataclasses, each with its own defaults:

| Field                | Type             | Defined in                     |
|----------------------|------------------|---------------------------------|
| `config.solver`      | `SolverOptions`  | `gpurec/api/solver_options.py` |
| `config.newton`      | `NewtonOptions`  | `gpurec/config/newton.py`      |
| `config.rates`       | `RateBounds`     | `gpurec/config/rates.py`       |
| `config.regularizer` | `PenaltyOptions` | `gpurec/solver/penalties.py`   |
| `config.memory`      | `MemoryOptions`  | `gpurec/config/memory.py`      |

All five are re-exported from `gpurec.config` (`from gpurec.config import GpurecConfig,
SolverOptions, NewtonOptions, RateBounds, PenaltyOptions, MemoryOptions`), so a script never needs
to know which sub-module a given option class actually lives in.

`GpurecConfig()` with no arguments constructs every sub-option at its own dataclass default --
identical to building each of the five independently. `config.validate()` delegates to each
sub-option's own `validate()` (where one exists).

## Source of truth

The dataclass defaults in the five files above are authoritative. `gpurec/config/defaults.toml` is
a hand-written TOML mirror of those same defaults, and the test
`test_defaults_toml_matches_dataclass_defaults` (`tests/test_config_toml.py`) asserts
`GpurecConfig.from_toml(DEFAULTS_TOML_PATH) == GpurecConfig()`. If you change a dataclass default,
update `defaults.toml` to match (or that test fails) -- the TOML file is a checked mirror, not an
independent source.

A field whose dataclass default is `None` (e.g. `SolverOptions.e_adjoint_tol`,
`RateBounds.max_rate`) is simply omitted from `defaults.toml` -- TOML has no null literal, so the
deep-merge leaves the dataclass's own `None` in place.

## Overriding: two ways

**1. A Python dataclass in your script** (preferred for one-off experiments):

```python
from gpurec.config import GpurecConfig, SolverOptions

config = GpurecConfig(solver=SolverOptions(pi_iters=32, e_tol=1e-7))
model = GeneReconModel(species_tree, gene_trees, mode="genewise", config=config)
```

**2. A TOML file**, loaded with `GpurecConfig.from_toml(path)` or `load_config(path)`:

```toml
# my_run.toml -- a user TOML lists ONLY the overrides it wants; everything else
# is deep-merged onto the GpurecConfig() defaults.
[solver]
pi_iters = 32
e_tol = 1e-7
```

```python
from gpurec.config import load_config

config = load_config("my_run.toml")   # load_config(None) == GpurecConfig()
```

`from_dict`/`from_toml` raise `ValueError` on any unknown key (top-level or nested) rather than
silently ignoring a typo. The CLI (`gpurec --config file.toml`, see `gpurec/cli/_common.py`)
uses the same loader.

## How it's actually wired (read before assuming more than this)

- **`config.solver` -> `GeneReconModel(config=...)`.** `GeneReconModel.__init__`
  (`gpurec/api/model.py`) takes an optional `config: GpurecConfig`; when given (and no explicit
  `solver_options=` is also passed), it pulls `config.solver` out and uses it as the model's
  `SolverOptions`. That is the *only* field of `config` this constructor reads.
- **`config.solver` -> the CLI's `--config file.toml`.** `gpurec/cli/_common.py::make_solver_options`
  loads the file via `load_config(args.config).solver`, then layers any explicitly-passed
  `--pi-iters`/`--neumann-terms`/`--e-max-iter` flag on top (flag > `--config` file > hardcoded
  `SolverOptions` default), and hands the result to `GeneReconModel(solver_options=...)`.
- **Honesty caveat -- `config.newton`/`config.rates`/`config.memory` are NOT auto-threaded into the
  fit entry points.** `NewtonOptions`, `RateBounds`, and `MemoryOptions` are each real, in-use
  defaults: the curvature solvers (`gpurec/solver/curvature.py` and its three consumers
  `genewise_curvature.py`/`origination_curvature.py`/`receiver_curvature.py`) accept a `newton:
  NewtonOptions | None` kwarg that falls back to `NewtonOptions()`; the rate optimizer and the
  genewise fit recipe use `RateBounds()`/`RateBounds.genewise()`; the memory-policy helpers
  (`gpurec/core/memory_policy.py`, `gpurec/solver/value_and_grad.py`, `gpurec/solver/hvp_exact.py`)
  use `MemoryOptions()` field values as their signature defaults. **But** `fit_genewise`,
  `optimize`, and `map_cv` (`gpurec/fit/genewise_fit.py`, `gpurec/fit/optimize.py`,
  `gpurec/fit/map_cv.py`) do not accept a `GpurecConfig`/`NewtonOptions`/`RateBounds`/
  `MemoryOptions` argument that would let a caller override those three sub-options for a fit run --
  they only consume `config.solver` indirectly (via the `*_reference()` factories below). Passing a
  custom `newton=`/`rates=`/`memory=` into a fit entry point is a noted future extension, not
  something this task implements. If you need a non-default `NewtonOptions` today, pass it directly
  to the curvature function you're calling (e.g. `newton_min(..., newton=my_newton_options)`).
- **`config.regularizer` is even less wired -- it is a passive facade consumed by no code path
  today.** The TV penalty's `eps` used at runtime is the module constant `DEFAULT_TV_EPS`
  (`gpurec/solver/penalties.py`), not `PenaltyOptions.tv_eps`; the origination-penalty call sites in
  `gpurec/solver/value_and_grad.py` are handed an `OriginationPenalty` instance directly rather than
  reading `PenaltyOptions.origination`; and the ridge hyperparameters (`lam_margin`/`lam_floor` in
  `gpurec/fit/map_fit.py`, `lambdas` in `gpurec/fit/map_cv.py`) are hardcoded function-signature
  defaults, not `PenaltyOptions` fields. Setting `regularizer.*` in a TOML today changes nothing
  observable.

## Recipe presets vs fit-hyperparameter dicts

`GpurecConfig` has three classmethod factories that reproduce the solver config baked into each
fit recipe:

- `GpurecConfig.genewise_reference()` -- `fit_genewise`'s `SolverOptions` (`e_max_iter=128,
  e_tol=1e-8, e_adjoint_tol=1e-7, ...`) + `RateBounds.genewise()` (floor `1e-6`, cap `2.0`).
- `GpurecConfig.map_cv_reference()` -- `map_cv`'s converged `SolverOptions` (`pi_iters=64,
  neumann_terms=64, ...`).
- `GpurecConfig.optimize_reference()` -- `optimize()` has no separate solver recipe, so this is
  exactly `GpurecConfig()`.

These are the single source for the numbers `gpurec/fit/genewise_fit.py::_BASE_SOLVER` and
`gpurec/fit/map_cv.py::_CV_SO` derive from (both are built from the matching factory's `.solver`
via a dict comprehension) -- the values must never be edited independently in two places.

This is a *different* layer from the fit-hyperparameter dicts `GENEWISE_REFERENCE`,
`OPTIMIZE_REFERENCE`, `MAP_CV_REFERENCE` (still living in the same three `gpurec/fit/*.py` files),
which cover recipe-level hyperparameters (`tol`, `max_steps`, `schedule`, ridge/lambda grid, ...)
that are not part of `GpurecConfig` at all. Use them as before:
`fit_genewise(sp, genes, **{**GENEWISE_REFERENCE, "tol": 5e-4})`.

## Four deliberate inconsistency resolutions

Consolidating scattered signature defaults into the dataclasses above surfaced four real
inconsistencies in the pre-config code. Each was resolved once, explicitly, rather than silently
picking one value:

1. **E-step tolerance vs tangent tolerance.** The primal E-step fixed point (`e_tol`) and the
   forward-mode JVP's E-step tangent fixed point (`e_tangent_tol`) used different tolerances
   (`1e-8` vs `1e-9`). Both are now explicit, separately-named `SolverOptions` fields -- neither
   was changed, they are just no longer implicit/undocumented.
2. **Rate floor.** The global rate floor is unified at `RateBounds()` = `min_rate=1e-10` (no cap);
   the genewise fit recipe's tighter box (`min_rate=1e-6, max_rate=2.0`) is its own named preset,
   `RateBounds.genewise()`, rather than a second copy-pasted literal.
3. **Divergent solver-signature defaults.** `_bicgstab`'s historical default (`max_iter=500`) and
   the Neumann self-loop's historical default (`3` terms) both now fall back to the single
   `SolverOptions` values (`bicgstab_max_iter=128`, `neumann_terms=64`) instead of their own
   independent literals.
4. **Dtype-tolerance helper.** The `dtype_rel_tol_default`/`dtype_rel_tol_floor` helpers (fp32 vs
   fp64 relative-residual targets) existed as near-duplicate local functions in more than one
   solver file; they are de-duplicated into `gpurec/config/gpurec_config.py` and imported from
   there.

None of these changed the observable numerics of any existing recipe -- they are the same values,
single-sourced. If a golden-output test ever disagrees with this list of four, that is a
regression, not a fifth resolution to add here.

## Non-goal: kernel launch knobs

Triton kernel launch parameters (e.g. `BLOCK_S=512/256`, `num_warps=8` in
`gpurec/core/kernels/`) are intentionally **not** part of `GpurecConfig`. They are derived
defaults tuned for the GPU/kernel, not per-dataset science knobs, and stay as local constants in
the kernel modules.

## If you catch yourself editing a library default

If you find yourself editing a default inside `gpurec/fit/`, `gpurec/solver/`, or
`gpurec/config/*.py` to make a specific dataset work, that value belongs in your script (as a
`GpurecConfig`/`SolverOptions`/... override) or a TOML file instead -- not a code edit.

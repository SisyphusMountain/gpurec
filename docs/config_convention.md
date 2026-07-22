# gpurec config convention

**Rule:** per-dataset configuration lives in your experiment script (or a TOML file you pass in),
never edited into a library function body. The library ships only reference/neutral defaults.

## The hierarchy: `GpurecConfig`

`gpurec.config.GpurecConfig` is the single composition root. It is a plain dataclass of six
per-area option dataclasses, each with its own defaults:

| Field                | Type             | Defined in                     |
|----------------------|------------------|---------------------------------|
| `config.solver`      | `SolverOptions`  | `gpurec/api/solver_options.py` |
| `config.precision`   | `PrecisionOptions` | `gpurec/config/precision.py` |
| `config.newton`      | `NewtonOptions`  | `gpurec/config/newton.py`      |
| `config.rates`       | `RateBounds`     | `gpurec/config/rates.py`       |
| `config.regularizer` | `PenaltyOptions` | `gpurec/solver/penalties.py`   |
| `config.memory`      | `MemoryOptions`  | `gpurec/config/memory.py`      |

All six are re-exported from `gpurec.config` (`from gpurec.config import GpurecConfig,
SolverOptions, PrecisionOptions, NewtonOptions, RateBounds, PenaltyOptions, MemoryOptions`), so a
script never needs to know which sub-module a given option class actually lives in.

`GpurecConfig()` with no arguments constructs every sub-option at its own dataclass default --
identical to building each of the six independently. `config.validate()` delegates to each
sub-option's own `validate()` (where one exists).

## Source of truth

The dataclass defaults in the six files above are authoritative. `gpurec/config/defaults.toml` is
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
from gpurec.config import GpurecConfig, PrecisionOptions, SolverOptions

config = GpurecConfig(
    solver=SolverOptions(pi_iters=32, e_tol=1e-7),
    precision=PrecisionOptions(model_dtype="float32", accumulator_dtype="float64"),
)
model = GeneReconModel(species_tree, gene_trees, mode="genewise", config=config)
```

**2. A TOML file**, loaded with `GpurecConfig.from_toml(path)` or `load_config(path)`:

```toml
# my_run.toml -- a user TOML lists ONLY the overrides it wants; everything else
# is deep-merged onto the GpurecConfig() defaults.
[solver]
pi_iters = 32
e_tol = 1e-7

[precision]
model_dtype = "float32"
accumulator_dtype = "float64"
```

```python
from gpurec.config import load_config

config = load_config("my_run.toml")   # load_config(None) == GpurecConfig()
```

`from_dict`/`from_toml` raise `ValueError` on any unknown key (top-level or nested) rather than
silently ignoring a typo. The CLI (`gpurec --config file.toml`, see `gpurec/cli/_common.py`)
uses the same loader.

## How it's actually wired (read before assuming more than this)

- **`config.solver` and `config.precision` -> `GeneReconModel(config=...)`.**
  `GeneReconModel.__init__` (`gpurec/api/model.py`) reads both sections. An explicit
  `solver_options=` overrides `config.solver`, and an explicit `dtype=` overrides
  `config.precision.model_dtype`. `config.precision.accumulator_dtype` still applies when
  `dtype=` overrides the model dtype and must be at least as wide as that effective dtype.
- **Precision responsibilities.** `model_dtype` controls parameters and dense E/Pi residual
  state and dense-kernel wave metadata. `accumulator_dtype` controls centered row offsets, the
  likelihood head and streamed reductions, small parameter softmaxes, and accumulator-domain
  preprocessing statics. Rust floating outputs remain f64 until Python materializes each tensor
  directly in its owning domain's dtype; dense wave metadata keeps directly constructed fp32 and
  fp64 variants for runtime dtype changes. Both accept
  `"float32"` and `"float64"`. The supported pairs are `float32/float32`, `float32/float64`, and
  `float64/float64` (model/accumulator); `float64/float32` is rejected because an accumulator may
  be wider than model state but never narrower. This policy covers model execution and its
  centered derivative paths; legacy outer optimization-vector policies are unchanged.
- **CLI precedence.** For model precision, explicit `--dtype` >
  `[precision].model_dtype` in `--config` > the `PrecisionOptions` default. There is no separate
  accumulator flag: set `[precision].accumulator_dtype` in TOML or pass a `GpurecConfig` from
  Python. For solver fields, an explicitly passed `--pi-iters`/`--neumann-terms`/`--e-max-iter`
  flag > the matching `[solver]` value > the `SolverOptions` default.
- **Fit and rebuild paths retain precision.** `fit_dtl`, `fit_genewise`, and `fit_global`
  pass their `config` through model construction and internal rebuilds. An explicit
  Python `dtype=` remains the model-dtype override; the configured accumulator policy is retained.
- **Honesty caveat -- `config.newton`/`config.rates`/`config.memory` are not universally threaded
  into fit entry points.** `NewtonOptions`, `RateBounds`, and `MemoryOptions` are each real, in-use
  defaults: the curvature solvers (`gpurec/solver/curvature.py` and its three consumers
  `genewise_curvature.py`/`origination_curvature.py`/`receiver_curvature.py`) accept a `newton:
  NewtonOptions | None` kwarg that falls back to `NewtonOptions()`; the rate optimizer and the
  genewise fit recipe use `RateBounds()`/`RateBounds.genewise()`; the memory-policy helpers
  (`gpurec/core/memory_policy.py`, `gpurec/solver/value_and_grad.py`, `gpurec/solver/hvp_exact.py`)
  use `MemoryOptions()` field values as their signature defaults. `fit_genewise` also consumes its
  documented `config.rates` fields. The entry points above accept `GpurecConfig` for
  solver/precision propagation, and individual recipes may consume another documented section,
  but passing a
  custom `newton=`/`rates=`/`memory=` into a fit entry point is a noted future extension, not
  something this task implements. If you need a non-default `NewtonOptions` today, pass it directly
  to the curvature function you're calling (e.g. `newton_min(..., newton=my_newton_options)`).
- **`config.regularizer` is only partly wired.** `map_cv` consumes
  `config.regularizer.lambdas` when an explicit `lambdas=` override is absent. The TV penalty's
  `eps` used at runtime is the module constant `DEFAULT_TV_EPS`
  (`gpurec/solver/penalties.py`), not `PenaltyOptions.tv_eps`; the origination-penalty call sites in
  `gpurec/solver/value_and_grad.py` are handed an `OriginationPenalty` instance directly rather than
  reading `PenaltyOptions.origination`; and the remaining ridge hyperparameters
  (`lam_margin`/`lam_floor` in `gpurec/fit/map_fit.py`) are function-signature defaults rather than
  automatically consuming their `PenaltyOptions` fields.

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

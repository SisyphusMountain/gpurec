# gpurec CLI

Installed as the `gpurec` console script (`pip install -e .`). Importing `gpurec` (hence
the CLI, even for `--help`) pulls in the Triton/CUDA stack via the package `__init__`, so
Triton must be installed to run any subcommand; the actual GPU forward/optimize work
happens when a subcommand executes.

## reconcile — log-likelihood at fixed rates

```
gpurec reconcile --species sp.nwk --gene g.nwk [g2.nwk ...] \
  --delta 0.1 --tau 0.05 --lambda 0.1 [--mode global|genewise] \
  [--device cuda] [--config run.toml] [--dtype float32|float64] \
  [--forward-self-loop exact|log|linear] [--adjoint-self-loop exact|series] \
  [--pi-iters N] [--neumann-terms N] [--e-max-iter N] \
  [--out per_fam.txt]
```
Prints `<family> <logL_nats>`. `global` mode prints one total line; `genewise` prints one
line per family. `--out` writes the AleRax `per_fam_likelihoods.txt` format.

**Units:** log-likelihood is reported in **nats** (AleRax-comparable). Internally the model
returns NLL in bits; the CLI converts (`logL_nats = -loss_bits * ln 2`).

**Rate order:** `--delta/--tau/--lambda` are D/T/L; `theta = [log₂ D, log₂ L, log₂ T]` (theta[2] = transfer; same column order as AleRax `D L T`).

## fit — optimize rates

```
gpurec fit --species sp.nwk --gene DIR_OR_GLOB \
  --mode global|specieswise|genewise [--steps 300] [--init-rate 0.1] \
  [--device cuda] [--config run.toml] [--dtype float32|float64] \
  [--forward-self-loop exact|log|linear] [--adjoint-self-loop exact|series] \
  [--pi-iters N] [--neumann-terms N] [--e-max-iter N] \
  [--out rates.txt]
```
`global` uses `fit_global`; `genewise` uses `fit_genewise`. A one-shot
`specieswise` fit is not supported and reports guidance to the MAP/CV entry
points. `--out`: for `global`, writes fitted rates (AleRax `# node D L T` order)
to `<out>` plus a `<out>.json` sidecar (`nll_bits`, `nll_nats`, `elapsed_s`,
`mode`, `n_families`). For `genewise`, writes ONLY `<out>.json` (schema:
`theta_log2`, `rates`, plus the same nll/timing fields) — no `<out>` rates file.

## Solver selection (`--forward-self-loop`, `--adjoint-self-loop`)

Inside each wave, every clade row obeys a small fixed-point equation — the row depends
on itself through "transfer out, then the donor lineage is lost" and "duplicate, then one
copy is lost". That is the **self-loop**, and these two flags choose how it is solved.

| flag | value | what it does |
|---|---|---|
| `--forward-self-loop` | `exact` (default) | Solves the fixed point outright. Every entry is a likelihood, so the equation is a linear system on the species tree and is eliminated in a fixed number of passes. |
| | `log` | Iterates it in log2 space, `--pi-iters` times. The reference implementation. |
| | `linear` | Iterates it in scaled linear space with an early exit. Holds one scale per row, so it cannot represent a row spanning more than ~126 binary orders in float32. |
| `--adjoint-self-loop` | `exact` (default) | Solves the transposed system outright. The Hessian-probe tangent follows this setting. |
| | `series` | Sums up to `--neumann-terms` Neumann terms, stopping early once a term can no longer move the result. |

**`--pi-iters` and `--neumann-terms` only apply to the iterated modes.** They are
iteration *counts*. Under the default exact solves the answer is the converged fixed
point no matter how many iterations are requested, so passing them changes nothing —
they are accepted so that a command line can select an iterated mode and its iteration
count together. `--e-max-iter` is unrelated to the self-loop: it caps the resident `E`
(survival) fixed-point solve and applies in every mode.

The exact solves are the library-wide defaults, so these flags are only needed to
reproduce the older iterated behaviour or to compare the two. On the toy fixtures under
`tests/data/alerax`, `reconcile` in float64 gives **bit-identical** per-family
log-likelihoods either way, and both land within 9.1e-13 nats of the AleRax reference;
in float32 the two paths differ by at most 2.8e-5 nats (on the 2336-leaf
`test_mixed_200`, whose log-likelihood is about -6221 nats).

## Precision configuration

With no `--dtype`, the model dtype comes from
`[precision].model_dtype` in `--config`; with no config it defaults to
`"float32"`. An explicit `--dtype` overrides that field. The accumulator dtype
has no separate CLI flag and always comes from
`[precision].accumulator_dtype` (default `"float64"`):

```toml
[precision]
model_dtype = "float32"
accumulator_dtype = "float64"
```

`model_dtype` controls parameters and dense E/Pi residual state.
`accumulator_dtype` controls centered row offsets, likelihood heads and
streamed reductions, small parameter softmaxes, and floating preprocessing
statics. Supported model/accumulator pairs are `float32/float32`,
`float32/float64`, and `float64/float64`; `float64/float32` is rejected. Thus
`--dtype float64` with the default accumulator gives a fully float64 model,
while the default `float32/float64` policy keeps dense kernels in fp32 and the
small numerically sensitive components in fp64.

On the toy AleRax fixtures, the float64/float64 likelihood matches
AleRax_fixed to ≤3e-12 nats (machine round-off), versus approximately 2.9e-5
in the historical fp32 comparison; see
`tests/test_fidelity_alerax.py::test_fidelity_float64_reaches_machine_precision`.

## `--config` and genewise rate bounds

`gpurec fit --mode genewise` fits per-family rates inside a box: rates may not fall below
`min_rate` (1e-6) nor rise above `max_rate` (2.0). That box is `fit_genewise`'s own preset,
and it is deliberately tighter than the library-wide `RateBounds()` default, which has a
floor of 1e-10 and **no** cap at all.

A `--config` file only replaces that box when it actually contains a `[rates]` table:

```toml
[rates]
min_rate = 1e-6
max_rate = 2.0
```

A config that leaves `[rates]` out keeps the genewise preset, so `gpurec fit --mode genewise
--config run.toml` works whether or not the file mentions rates. When a `[rates]` table *is*
present it wins as a whole, not field by field: a table that names only `min_rate` also
imposes that table's `max_rate` (i.e. no cap), because `[rates]` is read as one box. If you
want a floor without losing the cap, write both keys.

(Before 2026-09-04 the preset was replaced by any config at all. A config without `[rates]`
therefore imposed the capless global box; `log2_rate_bounds` passed the missing cap through as
`None`, and the fit's first Newton bound test killed the run with
`TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'`.)

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
  [--out rates.txt]
```
`global` uses `fit_global`; `genewise` uses `fit_genewise`. A one-shot
`specieswise` fit is not supported and reports guidance to the MAP/CV entry
points. `--out`: for `global`, writes fitted rates (AleRax `# node D L T` order)
to `<out>` plus a `<out>.json` sidecar (`nll_bits`, `nll_nats`, `elapsed_s`,
`mode`, `n_families`). For `genewise`, writes ONLY `<out>.json` (schema:
`theta_log2`, `rates`, plus the same nll/timing fields) — no `<out>` rates file.

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

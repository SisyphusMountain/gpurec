# gpurec CLI

Installed as the `gpurec` console script (`pip install -e .`). Importing `gpurec` (hence
the CLI, even for `--help`) pulls in the Triton/CUDA stack via the package `__init__`, so
Triton must be installed to run any subcommand; the actual GPU forward/optimize work
happens when a subcommand executes.

## reconcile — log-likelihood at fixed rates
```
gpurec reconcile --species sp.nwk --gene g.nwk [g2.nwk ...] \
  --delta 0.1 --tau 0.05 --lambda 0.1 [--mode global|genewise] \
  [--device cuda] [--dtype float64] [--out per_fam.txt]
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
  [--device cuda] [--dtype float64] [--out rates.txt]
```
`global`/`specieswise` use `gpurec.optim.optimize` + `final_eval`; `genewise` uses
`fit_genewise`. `--out`: for `global`/`specieswise`, writes fitted rates (AleRax
`# node D L T` order) to `<out>` plus a `<out>.json` sidecar (`nll_bits`, `nll_nats`,
`elapsed_s`, `mode`, `n_families`). For `genewise`, writes ONLY `<out>.json` (schema:
`theta_log2`, `rates`, plus the same nll/timing fields) — no `<out>` rates file.
Specieswise rows are labelled `s{i}` by species index (mapping to AleRax node labels is
the scale kit's job).

**Note:** `--dtype float64` sets the model's compute dtype (`theta`, `receiver_weights`,
`origination_weights` are built as float64 `nn.Parameter`s; `batch_statics` stay float32
and the forward upcasts to the parameter dtype). This means `reconcile` and the
likelihood evaluation run in float64, and `fit --mode genewise` is fully float64.
**Nuance:** the first-order optimizer used by `fit --mode global/specieswise`
(`gpurec.optim.optimize.first_order`) casts to float32 internally by design, so those fits
still optimize in float32 — `--dtype float64` does not change their optimization
trajectory, only the reported final NLL, which is evaluated (and reported) in float64.

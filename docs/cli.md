# gpurec CLI

Installed as the `gpurec` console script (`pip install -e .`). Requires CUDA + Triton
for the forward path (`--help` and argument parsing do not).

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
`fit_genewise`. `--out` writes fitted rates (AleRax `# node D L T` order) plus a
`<out>.json` sidecar (`nll_bits`, `nll_nats`, `elapsed_s`, `mode`, `n_families`).
Specieswise rows are labelled `s{i}` by species index (mapping to AleRax node labels is
the scale kit's job).

# HOGENOM Fast Optimization Results

Current best gradient-based run:

- rates: `model_rates_gradient_adam_line_search.csv`
- validation: `fixed_pass_validation_gradient_adam_line_search.csv`
- families: 1055
- optimization evaluator: fixed 20 Pi passes, adaptive E to `1e-10`
- reference evaluator: fixed 160 Pi passes, adaptive E to `1e-10`
- gradient adjoint: `neumann_terms=32`
- optimizer: Adam-style direction with per-family line search along the full
  gradient direction
- fixed20 summed NLL: 606570.9375 bits
- fixed160 summed NLL: 606570.8125 bits
- worst per-family `abs(fixed20 - fixed160)`: 0.0576171875 bits
- families above 0.1 bits: 0

Reproducible command:

```bash
python profiling/optimize_hogenom_gradient.py \
  --dataset tests/data/hogenom_bench \
  --init-csv tests/data/hogenom_bench/gpurec/model_rates.csv \
  --fixed-pi 20 \
  --reference-pi 160 \
  --neumann-terms 32 \
  --stages 8:0.125,8:0.0625
```

The main lesson is that low `neumann_terms` gave poor optimization directions.
With `neumann_terms=4`, gradient-based methods stalled well above the coordinate
baseline. Increasing to 8 or more fixed the direction quality; the saved run uses
32 terms.

Previous coordinate-search baseline:

- rates: `model_rates_fast_coord_refined_armijo2_fixed20.csv`
- validation: `fixed_pass_validation_best.csv`
- families: 1055
- optimization evaluator: fixed 20 Pi passes, adaptive E to `1e-10`
- reference evaluator: fixed 160 Pi passes, adaptive E to `1e-10`
- fixed20 summed NLL: 606592.75 bits
- fixed160 summed NLL: 606592.6875 bits
- worst per-family `abs(fixed20 - fixed160)`: 0.0218505859375 bits
- families above 0.1 bits: 0

Timing on the RTX 4090 in this workspace:

- coarse coordinate search: 72.55s, 43 no-grad fixed20 evaluations, summed NLL 606598.0625
- fine coordinate refinement: 41.46s, 25 no-grad fixed20 evaluations, summed NLL 606595.4375
- two Armijo BFGS polish steps: 49.45s, 3 gradient evaluations and 24 no-grad probes, summed NLL 606592.75

The coordinate search was useful as a baseline, but it does not scale with
parameter dimension because each coordinate requires separate probes.

Reproducible command for the coordinate baseline:

```bash
python profiling/optimize_hogenom_fast.py \
  --dataset tests/data/hogenom_bench \
  --init-csv tests/data/hogenom_bench/gpurec/model_rates.csv \
  --fixed-pi 20 \
  --reference-pi 160 \
  --polish-armijo-iters 2
```

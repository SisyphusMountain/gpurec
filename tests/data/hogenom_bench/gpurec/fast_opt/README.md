# HOGENOM Fast Optimization Results

Current best run:

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

The coordinate search is the main speed win: each no-grad fixed20 evaluation of all
1055 families is much cheaper than a full backward pass, and all families can accept
or reject coordinate moves independently.

Reproducible command for the same style of run:

```bash
python profiling/optimize_hogenom_fast.py \
  --dataset tests/data/hogenom_bench \
  --init-csv tests/data/hogenom_bench/gpurec/model_rates.csv \
  --fixed-pi 20 \
  --reference-pi 160 \
  --polish-armijo-iters 2
```

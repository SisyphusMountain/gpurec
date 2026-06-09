# HOGENOM Specieswise L2 Penalty CV Pilot

## Scope

This is a bounded pilot, not the full 1055-family production optimization.  It uses the local HOGENOM fixture under `tests/data` with 80 deterministically sampled families from 1055 available families.

## Data Split

- Seed: `20260609`
- Train families: `64`
- Validation families: `16`
- Split method: shuffle sorted `*.trees` with Python `random.Random(seed)`, then take validation first and training second from the bounded sample.
- Species tree: `/home/enzo/Documents/git/gpurec/gpurec/tests/data/hogenom_S.tree`
- Gene tree directory: `/home/enzo/Documents/git/gpurec/gpurec/tests/data/hogenom_trees`

## Penalty Grid

L2 penalty on specieswise log2-rate parameters:

```text
objective = train_raw_nll_bits + 0.5 * lambda * sum((theta - log2(0.05))^2)
```

Grid: `[0.01, 0.1, 1.0, 10.0]`

## Optimizer Settings

- Mode: `specieswise`
- Optimizer: `Adam`
- Steps per penalty: `40`
- Learning rate: `0.04`
- Gradient clipping: `500.0`
- Rate bounds: `[1e-10, 2.0]`
- Initial D/T/L rates: `0.05`
- Solver: `gmres`
- Forward Pi solver: `gmres`
- Forward Pi iterations: `16`
- Backward/self-loop iterations: `16`
- Device: `cuda`

## Validation Objective

The validation objective is raw held-out negative log likelihood in bits, evaluated on the validation families after copying the trained specieswise theta into a separate validation model.  The L2 penalty is not included in validation.

## Results

| lambda | val NLL/family | train NLL/family | rate min | rate max | theta min | theta max | stable | runtime s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 0.01 | 442.984314 | 474.974792 | 1.420e-02 | 1.853e-01 | -6.138 | -2.432 | True | 21.78 |
| 0.1 | 443.563629 | 476.138733 | 1.585e-02 | 1.825e-01 | -5.979 | -2.454 | True | 21.09 |
| 1 | 451.304504 | 487.792053 | 2.554e-02 | 1.652e-01 | -5.291 | -2.597 | True | 21.06 |
| 10 | 467.233582 | 521.611938 | 4.391e-02 | 1.381e-01 | -4.509 | -2.856 | True | 20.88 |

## Selected Penalty Recommendation

Recommend `lambda=0.01` for the next larger specieswise run from this pilot.  It had the best stable held-out raw NLL per validation family: `442.984314` bits/family.

## Theta And Rate Stability

All trials were considered stable when theta stayed finite and natural rates stayed within the configured bounds.  Final selected stats:

```json
{
  "step": 40,
  "train_raw_nll_bits": 30398.38671875,
  "train_penalty_bits": 39.102664947509766,
  "train_objective_bits": 30437.490234375,
  "val_raw_nll_bits": 7087.7490234375,
  "grad_norm_before_projection": 83.47476196289062,
  "projected_grad_norm": 83.47476196289062,
  "clipped_grad_norm": 83.47476196289062,
  "step_s": 0.46345869405195117,
  "val_s": 0.05555405397899449,
  "theta_min": -6.138038158416748,
  "theta_max": -2.4316906929016113,
  "theta_mean": -4.178032398223877,
  "theta_std": 1.3952504396438599,
  "rate_min": 0.014199282042682171,
  "rate_max": 0.18534810841083527,
  "rate_mean": 0.08257447928190231,
  "active_lower": 0,
  "active_upper": 0,
  "param_count": 3975,
  "finite": true
}
```

## Runtime

- Total elapsed: `85.19` s
- Output JSON: `benchmarks/large_dataset_capacity/output/hogenom_penalty_cv_pilot/20260609_013624/results.json`

## Commands Run

```bash
python /home/enzo/Documents/git/gpurec/gpurec/benchmarks/large_dataset_capacity/hogenom_penalty_cv_pilot.py --max-families 80 --val-families 16 --penalties 0.01,0.1,1.0,10.0 --steps 40 --lr 0.04 --pi-solver gmres --self-loop-solver gmres --output-root benchmarks/large_dataset_capacity/output/hogenom_penalty_cv_pilot --report benchmarks/large_dataset_capacity/reports/hogenom_penalty_cv_long_stability.md
```

## Blockers

- Full 1055-family specieswise CV was not run in this bounded pilot because multiplying the full dataset by the penalty grid would be substantially more expensive.
- No GBM/tree prior was exercised; no ready script-level GBM prior hook was found during the bounded inspection, so this pilot focused on L2 as requested.
- This script uses the current worktree implementation of GMRES; it does not modify or independently validate core GMRES internals.

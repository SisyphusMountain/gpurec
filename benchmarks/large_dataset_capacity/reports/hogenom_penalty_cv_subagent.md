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

Grid: `[0.0, 0.001, 0.01, 0.1, 1.0]`

## Optimizer Settings

- Mode: `specieswise`
- Optimizer: `Adam`
- Steps per penalty: `12`
- Learning rate: `0.08`
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
| 0 | 452.831482 | 496.644806 | 2.532e-02 | 9.878e-02 | -5.304 | -3.340 | True | 9.20 |
| 0.001 | 452.829651 | 496.645905 | 2.532e-02 | 9.878e-02 | -5.304 | -3.340 | True | 8.53 |
| 0.01 | 452.822113 | 496.656952 | 2.532e-02 | 9.877e-02 | -5.304 | -3.340 | True | 8.51 |
| 0.1 | 453.002106 | 496.911896 | 2.540e-02 | 9.876e-02 | -5.299 | -3.340 | True | 8.54 |
| 1 | 456.483521 | 501.152283 | 2.730e-02 | 9.797e-02 | -5.195 | -3.351 | True | 8.54 |

## Selected Penalty Recommendation

Recommend `lambda=0.01` for the next larger specieswise run from this pilot.  It had the best stable held-out raw NLL per validation family: `452.822113` bits/family.

## Theta And Rate Stability

All trials were considered stable when theta stayed finite and natural rates stayed within the configured bounds.  Final selected stats:

```json
{
  "step": 12,
  "train_raw_nll_bits": 31786.044921875,
  "train_penalty_bits": 16.575456619262695,
  "train_objective_bits": 31802.62109375,
  "val_raw_nll_bits": 7245.15380859375,
  "grad_norm_before_projection": 91.36837768554688,
  "projected_grad_norm": 91.36837768554688,
  "clipped_grad_norm": 91.36837768554688,
  "step_s": 0.5508644841611385,
  "val_s": 0.10154196317307651,
  "theta_min": -5.303827285766602,
  "theta_max": -3.339754343032837,
  "theta_mean": -4.239127159118652,
  "theta_std": 0.9094665050506592,
  "rate_min": 0.02531563863158226,
  "rate_max": 0.09877198189496994,
  "rate_mean": 0.06349358707666397,
  "active_lower": 0,
  "active_upper": 0,
  "param_count": 3975,
  "finite": true
}
```

## Runtime

- Total elapsed: `43.68` s
- Output JSON: `benchmarks/large_dataset_capacity/output/hogenom_penalty_cv_pilot/20260609_011651/results.json`

## Commands Run

```bash
python /home/enzo/Documents/git/gpurec/gpurec/benchmarks/large_dataset_capacity/hogenom_penalty_cv_pilot.py --max-families 80 --val-families 16 --penalties 0,0.001,0.01,0.1,1.0 --steps 12 --lr 0.08 --pi-solver gmres --self-loop-solver gmres --output-root benchmarks/large_dataset_capacity/output/hogenom_penalty_cv_pilot --report benchmarks/large_dataset_capacity/reports/hogenom_penalty_cv_subagent.md
```

## Blockers

- Full 1055-family specieswise CV was not run in this bounded pilot because multiplying the full dataset by the penalty grid would be substantially more expensive.
- No GBM/tree prior was exercised; no ready script-level GBM prior hook was found during the bounded inspection, so this pilot focused on L2 as requested.
- This script uses the current worktree implementation of GMRES; it does not modify or independently validate core GMRES internals.

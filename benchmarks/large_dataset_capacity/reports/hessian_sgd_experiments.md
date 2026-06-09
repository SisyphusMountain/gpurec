# Hessian-SGD HOGENOM Experiments

Date: 2026-06-04

## Context

The current checkout's benchmark harness did not contain the production
`hessian-sgd` workflow optimizer. The implementation was found in the sibling
worktree:

```text
/home/enzo/Documents/git/gpurec/gpurec-e-step-bench
```

The earlier local benchmark-harness finite-difference diagonal fallback has
been renamed to `fd-diag-hessian-sgd` to avoid confusing it with the production
workflow optimizer.

## Inputs

- Species tree:
  `benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core/hogenom/runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick`
- Generated family manifests:
  - `benchmarks/large_dataset_capacity/generated/alerax_hogenom_core_100_families.txt`
  - `benchmarks/large_dataset_capacity/generated/alerax_hogenom_core_1000_families.txt`

Both manifests passed `gpurec validate-inputs --check-preprocess
--require-cuda-backward-ready`.

## Results

| Run | Families | Settings | Status | Steps | Wall time | Final NLL bits | Final projected grad |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `alerax_hogenom_core_100_real_hessian_sgd` | 100 | default `lr=0.01`, `family_chunk_size=25` | not converged, max steps | 120 | 29.78s | 33321.77 | 5.32 |
| `alerax_hogenom_core_100_real_hessian_sgd_lr05_full` | 100 | `lr=0.5`, `solver_warmup_iters=0`, `family_chunk_size=25` | not converged, max steps | 120 | 40.54s | 30608.56 | 27.40 |
| `alerax_hogenom_core_100_real_hessian_sgd_lr05_onebatch` | 100 | `lr=0.5`, `solver_warmup_iters=0`, one resident batch | converged, loss plateau | 67 | 38.66s | 30478.38 | 27.32 |
| `alerax_hogenom_core_1000_real_hessian_sgd_lr05_onebatch` | 1000 | `lr=0.5`, `solver_warmup_iters=0`, default clade budget, 2 resident batches | not converged, max steps | 80 | 198.91s | 200547.53 | 30.47 |
| `alerax_hogenom_core_1000_real_hessian_sgd_lr05_clade2m` | 1000 | `lr=0.5`, `solver_warmup_iters=0`, `clade_budget=2000000`, one resident batch | converged, loss plateau | 80 | 264.27s | 188134.17 | 30.04 |
| `alerax_hogenom_core_100_real_hessian_sgd_refresh5_val32` | 100 | `lr=0.5`, `solver_warmup_iters=0`, one resident batch, Hessian refresh every 5 steps, validation Pi/Neumann 32 every 10 steps | converged, loss plateau | 61 | 37.68s | 30552.76 | 27.93 |
| `alerax_hogenom_core_100_real_hessian_sgd_refresh5_val5_32` | 100 | `lr=0.5`, `solver_warmup_iters=0`, one resident batch, Hessian refresh every 5 steps, validation Pi/Neumann 32 every 5 steps | converged, loss plateau | 59 | 37.96s | 30526.34 | 27.18 |
| `alerax_hogenom_core_1000_real_hessian_sgd_refresh5_val5_32_clade2m` | 1000 | `lr=0.5`, `solver_warmup_iters=0`, `clade_budget=2000000`, one resident batch, Hessian refresh every 5 steps, validation Pi/Neumann 32 every 5 steps | not converged, max steps | 80 | 306.90s | 188657.16 | 30.41 |

## Adaptive Rebatching

These runs used the same 1000-family one-batch baseline settings as
`alerax_hogenom_core_1000_real_hessian_sgd_lr05_clade2m`, but enabled
`adaptive_rebatch` and varied `adaptive_rebatch_fraction`.

| Run | Fraction | First rebatch step | Status | Steps | Wall time | Final NLL bits | Final projected grad | Delta vs no rebatch |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `alerax_hogenom_core_1000_real_hessian_sgd_lr05_clade2m` | disabled | n/a | converged, loss plateau | 80 | 264.27s | 188134.17 | 30.04 | n/a |
| `alerax_hogenom_core_1000_real_hessian_sgd_adaptive025_clade2m` | 0.25 | 2 | not converged, max steps | 80 | 76.96s | 195659.23 | 31.00 | +7525.06 |
| `alerax_hogenom_core_1000_real_hessian_sgd_adaptive050_clade2m` | 0.50 | 47 | not converged, max steps | 80 | 183.31s | 188246.16 | 30.18 | +111.98 |
| `alerax_hogenom_core_1000_real_hessian_sgd_adaptive067_clade2m` | 0.67 | 62 | not converged, max steps | 80 | 219.42s | 188045.50 | 30.71 | -88.67 |
| `alerax_hogenom_core_1000_real_hessian_sgd_adaptive075_clade2m` | 0.75 | 71 | not converged, max steps | 80 | 240.33s | 188127.58 | 29.73 | -6.59 |

## Interpretation

The production `hessian-sgd` workflow can optimize the HOGENOM-Core subsets and
reports convergence by likelihood plateau on the one-batch 100- and
1000-family runs. However, the final projected gradients remain large, so these
should not be treated as projected-gradient convergence. The default
`lr=0.01` route was too conservative for the 100-family subset within 120
steps.

The tested "refresh Hessians every 5 steps, periodically increase Pi/Neumann"
strategy was implemented with `fd_hessian_refresh_steps=5` and periodic
`hessian_sgd_validation_*` budgets of 32. Under the tested `lr=0.5` settings it
did not improve the 1000-family endpoint versus the earlier refresh-16 run:
the 1000-family run was slower, stopped at the max-step cap, and ended with a
higher NLL.

Adaptive rebatching helped only when the threshold was late enough. Fraction
0.25 removed families too aggressively and damaged the final objective. Fraction
0.50 saved time but ended slightly worse. Fractions 0.67 and 0.75 both improved
wall time and final NLL versus no rebatching, with 0.67 the best tested
tradeoff: 17.0% faster and 88.67 bits lower NLL. All adaptive runs still hit
the max-step cap, and their final-check loss deltas are not directly comparable
because final validation is reported on the post-rebatch active subset while
`final_nll_bits` and `per_fam_likelihoods.tsv` cover all 1000 families.

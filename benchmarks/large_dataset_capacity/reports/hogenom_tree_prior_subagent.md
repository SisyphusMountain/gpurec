# HOGENOM Tree Prior Bounded Experiment

## Scope

This bounded experiment tests the current `benchmarks/large_dataset_capacity/hogenom_specieswise_map_optimize.py` tree/root prior controls without editing core implementation.

All runs:

- Initialized from `benchmarks/large_dataset_capacity/output/hogenom_specieswise_map/20260609_012044/theta_final.pt`.
- Used `specieswise` mode on the first 160 HOGENOM families.
- Used forward Pi GMRES with `--pi-iters 16`.
- Used backward/self-loop GMRES with `--neumann-terms 16`.
- Used weak eventwise L2: `--penalty-lambdas 0.01,0.01,0.01`.
- Used `--steps 4`, `--lr 0.02`, `--device cuda`.

The tree prior is applied in log2-rate space as a unit-branch Brownian edge penalty:

`0.5 * sum(lambda_tree_event * (theta_child - theta_parent)^2)`

The root prior anchors root log2 rates to `log2(0.05)`:

`0.5 * sum(lambda_root_event * (theta_root - log2(0.05))^2)`

## Commands

Baseline weak L2 only:

```bash
python benchmarks/large_dataset_capacity/hogenom_specieswise_map_optimize.py \
  --max-families 160 \
  --steps 4 \
  --lr 0.02 \
  --penalty-lambdas 0.01,0.01,0.01 \
  --init-theta benchmarks/large_dataset_capacity/output/hogenom_specieswise_map/20260609_012044/theta_final.pt \
  --pi-solver gmres \
  --pi-iters 16 \
  --self-loop-solver gmres \
  --neumann-terms 16 \
  --device cuda \
  --print-every 1 \
  --output-root benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/baseline_l2_weak
```

Uniform and eventwise tree/root prior grid:

```bash
python benchmarks/large_dataset_capacity/hogenom_specieswise_map_optimize.py \
  --max-families 160 --steps 4 --lr 0.02 \
  --penalty-lambdas 0.01,0.01,0.01 \
  --tree-penalty-lambdas 1,1,1 \
  --root-penalty-lambdas 1,1,1 \
  --init-theta benchmarks/large_dataset_capacity/output/hogenom_specieswise_map/20260609_012044/theta_final.pt \
  --pi-solver gmres --pi-iters 16 --self-loop-solver gmres --neumann-terms 16 \
  --device cuda --print-every 4 \
  --output-root benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/tree_root_1

python benchmarks/large_dataset_capacity/hogenom_specieswise_map_optimize.py \
  --max-families 160 --steps 4 --lr 0.02 \
  --penalty-lambdas 0.01,0.01,0.01 \
  --tree-penalty-lambdas 10,10,10 \
  --root-penalty-lambdas 10,10,10 \
  --init-theta benchmarks/large_dataset_capacity/output/hogenom_specieswise_map/20260609_012044/theta_final.pt \
  --pi-solver gmres --pi-iters 16 --self-loop-solver gmres --neumann-terms 16 \
  --device cuda --print-every 4 \
  --output-root benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/tree_root_10

python benchmarks/large_dataset_capacity/hogenom_specieswise_map_optimize.py \
  --max-families 160 --steps 4 --lr 0.02 \
  --penalty-lambdas 0.01,0.01,0.01 \
  --tree-penalty-lambdas 100,100,100 \
  --root-penalty-lambdas 100,100,100 \
  --init-theta benchmarks/large_dataset_capacity/output/hogenom_specieswise_map/20260609_012044/theta_final.pt \
  --pi-solver gmres --pi-iters 16 --self-loop-solver gmres --neumann-terms 16 \
  --device cuda --print-every 4 \
  --output-root benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/tree_root_100

python benchmarks/large_dataset_capacity/hogenom_specieswise_map_optimize.py \
  --max-families 160 --steps 4 --lr 0.02 \
  --penalty-lambdas 0.01,0.01,0.01 \
  --tree-penalty-lambdas 1,10,10 \
  --root-penalty-lambdas 1,10,10 \
  --init-theta benchmarks/large_dataset_capacity/output/hogenom_specieswise_map/20260609_012044/theta_final.pt \
  --pi-solver gmres --pi-iters 16 --self-loop-solver gmres --neumann-terms 16 \
  --device cuda --print-every 4 \
  --output-root benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/tree_root_1_10_10

python benchmarks/large_dataset_capacity/hogenom_specieswise_map_optimize.py \
  --max-families 160 --steps 4 --lr 0.02 \
  --penalty-lambdas 0.01,0.01,0.01 \
  --tree-penalty-lambdas 10,100,100 \
  --root-penalty-lambdas 10,100,100 \
  --init-theta benchmarks/large_dataset_capacity/output/hogenom_specieswise_map/20260609_012044/theta_final.pt \
  --pi-solver gmres --pi-iters 16 --self-loop-solver gmres --neumann-terms 16 \
  --device cuda --print-every 4 \
  --output-root benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/tree_root_10_100_100
```

## Run Directories

| Label | Run directory |
| --- | --- |
| Weak L2 only | `benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/baseline_l2_weak/20260609_031237` |
| Tree/root 1,1,1 | `benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/tree_root_1/20260609_031340` |
| Tree/root 10,10,10 | `benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/tree_root_10/20260609_031412` |
| Tree/root 100,100,100 | `benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/tree_root_100/20260609_031444` |
| Tree/root 1,10,10 | `benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/tree_root_1_10_10/20260609_031516` |
| Tree/root 10,100,100 | `benchmarks/large_dataset_capacity/output/hogenom_tree_prior_subagent/tree_root_10_100_100/20260609_031547` |

## Final Metrics

| Label | Tree/root D,T,L | Raw NLL start | Raw NLL final | Objective final | Penalty final | Tree penalty final | Root penalty final | Rate max final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Weak L2 only | none | 135617.28 | 134904.31 | 134947.92 | 43.61 | 0.00 | 0.00 | 0.170929 |
| Tree/root 1,1,1 | 1,1,1 | 135617.28 | 134968.59 | 136748.61 | 1780.02 | 1728.15 | 9.56 | 0.170930 |
| Tree/root 10,10,10 | 10,10,10 | 135617.28 | 135192.55 | 151829.86 | 16637.31 | 16500.16 | 95.63 | 0.170146 |
| Tree/root 100,100,100 | 100,100,100 | 135617.28 | 135502.27 | 300541.19 | 165038.91 | 164042.28 | 956.30 | 0.166971 |
| Tree/root 1,10,10 | 1,10,10 | 135617.28 | 135103.58 | 144455.75 | 9352.17 | 9242.14 | 68.22 | 0.170146 |
| Tree/root 10,100,100 | 10,100,100 | 135617.28 | 135424.62 | 227653.84 | 92229.22 | 91506.58 | 682.21 | 0.166969 |

## Per-Event Rate Tail Summaries

Counts are across specieswise theta rows for each event column. All runs had `gt1 = 0` for D, T, and L.

| Label | Event | Min | Median | q95 | q99 | Max | Count > 0.15 | Count > 0.25 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Weak L2 only | D | 0.015463 | 0.019485 | 0.143126 | 0.160397 | 0.168842 | 62 | 0 |
| Weak L2 only | T | 0.014680 | 0.161704 | 0.166731 | 0.169169 | 0.170929 | 991 | 0 |
| Weak L2 only | L | 0.014697 | 0.153212 | 0.161376 | 0.165578 | 0.167775 | 761 | 0 |
| Tree/root 1,1,1 | D | 0.016336 | 0.019813 | 0.140995 | 0.160146 | 0.168305 | 46 | 0 |
| Tree/root 1,1,1 | T | 0.016389 | 0.161241 | 0.166448 | 0.168530 | 0.170930 | 918 | 0 |
| Tree/root 1,1,1 | L | 0.015285 | 0.152907 | 0.161319 | 0.165578 | 0.167774 | 739 | 0 |
| Tree/root 10,10,10 | D | 0.016856 | 0.019672 | 0.139409 | 0.158399 | 0.160786 | 21 | 0 |
| Tree/root 10,10,10 | T | 0.016390 | 0.157955 | 0.165452 | 0.167883 | 0.170146 | 808 | 0 |
| Tree/root 10,10,10 | L | 0.015968 | 0.150642 | 0.161019 | 0.165588 | 0.167777 | 673 | 0 |
| Tree/root 100,100,100 | D | 0.017276 | 0.019618 | 0.138895 | 0.147153 | 0.160368 | 7 | 0 |
| Tree/root 100,100,100 | T | 0.016390 | 0.151192 | 0.163072 | 0.165184 | 0.166971 | 705 | 0 |
| Tree/root 100,100,100 | L | 0.016041 | 0.145560 | 0.157868 | 0.159240 | 0.161292 | 520 | 0 |
| Tree/root 1,10,10 | D | 0.016335 | 0.019815 | 0.140973 | 0.160157 | 0.168293 | 46 | 0 |
| Tree/root 1,10,10 | T | 0.016390 | 0.157954 | 0.165451 | 0.167879 | 0.170146 | 808 | 0 |
| Tree/root 1,10,10 | L | 0.015968 | 0.150641 | 0.161017 | 0.165588 | 0.167776 | 673 | 0 |
| Tree/root 10,100,100 | D | 0.016855 | 0.019672 | 0.139408 | 0.158399 | 0.160786 | 21 | 0 |
| Tree/root 10,100,100 | T | 0.016390 | 0.151194 | 0.163072 | 0.165184 | 0.166969 | 705 | 0 |
| Tree/root 10,100,100 | L | 0.016041 | 0.145561 | 0.157867 | 0.159239 | 0.161291 | 520 | 0 |

## Interpretation

The weak L2 continuation gives the best short-run raw NLL on this 160-family subset, but it also leaves the largest T/L tail counts. The tree prior immediately regularizes spatial variation in theta; even lambda 1 reduces the high-rate event tails with a modest raw NLL cost over four steps.

Uniform lambda 10 and eventwise `1,10,10` are the most plausible bounded candidates from this pass. They reduce T/L tail counts by the same amount, but `1,10,10` has a substantially lower final objective because it avoids over-penalizing duplication edges. Uniform lambda 100 and eventwise `10,100,100` are too strong for this checkpoint and short continuation: the objective is dominated by the tree penalty and raw NLL improvement is poor.

## Recommendation

Use an eventwise tree/root prior around `D,T,L = 1,10,10` as the next full-dataset candidate, with weak eventwise L2 retained as a small anchor. This setting is a better compromise than scalar L2 alone because it directly penalizes phylogenetic roughness while preserving duplication flexibility. It is also clearly less aggressive than the `10,100,100` or uniform 100 settings.

Suggested next full or longer bounded continuation:

```bash
python benchmarks/large_dataset_capacity/hogenom_specieswise_map_optimize.py \
  --steps 40 \
  --lr 0.01 \
  --penalty-lambdas 0.01,0.01,0.01 \
  --tree-penalty-lambdas 1,10,10 \
  --root-penalty-lambdas 1,10,10 \
  --init-theta benchmarks/large_dataset_capacity/output/hogenom_specieswise_map/20260609_012044/theta_final.pt \
  --pi-solver gmres \
  --pi-iters 16 \
  --self-loop-solver gmres \
  --neumann-terms 16 \
  --device cuda \
  --print-every 5 \
  --output-root benchmarks/large_dataset_capacity/output/hogenom_specieswise_map_tree_prior
```

## Blockers And Caveats

- This is not a full cross-validation result. It is a bounded subset continuation intended to screen tree prior strengths.
- The subset uses the first 160 sorted gene-family tree files, not a randomized split.
- The checkpoint was optimized on all 1055 families with weak scalar L2, so this experiment measures continuation behavior from an existing solution rather than from a neutral initialization.
- The current tree penalty is unit-branch in log2-rate space. If branch lengths should scale Brownian variance, the script would need an implementation update before a final GBM prior claim.

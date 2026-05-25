# test_trees_1000 Resident Likelihood Timing

Date: 2026-05-25

This records the current fast path for full-dataset resident likelihood
evaluation on `tests/data/test_trees_1000` in specieswise mode.  The generated
tree dataset has a different shape from HOGENOM: `1999` species, `1000`
families, about `6.4M` clades, and `21` resident batches at a `315000` clade
budget.

For end-to-end timing on this dataset, `clade_first_fit` is better than the
HOGENOM-style `depth_first_fit` layout.  It preserves the steady likelihood
time while avoiding the depth-first prepass over per-family scheduler summaries.

Cold end-to-end command:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters 4 \
  --measure loss-only \
  --warmups 1 \
  --reps 1 \
  --family-chunk-size 300 \
  --clade-budget 315000 \
  --batch-packing clade_first_fit \
  --max-wave-size 8192 \
  --materialize-batches none
```

Cold result:

| Stage | Time |
|---|---:|
| model init / first resident batch | `5.597596463048831s` |
| first likelihood pass plus lazy remaining batches | `7.464512146951165s` |
| total to first fixed4 likelihood | `13.062108609999996s` |

The same lazy end-to-end path with `depth_first_fit` took `17.548588357982226s`
in a manual timing split (`9.950179827981628s` build, `7.598408530000597s`
first likelihood), so `clade_first_fit` saves about `4.9s` for this generated
dataset.

Steady-state command:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters 4,6,8,32,128 \
  --measure loss-only \
  --warmups 1 \
  --reps 3 \
  --family-chunk-size 300 \
  --clade-budget 315000 \
  --batch-packing clade_first_fit \
  --max-wave-size 8192 \
  --materialize-batches all
```

| Pi/E/Neumann budget | loss-only median | loss bits | delta vs fixed128 |
|---:|---:|---:|---:|
| 4 | `1.6204433359671384s` | `2156427.0` | `670.25` |
| 6 | `2.1482912749634124s` | `2157095.0` | `2.25` |
| 8 | `2.687740627967287s` | `2157097.25` | `0.0` |
| 128 | `34.98644854099257s` | `2157097.25` | `0.0` |

With `--materialize-batches all`, the clade-first resident build split was
`5.495721116021741s` for model init plus `5.5996280499966815s` for full
materialization in the fixed128 reference run.

Gradient timing for the same resident layout:

```bash
env PYTHONDONTWRITEBYTECODE=1 GPUREC_MEMORY_POLICY_RESERVE_GIB=0 \
  python profiling/bench_resident_likelihood.py \
  --dataset tests/data/test_trees_1000 \
  --mode specieswise \
  --fixed-iters 4,6,8 \
  --measure loss-grad \
  --warmups 1 \
  --reps 3 \
  --family-chunk-size 300 \
  --clade-budget 315000 \
  --batch-packing clade_first_fit \
  --max-wave-size 8192 \
  --materialize-batches all
```

| Pi/E/Neumann budget | loss+backward median | loss bits | grad inf |
|---:|---:|---:|---:|
| 4 | `6.295372458000202s` | `2156427.0` | `311.9596252441406` |
| 6 | `7.550051988975611s` | `2157095.0` | `312.9272766113281` |
| 8 | `8.658673683996312s` | `2157097.25` | `312.9301452636719` |

The first clade-first loss+backward warmup in this process took
`43.41080819600029s` because it compiled additional backward kernel
specializations.  The table above is steady-state after that compilation.

Interpretation:

- Starting at `4` Pi iterations is a good warm phase.  It is about `40%`
  cheaper than `8` for likelihood-only and about `27%` cheaper than `8` for
  loss+backward.
- Do not treat `4` as final fidelity on this dataset.  It is still `670` bits
  away from the fixed128 reference, in the optimistic direction.  Promote to at
  least `6` once a cheap phase stops making progress; `6` is within `2` bits of
  fixed128 here.
- `8`, `32`, and `128` agree at the printed precision, so higher fixed budgets
  are useful mainly as periodic validation points, not every-step work.

Differences from HOGENOM:

- HOGENOM's successful counts-free specieswise route used a `8 -> 16 -> 32`
  optimizer budget ladder with a `128` validation check.  On `test_trees_1000`,
  the first useful fidelity point is lower: `4` is cheap enough to use as the
  startup phase, then `6` is already nearly at the high-budget likelihood.
- HOGENOM worked best with `depth_first_fit`.  On `test_trees_1000`, depth-first
  gives similar steady likelihood timing but pays an extra construction prepass;
  `clade_first_fit` is therefore the better end-to-end default for the generated
  tree benchmark.
- HOGENOM resident batching had fewer batches under the accepted policy.  This
  dataset splits into `21` batches, so reusing a single global/specieswise
  resident E solve across no-grad batches removes repeated E work and is worth
  about two percent on likelihood-only timing.
- Larger batches and larger wave caps helped little here.  The
  `clade_first_fit`, `315000` clade-budget, `8192` max-wave policy keeps peak
  allocated memory near `5.13 GiB` for likelihood-only, while larger batches
  mostly spend more memory for small timing changes.

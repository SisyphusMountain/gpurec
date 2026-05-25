# test_trees_1000 Resident Likelihood Timing

Date: 2026-05-25

This records the current fast path for full-dataset resident likelihood
evaluation on `tests/data/test_trees_1000` in specieswise mode.  The useful
configuration is the HOGENOM-style resident batching policy, but the generated
tree dataset has a different shape: `1999` species, `1000` families, about
`6.4M` clades, and `21` resident batches at a `315000` clade budget.

Command:

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
  --batch-packing depth_first_fit \
  --max-wave-size 8192
```

| Pi/E/Neumann budget | loss-only median | loss bits | delta vs fixed128 |
|---:|---:|---:|---:|
| 4 | `1.624646694981493s` | `2156427.0` | `670.0` |
| 6 | `2.1562258880003355s` | `2157095.0` | `2.0` |
| 8 | `2.695869004004635s` | `2157097.0` | `0.0` |
| 32 | `9.193076923955232s` | `2157097.0` | `0.0` |
| 128 | `35.489344446978066s` | `2157097.0` | `0.0` |

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
  --batch-packing depth_first_fit \
  --max-wave-size 8192
```

| Pi/E/Neumann budget | loss+backward median | loss bits | grad inf |
|---:|---:|---:|---:|
| 4 | `6.338703259010799s` | `2156427.0` | `311.95794677734375` |
| 6 | `7.653619892022107s` | `2157095.0` | `312.9256591796875` |
| 8 | `8.74490552203497s` | `2157097.0` | `312.9285888671875` |

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
- HOGENOM resident batching had fewer batches under the accepted policy.  This
  dataset splits into `21` batches, so reusing a single global/specieswise
  resident E solve across no-grad batches removes repeated E work and is worth
  about two percent on likelihood-only timing.
- Larger batches and larger wave caps helped little here.  The
  `depth_first_fit`, `315000` clade-budget, `8192` max-wave policy keeps peak
  allocated memory near `5.13 GiB` for likelihood-only, while larger batches
  mostly spend more memory for small timing changes.

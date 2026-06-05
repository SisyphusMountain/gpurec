# Adaptive GMRES CGS2 Interval 4 Backward-Only Run

Date: `2026-06-05 23:12 CEST`

Implementation commit:

```text
1e71da73e8ab1edbc801759cf33a88534ac6dcfb
```

Command:

```bash
PYTHONPATH="$PWD" python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
  --self-loop-solver gmres \
  --gmres-iters 10 \
  --gmres-check-interval 4 \
  --warmup 1 \
  --output-json benchmarks/large_dataset_capacity/output/gmres_adaptive_cgs2_i4_20260606_231244/run_result.json
```

Run summary:

```text
family: CLU_000680_20_4_C
solver: gmres
max m: 10
check interval: 4
wave_count: 68
total_backward_iterations: 638
total_gmres_checks: 251
elapsed_s: 0.1872657008934766
relative L2 error vs Neumann=512: 6.588813e-06
```

This point keeps fewer VJPs than fixed `m=10` (`638` vs `680`) while cutting
the residual least-squares/check count from `598` to `251`.

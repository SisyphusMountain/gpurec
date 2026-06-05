# Adaptive GMRES CGS2 Interval 5 Backward-Only Run

Date: `2026-06-05 23:13 CEST`

Implementation commit:

```text
1e71da73e8ab1edbc801759cf33a88534ac6dcfb
```

Command:

```bash
PYTHONPATH="$PWD" python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
  --self-loop-solver gmres \
  --gmres-iters 10 \
  --gmres-check-interval 5 \
  --warmup 1 \
  --output-json benchmarks/large_dataset_capacity/output/gmres_adaptive_cgs2_i5_20260606_231304/run_result.json
```

Run summary:

```text
family: CLU_000680_20_4_C
solver: gmres
max m: 10
check interval: 5
wave_count: 68
total_backward_iterations: 670
total_gmres_checks: 202
elapsed_s: 0.17927237902767956
relative L2 error vs Neumann=512: 6.588437e-06
```

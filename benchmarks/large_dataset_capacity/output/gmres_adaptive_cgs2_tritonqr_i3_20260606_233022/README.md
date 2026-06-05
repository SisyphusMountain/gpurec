# Adaptive GMRES CGS2 Triton QR Interval 3 Backward-Only Run

Date: `2026-06-05 23:30 CEST`

Implementation commit:

```text
8f5f50c1149513f1219cc134efd339f04b968681
```

Command:

```bash
PYTHONPATH="$PWD" python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
  --self-loop-solver gmres \
  --gmres-iters 10 \
  --gmres-check-interval 3 \
  --warmup 1 \
  --output-json benchmarks/large_dataset_capacity/output/gmres_adaptive_cgs2_tritonqr_i3_20260606_233022/run_result.json
```

Run summary:

```text
family: CLU_000680_20_4_C
solver: gmres
max m: 10
check interval: 3
wave_count: 68
total_backward_iterations: 619
total_gmres_checks: 299
elapsed_s: 0.16277408390305936
relative L2 error vs Neumann=512: 6.589550e-06
```

# Fixed GMRES CGS2 Triton QR M10 Backward-Only Run

Date: `2026-06-05 23:30 CEST`

Implementation commit:

```text
8f5f50c1149513f1219cc134efd339f04b968681
```

Command:

```bash
PYTHONPATH="$PWD" python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
  --self-loop-solver gmres_fixed \
  --gmres-iters 10 \
  --warmup 1 \
  --output-json benchmarks/large_dataset_capacity/output/gmres_fixed_cgs2_tritonqr_m10_20260606_233057/run_result.json
```

Run summary:

```text
family: CLU_000680_20_4_C
solver: gmres_fixed
fixed m: 10
wave_count: 68
total_backward_iterations: 680
total_gmres_checks: 68
elapsed_s: 0.15153281693346798
relative L2 error vs Neumann=512: 6.588901e-06
```

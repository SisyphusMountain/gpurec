# Adaptive GMRES CGS2 Triton Residual Interval 3 Backward-Only Run

Date: `2026-06-05 23:25 CEST`

Implementation commit:

```text
198fa4b92e5f3db005f86fb7c1af2fda386c9a7f
```

Command:

```bash
PYTHONPATH="$PWD" python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
  --self-loop-solver gmres \
  --gmres-iters 10 \
  --gmres-check-interval 3 \
  --warmup 1 \
  --output-json benchmarks/large_dataset_capacity/output/gmres_adaptive_cgs2_tritonres_guard_i3_20260606_232528/run_result.json
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
elapsed_s: 0.16822252608835697
relative L2 error vs Neumann=512: 6.589550e-06
```

This is the best measured VJP/time compromise after replacing checkpoint
`torch.linalg.lstsq` residual checks with the Triton Hessenberg residual
kernel: it uses `619` VJPs instead of fixed `m=10`'s `680`, with backward-only
time close to fixed CGS2.

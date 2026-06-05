# Nsys Adaptive GMRES CGS2 Triton QR Interval 3 Profile

Date: `2026-06-05 23:31 CEST`

Implementation commit:

```text
8f5f50c1149513f1219cc134efd339f04b968681
```

Command:

```bash
PYTHONPATH="$PWD" nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --export=sqlite \
  -o benchmarks/large_dataset_capacity/output/nsys_gmres_adaptive_cgs2_tritonqr_i3_profile_20260606_233120/gmres_adaptive_cgs2_tritonqr_i3_family2461_max10_backward_only \
  python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
    --self-loop-solver gmres \
    --gmres-iters 10 \
    --gmres-check-interval 3 \
    --warmup 1 \
    --output-json benchmarks/large_dataset_capacity/output/nsys_gmres_adaptive_cgs2_tritonqr_i3_profile_20260606_233120/run_result.json
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
mean_wave_iterations: 9.102941176470589
mean_gmres_checks: 4.397058823529412
max_rel_res: 3.848858101780482e-06
```

Key Nsys comparison:

| Category | Old Adaptive MGS | Adaptive CGS2 I3 Triton QR | Previous Fixed CGS2 M10 |
|---|---:|---:|---:|
| summed GPU kernels | `206.592 ms`, `30,093` launches | `122.871 ms`, `10,847` launches | `129.190 ms`, `12,368` launches |
| CUDA API time | `142.419 ms`, `69,317` calls | `87.574 ms`, `24,671` calls | `87.620 ms`, `26,723` calls |
| `cudaStreamSynchronize` API time | `10.039 ms`, `1,875` calls | `0.568 ms`, `747` calls | `2.026 ms`, `254` calls |
| `_wave_backward_uniform_2d_jt_kernel` | `8.134 ms`, `598` calls | `8.326 ms`, `619` calls | `9.104 ms`, `680` calls |
| `_gmres_hessenberg_residual_kernel` | none | `2.865 ms`, `367` calls | none |
| PyTorch sum reductions | `50.868 ms`, `4,102` launches | `4.467 ms`, `489` launches | `4.618 ms`, `557` launches |
| QR/lstsq kernels | `26.721 ms`, `2,324` launches | `0.000 ms`, `0` launches | `5.457 ms`, `272` launches |

Conclusion:

For the first time on this family, adaptive GMRES has lower summed GPU kernel
time than the previous fixed CGS2 profile while using fewer VJPs. The measured
backward-only wall time remains slightly above the new fixed Triton-QR run, so
the remaining question is whether fewer VJPs improves the end-to-end optimizer
trajectory enough to offset the extra adaptive checks.

# Nsys Adaptive GMRES CGS2 Triton Residual Interval 3 Profile

Date: `2026-06-05 23:26 CEST`

Implementation commit:

```text
198fa4b92e5f3db005f86fb7c1af2fda386c9a7f
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
  -o benchmarks/large_dataset_capacity/output/nsys_gmres_adaptive_cgs2_tritonres_guard_i3_profile_20260606_232600/gmres_adaptive_cgs2_tritonres_guard_i3_family2461_max10_backward_only \
  python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
    --self-loop-solver gmres \
    --gmres-iters 10 \
    --gmres-check-interval 3 \
    --warmup 1 \
    --output-json benchmarks/large_dataset_capacity/output/nsys_gmres_adaptive_cgs2_tritonres_guard_i3_profile_20260606_232600/run_result.json
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
max_rel_res: 3.848858101780477e-06
```

Key Nsys comparison:

| Category | Old Adaptive MGS | Adaptive CGS2 I3 Triton Residual | Fixed CGS2 M10 |
|---|---:|---:|---:|
| summed GPU kernels | `206.592 ms`, `30,093` launches | `127.934 ms`, `11,935` launches | `129.190 ms`, `12,368` launches |
| CUDA API time | `142.419 ms`, `69,317` calls | `91.472 ms`, `27,119` calls | `87.620 ms`, `26,723` calls |
| `cudaStreamSynchronize` API time | `10.039 ms`, `1,875` calls | `2.013 ms`, `815` calls | `2.026 ms`, `254` calls |
| `_wave_backward_uniform_2d_jt_kernel` | `8.134 ms`, `598` calls | `8.318 ms`, `619` calls | `9.104 ms`, `680` calls |
| `_gmres_hessenberg_residual_kernel` | none | `1.863 ms`, `299` calls | none |
| PyTorch sum reductions | `50.868 ms`, `4,102` launches | `4.624 ms`, `557` launches | `4.618 ms`, `557` launches |
| QR/lstsq kernels | `26.721 ms`, `2,324` launches | `5.030 ms`, `272` launches | `5.457 ms`, `272` launches |

Conclusion:

The Triton residual checker removes repeated checkpoint cuSOLVER solves. For
this hard family, adaptive CGS2 interval `3` now has slightly lower summed GPU
kernel time than fixed CGS2 while doing fewer `J^T` applications, though
wall-clock still remains close rather than decisively faster.

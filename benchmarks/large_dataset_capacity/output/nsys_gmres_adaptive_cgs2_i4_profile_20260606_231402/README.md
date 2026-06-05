# Nsys Adaptive GMRES CGS2 Interval 4 Backward Profile

Date: `2026-06-05 23:14 CEST`

Implementation commit:

```text
1e71da73e8ab1edbc801759cf33a88534ac6dcfb
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
  -o benchmarks/large_dataset_capacity/output/nsys_gmres_adaptive_cgs2_i4_profile_20260606_231402/gmres_adaptive_cgs2_i4_family2461_max10_backward_only \
  python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
    --self-loop-solver gmres \
    --gmres-iters 10 \
    --gmres-check-interval 4 \
    --warmup 1 \
    --output-json benchmarks/large_dataset_capacity/output/nsys_gmres_adaptive_cgs2_i4_profile_20260606_231402/run_result.json
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
mean_wave_iterations: 9.382352941176471
mean_gmres_checks: 3.6911764705882355
max_rel_res: 3.848858102696082e-06
```

Key Nsys comparison:

| Category | Old Adaptive MGS | Adaptive CGS2 I4 | Fixed CGS2 M10 |
|---|---:|---:|---:|
| summed GPU kernels | `206.592 ms`, `30,093` launches | `137.375 ms`, `15,362` launches | `129.190 ms`, `12,368` launches |
| CUDA API time | `142.419 ms`, `69,317` calls | `101.383 ms`, `34,754` calls | `87.620 ms`, `26,723` calls |
| `cudaStreamSynchronize` API time | `10.039 ms`, `1,875` calls | `3.750 ms`, `902` calls | `2.026 ms`, `254` calls |
| `_wave_backward_uniform_2d_jt_kernel` | `8.134 ms`, `598` calls | `8.596 ms`, `638` calls | `9.104 ms`, `680` calls |
| PyTorch sum reductions | `50.868 ms`, `4,102` launches | `5.064 ms`, `740` launches | `4.618 ms`, `557` launches |
| QR/lstsq kernels | `26.721 ms`, `2,324` launches | `11.928 ms`, `1,004` launches | `5.457 ms`, `272` launches |

Conclusion:

Adaptive CGS2 with coarser residual checks gives a middle point between old
adaptive GMRES and fixed CGS2: fewer VJPs than fixed `m=10`, much lower
orchestration overhead than old adaptive GMRES, but still not as fast as fixed
because residual checks remain expensive.

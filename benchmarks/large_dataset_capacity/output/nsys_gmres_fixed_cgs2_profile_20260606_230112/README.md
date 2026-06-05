# Nsys GMRES Fixed CGS2 Backward Profile

Date: `2026-06-05 23:01 CEST`

Implementation commit:

```text
01a0faa90856a9aa730b21957baac59e0084717c
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
  -o benchmarks/large_dataset_capacity/output/nsys_gmres_fixed_cgs2_profile_20260606_230112/gmres_fixed_cgs2_family2461_max10_backward_only \
  python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
    --self-loop-solver gmres_fixed \
    --gmres-iters 10 \
    --warmup 1 \
    --output-json benchmarks/large_dataset_capacity/output/nsys_gmres_fixed_cgs2_profile_20260606_230112/run_result.json
```

Run summary:

```text
family: CLU_000680_20_4_C
solver: gmres_fixed
fixed m: 10
wave_count: 68
total_backward_iterations: 680
mean_wave_iterations: 10.0
max_rel_res: 3.848858102743517e-06
```

Key Nsys totals:

| Category | Previous Fixed MGS | Fixed CGS2 |
|---|---:|---:|
| summed GPU kernels | `187.624 ms`, `22,024` launches | `129.190 ms`, `12,368` launches |
| CUDA API time under tracing | `111.976 ms`, `48,501` calls | `87.620 ms`, `26,723` calls |
| `cudaStreamSynchronize` API time | `5.076 ms`, `254` calls | `2.026 ms`, `254` calls |
| `_wave_backward_uniform_2d_jt_kernel` | `9.112 ms`, `680` calls | `9.104 ms`, `680` calls |
| PyTorch sum reductions | `58.178 ms`, `4,365` launches | `4.618 ms`, `557` launches |
| cuBLAS GEMV/dot/update kernels | `0.105 ms`, `64` launches | `11.268 ms`, `4,212` launches |
| elementwise kernels | `23.063 ms`, `14,923` launches | `7.277 ms`, `4,927` launches |
| QR/lstsq kernels | `5.457 ms`, `272` launches | `5.457 ms`, `272` launches |

Conclusion:

The actual self-loop `J^T` work is unchanged. The win comes from replacing the
per-coefficient Python/PyTorch MGS loop with two batched CGS passes per Arnoldi
step, reducing small reduction and elementwise launch overhead. The next
remaining target is still a GPU-resident Arnoldi/Givens implementation, but
this confirms the bottleneck was the vector algebra around the matvec.

# Nsys GMRES Fixed-M Mask-Safe Backward Profile

Date: `2026-06-05 22:42 Europe/Paris`

Repository commit:

```text
5d63743f0be5e718d4fd515d57e7fb3d97f16bb1
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
  -o benchmarks/large_dataset_capacity/output/nsys_gmres_fixed_masksafe_profile_20260605_224244/gmres_fixed_masksafe_family2461_max10_backward_only \
  python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
    --self-loop-solver gmres_fixed \
    --gmres-iters 10 \
    --warmup 1 \
    --output-json benchmarks/large_dataset_capacity/output/nsys_gmres_fixed_masksafe_profile_20260605_224244/run_result.json
```

Run summary:

```text
family: CLU_000680_20_4_C
solver: gmres_fixed
fixed m: 10
wave_count: 68
total_backward_iterations: 680
mean_wave_iterations: 10.0
max_rel_res: 3.848858102743524e-06
```

Key Nsys totals:

```text
summed GPU kernels: 187.624 ms across 22,024 launches
CUDA API time under tracing: 111.976 ms across 48,501 calls
GPU memcpy + memset activity: 4.992 ms across 5,240 ops
cudaStreamSynchronize API time: 5.076 ms across 254 calls
```

Top kernel groups:

```text
PyTorch GMRES dot-product reductions: 58.170 ms, 4,363 launches
DTS backward kernels: 36.922 ms, 264 launches
_wave_backward_uniform_2d_precompute_kernel: 24.919 ms, 68 launches
PyTorch elementwise/copy/fill kernels: 23.246 ms, 15,127 launches
PyTorch norm reductions: 10.102 ms, 834 launches
_wave_backward_uniform_2d_jt_kernel: 9.112 ms, 680 launches
cuSOLVER QR from final torch.linalg.lstsq: 4.420 ms, 68 launches
```

Conclusion:

Fixed-m GMRES shifts work from host orchestration back toward actual matvecs.
Compared with adaptive max-10, it performs more `J^T` applications but sharply
reduces QR calls, launch count, API calls, and explicit stream synchronization
count.

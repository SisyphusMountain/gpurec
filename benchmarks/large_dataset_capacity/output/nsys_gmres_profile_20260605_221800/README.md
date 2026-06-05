# Nsys GMRES Backward Profile

Profile date: `2026-06-05 22:18 Europe/Paris`

Repository commit:

```text
5d63743f0be5e718d4fd515d57e7fb3d97f16bb1
```

The worktree had local GMRES experiment edits in
`gpurec/core/kernels/wave_backward.py` when this profile was recorded.

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
  -o benchmarks/large_dataset_capacity/output/nsys_gmres_profile_20260605_221800/gmres_family2461_max10_backward_only \
  python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
    --gmres-iters 10 \
    --warmup 1 \
    --output-json benchmarks/large_dataset_capacity/output/nsys_gmres_profile_20260605_221800/run_result.json
```

Run summary:

```text
family: CLU_000680_20_4_C
family_index: 2461
gmres_iters: 10
gmres_tol: 1e-10
wave_count: 68
total_backward_iterations: 598
mean_wave_iterations: 8.794117647058824
max_wave_iterations: 10
max_rel_res: 3.848858101747602e-06
```

Timing caveat:

```text
No-Nsys backward-only elapsed_s: 0.265298
Nsys-instrumented elapsed_s: 5.232288
```

CUDA tracing heavily perturbs this API-heavy prototype. Use the Nsys report for
attribution, not wall-clock timing.

Key Nsys totals:

```text
summed GPU kernels: 206.592 ms across 30,093 launches
CUDA API time under tracing: 142.419 ms across 69,317 calls
GPU memcpy + memset activity: 6.681 ms across 7,114 ops
cudaStreamSynchronize API time: 10.039 ms across 1,875 calls
```

Top kernel groups:

```text
PyTorch GMRES dot-product reductions: 50.859 ms, 4,100 launches
DTS backward kernels: 36.900 ms, 264 launches
PyTorch elementwise/copy/fill kernels: 28.953 ms, 20,973 launches
_wave_backward_uniform_2d_precompute_kernel: 24.920 ms, 68 launches
cuSOLVER QR from torch.linalg.lstsq: 19.708 ms, 598 launches
PyTorch norm reductions: 10.477 ms, 1,282 launches
_wave_backward_uniform_2d_jt_kernel: 8.134 ms, 598 launches
```

Conclusion:

The current GMRES prototype is not dominated by the retained Triton `J^T`
matvec, and it is not mainly sync-bound. The biggest cost is thousands of small
PyTorch reduction/elementwise kernels plus launch/API overhead from the
Python-level Arnoldi and least-squares loop.

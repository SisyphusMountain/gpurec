# Uniform Backward Profile

Last updated: 2026-05-02.

This note records the current Nsight Systems and Nsight Compute results for the
CUDA uniform-mode backward pass. It focuses on the global fp32 path used by the
batched `GeneReconModel` workload.

## Workload

- Dataset: `tests/data/test_trees_1000`
- Gene families: first 10 `g_*.nwk` files
- Model: `GeneReconModel.from_trees(..., mode="global", pibar_mode="uniform")`
- Device: RTX 4090, driver 580.126.20
- dtype: `torch.float32`
- Initial rates: `(D, L, T) = (0.05, 0.05, 0.05)`
- Pi forward: `fixed_iters_Pi=6`
- Backward pruning: enabled, `pruning_threshold=1e-6`
- Nsight Systems: 2025.5.1, `--trace=cuda,nvtx,osrt`
- Nsight Compute: 2025.3.1, `--set detailed`

The timed interval is one warmed backward pass:

```python
loss = model()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
loss.backward()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
```

The final synchronize is part of the measurement harness.

## Current Result

Warm wall-clock timings outside Nsight:

| Path | Forward | Backward | Peak Memory | Notes |
| --- | ---: | ---: | ---: | --- |
| old sparse `Pibar` VJP | 31.43 ms | 211.39 ms | 4.58 GB | pre-fusion baseline |
| tree `Pibar` VJP + PyTorch `dts_r` recompute | 31.51 ms | 159.33 ms | 4.19 GB | previous default |
| kernelized `dts_r` + PyTorch `index_add_` | 31.33 ms | 72.87 ms | 3.59 GB | `GPUREC_FUSED_DTS_BACKWARD_ACCUM=0` |
| kernelized `dts_r` + direct DTS accumulation | 31.35 ms | 72.17 ms | 3.50 GB | current default |

Correctness against the previous default on this workload:

| Path | Loss Difference | Max Theta-Gradient Abs Diff | Max Theta-Gradient Rel Diff |
| --- | ---: | ---: | ---: |
| kernelized `dts_r` + PyTorch `index_add_` | 0 | 0 | 0 |
| kernelized `dts_r` + direct DTS accumulation | 0 | 0 | 0 |

The current default is about 2.2x faster than the previous default and about
2.9x faster than the old sparse baseline. The big change is replacing the
backward-only PyTorch `_dts_cross_differentiable(...)` recomputation with the
same fused Triton cross-DTS path used by the forward pass. The direct DTS
accumulation is a smaller improvement, mainly reducing allocation and removing
the materialized `grad_Pi_l` / `grad_Pi_r` tensors.

The kernelized `dts_r` path is enabled for CUDA fp32 and fp64. The fused DTS
kernel supports compact per-split scalar parameters (`[N]` / `[N, 1]`) and
per-split species parameters (`[N, S]`), so the fp64 genewise gradient path can
use the same recomputation without expanding parameters.

## Correctness Checks

Commands run after the changes:

```bash
python -m py_compile gpurec/core/backward.py gpurec/core/kernels/wave_backward.py
pytest tests/kernels/test_uniform_cross_pibar_vjp_kernel.py -q
pytest tests/gradients/test_autograd_bridge.py -q
```

Results:

- `tests/kernels/test_uniform_cross_pibar_vjp_kernel.py`: 1 passed
- `tests/gradients/test_autograd_bridge.py`: 15 passed
- The gradient suite now includes `torch.autograd.gradcheck` on a small fp64
  global-uniform model. The standalone gradcheck run also returned `True`.

## What Is Kernelized Now

The backward pass now has these CUDA fused pieces on the uniform path:

- `_wave_backward_uniform_kernel`: self-loop adjoint solve and per-element
  parameter-gradient contributions.
- `_dts_fused_kernel` plus `_seg_lse_hdim_kernel`: recompute cross-DTS `dts_r`
  for the self-loop VJP without PyTorch `cat`, `stack`, or scatter reductions.
- `_dts_cross_backward_accum_kernel`: cross-DTS VJP, with direct `Pi` adjoints
  atomically accumulated into `accumulated_rhs`.
- `_uniform_cross_pibar_vjp_tree_kernel`: uniform `Pibar` VJP using bottom-up
  descendant/subtree gather instead of ancestor scatter.

Fallbacks remain available:

- `GPUREC_KERNELIZED_BACKWARD_DTS=0`: use old PyTorch `dts_r` recomputation.
- `GPUREC_FUSED_DTS_BACKWARD_ACCUM=0`: materialize `grad_Pi_l` / `grad_Pi_r`
  and use PyTorch `index_add_`.
- `GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=scatter`: use the first fused scatter
  `Pibar` VJP instead of the tree/gather VJP.

## Nsight Systems Summary

Nsight Systems adds profiling overhead, so absolute interval times are higher
than the wall-clock benchmark above. The relative breakdown is still useful.

| Path | Captured Backward Interval | Kernel Time, Summed | Memcpy Time, Summed | Memset Time, Summed |
| --- | ---: | ---: | ---: | ---: |
| old sparse `Pibar` VJP | 227.42 ms | 194.12 ms | 2.48 ms | 0.015 ms |
| tree `Pibar` VJP + PyTorch `dts_r` recompute | 169.50 ms | 134.20 ms | 2.07 ms | 0.008 ms |
| kernelized `dts_r` + direct DTS accumulation | 81.32 ms | 44.99 ms | 1.98 ms | 0.008 ms |

The old `CatArrayBatchedCopy` bucket from `_dts_cross_differentiable(...)` was
about 40.5 ms. In the current trace, `CatArrayBatchedCopy` is about 0.02 ms.

### Current Top GPU Operations

| Operation | Count | Total | Max Launch | Notes |
| --- | ---: | ---: | ---: | --- |
| `_wave_backward_uniform_kernel` | 32 | 12.78 ms | 6.40 ms | self-loop adjoint solve |
| `_uniform_cross_pibar_vjp_tree_kernel` | 31 | 10.24 ms | 1.53 ms | uniform `Pibar` VJP |
| `_dts_cross_backward_accum_kernel` | 31 | 6.06 ms | 0.98 ms | cross-DTS VJP + direct Pi accumulation |
| PyTorch scatter/gather kernels | 224 | 3.80 ms | 0.43 ms | remaining scatter/gather bookkeeping |
| `_dts_fused_kernel` | 31 | 2.46 ms | 0.41 ms | cross-DTS recompute before reduction |
| PyTorch reduction kernels | 265 | 2.40 ms | 0.18 ms | parameter reductions and masks |
| PyTorch fill kernels | 343 | 1.26 ms | 0.51 ms | temporary initialization |
| PyTorch add kernels | 321 | 1.00 ms | 0.37 ms | small elementwise updates |
| `_seg_lse_hdim_kernel` | 5 | 0.34 ms | 0.08 ms | multi-split segment logsumexp |

The previous top operation was `CatArrayBatchedCopy_alignedK_contig` at
40.49 ms. It is no longer a meaningful part of the profile.

## Nsight Compute: Uniform `Pibar` VJP

Representative largest launches:

| Path | Kernel | Grid | Block | Duration |
| --- | --- | ---: | ---: | ---: |
| fused scatter | `_uniform_cross_pibar_vjp_kernel` | 17962 | 128 | 8.42 ms |
| fused tree/gather | `_uniform_cross_pibar_vjp_tree_kernel` | 17962 | 128 | 1.57 ms |

| Resource / Metric | Scatter | Tree/Gather |
| --- | ---: | ---: |
| Memory throughput | 85.27% | 66.68% |
| DRAM throughput | 10.74% | 53.52% |
| L1/TEX hit rate | 22.99% | 74.51% |
| L2 hit rate | 97.15% | 85.75% |
| Occupancy | 97.98% | 98.04% |
| Active warps per SM | 47.03 / 48 | 47.06 / 48 |
| Issue slots busy | 8.24% | 41.27% |
| Registers per thread | 40 | 40 |
| Shared memory per block | 1.04 KB | 1.04 KB |
| DRAM read | 621.43 MB | 588.55 MB |
| DRAM write | 289.14 MB | 259.54 MB |
| DRAM bandwidth | 108.13 GB/s | 539.01 GB/s |
| Global load instructions | 75.82 M | 58.27 M |
| Global store instructions | 1.13 M | 1.96 M |
| Global reduction instructions | 19.09 M | 1.13 M |
| L2 atomic input cycles active | 44.78% | 4.86% |
| Branch target uniformity | 77.96% | 100.00% |

The descendant/tree gather removed the expensive ancestor-scatter atomics. The
remaining atomics in this kernel are the final adds into `accumulated_rhs` for
duplicate child clades across splits.

## Nsight Compute: DTS Backward Accumulation

Representative largest launch:

- Kernel: `_dts_cross_backward_accum_kernel`
- Grid: `8981`
- Block: `128`
- Duration: `0.99 ms`

| Resource / Metric | Value |
| --- | ---: |
| Compute throughput | 25.86% |
| Memory / DRAM throughput | 75.45% |
| L1/TEX hit rate | 62.33% |
| L2 hit rate | 73.10% |
| Occupancy | 73.35% |
| Active warps per SM | 35.21 / 48 |
| Issue slots busy | 14.82% |
| Registers per thread | 56 |
| Shared memory per block | 3.07 KB |
| DRAM read | 452.79 MB |
| DRAM write | 298.66 MB |
| DRAM bandwidth | 759.82 GB/s |
| Global load instructions | 11.64 M |
| Global store instructions | 1.15 M |
| Global reduction instructions | 3.39 M |
| L2 atomic input cycles active | 24.86% |
| Branch target uniformity | 42.27% |
| Divergent branch targets | 2.28 M |

Warp stall sampling:

| Stall Reason | Share |
| --- | ---: |
| Barrier | 29.08% |
| Long scoreboard | 27.35% |
| MIO throttle | 25.15% |
| Short scoreboard | 7.52% |
| LG throttle | 5.29% |
| Wait | 2.40% |

Interpretation:

- The direct accumulation kernel trades two PyTorch `index_add_` calls and two
  `[n_splits, S]` temporary tensors for in-kernel global reductions.
- The wall-clock gain is small but positive on this workload: about 0.7 ms and
  about 90 MB less peak allocation.
- The kernel is now visibly affected by atomics and branch divergence. Keeping
  `GPUREC_FUSED_DTS_BACKWARD_ACCUM=0` is useful for A/B profiling on other
  workloads.

## Nsight Compute: Self-Loop Backward Kernel

Representative largest launch:

- Kernel: `_wave_backward_uniform_kernel`
- Grid: `16645`
- Block: `128`
- Duration: `6.52 ms`

| Resource / Metric | Value |
| --- | ---: |
| Compute throughput | 12.26% |
| Memory throughput | 89.67% |
| DRAM throughput | 89.67% |
| L1/TEX hit rate | 57.47% |
| L2 hit rate | 74.47% |
| Occupancy | 82.18% |
| Active warps per SM | 39.45 / 40 |
| Issue slots busy | 12.26% |
| Registers per thread | 48 |
| Shared memory per block | 3.07 KB |
| DRAM read | 3.41 GB |
| DRAM write | 2.48 GB |
| DRAM bandwidth | 903.10 GB/s |
| Global load instructions | 73.50 M |
| Global store instructions | 29.36 M |
| Global reduction instructions | 0 |

This kernel is genuinely DRAM-bandwidth bound. The optimization target here is
reducing bytes moved, especially the scratch arrays used for Neumann iterations
and parameter-gradient contributions.

## Resource Utilization Overview

| Component | Main Saturated Resource | Underused Resource | Practical Meaning |
| --- | --- | --- | --- |
| `_wave_backward_uniform_kernel` | DRAM bandwidth | tensor cores, most compute pipes | reduce scratch traffic and stores |
| `_uniform_cross_pibar_vjp_tree_kernel` | memory dependency and level barriers | tensor cores, most compute pipes | aggregate duplicate child work or reduce tree-reduction traffic |
| `_dts_cross_backward_accum_kernel` | DRAM, atomics, divergence | tensor cores | direct accumulation helps memory but introduces atomic pressure |
| `_dts_fused_kernel` + `_seg_lse_hdim_kernel` | launch count and memory traffic | compute pipes | a single reduced-DTS kernel could remove the temporary `[n_splits, S]` buffer |
| remaining PyTorch ops | launch count and small reductions | compute pipes | fuse parameter reductions and bookkeeping where worthwhile |

Tensor cores are not used by the current hot uniform kernels. The backward pass
is dominated by scalar log-space transforms, gathers, tree reductions,
segmented logsumexp, and atomic accumulation. There is no dense MMA-shaped
operation in the current fused uniform forward or backward path.

## Tensor-Core Opportunities

The current uniform path does not naturally contain dense matrix products. The
main kernels are row-wise Triton programs:

- forward `_wave_step_uniform_kernel`: row max/sum, ancestor correction, DTS_L
  term construction, and logsumexp;
- forward `_wave_step_uniform_linear_kernel`: row max/sum plus a short signed
  sparse operator over each species row;
- backward `_wave_backward_uniform_kernel`: softmax weights, Neumann VJP
  passes, and parameter-gradient scratch writes;
- backward `_uniform_cross_pibar_vjp_tree_kernel`: row max/sum, ancestor
  denominator, bottom-up subtree reduction, and final accumulation;
- cross-DTS kernels: gathers, elementwise logsumexp weights, reductions, and
  atomics.

Using tensor cores would therefore require reformulating part of the uniform
math as batched dense or block-sparse matrix multiplication. This can still be
worth testing, not because matrix multiplication is the mathematical center of
the algorithm, but because tensor cores are currently idle and could offload
some FMA/reduction work away from scalar CUDA cores, irregular memory loads,
atomics, and branch-heavy tree walks.

Important dtype constraint:

- On the RTX 4090, tensor cores are useful for TF32, fp16, and bf16 style
  matrix products. They do not solve strict fp64 performance on this GPU.
- On A100/H100-class GPUs, fp64 tensor-core MMA exists, so the dense/block-sparse
  reformulations below could also be relevant for strict fp64.
- On the current 4090, tensor-core proposals should be treated as
  mixed-precision experiments unless they only affect fp32/TF32 paths.

### 1. Batched Dense or Block-Sparse Uniform `Pibar`

Uniform `Pibar` computes, for a batch of rows:

```text
P = exp2(Pi - row_max)
ancestor_sum = P @ ancestors_T
Pibar = log2(P.sum(dim=1) - ancestor_sum) + row_max + mt
```

The current fused path avoids materializing `P` and walks ancestors inside a
row kernel. That is good for avoiding dense zero work, but it creates irregular
loads and loop-carried pointer/tree work. A tensor-core version would batch many
clade rows and compute `P @ ancestors_T` as:

- dense GEMM with a low-precision dense ancestor matrix;
- block-sparse GEMM after reordering species by DFS/Euler order so nonzeros
  cluster into denser tiles;
- hybrid dense-block + scalar-tail computation, where only high-density tiles
  use tensor cores and the sparse tail stays in the current Triton walk.

Potential resource benefit:

- Moves the ancestor correction from scalar gather loops to tensor-core MMA.
- Converts random/tree-dependent loads into more regular tile loads.
- Can reduce pressure on L1/L2 dependency chains and branch issue slots.
- Leaves scalar/SFU resources for `exp2`, `log2`, max, and final DTS work.

Risk:

- The ancestor matrix is very sparse. For a balanced tree with `S ~= 1999`, the
  useful nonzeros are roughly `S * depth`, while dense GEMM pays `S^2`. Dense
  tensor-core GEMM may still lose unless the wave batch is very large or the
  block-sparse tiling is effective.
- Mixed-precision ancestor sums can perturb denominators near zero; this needs
  the existing debug guard and fp64 parity checks.

### 2. Tensor-Core Cross-`Pibar` VJP

The exact cross-`Pibar` VJP has the same matrix shape in reverse:

```text
u_d = grad_Pibar / denom
correction = u_d @ ancestors_dense
grad_Pi = p_prime * (u_d.sum(dim=1) - correction)
```

The current tree/gather kernel avoids ancestor-scatter atomics by doing a
bottom-up subtree reduction per split side. It is much faster than the original
scatter path, but still uses tree-level barriers, random child/parent loads,
and final atomics into `accumulated_rhs`.

A tensor-core variant would group rows by child clade, aggregate `grad_Pibar`
for duplicate children, then compute `correction` for many child rows with a
dense or block-sparse `u_d @ ancestors_dense`.

Potential resource benefit:

- Replaces tree-level barriers and subtree-buffer traffic with tile MMA.
- Reduces final atomic pressure if duplicate child clades are grouped before
  the VJP.
- Makes memory access more regular and may improve issue-slot utilization.
- Uses tensor cores while scalar units handle denominator formation and final
  elementwise scaling.

Risk:

- Same sparsity problem as forward `Pibar`.
- Grouping by child clade adds preprocessing/scheduling complexity.
- For fp64 on 4090 this is only a mixed-precision option.

### 3. Tensor-Core Uniform Linear Operator

The current `GPUREC_UNIFORM_IMPL=linear` path builds a signed sparse operator
for the Pi-dependent part of the uniform DTS_L update:

```text
raw = v_scaled * P.sum(dim=1) + P @ M_scaled.T
local_log = log2(raw) + row_max + row_scale
```

The Triton implementation stores `M_scaled` in an ELL-like layout and loops over
`MAX_OP_NNZ` per species row. This is sparse and cache-friendly, but it is still
scalar FMA/gather work. For large batches of clade rows, the `P @ M_scaled.T`
piece could be tested as:

- dense TF32/bf16/fp16 GEMM;
- block-sparse tensor-core GEMM with DFS species ordering;
- a hybrid where high-density blocks of `M_scaled` use tensor cores and the
  remaining short rows use the current ELL kernel.

Potential resource benefit:

- Moves the signed sparse operator's FMA work off scalar CUDA cores.
- Removes repeated per-row sparse-gather loops from the forward fixed-point
  iteration.
- Could reduce pressure on instruction issue and memory dependency stalls if
  the block layout is dense enough.

Risk:

- `M_scaled` has only `max_depth + 2` entries per row, so dense GEMM is likely
  too much arithmetic unless the tensor-core throughput compensates.
- `raw` can be close to zero or negative in debug mode. Mixed precision must
  preserve the guard behavior and pass likelihood/gradient parity.

### 4. Batched Reductions as GEMM-Like Operations

Several backward reductions are currently PyTorch reductions or scalar Triton
sums over `[rows, S]` tensors, for example parameter-gradient sums and
`u_d.sum(dim=1)`. In principle, these can be written as multiplication by a
vector/block of ones and executed by tensor cores.

This is lower priority than the `Pibar` and linear-operator candidates.

Potential resource benefit:

- Could reduce launch count and remove some DRAM round trips if fused with
  surrounding tiled work.
- Might help if many reductions are batched together into one large MMA.

Risk:

- Standalone reductions are memory-bound and usually not good tensor-core
  workloads.
- A GEMM-with-ones formulation may increase reads/writes unless it is fused
  into a larger tensor-core kernel.

### 5. Dense/Top-K Transfer Paths

This profile focuses on uniform mode, but the non-uniform dense/top-k paths
already have operations shaped like matrix products:

```text
Pibar = exp2(Pi - max) @ transfer_mat_T
```

The forward code enables `torch.backends.cuda.matmul.allow_tf32` around the
global `Pibar` loop, so PyTorch/cuBLAS dense matmuls can use tensor cores where
applicable. The custom Triton dense fused kernel still uses scalar `tl.sum`
rather than `tl.dot`, so if dense/top-k modes become important again, that path
should be revisited separately.

### Recommended Tensor-Core Experiments

1. Measure matrix sparsity and tile density after DFS/Euler species ordering:
   `ancestors_dense`, `ancestors_T`, and `M_scaled`.
2. Prototype the simplest forward-only experiment first: compute uniform
   `ancestor_sum = P @ ancestors_T` with dense TF32/cuBLAS for large waves, then
   run the existing DTS step. This will quickly show whether extra dense FLOPs
   can beat irregular ancestor walks.
3. If dense loses, prototype block-sparse tiles with a density threshold and
   scalar fallback for sparse tiles.
4. For backward, group cross-`Pibar` VJP rows by child clade before trying
   tensor-core correction. This targets both tensor-core utilization and atomic
   pressure at the same time.
5. Track NCU tensor-pipe utilization together with scalar/SFU utilization,
   DRAM bandwidth, L2 hit rate, branch uniformity, and atomic cycles. The goal
   is not just to light up tensor cores; it is to reduce the currently saturated
   resources without increasing total memory traffic too much.
6. Keep all tensor-core variants behind explicit flags and validate in this
   order: small fp64 `gradcheck`, large-S finite differences, then full
   1000-tree likelihood/gradient parity.

## FP64 Relative Performance

After making the uniform fused backward kernels dtype-parametric, fp64 uses the
same algorithmic backward path as fp32. The remaining fp64 slowdown is therefore
mostly hardware and arithmetic mix, not a separate PyTorch fallback path.

Recent warm CUDA-event timings on the same RTX 4090:

| Workload | dtype | Forward | Backward | Total | Peak Memory |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 tree | fp32 fused | 8.99 ms | 33.81 ms | 42.79 ms | 0.35 GB |
| 1 tree | fp64 fused | 184.33 ms | 89.49 ms | 273.82 ms | 0.68 GB |
| 1 tree | fp64 fallback self-loop | 184.60 ms | 147.37 ms | 331.97 ms | 2.81 GB |
| 10 trees | fp32 fused | 31.53 ms | 72.30 ms | 103.83 ms | 3.50 GB |
| 10 trees | fp64 fused | 1251.91 ms | 395.58 ms | 1647.49 ms | 6.98 GB |

For 10 trees, fp64 is about 15.9x slower overall. That is a weighted average:
the forward pass is about 39.7x slower, while backward is about 5.5x slower.
The forward pass dominates the fp64 total.

The machine is an NVIDIA GeForce RTX 4090. This card has very limited scalar
fp64 throughput compared with fp32, and these kernels do not use tensor cores.
The fp64 forward pass is heavy in scalar `exp2`, `log2`, max/sum reductions,
and log-space arithmetic, so it naturally exposes the fp64 throughput gap. The
backward pass moves more memory, performs atomics, and has more indexing/tree
work, so it is less purely fp64-compute-bound and slows down less.

### What Can Improve True FP64

If the requirement is strict fp64 arithmetic throughout, the only viable path is
to reduce the number of fp64 operations and bytes moved. We cannot make the
4090's fp64 lanes substantially faster in software.

Promising true-fp64 directions:

- Profile fp64 with NCU directly. The current detailed resource tables are
  fp32. We should confirm whether the fp64 forward kernels are dominated by
  fp64 arithmetic pipes, transcendental latency, register pressure, or memory.
- Store final forward row normalizers. The backward self-loop and `Pibar` VJP
  repeatedly recompute `row_max` and shifted row sums from `Pi_star`. Storing
  final `Pi_max` and shifted `Pi_sum` per clade is cheap (`[C, 1]`) and could
  remove one full row-statistics pass in several fp64 kernels.
- Reduce fp64 scratch traffic in `_wave_backward_uniform_kernel`. fp64 doubles
  the global traffic for the Neumann ping-pong buffers and per-element weight
  outputs. The fp32 NCU profile already shows this kernel is DRAM-bound; in
  fp64 the same design also increases register footprint.
- Add fp64-specific Triton tuning. The current block sizes were chosen for
  fp32-like behavior. fp64 values consume pairs of registers and can reduce
  effective occupancy. NCU should drive an autotune over `BLOCK_S`, `num_warps`,
  and possibly smaller per-program work for fp64.
- Avoid materializing intermediate `[n_splits, S]` buffers in fp64. A direct
  reduced cross-DTS recompute kernel would be more valuable in fp64 because the
  temporary is twice as large and its writes are more expensive.
- Group duplicate child-clade `Pibar` VJPs. This reduces repeated row scans,
  row-stat reductions, subtree reductions, and final atomics. It benefits both
  dtypes but is more valuable when each row scan is fp64.
- Use hardware with real fp64 throughput for production fp64 runs. If strict
  fp64 is scientifically required, data-center GPUs such as A100/H100-class
  parts are a better match than an RTX 4090.

### Mixed-Precision Options

If the goal is fp64-quality results rather than bitwise full-fp64 arithmetic,
mixed precision could reduce the fp64/fp32 gap much more. These should be kept
behind explicit flags and validated with likelihood parity, gradient parity,
and `gradcheck`.

Candidate mixed-precision experiments:

- Compute stable softmax weights in fp32 after fp64 max subtraction. The terms
  entering `exp2(term - max)` are non-positive and usually live in a bounded
  range. We can keep `Pi`, parameters, and final accumulators fp64 while casting
  the shifted deltas and weights to fp32.
- Store backward scratch weights in fp32. Buffers such as `diag_wt`,
  `pibar_wt`, `inv_denom`, `p_prime`, and speciation weights are probabilities
  or normalized factors. Keeping only the final adjoints and parameter
  reductions in fp64 could halve scratch bandwidth.
- Use fp32 Neumann ping-pong terms with fp64 accumulation into `v_k`. The
  Neumann correction terms decay quickly; using fp32 for intermediate terms may
  preserve gradients while cutting memory traffic and fp64 arithmetic.
- Use compensated fp32 or fp64-only reductions selectively. For example, keep
  row sums, segment reductions, and final theta gradients in fp64, but compute
  per-element weights in fp32.

These mixed-precision ideas are not semantics-preserving by construction. The
first acceptance gate should be a small fp64 `gradcheck`, followed by large-S
finite differences and parity against the current fused fp64 path.

## Optimization Opportunities

### 1. Reduce Scratch Traffic in `_wave_backward_uniform_kernel`

The largest remaining kernel is the self-loop backward solve. It is
DRAM-bandwidth bound and moves several GB in the largest launch under NCU
replay.

Likely directions:

- Recompute cheap weights instead of storing them when this reduces global
  traffic.
- Reduce the number of `[W, S]` scratch buffers used for Neumann ping-pong and
  per-element parameter contributions.
- Accumulate parameter gradients in a second reduction kernel so the self-loop
  kernel does not need to preserve as many large output buffers.

### 2. Aggregate Duplicate Child Work for `Pibar` VJP

The tree/gather `Pibar` VJP still runs one program per split side and atomically
adds into `accumulated_rhs[child]`. If the same child clade appears many times
in a wave, the row max, denominator, subtree reduction, and final atomic are
repeated.

Likely direction:

- Precompute a per-wave schedule grouped by child clade.
- Aggregate `grad_Pibar` by child before applying the uniform `Pibar` VJP.

### 3. Combine Cross-DTS Recompute and Reduction

The current kernelized `dts_r` recompute uses the forward path:

```python
dts_term = dts_fused(...)
dts_r = segmented_logsumexp(dts_term)
```

This is much faster than the old PyTorch recomputation, but it still allocates
and writes the full `[n_splits, S]` `dts_term` buffer.

Likely direction:

- Write a direct reduced-DTS kernel that computes the five DTS terms and reduces
  into `[W, S]` without materializing `dts_term`.
- Keep the existing forward path as a reference until parity is established.

### 4. Fuse Remaining Parameter Reductions

The current trace still has about 2.4 ms of PyTorch reduction kernels and about
3.8 ms of PyTorch scatter/gather kernels. These are now small compared with the
pre-kernelization `cat` cost, but they are visible.

Likely direction:

- Add wave-local Triton reductions for `aw0`, `aw1`, `aw2`, `aw345`, `aw3`, and
  `aw4`.
- Reuse output buffers across waves to reduce allocator and fill overhead.

## Recommended Next Steps

1. If fp64 throughput matters, run NCU on the fp64 forward and backward kernels
   first. The current fp32 resource profile is not enough to choose between
   arithmetic-pipe, register-pressure, and memory-traffic fixes.
2. Attack `_wave_backward_uniform_kernel` scratch traffic first; it is the
   largest remaining kernel and is clearly DRAM-bound.
3. Measure duplicate child frequency per wave and decide whether grouped
   `Pibar` VJP scheduling is worth the preprocessing complexity.
4. Prototype a direct reduced-DTS recompute kernel only after the two larger
   buckets above are understood.
5. Only after the strict-fp64 profile is understood, try mixed-precision
   variants behind explicit flags and accept them only if fp64 parity and
   `gradcheck` remain clean.

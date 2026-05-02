# Uniform Forward Profile

Last updated: 2026-05-01.

This note records the Nsight Systems and Nsight Compute results for the current
CUDA uniform-mode forward pass. The current high-level `GeneReconModel` default
keeps `fixed_iters_Pi=6`, so the Pi wave loop does not poll convergence with
`.item()` on every wave iteration. Passing `fixed_iters_Pi=None` restores the
adaptive convergence behavior.

## Workload

- Dataset: `tests/data/test_trees_1000`
- Gene families: first 3 `g_*.nwk` files
- Model: `GeneReconModel.from_trees(..., mode="global", pibar_mode="uniform")`
- Device: RTX 4090
- dtype: `torch.float32`
- Initial rates: `(D, L, T) = (0.05, 0.05, 0.05)`
- Solver knobs: `max_iters_E=2000`, `tol_E=1e-6`, `max_iters_Pi=50`, `tol_Pi=1e-3`, `fixed_iters_Pi=6`
- Nsight Systems: 2025.5.1, `--trace=cuda,nvtx,osrt`
- Nsight Compute: 2025.3.1, `SpeedOfLight`, `LaunchStats`, `Occupancy`,
  `MemoryWorkloadAnalysis`, `ComputeWorkloadAnalysis`, `SchedulerStats`,
  `WarpStateStats`

The measured interval is an NVTX `profile_forward` range around:

```python
loss = model()
torch.cuda.synchronize()
```

The final `torch.cuda.synchronize()` is part of the timing harness. It is not a
production synchronization point, but it is included so the measured interval is
full forward latency rather than only CPU enqueue time.

## Current Forward Summary

Current fixed-6 forward time is 15.36 ms.

| Quantity | Value |
| --- | ---: |
| Forward wall time, including final timing sync | 15.362 ms |
| GPU active time inside the interval | 14.053 ms |
| GPU idle/gap time inside the interval | 1.309 ms |
| Kernel execution time, summed | 12.838 ms |
| Device-to-device copy time, summed | 1.213 ms |
| CUDA stream count used by GPU work | 1 |
| Pi-loop host synchronizations | 0 |

All CUDA work in this trace ran on stream `7`; no kernels or copies overlap with
each other. The fixed-6 path removes the old Pi convergence synchronizations, so
the main remaining synchronization in the interval is the harness's final
`cudaDeviceSynchronize`, which waits for already-enqueued GPU work.

## 15.36 ms Timeline

The NVTX ranges below are CPU-side ranges. They measure enqueue/setup time and
include any explicit synchronizations inside the range, but asynchronous kernels
can continue after a range closes and are then paid by the final timing sync.

| CPU/NVTX range | Time |
| --- | ---: |
| `profile_forward` | 15.362 ms |
| `forward extract parameters` | 0.210 ms |
| `forward E fixed point` | 0.650 ms |
| `forward Pi waves` | 11.176 ms |
| `forward root likelihood` | 0.184 ms |
| `forward save outputs` | 0.001 ms |
| `forward reduce` | 0.012 ms |
| Final timing sync after `model()` | 3.058 ms |

The `forward Pi waves` range breaks down as:

| Pi subrange | Time |
| --- | ---: |
| `Pi setup tensors` | 0.149 ms |
| `Pi setup species helpers` | 0.008 ms |
| `Pi setup DTS constants` | 0.025 ms |
| `Pi setup pibar mode` | 0.283 ms |
| `Pi setup uniform linear` | 0.002 ms |
| `Pi wave forward v2` | 10.654 ms |
| `Pi finalize permute` | 0.021 ms |

The previous fixed-6 profile had `Pi setup species helpers` at about 1.1 ms.
That was a cache miss caused by comparing `torch.device("cuda")` with
`torch.device("cuda:0")`. Normalizing the target CUDA device removed that
CPU-side cost.

## GPU Work Attribution

Attributing GPU operations by the CPU NVTX range that launched them gives this
breakdown. This is usually more informative than asking whether the GPU
operation's start/end timestamp falls inside the CPU range, because CUDA launch
is asynchronous.

| Launch site | GPU ops | GPU time | Notes |
| --- | ---: | ---: | --- |
| `Pi wave forward v2` | 763 | 12.476 ms | 11.276 ms kernels, 1.200 ms D2D copies |
| `Pi finalize permute` | 2 | 0.615 ms | two large PyTorch index kernels |
| `Pi setup pibar mode` | 4 | 0.474 ms | fill/elementwise setup kernels |
| `Pi setup tensors` | 4 | 0.271 ms | initial Pi/Pibar allocation fills and leaf scatter |
| `E iter 0` | 76 | 0.143 ms | E warm-start converges quickly |
| `forward root likelihood` | 21 | 0.034 ms | small reductions/elementwise kernels |
| `forward extract parameters` | 17 | 0.022 ms | small parameter kernels |
| Other setup/reduce ranges | 11 | 0.018 ms | negligible |

The actual bottleneck is the launched GPU work from `Pi wave forward v2`, not E
or likelihood reduction.

## Top GPU Operations

Top GPU operations inside `profile_forward`:

| Operation | Count | Total | Average | Max |
| --- | ---: | ---: | ---: | ---: |
| `_wave_step_uniform_kernel` | 270 | 9.957 ms | 36.88 us | 239.59 us |
| Device-to-device copies | 282 | 1.213 ms | 4.30 us | 59.36 us |
| `_dts_fused_kernel` | 44 | 0.808 ms | 18.35 us | 272.20 us |
| PyTorch index kernels for final permute/setup | 3 | 0.619 ms | 206.43 us | 323.24 us |
| PyTorch fill kernels | 60 | 0.512 ms | 8.54 us | 150.69 us |
| PyTorch elementwise setup kernels | 15 | 0.339 ms | 22.62 us | 318.24 us |
| Small PyTorch index kernels in wave loop | 47 | 0.208 ms | 4.43 us | 12.83 us |
| Other small elementwise kernels | 89 | 0.117 ms | 1.32 us | 1.66 us |
| `_seg_lse_hdim_kernel` | 2 | 0.113 ms | 56.45 us | 56.70 us |

The Pi wave loop specifically launches:

| Operation launched by `Pi wave forward v2` | Count | GPU time |
| --- | ---: | ---: |
| `_wave_step_uniform_kernel` | 270 | 9.957 ms |
| Device-to-device copies | 270 | 1.200 ms |
| `_dts_fused_kernel` | 44 | 0.808 ms |
| Small PyTorch index kernels | 45 | 0.198 ms |
| Small PyTorch elementwise kernels | 88 | 0.116 ms |
| `_seg_lse_hdim_kernel` | 2 | 0.113 ms |
| Small fill kernels | 44 | 0.085 ms |

The 270 device-to-device copies are the `Pi[ws:we] = Pi_new` updates after each
fused wave-step kernel. They move about 900 MB total in this workload and cost
about 1.2 ms.

## Synchronization

Current fixed-6 synchronization inside `profile_forward`:

| Sync location | Count | Time |
| --- | ---: | ---: |
| Final timing sync after `model()` | 1 | 3.058 ms |
| `Pi setup pibar mode` | 1 | 0.256 ms |
| `Pi setup tensors` | 1 | 0.100 ms |
| `E iter 0` | 6 | 0.046 ms |

There are no synchronizations inside `Pi wave forward v2` in fixed-6 mode. In
the old adaptive mode, the convergence check:

```python
if compute_diff and max_diff.item() < local_tolerance:
    ...
```

created 104 Pi-loop synchronizations, costing about 6.77 ms on this workload.

## Resource Use

Nsight Compute was run on representative large launches from the fixed-6
forward. The representative `_wave_step_uniform_kernel` launch is the first
large wave in the profiled forward: grid `4684`, block size `128`.

| Metric | `_wave_step_uniform_kernel` |
| --- | ---: |
| Duration | 267.136 us |
| Compute throughput | 74.14% |
| Memory throughput | 65.03% |
| DRAM throughput | 44.24% |
| L1/TEX throughput | 66.49% |
| L2 throughput | 52.88% |
| L1/TEX hit rate | 81.36% |
| L2 hit rate | 88.62% |
| Issue slots busy | 73.01% |
| Achieved occupancy | 88.69% |
| Active warps per scheduler | 10.66 / 12 |
| Eligible warps per scheduler | 3.48 |
| Registers per thread | 40 |
| Local memory spills | 0 |
| Tensor core use | 0% |

Nsight Compute classifies this kernel as balanced between compute and memory.
It is not saturating DRAM; the hottest pipeline is ALU/instruction issue. That
means a pure memory optimization will not fully solve the wave-step cost.
Reducing both instruction count and memory traffic is the likely direction.

The representative `_dts_fused_kernel` launch is different:

| Metric | `_dts_fused_kernel` |
| --- | ---: |
| Duration | 57.280 us |
| Compute throughput | 23.74% |
| Memory throughput | 90.10% |
| DRAM throughput | 90.10% |
| L1/TEX throughput | 22.84% |
| L2 throughput | 38.90% |
| L1/TEX hit rate | 67.11% |
| L2 hit rate | 37.25% |
| Issue slots busy | 23.74% |
| Achieved occupancy | 87.84% |
| Active warps per scheduler | 11.32 / 12 |
| Eligible warps per scheduler | 0.39 |
| Registers per thread | 34 |

The DTS kernel is DRAM-bound and mostly stalled on memory dependencies, but it
only contributes about 0.8 ms total in this workload. Optimizing DTS alone has
a limited ceiling unless it also enables overlap with the wave-step work.

## Underused Resources

- Tensor cores are unused, which is expected for these scalar log-space kernels.
- DRAM is under the limit for `_wave_step_uniform_kernel` at about 44% DRAM
  throughput, even though total memory throughput is 65%.
- The fused wave-step has high but not full issue utilization: about 73% issue
  slots busy and 25% cycles with no eligible warp.
- The DTS kernel has high occupancy but poor issue eligibility because it waits
  on memory; it is using DRAM heavily but not SM compute.
- The whole forward uses one CUDA stream, so there is no GPU overlap between
  wave-step kernels, D2D copies, DTS kernels, setup kernels, or final permute.

## Overlap Opportunities

The current fixed-6 trace has one stream and serialized GPU work. Potential
overlap is therefore still available in principle, but dependencies constrain
where it is safe:

- DTS preparation for wave `k+1` may overlap with the self-loop of wave `k` if
  the inputs are already available. There is an `overlap_streams` path in
  `Pi_wave_forward`, but it is not currently wired through the high-level model.
  This is worth profiling because DTS is DRAM-bound while the wave-step kernel
  is more balanced. The overlap may be partial because both still consume memory
  bandwidth.
- The 270 D2D copies after wave-step launches cost about 1.2 ms. A ping-pong
  global Pi buffer or a kernel variant that writes directly into the next buffer
  could remove these copies. In-place writes are not obviously safe because the
  kernel still reads old row values while computing later species entries.
- Setup and final permutation account for about 1.36 ms of GPU work
  (`Pi setup tensors`, `Pi setup pibar mode`, and `Pi finalize permute`). Some
  of this is static leaf-mask or layout work that may be cacheable across
  forwards.
- Kernel launch overhead is about 1.1 ms of CUDA runtime API time
  (`cudaLaunchKernel` plus `cuLaunchKernelEx`). CUDA graphs or a persistent
  wave-loop kernel could reduce CPU launch overhead, though the GPU is already
  busy enough that not all launch overhead is on the critical path.
- Nsight Compute reports a tail effect for the largest wave-step launch:
  grid `4684` with 128-thread blocks creates 3 full block waves plus a partial
  wave. This is data-shape dependent; changing block size or grouping work may
  help some waves but needs measurement.

## Adaptive Versus Fixed Six

Historical comparison from the same workload before making fixed 6 the retained
default:

| Mode | Forward wall time | `forward Pi waves` | `_wave_step_uniform_kernel` count | Pi-loop sync |
| --- | ---: | ---: | ---: | ---: |
| Adaptive convergence | 21.710 ms | 20.144 ms | 239 | 104 syncs / 6.765 ms |
| Fixed 6 before CUDA-device cache fix | 16.466 ms | 12.772 ms | 270 | 0 syncs |
| Current fixed 6 | 15.362 ms | 11.176 ms | 270 | 0 syncs |

Fixed 6 performs more wave-step kernels than adaptive convergence, but it
removes the host polling cost. The CUDA-device cache fix then removes another
about 1.1 ms of CPU-side setup from the fixed-6 path.

## Ancestor Table And SpMM Experiments

Additional uniform implementations were profiled after the fixed-6 change.
They are selectable through `GPUREC_UNIFORM_IMPL`:

- `GPUREC_UNIFORM_IMPL=ancestor`: keep the fused wave-step structure, but use a
  precomputed padded `ancestor_cols[depth, species]` table instead of following
  `sp_parent` pointers inside the kernel.
- `GPUREC_UNIFORM_IMPL=spmm`: keep the fused wave-step structure, but use a
  CSR ancestor matrix inside the Triton wave kernel. This is the fused
  hand-written SpMM-style path: it reuses the existing row max, exp scaling,
  subtract, log, and DTS computations.
- `GPUREC_UNIFORM_IMPL=torch_spmm`: compute uniform `Pibar` with
  `torch.sparse.mm(ancestors_sparse, Pi_exp.T.contiguous()).T`, then run the
  existing non-Pibar wave-step kernel. The contiguous RHS is important:
  `Pi_exp.T` is a strided view, and the first SpMM prototype fed that directly
  to cuSPARSE.

End-to-end Nsight Systems results:

| Implementation | Forward wall time | `forward Pi waves` | Main wave-step/Pibar kernels | GPU active time |
| --- | ---: | ---: | ---: | ---: |
| Parent pointer fused, current default | 15.460 ms | 12.103 ms | 270 fused wave-step kernels, 9.982 ms | 14.081 ms |
| Parent pointer fused, after CSR patch | 15.406 ms | 11.793 ms | 270 fused wave-step kernels, 9.989 ms | 14.089 ms |
| Fused CSR ancestor correction | 18.257 ms | 11.715 ms | 270 fused wave-step kernels, 12.838 ms | 16.936 ms |
| Precomputed ancestor table | 20.915 ms | 11.875 ms | 270 fused wave-step kernels, 15.477 ms | 19.568 ms |
| Torch sparse-matmul Pibar, strided RHS | 64.767 ms | 59.288 ms | 264 cuSPARSE CSRMM kernels, 26.247 ms; 270 wave-step kernels, 5.599 ms | 56.687 ms |
| Torch sparse-matmul Pibar, contiguous RHS | 50.335 ms | 48.761 ms | 264 cuSPARSE CSRMM kernels, 2.431 ms; 270 wave-step kernels, 5.629 ms | 36.157 ms |

The ancestor-table and fused CSR variants were correct, but slower. The
representative large wave-step launch changed as follows:

| Metric | Parent pointer | Ancestor table | Fused CSR |
| --- | ---: | ---: | ---: |
| Duration | 260.000 us | 324.896 us | 342.976 us |
| Compute throughput | 74.35% | 49.73% | 48.12% |
| Memory throughput | 65.27% | 65.28% | 61.29% |
| DRAM throughput | 45.31% | 36.58% | 34.50% |
| L1/TEX throughput | 66.67% | 72.61% | 62.28% |
| L2 throughput | 54.64% | 52.86% | 61.29% |
| L1/TEX hit rate | 81.29% | 87.06% | 84.91% |
| L2 hit rate | 88.38% | 90.18% | 91.12% |
| Issue slots busy | 73.22% | 49.66% | 47.13% |
| No eligible warp cycles | 25.21% | 45.49% | 48.26% |
| Eligible warps per scheduler | 3.48 | 1.17 | 1.17 |
| Achieved occupancy | 88.59% | 93.69% | 92.15% |

The precomputed table and fused CSR variants remove the parent-pointer
dependency, but both add an explicit ancestor-index stream. The CSR path avoids
padded invalid ancestor loads, but adds row-pointer loads and reads a flattened
column-index array with a poor per-depth access pattern. On this workload both
variants made the kernel more scheduler limited: occupancy went up, but
eligible warps and issue utilization dropped sharply. The parent-pointer version
appears to benefit from a smaller, cache-hot index working set and better
converging access behavior, despite the loop-carried dependency.

The corrected-layout Torch SpMM path is still much worse because it decomposes
one fused wave iteration into many PyTorch/cuSPARSE operations:

| SpMM-path operation | Count | GPU time |
| --- | ---: | ---: |
| PyTorch elementwise/copy kernels | 3766 | 19.156 ms |
| Non-Pibar `_wave_step_kernel` | 270 | 5.629 ms |
| PyTorch reductions | 555 | 3.598 ms |
| cuSPARSE `csrmm_alg2_kernel` | 264 | 2.431 ms |
| cuSPARSE partition/scaling helpers | 549 | 1.876 ms |
| Device-to-device copies | 558 | 2.419 ms |

Making the RHS contiguous fixes the CSRMM layout problem and reduces CSRMM time
by about 10.8x in this trace. The path still is not a good replacement for the
fused uniform kernel: the surrounding exp/log/reduction/materialization kernels
and D2D copies dominate launch count and scheduling overhead.

## Correctness Check

The fixed-iteration comparison used the same model parameters and E solve, then
compared Pi/NLL against the adaptive result.

| Case | NLL | Pi iterations | NLL delta vs adaptive | Max significant Pi delta |
| --- | ---: | ---: | ---: | ---: |
| Adaptive | 6421.173339844 | 239 | 0 | 0 |
| Fixed 4 | 6421.194335938 | 180 | 2.099609375e-2 | 1.686889648 |
| Fixed 5 | 6421.174316406 | 225 | 9.765625000e-4 | 1.219329834e-1 |
| Fixed 6 | 6421.173339844 | 270 | 0 | 5.058288574e-3 |
| Fixed 7 | 6421.173339844 | 315 | 0 | 2.059936523e-4 |
| Fixed 8 | 6421.173339844 | 360 | 0 | 2.479553223e-5 |

Backward comparison:

| Quantity | Value |
| --- | ---: |
| Adaptive loss | 6421.173339844 |
| Fixed-6 loss | 6421.173339844 |
| Loss delta | 0 |
| Gradient max absolute delta | 3.967285156e-4 |

Fixed 6 is sufficient for the likelihood and gradient on this workload, but it
does not strictly satisfy the old max-entry Pi tolerance if estimated by
fixed6-to-fixed7 drift. Fixed 7 is the safer fixed-count candidate for matching
`tol_Pi=1e-3`; it measured about 17.84 ms before the CUDA-device cache fix.

## Main Takeaways

- Current forward time is 15.36 ms for this workload.
- The dominant cost is still the fixed uniform wave loop: 270 fused wave-step
  launches consume 9.96 ms of GPU time.
- The wave-step kernel is balanced and instruction-heavy, not purely DRAM-bound.
- DTS is DRAM-bound but small in aggregate.
- D2D copies after each wave step are the clearest remaining standalone cost at
  about 1.2 ms.
- There is no GPU overlap in the current trace; all work is serialized on one
  stream.

## 1000-Family Chunked Likelihood

Workload: `tests/data/test_trees_1000`, global uniform mode, fp32, fixed
`Pi` iterations = 6, RTX 4090. A single all-family resident run is not viable:
the 1000 families contain 6,417,248 clades, so one `[C,S]` fp32 matrix is about
51.3 GB at `S=1999`; `Pi + Pibar` alone would be about 102.6 GB. A first
all-family preprocessing attempt also reached about 93 GB RSS before GPU
layout/forward.

Implemented changes for this workload:

- Uniform fused leaf handling now uses `leaf_species_index[C]` instead of a
  persistent `[C,S]` leaf-term matrix.
- High-level likelihood paths skip the final full `Pi[perm]` copy and compute
  root likelihood from wave-ordered root ids.
- Fixed-6 uniform mode can use `Pibar[ws:we]` as ping-pong scratch for odd
  iterations, then recompute final `Pibar` once per wave. This removes the
  per-iteration `Pi_new -> Pi` device copy.
- The light batched preprocessing path now skips the C++ clade inclusion DAG,
  which is only needed for full debug/detail outputs and is not consumed by the
  likelihood path.
- `max_wave_size` can split oversized index-merged waves, but the stable result
  below uses the default index-merged waves. The tested large capped run needs
  more validation before using it for production.

Chunk-size progression on the first families:

| Version | Chunk | Forward | Peak GPU | Result |
| --- | ---: | ---: | ---: | --- |
| Previous committed path | 50 | 178.2 ms | 15.5 GB | Fits |
| Previous committed path | 70 | 245.8 ms | 21.4 GB | Fits |
| Previous committed path | 75 | - | - | OOM |
| Leaf index only | 100 | 321.7 ms | 15.3 GB | Fits |
| Leaf index + skip final `Pi[perm]` | 150 | 467.8 ms | 19.2 GB | Fits |
| Leaf index + skip final copy + ping-pong | 150 | 401.4 ms | 16.8 GB | Fits |
| Leaf index + skip final copy + ping-pong | 175 | - | - | Not stable; DTS path hit illegal access near memory limit |

Full 1000-family run with 150-family chunks:

| Chunk | Families | Clades | Waves | Max wave | Forward | Peak GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 150 | 954,706 | 48 | 238,864 | 401.441 ms | 16.770 GB |
| 1 | 150 | 969,482 | 48 | 242,558 | 405.702 ms | 17.029 GB |
| 2 | 150 | 979,570 | 52 | 245,080 | 408.183 ms | 17.208 GB |
| 3 | 150 | 961,630 | 48 | 240,595 | 407.063 ms | 16.893 GB |
| 4 | 150 | 954,750 | 48 | 238,875 | 397.762 ms | 16.770 GB |
| 5 | 150 | 962,290 | 48 | 240,760 | 400.981 ms | 16.904 GB |
| 6 | 100 | 634,820 | 50 | 158,830 | 266.704 ms | 11.160 GB |
| Total | 1000 | 6,417,248 | - | - | 2,687.836 ms | - |

An earlier same-chunk run spent 590.0 s in repeated Newick
preprocessing/model construction. After switching to batched light
preprocessing this dropped to 396.0 s, but profiling showed that the light path
still spent nearly all of its time computing an unused inclusion DAG.

10-family CPU/profile breakdown before skipping the inclusion DAG:

| Stage | Time |
| --- | ---: |
| C++ `preprocess_multiple_families(...)` light default | 3.929 s |
| Python family normalization | 0.006 s |
| `collate_gene_families` | 0.005 s |
| Python wave scheduling/merge | 0.003 s |
| GPU `build_wave_layout` | 0.027 s |
| Species helper move/sparse conversion | 0.003 s |
| Total measured construction work | 3.971 s |

The C++ `bench_parse` hook narrowed that down further:

| C++ stage, 10 families | With inclusion DAG | Without inclusion DAG |
| --- | ---: | ---: |
| `amalgamate_clades_and_splits` | 0.040 s | 0.037 s |
| `build_ccp_arrays` | 3.786 s | 0.003 s |
| Scheduling adjacency build | 0.011 s | 0.010 s |
| `compute_clade_waves` | 0.013 s | 0.012 s |
| Total | 3.851 s | 0.063 s |

The hotspot was the O(C^2) subset test in the inclusion-DAG construction inside
`build_ccp_arrays`. It is now guarded by the same light/full flag that controls
whether those debug fields are returned.

The current construction path now uses `preprocess_multiple_families`, whose
default output is the light likelihood-only payload. Full debug/detail output is
still available with `include_details=True`. The light payload skips large
unused CCP detail structures (`clade_leaves`, `clade_leaf_labels`,
`clade_is_leaf`, and inclusion-DAG debug fields), and no longer computes the
inclusion DAG. The old single-family preprocess path is still available for
debugging with
`GPUREC_PREPROCESS_MODE=single`. `GeneReconModel.from_trees` also accepts
`preprocess_cache_dir=...`; cache keys include the species tree content hash,
gene tree content hash, and a format version.

Current construction timings, same `test_trees_1000` data and same likelihood:

| Families | Path | Build/model construction | Forward | Loss |
| ---: | --- | ---: | ---: | ---: |
| 10 | old single-family preprocess | 4.772 s | - | 22182.916016 |
| 10 | batched light preprocess | 0.157 s | - | 22182.916016 |
| 10 | cache populate | 0.398 s | - | 22182.916016 |
| 10 | cache hit | 0.038 s | - | 22182.916016 |
| 50 | old single-family preprocess | 23.330 s | 136.050 ms | 107804.273438 |
| 50 | batched light preprocess | 0.726 s | 136.274 ms | 107804.273438 |
| 50 | cache populate | 0.810 s | 136.558 ms | 107804.273438 |
| 50 | cache hit | 0.116 s | 137.107 ms | 107804.273438 |
| 150 | batched light preprocess | 2.162 s | 402.325 ms | 323018.687500 |
| 150 | cache populate | 2.387 s | 403.586 ms | 323018.687500 |
| 150 | cache hit | 0.307 s | 403.032 ms | 323018.687500 |

Full 1000-family chunked construction and forward, with 150-family chunks:

| Pass | Build/model construction | Forward | Total loss | Peak GPU | Max RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| No cache, batched light preprocess | 15.586 s | 2690.811 ms | 2157097.125 | 17.280 GB | 1.490 GB |
| Cache populate | 17.261 s | 2704.248 ms | 2157097.125 | 17.280 GB | 1.578 GB |
| Cache hit | 2.129 s | 2712.187 ms | 2157097.125 | 17.280 GB | 1.578 GB |

The uncached 1000-family construction path is now down from about 590 s
originally, to 396 s after light output pruning, to 15.6 s after skipping the
unused inclusion-DAG computation. Reusing the cache still reduces construction
overhead to about 2.1 s; at that point end-to-end likelihood evaluation is
again dominated by the 2.7 s CUDA forward time.

Nsight Systems comparison for a 150-family chunk:

| Metric | Before ping-pong | After ping-pong |
| --- | ---: | ---: |
| Profiled forward interval | 465.557 ms | 400.892 ms |
| CUDA kernel time | 369.893 ms | 398.841 ms |
| Device-to-device memcpy | 93.759 ms / 45.805 GB | 0.0166 ms / 1.53 MB |
| `_wave_step_uniform_kernel` | 276.682 ms | 269.677 ms |
| `_dts_fused_kernel` | 55.573 ms | 55.350 ms |
| Final `_wave_pibar_uniform_parent_kernel` | - | 36.111 ms |

Ping-pong removes about 93.7 ms of D2D copy work and adds about 36.1 ms of
final Pibar recomputation, for a net improvement of about 65 ms on the
150-family chunk.

Nsight Compute sample for the largest 150-family `_wave_step_uniform_kernel`
launch (`grid=(238864,1,1)`, `block=(128,1,1)`):

| Metric | Value |
| --- | ---: |
| Duration | 11.556 ms |
| Compute throughput | 84.55% |
| Memory busy | 71.76% |
| DRAM throughput | 32.57% |
| L1/TEX throughput | 71.79% |
| L2 throughput | 62.57% |
| L1/TEX hit rate | 80.48% |
| L2 hit rate | 94.25% |
| Eligible warps / scheduler | 4.28 |
| No eligible warp cycles | 17.46% |
| Achieved occupancy | 99.59% |
| Local spill requests | 0 |

The large wave-step launch is compute-heavy with high occupancy and no spilling.
The most useful remaining optimization target in the profiled 150-family chunk
is now the algorithmic cost of the uniform wave-step/Pibar work itself, not
copy overhead.

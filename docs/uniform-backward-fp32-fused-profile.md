# Uniform FP32 fused backward profile

Date: 2026-05-02

This document profiles the current FP32 fused backward path for uniform
`Pibar`, identifies the concrete bottlenecks, and lists the next performance
work in priority order.

## Scope

Hardware and tools:

| Item | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 4090 |
| GPU memory | 24,564 MiB |
| Driver | 580.126.20 |
| Nsight Systems | 2025.5.1 |
| Nsight Compute | 2025.3.1 |

Runtime configuration:

```bash
GPUREC_KERNELIZED_BACKWARD_DTS=1
GPUREC_FUSED_DTS_BACKWARD_ACCUM=1
GPUREC_FUSED_CROSS_PIBAR_VJP=1
GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=tree
GPUREC_FUSED_UNIFORM_BACKWARD=1
GPUREC_UNIFORM_PINGPONG=1
```

Model configuration:

```text
mode=global
pibar_mode=uniform
dtype=torch.float32
fixed_iters_Pi=6
neumann_terms=3
use_pruning=True
dataset=tests/data/test_trees_1000
```

The measured interval is a warmed `loss.backward()` only. Forward is computed
before the profiler capture starts. The detailed Nsight Compute counters were
collected on the 10-family batch because NCU replays selected kernels and is
too intrusive for larger batches. Nsight Systems was collected for both 10 and
50 families.

## Workload shape

| Families | Species S | Waves K | Clade rows C | Split rows | Leaves | Roots |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1,999 | 45 | 66,530 | 83,135 | 16,645 | 10 |
| 50 | 1,999 | 47 | 321,930 | 402,275 | 80,545 | 50 |

The biggest waves are large enough to saturate memory bandwidth:

| Families | Largest wave rows | Largest split wave | Root-like split fanout examples |
|---:|---:|---:|---|
| 10 | wave 0: 16,645 rows | wave 41: 8,981 splits over 101 parents | wave 44: 7,330 splits over 2 parents |
| 50 | wave 0: 80,545 rows | wave 42: 42,155 splits over 247 parents | wave 44: 24,229 splits over 39 parents |

Child clade duplication across split sides is modest:

| Families | Split sides | Unique child rows | Duplicate factor |
|---:|---:|---:|---:|
| 10 | 166,270 | 149,635 | 1.111x |
| 50 | 804,550 | 724,051 | 1.111x |

That matters for the uniform `Pibar` VJP: grouping by child row can help, but it
only has an average upper bound near 10% for that kernel on this workload.

## End-to-end backward time

Normal timing, outside Nsight:

| Families | Backward CUDA time | Peak memory | Time per family |
|---:|---:|---:|---:|
| 10 | 72.676 ms | 3.50 GB | 7.268 ms |
| 50 | 229.919 ms | 16.81 GB | 4.598 ms |

The 50-family batch is much more efficient per tree because larger waves keep
the GPU busier. It is also close to the largest useful batch size on this 24 GB
card: peak memory is already 16.8 GB.

For comparison, disabling pruning on the 10-family batch took 101.283 ms. So
the current pruning is still a net win, even though its implementation causes
host/device synchronization.

## Nsight Systems breakdown

Nsight adds some overhead, so the captured wall times are higher than the
normal timings. The GPU-side proportions are still useful.

| Families | Captured backward | GPU kernel time | D2D/memcpy time | GPU active span | GPU idle/gap span | GPU kernels | Copies |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 81.031 ms | 45.062 ms | 1.974 ms | 47.044 ms (58.3%) | 33.687 ms (41.7%) | 2,886 | 688 |
| 50 | 244.036 ms | 197.552 ms | 8.274 ms | 205.842 ms (84.5%) | 37.834 ms (15.5%) | 3,809 | 755 |

The 10-family profile is dominated by launch/sync gaps. At 50 families those
gaps are mostly amortized, and kernel memory traffic becomes the dominant
problem.

CUDA API summary:

| Families | `cudaStreamSynchronize` | `cudaDeviceSynchronize` | `cudaLaunchKernel` | `cudaMemcpyAsync` |
|---:|---:|---:|---:|---:|
| 10 | 308 calls, 27.901 ms API time | 7 calls, 9.136 ms | 2,726 calls, 4.900 ms | 688 calls, 1.930 ms |
| 50 | 352 calls, 146.504 ms API time | 7 calls, 45.590 ms | 3,559 calls, 6.381 ms | 755 calls, 2.153 ms |

The API sync time is not additive with GPU kernel time: much of it is the host
waiting for already submitted kernels. The actionable signal is the GPU idle/gap
span and the fact that CPU decisions repeatedly gate future launches.

The largest code sources are in `gpurec/core/backward.py`:

| Source | Why it synchronizes or creates many kernels |
|---|---|
| `active_mask.any()` in the wave loop | Host branch on CUDA scalar |
| `active_mask.sum().item()` | GPU-to-CPU scalar transfer |
| `active_mask.nonzero(...)` | Dynamic shape compaction |
| `_get_leaf_mask(...): if mask.any()` | Host branch on CUDA scalar |
| `_scatter_accum(...)` | Per-wave PyTorch reductions and scatter adds |

Relevant locations: `gpurec/core/backward.py:641-647`,
`gpurec/core/backward.py:682-740`, and `gpurec/core/backward.py:760-782`.

## Kernel time breakdown

Nsight Systems kernel totals:

| Component | 10 families | % of 10 kernel time | 50 families | % of 50 kernel time | Main issue |
|---|---:|---:|---:|---:|---|
| `_wave_backward_uniform_kernel` | 12.776 ms | 28.4% | 64.022 ms | 32.4% | DRAM bandwidth, scratch traffic |
| `_uniform_cross_pibar_vjp_tree_kernel` | 10.337 ms | 22.9% | 39.555 ms | 20.0% | Memory traffic plus tree barriers |
| `_dts_cross_backward_accum_kernel` | 6.040 ms | 13.4% | 29.441 ms | 14.9% | Atomics, divergence, memory traffic |
| PyTorch scatter/gather reduce-add | 3.801 ms | 8.4% | 16.176 ms | 8.2% | Residual PyTorch scatter reductions |
| `_dts_fused_kernel` | 2.455 ms | 5.4% | 12.393 ms | 6.3% | DRAM bandwidth, split-term materialization |
| PyTorch reductions | 2.477 ms | 5.5% | 8.767 ms | 4.4% | Residual reductions |
| PyTorch add/fill/abs/index | 3.910 ms | 8.7% | 19.329 ms | 9.8% | Temporary setup and mask work |
| `_seg_lse_hdim_kernel` | 0.341 ms | 0.8% | 1.515 ms | 0.8% | Small, low occupancy |

The important split for the 50-family batch is:

```text
Fused Triton kernels named above:     146.9 ms
Residual PyTorch kernels:             ~44.3 ms
Memcpy/copy kernels:                    8.3 ms
GPU idle/gaps:                         37.8 ms
```

So there are two comparable opportunities:

1. Reduce the 146.9 ms of fused-kernel memory traffic.
2. Remove most of the 44.3 ms residual PyTorch work plus the 37.8 ms scheduling
   gaps.

## Nsight Compute resource counters

These are representative large launches from the 10-family capture. The
selected launches were the maximum or near-maximum duration launches for each
kernel class.

| Kernel | Launch duration | Memory throughput | DRAM peak | Compute peak | Issue active | Occupancy | Registers/thread | L1 hit | L2 hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `_wave_backward_uniform_kernel` | 6,502.9 us | 907 GB/s | 90.0% | 22.8% | 12.3% | 82.1% | 48 | 57.4% | 74.4% |
| `_uniform_cross_pibar_vjp_tree_kernel` | 1,547.6 us | 547 GB/s | 54.3% | 52.0% | 42.0% | 98.0% | 40 | 73.4% | 84.6% |
| `_dts_cross_backward_accum_kernel` | 988.0 us | 761 GB/s | 75.6% | 25.8% | 14.8% | 73.6% | 56 | 62.2% | 73.5% |
| `_dts_fused_kernel` | 386.9 us | 909 GB/s | 90.3% | 19.0% | 19.0% | 89.8% | 34 | 69.0% | 32.5% |
| `_seg_lse_hdim_kernel` | 82.7 us | 889 GB/s | 88.2% | 8.7% | 8.7% | 8.0% | 155 | 12.5% | 7.0% |
| PyTorch scatter reduce-add | 472.2 us | 289 GB/s | 28.6% | 11.0% | 10.4% | 95.4% | 32 | 9.8% | 55.1% |

Stall and atomic counters:

| Kernel | Branch uniform | Atomic active | DRAM read | DRAM write | Global loads | Global stores | RED ops | Main stalls |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `_wave_backward_uniform_kernel` | 93.8% | 0.0% | 3,383.9 MB | 2,484.7 MB | 73.50 M | 29.36 M | 0.00 M | long scoreboard 54.6%, barrier 19.2%, MIO throttle 13.3% |
| `_uniform_cross_pibar_vjp_tree_kernel` | 100.0% | 4.8% | 586.7 MB | 259.3 MB | 58.27 M | 1.96 M | 1.13 M | long scoreboard 60.2%, barrier 13.5% |
| `_dts_cross_backward_accum_kernel` | 42.3% | 24.8% | 453.1 MB | 298.8 MB | 11.64 M | 1.15 M | 3.39 M | barrier 28.5%, long scoreboard 26.8%, MIO throttle 25.2% |
| `_dts_fused_kernel` | 0.0% | 0.0% | 290.3 MB | 61.4 MB | 8.51 M | 0.57 M | 0.00 M | long scoreboard 90.3% |
| `_seg_lse_hdim_kernel` | 100.0% | 0.0% | 71.1 MB | 2.4 MB | 0.56 M | 0.00 M | 0.00 M | long scoreboard 36.8%, wait 19.4% |
| PyTorch scatter reduce-add | 100.0% | 15.0% | 133.3 MB | 2.9 MB | 2.08 M | 0.00 M | 1.04 M | long scoreboard 93.0% |

The largest self-loop launch processes the leaf wave:

```text
W = 16,645 rows
S = 1,999 species
elements = 33.27 M
NCU measured DRAM traffic = 5.87 GB
traffic per element = ~176 bytes
```

This kernel is memory-bandwidth bound. Occupancy is already good, and issue
active is low because warps wait on global memory. Optimizing occupancy alone
will not move the needle much.

## Bottleneck 1: self-loop wave kernel memory traffic

This is the largest kernel bucket:

```text
10 families: 12.776 ms total, 28.4% of kernel time
50 families: 64.022 ms total, 32.4% of kernel time
largest 50-family launch: 32.025 ms
```

The key resource counters are:

```text
DRAM utilization: 90.0%
Memory throughput: 907 GB/s
Compute throughput: 22.8%
Issue active: 12.3%
Long scoreboard stalls: 54.6%
Atomics: none
```

So the kernel is not slow because of atomics or low occupancy. It is slow
because it streams too much global memory.

The current implementation in `gpurec/core/kernels/wave_backward.py` stores and
reloads several full `[W, S]` buffers:

| Buffer class | Purpose |
|---|---|
| `rhs` | Input adjoint, overwritten as Neumann scratch |
| `v_k` | Accumulated Neumann solution |
| `aw0`, `aw1`, `aw2`, `aw345`, `aw3`, `aw4` | First used as self-loop weight scratch, then overwritten with parameter-gradient contributions |
| `spec_buf` | Ping-pong buffer for species-child scatter |
| `leaf_term_wt` | Dense `[W, S]` leaf term, almost entirely `-inf` |

The dense leaf term is especially wasteful. For a leaf row only one species
entry is finite, but the current kernel reads `leaf_term_wt` for every species
twice (`wave_backward.py:139` and `wave_backward.py:338`), and Python builds it
as a dense full tensor in `backward.py:641-647`.

### Next work

1. Replace dense `leaf_term_wt` with a compact per-row leaf species index.
   The kernel can compute:

   ```text
   t5 = log_pS[s] if leaf_species_for_row[w] == s else -inf
   ```

   This removes dense `[W, S]` allocation/fill and two global reads per element
   in the self-loop kernel.

2. Stop writing full parameter-gradient contribution tensors when possible.
   The kernel currently writes six `[W, S]` arrays and Python later reduces
   them with scatter/reduction kernels. For global mode, accumulate partial
   reductions inside the Triton kernel into a compact per-wave or per-block
   buffer, then reduce that small buffer. This removes:

   - final `aw*` stores from `_wave_backward_uniform_kernel`
   - PyTorch scatter/reduce kernels in `_scatter_accum`
   - `aw0 + aw2` and `aw3 + aw4 + aw5` temporary add kernels

3. Consider recomputing some self-loop weights instead of storing them in
   `aw*` scratch. The kernel is memory-bound, not compute-bound. Recomputing
   exp/log weights in the Neumann passes may be cheaper than storing and
   reloading six full arrays, especially on the large leaf wave.

4. Specialize the no-split leaf wave. The leaf wave has `has_splits=False` and
   sparse leaf observations. It is also the largest wave. A specialized path can
   avoid `dts_r` logic, dense leaf terms, and unnecessary softmax branches.

## Bottleneck 2: CPU-driven pruning and dynamic compaction

The wave loop still performs host decisions on CUDA tensors:

```python
rhs_k = accumulated_rhs[ws:we].clone()
clade_max = rhs_k.abs().max(dim=1).values
active_mask = clade_max >= pruning_threshold
if not active_mask.any():
    ...
n_active = int(active_mask.sum().item())
active_idx = active_mask.nonzero(as_tuple=True)[0]
```

This shows up as hundreds of synchronization points:

```text
10 families: 308 cudaStreamSynchronize calls, 33.7 ms GPU idle/gap span
50 families: 352 cudaStreamSynchronize calls, 37.8 ms GPU idle/gap span
```

Pruning itself is worth keeping: without pruning, the 10-family backward went
from 72.676 ms to 101.283 ms. The problem is the host-controlled pruning
implementation.

### Next work

1. Move pruning into the kernels instead of compacting on the host.

   The simplest design is to pass `active_mask` to the self-loop, DTS backward,
   and `Pibar` VJP kernels. A program checks its row or split parent and returns
   immediately when inactive. This preserves most pruning benefit without
   `any()`, `.item()`, or `nonzero()` host synchronization.

2. Stop skipping whole waves on the CPU.

   Launching a tiny no-op grid for an inactive wave is cheaper than a blocking
   CPU decision. There are only 45 to 47 waves in these profiles.

3. Use a compact leaf-index representation so `_get_leaf_mask(...).any()` is
   eliminated at the same time.

Expected gain:

```text
10-family upper bound: up to 33.7 ms of GPU idle/gaps
50-family upper bound: up to 37.8 ms of GPU idle/gaps
```

The full upper bound is not achievable because some gaps include unavoidable
launch sequencing and allocator activity, but removing host compaction should be
one of the highest-leverage changes.

## Bottleneck 3: uniform Pibar VJP tree kernel

This kernel applies the VJP of uniform `Pibar` for cross-DTS child gradients.

```text
10 families: 10.337 ms total, 22.9% of kernel time
50 families: 39.555 ms total, 20.0% of kernel time
```

Resource profile:

```text
Memory throughput: 547 GB/s
DRAM peak: 54.3%
Compute peak: 52.0%
Issue active: 42.0%
Occupancy: 98.0%
Long scoreboard stalls: 60.2%
Barrier stalls: 13.5%
Atomic active cycles: 4.8%
```

The final atomic add into `accumulated_rhs` is not the main problem. The main
costs are:

1. Recomputing row max and denominator for each split side.
2. Writing `u_d` into `subtree_buf`.
3. Bottom-up tree reductions with barriers.
4. Global reads of `Pi_star`, child gradients, and tree metadata.

Grouping by child clade would reduce duplicate work, but only moderately:

```text
average split-side duplicate factor = 1.111x
best observed waves = roughly 1.33x to 1.38x
```

So child grouping is useful, but it is not enough by itself to transform the
profile.

### Next work

1. Prototype child-grouped `Pibar` VJP only after the bigger self-loop and
   pruning fixes.

   The grouped design should first reduce `grad_Pibar_l/r` by child clade, then
   run one uniform `Pibar` VJP per unique child. Based on the measured duplicate
   factor, the likely gain is around 5% to 12% of this kernel on this workload.

2. Reuse forward-side uniform `Pibar` statistics if available.

   The kernel recomputes row max, row sum, and ancestor denominators from
   `Pi_star`. If forward can store compact per-row statistics, backward can
   avoid part of the repeated work. The tradeoff is extra forward memory.

3. Investigate a species-tree-specific kernel for the denominator.

   The bottom-up gather path avoided ancestor scatter atomics, but it still
   does a full-tree pass per split side. A species-major layout may let multiple
   child rows share metadata and improve cache behavior.

## Bottleneck 4: DTS cross backward accumulation

This kernel directly accumulates Pi adjoints from cross-clade DTS terms.

```text
10 families: 6.040 ms total, 13.4% of kernel time
50 families: 29.441 ms total, 14.9% of kernel time
```

Resource profile:

```text
Memory throughput: 761 GB/s
DRAM peak: 75.6%
Compute peak: 25.8%
Issue active: 14.8%
Branch uniformity: 42.3%
Atomic active cycles: 24.8%
RED instructions: 3.39 M in representative launch
```

The atomics are real here. The direct accumulation path saves the
`grad_Pi_l/grad_Pi_r` materialization and PyTorch `index_add_`, but it pays in
atomic adds and divergence:

```text
wave_backward.py:815-856
```

The largest split waves are root-like and have many splits per parent:

```text
50-family wave 42: 42,155 splits over 247 parents
50-family wave 44: 24,229 splits over 39 parents
```

For those waves, a split-major atomic kernel may not be the best layout.

### Next work

1. Add a second path for high fanout waves.

   Use the current direct atomic path for low and medium fanout. For root-like
   waves, try a parent-grouped or child-grouped two-stage reduction:

   ```text
   stage 1: compute per-split/per-tile contributions into a compact scratch
   stage 2: reduce by child row and species, then add once to accumulated_rhs
   ```

   The switch criterion can be based on `n_splits / W` and measured duplicate
   child factor.

2. Avoid doing speciation scatter with atomics where tree injectivity is known.

   Within one split output row, child-species scatter is conflict-free. The
   direct accumulation kernel uses atomics because multiple splits can target
   the same child clade. A grouped path can recover normal stores for the local
   scatter part.

3. Keep the current direct path as the fallback.

   Previous tests showed it was faster overall than the materializing path, and
   NCU confirms the cost is meaningful but not catastrophic.

## Bottleneck 5: residual PyTorch kernels

At 50 families, residual PyTorch kernels are about 44 ms of GPU kernel time:

```text
scatter/gather reduce-add: 16.176 ms
reductions:                 8.767 ms
add/fill/abs/index:        19.329 ms
```

The most important sources are:

1. `_scatter_accum(...)` in `gpurec/core/backward.py:760-782`.
2. `clade_max = rhs_k.abs().max(...)` and active-mask kernels.
3. Dense leaf-mask allocation/fill.
4. Temporary additions such as `aw0 + aw2` and `aw3 + aw4 + aw5`.
5. Some remaining fallback paths for non-fused pieces.

### Next work

1. Fuse parameter-gradient reductions out of `_scatter_accum`.

   For the global mode benchmark, all family indices map to one parameter row.
   The fused wave kernel can write compact partial sums instead of full `[W, S]`
   contribution tensors.

2. Remove dense leaf-mask creation.

   This removes fill kernels and one host sync from `_get_leaf_mask`.

3. Replace active-mask compaction with in-kernel row skipping.

   This removes PyTorch `abs`, `max`, compare, `any`, `sum.item`, and `nonzero`
   from the critical path, or at least reduces them to asynchronous device-side
   work.

## Bottleneck 6: D2D copies

Copies are not the biggest wall-time issue, but they are an easy cleanup:

| Families | D2D/memcpy calls | D2D/memcpy bytes | D2D/memcpy time |
|---:|---:|---:|---:|
| 10 | 688 | 852.8 MB | 1.974 ms |
| 50 | 755 | 3,979.9 MB | 8.274 ms |

Almost all of this volume matches two wave-RHS copies per processed wave:

```python
rhs_k = accumulated_rhs[ws:we].clone()        # backward.py:682
...
wave_backward_uniform_fused(..., rhs_k.clone(), ...)
```

`wave_backward_uniform_fused` documents that `rhs` is overwritten as scratch.
But `rhs_k` is already a private clone of `accumulated_rhs[ws:we]`, and it is
not needed after the fused call. Passing `rhs_k` directly should remove roughly
half of the D2D copy volume:

```text
10 families: about 0.4 GB and about 1 ms upper bound
50 families: about 2.0 GB and about 4 ms upper bound
```

This is a small but low-risk first patch.

## What not to prioritize first

### Tensor cores

The hot kernels are not GEMM-like. They are dominated by:

```text
log/exp
row reductions
tree gathers
scatter/reduce-add
global scratch traffic
atomics
```

The NCU counters show memory bandwidth and synchronization stalls, not matrix
multiply throughput. Tensor cores could only help after a larger reformulation
of the uniform `Pibar` denominator or DTS reductions into dense or block-sparse
matrix operations. That is not the next best step for FP32 performance.

### Occupancy tuning alone

The largest self-loop kernel already has 82.1% achieved occupancy and 90% DRAM
throughput. More occupancy does not fix a 5.87 GB representative-launch memory
stream. The right target is fewer bytes.

## Recommended plan

### P0: remove obvious copies and dense leaf tensors

1. Pass `rhs_k` directly into `wave_backward_uniform_fused` instead of
   `rhs_k.clone()`.
2. Replace dense `leaf_wt` with compact per-row leaf species indices.
3. Validate gradients against the current implementation and finite differences.
4. Re-profile 10 and 50 families with Nsight Systems.

Expected gain:

```text
copies: 1 to 4 ms depending on batch size
leaf tensor removal: likely several ms on 50 families, plus lower self-loop bandwidth
```

### P1: device-resident pruning

1. Keep pruning semantics, but stop branching on CUDA scalars from Python.
2. Pass active masks to the Triton kernels and skip inactive rows or splits
   inside the kernels.
3. Launch all waves in the fixed schedule.
4. Re-measure the no-op cost versus the removed sync gaps.

Expected gain:

```text
10-family upper bound: 33.7 ms GPU idle/gap span
50-family upper bound: 37.8 ms GPU idle/gap span
```

The real gain will be lower, but this should materially improve small and
medium batches.

### P2: fuse parameter-gradient reductions

1. Stop returning six full `[W, S]` `aw*` contribution tensors when the caller
   only needs reductions.
2. For global mode, emit compact per-block partials from Triton and reduce them
   in one follow-up kernel.
3. Later generalize to genewise/specieswise modes.

Expected gain:

```text
50-family residual PyTorch scatter/reduction/add/fill budget: about 44 ms
self-loop final-store traffic also decreases
```

This is the biggest structural cleanup after pruning.

### P3: specialize high-fanout split waves

1. Keep the current split-major DTS path for ordinary waves.
2. Add a parent-grouped path for high `n_splits / W` root-like waves.
3. Benchmark the switch threshold with Nsight Systems.

Expected gain:

```text
targets: 29.4 ms DTS backward accum + 12.4 ms DTS forward recompute at 50 families
likely gain: workload dependent, probably single-digit to low-double-digit ms
```

### P4: revisit uniform Pibar VJP grouping

1. Reduce `grad_Pibar_l/r` by child clade.
2. Run the uniform `Pibar` VJP once per unique child.
3. Keep the current tree gather as fallback.

Expected gain:

```text
average duplicate factor: 1.111x
likely gain: 5% to 12% of the Pibar VJP kernel, not a first-order win
```

## Summary

For the FP32 fused backward, the primary bottleneck is memory traffic, especially
in `_wave_backward_uniform_kernel`. On the largest representative launch it
moves 5.87 GB, reaches 90% DRAM utilization, and spends 54.6% of warp cycles in
long scoreboard stalls. That is the clearest sign that we should reduce bytes,
not chase tensor cores or occupancy.

The second bottleneck is still Python/PyTorch orchestration around the Triton
kernels. On the 50-family batch there are 37.8 ms of GPU idle/gaps and about
44 ms of residual PyTorch kernel work. The highest-value next implementation
work is therefore:

1. remove duplicate RHS copies and dense leaf tensors,
2. make pruning device-resident,
3. fuse parameter-gradient reductions,
4. specialize high-fanout DTS waves,
5. only then optimize uniform `Pibar` VJP grouping.

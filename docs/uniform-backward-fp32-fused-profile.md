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
GPUREC_COMPACT_PIBAR_SCRATCH=1
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

## Optimization Update: Tested Proposals

After this profile, seven production changes were implemented in the fused
uniform backward path. Several additional Bottleneck 1 experiments were
implemented behind environment flags and left disabled because they were slower
or only diagnostic.

| Commit | Change | Main effect |
|---|---|---|
| `a06aac1` | Use `leaf_species_index[C]` plus `[S]` `log_pS` in the Triton kernel | Avoids constructing and reading dense `[W,S]` leaf tensors |
| `ed6b5b7` | Treat `rhs` as read-only and use an internal ping-pong scratch buffer | Removes one full `[W,S]` device-to-device copy per processed wave |
| `05d7c48` | Allocate `spec_buf` with `empty` instead of `zeros` | Removes redundant scratch zero-fill launches |
| `8077a50` | Use plain reductions when `G == 1` | Replaces indexed `scatter_add_` with direct reductions for global-mode gradients |
| `dcda611` | Atomically accumulate global wave param reductions inside `_wave_backward_uniform_kernel` | Removes six full `[W,S]` contribution tensors from the hot reduction path |
| `fa88780` | Default cross-family `max_wave_size` to 32,768 | Keeps large batches below memory limits without slowing the 50-family case |
| `980ecac` | Store compact Pibar VJP coefficient scratch by default | Replaces `pibar_wt`, `inv_denom`, and `p_prime` scratch with `pibar_wt * inv_denom` plus one recomputed `p_prime` |

Correctness was checked by comparing the old dense-leaf path
(`GPUREC_BACKWARD_LEAF_INDEX=0`) with the new leaf-index path and by running the
autograd finite-difference bridge tests:

| Check | Result |
|---|---:|
| 3 families, loss difference | 0 |
| 3 families, max gradient abs diff | 9.16e-5 |
| 3 families, max gradient relative diff | 2.45e-7 |
| 10 families, loss difference | 0 |
| 10 families, max gradient abs diff | 0 |
| `tests/gradients/test_autograd_bridge.py` | 15 passed |
| `GPUREC_FUSED_WAVE_PARAM_ACCUM=0 tests/gradients/test_autograd_bridge.py` | 15 passed |
| `GPUREC_COMPACT_PIBAR_SCRATCH=0 tests/gradients/test_autograd_bridge.py` | 15 passed |
| `tests/kernels/test_uniform_cross_pibar_vjp_kernel.py` | 2 passed |

The fused parameter accumulation path changes fp32 summation order because it
uses RED operations inside the Triton kernel. Against the previous reduction
path, the largest observed 10-family gradient relative difference was
`2.85e-4`; the loss was unchanged and the finite-difference bridge tests passed.

Normal backward-only timing, same setup as above:

| State | 10 families | 50 families | 50-family peak memory |
|---|---:|---:|---:|
| Baseline in this document | 73.252 ms | 230.795 ms | 16.806 GB |
| Leaf index only | 68.373 ms | 225.373 ms | 16.161 GB |
| Leaf index + read-only RHS | 67.984 ms | 218.372 ms | 16.161 GB |
| Leaf index + read-only RHS + empty scratch | 68.160 ms | 214.755 ms | 16.162 GB |
| Plus direct global reductions | 65.428 ms | 205.489 ms | 16.162 GB |
| Plus fused wave param accumulation | 60.614 ms | 181.570 ms | 14.865 GB |
| Plus default `max_wave_size=32768` | 61.095 ms | 181.255 ms | 11.091 GB |
| Plus compact Pibar scratch | 59.462 ms | 174.174 ms | 10.567 GB |

The final default state is about 18.8% faster on the 10-family benchmark and
24.5% faster on the 50-family benchmark, while cutting 50-family peak memory by
about 37.1%. With the 32,768 wave cap, a 100-family backward run that OOMed in
the uncapped layout completed with `maxW=32768`, peak memory `18.73 GB`, and
best measured backward time `326.207 ms` before the compact-scratch change.

Nsight Systems on the current 50-family backward interval, compared with the
same code run under `GPUREC_COMPACT_PIBAR_SCRATCH=0`:

| Metric | Compact off | Compact default |
|---|---:|---:|
| Captured backward CUDA time | 198.034 ms | 188.684 ms |
| `_wave_backward_uniform_kernel` total | 50.154 ms | 41.119 ms |
| Largest `_wave_backward_uniform_kernel` launch | 9.658 ms | 7.655 ms |
| `_uniform_cross_pibar_vjp_tree_kernel` total | 39.777 ms | 38.633 ms |
| `_dts_cross_backward_accum_kernel` total | 29.665 ms | 29.659 ms |
| `_dts_fused_kernel` total | 12.153 ms | 12.156 ms |
| GPU memops time | 5.424 ms | 5.422 ms |
| Kernel launches | 2,982 | 2,982 |
| Copies | 690 | 690 |

The compact scratch change is isolated: it changes only the self-loop kernel
bucket. DTS, cross-Pibar VJP, memops, launch count, and copy count are
essentially unchanged. The top compact-default kernel totals are:

| Component | Optimized 50-family kernel time |
|---|---:|
| `_wave_backward_uniform_kernel` | 41.119 ms |
| `_uniform_cross_pibar_vjp_tree_kernel` | 38.633 ms |
| `_dts_cross_backward_accum_kernel` | 29.659 ms |
| `_dts_fused_kernel` | 12.156 ms |
| PyTorch reductions | 5.482 ms |
| PyTorch abs kernels | 4.999 ms |

Nsight Compute on the optimized largest 10-family leaf wave
(`_wave_backward_uniform_kernel`, `W=16645`, `S=1999`):

| Metric | Baseline | Optimized |
|---|---:|---:|
| Duration | 6.503 ms | 5.128 ms |
| DRAM read bytes | 3.384 GB | 2.451 GB |
| DRAM write bytes | 2.485 GB | 1.602 GB |
| Total DRAM bytes | 5.869 GB | 4.053 GB |
| DRAM throughput | 90.0% | 78.5% |
| Compute throughput | 22.8% | 28.7% |
| Issue slots busy | 12.3% | 15.9% |
| Achieved occupancy | 82.1% | 82.4% |
| Registers per thread | 48 | 48 |
| L1/TEX hit rate | 57.4% | 55.7% |
| L2 hit rate | 74.4% | 82.2% |
| Global RED accesses | 0 | 10.75 M |

The kernel is still DRAM-bandwidth bound after the optimization. The leaf-index
and fused-accumulation changes removed roughly 1.82 GB of DRAM traffic from the
largest leaf-wave launch, explaining the local 21% kernel speedup. The RED
operations are visible, but they are cheaper than writing six full contribution
tensors and reducing them with PyTorch.

After the compact Pibar scratch change, a fresh NCU pass on the same 10-family
leaf wave showed:

| Metric | Previous default | Compact scratch default |
|---|---:|---:|
| DRAM read bytes | 2.431 GB | 1.483 GB |
| DRAM write bytes | 1.601 GB | 1.211 GB |
| Total DRAM bytes | 4.032 GB | 2.694 GB |
| DRAM throughput | 75.0% | 61.5% |
| Compute throughput | 27.5% | 31.8% |
| Issue slots busy | 15.2% | 18.0% |
| Achieved occupancy | 82.0% | 81.8% |
| L1/TEX bytes | 27.265 GB | 25.190 GB |
| L2 bytes | 13.784 GB | 12.459 GB |
| Global RED accesses | 10.75 M | 10.75 M |

The compact path removes about `1.34 GB` of DRAM traffic from this one launch.
Occupancy is unchanged; the improvement is from issuing fewer memory
transactions, not from launching more resident work.

Other tested proposals:

| Proposal | 10-family result | 50-family result | Decision |
|---|---:|---:|---|
| Disable pruning | 91.015 ms | 324.348 ms | Reject; host pruning still avoids too much work |
| Pruning threshold `1e-8` | 65.203 ms | 218.066 ms | Reject; slower at 50 families |
| Pibar VJP `tree` | 61.135 ms | 183.883 ms | Keep current default |
| Pibar VJP ancestor columns | 85.586 ms | 341.054 ms | Reject |
| Pibar VJP PyTorch fallback | 114.400 ms | 481.529 ms | Reject |
| DTS backward direct accumulation | 61.202 ms | 180.732 ms | Keep current default |
| DTS backward materialized child grads | 63.098 ms | 187.895 ms | Reject |
| Disable kernelized backward DTS | 148.823 ms | 649.828 ms | Reject |
| `max_wave_size=32768` | 60.483 ms | 181.742 ms | Accept as default cap |
| `max_wave_size=16384` | not retested | 184.092 ms | Reject for default; useful only if memory constrained |
| `max_root_wave_size=32` | not retested | 194.434 ms | Reject; too many root-wave launches |
| no-split final VJP shortcut | 61.828 ms | 184.176 ms | Reject for default; algebra valid but slower |
| recompute Pibar denominator scratch | 60.090 ms | 174.754 ms | Diagnostic only; compact coefficient is better overall |
| compact Pibar coefficient scratch | 59.462 ms | 174.174 ms | Accept as default; best memory reduction and stable correctness |
| leaf-logp hit-only/scalar specialization | 62.137 ms | 182.300 ms | Reject; leaf load was not the limiting traffic |
| compact scratch only on no-split waves | 65.886 ms | 176.708 ms | Reject; split waves also need the compact scratch win |

The no-split final VJP shortcut was the targeted Bottleneck 1 subtask. An
explorer agent checked the algebra: when `has_splits=false`, `w_L=1`, so the
precomputed Neumann scratch weights satisfy
`leaf_wt = 1 - diag_wt - pibar_wt - sl1_wt - sl2_wt`. The D/Ebar split is
`exp2(DL_const) / (exp2(DL_const) + exp2(Ebar))` because the shared `Pi` factor
cancels from terms 0 and 1. The mapping caveat is that `sl1_wt` contributes to
`grad_E_s2`, while `sl2_wt` contributes to `grad_E_s1`.

The shortcut was implemented behind `GPUREC_FAST_NOSPLIT_PARAM_ACCUM=1` and
validated with the autograd bridge in both modes, but profiling showed why it
lost. On the representative 10-family leaf wave, default fused accumulation
used about `2.451 GB` DRAM reads and `1.602 GB` DRAM writes; the no-split
shortcut used `2.615 GB` reads and `1.632 GB` writes. In other words, reading
the stored scratch weights was less cache-friendly than recomputing the final
softmax terms, so the shortcut increased DRAM traffic and was left disabled.
After disabling it, the default path rechecked at `61.106 ms` for 10 families
and `180.791 ms` for 50 families, so the recorded experiment does not affect
production behavior unless `GPUREC_FAST_NOSPLIT_PARAM_ACCUM=1` is set.

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

### Progress on the four proposals

For proposals 3 and 4, three subagents were used and this document was updated
from the supervisor pass. Pasteur analyzed proposal 3 and proposed the compact
Pibar coefficient variant. Avicenna analyzed proposal 4 and pointed out that a
source-only no-split kernel clone would remove little because `has_splits` and
`USE_LEAF_INDEX` are already Triton constexprs. Leibniz reconstructed the
reproducible CUDA-event, Nsight Systems, and Nsight Compute harnesses used for
the measurements below.

| Proposal | Status | Benchmark result | Decision |
|---|---|---:|---|
| 1. Compact leaf species index instead of dense `leaf_term_wt` | Implemented in `a06aac1` | 10 families: `73.252 ms` to `68.373 ms`; 50 families: `230.795 ms` to `225.373 ms`; 50-family peak memory: `16.806 GB` to `16.161 GB` | Keep |
| 2. Avoid full parameter-gradient contribution tensors | Implemented in `dcda611` after direct reductions in `8077a50` | 10 families: `65.428 ms` to `60.614 ms`; 50 families: `205.489 ms` to `181.570 ms`; 50-family peak memory: `16.162 GB` to `14.865 GB` | Keep |
| 3. Recompute self-loop weights instead of storing/reloading scratch | Tested with two variants: full denominator recompute and compact Pibar coefficient scratch | Compact default: 10 families `59.462 ms`, 50 families `174.174 ms`; peak `10.567 GB` at 50 families | Keep compact coefficient scratch |
| 4. Specialize the no-split leaf wave | Tested three low-risk specializations: final-VJP shortcut, leaf-logp hit-only/scalar loads, and compact scratch restricted to no-split waves | Best no-split-only variant: 50 families `176.708 ms`; still slower than compact-all-waves `174.174 ms` | Reject for now |

Proposal 1 replaced the dense leaf tensor with `leaf_species_idx[C]` plus
`leaf_logp[S]`. Instead of reading a mostly `-inf` `[W, S]` `leaf_term_wt`
tile, the kernel computes the leaf term locally:

```text
t5 = leaf_logp[s] if leaf_species_idx[row] == s else -inf
```

This removes dense allocation/fill work in Python and removes two large
leaf-term global reads from the self-loop kernel. The measured gain was modest
but robust because it only affects the leaf-like rows; it also reduced
50-family peak memory by about `645 MB`.

Proposal 2 moved global-mode parameter-gradient accumulation into
`_wave_backward_uniform_kernel`. Before this change, the kernel wrote six full
`[W, S]` contribution tensors and Python reduced them later with PyTorch
scatter/reduction/add kernels. The current default uses Triton RED operations
to accumulate:

```text
grad_log_pD += sum(term0)
grad_log_pS += sum(term3 + term4 + leaf)
grad_E      += term0 + term2
grad_Ebar   += term1
grad_E_s1   += term4
grad_E_s2   += term3
grad_mt     += term2
```

The RED operations are visible in NCU, but they are cheaper than writing and
then rereading full contribution tensors. On the optimized largest 10-family
leaf wave, total DRAM traffic dropped from `5.869 GB` to `4.053 GB`, and the
launch time dropped from `6.503 ms` to `5.128 ms`.

Proposal 3 was tested with the same three-agent workflow. Pasteur identified a
middle path that is better than fully recomputing every self-loop weight. The
old Neumann scratch stored three Pibar-related arrays:

```text
aw1 = pibar_wt
aw2 = inv_denom
aw3 = p_prime
```

The accepted compact path stores only:

```text
aw1 = pibar_wt * inv_denom
```

Then Neumann pass A computes:

```text
u_d = term * aw1
```

and pass B recomputes only:

```text
p_prime = exp2(Pi[row, s] - row_max)
```

This saves two full `[W,S]` scratch allocations when in-kernel parameter
accumulation is enabled, removes two precompute stores, removes several Neumann
loads, and adds one `Pi` load plus one `exp2` in pass B. That trade is favorable
because the kernel is memory-bound and `Pi` has better locality than the
separate scratch streams. The more aggressive
`GPUREC_RECOMPUTE_PIBAR_DENOM=1` variant also passed correctness, but the
compact coefficient path has the cleaner memory reduction and is now the
default. `GPUREC_COMPACT_PIBAR_SCRATCH=0` restores the old behavior.

Proposal 4 was tested as far as the low-risk variants justified. The first
implemented shortcut used the fact that no-split waves have `w_L=1`, so the
final parameter VJP can recover:

```text
leaf_wt = 1 - diag_wt - pibar_wt - sl1_wt - sl2_wt
```

and split the combined diagonal weight between the D and Ebar terms using:

```text
D_frac = exp2(DL_const) / (exp2(DL_const) + exp2(Ebar))
```

This algebra was checked by an explorer agent and validated with
`tests/gradients/test_autograd_bridge.py` in both default and opt-in modes.
However, NCU showed that it increased the representative leaf-wave DRAM traffic
from `4.053 GB` to `4.246 GB`. The reason is that the shortcut replaces some
local recomputation with extra global scratch reads, and the access pattern is
worse for this already memory-bound kernel. Therefore the shortcut is disabled
by default behind `GPUREC_FAST_NOSPLIT_PARAM_ACCUM=1`.

Two other no-split/leaf specializations were tested. `GPUREC_LEAF_HIT_ONLY_LOGP`
with `GPUREC_SCALAR_LEAF_LOGP` avoids loading leaf log-probabilities for every
species lane, but this did not help because those loads are tiny compared with
the Neumann scratch traffic. Restricting compact Pibar scratch to no-split waves
with `GPUREC_COMPACT_PIBAR_SCRATCH=leaf` was also slower than applying compact
scratch to every fused wave. A source-only no-split clone would remove little:
`has_splits` and `USE_LEAF_INDEX` are already Triton constexprs, so the compiler
already specializes those branches per launch. A future separate leaf kernel
would only be worthwhile if it also changes the Neumann scratch layout; that
overlaps proposal 3, where the all-wave compact scratch path won.

## Bottleneck 2: CPU-driven pruning and dynamic compaction

The wave loop still performs host decisions on CUDA tensors:

```python
rhs_k = accumulated_rhs[ws:we]  # fused uniform default; clone fallback is opt-in
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

### Initial hypotheses

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

Pre-test upper bound:

```text
10-family upper bound: up to 33.7 ms of GPU idle/gaps
50-family upper bound: up to 37.8 ms of GPU idle/gaps
```

The full upper bound was never expected to be achievable because some gaps
include unavoidable launch sequencing and allocator activity. The experiment
below shows the more important issue: host pruning was also skipping whole
waves, not only synchronizing.

### Progress: tested device-resident pruning

We used the same three-agent workflow as for Bottleneck 1:

| Agent | Target | Result |
|---|---|---|
| Agent 1 | Device-side `active_mask` passed into Triton kernels | Correct, but fixed-schedule version is slower than host wave skipping |
| Agent 2 | Remove CPU whole-wave decisions and quantify sync overhead | Isolated sync cost is real, but CPU skipping avoids more GPU work |
| Agent 3 | Remove `_get_leaf_mask(...).any()` / compact leaf representation | Already solved on the default fused path by `leaf_species_index[C]` |

The main prototype is behind environment flags and is **not enabled by
default**:

```bash
GPUREC_BACKWARD_NO_CPU_PRUNING=1  # no per-wave host any/sum/nonzero, no stats sync
GPUREC_DEVICE_PRUNING=1           # same masked kernels, with deferred pruning stats
GPUREC_DENSE_LEAF_MASK_FROM_INDEX=1  # fallback-only dense leaf mask experiment
```

The device-pruning path computes the same row activity mask on the GPU, passes
it into `_wave_backward_uniform_kernel`, `_dts_cross_backward_accum_kernel`,
`_uniform_cross_pibar_vjp_*`, and `dts_fused`, and makes inactive programs
return early or write zero/`-inf` outputs. This removes the repeated host
`any()`, `sum().item()`, and `nonzero()` decisions from the fused CUDA path.

Correctness checks passed:

| Check | Result |
|---|---:|
| `GPUREC_BACKWARD_NO_CPU_PRUNING=1 tests/gradients/test_autograd_bridge.py` | 15 passed |
| `GPUREC_BACKWARD_NO_CPU_PRUNING=1 tests/kernels/test_uniform_cross_pibar_vjp_kernel.py` | 2 passed |
| `GPUREC_DEVICE_PRUNING=1 tests/gradients/test_autograd_bridge.py` | 15 passed |
| `GPUREC_DEVICE_PRUNING=1 tests/kernels/test_uniform_cross_pibar_vjp_kernel.py` | 2 passed |
| `GPUREC_DEVICE_PRUNING=1 GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=flat tests/gradients/test_autograd_bridge.py` | 15 passed |
| 10-family parity, loss diff | 0 |
| 10-family parity, max theta grad rel diff | `2.75e-7` |
| 50-family parity, loss diff | 0 |
| 50-family parity, max theta grad rel diff | `3.04e-7` |

Normal backward-only timings after adding active-aware DTS construction:

| State | 10 families mean / median | 50 families mean / median | Verdict |
|---|---:|---:|---|
| Default host pruning | `59.927 / 59.739 ms` | `174.785 / 174.387 ms` | Keep default |
| `GPUREC_BACKWARD_NO_CPU_PRUNING=1` | `66.862 / 59.250 ms` | `178.648 / 177.943 ms` | Rejected |
| `GPUREC_DEVICE_PRUNING=1` | `58.969 / 58.911 ms` | `178.303 / 177.588 ms` | Rejected |

The 10-family no-CPU mean includes one outlier (`97.856 ms`); the median shows
that fixed-schedule device pruning can match host pruning for small batches. At
50 families, where the production workload is more relevant, both no-CPU paths
are still slower by about `3.5 ms` to `3.9 ms`.

Nsight Systems, 10-family default host pruning vs `GPUREC_DEVICE_PRUNING=1`:

| Metric | Host pruning | Device pruning |
|---|---:|---:|
| Captured backward CUDA-event time | `70.513 ms` | `70.432 ms` |
| `cudaStreamSynchronize` calls | 273 | 213 |
| D2H copies | 210 | 150 |
| CUDA kernel launches | 2,171 | 2,314 |
| `_wave_backward_uniform_kernel` total | `9.334 ms` | `9.206 ms` |
| `_uniform_cross_pibar_vjp_tree_kernel` total | `10.488 ms` | `10.313 ms` |
| `_dts_cross_backward_accum_kernel` total | `6.095 ms` | `6.419 ms` |
| `_dts_fused_kernel` total | `2.389 ms` | `2.634 ms` |

Nsight Systems, 50-family default host pruning vs `GPUREC_DEVICE_PRUNING=1`:

| Metric | Host pruning | Device pruning |
|---|---:|---:|
| Captured backward CUDA-event time | `190.652 ms` | `194.487 ms` |
| `cudaStreamSynchronize` calls | 319 | 249 |
| D2H copies | 260 | 190 |
| CUDA kernel launches | 2,982 | 3,117 |
| `_wave_backward_uniform_kernel` total | `41.409 ms` | `39.986 ms` |
| `_uniform_cross_pibar_vjp_tree_kernel` total | `39.657 ms` | `38.163 ms` |
| `_dts_cross_backward_accum_kernel` total | `29.653 ms` | `31.257 ms` |
| `_dts_fused_kernel` total | `12.158 ms` | `13.287 ms` |

The result is technically useful but not a production improvement. The host
pruning path is expensive because it synchronizes, but it also skips entire
waves. The fixed-schedule device path removes about 60 to 70 D2H/sync events,
yet it still launches kernels for waves that the CPU path never launched. Even
with early returns, those launches still allocate outputs, clear buffers, run
small no-op programs, and create extra reduction/stat kernels. At 50 families,
the extra scheduled work is larger than the sync savings.

The active-aware DTS change was important: before it, 50-family no-CPU pruning
was about `181.5 ms`; after masking `dts_fused`, it was about `178.6 ms`. The
remaining regression is not from the self-loop or Pibar kernels themselves
(`_wave_backward_uniform_kernel` and `_uniform_cross_pibar_vjp_tree_kernel`
both got slightly faster in the 50-family trace). It is mainly from fixed
scheduling overhead around split waves: more `_dts_cross_backward_accum_kernel`
work, more `_dts_fused_kernel` launches, and extra launch/stat bookkeeping.

So the next version should not be "always launch every wave with an active
mask." It should preserve CPU pruning's scheduling property without CPU scalar
decisions. The likely design is a device-built active worklist:

1. Compute active row counts and active split-parent counts on device.
2. Compact active rows/splits into reusable work buffers with prefix sums.
3. Launch self-loop, DTS backward, and Pibar VJP on compacted active work only.
4. Either drop per-iteration pruning counters from the hot path or update them
   asynchronously and read them only for diagnostics.

That would remove host syncs while also avoiding the fixed-schedule launch and
no-op work that made this prototype slower.

### Progress: leaf-mask sync proposal

The default global CUDA path already avoids `_get_leaf_mask(...).any()`:
`leaf_species_index[C]` is passed directly to the fused Triton wave kernel, so
no dense `[W, S]` leaf mask is materialized in the hot path.

Forcing the legacy dense fallback (`GPUREC_BACKWARD_LEAF_INDEX=0`) costs about
`12.7 ms` at 50 families and raises peak memory by about `0.26 GB`. An opt-in
fallback, `GPUREC_DENSE_LEAF_MASK_FROM_INDEX=1`, was tested to construct the
dense mask from `leaf_species_index` without the CUDA scalar `.any()` branch.
It reduced the 10-family fallback profile from 308 to 275
`cudaStreamSynchronize` calls and D2H copies from 244 to 210, but runtime did
not improve because it still materializes the dense `[W, S]` mask. This remains
disabled; the correct production path is the existing fused leaf-index path.

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

### Bottleneck 3 proposal workflow

The three-agent workflow was reused for this bottleneck:

| Agent | Proposal | Implementation flag | Result |
|---|---|---|---|
| Agent 1 | Group split-side `Pibar` adjoints by child clade | `GPUREC_GROUPED_CROSS_PIBAR_VJP=1` and optionally `GPUREC_GROUPED_CROSS_PIBAR_REDUCE_IMPL=triton` | Correct, slower |
| Agent 2 | Reuse forward-side `Pibar` row information | `GPUREC_REUSE_FORWARD_PIBAR_STATS=1` | Correct, small gain on 10 families, neutral/slower on 50 |
| Agent 3 | Replace ancestor-list denominator with a species-tree prefix pass | `GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=prefix` | Correct, slower |

All paths are default-off. The production default remains the existing
bottom-up tree VJP kernel.

Correctness checks:

| Command | Result |
|---|---:|
| `GPUREC_CROSS_PIBAR_ROW_STATS=1 pytest -q tests/gradients/test_autograd_bridge.py tests/kernels/test_uniform_cross_pibar_vjp_kernel.py -q` | 17 passed |
| `GPUREC_REUSE_FORWARD_PIBAR_STATS=1 pytest -q tests/gradients/test_autograd_bridge.py -q` | 15 passed |
| `GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=prefix pytest -q tests/gradients/test_autograd_bridge.py tests/kernels/test_uniform_cross_pibar_vjp_kernel.py -q` | 17 passed |
| `GPUREC_GROUPED_CROSS_PIBAR_VJP=1 GPUREC_GROUPED_CROSS_PIBAR_REDUCE_IMPL=triton pytest -q tests/gradients/test_autograd_bridge.py tests/kernels/test_uniform_cross_pibar_vjp_kernel.py -q` | 17 passed |

Real-workload parity was also checked by the agents on 10 and 50 families.
The loss matched exactly in these runs. The largest reported theta-gradient
relative errors were `3.18e-7` for forward-stat reuse, `2.28e-7` for child
grouping, and `1.52e-7` for the prefix-denominator kernel.

Supervisor benchmark conditions:

```bash
FAMS={10,50} REPS=9 WARMUPS=5 python /tmp/gpurec_profile/bench_uniform_backward.py
```

Backward-only CUDA event timings:

| Variant | 10 families median | 10 families min | 50 families median | 50 families min | Peak alloc, 50 families |
|---|---:|---:|---:|---:|---:|
| Current default tree VJP | 47.769 ms | 47.662 ms | 162.704 ms | 161.829 ms | 10.567 GB |
| Forward `row_max` + `Pibar` denominator reuse | 47.290 ms | 46.898 ms | 166.956 ms | 162.592 ms | 10.568 GB |
| Backward row-stat precompute | 48.876 ms | 48.201 ms | 217.632 ms | 163.776 ms | 10.570 GB |
| Child-grouped VJP, Triton reduction | 50.138 ms | 49.388 ms | 177.791 ms | 175.892 ms | 10.575 GB |
| Prefix denominator kernel | 50.185 ms | 49.654 ms | 180.791 ms | 178.578 ms | 10.567 GB |

The 50-family row-stat-precompute run was bimodal. Nsight Systems gives the
more useful kernel-level explanation: it reduced the tree VJP kernel by only
`0.848 ms`, then added a `2.690 ms` precompute kernel, so it is not a useful
default even ignoring the timing jitter.

### Proposal 1: child-grouped uniform Pibar VJP

The idea was to reduce `grad_Pibar_l/r` by child clade first, then run one
uniform `Pibar` VJP per unique child instead of one per split side. Two variants
were tested:

1. A simple PyTorch path using `cat`, `unique`, `index_add_`, and a grouped VJP.
2. A cached-metadata Triton path using a custom `_group_cross_pibar_grad_kernel`
   followed by `_uniform_cross_pibar_vjp_tree_grouped_kernel`.

Nsight Systems, 50 families:

| Kernel group | Default | Grouped Triton |
|---|---:|---:|
| Uniform `Pibar` VJP | 37.784 ms | 32.119 ms |
| Group-reduce split-side gradients | 0 ms | 16.102 ms |
| Net `Pibar` VJP path | 37.784 ms | 48.221 ms |
| Backward interval in profiled run | 175.673 ms | 190.433 ms |

Grouping therefore does reduce the VJP kernel itself by `5.665 ms` at 50
families, but the required reduction pass costs `16.102 ms`. The duplicate
split-side factor is only about `1.111x` overall, so a separate full
`[2 * n_ws, S]` reduction pass is too much work.

The PyTorch grouped path is worse for the same reason plus extra allocation and
library overhead. It also raises 50-family peak allocation from about `10.567 GB`
to about `11.433 GB` in the agent run because it materializes `cat`,
`unique`/inverse metadata, and a grouped gradient buffer.

Recommendation: keep this disabled. A future version would have to fuse child
grouping into DTS cross backward so `grad_Pibar_l/r` is never materialized and
then reduced. Even then, the measured `1.11x` duplicate factor limits the upside.

### Proposal 2: reuse forward-side Pibar statistics

Two forms of row-stat reuse were tested.

The first precomputes backward row stats with `_pibar_row_stats_kernel`, then
loads `row_max` and `row_sum` in `_uniform_cross_pibar_vjp_tree_kernel`. This is
correct, but the extra full pass over `Pi_star` is not paid back. In Nsight
Systems at 50 families:

```text
default _uniform_cross_pibar_vjp_tree_kernel: 37.784 ms
row-stats _uniform_cross_pibar_vjp_tree_kernel: 36.936 ms
_pibar_row_stats_kernel: 2.690 ms
```

Net kernel time therefore regresses by about `1.842 ms`.

The second form stores only the final forward `row_max[C]` tensor and reuses
the already-materialized `Pibar` value to recover the inverse denominator:

```text
inv_denom[s] = 2 ** (row_max[row] + mt[s] - Pibar[row, s])
```

This skips both the backward row-max/row-sum pass and the ancestor-denominator
walk. It still needs the subtree correction because that depends on the
incoming adjoint.

Memory cost in fp32 is small for `row_max`:

| Families | C | `row_max` | `row_max + row_sum` | Full denom/inv-denom |
|---:|---:|---:|---:|---:|
| 10 | 66,530 | 0.266 MB | 0.532 MB | 532 MB |
| 50 | 321,930 | 1.288 MB | 2.575 MB | 2.57 GB |
| 100 | 635,372 | 2.541 MB | 5.083 MB | 5.08 GB |

This path helps small waves but not the 50-family schedule:

```text
10 families, event median: 47.769 ms -> 47.290 ms
50 families, event median: 162.704 ms -> 166.956 ms
50 families, Nsight Pibar VJP: 37.784 ms -> 39.157 ms
```

The agent's 10-family NCU profile explains the mixed result:

```text
L1/TEX bytes: 5.43 GB -> 2.75 GB
DRAM read: 201.7 MB -> 251.1 MB
DRAM write: 76.7 MB -> 72.5 MB
Issue active: 33.2% -> 23.4%
Occupancy: 73.3% -> 96.4%
```

So this does remove a large amount of L1/TEX ancestor-gather work, but it
replaces mostly cache-resident ancestor reads with extra global `Pibar` and
`mt` reads. At 50 families, that additional DRAM pressure cancels the arithmetic
savings.

Recommendation: leave default-off. It may be useful for smaller batches or a
future layout where `Pibar` is already hot in cache, but it is not a default
win for the 50-family occupancy-optimized schedule.

### Proposal 3: species-tree prefix denominator

The prefix prototype replaces the per-species padded-ancestor denominator loop
with a top-down pass over the species tree:

```text
prefix[s] = sum_{a in ancestors(s)} p[a]
denom[s] = row_sum - prefix[s]
```

It then reuses the existing bottom-up subtree VJP. This is controlled by
`GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=prefix`.

The result is correct but slower:

```text
50 families, event median: 162.704 ms -> 180.791 ms
50 families, Nsight Pibar VJP: 37.784 ms -> 56.475 ms
```

Representative NCU counters from the agent show the resource tradeoff:

| Metric | Default tree | Prefix denominator |
|---|---:|---:|
| DRAM read | 587.2 MB | 727.8 MB |
| DRAM write | 259.4 MB | 258.6 MB |
| L1/TEX traffic | 15.36 GB | 12.48 GB |
| L2 traffic | 4.56 GB | 7.07 GB |
| SM throughput | 51.1% | 47.0% |
| Issue active | 40.6% | 24.1% |
| Long scoreboard stalls | 62.5% | 37.5% |
| Barrier stalls | 13.6% | 39.3% |

The prefix kernel does reduce long-scoreboard stalls and L1/TEX traffic, which
means the original ancestor gathers really are part of the cost. But the
top-down prefix pass adds level barriers and scratch traffic. The result is
lower issue activity, higher L2/DRAM pressure, and much worse barrier stalls.

Recommendation: reject this species-major/prefix direction for now. The current
row-major split-side kernel has ugly gathers, but those gathers are often
cache-resident and keep the rest of the row work contiguous. Sharing species
metadata does not share the row-specific `Pi` distribution, which is the data
that dominates the VJP.

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

### Tested proposals

Two default-off alternatives were implemented:

| Path | Env flag | Idea |
|---|---|---|
| Child-grouped two-stage accumulation | `GPUREC_DTS_BACKWARD_ACCUM_IMPL=grouped` | Materialize local `grad_Pi_l/r`, reduce by child row, then add one row per child to `accumulated_rhs`. |
| Child-unique no-atomic accumulation | `GPUREC_DTS_BACKWARD_ACCUM_IMPL=noatomic` | For selected high-fanout waves whose child rows are unique, replace atomic add with load/add/store. |

Both paths keep the direct atomic kernel as the default fallback. The grouped
path also has aliases `GPUREC_GROUPED_DTS_BACKWARD_ACCUM=1` and thresholds
`GPUREC_GROUPED_DTS_BACKWARD_MIN_SPLITS` / `GPUREC_GROUPED_DTS_BACKWARD_MIN_FANOUT`.
The no-atomic path has alias `GPUREC_NOATOMIC_DTS_BACKWARD_ACCUM=1`.

Correctness checks against the current direct path:

```text
FAMS=2 forced grouped_all: max abs 6.10e-05, max rel 9.68e-07
FAMS=2 forced noatomic_all: exact
FAMS=10 threshold grouped: max abs 1.22e-04, max rel 3.18e-07
FAMS=10 threshold noatomic: max abs 3.66e-04, max rel 3.18e-07
FAMS=50 threshold grouped: exact
FAMS=50 threshold noatomic: exact
```

Backward-only CUDA event timings from the worker:

| Families | Path | Mean | Median | Min |
|---:|---|---:|---:|---:|
| 10 | direct | 48.033 ms | 47.810 ms | 47.523 ms |
| 10 | grouped | 48.311 ms | 48.313 ms | 47.981 ms |
| 10 | noatomic | 48.343 ms | 48.324 ms | 48.054 ms |
| 50 | direct | 163.760 ms | 164.352 ms | 162.431 ms |
| 50 | grouped | 175.255 ms | 175.860 ms | 173.495 ms |
| 50 | noatomic | 163.325 ms | 163.053 ms | 160.907 ms |

Nsight Systems, 50 families:

| Path | DTS accumulation kernel total |
|---|---:|
| direct `_dts_cross_backward_accum_kernel` | 29.639 ms |
| noatomic `_dts_cross_backward_accum_kernel` | 29.630 ms |
| grouped DTS pieces combined | 39.348 ms |

Grouped split:

```text
_dts_cross_backward_accum_kernel: 18.234 ms
_dts_cross_backward_kernel:        7.912 ms
_group_cross_pibar_grad_kernel:    7.331 ms
_add_grouped_dts_pi_accum_kernel:  5.870 ms
```

The child-grouped path regresses because the selected high-fanout waves are not
child-duplicate-heavy. In the 50-family workload, the selected waves have child
duplicate factor about `1.00`, so grouping adds materialization and reduction
kernels without reducing much contention.

The no-atomic path is correctness-safe for the observed child-unique high
fanout waves, but Nsight shows no meaningful kernel improvement. The atomics
are visible in NCU, but for this layout they are not the dominant reason the
DTS accumulation kernel costs about 30 ms.

Recommendation: keep the direct atomic path as the default. The next useful
DTS experiment is parent-grouped, not child-grouped: the problematic waves have
many splits per parent, while child rows are mostly unique.

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

### Tested proposal: fused active-mask construction

An active-mask builder was added behind `GPUREC_KERNELIZED_ACTIVE_MASK=1`.
It replaces:

```python
clade_max = rhs_k.abs().max(dim=1).values
active_mask = clade_max >= pruning_threshold
```

with one Triton kernel, `_active_mask_from_rhs_absmax_kernel`.

Correctness:

```text
tests/kernels/test_active_mask_kernel.py: 3 passed
GPUREC_KERNELIZED_ACTIVE_MASK=1 targeted backward/kernel tests: 20 passed
```

Worker benchmark summary:

| Families | Default median | Fused-mask median | Result |
|---:|---:|---:|---|
| 10 | 47.527 ms | 47.802 ms | slight regression |
| 50 | 167.547 ms | 160.386 ms then 163.149 ms | noisy, sometimes faster |

10-family Nsight Systems confirms the intended PyTorch kernels are removed:

| Kernel family | Default | Fused mask |
|---|---:|---:|
| PyTorch `AbsFunctor` active-mask kernels | 47 instances / 0.587 ms | 2 / 0.002 ms |
| PyTorch row `MaxOps` reductions | 62 / 0.624 ms | 17 / 0.058 ms |
| PyTorch compare-scalar kernels | 64 / 0.084 ms | 19 / 0.024 ms |
| `_active_mask_from_rhs_absmax_kernel` | 0 | 45 / 0.450 ms |

The fused mask is a cleaner implementation, but it does not remove the real
host-pruning bottleneck. `active_mask.any()`, `active_mask.sum().item()`, and
`active_mask.nonzero(...)` still force synchronization and dynamic work in the
wave loop. In the 50-family combined profile, the fused mask added `2.947 ms`
of Triton kernel time and made the profiled interval slower despite removing
some PyTorch launches.

Recommendation: keep `GPUREC_KERNELIZED_ACTIVE_MASK=1` default-off. The next
step should be device-side active worklists or a fixed schedule that avoids
host scalar decisions, not just a fused mask builder.

## Bottleneck 6: D2D copies

This proposal was implemented and made the default. The old behavior can still
be forced with `GPUREC_FUSED_UNIFORM_BACKWARD_VIEW_RHS=0`.

The earlier profile showed large D2D volume from wave RHS snapshots. One
documented clone had already been removed, but one per-wave clone remained:

```python
rhs_k = accumulated_rhs[ws:we].clone()
```

For the fused uniform CUDA path, `_wave_backward_uniform_kernel` treats
`rhs_ptr` as read-only. Later cross-DTS and uniform-`Pibar` adjoints accumulate
into child rows from earlier waves, not into the current wave slice. Therefore
the kernel can read the contiguous `accumulated_rhs[ws:we]` view directly.

Correctness:

```text
Default RHS-view targeted backward/kernel tests: 20 passed
GPUREC_FUSED_UNIFORM_BACKWARD_VIEW_RHS=0 tests/gradients/test_autograd_bridge.py: 15 passed
10-family off/on parity: loss diff 0, theta-grad max rel 1.01e-7
Direct fused-wrapper mutation check: rhs_mutation_max_abs 0
Wave layout overlap check: child_row_overlap_violations []
```

Backward-only CUDA event timings:

| Families | RHS clone median | RHS view median | Final default median | Peak alloc before | Peak alloc after |
|---:|---:|---:|---:|---:|---:|
| 10 | 48.993 ms | 48.340 ms | 46.641 ms | 2.829 GB | 2.695 GB |
| 50 | 163.035 ms | 159.055 ms | 158.463 ms | 10.567 GB | 10.305 GB |

The final default medians are from the supervisor's sequential `REPS=9`,
`WARMUPS=5` rerun after enabling RHS view by default.

Nsight Systems D2D copy summary:

| Families | RHS clone | D2D calls | D2D bytes | D2D time |
|---:|---|---:|---:|---:|
| 10 | on | 408 | 570.080 MB | 1.238 ms |
| 10 | view | 363 | 38.106 MB | 0.377 ms |
| 50 | on | 427 | 2609.790 MB | 5.159 ms |
| 50 | view | 378 | 35.638 MB | 0.391 ms |

The removed copy count is exactly one per wave: `45` copies for 10 families and
`49` copies for 50 families. D2D volume drops by `531.974 MB` and
`2574.152 MB`; D2D time drops by `0.861 ms` and `4.768 ms`.

Recommendation: keep RHS view as the default. This is the only Bottleneck 4-6
proposal in this round with a clear mechanism, a direct profile confirmation,
lower peak memory, and a repeatable wall-time improvement.

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

### P0: keep removing avoidable bytes

The RHS clone has been removed for the fused uniform CUDA path and is now the
default. The remaining byte-reduction work is inside the self-loop kernel and
around residual scratch tensors:

1. Audit the largest remaining D2D copies after RHS-view mode.
2. Continue reducing `_wave_backward_uniform_kernel` scratch traffic.
3. Keep the compact leaf-index path as the default and avoid reintroducing dense
   `[W, S]` leaf tensors.

Measured gain from the accepted RHS-view change:

```text
10 families: 48.993 ms -> 48.340 ms in worker run, final default 46.641 ms
50 families: 163.035 ms -> 159.055 ms in worker run, final default 158.463 ms
D2D bytes at 50 families: 2609.790 MB -> 35.638 MB
```

### P1: device-resident pruning/worklists

1. Keep pruning semantics, but stop branching on CUDA scalars from Python.
2. Pass active masks to the Triton kernels and skip inactive rows or splits
   inside the kernels.
3. Compact active rows/splits into device worklists, or launch all waves in a
   fixed schedule only if the no-op work is cheaper than the current syncs.
4. Re-measure the no-op cost versus the removed sync gaps.

Expected gain:

```text
10-family upper bound: 33.7 ms GPU idle/gap span
50-family upper bound: 37.8 ms GPU idle/gap span
```

The fused active-mask experiment proved that simply replacing `abs().max()` is
not enough. It removed the intended PyTorch kernels, but left `any`,
`sum().item`, and `nonzero` synchronization in place and therefore remains
default-off.

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

### P3: parent-specialize high-fanout split waves

1. Keep the current split-major DTS path for ordinary waves.
2. Add a parent-grouped path for high `n_splits / W` root-like waves.
3. Benchmark the switch threshold with Nsight Systems.

Expected gain:

```text
targets: 29.4 ms DTS backward accum + 12.4 ms DTS forward recompute at 50 families
likely gain: workload dependent, probably single-digit to low-double-digit ms
```

Child-grouping was tested and rejected for this workload: selected high-fanout
waves had child duplicate factor about `1.00`, and the grouped DTS pieces cost
`39.348 ms` versus `29.639 ms` for the direct atomic kernel.

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

1. keep reducing self-loop scratch/copy bytes after the accepted RHS-view fix,
2. make pruning device-resident with active worklists rather than only a fused
   mask builder,
3. fuse parameter-gradient reductions,
4. try parent-grouped high-fanout DTS waves,
5. only then revisit uniform `Pibar` VJP grouping.

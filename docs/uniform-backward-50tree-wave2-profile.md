# Uniform backward 50-tree profile, improvement wave 2

Date: 2026-05-02

This is a fresh profiling pass on the current `main` state after the previous
backward optimization commits, especially the RHS-view default. The scope is
the warmed FP32 uniform-mode backward pass for the first 50 trees in
`tests/data/test_trees_1000`.

The goal is not to restate the older bottleneck document. This pass looks for
the next useful optimization targets in the current 50-tree schedule.

## Environment

| Item | Value |
|---|---:|
| Commit | `ddbfa22` |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU memory | 24,564 MiB |
| Driver | 580.126.20 |
| Nsight Systems | 2025.5.1 |
| Nsight Compute | 2025.3.1 |
| Dataset | `tests/data/test_trees_1000`, first 50 gene trees |
| Species | `S=1999` |
| Families | `G=50` |
| Clade rows | `C=321930` |
| Waves | `49` |
| Max wave rows | `32768` |
| Split rows | `402275` |
| Leaves | `80545` |
| Roots | `50` |

Benchmark command:

```bash
source /home/enzo/Documents/git/gpurec/gpurec/.venv/bin/activate
FAMS=50 REPS=15 WARMUPS=5 python /tmp/gpurec_profile/bench_uniform_backward.py
```

The benchmark sets the current fused uniform defaults:

```bash
GPUREC_KERNELIZED_BACKWARD_DTS=1
GPUREC_FUSED_DTS_BACKWARD_ACCUM=1
GPUREC_FUSED_CROSS_PIBAR_VJP=1
GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=tree
GPUREC_FUSED_UNIFORM_BACKWARD=1
GPUREC_UNIFORM_PINGPONG=1
GPUREC_BACKWARD_LEAF_INDEX=1
GPUREC_FUSED_WAVE_PARAM_ACCUM=1
```

Model configuration:

```text
mode=global
pibar_mode=uniform
dtype=torch.float32
fixed_iters_Pi=6
neumann_terms=3
use_pruning=True
pruning_threshold=1e-6
max_wave_size=32768
```

Nsight report files:

| Capture | Path |
|---|---|
| Current default, Nsight Systems | `/tmp/gpurec_profile/wave2_bwd50_current.nsys-rep` |
| Current default, SQLite export | `/tmp/gpurec_profile/wave2_bwd50_current.sqlite` |
| Active-mask candidate, Nsight Systems | `/tmp/gpurec_profile/wave2_bwd50_active_mask.nsys-rep` |
| No-CPU-pruning diagnostic, Nsight Systems | `/tmp/gpurec_profile/wave2_bwd50_no_cpu_pruning.nsys-rep` |
| Wave kernel NCU | `/tmp/gpurec_profile/wave2_ncu_wave50.ncu-rep` |
| Uniform Pibar VJP NCU | `/tmp/gpurec_profile/wave2_ncu_pibar50.ncu-rep` |
| DTS backward accum NCU | `/tmp/gpurec_profile/wave2_ncu_dtsaccum50.ncu-rep` |
| DTS forward recompute NCU | `/tmp/gpurec_profile/wave2_ncu_dtsfwd50_bwd.ncu-rep` |
| Active-mask kernel NCU | `/tmp/gpurec_profile/wave2_ncu_active_mask50.ncu-rep` |

## Baseline timings

Normal CUDA-event timings, outside Nsight:

| Variant | Mean | Median | Min | Peak allocation |
|---|---:|---:|---:|---:|
| Current default | `157.263 ms` | `157.329 ms` | `155.283 ms` | `10.305 GB` |
| Disable pruning | `243.442 ms` | `241.924 ms` | `240.486 ms` | `10.305 GB` |
| `GPUREC_BACKWARD_NO_CPU_PRUNING=1` | `161.227 ms` | `161.736 ms` | `159.744 ms` | `10.305 GB` |
| `GPUREC_DEVICE_PRUNING=1` | `161.574 ms` | `161.084 ms` | `160.017 ms` | `10.305 GB` |
| `GPUREC_KERNELIZED_ACTIVE_MASK=1` | `152.612 ms` | `152.445 ms` | `150.852 ms` | `10.305 GB` |
| `GPUREC_REUSE_FORWARD_PIBAR_STATS=1` | `158.960 ms` | `159.290 ms` | `157.571 ms` | `10.306 GB` |
| Pibar stat reuse + active-mask kernel | `153.882 ms` | `153.629 ms` | `151.951 ms` | `10.306 GB` |

Main conclusions:

1. Pruning is still required. Disabling it is about `1.54x` slower.
2. Fully avoiding CPU pruning decisions is not enough. It removes some host
   gating, but it processes more wave/split work and ends up about `4 ms`
   slower than the default.
3. The active-mask Triton kernel is now a clear 50-tree win: median
   `157.329 ms -> 152.445 ms`, a `4.884 ms` improvement. It passed the focused
   correctness suite:

```bash
GPUREC_KERNELIZED_ACTIVE_MASK=1 \
pytest -q tests/gradients/test_autograd_bridge.py \
          tests/kernels/test_uniform_cross_pibar_vjp_kernel.py \
          tests/kernels/test_active_mask_kernel.py -q
# 20 passed
```

The active-mask flag should be considered for promotion to default in the next
implementation wave, after checking 10-tree and larger-batch behavior again.

## Workload shape

Largest wave-row batches:

| Wave | Start | Rows `W` | Has splits | Split rows | Split fanout |
|---:|---:|---:|---|---:|---:|
| 0 | 0 | 32768 | no | 0 | 0 |
| 1 | 32768 | 32768 | no | 0 | 0 |
| 3 | 80545 | 27023 | yes | 27023 | 1.0 |
| 34 | 234772 | 19412 | yes | 19412 | 1.0 |
| 33 | 215656 | 19116 | yes | 19116 | 1.0 |
| 35 | 254184 | 17886 | yes | 17886 | 1.0 |
| 32 | 198764 | 16892 | yes | 16892 | 1.0 |
| 4 | 107568 | 16281 | yes | 16281 | 1.0 |
| 36 | 272070 | 15206 | yes | 15206 | 1.0 |
| 2 | 65536 | 15009 | no | 0 | 0 |

Largest split waves:

| Wave | Rows `W` | Split rows | Split fanout |
|---:|---:|---:|---:|
| 44 | 247 | 42155 | 170.7 |
| 43 | 584 | 36200 | 62.0 |
| 3 | 27023 | 27023 | 1.0 |
| 46 | 39 | 24229 | 621.3 |
| 45 | 93 | 23345 | 251.0 |
| 42 | 1236 | 19642 | 15.9 |
| 34 | 19412 | 19412 | 1.0 |
| 33 | 19116 | 19116 | 1.0 |
| 35 | 17886 | 17886 | 1.0 |
| 32 | 16892 | 16892 | 1.0 |

This explains why there are two different hot shapes:

- leaf/no-split waves with `W=32768`, dominated by the self-loop kernel;
- small-root-ish waves with huge split counts, dominated by DTS recomputation,
  DTS backward accumulation, and uniform Pibar VJP.

## Nsight Systems: current default

The current default single-capture backward interval had CUDA-event time
`169.534 ms` under Nsight overhead. Direct event timing outside Nsight is the
reliable wall-time baseline; Nsight is used here for proportions.

GPU event interval:

| Metric | Value |
|---|---:|
| GPU span from first to last event | `169.188 ms` |
| Merged GPU active time | `143.400 ms` |
| GPU gaps | `25.789 ms` |
| Kernel time, summed | `142.761 ms` |
| Memcpy/memset time, summed | `0.638 ms` |
| CUDA kernels | `3234` |
| Memcpy/memset events | `662` |
| CUDA stream IDs used by kernels and memops | one stream, stream `7` |

The one-stream execution is dependency-driven. Within one wave, the pipeline is
mostly serial:

```text
active mask -> DTS forward recompute -> self-loop solve -> DTS backward accum
             -> uniform Pibar VJP -> next earlier wave
```

There is limited opportunity for profitable overlap because the dominant
kernels are all memory-bound and most later kernels consume outputs from earlier
kernels. The one possible local overlap is parameter reductions against Pibar
VJP, but both read the same large `grad_Pibar_*` buffers and would compete for
bandwidth. Fusion is more promising than extra streams.

Kernel and residual work breakdown:

| Component | Time | Launches | Share of kernel time | Notes |
|---|---:|---:|---:|---|
| `_wave_backward_uniform_kernel` | `39.578 ms` | 36 | 27.7% | self-loop Neumann VJP and wave param accumulation |
| `_uniform_cross_pibar_vjp_tree_kernel` | `37.954 ms` | 33 | 26.6% | uniform Pibar VJP for split child gradients |
| `_dts_cross_backward_accum_kernel` | `29.678 ms` | 33 | 20.8% | direct Pi adjoints plus Pibar-gradient materialization |
| `_dts_fused_kernel` | `12.136 ms` | 33 | 8.5% | backward-side DTS forward recomputation |
| PyTorch float reductions | `5.498 ms` | 248 | 3.9% | mostly remaining per-wave reductions |
| PyTorch abs kernels | `4.726 ms` | 51 | 3.3% | active-mask construction |
| PyTorch fill kernels | `3.507 ms` | 495 | 2.5% | temporary setup and zero/inf fills |
| PyTorch max reductions | `3.184 ms` | 106 | 2.2% | active-mask construction and other row reductions |
| `_seg_lse_hdim_kernel` | `1.518 ms` | 7 | 1.1% | split logsumexp for multi-split parents |
| PyTorch index kernels | `1.482 ms` | 66 | 1.0% | `nonzero`, indexing, compaction |
| CUB sort/select/compact kernels | `0.747 ms` | 257 | 0.5% | dynamic CUDA indexing support |

CUDA API summary:

| API | Calls | API time | Notes |
|---|---:|---:|---|
| `cudaStreamSynchronize` | 318 | `125.976 ms` | mostly host waiting for submitted GPU work |
| `cudaDeviceSynchronize` | 6 | `7.292 ms` | includes benchmark synchronization |
| `cudaLaunchKernel` | 2982 | `5.260 ms` | launch overhead, not the main wall-time issue at 50 trees |
| `cudaMemcpyAsync` | 640 | `1.716 ms` | tiny copies; raw GPU copy time is only `0.621 ms` |
| `cuLaunchKernelEx` | 142 | `0.365 ms` | Triton launches |
| `cuLaunchKernel` | 110 | `0.175 ms` | library launches |

Memcpy/memset summary:

| Operation | Count | Bytes | GPU time |
|---|---:|---:|---:|
| Device-to-device copy | 378 | `35.638 MB` | `0.391 ms` |
| Device-to-host copy | 260 | `0.033 MB` | `0.227 ms` |
| Host-to-device copy | 2 | `0.032 MB` | `0.003 ms` |
| Memset | 22 | `0.006 MB` | `0.017 ms` |

The RHS-view change did its job: D2D copies are no longer a meaningful
bandwidth cost. The remaining D2H copies are tiny but important because many of
them are scalar decisions that gate future kernel launches.

Likely D2H/sync sources in the current code:

- `active_mask.any()` in the wave loop;
- `active_mask.sum().item()` for `n_active`;
- `active_mask.nonzero(...)`, because CUDA `nonzero` has a dynamic output shape;
- scalar extraction of `log_pD` and `log_pS` in
  `dts_cross_backward_accum_fused`, which converts CUDA scalar tensors to Python
  floats before launching Triton;
- small scalar reductions in the final autograd/E-adjoint path.

The no-CPU-pruning diagnostic quantifies the tradeoff. It reduced D2H copies
from `260` to `188` and `cudaStreamSynchronize` calls from `318` to `246`, but
the profiled GPU active time increased from `143.400 ms` to `149.692 ms`. It
processed 49 wave-self-loop launches and 46 split launches instead of 36 and
33. The net result was slower (`161.736 ms` median outside Nsight).

## Active-mask candidate

With `GPUREC_KERNELIZED_ACTIVE_MASK=1`, event median improved from
`157.329 ms` to `152.445 ms`. Nsight Systems shows the mechanism:

| Metric | Default | Active-mask kernel |
|---|---:|---:|
| GPU span under Nsight | `169.188 ms` | `165.741 ms` |
| Merged GPU active time | `143.400 ms` | `138.337 ms` |
| GPU gaps | `25.789 ms` | `27.404 ms` |
| CUDA kernels | `3234` | `3136` |
| `_wave_backward_uniform_kernel` | `39.578 ms` | `39.464 ms` |
| `_uniform_cross_pibar_vjp_tree_kernel` | `37.954 ms` | `37.833 ms` |
| `_dts_cross_backward_accum_kernel` | `29.678 ms` | `29.680 ms` |
| `_dts_fused_kernel` | `12.136 ms` | `12.137 ms` |
| PyTorch abs kernels | `4.726 ms` | `0.002 ms` |
| PyTorch max reductions | `3.184 ms` | `0.199 ms` |
| PyTorch compare kernels | `0.145 ms` | `0.072 ms` |
| `_active_mask_from_rhs_absmax_kernel` | `0 ms` | `2.948 ms` |

This flag does not remove the host pruning syncs. It simply replaces roughly
`8.1 ms` of PyTorch active-mask kernels with `2.95 ms` of Triton row-reduction
kernels. That is now a robust win at 50 trees because the fused-kernel buckets
are unchanged and the reduced active GPU work dominates the slightly larger
gap span.

NCU on a representative active-mask launch (`W=32768`) confirms it is already a
simple memory-streaming kernel:

| Metric | Value |
|---|---:|
| Duration | `0.275 ms` |
| DRAM throughput | `95.9%` |
| SM throughput | `35.2%` |
| DRAM read | `0.262 GB` |
| DRAM write | `0.003 GB` |
| Memory throughput | `965 GB/s` |
| L1 hit rate | `17.7%` |
| L2 hit rate | `1.0%` |
| Achieved occupancy | `96.2%` |
| Registers/thread | 18 |
| Dominant stall | long scoreboard, `83.1%` of sampled issue-stall cycles |

This kernel itself should not be the next target. It is a single read of
`rhs[W,S]` plus one byte-scale output per row. The better follow-up is to use
the resulting mask without going back to the CPU for `any`, `sum().item`, and
`nonzero`.

## Nsight Compute resource summary

NCU was collected on one representative large launch from each dominant kernel
class. These are replayed single-kernel measurements, so use them for resource
diagnosis rather than direct wall-time accounting.

| Kernel | Launch shape | Duration | DRAM peak | SM peak | DRAM read | DRAM write | BW | L1 hit | L2 hit | Occ. | Eligible warps/sched | Regs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `_wave_backward_uniform_kernel` | `(32768,1)x128` | `7.714 ms` | `70.1%` | `35.3%` | `3.009 GB` | `2.440 GB` | `706 GB/s` | `56.9%` | `87.8%` | `82.5%` | `0.30` | 48 |
| `_uniform_cross_pibar_vjp_tree_kernel` | `(84310,1)x128` | `6.340 ms` | `63.0%` | `59.3%` | `2.703 GB` | `1.322 GB` | `635 GB/s` | `75.8%` | `86.6%` | `74.6%` | `0.81` | 56 |
| `_dts_cross_backward_accum_kernel` | `(42155,1)x128` | `4.752 ms` | `76.1%` | `25.2%` | `2.122 GB` | `1.519 GB` | `766 GB/s` | `61.4%` | `74.1%` | `74.6%` | `0.26` | 54 |
| `_dts_fused_kernel` | `(42155,16)x128` | `2.120 ms` | `88.5%` | `16.3%` | `1.540 GB` | `0.350 GB` | `891 GB/s` | `69.4%` | `23.0%` | `92.0%` | `0.20` | 34 |
| `_active_mask_from_rhs_absmax_kernel` | `(32768,1)x128` | `0.275 ms` | `95.9%` | `35.2%` | `0.262 GB` | `0.003 GB` | `965 GB/s` | `17.7%` | `1.0%` | `96.2%` | `0.17` | 18 |

Stall profile:

| Kernel | Main issue stalls |
|---|---|
| `_wave_backward_uniform_kernel` | long scoreboard `44.6%`, barrier `23.1%`, MIO throttle `12.9%`, short scoreboard `7.6%` |
| `_uniform_cross_pibar_vjp_tree_kernel` | long scoreboard `61.8%`, barrier `9.3%`, wait `6.8%`, short scoreboard `4.7%` |
| `_dts_cross_backward_accum_kernel` | barrier `29.4%`, long scoreboard `26.1%`, MIO throttle `25.8%`, short scoreboard `7.6%` |
| `_dts_fused_kernel` | long scoreboard `93.9%` |
| `_active_mask_from_rhs_absmax_kernel` | long scoreboard `83.1%`, short scoreboard `6.4%`, barrier `5.9%` |

Uncoalesced/excess sector signal:

| Kernel | Excessive sectors reported by NCU |
|---|---:|
| `_wave_backward_uniform_kernel` | `777.6 MB`, about 50% of L2 theoretical global sectors |
| `_uniform_cross_pibar_vjp_tree_kernel` | `618.5 MB`, about 27% of L2 theoretical global sectors |
| `_dts_cross_backward_accum_kernel` | `356.3 MB`, about 55% of L2 theoretical global sectors |
| `_dts_fused_kernel` | `44.0 MB`, about 23% of L2 theoretical global sectors |
| `_active_mask_from_rhs_absmax_kernel` | `1.8 MB`, about 18% of L2 theoretical global sectors |

These counters point to memory dependency and data movement, not raw FP32 ALU.
Tensor cores are not involved. The hot kernels either stream large `[rows,S]`
matrices or perform gather/scatter/atomic patterns that do not map to MMA.

## New bottlenecks and next proposals

### 1. Promote and extend the active-mask Triton path

Current impact:

```text
event median: 157.329 ms -> 152.445 ms
GPU active time under Nsight: 143.400 ms -> 138.337 ms
old PyTorch active-mask kernels: ~8.1 ms
new Triton active-mask kernels: 2.948 ms
```

This is the cleanest immediate win. It is correctness-tested and does not
change numerical ordering in the core adjoint kernels. The main reason it was
not promoted earlier was that the 50-family measurements were noisy; in the
current RHS-view baseline it is stable.

Follow-up after promotion:

1. Keep the Triton mask builder as default for CUDA fp32/fp64 uniform backward.
2. Continue using the CPU decisions initially, so the change is small and easy
   to validate.
3. Then build a second stage that avoids the CPU shape decisions:
   device-side active counts, persistent/overallocated active index buffers, or
   kernel-side masking without `nonzero`.

The no-CPU-pruning diagnostic shows that simply processing all rows with an
active mask is not enough. We need to remove syncs while preserving most of the
current compaction/skipping benefit.

### 2. Merge the two species loops in `_dts_cross_backward_accum_kernel`

This is the most concrete new fused-kernel opportunity found in this pass.

The current kernel computes DTS backward in two full `S` loops per split:

1. First loop:
   - loads `Pi_l`, `Pi_r`, `Pibar_l`, `Pibar_r`;
   - loads `Pi_l[c1]`, `Pi_l[c2]`, `Pi_r[c1]`, `Pi_r[c2]`;
   - computes `vd0..vd4`;
   - atomically accumulates the D/T direct `Pi` adjoints;
   - stores `grad_Pibar_l/r`;
   - accumulates scalar `param_pD/param_pS`.
2. Second loop:
   - reloads species children;
   - reloads `Pi_l[c1]`, `Pi_l[c2]`, `Pi_r[c1]`, `Pi_r[c2]`;
   - reloads `Pi_parent` and `v_k`;
   - recomputes `vd3/vd4`;
   - atomically accumulates the S-term direct `Pi` adjoints.

The first loop already has everything needed to do the second loop's S-term
atomic adds. Merging them should remove the second pass over `S`.

Why this matters:

| Metric | Largest launch |
|---|---:|
| Kernel total at 50 trees | `29.678 ms` across 33 launches |
| Representative launch | `42155` splits |
| Representative duration | `4.752 ms` |
| DRAM read | `2.122 GB` |
| DRAM write | `1.519 GB` |
| DRAM peak | `76.1%` |
| SM peak | `25.2%` |
| Top stalls | barrier, long scoreboard, MIO throttle |

Expected benefit: if the merged loop removes even 20-30% of this kernel's
traffic, the 50-tree backward should improve by roughly `6-9 ms` before
secondary effects. This proposal also reduces recomputation and should not
increase launch count.

Risks:

- Register pressure may increase because the first loop must keep `vd3/vd4`,
  child indices, masks, and the existing D/T values live around more atomics.
  The current kernel already uses 54 registers/thread and is register-limited
  to 75% theoretical occupancy.
- Adding more atomics to the first loop may worsen MIO throttle. The current
  split kernel already has global RED activity and only 0.26 eligible warps per
  scheduler.

This should be tested as a guarded kernel variant, with NCU comparing DRAM
bytes, register count, eligible warps, and MIO throttle.

### 3. Fuse or kernelize the remaining DTS parameter reductions

After `_dts_cross_backward_accum_kernel`, Python still performs per-wave
reductions:

```python
grad_log_pD[0] += param_pD.sum()
grad_log_pS[0] += param_pS.sum()
mt_contrib = grad_Pibar_l.sum(dim=0) + grad_Pibar_r.sum(dim=0)
grad_mt[0] += mt_contrib
```

In the default Nsight trace, PyTorch float reductions cost `5.498 ms` across
248 launches. Some of those reductions are unrelated, but the DTS parameter and
`grad_mt` reductions are a visible part of the residual PyTorch bucket. They
also force launch sequencing before the Pibar VJP runs.

Possible implementation:

- Make `_dts_cross_backward_accum_kernel` optionally accumulate `grad_log_pD`,
  `grad_log_pS`, and `grad_mt` directly.
- `grad_log_pD/grad_log_pS` are scalar reductions and should be cheap to
  accumulate with block-level partial sums plus one atomic per split/program.
- `grad_mt[s] += grad_Pibar_l[:,s] + grad_Pibar_r[:,s]` is more delicate:
  doing this inside the split kernel means many atomics into only `S=1999`
  species lanes. That may be cheaper than writing and rereading both
  `grad_Pibar_*` matrices, but it must be profiled.

Alternative:

- A custom Triton reduction over `grad_Pibar_l/r` by species. This keeps the
  reduction separate but removes many tiny PyTorch reductions and gives control
  over tiling. It will still read `2 * n_ws * S` floats, so it is less
  attractive than fusing if atomics are tolerable.

This proposal should be measured both standalone and after the DTS two-loop
merge, because the best placement may change after register pressure changes.

### 4. Remove per-wave CUDA-scalar extraction for DTS scalar parameters

`dts_cross_backward_accum_fused` currently converts `log_pD` and `log_pS` to
Python floats before launching Triton:

```python
pD_val = float(log_pD) if log_pD.ndim == 0 else float(log_pD.item())
pS_val = float(log_pS) if log_pS.ndim == 0 else float(log_pS.item())
```

In this benchmark there are 33 split waves, so this pattern can contribute up
to 66 scalar device-to-host sync points in the backward interval. The D2H bytes
are tiny and will not show as bandwidth, but they gate launches and contribute
to the `cudaStreamSynchronize` count.

Better design:

- Pass `log_pD` and `log_pS` as device pointers, like `dts_fused` already does.
- In the shared-parameter path, load `log_pD[0]` and `log_pS[0]` once inside
  the Triton program.
- Keep the Python-float path only for actual Python floats, not CUDA tensors.

Expected benefit is probably small by pure GPU time, but it attacks the
remaining 260 D2H copies and 318 stream synchronizations. It is also a clean
correctness-preserving cleanup.

### 5. Revisit DTS-to-Pibar fusion, but only after reducing DTS traffic

The current split-side path materializes `grad_Pibar_l` and `grad_Pibar_r` in
the DTS backward kernel, then the uniform Pibar VJP kernel immediately reads
them.

For the largest split wave:

```text
n_ws = 42155
S = 1999
grad_Pibar_l/r size = 2 * n_ws * S * 4 bytes ~= 674 MB
```

At 50 trees, the measured kernels are:

```text
_dts_cross_backward_accum_kernel:     29.678 ms
_uniform_cross_pibar_vjp_tree_kernel: 37.954 ms
```

A fused or semi-fused design could avoid writing and rereading the
`grad_Pibar_*` matrices. The hard part is that uniform Pibar VJP needs row
normalization and subtree correction for each child row, with barriers over the
species tree. It is not a simple elementwise continuation of DTS.

A plausible direction is not one giant kernel. It is:

1. Merge the DTS two-loop kernel first.
2. Add a variant that writes a more compact Pibar-VJP input or directly writes
   `u_d` into the Pibar subtree buffer.
3. Run the existing tree correction from that buffer.

This keeps the tree-reduction algorithm intact while trying to remove the
largest avoidable intermediate traffic.

### 6. Self-loop kernel remains the largest single bucket, but fewer easy wins

The self-loop kernel is still the largest bucket:

```text
_wave_backward_uniform_kernel total: 39.578 ms
largest launches: 7.323 ms and 7.297 ms at W=32768
representative DRAM traffic: 3.009 GB read + 2.440 GB write
```

NCU says it is memory-latency/bandwidth limited:

```text
DRAM peak: 70.1%
SM peak: 35.3%
eligible warps/scheduler: 0.30
main stalls: long scoreboard, barrier, MIO throttle
```

This is still a real bottleneck, but the previous wave already removed the
largest obvious scratch traffic. New self-loop work should be more targeted:

- inspect NCU source counters for the uncoalesced global sectors in this
  kernel;
- test whether any remaining scratch arrays can be recomputed cheaper than
  loaded/stored;
- consider a leaf/no-split specialization only if source counters show live
  branches or memory traffic that the current `HAS_SPLITS=False` and
  `USE_LEAF_INDEX=True` constexprs do not already eliminate.

The expected win is less certain than the active-mask and DTS-loop proposals.

## Ranked plan for the next implementation wave

1. **Promote `GPUREC_KERNELIZED_ACTIVE_MASK=1` to default for CUDA uniform
   backward.** Completed in the proposal 1 follow-up below.
2. **Merge the two species loops in `_dts_cross_backward_accum_kernel`.**
   Completed in the proposal 2 follow-up below.
3. **Remove CUDA scalar-to-Python extraction for `log_pD/log_pS` in DTS backward
   kernels**. Completed in the proposal 3 follow-up below.
4. **Kernelize/fuse DTS parameter and `grad_mt` reductions**. Completed in the
   proposal 4 follow-up below for scalar `grad_log_pD/grad_log_pS` reductions.
   The `grad_mt` accumulation variant was tested and left opt-in because it is
   slower.
5. **Only then revisit DTS/Pibar intermediate fusion**. The materialized
   `grad_Pibar_l/r` traffic is large, but the tree VJP makes a monolithic kernel
   risky.
6. **Use NCU source-counter-guided work on `_wave_backward_uniform_kernel`**.
   It remains the largest bucket, but the next change should be driven by exact
   source lines for excessive sectors, not by broad guesses.

## Measurement notes

- NCU replay times are not the same as normal execution time. The NCU tables are
  for resource diagnosis only.
- The `_dts_fused_kernel` is also used in the forward pass. Its NCU capture was
  taken with `PROFILE_CUDA_API=1` and `--profile-from-start off` so the selected
  launch comes from the backward interval.
- A first attempt to time two pruning variants in parallel was discarded. The
  timings reported above were rerun sequentially.

## Proposal 1 follow-up: active-mask default

Proposal 1 was tested with the same three-agent workflow used in earlier
optimization passes:

| Role | Task | Result |
|---|---|---|
| Static reviewer | Check the exact default flip and scope risks | One-line default flip is correct, but should be scoped to `pibar_mode == 'uniform'` to avoid changing dense/topk behavior |
| Correctness worker | Run focused tests and parity checks | Passed; see below |
| Benchmark worker | Compare default and explicit active-mask runs | Found timings in the active-mask range; supervisor reran explicit `GPUREC_KERNELIZED_ACTIVE_MASK=0` vs default to remove ambiguity after the local default changed |

The implemented guard is:

```python
kernelized_active_mask_enabled = (
    os.environ.get("GPUREC_KERNELIZED_ACTIVE_MASK", "1") != "0"
    and _HAS_FUSED_BACKWARD
    and pibar_mode == 'uniform'
    and device.type == 'cuda'
    and dtype in (torch.float32, torch.float64)
)
```

So the Triton mask path is now default-on only for CUDA fp32/fp64 uniform
backward. `GPUREC_KERNELIZED_ACTIVE_MASK=0` remains the override to force the
old PyTorch `abs().max()` mask construction.

Correctness checks:

| Command/check | Result |
|---|---:|
| `pytest -q tests/gradients/test_autograd_bridge.py tests/kernels/test_uniform_cross_pibar_vjp_kernel.py tests/kernels/test_active_mask_kernel.py -q` | 20 passed |
| `GPUREC_KERNELIZED_ACTIVE_MASK=0 pytest -q tests/gradients/test_autograd_bridge.py -q` | 15 passed |
| `test_trees_20`, 10-family default vs explicit active mask | loss diff `0`, theta max abs `1.91e-6`, max rel `1.91e-7` |
| `test_trees_1000`, 10-family default vs explicit active mask | exact loss and theta-gradient match |
| `test_trees_1000`, forced old mask vs active mask | loss diff `0`, theta max abs `3.66e-4`, max rel `9.55e-7` |

The larger old `tests/gradients/test_wave_gradient.py -k "Pruning"` check was
also tried, but it fails in both active-mask modes before reaching the mask
logic:

```text
TypeError: unsupported operand type(s) for @: 'Tensor' and 'NoneType'
```

The failure happens in the generic uniform self-loop path with `ancestors_T=None`
and reproduces with `GPUREC_KERNELIZED_ACTIVE_MASK=0`, so it is not a proposal 1
regression.

Supervisor timings after the default flip, with the old path forced by
`GPUREC_KERNELIZED_ACTIVE_MASK=0`:

```bash
FAMS={10,50} REPS=9 WARMUPS=5 python /tmp/gpurec_profile/bench_uniform_backward.py
```

| Families | Mask path | Mean | Median | Min | Peak allocation |
|---:|---|---:|---:|---:|---:|
| 10 | old PyTorch mask, forced `0` | `47.387 ms` | `47.386 ms` | `47.222 ms` | `2.695 GB` |
| 10 | new default Triton mask | `46.711 ms` | `46.615 ms` | `46.450 ms` | `2.695 GB` |
| 50 | old PyTorch mask, forced `0` | `157.464 ms` | `157.820 ms` | `155.726 ms` | `10.305 GB` |
| 50 | new default Triton mask | `152.798 ms` | `151.909 ms` | `151.252 ms` | `10.305 GB` |

A reversed-order 50-family rerun confirmed the direction:

| Families | Mask path | Mean | Median | Min |
|---:|---|---:|---:|---:|
| 50 | new default Triton mask | `152.924 ms` | `153.090 ms` | `151.432 ms` |
| 50 | old PyTorch mask, forced `0` | `157.737 ms` | `157.997 ms` | `156.014 ms` |

Decision: promote proposal 1. The measured win is about `0.8 ms` at 10
families and `4.9-5.9 ms` at 50 families, with no peak-memory change and no
meaningful numerical difference.

## Proposal 2 follow-up: merged DTS S-term accumulation

Proposal 2 merges the two species loops in `_dts_cross_backward_accum_kernel`.
The old direct DTS backward accumulation kernel did this:

1. First full pass over species:
   - compute `vd0..vd4`;
   - accumulate D/T direct `Pi` adjoints;
   - write `grad_Pibar_l/r`;
   - accumulate `param_pD/param_pS`.
2. Second full pass over species:
   - reload `Pi_l[c1]`, `Pi_l[c2]`, `Pi_r[c1]`, `Pi_r[c2]`;
   - reload parent `Pi` and `v_k`;
   - recompute `vd3/vd4`;
   - accumulate S-term direct `Pi` adjoints.

The new path does the S-term `vd3/vd4` `accumulated_rhs` updates inside the
first pass, while those values and species-child masks are already live. The
old two-loop path remains available with:

```bash
GPUREC_MERGED_DTS_BACKWARD_ACCUM=0
```

The alias `GPUREC_DTS_BACKWARD_ACCUM_IMPL=merged` also enables the merged path,
but the production default is now direct accumulation with merged S-term
updates enabled.

### Subagent workflow

| Role | Task | Result |
|---|---|---|
| Static reviewer | Check algebra and risks | Equivalent for atomic direct accumulation; non-atomic remains valid only under the existing unique-child-row guard |
| Correctness worker | Run focused tests and 10-family parity | Passed; exact 10-family loss and theta-gradient match against old direct |
| Performance worker | Run 10/50-family timings and 50-family Nsight Systems | Merged path consistently faster; DTS accum kernel bucket down by `3.535 ms` in Nsight |
| Supervisor | Add focused direct-kernel parity test, promote default, run NCU old-vs-merged | Confirmed reduced DRAM bytes but higher register pressure |

The static review specifically called out these constraints:

- atomic direct accumulation is algebraically safe when the S-term atomics move
  into the first loop;
- `c1_valid` and `c2_valid` masks must remain unchanged, because invalid
  sentinel children otherwise write to species `0`;
- the active-mask early return must still zero `grad_Pibar_l/r` and scalar
  params because those outputs are allocated with `torch.empty`;
- the non-atomic path must not be widened beyond its existing
  `child_rows_unique` guard.

The implementation therefore adds a `MERGE_S_TERM` Triton constexpr and passes
it only through `dts_cross_backward_accum_fused`. Grouped DTS and no-atomic DTS
paths are left as separate experimental paths.

### Correctness

Focused checks:

| Check | Result |
|---|---:|
| `py_compile` over changed files | passed |
| default focused suite with merged default | 23 passed |
| forced old path with `GPUREC_MERGED_DTS_BACKWARD_ACCUM=0` | 18 passed |
| correctness worker default focused tests | 13 passed |
| correctness worker with `GPUREC_MERGED_DTS_BACKWARD_ACCUM=1` | 13 passed |
| correctness worker old fallback `GPUREC_FUSED_DTS_BACKWARD_ACCUM=0` | 3 passed |
| correctness worker alias `GPUREC_DTS_BACKWARD_ACCUM_IMPL=merged` | 3 passed |

The new direct-kernel test
`tests/kernels/test_dts_backward_accum_kernel.py` compares old two-loop and
merged accumulation outputs directly. It covers:

- fp32 and fp64;
- duplicated child clade rows, so atomics are exercised;
- inactive parent rows through `active_mask`.

It passed:

```text
tests/kernels/test_dts_backward_accum_kernel.py: 3 passed
```

10-family end-to-end parity from the correctness worker, toggling only
`GPUREC_MERGED_DTS_BACKWARD_ACCUM`:

| Quantity | Difference |
|---|---:|
| loss abs diff | `0` |
| theta grad max abs diff | `0` |
| theta grad max rel diff | `0` |
| theta grad L2 diff | `0` |

### Timings

Benchmark command:

```bash
FAMS={10,50} REPS=9 WARMUPS=5 python /tmp/gpurec_profile/bench_uniform_backward.py
```

Performance worker timings before promoting the default:

| Families | Path | Mean | Median | Min | Peak allocation |
|---:|---|---:|---:|---:|---:|
| 10 | old direct | `47.197 ms` | `47.194 ms` | `46.856 ms` | `2.695 GB` |
| 10 | merged | `45.974 ms` | `45.955 ms` | `45.722 ms` | `2.695 GB` |
| 50 | old direct, run 1 | `153.055 ms` | `152.699 ms` | `151.524 ms` | `10.305 GB` |
| 50 | merged, run 1 | `149.925 ms` | `149.759 ms` | `148.211 ms` | `10.305 GB` |
| 50 | old direct, run 2 | `152.802 ms` | `153.575 ms` | `151.054 ms` | `10.305 GB` |
| 50 | merged, run 2 | `148.997 ms` | `148.606 ms` | `147.457 ms` | `10.305 GB` |

After promoting the merged path to default, the supervisor reran a direct
default-vs-forced-old comparison:

| Families | Path | Mean | Median | Min | Peak allocation |
|---:|---|---:|---:|---:|---:|
| 50 | new default merged | `148.942 ms` | `148.625 ms` | `147.666 ms` | `10.305 GB` |
| 50 | old path forced with `GPUREC_MERGED_DTS_BACKWARD_ACCUM=0` | `152.777 ms` | `153.198 ms` | `151.494 ms` | `10.305 GB` |

So proposal 2 saves about:

```text
10 families:  ~1.2 ms, 2.6%
50 families:  ~3.1-4.6 ms depending on run/order, about 2-3%
```

Combined with proposal 1, the current 50-family backward is now below
`150 ms` median on this benchmark.

### Nsight Systems

Worker Nsight Systems captures:

| Path | Script backward time | `_dts_cross_backward_accum_kernel` total | DTS launches | Total kernel time | Kernel interval | Kernel launches |
|---|---:|---:|---:|---:|---:|---:|
| old direct | `164.218 ms` | `29.681 ms` | 33 | `137.765 ms` | `163.910 ms` | 3136 |
| merged | `161.713 ms` | `26.145 ms` | 33 | `134.359 ms` | `161.418 ms` | 3136 |

The launch count is unchanged. The DTS accum bucket improves by:

```text
29.681 ms -> 26.145 ms
delta = -3.535 ms
relative DTS bucket speedup = 11.9%
```

### Nsight Compute

Supervisor NCU on the same representative largest DTS accum launch
(`42155` splits):

| Metric | Old two-loop | Merged S-term |
|---|---:|---:|
| Duration | `4.751 ms` | `4.371 ms` |
| DRAM read | `2.123 GB` | `2.026 GB` |
| DRAM write | `1.521 GB` | `1.319 GB` |
| Total DRAM bytes | `3.644 GB` | `3.345 GB` |
| DRAM throughput | `76.2%` | `76.0%` |
| SM throughput | `25.2%` | `23.9%` |
| L1 hit rate | `61.3%` | `55.5%` |
| L2 hit rate | `74.1%` | `67.4%` |
| Achieved occupancy | `74.6%` | `49.6%` |
| Registers/thread | 54 | 78 |
| Register occupancy limit | 9 blocks/SM | 6 blocks/SM |
| Eligible warps/scheduler | `0.264` | `0.149` |
| Global RED ops | `15.93 M` | `15.93 M` |
| Excess global sectors | `356 MB` | `224 MB` |

The result is a useful but not perfect tradeoff. The merged kernel removes about
`299 MB` of DRAM traffic from the representative launch and cuts excessive
global sectors by about `132 MB`. However, live range pressure increases
registers/thread from `54` to `78`, dropping achieved occupancy from `74.6%` to
`49.6%`. This explains why the single-launch speedup is `8.0%` instead of the
larger speedup one might expect from removing the whole second logical loop.

Stall mix:

| Stall | Old | Merged |
|---|---:|---:|
| Long scoreboard | `26.2%` | `30.4%` |
| Barrier | `29.4%` | `29.3%` |
| MIO throttle | `25.7%` | `18.1%` |
| Short scoreboard | `7.6%` | `11.0%` |
| LG throttle | `5.0%` | `4.1%` |

Decision: promote proposal 2. It gives a stable end-to-end improvement and
reduces the DTS accum bucket by almost `12%` in Nsight Systems. The next DTS
kernel work should target the new register-pressure/occupancy cost rather than
more blind fusion.

## Proposal 3 follow-up: DTS scalar parameters stay on device

Proposal 3 removes the per-wave CUDA-scalar extraction for `log_pD` and
`log_pS` in the fused DTS backward wrappers. Before this change,
`dts_cross_backward_fused` and `dts_cross_backward_accum_fused` converted the
two scalar CUDA tensors to Python floats:

```python
pD_val = float(log_pD) if log_pD.ndim == 0 else float(log_pD.item())
pS_val = float(log_pS) if log_pS.ndim == 0 else float(log_pS.item())
```

In the 50-family workload there are 33 split waves. That means the old default
did two scalar device-to-host extractions per split wave, for 66 tiny D2H
copies and 66 host-side stream synchronizations in the backward interval.

The new default passes one-element device tensors to the Triton kernels. Each
split program loads `log_pD` and `log_pS` once with `tl.load`. The old behavior
is still available for comparison with:

```bash
GPUREC_DTS_BACKWARD_DEVICE_SCALARS=0
```

The call sites in `Pi_wave_backward` now pass `log_pD` and `log_pS` directly
into the scalar-gated fused block instead of indexing `reshape(-1)[0]`. The
fused wrappers still reject non-scalar tensors through a `numel() == 1` helper,
so specieswise/genewise parameters continue to use the generic path.

### Subagent workflow

| Role | Task | Result |
|---|---|---|
| Static reviewer | Check the scalar-extraction sites and safest API shape | Recommended pointer scalar params, `numel()==1` guard, dtype/device normalization, and preserving the generic fallback |
| Correctness worker | Test old-vs-new parity for DTS backward wrappers | Passed for fp32/fp64, active-mask on/off, merged S-term on/off |
| Performance worker | Coordinate profiling plan | Paused before Nsight to avoid concurrent GPU profiling after seeing another worker's large CUDA test process |
| Supervisor | Implement, test, benchmark, run Nsight Systems and one NCU sample | Default device-scalar path is faster end-to-end by removing exactly 66 sync/copy pairs |

### Correctness

Focused supervisor checks:

| Check | Result |
|---|---:|
| `py_compile` over changed files | passed |
| `tests/kernels/test_dts_backward_accum_kernel.py` | 7 passed |
| default focused suite | 27 passed |
| forced old scalar path focused suite | 22 passed |

The new direct kernel tests compare:

- `GPUREC_DTS_BACKWARD_DEVICE_SCALARS=1` against forced fallback `0`;
- 0-d and `[1]` scalar tensor shapes;
- fp32 and fp64;
- `_dts_cross_backward_accum_kernel`;
- `_dts_cross_backward_kernel`, which is still used by the grouped accum path;
- inactive parent rows through `active_mask`.

The correctness worker also ran a direct old-vs-new parity script over
`dts_cross_backward_fused` and `dts_cross_backward_accum_fused`:

| dtype | active mask | merged S-term | max abs diff, non-accum | max abs diff, accum |
|---|---|---|---:|---:|
| fp32 | off | off | `0` | `4.66e-10` |
| fp32 | off | on | `0` | `0` |
| fp32 | on | off | `0` | `0` |
| fp32 | on | on | `0` | `0` |
| fp64 | off | off | `0` | `3.47e-18` |
| fp64 | off | on | `0` | `0` |
| fp64 | on | off | `0` | `0` |
| fp64 | on | on | `0` | `0` |

`tests/kernels/test_wave_backward_kernel.py` still has pre-existing failures:
the correctness worker saw the same `5 failed, 5 passed` result with the new
path and with `GPUREC_DTS_BACKWARD_DEVICE_SCALARS=0`, so those failures are not
attributable to this proposal.

### Timings

Benchmark command:

```bash
FAMS={10,50} REPS={9,15} WARMUPS=5 python /tmp/gpurec_profile/bench_uniform_backward.py
```

Supervisor event timings:

| Families | Scalar path | Mean | Median | Min | Peak allocation |
|---:|---|---:|---:|---:|---:|
| 10 | old Python scalar extraction | `46.744 ms` | `46.648 ms` | `46.317 ms` | `2.695 GB` |
| 10 | new device scalar load | `44.159 ms` | `44.155 ms` | `43.635 ms` | `2.695 GB` |
| 50 | old Python scalar extraction | `149.638 ms` | `149.606 ms` | `147.924 ms` | `10.305 GB` |
| 50 | new device scalar load | `148.596 ms` | `148.385 ms` | `147.268 ms` | `10.305 GB` |

The 50-family pair was rerun in reverse order:

| Families | Scalar path | Mean | Median | Min |
|---:|---|---:|---:|---:|
| 50 | new device scalar load | `147.902 ms` | `147.850 ms` | `146.017 ms` |
| 50 | old Python scalar extraction | `149.403 ms` | `149.778 ms` | `147.650 ms` |

The 50-family win is therefore about `1.0-1.5 ms`; the 10-family win was
larger, about `2.5 ms`, because fixed host synchronization costs are a larger
share of the smaller workload.

### Nsight Systems

Nsight Systems captures:

| Capture | Path |
|---|---|
| old scalar extraction | `/tmp/gpurec_profile/d71cf11_device_scalars_old_fams50.nsys-rep` |
| new device scalar load | `/tmp/gpurec_profile/d71cf11_device_scalars_new_fams50.nsys-rep` |

Single-capture backward event times under Nsight overhead:

| Scalar path | Backward event time | GPU kernel span | Summed kernel time | Kernel launches |
|---|---:|---:|---:|---:|
| old Python scalar extraction | `161.802 ms` | `161.522 ms` | `134.370 ms` | 3136 |
| new device scalar load | `159.916 ms` | `159.745 ms` | `134.712 ms` | 3136 |

The kernel launch count is unchanged. The summed kernel time is essentially the
same and is slightly higher in this one capture, which is consistent with the
new path adding two scalar `tl.load`s per split program. The wall-time gain
comes from host-side dependency removal:

| CUDA API / mem event | Old | New | Delta |
|---|---:|---:|---:|
| `cudaStreamSynchronize` calls | 318 | 252 | `-66` |
| `cudaStreamSynchronize` API time | `118.277 ms` | `116.564 ms` | `-1.713 ms` |
| `cudaMemcpyAsync` calls | 640 | 574 | `-66` |
| D2H copy events | 260 | 194 | `-66` |
| D2H GPU copy time | `0.227 ms` | `0.171 ms` | `-0.056 ms` |
| D2H bytes | `33.191 KB` | `32.927 KB` | `-264 B` |
| `cudaLaunchKernel` calls | 2835 | 2835 | `0` |

The `-66` deltas are exactly `2 * 33 split waves`, which confirms that these
were the `log_pD/log_pS` scalar extractions.

The DTS accum kernel bucket did not improve:

| Scalar path | `_dts_cross_backward_accum_kernel` total | Launches | Max launch |
|---|---:|---:|---:|
| old Python scalar extraction | `26.131 ms` | 33 | `4.306 ms` |
| new device scalar load | `26.409 ms` | 33 | `4.356 ms` |

That is expected. This proposal does not reduce DTS math or memory traffic; it
moves scalar parameter access from the host into the kernel. The right success
metric is synchronization/copy count and end-to-end wall time, not the DTS
kernel bucket.

### Nsight Compute

NCU was run on the largest DTS accum launch, using `--launch-skip 4` under the
CUDA profiler range:

| Metric | Old Python scalar extraction | New device scalar load |
|---|---:|---:|
| Duration | `4.368 ms` | `4.413 ms` |
| DRAM read | `2.026 GB` | `2.027 GB` |
| DRAM write | `1.319 GB` | `1.319 GB` |
| DRAM throughput | `76.0%` | `75.3%` |
| SM throughput | `23.9%` | `23.7%` |
| L2 throughput | `51.3%` | `51.3%` |
| Registers/thread | 78 | 72 |
| Register occupancy limit | 6 blocks/SM | 7 blocks/SM |
| Achieved occupancy | `49.6%` | `58.0%` |
| Eligible warps/scheduler | `0.149` | `0.161` |
| Global RED ops | `15.93 M` | `15.93 M` |
| Constant-cache requests | 2527 | 2546 |

The NCU sample confirms that the kernel resource profile is basically the same.
The new path adds only a tiny scalar-load footprint: constant-cache requests
increase by 19 on this launch, while bulk DRAM traffic and global reductions
are unchanged. The measured single-launch duration is about `1%` slower under
NCU replay, but the end-to-end Nsight Systems interval is still faster because
66 host synchronization points are gone.

Decision: promote proposal 3. It is a low-risk cleanup, improves the measured
50-family backward by about `1-1.5 ms`, improves the 10-family case more, and
removes exactly the synchronization/copy pattern it targeted. The next ranked
proposal should move to DTS parameter and `grad_mt` reductions; proposal 3 does
not change the remaining PyTorch reduction bucket.

## Proposal 4 follow-up: DTS reduction accumulation

Proposal 4 targets the post-DTS reductions in the scalar-parameter uniform path:

```python
grad_log_pD[0] += param_pD.sum()
grad_log_pS[0] += param_pS.sum()
mt_contrib = grad_Pibar_l.sum(dim=0) + grad_Pibar_r.sum(dim=0)
grad_mt[0] += mt_contrib
```

The implementation adds optional accumulation targets to
`_dts_cross_backward_accum_kernel`. For the promoted path, the Triton program
keeps its existing per-split `sum_pD` and `sum_pS` values and atomically adds
them directly into one-element `grad_log_pD` and `grad_log_pS` tensors. That
lets Python skip materializing `param_pD/param_pS` and skip the two PyTorch
scalar reductions for selected split waves.

The `grad_mt` path was also implemented behind the same flag family, but was
not promoted. Accumulating `grad_mt[s]` inside the split kernel creates
species-lane atomics into only `S=1999` targets, so its contention profile is
much worse than the two scalar parameter atomics.

Current defaults:

```bash
GPUREC_DTS_BACKWARD_REDUCTION_ACCUM=scalar
GPUREC_DTS_BACKWARD_REDUCTION_ACCUM_MIN_SPLITS=8192
```

Useful overrides:

```bash
GPUREC_DTS_BACKWARD_REDUCTION_ACCUM=0      # old materialized reduction path
GPUREC_DTS_BACKWARD_REDUCTION_ACCUM=all    # scalar params plus grad_mt, slower
```

The `8192` split threshold is deliberately conservative. It avoids increasing
register pressure on small split waves where the launch-saving benefit is too
small, while still catching the large 50-family split waves where the PyTorch
reduction launches are visible.

### Subagent workflow

| Role | Task | Result |
|---|---|---|
| Static reviewer | Check algebra and where direct accumulation is safe | Scalar `grad_log_pD/grad_log_pS` accumulation is equivalent for the current `G=1` scalar-parameter uniform path; vector/specieswise parameter cases should stay on generic paths |
| Correctness worker | Test direct-kernel parity, active-mask cases, and finite differences | Passed for fp32/fp64, active mask off/on, scalar-only and `grad_mt` variants |
| Performance worker | Check whether the proposal was present at the starting commit | Found no proposal-4 flags before the supervisor patch; supervisor ran the actual benchmarks and Nsight profiles |
| Supervisor | Implement guarded kernel targets, benchmark thresholds, run Nsys/NCU, choose default | Promoted scalar-only thresholded accumulation; left `grad_mt` opt-in |

The static review also called out the important restriction: this is safe for
scalar `log_pD/log_pS` and the current `G=1` uniform fused path. A familywise or
specieswise parameter layout would need indexed/scattered targets rather than
one scalar accumulation target.

### Correctness

Focused checks after the implementation:

| Check | Result |
|---|---:|
| `py_compile` over changed files | passed |
| `tests/kernels/test_dts_backward_accum_kernel.py` after adding direct reduction-target parity | 15 passed |
| default focused suite after promoting scalar threshold | 35 passed |
| forced old materialized path, `GPUREC_DTS_BACKWARD_REDUCTION_ACCUM=0` | 30 passed |
| full opt-in path, `GPUREC_DTS_BACKWARD_REDUCTION_ACCUM=all` | 30 passed |

The new direct-kernel test compares materialized reductions against in-kernel
accumulation. It covers fp32/fp64, active-mask off/on, scalar-only
accumulation, and the opt-in `grad_mt` accumulation target.

The correctness worker also ran a direct parity script across dtype and mask
variants:

| Variant | Max observed diff |
|---|---:|
| scalar-only fp32 | `1.49e-08` |
| scalar-only fp64 | `5.55e-17` |
| `grad_mt` vector/scalar targets | same tolerance range |

Finite-difference smoke checks also passed:

```bash
GPUREC_DTS_BACKWARD_REDUCTION_ACCUM=scalar pytest ... test_grad_log_pD_vs_fd test_grad_log_pS_vs_fd
GPUREC_DTS_BACKWARD_REDUCTION_ACCUM=all    pytest ... test_grad_log_pD_vs_fd test_grad_log_pS_vs_fd test_grad_mt_vs_fd
```

### Timings

Initial all-wave benchmark, before adding the split threshold:

| Families | Reduction mode | Mean | Median | Min | Decision |
|---:|---|---:|---:|---:|---|
| 10 | old materialized | `44.951 ms` | `44.947 ms` | `44.724 ms` | baseline |
| 10 | scalar params, all split waves | `45.289 ms` | `45.129 ms` | `44.760 ms` | slower |
| 10 | scalar params + `grad_mt` | `45.797 ms` | `45.820 ms` | `45.234 ms` | slower |
| 50 | old materialized | `149.128-151.349 ms` | `148.885-149.382 ms` | `147.492-148.093 ms` | baseline range |
| 50 | scalar params, all split waves | `148.258-149.311 ms` | `147.909-149.235 ms` | `145.996-147.123 ms` | small win/noise |
| 50 | scalar params + `grad_mt` | `150.722-151.115 ms` | `150.704-150.818 ms` | `148.978-149.835 ms` | slower |

Threshold sweep for scalar-only accumulation:

| Families | Threshold | Mean | Median | Min |
|---:|---:|---:|---:|---:|
| 10 | `4096` | `44.029 ms` | `43.989 ms` | `43.813 ms` |
| 10 | `8192` | `44.266 ms` | `43.889 ms` | `43.781 ms` |
| 10 | `16000` | `44.424 ms` | `44.293 ms` | `43.993 ms` |
| 50 | `4096` | `147.733 ms` | `147.662 ms` | `146.048 ms` |
| 50 | `8192` | `147.623 ms` | `147.163 ms` | `146.185 ms` |
| 50 | `16000` | `148.480 ms` | `148.341 ms` | `147.014 ms` |

Final event checks after making the thresholded scalar path the default:

| Families | Path | Mean | Median | Min | Peak allocation |
|---:|---|---:|---:|---:|---:|
| 10 | old forced with `GPUREC_DTS_BACKWARD_REDUCTION_ACCUM=0` | `44.676 ms` | `44.572 ms` | `44.323 ms` | `2.695 GB` |
| 10 | new default scalar threshold `8192` | `44.725 ms` | `44.591 ms` | `44.246 ms` | `2.695 GB` |
| 50 | old forced with `GPUREC_DTS_BACKWARD_REDUCTION_ACCUM=0` | `148.449 ms` | `147.777 ms` | `146.245 ms` | `10.305 GB` |
| 50 | new default scalar threshold `8192` | `147.563 ms` | `147.752 ms` | `145.985 ms` | `10.305 GB` |

The 10-family result is effectively neutral in the final short run. The
50-family mean improves by about `0.9 ms` in that run, and the threshold sweep
showed `8192` as the best 50-family setting. Since this document's target is
the 50-tree backward pass, the thresholded scalar path is a reasonable default.

### Nsight Systems

Nsight Systems captures:

| Capture | Path |
|---|---|
| old materialized reductions | `/tmp/gpurec_profile/prop4_reductions_old_fams50.nsys-rep` |
| scalar threshold `8192` | `/tmp/gpurec_profile/prop4_reductions_scalar8192_fams50.nsys-rep` |

Single-capture backward event times under Nsight overhead:

| Path | Backward event time | Kernel launches | Summed kernel time |
|---|---:|---:|---:|
| old materialized reductions | `159.983 ms` | 3136 | `134.555 ms` |
| scalar threshold `8192` | `158.519 ms` | 3100 | `134.363 ms` |

The mechanism is launch cleanup, not a major change to bulk GPU work:

| Metric | Old | Scalar threshold `8192` | Delta |
|---|---:|---:|---:|
| `cudaLaunchKernel` calls | 2835 | 2799 | `-36` |
| `cudaLaunchKernel` API time | `5.044 ms` | `4.922 ms` | `-0.122 ms` |
| PyTorch float-sum reductions | 245 launches, `5.457 ms` | 227 launches, `5.353 ms` | `-18`, `-0.104 ms` |
| PyTorch vectorized add kernels | 366 launches, `0.440 ms` | 348 launches, `0.417 ms` | `-18`, `-0.023 ms` |
| `_dts_cross_backward_accum_kernel` bucket | 33 launches, `26.411 ms` | 33 launches, `26.266 ms` | `-0.145 ms` |
| `_wave_backward_uniform_kernel` bucket | `39.545 ms` | `39.542 ms` | unchanged |
| `_uniform_cross_pibar_vjp_tree_kernel` bucket | `37.903 ms` | `37.986 ms` | unchanged/noise |
| `_dts_fused_kernel` bucket | `12.134 ms` | `12.131 ms` | unchanged |

CUDA API synchronization and copies do not change:

| API / mem event | Old | Scalar threshold `8192` |
|---|---:|---:|
| `cudaStreamSynchronize` | 252 calls, `116.377 ms` | 252 calls, `117.046 ms` |
| `cudaDeviceSynchronize` | 6 calls, `7.239 ms` | 6 calls, `7.217 ms` |
| `cudaMemcpyAsync` | 574 calls, `1.633 ms` | 574 calls, `1.471 ms` |
| D2H copies | 194 events, `0.175 ms`, `32.927 KB` | 194 events, `0.170 ms`, `32.927 KB` |
| D2D copies | 378 events, `0.391 ms`, `35.638 MB` | 378 events, `0.391 ms`, `35.638 MB` |

This is expected. Proposal 3 removed the scalar host extractions. Proposal 4
removes follow-up GPU reductions and their launch overhead, but it does not
remove any CPU pruning decisions or D2H shape queries.

### Nsight Compute

NCU was run on the largest DTS accumulation launch (`42155` splits):

| Metric | Old materialized reductions | Scalar threshold `8192` |
|---|---:|---:|
| Duration | `4.414 ms` | `4.375 ms` |
| DRAM read | `2.027 GB` | `2.026 GB` |
| DRAM write | `1.320 GB` | `1.318 GB` |
| Total DRAM bytes | `3.346 GB` | `3.345 GB` |
| DRAM throughput | `75.26%` | `75.91%` |
| Memory throughput | `758.0 GB/s` | `764.5 GB/s` |
| L1/TEX throughput | `25.29%` | `25.29%` |
| L2 throughput | `51.26%` | `51.22%` |
| Compute throughput | `23.68%` | `23.87%` |
| SM busy | `11.20%` | `11.25%` |
| Issue slots busy | `11.20%` | `11.25%` |
| Achieved occupancy | `57.98%` | `49.57%` |
| Registers/thread | 72 | 78 |
| Register occupancy limit | 7 blocks/SM | 6 blocks/SM |
| Active warps/scheduler | `6.97` | `5.96` |
| Eligible warps/scheduler | `0.161` | `0.154` |
| Global RED ops | `15.93 M` | `16.02 M` |
| L2 excessive global sectors | `224.1 M` | `224.1 M` |
| L1 hit rate | `54.83%` | `55.69%` |
| L2 hit rate | `68.04%` | `67.44%` |

Stall mix:

| Stall | Old | Scalar threshold `8192` |
|---|---:|---:|
| Barrier | `18.31` warps/issue | `15.17` warps/issue |
| Long scoreboard | `16.97` | `15.26` |
| MIO throttle | `13.40` | `10.04` |
| Short scoreboard | `6.69` | `6.09` |
| LG throttle | `2.75` | `2.60` |

The extra scalar atomics are visible as a small increase in global RED
instructions (`15.93 M -> 16.02 M`), but the kernel remains dominated by the
same memory/reduction behavior as before. The added constexpr path increases
registers from 72 to 78 in this compiled variant, lowering occupancy, so the
single-kernel improvement is small. The end-to-end improvement comes mostly
from removing 36 tiny PyTorch launches around the large split waves.

The `grad_mt` accumulation variant is slower for the same reason the static
review predicted. It does not add two scalar atomics per split program; it adds
many atomics into a small species vector, creating contention while also keeping
more values live in the already register-sensitive DTS accumulation kernel.

Decision: promote only scalar `grad_log_pD/grad_log_pS` accumulation for large
split waves. Keep `grad_mt` accumulation available for experiments with
`GPUREC_DTS_BACKWARD_REDUCTION_ACCUM=all`, but do not use it by default.

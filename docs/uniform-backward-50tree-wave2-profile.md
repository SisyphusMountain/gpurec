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
   backward**, after rechecking 10-tree and 50-tree correctness/timing. This is
   the simplest measured win: about `4.9 ms` median at 50 trees.
2. **Merge the two species loops in `_dts_cross_backward_accum_kernel`** behind
   an env flag. Profile with NCU before making it default. Watch register count,
   occupancy, MIO throttle, and DRAM bytes.
3. **Remove CUDA scalar-to-Python extraction for `log_pD/log_pS` in DTS backward
   kernels**. This should reduce sync count and is low-risk if implemented as a
   device-pointer load.
4. **Kernelize/fuse DTS parameter and `grad_mt` reductions**. Start with scalar
   `grad_log_pD/grad_log_pS`; handle `grad_mt` carefully because species-level
   atomic contention may or may not beat the current PyTorch reductions.
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

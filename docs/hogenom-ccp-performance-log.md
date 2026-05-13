# HOGENOM CCP Performance Log

## Objective

Minimize warm forward+backward runtime over the full HOGENOM CCP dataset while
preserving likelihood and gradient correctness.  Every optimization attempt
should be backed by measurements, preferably Nsight Systems / Nsight Compute
for GPU behavior and focused CUDA tests for parity.

Dataset/settings used unless noted:

- `tests/data/HOGENOM/hogenom/hogenom_families.local.txt`
- species tree: `output_alerax_corrected/species_trees/inferred_species_tree.newick`
- `float32`
- specieswise rates
- uniform origination probabilities
- `fixed_iters_E=6`, `fixed_iters_Pi=6`, `neumann_terms=6`

## Current Diagnosis

The previous lockstep resident schedule was wrong for performance.  It built
per-family phased waves first, then merged families by local wave index.  That
does not globally pack all ready clades in a resident batch, and it leaves many
waves well below the intended wave cap.

The new scheduler handles split-count-zero leaf clades first, then uses a
global ready queue across the resident batch and packs ready clades up to
`max_wave_size`.  This is closer to the DAG schedule we want.

The forward ready-queue policy is not optimal on all CCP-like DAGs: a low-level
regression case with cap 2 used seven total waves where a latest-valid reverse
compaction uses six.  The scheduler now keeps the forward policy when it hits
the simple leaf-first lower bound, but if it exceeds that bound it also builds a
reverse non-leaf schedule from roots/sinks backward and uses it when it reduces
the wave count.  The accepted HOGENOM `depth_first_fit, clade_budget=315000`
layout remains at 258 waves (`[102, 65, 48, 30, 13]` by batch), so this is a
guard against wasted waves rather than a new HOGENOM timing win.  A post-change
warm event run measured median forward+backward at 1.2468 s, peak allocated
5.904 GiB.

Measured warm runtime after the scheduler change:

| family chunk | batches | total waves | max wave | warm fwd+bwd | peak alloc | peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 43 | 3185 | 3217 | 3.145 s | 1.08 GiB | 1.72 GiB |
| 100 | 11 | 927 | 8192 | 1.624 s | 2.65 GiB | 2.83 GiB |
| 200 | 6 | 522 | 8192 | 1.390 s | 4.23 GiB | 8.47 GiB |
| 250 | 5 | 430 | 8192 | 1.360 s | 5.05 GiB | 10.27 GiB |
| 300 | 4 | 371 | 8192 | 1.256 s | 5.92 GiB | 11.95 GiB |
| 400 | 3 | 286 | 8192 | 1.307 s | 7.92 GiB | 16.31 GiB |
| 600 | 2 | 193 | 8192 | 1.289 s | 15.24 GiB | 16.71 GiB |

The chunk-300 row includes the later DTS launch-warp tuning, no-host pruning,
2D `J^T` warp retuning, and forward wave-step warp retuning.  Without the
no-host-pruning override the same tuned code measured 1.321 s before the
wave-step retune.  The row also includes the DTS parent-reduction block retune.
The other rows are the scheduler/self-loop-tuned measurements used for memory
tradeoff decisions.

The best warm value inside the 5-6 GiB allocated target is chunk size 300 at
about 1.26 s.  Larger chunks keep reducing waves but give small returns relative
to memory: chunk 600 uses 15.24 GiB for only about 44 ms over chunk 300.

The first pass for large chunks is still expensive because Triton compiles
larger wave/kernel variants.  Removing `W: tl.constexpr` from the retained 2D
backward self-loop kernels reduced one source of wave-size-specific
compilation, but warm-up is still much slower than steady state.

## Nsight Findings

Nsight Systems on tuned chunk size 300 with
`GPUREC_BACKWARD_NO_CPU_PRUNING=1` measured one profiled pass at 1.371 s
(`nsys` overhead relative to the uninstrumented 1.256 s median).  The measured
pass had 52,557 CUDA kernel launches and 1.147 s of GPU kernel time.

Top kernel families in the chunk-300 `nsys` report:

| kernel | launches | total GPU time | avg launch |
| --- | ---: | ---: | ---: |
| `_wave_backward_uniform_2d_jt_kernel` | 2226 | 0.268 s | 120.4 us |
| `_wave_step_uniform_kernel` | 2226 | 0.195 s | 87.5 us |
| `_dts_cross_backward_accum_kernel` | 358 | 0.161 s | 451.1 us |
| `_dts_parent_reduced_ge2_stage1_kernel` | 708 | 0.101 s | 142.9 us |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | 358 | 0.106 s | 296.9 us |
| `_wave_backward_uniform_2d_precompute_kernel` | 371 | 0.067 s | 181.8 us |

The corresponding `nsys` export did not include GPU metrics counters.  The
previous profiled pass with the same kernel defaults but host pruning enabled
reported GR active 83.5%, SMs active 72.6%, SM issue 24.7%, compute warps in
flight 48.1%, DRAM read 27.7%, and DRAM write 20.5%.

Nsight Compute on representative chunk-300 kernels:

- `_wave_step_uniform_kernel` at grid 8192 is healthy: 81.3% compute/memory
  throughput, 93% achieved occupancy, 5.33 waves per SM.
- `_wave_backward_uniform_2d_jt_kernel` at a representative grid 4337 is
  memory-heavy: about 67% DRAM throughput, 56-57% compute throughput, 48-49%
  achieved occupancy, 11.29 waves per SM.  Registers cap theoretical occupancy
  at 50%.
- Early `_wave_backward_uniform_2d_jt_kernel` launches still include tiny grids
  such as grid 1, which are expected after the leaf/frontier phase and are
  deeply underutilized individually.  They are not the main total-time driver.
- `_dts_cross_backward_accum_kernel` improved after forcing 8 launch warps:
  a representative grid 2718 now uses 40 registers/thread, reaches 100%
  theoretical occupancy and 94.1% achieved occupancy, and takes 225.6 us.
  Throughput is still only 55.9% memory, 52.1% DRAM, and 31.3% compute, so the
  kernel remains a target, but the launch shape is better than the earlier
  96-register, 41.7%-theoretical-occupancy version.  Some later launches are
  very small grids and show tail/underfill effects.

Self-loop configuration sweep at chunk size 300:

| setting | warm fwd+bwd | conclusion |
| --- | ---: | --- |
| default before sweep | 1.371 s | baseline |
| `GPUREC_SELF_LOOP_2D_BLOCK_W=2` | 1.406 s | worse |
| `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=4` | 1.341 s | better |
| `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=2` | 1.341 s | roughly tied |
| `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=1` | 1.651 s | much worse |
| `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=4 GPUREC_SELF_LOOP_2D_BLOCK_NODES=32` | 1.353 s | worse than 128 |
| `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=4 GPUREC_SELF_LOOP_2D_BLOCK_NODES=128` | 1.333 s | best measured |
| `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=4 GPUREC_SELF_LOOP_2D_BLOCK_NODES=256` | 1.340 s | worse than 128 |

The retained 2D path now defaults to `JT_NUM_WARPS=4` and
`BLOCK_NODES=128`.  The environment variables still override those defaults.

## Main Branch Notes

The `main` branch docs mention several older backward alternatives:

- the original fused self-loop baseline;
- Proposal 0, the retained 2D Triton self-loop path, previously fastest on
  `test_trees_1000` but memory-heavy;
- an exact CUDA no-split path behind `GPUREC_CUDA_SELF_LOOP_NOSPLIT`;
- a staged tree prototype behind `GPUREC_SELF_LOOP_TREE_STAGED`.

The current branch only retains the 2D Triton backward path.  If the next
optimization targets backward kernels, the best candidates are either reducing
the 2D backward memory traffic/register pressure or selectively restoring a
leaner no-split/path-specific kernel from `main` for the cases where it applies.

## Verified So Far

- Scheduler CPU unit tests cover global packing, wave cap, and topological
  ordering.
- Targeted CUDA tests pass for `GeneReconModel` forward/backward modes and
  resident batch parity.
- Targeted CUDA tests pass for `UniformChunkedReconModel` parity and chunk
  subset gradients.
- HOGENOM timings above were measured with `scripts/profile_hogenom_ccp_pass.py`
  after a warm pass.
- HOGENOM chunk-size 300 and 200 were profiled with Nsight Systems.
- HOGENOM chunk-size 300 top kernels were profiled with Nsight Compute.
- Tuned chunk-size 300 was re-profiled with Nsight Systems after changing the
  retained 2D JT launch defaults.
- DTS launch-warp tuning was re-profiled with Nsight Systems and Nsight Compute
  before promoting `GPUREC_DTS_NUM_WARPS=8` to the default.
- The no-host-pruning mode was timed warm, checked with Nsight Systems, and
  covered by the same targeted CUDA parity suite as the default path.
- The 2D `J^T` warp retune was checked warm and with Nsight Systems before
  changing the default to `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=2`.
- The forward wave-step warp retune was checked warm, with Nsight Systems, and
  with Nsight Compute before changing the default to
  `GPUREC_WAVE_STEP_NUM_WARPS=8`.
- The DTS parent-reduction block retune was checked warm, with Nsight Systems,
  and with Nsight Compute before changing the default to
  `GPUREC_DTS_PARENT_BLOCK_S=256`.
- First-fit clade packing was timed and rejected as a default because it raised
  peak allocation without improving warm runtime.

## Next Hypotheses

1. Treat chunk size 300 as the current 5-6 GiB target configuration.
2. Do not promote clade-only first-fit packing.  It reduced scheduled waves but
   increased peak allocation, so the next batching attempt needs a better memory
   proxy.
3. Inspect the DTS backward accumulation path first.  It is now a larger share
   of wall time than launch overhead and has lower utilization than the packed
   forward wave kernel.
4. If profiling supports it, prototype a lower-scratch or lower-register
   backward path and compare against the retained 2D path for correctness and
   warm runtime.

## Host Pruning Sync Plan

The remaining chunk-300 `nsys` pass has 1.190 s of GPU kernel time inside a
1.458 s profiled pass.  One possible source of the CPU/GPU gap is the backward
wave loop: for every wave it builds a device active mask, then synchronizes on
`active_mask.any()` and `active_mask.sum().item()` before launching the actual
wave kernels.  That makes sense when pruning skips enough waves, but it can be
counterproductive after global scheduling if most waves are active.

Next experiment:

- add an opt-in `GPUREC_BACKWARD_NO_CPU_PRUNING=1` mode;
- in that mode, keep the device active mask when pruning is enabled, but do not
  query it from the host and do not skip entire waves on the CPU;
- benchmark HOGENOM chunk 300 against the current default, then confirm any
  improvement with `nsys`;
- only promote it if likelihood/gradient parity holds and measured
  forward+backward time improves.

Results:

| setting | warm fwd+bwd | backward | `nsys` pass | GPU kernel time | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| default host pruning | 1.321 s | 0.971 s | 1.458 s | 1.190 s | baseline |
| `GPUREC_BACKWARD_NO_CPU_PRUNING=1` | 1.292 s | 0.944 s | 1.407 s | 1.186 s | accepted opt-in |
| no active pruning/mask | 1.515 s | 1.169 s | not run | not run | rejected |

The accepted mode still builds and passes the device active mask to kernels, so
it preserves the default pruning approximation.  It only stops synchronizing on
per-wave host `.any()` and `.sum().item()` decisions.  The no-mask run changes
the gradient slightly and is slower, so the active mask itself is still useful.

Follow-up: promote no-host-pruning to the default.  Every accepted HOGENOM
profile now uses `GPUREC_BACKWARD_NO_CPU_PRUNING=1`; leaving it opt-in makes
default model calls slower than the measured path.  Change the default so the
device active mask is still computed and passed to kernels, but host-side
wave-skipping synchronizations are disabled unless
`GPUREC_BACKWARD_NO_CPU_PRUNING=0` is set for diagnostics.  Accept only if the
targeted parity tests pass with the new default and with the diagnostic
host-pruning path.

Result: promoted.  The environment variable now defaults to `1`, so ordinary
calls use the measured no-host-pruning path.  Setting
`GPUREC_BACKWARD_NO_CPU_PRUNING=0` restores the old host wave-skipping path for
diagnostics.  Targeted parity tests passed under both defaults.  On the accepted
HOGENOM depth-first 315k layout, the new default measured 1.253 s in one noisy
three-pass run, while forcing the old path measured 1.274 s in a one-pass
diagnostic run.

## 2D Self-Loop Retuning Plan

After no-host pruning, the largest remaining kernel bucket is still
`_wave_backward_uniform_2d_jt_kernel`: 2,226 launches and 0.271 s total GPU time
in the chunk-300 `nsys` run.  Earlier tuning selected
`GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=4` and `GPUREC_SELF_LOOP_2D_BLOCK_NODES=128`,
but that was measured before the host-pruning change.

Next experiment: rerun a narrow launch-shape sweep under
`GPUREC_BACKWARD_NO_CPU_PRUNING=1`.

- keep `BLOCK_W=1`, since `BLOCK_W=2` was already worse;
- test `JT_NUM_WARPS` in `{2, 4, 8}`;
- test `BLOCK_NODES` in `{64, 128, 256}`;
- accept a change only if warm whole-dataset time improves and a follow-up
  profiler run confirms the 2D `J^T` bucket shrinks.

Results:

| setting | warm fwd+bwd | backward | `nsys` pass | 2D `J^T` GPU time | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| no-host baseline, `JT_NUM_WARPS=4`, `BLOCK_NODES=128` | 1.292 s | 0.944 s | 1.407 s | 0.2708 s | baseline |
| `JT_NUM_WARPS=2` | 1.284 s | 0.937 s | 1.404 s | 0.2680 s | accepted |
| `JT_NUM_WARPS=8` | 1.298 s | 0.949 s | not run | not run | rejected |
| `BLOCK_NODES=64` | 1.289 s | 0.942 s | not run | not run | small win, not best |
| `BLOCK_NODES=256` | 1.290 s | 0.940 s | not run | not run | small win, not best |
| `JT_NUM_WARPS=2 BLOCK_NODES=64` | 1.288 s | 0.941 s | not run | not run | does not stack |
| `JT_NUM_WARPS=2 BLOCK_NODES=256` | 1.290 s | 0.941 s | not run | not run | does not stack |

Set the retained 2D `J^T` default to 2 warps and leave `BLOCK_NODES=128`.

Follow-up precompute/store sweep plan:

The retained 2D self-loop path has a separate
`GPUREC_SELF_LOOP_2D_NUM_WARPS` knob for the precompute and parameter-store
kernels.  The accepted retune above targeted the `J^T` kernel via
`GPUREC_SELF_LOOP_2D_JT_NUM_WARPS`, but the current HOGENOM profile still spends
about 0.066 s in `_wave_backward_uniform_2d_precompute_kernel` and 0.049 s in
`_wave_backward_uniform_param_store_kernel`.  Test
`GPUREC_SELF_LOOP_2D_NUM_WARPS` in `{4, 8, 16}` on the accepted depth-first
315k layout.  Accept only if event timing improves and an `nsys` profile shows
the precompute/store buckets shrink without increasing the dominant `J^T`
bucket.

Result: rejected without `nsys` follow-up.  Neither alternative was a clear
event-timing improvement over the default 8-warps setting:

| `GPUREC_SELF_LOOP_2D_NUM_WARPS` | event median fwd+bwd | median backward | peak alloc | decision |
| ---: | ---: | ---: | ---: | --- |
| 4 | 1.2562 s | 0.9377 s | 5.904 GiB | rejected |
| 8 | 1.2468-1.2531 s | 0.9276-0.9341 s | 5.904 GiB | keep default |
| 16 | 1.2508 s | 0.9326 s | 5.904 GiB | rejected |

## Current 2D Jt NCU Plan

The accepted depth-first 315k profile still spends about 0.263-0.265 s in
`_wave_backward_uniform_2d_jt_kernel`, making it the largest single kernel
bucket.  Before changing that path again, refresh Nsight Compute on the current
default layout because the scheduler, no-host-pruning default, and several
launch-shape defaults have changed since the earlier representative NCU note.

Plan:

- run Nsight Compute on a representative `_wave_backward_uniform_2d_jt_kernel`
  launch from the accepted HOGENOM layout;
- capture occupancy, register count, memory throughput, DRAM throughput, and
  stall/memory-pressure evidence;
- only prototype another 2D-kernel change if NCU points to a concrete knob such
  as row blocking, species blocking, or register pressure.  Otherwise continue
  treating the 2D path as tuned and move to a different bucket.

Initial NCU result:

- `ncu_jt_current_depthff315_skip20.ncu-rep` captured a tail wave with grid 31
  and was not representative.
- `ncu_jt_current_depthff315_skip540.ncu-rep` captured a full wave with grid
  8192 and block size 64.  Duration was 402.6 us for the profiled launch.
- The kernel is memory-bound on the full wave: DRAM throughput 86.4%, L2
  throughput 66.0%, SM throughput 28.9%.
- It uses 255 registers/thread, giving only 16.7% theoretical occupancy and
  16.0% achieved occupancy.
- NCU reports 1.18M local spill instructions and 1.18M local-memory spill
  requests for the launch, plus 53% excessive global sectors.

Next experiment:

- sweep existing Jt launch-shape knobs before rewriting the kernel:
  `GPUREC_SELF_LOOP_2D_BLOCK_NODES` in `{32, 64, 128, 256}` and
  `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS` in `{1, 2, 4}`;
- accept a setting only if warm whole-dataset forward+backward improves and a
  follow-up NCU/`nsys` profile confirms reduced Jt cost or lower register/local
  memory pressure;
- if the knobs do not help, the next code-level target is reducing the amount
  of per-program species-tree state kept live in `_wave_backward_uniform_2d_jt_kernel`.

Sweep result:

| setting | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `GPUREC_SELF_LOOP_2D_BLOCK_NODES=32` | 1.2589 s | 0.3188 s | 0.9401 s | 5.904 GiB | rejected |
| `GPUREC_SELF_LOOP_2D_BLOCK_NODES=64` | 1.2460 s | 0.3185 s | 0.9283 s | 5.904 GiB | tied |
| `GPUREC_SELF_LOOP_2D_BLOCK_NODES=128` | 1.2466 s | 0.3193 s | 0.9274 s | 5.904 GiB | keep default |
| `GPUREC_SELF_LOOP_2D_BLOCK_NODES=256` | 1.2549 s | 0.3201 s | 0.9348 s | 5.904 GiB | rejected |
| `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=1` | 1.5681 s | 0.3176 s | 1.2506 s | 5.904 GiB | rejected |
| `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=2` | 1.2448 s | 0.3203 s | 0.9242 s | 5.904 GiB | keep default |
| `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS=4` | 1.2491 s | 0.3197 s | 0.9299 s | 5.904 GiB | rejected |

Conclusion: keep the current `BLOCK_NODES=128` and `JT_NUM_WARPS=2` defaults.
The existing knobs do not remove the register/local-memory pressure shown by
NCU.  Further Jt work should be a code-level reduction in live per-program
state or memory traffic, not another launch-shape sweep.

Row-blocking follow-up:

- test `GPUREC_SELF_LOOP_2D_BLOCK_W` in `{2, 4}` against the current default
  1;
- accept only if whole-dataset timing improves and NCU does not show worse
  register spilling;
- reject immediately if Triton compilation fails or if the larger row block
  worsens warm timing, because larger row blocks multiply the already large
  `[S, W]` live vectors in the Jt program.

Row-blocking result:

| `GPUREC_SELF_LOOP_2D_BLOCK_W` | median fwd+bwd | median forward | median backward | peak alloc | decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.2448-1.2477 s | 0.3200 s | 0.9242-0.9283 s | 5.904 GiB | keep default |
| 2 | 1.9433 s | 0.3178 s | 1.6256 s | 5.904 GiB | rejected |
| 4 | 5.2160 s | 0.3161 s | 4.8995 s | 5.904 GiB | rejected |

Conclusion: keep one row per Jt program.  Larger row blocks reduce the number
of row programs but multiply the already register-heavy live tensor shape and
make backward much slower.

## CUDA No-Split Specieswise Plan

The next code-level 2D alternative should be narrow: restore the exact CUDA
no-split row kernel from `main` only for no-split waves, then adapt it for the
HOGENOM specieswise parameter layout.  The main-branch kernel is attractive
because it runs all Neumann terms for one no-split wave inside one CUDA launch
using shared row-local state, avoiding the retained 2D path's precompute,
six Jt launches, parameter-store launch, and full `[W, S]` Jt scratch traffic
for leaf/no-split waves.

Constraints before promotion:

- keep it opt-in behind an environment variable until correctness and profiling
  are clear;
- route only auto-wrapped specieswise/shared no-split waves where constants are
  `[S]` and `dts_r is None`;
- extend the CUDA kernel's old scalar `grad_log_pD` and `grad_log_pS`
  accumulation to species-vector gradients by atomically accumulating per
  species, matching the existing vector E/Ebar/transfer accumulators;
- return only `v_k` and skip the external per-element `aw*` reductions when the
  CUDA path already accumulated all self-loop parameter gradients;
- verify with targeted resident/model parity tests before timing;
- accept as a default only if whole-HOGENOM stream timing improves and `nsys`
  confirms the self-loop bucket shrinks enough to offset the extra atomics and
  NVRTC path overhead.

Opt-in implementation result:

- added `gpurec/core/kernels/wave_backward_cuda.py`, an NVRTC no-split fp32
  CUDA row kernel adapted from `main`;
- extended old scalar D/S accumulation to species-vector D/S accumulation;
- routes only when `GPUREC_CUDA_SELF_LOOP_NOSPLIT=1`, `_auto_wrapped` is true,
  `dts_r is None`, dtype is fp32, and compact species-tree levels are present;
- preloads wheel-provided `libnvrtc-builtins.so` from `nvidia/cu13/lib` so the
  CUDA Python NVRTC bindings can compile in this venv.

Correctness checks:

- default path targeted suite: 15 passed;
- opt-in CUDA no-split targeted model/chunked suite: 8 passed;
- HOGENOM same-process comparison against the retained Triton path:
  loss diff 0, max gradient abs diff 0.0224, mean gradient abs diff 0.0006,
  gradient-norm relative delta 7.9e-6.

Performance result on HOGENOM depth-first 315k / wave-cap 8192:

| setting | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| retained 2D default | 1.2477 s | 0.3200 s | 0.9283 s | 5.904 GiB | baseline |
| `GPUREC_CUDA_SELF_LOOP_NOSPLIT=1` | 1.2171 s | 0.3190 s | 0.8981 s | 5.904 GiB | opt-in win |

Nsight Systems result:

- report: `profiling/hogenom_ccp/nsys_stream_depthff315_cuda_nosplit.nsys-rep`;
- total CUDA kernel time: 1.114 s versus prior 1.143 s baseline;
- `_wave_backward_uniform_2d_jt_kernel`: 0.232 s / 1464 launches versus
  roughly 0.265 s / 1548 launches in the baseline;
- `_wave_backward_uniform_2d_precompute_kernel`: 0.061 s / 244 launches;
- `_wave_backward_uniform_param_store_kernel`: 0.045 s / 244 launches;
- `gpurec_wave_backward_nosplit_uniform_fp32`: 0.0187 s / 14 launches.

Conclusion: the no-split CUDA router is a real HOGENOM speed win, currently
kept opt-in while deciding whether to make the auto/fallback behavior robust
enough for the default path.

Promotion plan:

- change `GPUREC_CUDA_SELF_LOOP_NOSPLIT` default from off to `auto`;
- in `auto`, try the CUDA no-split path for eligible waves but fall back to the
  retained 2D Triton path if the NVRTC/CUDA loader is unavailable;
- keep `GPUREC_CUDA_SELF_LOOP_NOSPLIT=0` as an explicit disable switch;
- treat `1`, `true`, `on`, `yes`, `force`, and `required` as explicit modes
  that should raise on CUDA no-split setup failure rather than silently falling
  back;
- rerun the targeted suite in default/auto mode and with the disable switch;
- rerun HOGENOM default timing to confirm the promoted behavior matches the
  opt-in result.

Promotion result:

- default mode is now `auto`: eligible no-split waves use the CUDA row kernel
  when available, and `GPUREC_CUDA_SELF_LOOP_NOSPLIT=0` disables it;
- explicit modes `1`, `true`, `on`, `yes`, `force`, and `required` still raise
  on setup failure;
- targeted suite in default/auto mode: 15 passed;
- targeted suite with `GPUREC_CUDA_SELF_LOOP_NOSPLIT=0`: 15 passed;
- HOGENOM default timing after promotion: median forward+backward 1.2186 s,
  median forward 0.3189 s, median backward 0.8996 s, peak allocated 5.904 GiB.

## Post-Leaf DAG Scheduling Plan

The current resident scheduler already runs a leaf-only phase first, then packs
ready non-leaf clades from all families into capped waves.  That is the right
dependency model, but the implementation is still a list-scheduling heuristic:
it uses a forward ready queue and, only when the simple lower bound is missed, a
reverse latest-valid compaction pass.  The user concern is valid: after the leaf
phase this should be treated as a single capacity-constrained DAG layering
problem, not as per-CCP waves.

Next implementation step:

- keep the leaf phase unchanged because leaves use different initialization;
- build additional post-leaf schedule candidates over the full batch DAG;
- add a Coffman-Graham-style candidate, because it is designed to layer a DAG
  under a fixed width cap and should improve cases where ready-queue order
  wastes slots;
- choose the valid candidate with the fewest non-leaf waves, using the existing
  forward and reverse schedules as fallbacks;
- preserve the DTS partial-row guard as an optional secondary capacity
  constraint;
- accept only after topological/unit tests, resident parity tests, and a fresh
  HOGENOM timing check.

Implementation result:

- added a Coffman-Graham-style layered compaction candidate for the post-leaf
  non-leaf DAG;
- the scheduler now compares forward ready-queue, reverse latest-valid, and
  layered candidates when the forward schedule misses the simple lower bound;
- added a unit regression where forward and reverse ready-queue schedules need
  four non-leaf waves at cap 2, while the layered candidate reaches the feasible
  three-wave schedule;
- targeted scheduler/model parity tests pass.

HOGENOM result for the accepted depth-first 315k / wave-cap 8192 layout:

| batch | leaf clades | non-leaf clades | max depth | leaf waves | work lower bound | total lower bound | scheduled waves |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 27,830 | 287,155 | 98 | 4 | 36 | 102 | 102 |
| 1 | 24,939 | 289,963 | 61 | 4 | 36 | 65 | 65 |
| 2 | 23,300 | 250,598 | 45 | 3 | 31 | 48 | 48 |
| 3 | 8,739 | 95,381 | 28 | 2 | 12 | 30 | 30 |
| 4 | 1,783 | 13,425 | 12 | 1 | 2 | 13 | 13 |

The HOGENOM wave counts are already equal to the lower bound
`leaf_waves + max(max_depth, ceil(nonleaf_clades / 8192))`, so this scheduling
change fixes non-optimal DAG layouts but does not reduce the accepted HOGENOM
wave count.  A fresh stream timing run measured median forward+backward
1.2477 s, median forward 0.3200 s, median backward 0.9283 s, and peak allocated
5.904 GiB, matching the previous HOGENOM result within noise.

## Forward Wave-Step Retuning Plan

With the latest no-host-pruning and 2D `J^T` defaults, the next largest kernel
bucket is `_wave_step_uniform_kernel`: 2,226 launches and 0.214 s total GPU time
in the chunk-300 `nsys` run.  Earlier NCU showed a representative full wave as
healthy, so expect only small gains, but this bucket is now large enough to test
instead of guessing.

Next experiment:

- add diagnostic environment overrides for the shared uniform wave-step launch
  shape;
- test `GPUREC_WAVE_STEP_NUM_WARPS` in `{2, 4, 8}` and
  `GPUREC_WAVE_STEP_BLOCK_S` in `{128, 256, 512}`;
- accept a default change only if warm whole-dataset timing improves and an
  `nsys` run confirms the wave-step bucket shrinks without moving cost into
  final Pibar row recomputation.

Results:

| setting | warm fwd+bwd | forward | `nsys` pass | wave-step GPU time | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| `NUM_WARPS=4 BLOCK_S=256` | 1.284 s | 0.348 s | 1.404 s | 0.2136 s | baseline |
| `NUM_WARPS=2 BLOCK_S=256` | 1.291 s | 0.355 s | not run | not run | rejected |
| `NUM_WARPS=8 BLOCK_S=256` | 1.264 s | 0.328 s | 1.380 s | 0.1979 s | accepted |
| `NUM_WARPS=16 BLOCK_S=256` | 1.480 s | 0.542 s | not run | not run | rejected |
| `NUM_WARPS=8 BLOCK_S=128` | 1.459 s | 0.520 s | not run | not run | rejected |
| `NUM_WARPS=8 BLOCK_S=512` | 1.274 s | 0.341 s | not run | not run | worse than 256 |

Nsight Compute on `NUM_WARPS=8 BLOCK_S=256` full-wave launches reports about
89% compute/memory throughput, 93% achieved occupancy, 40 registers/thread, and
10.67 waves per SM.  The final Pibar recomputation bucket also improved in the
`nsys` run (`0.0305 s -> 0.0276 s`) because it shares the uniform wave-step
launch helper.  Set the default `GPUREC_WAVE_STEP_NUM_WARPS` value to 8 and keep
`BLOCK_S=256`.

## Pibar From-UD Retuning Plan

After the forward wave-step retune, `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel`
is still about 0.106 s of GPU time.  It uses the same compact species-level
tree reduction shape for every split side and currently hardcodes
`BLOCK_S=256` and `num_warps=4`.

Next experiment:

- add diagnostic overrides for this Pibar-from-UD correction kernel;
- test `GPUREC_PIBAR_UD_NUM_WARPS` in `{2, 4, 8}`;
- test `GPUREC_PIBAR_UD_BLOCK_S` in `{128, 256, 512}`;
- accept a default change only if warm whole-dataset timing improves and `nsys`
  confirms the Pibar-from-UD bucket shrinks.

Results:

| setting | warm fwd+bwd | backward | `nsys` pass | Pibar-from-UD GPU time | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| `NUM_WARPS=4 BLOCK_S=256` | 1.264 s | 0.936 s | 1.380 s | 0.1064 s | baseline |
| `NUM_WARPS=2 BLOCK_S=256` | 1.270 s | 0.942 s | not run | not run | rejected |
| `NUM_WARPS=8 BLOCK_S=256` | 1.262 s | 0.935 s | 1.386 s | 0.1041 s | rejected |
| `NUM_WARPS=8 BLOCK_S=128` | 1.276 s | 0.948 s | not run | not run | rejected |
| `NUM_WARPS=8 BLOCK_S=512` | 1.264 s | 0.936 s | not run | not run | tied baseline |

The 8-warp variant shrank the target bucket, but the full `nsys` pass regressed
and total GPU kernel time increased slightly (`1.1614 s -> 1.1624 s`).  Keep the
default at `NUM_WARPS=4 BLOCK_S=256`; retain the environment overrides only for
future diagnostics.

## Parent-Reduced DTS Retuning Plan

The parent-reduced DTS forward recompute still accounts for a meaningful share
of the pass: `_dts_parent_reduced_ge2_stage1_kernel` is about 0.108 s,
`_dts_eq1_to_rows_kernel` about 0.020 s, and stage 2 about 0.009 s.  This path
uses `BLOCK_S=128` and `tile_splits=64` today.

Next experiment:

- add diagnostic overrides for the parent-reduced DTS launch shape;
- test `GPUREC_DTS_PARENT_NUM_WARPS` in `{2, 4, 8}`;
- test `GPUREC_DTS_PARENT_BLOCK_S` in `{64, 128, 256}`;
- test `GPUREC_DTS_PARENT_TILE_SPLITS` in `{32, 64, 128}`;
- accept a default change only if whole-dataset timing improves and `nsys`
  confirms the combined DTS stage buckets shrink.

Results:

| setting | warm fwd+bwd | forward | peak alloc | `nsys` pass | stage1 GPU time | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `BLOCK_S=128 TILE_SPLITS=64` | 1.264 s | 0.328 s | 5.92 GiB | 1.380 s | 0.1078 s | baseline |
| `BLOCK_S=256 TILE_SPLITS=64` | 1.256 s | 0.324 s | 5.92 GiB | 1.371 s | 0.1012 s | accepted |
| `BLOCK_S=512 TILE_SPLITS=64` | 1.260 s | 0.325 s | 5.92 GiB | not run | not run | worse than 256 |
| `BLOCK_S=128 TILE_SPLITS=32` | 1.277 s | 0.332 s | 7.39 GiB | not run | not run | rejected |
| `BLOCK_S=128 TILE_SPLITS=128` | 1.272 s | 0.329 s | 5.47 GiB | not run | not run | memory lower but slower |
| `NUM_WARPS=8 BLOCK_S=128` | 1.272 s | 0.333 s | 5.92 GiB | not run | not run | rejected |

Nsight Compute on `BLOCK_S=256` stage-1 launches reports roughly 87-88% DRAM
throughput, 97-100% achieved occupancy, and 40 registers/thread.  The kernel is
memory-bound but better packed with 6 species blocks per row instead of 11.
Set the default DTS parent block size to 256 and keep `tile_splits=64`.

## Next Batch-Policy Plan

Equal family chunks are a blunt proxy for active runtime tensor size.  HOGENOM
families vary substantially in clade and split count, and the current best
family-count chunk (`300`) has four batches with a 5.92 GiB peak.  Before
rewriting kernels, test these batch layouts:

- family-count `300` baseline;
- sequential clade budgets around the same largest active batch size;
- if sequential clade budgets help, consider a first-fit decreasing clade/split
  bin packer as an explicit opt-in mode.

Acceptance criteria for changing batching defaults or adding a new policy:

- likelihood and specieswise gradient must match the current stream baseline;
- measured warm forward+backward must improve at comparable peak allocation;
- any result that changes defaults must be profiled with `nsys` after timing.

Initial metadata-only results:

| policy | budget/chunk | batches | total waves | largest clades | largest splits |
| --- | ---: | ---: | ---: | ---: | ---: |
| family count | 300 | 4 | 371 | 305750 | 806746 |
| sequential clade | 300000 | 4 | 371 | 299999 | 793473 |
| sequential clade | 325000 | 4 | 349 | 324477 | 861792 |
| sequential clade | 350000 | 3 | 278 | 349853 | 924182 |
| first-fit decreasing clade | 275000 | 4 | 315 | 275000 | 743092 |
| first-fit decreasing clade | 300000 | 4 | 291 | 300000 | 822127 |
| first-fit decreasing clade | 325000 | 4 | 279 | 325000 | 905606 |
| first-fit decreasing clade | 350000 | 3 | 243 | 349998 | 922425 |

Sequential clade budgets do not improve the 5-6 GiB target in timing:
325k clades measured 1.337 s at 6.53 GiB, compared with the family-count 300
baseline at 1.332 s and 5.92 GiB.  First-fit decreasing clade packing was worth
testing as an explicit opt-in policy because its metadata suggested many fewer
waves at similar clade caps.

Timing the first-fit policy exposed a missing memory proxy.  First-fit
decreasing clade packing with `clade_budget=300000` and no family-count cap
reduced the metadata wave count to 291, but the measured run used 8.85 GiB and
1.331 s.  Adding `family_chunk_size=300` reduced the metadata wave count to 316,
but still used 8.85 GiB and measured 1.341 s.  This is not an improvement over
plain family-count 300, so clade-only bin packing should not become the default.
If we revisit non-contiguous batches, the policy needs a better memory proxy
than clade count alone, likely involving the dense static/runtime layout induced
by family composition.

## Depth-Aware Batch-Packing Plan

The per-batch global scheduler now matches the leaf-first lower bound on the
HOGENOM chunk-300 layout:

| batch | waves | lower bound | leaf waves | critical depth | non-leaf work waves |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 85 | 85 | 3 | 82 | 33 |
| 1 | 88 | 88 | 4 | 84 | 30 |
| 2 | 98 | 98 | 4 | 94 | 35 |
| 3 | 100 | 100 | 2 | 98 | 19 |

So the remaining scheduling waste is not inside a resident batch.  It is that
equal family-count chunks pay the long CCP critical-path tail four times.  The
next experiment is an opt-in depth-aware first-fit batch policy:

- compute per-family leaf count, non-leaf count, and CCP critical depth;
- sort families by descending critical depth and clade count;
- place each family into the existing clade-budgeted batch that minimizes the
  increase in the batch lower bound
  `ceil(leaves / max_wave_size) + max(depth, ceil(nonleaves / max_wave_size))`;
- fall back to opening a new batch when no existing batch fits the clade budget;
- accept only if warm HOGENOM timing improves without exceeding the current
  5-6 GiB target materially and the standard likelihood/gradient tests pass.

Metadata and timing results:

| packing | budget/chunk | batches | total waves | warm fwd+bwd | peak alloc | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sequential family count | 300 | 4 | 371 | 1.256 s | 5.92 GiB | previous lean baseline |
| clade first-fit | 300000 | 4 | 291 | 1.331 s | 8.85 GiB | rejected earlier |
| depth first-fit | 300000 | 5 | 263 | 1.251 s | 5.44 GiB | accepted opt-in |
| depth first-fit | 315000 | 5 | 258 | 1.247 s | 5.90 GiB | best lean setting |
| depth first-fit | 320000 | 5 | 256 | 1.252 s | 8.02 GiB | rejected |
| depth first-fit | 325000 | 5 | 255 | 1.249 s | 8.09 GiB | rejected |

Use `batch_packing="depth_first_fit"` with `clade_budget=315000` as the current
best HOGENOM CCP lean scheduling option.  It improves the median stream
forward+backward pass by about 9 ms over the chunk-300 baseline and keeps peak
allocation in the same 5-6 GiB envelope.  Larger budgets show that fewer waves
alone are not enough; the induced per-wave split layout can cross a memory
threshold and erase the timing gain.

Nsight follow-up plan:

- profile the accepted `depth_first_fit, clade_budget=315000` layout with
  Nsight Systems using the same no-host-pruning fast path;
- compare total launches, total kernel time, and the dominant kernel buckets
  against the previous sequential chunk-300 `nsys` report;
- accept the policy as an opt-in scheduling improvement only if the profiler
  confirms that the wave-count reduction does not move time into another kernel
  bucket or CPU-side gap.

Nsight Systems result for `depth_first_fit, clade_budget=315000`:

| layout | waves | `nsys` pass | kernel launches | GPU kernel time | peak alloc |
| --- | ---: | ---: | ---: | ---: | ---: |
| sequential chunk 300 | 371 | 1.371 s | 52,557 | 1.147 s | 5.92 GiB |
| depth first-fit 315k | 258 | 1.343 s | 48,030 | 1.143 s | 5.90 GiB |

Top kernel comparison:

| kernel | sequential total | depth first-fit total | note |
| --- | ---: | ---: | --- |
| `_wave_backward_uniform_2d_jt_kernel` | 0.267 s / 2226 launches | 0.265 s / 1548 launches | fewer but larger launches |
| `_wave_step_uniform_kernel` | 0.195 s / 2226 launches | 0.192 s / 1548 launches | fewer but larger launches |
| `_dts_cross_backward_accum_kernel` | 0.161 s / 358 launches | 0.166 s / 244 launches | regresses per launch |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | 0.106 s / 358 launches | 0.107 s / 244 launches | unchanged total |
| `_dts_parent_reduced_ge2_stage1_kernel` | 0.101 s / 708 launches | 0.097 s / 478 launches | small win |

The scheduling change lowers launch count and profiler wall time, but most GPU
time is still row/split work in the same backward buckets.  Further scheduling
work should use a better memory/work proxy than clade count alone; otherwise
larger waves can simply trade fewer launches for heavier DTS and Pibar work.
The next kernel-side target is still the retained 2D self-loop backward path
and its parameter-gradient/reduction overhead.

## 2D Backward Alternative Survey Plan

The accepted HOGENOM scheduling policy still spends about 0.265 s in
`_wave_backward_uniform_2d_jt_kernel`, 0.066 s in its precompute kernel,
0.049 s in parameter-store, and about 0.054 s in PyTorch reductions.  Before
rewriting that path, inspect `main` and archived docs for the older non-2D /
no-split / staged alternatives:

- grep `main` docs and kernels for `2D`, `nosplit`, `staged`, and
  self-loop backward variants;
- identify which alternatives still match the current uniform resident layout
  and which are obsolete because they assume removed metadata;
- prototype only an opt-in path that can be benchmarked against the current
  retained 2D path with identical likelihood/gradient tests;
- reject any alternative that loses the current 1.247 s HOGENOM stream timing
  or materially worsens the 5-6 GiB memory target.

Survey outcome:

- The exact CUDA no-split kernel in `main` is implementation-ready but was
  routed only for global/shared no-split waves with scalar D/S parameter
  gradients.  HOGENOM specieswise rates use species-vector D/S gradients, so
  adopting that path would require a new gradient layout in the CUDA kernel and
  would only affect leaf/no-split waves.
- The staged tree prototype exists in `main`, but the archived timings reject
  it versus Proposal 0 2D in its current form.
- Proposal 0 2D remains the fastest documented broad self-loop strategy, but
  it returns full per-element parameter VJPs.  In the current HOGENOM `nsys`
  report, parameter-store plus PyTorch reductions account for about 0.103 s of
  GPU kernel time.

Next experiment: add an opt-in 2D parameter-accumulation path for the
auto-wrapped shared/specieswise layout.  It should keep the existing 2D
precompute and `J^T` kernels, but replace the final per-element parameter-store
outputs and host-side PyTorch reductions with a Triton atomic accumulation
kernel.  This is narrower than porting the CUDA no-split path and directly
targets overhead visible in the current HOGENOM profile.  Accept it only if
the targeted parity suite passes and warm HOGENOM timing improves under
`nsys`/event timing.

Result: rejected and removed.  The opt-in prototype passed the targeted parity
suite, but warm HOGENOM timing regressed:

| setting | warm fwd+bwd | backward | peak alloc | `nsys` pass | GPU kernel time |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth first-fit 315k baseline | 1.247 s | 0.927 s | 5.90 GiB | 1.343 s | 1.143 s |
| 2D param accumulation prototype | 1.267 s | 0.948 s | 5.78 GiB | 1.429 s | 1.223 s |

Nsight Systems showed why: `_wave_backward_uniform_param_accum_kernel` cost
0.193 s, replacing `_wave_backward_uniform_param_store_kernel` at 0.049 s plus
PyTorch reductions at 0.054 s.  The atomic accumulation path saved some memory
and launch count, but the atomics were much slower than storing row
contributions and reducing them with PyTorch.  Do not revive this approach
unless it uses a staged reduction rather than per-row atomics.

## Depth-Aware Wave-Cap Sweep Plan

The accepted depth-aware policy was measured only at the default
`max_wave_size=8192`.  Since the per-batch scheduler is lower-bound optimal for
that cap, the next scheduling knob is the cap itself:

- expose `max_wave_size` in `scripts/profile_hogenom_ccp_pass.py` so HOGENOM
  timing runs can vary it without ad hoc scripts;
- test the accepted depth-aware packing around `clade_budget=315000` with
  wave caps above and below 8192;
- reject larger caps if the 2D self-loop scratch or static layout pushes peak
  allocation outside the 5-6 GiB target or if whole-dataset timing regresses;
- profile any apparent win with Nsight Systems before promoting it.

Follow-up experiment after adding the scheduler compaction guard:

- rerun event timings for the current accepted setting
  `depth_first_fit, clade_budget=315000, max_wave_size=8192`;
- rerun candidate caps 12288 and 16384 with the same warm cache and
  `GPUREC_BACKWARD_NO_CPU_PRUNING=1`;
- accept a larger cap only if the median full-stream forward+backward improves
  by more than run-to-run noise, the peak allocation remains acceptable for the
  5-6 GiB lean target, and an `nsys` pass shows the improvement is not just
  shifted into larger DTS/Pibar kernels;
- if the larger cap is only marginally faster but crosses the memory target,
  keep 8192 as the default and record the larger cap as an opt-in borderline
  speed/memory tradeoff.

Wave-cap sweep result:

| `max_wave_size` | waves by batch | total waves | event median fwd+bwd | median forward | median backward | peak alloc | peak reserved | decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8192 | `[102, 65, 48, 30, 13]` | 258 | 1.2468 s | 0.3197 s | 0.9276 s | 5.904 GiB | 7.234 GiB | keep lean default |
| 12288 | `[101, 64, 47, 29, 13]` | 254 | 1.2530 s | 0.3194 s | 0.9335 s | 6.008 GiB | 6.656 GiB | rejected |
| 16384 | `[100, 63, 47, 29, 13]` | 252 | 1.2407 s | 0.3191 s | 0.9214 s | 6.347 GiB | 6.770 GiB | not default; borderline opt-in |

Nsight Systems for `max_wave_size=16384` produced
`profiling/hogenom_ccp/nsys_stream_depthff315_wave16384.nsys-rep`.  The
profiled pass took 1.355 s with 47,844 CUDA kernel launches and 1.148 s of GPU
kernel time.  Compared with the accepted 8192-cap `nsys` report (48,030
launches, 1.143 s GPU kernel time, 1.343 s profiled pass), the larger cap
removes only 186 launches and shifts work into larger kernels:

| kernel | 8192 total | 16384 total | note |
| --- | ---: | ---: | --- |
| `_wave_backward_uniform_2d_jt_kernel` | 0.265 s / 1548 launches | 0.264 s / 1512 launches | launch count slightly lower |
| `_wave_step_uniform_kernel` | 0.192 s / 1548 launches | 0.193 s / 1512 launches | no real improvement |
| `_dts_cross_backward_accum_kernel` | 0.166 s / 244 launches | 0.170 s / 244 launches | worse per launch |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | 0.107 s / 244 launches | 0.107 s / 244 launches | unchanged total |
| `_wave_backward_uniform_2d_precompute_kernel` | 0.066 s / 258 launches | 0.067 s / 252 launches | unchanged total |

Conclusion: keep `max_wave_size=8192` as the lean default.  `16384` can be an
explicit speed/memory experiment because event timings were about 6 ms faster,
but `nsys` does not show a GPU-time win and peak allocation rises to 6.35 GiB.

## Batch Memory-Proxy Diagnosis Plan

The accepted depth-first batch policy still uses clade count as its hard memory
cap.  The HOGENOM runs show that this is incomplete: moving from 315k to 320k
clades saves only two waves but raises peak allocation from about 5.90 GiB to
about 8.02 GiB.  Before adding another packing heuristic, measure what changes
across the accepted and rejected layouts:

- total clades and total splits per batch;
- max split rows in any wave and total split rows in split-bearing waves;
- max ge2 parent count and max ge2 fanout;
- wave-count lower bound versus actual waves;
- which batch owns the measured peak allocation.

Use that table to decide whether the next policy should add a split budget,
a max-wave-split budget, a family composition penalty, or no new policy at all.
Do not promote a batch policy unless likelihood/gradient tests pass and HOGENOM
timing plus `nsys` improve within the 5-6 GiB target.

Diagnosis result:

The memory cliff is not explained by total clades or total splits alone.  It is
caused by the parent-reduced DTS forward scratch for GE2 clades.  That scratch
allocates two `[n_ge2_groups * ceil(max_ge2_fanout / tile_splits), S]` partial
arrays for a wave.  At the default `tile_splits=64`, the accepted 315k layout's
largest batch-0 parent partial is about 0.844 GiB, while the rejected 320k
layout puts a `ge2_max_fanout=9774` clade in a wave with 1910 GE2 groups,
making the parent partial about 2.885 GiB.  That accounts for the observed
batch-0 peak allocation jump from about 5.90 GiB to about 8.05 GiB.

Next experiment:

- add an opt-in scheduler cap on the GE2 DTS partial-row proxy
  `n_ge2_groups * ceil(max_ge2_fanout / tile_splits)`;
- expose it in the HOGENOM profiler as `--max-dts-partial-rows`;
- test whether rejected larger clade budgets such as 320k or 325k can stay in
  the 5-6 GiB envelope by splitting only the problematic high-fanout waves;
- accept only if likelihood/gradient tests pass and warm timing plus `nsys`
  beat or tie the current 315k/8192 default without a memory cliff.

Result: implemented as an opt-in scheduler guard, not a new default.  The
profiler script exposes it as `--max-dts-partial-rows`, and the model accepts
`max_dts_partial_rows`.

| layout | DTS partial-row cap | total waves | event median fwd+bwd | peak alloc | `nsys` pass | GPU kernel time | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| depth-first 315k | none | 258 | 1.2468 s | 5.904 GiB | 1.343 s | 1.143 s | current lean default |
| depth-first 320k | none | 256 | 1.252 s earlier | 8.02 GiB | not profiled | not profiled | rejected memory cliff |
| depth-first 320k | 100000 | 256 | 1.2456 s | 5.928 GiB | 1.347 s | 1.142 s | opt-in memory guard |
| depth-first 325k | 100000 | 255 | 1.2413 s | 6.305 GiB | not profiled | not profiled | too much memory |
| depth-first 325k | 80000 | 255 | 1.2444 s | 6.024 GiB | not profiled | not profiled | borderline memory |
| depth-first 325k | 70000 | 255 | 1.2566 s | 5.948 GiB | not profiled | not profiled | slower |

The 320k capped layout produced
`profiling/hogenom_ccp/nsys_stream_depthff320_dtsrows100k.nsys-rep`: 47,920
CUDA kernel launches and 1.142 s GPU kernel time.  That is only a fractional GPU
kernel-time improvement over the 315k baseline, while profiled wall time was
slightly worse.  The value of the cap is avoiding pathological scratch when
experimenting with larger clade budgets; it should not replace the 315k default
unless a follow-up profile shows a clear end-to-end win.

Follow-up stacking plan:

- combine the DTS partial-row cap with larger `max_wave_size` values, because
  the cap may remove the memory cliff that made larger waves risky;
- test `clade_budget=320000, max_dts_partial_rows=100000` with
  `max_wave_size` 12288 and 16384;
- reject the combination if peak allocation crosses the lean target or if
  `nsys` shows the same pattern as the uncapped large-wave sweep: fewer launches
  but no GPU-kernel-time improvement.

Stacking result: rejected as a default.  The capped 320k layout gets fewer
waves with larger caps, but timing does not improve materially and memory
crosses the lean target:

| clade budget | DTS partial-row cap | `max_wave_size` | total waves | event median fwd+bwd | peak alloc | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 320000 | 100000 | 8192 | 256 | 1.2456 s | 5.928 GiB | opt-in memory guard |
| 320000 | 100000 | 12288 | 252 | 1.2453 s | 6.081 GiB | rejected default |
| 320000 | 100000 | 16384 | 250 | 1.2450 s | 6.397 GiB | rejected |

The larger wave caps also reintroduced very expensive first-run compilation /
setup costs, with warmup passes around 58-71 s.  Keep the lean default at
315k/8192 and keep the capped 320k layout as an opt-in memory-guarded
experiment only.

## DTS Accumulation Plan

The current tuned chunk-300 profile still spends about 0.179 s of GPU time in
`_dts_cross_backward_accum_kernel`.  NCU showed this kernel at roughly 52%
memory throughput, 26% compute throughput, 96 registers/thread, and 41.7%
theoretical occupancy.  Before rewriting it, sweep the existing two-stage
`grad_mt` reduction tile size:

- current hardcoded tile: 128 split rows per partial tile;
- candidate tile sizes: 32, 64, 128, 256, 512;
- acceptance criterion: whole-dataset warm forward+backward improves without
  changing likelihood/gradient and without increasing peak allocation
  materially.

If none of those improve the pass, the next kernel-level work should target
register pressure and memory traffic inside `_dts_cross_backward_accum_kernel`
rather than its reduction staging.

Tile-size sweep results at chunk size 300:

| `GPUREC_DTS_GRAD_MT_TILE_SPLITS` | warm fwd+bwd | peak alloc | conclusion |
| ---: | ---: | ---: | --- |
| 64 | 1.336 s | 5.92 GiB | worse |
| 128 | 1.332 s | 5.92 GiB | current default |
| 192 | 1.332 s | 5.92 GiB | no clear win |
| 256 | 1.326-1.331 s | 5.92 GiB | not confirmed by `nsys` |
| 512 | 1.338 s | 5.92 GiB | worse |

The 256 value looked slightly better in one uninstrumented run, but a follow-up
Nsight Systems run measured 1.495 s under profiler overhead and
`_dts_cross_backward_accum_kernel` at 0.187 s, worse than the tuned-default
`nsys` report at 1.471 s / 0.179 s.  Keep 128 as the default; retain the
environment override only for future diagnostics.

Retest after depth-aware packing and the no-host-pruning fast path:

| `GPUREC_DTS_GRAD_MT_TILE_SPLITS` | event median fwd+bwd | median forward | median backward | peak alloc | decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 1.2504 s | 0.3191 s | 0.9307 s | 5.904 GiB | rejected |
| 64 | 1.2465 s | 0.3179 s | 0.9284 s | 5.904 GiB | tied |
| 128 | 1.2468 s | 0.3197 s | 0.9276 s | 5.904 GiB | current default |
| 256 | 1.2426 s | 0.3192 s | 0.9235 s | 5.904 GiB | rejected after `nsys` |
| 512 | 1.2447 s | 0.3183 s | 0.9265 s | 5.904 GiB | rejected |

The 256 retest again looked better in event timings, but Nsight Systems still
did not confirm a GPU-time win.  The report
`profiling/hogenom_ccp/nsys_stream_depthff315_dtsgradmt256.nsys-rep` measured
1.347 s profiled, 48,030 CUDA kernel launches, and 1.145 s of GPU kernel time;
the accepted depth-first 315k baseline measured 1.343 s profiled, 48,030
launches, and 1.143 s GPU kernel time.  `_dts_cross_backward_accum_kernel`
measured 0.167 s at tile 256 versus 0.166 s in the baseline.  Keep the tile
default at 128.

Next DTS experiment: reduce the Triton species block size for
`_dts_cross_backward_accum_kernel`.  The current block size is 256 species,
which NCU reports at 96 registers/thread and only 41.7% theoretical occupancy.
Test `GPUREC_DTS_BLOCK_S=64` and `128` against the default `256`; accept a
change only if whole-dataset warm timing improves and the follow-up `nsys`
kernel table confirms lower DTS cost.

DTS launch-shape sweep:

| setting | warm fwd+bwd | `nsys` pass | DTS kernel GPU time | conclusion |
| --- | ---: | ---: | ---: | --- |
| default before sweep | 1.332 s | 1.471 s | 0.179 s | baseline |
| `GPUREC_DTS_BLOCK_S=128` | 1.332 s | not run | not run | no clear win |
| `GPUREC_DTS_BLOCK_S=64` | 1.329 s | 1.479 s | 0.183 s | rejected |
| `GPUREC_DTS_NUM_WARPS=8` | 1.315 s | 1.458 s | 0.166 s | accepted |
| `GPUREC_DTS_NUM_WARPS=16` | 1.315 s | not run | not run | tied with 8 |

Set `_dts_cross_backward_accum_kernel` default `num_warps` to 8.  Keep
`GPUREC_DTS_NUM_WARPS` and `GPUREC_DTS_BLOCK_S` as diagnostic overrides.

## Post-Promotion Batch-Scheduling Sweep Plan

After promoting the CUDA no-split self-loop path to default `auto`, the accepted
HOGENOM CCP stream baseline is about 1.22 s forward+backward at 5.90 GiB peak
allocation with `batch_packing="depth_first_fit"`, `clade_budget=315000`, and
`max_wave_size=8192`.  The per-batch scheduler is already at the leaf-first
lower bound for that layout, so further scheduling work should change batch
composition or memory guards rather than only reorder clades within a batch.

Next experiment:

- use the current default auto no-split path, not the older retained-2D
  baseline;
- retest larger depth-first clade budgets with the DTS partial-row scheduler
  cap, because that cap directly targets the GE2 scratch memory cliff observed
  at 320k+ clades;
- start with `max_wave_size=8192` to isolate batch composition from larger-wave
  effects;
- measure warm whole-dataset stream timing and peak allocation for candidate
  budgets 320k, 325k, 340k, and 350k with `max_dts_partial_rows=100000`;
- profile any apparent event-time win with Nsight Systems and accept it only if
  the profiler shows lower total GPU kernel time or a meaningful launch-count
  reduction without moving cost into DTS/Pibar buckets;
- keep the 315k/8192 layout as the default if larger batches only improve event
  timing within noise or exceed the 5-6 GiB lean target.

Result: rejected as a default.  The only lean-memory candidate with an apparent
event-time improvement was `clade_budget=320000, max_dts_partial_rows=100000`,
but Nsight Systems did not confirm a GPU-time win.

| clade budget | DTS partial-row cap | batches | event median fwd+bwd | median forward | median backward | peak alloc | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 315000 | none | 5 | 1.2197 s | 0.3182 s | 0.9012 s | 5.904 GiB | current default |
| 320000 | 100000 | 5 | 1.2151 s | 0.3184 s | 0.8965 s | 5.928 GiB | profiled, rejected |
| 325000 | 100000 | 5 | 1.2159 s | 0.3194 s | 0.8965 s | 6.305 GiB | rejected memory |
| 340000 | 100000 | 5 | 1.2185 s | 0.3206 s | 0.8990 s | 6.665 GiB | rejected memory |
| 350000 | 100000 | 5 | 1.2211 s | 0.3197 s | 0.9009 s | 6.722 GiB | rejected |

Nsight Systems comparison:

| layout | profiled pass | CUDA launches | GPU kernel time | note |
| --- | ---: | ---: | ---: | --- |
| 315k default, auto no-split | 1.343 s earlier | 47,722 | 1.1136 s | baseline |
| 320k + DTS rows 100k, auto no-split | 1.340 s | 47,612 | 1.1199 s | fewer launches but slower kernels |

The 320k layout removes 110 launches and slightly reduces the 2D `J^T` bucket
(`0.2319 s -> 0.2309 s`), but `_dts_cross_backward_accum_kernel` regresses
(`0.1633 s -> 0.1716 s`).  Keep the 315k/8192 layout as the default and treat
the DTS partial-row cap as a diagnostic/memory-guard option only.

## Current DTS Accumulation NCU Plan

The current auto no-split baseline still spends about 0.163 s of GPU time in
`_dts_cross_backward_accum_kernel`.  The previous DTS launch-shape sweeps found
`GPUREC_DTS_NUM_WARPS=8` best, but those measurements predate the promoted
CUDA no-split default and were mostly Nsight Systems summaries.  Before making
another DTS kernel change, capture Nsight Compute for a representative heavy
current-default DTS launch.

Plan:

- use the accepted `depth_first_fit, clade_budget=315000, max_wave_size=8192`
  layout and current default auto no-split path;
- select a heavy launch from the current `nsys` report rather than guessing;
  launch index 172 is the largest DTS launch in the captured pass at about
  3.1 ms;
- run `ncu --set full` on that launch;
- inspect occupancy, register pressure, memory throughput, cache behavior, and
  replay/spill diagnostics before deciding whether to rewrite the kernel or
  only test a narrow launch-shape option;
- accept no code change without parity tests and an `nsys` whole-dataset
  confirmation.

NCU result for `profiling/hogenom_ccp/ncu_dts_current_depthff315_skip172.ncu-rep`:

- launch shape: grid 43,245, block 256, duration 3.22 ms;
- registers/thread: 40; no local memory spills;
- theoretical occupancy: 100%; achieved occupancy: 98.72%;
- DRAM throughput: 60.05%; L2 throughput: 53.77%; L2 hit rate: 72.08%;
- compute throughput: 27.84%; issue slots busy: 16.91%;
- branch efficiency: 67.34%; average eligible warps/scheduler: 0.40.

Diagnosis: the current DTS accumulation kernel is not register/occupancy
limited.  Another broad launch-shape sweep is unlikely to help.  The profile
instead points to memory traffic and irregular branch behavior.

Next narrow experiment: when a split parent is inactive, the DTS kernel already
writes `pibar_side_active=false` for both child-side rows.  The downstream
Pibar-from-UD kernel checks that mask before reading `pibar_ud` or `pibar_A`.
Therefore, test skipping the expensive inactive-row zero fill of `pibar_ud`
and `pibar_A`.  Accept only if targeted parity tests pass and `nsys` confirms
that the DTS bucket or total GPU kernel time shrinks.

Result: accepted.  `GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO` now defaults to `1`;
set it to `0` to restore the old zero-fill behavior for diagnostics.

Correctness:

- targeted scheduler/model/chunked parity suite: 15 passed;
- HOGENOM loss/gradient are unchanged within the existing fp32 run noise:
  loss 667283.5625 bits, gradient infinity norm about 645.922.

Event timing:

| setting | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| skip inactive zero fill | 1.2080 s | 0.3193 s | 0.8884 s | 5.904 GiB | accepted |
| old zero fill (`GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO=0`) | 1.2209 s | 0.3191 s | 0.9015 s | 5.904 GiB | slower |

Nsight Systems confirmation:

| setting | profiled pass | CUDA launches | GPU kernel time | DTS accumulation bucket |
| --- | ---: | ---: | ---: | ---: |
| old auto no-split baseline | 1.343 s earlier | 47,722 | 1.1136 s | 0.1633 s |
| skip inactive zero fill | 1.313 s | 47,722 | 1.1004 s | 0.1524 s |

The launch count is unchanged; the win is from removing unnecessary memory
writes in inactive DTS split rows.  The Pibar-from-UD bucket remains stable
(`0.1071 s -> 0.1068 s`), so the skipped rows are correctly handled by the
existing side-active mask.

## Commands

Warm whole-dataset stream timing:

```bash
python scripts/profile_hogenom_ccp_pass.py \
  --chunk-size 200 \
  --warmup-runs 1 \
  --profile-runs 1 \
  --mode stream-batches \
  --no-cuda-profiler-api
```

Production closure timing:

```bash
python scripts/profile_hogenom_ccp_pass.py \
  --chunk-size 200 \
  --warmup-runs 1 \
  --profile-runs 1 \
  --mode full \
  --no-cuda-profiler-api
```

Targeted correctness tests:

```bash
pytest -q \
  tests/unit/test_global_wave_scheduler.py \
  tests/integration/test_gene_recon_model.py::test_gene_recon_model_forward_backward_modes \
  tests/integration/test_gene_recon_model.py::test_memory_safe_resident_batches_match_resident_and_slice \
  tests/integration/test_uniform_chunked_model.py::test_chunked_uniform_matches_resident_global_model \
  tests/integration/test_uniform_chunked_model.py::test_chunked_uniform_chunk_subset_nll_and_gradient
```

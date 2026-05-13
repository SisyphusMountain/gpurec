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

## 2D Self-Loop Inactive Scratch-Zero Plan

The DTS inactive-row skip shows that zero-filling masked rows is a real
backward cost.  The retained 2D self-loop path has a similar pattern: the
precompute and `J^T` kernels use the active mask for computation, but still
write zeros into temporary scratch rows for inactive clades.  Those scratch
rows are not read by later 2D work because every later load is also guarded by
the active mask.  The final parameter-store kernel is different: its per-element
parameter output arrays are reduced across all rows, so inactive parameter rows
must remain explicitly zero.

Next experiment:

- add an environment-controlled default-on skip for inactive scratch zero fills
  in `_wave_backward_uniform_2d_precompute_kernel` and
  `_wave_backward_uniform_2d_jt_kernel`;
- do not change `_wave_backward_uniform_param_store_kernel`;
- verify targeted parity tests and HOGENOM loss/gradient;
- compare default skip against `GPUREC_SELF_LOOP_2D_SKIP_INACTIVE_SCRATCH_ZERO=0`;
- accept only if `nsys` confirms the 2D precompute/`J^T` buckets or total GPU
  kernel time shrink.

Result: accepted.  `GPUREC_SELF_LOOP_2D_SKIP_INACTIVE_SCRATCH_ZERO` now defaults
to `1`; set it to `0` to restore the old scratch zero-fill behavior for
diagnostics.  The parameter-store kernel is unchanged and still zero-fills
inactive rows before reduction.

Correctness:

- targeted scheduler/model/chunked parity suite: 15 passed;
- HOGENOM loss/gradient are unchanged within the existing fp32 run noise:
  loss 667283.5625 bits, gradient infinity norm about 645.922.

Event timing:

| setting | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| skip inactive 2D scratch zero fill | 1.1742 s | 0.3189 s | 0.8552 s | 5.904 GiB | accepted |
| old scratch zero fill (`GPUREC_SELF_LOOP_2D_SKIP_INACTIVE_SCRATCH_ZERO=0`) | 1.2071 s | 0.3187 s | 0.8877 s | 5.904 GiB | slower |

Nsight Systems confirmation:

| setting | profiled pass | CUDA launches | GPU kernel time | 2D `J^T` | 2D precompute |
| --- | ---: | ---: | ---: | ---: | ---: |
| DTS inactive skip only | 1.313 s | 47,722 | 1.1004 s | 0.2320 s | 0.0610 s |
| skip inactive 2D scratch zero fill | 1.283 s | 47,722 | 1.0709 s | 0.2097 s | 0.0521 s |

The launch count is unchanged.  The win comes from avoiding inactive-row
temporary-buffer writes in the retained 2D self-loop kernels.

## Current 2D Jt NCU After Scratch-Zero Skip Plan

After the inactive scratch-zero skip, the split-wave 2D `J^T` kernel remains
the largest single GPU bucket: 1,464 launches and 0.2097 s in
`profiling/hogenom_ccp/nsys_stream_depthff315_skip_inactive_2d_scratch.sqlite`.
The previous NCU capture predates the scratch-zero change, so the next step is
to profile the current kernel before changing it again.

Plan:

- use the accepted `depth_first_fit, clade_budget=315000, max_wave_size=8192`
  layout and all current defaults;
- select a heavy current launch from the latest `nsys` report; matching launch
  index 1218 is the largest observed 2D `J^T` launch at about 455 us;
- run `ncu --set full` on that launch;
- compare the post-skip NCU result with the older Jt diagnosis, especially
  register pressure, local spills, occupancy, memory throughput, and excessive
  sector traffic;
- only prototype another 2D `J^T` change if NCU identifies a concrete source of
  remaining overhead; otherwise move to the next profiler bucket.

NCU result for
`profiling/hogenom_ccp/ncu_jt_current_after_scratch_skip1218.ncu-rep`:

- launch shape: grid 8192, block 64, duration 440.10 us;
- registers/thread: 255; local spilling requests: 1,179,648;
- theoretical occupancy: 16.67%; achieved occupancy: 16.26%;
- DRAM throughput: 87.10%; L2 throughput: 65.95%; L2 hit rate: 65.56%;
- compute throughput: 26.45%; issue slots busy: 17.18%;
- branch efficiency: 100%.

Diagnosis: the inactive scratch-zero skip reduced total Jt time by avoiding
unnecessary stores, but the remaining Jt kernel is still the same full-row 2D
strategy: one program owns a whole species row, carries a very wide live vector,
spills to local memory, and is occupancy-limited by registers.  The profiler
does not point to another launch-shape knob; earlier `BLOCK_W`, `BLOCK_NODES`,
and `JT_NUM_WARPS` sweeps already rejected those.  A meaningful Jt improvement
would need a different algorithm, likely staged level kernels or a CUDA split
path, and must be weighed against extra launches and prior staged-tree results.

Next concrete target should therefore be a different profiler bucket unless we
are ready to prototype a broader replacement for the retained 2D split-wave
path.

## Self-Loop Gradient Reduction Reuse Plan

The current accepted `nsys` profile still shows the self-loop parameter-store
kernel at about 0.044 s and PyTorch reduction kernels at about 0.050 s.  The
2D parameter-accumulation prototype was rejected because per-row atomics were
slower than store-plus-reduce, so do not revive that path.  There is, however,
a narrower Python-side redundancy in the specieswise auto-wrapped path:

- `grad_log_pD` reduces `aw0.sum(dim=0)`;
- `grad_max_transfer_mat` reduces `aw2.sum(dim=0)`;
- `grad_E` currently reduces `(aw0 + aw2).sum(dim=0)`, which materializes a
  full `[W, S]` add and launches another full reduction even though the needed
  two row sums are already available.

Next experiment:

- add a narrow `G == 1`, vector-parameter fast path that computes
  `aw0_sum = aw0.sum(dim=0)` and `aw2_sum = aw2.sum(dim=0)` once and reuses
  them for `grad_log_pD`, `grad_max_transfer_mat`, and `grad_E`;
- leave genewise/family-indexed and scalar-global behavior on the existing
  `_scatter_accum` path;
- verify targeted parity tests;
- benchmark HOGENOM event timing and profile with `nsys`;
- accept only if the PyTorch add/reduction bucket or total GPU kernel time
  shrinks without changing likelihood/gradient.

Result: accepted for the specieswise auto-wrapped path.  The change reuses
`aw0.sum(dim=0)` and `aw2.sum(dim=0)` for `grad_E` instead of materializing
and reducing `(aw0 + aw2)`.  Genewise/family-indexed and scalar-global paths
still use the existing `_scatter_accum` implementation.

Correctness:

- targeted scheduler/model/chunked parity suite: 15 passed;
- HOGENOM loss/gradient are unchanged within the existing fp32 run noise:
  loss 667283.5625 bits, gradient infinity norm about 645.922.

Event timing:

| setting | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| reduction reuse | 1.1548 s | 0.3185 s | 0.8365 s | 5.904 GiB | accepted |
| previous accepted baseline | 1.1742 s | 0.3189 s | 0.8552 s | 5.904 GiB | baseline |

Nsight Systems confirmation:

| setting | profiled pass | CUDA launches | GPU kernel time | PyTorch sum-reduce bucket | CUDA add bucket |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous accepted baseline | 1.283 s | 47,722 | 1.0709 s | 0.0497 s / 3131 launches | 0.0166 s |
| reduction reuse | 1.281 s | 47,478 | 1.0605 s | 0.0448 s / 2887 launches | 0.0057 s |

The measured profiler improvement is modest but real: one full-wave reduction
launch per split wave is removed, and the large `aw0 + aw2` elementwise add is
eliminated for the HOGENOM specieswise path.

## Final Pibar Recompute Fusion Plan

The current accepted profile still launches `_wave_pibar_uniform_parent_kernel`
258 times, costing about 0.027 s of GPU kernel time.  This kernel runs after
the last fixed Pi iteration to recompute final Pibar rows and row maxima for
backward.  The last wave-step launch already has the final Pi output in hand,
but it cannot compute Pibar during the main pass because row max/sum of the
final result are only known after all species entries have been produced.

Next experiment:

- add an optional final-Pibar mode to `_wave_step_uniform_kernel`;
- on the last Pi iteration, after storing final Pi, perform the same row
  max/sum and ancestor-walk Pibar computation inside the wave-step kernel;
- keep the separate `_wave_pibar_uniform_parent_kernel` as a fallback and for
  non-final callers;
- preserve the existing root-wave skip, since all-root waves do not need saved
  Pibar rows for backward;
- verify forward/backward parity and HOGENOM loss/gradient;
- benchmark event timing and confirm with `nsys`;
- accept only if the removed Pibar launches are not replaced by an equal or
  larger increase in `_wave_step_uniform_kernel` time.

Result: accepted as a small launch/GPU-time cleanup, with
`GPUREC_FUSE_FINAL_PIBAR=0` retained as the old separate-recompute path.

Correctness:

- targeted scheduler/model/chunked parity suite: 15 passed;
- HOGENOM loss/gradient are unchanged within the existing fp32 run noise:
  loss 667283.5625 bits, gradient infinity norm about 645.922.

Event timing:

| setting | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| fused final Pibar | 1.1568 s | 0.3193 s | 0.8375 s | 5.904 GiB | accepted, tied |
| old separate final Pibar (`GPUREC_FUSE_FINAL_PIBAR=0`) | 1.1568 s | 0.3201 s | 0.8376 s | 5.904 GiB | tied |

Nsight Systems confirmation:

| setting | profiled pass | CUDA launches | GPU kernel time | wave-step bucket | final Pibar bucket |
| --- | ---: | ---: | ---: | ---: | ---: |
| reduction-reuse baseline | 1.281 s | 47,478 | 1.0605 s | 0.1937 s | 0.0270 s |
| fused final Pibar | 1.265 s | 47,220 | 1.0528 s | 0.2194 s | removed |

The final Pibar work moves into `_wave_step_uniform_kernel`, but not one-for-one:
the wave-step bucket grows by 0.0257 s while the separate Pibar bucket
disappears at 0.0270 s, and 258 launches are removed.  The event timing is
within noise, so this is a cleanup rather than a user-visible speed jump.

## Non-Leaf Leaf-Term Specialization Plan

The global scheduler already emits leaf waves first, followed by non-leaf
internal/root waves.  However, the forward wave-step kernel and the retained 2D
self-loop backward kernels still carry the leaf-hit term on every wave by
checking `leaf_species_index[row] == species`.  For non-leaf waves that term is
always impossible and should be the constant `-inf`.

Next experiment:

- store the wave phase in `wave_metas` so forward/backward can identify
  leaf-only waves without re-inspecting rows;
- add a `HAS_LEAF_TERM` constexpr to the forward wave-step and retained 2D
  backward kernels;
- pass `HAS_LEAF_TERM=false` for non-leaf waves so those kernels skip the
  leaf-species load, equality mask, leaf-logp load, and `t5` contribution;
- keep the conservative old behavior when phase metadata is absent;
- verify targeted parity tests and HOGENOM loss/gradient;
- benchmark event timing and confirm with `nsys`, accepting only if the
  wave-step and/or retained 2D buckets shrink.

Result: accepted.  Wave metadata now records the scheduler phase, and
`GPUREC_SPECIALIZE_NONLEAF_LEAF_TERM` defaults to `1`; set it to `0` to force
the old behavior where every wave carries the leaf-hit term.

Correctness:

- targeted scheduler/model/chunked parity suite: 15 passed;
- HOGENOM loss/gradient are unchanged within the existing fp32 run noise:
  loss 667283.5625 bits, gradient infinity norm about 645.922.

Event timing:

| setting | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| non-leaf leaf-term specialization | 1.1470 s | 0.3135 s | 0.8336 s | 5.904 GiB | accepted |
| old leaf term on every wave (`GPUREC_SPECIALIZE_NONLEAF_LEAF_TERM=0`) | 1.1513 s | 0.3185 s | 0.8327 s | 5.904 GiB | slower |

Nsight Systems confirmation:

| setting | profiled pass | CUDA launches | GPU kernel time | wave-step bucket | 2D precompute |
| --- | ---: | ---: | ---: | ---: | ---: |
| final-Pibar fusion baseline | 1.265 s | 47,220 | 1.0528 s | 0.2194 s | 0.0522 s |
| non-leaf leaf-term specialization | 1.258 s | 47,220 | 1.0449 s | 0.2151 s | 0.0491 s |

The launch count is unchanged; the win comes from compiling non-leaf wave
kernels without the impossible leaf-hit term.

## Current Wave-Step NCU Plan

After final-Pibar fusion and non-leaf leaf-term specialization,
`_wave_step_uniform_kernel` is the largest current GPU bucket:
1,548 launches and 0.2151 s in
`profiling/hogenom_ccp/nsys_stream_depthff315_nonleaf_leafterm.sqlite`.  The
previous wave-step NCU diagnosis predates both changes, so profile the current
kernel before trying another launch-shape or code change.

Plan:

- use the accepted `depth_first_fit, clade_budget=315000, max_wave_size=8192`
  layout and all current defaults;
- select a heavy current wave-step launch from the latest `nsys` report; launch
  index 47 is one of the full fused-final-Pibar launches at about 448 us;
- run `ncu --set full` on that launch;
- inspect memory throughput, occupancy, registers, spills, branch efficiency,
  and scheduler statistics;
- only run another launch-shape sweep or kernel edit if NCU points to a
  concrete bottleneck that has not already been covered by the earlier
  `GPUREC_WAVE_STEP_NUM_WARPS` / `BLOCK_S` sweeps.

NCU result for `profiling/hogenom_ccp/ncu_wave_step_current_skip47.ncu-rep`:

- launch shape: grid 8192, block 256, duration 508.99 us;
- registers/thread: 40; no local memory spills;
- theoretical occupancy: 100%; achieved occupancy: 95.10%;
- compute throughput: 90.35%; memory throughput: 90.35%;
- DRAM throughput: 28.67%; L1/TEX throughput: 90.88%; L2 throughput: 19.09%;
- L1/TEX hit rate: 94.14%; L2 hit rate: 75.89%;
- issue slots busy: 62.60%; branch efficiency: 99.38%.

Diagnosis: the current full fused-final-Pibar wave-step launch is healthy.  It
is not register- or occupancy-limited and has no spills.  The bottleneck is the
kernel doing substantial real work with high compute/L1 pipe utilization, so
another launch-shape sweep is unlikely to help.  Further wave-step improvement
would require reducing math/loads algorithmically, not just changing
`num_warps` or `BLOCK_S`.

## Deadline Post-Leaf Scheduling Plan

The scheduler already performs the structurally correct leaf-first global pass.
The remaining risk is that the post-leaf heuristics are still not explicitly
trying a fixed wave horizon: they compare forward ready-queue, reverse
compaction, and a Coffman-Graham-style layering, but none of those attempts says
"can this DAG fit in exactly the lower-bound number of non-leaf waves?"  The
simple lower bound is only necessary, not sufficient, so a failed lower-bound
attempt is still useful evidence rather than a guarantee that the existing
heuristics are wasting waves.

Next implementation step:

- keep the leaf phase unchanged;
- add a deadline/latest-fit post-leaf candidate that tries target non-leaf wave
  counts from the lower bound upward;
- schedule backward from the target horizon, prioritizing ready clades with the
  latest bottom-up earliest level so no clade falls below its legal wave;
- keep the existing forward ready queue, reverse compaction, and
  Coffman-Graham-style candidate as fallbacks;
- preserve the optional DTS partial-row guard and root wave split behavior;
- accept only after a fixed-horizon scheduler regression plus the targeted
  scheduler/model/chunked parity suite.

Implementation result:

- added a deadline/latest-fit candidate that attempts non-leaf wave horizons
  from the simple lower bound up to the current best heuristic count;
- the candidate schedules backward from the target horizon and rejects a target
  as soon as a ready clade would fall below its bottom-up earliest wave;
- kept the existing forward, reverse, and Coffman-Graham candidates as
  fallbacks;
- added a direct regression for the fixed-horizon candidate.

Correctness and HOGENOM check:

- targeted scheduler/model/chunked parity suite: 16 passed;
- accepted HOGENOM depth-first 315k / wave-cap 8192 layout remains
  `[102, 65, 48, 30, 13]` waves by batch, 258 total;
- fresh stream timing measured median forward+backward 1.1434 s, median
  forward 0.3145 s, median backward 0.8288 s, peak allocated 5.904 GiB;
- loss and gradient match the existing run: loss 667283.5625 bits, gradient
  infinity norm about 645.922.

## CUDA Split-Wave Self-Loop Prototype Plan

The deadline scheduler check confirms that the accepted HOGENOM batches already
hit the leaf-first wave-count lower bound.  The next scheduling-adjacent
overhead is therefore inside each split-bearing wave: split waves still use the
retained Triton 2D self-loop path, which launches precompute, one `J^T` kernel
per Neumann term, a parameter-store kernel, and follow-up reductions.  The CUDA
no-split path already avoids that scratch-heavy sequence for no-split waves by
doing the row-local self-loop solve and shared/specieswise gradient
accumulation in one launch per wave.

Next prototype:

- extend the opt-in CUDA row kernel to accept an optional split-side `dts_r`
  matrix;
- compute the left-side mixture weight
  `w_L = exp2(dts_l - logaddexp2(dts_l, dts_r))` inside the CUDA weights helper;
- multiply the self-loop diagonal, Pibar coefficient, speciation weights, and
  DTL parameter adjoints by `w_L`, matching the retained 2D Triton formulas;
- route only auto-wrapped shared/specieswise fp32 CUDA waves with compact
  species levels available;
- keep the existing Triton 2D path as the fallback and make split CUDA
  separately controlled by `GPUREC_CUDA_SELF_LOOP_SPLIT`;
- verify with the targeted parity suite forced on/off, then benchmark HOGENOM
  and run `nsys` only if the event timing improves.

Result: accepted and promoted to default `auto`.  Set
`GPUREC_CUDA_SELF_LOOP_SPLIT=0` to force the retained Triton 2D split-wave
self-loop path.

Correctness:

- targeted scheduler/model/chunked parity suite with split CUDA forced on:
  16 passed;
- same targeted suite with the promoted default: 16 passed;
- same targeted suite with `GPUREC_CUDA_SELF_LOOP_SPLIT=0`: 16 passed;
- direct HOGENOM fallback-vs-split comparison had identical loss
  667283.5625 bits; the fp32 gradient max absolute delta was 0.0116, mean
  absolute delta 0.00044, on a gradient with infinity norm about 646.

Event timing on the accepted depth-first 315k / wave-cap 8192 layout:

| setting | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| promoted CUDA split self-loop | 0.9330 s | 0.3154 s | 0.6180 s | 5.779 GiB | accepted |
| previous default after deadline scheduling | 1.1434 s | 0.3145 s | 0.8288 s | 5.904 GiB | baseline |

Nsight Systems result for
`profiling/hogenom_ccp/nsys_stream_depthff315_cuda_split.nsys-rep`:

| setting | profiled pass | CUDA launches | GPU kernel time |
| --- | ---: | ---: | ---: |
| previous non-leaf leaf-term specialization | 1.258 s | 47,220 | 1.0449 s |
| CUDA split self-loop | 1.052 s | 42,096 | 0.8330 s |

Top bucket movement:

| kernel bucket | previous | CUDA split |
| --- | ---: | ---: |
| `_wave_backward_uniform_2d_jt_kernel` | 0.2098 s / 1464 launches | removed |
| `_wave_backward_uniform_2d_precompute_kernel` | 0.0491 s / 244 launches | removed |
| `_wave_backward_uniform_param_store_kernel` | 0.0441 s / 244 launches | removed |
| `gpurec_wave_backward_nosplit_uniform_fp32` | 0.0187 s / 14 launches | 0.1485 s / 258 launches |
| PyTorch sum reductions | 0.0447 s / 2887 launches | 0.0032 s / 1423 launches |

Nsight Compute on a full 8192-row CUDA split launch
(`profiling/hogenom_ccp/ncu_cuda_split_wave164.ncu-rep`):

- duration 1.93 ms; grid 8192, block 256;
- registers/thread 40, no local spilling;
- dynamic shared memory 37.1 KB/block, limiting residency to two blocks/SM;
- achieved occupancy about 33%, matching the shared-memory limit;
- compute and memory throughput both about 27%, DRAM throughput about 8%;
- branch efficiency 95.4%;
- main stalls are barrier and long scoreboard, consistent with the in-block
  species-tree reductions.

Diagnosis: the CUDA split path is not a perfect kernel, but it removes enough
scratch traffic, follow-up reductions, and launches from the retained 2D path
to reduce whole-dataset forward+backward by about 18% and slightly lower peak
allocation.  The next kernel work should target the CUDA row kernel's shared
memory footprint/barriers, not return to the 2D Triton strategy.

## CUDA Self-Loop Block-Size Plan

The promoted CUDA split/no-split self-loop kernel uses 37.1 KB dynamic shared
memory per block, so full HOGENOM split waves are limited to two resident
blocks per SM.  With the current hardcoded 256-thread launch, that is only 16
resident warps per SM and about 33% achieved occupancy.  A 512-thread block
would still be limited to two blocks by shared memory, but would expose 32
resident warps per SM.  That may hide the barrier and long-scoreboard stalls
seen in NCU without changing the algorithm.

Next experiment:

- add a diagnostic `GPUREC_CUDA_SELF_LOOP_BLOCK` launch override, keeping 256 as
  the initial default;
- test 512 against 256 on the accepted HOGENOM layout with split CUDA default
  enabled;
- optionally test 1024 if 512 is promising, since it may also reach 32 resident
  warps but with only one block per SM under the thread limit;
- accept a new default only if targeted parity still passes and whole-dataset
  event timing improves; profile with `nsys` if the timing win is material.

Result: accepted.  `GPUREC_CUDA_SELF_LOOP_BLOCK` now defaults to 512; set it to
256 to restore the previous launch shape for diagnostics.

Correctness:

- targeted scheduler/model/chunked parity suite with block 512 default:
  16 passed.

Event timing:

| block size | median fwd+bwd | median forward | median backward | peak alloc | decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 0.9330 s | 0.3154 s | 0.6180 s | 5.779 GiB | previous default |
| 512 | 0.9118 s | 0.3145 s | 0.5976 s | 5.779 GiB | accepted |
| 768 | 0.9245 s | 0.3152 s | 0.6098 s | 5.779 GiB | rejected |
| 1024 | 0.9867 s | 0.3141 s | 0.6721 s | 5.779 GiB | rejected |

Nsight Systems result for
`profiling/hogenom_ccp/nsys_stream_depthff315_cuda_split_block512.nsys-rep`:

| setting | profiled pass | CUDA launches | GPU kernel time | CUDA self-loop bucket |
| --- | ---: | ---: | ---: | ---: |
| block 256 | 1.052 s | 42,096 | 0.8330 s | 0.1485 s |
| block 512 | 1.036 s | 42,096 | 0.8285 s | 0.1393 s |

Nsight Compute on a full 8192-row launch
(`profiling/hogenom_ccp/ncu_cuda_split_block512_wave164.ncu-rep`) confirms the
intended occupancy effect:

- block 512, grid 8192, duration 1.87 ms;
- registers/thread 40, no local spilling;
- dynamic shared memory 37.1 KB/block, still limiting residency to two
  blocks/SM;
- achieved occupancy rises to about 66%;
- compute/memory throughput rises to about 34%, DRAM throughput stays low at
  about 8.7%;
- long-scoreboard stalls fall versus block 256, while barrier stalls remain
  prominent from the in-block reductions.

## CUDA Self-Loop Shared-Array Reduction Plan

The block-512 CUDA self-loop kernel is still shared-memory constrained: it
allocates seven row-sized shared arrays,
`term/work/vacc/diag/pcoef/sl1w/sl2w`.  The `sl1w` and `sl2w` arrays are only
used to add the speciation contribution from a species' parent:
`term[parent] * side_weight(parent -> child)`.  Because the species topology is
a tree, every child species has at most one parent.  We can precompute one
child-indexed `edgew[S]` array instead of two parent-indexed arrays:

```text
edgew[sp_child1[parent]] = q3(parent)
edgew[sp_child2[parent]] = q4(parent)
```

Then the Neumann update uses `term[parent[s]] * edgew[s]`.

Next experiment:

- add an environment-controlled CUDA mode that replaces `sl1w/sl2w` with one
  `edgew` array;
- reduce dynamic shared memory from `7*S*sizeof(float)` to
  `6*S*sizeof(float)` in that mode;
- keep the old layout as a diagnostic fallback;
- verify targeted parity with the new mode forced on;
- benchmark HOGENOM, then run `nsys` and optionally `ncu` if the event timing
  improves.

Result: accepted and promoted to default.  Set
`GPUREC_CUDA_SELF_LOOP_CHILD_EDGE_WEIGHT=0` to restore the old two-array
`sl1w/sl2w` layout for diagnostics.

Correctness:

- targeted scheduler/model/chunked parity suite with child-edge mode forced on:
  16 passed;
- same targeted suite with the promoted default: 16 passed;
- same targeted suite with `GPUREC_CUDA_SELF_LOOP_CHILD_EDGE_WEIGHT=0`:
  16 passed.

Event timing:

| setting | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| child-edge shared array | 0.8922 s | 0.3145 s | 0.5772 s | 5.779 GiB | accepted |
| two parent-side arrays | 0.9118 s | 0.3145 s | 0.5976 s | 5.779 GiB | previous default |

Nsight Systems result for
`profiling/hogenom_ccp/nsys_stream_depthff315_cuda_child_edge.nsys-rep`:

| setting | profiled pass | CUDA launches | GPU kernel time | CUDA self-loop bucket |
| --- | ---: | ---: | ---: | ---: |
| two parent-side arrays | 1.036 s | 42,096 | 0.8285 s | 0.1393 s |
| child-edge shared array | 0.996 s | 42,096 | 0.7856 s | 0.1008 s |

Nsight Compute on a full 8192-row launch
(`profiling/hogenom_ccp/ncu_cuda_child_edge_wave164.ncu-rep`):

- duration 1.27 ms versus 1.87 ms for the two-array block-512 launch;
- dynamic shared memory drops from 37.1 KB/block to 31.8 KB/block;
- occupancy limit rises from two to three blocks/SM, and achieved occupancy is
  about 99%;
- registers/thread: 39; no local spilling;
- compute/memory throughput rises to about 49%, with DRAM still only about
  12.7%;
- barrier and long-scoreboard stalls both fall substantially.

Diagnosis: this is the intended next step after the block-size sweep.  It
removes one row-sized shared array, unlocks the third resident block per SM,
and reduces the CUDA self-loop bucket by about 38 ms in the profiled pass
without changing launch count or memory footprint at the model level.

## Post-CUDA-Split Scheduling Retest Plan

After the CUDA split self-loop changes, the accepted HOGENOM run is no longer
dominated by the retained 2D self-loop path.  The largest bucket is now the
forward wave-step launch family, with 1,548 launches over 258 waves.  Earlier
batch/wave scheduling sweeps were measured before split waves used the CUDA
self-loop path, so the memory and timing tradeoffs may have shifted.

Next experiment:

- keep the accepted depth-first packing policy and uniform origination setup;
- retest a narrow set of larger clade budgets and wave caps under the new
  CUDA split default;
- start with metadata-only wave counts to avoid blind full runs;
- then benchmark only candidates that reduce total waves or plausibly remain
  inside the 5-6 GiB lean memory target;
- accept a scheduling change only if whole-dataset event timing improves and
  `nsys` confirms lower GPU kernel time without shifting more cost into DTS or
  Pibar buckets.

## Post-CUDA-Split Scheduling Metadata Sweep

The HOGENOM CCP intra-batch scheduler is already doing the intended
leaf-first, then globally packed post-leaf DAG schedule.  On the current
`depth_first_fit` HOGENOM batches, each inspected batch reaches the simple
lower bound:

```text
leaf_waves + max(longest_post_leaf_depth, ceil(nonleaf_clades / wave_cap))
```

For the accepted `clade_budget=315000, max_wave_size=8192` configuration:

| batch | clades | leaves | nonleaves | depth | lower bound | actual waves |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 314,985 | 27,830 | 287,155 | 98 | 102 | 102 |
| 1 | 314,902 | 24,939 | 289,963 | 61 | 65 | 65 |
| 2 | 273,898 | 23,300 | 250,598 | 45 | 48 | 48 |
| 3 | 104,120 | 8,739 | 95,381 | 28 | 30 | 30 |
| 4 | 15,208 | 1,783 | 13,425 | 12 | 13 | 13 |

This means a better ready-queue heuristic cannot reduce the wave count for
this batch layout unless it violates dependencies or mixes leaf/non-leaf
semantics.  The underfilled tail waves are caused by depth-dominated batches:
for example, the deepest batch has depth 98, so it needs 98 post-leaf waves
even though its non-leaf work alone would need only 36 waves at cap 8192.

Metadata-only retest of larger batch budgets and wave caps:

| clade budget | wave cap | batches | total waves | max clades | max splits |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 315,000 | 8,192 | 5 | 258 | 314,985 | 805,558 |
| 350,000 | 8,192 | 5 | 247 | 349,825 | 899,027 |
| 400,000 | 8,192 | 5 | 229 | 399,997 | 1,056,554 |
| 400,000 | 16,384 | 5 | 224 | 399,994 | 1,056,233 |
| 450,000 | 8,192 | 5 | 220 | 449,992 | 1,135,851 |
| 500,000 | 8,192 | 4 | 210 | 499,957 | 1,265,563 |

Next benchmark: time the 350k, 400k, and 500k candidates with the current
CUDA self-loop defaults.  Accept a larger budget only if event timing improves
without exceeding the lean memory target.

Event timing result:

| clade budget | wave cap | median fwd+bwd | median forward | median backward | peak alloc | peak reserved | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 315,000 | 8,192 | 0.8866 s | 0.3159 s | 0.5705 s | 5.779 GiB | 6.961 GiB | baseline |
| 320,000 | 8,192 | 0.8861 s | 0.3162 s | 0.5699 s | 7.922 GiB | 8.273 GiB | rejected: memory |
| 350,000 | 8,192 | 0.8866 s | 0.3164 s | 0.5702 s | 8.370 GiB | 8.941 GiB | rejected: memory |
| 315,000 | 16,384 | 0.8840 s | 0.3157 s | 0.5684 s | 5.728 GiB | 6.268 GiB | neutral |
| 315,000 | 24,576 | 0.8825 s | 0.3146 s | 0.5681 s | 5.522 GiB | 11.051 GiB | rejected: reserved memory |
| 315,000 | 32,768 | 0.8803 s | 0.3153 s | 0.5645 s | 5.523 GiB | 11.047 GiB | rejected: nsys neutral/reserved memory |

Nsight Systems for
`profiling/hogenom_ccp/nsys_stream_depthff315_wave32768.nsys-rep`:

| setting | profiled pass | CUDA launches | GPU kernel time | wave-step bucket | CUDA self-loop bucket |
| --- | ---: | ---: | ---: | ---: | ---: |
| 315k / 8192 | 0.996 s | 42,096 | 0.7856 s | 0.2151 s / 1,548 | 0.1008 s / 258 |
| 315k / 32768 | 0.998 s | 42,024 | 0.7862 s | 0.2149 s / 1,494 | 0.1031 s / 249 |

Diagnosis: increasing the wave cap removes only nine resident waves and does
not reduce total GPU kernel time in Nsight.  Larger clade budgets reduce waves
more, but they immediately leave the lean memory envelope and do not improve
the measured median at 320k or 350k.  The current HOGENOM scheduling bottleneck
is therefore not an intra-batch wave heuristic: the scheduler already reaches
the leaf-first DAG lower bound for the accepted layout.  Meaningful reductions
in wave count would require either substantially larger resident batches, a
different memory representation that makes those larger batches cheap, or a
kernel path that can process multiple dependency levels inside one launch.

## Prepared Origination Fast-Path Plan

Nsight Systems NVTX ranges in the accepted child-edge profile show a surprising
forward-side overhead:

| NVTX range | count | total wall time |
| --- | ---: | ---: |
| `forward root likelihood` | 5 | 0.2151 s |
| `forward Pi waves` | 5 | 0.0918 s |
| `forward E fixed point` | 5 | 0.0170 s |

The model constructor already validates and normalizes `origination_probs`, and
batched resident mode slices those prepared rows for each batch.  However the
hot likelihood and implicit-gradient paths call `prepare_origination_probs`
again.  That helper performs CUDA `.item()` validations (`isfinite`, negative
check, row-sum check), so each batch can introduce host synchronization and
extra tiny PyTorch kernels even when the probability tensor is already resident
and normalized.

Next experiment:

- add internal `origination_probs_prepared` flags to the likelihood, Pi
  backward root-adjoint initialization, and E-adjoint denominator path;
- pass `origination_probs_prepared=True` only from model/static paths where the
  constructor has already called `prepare_origination_probs`;
- keep public likelihood helpers validating by default;
- verify origination-probability unit tests and the targeted model parity
  suite;
- benchmark the accepted HOGENOM stream pass and run `nsys` to confirm the
  `forward root likelihood` range and tiny validation kernels shrink.

Result: accepted.  Internal model paths now pass
`origination_probs_prepared=True`, keeping public helper validation as the
default.  The Pi backward root adjoint is also initialized with one vectorized
root-row operation instead of a Python loop over family roots.

Correctness:

- origination-probability unit tests plus targeted scheduler/model/chunked
  parity suite: 19 passed.

Event timing:

| setting | median fwd+bwd | median forward | median backward | peak alloc | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| before prepared-orig fast path | 0.8866 s | 0.3159 s | 0.5705 s | 5.779 GiB | 6.961 GiB |
| prepared-orig fast path | 0.7796 s | 0.3092 s | 0.4706 s | 5.780 GiB | 6.965 GiB |

Nsight Systems for
`profiling/hogenom_ccp/nsys_stream_depthff315_prepared_orig.nsys-rep`:

| setting | profiled pass | CUDA launches | GPU kernel time | `forward root likelihood` NVTX |
| --- | ---: | ---: | ---: | ---: |
| before prepared-orig fast path | 0.9965 s | 42,096 | 0.7856 s | 0.2151 s / 5 |
| prepared-orig fast path | 0.8190 s | 15,606 | 0.7457 s | 0.0016 s / 5 |

Top GPU buckets after the change:

| kernel bucket | launches | GPU time |
| --- | ---: | ---: |
| `_wave_step_uniform_kernel` | 1,548 | 0.2128 s |
| `_dts_cross_backward_accum_kernel` | 244 | 0.1540 s |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | 244 | 0.1063 s |
| `gpurec_wave_backward_nosplit_uniform_fp32` | 258 | 0.0994 s |
| `_dts_parent_reduced_ge2_stage1_kernel` | 478 | 0.0969 s |

Diagnosis: the weighted-origination support had accidentally put validation
and root-row adjoint setup in the hot path.  The repeated CUDA `.item()`
checks synchronized the host after Pi forward, and the per-root backward loop
launched thousands of tiny PyTorch kernels.  Treating model-owned origination
probabilities as already prepared removes that overhead without changing the
public validation path.

## E-Adjoint Host Synchronization Audit Plan

After the prepared-origination fast path, the accepted HOGENOM `nsys` profile is
much cleaner but still reports 599 `cudaStreamSynchronize` runtime calls during
one measured streamed pass.  The likely source is the E-adjoint CG solve in
`gpurec/optimization/implicit_grad.py`, which currently converts device
residual dot products and norms to Python floats each iteration.

Next experiment:

- instrument the current code externally, without changing solver math, to
  count CG iterations per resident batch and measure how much of the backward
  pass sits in the E-adjoint solve;
- use Nsight Systems runtime tables to compare synchronization count/time
  before and after any candidate change;
- only prototype a solver-side change if the measurement shows that scalar CG
  synchronization is material relative to the 0.78 s whole-dataset pass;
- preserve exact gradient semantics unless a deliberate solver tolerance/change
  is separately tested for likelihood/gradient parity.

Instrumentation result: do not change the solver yet.

One measured HOGENOM streamed pass after warmup:

| batch | CG iterations | CG wall time |
| ---: | ---: | ---: |
| 0 | 7 | 0.00495 s |
| 1 | 7 | 0.00485 s |
| 2 | 7 | 0.00485 s |
| 3 | 8 | 0.00552 s |
| 4 | 8 | 0.00544 s |

Total measured CG wall time was about 0.0256 s inside a 0.7935 s
forward+backward pass.  That is not zero, but the larger remaining costs are
still the GPU buckets in wave-step, DTS accumulation, Pibar VJP, CUDA self-loop,
and parent-reduced DTS.  A solver-side rewrite would add correctness risk for
too little expected gain at this point.

## Current DTS Backward NCU Plan

After the prepared-origination fast path, `_dts_cross_backward_accum_kernel` is
the second largest kernel bucket at about 0.154 s over 244 launches.  The
largest observed launch in
`profiling/hogenom_ccp/nsys_stream_depthff315_prepared_orig.sqlite` is matching
DTS launch index 172, at about 3.03 ms.

Next experiment:

- capture one Nsight Compute report for that launch under the current accepted
  defaults;
- inspect occupancy, registers, spills, memory throughput, branch/barrier
  stalls, and source counters before changing kernel code;
- only prototype a DTS accumulation change if NCU points to a concrete
  bottleneck that is not already covered by the earlier launch-shape sweeps.

NCU result for
`profiling/hogenom_ccp/ncu_dts_backward_prepared_launch172.ncu-rep`:

- launch shape: grid 43,245 blocks, block size 256;
- duration: 3.21 ms for the sampled large launch;
- registers/thread: 40; no local memory spilling;
- theoretical occupancy: 100%; achieved occupancy: 98.75%;
- DRAM throughput: 57.49%; L2 throughput: 52.04%;
- compute throughput: 27.72%; issue slots busy: 16.89%;
- active warps per scheduler: 11.86, but eligible warps per scheduler only
  0.40;
- branch efficiency: 67.34%;
- NCU reports about 48.4M excessive global sectors, about 26% of total sectors.

Diagnosis: this kernel is no longer register- or occupancy-limited under the
current defaults.  The remaining DTS accumulation cost is dominated by latency,
branch divergence, and imperfect global-memory access patterns.  The earlier
launch-shape sweeps are consistent with this NCU result: changing block size or
warps is unlikely to move the total pass much.  A real improvement would need a
structural rewrite of how split-side rows are grouped or how child/parent rows
are accessed, and should not be attempted without a more specific design.

## DTS-R Output Fill Skip Plan

The current prepared-origination profile still contains 1,809 PyTorch
`FillFunctor<float>` launches, about 0.0228 s of GPU time.  One repeated source
is `dts_fused_parent_reduced`, which allocates every DTS-R output as
`torch.full((W, S), -inf)` before launching the eq1/ge2 parent-reduced kernels.

For wave-ordered CCP batches, a wave with `meta["has_splits"]` should contain
only non-leaf/root clades.  The layout metadata partitions all parent rows into
exactly-one-split parents (`n_eq1`) and multi-split parents (`ge2_parent_ids`).
If `n_eq1 + len(ge2_parent_ids) == W`, every output row is overwritten by the
Triton kernels.  With pruning active, inactive eq1/ge2 rows are explicitly
written as `-inf`, so the initial full-tensor fill is still redundant.

Next experiment:

- allocate DTS-R with `torch.empty` when parent coverage is complete;
- keep the existing `-inf` fill fallback if a future layout has incomplete
  coverage or a caller passes an output tensor that cannot be assumed complete;
- verify targeted model/chunked tests;
- benchmark the accepted HOGENOM stream pass;
- run `nsys` only if event timing improves enough to check that FillFunctor
  launches and total GPU kernel time decrease.

Result: rejected.  Targeted correctness passed (19 tests) and event timing
looked slightly better, but Nsight Systems did not confirm a real improvement.

| setting | median fwd+bwd | CUDA launches | GPU kernel time | FillFunctor launches/time | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| prepared-orig baseline | 0.7796 s | 15,606 | 0.7457 s | 1,809 / 0.0228 s | baseline |
| DTS-R fill skip | 0.7766 s | 15,118 | 0.7524 s | 1,321 / 0.0178 s | rejected |

The fill skip removed 488 fill launches and about 5 ms from the FillFunctor
bucket, but profiler noise or allocator/cache effects moved more time into the
main kernels (`_dts_cross_backward_accum_kernel` and the CUDA self-loop both
rose in the profiled pass).  Since the acceptance criterion required lower
total GPU kernel time, the code change was reverted.

## Current Pibar VJP NCU Plan

After the prepared-origination fast path, the Pibar-from-UD VJP kernel remains
one of the largest single GPU buckets:
`_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` takes about 0.106 s
over 244 launches in
`profiling/hogenom_ccp/nsys_stream_depthff315_prepared_orig.sqlite`.  The
largest observed launch is match index 172, at about 2.05 ms.

Next experiment:

- capture one Nsight Compute report for match index 172 under the current
  accepted HOGENOM settings;
- inspect achieved occupancy, register pressure, spills, memory throughput,
  branch divergence, and warp issue stalls;
- compare the result with the DTS NCU result before changing code, because both
  kernels walk cross-clade CCP structure and may be limited by similar memory
  and divergence patterns;
- only prototype a Pibar VJP change if NCU points to a concrete structural
  bottleneck that can be improved without changing likelihood or gradient
  semantics.

NCU result for
`profiling/hogenom_ccp/ncu_pibar_vjp_prepared_launch172.ncu-rep`:

- launch shape: grid 86,490 blocks, block size 128;
- duration: 2.06 ms for the sampled large launch;
- registers/thread: 36; no local memory spilling;
- theoretical occupancy: 100%; achieved occupancy: 97.49%;
- DRAM throughput: 88.45%; L2 throughput: 68.71%;
- compute throughput: 47.40%; issue slots busy: 34.71%;
- active warps per scheduler: 11.63, but eligible warps per scheduler only
  0.82;
- branch efficiency: 99.74%;
- NCU reports about 153.0M excessive global sectors, about 60% of total
  sectors;
- average useful bytes per global-load sector is 10.1 / 32; average useful
  bytes per global-store sector is 4.4 / 32.

Diagnosis: this launch is not occupancy-, register-, spill-, or branch-limited.
It is primarily a DRAM/coalescing problem, likely from each split-side program
walking species-tree rows and then atomically adding into child rows of
`accumulated_rhs`.  A real fix probably requires a structural change to the
Pibar VJP layout or accumulation order, not a simple occupancy tweak.

Before attempting that rewrite, run a narrow launch-option sweep for the
existing `GPUREC_PIBAR_UD_BLOCK_S` and `GPUREC_PIBAR_UD_NUM_WARPS` knobs.  This
is low risk and may reveal a suboptimal default, but should only be promoted if
whole-dataset timing improves and a follow-up `nsys` pass confirms lower total
GPU kernel time.

Event timing sweep:

| setting | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| default | 0.7856 s | 0.3106 s | 0.4750 s | 5.780 GiB | baseline |
| `BLOCK_S=64, WARPS=4` | 0.7951 s | 0.3099 s | 0.4852 s | 5.780 GiB | rejected |
| `BLOCK_S=128, WARPS=4` | 0.7910 s | 0.3095 s | 0.4815 s | 5.780 GiB | rejected |
| `BLOCK_S=256, WARPS=4` | 0.7849 s | 0.3108 s | 0.4741 s | 5.780 GiB | tied |
| `BLOCK_S=512, WARPS=4` | 0.7843 s | 0.3093 s | 0.4750 s | 5.780 GiB | tied |
| `BLOCK_S=128, WARPS=8` | 0.7907 s | 0.3109 s | 0.4798 s | 5.780 GiB | rejected |
| `BLOCK_S=256, WARPS=8` | 0.7793 s | 0.3088 s | 0.4707 s | 5.780 GiB | validate with `nsys` |
| `BLOCK_S=128, WARPS=2` | 0.8005 s | 0.3098 s | 0.4904 s | 5.780 GiB | rejected |

The only candidate with a visible event-time improvement is
`BLOCK_S=256, WARPS=8`, and the likelihood/gradient reported by the profiler
remain unchanged.  Next validation: run Nsight Systems for this candidate and
compare CUDA launch count, total GPU kernel time, and the Pibar VJP bucket
against the prepared-origination baseline before promoting a default.

Nsight Systems validation for
`profiling/hogenom_ccp/nsys_stream_depthff315_pibar_w8.nsys-rep`:

| setting | CUDA launches | GPU kernel time | Pibar VJP bucket | DTS backward bucket | CUDA self-loop bucket | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| default prepared-origination baseline | 15,606 | 0.7457 s | 0.1063 s / 244 | 0.1540 s / 244 | 0.0994 s / 258 | baseline |
| `BLOCK_S=256, WARPS=8` | 15,606 | 0.7459 s | 0.1038 s / 244 | 0.1519 s / 244 | 0.1047 s / 258 | rejected |

The candidate does reduce the targeted Pibar VJP bucket by about 2.5 ms, and
DTS accumulation is also slightly lower in this profiled pass.  The CUDA
self-loop bucket rises by about 5.3 ms, leaving total GPU kernel time
unchanged/slightly worse.  Because the acceptance criterion requires a real
whole-pass `nsys` improvement, do not promote the Pibar VJP launch-option
change.  The NCU result remains useful: the next Pibar improvement would need
to address uncoalesced loads/stores structurally rather than retuning block
shape.

## Pibar VJP Compact-Level Ordering Plan

The Pibar VJP compact tree walk processes species-tree nodes level by level.
Nodes within one level are independent, so their order is semantically
irrelevant but affects memory coalescing for:

- `pibar_ud[row, parent]` loads/stores;
- `pibar_ud[row, child1]` and `pibar_ud[row, child2]` loads;
- the final `accumulated_rhs[child, species]` atomic-add pattern indirectly
  through the per-row tree accumulation.

The current ordering is ascending parent species id.  That should make parent
loads/stores relatively regular, but child accesses can be scattered.  Since
NCU reports about 60% excessive global sectors in the Pibar VJP kernel, test
whether alternative within-level orderings improve the whole pass:

- add a diagnostic `GPUREC_SPECIES_COMPACT_LEVEL_ORDER` option with current
  `parent` ordering as the default;
- prototype `child_min`, `child1`, and `child2` orderings;
- include the ordering in the species-topology cache key so same-process
  diagnostics are not silently reused;
- verify targeted correctness, then benchmark HOGENOM event timing;
- run `nsys` only if an ordering shows a real event-time improvement; promote
  nothing unless total GPU kernel time improves.

Result: rejected and removed.  The diagnostic ordering knob passed the focused
species-topology/kernel tests, but whole-dataset event timing did not improve:

| compact level order | median fwd+bwd | median forward | median backward | peak alloc | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| parent/current | 0.7802 s | 0.3075 s | 0.4729 s | 5.780 GiB | baseline |
| child_min | 0.7868 s | 0.3098 s | 0.4769 s | 5.780 GiB | rejected |
| child1 | 0.7815 s | 0.3082 s | 0.4732 s | 5.780 GiB | tied/slower |
| child2 | 0.7924 s | 0.3100 s | 0.4830 s | 5.780 GiB | rejected |

The existing parent order is already the best measured option.  Because the
prototype added an unused diagnostic knob without a performance win, the code
change was removed and no `nsys` follow-up was run.

## Current CPU And Launch Overhead Audit Plan

After the prepared-origination fast path and rejected Pibar follow-ups, the
accepted uninstrumented stream pass is around 0.78 s and the current `nsys`
profile reports about 0.746 s of GPU kernel time.  Before considering larger
changes such as launch consolidation or CUDA graph capture, audit the existing
`nsys_stream_depthff315_prepared_orig.sqlite` report for:

- measured-pass NVTX wall time versus GPU kernel time;
- CUDA kernel launch count and launch API self time;
- synchronization and memcpy API time;
- whether the CPU/GPU gap is large enough to justify a graph/capture prototype
  with the added complexity of autograd, batch switching, and dynamic tensor
  lifetimes.

Only plan a launch-overhead prototype if the audit shows a meaningful
recoverable CPU gap.  Otherwise keep focusing on the dominant GPU kernel
buckets.

Audit result from
`profiling/hogenom_ccp/nsys_stream_depthff315_prepared_orig.sqlite`:

- measured-pass NVTX wall time: 0.8190 s;
- GPU kernel active sum: 0.7457 s;
- first-to-last kernel window: 0.8187 s;
- inferred inter-kernel/non-kernel gap: 0.0731 s across 15,605 gaps;
- largest single inter-kernel gap: 292.8 us;
- gaps at least 50 us: 76 gaps, totaling 0.0063 s;
- gaps at least 10 us: 2,092 gaps, totaling 0.0434 s;
- CUDA runtime launch API self time: about 0.173 s across
  `cudaLaunchKernel*` / `cuLaunchKernel*` calls, but this is not directly
  additive with wall time because the CPU is launching while GPU kernels are
  executing;
- CUDA memcpy activity: 1,983 copies, 0.0019 s GPU copy time, about 171 MB;
- CUDA synchronization API time is about 0.412 s, mostly CPU waiting for queued
  GPU work rather than standalone overhead.

Diagnosis: launch/CPU overhead exists, but the recoverable envelope in the
current stream pass is at most about 73 ms and is spread over thousands of
small gaps.  A CUDA graph or launch-consolidation prototype would be
non-trivial because the stream pass switches resident batches, allocates
batch-local tensors, and uses autograd.  Do not pursue that before exhausting
lower-risk GPU-kernel work; the dominant cost is still real kernel time.

## Dense Runtime Fill Audit Plan

The current prepared-origination `nsys` profile still has a PyTorch
`FillFunctor<float>` bucket of about 0.0228 s over 1,809 launches.  A previous
DTS-R output fill skip removed many fill launches but did not improve total GPU
kernel time, so do not repeat that change blindly.  First identify whether any
remaining large dense fills are provably redundant.

Initial audit from
`profiling/hogenom_ccp/nsys_stream_depthff315_prepared_orig.sqlite`:

- 12 fill launches exceed 0.5 ms each;
- the largest pairs occur around forward setup and line up with dense `Pi` and
  `Pibar` initialization for large batches;
- other large fills occur around backward setup and likely include dense
  accumulated adjoint buffers;
- the ten largest dense fills alone account for roughly 15-16 ms of GPU time.

Next experiment:

- inspect forward and backward consumers before changing code;
- only replace a `torch.full`/`torch.zeros` with `empty` if every read is
  guarded or every element is overwritten before use;
- verify targeted model/chunked parity and HOGENOM loss/gradient before any
  timing conclusion;
- run whole-dataset event timing first, and only run `nsys` if the timing
  suggests the removed fill launches also reduce total GPU kernel time.

Forward `Pibar` initialization is the narrow candidate.  `Pi` must keep its
`-inf` initialization because each wave's first self-loop iteration reads the
current clade row as the fixed-point initial state.  `Pibar`, however, is first
written as the ping-pong output for a wave before that wave reads it, and
cross-clade DTS only reads child rows from earlier waves.  Final root-only
waves may intentionally skip final Pibar storage, but those rows were already
being treated as not needed by backward.  Test this as an opt-in
`GPUREC_FORWARD_EMPTY_PIBAR_INIT=1` mode before considering a default change.

Correctness with `GPUREC_FORWARD_EMPTY_PIBAR_INIT=1`:

- targeted species/scheduler/model/chunked parity suite: 21 passed.

Event timing:

| setting | median fwd+bwd | median forward | median backward | peak alloc | loss / grad |
| --- | ---: | ---: | ---: | ---: | --- |
| default Pibar fill | 0.7790 s | 0.3090 s | 0.4700 s | 5.780 GiB | unchanged |
| empty Pibar init | 0.7706 s | 0.3024 s | 0.4683 s | 5.780 GiB | unchanged |

Next validation: run `nsys` for the empty-Pibar mode and accept it only if total
GPU kernel time and the FillFunctor bucket shrink rather than merely shifting
time elsewhere.

Nsight Systems validation for
`profiling/hogenom_ccp/nsys_stream_depthff315_empty_pibar.nsys-rep`:

| setting | CUDA launches | GPU kernel time | FillFunctor bucket | wave-step bucket | DTS backward bucket | CUDA self-loop bucket | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default Pibar fill | 15,606 | 0.7457 s | 0.0228 s / 1,809 | 0.2128 s | 0.1540 s | 0.0994 s | baseline |
| empty Pibar init | 15,601 | 0.7523 s | 0.0172 s / 1,804 | 0.2146 s | 0.1572 s | 0.1063 s | rejected |

The opt-in path removes exactly the expected five large Pibar fill launches and
shrinks the fill bucket by about 5.6 ms, but the main wave-step, DTS backward,
and CUDA self-loop buckets all move up in the profiled pass.  Because total GPU
kernel time regresses by about 6.7 ms, do not promote the change.  The code path
was removed instead of kept as a diagnostic knob.

## Family-Count Cap Scheduling Audit Plan

The accepted `depth_first_fit` HOGENOM layout uses both
`family_chunk_size=300` and `clade_budget=315000`.  The current batch metadata
shows that the first two batches are clade-budget limited, but batches 2 and 3
hit the 300-family cap with substantial clade headroom:

| batch | families | clades | waves | max depth | leaf waves | work waves |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 91 | 314,985 | 102 | 98 | 4 | 36 |
| 1 | 146 | 314,902 | 65 | 61 | 4 | 36 |
| 2 | 300 | 273,898 | 48 | 45 | 3 | 31 |
| 3 | 300 | 104,120 | 30 | 28 | 2 | 12 |
| 4 | 218 | 15,208 | 13 | 12 | 1 | 2 |

This suggests the family-count cap may be a stale constraint from older
resident batching.  Next experiment:

- build metadata for `family_chunk_size=0, clade_budget=315000,
  batch_packing=depth_first_fit`;
- compare total batches, lower-bound waves, max clades, max split count, and
  DTS partial-row memory proxy against the accepted 300-family cap layout;
- only run full timing if the metadata reduces waves or batch count while
  staying inside the same clade/memory envelope;
- accept no default change without HOGENOM loss/gradient parity and `nsys`
  confirmation.

Metadata result:

| family cap | batches | total waves | max clades | max splits | max DTS partial rows | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 300 | 5 | 258 | 314,985 | 805,558 | 85,464 | baseline |
| 0 / no cap | 4 | 240 | 314,585 | 873,723 | 85,464 | benchmark |

The no-family-cap layout keeps the worst clade count and DTS partial-row proxy
inside the current envelope while removing one resident batch and 18 waves.  It
does increase the largest split count, so full timing and memory must decide
whether the grouping is actually better.

Event timing:

| family cap | batches | median fwd+bwd | median forward | median backward | peak alloc | peak reserved | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 300 | 5 | 0.7788 s | 0.3070 s | 0.4720 s | 5.780 GiB | 6.965 GiB | baseline |
| 0 / no cap | 4 | 0.7734 s | 0.3067 s | 0.4667 s | 5.980 GiB | 10.992 GiB | validate with `nsys` |

The no-cap layout is modestly faster and still under 6 GiB allocated, but it
substantially increases reserved memory.  Loss and gradient differ only at the
usual fp32 accumulation-order scale.  Run Nsight Systems before deciding
whether the lower wave/batch count translates into lower GPU kernel time.

Nsight Systems validation for
`profiling/hogenom_ccp/nsys_stream_depthff315_no_family_cap.nsys-rep`:

| family cap | CUDA launches | GPU kernel time | wave-step bucket | DTS backward bucket | Pibar VJP bucket | CUDA self-loop bucket | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 300 | 15,606 | 0.7457 s | 0.2128 s / 1,548 | 0.1540 s / 244 | 0.1063 s / 244 | 0.0994 s / 258 | baseline |
| 0 / no cap | 13,464 | 0.7468 s | 0.2134 s / 1,440 | 0.1553 s / 227 | 0.1066 s / 227 | 0.1003 s / 240 | rejected default |

The no-cap layout removes 2,142 CUDA launches, 18 self-loop waves, and one
resident batch, but larger batches make the main kernels heavier per launch.
Total GPU kernel time is slightly worse, and peak reserved memory rises to about
11 GiB.  Keep the 300-family cap as the default for now; `--chunk-size 0`
remains a useful diagnostic for lower-launch scheduling experiments but is not
a profiler-confirmed speed win.

## Current Scheduling Audit

The desired scheduling model is:

- process all leaf clades first, because leaf initialization is a distinct
  operation;
- schedule every remaining non-leaf clade from all resident-batch DAGs into
  globally packed ready waves;
- cap each wave at `max_wave_size` clades;
- minimize the number of non-leaf waves subject to the cap and topological
  dependencies.

For this model, a hard lower bound for each resident batch is:

```text
ceil(leaves / max_wave_size)
+ max(max_bottom_up_depth, ceil(nonleaves / max_wave_size))
```

The current `schedule_global_phased_waves` implementation already applies this
leaf phase plus global non-leaf scheduling, including deadline and
Coffman-Graham-style compaction attempts when greedy ready scheduling leaves a
gap.

Audit command:

```bash
python - <<'PY'
# Build the HOGENOM+CCP model with the profiling configuration and compare each
# resident batch's scheduled wave count with the leaf-first lower bound.
PY
```

Result for the warm HOGENOM+CCP configuration
`--chunk-size 300 --clade-budget 315000 --batch-packing depth_first_fit
--max-wave-size 8192`:

| batch | families | clades | waves | lower bound | gap | leaf waves | non-leaf bound | max depth |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 91 | 314985 | 102 | 102 | 0 | 4 | 98 | 98 |
| 1 | 146 | 314902 | 65 | 65 | 0 | 4 | 61 | 61 |
| 2 | 300 | 273898 | 48 | 48 | 0 | 3 | 45 | 45 |
| 3 | 300 | 104120 | 30 | 30 | 0 | 2 | 28 | 28 |
| 4 | 218 | 15208 | 13 | 13 | 0 | 1 | 12 | 12 |

Total scheduled waves are 258 and the summed lower bound is also 258.  This
means the current within-batch scheduler cannot reduce the wave count further
without changing the batching/memory constraints, increasing the wave cap, or
removing the separate leaf phase.

The earlier no-family-cap diagnostic
`--chunk-size 0 --clade-budget 315000 --batch-packing depth_first_fit` also hit
its per-batch lower bounds:

| batch | families | clades | waves | lower bound | gap | max depth |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90 | 314552 | 102 | 102 | 0 | 98 |
| 1 | 145 | 314585 | 65 | 65 | 0 | 61 |
| 2 | 381 | 313923 | 49 | 49 | 0 | 45 |
| 3 | 439 | 80053 | 24 | 24 | 0 | 23 |

The 240-wave no-family-cap layout reduces launches by changing resident batch
composition, not by finding a better within-batch DAG schedule.  The prior
`nsys` result rejected it as a default because total GPU kernel time did not
improve and reserved memory rose substantially.

Current timing check after the audit, with the accepted 5-batch layout and one
warmup:

```bash
python scripts/profile_hogenom_ccp_pass.py \
  --chunk-size 300 \
  --clade-budget 315000 \
  --batch-packing depth_first_fit \
  --max-wave-size 8192 \
  --warmup-runs 1 \
  --profile-runs 5 \
  --mode stream-batches \
  --no-cuda-profiler-api
```

Median forward is `0.3094 s`, median backward is `0.4777 s`, and median
forward+backward is `0.7875 s`.  Peak allocated memory is `5.780 GiB`; peak
reserved memory is `6.963 GiB`.

## Backward Root RHS Initialization Plan

The next small overhead target is the backward initialization of
`accumulated_rhs` in `Pi_wave_backward`.  Each resident batch currently does:

```python
accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
accumulated_rhs.index_copy_(0, root_ids_device, root_rhs)
```

The zero initialization is required: every non-root Pi adjoint starts at exact
zero, and the pruning mask depends on those zeros.  The root rows are the only
non-zero initial rows.  The current path therefore performs one dense
initialization plus a separate root-row scatter per batch.

Experiment:

- add an opt-in `GPUREC_BACKWARD_FUSED_ROOT_RHS_INIT=1` path;
- prepare a static `root_position_index` in the wave layout, mapping every
  wave-ordered clade row to its root-row position or `-1`;
- allocate `accumulated_rhs` with `torch.empty` and initialize zeros/root RHS
  in one Triton kernel;
- keep the existing `torch.zeros` + `index_copy_` path as the default until
  correctness and profiling justify changing it;
- verify targeted model and chunked parity with the flag enabled;
- benchmark HOGENOM event timing and run `nsys` only if the median improves
  enough to check total GPU kernel time and the fill bucket.

Result: rejected.  The opt-in path passed the targeted parity suite, but event
timing did not improve, so no `nsys` run was warranted.

Correctness:

```bash
GPUREC_BACKWARD_FUSED_ROOT_RHS_INIT=1 pytest -q \
  tests/unit/test_specieswise_uniform.py \
  tests/unit/test_global_wave_scheduler.py \
  tests/integration/test_gene_recon_model.py::test_gene_recon_model_forward_backward_modes \
  tests/integration/test_gene_recon_model.py::test_memory_safe_resident_batches_match_resident_and_slice \
  tests/integration/test_uniform_chunked_model.py::test_chunked_uniform_matches_resident_global_model \
  tests/integration/test_uniform_chunked_model.py::test_chunked_uniform_chunk_subset_nll_and_gradient
```

Result: 21 passed.

Event timing:

| setting | median fwd+bwd | median forward | median backward | peak alloc | peak reserved | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| default root RHS init | 0.7770 s | 0.3087 s | 0.4682 s | 5.786 GiB | 6.982 GiB | baseline |
| fused root RHS init, block 512 | 0.7792 s | 0.3083 s | 0.4709 s | 5.786 GiB | 6.982 GiB | rejected |
| fused root RHS init, block 1024 | 0.7880 s | 0.3104 s | 0.4777 s | 5.786 GiB | 6.982 GiB | rejected |
| fused root RHS init, block 2048 | 0.7851 s | 0.3098 s | 0.4753 s | 5.786 GiB | 6.982 GiB | rejected |
| fused root RHS init, block 4096 | 0.7877 s | 0.3113 s | 0.4764 s | 5.786 GiB | 6.984 GiB | rejected |

The dense zero fill is apparently faster than the prototype's per-element root
mapping work.  The code change was removed; keep the default PyTorch
`zeros + index_copy_` initialization.

## CUDA Pibar-From-UD Plan

The current accepted profile still spends about `0.106 s` in
`_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel`.  The NCU diagnosis for a
large launch showed high achieved occupancy but poor memory coalescing:
DRAM throughput was high, useful bytes per sector were low, and excessive
global sectors were about 60% of total sectors.  Retuning
`GPUREC_PIBAR_UD_BLOCK_S` and `GPUREC_PIBAR_UD_NUM_WARPS` did not reduce total
GPU time.

The `main` branch contains an opt-in `GPUREC_CUDA_PIBAR_FROM_UD` prototype in
`gpurec/core/kernels/pibar_vjp_cuda.py`.  It runs one split-side row per CUDA
block, copies that row's `u_d` into shared memory, performs the species-tree
subtree reduction there, and atomically accumulates the final child-row RHS.
This directly targets the current compact Triton kernel's repeated global
loads/stores during the bottom-up tree walk.

Experiment:

- port only the compact-level CUDA path as an opt-in
  `GPUREC_CUDA_PIBAR_FROM_UD=1` route;
- keep the existing Triton compact path as the default and fallback;
- support fp32 CUDA only, matching the HOGENOM profile and the main-branch
  prototype;
- preserve `active_mask`, `side_active`, and compact species-level semantics;
- verify targeted parity with the flag enabled;
- benchmark HOGENOM event timing first, then run `nsys` only if the CUDA route
  improves the whole-dataset median.

## CUDA Pibar Shared-Padding Follow-Up Plan

The CUDA Pibar route improves the whole-dataset event median, and `nsys`
confirms lower total GPU kernel time.  NCU on the largest CUDA Pibar launch
shows:

- duration about `1.62 ms` for grid `86490`, block `256`;
- 26 registers/thread, no spills;
- theoretical occupancy 100%, achieved occupancy about 95%;
- DRAM throughput about 89%;
- shared-memory load/store bank conflicts are high, with NCU estimating
  uncoalesced shared accesses as a large remaining source of stalls.

Next experiment: add an opt-in
`GPUREC_CUDA_PIBAR_FROM_UD_PAD_SHARED=1` mode that stores species row scratch
at `s + floor(s / 32)` in dynamic shared memory.  This slightly increases
shared memory per CTA but may reduce bank conflicts in the bottom-up subtree
reduction.  Keep padding off unless event timing and `nsys` improve.

Result: accepted for the unpadded CUDA Pibar route; rejected for shared padding.
`GPUREC_CUDA_PIBAR_FROM_UD` now defaults to `auto`, with
`GPUREC_CUDA_PIBAR_FROM_UD=0` as the escape hatch and
`GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1` for experiments that must fail instead of
falling back.

Correctness:

```bash
GPUREC_CUDA_PIBAR_FROM_UD=1 \
GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1 \
pytest -q \
  tests/unit/test_specieswise_uniform.py \
  tests/unit/test_global_wave_scheduler.py \
  tests/integration/test_gene_recon_model.py::test_gene_recon_model_forward_backward_modes \
  tests/integration/test_gene_recon_model.py::test_memory_safe_resident_batches_match_resident_and_slice \
  tests/integration/test_uniform_chunked_model.py::test_chunked_uniform_matches_resident_global_model \
  tests/integration/test_uniform_chunked_model.py::test_chunked_uniform_chunk_subset_nll_and_gradient
```

Result: 21 passed.  The same suite also passed with
`GPUREC_CUDA_PIBAR_FROM_UD_PAD_SHARED=1`, and passed again with the promoted
default `auto` route and no environment variable.

Event timing:

| setting | median fwd+bwd | median forward | median backward | peak alloc | peak reserved | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Triton compact Pibar | 0.7843 s | 0.3103 s | 0.4749 s | 5.780 GiB | 6.965 GiB | baseline |
| CUDA Pibar, strict env | 0.7540 s | 0.3085 s | 0.4452 s | 5.780 GiB | 6.965 GiB | accept |
| CUDA Pibar, default auto | 0.7609 s | 0.3094 s | 0.4515 s | 5.780 GiB | 6.963 GiB | promoted |
| CUDA Pibar + shared padding | 0.7608 s | 0.3114 s | 0.4494 s | 5.780 GiB | 6.965 GiB | reject padding |

Nsight Systems for
`profiling/hogenom_ccp/nsys_stream_depthff315_cuda_pibar.nsys-rep`:

| setting | profiled pass | CUDA launches | GPU kernel time | Pibar VJP bucket |
| --- | ---: | ---: | ---: | ---: |
| Triton compact Pibar baseline | 0.8190 s | 15,606 | 0.7457 s | 0.1063 s / 244 |
| CUDA Pibar | 0.8050 s | 15,606 | 0.7305 s | 0.0827 s / 244 |

The CUDA path reduces the Pibar VJP bucket by about `23.6 ms`; total GPU kernel
time drops by about `15.2 ms`.  Some neighboring buckets move up slightly in
the profiled pass (`wave_step`, DTS backward, CUDA self-loop), but not enough
to erase the Pibar win.

Nsight Compute on the largest CUDA Pibar launch
(`profiling/hogenom_ccp/ncu_cuda_pibar_launch172.ncu-rep`):

- grid `86490`, block `256`, duration `1.62 ms`;
- 26 registers/thread, no spills;
- theoretical occupancy 100%, achieved occupancy about 95%;
- DRAM throughput about 89%, L2 hit rate about 16%;
- branch efficiency about 95%;
- uncoalesced global accesses are much lower than the Triton compact path
  (about 13% excessive sectors here versus about 60% in the Triton NCU);
- shared-memory bank conflicts remain significant, but the simple
  `s + floor(s / 32)` padding experiment did not improve end-to-end timing.

Diagnosis: moving the per-split-side species-tree reduction into row-local CUDA
shared memory is a real improvement for the current HOGENOM CCP workload.  It
does not reduce launch count or memory footprint, but it improves the dominant
Pibar VJP kernel enough to promote as the default auto path.

## CUDA Pibar Block-Size Sweep Plan

The promoted CUDA Pibar kernel still has one low-risk launch-shape knob:
`GPUREC_CUDA_PIBAR_FROM_UD_BLOCK`.  The accepted route uses 256 threads per
split-side row.  NCU reports high achieved occupancy but low eligible warps per
scheduler, with stalls dominated by L1TEX/global-memory dependencies.  A
different block size may change how quickly each row copies `u_d`, performs
the compact-level shared-memory reduction, and writes child RHS contributions.

Experiment:

- keep `GPUREC_CUDA_PIBAR_FROM_UD=auto` semantics unchanged;
- sweep `GPUREC_CUDA_PIBAR_FROM_UD_BLOCK` in `{128, 512, 1024}` against the
  default 256;
- use the accepted HOGENOM stream pass with one warmup and three measured runs
  for the first screen;
- accept a new default only if event timing improves and a follow-up `nsys`
  run confirms lower total GPU kernel time, not just a local CUDA-event fluctuation.

Result: rejected.  Keep the default block size at 256.

First screen:

| block | measured runs | median fwd+bwd | median forward | median backward | decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 3 | 0.7574 s | 0.3092 s | 0.4490 s | paired check |
| 512 | 3 | 0.7799 s | 0.3082 s | 0.4718 s | rejected |
| 1024 | 3 | 0.8744 s | 0.3081 s | 0.5663 s | rejected |

Paired five-run check:

| block | measured runs | median fwd+bwd | median forward | median backward | peak alloc | peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 5 | 0.7562 s | 0.3103 s | 0.4459 s | 5.780 GiB | 6.963 GiB |
| 128 | 5 | 0.7634 s | 0.3102 s | 0.4534 s | 5.780 GiB | 6.963 GiB |

The apparent 128-thread improvement in the three-run screen was noise.  Since
the paired check favors 256, no `nsys` run is needed.

## Post-CUDA-Pibar DTS Launch-Shape Sanity Plan

After promoting CUDA Pibar, the largest remaining buckets in
`nsys_stream_depthff315_cuda_pibar` are:

- `_wave_step_uniform_kernel`: about `0.215 s`;
- `_dts_cross_backward_accum_kernel`: about `0.157 s`;
- `gpurec_wave_backward_nosplit_uniform_fp32`: about `0.103 s`;
- `_dts_parent_reduced_ge2_stage1_kernel`: about `0.096 s`;
- `gpurec_pibar_from_ud_shared_fp32`: about `0.083 s`.

The DTS NCU result says `_dts_cross_backward_accum_kernel` is not occupancy- or
register-limited, so a broad rewrite is not justified by launch-shape tuning.
However, the old DTS launch-shape sweep predates CUDA Pibar and several
accepted changes.  Run a narrow sanity check of existing diagnostic knobs only:

- default `GPUREC_DTS_NUM_WARPS=8`, implicit `BLOCK_S=256`;
- `GPUREC_DTS_NUM_WARPS=4`;
- `GPUREC_DTS_NUM_WARPS=16`;
- `GPUREC_DTS_BLOCK_S=128` with default 8 warps.

Use the accepted HOGENOM stream timing with one warmup and three measured runs.
Run `nsys` only if a setting gives a clear event-time improvement.

Result: no change.  Keep `GPUREC_DTS_NUM_WARPS=8` and implicit
`GPUREC_DTS_BLOCK_S=256`.

First screen:

| setting | measured runs | median fwd+bwd | median forward | median backward | peak reserved | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| default | 3 | 0.7648 s | 0.3108 s | 0.4546 s | 6.963 GiB | baseline |
| `GPUREC_DTS_NUM_WARPS=4` | 3 | 0.7760 s | 0.3105 s | 0.4655 s | 6.965 GiB | rejected |
| `GPUREC_DTS_NUM_WARPS=16` | 3 | 0.7564 s | 0.3110 s | 0.4455 s | 6.965 GiB | paired check |
| `GPUREC_DTS_BLOCK_S=128` | 3 | 0.7631 s | 0.3093 s | 0.4538 s | 6.963 GiB | rejected |

Paired five-run check:

| setting | measured runs | median fwd+bwd | median forward | median backward | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | 5 | 0.7575 s | 0.3097 s | 0.4471 s | 6.963 GiB |
| `GPUREC_DTS_NUM_WARPS=16` | 5 | 0.7572 s | 0.3111 s | 0.4460 s | 6.965 GiB |

The paired result is a tie, not a clear win.  Since NCU already indicated this
kernel is memory/divergence limited, do not spend another `nsys` run or promote
a 16-warp default.

## Post-CUDA-Pibar No-Family-Cap Scheduling Retest Plan

The accepted `family_chunk_size=300, clade_budget=315000` HOGENOM layout has 5
resident batches and 258 waves.  The earlier no-family-cap diagnostic
(`--chunk-size 0 --clade-budget 315000`) produced 4 resident batches and 240
waves, but was rejected because total GPU kernel time did not improve and peak
reserved memory rose to about 11 GiB.

That rejection predates the promoted CUDA Pibar route, which changes the
relative cost of split-side Pibar VJP work.  Retest exactly this one lower-wave
layout under current defaults before looking at more aggressive clade budgets.

Experiment:

- benchmark the accepted 300-family cap and no-family-cap layouts with one
  warmup and five measured HOGENOM stream passes;
- require identical loss/gradient within existing fp32 run noise;
- accept the no-family-cap layout only if event timing improves materially and
  a follow-up `nsys` run confirms lower total GPU kernel time;
- keep the 300-family cap if the win is only a CUDA-event fluctuation or if the
  memory-reservation tradeoff remains poor.

Result: keep the 300-family cap as the conservative memory-default layout, but
record `--chunk-size 0 --clade-budget 315000` as a faster high-memory
scheduling option under the CUDA Pibar default.

Event timing:

| family cap | batches | waves | median fwd+bwd | median forward | median backward | peak alloc | peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 5 | 258 | 0.7569 s | 0.3090 s | 0.4483 s | 5.780 GiB | 6.963 GiB |
| none (`--chunk-size 0`) | 4 | 240 | 0.7456 s | 0.3079 s | 0.4377 s | 5.980 GiB | 10.994 GiB |

Nsight Systems for
`profiling/hogenom_ccp/nsys_stream_depthff315_cuda_pibar_no_family_cap.nsys-rep`:

| family cap | profiled pass | CUDA launches | GPU kernel time | wave-step | DTS backward | CUDA self-loop | CUDA Pibar |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 0.8050 s | 15,606 | 0.7305 s | 0.2151 s / 1,548 | 0.1572 s / 244 | 0.1030 s / 258 | 0.0827 s / 244 |
| none | 0.7869 s | 13,464 | 0.7239 s | 0.2140 s / 1,440 | 0.1508 s / 227 | 0.1063 s / 240 | 0.0830 s / 227 |

The no-family-cap layout removes 18 waves and 2,142 launches.  GPU kernel time
drops by about `6.6 ms`; measured-pass wall time drops by about `18 ms`; median
event timing drops by about `11 ms`.  The main tradeoff is memory reservation:
peak allocated remains near the lean envelope at `5.98 GiB`, but PyTorch peak
reserved memory rises to about `11.0 GiB`.  Keep the 300-family cap for
memory-conservative runs; use `--chunk-size 0` when runtime is prioritized and
the larger CUDA reservation is acceptable.

## No-Family-Cap Clade-Budget Knee Search Plan

The no-family-cap `315000` layout improves runtime but increases peak reserved
memory substantially.  Before treating that as the only high-performance
scheduling option, look for a nearby clade-budget knee:

- keep `family_chunk_size=0` and `batch_packing=depth_first_fit`;
- sweep clade budgets below `315000` with metadata only first;
- record batch count, total waves, max clades, max splits, and max DTS partial
  rows;
- benchmark only layouts that retain most of the 240-wave improvement while
  reducing max active batch size;
- require event timing and, for any promoted candidate, `nsys` confirmation.

Metadata sweep:

| clade budget | batches | total waves | max clades | max splits | max DTS partial rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 260,000 | 4 | 261 | 259,972 | 707,510 | 53,583 |
| 275,000 | 4 | 255 | 274,985 | 739,740 | 53,583 |
| 290,000 | 4 | 251 | 289,974 | 788,865 | 56,916 |
| 300,000 | 4 | 247 | 299,948 | 824,740 | 57,052 |
| 305,000 | 4 | 245 | 304,913 | 841,056 | 57,052 |
| 310,000 | 4 | 243 | 309,998 | 859,025 | 57,052 |
| 315,000 | 4 | 240 | 314,585 | 873,723 | 85,464 |

Event timing:

| clade budget | waves | median fwd+bwd | median forward | median backward | peak alloc | peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300,000 | 247 | 0.7443 s | 0.3056 s | 0.4388 s | 5.489 GiB | 10.309 GiB |
| 305,000 | 245 | 0.7435 s | 0.3056 s | 0.4379 s | 5.675 GiB | 8.027 GiB |
| 310,000 | 243 | 0.7406 s | 0.3070 s | 0.4341 s | 5.843 GiB | 8.027 GiB |
| 315,000 | 240 | 0.7456 s | 0.3079 s | 0.4377 s | 5.980 GiB | 10.994 GiB |

Nsight Systems validation:

| layout | waves | profiled pass | CUDA launches | GPU kernel time | DTS backward | CUDA self-loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 300-family cap / 315k | 258 | 0.8050 s | 15,606 | 0.7305 s | 0.1572 s | 0.1030 s |
| no family cap / 315k | 240 | 0.7869 s | 13,464 | 0.7239 s | 0.1508 s | 0.1063 s |
| no family cap / 310k | 243 | 0.8000 s | 13,560 | 0.7425 s | 0.1784 s | 0.0987 s |
| no family cap / 305k | 245 | 0.7840 s | 13,624 | 0.7184 s | 0.1532 s | 0.0992 s |

Result: the `305000` no-family-cap layout is the best validated
speed/memory knee.  The `310000` layout had the best five-run event median, but
`nsys` rejected it because DTS backward jumped to `0.1784 s` and total GPU
kernel time regressed.  The `300000` layout has lower peak allocation but worse
reserved memory and no timing advantage over 305k.  Keep the 300-family cap as
the conservative memory-default recommendation; use
`--chunk-size 0 --clade-budget 305000` as the best validated higher-memory
runtime option under the current CUDA Pibar default.

## 305k No-Family-Cap Wave-Cap Retest Plan

The best validated high-performance layout now uses
`family_chunk_size=0`, `clade_budget=305000`, `batch_packing=depth_first_fit`,
and `max_wave_size=8192`.  Larger wave caps were rejected earlier for the
300-family/315k layout because they removed few waves and did not reduce total
GPU time.  Retest this only for the new 305k no-family-cap layout:

- run metadata first for `max_wave_size` in `{8192, 12288, 16384, 24576,
  32768}`;
- benchmark only caps that remove a meaningful number of waves without
  increasing max active batch size;
- accept a wave-cap change only if event timing improves and `nsys` confirms
  lower total GPU kernel time.

Metadata sweep, forcing all batch statics to build:

| `max_wave_size` | waves by batch | total waves | max wave | max clades | max splits | max DTS partial rows |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8,192 | `[102, 64, 51, 28]` | 245 | 8,192 | 304,913 | 841,056 | 68,969 |
| 12,288 | `[101, 63, 50, 27]` | 241 | 12,288 | 304,923 | 841,607 | 62,510 |
| 16,384 | `[100, 63, 49, 27]` | 239 | 16,384 | 304,923 | 841,607 | 75,840 |
| 24,576 | `[100, 62, 49, 27]` | 238 | 24,576 | 304,923 | 841,607 | 62,510 |
| 32,768 | `[99, 62, 48, 27]` | 236 | 27,144 | 304,923 | 841,607 | 62,510 |

The max-DTS-partial-row values here come from fully materialized batch statics.
They should be preferred over the earlier lazy metadata table for this field.

Event timing:

| `max_wave_size` | total waves | event median fwd+bwd | median forward | median backward | peak alloc | peak reserved | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8,192 | 245 | 0.7435 s | 0.3056 s | 0.4379 s | 5.675 GiB | 8.027 GiB | baseline |
| 12,288 | 241 | 0.7373 s | 0.3060 s | 0.4316 s | 5.572 GiB | 6.852 GiB | send to `nsys` |
| 32,768 | 236 | 0.7440 s | 0.3068 s | 0.4370 s | 5.492 GiB | 7.170 GiB | rejected |

Nsight Systems validation:

| layout | waves | profiled pass | CUDA launches | GPU kernel time | wave-step | DTS backward | CUDA self-loop | CUDA Pibar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 305k / wave 8,192 | 245 | 0.7840 s | 13,624 | 0.7184 s | 0.2136 s / 1,470 | 0.1532 s / 232 | 0.0992 s / 245 | 0.0829 s / 232 |
| 305k / wave 12,288 | 241 | 0.7921 s | 13,592 | 0.7345 s | 0.2134 s / 1,446 | 0.1658 s / 232 | 0.1023 s / 241 | 0.0830 s / 232 |

Result: reject the larger wave cap for now.  `max_wave_size=12288` removes four
waves and 32 launches, but the profiler run is slower: total GPU kernel time
increases by about `16 ms`, mainly from DTS backward and the CUDA self-loop
bucket.  Keep `max_wave_size=8192` for both the conservative 300-family-cap
layout and the higher-memory `--chunk-size 0 --clade-budget 305000` runtime
layout.

## Main-Branch 2D Backward Alternative Audit Plan

The retained split-wave backward path still spends meaningful time in the 2D
Triton `J^T` strategy, and the objective calls out the possibility that an
older/main-branch alternative may trade memory or compile overhead differently.
Before changing code, audit the main branch rather than guessing:

- grep main-branch docs for `2D` and related backward-kernel notes;
- inspect the main-branch backward kernel code around any alternative path;
- compare the alternative against the current retained path in terms of launch
  count, scratch memory, supported parameter modes, and expected correctness
  surface;
- only prototype if there is a concrete low-risk switch or environment-gated
  candidate;
- require targeted parity tests, warm HOGENOM timing, and `nsys` confirmation
  before accepting any change.

Audit result:

- main's Proposal 1 staged tree path is not a good restoration candidate as
  implemented; the archived profile says it passed parity but regressed in
  timing, added many launches/scratch stages, and OOMed at larger sizes;
- main's Proposal 0 2D path is the retained branch that was later superseded
  for HOGENOM split/no-split waves by the CUDA self-loop route already present
  here;
- the useful main-branch alternatives already acted on in this branch are the
  exact CUDA self-loop and CUDA Pibar routes, so do not spend time re-porting
  the staged tree prototype without a new design that reduces launches and
  scratch traffic.

## Batch-Composition Local-Search Plan

The intra-batch scheduler is already leaf-first and lower-bound optimal for
the accepted HOGENOM batch layouts, so a better ready-queue heuristic cannot
remove waves by itself.  The remaining scheduling lever is batch composition:
assign families to resident batches so the sum of per-batch lower bounds is
small while staying within clade, split, and observed memory envelopes.

Next experiment:

- load HOGENOM family scheduling metadata from the preprocessing cache only;
- compare the existing `depth_first_fit` batches against metadata-only local
  search / random-restart packings under `clade_budget` values near the current
  best (`300000`, `305000`, `310000`, `315000`);
- score candidates by summed leaf-first lower bound, max clades, max splits,
  and an approximate DTS partial-row proxy before materializing expensive
  statics;
- only build and time a candidate if metadata reduces waves or improves the
  DTS/split envelope relative to the validated `305000` no-family-cap layout;
- accept no new packing policy without HOGENOM loss/gradient parity, warm
  event timing, and `nsys` confirmation of lower total GPU kernel time.

Metadata search result: no new benchmark candidate.

The search loaded 1,055 HOGENOM families from the preprocessing cache and
tested the existing `depth_first_fit` heuristic against deterministic orderings
by clade count, non-leaf count, per-family wave cost, split count, plus 30
random greedy starts per budget and a short move/swap local search.  The local
search used the same leaf-first lower-bound objective that the exact scheduler
matches on these layouts.

| clade budget | best metadata packing | batches | lower-bound waves | waves by batch | max clades | max splits | decision |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 300,000 | `depth_first_fit` | 4 | 247 | `[102, 64, 51, 30]` | 299,948 | 824,740 | no improvement |
| 305,000 | `depth_first_fit` | 4 | 245 | `[102, 64, 51, 28]` | 304,913 | 841,056 | keep validated runtime candidate |
| 310,000 | `depth_first_fit` + tiny local-search split reduction | 4 | 243 | `[102, 64, 50, 27]` | 309,993 | 858,401 | no material change; prior `nsys` rejected 310k |
| 315,000 | `depth_first_fit` | 4 | 240 | `[102, 65, 49, 24]` | 314,585 | 873,723 | no improvement |

The only metadata change was a tiny max-split reduction for the already
rejected 310k budget (`859,025 -> 858,401`) with the same wave count.  That is
not enough to justify materializing a custom packing and re-running Nsight,
because the prior 310k profile regressed in DTS backward by much more than this
split-count difference can plausibly explain.  Keep the validated
`--chunk-size 0 --clade-budget 305000` high-performance option and do not add a
new packing policy from this search.

## Commands

Warm whole-dataset stream timing, conservative memory layout:

```bash
python scripts/profile_hogenom_ccp_pass.py \
  --chunk-size 300 \
  --clade-budget 315000 \
  --batch-packing depth_first_fit \
  --max-wave-size 8192 \
  --warmup-runs 1 \
  --profile-runs 5 \
  --mode stream-batches \
  --no-cuda-profiler-api
```

Warm whole-dataset stream timing, best validated runtime layout:

```bash
python scripts/profile_hogenom_ccp_pass.py \
  --chunk-size 0 \
  --clade-budget 305000 \
  --batch-packing depth_first_fit \
  --max-wave-size 8192 \
  --warmup-runs 1 \
  --profile-runs 5 \
  --mode stream-batches \
  --no-cuda-profiler-api
```

Production closure timing, conservative memory layout:

```bash
python scripts/profile_hogenom_ccp_pass.py \
  --chunk-size 300 \
  --clade-budget 315000 \
  --batch-packing depth_first_fit \
  --max-wave-size 8192 \
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

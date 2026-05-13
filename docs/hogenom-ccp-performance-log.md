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

Measured warm runtime after the scheduler change:

| family chunk | batches | total waves | max wave | warm fwd+bwd | peak alloc | peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 43 | 3185 | 3217 | 3.145 s | 1.08 GiB | 1.72 GiB |
| 100 | 11 | 927 | 8192 | 1.624 s | 2.65 GiB | 2.83 GiB |
| 200 | 6 | 522 | 8192 | 1.390 s | 4.23 GiB | 8.47 GiB |
| 250 | 5 | 430 | 8192 | 1.360 s | 5.05 GiB | 10.27 GiB |
| 300 | 4 | 371 | 8192 | 1.264 s | 5.92 GiB | 11.95 GiB |
| 400 | 3 | 286 | 8192 | 1.307 s | 7.92 GiB | 16.31 GiB |
| 600 | 2 | 193 | 8192 | 1.289 s | 15.24 GiB | 16.71 GiB |

The chunk-300 row includes the later DTS launch-warp tuning, no-host pruning,
2D `J^T` warp retuning, and forward wave-step warp retuning.  Without the
no-host-pruning override the same tuned code measured 1.321 s before the
wave-step retune.  The other rows are the scheduler/self-loop-tuned measurements
used for memory tradeoff decisions.

The best warm value inside the 5-6 GiB allocated target is chunk size 300 at
about 1.26 s.  Larger chunks keep reducing waves but give small returns relative
to memory: chunk 600 uses 15.24 GiB for only about 44 ms over chunk 300.

The first pass for large chunks is still expensive because Triton compiles
larger wave/kernel variants.  Removing `W: tl.constexpr` from the retained 2D
backward self-loop kernels reduced one source of wave-size-specific
compilation, but warm-up is still much slower than steady state.

## Nsight Findings

Nsight Systems on tuned chunk size 300 with
`GPUREC_BACKWARD_NO_CPU_PRUNING=1` measured one profiled pass at 1.380 s
(`nsys` overhead relative to the uninstrumented 1.264 s median).  The measured
pass had 52,557 CUDA kernel launches and 1.161 s of GPU kernel time.

Top kernel families in the chunk-300 `nsys` report:

| kernel | launches | total GPU time | avg launch |
| --- | ---: | ---: | ---: |
| `_wave_backward_uniform_2d_jt_kernel` | 2226 | 0.268 s | 120.4 us |
| `_wave_step_uniform_kernel` | 2226 | 0.198 s | 88.9 us |
| `_dts_cross_backward_accum_kernel` | 358 | 0.161 s | 451.1 us |
| `_dts_parent_reduced_ge2_stage1_kernel` | 708 | 0.108 s | 152.5 us |
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

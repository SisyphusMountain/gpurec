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
| 300 | 4 | 371 | 8192 | 1.333 s | 5.92 GiB | 11.95 GiB |
| 400 | 3 | 286 | 8192 | 1.307 s | 7.92 GiB | 16.31 GiB |
| 600 | 2 | 193 | 8192 | 1.289 s | 15.24 GiB | 16.71 GiB |

The best warm value inside the 5-6 GiB allocated target is chunk size 300 at
about 1.33 s.  Larger chunks keep reducing waves but give small returns relative
to memory: chunk 600 uses 15.24 GiB for only about 44 ms over chunk 300.

The first pass for large chunks is still expensive because Triton compiles
larger wave/kernel variants.  Removing `W: tl.constexpr` from the retained 2D
backward self-loop kernels reduced one source of wave-size-specific
compilation, but warm-up is still much slower than steady state.

## Nsight Findings

Nsight Systems on tuned chunk size 300 measured one profiled pass at 1.471 s
(`nsys` overhead relative to the uninstrumented 1.332 s median).  The measured
pass had 53,309 CUDA kernel launches and 1.206 s of GPU kernel time.  NVTX
forward time was 0.360 s and backward time was 1.111 s.

Top kernel families in the chunk-300 `nsys` report:

| kernel | launches | total GPU time | avg launch |
| --- | ---: | ---: | ---: |
| `_wave_backward_uniform_2d_jt_kernel` | 2172 | 0.271 s | 125.0 us |
| `_wave_step_uniform_kernel` | 2226 | 0.216 s | 97.1 us |
| `_dts_cross_backward_accum_kernel` | 349 | 0.179 s | 514.3 us |
| `_dts_parent_reduced_ge2_stage1_kernel` | 699 | 0.108 s | 154.4 us |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | 349 | 0.107 s | 306.2 us |
| `_wave_backward_uniform_2d_precompute_kernel` | 362 | 0.068 s | 186.5 us |

GPU metrics over the measured pass: GR active 83.4%, SMs active 72.5%, SM issue
24.0%, compute warps in flight 41.8%, DRAM read 27.4%, DRAM write 20.2%.

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
- `_dts_cross_backward_accum_kernel` is a more concerning backward target:
  a representative grid 2718 reaches only about 52% memory throughput and 26%
  compute throughput, with 96 registers/thread and 41.7% theoretical occupancy.
  Some later launches are very small grids and show tail/underfill effects.

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

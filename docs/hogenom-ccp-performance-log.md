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
| 200 | 6 | 522 | 8192 | 1.440 s | 4.23 GiB | 8.47 GiB |
| 250 | 5 | 430 | 8192 | 1.395 s | 5.05 GiB | 10.27 GiB |
| 300 | 4 | 371 | 8192 | 1.371 s | 5.92 GiB | 11.95 GiB |
| 400 | 3 | 286 | 8192 | 1.321 s | 7.92 GiB | 16.31 GiB |
| 600 | 2 | 193 | 8192 | 1.296 s | 15.24 GiB | 16.71 GiB |

The best warm value inside the 5-6 GiB allocated target is chunk size 300 at
about 1.37 s.  Larger chunks keep reducing waves but give small returns relative
to memory: chunk 600 uses 15.24 GiB for only about 75 ms over chunk 300.

The first pass for large chunks is still expensive because Triton compiles
larger wave/kernel variants.  Removing `W: tl.constexpr` from the retained 2D
backward self-loop kernels reduced one source of wave-size-specific
compilation, but warm-up is still much slower than steady state.

## Nsight Findings

Nsight Systems on chunk size 300 measured one profiled pass at 1.496 s
(`nsys` overhead relative to the uninstrumented 1.371 s median).  The measured
pass had 53,309 CUDA kernel launches and 1.231 s of GPU kernel time.  NVTX
forward time was 0.360 s and backward time was 1.135 s.

Top kernel families in the chunk-300 `nsys` report:

| kernel | launches | total GPU time | avg launch |
| --- | ---: | ---: | ---: |
| `_wave_backward_uniform_2d_jt_kernel` | 2172 | 0.294 s | 135.6 us |
| `_wave_step_uniform_kernel` | 2226 | 0.217 s | 97.3 us |
| `_dts_cross_backward_accum_kernel` | 349 | 0.181 s | 517.7 us |
| `_dts_parent_reduced_ge2_stage1_kernel` | 699 | 0.108 s | 154.4 us |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | 349 | 0.106 s | 304.9 us |
| `_wave_backward_uniform_2d_precompute_kernel` | 362 | 0.067 s | 186.1 us |

GPU metrics over the measured pass: GR active 82.9%, SMs active 72.3%, SM issue
27.0%, compute warps in flight 45.5%, DRAM read 26.7%, DRAM write 20.0%.

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

## Next Hypotheses

1. Treat chunk size 300 as the current 5-6 GiB target configuration.
2. Try suboptimal-option diagnostics before redesign:
   `GPUREC_SELF_LOOP_2D_BLOCK_W`, `GPUREC_SELF_LOOP_2D_NUM_WARPS`,
   `GPUREC_SELF_LOOP_2D_BLOCK_NODES`, and pruning on/off.
3. Inspect the DTS backward accumulation path first.  It is now a larger share
   of wall time than launch overhead and has lower utilization than the packed
   forward wave kernel.
4. If profiling supports it, prototype a lower-scratch or lower-register
   backward path and compare against the retained 2D path for correctness and
   warm runtime.

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

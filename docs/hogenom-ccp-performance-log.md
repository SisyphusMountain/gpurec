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

| family chunk | batches | total waves | max wave | warm fwd+bwd | peak alloc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 43 | 3185 | 3217 | 3.145 s | 1.08 GiB |
| 100 | 11 | 927 | 8192 | 1.624 s | 2.65 GiB |
| 200 | 6 | 522 | 8192 | 1.439 s | 4.23 GiB |

The best warm value so far is chunk size 200 at about 1.44 s.  The first pass
for large chunks is still expensive because Triton compiles larger wave/kernel
variants.  Removing `W: tl.constexpr` from the retained 2D backward self-loop
kernels reduced one source of wave-size-specific compilation, but warm-up is
still much slower than steady state.

## Verified So Far

- Scheduler CPU unit tests cover global packing, wave cap, and topological
  ordering.
- Targeted CUDA tests pass for `GeneReconModel` forward/backward modes and
  resident batch parity.
- Targeted CUDA tests pass for `UniformChunkedReconModel` parity and chunk
  subset gradients.
- HOGENOM timings above were measured with `scripts/profile_hogenom_ccp_pass.py`
  after a warm pass.

## Next Hypotheses

1. Inspect `main` for the older non-2D backward path or documented alternative.
   The retained 2D path may trade speed for memory and compile overhead.
2. Profile chunk size 200 with `nsys` after the scheduler change to confirm
   kernel count, kernel time, SM activity, and remaining overhead.
3. Use `ncu` on the dominant chunk-200 backward kernels.  Check whether the
   larger packed waves fix small-grid underutilization or whether the 2D
   self-loop remains inefficient.
4. Try suboptimal-option diagnostics before redesign:
   `GPUREC_SELF_LOOP_2D_BLOCK_W`, pruning on/off, and chunk size vs memory.
5. If profiling supports it, prototype a non-2D or lower-scratch backward path
   and compare against the retained 2D path for correctness and warm runtime.

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

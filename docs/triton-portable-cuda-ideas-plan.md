# Plan: Port Portable CUDA Performance Ideas Back To Triton

## Goal

Keep the maintainability of the retained Triton backend while testing whether the useful CUDA prototype ideas can be expressed in Triton with comparable performance.

Target area: backward self-loop and Pibar VJP in `gpurec/core/kernels/wave_backward.py`.

## Baseline

Establish a clean Triton-only baseline by disabling native CUDA prototypes:

```bash
GPUREC_CUDA_SELF_LOOP_NOSPLIT=off \
GPUREC_CUDA_SELF_LOOP_SPLIT=off \
GPUREC_CUDA_PIBAR_FROM_UD=off
```

Record:

- HOGENOM median forward+backward, forward, and backward time.
- Peak allocated and reserved memory.
- Nsight Systems kernel buckets for self-loop, Pibar VJP, DTS backward, and PyTorch reductions.
- Current default `auto` behavior as the performance target.

## Candidate 1: Triton Child-Edge Self-Loop Weights

CUDA improvement:

- Replaces separate `sl1w` and `sl2w` parent-side arrays with one child-indexed edge-weight array.
- Reduces row-local scratch and improves occupancy in the CUDA kernel.

Triton port idea:

- Replace or supplement the current `sl1_ptr` and `sl2_ptr` scratch layout in `_self_loop_coefficients_kernel`.
- Store edge weight by child species index where possible.
- In `_self_loop_adjoint_update_kernel`, read the child-indexed weight directly instead of loading the parent and choosing `sl1` versus `sl2`.

Expected benefit:

- Fewer `[W, S]` scratch arrays.
- Less global memory traffic.
- Simpler `J^T` child contribution logic.

Gate:

- Match existing gradients within current fp32 tolerance.
- Reduce the self-loop bucket or total backward median versus the Triton-only baseline.

## Candidate 2: Triton Direct Gradient Accumulation For Eligible Shared/Fp32 Path

CUDA improvement:

- Accumulates parameter gradients inside the row kernel.
- Avoids returning large `aw*` tensors and many PyTorch reductions.

Triton port idea:

- Add an eligible fast path for auto-wrapped shared/specieswise fp32 cases.
- In the final Triton param-store kernel, atomically accumulate directly into gradient outputs.
- Avoid materializing some or all of `aw0`, `aw1`, `aw2`, `aw345`, `aw3`, and `aw4`.

Expected benefit:

- Lower memory allocation.
- Fewer post-kernel PyTorch reductions.
- Potential launch-count reduction.

Risk:

- Atomics may be slower for dense or high-contention cases.
- Scalar, `[S]`, and `[G, S]` gradient layouts may need separate handling.

Gate:

- Keep the old materialized path as fallback.
- Accept only if whole backward median improves, not just allocation size.

## Candidate 3: Fuse Triton Self-Loop Iterations For Small Or Medium `S`

CUDA improvement:

- Keeps `term`, `work`, `vacc`, coefficients, and correction scratch row-local inside one kernel.
- Loops over all Neumann terms inside the kernel.

Triton port idea:

- Prototype a specialized Triton kernel for eligible fp32/shared rows where `S` is small enough to keep row vectors inside one program.
- Fuse precompute, the Neumann loop, and parameter contribution calculation.
- Avoid global round trips between Neumann iterations.

Expected benefit:

- Fewer launches.
- Less global scratch traffic.

Risk:

- Register pressure or spills may erase the win.
- Triton lacks CUDA-style dynamic shared-memory arrays, so this may not scale to large `S`.

Gate:

- Prototype only after Candidates 1 and 2.
- Accept only with Nsight confirmation of lower total GPU kernel time.

## Candidate 4: Pibar VJP Triton Retune Inspired By CUDA

CUDA improvement:

- Copies each split-side `u_d` row into shared memory.
- Performs bottom-up species-tree reduction locally.
- Avoids repeated uncoalesced global loads and stores.

Triton limitation:

- Triton does not expose the same simple dynamic shared-memory scratch model for arbitrary indexed `work[parent]`, `work[c1]`, and `work[c2]`.

Possible Triton experiments:

- Try a register-vector version for small enough `S`.
- Try a level-tiled version that reduces global memory traffic by processing more species per program.
- Revisit layout of compact level arrays to improve coalescing.
- Keep CUDA Pibar as the performance target unless a Triton version clearly wins.

Expected benefit:

- Lower Pibar global memory traffic if the tree reduction can stay local enough.

Risk:

- Likely lower return than self-loop work.
- May become more complex than the CUDA prototype without matching its performance.

Gate:

- Do not replace the current Triton compact kernel unless the Pibar bucket and total GPU time improve.

## Correctness Suite

For each candidate:

```bash
pytest -q \
  tests/unit/test_specieswise_uniform.py \
  tests/unit/test_global_wave_scheduler.py \
  tests/integration/test_gene_recon_model.py::test_gene_recon_model_forward_backward_modes \
  tests/integration/test_gene_recon_model.py::test_memory_safe_resident_batches_match_resident_and_slice \
  tests/integration/test_uniform_chunked_model.py::test_chunked_uniform_matches_resident_global_model \
  tests/integration/test_uniform_chunked_model.py::test_chunked_uniform_chunk_subset_nll_and_gradient
```

Also run a direct HOGENOM comparison:

- Same loss.
- Gradient max absolute delta comparable to current CUDA/Triton differences.
- Gradient norm relative delta acceptable.

## Benchmark Gate

For every candidate:

1. Compare against the Triton-only baseline.
2. Compare against the current default CUDA-auto path.
3. Use at least one warmup and three measured runs for the first screen.
4. Use five measured runs before promotion.
5. Run Nsight Systems only if CUDA-event timing improves materially.
6. Promote only if median forward+backward time improves, backward median improves, total GPU kernel time does not regress, and memory does not materially regress.

## Recommended Order

1. Child-edge Triton self-loop weights.
2. Direct Triton gradient accumulation for eligible shared/fp32 path.
3. Small/medium `S` fused Triton self-loop prototype.
4. Pibar VJP Triton experiments only after self-loop wins are exhausted.

## Non-Goals

- Do not remove native CUDA prototypes during this experiment.
- Do not change default routing until benchmark evidence exists.
- Do not force the CUDA shared-memory design into Triton if it becomes more complex than the native CUDA path.

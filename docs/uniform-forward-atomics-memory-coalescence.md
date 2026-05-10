# Uniform Forward Atomics, Memory, and Coalescence Notes

Date: 2026-05-05

This document records a code-read pass over the current uniform-mode forward
path.  The goal is to identify forward-pass atomics, memory-management issues,
uncoalesced accesses, and concrete experiments to prepare for implementation
and benchmarking.

Related documents:

- `docs/uniform-forward-profile.md`
- `docs/uniform-forward-optimization-proposals.md`
- `docs/uniform-forward-backward-full-pipeline-profile.md`
- `docs/uncoalesced-atomic-backward.md`

## Scope

The production path considered here is `pibar_mode="uniform"` on CUDA through
`Pi_wave_forward` in `gpurec/core/forward.py`.

Important active defaults in the optimized path are:

```text
GPUREC_FORWARD_LEAF_INDEX=1
GPUREC_UNIFORM_PINGPONG=1
GPUREC_FORWARD_PARENT_REDUCED_DTS=1
GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL=tiled
GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY=1
GPUREC_FORWARD_TOPOLOGY_INT32=1
```

The main custom kernels are:

- `_wave_step_uniform_kernel` in `gpurec/core/kernels/wave_step.py`
- `_wave_pibar_uniform_parent_kernel` in `gpurec/core/kernels/wave_step.py`
- `_dts_fused_kernel` and parent-reduced DTS kernels in
  `gpurec/core/kernels/dts_fused.py`

## Executive Summary

There is no obvious custom atomic hotspot in the optimized uniform forward
path.  The Triton forward kernels avoid atomics by assigning ownership so that
one program writes each output row/species tile or each reduced parent row.

The forward opportunities are instead:

1. reduce or avoid final Pibar recomputation/storage where Pibar is not live;
2. reduce DTS scratch allocation, fill traffic, and partial-buffer traffic;
3. improve locality for species parent/child gathers, likely through a
   benchmarked species-index remap rather than a local kernel tweak;
4. keep fp32 on the current parent-walk path unless a full fused replacement
   beats it, but consider fp64-specific Pibar variants separately;
5. treat CUDA graph capture, stream overlap, and chunk auto-selection as
   secondary memory/launch-management experiments.

Forward optimization should not start by replacing atomics.  There are no
Triton atomics in the current optimized forward kernels.

## Atomic Audit

### Optimized Uniform CUDA Path

No `tl.atomic_*` use was found in the forward wave-step or DTS Triton kernels.

The main ownership patterns are:

| Area | Ownership pattern | Atomic status |
|---|---|---|
| Uniform wave step | one Triton program per clade row, iterating species tiles | no atomics |
| Final parent Pibar | one Triton program per clade row | no atomics |
| Standard DTS | one program per split/species tile writes `[n_splits, S]` | no atomics |
| Parent-reduced eq1 DTS | one split writes directly to its unique parent row | no atomics |
| Parent-reduced ge2 DTS | stage 1 writes unique partial rows, stage 2 writes parent rows | no atomics |

The parent-reduced DTS design is the right forward analogue of "avoid atomics":
it changes the reduction layout so that each output location has a unique
writer at each stage.

### Fallback PyTorch Paths

The generic reduced-DTS fallback in `gpurec/core/forward.py` uses
`scatter_reduce_` and `scatter_add_`.  On CUDA, those PyTorch operations may
use atomics internally.

That path is not the optimized large uniform CUDA default.  It should remain a
fallback and parity reference, not the first target for forward optimization.

### Indexed Leaf Initialization

Forward setup initializes leaves with indexed writes:

```python
Pi[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0
```

This is an indexed write kernel, but the expected layout has one finite species
per leaf row, so it is not an algorithmic atomic-reduction issue.  The compact
leaf-index path is already the important optimization: it avoids dense
`[W, S]` leaf masks in the hot uniform fused path.

## Memory and Coalescence Audit

### Pi/Pibar Setup

`Pi_wave_forward` allocates full `[C, S]` tensors for `Pi` and `Pibar`.
Training needs the full saved state for backward.  Likelihood-only modes can
avoid returning or materializing some final outputs, but forward DTS still needs
child `Pibar` rows while building later waves.

Current setup cost includes:

- `torch.full((C, S), ...)` for `Pi`;
- leaf indexed write into `Pi`;
- `torch.full((C, S), -inf)` for `Pibar`;
- optional `uniform_pibar_row_max[C]` when backward/Pibar VJP stats are reused.

The dense leaf fallback still exists for non-default variants, but the default
uniform fused path uses compact `leaf_species_index[C]` plus `leaf_logp`.

### Uniform Wave Step

The hot forward kernel does two row-local passes:

1. a coalesced scan of `Pi[row, :]` to compute `row_max` and `row_sum`;
2. a species pass that computes uniform `Pibar`, DTS_L terms, child species
   gathers, leaf terms, optional DTS, and the new `Pi`.

The first pass is coalesced.  The second pass has the important irregular
patterns:

- parent-pointer walk through `sp_parent` for each species;
- uncoalesced `Pi[row, cur]` ancestor loads during that walk;
- child-species gathers through `sp_child1` and `sp_child2`;
- optional `DTS_reduced[row, s]` load;
- global store of `Pi_new` and, outside ping-pong mode, `Pibar`.

The current parent-pointer implementation is deliberately still the fp32
default.  Earlier ancestor-table, CSR, linear, two-kernel, and sparse-matmul
variants were correct but slower in fp32 end-to-end forward profiles.  The
parent-pointer path has loop-carried dependency, but it keeps the index working
set small and cache-friendly.

### Ping-Pong Store Pattern

Fixed even uniform iterations use ping-pong mode:

```text
iteration 0: Pi -> Pibar scratch
iteration 1: Pibar scratch -> Pi
...
final: recompute/store valid Pibar rows if needed
```

This removes the old per-iteration `Pi_new -> Pi[wave]` device-to-device copy
traffic.  The tradeoff is a final `_wave_pibar_uniform_parent_kernel` launch
for rows whose final Pibar is needed.

For large fp32 likelihood profiles, ping-pong already solved the major D2D copy
problem.  Remaining work is final Pibar liveness and the wave-step algorithm
itself.

### DTS

The standard DTS kernel reads child clade rows:

```text
Pi[left, s], Pi[right, s], Pibar[left, s], Pibar[right, s]
Pi[left, child1[s]], Pi[right, child2[s]]
Pi[right, child1[s]], Pi[left, child2[s]]
```

Within a child row, same-species loads are coalesced.  Across split rows,
`left` and `right` are irregular clade indices.  The child-species loads are
also species-tree gathers.

The parent-reduced DTS path avoids materializing full `[n_splits, S]` when only
`[W, S]` parent rows are needed.  Remaining memory costs are:

- filling `out[W, S]` with `-inf`;
- allocating and writing `partial_max`/`partial_sum` for high-fanout ge2
  groups;
- reading those partial buffers in stage 2;
- tuning `tile_splits` for fanout and occupancy.

### Output and Chunk Memory

Full-output training needs full `Pi` and `Pibar`.  Likelihood-only/root-row
calls should prefer root-row output and `need_pibar=False` where backward is
not required.

The full pipeline profile shows chunk size is a memory-management parameter,
not just a launch-count parameter.  Larger chunks reduce chunk count, but can
increase memory/L2 pressure and peak allocation.  Chunk `75` was the best
reported 1000-family default on the profiled GPU, while chunk `100` was slower
and chunk `125` OOMed.

## Proposal 1: Pibar Liveness for Likelihood-Only Forward

### Idea

In ping-pong mode, final Pibar recomputation is required only for rows whose
Pibar will be read after the wave completes, or for rows that must be returned
or saved for backward.

Current code can skip final Pibar for an all-root wave when `need_pibar=False`.
That is conservative.  A finer liveness analysis could skip final Pibar rows
that have no future DTS consumers in likelihood-only forward.

### Implementation Sketch

Precompute, per wave, whether each clade row will be used as a left/right child
in any later split:

```text
pibar_live_after[row] = row appears in future split left/right arrays
```

Use that information in the final Pibar recompute after the last ping-pong
iteration:

- if no rows in the wave are live, skip `_wave_pibar_uniform_parent_kernel`;
- if all rows are live, keep the current dense wave kernel;
- if some rows are live, benchmark either:
  - a row-indexed Pibar kernel over live rows only;
  - a masked dense kernel if the live fraction is high.

### Applicability

This is primarily likelihood-only.  Training backward consumes saved forward
state, so full `Pibar` remains necessary unless backward is changed to
recompute or consume a compact saved representation.

### Benchmark Gate

Measure:

- total `_wave_pibar_uniform_parent_kernel` time removed;
- added metadata and masked/indexed-kernel overhead;
- NLL parity;
- peak memory and D2D copies.

Run at least 50-family, 150-family, and full chunked 1000-family likelihood
profiles.  The win should be visible in total forward time, not only in a
microbenchmark.

## Proposal 2: DTS Scratch, Fill, and Partial-Buffer Cleanup

### Idea

Parent-reduced DTS is already enabled, but the implementation still pays for
`out.fill_(-inf)` and per-wave partial-buffer allocation.  These are good
memory-management targets because they do not change the forward math.

### Implementation Sketch

1. Add metadata that says whether every parent row in a wave has at least one
   DTS split and no active-mask holes.
2. Skip `out.fill_(-inf)` only when every output row is guaranteed to be
   written by eq1 or ge2 kernels.
3. Reuse parent-reduced DTS scratch buffers across waves inside one forward
   call:
   - `out[W_max, S]`;
   - `partial_max[(n_groups * max_tiles)_max, S]`;
   - `partial_sum[(n_groups * max_tiles)_max, S]`.
4. Keep the current allocation path as a debug fallback.

### Risks

Skipping the fill is only safe when all rows are overwritten.  Rows without DTS
splits must remain `-inf` because the wave-step kernel reads `[W, S]` DTS rows
by wave-local row index.

Partial-buffer pooling reduces allocator overhead and memory churn, but not
the stage1/stage2 DRAM traffic itself.  The benchmark should separate allocator
time from kernel memory traffic.

### Benchmark Gate

Use Nsys to compare:

- `cudaMalloc`/allocator-visible events, if present;
- fill kernels;
- `_dts_parent_reduced_ge2_stage1_kernel`;
- `_dts_parent_reduced_ge2_stage2_kernel`;
- total forward interval.

Also sweep `GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS` over at least
`32`, `64`, and `128` for 50-family and 150-family chunks.

## Proposal 3: Species Locality / Coalescence Remap

### Idea

The remaining uncoalesced forward loads are topology-indexed:

- `sp_parent` parent walk in uniform Pibar;
- `sp_child1`/`sp_child2` child gathers in wave-step and DTS;
- split child clade row loads in DTS.

The species-tree arrays are small enough to be cache-hot, but `Pi[row, species]`
gathers can still miss coalescing.  A species-id remap may improve locality if
it makes parent/child/subtree accesses closer in memory.

### Implementation Sketch

Build an experimental preprocessing-only species permutation, then transform
all species-indexed tensors consistently:

- `sp_parent`, `sp_child1`, `sp_child2`;
- `log_pS`, `log_pD` when specieswise;
- `E`, `Ebar`, max-transfer rows;
- leaf species indices;
- returned root rows if caller expects original species order.

Candidate orders:

- current order as baseline;
- postorder or Euler order with subtrees contiguous;
- breadth-by-level order to improve parent-walk locality;
- child-pair-adjacent order to improve `sp_child1`/`sp_child2` gathers.

### Risks

This is invasive because all species-indexed data and tests must agree on the
permutation.  It should first be implemented as an opt-in preprocessing
experiment with exact parity and no kernel changes.

Earlier ancestor-table and CSR variants show that "more structured indexing"
does not automatically beat the current parent-pointer path.  The remap should
be promoted only if NCU shows better L2 behavior and total forward time drops.

### Benchmark Gate

For representative large waves, compare:

- `_wave_step_uniform_kernel` duration;
- L2 hit rate and L1/TEX scoreboard stalls;
- `_dts_fused_kernel` and parent-reduced DTS stage1 duration;
- NLL and `Pi` parity after inverse permutation.

Promote only if fp32 total forward improves by at least `3-5%` on a 150-family
chunk without increasing peak memory materially.

## Proposal 4: fp64-Specific Pibar Variants

### Idea

The fp32 parent-walk Pibar path is already high-occupancy and previous
alternatives regressed.  fp64 is different: profiles show scalar fp64 pipeline
pressure, and scratch prototypes found row-prefix Pibar promising for fp64
Pibar-only workloads.

Do not replace the fp32 path with row-prefix Pibar.  Keep any row-prefix or
mixed-internal experiment fp64-specific and explicitly gated.

### Implementation Sketch

Candidate A: fp64 final-Pibar-only replacement.

- Replace `_wave_pibar_uniform_parent_kernel` with a CUDA shared-memory
  row-prefix kernel only for `dtype=torch.float64`.
- Do not change `_wave_step_uniform_kernel` initially.
- Gate with `GPUREC_UNIFORM_FP64_FINAL_PIBAR_PREFIX=1`.

Candidate B: explicit approximate inference-only Pibar32.

- Compute Pibar normalization internals in fp32 and cast Pibar back to fp64.
- Expose as an explicit approximate inference mode, not as hidden behavior
  under `dtype=torch.float64`.
- Do not allow training/backward through this mode until finite-difference
  gradient checks pass.

Candidate C: full fused fp64 wave-step replacement.

- Only consider after Candidate A proves useful in full forward timing.
- The acceptance metric must be `_wave_step_uniform_kernel` bucket reduction,
  not Pibar-only microkernel speed.

### Benchmark Gate

For exact fp64:

- NLL parity must be exact within fp64 tolerance;
- gradients must pass existing fd/gradcheck tests if training is supported;
- full forward time must improve, not only final Pibar microbenchmarks.

For approximate fp64/Pibar32:

- expose an explicit flag and report NLL drift;
- inference-only until backward parity and finite-difference checks are added;
- document tolerances per workload.

## Proposal 5: Readiness-Aware DTS Overlap

### Idea

DTS can be DRAM-bound while wave-step is more balanced.  There may be overlap
opportunity if future-wave DTS becomes ready before the wave that consumes it.

The code already has overlap controls, but documented optimized profiles use a
serialized stream path by default.

### Implementation Sketch

Precompute the wave dependency for every DTS group:

```text
dts_ready_after[wave] = max(child_wave(left), child_wave(right))
```

Then schedule DTS as soon as dependencies are ready, bounded by a small pending
queue:

```text
while computing wave k:
    launch any future DTS j where dts_ready_after[j] <= k
    keep at most max_pending DTS jobs
before wave j self-loop:
    wait only for dts[j]
```

### Risks

Overlap can increase resident scratch memory and hurt cache locality.  It can
also add stream/event overhead if dependencies are too tight.

### Benchmark Gate

Use Nsys, not just CUDA events:

- check real overlap between DTS kernels and wave-step kernels;
- compare total forward time and peak memory;
- verify no added synchronization appears on the critical path.

Keep off by default unless total forward improves on large chunks.

## Proposal 6: CUDA Graph Capture for Fixed Schedules

### Idea

Fixed-6 uniform forward has a stable launch pattern per wave layout.  CUDA graph
capture may reduce CPU launch overhead, especially in repeated training steps.

### Priority

This is lower priority than Pibar/DTS memory work.  Existing profiles show
kernel time dominates launch overhead in the optimized path.  Graph capture is
most useful after the kernel set stabilizes.

### Benchmark Gate

Measure repeated same-layout forward calls:

- wall time outside CUDA events;
- CUDA API launch overhead;
- graph capture/replay setup cost amortization;
- compatibility with autograd saved tensors and chunking.

## Proposal 7: Chunk and Output Memory Policy

### Idea

Chunk size affects forward/backward throughput and peak memory.  The best
observed 1000-family setting was not the largest fitting chunk.  Output mode
also affects lifetime and memory pressure.

### Implementation Sketch

Add an auto-selection helper that estimates resident memory from:

```text
chunk_clades * S * sizeof(dtype) * saved_state_count
```

Then benchmark a small candidate set and choose the largest safe chunk only
when it is also empirically fast.  On the profiled GPU/dataset, this rule
should prefer chunk `75` over chunk `100`.

For likelihood-only API calls:

- prefer `return_root_rows=True`;
- use `need_pibar=False`;
- avoid full original-order `Pi` permutation unless the caller needs it.

### Benchmark Gate

Measure full 1000-family training and likelihood-only runs:

- total forward;
- total backward;
- peak memory;
- number of chunks;
- loss/gradient parity.

## Lower-Priority or Rejected Directions

These should not be repeated unchanged:

| Direction | Current decision |
|---|---|
| Replace fp32 parent walk with ancestor-table, CSR, linear, two-kernel, or sparse-mm Pibar | previously slower in fp32 forward |
| Search for custom forward atomics to remove | no optimized-path custom atomics found |
| Dense leaf masks in uniform fused path | keep compact leaf index on |
| Adaptive Pi convergence polling | fixed-6 removed host polling and is faster |
| Leaf/no-split special kernel | earlier NCU did not show an easy fp32 win |
| Tensor cores for current kernels | not applicable to scalar log-space reductions |

## Implementation Order

Recommended next order:

| Rank | Proposal | Reason |
|---:|---|---|
| 1 | DTS scratch/fill/partial-buffer cleanup | localized memory-management change; math-preserving |
| 2 | Pibar liveness for likelihood-only forward | can remove final Pibar work where rows are dead |
| 3 | Species locality remap experiment | attacks uncoalesced gathers without changing kernels first |
| 4 | fp64 final-Pibar or Pibar32 inference experiments | promising only for fp64 or explicit approximate inference |
| 5 | readiness-aware DTS overlap | possible but memory-pressure sensitive |
| 6 | chunk/output auto-policy | improves robustness and full-pipeline memory behavior |
| 7 | CUDA graph capture | useful after kernel structure stabilizes |

## Benchmark Checklist

Correctness:

```bash
pytest -q tests/kernels/test_wave_step_uniform_forward_kernel.py \
          tests/kernels/test_dts_fused_kernel.py \
          tests/unit/test_uniform_forward_scheduling.py \
          tests/unit/test_genewise_wave.py
```

For changes that alter saved forward state or training behavior:

```bash
pytest -q tests/gradients/test_autograd_bridge.py \
          tests/gradients/test_fd_all_modes.py::test_analytic_matches_fd
```

Forward timing:

```bash
.venv/bin/python profiling/bench_uniform_forward_parent_dts.py \
  --dataset tests/data/test_trees_1000 \
  --fams 50 \
  --fixed-iters 6 \
  --dtype fp32 \
  --no-need-pibar \
  --root-rows

.venv/bin/python profiling/bench_uniform_forward_parent_dts.py \
  --dataset tests/data/test_trees_1000 \
  --fams 150 \
  --fixed-iters 6 \
  --dtype fp32 \
  --no-need-pibar \
  --root-rows
```

Full pipeline:

```bash
.venv/bin/python profiling/bench_uniform_forward_backward_pipeline.py \
  --dataset tests/data/test_trees_1000 \
  --chunk-size 75 \
  --fixed-iters 6 \
  --dtype fp32
```

Nsight Systems should be used for launch counts, memory copies, stream overlap,
fill kernels, and kernel buckets.  Nsight Compute should be used for the
representative largest wave-step launch, final Pibar launch, and high-fanout
DTS stage1 launch.

Promotion criteria:

- exact NLL parity for exact-mode changes;
- gradient parity for any training path change;
- no meaningful peak-memory regression;
- forward interval improvement on at least 50-family and 150-family workloads;
- full 1000-family chunked run does not regress.

# Context For `gpurec/core/kernels/wave_backward.py`

This note is meant to accompany `gpurec/core/kernels/wave_backward.py` for GPU
kernel review.  It focuses on the backward-pass dataflow and on why the file
currently uses atomic additions in several places.

## Short Version

`wave_backward.py` contains the retained Triton CUDA kernels used by
`Pi_wave_backward()` in `gpurec/core/backward.py`.  The kernels explicitly
compute the VJP of the wave-ordered fixed-point recurrence for the clade/species
probability tensor `Pi`.

The main performance question is not "are atomics mathematically necessary?"
but "are atomics the best reduction strategy for this layout?"  Several atomic
sites are necessary for correctness under the current one-program-per-row or
one-program-per-split launch geometry because many programs can write the same
`accumulated_rhs[child_clade, species]` or the same parameter-gradient cell.
They could be replaced by staged or segmented reductions, but that would require
additional metadata, memory, and launches.

The most important shared output is:

```text
accumulated_rhs: [C, S]
```

It is the wave-ordered adjoint of `Pi`.  Backward processes waves from roots
toward leaves.  Each wave reads its own row block from `accumulated_rhs`, solves
the self-loop adjoint for that wave, then scatters split contributions into
child rows of `accumulated_rhs` for earlier waves to consume.

## Where This File Sits

The high-level gradient flow is:

```text
public model/autograd API
  -> gpurec/optimization/implicit_grad.py::implicit_grad_loglik_vjp_wave()
     -> gpurec/core/backward.py::Pi_wave_backward()
        -> gpurec/core/kernels/wave_backward.py
     -> _e_adjoint_and_theta_vjp()
```

`Pi_wave_backward()` is `torch.no_grad()` code.  It does not rely on PyTorch
autograd for the `Pi` recurrence.  It returns explicit gradient buffers:

```text
v_Pi
grad_E
grad_Ebar
grad_E_s1
grad_E_s2
grad_log_pD
grad_log_pS
grad_max_transfer_mat
```

The E-adjoint and final theta VJP are completed later in
`_e_adjoint_and_theta_vjp()`.

## Runtime Constraints

The retained production path currently assumes:

- CUDA tensors only.
- `float32` or `float64` for `Pi_wave_backward()`.
- `S > 256` species nodes.
- Wave-ordered clade rows.  Each wave is a contiguous `Pi[ws:we]` row block.
- Compact species topology is available:
  `compact_level_ptr`, `compact_level_parents`, `compact_level_child1`,
  `compact_level_child2`.
- `uniform_pibar_row_max` was saved by the forward pass.

The code supports shared/global and family-indexed layouts.  Shared mode is
internally normalized to one family row (`G == 1`) in `Pi_wave_backward()`.

## Key Tensor Shapes

Common symbols:

```text
C      total clades in the current resident batch/chunk
S      species-tree nodes
K      number of waves
W      rows in one wave
n_ws   number of split split rows in one wave
G      number of family parameter rows
```

Important tensors:

```text
Pi_star_wave           [C, S]
Pibar_star_wave        [C, S]
uniform_pibar_row_max  [C]
accumulated_rhs        [C, S]
rhs_k                 [W, S] view into accumulated_rhs[ws:we]
v_k                   [W, S] wave-local solved adjoint
```

The wave metadata gives split rows for one wave:

```text
sl          [n_ws] child clade id, left side
sr          [n_ws] child clade id, right side
reduce_idx  [n_ws] wave-local parent row id
wlsp        [n_ws] log split probability
```

The metadata is grouped primarily by parent/reduce index for forward DTS
reduction, not by child destination for backward scatter.

## Forward Recurrence Being Differentiated

Forward uses `gpurec/core/kernels/wave_step.py`.  For each wave, it combines:

1. A same-row self-loop update over species.
2. Optional split DTS terms from child clade rows.
3. Optional leaf terms.
4. Uniform `Pibar` terms based on row sums minus ancestor-subtree sums.

Forward stores:

```text
Pi_wave_ordered
Pibar_wave_ordered
uniform_pibar_row_max
```

Those saved tensors are the inputs to `Pi_wave_backward()`.

## Backward Driver Flow

`Pi_wave_backward()` initializes `accumulated_rhs` from the NLL derivative at
root clades:

```text
accumulated_rhs[root_rows, :] = root NLL adjoint
```

Then it walks waves in reverse order:

```text
for wave k from K - 1 down to 0:
    rhs_k = accumulated_rhs[ws:we]
    active_mask = absmax(rhs_k) >= pruning_threshold

    if wave has split splits:
        recompute dts_r for the wave

    v_k, self_loop_param_contribs = compute_wave_adjoint(...)

    reduce or directly accumulate self-loop parameter gradients

    if wave has split splits:
        accumulate_split_dts_vjp(...)
        accumulate_split_pibar_vjp(...)
```

After this, child rows in `accumulated_rhs` have received all adjoints from the
current wave, and the next reverse wave can read them.

## Kernel Groups In `wave_backward.py`

### Active Mask

`compute_active_wave_rows_from_adjoint()` builds a `[W]` boolean mask by reducing
`abs(rhs_k[w, :])`.  It is used to skip rows whose adjoint is effectively zero.

This helper accepts bf16 for standalone experiments, but the public
`Pi_wave_backward()` path rejects bf16 before reaching it.

### Self-Loop Backward

Public wrapper:

```text
compute_wave_adjoint(...)
```

Implementation:

```text
_self_loop_coefficients_kernel
_self_loop_adjoint_update_kernel
_self_loop_parameter_gradient_kernel
```

The self-loop is an implicit fixed-point recurrence inside one clade row.  The
backward path approximates the inverse with a truncated Neumann series:

```text
v_k = rhs + J^T rhs + (J^T)^2 rhs + ...
```

The precompute kernel builds per-row/per-species coefficients.  The Jt kernel
applies one Neumann term.  Python launches the Jt kernel once per requested
Neumann term.  The parameter-store kernel computes VJP contributions for
self-loop parameters.

Important property: each self-loop Jt program owns one wave row.  Inside that
row it performs species-tree reductions with normal stores and barriers, not
global atomics.

### Cross-Clade DTS Backward

Public wrapper:

```text
accumulate_split_dts_vjp(...)
```

Implementation:

```text
_split_dts_vjp_kernel
```

This runs one Triton program per split row.  It:

- reads parent adjoint `v_k[reduce_idx[i], :]`;
- computes the VJP of the five DTS terms for that split;
- adds direct child `Pi` adjoints into `accumulated_rhs`;
- stages `Pibar` side adjoints as `pibar_ud` and `pibar_A`;
- accumulates DTS parameter gradients.

The direct child `Pi` adjoints are the first major atomic site.

### Uniform Pibar VJP From Staged `u_d`

Public wrapper:

```text
accumulate_split_pibar_vjp(...)
```

Implementation:

```text
_pibar_vjp_kernel
```

This runs one program per split side, so the grid has `2 * n_ws` programs.  Each
program walks the compact species topology to reduce the staged `pibar_ud` row,
then atomically adds final contributions into the child row of
`accumulated_rhs`.

This is the second major `accumulated_rhs` atomic site.

## Atomic Sites And What They Mean

### 1. Self-Loop Parameter Gradients

In `_self_loop_parameter_gradient_kernel`, when `ACCUM_GRADS=True`, programs
atomic-add directly into gradient tensors such as:

```text
grad_log_pD
grad_log_pS
grad_E
grad_Ebar
grad_E_s1
grad_E_s2
grad_max_transfer_mat
```

Why atomics are needed in this launch geometry:

- There is one program per wave row.
- Every row contributes to the same shared `[S]` gradient vectors or scalar
  gradients in shared mode.
- Without atomics, concurrent programs would race on the same gradient cells.

Atomics are not mathematically required.  Existing code still supports the
alternative materialized path where the kernel stores per-row contribution
tensors `aw0`, `aw1`, `aw2`, `aw345`, `aw3`, `aw4`, and Python/PyTorch reduces
them afterward.  That avoids Triton atomics but costs full `[W, S]` scratch
traffic and reduction launches.

Potential alternative:

- Stage tiled reductions as `[tiles, S]` or `[tiles]`, then run a second
  reduction kernel.  This is likely the most plausible non-atomic replacement
  for dense shared gradients.

### 2. Cross-Clade Direct Child `Pi` Adjoint Accumulation

In `_split_dts_vjp_kernel`, the direct `Pi` adjoints write to:

```text
accumulated_rhs[sl[i], s] += ...
accumulated_rhs[sr[i], s] += ...
```

and, for the speciation terms:

```text
accumulated_rhs[sl[i], child_species] += ...
accumulated_rhs[sr[i], child_species] += ...
```

Why atomics are needed in this launch geometry:

- There is one program per split row.
- Multiple split rows can target the same child clade row in the same launch.
- The current wave metadata does not guarantee one writer per
  `(child_clade, species)` destination.

There is a `USE_ATOMICS=False` branch in the kernel, but it is only correct when
the caller can prove unique write destinations.  `Pi_wave_backward()` does not
prove that and uses the default atomic path.

A local metadata probe on `tests/data/test_trees_1000`, first 100 families,
showed the collision is real for a representative fixture:

```text
families                  100
S                         1999
waves                     201
split_waves               181
splits                    793940
waves_with_child_conflict 101
extra_child_writers       152710
max_child_multiplicity    2
waves_with_parent_ge2     25
max_parent_split_multiplicity 4577
```

Those numbers are workload-specific, but they show that non-atomic direct
stores would be unsafe under ordinary metadata.

Potential alternatives:

- Materialize per-split child contributions and reduce by child row with
  `index_add` or a custom segmented reduction.  A comment in the kernel notes
  this used to be done with materialized `grad_Pi_l/grad_Pi_r` plus PyTorch
  `index_add_`.
- Add reverse adjacency metadata grouped by child row and launch one program per
  child row or child-row/species tile.
- Split launches so each launch has unique child destinations.  This would
  reduce races but likely increases wave count and launch overhead.

### 3. Cross-Clade DTS Parameter Gradients

The DTS kernel also atomically accumulates:

```text
grad_log_pD
grad_log_pS
grad_max_transfer_mat
```

Layouts include shared scalar, shared species vector, family scalar, and
family/species.  The scalar and vector layouts are naturally many-writer
reductions.  Family layouts may reduce contention if many families are active,
but they still need a reduction strategy.

There is already one staged-reduction special case:

```text
stage_max_transfer_gradient_by_tile=True
```

For eligible shared `[S]` `grad_max_transfer_mat`, the DTS kernel atomically accumulates into
`grad_max_transfer_tiles[tile, s]`, then `_dts_max_transfer_gradient_kernel` reduces
tiles into `grad_max_transfer_mat`.  That pattern is a useful precedent for replacing
other high-contention parameter atomics.

Potential alternatives:

- Extend two-stage partial reduction to `grad_log_pD` and `grad_log_pS` for
  shared species-vector layouts.
- For family-indexed layouts, stage by `(family_tile, species)` or sort/group
  split rows by family before reduction.
- For scalar layouts, accumulate per program to a scalar and reduce per wave or
  per block rather than one atomic per split.

### 4. Uniform Pibar VJP Final Add

In `_pibar_vjp_kernel`, each split-side
program eventually writes:

```text
accumulated_rhs[child_clade, s] += p_prime * (A - subtree_sum)
```

Why atomics are needed in this launch geometry:

- There is one program per split side, not one program per child row.
- Multiple split sides can contribute to the same child row.
- This is another scatter-add into `accumulated_rhs`.

Potential alternatives:

- Keep `pibar_ud` materialized, but reduce by child row in a second stage.
- Build child-grouped metadata and process all split sides for a child row in
  one program or one cooperative group.
- Change the staged `pibar_ud` layout from split-side-major to child-major if
  the downstream access pattern dominates.

Performance notes in `docs/hogenom-ccp-performance-log.md` indicate this kernel
has historically looked memory/coalescing limited rather than occupancy limited.
That points toward structural layout/reduction changes rather than simple
launch-option retuning.

## Correctness Invariants To Preserve

Any non-atomic rewrite must preserve these ordering constraints:

1. A wave may only read `accumulated_rhs[ws:we]` after all later/rootward waves
   have added into it.
2. Within a wave, self-loop backward reads `rhs_k` as the incoming parent-row
   adjoint and writes parameter gradients plus `v_k`.
3. Split DTS and Pibar VJP for the current wave add into child rows, not
   into the current `rhs_k` row block for reuse by the same wave.
4. If pruning is enabled, inactive parent rows must contribute zeros, and
   staged Pibar side rows may be skipped only when their side-active wave-row proves
   they are zero.
5. The accumulation order is not numerically deterministic with atomics.  A
   staged reduction may slightly change fp32 results.  Existing tests use fp32
   tolerances for that reason.

## What Is Worth Reviewing

High-value GPU review questions:

1. Are the `accumulated_rhs` scatter-adds best handled as atomics, or should
   split metadata include child-destination grouping for a staged reduction?
2. Is the Pibar VJP better launched per split side, as it is now, or per child
   row with gathered split-side inputs?
3. Should DTS parameter gradients use the same two-stage tiled pattern already
   used for `grad_max_transfer_mat[S]`?
4. For self-loop parameter gradients, is direct atomic accumulation actually
   faster than materialized `aw*` plus reductions on the target workloads?
5. Is there a good CUDA-specific implementation strategy that Triton cannot
   express cleanly, especially for child-grouped reductions or shared-memory
   species-tree walks?

## Suggested Experimental Path

A safe path for trying to remove atomics would be:

1. Start with one atomic family at a time.  Do not change self-loop, DTS direct
   child RHS, DTS parameter gradients, and Pibar VJP together.
2. Add an opt-in path or direct benchmark hook so the existing kernels remain
   the correctness baseline.
3. For `accumulated_rhs`, first build a verifier that checks destination
   multiplicity per wave.  This prevents accidentally using the unsafe
   non-atomic branch on colliding metadata.
4. Prototype a staged reduction using extra metadata for either:
   - DTS direct child `Pi` contributions, or
   - Pibar final child-row contributions.
5. Compare both end-to-end time and GPU kernel buckets.  Saving atomics in one
   kernel can easily lose if it adds global-memory traffic or extra launches.

## Relevant Files

```text
gpurec/core/backward.py
    Driver for Pi_wave_backward(), root RHS initialization, wave loop,
    self-loop calls, DTS calls, Pibar VJP calls, and final result dict.

gpurec/core/kernels/wave_backward.py
    Triton kernels under review.

gpurec/core/kernels/wave_step.py
    Forward wave kernels whose saved state is differentiated here.

gpurec/core/kernels/dts_fused.py
    Forward split DTS parent-reduced recompute used by backward.

gpurec/core/batching.py
crates/gpurec-preprocess/src/layout.rs
    Wave-layout and split metadata construction.

gpurec/core/species.py
    Compact species topology creation/caching for wave kernels.

gpurec/optimization/implicit_grad.py
    Caller that consumes Pi_wave_backward() outputs and completes the gradient.

profiling/bench_uniform_forward_backward_pipeline.py
    Maintained benchmark harness for global/uniform forward plus backward.
```


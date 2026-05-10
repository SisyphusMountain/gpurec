# Forward + Backward Full Pipeline Plan

Date: 2026-05-05.

This plan describes the target implementation for the fastest production
pipeline that evaluates the reconciliation likelihood and, when requested, its
gradient.  The scope is the optimized `pibar_mode="uniform"` path for global,
genewise, and specieswise parameterizations.  Pairwise transfer is outside this
plan because it does not share the same uniform `Pibar` algebra and does not
currently have a comparable optimized backward path.

## Target Shape

The production pipeline should have two explicit modes:

| Mode | Output | State kept | Intended use |
|---|---|---|---|
| likelihood-only | root rows or reduced loss | minimal; no full backward tensors | inference, line-search scoring, benchmarking |
| training | loss and gradients | chunk-local `Pi`, `Pibar`, row stats, masks, compact VJP state | optimization |

The key rule is that the likelihood-only path must stay root-row/minimal-state,
while the training path must keep the full forward state for only the current
resident family chunk and release it immediately after its reverse pass.

## High-Level Execution

```python
def loss_and_grad(theta, families, mode, need_grad):
    static = get_or_build_static_state(families, mode)
    chunks = scheduler.plan(static, need_grad=need_grad, memory_budget=budget)

    params = extract_uniform_params(theta, mode)
    E_state = solve_E_fixed_point(params, static.species_helpers)

    total_loss = 0
    grad_accum = init_grad_accumulators(theta, E_state)

    for chunk in chunks:
        state = forward_chunk(
            chunk,
            params=params,
            E_state=E_state,
            save_for_backward=need_grad,
            root_rows_only=not need_grad,
        )
        total_loss += reduce_root_likelihood(state.root_rows, E_state.E)

        if need_grad:
            backward_chunk(
                state,
                params=params,
                E_state=E_state,
                grad_accum=grad_accum,
            )
            release_chunk_state(state)

    if need_grad:
        apply_E_adjoint(params, E_state, grad_accum)
        return total_loss, grad_accum.theta
    return total_loss
```

All scalar results should remain on device until the end of the call.  The
pipeline should not perform per-wave or per-chunk `.item()` calls in the hot
path.

## Static Preprocessing

Static work should happen once per dataset/model state:

1. Parse species and gene trees.
2. Build per-family clade DAGs, CCP split arrays, leaf indices, root ids, and
   family metadata.
3. Build uniform species helpers:
   - dense ancestor matrix for preprocessing;
   - compact/sparse ancestor structures needed by kernels;
   - `unnorm_row_max`.
4. Cache light preprocessing artifacts on disk.
5. Build memory-aware chunk plans and wave layouts for the supported chunk
   sizes.

The default preprocessing mode should remain the light mode already promoted:
do not build unused clade-inclusion DAGs for the forward/backward wave
scheduler.  Layout construction should not be repeated inside every optimizer
step unless the family set or scheduling policy changes.

## Scheduling Policy

Use separate default policies for likelihood-only and training:

| Workload | Default policy on 24 GB RTX 4090 | Rationale |
|---|---|---|
| global/uniform likelihood-only, 1000 trees | `family_chunk_size=150`, `max_wave_size=32768` | stable near-optimum; `125/65536` is lower-memory but not a stable speed win |
| genewise likelihood-only, 1000 trees | same shape as global, with family-indexed constants | current forward is within about `0.4%` of global |
| specieswise likelihood-only, 1000 trees | `family_chunk_size=150` | forward is essentially global-speed |
| genewise training | chunk 50 default, chunk 100 high-memory option | chunk 100 is faster, chunk 50 has safer memory margin |
| specieswise training | start conservative; optimize backward first | current specieswise backward is still generic-heavy |

Do not promote split-row, fanout, or DTS-overlap scheduling as defaults:

- split-row caps isolate high-fanout waves but the best observed gain was about
  `0.4%`, inside timing noise;
- fanout caps create too many waves and are slower;
- DTS overlap creates real stream overlap, but Nsight showed the added
  scheduling, event, and allocator overhead makes total time worse.

The scheduler should still record diagnostic counters:

```text
chunks, total waves, max wave rows, split rows, max phase-3 split rows,
family locality, peak allocation estimate
```

These counters are useful for detecting when a new dataset falls outside the
profiled envelope.

## Forward Chunk Plan

Each forward chunk should run the current optimized uniform path:

1. Initialize leaf rows through compact leaf indices.
2. Traverse waves in topological order.
3. For each wave:
   - compute parent-reduced DTS for split rows;
   - run fixed 6 local Pi iterations;
   - use ping-pong buffers for even fixed iterations;
   - compute/store `Pibar` required by either the next wave or backward;
   - store row maxima/statistics used by compact Pibar VJP.
4. Return only root rows in likelihood-only mode.
5. Return full saved tensors in training mode.

The forward path should keep these defaults:

```text
GPUREC_UNIFORM_PINGPONG=1
GPUREC_FORWARD_PARENT_REDUCED_DTS=1
GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY=1
GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL=tiled
GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS=64
fixed_iters_Pi=6
need_pibar=False for likelihood-only
need_pibar=True/full saved state for training
```

Mode-specific forward handling:

| Mode | Parameter access |
|---|---|
| global | scalar/shared `log_pD`, `log_pS`, `log_pL`, `mt` |
| genewise | family-indexed constants and family-indexed DTS parameters |
| specieswise | direct `[S]` species vectors; no family-indexed materialization |

The implementation should not materialize `[W, S]` constants or
`[split_rows, S]` parameter slabs.  Those materializations were the main
genewise forward regressions and have already been removed.

## Saved State Contract

For training, define a compact explicit saved-state object per chunk:

```text
Pi or ping-pong-resolved Pi        [C, S]
Pibar                              [C, S]
root rows / root ids
wave layout and split metadata
row maxima and Pibar row stats
family ids / family offsets
leaf index metadata
parameter layout tags
```

The saved state must be exactly what the fused backward kernels expect.  Avoid
falling back to generic PyTorch reconstruction paths.  The benchmark wrappers
should keep strict optimized guards that print or fail on:

```text
full_saved_tensors_for_backward
fused_uniform_backward
fused_genewise_backward
fused_specieswise_backward
kernelized_backward_dts
compact_tree_pibar_vjp
generic_pytorch_fallback == 0
```

## Backward Chunk Plan

Backward should traverse the chunk waves in reverse topological order:

```text
seed dPi[root rows]
for wave in reversed(waves):
    build active/pruning mask on device
    recompute or read compact DTS/Pibar coefficients
    apply fused self-loop VJP
    accumulate DTS child adjoints and parameter-gradient partials
    run compact/tree uniform Pibar VJP
accumulate chunk gradients into global device buffers
```

The default uniform backward path should use the promoted fused components:

| Component | Required implementation |
|---|---|
| active mask | Triton/kernelized active mask |
| self-loop VJP | `_wave_backward_uniform_kernel` with current tuned block size |
| topology | int32 topology where valid |
| speciation in self-loop | parent-gather path |
| row statistics | reuse forward `Pibar` row maxima/statistics |
| DTS backward | merged species-loop accumulation; scalar parameters stay on device |
| DTS reductions | direct scalar accumulation where applicable |
| Pibar VJP | compact staged `u_d` / tree VJP path, not dense generic materialization |

Mode-specific backward handling:

| Mode | Gradient target |
|---|---|
| global | scalar `dtheta[3]`; direct reductions are safe |
| genewise | `dtheta[G, 3]`; use family-indexed fused self-loop, DTS, and Pibar VJP |
| specieswise | `dtheta[S, 3]`; use fused specieswise reductions, avoid generic ATen/cusparse path |

Specieswise backward is the largest remaining gap.  The current forward is
already global-speed, but specieswise backward profiles show generic PyTorch,
cuSPARSE, indexing/scatter/gather, cat materialization, and many syncs.  The
pipeline is not complete until specieswise backward uses the same fused
self-loop/DTS/Pibar structure as global and genewise, with species-indexed
gradient reductions.

## E Fixed Point And E Adjoint

The E fixed point should be solved once per optimizer step for global and
specieswise modes, and batched over families for genewise mode.  The E state is
small compared with chunk-local `Pi/Pibar`, so it can remain resident for the
whole pipeline.

During chunk backward, accumulate all contributions to:

```text
dE
dEbar
d log_pD, d log_pS, d log_pL, d mt
```

After all chunks are processed, run the E adjoint/VJP once using the accumulated
device buffers.  This avoids re-solving or synchronizing E gradients per chunk.

## Memory Management

Use a device buffer pool keyed by:

```text
(dtype, S, max_C_in_chunk, max_wave_rows, max_split_rows, mode, need_grad)
```

The pool should own reusable tensors for:

```text
Pi, Pibar, dPi/rhs, active masks, DTS scratch, compact Pibar VJP scratch,
parameter partials, root rows
```

The training loop must release chunk-local state before moving to the next
chunk.  It should never keep the full 1000-family `Pi/Pibar/dPi` state resident.

Approximate scheduling objective:

```text
saved forward state
+ backward adjoints
+ DTS/Pibar VJP scratch
+ gradient partials
+ static metadata
+ safety margin
< memory budget
```

For likelihood-only mode, the scheduler can use larger chunks because full
backward tensors are not needed.  For training, prefer a slightly smaller
default chunk that leaves allocator and temporary-buffer headroom.

## CUDA Graphs And Launch Overhead

CUDA graphs are not the first optimization for the current likelihood-only
forward, because large chunks are dominated by Triton kernel time rather than
Python enqueue overhead.  They become more attractive for training and
specieswise backward because those paths still have many small launches.

Graph capture should be attempted only after:

1. static chunk layouts are reused;
2. buffer shapes are stable for a chunk bucket;
3. all hot-path scalar decisions stay on device;
4. no allocator calls happen inside the captured interval.

A practical design is one graph per chunk-shape bucket:

```text
(mode, dtype, S, C_bucket, wave_count, max_wave_rows, max_split_rows)
```

Use graph replay for optimizer iterations after the first warmup compile/capture
step.

## Triton Autotuning

Use offline autotuning, not production-time `@triton.autotune`, for the shared
global/genewise/specieswise kernels.

Initial grid:

```text
forward wave step:
    BLOCK_S in {128, 256, 512}
    num_warps in {4, 8}

backward self-loop:
    BLOCK_S in {128, 256, 512}
    num_warps in {4, 8, 16}

parent-reduced DTS:
    TILE_SPLITS in {32, 64, 128, 256}

DTS backward accumulation:
    BLOCK_S in {128, 256, 512}
    num_warps in {4, 8}

compact/tree Pibar VJP:
    BLOCK_S in {128, 256, 512}
    num_warps in {4, 8}
```

Cache the chosen parameters by:

```text
(kernel, mode, dtype, S, wave_size_bucket, fanout_bucket, pruning_bucket)
```

Acceptance requires Nsys/NCU confirmation that the selected config improves the
intended bucket without moving time into another bucket or increasing launch
count enough to erase the gain.

## Correctness Matrix

Every promoted implementation path must pass:

| Check | Purpose |
|---|---|
| high-iteration fp64 parity | proves mathematically equivalent paths agree |
| fixed-6 fp32 parity | validates production numerical envelope |
| gradcheck / finite differences on small trees | checks backward formulas |
| chunk-size invariance | proves streaming chunks do not change results |
| genewise vs individual global trees | validates family-indexed parameters |
| constant specieswise vs global | validates specieswise semantics |
| AleRax specieswise fixture | validates real specieswise parameter files |
| strict optimized-path subprocess checks | prevents silent fallback to PyTorch |

Do not relax high-iteration fp64 tests when two implementations should be
identical.  For fp32, tolerate only the small order-dependent differences that
remain after fp64 and finite-difference checks pass.

## Profiling Gates

Use Nsight Systems and Nsight Compute for promotion decisions.  The default
Torch profiler is not sufficient for this pipeline.

For every major change, collect:

```text
end-to-end wall/event time
GPU kernel-sum time
kernel bucket table
launch count
CUDA API synchronization count
peak allocation
top NCU kernels with SM/L1/L2/DRAM/occupancy/stall metrics
loss and gradient parity
```

Minimum benchmark matrix:

| Dataset slice | Use |
|---|---|
| 3 families | quick correctness and regression smoke |
| 10 families | gradcheck-adjacent performance smoke |
| 50 families | memory-safe backward profiling |
| 100 families | high-memory resident backward comparison |
| 1000 families | full streaming throughput and memory policy |

## Implementation Phases

### Phase 0: Contracts And Baselines

- Add one supported benchmark entrypoint for full forward+backward streaming.
- Print the active optimized-path contract for global, genewise, and
  specieswise modes.
- Record current 50/100/1000-family baselines for likelihood-only and training.

### Phase 1: Unified Chunk Scheduler

- Move chunk planning into a reusable scheduler object.
- Estimate memory separately for inference and training.
- Preserve current family locality.
- Keep split/fanout counters as diagnostics only.

### Phase 2: Unified Forward Executor

- Route global, genewise, and specieswise through one optimized uniform forward
  executor.
- Keep root-row output for inference.
- Keep full saved tensors only for training.
- Enforce no `[W,S]` or `[split_rows,S]` parameter materialization.

### Phase 3: Saved-State And Buffer Pool

- Define a typed saved-state object for backward.
- Add reusable buffer pools for forward and backward scratch.
- Remove allocator calls from repeated chunk execution where practical.

### Phase 4: Unified Backward Executor

- Route global and genewise through the existing fused self-loop/DTS/Pibar VJP
  kernels.
- Port specieswise backward to the same fused structure with species-indexed
  reductions.
- Keep all reductions and scalar parameters on device.

### Phase 5: E Adjoint Integration

- Accumulate chunk-local E/theta contributions on device.
- Run the E adjoint once after all chunks.
- Validate global/genewise/specieswise gradients against existing references.

### Phase 6: CUDA Graph And Autotune Pass

- Capture stable chunk buckets after allocator cleanup.
- Add offline Triton autotune scripts and cache.
- Promote only settings that improve full forward+backward time, not just one
  isolated kernel.

### Phase 7: Production API

- Expose one high-level path for:
  - `loss(theta)`;
  - `loss_and_grad(theta)`;
  - per-family likelihoods when requested.
- Make strict optimized guards available in debug/profile mode.
- Keep environment overrides for experiments, but make the measured defaults
  explicit in docs and benchmark output.

## Expected Outcome

The target is not a radically different algorithm.  The fastest path is the
current wave dynamic program, streamed over memory-safe chunks, with all generic
backward pieces removed and with no full-dataset dense state kept resident.

Expected performance direction:

- likelihood-only global/genewise/specieswise should stay near the current
  `~2.3 s` 1000-family envelope on the RTX 4090;
- genewise training should use the chunked optimized path already measured at
  about `9-11 s` depending on chunk size and memory margin;
- specieswise training should improve substantially once its backward path stops
  using generic PyTorch/cuSPARSE-heavy work;
- remaining gains after that are likely incremental and profile-driven:
  autotuned Triton launch parameters, fewer allocator calls, graph replay for
  stable chunks, and further fusion of the split-side backward buckets.

## Current Uniform Round

The current global/uniform implementation, benchmark, and profile report is in:

```text
docs/uniform-forward-backward-full-pipeline-profile.md
```

That report supersedes the worker draft below for this uniform-focused round.

## Superseded Worker Draft: Specieswise Uniform Forward + Backward

Date: 2026-05-05.

This round measured the current specieswise `pibar_mode="uniform"` full
forward+backward path through a profiling-only harness:

```text
profiling/specieswise_worker3/bench_specieswise_uniform.py
```

The harness has two engines:

| Engine | Status | Notes |
|---|---|---|
| `autograd` | unsupported for large specieswise training | resident public model path OOMed on 50 families |
| `chunked` | supported profiling path for this round | streams family chunks, runs full forward and backward per chunk, and accumulates gradients |

The resident OOM happened in the generic specieswise backward while allocating
another `1.57 GiB`; PyTorch already had about `21.87 GiB` allocated on a
`23.51 GiB` RTX 4090.  This validates the training scheduler requirement:
specieswise training must stream chunks and release chunk-local state.

### Active Path

The active optimized pieces for the chunked specieswise harness were:

```text
GPUREC_UNIFORM_PINGPONG=1
GPUREC_FORWARD_PARENT_REDUCED_DTS=1
GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL=tiled
GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY=1
GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS=0
GPUREC_FORWARD_TOPOLOGY_INT32=1
GPUREC_FORWARD_LEAF_INDEX=1
GPUREC_KERNELIZED_BACKWARD_DTS=1
GPUREC_FUSED_DTS_BACKWARD_ACCUM=1
GPUREC_FUSED_CROSS_PIBAR_VJP=1
GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL=tree
GPUREC_BACKWARD_LEAF_INDEX=1
GPUREC_FUSED_WAVE_PARAM_ACCUM=1
GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES=1
GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS=1
GPUREC_DTS_GRAD_MT_TWO_STAGE=1
```

The harness explicitly forced:

```text
GPUREC_FUSED_UNIFORM_BACKWARD=0
```

for `engine=chunked` specieswise runs.  The current fused uniform self-loop gate
supports the scalar/global path and the family-indexed genewise path; it does
not yet support the species-indexed `[S, 3]` gradient target.  Therefore the
measured specieswise backward is still dominated by generic PyTorch/cuSPARSE
work even though several DTS and Pibar subpaths are fused.

The implementation shape used by the harness is:

```python
E_state = E_fixed_point(...)
grad_theta = zeros_like(theta)
nll = 0.0

for family_chunk in chunks(family_batch_size):
    wave_layout = build_chunk_layout(family_chunk)
    pi_out = Pi_wave_forward(
        wave_layout,
        E_state,
        fixed_iters=6,
        pibar_mode="uniform",
        return_original=False,
    )
    nll += compute_log_likelihood(
        pi_out["Pi_wave_ordered"],
        E_state["E"],
        wave_layout["root_clade_ids"],
    ).sum()

    grad_theta_chunk, stats = implicit_grad_loglik_vjp_wave(
        wave_layout,
        Pi_star_wave=pi_out["Pi_wave_ordered"],
        Pibar_star_wave=pi_out["Pibar_wave_ordered"],
        specieswise=True,
        pibar_mode="uniform",
        uniform_pibar_row_max=pi_out.get("uniform_pibar_row_max"),
    )
    grad_theta += grad_theta_chunk
    release_chunk_state()
```

The current script then divides `grad_theta` by `n_chunks`.  That makes the
reported `grad_inf` chunk-size dependent while the reported `nll` is summed.
The benchmark timings are still useful, but the gradient magnitude is not a
valid chunk-size parity metric until the harness reports an explicitly chosen
sum or mean loss convention.

### Benchmark Table

Dataset: `tests/data/test_trees_1000`.  Device: `NVIDIA GeForce RTX 4090`.
Software: `torch 2.11.0+cu128`, fp32, `S=1999`, fixed Pi iterations `6`.

Times are CUDA event medians in milliseconds.  Ranges are min to max across the
available repetitions.  `Total` is `E + forward + backward + optimizer` and
does not include layout rebuild wall time; the adjacent wall note includes
layout and Python overhead.  Peak memory is the maximum recorded
`torch.cuda.max_memory_allocated()` for the measured reps.

| Families | Family chunk | Reps | Forward ms | Backward ms | Total ms | Wall ms | Peak GiB | Loss | Grad finite/norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 10 | 1 | `317.5` | `747.3` | `1089.9` | `1115.9` | `10.64` | `22182.9141` | not emitted; `grad_inf=4.4980` |
| 50 | 10 | 3 | `129.3` (`129.1-129.4`) | `2666.9` (`2628.2-2687.8`) | `2796.9` (`2757.9-2817.7`) | `2882.3` | `10.66` | `107804.2656` | not emitted; `grad_inf=3.1005` |
| 50 | 5 | 3 | `164.4` (`161.7-172.0`) | `3480.0` (`3170.2-5435.6`) | `3645.1` (`3332.4-5608.9`) | `3790.0` | `5.52` | `107804.2686` | not emitted; `grad_inf=1.5502` |
| 150 | 5 | 3 | `490.6` (`472.4-523.2`) | `11735.9` (`9412.4-12367.2`) | `12229.2` (`9885.3-12899.9`) | `12698.0` | `5.97` | `323018.6934` | not emitted; `grad_inf=1.3446` |
| 1000 | 5 | 2 | `54249.3` (`3168.4-105330.3`) | `94746.8` (`59809.6-129684.1`) | `149002.3` (`62978.4-235026.2`) | `151823.4` | `6.96` | `2157097.0537` | not emitted; `grad_inf=1.5644` |

The 1000-family run is not a stable promotion number.  Only two reps were
captured, and the forward phase varied from `3.17 s` to `105.33 s`, likely due
to first-measured-rep overhead or GPU contention.  It is useful only as evidence
that family chunk `5` can stream the full dataset within memory.

The 50-family chunk-size comparison is still useful:

- chunk `10` is faster for the current generic-heavy specieswise backward:
  `2.80 s` total event time versus `3.65 s`;
- chunk `5` roughly halves peak allocation: `5.52 GiB` versus `10.66 GiB`;
- the `grad_inf` difference is an artifact of the harness divide-by-chunks
  behavior and should not be interpreted as a numerical difference in the
  backward formulas.

### Nsight Systems Breakdown

Nsight Systems captures were collected for the chunked specieswise path with
family chunk `5`.

| Families | Script total under Nsight | Kernel launches | Kernel total GPU ms | Notes |
|---:|---:|---:|---:|---|
| 50 | `4781.5 ms` | `251829` | `2455.4` | one profiled rep |
| 150 | `12985.0 ms` | `747428` | `7272.2` | one profiled rep |

Kernel bucket distribution:

| Bucket | 50 families | 150 families | Interpretation |
|---|---:|---:|---|
| ATen elementwise | `977.6 ms` / `39.8%` | `2891.3 ms` / `39.8%` | largest generic bucket |
| cuSPARSE sparse | `438.1 ms` / `17.8%` | `1298.3 ms` / `17.9%` | species-tree sparse reductions remain generic |
| ATen index/scatter/gather | `228.6 ms` / `9.3%` | `678.1 ms` / `9.3%` | many small indexing kernels |
| ATen cat/materialize | `222.4 ms` / `9.1%` | `657.3 ms` / `9.0%` | materializes side buffers repeatedly |
| ATen copy/fill | `207.9 ms` / `8.5%` | `614.2 ms` / `8.4%` | allocation/zero/copy traffic |
| gpurec custom Triton/CUDA | `147.9 ms` / `6.0%` | `446.0 ms` / `6.1%` | optimized kernels are no longer the dominant bucket |
| ATen reductions | `90.7 ms` / `3.7%` | `267.7 ms` / `3.7%` | remaining generic reductions |
| cuBLAS dot/GEMV | `54.7 ms` / `2.2%` | `162.5 ms` / `2.2%` | small GEMV/dot work |

Top individual kernels in the 50-family profile:

| Kernel bucket | Launches | GPU ms | Avg us | Kernel |
|---|---:|---:|---:|---|
| cuSPARSE sparse | `1909` | `391.5` | `205.1` | `cusparse::csrmm_alg2_kernel` |
| ATen elementwise | `29686` | `232.4` | `7.8` | vectorized add |
| ATen cat/materialize | `843` | `195.8` | `232.2` | `CatArrayBatchedCopy_alignedK_contig` |
| ATen elementwise | `4482` | `141.2` | `31.5` | vectorized `where` |
| ATen elementwise | `7218` | `117.3` | `16.3` | elementwise multiply |
| gpurec custom Triton/CUDA | `2730` | `112.6` | `41.2` | `_wave_step_uniform_kernel` |

The 150-family profile has the same shape: `cusparse::csrmm_alg2_kernel` is the
largest individual kernel family (`1160.0 ms`), followed by ATen elementwise
and cat/materialization kernels.  This is a strong signal that the next
specieswise work should remove generic backward structure before retuning the
already optimized forward kernels.

### Nsight Compute Resource Summary

NCU was collected on representative launches from a 5-family chunk.  These are
single-launch resource diagnostics under replay overhead, not end-to-end timing
numbers.

| Kernel | Grid/block | Time us | SM % | DRAM % | L2 % | Active warps % | Regs/thread | Dynamic smem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `_wave_step_uniform_kernel` | `(1001,1,1)` / `128` | `60.5` | `60.7` | `27.1` | `23.6` | `53.3` | `40` | `1024 B` |
| `CatArrayBatchedCopy_alignedK_contig` | `(51,6,1)` / `128` | `3.4` | `3.9` | `19.1` | `11.1` | `16.8` | `30` | `0 B` |
| `cusparse::csrmm_alg2_kernel` | `(259,1,1)` / `128` | `6.4` | `18.5` | `4.3` | `4.6` | `14.7` | `53` | `0 B` |

The forward wave kernel is doing meaningful GPU work and has moderate SM
utilization.  The generic cat and cuSPARSE samples are low-occupancy kernels
that are individually small but numerous enough to dominate the Nsight Systems
trace.

### Experiments And Decisions

| Experiment | Result | Decision |
|---|---|---|
| Resident specieswise `autograd` path on 50 families | OOM during backward at about `21.87 GiB` allocated | Drop as a large-specieswise training strategy |
| Chunked specieswise path, family chunk `10` on 50 families | fastest measured 50-family total: `2.80 s`, but peak allocation `10.66 GiB` | Keep as high-memory local benchmark |
| Chunked specieswise path, family chunk `5` on 50/150/1000 families | lower memory (`5.52-6.96 GiB`) and streams 1000 families, but slower and noisy | Keep as conservative profiling harness default |
| Force specieswise through `GPUREC_FUSED_UNIFORM_BACKWARD=1` | not supported by current fused self-loop gate; harness disables it | Retest only after adding species-indexed fused self-loop/reduction support |
| NCU on `_wave_step_uniform_kernel`, cat, and cuSPARSE | confirms custom forward is not the main specieswise bottleneck | Use NCU for resource diagnostics; optimize generic backward buckets first |
| 1000-family chunked timing | completed within memory, but only two noisy reps | Retest before using as a throughput claim |

### Incomplete Data

The worker artifacts do not include:

- all-finite gradient checks;
- gradient L2 norm;
- chunk-size parity for gradients under a corrected sum/mean convention;
- resident-vs-chunked parity on a small slice that fits resident memory;
- stable repeated 1000-family timings without profiler or GPU contention;
- Nsight Systems for the 1000-family full streaming path;
- NCU on the specieswise backward kernels after removing generic fallback.

Until these are filled in, this round should be treated as profiling evidence
and implementation guidance, not as a correctness or promotion gate.

### Next Bottlenecks And Steps

1. Fix the profiling harness contract:
   - choose summed loss or mean loss explicitly;
   - remove or make explicit the `grad_theta / n_chunks` scaling;
   - emit `torch.isfinite(grad_theta).all()`, gradient L2 norm, and
     `grad_inf`;
   - report chunk-size parity for loss and gradients.
2. Add a small correctness matrix for the chunked specieswise path:
   - 3/10-family resident-vs-chunked parity where resident fits;
   - chunk `5` versus chunk `10` parity after the gradient convention is fixed;
   - constant specieswise parameters versus global uniform where applicable.
3. Port specieswise self-loop backward to the fused uniform structure:
   - accept `[S]` species-indexed `log_pD/log_pS/log_pL/mt` and E arrays;
   - accumulate `dtheta[S,3]` directly in fused kernels;
   - preserve the existing global and genewise gates.
4. Remove generic specieswise sparse/index/materialization buckets:
   - replace cuSPARSE species-tree reductions with compact tree kernels;
   - avoid `cat` materialization by passing side buffers or compact side-major
     scratch directly;
   - fuse the indexing/scatter/gather and copy/fill-heavy update sequences.
5. Retest chunk sizes after the fused specieswise backward exists:
   - benchmark family chunks `5`, `10`, `20`, and a high-memory option;
   - collect 50/150/1000-family timings with at least 5 stable reps;
   - promote the largest stable chunk that leaves enough memory headroom.
6. Re-run Nsight Systems and NCU after the specieswise port:
   - verify `gpurec_custom_triton_cuda` replaces the generic ATen/cuSPARSE
     buckets;
   - check whether launch count or allocator overhead then justifies CUDA graph
     capture and buffer pooling.

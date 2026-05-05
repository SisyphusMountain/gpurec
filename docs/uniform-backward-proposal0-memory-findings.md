# Uniform Backward Proposal 0 Memory Findings

Date: 2026-05-05

This document records the memory audit and tuning sweep that led to making
Proposal 0, the 2D Triton self-loop backward path, the default uniform backward
path when it is eligible. The default is no longer controlled by a fixed species
count threshold. It is controlled by an explicit tensor-payload memory estimate
and the current GPU memory budget.

## New Default

The practical default for the full global/uniform forward+backward pipeline is:

```bash
GPUREC_SELF_LOOP_2D_TRITON=auto
GPUREC_SELF_LOOP_2D_BLOCK_W=1
FAMILY_CHUNK_SIZE=auto
MAX_WAVE_SIZE=auto
```

The core backward path enables Proposal 0 by default only when it is eligible:

- CUDA tensors;
- uniform `Pibar` mode;
- fused uniform backward active;
- `fp32` or `fp64`;
- estimated Proposal 0 payload fits the current GPU memory budget.

This eligibility gate is important because Proposal 0 does not currently support
`bf16`. If the default were a plain environment flip, `bf16` would disable the
baseline in-kernel parameter accumulation and then fall back to the older kernel,
which would be a silent regression.

Set `GPUREC_SELF_LOOP_2D_TRITON=0` to force the older fused self-loop kernel.
Set `GPUREC_SELF_LOOP_2D_TRITON=force` only for debugging or profiling a case
that the memory policy rejects.

## Runtime And Memory Frontier

The sweep used the full global/uniform forward+backward pipeline on
`tests/data/test_trees_1000`, fixed 6 Pi iterations, and Proposal 0 enabled.

100-family medians, 3 timed reps:

| Families | Chunk | Max wave | `BLOCK_W` | Forward | Backward | Total | Peak allocated | Peak reserved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 100 | 32768 | 1 | `513.4 ms` | `365.6 ms` | `879.0 ms` | `21.33 GiB` | `22.67 GiB` |
| 100 | 50 | 16384 | 1 | `241.3 ms` | `348.4 ms` | `589.2 ms` | `10.78 GiB` | `14.11 GiB` |
| 100 | 25 | 8192 | 1 | `248.7 ms` | `350.9 ms` | `599.6 ms` | `5.44 GiB` | `5.73 GiB` |
| 100 | 25 | 4096 | 1 | `248.9 ms` | `356.7 ms` | `606.0 ms` | `4.94 GiB` | `5.01 GiB` |
| 100 | 10 | 32768 | 1 | `265.2 ms` | `412.6 ms` | `677.8 ms` | `4.39 GiB` | `6.00 GiB` |

Direct 1000-family confirmations, 1 timed rep:

| Families | Chunk | Max wave | Chunks | Forward | Backward | Total | Peak allocated | Peak reserved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 50 | 16384 | 20 | `2487.9 ms` | `3521.1 ms` | `6009.0 ms` | `11.32 GiB` | `21.18 GiB` |
| 1000 | 25 | 8192 | 40 | `2441.9 ms` | `3542.5 ms` | `5984.3 ms` | `5.94 GiB` | `16.17 GiB` |

The `chunk=25, max_wave=8192` point is the best practical default. It is
slightly faster than the higher-memory `chunk=50, max_wave=16384` direct run in
this sweep, while cutting peak allocated memory from `11.32 GiB` to `5.94 GiB`.
Compared with the previously documented default full pass of about `8684.9 ms`,
it is about `1.45x` faster end to end.

Other tunables:

- `GPUREC_SELF_LOOP_2D_BLOCK_W=2` was slower at the same allocated peak, so
  `BLOCK_W=1` remains the default.
- `MAX_WAVE_SIZE=4096` saved about `0.5 GiB` at 100 families but added launch
  overhead; it is useful only for unusually tight memory budgets.
- `GPUREC_BACKWARD_NO_CPU_PRUNING=1` and `GPUREC_DEVICE_PRUNING=1` were slower
  at the same memory footprint.
- DTS tile-size tweaks were neutral in this sweep.

## Memory Model

For the 1000-tree species set, `S=1999`. A dense fp32 `[W, S]` scratch tensor at
`W=32768` is:

```text
32768 * 1999 * 4 bytes = 0.244 GiB
```

Proposal 0 currently allocates ten full `[W, S]` scratch tensors in
`_wave_backward_uniform_2d_prototype`:

```text
v_k, aw0, aw1, aw2, aw345, aw3, aw4, spec_buf, term_buf, pibar_corr
```

At `MAX_WAVE_SIZE=32768`, that is about `2.44 GiB` of per-wave scratch before
allocator rounding and overlap with other live tensors.

The full training pass also keeps several resident `[C, S]` tensors per chunk:

- `Pi_wave_ordered`, saved from forward for backward;
- `Pibar_wave_ordered`, saved from forward for backward;
- `accumulated_rhs`, the backward RHS/adjoint state.

At 50 families these are each about `2.4 GiB`; at 100 families each is about
`4.7 GiB`. This is why resident family chunk size dominates memory, while
`MAX_WAVE_SIZE` mainly controls per-wave scratch and launch granularity.

The exact payload formulas used by the policy are:

```text
bytes(dtype) = tensor element size in bytes

dense_state_bytes(C, S, dtype)
    = 3 * C * S * bytes(dtype)

proposal0_wave_scratch_bytes(W, S, dtype)
    = 10 * W * S * bytes(dtype)

baseline_wave_scratch_bytes(W, S, dtype)
    = 8 * W * S * bytes(dtype)

uniform_training_payload_bytes(C, S, W, dtype, proposal0)
    = dense_state_bytes(C, S, dtype)
      + selected_wave_scratch_bytes(W, S, dtype)
      + small row/topology allowance
```

`C` is the number of clades resident in the current family chunk, `S` is the
number of species, and `W` is the emitted maximum wave size after applying
`MAX_WAVE_SIZE`.

The formula is exact for tensor payload bytes. The policy then compares it to a
conservative GPU memory budget:

```text
budget = min(total_vram * GPUREC_MEMORY_POLICY_FRACTION,
             current_free_vram - GPUREC_MEMORY_POLICY_RESERVE_GIB)
```

Defaults:

```bash
GPUREC_MEMORY_POLICY_FRACTION=0.85
GPUREC_MEMORY_POLICY_RESERVE_GIB=1.0
```

The margin is necessary because PyTorch allocator rounding, cached blocks,
profiler state, and temporary tensors are not captured by the payload formula.

## Highest-Value Memory Improvements

### 1. Dispatch Proposal 0 Before Generic Scratch Allocation

Current issue: `wave_backward_uniform_fused` allocates generic fallback scratch
before trying the Proposal 0 prototype. When Proposal 0 is enabled and no scratch
pool is passed, peak memory can transiently include both scratch sets.

Expected gain: up to one generic scratch set, roughly `2.44 GiB` at
`MAX_WAVE_SIZE=32768`, proportionally less at `8192`.

Risk: low. The main requirement is to preserve fallback behavior when Proposal 0
is requested but ineligible.

### 2. Add In-Kernel Parameter Accumulation To Proposal 0

Current issue: Proposal 0 rejects `accum_param_grads`, returns full per-element
VJP tensors, and the Python side reduces:

```python
scatter(grad_log_pD, aw0)
scatter(grad_log_pS, aw345)
scatter(grad_E, aw0 + aw2)
scatter(grad_Ebar, aw1)
scatter(grad_E_s1, aw4)
scatter(grad_E_s2, aw3)
scatter(grad_mt, aw2)
```

A Proposal 0 accumulated-parameter mode could avoid keeping several `aw*`
tensors live after the self-loop solve and remove the `aw0 + aw2` temporary.

Expected gain: around one to several `[W, S]` buffers, plus less Python/Torch
reduction work.

Risk: medium. The accumulation order changes, so fp32 parity should be checked
against both the older fused kernel and current Proposal 0.

### 3. Reuse `term_out` As `pibar_corr`

The 2D `J^T` kernel writes the subtree correction input to `pibar_corr`, reduces
it bottom-up, then writes the final term to `term_out`. Since `term_out` has no
useful old contents before that final write, it can probably serve as the
temporary correction buffer.

Expected gain: one `[W, S]` buffer, `0.244 GiB` at `MAX_WAVE_SIZE=32768` or
`0.061 GiB` at `8192`.

Risk: low to medium. The tree-reduction barriers and active-row zeroing need a
careful review.

### 4. Use Saved `Pibar` And Row Max For `inv_denom`

Proposal 0 currently recomputes the uniform denominator through row sums and
ancestor walks in the precompute kernel. Since the forward pass saved `Pibar` and
the row max, the denominator inverse can be derived as:

```text
inv_denom = exp2(row_max + mt - Pibar)
```

This does not directly remove a large allocation, but it should reduce parent
walk traffic and register pressure in the precompute kernel.

Risk: medium-low. It must handle `Pibar = -inf`, family-indexed `mt`, and exact
uniform semantics.

### 5. Recompute Or Narrow `p_prime`

`p_prime` is stored as a full `[W, S]` coefficient and loaded in every `J^T`
application. Recomputing it would save a buffer but add a Pi load and `exp2` per
Neumann term; lower-precision storage could save half of that buffer with some
numerical risk.

This is not the first target because the runtime tradeoff is less clear than
the scratch-dispatch and parameter-accumulation fixes.

### 6. Recompute Full Or Partial `Pibar`

Avoiding saved `Pibar_wave_ordered` would save one full resident `[C, S]` tensor,
which is a multi-GiB win at large chunk sizes. However, both self-loop backward
and DTS backward consume `Pibar`, so this would trade memory for repeated
forward-style denominator/tree work.

This should be considered only after the lower-risk Proposal 0 scratch fixes.

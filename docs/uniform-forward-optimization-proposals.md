# Uniform Forward Optimization Proposals

Date: 2026-05-04

This document is a proposal pass for the `pibar_mode="uniform"` forward pass.
It uses the current forward profile in `docs/uniform-forward-profile.md`, the
uniform backward optimization documents, and a fresh read-only code inspection.
The goal is to identify the next experiments that are likely to improve
likelihood throughput without repeating paths that were already measured and
rejected.

## Reference point

The small forward profile in `docs/uniform-forward-profile.md` uses the first
3 gene trees from `tests/data/test_trees_1000`, global uniform mode, fp32, an
RTX 4090, and fixed 6 Pi iterations.

| Quantity | Current value |
|---|---:|
| Forward wall time | `15.362 ms` |
| GPU active time | `14.053 ms` |
| Summed kernel time | `12.838 ms` |
| Device-to-device copy time | `1.213 ms` |
| Pi-loop host synchronizations | `0` |
| CUDA streams used | `1` |

The hot path is the Pi wave loop:

| GPU operation | Count | Time | Notes |
|---|---:|---:|---|
| `_wave_step_uniform_kernel` | `270` | `9.957 ms` | fixed 6 iterations over 45 waves |
| D2D wave-step copies | `270` | `1.200 ms` | mostly removed by ping-pong in large fixed-6 runs |
| `_dts_fused_kernel` | `44` | `0.808 ms` | small in the 3-family profile |
| Final/setup index/fill work | mixed | about `1.4 ms` | depends on caller and output mode |

The representative forward `_wave_step_uniform_kernel` launch is not a pure
DRAM-bandwidth problem:

| Metric | Value |
|---|---:|
| Compute throughput | `74.14%` |
| Memory throughput | `65.03%` |
| DRAM throughput | `44.24%` |
| L1/TEX throughput | `66.49%` |
| Issue slots busy | `73.01%` |
| Achieved occupancy | `88.69%` |
| Registers/thread | `40` |
| Tensor core use | `0%` |

So the primary small-batch target is instruction count and fused-kernel
algorithmic work, not only bytes moved.

For the 1000-tree workload, all families cannot be resident at once. The
profiled stable path used 150-family chunks. A representative 150-family chunk
after ping-pong measured:

| Component | Time |
|---|---:|
| Forward interval | `400.892 ms` |
| `_wave_step_uniform_kernel` | `269.677 ms` |
| `_dts_fused_kernel` | `55.350 ms` |
| final `_wave_pibar_uniform_parent_kernel` | `36.111 ms` |
| D2D copy time | `0.0166 ms` |

The largest 150-family wave-step launch had `W=238864`, `S=1999`, duration
`11.556 ms`, compute throughput `84.55%`, memory busy `71.76%`, DRAM
throughput `32.57%`, achieved occupancy `99.59%`, and no spills. At this
scale, copy traffic is no longer the main issue; the wave-step/Pibar algorithm
itself dominates.

## Already measured

These results should not be repeated unchanged:

| Idea | Result |
|---|---|
| Adaptive Pi convergence polling | Slower: `21.710 ms` versus fixed-6 `15.362 ms`; the old path paid `104` Pi-loop syncs and `6.765 ms` of sync time. |
| Fixed 4 or 5 iterations | Faster but not as accurate. Fixed 5 had `9.77e-4` NLL delta and larger Pi drift. Fixed 6 matched the adaptive NLL on the profiled workload. |
| Ancestor table inside the fused forward kernel | Correct but slower: about `20.915 ms` versus about `15.4 ms`. |
| Fused CSR ancestor correction | Correct but slower: about `18.257 ms`. |
| `torch.sparse.mm` Pibar | Much slower even after fixing RHS layout: about `50.335 ms`; surrounding exp/log/reduction/copy kernels dominated. |
| Ping-pong scratch for fixed even iterations | Accepted. On a 150-family chunk it removed `93.759 ms / 45.805 GB` of D2D copies, added `36.111 ms` final Pibar recompute, and improved the interval by about `65 ms`. |
| Light preprocessing and skipping the inclusion DAG | Accepted for construction. This fixed preprocessing, not the GPU forward interval. |

The important lesson from the sparse-matmul experiments is that the parent
pointer walk is not automatically worse than a "better" sparse layout. The
default fused kernel keeps the index working set small and cache-hot. The CSR
and ancestor-table variants remove loop-carried parent dependency, but they add
index streams and reduce eligible warps per scheduler.

## What transfers from backward

The backward optimization documents suggest several useful patterns, but not
all of them transfer directly.

| Backward lesson | Forward interpretation |
|---|---|
| Avoid materializing large `[rows, S]` intermediates when a reduced result is enough. | Strongly applies to high-fanout forward DTS. |
| Removing D2D copies is high value only when copies are still large. | Already mostly solved for fixed-6 large chunks by ping-pong. |
| Host sync/launch cleanup helps, but is secondary after hot kernels dominate. | Fixed 6 already removed Pi convergence syncs. CUDA graphs are lower priority unless combined with larger fusion. |
| Pruning helps backward because many adjoint rows are numerically inactive. | Forward probabilities are dense after a few waves. Do not add host-polled forward pruning without device-side sparsity evidence. |
| Tensor cores are not useful for scalar log-space tree reductions. | Still true for the current uniform forward hot path. |
| Parent-reduced DTS recompute helped backward. | This is the most direct forward candidate, especially for 150-family chunks. |

## Proposal 0: enable parent-reduced DTS in forward

Current forward calls `_compute_dts_cross(...)` without enabling its
`parent_reduced` path. The machinery already exists in
`gpurec/core/kernels/dts_fused.py`:

```text
old path:
    dts_fused(Pi, Pibar, splits) -> dts_term[n_splits, S]
    dts_r[eq1 parents] = dts_term[eq1]
    dts_r[ge2 parents] = seg_logsumexp(dts_term[ge2], ge2_ptr)

parent-reduced path:
    eq1 kernel writes directly to dts_r[parent, :]
    ge2 tiled kernel reduces split tiles by parent
    stage 2 combines partial max/sum into dts_r[parent, :]
```

This avoids writing a full `[n_splits, S]` split matrix when the wave-step only
needs `[W, S]` parent rows. It is most valuable for root-like waves with small
`W` and huge split fanout.

Why this should be first:

- The forward 3-family DTS bucket is only `0.808 ms`, so small batches will not
  move much.
- The 150-family chunk DTS bucket is `55.350 ms`, so the same idea has a real
  ceiling in the large workload.
- Backward already accepted the parent-reduced DTS recompute as a default path
  after parity and profiling.
- The forward wrapper already has the function parameters; the main work is
  policy, wiring, and profiling.

Suggested guarded implementation:

```python
parent_reduced = (
    os.environ.get("GPUREC_FORWARD_PARENT_REDUCED_DTS", "1") != "0"
    and pibar_mode == "uniform"
    and torch.device(device).type == "cuda"
)

dts_r = _compute_dts_cross(
    Pi, Pibar, meta, sp_child1, sp_child2,
    pD_dts, pS_dts, S, device, dtype,
    parent_reduced=parent_reduced,
    parent_reduced_min_splits=int(os.environ.get(
        "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS", "8192")),
    parent_reduced_impl=os.environ.get(
        "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL", "tiled"),
    parent_reduced_tile_splits=int(os.environ.get(
        "GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS", "64")),
)
```

The default should be gated by split count and fanout. A useful first policy is
`n_splits >= 8192`, using the existing `meta["ge2_max_fanout"]` to avoid a
device scalar sync inside the kernel wrapper.

Expected gain:

| Workload | Expected effect |
|---|---|
| 3 families | likely noise to `1 ms`, because DTS is only `0.808 ms` total |
| 50 to 150 families | likely useful; target `5-15 ms` on 150-family chunks if high-fanout waves dominate the DTS bucket |
| 1000 families chunked | same per-chunk gain multiplied over 7 chunks |

Risks:

- Parent-reduced DTS allocates partial buffers for high-fanout ge2 waves.
- Eq1-heavy waves may not benefit because direct write plus fill can be more
  expensive than the current materialization path.
- The wrapper currently fills `out[W,S]` with `-inf`; for waves where every
  row is written, this fill may be avoidable in a later pass.

Profiling gate:

- Nsys: `_dts_fused_kernel + _seg_lse_hdim_kernel + fill` versus
  `_dts_eq1_to_rows_kernel + _dts_parent_reduced_*`.
- NCU: DRAM writes and reads for the largest high-fanout wave.
- Correctness: compare NLL and `Pi_wave_ordered` against the current path for
  fixed 6 on 3, 50, and 150 families.

## Proposal 1: skip final Pibar for likelihood-only forward

The fixed-6 ping-pong path alternates `Pi` and `Pibar` as row buffers:

```text
iteration 0: Pi    -> Pibar scratch
iteration 1: Pibar -> Pi
iteration 2: Pi    -> Pibar scratch
iteration 3: Pibar -> Pi
iteration 4: Pi    -> Pibar scratch
iteration 5: Pibar -> Pi
```

After an even number of iterations, final `Pi` is already in `Pi`. The current
training path then recomputes final `Pibar` so backward can use
`Pibar_wave_ordered`. On a 150-family chunk, that final
`_wave_pibar_uniform_parent_kernel` costs `36.111 ms`.

For pure likelihood evaluation, final `Pibar` is not needed. The root
likelihood only consumes `Pi[root_clade_ids]`. Therefore a loss-only API can
skip final Pibar recomputation and avoid saving `Pibar_wave_ordered`.

Suggested API shape:

```python
Pi_wave_forward(..., need_pibar=True, return_roots_only=False)
```

Policy:

- Autograd/training path: `need_pibar=True`.
- Inference/log-likelihood-only path: `need_pibar=False`.
- If `need_pibar=False`, allow `Pibar` to remain scratch and return `None` for
  `Pibar_wave_ordered`.

Expected gain:

- About `36 ms` per 150-family chunk in the documented profile.
- About `250 ms` across seven 150-family chunks, if the same chunk profile
  holds for all 1000 trees.
- No numerical change to the NLL, because final `Pi` is unchanged.

Risks:

- The autograd path must not accidentally use this mode.
- Some callers may assume `Pibar_wave_ordered` is always a tensor. The safer
  implementation is an explicit argument, not an environment-only behavior.

Correctness gate:

- Loss-only NLL parity must be exact or within existing fp32 ordering noise.
- Backward tests must run with the default `need_pibar=True`.

## Proposal 2: source-counter-driven wave-step kernel cleanup

The dominant kernel is `_wave_step_uniform_kernel`. It is already fused and has
high occupancy, so broad rewrites should be justified by NCU source counters.
The next low-risk variants are small and measurable.

### 2a. Use int32 species topology in forward kernels

Backward gained from passing int32 topology and split metadata into hot
kernels. Forward still constructs species child/parent helpers as `torch.long`
and the Triton kernel uses int64 parent-chain values.

Experiment:

- Build `sp_child1`, `sp_child2`, and `sp_parent` as int32 for CUDA uniform
  forward when `S < 2^31`.
- Keep Python-visible helpers compatible, but pass int32 tensors to the Triton
  kernels.
- Compare generated registers, instruction count, L1/L2 traffic, and duration.

Expected gain is small but plausible. The forward kernel is instruction-heavy,
and parent-chain index operations sit inside the pass-2 loop.

### 2b. Sweep `BLOCK_S` and `num_warps`

Current uniform wave-step uses:

```python
BLOCK_S = min(256, triton.next_power_of_2(S))
num_warps = 4
```

For `S=1999`, that means eight species tiles per row. The largest 150-family
launch reaches almost full occupancy and high compute throughput. It is worth
profiling:

| Variant | What it tests |
|---|---|
| `BLOCK_S=128`, `num_warps=4` | smaller vectors, potentially lower register pressure |
| `BLOCK_S=256`, `num_warps=4` | current default |
| `BLOCK_S=256`, `num_warps=8` | more warp parallelism per row tile |
| `BLOCK_S=512`, `num_warps=8` | fewer loop tiles, more registers and vector lanes |

The acceptance criterion is end-to-end forward time, not just one launch.
Different wave sizes may prefer different settings, so a simple size-based
policy may beat a single global default.

### 2c. Disable row-max storage outside backward

When forward is used for backward, storing `uniform_pibar_row_max[C]` can avoid
some later denominator work. For likelihood-only forward, it is dead output.
The store is only one scalar per clade row, so the speed gain is likely small,
but it is an easy cleanup once `need_pibar=False` exists.

### 2d. Leaf/no-split source audit before specialization

The largest early waves are leaf/no-split waves. The current Triton kernel
already receives `has_splits` and `USE_LEAF_INDEX` as compile-time constants,
so easy branches are already specialized away. A separate leaf kernel should
only be built if NCU source counters show a specific remaining cost, such as:

- unnecessary `exp2(t5 - m)` work for non-leaf species lanes;
- repeated `leaf_logp` loads that are not cached;
- topology loads that could be bypassed for known species-leaf rows.

The backward no-split specialization experiments mostly lost because they
traded local recomputation for worse global-memory scratch access. Forward has
less scratch, so the only credible leaf specialization is one that reduces
instructions, not one that adds new row buffers.

Expected gain for Proposal 2:

- Small-batch: probably `0.5-2 ms` if a setting is clearly better.
- 150-family chunk: `5-15 ms` is plausible because the wave-step bucket is
  `269.677 ms`, but only if the change reduces instruction count without
  lowering occupancy.

## Proposal 3: CUDA shared-memory row-prefix Pibar prototype

This is the higher-risk version of Proposal 2. The current kernel computes
uniform Pibar by scanning every species row twice:

```text
pass 1:
    row_max, row_sum = reduce(Pi[row, :])

pass 2:
    for species s:
        ancestor_sum[s] = sum_{a in ancestors(s)} exp2(Pi[row, a] - row_max)
        pibar[s] = log2(row_sum - ancestor_sum[s]) + row_max + mt[s]
        compute DTS_L terms and logsumexp
```

The parent-pointer walk repeats ancestor work across species. A species-tree
prefix formulation can compute all ancestor sums for one clade row in
approximately `O(S)` tree work rather than `O(S * depth)` ancestor walks:

```text
for row c:
    sh_pi[s] = Pi[c, s]
    sh_exp[s] = exp2(sh_pi[s] - row_max)

    # species are in parent-before-child order
    ancestor_sum[root] = sh_exp[root]
    for s in species_topological_order[1:]:
        ancestor_sum[s] = sh_exp[s] + ancestor_sum[parent[s]]

    for s:
        pibar[s] = log2(row_sum - ancestor_sum[s]) + row_max + mt[s]
        Pi_new[c, s] = logsumexp(DTS_L terms)
```

This is awkward in current Triton because one program handles only a species
tile and cannot synchronize across all species tiles for a row. A CUDA block
can keep row vectors in shared memory and use block-wide synchronization.

Why it might help:

- It attacks the algorithmic ancestor-walk cost inside the largest forward
  bucket.
- It can also avoid rereading the Pi row in pass 2 by keeping row values in
  shared memory.
- It may be more valuable for fp64, where the current scalar exp/log path is
  extremely expensive on RTX 4090.

Why it might fail:

- The backward shared-memory Neumann prototype reduced global memory traffic
  but lost occupancy due to shared-memory footprint. Forward has less scratch,
  but the same occupancy risk exists.
- A row block for `S=1999` needs several shared arrays. Two fp32 row arrays are
  about `16 KB`; four are about `32 KB`. More arrays will reduce resident
  blocks per SM.
- Tree-prefix order and child gathers must be carefully laid out to avoid
  shared-memory bank conflicts and serial bottlenecks.

Recommended scope:

1. Prototype only no-split waves first, because that isolates Pibar/DTS_L from
   cross-clade DTS input.
2. Keep the existing Triton path as default.
3. Compare one large leaf wave and one large internal wave with NCU.

Acceptance gate:

- At least `5%` faster on the `_wave_step_uniform_kernel` bucket for 150-family
  chunks.
- No increase in peak memory.
- Exact or near-exact parity with the Triton path under fixed 6.

## Proposal 4: readiness-aware DTS overlap

The forward code has an `overlap_streams` argument, but the current high-level
model does not expose it, and the documented traces use one serialized CUDA
stream. Also, the existing overlap implementation schedules DTS for wave
`k+1` only after wave `k` finishes, then immediately waits before wave `k+1`.
That is not enough to guarantee useful overlap.

The useful version needs dependency readiness:

```text
for each wave j:
    dts_ready_after[j] = max(wave_index(child) for every split child of wave j)

while computing wave k self-loop:
    for future wave j with dts_ready_after[j] < k and dts not computed:
        launch DTS(j) on prep stream
```

This can overlap DTS for a future wave with the current wave-step only when all
children of that future wave are already finalized. It will not help if every
next wave directly depends on the current wave.

Expected gain:

- Small batches: low, because DTS is only `0.808 ms`.
- 150-family chunks: data-dependent. DTS is `55.350 ms`, but only the portion
  whose inputs are ready early can overlap. A realistic target is `0-10 ms`.

Profiling gate:

- First add instrumentation: for each wave, record `dts_ready_after`,
  `n_splits`, and whether the wave's DTS could have overlapped any earlier
  self-loop.
- Then expose `overlap_streams` or add a new lookahead flag and profile with
  Nsys stream timelines.

Risk:

- Extra streams may compete for L2/L1 and math pipelines. The wave-step kernel
  is already compute-heavy, while DTS is memory-bound. Partial overlap is
  plausible but not guaranteed.

## Proposal 5: forward-specific chunking and wave-size policy

Backward uses `max_wave_size=32768` as a good time/memory tradeoff because
backward scratch is large. Forward has different memory pressure:

```text
resident Pi/Pibar storage ~= 2 * C_chunk * S * sizeof(dtype)
```

For `S=1999` and fp32, that is about `15.99 KB` per clade row for the two
matrices. A chunk with about `960k` clades therefore needs about `15.3 GB` just
for `Pi + Pibar`, which matches the 150-family profile.

Recommendations:

1. Choose chunks by clade budget and available GPU memory, not by family count.
2. Solve `E` once for global/shared mode, then stream Pi chunks and accumulate
   the root NLL.
3. Sweep `max_wave_size` separately for forward-only and training modes.
4. Keep larger wave caps for forward-only if memory fits, because the largest
   wave-step launches already saturate occupancy well.

Suggested chunk-budget policy:

```python
usable = 0.80 * torch.cuda.mem_get_info()[0]
bytes_per_clade = 2 * S * dtype.itemsize
extra = estimated_dts_scratch + constants + allocator_margin
C_budget = floor((usable - extra) / bytes_per_clade)
```

The scheduler should then pack families until the next family would exceed
`C_budget`, with an optional hard cap for split fanout scratch.

Expected gain:

- Mostly memory safety and throughput stability.
- It can improve time if it allows larger stable chunks than a fixed family
  count, or avoids near-OOM chunks like the previously unstable 175-family run.

Correctness gate:

- Chunked NLL must equal the unchunked result on small subsets that fit.
- Chunk order must not change per-family likelihoods beyond expected fp32
  accumulation order differences.

## Proposal 6: root-only and liveness-aware forward outputs

The high-level autograd path already computes root likelihood from
wave-ordered roots and avoids returning `Pi[perm]`. Some direct optimizer or
debug callers may still request original-order `Pi`, and therefore pay an
avoidable final index copy.

Immediate cleanup:

- Audit all likelihood callers and pass `return_original=False` when only the
  root likelihood is needed.
- Add a root-only return mode for inference:

```python
Pi_wave_forward(..., return_root_rows=True)
```

This does not reduce peak memory by itself because internal rows are needed
while waves are computed. It does reduce output lifetime and final copies.

Larger rewrite:

- Compute last use for every clade row from split metadata.
- Evict or compact rows whose last parent has already been computed.
- Rewrite DTS and wave-step kernels to read from a compact live-row store
  instead of dense global clade ids.

This is a large change and should not be first. Current kernels assume dense
global `[C, S]` addressing for `Pi` and `Pibar`. Row eviction would save memory,
but it would complicate every split gather and probably reduce coalescing unless
the live-row layout is carefully designed.

## Proposal 7: genewise uniform leaf-index path

The fast uniform leaf path uses `leaf_species_index[C]` plus `leaf_logp[S]`.
It is disabled when `family_idx` is present. In batched genewise uniform mode,
the fallback can allocate dense `[C, S]` leaf/clade-species tensors even though
each leaf row has only one finite species.

Extend the Triton leaf hit path:

```text
if log_pS is [G]:
    t_leaf = log_pS[family_idx[row]] if leaf_species[row] == s else -inf

if log_pS is [G, S]:
    t_leaf = log_pS[family_idx[row], s] if leaf_species[row] == s else -inf
```

This does not affect the current global-mode profile, but it matters for
genewise training and for memory scaling when many families are batched.

Tests:

- `tests/unit/test_genewise_wave.py`
- `tests/integration/test_gene_recon_model.py`
- gradient bridge tests with genewise uniform mode

## Proposal 8: uniform-only species preprocessing

This is outside the GPU forward interval, but it improves end-to-end
likelihood throughput and memory footprint.

Current preprocessing still builds dense species helpers such as
`Recipients_mat` and `ancestors_dense`. Uniform parameter extraction only needs
the row maximum of the unnormalized recipient matrix, and uniform forward needs
species parent/child topology plus ancestor information.

Uniform-only preprocessing should:

- return `unnorm_row_max` directly without materializing full
  `Recipients_mat`;
- return parent/child arrays and compact ancestor CSR/table helpers directly;
- skip dense `ancestors_dense` unless a dense/debug path explicitly asks for it.

This will not reduce a warmed GPU forward profile, but it reduces construction
RSS, cache size, and repeated model setup for large datasets.

## Proposal 9: fp64 and tensor-core direction

The backward docs record a severe fp64 forward penalty on RTX 4090:

```text
10-tree fp32 fused forward/backward: 31.53 ms / 72.30 ms
10-tree fp64 fused forward/backward: 1251.91 ms / 395.58 ms
```

The forward fp64 path is about `39.7x` slower in that measurement. This is
expected on a consumer Ada GPU for scalar fp64 exp/log/reduction-heavy kernels.

Useful fp64 experiments:

- reduce the number of fp64 `exp2`/`log2` evaluations through the row-prefix
  Pibar prototype;
- test mixed precision for selected Pibar normalization internals, guarded by
  strict NLL and gradient parity;
- tune Triton block sizes separately for fp64.

Tensor cores are not a promising near-term target for uniform forward. The hot
work is scalar log-space reductions, parent-chain/tree gathers, and exp/log.
Using tensor cores would require reformulating a large part of uniform Pibar as
dense or block-sparse MMA. The earlier sparse-matmul path showed that even a
fast sparse matmul is not enough if the surrounding exp/log/materialization
work is not fused.

## Ranked plan

| Rank | Proposal | Why now | Main metric |
|---:|---|---|---|
| 0 | Enable/profile forward parent-reduced DTS | Existing tested machinery; targets `55 ms` DTS bucket on 150-family chunks | DTS bucket and total forward |
| 1 | Add likelihood-only `need_pibar=False` path | Directly saves documented `36 ms` final Pibar recompute per 150-family chunk | NLL parity and total forward |
| 2 | NCU-guided wave-step cleanup | Main bucket is `269 ms` per 150-family chunk | `_wave_step_uniform_kernel` duration and instruction counters |
| 3 | Forward-specific chunk and wave-size sweep | Current chunking is memory-limited and workload-dependent | total 1000-tree time, peak memory |
| 4 | Readiness-aware DTS overlap | Possible but dependency-limited | Nsys overlap and total forward |
| 5 | CUDA shared-memory row-prefix Pibar prototype | Higher-risk attack on the dominant algorithmic cost | wave-step bucket improves by at least `5%` |
| 6 | Genewise leaf-index and uniform-only preprocessing | Important for other modes and setup, not the current hot profile | memory, construction time, genewise forward |

## Profiling protocol for each accepted experiment

Correctness:

```bash
pytest -q tests/unit/test_wave_v2.py \
          tests/unit/test_cross_family_wave.py \
          tests/unit/test_genewise_wave.py \
          tests/kernels/test_wave_step_uniform_forward_kernel.py
```

For changes that affect autograd saves or training:

```bash
pytest -q tests/gradients/test_autograd_bridge.py \
          tests/gradients/test_fd_all_modes.py::test_analytic_matches_fd
```

Forward parity checks:

- Compare NLL against the current fixed-6 path.
- Compare `Pi_wave_ordered` on significant entries for small batches.
- For loss-only `need_pibar=False`, verify the NLL is unchanged and backward is
  never called through that path.

Performance:

- CUDA-event timing outside Nsight for 3, 50, 150, and full 1000-family
  chunked workloads.
- Nsys for total kernel buckets, D2D/D2H copies, launch count, and stream
  overlap.
- NCU for representative `_wave_step_uniform_kernel`, DTS kernels, and final
  Pibar kernels.
- Always record `torch.cuda.max_memory_allocated()`.

The most useful near-term success criterion is reducing the 150-family chunk
from about `400 ms` while preserving the same NLL. Small 3-family wins are
valuable only if they do not make the large-chunk path worse.

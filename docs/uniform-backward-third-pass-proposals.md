# Uniform Backward Optimization Proposals, Pass 3

This note is a third pass over the uniform backward pass after the two
optimization waves documented in:

- `docs/uniform-backward-fp32-fused-profile.md`
- `docs/uniform-backward-50tree-wave2-profile.md`
- `docs/uniform-backward-pruning-granularity.tex`
- `docs/uniform-backward-optimizations-summary.tex`

The goal here is not to repeat previous suggestions. The previous passes
already removed the largest obvious scratch streams, made most scalar
reductions device-side, staged the Pibar VJP input in the DTS kernel, and
changed the self-loop Neumann speciation update from scatter-scratch to
parent-gather. The next optimizations need to attack the remaining split-side
schedule and the places where pruning is measured but not actually used.

## Current reference point

The most recent documented 50-family default is:

```text
CUDA-event backward median: about 121.016 ms
Nsight captured backward event: 135.759 ms
Nsight summed kernel time: 109.177 ms
Kernel launches: 2970
```

The largest post-proposal-6 Nsight buckets are:

| Component | 50-family Nsys bucket | Status |
|---|---:|---|
| `_dts_cross_backward_accum_kernel` | `32.606 ms` | now largest bucket |
| `_uniform_cross_pibar_vjp_tree_from_ud_kernel` | `25.618 ms` | staged Pibar tree correction |
| `_wave_backward_uniform_kernel` | `25.412 ms` | much improved, still memory-latency limited |
| `_dts_fused_kernel` | `12.207 ms` | backward-side DTS forward recompute |
| `_active_mask_from_rhs_absmax_kernel` | `2.919 ms` | simple streaming row max |

Representative resource counters from the final/near-final profiles:

| Kernel | Representative launch | Main resource signal |
|---|---:|---|
| `_dts_cross_backward_accum_kernel`, staged path | `4.855 ms`, `2.027 GB` read, `1.319 GB` write, `96` regs/thread, `41.5%` occupancy | register pressure plus memory/RED traffic |
| `_uniform_cross_pibar_vjp_tree_from_ud_kernel` | `4.220 ms`, `2.023 GB` read, `1.318 GB` write, `40` regs/thread, `98.4%` occupancy | memory-bound tree-buffer traffic |
| `_wave_backward_uniform_kernel`, final gather path | `5.102 ms`, `1.008 GB` read, `1.895 GB` write, `40` regs/thread, `99.4%` occupancy | load-latency and memory throughput |
| `_dts_fused_kernel` | `2.120 ms`, `1.540 GB` read, `0.350 GB` write | pure split-term streaming/recompute |

So the next useful work is mostly not "make occupancy higher". The self-loop
kernel already has high occupancy, and the staged Pibar tree also has high
occupancy. The remaining cost is repeated memory traffic, split-side work that
is scheduled too broadly, and some host decisions that still serialize the wave
loop.

## What has already been tried

These ideas should not be repeated without a substantially different dataflow.

| Idea | Result | Why it failed or plateaued |
|---|---:|---|
| Disable pruning | `157.329 ms -> 241.924 ms` in a later 50-family profile | whole-wave pruning skips real work |
| Fixed-schedule device pruning | about `157.329 ms -> 161.084-161.736 ms` | removed syncs but launched waves the host skipped |
| Child-grouped uniform Pibar VJP | tree kernel `37.784 -> 32.119 ms`, but group reduction `16.102 ms` | duplicate child factor was only about `1.111x` |
| Forward Pibar stat reuse alone | 10-family small win, 50-family median regression to `166.956 ms` | replaced cache-friendly ancestor work with extra DRAM reads |
| Backward row-stat precompute | tree kernel saved `0.848 ms`, precompute cost `2.690 ms` | full extra pass over rows did not pay back |
| Prefix-denominator Pibar VJP | 50-family median `162.704 -> 180.791 ms` | level barriers and L2/DRAM traffic dominated |
| Child-grouped DTS accumulation | 50-family median `164.352 -> 175.860 ms` | high-fanout waves were not child-duplicate-heavy |
| No-atomic DTS accumulation | 50-family median `164.352 -> 163.053 ms`, Nsys DTS bucket unchanged | atomics were visible but not the first-order cost |
| Full `grad_mt` accumulation inside DTS | slower than scalar-only reduction accumulation | many atomics into only `S` species lanes |
| Monolithic DTS plus Pibar fusion | rejected after staged DTS reached `96` regs/thread | more fusion would likely reduce occupancy further |
| Bigger wave block/warp sweeps | `BLOCK_S=512` and 16 warps slower | the winning self-loop point is already `BLOCK_S=256`, 8 warps |
| No-split leaf specializations | slower than compact all-wave scratch | compiler already specializes major branches; extra scratch reads hurt |
| Tensor cores | not used | hot kernels are log/exp, tree gathers, reductions, atomics, and memory streams |

The important lesson is that separate reduction/materialization passes usually
lose. Successful changes either removed a large memory stream, removed a host
sync without adding GPU work, or fused a reduction into values already live in
the hot kernel.

## Proposal 0: remove unused fused-path pruning work

This is the lowest-risk cleanup. In the default fused uniform path, the loop
still computes:

```python
n_active = int(active_mask.sum().item())
if n_active < W:
    active_idx = active_mask.nonzero(as_tuple=True)[0]
    rhs_active = rhs_k[active_idx]
```

For the fused path, `active_idx` and `rhs_active` are not used by
`wave_backward_uniform_fused`. They are useful in the generic PyTorch fallback,
but the default CUDA path only uses the mask for whole-wave skipping unless
the experimental device-pruning flags are enabled.

The cleanup should split pruning into two modes:

```python
wave_active = active_mask.any()      # still needed for host whole-wave skip
if not wave_active:
    continue

if use_fused:
    # no active_idx, no rhs_active, no nonzero
    # optionally skip active_mask.sum().item() outside debug/stats mode
    pass
else:
    n_active = int(active_mask.sum().item())
    active_idx = active_mask.nonzero(as_tuple=True)[0]
```

Expected gain:

- small but almost free;
- likely around `0.5-2 ms` at 50 families if it removes CUB `nonzero` work and
  one scalar sync for many partially active waves;
- also fixes misleading statistics: "rows below threshold" should not be
  reported as "rows whose fused work was skipped".

Correctness risk is low because this does not change computed adjoints.

Profiling gate:

- Nsight Systems should show fewer CUB select/compact kernels, fewer PyTorch
  index kernels, and possibly fewer D2H scalar syncs.
- Numerical output should be bitwise identical because the fused kernels
  receive the same inputs as before.

### Proposal 0 implementation result

Implemented in `gpurec/core/backward.py`.  The production fused uniform path now
keeps only the whole-wave pruning decision:

```python
active_mask = _compute_active_mask(rhs_k)
if not active_mask.any():
    n_waves_skipped += 1
    n_clades_skipped += W
    continue

if use_fused:
    if GPUREC_BACKWARD_PRUNING_ROW_STATS:
        n_active = int(active_mask.sum().item())
        n_clades_skipped += W - n_active
    else:
        n_active = W
else:
    n_active = int(active_mask.sum().item())
    n_clades_skipped += W - n_active
```

Then compaction is also restricted to the generic fallback:

```python
if not kernel_pruning_wave and not use_fused:
    active_idx = active_mask.nonzero(as_tuple=True)[0]
    rhs_active = rhs_k[active_idx]
else:
    active_idx = None
    rhs_active = rhs_k
```

This keeps fallback behavior unchanged, while the default fused Triton path no
longer pays for compacted rows it never consumes.  `n_clades_skipped` in the
default fused path now means clades skipped by whole-wave pruning.  The old
partial-row count remains available only as the diagnostic
`GPUREC_BACKWARD_PRUNING_ROW_STATS=1`.

Correctness checks:

| Check | Result |
|---|---:|
| `python -m py_compile gpurec/core/backward.py` | pass |
| `pytest -q tests/kernels/test_active_mask_kernel.py tests/kernels/test_uniform_cross_pibar_vjp_kernel.py tests/kernels/test_dts_backward_accum_kernel.py` | `24 passed` |
| `pytest -q tests/gradients/test_autograd_bridge.py` | `15 passed` |
| `GPUREC_FUSED_UNIFORM_BACKWARD=0 pytest -q tests/gradients/test_autograd_bridge.py` | `15 passed` |
| `/tmp/gpurec_profile/check_b2_pruning_parity.py`, `FAMS=10` | loss diff `0`, theta max-rel `3.11e-6` for no-CPU/device pruning diagnostics |
| clean `HEAD` old path vs proposal-0 path, `FAMS=10` | loss diff `0`, theta max-rel `0` |

One broader standalone kernel file was also tried:
`pytest tests/kernels/test_active_mask_kernel.py tests/kernels/test_wave_backward_kernel.py`.
It still has pre-existing direct-kernel tolerance failures unrelated to this
host-pruning branch, so it was not used as the acceptance gate for proposal 0.

Event benchmark commands:

```bash
FAMS=10 REPS=15 WARMUPS=5 python /tmp/gpurec_profile/bench_uniform_backward.py
FAMS=50 REPS=15 WARMUPS=5 python /tmp/gpurec_profile/bench_uniform_backward.py
```

For old-code measurements I used a clean detached worktree at `HEAD` and forced
imports with `PYTHONPATH=/tmp/gpurec_prop0_old`, because the benchmark script is
executed by absolute path and otherwise imports the editable checkout.

| Workload | Old median | Proposal 0 median | Delta |
|---|---:|---:|---:|
| 10 families | `37.137 ms` | `36.419 ms` | `-0.718 ms` / `-1.9%` |
| 50 families | `120.771 ms` | `119.707 ms` | `-1.064 ms` / `-0.9%` |

Nsight Systems 50-family single-backward capture:

| Metric | Old clean `HEAD` | Proposal 0 | Delta |
|---|---:|---:|---:|
| CUDA event in capture | `134.394 ms` | `132.435 ms` | `-1.959 ms` |
| summed GPU kernel time | `109.003 ms` | `108.682 ms` | `-0.321 ms` |
| kernel launches | `2970` | `2846` | `-124` |
| `cudaStreamSynchronize` calls | `254` | `205` | `-49` |
| `cudaMemcpyAsync` calls | `576` | `527` | `-49` |
| D2H memcpy events | `194` | `145` | `-49` |
| `cudaLaunchKernel` API time | `4.952 ms` | `4.603 ms` | `-0.349 ms` |

The removed work breaks down as:

| Removed bucket | Old launches | Proposal 0 launches | Old GPU time |
|---|---:|---:|---:|
| active row `sum().item()` long reductions | `36` | `0` | `0.104 ms` |
| `nonzero` count/reduce kernels | `26` | `0` | `0.043 ms` |
| CUB device select kernels | `13` | `0` | `0.020 ms` |
| compacted row index-gather kernels | `13` | `0` | `0.048 ms` |

A diagnostic run with `GPUREC_BACKWARD_PRUNING_ROW_STATS=1` kept the row-count
reduction but still disabled fused-path compaction.  It had `2918` kernel
launches, `241` `cudaStreamSynchronize` calls, and a 50-family event median of
`119.896 ms`, essentially inside timing noise from the production proposal-0
path.  This shows that the row-count sync is measurable in Nsys, but the runtime
impact is small; the useful cleanup is mainly semantic and removes avoidable
host decisions.

The hot fused kernels were unchanged within noise:

| Kernel | Old Nsys bucket | Proposal 0 bucket |
|---|---:|---:|
| `_dts_cross_backward_accum_kernel` | `32.561 ms` | `32.580 ms` |
| `_uniform_cross_pibar_vjp_tree_from_ud_kernel` | `25.527 ms` | `25.467 ms` |
| `_wave_backward_uniform_kernel` | `25.379 ms` | `25.369 ms` |
| `_dts_fused_kernel` | `12.204 ms` | `12.214 ms` |
| `_active_mask_from_rhs_absmax_kernel` | `2.919 ms` | `2.919 ms` |

Conclusion: accept proposal 0 as a small cleanup.  It removes 49 host-visible
scalar decisions and 124 launches in the 50-family backward pass, but it does
not move the major compute buckets because only 13 processed waves were
partially active enough to trigger the old `nonzero` path, and the removed
kernels were tiny compared with DTS/Pibar/self-loop work.  The remaining
pruning bottleneck is still the host whole-wave `active_mask.any()` decision;
changing that belongs to proposal 1, not this cleanup.

## Proposal 1: hybrid host-wave skip plus fused row masks

The failed device-pruning experiment launched a fixed schedule for all waves.
That was the wrong comparison: it removed CPU syncs but lost whole-wave
skipping. The more useful hybrid is:

```python
mask = active_rows(rhs)
if not mask.any():
    skip the entire wave pipeline
else:
    pass mask into the fused wave, DTS forward, DTS backward, and Pibar kernels
```

This keeps the strongest property of host pruning, namely skipping whole waves,
while making row-level pruning real inside active waves. It is different from
the rejected `GPUREC_DEVICE_PRUNING=1` path because inactive waves remain
unscheduled.

Policies to test:

| Policy | Purpose |
|---|---|
| always pass mask for active waves | maximum row pruning, measures branch/mask overhead |
| pass mask only when `n_active < W` | avoids overhead on fully active waves |
| pass mask only when inactive fraction is at least `10%` or `25%` | avoids branch overhead for almost-full waves |
| pass mask only to split-side kernels | targets DTS/Pibar buckets first |
| pass mask only to self-loop kernel | isolates self-loop row-pruning value |

Important implementation detail: policies that require `n_active` should not
use `sum().item()` in production. First implement the simple "always pass mask
after whole-wave skip" variant, then add a debug/profiling variant that
collects active fractions. If active-fraction thresholds are needed in
production, compute them with a device-side counter or preallocated worklist.

Expected gain:

- upper bound is the work spent on inactive rows inside waves that are not
  wholly inactive;
- the pruning document suggests the current default counts many inactive rows
  that are not actually skipped, but we still need exact post-proposal-6
  inactive-row fractions;
- likely range is `0-8 ms` at 50 families, with the high end only if many
  processed split waves are partially inactive.

Main risk:

- every masked kernel has an extra branch and may write zeros for inactive
  rows unless carefully structured;
- if most active waves are almost full, the branch overhead can exceed savings.

Profiling gate:

- compare Nsys buckets for `_dts_fused_kernel`,
  `_dts_cross_backward_accum_kernel`, `_uniform_cross_pibar_vjp_tree_from_ud`,
  and `_wave_backward_uniform_kernel`;
- collect actual skipped row and skipped split-side counts separately from
  threshold-inactive diagnostics;
- reject if total kernel time does not fall even when row counts look good.

## Proposal 2: parent-reduced DTS forward recompute

The backward pass recomputes cross-DTS forward terms before the self-loop VJP:

```text
dts_fused(Pi, Pibar, splits) -> dts_term[n_splits, S]
seg_logsumexp(dts_term, reduce_idx) -> dts_r[W, S]
wave_backward_uniform_fused(..., dts_r) -> v_k
```

For high-fanout root-like waves this materializes a very large
`[n_splits, S]` tensor only to reduce it by parent. The largest documented
50-family split wave has:

```text
n_splits = 42155
S = 1999
dts_term size ~= 42155 * 1999 * 4 bytes = 337 MB
```

The current buckets are:

```text
_dts_fused_kernel:       12.207 ms
_seg_lse_hdim_kernel:     about 1.5 ms in earlier profiles
```

A parent-reduced DTS kernel should compute `dts_r` directly for multi-split
parents, avoiding the full split-by-species materialization.

The wave metadata already sorts multi-split parents contiguously:

```text
eq1 splits first, then ge2 splits sorted by reduce_idx
meta["ge2_ptr"] gives CSR parent ranges
meta["ge2_parent_ids"] maps CSR groups to wave-local parents
```

A practical implementation is a two-stage tiled reduction over split ranges:

```text
stage 1 grid: (parent_group, split_tile, species_block)
    for split in split_tile:
        compute the 5 DTS terms for species_block
        update local max and shifted sum
    write partial_max[parent_group, split_tile, species_block]
    write partial_sum[parent_group, split_tile, species_block]

stage 2 grid: (parent_group, species_block)
    reduce partial max/sum over split_tile
    write dts_r[parent, species_block]

eq1 path:
    compute dts_r directly without staging split terms
```

Why this is different from rejected child grouping:

- child grouping reduced duplicate child adjoints after DTS;
- this proposal reduces the split dimension by parent before materializing
  `dts_r`;
- the high-fanout structure is parent-heavy, not child-duplicate-heavy, so this
  matches the observed shape.

Expected gain:

- hard upper bound is roughly `_dts_fused_kernel + seg_lse`, about `13-14 ms`
  in the 50-family profile;
- realistic first implementation target is `4-8 ms`, because the new kernel
  still reads all child `Pi`/`Pibar` rows and does the same log/exp work;
- memory peak should also fall because the full `dts_term[n_splits, S]` buffer
  is avoided or replaced by much smaller partials.

Risks:

- if split tiles are too large, one program serializes too much work and loses
  parallelism;
- if partial buffers are too large or badly laid out, the second stage becomes
  another memory-bound reduction;
- the eq1 path may not benefit, so the implementation should switch only for
  high fanout or large `n_splits`.

Profiling gate:

- first instrument per-wave `n_splits`, number of ge2 parents, max/mean fanout,
  and bytes allocated for `dts_term`;
- enable the parent-reduced path only when `n_splits / n_ge2_parents` is high;
- NCU should show lower DRAM writes than `_dts_fused_kernel + seg_lse`;
- Nsys should show the combined DTS-forward bucket falling, not just moving
  time into a new partial-reduction kernel.

## Proposal 3: parent-tiled DTS backward accumulation

The current DTS backward accumulation is split-major:

```text
grid = (n_splits,)
program i:
    parent = reduce_idx[i]
    load parent Pi and v_k for every species block
    load left/right child Pi and Pibar
    compute vd0..vd4
    atomic_add direct Pi adjoints into child rows
    write staged Pibar u_d and A
```

This is now the largest bucket:

```text
_dts_cross_backward_accum_kernel: 32.606 ms at 50 families
representative staged launch: 4.855 ms
registers/thread: 96
achieved occupancy: 41.5%
global RED ops: 18.72 M
```

The high-fanout waves have many splits per parent:

```text
42155 splits over 247 parents: 170.7 splits/parent
24229 splits over 39 parents: 621.3 splits/parent
```

Split-major scheduling reloads the same parent `Pi_parent[s]` and `v_k[parent,s]`
for every split. A parent-tiled layout would instead make parent and species
block explicit:

```text
grid = (parent_group, split_tile, species_block)
load Pi_parent[species_block] once
load v_k[parent, species_block] once
for split in this parent's split_tile:
    load child rows
    compute vd0..vd4
    write or accumulate child adjoints and staged u_d
```

This is not the same as the rejected grouped DTS path. The rejected path grouped
by child after materializing local gradients, but the measured child duplicate
factor was about `1.00`. This proposal groups by parent, where the measured
fanout is large.

There are two variants to test.

### 3A. Parent-tiled split kernel with the same atomic outputs

Keep the same outputs as the current staged DTS kernel:

- direct atomic adds into `accumulated_rhs`;
- staged `pibar_ud[2 * n_splits, S]`;
- `pibar_A[2 * n_splits]`;
- scalar parameter reductions as today.

This isolates the benefit of parent reuse without changing the downstream
Pibar path.

Expected gain:

- parent `Pi_parent` and `v_k` loads become roughly one per split tile instead
  of one per split;
- if parent loads are a meaningful part of the current `2.027 GB` DRAM read,
  this can save several ms;
- realistic target is `3-6 ms` at 50 families.

Risk:

- looping over split tiles can increase live ranges and worsen the already high
  `96` register/thread pressure;
- direct child atomics and staged `u_d` writes remain, so the total gain is
  capped.

### 3B. Parent-tiled backward plus local split-side pruning stats

While parent tiling computes `vd1/vd2`, also compute cheap split-side magnitude
statistics:

```text
side_abs_l[i] = max_s abs(vd2 * inv_denom_l)
side_abs_r[i] = max_s abs(vd1 * inv_denom_r)
```

This does not change results yet. It tells us whether the staged Pibar tree is
processing many split sides whose `u_d` is exactly zero or below a threshold.
If many sides are inactive, Proposal 4 becomes much more attractive.

Profiling gate:

- compare representative NCU register count, occupancy, DRAM bytes, and RED
  operations against the current staged kernel;
- reject parent tiling if register pressure rises enough to erase memory-load
  savings;
- keep the first implementation restricted to root-like waves with high
  `n_splits / W`.

## Proposal 4: split-side worklists for staged Pibar VJP

The staged DTS path writes `u_d` and `A`, then the Pibar tree correction runs
one row per split side:

```text
_uniform_cross_pibar_vjp_tree_from_ud_kernel[(2 * n_splits,)]
```

The kernel no longer computes denominators, but it still reads and writes a
large tree buffer:

```text
representative launch: 4.220 ms
DRAM read: 2.023 GB
DRAM write: 1.318 GB
50-family Nsys bucket: 25.618 ms
```

A split side with zero or negligible `u_d` contributes nothing useful to the
tree correction:

```text
u_d = grad_Pibar * inv_denom
A = sum_s u_d[s]
contrib[s] = p_prime[s] * (A - subtree_sum[s])
```

If `u_d` is exactly zero, then `A` and every subtree sum are zero, so the whole
split-side tree program can be skipped exactly. For threshold pruning, the same
idea is approximate and must be guarded by the pruning threshold semantics.

Implementation sequence:

```text
1. In DTS staged kernel, compute side_abs_l/side_abs_r or exact side_nonzero.
2. For debugging only, report:
       active sides / total sides
       active sides by wave
       active sides inside already-active parent rows
3. If sparse enough, compact active split sides into a device worklist:
       side_child[side_id]
       side_ud_offset[side_id]
       side_A[side_id]
4. Launch Pibar tree only on active side_id rows.
```

The worklist should be generated on device. A host `nonzero().item()` version
would likely repeat the same mistake as fixed-schedule device pruning or
CPU-driven compaction.

Expected gain:

- upper bound is the `25.618 ms` staged Pibar tree bucket;
- realistic gain depends entirely on measured side sparsity;
- if `20%` of split sides are exactly inactive in processed split waves, the
  target is roughly `4-5 ms` before worklist overhead;
- if almost every side is active, this proposal should be dropped.

Risks:

- worklist construction is another reduction/compaction pass;
- if side sparsity is low, the worklist pass will cost more than it saves;
- approximate split-side thresholding can perturb gradients, so start with
  exact-zero skipping and only later evaluate thresholded pruning.

Profiling gate:

- add a stats-only mode first; do not implement compaction until side sparsity
  is known;
- measure worklist construction separately from the tree kernel reduction;
- verify old/new parity exactly for exact-zero mode and with finite-difference
  checks for thresholded mode.

## Proposal 5: reduce Pibar tree level-padding and species-topology traffic

The staged Pibar tree kernel is now mostly a tree-buffer correction:

```text
for level in range(N_LEVELS):
    for p_start in range(0, MAX_LEVEL_WIDTH, BLOCK_S):
        parent = level_parents[level, p_offs]
        parent_val += child1_val + child2_val
        store parent_val
```

This representation is simple, but it pads every level to `MAX_LEVEL_WIDTH`.
If the species tree is unbalanced, many lanes are masked off. Even in a
balanced tree, every split side reloads the same `level_parents`, `sp_child1`,
and `sp_child2` metadata. The current from-`u_d` tree kernel is memory-bound,
so topology and tree-buffer layout still matter.

The first step is instrumentation:

```text
level_widths[level]
padding_work = N_LEVELS * MAX_LEVEL_WIDTH - sum(level_widths)
padding_fraction = padding_work / (N_LEVELS * MAX_LEVEL_WIDTH)
```

If padding is significant, test a compact level layout:

```text
level_nodes_flat = concat(nodes_at_level_0, nodes_at_level_1, ...)
level_offsets[level], level_counts[level]
```

Triton cannot easily make the inner loop trip count dynamic without still
looping to a compile-time maximum, so there are two realistic implementations:

1. Generate a small family of specialized Triton kernels for the observed
   species tree shape, with per-level widths baked as constexprs.
2. Write this one kernel in CUDA/C++ if the level-width padding is confirmed to
   be a first-order cost.

This is more suitable for CUDA than most of the current kernels because the
algorithm is tree-topology-specific and not a regular row-wise map. The
maintenance cost is also bounded: one specialized Pibar tree-correction kernel,
not a wholesale rewrite.

Expected gain:

- if padding is low, no gain;
- if padding is high, target `2-5 ms` out of the `25.618 ms` Pibar tree bucket;
- this can also reduce instruction count and barrier pressure without changing
  the DTS staged dataflow.

Risks:

- prefix-denominator work already failed because extra passes and barriers
  dominated; this proposal must only change the bottom-up correction layout,
  not reintroduce denominator construction;
- generated Triton variants add complexity;
- CUDA/C++ may be justified only after padding is quantified.

Profiling gate:

- report padding fraction and actual active parent nodes per level;
- NCU should show lower executed instructions and lower DRAM/L2 traffic in the
  from-`u_d` tree kernel;
- Nsys must show the Pibar tree bucket falling without a new setup kernel
  erasing the win.

## Proposal 6: two-stage `grad_mt` reduction for the staged DTS path

The staged DTS path must still accumulate the vector `grad_mt` contribution:

```text
grad_mt[s] += sum_over_splits(vd1[i, s] + vd2[i, s])
```

Before DTS-to-Pibar staging, the `grad_mt` accumulation variant was tested and
rejected because it added many atomics into only `S=1999` species lanes. After
staging, however, the default path effectively has to do this work somewhere,
because `grad_Pibar_l/r` are no longer materialized for a later PyTorch
`sum(dim=0)`.

The current staged kernel does the vector accumulation directly with atomics.
The cost is visible in the proposal-5 NCU comparison:

```text
old DTS accum:     78 regs/thread, 49.7% occupancy, 16.02 M global RED ops
staged DTS accum:  96 regs/thread, 41.5% occupancy, 18.72 M global RED ops
```

The goal is to keep the staged Pibar win while taking vector `grad_mt` pressure
out of the already register-heavy DTS program.

A split-tiled two-stage reduction would look like this:

```text
stage 1, inside a parent-tiled or split-tiled DTS kernel:
    for split in split_tile:
        compute vd1, vd2
        local_mt[species_block] += vd1 + vd2
    store mt_partial[tile_id, species_block]

stage 2:
    reduce mt_partial[:, s] over tile_id
    add to grad_mt[s]
```

For the largest documented split wave, a tile size of `64` gives:

```text
ceil(42155 / 64) * 1999 * 4 bytes ~= 5.3 MB of partials
```

That is tiny compared with the `pibar_ud` staging buffer and much smaller than
the full `grad_Pibar_l/r` materialization that staging removed.

Why this is different from the rejected `grad_mt` atomics:

- the rejected path atomically added every split/species lane into `S` global
  addresses;
- this path reduces many splits locally first, then writes one partial per
  split tile and species lane;
- it is most naturally paired with Proposal 3's parent/split tiling, which
  already loops over multiple splits per program.

Expected gain:

- target is the extra register/RED pressure introduced by DTS-to-Pibar staging;
- plausible `1-4 ms` at 50 families if it lowers registers or MIO/RED stalls
  in `_dts_cross_backward_accum_kernel`;
- if implemented as an entirely separate pass that rereads large DTS values, it
  will likely lose.

Risks:

- if the partial-reduction kernel is memory-bound or launch-heavy, it may erase
  the savings;
- moving `grad_mt` summation changes fp32 reduction order, so exact bitwise
  parity should not be expected;
- this should be tested after or together with parent-tiled DTS, not as a
  blind extra pass on the current split-major kernel.

Profiling gate:

- NCU should show lower global RED operations and preferably lower registers in
  the DTS kernel;
- Nsys should show the DTS bucket plus the new reduction bucket lower than the
  current DTS bucket;
- gradient differences should match the expected reordered-fp32 tolerance and
  pass finite-difference checks.

## Proposal 7: scratch-buffer pooling for large wave temporaries

The hot wrappers allocate several large temporary tensors per wave:

- self-loop `v_k`, `aw*`, `spec_buf`, `term_buf`;
- DTS staged `pibar_ud[2 * n_splits, S]`;
- `pibar_A[2 * n_splits]`;
- `dts_r[W, S]`;
- sometimes full `dts_term[n_splits, S]`.

Most `torch.empty` calls are handled by the PyTorch caching allocator, so this
is not expected to be a large kernel-time win. However, the current backward
still has a nontrivial gap between summed kernel time and event time, and the
large buffers drive peak memory. A small scratch pool could help larger
batches or reduce allocator overhead variance.

Implementation:

```text
At backward setup:
    allocate max-sized scratch buffers for the largest wave and largest split wave.
Per wave:
    pass views/slices into Triton wrappers.
```

Expected gain:

- probably less than `1 ms` at 50 families;
- more valuable for 1000-tree scheduling because it bounds peak memory and
  avoids allocator churn;
- can be combined with Proposal 2, where avoiding `dts_term` may reduce the
  required scratch pool substantially.

Risks:

- buffer lifetime mistakes are easy because later kernels consume earlier
  scratch;
- overlapping views must not alias outputs that are simultaneously live;
- this should not be attempted before a liveness table is written for each
  wave phase.

## Proposal 8: family/chunk-aware pruning for very large batches

Whole-wave pruning weakens as more families are batched together. A global wave
is active if any family has an active row in that wave. With 1000 trees, most
global waves may remain active even if most families are locally inactive.

The current 50-family profile still benefits from host whole-wave pruning, but
larger batches likely need a second granularity:

```text
global wave
    -> chunks of rows, preferably family-contiguous or active-row-contiguous
    -> split chunks for high-fanout parents
```

This interacts with memory limits. The goal is not to return to tiny per-tree
waves. The goal is to keep chunks large enough for occupancy while letting
inactive families drop out.

Possible schedule:

```text
For each static wave:
    build active row mask
    if active rows are sparse by family:
        compact active row chunks on device
        process chunks with fixed max rows/splits
    else:
        process the full global wave
```

This should be considered after Proposal 1 gives real row masks inside active
waves. It is likely more important for 1000-family throughput than for the
50-family profile.

## Proposed implementation order

The order below maximizes learning per engineering hour.

1. **Remove unused fused-path active-index work and fix pruning statistics.**
   This should be exact and low risk.
2. **Instrument active rows, active splits, active split sides, DTS fanout, and
   Pibar level-padding.** Do this before building any more schedulers.
3. **Test hybrid host-wave skip plus fused row masks.** Preserve whole-wave
   skipping; do not repeat fixed-schedule device pruning.
4. **Implement parent-reduced DTS forward recompute for high-fanout ge2
   waves.** This attacks the `dts_fused + seg_lse` materialization.
5. **Prototype parent-tiled DTS backward accumulation on high-fanout waves.**
   Keep outputs unchanged first; only then add split-side stats.
6. **Move staged-path `grad_mt` into split-tile partial reductions if parent
   tiling exposes a cheap local reduction.** Do not add a blind full pass.
7. **If side sparsity is high, build staged-Pibar split-side worklists.**
   Start with exact-zero mode.
8. **If Pibar level padding is high, specialize the from-`u_d` tree correction
   layout.** Consider CUDA/C++ only for this one topology-specific kernel.
9. **Pool scratch buffers once the winning dataflow is clearer.**

## Measurement plan

Every proposal above should be measured with the same discipline as the earlier
passes:

```text
1. Correctness:
   - old-vs-new loss and theta-gradient parity on 3, 10, and 50 families;
   - focused Triton kernel parity tests;
   - finite-difference/autograd bridge tests;
   - fp32 and fp64 coverage when the kernel supports both.

2. Timing:
   - CUDA-event backward-only timings outside Nsight;
   - at least two run orders for small expected gains;
   - peak memory.

3. Nsight Systems:
   - total backward interval;
   - summed kernel buckets;
   - kernel launch count;
   - D2H sync/copy counts;
   - D2D copy volume;
   - whether time moved into a new helper kernel.

4. Nsight Compute:
   - representative largest launches;
   - DRAM read/write bytes;
   - register count and occupancy;
   - eligible warps per scheduler;
   - global RED operations;
   - excessive L2 sectors;
   - stall mix.
```

## Expected upper bound

The remaining 50-family median is about `121 ms`. A direct language rewrite to
CUDA is unlikely to cut this in half because the hot work is memory movement,
tree reductions, and split scheduling. But a third optimization pass still has
room:

| Direction | Plausible gain at 50 families | Notes |
|---|---:|---|
| Remove unused active-index/sync work | `0.5-2 ms` | exact cleanup |
| Hybrid row masks inside active waves | `0-8 ms` | depends on measured partial-wave inactivity |
| Parent-reduced DTS forward | `4-8 ms` | avoids large `[n_splits, S]` materialization for high-fanout waves |
| Parent-tiled DTS backward | `3-6 ms` | if parent reloads are significant and register pressure stays controlled |
| Two-stage `grad_mt` reduction | `1-4 ms` | only if it lowers staged DTS RED/register pressure |
| Split-side Pibar worklists | `0-5 ms` | only if side sparsity is real |
| Pibar tree layout specialization | `0-5 ms` | only if level padding/topology traffic is large |
| Scratch pooling | `<1 ms` at 50 families | more useful for memory and larger batches |

A realistic third-pass target is therefore about `10-20 ms` on the 50-family
backward if the parent-reduced DTS and parent-tiled DTS ideas work. A more
conservative target is `5-10 ms` if pruning and split-side sparsity are weak.
The hard upper bound is much lower than the full `109 ms` summed kernel time:
most of that work is still the real dense species-by-row math that must be
performed.

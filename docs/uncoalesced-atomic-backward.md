# Atomics and Uncoalesced Accesses in Uniform Backward

This note summarizes the remaining atomic operations and irregular memory
access patterns in the optimized uniform backward pass. It separates true
atomic contention from ordinary uncoalesced gathers/scatters, because they have
different optimization paths.

## Main Observation

The uniform backward pass is no longer dominated by generic PyTorch fallback
work. The remaining hard costs are mostly structural:

- multiple split terms can contribute to the same child clade;
- uniform Pibar corrections require subtree/ancestor reductions on the species
  tree;
- speciation terms naturally gather from or scatter to species-tree children;
- scalar or shared parameter gradients aggregate many row/species
  contributions into a small number of outputs.

Some of these require atomics unless we introduce a staged reduction. Others are
not atomic but still have poor coalescing because the addresses follow tree
topology rather than contiguous species order.

## Atomic Operations

### Baseline Self-Loop Ancestor Correction

In the baseline fused self-loop kernel, each species lane walks its parent chain
and atomically adds into the ancestor correction buffer:

```python
cur = s_offs
for _depth in range(MAX_ANCESTOR_DEPTH):
    tl.atomic_add(pibar_corr_ptr + out_base + cur, u_d, mask=anc_mask)
    cur = tl.load(sp_parent_ptr + cur, mask=anc_mask, other=-1)
```

Code: `gpurec/core/kernels/wave_backward.py`, around the baseline
`_wave_backward_uniform_kernel` ancestor correction.

This is one of the worst remaining patterns in the baseline path. It combines:

- parent-pointer chasing;
- non-contiguous writes;
- atomic accumulation;
- repeated tree-depth work per species.

This is the pattern Proposal 0 and the CUDA no-split tree path try to avoid by
turning the operation into a bottom-up tree reduction.

### Self-Loop Parameter Gradient Accumulation

The fused self-loop kernel can atomically accumulate parameter gradients
directly:

```python
tl.atomic_add(grad_log_pD_ptr + family, sum_aw0)
tl.atomic_add(grad_log_pS_ptr + family, sum_aw345)
tl.atomic_add(grad_E_ptr + grad_family_base + s_offs, aw0 + aw2)
tl.atomic_add(grad_Ebar_ptr + grad_family_base + s_offs, aw1)
tl.atomic_add(grad_E_s1_ptr + grad_family_base + s_offs, aw4)
tl.atomic_add(grad_E_s2_ptr + grad_family_base + s_offs, aw3)
tl.atomic_add(grad_mt_ptr + grad_family_base + s_offs, aw2)
```

These atomics are less problematic than the ancestor walk because the `[S]`
vector writes are contiguous within each program. The scalar `log_pD/log_pS`
atomics can serialize more strongly, but they are small compared with the
self-loop state traffic.

Proposal 0 currently disables this in-kernel accumulation and returns
per-element `aw*` tensors for Python-side reductions. That avoids these atomics
inside the Proposal 0 kernels, but increases memory footprint and adds external
reduction work.

### DTS Cross Backward Into `accumulated_rhs`

The DTS cross-clade backward kernel atomically adds direct child-clade
contributions into the global accumulated RHS:

```python
tl.atomic_add(accumulated_rhs_ptr + pi_l_base + s_offs, vd0 + vd1)
tl.atomic_add(accumulated_rhs_ptr + pi_r_base + s_offs, vd0 + vd2)
```

This is structurally needed because many splits can target the same child clade.
The current fused accumulation avoids materializing separate `grad_Pi_l` and
`grad_Pi_r` tensors followed by PyTorch `index_add_`, but the underlying
many-to-one accumulation still exists.

Possible improvement: staged per-parent or per-child reductions for high-fanout
waves. This only becomes high priority after the self-loop bucket is reduced,
because DTS cross backward is currently a secondary bottleneck.

### DTS Parameter and Transfer-Matrix Gradients

The DTS backward kernel also has atomics for parameter and transfer-gradient
accumulation:

```python
tl.atomic_add(grad_log_pD_ptr + ..., vd0)
tl.atomic_add(grad_log_pS_ptr + ..., vd3 + vd4)
tl.atomic_add(grad_mt_ptr + ..., vd1 + vd2)
```

Some transfer-gradient accumulation already has a two-stage path. Remaining
atomics are mostly acceptable until DTS becomes a larger share of total runtime.

### Uniform Cross-Pibar VJP Into `accumulated_rhs`

The uniform Pibar VJP tree kernels eventually add corrected contributions back
to child-clade Pi adjoints:

```python
contrib = p_prime * (A - subtree_sum)
tl.atomic_add(accumulated_rhs_ptr + pi_base + s_offs, contrib)
```

This appears in the direct tree kernel and the staged `from_ud` kernels. The
atomic is required because multiple split-side Pibar adjoints can hit the same
child clade.

The CUDA shared-memory Pibar VJP path has the same final atomic:

```cuda
atomicAdd(accumulated_rhs + rhs_base + s, contrib);
```

The tree reduction improves the ancestor/subtree correction, but it does not
remove the cross-split many-to-one accumulation into child clades.

### CUDA No-Split Self-Loop Path

The CUDA no-split row kernel also uses atomics for parameter gradients:

```cuda
atomicAdd(grad_E + s, aw0 + aw2);
atomicAdd(grad_Ebar + s, aw1);
atomicAdd(grad_E_s1 + s, aw4);
atomicAdd(grad_E_s2 + s, aw3);
atomicAdd(grad_mt + s, aw2);
atomicAdd(grad_log_pD, sum_pD);
atomicAdd(grad_log_pS, sum_pS);
```

This path is still faster for eligible no-split waves because it removes enough
global scratch and ancestor-correction traffic to dominate the cost of these
remaining atomics.

## Uncoalesced or Irregular Accesses

### Species Child Gathers

Speciation terms frequently read child species values:

```python
c1 = tl.load(sp_child1_ptr + s_offs)
c2 = tl.load(sp_child2_ptr + s_offs)
pi_s1 = tl.load(Pi_star_ptr + pi_base + c1)
pi_s2 = tl.load(Pi_star_ptr + pi_base + c2)
```

The `sp_child1/sp_child2` vectors are read contiguously, but the subsequent
loads from `Pi[..., c1]` and `Pi[..., c2]` are tree-indexed gathers. They are not
perfectly coalesced unless the species numbering happens to put child indices
near the parent species lanes.

This pattern appears in both self-loop and DTS backward kernels.

### Baseline Ancestor Parent Walk

The baseline ancestor walk is both atomic and uncoalesced:

```python
cur = tl.load(sp_parent_ptr + cur)
tl.atomic_add(pibar_corr_ptr + out_base + cur, u_d)
```

This is worse than a simple gather because each lane follows a different
parent-chain length and writes to a different ancestor sequence.

### Speciation Scatter Stores

The baseline self-loop speciation contribution writes to child species slots:

```python
tl.store(spec_buf_ptr + out_base + c1, src1)
tl.store(spec_buf_ptr + out_base + c2, src2)
```

These writes are conflict-free because each species has only one parent in the
species tree, so atomics are not needed. However, the writes are still scattered
and therefore not naturally coalesced.

The previously explored gather formulation replaces these stores with parent
loads. That can reduce scatter writes, but it was not exact or not consistently
faster in all production cases, so the current production path keeps the scatter
semantics.

### Compact Tree Reductions

The compact tree kernels use contiguous metadata arrays:

```python
parent = tl.load(compact_level_parent_ptr + node_offs)
c1 = tl.load(compact_level_child1_ptr + node_offs)
c2 = tl.load(compact_level_child2_ptr + node_offs)
```

That part is coalesced. The row-buffer accesses are still indexed by tree nodes:

```python
parent_val = tl.load(buf + row_base + parent)
c1_val = tl.load(buf + row_base + c1)
c2_val = tl.load(buf + row_base + c2)
tl.store(buf + row_base + parent, parent_val + c1_val + c2_val)
```

This is more structured than the parent walk and avoids atomics inside the tree
reduction, but it is still not a dense contiguous vector pass over species.

### Split Child-Row Access

DTS cross backward reads child clade rows selected by `sl` and `sr`:

```python
sl = tl.load(sl_ptr + i)
sr = tl.load(sr_ptr + i)
Pi_l = tl.load(Pi_star_ptr + sl * stride + s_offs)
Pi_r = tl.load(Pi_star_ptr + sr * stride + s_offs)
```

Within a given child row, the species dimension is contiguous. Across programs,
however, split child rows are irregular because `sl/sr` follow the clade split
list rather than a dense child-row order.

## Optimization Implications

### Highest-Value Targets

1. Replace the baseline self-loop ancestor walk wherever possible.
   This is the worst combination: parent-pointer chasing, uncoalesced writes,
   and atomics. Proposal 0 and the CUDA no-split tree kernel are targeting this
   directly.

2. Extend tree/2D treatment to split waves.
   The CUDA no-split path leaves split-wave Triton work untouched. Profiling
   showed a large remaining split-wave bucket, so this is a natural next target.

3. Reduce Proposal 0 memory so it can be used at good chunk sizes.
   Proposal 0 removes the worst ancestor-walk atomics, but currently pays with
   large `[W, S]` scratch and external parameter reductions.

4. Add an in-kernel or staged parameter-gradient path for Proposal 0.
   This would reduce full `[W, S]` `aw*` materialization and avoid Python-side
   reductions. It may reintroduce parameter atomics, but those are probably less
   harmful than the current memory pressure.

### Secondary Targets

1. Revisit DTS cross backward after self-loop improves.
   DTS cross backward has real atomics and irregular split child-row access. In
   recent profiles it can be comparable to, or larger than, the self-loop
   bucket. The reason to rank the self-loop first is that the no-split CUDA
   path already gives a concrete implementation route; DTS needs a narrower
   high-fanout design to avoid repeating rejected grouped/parent-tiled variants.

2. Stage high-fanout child-clade accumulation.
   For waves where many splits target the same child clade, a grouped or
   two-stage accumulation could reduce atomic contention into `accumulated_rhs`.
   This needs wave-shape-aware profiling.

3. Revisit scalar parameter atomics.
   Scalar `log_pD/log_pS` reductions can serialize, but they are not currently
   the dominant cost. Two-stage scalar reductions are only worth pursuing if NCU
   shows atomic stalls after larger memory-traffic issues are fixed.

## Current Code-Read Snapshot

This section records the current implementation state, so the next
implementation pass can start from the real routing logic instead of from the
older proposal names.

### Uniform Backward Wave Pipeline

`Pi_wave_backward` in `gpurec/core/backward.py` still runs one reverse wave at a
time. For each active wave the production CUDA/uniform path does:

1. build or receive an active row mask;
2. recompute split-side DTS forward terms when the wave has splits;
3. run the self-loop Neumann VJP through `wave_backward_uniform_fused`;
4. run DTS cross backward accumulation for split waves;
5. run uniform cross-Pibar VJP, usually from DTS-staged `u_d` rows.

The default gates worth remembering are:

| Area | Current default path | Main opt-in alternatives |
|---|---|---|
| Active mask | `_active_mask_from_rhs_absmax_kernel` when CUDA/uniform | fixed-schedule device pruning, still not default |
| Self-loop | `_wave_backward_uniform_kernel` in Triton | `GPUREC_CUDA_SELF_LOOP_NOSPLIT`, `GPUREC_SELF_LOOP_2D_TRITON`, `GPUREC_SELF_LOOP_TREE_STAGED` |
| DTS backward accumulation | `_dts_cross_backward_accum_kernel` direct split-major accumulation | grouped/noatomic, parent-tiled, parent-ragged |
| DTS-to-Pibar handoff | `GPUREC_DTS_PIBAR_UD_FUSION=1`, staging `pibar_ud` and `pibar_A` | non-staged `grad_Pibar_l/r` fallback |
| Cross-Pibar VJP | compact Triton tree-from-UD consumer | `GPUREC_CUDA_PIBAR_FROM_UD`, Euler-prefix experiment |
| Self-loop parameter gradients | in-kernel atomics in the main Triton self-loop | `GPUREC_SELF_LOOP_PARAM_TWO_STAGE` for eligible no-split waves |

The hot path is therefore no longer a generic PyTorch fallback. It is a small
number of memory-bound custom kernels plus a host wave loop that still makes
whole-wave pruning decisions.

### Self-Loop Details

The default Triton self-loop kernel is still the main place where an ancestor
walk and atomic tree correction are combined. It now computes the full
ancestor/subtree correction:

- the precompute pass walks `sp_parent` for every species lane to compute the
  uniform Pibar denominator;
- each Neumann iteration zeroes `pibar_corr`, computes
  `u_d = term * pibar_wt * inv_denom`, and atomically scatters `u_d` along the
  parent chain into `pibar_corr`;
- the same iteration also moves row-sized term/speciation scratch through
  global memory.

`GPUREC_COMPACT_PIBAR_SCRATCH=1` reduces scratch by storing
`pibar_wt * inv_denom` instead of separate Pibar weight and denominator arrays.
That helps memory footprint, but it does not remove the parent walk or the
ancestor atomics.

The NVRTC CUDA no-split kernel in
`gpurec/core/kernels/wave_backward_cuda.py` is an important implementation
candidate, but the semantic mode matters. `correction_mode="self"` is a
historical control mode. The exact current semantics are
`correction_mode="tree"`, because the Triton baseline applies the full subtree
correction. Any promotion discussion should benchmark:

```bash
GPUREC_CUDA_SELF_LOOP_NOSPLIT=1
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree
```

The CUDA no-split path keeps row-local Neumann state in shared memory and avoids
most global row scratch traffic. It still performs global parameter-gradient
atomics and, in exact `tree` mode, pays per-iteration shared-memory tree
barriers.

### Split-Side Details

The current DTS accumulation kernel does much more than the older
`grad_Pi_l/grad_Pi_r` materialization path:

- it atomically accumulates direct child Pi adjoints into `accumulated_rhs`;
- it can merge the speciation child contribution in the same kernel;
- it can accumulate scalar parameter reductions in-kernel;
- it can accumulate or two-stage `grad_mt`;
- when `GPUREC_DTS_PIBAR_UD_FUSION=1`, it emits `pibar_ud`, `pibar_A`, and
  optional side-activity flags for the cross-Pibar tree consumer.

This fusion is why the kernel has high register pressure, but it also avoids
large external reductions. A future split-side optimization should measure the
combined DTS-accumulation plus Pibar-VJP bucket, not only one kernel in
isolation.

The compact Triton Pibar-from-UD kernel currently reuses `pibar_ud` as global
tree scratch: it reads and writes the same `[2 * n_splits, S]` rows while
building subtree sums. The CUDA `GPUREC_CUDA_PIBAR_FROM_UD=1` prototype keeps
that tree scratch in shared memory instead. It still reads `pibar_ud` and still
atomically adds into `accumulated_rhs`, but it removes the global tree-scratch
writeback stream.

## Implementation Opportunities

### 1. Promote Exact CUDA No-Split Self-Loop Behind a Router

The most implementation-ready self-loop opportunity is not a new kernel. It is
to route large eligible no-split waves to the existing CUDA no-split row kernel
in exact `tree` mode, with an explicit fallback when the CUDA/NVRTC path is not
available.

Current eligibility is already checked in `Pi_wave_backward`:

- global/shared mode (`_auto_wrapped`);
- no split DTS term for the wave (`dts_r is None`);
- in-kernel parameter accumulation is available;
- dtype is fp32;
- uniform leaf index and saved row maxima are available;
- species topology tensors are int32;
- scalar `grad_log_pD` and `grad_log_pS`;
- compact species levels are available.

Implementation notes:

- add an `auto` router rather than flipping the existing opt-in flag blindly;
- use exact `tree` correction for correctness parity with the current baseline;
- keep `self` mode available only as a diagnostic/lower-bound mode;
- add a minimum wave-size threshold, for example
  `GPUREC_CUDA_SELF_LOOP_NOSPLIT_MIN_W`, and start benchmarking thresholds at
  `8192`, `16384`, and `32768`;
- keep the old Triton kernel for split waves and unsupported dtypes.

Benchmark expectation:

- strongest on full no-split waves such as `W=32768`;
- likely more useful as family count increases, because large no-split waves
  and peak scratch pressure become more important;
- NCU should show lower DRAM stores and lower global scratch traffic, even if
  shared-memory occupancy is lower.

Promotion gate:

- exact-mode parity against the current default on synthetic and real uniform
  workloads;
- no 10-family regression larger than noise;
- clear 50-family and 100-family event-median improvement or a memory/OOM win;
- Nsys self-loop bucket improves enough that DTS/Pibar buckets become the next
  largest unchanged costs.

### 2. Reduce CUDA No-Split Parameter Atomics or Shared Footprint

The CUDA no-split kernel removes global row scratch, but it still uses global
atomics for parameter gradients:

```cuda
atomicAdd(grad_E + s, aw0 + aw2);
atomicAdd(grad_Ebar + s, aw1);
atomicAdd(grad_E_s1 + s, aw4);
atomicAdd(grad_E_s2 + s, aw3);
atomicAdd(grad_mt + s, aw2);
```

It also keeps seven fp32 row arrays in shared memory for `S=1999`, about
`56 KiB` per CTA. That is useful for DRAM traffic but limits occupancy.

Two follow-up variants are worth implementing only after the exact CUDA router
is measured:

1. **Parameter partials for CUDA no-split.** Let the CUDA row kernel write
   `v_k` without parameter atomics, then run a compact row-tile reduction, or
   add CUDA-side row-tile partial reductions directly. The existing
   `GPUREC_SELF_LOOP_PARAM_TWO_STAGE` machinery is the closest scaffold, but
   prior standalone timing regressed, so this should be tested only in
   combination with the CUDA no-split solve.
2. **Lower-shared-memory row kernel.** Keep only `term`, `next/work`, `vacc`,
   and possibly `pcoef` in shared memory; recompute or reload cheaper
   coefficients such as diagonal/speciation weights. The goal is to move from
   one resident CTA per SM toward two or three resident CTAs while preserving
   most of the DRAM-store reduction.

NCU gates for these variants:

- global reduction instruction count should fall for the parameter-partial
  variant;
- dynamic shared memory per block should fall for the low-shared-memory
  variant;
- active warps and occupancy should improve without reintroducing enough DRAM
  traffic to lose the original win;
- no local spills.

### 3. Promote CUDA Shared Pibar-From-UD With a Split Threshold

The two-kernel CUDA Pibar-from-UD prototype is the most implementation-ready
split-side memory optimization. It preserves the current DTS producer and
replaces only the compact Triton tree consumer.

Implementation notes:

- add an internal threshold such as `GPUREC_CUDA_PIBAR_FROM_UD_MIN_SPLITS`;
- require fp32 CUDA, staged `pibar_ud`, saved `pibar_row_max`, and compact
  species levels;
- keep `GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1` for test/profiling runs, but
  production auto-routing should fall back cleanly if NVRTC or CUDA bindings
  are unavailable;
- threshold on either `n_ws` or estimated row traffic
  `2 * n_ws * S * sizeof(dtype)`.

Expected benefit:

- removes the global tree-scratch writeback stream from the Pibar VJP consumer;
- does not reduce peak memory, because `pibar_ud` is still materialized by DTS;
- does not remove final atomics into `accumulated_rhs`.

Promotion gate:

- no 10-family regression after thresholding;
- 50-family Pibar-from-UD bucket improves by at least about 20%;
- combined backward median improves enough to justify the extra runtime CUDA
  kernel dependency;
- NCU confirms DRAM writes fall and occupancy remains high.

### 4. Fuse DTS `u_d` Construction With Shared-Memory Pibar Tree

This is the larger split-side opportunity after the two-kernel CUDA
Pibar-from-UD path is understood. The current dataflow writes `pibar_ud` in DTS
and then reads it in the Pibar tree kernel. A fused variant would compute
`u_d`, reduce it through the species tree in shared memory, and atomically add
the final Pibar contribution to `accumulated_rhs` without materializing the
full `pibar_ud` matrix.

Two designs should be benchmarked separately:

1. **Pibar-side fused CUDA block.** A block handles one split side, recomputes
   or receives the Pibar-side adjoint, builds `u_d` in shared memory, runs the
   tree reduction, and writes final Pi contribution.
2. **DTS-plus-Pibar kernel split.** Keep direct Pi adjoints and parameter
   reductions in the current Triton DTS kernel, but move only the Pibar-side
   `u_d` construction and tree consumer into CUDA to avoid pushing the existing
   96-register DTS kernel even higher.

Risk signals:

- register count above the current DTS accumulation kernel;
- lower occupancy from too much fusion;
- local spills;
- increased total time for DTS accumulation plus Pibar VJP even if one kernel
  improves.

Acceptance should be based on the combined DTS-accumulation plus Pibar-VJP Nsys
bucket and on peak allocation, not only on the new kernel's standalone time.

### 5. Revisit High-Fanout DTS Backward With a Narrower Scope

The code already contains grouped, noatomic, parent-tiled, and parent-ragged DTS
accumulation variants. The documented parent-ragged attempt removed rectangular
overlaunch but still was not suitable as a default. A next attempt should be
narrower:

- keep eq1 split rows on the current direct split-major kernel;
- route only true high-fanout `ge2` parent groups through a ragged worklist;
- build and cache the worklist during scheduling/preprocessing, not inside the
  backward wave loop;
- avoid multiplying scalar reductions by species-block count;
- compare only on waves whose `ge2_mean_fanout` and split count exceed the
  configured threshold.

This should not be prioritized ahead of the exact CUDA no-split router and the
CUDA Pibar-from-UD threshold, because both of those have more direct evidence
and lower implementation ambiguity.

## Benchmarking Plan

### Correctness Gates

Run focused tests before any timing claims:

```bash
python -m py_compile gpurec/core/backward.py gpurec/core/kernels/wave_backward.py
python -m py_compile gpurec/core/kernels/wave_backward_cuda.py gpurec/core/kernels/pibar_vjp_cuda.py

pytest -q tests/gradients/test_autograd_bridge.py
pytest -q tests/gradients/test_uniform_backward_ancestor_batching.py
pytest -q tests/kernels/test_dts_backward_accum_kernel.py
pytest -q tests/kernels/test_uniform_cross_pibar_vjp_kernel.py
```

For self-loop changes, include targeted direct-kernel checks from
`tests/kernels/test_wave_backward_kernel.py`, especially the tests that compare
CUDA `tree` correction to the PyTorch ancestor-corrected reference.

For production promotion, also run a real-workload parity check against the
default path:

- loss should match exactly for the same forward state;
- theta-gradient max relative difference should stay in the fp32 atomic-order
  range, roughly `1e-5` or better unless a broader test justifies a looser
  tolerance;
- compare all returned gradient entries, not only wall time.

### Timing Matrix

Use the warmed uniform backward harness:

```bash
FAMS=10 REPS=7 WARMUPS=4 MAX_WAVE_SIZE=32768 \
  python profiling/proposal8/bench_uniform_backward.py

FAMS=50 REPS=7 WARMUPS=4 MAX_WAVE_SIZE=32768 \
  python profiling/proposal8/bench_uniform_backward.py

FAMS=100 REPS=5 WARMUPS=3 MAX_WAVE_SIZE=32768 \
  python profiling/proposal8/bench_uniform_backward.py
```

Benchmark these variants as a minimum:

| Variant | Env |
|---|---|
| Baseline | current defaults |
| Exact CUDA no-split | `GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree` |
| CUDA Pibar-from-UD | `GPUREC_CUDA_PIBAR_FROM_UD=1 GPUREC_CUDA_PIBAR_FROM_UD_STRICT=1` |
| Combined exact CUDA paths | both exact no-split and CUDA Pibar-from-UD flags |
| Self-loop parameter partials | add `GPUREC_SELF_LOOP_PARAM_TWO_STAGE=1` only for the scoped experiment |
| DTS high-fanout candidate | selected `GPUREC_DTS_BACKWARD_ACCUM_IMPL=...` variant |

Report at least median, mean, min, peak allocation, clades/waves/split rows,
and whether 100-family completes without OOM. The 10-family case is important:
several prototypes only help once large waves amortize their overhead.

### Nsight Gates

For a 50-family representative run, capture Nsight Systems with CUDA API
tracing and compare:

- total CUDA-event backward time;
- summed GPU kernel time;
- top buckets for self-loop, DTS accumulation, Pibar-from-UD, and DTS forward
  recompute;
- kernel launch count;
- D2H scalar copies/synchronizations;
- peak allocation if available from the harness.

Then run Nsight Compute on one representative launch from each changed bucket.
Track:

- duration;
- DRAM read/write bytes;
- global load/store/reduction instructions;
- registers per thread;
- achieved occupancy/active warps;
- dynamic shared memory per block;
- local spills;
- long-scoreboard, barrier, LG throttle, and MIO throttle samples.

Do not promote a path only because a sub-kernel improves. The combined bucket
and end-to-end backward median are the deciding metrics.

## Summary

The uniform backward pass still contains atomics, but they are not all equally
problematic.

The most damaging per-row pattern is the baseline self-loop ancestor walk,
because it combines atomics with uncoalesced parent-pointer traversal. Proposal
0-style tree treatment and the exact CUDA no-split tree path are therefore
pointed at the right structural bottleneck, even when DTS accumulation is the
largest wall-time bucket in a particular profile.

The remaining atomics into `accumulated_rhs` are more structural: split terms and
Pibar VJP terms are many-to-one by definition. They can be reduced with staged
grouping, but probably not eliminated without changing the scheduling model.

The remaining uncoalesced accesses mostly come from species-tree topology:
child gathers, child scatters, compact tree row-buffer accesses, and irregular
split child rows. These are best attacked with tree-level scheduling, grouped
split layouts, or staged reductions rather than simple launch-parameter tuning.

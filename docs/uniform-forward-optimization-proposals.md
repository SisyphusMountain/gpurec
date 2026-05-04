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
    and (
        os.environ.get("GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY", "1") == "0"
        or meta.get("n_ge2_clades", 0) > 0
    )
)

dts_r = _compute_dts_cross(
    Pi, Pibar, meta, sp_child1, sp_child2,
    pD_dts, pS_dts, S, device, dtype,
    parent_reduced=parent_reduced,
    parent_reduced_min_splits=int(os.environ.get(
        "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS", "0")),
    parent_reduced_impl=os.environ.get(
        "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL", "tiled"),
    parent_reduced_tile_splits=int(os.environ.get(
        "GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS", "64")),
)
```

The profiled default is now gated by ge2 presence, not by a large split-count
threshold: `GE2_ONLY=1`, `MIN_SPLITS=0`, `IMPL=tiled`, and `TILE_SPLITS=64`.
The old path remains available with `GPUREC_FORWARD_PARENT_REDUCED_DTS=0`.
The existing `meta["ge2_max_fanout"]` is still passed to avoid a device scalar
sync inside the kernel wrapper.

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

### Proposal 0 follow-up: implemented and profiled

Implemented the parent-reduced DTS path for uniform forward. The wiring reaches
all three forward DTS call sites:

- initial overlap-stream DTS prep;
- next-wave overlap-stream DTS prep;
- normal non-overlap wave loop.

The controls are:

```text
GPUREC_FORWARD_PARENT_REDUCED_DTS=1
GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY=1
GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS=0
GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL=tiled
GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS=64
```

`impl=direct` remains available for experiments, but is not the default because
it regressed on the 50-family benchmark.

Correctness evidence:

| Check | Result |
|---|---:|
| Focused kernel and forward command with parent-reduced DTS forced | `21 passed in 10.03 s` |
| 3-family forced `MIN_SPLITS=0` NLL | old/new both `6421.17333984375` |
| 3-family `Pi/Pibar` max abs delta | `0.00244140625` |
| 50-family `MIN_SPLITS=8192` NLL | old `107804.2734375`, new `107804.265625`, diff `0.0078125` |
| 150-family separate-run NLL | old/new both `323018.6875` |

Focused command:

```bash
GPUREC_FORWARD_PARENT_REDUCED_DTS=1 \
GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS=0 \
.venv/bin/python -m pytest -q \
  tests/kernels/test_dts_fused_kernel.py::test_dts_parent_reduced_accepts_int32_split_metadata \
  tests/kernels/test_dts_fused_kernel.py::test_dts_parent_reduced_matches_existing_recompute \
  tests/unit/test_genewise_wave.py::test_genewise_wave_large_s \
  tests/unit/test_cross_family_wave.py::test_batched_wave_matches_individual_large_s
```

Timing summary:

| Workload | Policy | Old | New | Delta |
|---|---|---:|---:|---:|
| 3 families | `MIN_SPLITS=8192` | `13.0007 ms` | `12.9954 ms` | `-0.0053 ms` |
| 3 families | `MIN_SPLITS=0`, ge2-only | `13.0003 ms` | `12.8738 ms` | `-0.1265 ms` |
| 50 families | `MIN_SPLITS=8192`, ge2-only, 7 reps | `138.5752 ms` | `136.3438 ms` | `-2.2314 ms` |
| 150 families | `MIN_SPLITS=8192`, ge2-only, 5 reps | `407.217 ms` | `402.447 ms` | `-4.770 ms` |

50-family sweep:

| Variant | Median time |
|---|---:|
| tiled, `tile16`, `MIN_SPLITS=8192` | `136.061 ms` |
| tiled, `tile32`, `MIN_SPLITS=8192` | `135.952 ms` |
| tiled, `tile64`, `MIN_SPLITS=8192` | `135.999 ms` |
| tiled, `tile128`, `MIN_SPLITS=8192` | `136.175 ms` |
| tiled, `tile256`, `MIN_SPLITS=8192` | `135.702 ms` |
| tiled, `tile64`, `MIN_SPLITS=0` | `135.736 ms` |
| direct, `tile64`, `MIN_SPLITS=8192` | `139.093 ms` |
| direct, `tile64`, `MIN_SPLITS=0` | `140.707 ms` |

150-family sweep:

| Variant | Median time |
|---|---:|
| tiled, `tile32`, `MIN_SPLITS=8192` | `403.650 ms` |
| tiled, `tile64`, `MIN_SPLITS=8192` | `403.895 ms` |
| tiled, `tile256`, `MIN_SPLITS=8192` | `402.928 ms` |
| tiled, `tile64`, `MIN_SPLITS=0` | `401.654 ms` |
| tiled, `tile256`, `MIN_SPLITS=0` | `403.608 ms` |
| direct, `tile64`, `MIN_SPLITS=8192` | `402.072 ms` |

The direct 150-family number looks competitive in that short sweep, but the
same implementation is clearly slower on 50 families, so it stays experimental.
Peak allocation on the 150-family old/new run improved from `15.227 GiB` to
`15.022 GiB`.

Nsight Systems on 50 families, `MIN_SPLITS=0`, ge2-only, one profiled rep:

| Metric | Old | New |
|---|---:|---:|
| Event timing | `136.869 ms` | `135.151 ms` |
| `_wave_step_uniform_kernel` | `93.515 ms` | `93.883 ms` |
| final Pibar | `12.219 ms` | `12.252 ms` |
| DTS-related bucket | `19.439 ms` | `17.405 ms` |
| DTS-related launches | `53` | `59` |

Old DTS detail:

| Kernel | Time | Launches |
|---|---:|---:|
| `_dts_fused_kernel` | `17.870 ms` | `46` |
| `_seg_lse_hdim_kernel` | `1.569 ms` | `7` |

New DTS detail:

| Kernel | Time | Launches |
|---|---:|---:|
| `_dts_fused_kernel` | `9.731 ms` | `39` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `7.488 ms` | `7` |
| `_dts_parent_reduced_ge2_stage2_kernel` | `0.130 ms` | `7` |
| `_dts_eq1_to_rows_kernel` | `0.056 ms` | `6` |

The DTS bucket saves `2.034 ms` even though DTS-related launch count increases.
The win comes from reducing split-materialized work and memory traffic, not
from launch reduction. Device copies were negligible in this isolated Pi-wave
profile, about `1.7 us`.

Nsight Compute on the largest 50-family DTS launch:

| Metric | Old `_dts_fused_kernel` | New ge2 stage 1 |
|---|---:|---:|
| Grid | `(42155,16,1)` | `(13,60,16)` |
| Duration | `2.168 ms` | `1.983 ms` |
| Registers/thread | `34` | `40` |
| Spills | none | none |
| Compute/memory throughput | `88.43%` | `75.98%` |
| SM throughput | `15.07%` | `14.18%` |
| Achieved active warps | `92.28%` | `94.12%` |
| Eligible warps/scheduler | `0.184` | `0.235` |
| Issue active | `15.07%` | `14.28%` |
| Long scoreboard stall ratio | `69.39` | `68.10` |
| L1 hit rate | `69.37%` | `55.53%` |
| L2 hit rate | `23.20%` | `12.23%` |
| Instructions executed | `372.23M` | `320.28M` |

The new stage-1 kernel remains memory-dependency limited, but it executes about
`14%` fewer instructions and lowers DRAM pressure. The lower cache hit rates are
acceptable because the algorithm does less total split-output materialization.

Decision: promote tiled parent-reduced DTS for ge2-containing uniform forward
waves by default with `MIN_SPLITS=0`, `GE2_ONLY=1`, and `TILE_SPLITS=64`. Keep
the old path available with `GPUREC_FORWARD_PARENT_REDUCED_DTS=0`, and keep
`impl=direct` as an experimental option only.

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

### Proposal 1 follow-up: implemented, but not a speed win

Implemented the explicit likelihood-only API:

```python
Pi_wave_forward(..., need_pibar=False)
```

When `need_pibar=False`, `Pi_wave_forward` returns:

```python
{
    "Pi_wave_ordered": Pi,
    "Pibar_wave_ordered": None,
    "uniform_pibar_row_max": None,
    ...
}
```

The autograd and optimizer paths keep the default `need_pibar=True`, so
backward still receives a full `Pibar_wave_ordered`. The no-grad
`GeneDataset.compute_likelihood_batch(...)` path now passes `need_pibar=False`
and accepts `fixed_iters_Pi` so likelihood-only fixed-iteration runs can use
the API directly.

The important safety correction is that the optimistic interpretation of this
proposal was not valid. In fixed-even ping-pong,
`wave_step_uniform_fused_into(..., STORE_PIBAR=False)` leaves final `Pi` in
`Pi`, but it does not leave final `Pibar` in `Pibar`. Later cross-DTS waves read
`Pibar[left_child]` and `Pibar[right_child]`, so final Pibar recomputation is
still required for every row that can be consumed by a later split.

The first attempted guard skipped by `meta["phase"] == 3`, but cross-family
waves can be phase-mixed. In a 3-family run, a wave labelled phase 3 contained
some roots and some non-root rows. Skipping that wave's Pibar recompute produced
the same root NLL by accident, but full `Pi_wave_ordered` differed by
`3.668`, which would corrupt later use of those rows. The final guard skips only
waves whose entire contiguous row range consists of root clades. To avoid a
per-forward device-to-host sync, `build_wave_layout` now stores
`root_clade_ids_cpu` during layout construction.

Skipped work accounting:

| Workload | Total waves | Root-only waves skipped | Rows skipped |
|---|---:|---:|---:|
| 3 families | `45` | `1` | `1` |
| 50 families | `49` | `1` | `1` |
| 150 families | `65` | `1` | `1` |

This explains why the original `36 ms` target was not attainable: almost all of
the `_wave_pibar_uniform_parent_kernel` bucket is for non-root rows whose Pibar
is needed by later DTS.

Correctness evidence:

| Check | Result |
|---|---:|
| 3-family direct fixed-6 NLL, `need_pibar=True` | `6421.17333984375` |
| 3-family direct fixed-6 NLL, `need_pibar=False` | `6421.17333984375` |
| Direct NLL delta | `0.0` |
| Full `Pi_wave_ordered` max abs delta | `0.0` |
| `need_pibar=False` return value | `Pibar_wave_ordered is None` |
| Focused forward/backward smoke tests | `35 passed in 12.53 s` |
| Model API likelihood tests | `3 passed in 55.70 s` |

Commands:

```bash
.venv/bin/python -m py_compile \
  gpurec/core/batching.py \
  gpurec/core/forward.py \
  gpurec/core/model.py \
  profiling/bench_uniform_forward_parent_dts.py

.venv/bin/python -m pytest -q \
  tests/kernels/test_dts_fused_kernel.py::test_dts_parent_reduced_matches_existing_recompute \
  tests/unit/test_genewise_wave.py::test_genewise_wave_large_s \
  tests/unit/test_cross_family_wave.py::test_batched_wave_matches_individual_large_s \
  tests/gradients/test_autograd_bridge.py

.venv/bin/python -m pytest -q \
  tests/unit/test_wave_v2.py::test_model_api_wave_matches_fp \
  tests/unit/test_cross_family_wave.py::test_model_api_wave_vs_sequential \
  tests/unit/test_cross_family_wave.py::test_batched_wave_100_families_large_s
```

Timing, proposal 0 enabled, fixed 6, 9 reps after moving root ids to CPU layout
metadata:

| Workload | `need_pibar=True` median | `need_pibar=False` median | Delta | Peak change |
|---|---:|---:|---:|---:|
| 3 families | `12.868 ms` | `12.884 ms` | `+0.016 ms` | `0.322501 -> 0.322431 GiB` |
| 50 families | `135.997 ms` | `135.672 ms` | `-0.325 ms` | `5.273371 -> 5.272171 GiB` |
| 150 families | `403.306 ms` | `404.224 ms` | `+0.918 ms` | `15.022059 -> 15.018502 GiB` |

Nsight Systems, one warmed profiled rep:

| Workload | Variant | Event time | `_wave_pibar_uniform_parent_kernel` | Launches | `_wave_step_uniform_kernel` |
|---|---|---:|---:|---:|---:|
| 50 families | `need_pibar=True` | `135.671 ms` | `12.395 ms` | `49` | `94.233 ms` |
| 50 families | `need_pibar=False` | `133.750 ms` | `11.971 ms` | `48` | `92.833 ms` |
| 150 families | `need_pibar=True` | `395.309 ms` | `35.533 ms` | `65` | `271.068 ms` |
| 150 families | `need_pibar=False` | `395.480 ms` | `35.460 ms` | `64` | `271.292 ms` |

The one removed Pibar launch saves only `0.073 ms` in the 150-family Nsight
capture. Other buckets are unchanged within noise. The small peak-memory drop
comes from not retaining `uniform_pibar_row_max` and not returning Pibar as a
valid output, not from removing the full Pibar working set, which is still
needed internally by later DTS.

Decision: keep `need_pibar=False` as a semantic likelihood-only API and keep
`compute_likelihood_batch` on that path, but do not count proposal 1 as a
performance win. Recovering the original `36 ms` target would require a deeper
liveness design, for example recomputing/storing Pibar only for child rows that
future DTS will actually read, or folding child Pibar computation into DTS.

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

### Proposal 2 follow-up: implemented int32 topology, rejected tile/default leaf rewrites

Proposal 2 was tested with the same worker/supervisor split as proposal 1:
one workstream reviewed implementation risk, one ran correctness/parity checks,
one handled profiling, and the supervisor provided the documentation checklist.
The final local implementation and documentation were applied in this branch.

Implementation:

- `GPUREC_FORWARD_TOPOLOGY_INT32` now controls CUDA forward species topology
  dtype and defaults to enabled when `S < 2^31`.
- `_get_species_wave_helpers(...)` keeps CPU helper tensors as `torch.long` for
  Python/PyTorch indexing, but caches CUDA `sp_child1`, `sp_child2`,
  `sp_parent`, and padded `ancestor_cols` as either `int32` or `long`.
- The helper cache key now includes the chosen index dtype so a process can
  switch between int64 and int32 experiments without reusing stale tensors.
- `_wave_step_uniform_kernel` and `_wave_pibar_uniform_parent_kernel` receive a
  `TOPOLOGY_INT32` constexpr. The parent-walk cursor stays int32 only on the
  int32 path; the int64 path keeps `s_offs.to(tl.int64)`. This was necessary
  because Triton rejects loop-carried variables whose dtype changes across
  iterations.
- `GPUREC_FORWARD_WAVE_BLOCK_S` and `GPUREC_FORWARD_WAVE_NUM_WARPS` were added
  as explicit tuning knobs for uniform forward kernels. The default tile policy
  remains the previous `BLOCK_S=min(256, next_power_of_2(S))`, `num_warps=4`.

The key kernel change is deliberately small:

```python
if TOPOLOGY_INT32:
    cur = s_offs
else:
    cur = s_offs.to(tl.int64)

for _ in range(0, MAX_ANCESTOR_DEPTH):
    cur_valid = mask & (cur >= 0) & (cur < S)
    pi_anc = tl.load(Pi_ptr + pi_base + cur, mask=cur_valid, other=NEG_LARGE)
    ancestor_sum += ...
    cur = tl.load(sp_parent_ptr + cur, mask=cur_valid, other=-1)
```

This attacks the integer/index side of the parent walk directly. It does not
change the mathematical order of the default `BLOCK_S=256`, `num_warps=4`
configuration, so the default int32 path is bitwise identical to the old int64
path on the tested small parity workload.

Correctness evidence:

| Check | Result |
|---|---:|
| 3-family NLL, int64 default | `6421.17333984375` |
| 3-family NLL, int32 default | `6421.17333984375` |
| 3-family NLL delta, int32 default | `0.0` |
| 3-family significant `Pi` max abs delta, int32 default | `0.0` |
| 3-family significant `Pi` max abs delta, `BLOCK_S=128, warps=4` | `3.2806e-4` |
| 3-family significant `Pi` max abs delta, `BLOCK_S=256, warps=8` | `3.2806e-4` |
| 3-family significant `Pi` max abs delta, `BLOCK_S=512, warps=8` | `2.5940e-4` |
| Focused forward/backward smoke tests | `35 passed in 58.47 s` |
| Model API likelihood tests | `3 passed in 67.22 s` |
| Uniform wave-step kernel tests with `BLOCK_S=512, warps=8` | `3 passed in 1.17 s` |
| `py_compile` touched Python files | passed |

The non-default tile shapes preserve the NLL in the 3-family fixed-6 check, but
they are not bitwise-equivalent full-tensor rewrites because row max/sum
reductions happen in a different tile order. That is acceptable for an opt-in
profiling knob, but it is not a reason to change the production default.

Timing, proposal 0 enabled, fixed 6, `need_pibar=False`, 50 families:

| Variant | Median | Mean | Min | Delta vs int64 default | NLL |
|---|---:|---:|---:|---:|---:|
| int64 default | `135.496 ms` | `135.600 ms` | `134.844 ms` | - | `107804.265625` |
| int32 default | `116.706 ms` | `116.732 ms` | `116.143 ms` | `-18.790 ms` | `107804.265625` |
| int32, `BLOCK_S=128`, `warps=4` | `117.370 ms` | `117.315 ms` | `116.619 ms` | `-18.126 ms` | `107804.265625` |
| int32, `BLOCK_S=256`, `warps=8` | `115.441 ms` | `115.743 ms` | `114.639 ms` | `-20.054 ms` | `107804.265625` |
| int32, `BLOCK_S=512`, `warps=8` | `118.148 ms` | `119.659 ms` | `116.409 ms` | `-17.347 ms` | `107804.265625` |

Timing, same setup, 150 families:

| Variant | Median | Mean | Min | Delta vs int64 default | NLL |
|---|---:|---:|---:|---:|---:|
| int64 default | `403.379 ms` | `402.480 ms` | `400.092 ms` | - | `323018.6875` |
| int32 default | `346.532 ms` | `346.526 ms` | `345.625 ms` | `-56.847 ms` | `323018.6875` |
| int32, `BLOCK_S=128`, `warps=4` | `375.678 ms` | `372.588 ms` | `350.303 ms` | `-27.701 ms` | `323018.6875` |
| int32, `BLOCK_S=256`, `warps=8` | `347.310 ms` | `346.894 ms` | `345.540 ms` | `-56.069 ms` | `323018.6875` |
| int32, `BLOCK_S=512`, `warps=8` | `349.448 ms` | `349.114 ms` | `346.824 ms` | `-53.931 ms` | `323018.6875` |

After making int32 topology the default, the no-flag benchmark reports:

| Workload | Default median | NLL | Peak GPU |
|---|---:|---:|---:|
| 50 families | `116.663 ms` | `107804.265625` | `5.271977 GiB` |
| 150 families | `344.509 ms` | `323018.6875` | `15.018308 GiB` |

The 8-warp tile is slightly faster for 50 families, but not for 150 families.
Because the 150-family chunk is the current high-occupancy target, the default
tile shape stays unchanged and the tile sweep remains an experiment knob.

Nsight Systems, one warmed 150-family rep:

| Kernel bucket | int64 default | int32 default | Delta | Launches |
|---|---:|---:|---:|---:|
| Event time | `399.104 ms` | `344.755 ms` | `-54.349 ms` | - |
| Kernel span | `399.075 ms` | `344.659 ms` | `-54.416 ms` | - |
| `_wave_step_uniform_kernel` | `274.317 ms` | `225.193 ms` | `-49.124 ms` | `390 -> 390` |
| `_wave_pibar_uniform_parent_kernel` | `36.143 ms` | `30.710 ms` | `-5.433 ms` | `64 -> 64` |
| `_dts_fused_kernel` | `30.917 ms` | `30.896 ms` | `-0.021 ms` | `49 -> 49` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `23.263 ms` | `23.441 ms` | `+0.178 ms` | `8 -> 8` |
| PyTorch index kernels | `12.267 ms` | `12.270 ms` | `+0.003 ms` | `50 -> 50` |
| PyTorch fill kernels, main bucket | `8.697 ms` | `8.697 ms` | `0.000 ms` | `61 -> 61` |

The improvement is therefore not a scheduling or launch-count effect. It is
concentrated in the two kernels that walk species parent/child topology.

Nsight Compute, representative large 150-family `_wave_step_uniform_kernel`
launch `54`:

| Metric | int64 default | int32 default |
|---|---:|---:|
| Duration | `1.635520 ms` | `1.340448 ms` |
| Executed instructions | `1.504e9` | `1.158e9` |
| Issued instructions | `1.504e9` | `1.158e9` |
| Registers/thread | `40` | `40` |
| Local spill requests | `0` | `0` |
| Achieved occupancy | `98.18%` | `98.11%` |
| Achieved active warps/SM | `47.12` | `47.09` |
| Issue slots busy | `81.38%` | `76.17%` |
| Compute throughput | `81.67%` | `82.17%` |
| Memory throughput | `71.43%` | `82.17%` |
| DRAM throughput | `47.57%` | `58.04%` |
| L1/TEX throughput | `71.73%` | `82.39%` |
| L2 throughput | `62.28%` | `62.09%` |
| L1/TEX hit rate | `80.50%` | `81.83%` |
| L2 hit rate | `88.86%` | `86.37%` |
| Branch efficiency | `100%` | `100%` |
| Divergent branches | `0` | `0` |

The NCU interpretation is that int32 does not improve occupancy and does not
remove spills; it removes about `23%` of the executed instructions in the hot
launch. NCU's compute-workload rule reported the old int64 kernel as ALU
over-utilized (`80.3%` ALU pipeline), while the int32 kernel dropped that ALU
pipeline estimate to `47.2%`. Source counters still report uncoalesced global
accesses, so the change is best understood as reducing integer/index instruction
pressure around the same irregular ancestor-gather pattern, not as fixing memory
coalescing.

Leaf/no-split audit:

| Quantity, 150 families | Value |
|---|---:|
| No-split waves | `8` |
| No-split rows | `238864` |
| No-split rows that are leaves | `238864` |
| No-split fixed-6 wave-step launches | `48` |
| No-split wave-step time in int32 Nsys trace | `53.822 ms` |
| Split-wave wave-step time in int32 Nsys trace | `171.371 ms` |

NCU on the first leaf-only no-split launch showed the same basic profile as
the representative split launch: `1.303424 ms`, `40` registers/thread, no
spills, `98.09%` achieved occupancy, `1.141e9` executed instructions, `100%`
branch efficiency, and no divergent branches. The `has_splits` branch is already
a compile-time constant and the leaf term already uses `USE_LEAF_INDEX`; there
was no source-counter evidence for an easy leaf-only specialization. A separate
leaf/no-split kernel remains possible, but it should be treated as a new
algorithmic proposal rather than a small cleanup.

Row-max storage:

Proposal 2c was already handled by proposal 1. `reuse_forward_pibar_stats` is
now gated by `need_pibar`, and likelihood-only forward returns
`uniform_pibar_row_max=None`. This removes dead backward-only row-max output from
the likelihood path, but proposal 1 measured only a very small memory effect and
no meaningful speedup because the row-max store is one scalar per clade row.

Profiler artifacts:

```text
/tmp/gpurec_profile/forward_prop2/nsys_f150_base.nsys-rep
/tmp/gpurec_profile/forward_prop2/nsys_f150_i32.nsys-rep
/tmp/gpurec_profile/forward_prop2/ncu_wave_base_launch54.csv
/tmp/gpurec_profile/forward_prop2/ncu_wave_i32_launch54.csv
/tmp/gpurec_profile/forward_prop2/ncu_wave_base_launch54_instruction.csv
/tmp/gpurec_profile/forward_prop2/ncu_wave_i32_launch54_instruction.csv
/tmp/gpurec_profile/forward_prop2/ncu_wave_i32_leaf_launch0.csv
```

Decision:

- Promote int32 species topology as the CUDA uniform forward default.
- Keep `GPUREC_FORWARD_TOPOLOGY_INT32=0` as an escape hatch.
- Keep `GPUREC_FORWARD_WAVE_BLOCK_S` and `GPUREC_FORWARD_WAVE_NUM_WARPS` as
  profiling knobs, but do not change the default tile shape.
- Do not implement a leaf/no-split specialization in this pass.

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

### Proposal 3 follow-up: shared-memory row-prefix Pibar prototype

Proposal 3 was tested as a feasibility and performance study rather than a
production implementation.  The completed work covered the current code map,
baseline profiles, a Triton scratch row-prefix Pibar prototype, and a local
NVRTC CUDA shared-memory Pibar-only prototype.

The important distinction is scope: both prototypes implemented Pibar
normalization, not the full fused DTS_L wave-step update.  The original
acceptance gate was a reduction in the `_wave_step_uniform_kernel` bucket for
150-family chunks.  Since no full DTS_L fused CUDA row-prefix kernel was
implemented, Proposal 3 cannot be accepted as a production fp32 forward path.

#### Code map and feasibility

Worker A mapped the current uniform wave-step implementation:

| Location | Role |
|---|---|
| `gpurec/core/kernels/wave_step.py:_wave_step_uniform_kernel`, lines `381-393` | row max/sum reduction |
| lines `410-445` | Pibar ancestor correction modes |
| lines `433-445` | default parent-pointer walk |
| lines `447-450` | Pibar formula |
| lines `457-512` | DTS_L term loads and logsumexp update |
| `gpurec/core/forward.py:_run_wave_self_loop` | default ping-pong wiring through `wave_step_uniform_fused_into` and final `wave_pibar_uniform_parent_fused` |

For `tests/data/test_trees_1000`, species ids are child-before-parent: every
non-root species has `parent_index > child_index`.  Therefore a prefix can be
computed by descending species id, or by explicit species-tree levels.  The
tested species topology has `S=1999` and `max_depth=23`.

A minimal CUDA hook would likely follow the existing NVRTC/driver pattern in
`wave_backward_cuda.py`: one CUDA block per clade row and dynamic shared memory
for one species row.  Shared-memory footprint is manageable for Pibar-only
state but still a serious occupancy constraint:

| dtype | one row shared array |
|---|---:|
| fp32 | `7.81 KiB` |
| fp64 | `15.62 KiB` |

Adding more arrays for a fully fused DTS_L wave-step would reduce resident
blocks per SM further.

#### Representative commands

The baseline forward interval used the current optimized defaults:

```bash
GPUREC_FORWARD_PARENT_REDUCED_DTS=1 \
GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS=0 \
GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL=tiled \
GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS=64 \
GPUREC_FORWARD_LEAF_INDEX=1 \
GPUREC_FORWARD_TOPOLOGY_INT32=1 \
GPUREC_FORWARD_DTS_OVERLAP_MODE=off \
.venv/bin/python profiling/bench_uniform_forward_parent_dts.py \
  --dataset tests/data/test_trees_1000 \
  --fams 50 \
  --fixed-iters 6 \
  --dtype fp32 \
  --no-need-pibar \
  --root-rows
```

Nsys profiling used the same benchmark shape with one profiled repetition, for
example:

```bash
nsys profile --force-overwrite=true \
  -o /tmp/prop3_default_fp32_f50 \
  .venv/bin/python profiling/bench_uniform_forward_parent_dts.py \
    --dataset tests/data/test_trees_1000 \
    --fams 50 \
    --fixed-iters 6 \
    --dtype fp32 \
    --no-need-pibar \
    --root-rows \
    --reps 1 \
    --warmups 1
```

The Triton scratch prototype was run from:

```bash
.venv/bin/python /tmp/gpurec_row_prefix_pibar_proto.py
```

#### Baseline profiles

Worker B profiled the current optimized path with global uniform mode,
fixed-6 Pi iterations, `need_pibar=False`, `root_rows=True`, DTS overlap off,
parent-reduced DTS on, leaf-index on, and int32 topology on.

50-family fp32 shape:

| Quantity | Value |
|---|---:|
| `S` | `1999` |
| `G` | `50` |
| clades `C` | `321930` |
| waves | `49` |
| max wave size | `32768` |
| split rows | `402275` |
| first waves | `k0=32768` leaf-only, `k1=32768` leaf-only, `k2=15009` leaf-only, `k3=27023` internal split rows |

50-family fp32 likelihood-only interval:

| Variant | Median | Peak GPU | NLL |
|---|---:|---:|---:|
| default | `116.435 ms` | `5.256 GiB` | `107804.265625` |
| ancestor | `159.383 ms` | `5.458 GiB` | `107804.265625` |
| csr | `169.739 ms` | `5.458 GiB` | `107804.265625` |
| two_kernel | `204.512 ms` | `10.25 GiB` | `107804.265625` |
| linear | `413.727 ms` | `10.25 GiB` | `107804.265625` |

50-family fp32 Nsys default:

| Kernel bucket | Time | Launches | Share |
|---|---:|---:|---:|
| `_wave_step_uniform_kernel` | `76.998 ms` | `294` | `66.5%` |
| `_wave_pibar_uniform_parent_kernel` | `10.306 ms` | `48` | `8.9%` |
| `_dts_fused_kernel` | `9.703 ms` | `39` | `8.4%` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `7.521 ms` | `7` | `6.5%` |

Representative 50-family fp32 NCU launches:

| Launch | Duration | Throughput | DRAM | L1/TEX | L2 hit | Occupancy | Registers | Spills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default leaf wave `k0`, `W=32768` | `1.32 ms` | compute/memory `82.13%` | `38.05%` | `82.41%` | `92.01%` | `98.16%` | `40` | `0` |
| final parent-Pibar, `W=32768` | `1.10 ms` | compute/memory `83.58%` | `45.72%` | hit `93.83%` | `72.93%` | `98.00%` | `34` | `0` |

For fp32, the optimized Triton kernels are already high-occupancy and
throughput-heavy.  The existing ancestor/csr/two-kernel/linear variants all
regress in full forward timing.

10-family fp64 likelihood-only interval:

| Variant | Median | Peak GPU | NLL read |
|---|---:|---:|---|
| default | `1219.896 ms` | `2.172 GiB` | baseline |
| ancestor | `1101.922 ms` | `2.483 GiB` | delta `1.381e-4` |
| csr | `1102.028 ms` | `2.483 GiB` | delta `1.381e-4` |
| two_kernel | `1104.638 ms` | `4.46 GiB` | delta `1.381e-4` |
| linear | `1236.321 ms` | `4.46 GiB` | exact default NLL |

10-family fp64 Nsys default:

| Kernel bucket | Time | Launches | Share |
|---|---:|---:|---:|
| `_wave_step_uniform_kernel` | `1065.183 ms` | `270` | `85.5%` |
| `_wave_pibar_uniform_parent_kernel` | `132.761 ms` | `44` | `10.7%` |
| DTS combined | about `43 ms` | - | - |

10-family fp64 NCU default leaf wave: duration `51.27 ms`, compute throughput
`83.88%`, memory throughput `1.39%`, DRAM `0.99%`, fp64 pipeline bottleneck,
occupancy `81.16%`, `48` registers/thread, and no spills.  This confirms that
fp64 is scalar-pipeline limited, not memory-bandwidth limited.

#### Triton scratch prefix prototype

Worker C wrote a temporary scratch script:
`/tmp/gpurec_row_prefix_pibar_proto.py`.  No tracked files were edited.

The prototype decomposes Pibar into multiple Triton launches:

```text
1. compute row stats
2. store exp scratch [W, S]
3. launch one level kernel per species depth
4. launch final Pibar kernel
```

For this topology, `max_depth = n_levels = 23` and `max_level_width = 242`.

Parity against the parent-walk Pibar path:

| dtype | Max abs error |
|---|---:|
| fp32 | `0` to `9.54e-7` |
| fp64 | `8.88e-16` to `1.78e-15` |

Pibar-only median timings:

| dtype | `W` | Parent walk | Triton prefix | Result |
|---|---:|---:|---:|---|
| fp32 | `1` | `0.0316 ms` | `0.3931 ms` | prefix slower |
| fp32 | `512` | `0.0365 ms` | `0.3881 ms` | prefix slower |
| fp32 | `2048` | `0.0836 ms` | `0.4628 ms` | prefix slower |
| fp64 | `1` | `0.2785 ms` | `0.4260 ms` | prefix slower |
| fp64 | `512` | `1.0428 ms` | `0.5501 ms` | prefix faster |
| fp64 | `2048` | `4.1206 ms` | `1.1315 ms` | `3.64x` faster |

Nsys for fp64 `W=2048` showed why this is not production-ready:
the parent path was `6` launches for `24.64 ms` total, while the prefix path
spent `2.44 ms` in init kernels, `1.61 ms` in final kernels, and `0.845 ms` in
`138` level-kernel launches.  The math is promising for fp64, but the launch
structure is unsuitable for the real forward loop.

#### CUDA shared-memory Pibar-only scratch

A local inline NVRTC prototype tested one CUDA block per Pi row with dynamic
shared memory for a single exp/prefix row.  It uses the child-before-parent
species-id property:

```cuda
// prefix initially stores exp2(Pi[row, s] - row_max)
__syncthreads();
if (threadIdx.x == 0) {
    for (int s = S - 1; s >= 0; --s) {
        int p = parent[s];
        if (p >= 0) {
            prefix[s] += prefix[p];
        }
    }
}
__syncthreads();
// all threads write Pibar[row, s]
```

This prototype computes Pibar only.  It does not fuse DTS_L loads, D/S/T terms,
or the final logsumexp update.

Parity against `wave_pibar_uniform_parent_fused`:

| dtype | Max abs error | Read |
|---|---:|---|
| fp32 | `3.81e-6` | allclose under scratch tolerance |
| fp64 | `7.11e-15` | allclose under scratch tolerance |

CUDA event medians:

| dtype | `W` | Parent walk | Shared prefix | Result |
|---|---:|---:|---:|---|
| fp32 | `1` | `0.0307 ms` | `0.0666 ms` | `2.17x` slower |
| fp32 | `512` | `0.0358 ms` | `0.0696 ms` | `1.94x` slower |
| fp32 | `2048` | `0.0819 ms` | `0.1792 ms` | `2.19x` slower |
| fp32 | `8192` | `0.2721 ms` | `0.6458 ms` | `2.37x` slower |
| fp64 | `1` | `0.2774 ms` | `0.1207 ms` | `2.30x` faster |
| fp64 | `512` | `1.0414 ms` | `0.1986 ms` | `5.24x` faster |
| fp64 | `2048` | `4.1185 ms` | `0.7772 ms` | `5.30x` faster |
| fp64 | `8192` | `16.4274 ms` | `2.6880 ms` | `6.11x` faster |

Shared memory per block was one row only: `7.81 KiB` for fp32 and `15.62 KiB`
for fp64.  The prototype uses no global scratch allocation.  The single-thread
descending recurrence is serial inside each row, but for fp64 it still wins
because it avoids the repeated scalar fp64 parent-walk work.

#### Interpretation

The two prototypes agree on the main technical result:

- row-prefix Pibar is a bad fp32 tradeoff in the current forward path;
- row-prefix Pibar is promising for fp64, especially for larger `W`;
- avoiding global scratch and many level launches matters;
- the remaining hard problem is fusing the prefix result into the DTS_L
  wave-step update without losing occupancy to shared memory, barriers, and
  additional per-row scratch.

The current fp32 baseline already has `98%` achieved occupancy and no spills in
the representative leaf and final-Pibar kernels.  Replacing the parent walk
with a shared-memory row prefix adds synchronization and shared-memory
footprint, while removing work that is not yet expensive enough in fp32.  That
is why the CUDA Pibar-only prefix is `2.0-2.4x` slower in fp32.

The fp64 situation is different.  NCU shows the default fp64 leaf wave at
`51.27 ms`, with DRAM almost idle and the scalar fp64 pipeline saturated.  The
shared-memory prefix removes repeated fp64 ancestor-walk arithmetic and gives a
`2.3-6.1x` Pibar-only speedup.  This supports an fp64-specific Pibar/final-Pibar
follow-up, but not a general fp32 production rewrite.

#### Decision

Do not promote a production fp32 path from Proposal 3.

The original acceptance gate was at least `5%` faster on the
`_wave_step_uniform_kernel` bucket for 150-family chunks.  That gate was not
met because the implemented CUDA scratch was Pibar-only and no full fused
DTS_L CUDA wave-step kernel was built.  The fp32 Pibar-only CUDA prefix also
regressed relative to the current Triton parent-walk Pibar kernel.

Keep the result as evidence for two narrower future directions:

- fp64 Pibar-only or final-Pibar replacement, where the shared-memory CUDA
  prefix showed `2.3-6.1x` Pibar-only speedups without global scratch;
- a later full DTS_L fused CUDA prototype only if it can keep shared-memory
  footprint low enough to preserve occupancy and demonstrate full wave-step
  bucket improvement, not just Pibar microkernel speedup.

Artifacts:

```text
/tmp/gpurec_row_prefix_pibar_proto.py
/tmp/gpurec_row_prefix_fp32.log
/tmp/gpurec_row_prefix_fp64.log
/tmp/gpurec_row_prefix_w2048_fp64.nsys-rep
/tmp/gpurec_row_prefix_w2048_fp64_kernels.txt
/tmp/prop3_default_fp32_f50.nsys-rep
```

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

### Proposal 4 follow-up: implemented as opt-in, not promoted

This proposal was tested with the same three-worker plus supervisor workflow:
one worker checked dependency safety, one checked correctness, one profiled the
CUDA timeline, and the supervisor consolidated the result. The implementation
was kept opt-in because it changes stream ordering and can retain future `dts_r`
buffers longer than the serialized path.

Implementation:

- `build_wave_layout(...)` now stores `dts_ready_after[j]` and
  `dts_overlap_gap[j]` in each wave meta and in the layout dictionary.
- `_ensure_dts_ready_after(...)` backfills those fields for older cached
  layouts.
- `Pi_wave_forward(...)` recognizes `GPUREC_FORWARD_DTS_OVERLAP_MODE`:
  `off`, `next`, and `ready`.
- `GPUREC_FORWARD_DTS_OVERLAP_MAX_PENDING` bounds how many future DTS buffers
  can be live at once.
- `profiling/bench_uniform_forward_parent_dts.py` exposes the same controls as
  `--overlap-mode` and `--overlap-max-pending`, and prints readiness diagnostics
  in the shape summary.

The readiness rule is:

```text
dts_ready_after[j] =
    max(wave_index(left_child), wave_index(right_child)) over splits in wave j

DTS(j) may be launched while computing wave k only if:
    dts_ready_after[j] < k
```

The strict `< k` is important. Cross-DTS reads both child `Pi` and child
`Pibar`; with the fixed-iteration ping-pong path, a child's final `Pibar` is
only guaranteed after that child wave finishes its final Pibar recompute. The
root-only `need_pibar=False` skip remains safe because root-only rows have no
later cross-DTS consumers.

Readiness accounting:

| Workload | Waves | DTS waves | Split rows | Ready-early DTS waves | Ready-early split rows | Max ready gap |
|---|---:|---:|---:|---:|---:|---:|
| 3 families | `45` | `44` | `23393` | `0` | `0` | `0` |
| 50 families | `49` | `46` | `402275` | `0` | `0` | `0` |
| 150 families | `65` | `57` | `1192970` | `15` | `334349` | `4` |

This explains why 3-family and 50-family chunks cannot benefit from this
scheduler: every DTS wave becomes ready only immediately before its consumer.
The 150-family chunk has some theoretical window:

```text
wave 8:  ready after 3, gap 4, split rows 32768
wave 9:  ready after 5, gap 3, split rows 32768
wave 10: ready after 7, gap 2, split rows 14542
wave 11: ready after 9, gap 1, split rows 32768
```

Correctness evidence:

| Check | Result |
|---|---:|
| Direct 3-family `next` vs `off` NLL delta | `0.0` |
| Direct 3-family `next` vs `off` significant `Pi` max abs delta | `0.0` |
| Direct 3-family `ready` vs `off` NLL delta | `0.0` |
| Direct 3-family `ready` vs `off` significant `Pi` max abs delta | `0.0` |
| Direct 3-family `ready`, `need_pibar=False`, NLL delta | `0.0` |
| Direct 3-family `ready`, `need_pibar=False`, significant `Pi` max abs delta | `0.0` |
| Direct 3-family `ready`, `need_pibar=False`, returned Pibar | `None` |
| Focused forward/backward smoke tests | `35 passed in 11.95 s` |
| Model API likelihood tests | `3 passed in 46.14 s` |
| `py_compile` on touched Python files | passed |

Timing, 50 families, fixed 6, no Pibar output, `7` timed reps:

| Mode | Pending cap | Median ms | Mean ms | Min ms | Max ms | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| `off` | n/a | `116.169` | `116.260` | `115.800` | `117.056` | `5.272` |
| `next` | n/a | `116.827` | `116.713` | `115.975` | `117.436` | `5.272` |
| `ready` | `2` | `116.832` | `116.762` | `116.108` | `117.169` | `5.272` |
| `ready` | `4` | `116.400` | `116.433` | `115.579` | `117.469` | `5.272` |

Timing, 150 families, fixed 6, no Pibar output, `5` timed reps:

| Mode | Pending cap | Median ms | Mean ms | Min ms | Max ms | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| `off` | n/a | `344.612` | `344.595` | `343.136` | `346.139` | `15.018` |
| `next` | n/a | `345.573` | `345.353` | `343.806` | `346.492` | `15.018` |
| `ready` | `1` | `344.196` | `344.225` | `342.511` | `346.076` | `15.018` |
| `ready` | `2` | `345.135` | `356.412` | `343.251` | `403.293` | `15.194` |
| `ready` | `4` | `346.032` | `357.296` | `344.510` | `404.438` | `15.194` |

The best result was `ready` with a pending cap of `1`, but its apparent
`0.416 ms` median improvement over `off` is within run-to-run noise. Higher
pending caps increased peak memory by about `0.176 GiB` and introduced large
latency outliers because more future DTS outputs stayed live.

Nsys, 150 families, one timed repetition after warmup:

| Mode | Event time ms | Kernel span ms | Main stream busy ms | Prep stream busy ms | Main/prep overlap ms |
|---|---:|---:|---:|---:|---:|
| `off` | `341.569` | `341.470` | `341.470` | `0.000` | `0.000` |
| `ready`, cap `1` | `344.518` | `344.418` | `271.920` | `72.290` | `0.462` |

Main kernel buckets in the `off` trace:

| Kernel bucket | Time ms | Launches | Stream |
|---|---:|---:|---:|
| `_wave_step_uniform_kernel` | `222.662` | `390` | `7` |
| `_dts_fused_kernel` | `30.876` | `49` | `7` |
| `_wave_pibar_uniform_parent_kernel` | `30.029` | `64` | `7` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `23.490` | `8` | `7` |
| PyTorch indexing kernels | `12.265` | `50` | `7` |

Main kernel buckets in the `ready`, cap `1` trace:

| Kernel bucket | Time ms | Launches | Stream |
|---|---:|---:|---:|
| `_wave_step_uniform_kernel` | `225.222` | `390` | `7` |
| `_dts_fused_kernel` | `30.895` | `49` | `29` |
| `_wave_pibar_uniform_parent_kernel` | `30.699` | `64` | `7` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `23.493` | `8` | `29` |
| PyTorch indexing kernels | `12.245` | `49` | `29` |
| Prep-stream fill kernels | `4.818` | `57` | `29` |

The timeline confirms the practical bottleneck: the readiness-aware scheduler
does move DTS work to a second stream, but only `0.462 ms` of prep-stream work
actually overlaps the main stream. The dominant `_wave_step_uniform_kernel`
bucket already occupies the GPU heavily, so the memory-bound DTS kernels do not
find enough spare bandwidth or scheduling slots to hide meaningful time.

Decision:

- Keep readiness diagnostics and `ready` mode as opt-in profiling machinery.
- Keep `off` as the default.
- Do not use pending caps above `1` unless the buffer lifetime policy changes.

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

### Proposal 5 follow-up: clade-budget harness, memory win only

Proposal 5 was tested with the same static-review, correctness, profiling, and
supervisor split. This pass added a profiling harness rather than changing the
training API:

```text
profiling/bench_uniform_forward_chunking.py
```

The harness loads `test_trees_1000` once, solves the global uniform `E` fixed
point once, builds one wave layout per resident chunk, then times the full
Pi/root-likelihood sweep over all chunks. It supports:

- fixed family chunks, matching the old 150-family policy;
- order-preserving clade-budget chunks;
- a conservative `--auto-budget` mode based on free memory;
- `max_wave_size` and `max_root_wave_size` sweeps;
- Nsys capture through `--profile-cuda-api`.

The public API still has two separate behaviors:

- `GeneReconModel.forward()` builds one monolithic static layout and is the
  backward-capable path. It should not silently chunk because backward needs the
  saved `Pi/Pibar`.
- `GeneDataset.compute_likelihood_batch()` already chunks by family count, but
  each chunk re-extracts parameters and re-solves `E`. For global/specieswise
  uniform inference, reusing one shared `E` is valid if the theta is truly
  shared.

The tested configuration was fp32, global uniform mode, fixed 6 Pi iterations,
parent-reduced DTS enabled, int32 forward topology enabled, `need_pibar=False`,
and DTS overlap disabled.

Workstream summary:

| Workstream | Result |
|---|---|
| Static/API review | Clade-budget inference is safe as an order-preserving chunk planner plus shared-E loop. Do not change autograd forward silently. |
| Correctness | Fixed-family and clade-budget chunks matched unchunked small cases exactly. Existing model/wave tests passed. |
| Profiling | Clade budgeting reduces peak memory, but does not produce a robust speedup. Larger chunks and larger waves hit DTS scratch/stability limits. |

#### Correctness evidence

| Check | Result |
|---|---:|
| 10 families, two 5-family chunks vs unchunked | NLL abs diff `0.0` |
| 50 families, two 25-family chunks vs unchunked | NLL abs diff `0.0` |
| Worker custom 6-family fixed chunks vs unchunked | scalar diff `0`, max per-family diff `0` |
| Worker custom 6-family clade-budget chunks vs unchunked | scalar diff `0`, max per-family diff `0` |
| Permuted 6-family order with chunking | output order preserved, max diff `0` |
| `pytest -q tests/unit/test_cross_family_wave.py::test_batched_wave_matches_individual_large_s` | `3 passed in 12.90 s` |
| `pytest -q tests/unit/test_wave_v2.py::test_model_api_wave_matches_fp` | `1 passed in 4.63 s` |
| `pytest -q tests/gradients/test_autograd_bridge.py::test_model_nll_matches_compute_likelihood_batch` | `5 passed in 1.36 s` |
| `pytest -q tests/unit/test_cross_family_wave.py::test_batched_wave_100_families_large_s` | `1 passed in 47.34 s` |

The full 1000-family scalar NLL varies by up to `0.25` bits between some chunk
policies because the chunked script accumulates fp32 chunk sums in a different
order. The relative difference is about `1.2e-7`; per-family/order checks on
small cases were exact.

#### Harness validation

The new harness was cross-checked against the existing parent-DTS benchmark on
the first 150 families:

| Path | Median Pi/root interval | NLL | Peak GPU |
|---|---:|---:|---:|
| `bench_uniform_forward_parent_dts.py`, 150 families | `344.309 ms` | `323018.6875` | `15.018 GiB` |
| `bench_uniform_forward_chunking.py`, one 150-family chunk | `345.297 ms` | `323018.6875` | `15.018 GiB` |

The `~1 ms` difference is timing noise plus a small scalar-accumulation wrapper;
the harness is representative of the existing forward interval.

#### 1000-family chunk policy sweep

All rows use `max_wave_size=32768`, warmed allocator timing, `3` timed reps, and
one warmup. `C_max` is the largest resident clade count.

| Policy | Chunks | `C_max` | Total waves | Median ms | Peak GiB | NLL |
|---|---:|---:|---:|---:|---:|---:|
| fixed `150` families | `7` | `979570` | `453` | `2319.010` | `15.679` | `2157097.0` |
| clade budget `950000` | `7` | `949705` | `456` | `2323.972` | `15.230` | `2157097.0` |
| clade budget `900000` | `8` | `899877` | `500` | `2335.058` | `14.487` | `2157097.0` |
| conservative auto budget, `891353` clades | `8` | `890072` | `499` | `2319.109` | `14.343` | `2157097.25` |
| clade budget `1000000` | `7` | `999058` | `448` | `2499.909` | `15.969` | `2157097.0` |
| clade budget `1050000` | `7` | `1049674` | `451` | `2335.260` | `16.720` | `2157097.0` |
| clade budget `1100000` | `6` | `1098643` | `411` | failed | - | - |
| clade budget `1200000` | `6` | `1198398` | `425` | failed | - | - |

The accepted part of the proposal is memory stability, not speed. A
`900k-950k` clade budget cuts peak allocation by about `0.45-1.19 GiB` with
only `0.2-0.7%` timing cost. The conservative auto policy picked about `891k`
clades on the RTX 4090 and matched the fixed-150 median while reducing peak
memory by about `1.34 GiB`.

The rejected part is “larger stable chunks”. Budgets at `1.1M` and `1.2M`
clades failed during warmup with a Triton/CUDA illegal memory access in the DTS
path. This is consistent with the earlier 175-family instability. A production
packer therefore needs a DTS scratch/fanout guard in addition to the
`2*C*S*sizeof(dtype)` Pi/Pibar estimate.

#### Wave-size policy

Whole-1000 timing with fixed 150-family chunks:

| `max_wave_size` | Chunks | Total waves | Max `W` | Median ms | Peak GiB | Result |
|---:|---:|---:|---:|---:|---:|---|
| `32768` | `7` | `453` | `32768` | `2319.010` | `15.679` | current stable point |
| `65536` | `7` | `368` | `65536` | `3180.165` | `16.236` | slower |
| `131072` | `7` | `349` | `131072` | failed | - | DTS scratch OOM |
| uncapped | `7` | `342` | `245080` | `3193.939` | `16.297` | slower |

The earlier intuition that forward-only might prefer larger waves did not hold
for the full 1000-family sweep after the current parent-reduced/need-Pibar
changes. Larger waves reduce launch count, but they increase per-launch working
set and scratch pressure; `32768` remains the right default.

#### Nsight Systems

Nsys captures were restricted to one warmed timed repetition. Device copies
were negligible in all representative traces: `8-9` copies, `32-36` bytes, about
`0.005 ms` total. All kernels ran on one CUDA stream.

| Metric | fixed `150` | clade `950k` | clade `900k` |
|---|---:|---:|---:|
| Kernel span ms | `2507.159` | `2314.534` | `2323.195` |
| Summed kernel ms | `2501.619` | `2309.747` | `2318.104` |
| Kernel launches | `5919` | `5952` | `6581` |
| Peak GiB in non-Nsys timing | `15.679` | `15.230` | `14.487` |

Top kernel buckets:

| Kernel bucket | fixed `150` | clade `950k` | clade `900k` |
|---|---:|---:|---:|
| `_wave_step_uniform_kernel` | `1639.133 ms / 2718` | `1511.665 ms / 2736` | `1523.250 ms / 3000` |
| `_dts_fused_kernel` | `226.474 ms / 342` | `207.703 ms / 344` | `206.591 ms / 385` |
| `_wave_pibar_uniform_parent_kernel` | `215.294 ms / 446` | `205.751 ms / 449` | `204.451 ms / 492` |
| `_dts_parent_reduced_ge2_stage1_kernel` | `175.599 ms / 58` | `157.445 ms / 58` | `156.977 ms / 64` |
| PyTorch index kernels | `90.069 ms / 349` | `82.226 ms / 351` | `81.998 ms / 393` |

The Nsys picture matches the timing sweep qualitatively: lower clade budgets
shrink the largest chunk and can reduce the largest kernel buckets, but once the
budget gets too small the extra waves and launches eat the gain. The sweet spot
is a memory/robustness choice around `900k-950k`, not a throughput breakthrough.

Artifacts:

```text
/tmp/gpurec_profile/forward_prop5/chunk_fixed150.nsys-rep
/tmp/gpurec_profile/forward_prop5/chunk_fixed150.sqlite
/tmp/gpurec_profile/forward_prop5/chunk_clade950k.nsys-rep
/tmp/gpurec_profile/forward_prop5/chunk_clade950k.sqlite
/tmp/gpurec_profile/forward_prop5/chunk_clade900k.nsys-rep
/tmp/gpurec_profile/forward_prop5/chunk_clade900k.sqlite
```

#### Decision

Keep `max_wave_size=32768`. Do not promote `65536`, `131072`, or uncapped waves.

Keep fixed 150-family chunks as the throughput baseline when memory is
available. For inference deployments that need a safer memory envelope, use an
order-preserving clade budget around `900k-950k` clades on this RTX 4090
workload. This loses little or no throughput and saves roughly `0.45-1.34 GiB`
of peak allocation.

The long-term API change is still worthwhile: add a public likelihood-only
chunked inference path that reuses global/specieswise `E`, accumulates in fp64
or host `math.fsum` for stable scalar reporting, and chooses chunks by clades
plus a DTS scratch/fanout guard. Do not route autograd `forward()` through this
path without a separate backward design.

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

### Proposal 6 follow-up: root-row output mode

The proposal was tested with the same split used for the previous optimization
passes: one static/API audit, one correctness pass, one profiler pass, and a
supervisor pass that merged the findings into this document. The useful part of
the proposal is not an internal row-eviction rewrite yet. It is a narrower
root-row output mode for likelihood-only paths.

#### Implementation

`Pi_wave_forward(...)` now has a second output-lifetime switch:

```python
Pi_wave_forward(..., return_original=False, return_root_rows=True)
```

When `return_root_rows=True`, the forward pass gathers only:

```python
Pi_root_rows = Pi[wave_layout["root_clade_ids"]]
```

and does not return the full dense `Pi_wave_ordered` tensor. The likelihood can
then be computed directly:

```python
numerator = logsumexp2(Pi_root_rows, dim=-1) - log2(S)
denominator = log2(1 - exp2(E).mean(dim=-1))
nll = -(numerator - denominator)
```

This is implemented as `compute_log_likelihood_root_rows(...)`. The full
wave-ordered tensors are still returned by default, and they are still retained
for autograd/backward paths that need `Pi_wave_ordered`, `Pibar_wave_ordered`,
and the Pibar row maxima.

Call-site policy after the pass:

| Caller | Policy | Reason |
|---|---|---|
| `GeneDataset.compute_likelihood_batch` | `return_original=False`, `need_pibar=False`, `return_root_rows=True` | pure likelihood-only inference |
| genewise NLL-only evaluation | `return_original=False`, `need_pibar=False`, `return_root_rows=True` | pure likelihood-only inference |
| `wave_optimizer` forward/backward | `return_original=False`, full wave output kept | backward needs wave-ordered saved tensors |
| genewise mini-batch training | `return_original=False`, full wave output kept | backward needs wave-ordered saved tensors |
| implicit per-family gradient helper | `return_original=False`, full wave output kept | original-order copy was unused |
| direct/debug callers | defaults preserve old behavior | avoids API breakage |

#### Correctness evidence

The root-row mode is intentionally output-only: it changes which tensors survive
the call, not the recurrence. The dedicated unit test checks that root-row
output equals the corresponding rows of full `Pi_wave_ordered`, and that both
likelihood formulas are bitwise identical for the tested case.

| Check | Result |
|---|---:|
| `test_batched_wave_root_rows_match_full_wave_output` | passed |
| `test_model_nll_matches_compute_likelihood_batch` | passed |
| genewise wave batch/per-family parity | passed |
| model API wave vs fixed-point parity | passed |
| integration likelihood helper | passed |
| genewise optimization smoke test | passed |
| 10-family chunked root-row vs unchunked full-output NLL | `abs_diff = 0.0` |
| 3-family parent-DTS compare, root rows | `abs_diff = 0.0` |

Commands used:

```text
pytest -q tests/unit/test_cross_family_wave.py::test_batched_wave_root_rows_match_full_wave_output tests/gradients/test_autograd_bridge.py::test_model_nll_matches_compute_likelihood_batch
pytest -q tests/unit/test_genewise_wave.py::test_genewise_wave_batch_vs_per_family tests/unit/test_wave_v2.py::test_model_api_wave_matches_fp
pytest -q tests/integration/test_gene_recon_model.py::test_log_likelihood_helper tests/unit/test_optimize_genewise.py::test_nll_decreases
python profiling/bench_uniform_forward_chunking.py --fams 10 --family-chunk-size 5 --reps 1 --warmups 0 --compare-unchunked-max-fams 20 --root-rows
python profiling/bench_uniform_forward_parent_dts.py --fams 3 --reps 1 --warmups 0 --no-need-pibar --root-rows --compare
```

#### Timing and memory

Measured with fp32 global uniform mode, fixed 6 Pi iterations, proposal 0/1/2
defaults enabled, `max_wave_size=32768`, DTS overlap off, and
`need_pibar=False`. `live_after_last_gib` is measured immediately after the
timed forward returns and before the benchmark releases the returned output.

| Workload | Output mode | Median forward | NLL | Peak GPU | Live after return |
|---|---|---:|---:|---:|---:|
| 50 families | full wave output | `117.296 ms` | `107804.265625` | `5.272 GiB` | `2.431 GiB` |
| 50 families | root rows | `116.902 ms` | `107804.265625` | `5.272 GiB` | `0.034 GiB` |
| 150 families | full wave output | `346.679 ms` | `323018.6875` | `15.018 GiB` | `7.176 GiB` |
| 150 families | root rows | `346.812 ms` | `323018.6875` | `15.018 GiB` | `0.068 GiB` |
| 1000 families, 150-family chunks | full wave output | `2336.797 ms` | `2157097.0` | `15.679 GiB` | per-chunk full output retained during likelihood |
| 1000 families, 150-family chunks | root rows | `2331.556 ms` | `2157097.0` | `15.679 GiB` | per-chunk root rows retained during likelihood |

The timing difference is within run noise. That is expected because the root
rows are gathered either way: full-output likelihood gathers them inside
`compute_log_likelihood(...)`, while root-row mode gathers them inside
`Pi_wave_forward(...)`. The improvement is memory liveness. At 150 families the
returned resident output drops by about `7.108 GiB`, which is essentially one
dense fp32 `[C, S]` matrix:

```text
954706 clades * 1999 species * 4 bytes = 7.11 GiB
```

Peak memory is unchanged because the dense internal `Pi` and the current-wave
scratch are still needed while the recurrence is running. Root-row mode only
controls what survives the call.

#### Nsight Systems

Nsys was run on 150 families with one profiled repetition after warmup. The
kernel profile is intentionally almost identical, which confirms that this
change is an output-lifetime optimization rather than a compute-kernel rewrite.

| Metric | Full wave output | Root rows |
|---|---:|---:|
| CUDA-event time | `344.644 ms` | `344.947 ms` |
| kernel span | `344.529 ms` | `344.848 ms` |
| summed kernel time | `343.887 ms` | `344.209 ms` |
| kernel launches | `845` | `845` |
| CUDA streams with kernels | `1` | `1` |
| device memcpy | `2 copies, 8 bytes, 0.0017 ms` | `2 copies, 8 bytes, 0.0019 ms` |

Top kernel buckets:

| Kernel bucket | Full wave output | Root rows | Interpretation |
|---|---:|---:|---|
| `_wave_step_uniform_kernel` | `224.984 ms / 390` | `225.457 ms / 390` | recurrence unchanged |
| `_dts_fused_kernel` | `30.904 ms / 49` | `30.894 ms / 49` | DTS unchanged |
| `_wave_pibar_uniform_parent_kernel` | `30.691 ms / 64` | `30.616 ms / 64` | Pibar update unchanged |
| `_dts_parent_reduced_ge2_stage1_kernel` | `23.514 ms / 8` | `23.453 ms / 8` | parent DTS unchanged |
| PyTorch index kernels | `12.271 ms / 50` | `12.270 ms / 50` | root gather moved, not removed |
| fill kernels | `8.701 ms / 61` | `8.697 ms / 61` | unchanged initialization/setup |

The single-stream profile also shows no new overlap opportunity introduced by
root-row mode. The hot path is still dominated by the wave-step recurrence, then
DTS/Pibar kernels.

#### Liveness rewrite interpretation

The larger row-liveness idea remains future work. A safe version would need at
least:

```text
last_use[row] = max(wave_index(parent) for every split where row is a child)
after wave k:
    rows with last_use[row] <= k can be evicted
```

Current DTS kernels gather children by dense clade ids:

```text
Pi[left_child, species], Pi[right_child, species]
Pibar[left_child, species], Pibar[right_child, species]
```

Compacting live rows would require either an extra
`global_clade_id -> live_row_id` indirection in every split gather, or a
rewritten scheduler that emits compact split metadata per wave. This can reduce
peak memory, unlike root-row mode, but it is a larger algorithmic change and may
trade memory savings for worse gather locality.

#### Decision

Keep root-row mode for likelihood-only inference paths. It does not make one
forward chunk materially faster, but it removes multi-GiB returned tensors from
the lifetime of large chunks and makes subsequent batching/chunk scheduling
less fragile. Keep full wave output for autograd/backward until backward has a
separate saved-tensor strategy.

Artifacts:

```text
/tmp/gpurec_profile/forward_prop6/rootrows0_f150.nsys-rep
/tmp/gpurec_profile/forward_prop6/rootrows0_f150.sqlite
/tmp/gpurec_profile/forward_prop6/rootrows1_f150.nsys-rep
/tmp/gpurec_profile/forward_prop6/rootrows1_f150.sqlite
```

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

### Proposal 7 follow-up: implemented genewise leaf-index path

Proposal 7 was implemented as a real genewise uniform forward optimization.
The old genewise uniform path disabled `leaf_species_index` whenever
`family_idx` was present, so it fell back to dense per-wave leaf masks.  That
fallback is expensive for large `S`: each wave builds or touches dense `[W, S]`
leaf data even though each leaf row has only one finite species entry.

Implementation:

- `use_uniform_leaf_index` in `gpurec/core/forward.py` no longer excludes
  batched/genewise mode.  It is controlled by `GPUREC_FORWARD_LEAF_INDEX`
  defaulting to on, plus the existing exclusions for the linear, two-kernel,
  and SpMM paths.
- `_wave_step_uniform_kernel` in `gpurec/core/kernels/wave_step.py` now takes
  `family_idx_ptr` plus `LEAF_LOGP_MODE`.
- `LEAF_LOGP_MODE=0` handles shared `[S]` leaf log-probabilities.
- `LEAF_LOGP_MODE=1` handles genewise scalar `[G]` leaf log-probabilities.
- `LEAF_LOGP_MODE=2` handles genewise specieswise `[G, S]` leaf
  log-probabilities.
- The fused, ping-pong `fused_into`, ancestor, and CSR uniform wrappers pass
  `family_idx` through to the Triton kernel.
- The two-kernel and linear uniform paths keep the dense-leaf fallback.
- `profiling/bench_uniform_forward_parent_dts.py` now exposes `--mode` and
  `--leaf-index` so genewise leaf addressing can be benchmarked directly.

The fused kernel now computes the leaf term in the compact path:

```text
mode 0: leaf = leaf_logp[s]
mode 1: leaf = leaf_logp[family_idx[row]]
mode 2: leaf = leaf_logp[family_idx[row], s]

t_leaf = leaf if leaf_species[row] == s else -inf
```

Correctness evidence:

| Check | Result |
|---|---:|
| `test_wave_step_uniform_leaf_index_logp_modes_match_dense_leaf_term` for `[S]`, `[G]`, and `[G,S]` | `3 passed in 1.46 s` |
| Genewise leaf-index vs dense fallback tests for `[G]` and `[G,S]` | passed as part of `test_genewise_wave.py` |
| Genewise gradient bridge parity for supported `[G]` mode | passed |
| `tests/unit/test_genewise_wave.py` plus `tests/gradients/test_autograd_bridge.py::test_model_nll_matches_compute_likelihood_batch` | `14 passed in 6.45 s` |

The forward `[G,S]` path is covered against the dense leaf fallback.  The
remaining public API gap is that the model gradient bridge still does not
support combined genewise+specieswise training end to end, so `[G,S]` is
forward-tested but not promoted as a full training mode.

Timing, genewise uniform fp32, fixed 6, `test_trees_1000`, root-row output,
`need_pibar=False`, parent-reduced DTS on, int32 topology on:

| Workload | Leaf index | Ping-pong | Median forward | Peak GPU | NLL | Outcome |
|---:|---|---|---:|---:|---:|---|
| 10 families | disabled | default | `63.889 ms` | `2.793 GiB` | `22182.914062` | dense leaf fallback |
| 10 families | enabled | default | `51.062 ms` | `2.131 GiB` | `22182.914062` | `-12.827 ms`, `-0.662 GiB` |
| 25 families | disabled | default | `150.287 ms` | `5.845 GiB` | `54262.445312` | dense leaf fallback |
| 25 families | enabled | default | `114.116 ms` | `4.257 GiB` | `54262.445312` | `-36.171 ms`, `-1.588 GiB` |
| 50 families | disabled | default | `297.376 ms` | `10.646 GiB` | `107804.265625` | baseline |
| 50 families | enabled | default | `224.115 ms` | `7.761 GiB` | `107804.265625` | `-73.261 ms`, `-2.885 GiB` |
| 150 families | disabled | default | - | - | - | OOM on first timed rep |
| 150 families | enabled | default | `664.441 ms` | `17.959 GiB` | `323018.6875` | fits |

The 150-family disabled run reached `23.14 GiB` process memory and failed on a
`250 MiB` allocation.  The compact leaf path therefore changes memory scaling,
not only latency.

Isolating compact leaf addressing from ping-pong on 50 families:

| Leaf index | `GPUREC_UNIFORM_PINGPONG` | Median forward | Peak GPU |
|---|---:|---:|---:|
| disabled | `0` | `297.016 ms` | `10.646 GiB` |
| enabled | `0` | `265.583 ms` | `8.005 GiB` |
| enabled | default | `224.115 ms` | `7.761 GiB` |

So compact leaf addressing itself saves about `31.4 ms` and `2.64 GiB` at 50
families.  Once the path is compact enough to enter ping-pong, ping-pong
accounts for the remaining roughly `41.5 ms` and `0.24 GiB` improvement.

Nsight Systems, local 50-family pair:

| Metric | Leaf index disabled | Leaf index enabled |
|---|---:|---:|
| Event time | `301.304 ms` | `225.514 ms` |
| Kernel launches | `1506` | `1173` |
| Kernel sum | `266.523 ms` | `224.579 ms` |
| Kernel span | `301.166 ms` | `225.416 ms` |
| CUDA streams | `1` | `1` |
| Memcpy | `355 calls / 15.445 GB / 28.452 ms` | `2 calls / 8 bytes / 0.0016 ms` |

Top buckets:

| Bucket | Leaf index disabled | Leaf index enabled | Interpretation |
|---|---:|---:|---|
| `_wave_step_uniform_kernel` | `194.470 ms / 294` | `154.129 ms / 294` | compact leaf term plus ping-pong store pattern reduces wave-step work |
| PyTorch elementwise | `45.080 ms / 1074` | `33.979 ms / 766` | dense leaf mask construction mostly removed |
| `_dts_fused_kernel` | `14.764 ms / 39` | `14.746 ms / 39` | DTS unchanged |
| `_dts_parent_reduced_ge2_stage1_kernel` | `11.417 ms / 7` | `11.421 ms / 7` | DTS unchanged |
| `_wave_pibar_uniform_parent_kernel` | absent from disabled top buckets | `9.997 ms / 48` | enabled path enters ping-pong and pays final Pibar recompute |

The old dense path built per-wave dense leaf masks through PyTorch
fills/indexing/`any` checks and stayed on the non-ping-pong update pattern,
which created large device-to-device copy traffic.  The new path eliminates the
dense leaf masks and the memcpy-heavy non-ping-pong pattern.  DTS timings are
unchanged, confirming that the win is specifically leaf addressing plus enabling
the existing ping-pong fast path.

Decision:

- Keep `GPUREC_FORWARD_LEAF_INDEX=1` as the default.
- Treat this as a real genewise uniform forward win: `-73.3 ms` and
  `-2.9 GiB` at 50 families, and the 150-family case changes from OOM to
  fitting at `17.959 GiB`.
- Keep the dense fallback for the two-kernel and linear implementations.
- Keep documenting the public API limitation: genewise scalar `[G]` is covered
  by gradient bridge parity; genewise specieswise `[G,S]` is forward-tested
  but not yet a full combined genewise+specieswise training path.

Artifacts:

```text
/tmp/gpurec_profile/forward_prop7/genewise_leaf0_f50.nsys-rep
/tmp/gpurec_profile/forward_prop7/genewise_leaf0_f50.sqlite
/tmp/gpurec_profile/forward_prop7/genewise_leaf1_f50.nsys-rep
/tmp/gpurec_profile/forward_prop7/genewise_leaf1_f50.sqlite
```

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

### Proposal 8 follow-up: implemented compact species preprocessing

Proposal 8 is now implemented for the high-level uniform path.  This is a
setup/cache/RSS improvement, not a warmed GPU forward-kernel speedup.

Implementation:

- C++ `preprocess(...)` and `preprocess_multiple_families(...)` accept
  `include_species_matrices=true` by default.  When it is false, they skip
  dense `Recipients_mat` and `ancestors_dense`.
- Compact preprocessing still emits `unnorm_row_max`, computed directly from
  the species topology rather than from the dense unnormalized recipient
  matrix.
- `GeneDataset` keeps `retain_dense_species_matrices=true` by default for
  backward compatibility.
- `GeneReconModel.from_trees(...)` passes
  `retain_dense_species_matrices=(pibar_mode != "uniform")`, so high-level
  uniform model construction is compact by default.  Dense/topk/pairwise modes
  retain the dense matrices.
- The preprocessing cache key distinguishes
  `light-v2:dense-species` from `light-v2:compact-species`.
- Uniform static helpers no longer require `Recipients_mat` or
  `ancestors_dense`.  `ancestors_T` is built from compact topology through
  `gpurec/core/species.py`.
- `Pi_wave_forward`'s CUDA `torch_spmm` fallback also derives its sparse
  ancestor matrix from compact topology, so environment-selected fallback
  implementations do not reintroduce a dense `ancestors_dense` dependency.
- `wave_optimizer` and `genewise_optimizer` use the same compact topology
  helper, so the optimizer paths do not silently require dense species matrices.

The dense species mode for this dataset has three fp64 CPU matrices at
`S=1999`:

| Dense species tensor | Size |
|---|---:|
| `Recipients_mat` | `30.487 MiB` |
| `ancestors_dense` | `30.487 MiB` |
| `tr_mat_unnormalized` | `30.487 MiB` |

Compact mode removes all three live dense tensors and keeps `unnorm_row_max`
plus compact topology helpers.  A unique CPU tensor inventory for 10 families
measured `98.501 MiB` in dense species mode versus `7.024 MiB` in compact
species mode, saving about `91.5 MiB` of live CPU tensor memory.

Correctness evidence:

| Check | Result |
|---|---:|
| Full autograd bridge suite, including compact uniform construction, dense/topk retention, NLL batch match, and CUDA `torch_spmm` fallback | `20 passed` |
| Multi-family light preprocess default, preprocess cache parity with single path, and dtype roundtrip | `3 passed` |
| Genewise optimizer and genewise batch/per-family forward checks | `4 passed` |
| Follow-up sparse-warning check after wrapping sparse construction | `2 passed`, no sparse warning |
| CUDA `torch_spmm` fallback with compact uniform species helpers | matched default fused loss within `1e-7` |

NLL parity against a dense-retaining uniform model:

| Workload | Compact loss | Dense-retained loss | Abs diff |
|---:|---:|---:|---:|
| 10 families | `45471.82421875` | `45471.82421875` | `0.0` |
| 50 families | `221312.875` | `221312.875` | `0.0` |

This verifies that `unnorm_row_max` and compact topology reproduce the uniform
model state needed by the forward pass.

#### Construction and RSS

Separate-process construction/RSS measurements include static build for
`test_trees_1000`, fp32, fixed 6 Pi iterations.

| Workload | Species mode | Construct time | Elapsed | Max RSS | CUDA peak | Clades | `ancestors_T` nnz |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | compact | `0.874278 s` | `1.82 s` | `1167536 KB` | `45.466 MiB` | `321930` | `24809` |
| 50 | dense-retaining simulation | `0.948574 s` | `1.91 s` | `1207700 KB` | `45.466 MiB` | `321930` | `24809` |
| 150 | compact | `2.267044 s` | `3.24 s` | `1302944 KB` | `135.104 MiB` | `954706` | `24809` |
| 150 | dense-retaining simulation | `2.294321 s` | `3.25 s` | `1332536 KB` | `135.104 MiB` | `954706` | `24809` |

The RSS deltas are modest in these whole-process measurements because Python,
parsed family state, and cached tensors dominate the process footprint.  The
targeted tensor inventory above shows the actual dense species tensor removal
more directly.  GPU peak during static construction is unchanged because the
uniform static build no longer moves dense species matrices to GPU in either
compact mode or the dense-retaining simulation.

#### Cache size and dataset construction

Measurements use `GeneDataset(preprocess_cache_dir=...)`.

| Workload | Species mode | Cold dataset | Warm dataset | Species cache | Family cache | Total cache |
|---:|---|---:|---:|---:|---:|---:|
| 10 | compact | `0.258815 s` | `0.013052 s` | `0.105 MiB` | `7.192 MiB` | `7.297 MiB` |
| 10 | dense | `0.190419 s` | `0.031644 s` | `61.080 MiB` | `7.192 MiB` | `68.272 MiB` |
| 50 | compact | `0.676246 s` | `0.053537 s` | - | - | `34.944 MiB` |
| 50 | dense | `0.725685 s` | `0.074977 s` | - | - | `95.919 MiB` |
| 150 | compact | `2.158724 s` | `0.155700 s` | - | - | `103.456 MiB` |
| 150 | dense | `2.390087 s` | `0.179270 s` | - | - | `164.431 MiB` |

The species-cache delta is stable:

```text
61.080 MiB - 0.105 MiB = 60.975 MiB saved
```

The cold timing is not uniformly faster at 10 families because parsing and cache
write overhead dominate at that size.  At larger workloads compact mode is
slightly faster and consistently reduces warm-cache time and cache footprint.

#### Interpretation

This proposal removes dense species matrices from uniform setup and cache
state.  It does not change the warmed GPU forward interval: the `Pi` wave loop,
DTS kernels, and final likelihood kernels are unchanged.  The practical wins
are:

- lower CPU tensor/RSS pressure during model construction;
- much smaller species cache entries;
- lower warm-cache load time;
- clearer separation between uniform topology helpers and dense transfer modes.

The remaining setup opportunity is to precompute compact ancestor COO/CSR
directly during preprocessing.  The profiling worker measured about `19-20 ms`
of Python topology-to-sparse setup that still happens after compact
preprocessing.  Moving that sparse construction into preprocessing/cache would
not accelerate warmed kernel profiles, but it would further reduce repeated
model setup time.

#### Decision

Keep compact species preprocessing as the high-level default for
`pibar_mode="uniform"`.  Keep dense species matrices by default for dense/topk
and low-level `GeneDataset` compatibility.  Count this as a setup/cache/RSS
improvement, not as a forward GPU kernel optimization.

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

### Proposal 9 follow-up: tested fp64 and tensor-core experiments

Proposal 9 tested three directions:

1. row-prefix / linear uniform formulation;
2. mixed internal precision for fp64 inputs;
3. fp64 Triton tile tuning and tensor-core audit.

The benchmark helper now accepts `--dtype {fp32,fp64}` so the fp64 measurements
are reproducible.  The default remains fp32.  The reliable local measurements
below used an RTX 4090, `S=1999`, `tests/data/test_trees_1000`, global uniform
mode, fixed 6 Pi iterations, current optimized likelihood interval,
`--no-need-pibar --root-rows`, and cache directory
`/tmp/gpurec_prop9_cache`.  Earlier accidental parallel benchmark samples are
excluded.

#### Baseline dtype cost

Likelihood-only interval:

| Workload | dtype | Median | Peak GPU | NLL | Read |
|---:|---|---:|---:|---:|---|
| 3 families | fp32 | `11.550 ms` | `0.306619 GiB` | `6421.17333984375` | baseline |
| 3 families | fp64 | `389.793 ms` | `0.612526 GiB` | `6421.173588860177` | `33.7x`, NLL diff `2.49e-4` |
| 10 families | fp32 | `59.239 ms` | `1.088754 GiB` | `22182.9140625` | baseline |
| 10 families | fp64 | `2216.894-3071.235 ms` | `2.171539 GiB` | `22182.916951060826` | noisy `37-52x`, NLL diff `0.002889` |

Full-output interval, `--need-pibar --no-root-rows`:

| Workload | dtype | Median | Peak GPU | NLL | Ratio |
|---:|---|---:|---:|---:|---:|
| 3 families | fp32 | `27.178 ms` | `0.306689 GiB` | `6421.17333984375` | - |
| 3 families | fp64 | `847.117 ms` | `0.612666 GiB` | `6421.173588860177` | `31.2x` |

The end-to-end fp64 penalty is therefore about `31-52x` on this RTX 4090,
depending on workload and output mode.  The memory increase is the expected
roughly `2x` tensor-size effect.

#### NCU resource evidence

NCU was run on the first 3-family `_wave_step_uniform_kernel` launch:
`grid=(4684,1,1)`, block x dimension `128`.

| Metric | fp32 | fp64 |
|---|---:|---:|
| Kernel duration | `209.024 us` | `14.484 ms` |
| Duration ratio | - | about `69x` |
| SM throughput | `74.873%` | `83.554%` |
| DRAM throughput | `37.984%` | `1.066%` |
| FP64 pipe instructions | `0` | `216,382,013` |
| Tensor HMMA instructions | `0` | `0` |
| Registers/thread | `40` | `48` |
| Waves/SM | `3.05` | `3.66` |

Interpretation: fp64 is not memory-bound here.  DRAM is almost idle in the fp64
kernel, while scalar fp64 work saturates the SM side.  Tensor cores are exactly
unused in both runs.  The current kernel has no `tl.dot`/MMA structure and
executes scalar exp/log/reduction and tree-gather code.

Artifacts:

```text
/tmp/gpurec_prop9_ncu_fp32_default.ncu-rep
/tmp/gpurec_prop9_ncu_fp64_default.ncu-rep
```

#### Row-prefix / linear and existing fp64 variants

Worker A found that no usable forward row-prefix CUDA/shared-memory prototype
is wired today.  The prefix/Euler code is backward-only VJP machinery, mainly
in `gpurec/core/kernels/wave_backward.py`.  The available forward variants are
the default parent-walk ping-pong path plus `GPUREC_UNIFORM_IMPL=ancestor`,
`csr`, `two_kernel`, and `linear`; all are parameterized for fp64.

Correctness was acceptable for the existing variants:

| Check | Result |
|---|---|
| Synthetic fp64 kernel parity | default/two-kernel/ancestor/csr max abs `0.0`; linear `6.66e-16` |
| 3-family dataset NLL parity | default `6421.173588860177`; ancestor/csr/two-kernel delta `3.855e-05`; linear delta `0.0` |
| 10-family dataset NLL parity | default `22182.916951060826`; ancestor/csr/two-kernel delta `1.381e-04`; linear delta `0.0` |

Worker A warmed likelihood-only timings after E, fp64, fixed 6, root rows:

| Variant | 3 families | Peak | 10 families | Peak | Read |
|---|---:|---:|---:|---:|---|
| default | `410.197 ms` | `0.613 GiB` | `1239.073 ms` | `2.172 GiB` | baseline |
| ancestor | `371.897 ms` | `0.701 GiB` | `1118.671 ms` | `2.483 GiB` | possible `9-10%` fp64 win |
| csr | `373.647 ms` | `0.701 GiB` | `1118.716 ms` | `2.483 GiB` | similar to ancestor |
| two_kernel | `374.494 ms` | `1.260 GiB` | `1124.505 ms` | `4.464 GiB` | faster, but high memory |
| linear | `497.580 ms` | `1.261 GiB` | `1257.127 ms` | `4.466 GiB` | reject |

Worker A's Nsys comparison did not show a clean linear win:
default wave step `3126.176 ms / 270` launches versus linear
`3025.390 ms / 270`, but final Pibar increased from
`382.098 ms / 44` to `420.426 ms / 45`; DTS was similar.  The earlier local
guarded linear check was also slower (`1299.424 ms` on 3 families) and used
about `2x` GPU memory.  Artifacts:

```text
/tmp/gpurec_profile/prop9_fp64/default_f10.nsys-rep
/tmp/gpurec_profile/prop9_fp64/linear_f10.nsys-rep
```

Decision: reject row-prefix/linear promotion.  There is no forward row-prefix
implementation to promote, and the existing linear variant is slower.  The
ancestor/csr variants may deserve a later controlled fp64-only sweep because
they showed a possible `9-10%` speedup with modest memory growth and small NLL
deltas, but they should not become defaults from this evidence alone.

#### Mixed internal precision

A temporary, removed prototype tested:

```text
GPUREC_UNIFORM_INTERNAL_FP32=1
```

It kept external tensors fp64 but cast the default uniform wave-step and final
Pibar internal arithmetic to fp32.  This code was removed after testing and is
not a committed production path.

Performance was promising:

| Workload/mode | Default fp64 | Mixed internal fp32 | Peak GPU | NLL |
|---|---:|---:|---:|---:|
| 3 families, likelihood-only | `389.793 ms` | `33.215 ms` | `0.612526 GiB` | `6421.173355468305` |
| 10 families, likelihood-only | `2216.894-3071.235 ms` | `104.916 ms` | `2.171539 GiB` | `22182.916381341616` |
| 3 families, full-output | `847.117 ms` | `73.809 ms` | - | `6421.173355468305` |

Numerically, this moved the fp64 result close to the fp32 likelihood:

| Workload | Abs NLL diff vs default fp64 |
|---:|---:|
| 3 families | `2.33e-4` |
| 10 families | `5.70e-4` |

But finite-difference gradient checks failed under the mixed-internal prototype:

| Case | Analytic | FD | Diff |
|---|---:|---:|---:|
| global uniform, `idx=(1,)` | `3.656096` | `3.630517` | `2.558e-2` |
| specieswise uniform, `idx=(0,0)` | `0.160833` | `0.152741` | `8.091e-3` |
| genewise uniform, `idx=(0,0)` | `0.971214` | `0.975034` | `-3.820e-3` |

This finite-difference failure is the decisive reason not to promote the
broader mixed-internal path for training/backward.

Worker B tested a narrower scratch candidate: tensors, E, DTS, Pi values, and
the final likelihood remained fp64, but only Pibar normalization internals
(`row_max`, `row_sum`, `ancestor_sum`, `denom`, and
`log2(denom) + row_max + mt`) ran in fp32 before casting Pibar back to fp64.
No tracked files were edited.

| Workload/mode | Current fp64 | Pibar32 scratch | Speedup |
|---|---:|---:|---:|
| 3 families, likelihood-only | `389.97 ms` | `111.74 ms` | `3.49x` |
| 10 families, likelihood-only | `2558.32 ms` | `747.80 ms` | `3.42x` |
| 3 families, full-output | `844.74 ms` | `244.88 ms` | `3.45x` |
| 10 families, full-output | `2656.18 ms` | `754.16 ms` | `3.52x` |

Worker B's 3-family Nsys split showed wave step falling from `358.10 ms` to
`103.09 ms`, final Pibar from `44.82 ms` to `2.27 ms`, and DTS unchanged at
about `13 ms`.

| Workload | NLL diff vs fp64 | Pi max abs | Pi mean abs | Pibar max abs |
|---:|---:|---:|---:|---:|
| 3 families | `3.70e-05` | `1.07e-02` | `2.42e-04` | `1.32e-02` |
| 10 families | `1.20e-04` | `2.13e-02` | `2.11e-04` | `2.18e-02` |

This is a more conservative future direction than the broader local
`GPUREC_UNIFORM_INTERNAL_FP32=1` experiment: the speedup is smaller, but the
NLL drift is also smaller and the approximation is localized to Pibar
normalization.  Worker B did not run gradcheck, so this remains an
inference-only candidate until backward parity and finite-difference checks are
implemented.

Mixed-precision decision: do not promote any mixed internal fp32 path for
training/backward.  A hidden approximation inside `dtype=torch.float64` would
make gradients dtype-dependent in a way that already failed finite-difference
checks.  The narrower Pibar32 variant is worth considering only as an explicit
approximate inference mode with documented tolerances.

#### Block-size tuning and tensor-core audit

Worker C tested fp64 full-output runs on 10 families with exact NLL parity for
all block candidates: `22182.916951060826`, delta `0.0`, peak
`2.172035 GiB`.

| `BLOCK_S` / warps | Median | Mean | Min | Max |
|---|---:|---:|---:|---:|
| default/default | `1294.522 ms` | `1647.310 ms` | `1220.011 ms` | `2655.197 ms` |
| `128 / 4` | `2716.198 ms` | `2554.450 ms` | `1242.821 ms` | `3967.305 ms` |
| `256 / 4` | `1299.455 ms` | `1880.114 ms` | `1219.967 ms` | `3112.721 ms` |
| `256 / 8` | `1310.252 ms` | `1547.670 ms` | `1245.598 ms` | `2424.108 ms` |
| `512 / 8` | `1221.536 ms` | `1222.863 ms` | `1221.426 ms` | `1230.903 ms` |

`512/8` was the best and most stable in that full-output fp64 run.  However,
local 10-family likelihood-only `512/8` was slower: median `4424.032 ms`
versus noisy default medians in the `2216-3071 ms` range.  Treat fp64 block
tuning as workload-sensitive and inconclusive.  Keep the tuning knobs, but do
not change the default.

Worker Nsys for default full-output fp64 showed:

| Kernel bucket | Time | Launches | Share |
|---|---:|---:|---:|
| `_wave_step_uniform_kernel` | `2.285 s` | `270` | `86.0%` |
| `_wave_pibar_uniform_parent_kernel` | `269.247 ms` | `45` | `10.1%` |
| `_dts_fused_kernel` | `56.515 ms` | `39` | - |

The tensor-core audit found tensor-pipe counters exactly zero and scalar fp64
`dadd`/`dmul`/`dfma` instructions present.  There is no tensor-core opportunity
in the current scalar kernels.

#### Decision

No production fp64/tensor-core optimization is promoted from Proposal 9.  The
only retained implementation change is benchmark support for `--dtype` so fp64
measurements can be reproduced.

Near-term direction:

- do not use the linear/row-prefix direction for fp64 forward yet; no forward
  row-prefix implementation is wired;
- run a later controlled ancestor/csr fp64 sweep before considering an fp64
  default change;
- do not silently mix fp32 internals into fp64 training;
- keep fp64 block-size knobs for profiling, but leave defaults unchanged;
- consider an explicit inference-only Pibar32 approximate mode with
  per-family NLL bounds;
- require gradient-consistent mixed backward and finite-difference validation
  before any mixed mode can be used for training;
- otherwise focus on algorithmic reformulation that reduces scalar exp/log work
  in a gradient-consistent way.

Tensor cores are not applicable to the current uniform forward kernels.

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

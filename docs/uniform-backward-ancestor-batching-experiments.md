# Uniform Backward Ancestor-Batching Experiments

Date: 2026-05-05.

Scope: test whether the uniform self-loop backward can avoid the current
uncoalesced ancestor walk by batching the ancestor/subtree operation across the
wave dimension, and whether any of those variants should replace or augment the
current fused Triton path.

This document is the working log for the implementation, correctness, and
profiling agents. Each proposal below must be tested with an opt-in flag first;
no prototype should become the default without parity, finite-difference or
gradcheck coverage where applicable, warmed timing, and Nsight evidence.

Documentation-owner constraint for this pass: this document is the only file
edited by the documentation agent. Production and test commits by other workers
are observed and summarized here, but not modified here.

## Current Result At A Glance

The most important profiling result remains Proposal 5: the existing NVRTC CUDA
no-split row kernel, run in exact `tree` correction mode, is a strong control
for the current full-ancestor baseline. Since that result, four support commits
landed:

- `cdffa29 Add ancestor batching profiling harness`
- `256bff7 Add opt-in 2D self-loop backward prototype`
- `1b14648 Add uniform ancestor batching correctness tests`
- `a018169 Add staged tree self-loop backward prototype`

| Item | Current status | Decision |
|---|---|---|
| Proposal 0, 2D Triton self-loop | opt-in prototype landed in `256bff7`; parity passes; 10/50/100-family timing is promising but memory-heavy | keep opt-in; collect Nsys/NCU before any promotion discussion |
| Proposal 1, staged Triton tree DP | opt-in staged prototype landed in `a018169`; parity passes; current `W=4,S=256` timing regresses | reject current configuration for promotion; keep as correctness scaffold |
| Proposal 2, species-major/transposed scratch | no production implementation | lower priority; current species ids already have contiguous subtrees |
| Proposal 3, hybrid wave router | no threshold router implementation | needed if Proposal 1 or 5 is promoted |
| Proposal 4, forward path prefix | no production implementation in this pass | reject as fp32 production direction until a full fused forward kernel exists |
| Proposal 5, existing CUDA no-split row kernel | implemented earlier; retested on current semantics | keep opt-in, but exact `tree` mode passes the performance gate on 50 families |

The key decision nuance: `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=self` is a
historical lower-bound/control mode. The current Triton baseline, after
`221f16a`, uses the full ancestor/subtree correction. Therefore current
correctness and promotion discussions must use:

```bash
GPUREC_CUDA_SELF_LOOP_NOSPLIT=1
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree
```

## Background Read

Relevant older documentation and facts used here:

- `docs/uniform-mode.md` defines the uniform forward denominator and backward
  VJP. The self-loop VJP needs:

  ```text
  u[s] = grad_Pibar[s] / denom[s]
  A = sum_s u[s]
  correction[j] = sum_{s where j is an ancestor of s} u[s]
  grad_Pi[j] += p_prime[j] * (A - correction[j])
  ```

- `docs/bf16-backward-profile.md` documents `de2c5fd`, which fixed the bf16
  pathological global atomic path by using fp32 only for the internal
  `pibar_corr` scratch. After that fix, the remaining large-wave profile is
  memory/L2/topology-limited, not compute-limited.
- `docs/uniform-backward-fourth-pass-proposals.md` documents the CUDA no-split
  row kernel and the self-loop parameter two-stage reduction. Those results
  predate `221f16a`; the old `self` correction results are not a correctness
  target for the current full-ancestor baseline.
- `docs/uniform-backward-fp32-fused-profile.md` tested a top-down prefix
  denominator for cross-clade Pibar VJP. It was correct but slower because
  barriers and extra scratch traffic outweighed less ancestor-gather traffic.
- `docs/uniform-forward-optimization-proposals.md` tested Pibar-only row-prefix
  forward prototypes. The fp32 Pibar-only CUDA prefix was slower than the
  current parent walk; fp64 was promising, but no full fused forward
  DTS_L/Pibar kernel was built.

## Current Implementation Inventory

Present opt-in paths related to this document:

| Flag | Present | Default | Notes |
|---|---:|---:|---|
| `GPUREC_CUDA_SELF_LOOP_NOSPLIT` | yes | off | Routes no-split fp32 uniform waves to `gpurec_wave_backward_nosplit_uniform_fp32`. |
| `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION` | yes | `self` if the CUDA path is enabled | Use `tree` for current exact semantics. |
| `GPUREC_SELF_LOOP_PARAM_TWO_STAGE` | yes | off | Removes self-loop parameter REDs in eligible no-split waves, but regressed end-to-end in prior tests. |
| `GPUREC_DTS_PIBAR_UD_EULER_PREFIX` | yes | off | Cross-clade staged Pibar VJP interval-prefix path, not the self-loop kernel. |
| `GPUREC_SELF_LOOP_2D_TRITON` | yes | off | Proposal 0 prototype. CUDA fp32/fp64 only, default `GPUREC_SELF_LOOP_2D_MAX_S=2048`; returns per-element parameter VJPs, so the model path disables in-kernel self-loop parameter accumulation when enabled. |
| `GPUREC_SELF_LOOP_2D_BLOCK_W` | yes | `1` | Proposal 0 row block size. Larger values multiply the full-species vector width and are likely register-pressure limited. |
| `GPUREC_SELF_LOOP_TREE_STAGED` | yes | off | Proposal 1 staged tree-DP prototype. CUDA fp32/fp64 only; returns per-element parameter VJPs, so the model path disables in-kernel self-loop parameter accumulation when enabled. |
| `GPUREC_SELF_LOOP_TREE_TILE_W` | yes | `2` in implementation, `4` in harness variant | Proposal 1 row tile. Current benchmarked harness variant is `proposal1_tree_staged_w4_s256`. |
| `GPUREC_SELF_LOOP_TREE_TILE_S` | yes | `128` in implementation, `256` in harness variant | Proposal 1 species tile for staged kernels. |
| `GPUREC_SELF_LOOP_TREE_TRANSPOSED` | no | n/a | Proposal 2, not implemented in production. |
| `GPUREC_FORWARD_UNIFORM_PATH_PREFIX` | no | n/a | Proposal 4, not implemented in production. |

Reusable harness:

- `profiling/ancestor_batching/bench_uniform_backward.py` is the single-variant
  CUDA-event benchmark.
- `profiling/ancestor_batching/run_profiles.py` orchestrates timing, parity,
  Nsys, and NCU captures and writes ignored artifacts under
  `profiling/ancestor_batching/artifacts/`.
- `profiling/ancestor_batching/README.md` documents the harness contract.

Current harness availability dry-run:

```bash
python profiling/ancestor_batching/run_profiles.py \
  --dry-run \
  --phases timing \
  --fams 10 \
  --run-id doc_available_dryrun
```

Result on `1b14648`: selected `baseline`, `proposal0_2d_triton_bw1`,
`proposal0_2d_triton_bw2`, `proposal1_tree_staged_w4_s256`,
`proposal5_cuda_nosplit_self`, and `proposal5_cuda_nosplit_tree`; skipped
Proposal 2, Proposal 3, and Proposal 4 because their required production flags
or implementation markers were absent. At `1b14648`, Proposal 1 selection only
meant flag plumbing. After `a018169`, the same harness variant maps to the real
staged tree-DP prototype.

## Baseline Problem

The hot self-loop backward currently computes the uniform-Pibar correction by
walking species ancestors inside one Triton program per clade row:

```python
for s in species:
    u_d = term[w, s] * pibar_coeff[w, s]
    for a in ancestors_including_self(s):
        atomic_add(pibar_corr[w, a], u_d)

result[w, s] = term[w, s] * diag[w, s] \
             + p_prime[w, s] * (A[w] - pibar_corr[w, s]) \
             + speciation_parent_gather_or_scatter(w, s)
```

Mathematically, `pibar_corr[w, s]` is the subtree sum
`sum_{d in descendants(s)} u_d[w, d]`. The parent-pointer walk is therefore not
fundamental. It can be replaced by either a bottom-up species-tree reduction or
an Euler-interval prefix sum.

The fixed bf16 profile showed that, after avoiding bf16 atomics, the large
self-loop launch is again memory/L2 limited:

| metric | fixed bf16 large wave | fp32 large wave |
|---|---:|---:|
| L2 throughput | `70.75%` | `75.64%` |
| compute throughput | `18.36%` | `18.69%` |
| achieved occupancy | `82.31%` | `99.06%` |
| excessive global sectors | about `44%` | about `44%` |

After `221f16a`, current fp32 exact self-loop launches are much slower than the
fourth-pass self-only baseline. On the 50-family Nsys capture in this pass, the
last two no-split `W=32768` Triton launches take about `52.356 ms` and
`49.780 ms`.

The target is not just removing atomics. The target is reducing uncoalesced
topology traffic and repeated ancestor visits while preserving the full
ancestor/subtree VJP.

## Measurement Protocol

Correctness gates:

- `pytest -q tests/kernels/test_wave_backward_kernel.py`
- `pytest -q tests/gradients/test_autograd_bridge.py`
- direct parity on `tests/data/test_trees_1000` for 10 and 50 families:
  baseline vs prototype loss, theta gradient max absolute difference, and max
  relative difference versus the baseline gradient infinity norm.
- for any semantic change to the self-loop VJP, add a small synthetic reference
  test against the PyTorch analytical path, not only baseline parity.

Performance gates:

- warmed CUDA-event backward timing for 10, 50, and 100 families from
  `tests/data/test_trees_1000`;
- direct forward/backward model timing for first 100 families when the prototype
  affects the production model path;
- Nsight Systems kernel bucket comparison for the 50- or 100-family workload;
- Nsight Compute on one representative large `_wave_backward_uniform_kernel` or
  replacement launch, using `--kernel-id` rather than launch-skip heuristics.

Preferred harness commands for new batches:

```bash
python profiling/ancestor_batching/run_profiles.py --phases timing
python profiling/ancestor_batching/run_profiles.py \
  --phases nsys,ncu \
  --variants baseline,proposal5_cuda_nosplit_tree
python profiling/ancestor_batching/bench_uniform_backward.py \
  --fams 50 \
  --reps 9 \
  --warmups 5 \
  --cache-dir /tmp/gpurec_ancestor_batching_cache
```

The older `profiling/proposal8/bench_uniform_backward.py` commands remain
recorded below because they are the exact commands used for the Proposal 5
control result.

Promotion threshold:

- A prototype must reduce end-to-end backward time by at least `3%` on the
  50-family workload and not regress 100-family timing.
- If it only improves one kernel bucket, it must be left opt-in unless the
  end-to-end gain survives alternating paired runs.

For Proposal 5 specifically, current promotion must also require changing the
enabled correction mode to `tree`, or otherwise documenting `self` as an
intentional approximation. The current exact semantics are not `self`.

## Proposal 0: 2D Triton Self-Loop Kernel

`256bff7` implements this as an opt-in Triton prototype. One Triton program
processes `BLOCK_W` clade rows and the full species vector
`BLOCK_S=next_power_of_2(S)`. The implementation is intentionally conservative:
it supports CUDA fp32/fp64 only, refuses `S > GPUREC_SELF_LOOP_2D_MAX_S`
(`2048` by default), and returns per-element parameter VJPs instead of using
the fused in-kernel parameter accumulators.

```python
if not env("GPUREC_SELF_LOOP_2D_TRITON"):
    return current_fused_triton()

if dtype not in (fp32, fp64) or S > GPUREC_SELF_LOOP_2D_MAX_S:
    return unavailable_or_raise_if_strict()

precompute_kernel:
    # one program per row block, full species vector in-program
    rows = block_id * BLOCK_W + arange(BLOCK_W)
    species = arange(next_power_of_2(S))
    diag, pibar_coeff, p_prime, sl1_wt, sl2_wt = make_self_loop_weights(...)
    v_k = rhs

for n in range(neumann_terms):
    jt_kernel:
        term = rhs if n == 0 else previous_term_buffer
        u = term * pibar_coeff
        A = sum_species(u)
        corr = u

        # compact levels are built from the species children once and supplied
        # to every row-block program.
        for level in compact_bottom_up_levels:
            corr[parent] = corr[parent] + corr[child1] + corr[child2]
            barrier_inside_program()

        out = term * diag + p_prime * (A - corr)
        out[child1] += term[parent] * sl1_wt[parent]
        out[child2] += term[parent] * sl2_wt[parent]
        v_k += out

param_store_kernel:
    aw0, aw1, aw2, aw345, aw3, aw4 = per_element_param_vjp(v_k)
```

Expected benefit:

- reuse one species-tree schedule for `BLOCK_W` clade rows;
- remove global atomics from the correction path;
- reduce repeated parent-pointer loads.

Risks:

- `BLOCK_W * BLOCK_S` tensors can explode registers. For `S=1999`,
  `BLOCK_W=2` already means roughly 4096 vector lanes before temporaries.
- A single Triton program cannot synchronize with other programs, so this only
  works if the full species dimension is inside one program.
- It may under-occupy if register pressure forces spills.

Test matrix:

| env flag | values |
|---|---|
| `GPUREC_SELF_LOOP_2D_TRITON` | `0`, `1` |
| `GPUREC_SELF_LOOP_2D_BLOCK_W` | `1`, `2`, maybe `4` if compilation permits |
| `GPUREC_SELF_LOOP_2D_BLOCK_NODES` | default `64` |
| `GPUREC_SELF_LOOP_2D_NUM_WARPS` | default `8` |
| `GPUREC_SELF_LOOP_2D_JT_NUM_WARPS` | defaults to `GPUREC_SELF_LOOP_2D_NUM_WARPS` |
| `GPUREC_SELF_LOOP_2D_MAX_S` | default `2048` |

Correctness commands run by the documentation pass:

```bash
pytest -q \
  tests/kernels/test_wave_backward_kernel.py::test_uniform_pibar_jt_correction_is_subtree_sum_not_self_only \
  tests/kernels/test_wave_backward_kernel.py::test_documented_self_loop_prototype_flags_match_synthetic_reference \
  tests/kernels/test_wave_backward_kernel.py::test_cuda_nosplit_tree_correction_matches_synthetic_reference

pytest -q -m 'not slow' \
  tests/gradients/test_uniform_backward_ancestor_batching.py
```

Results:

```text
6 passed in 1.43s
1 passed, 1 deselected in 12.14s
```

After `a018169`, the staged prototype was added and the same focused checks
were rerun on current HEAD:

```bash
pytest -q \
  tests/kernels/test_wave_backward_kernel.py::test_documented_self_loop_prototype_flags_match_synthetic_reference

pytest -q -s -m 'not slow' \
  tests/gradients/test_uniform_backward_ancestor_batching.py
```

Results:

```text
4 passed in 1.24s
1 passed, 1 deselected in 5.73s
```

The current 10-family model parity deltas:

| Case | Loss abs diff | Theta grad max abs diff | Relative to baseline grad inf | Interpretation |
|---|---:|---:|---:|---|
| `proposal0_2d_triton` | `0.000e+00` | `4.578e-02` | `3.439e-05` | real Proposal 0 prototype plus external parameter accumulation |
| `proposal1_tree_staged` | `0.000e+00` | `4.602e-02` | `3.458e-05` | real Proposal 1 staged prototype plus external parameter accumulation |
| `proposal2_tree_transposed` | `0.000e+00` | `1.221e-04` | `9.171e-08` | no production flag effect |
| `proposal3_hybrid_tree_router` | `0.000e+00` | `4.578e-02` | `3.439e-05` | no threshold router yet; staged flag is active |
| `proposal4_forward_path_prefix` | `0.000e+00` | `1.221e-04` | `9.171e-08` | no production flag effect |
| `proposal5_cuda_nosplit_tree` | `0.000e+00` | `2.075e-03` | `1.559e-06` | exact CUDA tree no-split path |

The larger Proposal 0 deltas are still far inside the test tolerance
(`grad_rel_to_baseline_inf < 2e-3`) and are expected for changed fp32
accumulation order. They are not performance evidence.

Timing smoke attempt:

```bash
python profiling/ancestor_batching/run_profiles.py \
  --phases timing \
  --variants baseline,proposal0_2d_triton_bw1 \
  --fams 10 \
  --reps 3 \
  --warmups 1 \
  --cache-dir /tmp/gpurec_ancestor_batching_cache \
  --run-id doc_p0_smoke
```

Result on `1b14648`: both selected variants returned code `1` before timing.
The logs reached model shape reporting (`S=1999`, `C=66530`, `waves=45`,
`max_wave_rows=16645`, `split_rows=83135`) and then failed in forward
allocation with only about `186 MiB` free on the RTX 4090 because concurrent
Python GPU workers were holding roughly `22 GiB`. Treat this as an invalid
timing attempt caused by local GPU memory pressure, not as a Proposal 0
performance result.

That invalid smoke attempt was superseded by worker timing artifacts. The
cleanest current Proposal 0 timing batch is
`profiling/ancestor_batching/artifacts/20260505_ancestor_batching_head1b14648`.
Exact generated commands are in that artifact's `commands.sh`; the command
shape is:

```bash
GPUREC_SELF_LOOP_2D_TRITON=1 \
GPUREC_SELF_LOOP_2D_BLOCK_W=2 \
python profiling/ancestor_batching/bench_uniform_backward.py \
  --dataset tests/data/test_trees_1000 \
  --start 0 \
  --fams 50 \
  --variant-label proposal0_2d_triton_bw2 \
  --dtype fp32 \
  --cache-dir /tmp/gpurec_ancestor_batching_cache \
  --reps 9 \
  --warmups 5
```

Warmed CUDA-event timing, 10/50 families:

| Families | Variant | Commit | Median backward | Mean | Min | Peak allocation | Grad max abs diff vs baseline |
|---:|---|---|---:|---:|---:|---:|---:|
| 10 | baseline | `1b14648` | `102.813 ms` | `104.215 ms` | `94.122 ms` | `2.619 GB` | - |
| 10 | Proposal 0 `BLOCK_W=1` | `1b14648` | `52.161 ms` | `52.842 ms` | `51.677 ms` | `4.356 GB` | `0.045776` |
| 10 | Proposal 0 `BLOCK_W=2` | `1b14648` | `57.432 ms` | `60.169 ms` | `53.000 ms` | `4.356 GB` | `0.045776` |
| 10 | Proposal 5 CUDA `tree` | `1b14648` | `67.130 ms` | `67.157 ms` | `66.576 ms` | `1.952 GB` | `0.002319` |
| 50 | baseline | `1b14648` | `328.562 ms` | `328.350 ms` | `325.794 ms` | `9.823 GB` | - |
| 50 | Proposal 0 `BLOCK_W=1` | `1b14648` | `205.076 ms` | `219.436 ms` | `193.834 ms` | `14.217 GB` | `1.170410` |
| 50 | Proposal 0 `BLOCK_W=2` | `1b14648` | `176.359 ms` | `175.953 ms` | `174.438 ms` | `14.217 GB` | `1.168945` |
| 50 | Proposal 5 CUDA `tree` | `1b14648` | `230.705 ms` | `234.413 ms` | `227.347 ms` | `9.398 GB` | `0.028320` |

The 50-family Proposal 0 `BLOCK_W=2` median is `46.3%` faster than the baseline
and `23.5%` faster than exact Proposal 5 CUDA `tree` in this timing batch, but
it allocates about `4.39 GB` more than the baseline and about `4.82 GB` more
than Proposal 5.

100-family follow-up artifacts are mixed across commits because workers landed
new code during profiling. The commit field in each JSONL is authoritative:

| Families | Variant | Commit | Median backward | Peak allocation | Notes |
|---:|---|---|---:|---:|---|
| 100 | baseline | `1b14648` | `649.986 ms` | `16.935 GB` | clean run in `...head1b14648_fams100_clean` |
| 100 | Proposal 5 CUDA `tree` | `1b14648` | `428.945 ms` | `16.906 GB` | clean run in same artifact |
| 100 | Proposal 0 `BLOCK_W=1` | `a018169` | `320.891 ms` | `21.329 GB` | run in `...head1b14648_p0p1_fams100_clean`; baseline not rerun in that artifact |
| 100 | Proposal 0 `BLOCK_W=2` | `a018169` | `323.333 ms` | `21.329 GB` | same caveat |

Interpretation: Proposal 0 is no longer merely a high-risk sketch. Event timing
is promising, including at 50 and 100 families, but the memory cost is large and
there is no Nsys/NCU evidence yet. Promotion is blocked on profiling: we need
to know whether the speedup is coming from useful traffic reduction, whether
register spill appears at larger row blocks, and whether the peak allocation is
acceptable.

Status: implemented as an opt-in prototype. Decision: keep opt-in; promising
but not promotable until Nsys/NCU and a memory plan exist.

## Proposal 1: Extra Triton Kernels For Tree DP

`a018169` implements this as an opt-in staged prototype. It still reuses the
Proposal 0 full-species precompute kernel, but the Neumann `J^T` application is
split across tileable kernels with global synchronization between stages:

```python
precompute_full_species_weights(...)  # diag, pibar_coeff, p_prime, sl weights

for iter in range(neumann_terms):
    make_u_kernel:
        u[w, s] = term[w, s] * pibar_coeff[w, s]
        A[w] = sum_s u[w, s]              # atomic add over species tiles

    for level in compact_bottom_up_levels:
        reduce_level_kernel:
            corr[w, parent] += corr[w, child1] + corr[w, child2]

    combine_base_kernel:
        out[w, s] = term[w, s] * diag[w, s] \
                  + p_prime[w, s] * (A[w] - corr[w, s])

    spec_scatter_kernel:
        out[w, child1] += term[w, parent] * sl1_wt[w, parent]
        out[w, child2] += term[w, parent] * sl2_wt[w, parent]

    accumulate_kernel:
        v_k += out

param_store_kernel(...)
```

This implementation uses one `make_u` launch, one launch per compact tree
level, and three more launches per Neumann term, plus precompute and parameter
store launches. It therefore tests the staged synchronization idea directly,
but also pays a large launch-count and scratch-traffic cost.

Expected benefit:

- true 2D wave/species tiling is possible because each stage has kernel-boundary
  synchronization;
- the tree reduction is `O(W * S)` rather than `O(W * S * depth)`;
- row-major loads can remain coalesced for species tiles.

Risks:

- extra launches: roughly `3 * neumann_terms` more kernels per wave, or more if
  each level is separate;
- more global scratch traffic than the current fused path;
- the end-to-end time may regress even if the correction itself improves.

Test matrix:

| env flag | values |
|---|---|
| `GPUREC_SELF_LOOP_TREE_STAGED` | `0`, `1` |
| `GPUREC_SELF_LOOP_TREE_TILE_W` | `1`, `2`, `4`, `8` |
| `GPUREC_SELF_LOOP_TREE_TILE_S` | `128`, `256` |
| `GPUREC_SELF_LOOP_TREE_BLOCK_NODES` | default `128` |
| `GPUREC_SELF_LOOP_TREE_NUM_WARPS` | default `4` |
| `GPUREC_SELF_LOOP_TREE_REDUCE_WARPS` | default `4` |
| `GPUREC_SELF_LOOP_TREE_PRECOMPUTE_WARPS` | default `8` |

Correctness on current HEAD:

```bash
pytest -q \
  tests/kernels/test_wave_backward_kernel.py::test_documented_self_loop_prototype_flags_match_synthetic_reference

pytest -q -s -m 'not slow' \
  tests/gradients/test_uniform_backward_ancestor_batching.py
```

Result:

```text
4 passed in 1.24s
1 passed, 1 deselected in 5.73s
```

The 10-family model test reports `proposal1_tree_staged` loss diff `0`, theta
gradient max absolute diff `4.602e-02`, and relative-to-baseline-inf
`3.458e-05`.

Worker timing command for the staged harness variant:

```bash
GPUREC_SELF_LOOP_TREE_STAGED=1 \
GPUREC_SELF_LOOP_TREE_TILE_S=256 \
GPUREC_SELF_LOOP_TREE_TILE_W=4 \
python profiling/ancestor_batching/bench_uniform_backward.py \
  --dataset tests/data/test_trees_1000 \
  --start 0 \
  --fams 50 \
  --variant-label proposal1_tree_staged_w4_s256 \
  --dtype fp32 \
  --cache-dir /tmp/gpurec_ancestor_batching_cache \
  --reps 9 \
  --warmups 5
```

Timing artifact:
`profiling/ancestor_batching/artifacts/20260505_ancestor_batching_current_p1_10_50`.
That worker ran only Proposal 1, so the comparison baseline below is the
established `1b14648` baseline from
`20260505_ancestor_batching_head1b14648`; `a018169` does not alter the baseline
path when the staged flag is off.

| Families | Variant | Commit | Median backward | Mean | Min | Peak allocation |
|---:|---|---|---:|---:|---:|---:|
| 10 | baseline | `1b14648` | `102.813 ms` | `104.215 ms` | `94.122 ms` | `2.619 GB` |
| 10 | Proposal 1 staged `W=4,S=256` | `51f6e8d` | `124.250 ms` | `161.788 ms` | `120.624 ms` | `4.356 GB` |
| 50 | baseline | `1b14648` | `328.562 ms` | `328.350 ms` | `325.794 ms` | `9.823 GB` |
| 50 | Proposal 1 staged `W=4,S=256` | `51f6e8d` | `457.069 ms` | `457.157 ms` | `456.752 ms` | `14.218 GB` |

The 50-family staged prototype is `39.1%` slower than the baseline and roughly
`98%` slower than Proposal 5 CUDA `tree` in the same 50-family timing family.
It also has the same high scratch footprint as Proposal 0. The first warmup
includes enormous Triton compile or binary-load cost (`163.6 s` on 10 families,
`174.4 s` on 50 families); those warmups are excluded from the medians but make
the current configuration operationally unattractive.

100-family staged attempts did not produce valid timing. In
`20260505_ancestor_batching_head1b14648_p0p1_fams100_clean`, the `a018169`
Proposal 1 run reached model shape reporting and then failed with Triton CUDA
out-of-memory while loading the staged precompute binary. A separate
100-family artifact failed with GPU memory exhausted during staged scratch
allocation. Both point to the same decision: this implementation is memory and
launch-count limited before it becomes a useful split-wave replacement.

Status: implemented as an opt-in prototype. Decision: reject the current
`proposal1_tree_staged_w4_s256` configuration for promotion. Keep it as a
correctness scaffold and profiling reference, but future staged work needs a
different launch/memory design before more broad benchmarking.

## Proposal 2: Species-Major / Transposed Scratch

For batched tree DP, row-major `[W, S]` is good for per-row reductions but bad
if a kernel wants to process one species node across many clade rows. Test a
temporary species-major scratch:

```python
u_T[s, w] = u[w, s]

for level in postorder_levels:
    u_T[parent, w_tile] += u_T[child1, w_tile] + u_T[child2, w_tile]

corr[w, s] = u_T[s, w]
```

Expected benefit:

- species-level tree updates become contiguous across a `W` tile;
- no atomics if each `(species, w_tile)` owner writes a unique output;
- may work better for large waves where `W >> S`.

Risks:

- transposition costs may dominate;
- the scratch is another full `[W, S]` tensor;
- current backward already has large memory pressure, so peak allocation must be
  measured.

Test matrix:

| env flag | values |
|---|---|
| `GPUREC_SELF_LOOP_TREE_TRANSPOSED` | `0`, `1` |
| `GPUREC_SELF_LOOP_TREE_TILE_W` | `16`, `32`, `64`, `128` |

Status: not implemented in production. The new correctness test includes a
`GPUREC_SELF_LOOP_TREE_TRANSPOSED` case to ensure the documented flag does not
break the synthetic path, but `rg` finds no production reference to that flag.
Decision: lower priority than Proposal 1. Older Euler-layout diagnostics showed
that, for `tests/data/test_trees_1000`, every subtree is already one contiguous
current-order interval. That reduces the expected value of a broad
transposed/species relayout. A narrow interval-prefix self-loop variant may be
more promising than a full species-major scratch.

## Proposal 3: Hybrid Wave Router

Use the current fused kernel for small or awkward waves and route only large
waves to the staged or CUDA/tree path.

Candidate rule:

```python
if W >= threshold and not has_splits and dtype == torch.float32:
    use_cuda_tree_nosplit()
elif W >= staged_threshold and S <= max_s and dtype in supported_dtypes:
    use_tree_staged_variant()
else:
    use_current_fused_kernel()
```

Expected benefit:

- avoid launch overhead on tiny waves;
- keep the known-good fused kernel for split-heavy root waves if the tree path
  only helps no-split or leaf-like waves;
- allow different thresholds for fp32 and bf16.

Risks:

- wrong threshold can hide a good kernel or amplify a bad one;
- more code paths require more tests;
- scheduling interactions with active-mask pruning can change the best
  threshold.

Test matrix:

| env flag | values |
|---|---|
| `GPUREC_SELF_LOOP_TREE_MIN_W` | `0`, `1024`, `4096`, `8192`, `16384` |
| `GPUREC_SELF_LOOP_TREE_SPLIT_WAVES` | `0`, `1` |

Status: not implemented as a threshold router. The existing CUDA no-split flag
is already a coarse router for no-split fp32 waves, and the correctness tests
exercise a documented `proposal3_hybrid_tree_router` env bundle. In current
production code that bundle activates the Proposal 1 staged prototype globally;
the `GPUREC_SELF_LOOP_TREE_MIN_W` threshold is absent and the profiling harness
skips the Proposal 3 variant by default. Decision: required before any default
promotion. Current data support routing large no-split waves to CUDA `tree` or
possibly Proposal 0; the current Proposal 1 staged implementation is too slow
to be the split-wave route.

## Proposal 4: Forward Uniform-Pibar Tree Prefix

The forward denominator also computes:

```python
denom[w, s] = row_sum[w] - sum_{a in ancestors(s)} p_prime[w, a]
```

Since the current species order has contiguous subtrees on
`tests/data/test_trees_1000`, test an Euler/prefix formulation:

```python
prefix[w, t + 1] = prefix[w, t] + p_prime[w, species_in_euler[t]]
ancestor_sum[w, s] = prefix[w, euler_pos[s] + 1] - prefix[w, root_to_parent_start]
```

For the current denominator, the needed quantity is a root-path prefix, not a
subtree sum. If species ids are already topological, a top-down path-prefix by
levels may be cheaper than walking parents for every descendant.

Expected benefit:

- reduce repeated ancestor loads during forward self-loop/Pibar construction;
- create reusable row stats for backward if the model stores them.

Risks:

- forward was not the main bf16 bottleneck after the atomic fix;
- prefix/path kernels add launches unless fused carefully;
- root-path prefix and subtree interval prefix are different operations.

Test matrix:

| env flag | values |
|---|---|
| `GPUREC_FORWARD_UNIFORM_PATH_PREFIX` | `0`, `1` |
| `GPUREC_FORWARD_UNIFORM_PATH_TILE_W` | `1`, `2`, `4`, `8` |

Status: not implemented in production. The new correctness test includes a
`GPUREC_FORWARD_UNIFORM_PATH_PREFIX` env bundle, but current production code has
no such flag reference and the profiling harness skips Proposal 4 by default.
Decision: do not pursue a production fp32 forward path until a full fused
forward DTS_L/Pibar kernel exists. Older Pibar-only CUDA prefix results were
correct but `2.0-2.4x` slower than the current fp32 parent-walk Pibar kernel;
fp64 remained a separate promising direction.

## Proposal 5: Existing CUDA Shared-Memory Row Kernel As An Upper Bound

Previous work added an opt-in NVRTC CUDA no-split self-loop kernel. It keeps
one row's Neumann temporary vectors in shared memory:

```cuda
// one CUDA block per clade row
term[s]  = rhs[s]
vacc[s]  = rhs[s]
diag[s]  = q_D[s] + q_Tlocal[s]
pcoef[s] = q_Pibar[s] * inv_denom[s]

for iter in 0..neumann_terms-1:
    u[s] = term[s] * pcoef[s]
    A = block_sum_s(u[s])

    if correction_mode == tree:
        work[:] = bottom_up_subtree_sum(u[:])
        correction[s] = work[s]
    else:
        correction[s] = u[s]      // historical self-only control

    next[s] = term[s] * diag[s]
            + p_prime[s] * (A - correction[s])
            + term[parent[s]] * speciation_weight(parent[s] -> s)
    vacc[s] += next[s]
    swap(term, next)

write vacc[s]
atomic_add parameter gradients
```

Expected benefit:

- establishes whether shared-memory row-local tree DP can beat current Triton;
- gives an upper-bound/control for the Triton 2D and staged variants.

Risks:

- shared memory limits occupancy to about one CTA per SM for `S=1999`;
- currently fp32-only;
- the opt-in default correction mode is `self`, which is no longer the exact
  current semantic target after `221f16a`.

Test matrix:

| env flag | values |
|---|---|
| `GPUREC_CUDA_SELF_LOOP_NOSPLIT` | `0`, `1` |
| `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION` | `tree`, `self` |

Status: retested on the current full-ancestor baseline. Decision: keep opt-in
for now, but exact `tree` mode passes the 50-family performance threshold and
rescues the current 100-family OOM in the local control environment. Do not
promote until the implementation worker either changes the enabled default to
`tree` or adds a separate exact flag, and until a full `tests/kernels` pass plus
paired timing are run.

Additional coverage in `1b14648` now directly tests that CUDA no-split
`correction_mode="tree"` matches the synthetic PyTorch reference and that
`correction_mode="self"` differs on a topology where subtree correction matters.
That strengthens the semantic case for `tree` as the only exact Proposal 5
promotion candidate.

## Current Result Batch: Proposal 5 Control

Commit under test:

```text
38f42f5 Plan ancestor batching backward experiments
```

Workload:

```text
dataset=tests/data/test_trees_1000
families=first 50 unless stated otherwise
mode=global
pibar_mode=uniform
dtype=torch.float32
fixed_iters_Pi=6
neumann_terms=3
max_wave_size=32768
use_pruning=True
pruning_threshold=1e-6
```

### Correctness

Focused command run by the documentation pass:

```bash
GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 \
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree \
pytest -q \
  tests/gradients/test_autograd_bridge.py \
  tests/kernels/test_wave_backward_kernel.py::test_wave_backward_uniform_fused_supports_fp64_synthetic \
  tests/kernels/test_wave_backward_kernel.py::test_wave_backward_uniform_fused_supports_bf16_synthetic
```

Result:

```text
26 passed in 11.51s
```

Large-workload one-shot parity, documentation pass:

| Families | Mode | Loss diff | Theta grad max abs diff | Relative to baseline grad inf |
|---:|---|---:|---:|---:|
| 10 | CUDA `tree` vs default | `0` | `2.19726562e-03` | `1.65085469e-06` |
| 50 | CUDA `tree` vs default | `0` | `2.88085938e-02` | `4.47950811e-06` |

Local control parity supplied by the implementation/profiling worker for
50 families:

| Mode | Loss | Theta grad | Max abs diff vs default | Relative diff |
|---|---:|---|---:|---:|
| default | `107804.265625` | `[6431.1953, 5842.3198, 1833.2354]` | - | - |
| CUDA `tree` | `107804.265625` | `[6431.1680, 5842.3364, 1833.2520]` | `0.02734375` | `4.25e-6` |
| CUDA `self` | `107804.265625` | `[6431.1685, 5842.3315, 1833.2521]` | `0.02685547` | `4.18e-6` |

Interpretation: `tree` has the right semantics and the observed differences are
consistent with fp32 summation/order differences. `self` is also close on this
one theta-gradient projection, but it is not the exact VJP formula and should
remain a lower-bound diagnostic unless deliberately accepted as an
approximation.

### Event Timing

Canonical local control command supplied by the worker:

```bash
python profiling/proposal8/bench_uniform_backward.py \
  --fams 50 \
  --reps 7 \
  --warmups 4 \
  --cache-dir /tmp/gpurec_ancestor_batching_cache
```

with flags added for the CUDA variants:

```bash
GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 \
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree \
python profiling/proposal8/bench_uniform_backward.py \
  --fams 50 \
  --reps 7 \
  --warmups 4 \
  --cache-dir /tmp/gpurec_ancestor_batching_cache

GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 \
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=self \
python profiling/proposal8/bench_uniform_backward.py \
  --fams 50 \
  --reps 7 \
  --warmups 4 \
  --cache-dir /tmp/gpurec_ancestor_batching_cache
```

Worker local controls:

| Variant | Median backward | Mean | Min | Peak allocation | Loss | Notes |
|---|---:|---:|---:|---:|---:|---|
| default Triton | `331.434 ms` | `358.793 ms` | `326.854 ms` | `10.547 GB` | `107804.265625` | one `474 ms` outlier |
| CUDA no-split `tree` | `249.027 ms` | `274.346 ms` | `225.097 ms` | `10.091 GB` | `107804.265625` | one `489 ms` outlier |
| CUDA no-split `self` | `218.251 ms` | `218.238 ms` | `217.251 ms` | `10.091 GB` | `107804.265625` | semantic lower bound |

Against the default median, exact CUDA `tree` is `82.407 ms` faster
(`24.9%`) and reduces peak allocation by about `0.456 GB`. The `self` mode is
faster (`34.2%` median gain), but it should not be treated as promotable
without an explicit semantic decision.

Documentation-pass spot checks:

| Families | Variant | Reps/warmups | Median backward | Mean | Min | Peak allocation | Notes |
|---:|---|---|---:|---:|---:|---:|---|
| 10 | default Triton | `5/3` | `88.262 ms` | `88.509 ms` | `87.566 ms` | `2.812 GB` | current exact baseline |
| 10 | CUDA `tree` | `5/3` | `67.828 ms` | `67.941 ms` | `67.500 ms` | `2.096 GB` | `23.2%` median gain |
| 50 | default Triton | `5/3` | `328.482 ms` | `328.105 ms` | `326.542 ms` | `10.547 GB` | agrees with worker median direction |
| 50 | CUDA `tree` | `5/3` | `226.949 ms` | `227.481 ms` | `225.901 ms` | `10.091 GB` | faster run than worker median, same conclusion |
| 100 | default Triton | `3/2` | n/a | n/a | n/a | n/a | OOM allocating `accumulated_rhs` on this GPU snapshot |
| 100 | CUDA `tree` | `3/2` | `423.992 ms` | `423.906 ms` | `422.934 ms` | `18.153 GB` | completes |

The 100-family result should be read as a memory gate, not a paired speedup:
the default current exact path did not complete in this local environment, while
the CUDA no-split tree path did.

### Nsight Systems

Nsys commands:

```bash
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=4 MAX_WAVE_SIZE=32768 \
nsys profile --force-overwrite=true --stats=false \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  -o /tmp/gpurec_profile/ancestor_batching_doc/default_50 \
  python profiling/proposal8/bench_uniform_backward.py \
    --cache-dir /tmp/gpurec_ancestor_batching_cache

GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 \
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree \
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=4 MAX_WAVE_SIZE=32768 \
nsys profile --force-overwrite=true --stats=false \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  -o /tmp/gpurec_profile/ancestor_batching_doc/cuda_tree_50 \
  python profiling/proposal8/bench_uniform_backward.py \
    --cache-dir /tmp/gpurec_ancestor_batching_cache
```

Stats export:

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output /tmp/gpurec_profile/ancestor_batching_doc/default_50_kern \
  /tmp/gpurec_profile/ancestor_batching_doc/default_50.nsys-rep

nsys stats --report cuda_gpu_kern_sum --format csv \
  --output /tmp/gpurec_profile/ancestor_batching_doc/cuda_tree_50_kern \
  /tmp/gpurec_profile/ancestor_batching_doc/cuda_tree_50.nsys-rep
```

Single profiled backward intervals:

| Metric | Default Triton | CUDA `tree` |
|---|---:|---:|
| benchmark-reported backward under Nsys | `359.861 ms` | `249.268 ms` |
| `_wave_backward_uniform_kernel` | `267.494 ms` / 36 launches | `140.967 ms` / 33 launches |
| `gpurec_wave_backward_nosplit_uniform_fp32` | - | `16.503 ms` / 3 launches |
| total self-loop bucket including CUDA replacements | `267.494 ms` | `157.471 ms` |
| `_dts_cross_backward_accum_kernel` | `27.312 ms` / 33 | `27.338 ms` / 33 |
| `_uniform_cross_pibar_vjp_tree_from_ud_compact_kernel` | `16.297 ms` / 33 | `16.187 ms` / 33 |
| `_dts_parent_reduced_ge2_stage1_kernel` | `7.500 ms` / 6 | `7.473 ms` / 6 |
| kernel launches from Nsys summary | unchanged for dominant non-self-loop buckets | unchanged for dominant non-self-loop buckets |

Launch order for the three no-split waves:

| No-split launch | Default Triton | CUDA `tree` |
|---:|---:|---:|
| `W=15009` | `24.348 ms` | `3.080 ms` |
| `W=32768` | `52.356 ms` | `6.709 ms` |
| `W=32768` | `49.780 ms` | `6.714 ms` |
| total | `126.484 ms` | `16.503 ms` |

Interpretation: the end-to-end Nsys improvement is almost entirely explained by
the three no-split self-loop replacements. DTS accumulation, cross-Pibar VJP,
and other hot buckets are essentially unchanged. The remaining
`_wave_backward_uniform_kernel` bucket in CUDA `tree` mode is split-wave work,
which Proposal 5 intentionally does not handle.

### Nsight Compute

NCU commands, using `--kernel-id` selectors:

```bash
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=4 MAX_WAVE_SIZE=32768 \
ncu --target-processes all --profile-from-start off \
  --kernel-id ::regex:_wave_backward_uniform_kernel:35 \
  --launch-count 1 \
  --set detailed --csv --page raw \
  --log-file /tmp/gpurec_profile/ancestor_batching_doc/ncu_default_wave35.csv \
  python profiling/proposal8/bench_uniform_backward.py \
    --cache-dir /tmp/gpurec_ancestor_batching_cache

GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 \
GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree \
PROFILE_CUDA_API=1 FAMS=50 REPS=1 WARMUPS=4 MAX_WAVE_SIZE=32768 \
ncu --target-processes all --profile-from-start off \
  --kernel-id ::regex:gpurec_wave_backward_nosplit_uniform_fp32:2 \
  --launch-count 1 \
  --set detailed --csv --page raw \
  --log-file /tmp/gpurec_profile/ancestor_batching_doc/ncu_cuda_tree_wave2.csv \
  python profiling/proposal8/bench_uniform_backward.py \
    --cache-dir /tmp/gpurec_ancestor_batching_cache
```

Representative largest `W=32768` no-split launch:

| Metric | Default Triton exact | CUDA `tree` exact |
|---|---:|---:|
| NCU duration | `58.190 ms` | `7.352 ms` |
| grid / block | `32768 x 128` | `32768 x 256` |
| registers/thread | `40` | `40` |
| dynamic shared memory/block | `2.0 KiB` | `55,972 B` |
| shared-memory occupancy limit | `21 blocks/SM` | `1 block/SM` |
| active warps | `99.11%`, `47.57` warps/SM | `16.62%`, `7.98` warps/SM |
| DRAM read bytes | `7.452 GB` | `0.787 GB` |
| DRAM write bytes | `4.704 GB` | `0.246 GB` |
| total DRAM bytes | `12.156 GB` | `1.033 GB` |
| L2 throughput | `74.11%` | `21.60%` |
| DRAM throughput | `21.25%` | `14.29%` |
| SM throughput | `18.25%` | `22.62%` |
| global load instructions | `308.150 M` | `103.481 M` |
| global store instructions | `47.481 M` | `2.064 M` |
| global RED instructions | `11.010 M` | `10.387 M` |
| excessive L2 global-sector estimate | `1.258 GB` | `59.425 MB` |
| local spilling requests | `1,048,576` | `0` |
| long-scoreboard stall samples | `1,365,016` | `318,076` |
| barrier stall samples | `504,519` | `293,123` |

Interpretation:

- The current exact Triton launch is dominated by the full ancestor scatter:
  it moves about `12.16 GB` in the selected launch and issues hundreds of
  millions of global load/store instructions.
- CUDA `tree` still has the known shared-memory occupancy problem: one CTA per
  SM and only about `16.6%` active warps. On the old self-only baseline that
  occupancy problem made it only marginal. Against the current exact Triton
  ancestor-walk baseline, the reduction in global traffic is so large that the
  shared-memory row kernel wins anyway.
- Global RED instructions barely move because both kernels still use atomics
  for parameter-gradient accumulation. Proposal 5 solves the ancestor/subtree
  correction traffic, not final parameter-gradient atomics.
- The remaining performance work is to get the CUDA tree memory behavior
  without the one-row-per-block occupancy cap, or to extend tree batching to
  split waves.

## Proposal Decisions

| Proposal | Decision | Rationale |
|---|---|---|
| 0: 2D Triton self-loop | keep opt-in; promising but memory-heavy | The prototype passes parity and beats baseline/P5 on event timing at 50 families, but peak allocation rises to `14.217 GB` at 50 and `21.329 GB` at 100. Needs Nsys/NCU before promotion. |
| 1: staged Triton tree DP | reject current configuration | The `W=4,S=256` staged prototype passes parity, but regresses to `457.069 ms` median backward at 50 families and OOMs before valid 100-family timing. |
| 2: species-major/transposed scratch | defer | No production implementation. Current species ids already provide contiguous subtree intervals; transposition may cost more than it saves. |
| 3: hybrid router | required if promoting | No threshold router exists. Current evidence supports routing large no-split fp32 waves to exact CUDA `tree`; small/split/bf16 cases need separate thresholds. |
| 4: forward path prefix | reject for fp32 production for now | No production implementation. Older fp32 Pibar-only prefix prototypes were slower and forward is not the current bottleneck after the bf16 fix. |
| 5: CUDA no-split row kernel | keep opt-in; exact `tree` is promising | `tree` mode cuts the 50-family median by about `25%`, explains the Nsys self-loop bucket reduction, and completes 100 families where the current default OOMed locally. Needs full-test and default-mode cleanup before promotion. |

## Open Follow-Ups

1. Implementation worker: make exact `tree` mode the only candidate for any
   Proposal 5 promotion. Consider changing the opt-in default correction mode
   or adding a more explicit flag name so `self` cannot be mistaken for current
   exact semantics.
2. Correctness worker: run the full `tests/kernels/test_wave_backward_kernel.py`,
   `tests/gradients/test_autograd_bridge.py`, and
   `tests/gradients/test_uniform_backward_ancestor_batching.py` suites with
   `GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree`.
3. Profiling worker: collect alternating 50-family paired runs for default vs
   CUDA `tree`, and rerun the 100-family default on a clean-memory GPU to
   distinguish true OOM from local memory pressure.
4. Profiling worker: collect Nsys/NCU for Proposal 0 `BLOCK_W=2` on the
   50-family workload. Focus on registers, spills, DRAM/L2 traffic, launch
   count, and why peak allocation rises by about `4.4 GB`.
5. Implementation worker: do not promote the current Proposal 1 staged kernel.
   If staged work continues, reduce launch count and scratch footprint before
   rerunning broad timings.
6. Profiling worker: if Proposal 5 is considered for default, collect one Nsys
   capture with `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=self` only as a
   semantic lower-bound comparison, not as a promotion candidate.

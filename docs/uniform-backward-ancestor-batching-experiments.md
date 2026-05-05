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

Documentation-owner constraint for this pass: no production code or tests were
edited. All current measurements below were collected with existing opt-in
paths.

## Current Result At A Glance

The most important current result is Proposal 5: the existing NVRTC CUDA
no-split row kernel, run in exact `tree` correction mode, is a strong control
for the current full-ancestor baseline.

| Item | Current status | Decision |
|---|---|---|
| Proposal 0, 2D Triton self-loop | not implemented in current tree | keep as future experiment; high register-risk |
| Proposal 1, staged Triton tree DP | not implemented in current tree | highest-priority new implementation after Proposal 5 control |
| Proposal 2, species-major/transposed scratch | not implemented in current tree | lower priority; current species ids already have contiguous subtrees |
| Proposal 3, hybrid wave router | not implemented directly | needed if Proposal 1 or 5 is promoted |
| Proposal 4, forward path prefix | not implemented for this pass | reject as fp32 production direction until a full fused forward kernel exists |
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
| `GPUREC_SELF_LOOP_2D_TRITON` | no | n/a | Proposal 0, not implemented. |
| `GPUREC_SELF_LOOP_TREE_STAGED` | no | n/a | Proposal 1, not implemented. |
| `GPUREC_SELF_LOOP_TREE_TRANSPOSED` | no | n/a | Proposal 2, not implemented. |
| `GPUREC_FORWARD_UNIFORM_PATH_PREFIX` | no | n/a | Proposal 4, not implemented. |

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

Promotion threshold:

- A prototype must reduce end-to-end backward time by at least `3%` on the
  50-family workload and not regress 100-family timing.
- If it only improves one kernel bucket, it must be left opt-in unless the
  end-to-end gain survives alternating paired runs.

For Proposal 5 specifically, current promotion must also require changing the
enabled correction mode to `tree`, or otherwise documenting `self` as an
intentional approximation. The current exact semantics are not `self`.

## Proposal 0: 2D Triton Self-Loop Kernel

Implement a Triton variant where one program processes a block of clade rows and
the full species vector:

```python
offs_w = block_w + arange(BLOCK_W)
offs_s = arange(BLOCK_S)          # BLOCK_S ~= next_power_of_2(S)

term      = load(term_ptr      + offs_w[:, None] * S + offs_s[None, :])
pibar_c   = load(pibar_coeff   + offs_w[:, None] * S + offs_s[None, :])
u         = term * pibar_c
A         = sum(u, axis=1)

# bottom-up tree reduction, still inside the program
corr = u
for level in postorder_levels:
    corr[:, parent] = corr[:, parent] + corr[:, child1] + corr[:, child2]

result = term * diag + p_prime * (A[:, None] - corr) + speciation_gather(term)
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

Status: not implemented. Decision: keep as a high-risk experiment. The current
CUDA tree-mode result shows that full-row local tree DP is valuable, but a
Triton 2D implementation must prove it can avoid register spill and occupancy
collapse.

## Proposal 1: Extra Triton Kernels For Tree DP

Split the fused self-loop into explicit stages for large waves:

```python
precompute_weights(...)

for iter in range(neumann_terms):
    make_u_and_A(term, pibar_coeff)       # [W, S] u_d and [W] A
    subtree_reduce_by_levels(u)           # corr[w, s] = subtree_sum_s(u[w])
    combine_next(term, corr, A, weights)  # produces next term and accumulates v

param_vjp(...)
```

This may use one kernel per stage per Neumann iteration, or one kernel per tree
level for the reduction if global inter-level synchronization is needed.

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

Status: not implemented. Decision: this is now the highest-priority new
implementation proposal. Proposal 5 proves that no-split full-row tree
correction can remove about `110 ms` from the 50-family profiled interval, but
split waves still run through the current Triton ancestor-walk path and account
for about `141 ms` of self-loop time in the CUDA-tree Nsys capture.

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

Status: not implemented. Decision: lower priority than Proposal 1. Older
Euler-layout diagnostics showed that, for `tests/data/test_trees_1000`, every
subtree is already one contiguous current-order interval. That reduces the
expected value of a broad transposed/species relayout. A narrow interval-prefix
self-loop variant may be more promising than a full species-major scratch.

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

Status: not implemented as a new router, but the existing CUDA no-split flag is
already a coarse router for no-split fp32 waves. Decision: required before any
default promotion. Current data support routing only large no-split waves to
the CUDA tree path; split waves need Proposal 1 or another implementation.

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

Status: not implemented in this pass. Decision: do not pursue a production fp32
forward path until a full fused forward DTS_L/Pibar kernel exists. Older
Pibar-only CUDA prefix results were correct but `2.0-2.4x` slower than the
current fp32 parent-walk Pibar kernel; fp64 remained a separate promising
direction.

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
| 0: 2D Triton self-loop | pending, high-risk | Full-row tree correction is valuable, but a single Triton program over `S=1999` risks excessive vector lanes and spills. |
| 1: staged Triton tree DP | pursue next | Kernel-boundary synchronization can batch row/species work and may handle split waves, which remain the largest self-loop bucket after Proposal 5. |
| 2: species-major/transposed scratch | defer | Current species ids already provide contiguous subtree intervals; transposition may cost more than it saves. |
| 3: hybrid router | required if promoting | Current evidence supports routing large no-split fp32 waves to exact CUDA `tree`; small/split/bf16 cases need separate thresholds. |
| 4: forward path prefix | reject for fp32 production for now | Older fp32 Pibar-only prefix prototypes were slower and forward is not the current bottleneck after the bf16 fix. |
| 5: CUDA no-split row kernel | keep opt-in; exact `tree` is promising | `tree` mode cuts the 50-family median by about `25%`, explains the Nsys self-loop bucket reduction, and completes 100 families where the current default OOMed locally. Needs full-test and default-mode cleanup before promotion. |

## Open Follow-Ups

1. Implementation worker: make exact `tree` mode the only candidate for any
   Proposal 5 promotion. Consider changing the opt-in default correction mode
   or adding a more explicit flag name so `self` cannot be mistaken for current
   exact semantics.
2. Correctness worker: run the full `tests/kernels/test_wave_backward_kernel.py`
   and `tests/gradients/test_autograd_bridge.py` suites with
   `GPUREC_CUDA_SELF_LOOP_NOSPLIT=1 GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=tree`.
3. Profiling worker: collect alternating 50-family paired runs for default vs
   CUDA `tree`, and rerun the 100-family default on a clean-memory GPU to
   distinguish true OOM from local memory pressure.
4. Implementation worker: prototype Proposal 1 for split waves, because Nsys
   shows about `140.967 ms` of split-wave Triton self-loop time remains after
   routing no-split waves to CUDA `tree`.
5. Profiling worker: if Proposal 5 is considered for default, collect one Nsys
   capture with `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION=self` only as a
   semantic lower-bound comparison, not as a promotion candidate.

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
             + speciation_parent_gather(w, s)
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

The target is not just removing atomics. The target is reducing uncoalesced
topology traffic and repeated ancestor visits.

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

Status: not started.

## Proposal 1: Extra Triton Kernels for Tree DP

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

Status: not started.

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

Status: not started.

## Proposal 3: Hybrid Wave Router

Use the current fused kernel for small or awkward waves and route only large
waves to the staged or 2D tree path.

Candidate rule:

```python
if W >= threshold and S <= max_s and dtype in supported_dtypes:
    use_tree_variant()
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

Status: not started.

## Proposal 4: Forward Uniform-Pibar Tree Prefix

The forward denominator also computes:

```python
denom[w, s] = row_sum[w] - sum_{a in ancestors(s)} p_prime[w, a]
```

Since the current species order appears to have contiguous subtrees on
`test_trees_1000`, test an Euler/prefix formulation:

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

Status: not started.

## Proposal 5: Existing CUDA Shared-Memory Row Kernel as an Upper Bound

Previous work added an opt-in NVRTC CUDA no-split self-loop kernel. Re-test it
against the current post-bf16-atomic baseline, because it already keeps one
row's Neumann temporary vectors in shared memory:

```cuda
u[s] = term[s] * pibar_coeff[s]
A = block_sum(u)
subtree_reduce_in_shared_memory(u)
next[s] = term[s] * diag[s] + p_prime[s] * (A - u[s]) + speciation_gather(...)
```

Expected benefit:

- establishes whether shared-memory row-local tree DP can beat current Triton;
- gives an upper-bound/control for the Triton 2D and staged variants.

Risks:

- previous 50-family measurements were only marginal;
- shared memory limits occupancy to about one CTA per SM for `S=1999`;
- currently fp32-only unless extended.

Test matrix:

| env flag | values |
|---|---|
| `GPUREC_CUDA_SELF_LOOP_NOSPLIT` | `0`, `1` |
| `GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION` | `self`, `tree` |

Status: needs re-test on current baseline.

## Running Results

No new experiments have been run for this pass yet. Results should be appended
under this section with:

- commit hash;
- exact env flags and commands;
- correctness results;
- benchmark table before/after;
- Nsys kernel bucket deltas;
- NCU resource counters for the replacement kernel;
- decision: promote, keep opt-in, or reject.

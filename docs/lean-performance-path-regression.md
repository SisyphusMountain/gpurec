# Lean Branch Performance Path Regression

## Problem

The lean branch was supposed to keep only the highest-performance paths. It currently does the opposite for the main HOGENOM/genewise use case: genewise CUDA forward is routed away from the kernelized uniform path and back into a PyTorch sparse/dense Pibar fallback.

In `gpurec/core/forward.py`, `use_uniform_linear` is gated as CUDA and not batched. Genewise mode passes `family_idx`, which makes the forward path batched, so the fast uniform path is disabled. The code then calls `_compute_Pibar_uniform_inline`, which computes:

```python
Pi_exp @ ancestors_T
```

That is not the intended high-performance Pibar implementation.

## Intended Fast Path

For uniform Pibar, the efficient computation is:

1. Compute a stabilized row sum over `exp2(Pi)`.
2. Walk each species' ancestors using precomputed ancestor pointers or columns.
3. Subtract ancestor mass from the row sum.
4. Write `Pibar` directly from the kernel.

The remaining `wave_pibar_uniform_fused` kernel does this with `ancestor_cols`, but it only supports the shared/global `mt[S]` layout. It is therefore only used for non-genewise CUDA runs. The genewise path needs the same kernelized ancestor-walk computation with family-indexed constants, such as `mt[G, S]` addressed through `family_idx`, or an equivalent wave-local `mt[W, S]` layout.

## What Went Wrong

The codebase reduction removed several alternate uniform wrappers, including the CSR-style uniform forward wrapper, but did not preserve a genewise-capable version of the best Pibar path. The result is a leaner codebase on paper, but not a lean high-performance implementation.

This is especially bad because genewise HOGENOM is the performance-critical workload.

## Required Correction

The lean version should keep one production uniform CUDA path that works for all retained modes:

- global
- specieswise
- genewise

For genewise, that path must not fall back to `_compute_Pibar_uniform_inline` on CUDA. The retained Pibar kernel should support family-indexed `mt`/rate constants through `family_idx`, and the forward path should enable it for genewise.

The PyTorch sparse-matmul Pibar implementation should be CPU/debug fallback only, not the CUDA genewise hot path.

## Resolution

The retained `wave_pibar_uniform_fused` kernel now supports all production
uniform layouts:

- shared/global/specieswise `mt[S]`;
- wave-local `mt[W, S]`;
- genewise family-indexed `mt[G, S]` addressed through `family_idx`.

`Pi_wave_forward` now enables the CUDA ancestor-walk Pibar kernel whenever it
runs on CUDA. The sparse `_compute_Pibar_uniform_inline` path is left for
CPU/non-CUDA fallback only. Genewise CUDA forward recomputes final Pibar rows
with the kernel after the fixed-point loop so later waves and backward receive
Pibar values for the final Pi state.

Validation added:

- kernel parity for shared, wave-local, and family-indexed Pibar layouts;
- CUDA forward tests for all retained modes that monkeypatch
  `_compute_Pibar_uniform_inline` to fail if the sparse fallback is reached;
- HOGENOM strict-path smoke:
  `python profiling/bench_genewise_forward_chunking.py --dataset tests/data/hogenom_bench --fams 20 --family-chunk-size 20 --warmups 1 --reps 1 --cache-dir /tmp/gpurec_hogenom_fast_cache`
  reported `optimized_forward_status optimized 1` and `forward_ms 29.824`.

## Related DTS Issue

The current `_compute_dts_cross` still calls the fused DTS term kernel, but then materializes split terms and reduces them to parent rows outside the DTS kernel. The previously available parent-reduced DTS path was removed from the retained forward path. That may also conflict with the "only highest-performance path" goal and should be benchmarked before being discarded.

## `test_trees_1000` Benchmark Comparison

Date: 2026-05-11

Environment:

- branch: `lean-scheduled-optimizers`
- commit: `bcf23d4`
- GPU: NVIDIA GeForce RTX 4090
- dataset: `tests/data/test_trees_1000`
- note: an existing Jupyter kernel was using about `8.8 GiB` of GPU memory, so
  the chunk-100 genewise run could not be measured cleanly in this session.

Historical untracked reference documents from the original performance workspace:

- `docs/genewise-forward-backward-optimization-proposals.md`
- `docs/uniform-backward-50tree-wave2-profile.md`

Those reference names and several benchmark harnesses named below came from
the historical performance workspace and are not all tracked in the current
branch. Missing historical harness names include
`profiling/bench_genewise_forward_chunking.py`,
`profiling/bench_genewise_backward_chunking.py`, and
`profiling/ancestor_batching/bench_uniform_backward.py`. Treat the commands in
this section as provenance for the recorded numbers, not as a current
reproducible command set.

Historical commands run, not reproducible from a clean checkout:

```bash
PREPROCESS_CACHE_DIR=/tmp/gpurec_test_trees_1000_bench \
python profiling/bench_genewise_forward_chunking.py \
  --dataset tests/data/test_trees_1000 \
  --fams 1000 \
  --family-chunk-size 50 \
  --warmups 1 \
  --reps 3 \
  --max-wave-size 32768 \
  --cache-dir /tmp/gpurec_test_trees_1000_bench

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PREPROCESS_CACHE_DIR=/tmp/gpurec_test_trees_1000_bench \
python profiling/bench_genewise_backward_chunking.py \
  --dataset tests/data/test_trees_1000 \
  --fams 1000 \
  --family-chunk-size 50 \
  --warmups 1 \
  --reps 3 \
  --backward-path optimized-genewise \
  --strict-optimized-kernels \
  --max-wave-size 32768 \
  --cache-dir /tmp/gpurec_test_trees_1000_bench

PREPROCESS_CACHE_DIR=/tmp/gpurec_test_trees_1000_bench \
python profiling/ancestor_batching/bench_uniform_backward.py \
  --dataset tests/data/test_trees_1000 \
  --fams 50 \
  --warmups 5 \
  --reps 9 \
  --max-wave-size 32768 \
  --variant-label current-lean \
  --cache-dir /tmp/gpurec_test_trees_1000_bench
```

Results:

| Benchmark | Documented target | Current result | Delta |
|---|---:|---:|---:|
| Genewise forward, 1000 trees | `2.328 s` with chunks `150x6 + 100` | `5.819 s` median with chunk 50 | not comparable; chunk policy differs |
| Genewise chunked training, 1000 trees, chunk 50 | `10.652 s` total, `10.787 GB` peak | `11.856 s` median total, `13.732 GB` peak | `1.11x` slower, `1.27x` peak memory |
| Uniform/global backward, 50 trees | `157.329 ms` median backward, `10.305 GB` peak | `336.828 ms` median backward, `10.262 GiB` peak | `2.14x` slower |

Additional observations:

- The genewise forward-only wrapper still reports `optimized_forward_status
  optimized 1`, but the chunk-50 run is not directly comparable to the
  documented `2.328 s` forward-only envelope because that envelope used chunks
  `150x6 + 100`.
- The forward-only wrapper is also not the same semantic path as the forward
  phase inside the backward wrapper. It runs under `torch.no_grad()`, uses root
  rows only, sets `need_pibar=False`, saves no backward tensors, and hardcodes
  `use_pruning=False`. The backward wrapper runs with gradients enabled, saves
  full wave-ordered `Pi`/`Pibar` for implicit backward, and uses pruning by
  default.
- The full genewise backward path still reports the strict optimized verdict,
  and its warmed full-pass regression is moderate relative to the chunk-50
  training envelope. The forward component remains the largest gap:
  `5.914 s` of the `11.856 s` total in the backward wrapper.
- A 50-family resident genewise backward slice is fast in isolation:
  `286.278 ms` forward, `300.567 ms` backward, and `586.845 ms` total. The
  regression shows up across the full 20 chunk-50 training pass, especially
  when comparing the lean training forward component with current `main`.
- Chunk 100 could not be timed because the first 100-family genewise forward
  pass attempted an additional `4.73 GiB` allocation with only about `4.53 GiB`
  free under the existing Jupyter GPU allocation.

Conclusion:

The initial forward-only comparison was confounded by chunk policy: the
documented `2.328 s` result used larger chunks than the measured chunk-50 run.
The real lean regression is the chunk-50 training forward component: the same
benchmark on current `main` spends about `2.739 s` in forward, while lean spends
about `5.914 s`. The uniform/global backward mismatch is also real relative to
the old document, but it reproduces on current `main` and therefore is not
specific to this lean branch. Before treating this branch as a lean replacement,
the retained path needs a performance gate that checks full training
forward+backward numbers, not only optimized flag verdicts.

## Main-Branch Verification

Date: 2026-05-11

To avoid disturbing the dirty lean worktree, verification was run in a clean
main worktree:

```text
commit 5d1433f6ec90756265a64e0cb960eae538f6da71
```

The `test_trees_1000` dataset is untracked in that clean worktree, so commands
used the dataset path from the original workspace:

```text
$GPUREC_WORKSPACE/tests/data/test_trees_1000
```

Main results:

| Benchmark | Documented target | Main result | Lean result |
|---|---:|---:|---:|
| Genewise chunked training, 1000 trees, chunk 50 | `10.652 s` total, `10.787 GB` peak | `8.766 s` total, `13.799 GB` peak | `11.856 s` total, `13.732 GB` peak |
| Genewise training forward component, chunk 50 | `2.980 s` | `2.739 s` | `5.914 s` |
| Genewise training backward component, chunk 50 | `7.672 s` | `6.008 s` | `5.941 s` |
| Uniform/global backward, 50 trees | `157.329 ms` median backward | `335.349 ms` | `336.828 ms` |

Interpretation:

- Current local `main` still preserves the high-performance genewise training
  path. Its chunk-50 full training pass is faster than the documented Proposal
  5 number and much faster than the lean branch.
- The lean regression is concentrated in the forward component used inside the
  genewise training path: `main` spends about `2.739 s`, while lean spends
  about `5.914 s` for the same 20 chunk-50 forward passes.
- Current local `main` does not reproduce the older uniform/global backward
  `157 ms` profile. It matches lean at about `335-337 ms`, so that regression
  either predates the lean branch or depends on a benchmark harness/state from
  the documented `ddbfa22` profile. The `ddbfa22` checkout in this repository
  does not contain the later `profiling/ancestor_batching/bench_uniform_backward.py`
  harness, so the exact old command could not be rerun directly from that
  commit.

Cause of the lean genewise regression:

- `main` keeps the fused uniform wrappers in `gpurec/core/kernels/wave_step.py`,
  including `wave_step_uniform_fused_into`,
  `wave_step_uniform_ancestor_fused`, `wave_step_uniform_csr_fused`,
  `wave_step_uniform_two_kernel_fused`, and
  `wave_pibar_uniform_parent_fused`.
- Lean replaced those wrappers with a `_REMOVED_UNIFORM_WRAPPERS` stub and kept
  only the generic `wave_step_fused`, `wave_step_uniform_linear_fused`, and
  `wave_pibar_uniform_fused` route.
- `main` also still routes forward DTS through `dts_fused_parent_reduced`.
  Lean removed that parent-reduced DTS entrypoint from the retained forward
  path.
- The later lean patch made genewise Pibar kernelized again, but it did not
  restore the fused uniform wave-step/ping-pong path or parent-reduced DTS. That
  left the code with an "optimized" flag verdict but not the same optimized
  algorithm.

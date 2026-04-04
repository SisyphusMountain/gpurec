# Project State

## Architecture

- **Wave-only**: `Pi_wave_forward` is the sole Pi forward pass. The v1 gather/scatter implementation was removed 2026-03-14.
- **FP kept for debug**: `Pi_fixed_point` / `Pi_step` remain for debugging and genewise fallback.
- **API**:
  - `GeneDataset.compute_likelihood_batch()` — always uses wave (no `use_wave`/`wave_version` params). Falls back to FP only for genewise.
  - `GeneDataset.compute_likelihood(idx, use_wave=False)` — single-family, FP by default, `use_wave=True` for wave.

## Pibar Computation Strategies

Three strategies for computing Pibar (the transfer-weighted sum of Pi), distinguished by the transfer rate structure:

1. **Uniform** (scalar T): `Pibar[c,s] = sum_j(Pi[c,j]) - sum_{j∈ancestors(s)}(Pi[c,j])`, then scale by `mt[s]`. Unweighted sum + sparse ancestor correction. O(W×S + W×nnz_ancestors).
   - Code `pibar_mode='uniform'`: exact (uses `ancestors_T` sparse matmul for ancestor correction)

2. **Specieswise** (per-donor T[s]): `Pibar[c,s] = sum_j(Pi[c,j]·T[j]) - sum_{j∈ancestors(s)}(Pi[c,j]·T[j])`. Weighted sum + sparse ancestor correction. Same sparsity pattern as uniform, just weighted.

3. **Pairwise** (per-pair T[i,j]): full [S,S] log-matmul, or top-k sparsification of Pi columns. Code `pibar_mode='dense'`. O(W×S²) exact, or O(W×S×k) with top-k.

## Mode Support Matrix

Rows: Pibar strategy (code `pibar_mode`). Columns: parameter structure.

| pibar_mode \ param_mode | global | specieswise | pairwise |
|---|---|---|---|
| **pairwise** (`dense`) | fwd ✓, bwd ✓, FD ✓ (3.8e-7), tol 0.01%, 15+ tests | fwd ✓, bwd ✓, FD ✓ (2.3e-7), tol 0.01%, 1 test | fwd ✓, bwd N/A, 0 tests |
| **uniform** (`uniform`) | fwd ✓, bwd ✓, FD ✓ (2.6e-7), tol 0.1%, 10+ tests | fwd ✓, bwd ✓, FD ✓ (3.4e-7), tol 0.01%, 1 test | invalid (uniform assumes scalar T) |

FD error is max relative error vs central finite differences (float64). Tolerance is the assert threshold in the test.

**genewise**: per-family wave loop (not cross-family batched).

### Specieswise Notes
- Forward was broken before 2026-03-13: `dts_fused.py` collapsed per-species log_pD/log_pS to scalars via `.mean()`. Now fixed to pass per-species vectors through the kernel.
- Forward correctness test: `TestSpecieswiseForwardConsistency` verifies that specieswise with uniform rates matches global mode.
- Three FD gradient tests (`uniform`, `dense`, `uniform`) validate the full backward chain.

## Performance

| Metric | Value |
|---|---|
| S=1999 wave vs FP | 18.6x faster (~200ms/family) |
| S=19999 (uniform) | 0.26s/family |
| Fused kernel bandwidth | 81% peak BW at S=20K |
| Small-S (≤256) | Slower than FP (known issue) |
| Peak memory at S=20K | ~18 GB on 24 GB GPU |

## Optimizer Support

- **`optimize_theta_wave`**: L-BFGS optimizer with implicit gradient (single-family, any pibar_mode). Warm-start E across iterations. Armijo line search.
- **`optimize_theta_genewise`**: L-BFGS with **batched backward** across families. Single `Pi_wave_backward` call with `family_idx` for all families, per-gene convergence masking. Per-family forward → merged Pi → single batched backward → per-family E adjoint.
- **E adjoint**: CG solver with GMRES fallback (17 CG iterations typical).
- **Gradient pruning**: Per-clade adjoint thresholding in `Pi_wave_backward` — skips clades where adjoint magnitude < `pruning_threshold`. Achieves 49% clade pruning on test data.

## Benchmark: Forward Pass (1 family, float32)

Time for extract_parameters + E_fixed_point + Pi_wave_forward + logL.
Wave layout build excluded (one-time cost).
GPU: RTX 4090 (24 GB). Measured 2026-03-13.

### S = 1999 (test_trees_1000)

| pibar_mode \ param_mode | global | specieswise | pairwise |
|---|---|---|---|
| **pairwise** (`dense`) | 87 ms | 77 ms | N/A |
| **uniform** (`uniform`) | 52 ms | 57 ms | N/A |
| **uniform exact** (`uniform`) | 96 ms | 76 ms | N/A |

### S = 19999 (test_trees_10000)

| pibar_mode \ param_mode | global | specieswise | pairwise |
|---|---|---|---|
| **pairwise** (`dense`) | OOM | OOM | N/A |
| **uniform** (`uniform`) | 668 ms | 624 ms | N/A |
| **uniform exact** (`uniform`) | OOM | OOM | N/A |

Pairwise and uniform-exact require [S,S] matrices on GPU (~3 GB each at S=20K).

## Dead Code Cleanup (2026-03-14)

Removed v1 wave path and stale code:

**likelihood.py**: Deleted v1 `Pi_wave_forward`, `_compute_Pibar_wave`, `_compute_Pibar_wave_compressed` (~276 lines).

**model.py**: `compute_likelihood_batch` always uses wave (removed `use_wave`, `wave_version` params). `compute_likelihood` wave branch switched from v1 to wave-ordered layout.

**Tests converted v1 to wave**: `test_wave_vs_fp.py`, `test_cross_family_wave.py`, `bench_scale.py`. **test_wave_v2.py**: removed v1-comparison tests, kept wave-vs-FP + model API tests.

**Deleted 18 stale files**: 9 broken tests (`test_convergence_analysis`, `test_parameter_optimization_timing`, `test_wave_pi_equivalence`, `test_implicit_vjp`, `test_pi_update_rigorous`, `test_E_update_rigorous`, `test_scatter_rigorous`, `test_likelihood_comparison`, `converged_data`), 2 scripts (`optimize_theta`, `profile_likelihood`), 5 profiling scripts (`explore_compressed_pibar`, `explore_hybrid_pibar`, `explore_tl_separation`, `explore_precomputed_topk`, `explore_warmstart_compressed`), 2 docs (`scheduling_proposal.md`, `scheduling_proposal_2.md`).

**Tolerance widened**: wave-vs-FP tests use LOGL_ATOL=5e-2 (was 1e-2). The wave clade permutation changes FP operation ordering, causing ~0.03 absolute diffs on logL~2000 (~1.5e-5 relative).

## Known Issues & Gaps

- **Small-S kernel slower than FP**: fused Triton kernel underperforms fixed-point for S ≤ 256.
- **No pairwise gradient support**: forward-only for pairwise parameter mode.
- **Batched backward only validated at small S**: forward works at S=20K, batched backward not yet tested at large S.
- **uniform OOMs at large S**: uses dense [S,S] `recipients_T` matmul — should exploit ancestor sparsity (O(depth) per species) instead.
- **uniform coverage**: only 1 FD test at S=39; no large-S validation.
- **`build_wave_layout` memory**: ~8 MB at S=20K (perm arrays + split metadata). Previously misreported as 6.7 GB.

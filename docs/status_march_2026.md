# Project Status Report — gpurec (March 16, 2026)

## Overview

GPU-accelerated gene tree / species tree reconciliation under the DTL (Duplication, Transfer, Loss) model. Core computation: fixed-point iteration for Pi[C,S] and E[S] matrices in log2-space, with wave-ordered scheduling for parallelism.

---

## 1. Architecture

### Computational Pipeline

```
theta → extract_parameters → (log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat)
                                    ↓
                              E_fixed_point  (E_step iterated to convergence)
                                    ↓
                              Pi computation (wave or fixed-point)
                                    ↓
                              compute_log_likelihood
```

**Two Pi solvers:**
- `Pi_fixed_point`: Global Jacobi iteration. Simple but slow. Used as `use_wave=False` fallback.
- `Pi_wave_forward`: Wave-ordered layout with per-wave convergence. 18.6x faster than FP at S=1999. Default path.

**Backward pass** (for optimization):
- `Pi_wave_backward`: Neumann-series implicit gradient through Pi self-loop
- E adjoint: CG solve with GMRES fallback
- Full chain: `implicit_grad_loglik_vjp_wave` → dNLL/dtheta

### Key Files

| File | Role |
|------|------|
| `src/core/likelihood.py` | Pi_step, E_step, Pi_fixed_point, Pi_wave_forward, Pi_wave_backward |
| `src/core/terms.py` | compute_DTS (5-term), compute_DTS_L (6-term) |
| `src/core/extract_parameters.py` | theta → log2-space event probabilities + transfer matrix |
| `src/core/model.py` | GeneDataset, compute_likelihood, compute_likelihood_batch |
| `src/core/batching.py` | collate_gene_families, collate_wave, build_wave_layout |
| `src/core/scheduling.py` | compute_clade_waves (C++ phased scheduler) |
| `src/core/kernels/dts_fused.py` | Fused Triton DTS kernel |
| `src/core/kernels/wave_step.py` | Fused Triton Pibar+DTS_L+convergence kernel |
| `src/core/kernels/scatter_lse.py` | Segmented logsumexp Triton kernel |
| `src/optimization/theta_optimizer.py` | Adam, L-BFGS-B, implicit gradient VJP |
| `logmatmul/` | Log2-space matmul library (dense, sparse, compressed) |
| `rustree/` | Rust tree metrics + simulation (Python/R bindings) |

---

## 2. Parameter Modes

Two categorical axes (not three independent booleans):

- **transfer_mode** ∈ {uniform, specieswise, pairwise} — how transfer rates vary across species
- **gene_granularity** ∈ {uniform, genewise} — shared vs per-gene rates

Specieswise and pairwise are **mutually exclusive** (both are transfer_mode choices).

### 6 Valid Combinations

| transfer_mode | gene_granularity | theta shape | extract_parameters | extract_parameters_uniform | Forward | Backward | Optimizer | Tested e2e |
|---|---|---|---|---|---|---|---|---|
| uniform | uniform | `[3]` | line 84 | line 155 | all pibar_modes | yes | yes | **yes** |
| uniform | genewise | `[G,3]` | line 30 | line 134 | all pibar_modes | yes | yes | **yes** |
| specieswise | uniform | `[S,3]` | line 47 | line 145 | all pibar_modes | yes | yes | **yes** |
| specieswise | genewise | `[G,S,3]` | line 6 | line 123 | all pibar_modes | yes | yes | **yes** |
| pairwise | uniform | `[2]` | line 74 | N/A | dense, topk | untested | untested | **no** |
| pairwise | genewise | `[G,2]` | **NotImplementedError** (line 34) | N/A | N/A | N/A | N/A | N/A |

### Dead Code in extract_parameters

`specieswise + pairwise` branches exist at lines 9-17 and 50-58 but are invalid combinations. Should be replaced with guards.

---

## 3. Pibar Modes

| pibar_mode | Pibar cost | Ebar cost | Accuracy | Transfer modes | Tested |
|---|---|---|---|---|---|
| `dense` | O(W×S²) cuBLAS | O(S²) einsum | Exact | all | **yes** |
| `uniform_approx` | O(W×S) row_sum-self | O(S) logsumexp | ~1e-6 rel | uniform, specieswise | **yes** |
| `uniform` | O(W×S + nnz) sparse | O(S²) sparse | Exact | uniform, specieswise | partial (manual) |
| `topk` | O(W×k×S) gather+bmm or k-loop | dense fallback | ~0.01 abs per family | all (designed for pairwise at large S) | **yes** (forward + L-BFGS-B) |

**Notes:**
- `uniform_approx` and `uniform` don't apply to pairwise (transfer matrix is non-uniform)
- `topk` is the intended large-S path for pairwise; forward and L-BFGS-B convergence tested
- `uniform` mode verified manually (matches dense at fp64) but has no pytest coverage

---

## 4. Performance

### Benchmarks (S=1999, test_trees_1000)

| Path | 10 families | Per-family (after warmup) |
|---|---|---|
| Fixed-point (FP) | 48.0s | 4.8s |
| Wave v1 (gather/scatter) | — | 73ms |
| Wave (contiguous) | 2.6s | 200ms (18.6x vs FP) |
| Wave + uniform_approx | — | ~50ms (4x vs dense cuBLAS) |

### Large-S (S=19999, 10K-leaf species tree)

- 1 family: 0.26s, 18 GB peak on 24 GB GPU
- Pi_wave_forward is 96% of time

### Key Optimizations Applied

1. C++ phased wave scheduler (3-phase: leaf/internal/root)
2. Per-wave convergence (avg 5.4 iters vs 8 fixed)
3. Fused Triton DTS kernel (gather + 5 terms + logsumexp)
4. TF32 matmul for Pibar (~1.4x speedup)
5. Pi init with `finfo.min` (converges in 16 iters/wave)
6. Wave zero-copy layout (3.5-3.9x vs v1)
7. Uniform Pibar O(W×S) formula (4x vs cuBLAS, no [S,S] allocation)

---

## 5. Backward Pass & Optimization

### What Works

- **Pi backward**: Neumann-series implicit gradient, optional clade pruning (50-80% savings)
- **E adjoint**: CG solve with GMRES fallback
- **Full chain**: `implicit_grad_loglik_vjp_wave` → dNLL/dtheta
- **FD validation**: pD, pS, mt gradients match FD within 0.1%
- **Optimizers**: Adam (tested, 5-step NLL decrease), L-BFGS-B with FD (converges to AleRax reference)

### What's Untested

- L-BFGS-B with analytical gradient (path exists, never run e2e)
- Multi-family optimizer convergence (only 1 family tested)
- Large-S backward (forward works at S=20K, backward not validated)
- Pairwise gradient
- topk gradient
- E adjoint stability near spectral radius → 1

---

## 6. Test Coverage

### Overview

| Test File | Tests | Modes Covered | Status |
|---|---|---|---|
| `tests/unit/test_wave_vs_fp.py` | 11 | uniform/dense, small+large S | all pass |
| `tests/unit/test_wave_v2.py` | 7 | uniform/dense, batching | all pass |
| `tests/unit/test_cross_family_wave.py` | 13 | uniform/dense, collation, AleRax comparison | all pass |
| `tests/unit/test_genewise_wave.py` | 6 | genewise × (uniform_approx, dense) | all pass |
| `tests/unit/test_seg_logsumexp.py` | 4 | Triton kernel fp32/fp64, forward+backward | all pass |
| `tests/gradients/test_wave_gradient.py` | 30+ | all pibar modes × (global, specieswise, genewise) gradients | all pass |
| `tests/integration/test_e2e_alerax.py` | 1 | Full pipeline vs AleRax (simulated trees) | pass (slow) |
| `tests/cli/test_reconcile.py` | 14 | CLI arg parsing, devices, parameter ranges | pass |

---

### 6.1 `tests/unit/test_wave_vs_fp.py` — Wave vs Fixed-Point Correctness

**What it tests**: `Pi_wave_forward` (wave-ordered layout) produces the same per-family log-likelihoods as `Pi_fixed_point` (global Jacobi iteration). This is the fundamental correctness test for the wave solver.

**Setup**: Preprocesses real trees via the C++ extension (`ext.preprocess`), runs `extract_parameters` and `E_fixed_point` to get shared E matrices, then computes Pi via both solvers independently.

**Parameters**: D=0.05, L=0.05, T=0.05. Convergence tolerance 1e-3. LogL tolerance 5e-2 (the wave clade permutation changes FP operation ordering, giving ~0.03 absolute diffs on logL values of ~2000, i.e. ~1.5e-5 relative).

| Test | Dataset | S | Description | Assertion |
|---|---|---|---|---|
| `test_wave_matches_fp_small_s[0..4]` | test_trees_100 | 199 | 5 families, fused Triton small-S kernel path (S≤256) | \|logL_FP − logL_wave\| < 5e-2 |
| `test_wave_matches_fp_large_s[0..4]` | test_trees_1000 | 1999 | 5 families, cuBLAS Pibar large-S path (S>256) | \|logL_FP − logL_wave\| < 5e-2 |
| `test_wave_faster_than_fp_large_s` | test_trees_1000 | 1999 | 1 family, wall-clock comparison | wave/FP ratio < 0.5 (i.e. wave >2x faster) |

---

### 6.2 `tests/unit/test_wave_v2.py` — Wave Batching + Model API

**What it tests**: Cross-family wave batching via `collate_gene_families` + `collate_wave` + `build_wave_layout` produces the same results as per-family FP, and the high-level `GeneDataset` API agrees with the low-level functions.

**Setup**: Same as test_wave_vs_fp (real trees, shared E). For batched tests, multiple families are collated into cross-family waves.

| Test | Dataset | S | Families | Description | Assertion |
|---|---|---|---|---|---|
| `test_wave_matches_fp_small_s[2,5,10]` | test_trees_100 | 199 | 2/5/10 | Batched wave vs per-family FP | \|logL_FP − logL_wave\| < 5e-2 per family |
| `test_wave_matches_fp_large_s[2,5,10]` | test_trees_1000 | 1999 | 2/5/10 | Batched wave vs per-family FP | \|logL_FP − logL_wave\| < 5e-2 per family |
| `test_model_api_wave_matches_fp` | test_trees_1000 | 1999 | 10 | `GeneDataset.compute_likelihood_batch()` vs `GeneDataset.compute_likelihood(i)` per-family | \|wave − seq\| < max(5e-2, \|seq\| × 5e-5) |

---

### 6.3 `tests/unit/test_cross_family_wave.py` — Cross-Family Batching + AleRax Reference

**What it tests**: Two categories: (A) batched wave matches individual wave and FP at various scales, and (B) wave forward matches AleRax reference log-likelihoods on 4 curated datasets.

#### (A) Cross-Family Batching

**Setup**: Families preprocessed individually then collated via `collate_gene_families` + `collate_wave`. Cross-family phases computed as per-wave max across families.

| Test | Dataset | S | Families | Description | Assertion |
|---|---|---|---|---|---|
| `test_batched_wave_matches_individual_small_s[2,5,10]` | test_trees_100 | 199 | 2/5/10 | Batched vs per-family wave (small S) | \|individual − batched\| < 5e-2 |
| `test_batched_wave_matches_individual_large_s[2,5,10]` | test_trees_1000 | 1999 | 2/5/10 | Batched vs per-family wave (large S) | \|individual − batched\| < 5e-2 |
| `test_batched_wave_100_families_large_s` | test_trees_1000 | 1999 | 100 | 100 families in chunks of 20, all finite, spot-check 5 against individual wave | all finite + \|diff\| < 5e-2 |
| `test_batched_wave_timing_large_s` | test_trees_1000 | 1999 | 20 | Benchmark: batched vs sequential wave. Reports ms/family | correctness only (no speed assertion) |
| `test_batched_wave_vs_sequential_fp_large_s` | test_trees_1000 | 1999 | 10 | Batched wave vs per-family FP | \|FP − wave\| < 5e-2 |
| `test_batched_wave_vs_fp_100_families_large_s` | test_trees_1000 | 1999 | 100 | 100 families wave, spot-check 10 against FP. Reports mean/max diff | all finite + \|diff\| < 5e-2 |
| `test_batched_wave_vs_fp_small_s_20_families` | test_trees_20 | ~39 | 20 | Small trees: batched wave vs per-family FP | \|FP − wave\| < 5e-2 |
| `test_batched_wave_vs_fp_100_families_small_s` | test_trees_100 | 199 | 100 | 100 families small S: wave vs FP. Reports mean/max diff | all finite + \|diff\| < 5e-2 |
| `test_model_api_wave_vs_sequential` | test_trees_1000 | 1999 | 10 | `GeneDataset` high-level API: `compute_likelihood_batch` vs `compute_likelihood(i)` | \|wave − seq\| < 5e-2 |

#### (B) AleRax Reference Comparison

**What it tests**: Wave forward at AleRax's optimized DTL parameters produces the same per-family likelihood as AleRax 1.3.0. Corrects for the fact that AleRax omits the uniform origination prior (1/S) from its `per_fam_likelihoods.txt`, while our formula includes −log₂(S). The correction is: `wave_nats_no_orig = wave_nats + ln(S)`.

**Setup**: E solved to tolerance 1e-12, Pi to tolerance 1e-6, all in float32.

| Test | Dataset | S | D | L | T | AleRax logL (nats) | Assertion |
|---|---|---|---|---|---|---|---|
| `test_wave_matches_alerax[test_trees_1]` | test_trees_1 | 3 | 1e-10 | 1e-10 | 1e-10 | −2.56495 | \|diff\| < 0.05 |
| `test_wave_matches_alerax[test_trees_2]` | test_trees_2 | 3 | 1e-10 | 1e-10 | 0.0517 | −8.72486 | \|diff\| < 0.05 |
| `test_wave_matches_alerax[test_trees_3]` | test_trees_3 | 3 | 0.0556 | 1e-10 | 1e-10 | −6.75086 | \|diff\| < 0.05 |
| `test_wave_matches_alerax[test_mixed_200]` | test_mixed_200 | ~400 | 0.161 | 1e-10 | 0.156 | −6215.73 | \|diff\| < 0.05 |

---

### 6.4 `tests/unit/test_genewise_wave.py` — Genewise Mode

**What it tests**: Per-gene theta (different D/L/T rates per gene family) works through the wave path and matches the dense FP fallback.

**Setup**: `GeneDataset` with `genewise=True`. Each family gets a random theta (mean around log₂(0.03), i.e. `randn * 0.5 - 5.0`). The genewise path runs per-family wave loops (not cross-family batched) because each family has its own E solution.

| Test | Dataset | S | Families | Description | Assertion |
|---|---|---|---|---|---|
| `test_extract_params_uniform_genewise_shapes` | synthetic | — | — | Verifies tensor shapes for `extract_parameters_uniform` with genewise=True: pS/pD/pL [G], mt [G,S] for non-specieswise; pS/pD/pL [G,S], mt [G,S] for specieswise | shape checks |
| `test_extract_params_uniform_genewise_consistency` | synthetic | 50 | 4 | Per-family extraction == stacked genewise extraction | `allclose(atol=1e-6)` |
| `test_genewise_wave_vs_fp_uniform_approx` | test_trees_100 | 199 | 5 | Genewise wave (`uniform_approx`) vs genewise FP (`dense`) | \|wave − fp\| < 5e-2 per family |
| `test_genewise_wave_vs_fp_specieswise_uniform_approx` | test_trees_100 | 199 | 3 | Genewise + specieswise wave (`uniform_approx`) vs FP (`dense`). theta shape [S,3] per family | \|wave − fp\| < max(5e-2, \|fp\| × 5e-4) |
| `test_genewise_wave_batch_vs_per_family` | test_trees_100 | 199 | 4 | Full genewise batch == sum of single-family genewise batches | \|batch − seq\| < 1e-2 |
| `test_genewise_wave_large_s` | test_trees_1000 | 1999 | 3 | Genewise at large S: wave (`uniform_approx`) vs FP (`dense`) | \|wave − fp\| < max(5e-2, \|fp\| × 5e-4) |

---

### 6.5 `tests/unit/test_seg_logsumexp.py` — Triton Kernel

**What it tests**: The custom Triton `seg_logsumexp` kernel (segmented logsumexp over variable-length segments) matches a pure-PyTorch reference in both the forward pass and the backward pass.

**Setup**: Random segments with variable lengths (some empty), random values including ~15-20% `-inf` entries to test edge cases.

| Test | Dtype | Description | Assertion |
|---|---|---|---|
| `test_seg_logsumexp_matches_reference[float32]` | float32 | Forward: G=41 segments, S=13 columns | `allclose(rtol=1e-6, atol=1e-6)` |
| `test_seg_logsumexp_matches_reference[float64]` | float64 | Forward: same structure | `allclose(rtol=1e-12, atol=1e-12)` |
| `test_seg_logsumexp_gradients[float32]` | float32 | Backward: autograd gradient vs closed-form `softmax(seg) * grad_out` | `allclose(rtol=1e-6, atol=1e-6)` |
| `test_seg_logsumexp_gradients[float64]` | float64 | Backward: same | `allclose(rtol=1e-12, atol=1e-12)` |

---

### 6.6 `tests/gradients/test_wave_gradient.py` — Backward Pass & Gradient Validation

**What it tests**: The full implicit gradient pipeline: Pi backward (Neumann series), E adjoint (CG/GMRES), and full-chain `dL/dtheta` — validated against central finite differences in float64.

This is the most comprehensive test file. It covers 13 test classes:

#### Neumann Series (synthetic)

| Test | Description | Assertion |
|---|---|---|
| `TestNeumannSynthetic::test_neumann_matches_exact_inverse` | 20×20 matrix with ρ=0.04, 3 Neumann terms vs `linalg.solve` | rel error < 1e-4 |
| `TestNeumannSynthetic::test_neumann_4_terms_fp64` | 50×50 matrix, 4 terms | rel error < 3e-6 |

#### Wave Backward Smoke Tests (test_trees_20, S~39, float32)

| Test | Description | Assertion |
|---|---|---|
| `TestWaveBackward::test_backward_runs` | `Pi_wave_backward` completes, returns v_Pi of correct shape | no error, correct shape |
| `TestWaveBackward::test_backward_gradients_finite` | All accumulators (grad_E, grad_Ebar, grad_log_pD, grad_log_pS, grad_max_transfer_mat) are finite | `isfinite().all()` for each |
| `TestWaveBackward::test_root_rhs_nonzero` | Root clade gradient is nonzero | max abs > 0 |

#### Gradient Descent (test_trees_20, float32)

| Test | Description | Assertion |
|---|---|---|
| `TestGradientDescent::test_gradient_step_decreases_nll` | Verifies at least one of grad_log_pD, grad_log_pS, grad_mt is nonzero. Prints FD directional derivatives for each theta component at eps=1e-3 | at least one gradient nonzero |

#### Pi-Level Finite Difference Validation (test_trees_1000, S=1999, float64)

These tests validate `Pi_wave_backward` gradients (holding E fixed) against central FD at eps=1e-4.

| Test | Parameter | Description | Assertion |
|---|---|---|---|
| `TestWaveVsFiniteDifference::test_log_pD_gradient` | log_pD (scalar) | Perturb log_pD by ±eps, re-run Pi forward, central FD | rel error < 1e-4 (0.01%) |
| `TestWaveVsFiniteDifference::test_log_pS_gradient` | log_pS (scalar) | Same for log_pS | rel error < 1e-4 |
| `TestWaveVsFiniteDifference::test_mt_gradient` | max_transfer_mat [S] | Random direction, directional derivative FD vs dot product with grad_mt | rel error < 1e-4 |

#### Pruning

| Test | Description | Assertion |
|---|---|---|
| `TestPruning::test_pruning_runs_and_is_finite` | Pruned backward (threshold=−50) vs full backward. Checks finite, positive correlation | `isfinite().all()` + positive dot product |

#### Gradient Bounds

| Test | Description | Assertion |
|---|---|---|
| `TestGradientBoundsVectorized::test_vectorized_produces_valid_output` | `compute_gradient_bounds` with wave_metas: root bound=0, all bounds finite or −inf | shape + value checks |

#### Differentiable Self-Loop

| Test | Description | Assertion |
|---|---|---|
| `TestSelfLoopDifferentiable::test_self_loop_matches_forward` | `_self_loop_differentiable` produces correct shape, finite output on leaf wave | shape + finite |
| `TestSelfLoopDifferentiable::test_self_loop_gradients_flow` | Autograd gradients flow through `_self_loop_differentiable` (3 synthetic clades) | `Pi_W.grad` exists and finite |

#### Full-Chain FD: dL/dtheta through E and Pi (test_trees_1000, float64)

These are the most important gradient tests. They perturb theta, re-solve both E AND Pi from scratch, and compare to the analytical `implicit_grad_loglik_vjp_wave` which uses Pi backward + E adjoint CG.

| Test Class | pibar_mode | Dataset | S | theta shape | eps | Tolerance | What's validated |
|---|---|---|---|---|---|---|---|
| `TestFullChainFD` | uniform_approx | test_trees_1000 | 1999 | [3] | 1e-4 | rel < 1e-4 per component | dL/d(theta[0]), dL/d(theta[1]), dL/d(theta[2]) |
| `TestFullChainFDDense` | dense | test_trees_1000 | 1999 | [3] | 1e-4 | rel < 1e-3 | Same 3 components, via dense matmul Pibar path |
| `TestUniformExactFullChainFD` | uniform (exact) | test_trees_20 | ~39 | [3] | 1e-4 | rel < 1e-3 | Same, with ancestor-corrected sparse Pibar |

#### Specieswise Gradient FD (float64)

Specieswise mode: theta [S,3], per-species D/L/T rates. FD perturbs each species independently.

| Test Class | pibar_mode | Dataset | S | eps | Tolerance | What's validated |
|---|---|---|---|---|---|---|
| `TestSpecieswiseUniformFD` | uniform_approx | test_trees_20 | ~39 | 1e-4 | rel < 1e-3 | Random-direction FD on theta [S,3] |
| `TestSpecieswiseDenseFD` | dense | test_trees_20 | ~39 | 1e-4 | rel < 1e-3 | Same, dense Pibar path |
| `TestSpecieswiseUniformExactFD` | uniform (exact) | test_trees_20 | ~39 | 1e-4 | rel < 1e-3 | Same, ancestor-corrected sparse path |

#### Specieswise Forward Consistency (test_trees_20, float64)

| Test | Description | Assertion |
|---|---|---|
| `TestSpecieswiseForwardConsistency::test_E_matches` | Specieswise mode with identical rates per species == global mode: E matrices match | max abs diff < 1e-10 |
| `TestSpecieswiseForwardConsistency::test_Pi_matches` | Same for Pi | max abs diff < 1e-8 |
| `TestSpecieswiseForwardConsistency::test_logL_matches` | Same for logL | abs diff < 1e-6 |

#### Genewise Gradient FD (test_trees_1000, 3 families, float64)

| Test | Description | Assertion |
|---|---|---|
| `TestGenewiseGradient::test_genewise_gradient_matches_fd` | 3 families with different random theta [G,3]. Per-gene per-component FD (eps=1e-4) vs `implicit_grad_loglik_vjp_wave_genewise` analytical gradient | rel error < 1e-2 (1%) per component |

#### End-to-End Optimizer (test_trees_1000, 1 family, float64)

| Test | Description | Assertion |
|---|---|---|
| `TestEndToEnd::test_optimization_decreases_nll` | 5 steps of `optimize_theta_wave` (Adam, lr=0.1). Reports NLL, grad norm, step size per iteration | NLL_last < NLL_first |

#### Gradient Descent All Modes (test_trees_1000, 3 families, float64)

Parametrized over all 6 valid (genewise, specieswise, pibar_mode) combinations:

| genewise | specieswise | pibar_mode | theta shape | Method |
|---|---|---|---|---|
| False | False | uniform_approx | [3] | `optimize_theta_wave` (3 steps, lr=0.1) |
| False | False | dense | [3] | same |
| False | True | uniform_approx | [S,3] | same |
| False | True | dense | [S,3] | same |
| True | False | uniform_approx | [G,3] | Manual Adam loop with `implicit_grad_loglik_vjp_wave_genewise` |
| True | True | uniform_approx | [G,S,3] | same |

**Assertion**: NLL decreases after 3 Adam steps for each combination.

---

### 6.7 `tests/integration/test_e2e_alerax.py` — Full Pipeline vs AleRax

**What it tests**: End-to-end validation against the reference implementation AleRax 1.3.0. Simulates fresh trees, runs AleRax to infer DTL parameters, then runs gpurec at those parameters and compares per-family log-likelihoods.

**Pipeline**:
1. Simulate a 60-leaf species tree (birth=1, death=0) via `rustree.simulate_species_tree`
2. Simulate 100 gene trees (d=0.05, t=0.05, l=0.05) via `rustree` DTL simulator
3. Run AleRax (MPI, 4 procs) with `--rec-model UndatedDTL --model-parametrization GLOBAL`
4. Parse AleRax's inferred D/L/T and per-family likelihoods
5. Run gpurec (`Pi_fixed_point`, tolerance 1e-12, float64) with AleRax's parameters
6. Convert gpurec log₂ to nats, compare per-family

**Assertion**: All 100 families within 1e-3 nats absolute tolerance.

**Note**: Requires AleRax binary + MPI installed. Slow (~minutes). Runs as a standalone script, not a pytest test.

---

### 6.8 `tests/cli/test_reconcile.py` — CLI

**What it tests**: The `gpurec` CLI entrypoint (`src/cli/reconcile.py`) handles arguments, errors, and produces valid output.

| Test | Description | Assertion |
|---|---|---|
| `test_cli_script_exists` | Script exists and has Python shebang | file exists |
| `test_cli_help` | `--help` flag works | exit 0, output contains `--species`, `--gene`, `--delta`, `--tau`, `--lambda` |
| `test_cli_missing_required_args` | No args → error | exit ≠ 0, stderr contains "required" |
| `test_cli_with_test_trees_1` | Run on test_trees_1 (D=L=T=0.1, 10 iters, CPU) | exit 0, output contains "Log-likelihood", value in (−1000, 0) |
| `test_cli_with_test_trees_2` | Run on test_trees_2 | exit 0, output contains "Log-likelihood" |
| `test_cli_with_debug_flag` | `--debug` produces verbose output | exit 0, output > 100 chars |
| `test_cli_parameter_variations` | 3 parameter combos: (1e-10,1e-10,1e-10), (0.01,0.01,0.01), (0.5,0.3,0.4) | all exit 0 |
| `test_cli_invalid_file_path` | Nonexistent files → error | exit ≠ 0, "Error" in output |
| `test_cli_with_cuda` | Same as test_trees_1 but `--device cuda` (skip if no GPU) | exit 0 |
| `test_cli_with_test_trees_200` | Large tree (200 leaves), slow marker | exit 0 |
| `test_cli_with_mixed_200` | Mixed-event tree, slow marker | exit 0 |

---

### Coverage Gaps

| What | Status |
|------|--------|
| pairwise forward (any mode) | **zero tests** — manually verified to work with dense |
| pairwise gradient | **zero tests** |
| topk forward | **fixed** (2026-03-20) — replaced broken sparse-output (scatter k positions, 92% −inf) with full-output compressed-input: gather+bmm for small S, k-loop for large S. k=16 logL within 0.01 of dense. L-BFGS-B converges. See `tests/topk_lbfgs_convergence.py` |
| topk gradient | **untested** — backward falls back to dense, not FD-validated |
| multi-family optimizer convergence | **missing** — gradient descent tested (3 steps, NLL decreases) but no convergence test |
| L-BFGS-B with analytical gradient | **missing** — only FD path tested in e2e AleRax comparison |

---

## 7. Git State

### Branch: `batched` (39 commits ahead of main)

**Recent commits:**
```
8642cc5 Remove dead modules, stale tests, and update .gitignore
001ec7c Add sparse_corrected Pibar mode, optimizer functions, and full-chain gradient tests
d40584c Fix wave backward parameter gradients and add FD validation
3be1aae Fix cross-clade backward to use manual gradient computation
a3011b7 Add wave-decomposed implicit gradient backward pass (WIP)
```

### Uncommitted Changes

**Today's session (not yet committed):**
- Deleted `src/core/kernels/seg_log_matmul.py` (genewise segmented matmul — dead code)
- Removed `compute_DTS_independent`, `compute_DTS_L_independent` from `terms.py`
- Simplified `Pi_step` and `Pi_fixed_point` signatures (removed genewise/specieswise/pairwise/batch_info params)
- Simplified `compute_likelihood_batch` routing (removed dead `Pi_fixed_point` + `batch_info` branch)
- Fixed pairwise theta initialization in `GeneDataset.__init__` (was `[3]`, now correctly `[2]`)
- Fixed `root_clade_id` used-before-assignment in `compute_likelihood` wave path
- Added `specieswise + pairwise` mutual exclusivity guard
- Fixed genewise wave path to pass actual `transfer_mat` for dense/topk modes

**Other unstaged modifications (28 files):**
- Core: likelihood.py, model.py, theta_optimizer.py, terms.py, kernels, batching
- Rust: 11 rustree modules (sampling, metrics, surgery, etc.)
- Tests: test_wave_gradient.py, test_wave_v2.py
- Docs: status_march_2026.md

**Untracked (98 files):**
- `rustree/` subproject (Cargo.toml, 64 source/test files)
- `tests/integration/`, `tests/profiling/` benchmarks
- `docs/` (3 new docs: clade_scheduling, optim_opportunities, optimization_plan)

### Main branch

Appears stale — last commit "working program git add ." from pre-September 2025.

---

## 8. Auxiliary Components

### rustree (Rust)
Tree metrics + simulation library. Python/R bindings via pyo3/extendr.
- Newick parsing (pest grammar)
- Birth-death + DTL simulation
- Robinson-Foulds distance, pairwise metrics
- AleRax wrapper (external process)
- Tree surgery (prune, graft, induced subtrees)

### logmatmul (Python/Triton)
Log2-space matrix multiplication without SFU instructions.
- Dense: two-pass (bf16/tf32) or single-pass (ieee fp32)
- Streaming top-k selection
- Compressed matmul (k×k sub-blocks)
- ~83-84% of cuBLAS throughput on RTX 4090
- Autograd wrapper (`LogspaceMatmulFn`)

### C++ Preprocessor
JIT-compiled via PyTorch's `torch.utils.cpp_extension`. Computes CCP (Clades, Clades-Pairs) structure from Newick trees. Three source files under `src/core/cpp/`.

### CLI
`gpurec` entrypoint via pyproject.toml. Basic: `--species`, `--gene`, `--delta/tau/lambda`, `--device`, `--dtype`. Hardcodes `pairwise=False`.

---

## 9. Known Issues

| Issue | Severity | Location |
|---|---|---|
| `genewise + pairwise` raises NotImplementedError | Design decision (not wanted) | extract_parameters.py:34 |
| Dead `specieswise + pairwise` branches in extract_parameters | Minor dead code | extract_parameters.py:9-17, 50-58 |
| topk gradient not FD-validated | Low-medium risk | likelihood.py backward falls back to dense |
| 10 legacy test files fail to import (41 tests uncollectable) | Noise | Old `ThetaOptimizationProblem` tests |
| Gradient residual ~0.6-0.7 at FD optimum | FD artifact, not a real bug | Needs analytical gradient validation |
| `uniform` pibar mode has no pytest coverage | Fragile | Manually verified only |
| CLI hardcodes `pairwise=False` | Missing feature | cli/reconcile.py:63 |

---

## 10. Recommended Priorities

1. **Commit today's cleanup** — seg_log_matmul deletion, pairwise fix, signature simplifications
2. **Add pairwise e2e test** — forward pass works, needs pytest coverage (dense + topk forward now correct)
3. **Add uniform pibar pytest** — manually verified but fragile without CI
4. **Run L-BFGS-B with analytical gradient** — the key validation: does it converge to |g|→0?
5. **Multi-family optimizer test** — 5-10 families, verify convergence
6. **Clean up dead specieswise+pairwise branches** in extract_parameters
7. **Clean up legacy test files** — mark xfail/skip or delete

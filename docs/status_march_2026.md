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
| `topk` | O(W×k²) compressed | dense fallback | ~0.025% | all (designed for pairwise at large S) | **no** |

**Notes:**
- `uniform_approx` and `uniform` don't apply to pairwise (transfer matrix is non-uniform)
- `topk` is the intended large-S path for pairwise but has zero test coverage
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

### Test Suite Summary

| Test File | Tests | Modes Covered | Status |
|---|---|---|---|
| `tests/unit/test_wave_vs_fp.py` | 11 | uniform/dense, small+large S | all pass |
| `tests/unit/test_wave_v2.py` | 7 | uniform/dense, batching | all pass |
| `tests/unit/test_cross_family_wave.py` | 9 | uniform/dense, collation | all pass (1 flaky NVTX) |
| `tests/unit/test_genewise_wave.py` | 6 | genewise × (uniform_approx, dense) | all pass |
| `tests/unit/test_seg_logsumexp.py` | 2 | Triton kernel fp32/fp64 | all pass |
| `tests/gradients/test_wave_gradient.py` | 15+ | uniform_approx, dense, uniform gradients | all pass |
| `tests/integration/test_e2e_alerax.py` | 1 | Full pipeline vs AleRax | pass (slow) |
| `tests/cli/test_reconcile.py` | 14+ | CLI arg parsing, devices | pass |

### Coverage Gaps

| What | Status |
|------|--------|
| pairwise forward (any mode) | **zero tests** — manually verified to work with dense |
| pairwise gradient | **zero tests** |
| topk forward | **zero tests** — implemented but unvalidated |
| topk gradient | **zero tests** |
| uniform (sparse) forward pytest | **missing** — manually verified, no pytest |
| uniform (sparse) gradient | **missing** — only uniform_approx tested |
| multi-family optimizer | **missing** — only 1 family in TestEndToEnd |
| L-BFGS-B analytical gradient | **missing** — only FD path tested |

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
| topk mode fully implemented but zero test coverage | Medium risk | likelihood.py, theta_optimizer.py |
| 10 legacy test files fail to import (41 tests uncollectable) | Noise | Old `ThetaOptimizationProblem` tests |
| Gradient residual ~0.6-0.7 at FD optimum | FD artifact, not a real bug | Needs analytical gradient validation |
| `uniform` pibar mode has no pytest coverage | Fragile | Manually verified only |
| CLI hardcodes `pairwise=False` | Missing feature | cli/reconcile.py:63 |

---

## 10. Recommended Priorities

1. **Commit today's cleanup** — seg_log_matmul deletion, pairwise fix, signature simplifications
2. **Add pairwise e2e test** — forward pass works, needs pytest coverage (dense + topk)
3. **Add uniform pibar pytest** — manually verified but fragile without CI
4. **Run L-BFGS-B with analytical gradient** — the key validation: does it converge to |g|→0?
5. **Multi-family optimizer test** — 5-10 families, verify convergence
6. **Clean up dead specieswise+pairwise branches** in extract_parameters
7. **Clean up legacy test files** — mark xfail/skip or delete

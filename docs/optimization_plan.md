# Plan: gpurec Future — Fast Gradient Descent for ALE Reconciliation

## Context

**Goal**: Optimize parameters θ = (δ, τ, λ per species branch) by gradient descent, as fast as possible, for the ALE (Amalgamated Likelihood Estimation) algorithm. The bottleneck is the Π̄ = log(exp(Π) · M) matmul (O(C·S²) per iteration). The `logmatmul` repo already implements this exact operation on GPU with 103 TFLOP/s, plus top-k sparsification and compressed backward kernels. The wave scheduling infrastructure is already built in C++. This plan connects all the pieces.

**Current state**: Fixed-point iteration computes Π for ALL clades simultaneously, 3-4 iterations. Each iteration does a dense C×S×S matmul. For 1000-species, 1000-family datasets: C_total ≈ 4M clades, S = 2000, yielding ~128 TFLOP of matmul per optimization step.

---

## Phase A: logmatmul Integration (drop-in Π̄ speedup)

### A0. Standardize codebase to log2

Convert all log-space computations from ln to log2. This eliminates conversion overhead at every logmatmul boundary and every `exp`/`log` call.

**Files to modify**:
- `src/core/likelihood.py`: `torch.exp` → `torch.exp2`, `torch.log` → `torch.log2`, `math.log(...)` → `math.log2(...)`, `torch.logsumexp` needs a log2-space wrapper
- `src/core/terms.py`: All DTS/DTS_L term computations use log-space additions (unchanged) but the `log_2` constant and `logsumexp` calls need updating
- `src/core/extract_parameters.py`: `log_softmax` and parameter extraction must output log2-space
- `src/core/kernels/scatter_lse.py`: `seg_logsumexp` kernel — change `tl.exp` to `tl.exp2`, `tl.log` to `tl.log2`
- `src/core/kernels/seg_log_matmul.py`: Same treatment
- `src/core/model.py`: Initialization constants (`-log(10)` → `-log2(10)`)

**Key invariant**: All log-space tensors (Π, Π̄, E, Ē, DTS terms, log_pS/D/L) are in log2-space. Additions in log-space are unchanged. Only `exp`↔`log` pairs and `logsumexp` need updating.

**Helper**: Write `logsumexp2(x, dim)` = `log2(sum(2^x, dim))` using the standard max-subtract trick with `exp2`/`log2`.

### A1. Package logmatmul as importable module

- Add `logmatmul/pyproject.toml` with minimal `[project]` and `[build-system]`
- Change `logmatmul/src/` imports to `logmatmul.dense`, `logmatmul.sparse`, `logmatmul.compressed`
- Install as `pip install -e logmatmul/`

### A2. Write `torch.autograd.Function` for `logspace_matmul`

The dense kernel has no backward pass. Write `LogspaceMatmulFn`:
- **Forward**: `out = logspace_matmul(M, Y, precision="bf16")` — returns log2-space result
- **Backward w.r.t. Y**: `dL/dY[k,n] = Σ_b dL/dout[b,n] · M[b,k] · 2^(Y[k,n]-c[n]) / (Σ_j M[b,j]·2^(Y[j,n]-c[n]))` — this is another matmul of similar structure, implementable via `logspace_matmul(M.T, ...)`
- **Backward w.r.t. M**: `dL/dM[b,k] = Σ_n dL/dout[b,n] · 2^(Y[k,n]-out[b,n]) · ln(2)` — a matmul in mixed space

**File**: New `logmatmul/src/autograd.py`

### A3. Replace Π̄ computation in `Pi_step`

**File**: `src/core/likelihood.py`, lines 180-194 (non-genewise path)

Current:
```python
Pi_max = torch.max(Pi, dim=1, keepdim=True).values
Pi_linear = torch.exp(Pi - Pi_max)
Pibar_linear = Pi_linear.mm(transfer_mat_T)
Pibar = torch.log(Pibar_linear) + Pi_max + max_transfer_mat
```

Replace with:
```python
Pi_T = Pi.T.contiguous()                                        # [S, C] log2-space
Pibar_T = LogspaceMatmulFn.apply(transfer_mat, Pi_T)            # [S, C] log2-space
Pibar = Pibar_T.T + max_transfer_mat.squeeze(-1)                # [C, S] log2-space
```

The genewise path (`segmented_log_matmul`) stays unchanged — it handles per-gene transfer matrices.

**Note**: The entire codebase will be standardized to log2 (see Phase A0 below). This eliminates conversion overhead at every logmatmul call boundary and at every `torch.exp`/`torch.log` throughout likelihood.py and terms.py.

### A4. Validate correctness

- Run existing test suite: `pytest tests/`
- Compare `compute_log_likelihood` output against reference using `precision="ieee"` (fp32) for correctness (tolerance 1e-5). bf16/tf32 modes trade accuracy for speed and will NOT match to 1e-5.
- Profile with `precision="bf16"` to confirm matmul speedup; verify likelihood is still reasonable (tolerance ~1e-2)

**Expected speedup**: 2-3× on the matmul portion (103 vs ~50 TFLOP/s effective), overall ~1.5-2× on full Pi_step.

---

## Phase B: Wave-Based Forward Pass (eliminate global fixed-point)

### Key insight: no cross-clade dependency in Π̄

Π̄[γ, e] = Σ_f M[e,f] · exp(Π[γ, f]). Each row of Π̄ depends only on the same row of Π (the matmul mixes species, not clades). The cross-clade dependency is only through the DTS terms (parent depends on children's Π). So within a wave, the self-dependency (DTS_L terms: DL, SL, TL) only involves the wave's own Π rows.

### B1. New function `Pi_wave_forward`

**File**: `src/core/likelihood.py` (new function, replaces `Pi_fixed_point` when scheduling is available)

```
Pi_wave_forward(waves, ccp_helpers, species_helpers, ...):
  Pi = init_leaf_clades()      # wave 0: known from clade-species map

  for k in range(1, len(waves)):
    wave_ids = waves[k]         # clade indices in this wave

    # Cross-clade terms (DTS): children are in earlier waves, already fixed
    # Gather Pi[left], Pi[right], Pibar[left], Pibar[right] — all fixed
    DTS_term = compute_DTS_for_wave(...)  # computed ONCE, no iteration

    # Self-loop terms (DTS_L): require Pibar of THIS wave's clades
    for local_iter in range(3-4):
      Pi_W = Pi[wave_ids]                         # [|W|, S]
      Pibar_W = logspace_matmul(M, Pi_W.T).T      # [|W|, S] — small matmul
      DTS_L_term = compute_DTS_L_for_wave(Pi_W, Pibar_W, E, ...)
      Pi[wave_ids] = logaddexp(DTS_term, DTS_L_term)  # update only wave rows
```

**Matmul cost per wave**: |W|×S×S (typically 256×2000×2000 = 1 GFLOP) × 3-4 iterations = 3-4 GFLOP.
**Total**: ~43 waves × 4 GFLOP = 172 GFLOP per family (vs 16 TFLOP for 4 global iterations).
That's ~93× fewer FLOPs for the matmul. But small matmuls have poor GPU utilization — Phase E (cross-family batching) fixes this.

### B2. Adapt `compute_DTS` for per-wave subsets

**File**: `src/core/terms.py`

New `compute_DTS_wave(Pi, Pibar, wave_split_ids, split_leftrights, ...)`:
- Only processes splits whose parent is in the current wave
- Children Pi/Pibar are gathered from the full (partially-frozen) Pi tensor
- Returns DTS_term indexed over wave splits only

### B3. Cross-family wave batching (built-in from the start)

Go directly to cross-family batching — no intermediate single-family step. Reuse the C++ `compute_phased_cross_family_wave_stats` logic to assign global wave indices across families. The Python-side code batches all families' wave-k clades into a single matmul:
- Phase 1 (leaves): no matmul, just initialize
- Phase 2 (internal): batched logspace_matmul across families — all families' wave-k clades in one matmul
- Phase 3 (root Ω): scatter-logsumexp over many splits

The C++ scheduler already produces cross-family batches of size ≤ W=256 with packet-aware priority. Expose the full wave assignment (not just stats) from C++ to Python.

**File**: Extend `src/core/batching.py` with `collate_wave(families_data, wave_k)`.

### B4. Verification

- Assert `Pi_wave_forward` matches `Pi_fixed_point` within tolerance (1e-5)
- Wave-by-wave: check that each wave's output is consistent with the fixed-point result
- Benchmark: measure total time and compare with baseline

---

## Phase C: Top-k Sparsification (compressed Π̄)

### C1. After each wave, compress Π

```python
from logmatmul import streaming_topk

Pi_W = Pi[wave_ids]                              # [|W|, S]
topk_vals, topk_idx = streaming_topk(Pi_W.T, k)  # [k, |W|] each
# Store compressed representation for use by later waves
```

### C2. Compressed Π̄ for DTS_L self-loop

Within the local iteration of each wave, use compressed matmul:

```python
from logmatmul import logspace_matmul_compressed

Pibar_compressed = logspace_matmul_compressed(topk_idx, topk_vals, M)  # [k, |W|]
```

This reads k²×|W| entries from M (with k=32: 1024×256 = 262K reads, vs 2000²×256 = 1B for dense). ~4000× less memory traffic.

### C3. Compressed children for DTS terms

When computing DTS terms for wave k, the children (from earlier waves) are already compressed. Reconstruct sparse Pi[child] by scattering topk_vals into a -inf tensor of size S, then proceed with elementwise DTS computation. Or implement a sparse DTS kernel that works directly on compressed representations.

### C4. Fallback for non-sparse clades

Monitor `topk_vals[0] - topk_vals[k-1]` per clade. If the gap is small (probability mass is spread), fall back to dense computation for that clade. In practice, the thesis notes that "pour chaque clade, seul un petit nombre de branches de l'arbre d'espèces porte une vraisemblance significative" — most clades are very sparse.

### C5. Backward through compressed path

`logspace_matmul_compressed_backward` already exists in `logmatmul/src/compressed.py`. Wire it into the autograd function from A2.

**Expected speedup**: 5-10× on the matmul portion over Phase B (dense waves).

---

## Phase D: Gradient Pruning (prune clades from backward pass)

### D1. Record clade scores during forward

After each wave converges: `s[γ] = max_e Π[γ,e]` for γ in wave. This is already computed as the stabilization max (or as `topk_vals[0]` if using top-k).

### D2. Gradient bound propagation (CPU, O(N_splits))

After full forward pass:
```python
grad_bound = [-inf] * C
grad_bound[root] = 0.0

for γ in reverse_topological_order:
    if grad_bound[γ] < PRUNE_THRESHOLD:
        continue  # prune subtree
    for (γ', γ'') in bipartitions(γ):
        grad_bound[γ'] = logsumexp(grad_bound[γ'], grad_bound[γ] + s[γ''])
        grad_bound[γ''] = logsumexp(grad_bound[γ''], grad_bound[γ] + s[γ'])
```

### D3. Backward wave filtering

When constructing backward waves, exclude clades where `grad_bound[γ] < threshold`. Pruned clades get zero gradient. No kernel changes needed.

### D4. Threshold annealing

| Phase | Threshold (log2) | Expected pruning |
|-------|-----------------|------------------|
| Early optimization | -20 | 50-80% of clades |
| Mid optimization | -40 | 20-40% |
| Near convergence | -∞ (disabled) | 0% |

**Expected speedup**: 2-5× on backward pass, especially early in optimization.

---

## Phase E: Reorganization

### E1. Remove duplicate scheduling code

The C++ code in `preprocess.cpp` has `compute_clade_waves` (line ~970), `compute_wave_stats`, `compute_packet_wave_stats`, `compute_phased_wave_stats`, `compute_phased_cross_family_wave_stats`, `compute_cross_family_wave_stats`. The Python `scheduling.py` also has `compute_clade_waves`.

**Decision**:
- Keep C++ `build_sched_data` and `compute_phased_cross_family_wave_stats` — these run during preprocessing, produce wave assignments consumed by the Python likelihood code
- Remove Python `scheduling.py` — it duplicates the C++ version without the packet-aware priority
- Remove C++ `compute_wave_stats`, `compute_packet_wave_stats` — superseded by phased variants
- Keep `bench_parse` for diagnostics

### E2. Kernel consolidation

- `src/core/kernels/seg_log_matmul.py`: Keep for genewise path (per-gene transfer matrices)
- `src/core/kernels/scatter_lse.py`: Keep (segmented logsumexp for DTS reduction)
- `logmatmul/src/dense.py`: Use for shared-parameter Π̄ (single transfer matrix)
- `logmatmul/src/compressed.py`: Use for sparse Π̄ after top-k

### E3. C++ vs Rust delineation

- **C++** (keep): CCP extraction (`amalgamate_clades_and_splits`, `build_ccp_arrays`, `build_ccp_light`, `build_sched_data`) — deeply integrated with PyTorch tensor creation via pybind11
- **Rust** (rustree): Tree simulation, comparison, sampling, visualization — separate tool, used via PyO3 bindings for data generation/analysis
- **Python**: Likelihood computation, gradient descent, wave-based scheduling logic, optimization loop

---

## Performance Estimates (1000 species, 1000 families)

| Stage | Matmul FLOPs/step | Effective TFLOP/s | Time estimate |
|-------|-------------------|-------------------|---------------|
| Baseline (4 global iters) | 128 TFLOP | ~50 | ~2.5s matmul |
| A: logmatmul dense | 128 TFLOP | ~103 | ~1.2s matmul |
| B: wave-based (no batching) | 1.4 TFLOP | ~20 (small matmuls) | ~0.07s matmul |
| B+E: wave-based + cross-family | 1.4 TFLOP | ~80 (batched) | ~0.02s matmul |
| C: top-k (k=32) | ~0.04 TFLOP | memory-bound | ~0.005s |
| D: gradient pruning | 50-80% reduction in backward | — | backward 2-5× faster |

The DTS term computation (O(N_splits × S)) becomes the bottleneck after matmul optimization. For 1000 families: N_splits_total ≈ 5M, S = 2000 → 10 GFLOP of gather/scatter/logsumexp work per step.

---

## Implementation Order

```
Phase A (foundation):          A0 → A1 → A2 → A3 → A4
                                              ↓
Phase B (waves, cross-family): B1 → B2 → B3 → B4
                                              ↓
Phase C (sparsification):      C1 → C2 → C3 → C4 → C5    [can start C1-C3 in parallel with B]
                                              ↓
Phase D (gradient pruning):    D1 → D2 → D3 → D4          [after B is tested]
                                              ↓
Phase E (cleanup):             E1, E2, E3                  [ongoing]
```

**Critical path**: A → B → D.
**Decisions made**: log2 throughout (not ln), cross-family directly (no single-family intermediate), Phase A first.

---

## Critical Files

| File | Modification | Phase |
|------|-------------|-------|
| `logmatmul/pyproject.toml` | New — package config | A1 |
| `logmatmul/src/autograd.py` | New — backward pass wrapper | A2 |
| `src/core/likelihood.py` | Replace Π̄ matmul; new `Pi_wave_forward` | A3, B1 |
| `src/core/terms.py` | New `compute_DTS_wave` for per-wave subsets | B2 |
| `src/core/batching.py` | New `collate_wave` for cross-family wave batching | B3 |
| `src/core/cpp/preprocess.cpp` | Remove deprecated scheduling functions | E1 |
| `src/core/scheduling.py` | Remove (superseded by C++ phased scheduler) | E1 |

## Verification

1. **Phase A**: `compute_log_likelihood` matches reference within 1e-5 using `precision="ieee"` (fp32). For bf16/tf32: verify likelihood is reasonable (~1e-2 tolerance) and optimization converges similarly.
2. **Phase B**: `Pi_wave_forward` matches `Pi_fixed_point` within 1e-5 (both in fp32/ieee mode); wave-by-wave checks
3. **Phase C**: Compare dense vs compressed likelihood; validate k vs accuracy on real data
4. **Phase D**: Verify gradient matches full backward within threshold-dependent tolerance
5. **End-to-end**: Run optimization on 100-species dataset, compare convergence curves with both ieee and bf16 modes

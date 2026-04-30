# gpurec Project Diagnosis — April 2, 2026

## Summary

The forward+backward+optimizer pipeline is fully built. Wave-based forward is 18.6x faster than fixed-point. Uniform Pibar eliminates O(S²) for non-pairwise modes. L-BFGS with batched backward across families is implemented. The main gap is **end-to-end optimizer convergence validation** — the optimizer exists but has never been run to convergence on a real dataset with analytical gradients.

---

## 1. Must-Validate (correctness unproven)

### 1.1 L-BFGS convergence with analytical gradient

`optimize_theta_genewise` provides the full path: batched backward, per-gene convergence masking, CG/GMRES E adjoint. But it has never been run to convergence on a real dataset. The existing `test_e2e_alerax.py` validates the forward pass against AleRax, and `TestEndToEnd::test_optimization_decreases_nll` shows NLL decreases for 5 Adam steps — but neither proves the analytical-gradient L-BFGS converges to the correct parameters.

**What's needed**: Run `optimize_theta_genewise` on a multi-family dataset (e.g. 100 families, S=1999) and verify:
- NLL converges (gradient norm → 0)
- Inferred θ matches AleRax's parameters within tolerance
- No numerical instability in E adjoint or Neumann series at convergence

**Why it matters**: This is the whole point of the project. Everything else is incremental.

### 1.2 Batched backward at large S

Forward works at S=20K. Backward only validated at small S (~39, ~199). Need to confirm no numerical issues at S=1999+.

**What's needed**: Run `Pi_wave_backward` on a single family at S=1999, compare against per-family backward or FD. Check that gradient magnitudes are reasonable and finite.

---

## 2. Test Coverage Holes

| Gap | Risk | Current state |
|-----|------|---------------|
| Pairwise forward | Medium | Zero pytest tests. Manually verified with dense. |
| `uniform` pibar mode | Medium | 1 FD test at S=39 only. No large-S coverage. Manually verified to match dense at fp64. |
| topk gradient | Low-medium | Backward falls back to dense (correct but untested compressed path). Not FD-validated. |
| Batched backward | Low | `TestBatchedBackward` exists but only at small S. |

---

## 3. Functional Gaps

### 3.1 Pairwise backward

Forward works for `pibar_mode='dense'` and `pibar_mode='topk'` with pairwise transfer rates. Backward is completely untested. This blocks pairwise parameter optimization.

**Severity**: Medium. Non-pairwise (uniform/specieswise) covers the common use case. Pairwise is needed for per-donor-recipient transfer rate estimation.

### 3.2 Threshold annealing for gradient pruning

Per-clade adjoint thresholding achieves 49% pruning with a fixed threshold. Annealing (aggressive early → disabled near convergence) could speed up optimization by 2-5x on the backward pass, especially early when parameters are rough.

**Severity**: Low. Fixed threshold works. Annealing is a performance optimization, not a correctness issue.

---

## 4. Cleanup / Tech Debt

| Item | Effort | Impact |
|------|--------|--------|
| Merge `batched` → `main` | Low | Main is stale (pre-Sep 2025). 40+ commits behind. |
| Remove `scheduling.py` | Trivial | Duplicate of C++ phased scheduler. |
| Guard dead `specieswise + pairwise` branches in `extract_parameters.py` | Trivial | Lines 9-17, 50-58 are invalid combinations. |
| Clean `genewise + pairwise` NotImplementedError | None needed | Design decision — not wanted. |

---

## 5. Performance (low priority)

| Issue | Impact | Notes |
|-------|--------|-------|
| Small-S kernel slower than FP | Low | Fused Triton underperforms at S ≤ 256. Could fall back to FP. |
| `uniform` pibar OOMs at large S | Medium for `uniform` mode | Uses dense [S,S] `recipients_T`. Should exploit ancestor sparsity (O(depth) per species). `uniform` works fine at large S. |

---

## Recommended Sequence

1. **Validate optimizer convergence** (1.1) — run `optimize_theta_genewise` on 100 families at S=1999 against AleRax reference
2. **Validate large-S backward** (1.2) — single family at S=1999, compare to FD
3. **Add pairwise forward tests** — low effort, closes biggest test gap
4. **Add `uniform` pibar pytest** — low effort, currently fragile
5. **Merge to main** — once 1-2 are validated
6. Everything else as needed

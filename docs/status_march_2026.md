# Status: Optimization & Gradient Pipeline (March 12, 2026)

## What Works (Tested & Validated)

### Forward Pass — Pi_wave_forward_v2
- Wave-ordered layout with contiguous zero-copy `Pi[ws:we]` slices
- 3.5-3.9x faster than V1 (gather/scatter), 18.6x faster than global FP
- Three `pibar_mode` options for the transfer matmul:
  - `'dense'`: exact O(W*S^2) cuBLAS matmul
  - `'uniform'`: O(W*S) approximation (row_sum - self), 1e-6 relative error, 4x faster than cuBLAS
  - `'sparse_corrected'`: exact O(W*S^2) matmul via `(1 - ancestors)^T`, matches dense to fp64 precision
- `sparse_corrected` E_step Ebar: `expE @ recipients_T` (O(S^2), same as dense but avoids building the full normalized transfer_mat)
- Tested at S=399 (test_trees_dtl01): NLL=968.739 nats matches AleRax reference
- 16 tests in `tests/unit/test_wave_v2.py` — all pass

### Backward Pass — Pi_wave_backward_v2
- Neumann-series implicit gradient through the Pi self-loop (no autograd tape)
- Produces `v_Pi` (adjoint), `grad_E/Ebar/E_s1/E_s2`, `grad_log_pD/pS`, `grad_max_transfer_mat`
- Optional clade pruning (skip clades with Pi < threshold, saves 50-80% work)
- 15 tests in `tests/gradients/test_wave_gradient.py` — all pass:
  - `TestWaveVsFiniteDifference`: parameter gradients (pD, pS, mt) match FD within 0.1%
  - `TestFullChainFD`: full dL/dtheta through both E and Pi matches FD within 1%
  - `TestEndToEnd`: 5-step `optimize_theta_wave()` decreases NLL

### Full-Chain Implicit Gradient — implicit_grad_loglik_vjp_wave
- Pi backward -> E adjoint (CG solve with GMRES fallback) -> theta VJP
- Includes direct dNLL/dE from the likelihood denominator term
- Chains `grad_Ebar` through to `grad_mt` (Ebar = logsumexp2(E) + mt)
- FD-validated on test_trees_1000 (1 family, fp64)

### Optimizers
- `optimize_theta_wave()`: Adam + implicit gradient, tested (5-step NLL decrease)
- `optimize_theta_lbfgsb()`: scipy L-BFGS-B, supports `gradient_mode='analytical'|'fd'`
- Both support `pibar_mode` in {'uniform', 'dense', 'sparse_corrected'}

### Batching
- `build_wave_layout()`: eq1/ge2 split sorting for efficient DTS reduction (direct copy + seg_logsumexp)
- `original_root_clade_ids` field added — stores root IDs in original (unpermuted) clade space

## What's Untested / Needs Validation

### sparse_corrected mode — no automated tests
- Verified manually via ad-hoc scripts (`/tmp/test_v2_correct.py`, `/tmp/test_sparse_corrected_e2e.py`)
- Matches dense exactly (diff=0 at fp64) at S=399
- FD optimizer converges to AleRax reference (NLL=968.739 nats)
- **Needs**: pytest coverage in `test_wave_gradient.py` or `test_wave_v2.py`:
  - E_step sparse_corrected matches dense
  - Pi_wave_forward_v2 sparse_corrected matches dense
  - Full-chain gradient FD test with sparse_corrected
  - Optimizer convergence test with sparse_corrected

### Specieswise + sparse_corrected
- `specieswise=True, pairwise=False` works with `pibar_mode='uniform'` (tested)
- `sparse_corrected` supports specieswise: per-species `mt[s]` broadcasts correctly
- **Not validated**: specieswise + sparse_corrected end-to-end

### Pairwise transfers
- Require O(S^2) dense matmul — `sparse_corrected` and `uniform` don't apply
- `pibar_mode='dense'` with `pairwise=True` is the only valid mode
- **Not tested** end-to-end with the wave pipeline

### Multi-family optimizer convergence
- `optimize_theta_wave()` and `optimize_theta_lbfgsb()` are designed for batched families
- Only tested with 1 family in `TestEndToEnd`
- **Needs**: test with 5-10 families, verify NLL decreases and rates are reasonable

### Large-S (S >= 10K) with analytical gradient
- Forward pass works at S=20K (0.26s/family, 18 GB peak)
- Backward pass untested at large S — gradient pruning is designed for this but not validated at scale
- CG/GMRES solve may need tuning at large S (Jacobian structure changes)

### E adjoint numerical stability
- CG solve for `(I - G_E^T) w = q` assumes convergent fixed point (spectral radius < 1)
- Not tested with rates close to instability (high D or T rates where spectral radius -> 1)
- GMRES fallback exists but untested in practice

## Known Issues

### Legacy import errors (10 test files)
- `src.core.ccp` and `src.core.tree_helpers` no longer exist
- 10 test files fail to import (41 tests uncollectable)
- These test the old `ThetaOptimizationProblem` and non-wave gradient paths
- `theta_optimizer.py` guards legacy imports with `try/except` — importable but `ThetaOptimizationProblem.__init__` raises

### Gradient residual at optimum
- FD optimizer shows persistent |g| ~ 0.6-0.7 at convergence (FD artifacts at 1e-5 step size)
- Analytical gradient has not been tested at the optimum to see if it reaches |g| ~ 0
- This would validate that the gradient is truly zero at the converged rates

### optimize_theta_lbfgsb() with analytical gradient
- `gradient_mode='analytical'` path exists but has not been run end-to-end
- Only `gradient_mode='fd'` has been tested (via ad-hoc scripts)
- **The key test**: does L-BFGS-B + analytical gradient converge to AleRax's optimum?

## Architecture Notes

### Root clade ID spaces
- `wave_layout['root_clade_ids']`: **permuted** space — used internally by V2 self-loop and backward pass
- `wave_layout['original_root_clade_ids']`: **original** space — use with `compute_log_likelihood()` on V2's unpermuted Pi output
- `batched['root_clade_ids']`: **original** space — same as above
- Production code (`model.py`, `theta_optimizer.py`) already uses the correct IDs

### Transfer matrix modes
| Mode | Pibar cost | Ebar cost | Accuracy | When to use |
|------|-----------|-----------|----------|-------------|
| `dense` | O(W*S^2) | O(S^2) | Exact | Small S, pairwise rates |
| `uniform` | O(W*S) | O(S) | ~1e-6 rel | Non-pairwise, any S |
| `sparse_corrected` | O(W*S^2) | O(S^2) | Exact | Non-pairwise, need exact + no normalized transfer_mat |

`sparse_corrected` is exact like `dense` but works with the raw `(1 - ancestors)^T` matrix instead of the normalized transfer matrix. For constant or specieswise rates, it's equivalent to `dense`. For pairwise rates, only `dense` works.

## File Manifest (Unstaged Changes)

| File | Changes |
|------|---------|
| `src/core/likelihood.py` | sparse_corrected in E_step, _compute_Pibar_inline, Pi_wave_forward_v2; recipients_T parameter threading |
| `src/core/batching.py` | eq1/ge2 split sorting in build_wave_layout; original_root_clade_ids field |
| `src/optimization/theta_optimizer.py` | Guard legacy imports; implicit_grad fix (denominator E term, mt+Ebar gradient); optimize_theta_wave(); optimize_theta_lbfgsb() |
| `tests/gradients/test_wave_gradient.py` | TestFullChainFD, TestEndToEnd classes |

## Recommended Next Steps

1. **Add sparse_corrected to automated tests** — the code is validated but fragile without CI coverage
2. **Run optimize_theta_lbfgsb with analytical gradient** — verify convergence to AleRax optimum with |g| -> 0
3. **Multi-family optimizer test** — 5-10 families, check rates and NLL
4. **Commit the current changes** — batching.py and likelihood.py are interdependent (eq1/ge2 sorting required by _compute_dts_cross)
5. **Clean up legacy test files** — either fix imports or mark as xfail/skip

# Box ↔ prior substitution, and the two-floor |g| diagnosis (archaea species-wise CV)

**TL;DR.** The DTL rate **box** and the **Sanderson tree-smoothing prior** (λ·R, R = ½Σ_edges‖θ_c−θ_p‖²)
are *substitute* regularizers for the same non-identifiability. A **tight** box does the regularizing, so
cross-validation wants **no** prior (λ\*→0); a **loose/relaxed** box leaves the rates free, so CV recovers a
genuine **interior smoothing optimum** (λ\*≈0.02). Either way the *fit* — NLL and the per-species event
probabilities — is identical to <0.1%; the regularizer only controls **where the non-identifiable D–L rates
sit**, not the likelihood. A clean |g|=0 stationary point exists only with the tight box; with the loose box
the freed D–L turnover direction is non-identifiable and |Pg| floors at ~0.018 (the loss is still converged).

Model: species-wise UndatedDTL, θ∈R^{S×3}=[θ_D,θ_L,θ_T] = log2 rates; event probs are a log2-softmax with
speciation pinned at 1: p = [1, 2^θ_D, 2^θ_L, 2^θ_T]/Z (`extract_parameters_uniform`). Archaea: S=119
species, p=357. CV = K=5-fold held-out predictive NLL over families, GBM penalty, descending-λ homotopy
(`run_cv.py`; box added via `--min-rate/--max-rate`).

## 1. CV curves (mean held-out NLL over 5 folds)

### archaea-256, TIGHT box ρ∈[0.05, 2.0]  → λ\* = 0
| λ | 10 | 3 | 1 | 0.3 | 0.1 | 0.03 | 0.01 | 0 |
|---|---|---|---|---|---|---|---|---|
| CV | 5558.9 | 5510.4 | 5478.2 | 5453.0 | 5446.7 | 5445.45 | 5445.50 | **5445.43 ←λ\*** |

Monotone-decreasing toward λ=0: the box already removes the boundary runaway, so the prior adds nothing.
(refit % at box: λ=10→1.4%, λ=0.03→47%, λ=0→53% — the box does the regularizing.)

### archaea-256, LOOSE box ρ∈[1e-4, 16]  → λ\* ≈ 0.02   (figure: cv_loosebox_curve.pdf)
| λ | 3 | 1 | 0.3 | 0.1 | 0.05 | 0.03 | 0.02 | 0.01 | 0 |
|---|---|---|---|---|---|---|---|---|---|
| CV | 5509.1 | 5474.0 | 5450.3 | 5440.4 | 5437.7 | 5437.5 | **5437.46 ←λ\*** | 5438.1 | 5440.7 |
| Δ vs λ\* | +71.6 | +36.6 | +12.8 | +2.9 | +0.22 | +0.02 | 0 | +0.66 | +3.2 |

Genuine interior U (flat bottom λ∈[0.01,0.05]). Over-smoothing steeply worse; **no-prior (λ=0) is +3.2 worse**
than λ\* (0.063 NLL/family). refit % at box ~0 for all λ>0 (prior keeps the fit interior), 8.7% at λ=0.

### archaea-FULL 5446, TIGHT box ρ∈[0.05, 2.0]  (3-λ probe) → λ\* = 0.01 (flat)
| λ | 0.1 | 0.03 | 0.01 |
|---|---|---|---|
| CV | 72012.19 | 72011.99 | **72011.75 ←λ\*** |

Flat to 0.0005 NLL/family — same tight-box story at full scale. The bounded refit at λ=0.03 reproduces the
§5.2 certified bounded minimum (F=359,592 ≈ 359,591.9). *(Full + loose box = pending run.)*

## 2. The two |g| floors (loose box, archaea-256, λ=0.03)

The L-BFGS endpoint is **not** stationary: raw |g|=2.8, |Pg|=0.21. Driving it down hits **two** distinct floors:

**Floor 1 — fp32 precision (|Pg|≈0.21), real and fixable.**
The loss ≈26,501; fp32 resolves a sum that size to ≈0.003 NLL, but the remaining descent in the flat D–L
direction is ~1e-4 NLL/step — *below* the fp32 floor. Evidence: fp32 vs fp64 loss differ by 0.0044; FD of the
loss along −Pg is sign-flipping garbage in fp32 (−0.73,+9.2,+0.44) but matches the analytic gradient in fp64
(−0.213≈−0.211); line search uphill in fp32, downhill in fp64. **pi/neumann 64→128→256 leaves |Pg| unchanged**
→ not a solver-truncation floor, purely float precision. fp64 L-BFGS breaks through: |Pg| 0.21→0.018.

**Floor 2 — non-identifiability (|Pg|≈0.018), intrinsic and unfixable by any optimizer.**
fp64 L-BFGS, matrix-free Newton-CG, AND exact dense fp64 Newton all stall here. The Hessian is PD
(λ_min=+0.0154, λ_max=244, κ=1.6e4 — fully within fp64's reach), yet the exact Newton step (|d|=2.18) cannot
descend: Armijo cuts α→7e-9. Cause: the soft D–L eigendirection is ⊥ the gradient and the objective is
non-quadratic there, so Newton mis-steps; the gradient direction is stiff (~244) so gradient descent crawls.
The **loss is converged** (improvable by ~1e-6); |Pg|=0.018 IS the D–L confounding, not non-convergence.
A clean |g|=0 only exists when the box pins this direction (tight box → §5.2 |Pg|=3e-3).

## 3. fp64 vs fp32 changes nothing reportable (verified)

Optimizing the loose-box λ=0.03 fit in fp64 vs the fp32 endpoint:
- **NLL: −0.164 (−6e-4 %)**, penalized F: −0.016.  → the fit is unchanged.
- **θ: ‖Δθ‖=0.81 (0.9% rel)**; median rate change 0.36%, max 17.6% — but only on tiny near-floor rates.
- **Event probabilities (mean over 119 species) unchanged**: Speciation 0.583, Loss 0.236, Transfer 0.105,
  Duplication 0.076; largest single-species shift **0.4 pp**; 0/119 moved >0.8% total-L1.

So the fp32→fp64 work was diagnostically decisive but **practically inert**: it proved |Pg|=0.21 was a
precision artifact and the real floor is non-identifiability, while leaving NLL, probabilities, and λ\*
exactly where fp32 had them. **fp32 CV curves are reliable** (held-out differences 0.06–1.4/family ≫ fp32 noise).

## Reproduce
```
cd <worktree>; export PYTHONNOUSERSITE=1 PYTHONPATH=$PWD \
  GPUREC_PREPROCESS_PATH=$PWD/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so
PY=/home/enzo/miniforge3/bin/python
# loose-box CV curve (256 fam):
$PY experiments/sanderson_cv/run_cv.py --dataset archaea --families 256 --k 5 \
   --lambdas 3 1 0.3 0.1 0.05 0.03 0.02 0.01 0 --min-rate 1e-4 --max-rate 16 --no-wandb \
   --outdir experiments/sanderson_cv/runs/cv_archaea_n256_box_1e-4_16_grid
# tight-box (drop --min-rate/--max-rate or use 0.05 2.0); full archaea: --families 5446
```
Diagnostics (scratchpad copies): diagnose_stall.py / diagnose_fp64.py (precision), newton_fp64 /
dense_newton_fp64 (the |g| floor), compare_before_after.py (fp64 vs fp32 deltas). Figure: cv_loosebox_curve.pdf.

# Saddle-escape + Newton polish: the CV optimum is a saddle, not a minimum

## The finding

The Sanderson cross-validation selects a *light* smoothing (**λ\*=0.03** on both hogenom-1055 and the
60-species archaea set; see `sanderson_cv.md`). But at λ\*=0.03 the GBM-penalized Hessian is **not
positive-definite** — the CV-optimal MAP point is a **saddle**, not a local minimum. This is the
concrete form of the Sanderson tension: the regularization that generalizes best is too light to make
the likelihood identifiable/convex.

The standard pipeline (`run_cv`: Adam → scipy L-BFGS-B) **cannot escape this saddle** — L-BFGS keeps a
*positive-definite* Hessian model, so it has no descent direction for negative curvature and simply
parks with a small gradient. The fix is the classic three-stage endgame:

**Adam → L-BFGS → saddle-free escape (descend the negative-curvature eigenvector) → Newton polish.**

Each stage is load-bearing, demonstrated end-to-end on archaea λ=0.03 (exact 357×357 `eigh`, fp64):

| stage | F (penalized NLL) | ‖g‖ | λ_min(H+λL) | n_neg |
|---|---|---|---|---|
| CV refit (Adam→L-BFGS) | 26500.94 | 0.29 | **−0.0192** (saddle) | 2 |
| + saddle escape (line-search along v₀, re-converge) | 26500.64 | 0.031 | **+0.0137** (PD basin) | 0 |
| + one Newton step (δ=−H⁻¹g, ‖δ‖=0.14) | 26500.64 | **1.4e-4** | **+0.0138** (PD) | 0 |

So the escaped+polished point is a **certified true local minimum** — zero gradient (1.4e-4, ~230× below
the L-BFGS floor) and positive-definite Hessian (exact `eigh`, no residual ambiguity) — sitting **0.30 NLL
below** the saddle the CV refit reported. Note the Newton step refines the *gradient*, not the *loss*
(F unchanged to 4 dp): the residual ‖g‖≈0.03 after escape is entirely in the **soft direction**
(λ_min≈0.014), which a PD-Hessian-model optimizer (L-BFGS) can't resolve but a curvature-aware Newton
step nails in one shot (‖δ‖=0.14, mostly along that soft eigenvector).

**hogenom λ=0.03** is the same: its certified bottom Rayleigh quotient is `u₀ᵀHu₀ = −0.046 < 0` (a
constructive negative-curvature witness — decisive on the *sign* even though p=3993 leaves the precise
eigenvalue unresolved at M=200). For hogenom, λ=1 was PD (+0.066) while λ=0.03 is a saddle (−0.046), so
the curvature **crosses zero** as regularization lightens.

## The full λ-grid: every penalized λ is a certified minimum (archaea)

Running escape+Newton-polish+exact-`eigh`-cert on the whole homotopy grid (archaea, 256 families, each λ
seeded from its `run_cv` refit; A100 fp64, job `4635197`):

| λ | refit λ_min | refit a saddle? | final λ_min(H+λL) | final ‖g‖ | F(refit)→F(final) | certified min? |
|---|---|---|---|---|---|---|
| 10   | +0.7322 | no  | **+0.7373** | 4.7e-4 | 27713.907→27713.902 | ✅ |
| 3    | +0.4616 | no  | **+0.4600** | 1.4e-4 | 27342.562→27342.561 | ✅ |
| 1    | +0.2221 | no  | **+0.2211** | 4.9e-4 | 27036.110→27036.108 | ✅ |
| 0.3  | +0.0428 | no  | **+0.0561** | 4.9e-4 | 26768.181→26768.148 | ✅ |
| 0.1  | +0.0362 | no  | **+0.0356** | 2.7e-5 | 26604.186→26604.164 | ✅ |
| 0.03 | −0.0192 | **yes** (n_neg=2) | **+0.0138** | 1.4e-4 | 26500.942→26500.645 | ✅ (escaped) |
| 0    | −0.0123 | **yes**           | **−0.0015** | 3.6e-2 | 26407.781→26407.408 | ❌ (stays indefinite) |

Two readings:

1. **λ_min(λ) is monotone decreasing and crosses zero just below λ=0.03.** Curvature falls smoothly
   0.74 → 0.46 → 0.22 → 0.056 → 0.036 → **+0.014 (λ=0.03)** → **−0.0015 (λ=0)**. Every positive penalty
   produces an isolated PD minimum; the **bare MLE (λ=0) has none** — even after a saddle-escape it relaxes
   only from −0.012 to −0.0015 and remains indefinite. This is the raw-MLE non-identifiability made
   quantitative: the GBM penalty is what *creates* the minimum, and the CV-selected λ\*=0.03 is the
   lightest penalty that still does so.

2. **Two regimes, both needing curvature-awareness for different reasons.** At λ≥0.1 the refit is *already*
   PD but its gradient is large (0.2–0.3) and lives almost entirely in the **soft eigendirection** —
   L-BFGS floors there; Newton clears it in 1–3 steps. At λ=0.03 the refit is a *genuine saddle* (n_neg=2)
   that needs the negative-curvature escape first. The L-BFGS endpoint is never the answer at light λ.

**Why the line search is load-bearing (v1→v2).** The first pass used a single undamped Newton step
δ=−H⁻¹g and *failed to converge λ=0.3 and λ=0.1*: at near-flat points H⁻¹ amplifies the soft direction by
~1/λ_min, the full step overshoots the quadratic-trust region, and ‖g‖ *increases* (λ=0.3 ended at
‖g‖=0.23, λ=0.1 at 0.011 — both reported `certified=False`). The hardened `newton_polish` — iterated, with
a backtracking line search accepting the largest α∈{1,½,…} that decreases ‖g‖ — fixes both (λ=0.3:
0.216→0.120 @α=½ →0.073 @α=1 →4.9e-4 @α=1; λ=0.1: 0.289→0.011→2.7e-5). **All future runs/scripts use this
line-searched path** (it is the only Newton entry point in `saddle_escape.py` / `certify_all_lambdas.py`).

## Why L-BFGS floors, why Lanczos needed M≈p

- **L-BFGS ‖g‖ floor is precision-, not optimizer-limited**: extended fp32 L-BFGS gains 0 loss (already
  at the fp32 machine-eps wall); fp64 drives ‖g‖ ~10× lower for a negligible loss change. The floor
  scale tracks the objective magnitude (archaea ~0.05, hogenom ~3.6 — *not* comparable across datasets).
- **Lanczos resolves the bottom only when M is adequate vs p**: at archaea p=357, M=300 Lanczos matches
  the exact `eigh` to 1.9e-7 (resid 1e-4). At hogenom p=3993, M=200 leaves a large residual — but the
  negative Ritz *value* is still a valid Rayleigh quotient, so the saddle sign is proven regardless. For
  small problems prefer the **exact full Hessian** (form p HVPs → dense `eigh`): definitive, no ambiguity.

## Reproduce it

Code: `experiments/sanderson_cv/saddle_escape.py` (efficient — builds the exact-HVP cache **once** per
point; auto-uses exact `eigh` for p≤1200, converged Lanczos + CG-Newton above). Pinned checkpoints:
`experiments/sanderson_cv/_artifacts/archaea_lam0.03_{saddle,newton_polished}.pt`.

```bash
export GPUREC_PREPROCESS_PATH=<repo>/gpurec/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so
export PYTHONPATH=<worktree>

# 1. (optional) regenerate the saddle θ from scratch — the archaea λ=0.03 all-data refit:
python experiments/sanderson_cv/run_cv.py --dataset archaea --families 256 --k 5 \
    --lambdas 10 3 1 0.3 0.1 0.03 0 --lbfgs-iters 150 --no-wandb --outdir runs/cv_archaea
#   -> runs/cv_archaea/ckpt/refit_lam5.pt   (== _artifacts/archaea_lam0.03_saddle.pt)

# 2. escape + Newton-polish + re-certify (DATASET mode builds the model from the in-repo .ale data):
DATASET=archaea FAMILIES=256 LAM=0.03 \
  THETA=experiments/sanderson_cv/_artifacts/archaea_lam0.03_saddle.pt \
  OUT=experiments/sanderson_cv/_artifacts/archaea_lam0.03_newton_polished.pt \
  python experiments/sanderson_cv/saddle_escape.py
#   -> prints the table above; OUT holds theta_{saddle,escaped,newton} + λ_min/‖g‖/loss at each stage.
```

For hogenom (p=3993, fp64, A100): same script in **capture mode** —
`CAP=<capture_1055.pt> THETA=<hogenom λ=0.03 refit> LAM=0.03 FULL_HESSIAN=0 python saddle_escape.py`
(uses converged Lanczos for v₀ and a CG-based Newton solve). The capture + refit θ are staged at
`/work/SzollosiU/enzo-marsot/sanderson-polish/` on the cluster.

The pinned `*_newton_polished.pt` is the regression target: re-running `saddle_escape.py` on the pinned
saddle θ must reproduce λ_min_saddle≈−0.019 → λ_min_newton≈+0.0138 and ‖g‖_newton ≈ 1e-4.

**Full λ-grid (the certified table above).** `experiments/sanderson_cv/certify_all_lambdas.py` loops the
homotopy grid, calling `saddle_escape.run` (always the line-searched Newton) on each `refit_lam{i}.pt` and
printing the CERTIFIED-MINIMUM SUMMARY:

```bash
DATASET=archaea FAMILIES=256 CKPT_DIR=<run_cv outdir>/ckpt \
  OUT_DIR=experiments/sanderson_cv/_artifacts/certified_v2 LAMBDAS="10 3 1 0.3 0.1 0.03 0" \
  python experiments/sanderson_cv/certify_all_lambdas.py
```

Pinned outputs for all 7 λ + the cluster run log are in `_artifacts/certified_v2/`
(`archaea_lam{λ}_certified.pt`, `certify_all.4635197.log`; A100 fp64, ~65 min). Each holds
theta_{saddle,escaped,newton} and λ_min/‖g‖/F at every stage.

## Implication for the pipeline

`run_cv` reports the L-BFGS endpoint, which at light λ is a **saddle** (or, at λ≥0.1, a PD point whose
gradient still floors in the soft direction). To report a genuine MAP point estimate, chain
`saddle_escape.py` (escape + line-searched Newton) after the L-BFGS refit — at every penalized λ this
yields a certified PD minimum (see the grid table). The components exist in the `newton/` research layer;
this experiment is the minimal, verified version on the merged code. (The CV *curve* and λ\* are
unaffected — the saddle/floor→min gap is ~uniform across λ and a tiny fraction of the held-out NLL.)

The one λ that resists is **λ=0**: with no penalty there is no isolated PD minimum to find — the escape
relaxes λ_min only from −0.012 to −0.0015, still indefinite. That is the expected, quantified raw-MLE
non-identifiability, and the affirmative case for the Sanderson penalty: regularization is what makes the
rate estimate a well-posed minimum at all.

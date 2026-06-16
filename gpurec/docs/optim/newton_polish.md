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

## Implication for the pipeline

`run_cv` reports the L-BFGS endpoint, which at light λ is a **saddle**. To report a genuine MAP point
estimate, chain `saddle_escape.py` (escape + Newton) after the L-BFGS refit. The components exist in the
`newton/` research layer; this experiment is the minimal, verified version on the merged code. (The CV
*curve* and λ\* are unaffected — the 0.30-NLL saddle→min gap is ~uniform across λ and ~0.006% of the
held-out NLL.)

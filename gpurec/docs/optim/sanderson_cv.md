# Sanderson-style penalized-likelihood cross-validation (GBM rate prior)

Runs penalized maximum likelihood with an **autocorrelated-rates (GBM) roughness penalty** and
chooses the smoothing strength λ by **k-fold cross-validation over families**, on the full
1055-family hogenom set. This is the principled resolution of the raw-MLE non-identifiability
documented in `landscape-nonidentifiability` / `specieswise-basin-mapcv`: at λ=0 about half the
per-species DTL rates run to the 0/1 boundary (unidentifiable, flat likelihood); λ→∞ collapses to a
single clock rate; the CV minimum is the principled middle.

Code: `experiments/sanderson_cv/run_cv.py` (+ the `tree_penalty` arg added to
`gpurec.optim.value_and_grad.make_value_and_grad`).

## The penalty

`R(theta) = ½ · Σ_{edges (child c, parent p)} ||theta[c] − theta[p]||²` — a graph-Laplacian
quadratic over the species tree that shrinks each species' rates toward its **parent's** (Sanderson
2002 autocorrelated-rates / GBM flavor), **not** toward an arbitrary constant. So λ→∞ gives the
molecular clock with the common rate set by the data, not by a guessed center. Its Hessian is
`λ·L` (PSD, L = tree graph Laplacian), so it composes with the bare NLL Hessian and preserves the
certifiable-PD structure of the old centered ridge.

- Parent map: `model.species_helpers["sp_parent"]` (int32 `[S]`, `−1` at the root, postorder).
- Gradient: `+λ(θ_c−θ_p)` on the child, `−λ(θ_c−θ_p)` on the parent (verified vs autograd to 8e-17
  and FD to 2e-9; end-to-end through a real model loss-exact / grad 3e-8).
- `make_value_and_grad(..., tree_penalty=(lam, sp_parent))`. Composes with the centered `prior=`.

## Design decisions

| decision | choice | why |
|---|---|---|
| init θ | **θ=0** (all DTL probs 0.25) | the empirically better basin (`specieswise-basin-mapcv`) |
| solver | **converged** pi=64 / neumann=64 | pi=16 gradient is biased ~5%; would corrupt the per-fold optima |
| λ schedule | **homotopy high→low, warm-started** | each fit starts inside the previous (more convex/PD) basin |
| optimizer | Adam (basin entry) → scipy **unconstrained L-BFGS-B** (`bounds=None`) | the prior regularizes; θ is NOT boxed. scipy ships no plain L-BFGS — L-BFGS-B with no bounds *is* unconstrained L-BFGS |
| **solver dtype** | **fp32** on the 4090 | fp64 is **27× slower** on consumer cards (333 ms → 8974 ms) with **bit-identical** loss; the grad's atomic noise floor (~2e-4) dominates regardless of dtype. Use fp64 only on an A100 (`solve_dtype`) |
| fold unit | per-family NLL, train/test by **data-subsetting** | specieswise E is species-only ⇒ NLL additive over families (verified ~1e-8); held-out NLL is a true predictive quantity |

## The "true local minimum" check (PD certificate)

For each all-data refit we check whether θ is a genuine local minimum: smallest eigenvalue of
`H + λL` via Lanczos min-eig. **The HVP must be the analytic exact-HVP** (`hvp_exact`), summed over
the 5 batches (`H = Σ_b H_b`, batch NLLs are additive) plus the exact penalty term `λL`. We report
`lam_min` and the **Ritz residual**; `lam_min > 0` with a small residual ⇒ certified PD true local
minimum. Run post-hoc by `certify.py` (needs the whole GPU), which writes the result back into
`state.pt`.

**Why not the cheaper FD-of-gradient HVP** (diagnosed in `_cert_diag.py` against the exact-HVP as
ground truth on the single-batch 48-family refit): the bottom eigenvalue here is *tiny* — e.g. at
λ=10 the exact-HVP gives `lam_min = +0.0245` (residual 0.012), a true min but barely positive because
bare-H is near-singular (≈ −0.05) and `λL` only just lifts it. The FD HVP's error floor is
~0.5 %·‖H‖₂ (truncation-limited at eps=1e-2; `grad_avg_K` doesn't help, so it's not noise), which is
~0.3–0.5 in absolute terms — *larger than* the eigenvalue we need to resolve. FD Lanczos accordingly
returned residuals 0.28–0.46 and even flipped the sign (−0.011). FD with eps=1e-3 was far worse
(9.5 % HVP error — that was the original smoke-cert garbage). So FD is fine for Newton steps but
**cannot certify a near-zero `lam_min`**; the exact-HVP can (clean residual 0.012).

This is run on the all-data refits (the θ we report), not on every CV fold cell — a per-cell
certificate would add hours. The cheap necessary condition (gradient norm) is logged on **every** fit.

## Robustness / resume

A multi-hour job, so it is built to survive a crash and to be inspected mid-flight:

- **λ-level checkpoints**: `ckpt/fold{fi}_lam{li}.pt` and `ckpt/refit_lam{li}.pt` hold θ; a fold's
  homotopy resumes from the last completed λ, completed folds are skipped entirely.
- **`state.pt`** (atomic write): config, every cell's result (held-out NLL, final loss/grad-norm,
  θ-stats), the CV curve, λ*, and the certified refit table. Reloaded on restart.
- **`events.jsonl`**: one JSON record per logged iteration (append-only) — full trajectory for
  post-hoc inspection even if wandb is off.
- **wandb** (`project=gpurec-sanderson-cv`, `resume="allow"`): per-iteration NLL, grad norm, LR,
  phase, θ-stats (incl. `frac_extreme` = fraction of |θ|>5, the boundary-saturation indicator), and
  per-iteration wall time; per-cell summaries. `--no-wandb` to disable.

## Cost (RTX 4090, fp32, pi=64/neu=64)

| scale | build | value+grad | value-only | peak mem |
|---|---|---|---|---|
| 64 families | 0.8 s | 0.42 s | 0.18 s | 1.45 GB |
| 1055 families (5 batches) | 7.4 s | **6.58 s** | 2.68 s | 6.66 GB |

A full k=5 CV over the λ-grid is a ~3–5 h job; memory stays ~7 GB (batched), well within 24 GB.

## Running

```bash
export GPUREC_PREPROCESS_PATH=/home/enzo/Documents/git/gpurec/gpurec/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so
export PYTHONPATH=$PWD                      # the worktree root
# smoke first (minutes):
python experiments/sanderson_cv/run_cv.py --smoke 48 --k 2 --lambdas 10 1 0 --no-wandb \
    --adam-steps 20 --lbfgs-iters 40 --cert-m 16 --outdir experiments/sanderson_cv/runs/smoke
# the real run:
python experiments/sanderson_cv/run_cv.py --families 1055 --k 5 \
    --lambdas 1000 100 10 1 0.1 0 --outdir experiments/sanderson_cv/runs/cv_1055
```

Reads the family list from `experiments/sanderson_cv/families_1055.txt` (snapshot of the canonical
`hogenom_1055` set), the species tree + gene trees from `tests/data/alerax_hogenom_core/hogenom`.

## Results

_(to be filled after the 1055-family run: the CV curve, λ*, the certified-PD refit table, and
whether the boundary-saturation `frac_extreme` collapses and run-to-run variance drops vs the raw
MLE — the science signal that regularization resolves the non-identifiability.)_

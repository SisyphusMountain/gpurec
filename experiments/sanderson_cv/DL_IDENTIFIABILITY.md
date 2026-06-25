# D–L turnover: the weakly-curved direction of the species-wise DTL Hessian

**What this is.** A factual record of what we *measured* about the duplication–loss (D–L) confounding in
the species-wise undated DTL fit, plus the script that produced it. Scope is deliberately narrow: the
numbers and their immediate reading — no claim about *why* beyond the curvature itself, and no proposed fix.

## 1. Setup

Species-wise undated DTL on a fixed species tree. Per species `s`, log₂-rates `θ_s = (θ_D, θ_L, θ_T)`,
rate `r_X = 2^{θ_X}`; event probabilities are a soft-max with speciation pinned (`p_X = 2^{θ_X}/Z`).
Penalized loss `F = NLL + (λ/2)·Σ_{edges}‖θ_c − θ_p‖²` (GBM tree-Laplacian smoothing prior,
`penalty = λ·½ θᵀ(L⊗I₃)θ`). Archaea: `S = 119`, `p = 357`.

## 2. Measured Hessian signature

At a (regularized) species-wise optimum the Hessian is strongly anisotropic, per species:

- **soft** direction ≈ `θ_D + θ_L` ("turnover"): eigenvalue ≈ +0.015 (256-fam, fp64), +0.031 (full).
- **stiff** direction ≈ `θ_D − θ_L` ("net"): eigenvalue ≈ 244.
- condition number κ ≈ 1.6·10⁴ (256-fam) → ≈ 3.8·10⁴ (full).
- per-species D–L curvature coupling ≈ 0.93; transfer decoupled.
- the data Hessian alone is indefinite (4 negative eigenvalues, 90/357 below 0.1); the prior makes `F` PD.

Local algebra (why the eigenvectors are net / turnover): with `d = 2^{θ_D}`, `ℓ = 2^{θ_L}`,
`u = (θ_D+θ_L)/2`, `w = (θ_D−θ_L)/2`, the net `n = d−ℓ` has `∂n/∂u = (ln2)(d−ℓ)` and
`∂n/∂w = (ln2)(d+ℓ)`, so when `d ≈ ℓ` the dependence on the turnover coordinate `u` is ≪ that on the net
coordinate `w` — turnover is the soft direction. (Standard reading: a duplication then immediate loss of
one copy leaves a pattern similar to no event, so the data constrain net `d−ℓ` better than total `d+ℓ`.)

## 3. The Laplacian-null measurement (the new facts)

We separated **data** curvature from **prior** curvature on the turnover direction, prior-free. The
**global** turnover mode `v` (every species `δθ_D = δθ_L = +1`, `δθ_T = 0`) is constant across the tree, so
it lies in the null space of the tree Laplacian `L` — the prior adds *exactly zero* curvature to it.
Evaluated at the converged loose-box (rate ∈ [1e-4, 16]) λ=0.03 archaea-256 minimum, fp64:

| # | quantity | value |
|---|----------|-------|
| 1 | global turnover, data curvature `vᵀH_data v` | **+2.6498** (data+prior +2.6498; prior contribution −2.1·10⁻¹⁴) |
| 2 | global net, data curvature | +17.262 (net / turnover = 6.5×) |
| 3 | global turnover data curvature vs #families | m=64 → 0.689, 128 → 1.373, 256 → 2.650 (per-family ≈ 0.0107, constant) |
| 4 | turnover subspace (119-dim, `U` = per-species D=L) | `λ_min(UᵀH_data U) = +0.0184` (0/119 below 1e-3); `λ_min(UᵀH_F U) = +0.0438` |

## 4. What the numbers say (only this)

- The turnover direction is **weakly curved by the data, not flat**: every turnover direction has positive
  data curvature (test 4, none null); the global turnover mode has data curvature +2.65 with the prior
  contributing exactly zero (test 1).
- That data curvature **grows ~linearly with the number of families** (test 3, per-family ≈ 0.0107 constant)
  — it accumulates like Fisher information rather than sitting at a fixed numerical floor.
- The split is global-vs-local: the **global / shared** turnover level is data-determined and prior-immune;
  only the **per-species** turnover contrasts are data-poor, and the tree prior is what lifts those
  (test 4: 0.018 → 0.044).

No claim beyond this. In particular we do **not** establish any exact non-identifiability or a specific
generative mechanism — only that the measured curvature of the turnover direction is small, positive, and
sample-additive.

## 5. Reproduce

`experiments/sanderson_cv/dl_identifiability.py` (fp64): builds the model, loads a converged θ, forms
`H_data` (`make_lap` with λ=0) and `H_F` (`make_lap` with λ) HVP operators, and evaluates the quadratic
forms and the turnover-subspace `UᵀHU` spectrum. Converged point used:
`runs/cv_archaea_n256_box_1e-4_16/refit_lam0.03_fp64_converged.pt`.

Caveat the tests exploit: the GBM penalty does **not** penalize a globally constant shift of all `θ_D`
(or all `θ_L`) — that mode is in the Laplacian null space — which is exactly why test 1/3 isolate pure
data curvature.

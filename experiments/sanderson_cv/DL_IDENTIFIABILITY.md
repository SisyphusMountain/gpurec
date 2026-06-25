# The duplication–loss (D–L) non-identifiability in undated DTL reconciliation

**TL;DR.** In the species-wise undated DTL model, the per-species **duplication** and **loss** rates are
confounded: the data constrains their *difference* (net gene gain/loss, `θ_D−θ_L`) far better than their
*sum* (turnover, `θ_D+θ_L`). This produces a near-singular Hessian (κ ≈ 1.6·10⁴ on archaea-256, ≈ 3.8·10⁴
full) with a soft eigendirection along turnover. We tested whether this is an **exact** non-identifiability
(à la Louca–Pennell) or merely **weak**. The measurements are decisive: it is **weak, not exact**. Turnover
carries genuine, family-additive data information (Fisher info grows linearly in the number of families);
**global turnover is fully data-determined and grows stiff with sample size**; only the **per-species
turnover contrasts** are data-poor, and the tree-smoothing prior shrinks exactly those — legitimate
hierarchical shrinkage, not fabricated information. The Louca–Pennell exact null is the *pruned* limit this
model perturbs away from.

**Confirmed conclusion (canonical statement for handoff).**

> The D–L turnover direction is the weak coordinate of the undated DTL reconciliation model, but it is **not**
> an exact non-identifiability in the full scaffolded model: **global turnover has prior-immune,
> family-additive Fisher curvature**, while only **local per-species turnover contrasts remain data-poor**;
> the Louca–Pennell exact null is the **pruned** limiting model that DTL perturbs away from.

---

## 1. Model

A gene family evolves along a fixed species tree (`S` species) under an **undated DTL** process:
Speciation, Duplication, Loss, Transfer. Per species `s`, three free log₂-rate parameters

```
θ_s = (θ_D, θ_L, θ_T)_s ∈ ℝ³ ,   rate r_X = 2^{θ_X}.
```

Event probabilities are a soft-max with speciation pinned to logit 0:

```
p_X = 2^{θ_X} / Z ,   Z = 1 + 2^{θ_D} + 2^{θ_L} + 2^{θ_T} ,   X ∈ {D, L, T};   p_S = 1/Z.
```

The likelihood `P_θ(G)` of the observed gene-tree datum `G` (an ALE amalgamation) is computed by summing
over all latent reconciliations via a dynamic program on the species tree. We minimize a penalized NLL with
a Gaussian/Brownian (GBM) tree-smoothing prior:

```
F(θ) = NLL(θ) + (λ/2) · Σ_{(c,p)∈edges} ‖θ_c − θ_p‖²   (penalty = λ · ½ θᵀ(L ⊗ I₃)θ).
```

Archaea: `S = 119`, `p = 357`.

---

## 2. The confounding — mechanism and empirical signature

**Mechanism (why D and L confound).** The observed gene trees inform the *net flux* of gene copies (do
families grow or shrink, `~ r_D − r_L`) much better than the *total turnover* (`~ r_D + r_L`). The reason is
structural:

> A **duplication immediately followed by loss of one copy** leaves the surviving gene tree topologically
> unchanged — an *invisible* event pair.

So high-D + high-L (lots of cancelling churn) is nearly indistinguishable from low-D + low-L at fixed net
drift. The data pins `r_D − r_L`; it sees `r_D + r_L` only through weak second-order traces.

**Empirical signature (measured).** At a (regularized) optimum the Hessian `H = ∇²F` is strongly
anisotropic, per species:

- **Soft** direction ≈ `θ_D + θ_L` ("turnover"): eigenvalue λ_min ≈ +0.015 (archaea-256, fp64),
  +0.031 (full archaea).
- **Stiff** direction ≈ `θ_D − θ_L` ("net growth"): eigenvalue λ_max ≈ 244.
- Condition number κ ≈ 1.6·10⁴ (256 fam) → ≈ 3.8·10⁴ (full).
- Posterior `corr(θ̂_D, θ̂_L) ≈ 0.93` (inverse-Fisher) — covariance ellipse stretched along the D=L diagonal.
- Transfer is ~decoupled from the pair.
- The soft eigenvector is ⊥ the gradient at every low-loss point (a genuine flat valley).
- λ_min > 0 but tiny: **practical** (near-) non-identifiability. With **no prior** (λ=0), rates run to the
  boundary (~38–47 % saturate) and the raw MLE is ill-posed.

---

## 3. Local algebra — why the eigenvectors are net / turnover

Per species, with `d = 2^{θ_D}`, `ℓ = 2^{θ_L}`, and coordinates `u = (θ_D+θ_L)/2` (turnover),
`w = (θ_D−θ_L)/2` (net): `d = 2^{u+w}`, `ℓ = 2^{u−w}`, so for the net quantity `n = d − ℓ`,

```
∂n/∂u = (ln2)(d − ℓ),     ∂n/∂w = (ln2)(d + ℓ).
```

When `d ≈ ℓ`, `|∂n/∂u| / |∂n/∂w| = |d−ℓ|/|d+ℓ| ≪ 1`. If the NLL depends locally mostly on `n = d−ℓ`,
`NLL ≈ φ(d−ℓ)`, then at a stationary point (`φ'=0`) the (u,w)-Hessian is `φ''·[[∂_u n, ∂_w n]]ᵀ[[…]]` —
**rank one**, stiff along `w = θ_D−θ_L`, null along `u = θ_D+θ_L`. Any small dependence on `d+ℓ` perturbs
the zero into a small positive eigenvalue. That is exactly the observed pattern.

---

## 4. The principled frame — pulled process, Louca–Pennell, and the exact-vs-weak criterion

**Exact Fisher-null criterion.** For a direction `v`, `vᵀI(θ)v = E_θ[(∂_v log P_θ(G))²]`, so `vᵀI(θ)v = 0`
iff `∂_v log P_θ(G) = 0` for almost every observable `G` — i.e. moving along `v` leaves the *entire*
observed-data law unchanged. For the latent complete reconciliation with event counts `N_X`,
`∂ log P_θ(R)/∂θ_X = (ln2)(N_X − N_• p_X)`; the observed score replaces `N_X` by its posterior expectation
given `G`. The turnover score depends on `E[N_D + N_L | G]`, dominated by events whose side branches die
(weakly constrained); the net score depends on `E[N_D − N_L | G]`, tightly constrained by copy-number
change and visible paralogy. This is the missing-information account of the Hessian.

**Pulled process (the Louca–Pennell analogue).** In the continuous-time D/L birth–death analogue, with
`q(t)` = probability a lineage leaves no sampled descendant, a duplication is *visible* only if the other
daughter also survives, so the reconstructed tree sees the **pulled** rate `d_pull = d·(1−q)`, not `d` and
`ℓ` separately. Given `d_pull(t)` and many choices of `q(t)`, one recovers admissible `(d, ℓ)` —
**infinitely many** raw pairs give the same pulled process. This is the exact birth–death non-identifiability
(Louca & Pennell, *Nature* 2020): extant timetrees identify only certain pulled combinations.

**Why the full DTL model can have small but positive curvature.** The species scaffold observes more than a
bare timetree (where paralogies map, which species lack a family, which losses reconciliation requires,
duplication-vs-transfer placement). A branching-process moment calculation shows the hierarchy: the first
moment `E[N_t] = e^{(d−ℓ)t}` depends only on net `d−ℓ`; the variance
`Var(N_t) = ((d+ℓ)/(d−ℓ)) e^{(d−ℓ)t}(e^{(d−ℓ)t}−1)` carries `d+ℓ` — but only at second order. So turnover is
present in the likelihood, weakly. **Predicted:** exact non-identifiability in the pruned/extant limit; weak
identifiability in the full scaffolded model.

---

## 5. The decisive measurement (this is the new part)

We separated **data** curvature from **prior** curvature on the turnover subspace, prior-free, using a
Laplacian-null trick: the **global turnover mode** `v_glob` (every species `δθ_D = δθ_L = +1`, `δθ_T = 0`)
is *constant across the tree*, hence in the null space of the tree Laplacian `L ⊗ I`. So `λ·L` contributes
**exactly zero** to its curvature — whatever curvature it has is **pure data information**. Evaluated at the
converged loose-box (rate ∈ [1e-4, 16]) λ=0.03 archaea-256 minimum, fp64:

| # | test | result | reading |
|---|------|--------|---------|
| **1** | global turnover `v_glob`, curvature | data **+2.6498**, data+prior **+2.6498** (prior contrib **−2.1e-14**) | prior is exactly Laplacian-null (machine precision); **data curvature is clearly nonzero** |
| **2** | global net `(δθ_D=+1, δθ_L=−1)`, data curvature | **+17.262** → net/turnover = **6.5×** | global anisotropy is only 6.5× — the κ≈10⁴ is **not** the global mode |
| **3** | family scaling of `v_glob` data curvature | m=64 → **0.689**, m=128 → **1.373**, m=256 → **2.650**; per-family **0.01077 / 0.01073 / 0.01035** | **linear in #families** → intrinsic Fisher information (additive over independent families), not a noise floor |
| **4** | turnover-subspace spectrum (U = per-species D=L basis, 119 cols) | `UᵀH_dataU`: λ_min **+0.01838**, λ_max +19.37, **0/119** below 1e-3; `UᵀH_FU`: λ_min **+0.04379** | **every** turnover direction has positive data curvature; the prior roughly **doubles** the softest (0.018 → 0.044) |

**Verdict: weak identifiability, not an exact null.** Turnover carries genuine, family-additive data
information; no turnover direction is null at the data level. (This overturned the a-priori guess — including
ours — that the global mode would be an exact null. It is not: it is data-stiff and prior-immune.)

---

## 6. Interpretation — global vs local, and the prior's real role

The family-scaling decomposition (test 3) is the key:

- **Global / shared turnover is data-determined.** Per-family curvature ≈ 0.0107, so at full archaea
  (5446 fam) the global turnover curvature extrapolates to ≈ **58** — *stiff*. The overall level of turnover
  is **not** the problem, and the prior is provably null on it.
- **Per-species turnover contrasts are the data-poor part.** Each species' turnover is informed only by the
  families occurring in it, so the *local* contrasts are thin — that is where λ_min = +0.018 sits (test 4),
  and where the prior roughly doubles the curvature (0.018 → 0.044).

So the GBM tree prior is **not** "fabricating global turnover information as if it were data." Global
turnover is data-determined; the prior is **null** on it. What the prior does is **borrow strength across
neighboring species for the thin per-species turnover contrasts** — legitimate hierarchical (GMRF) shrinkage,
and test 4 shows it lifts precisely the most-local, data-poorest directions. This is a more defensible
picture of the penalty than "a prior masquerading as data."

The honest one-line summary:

> **Turnover is weakly but genuinely identified; global turnover is fully data-determined and grows stiff
> with sample size; only per-species turnover contrasts are data-poor, and the tree prior shrinks exactly
> those. It is not the Louca–Pennell exact null — that is the pruned limit this model perturbs away from.**

---

## 7. Resolutions (what counts as principled)

The penalty resolves *conditioning*, not *identifiability* — it is a legitimate prior, but (for the local
contrasts) it is a prior, not data. Defensible options:

- **A. Report identifiable coordinates.** Fit/report the pulled/net quantities (net duplication signal,
  visible paralogy / missing-taxon / transfer-placement probabilities, the no-descendant probabilities
  `q_s`); treat raw turnover as nuisance. Cleanest frequentist resolution.
- **B. Reparametrize the prior (recommended, contained change).** Our GBM penalty currently smooths
  `(θ_D, θ_L, θ_T)` *independently*, i.e. it smooths the non-identified turnover direction. Better: change
  variables **before** penalizing — keep the well-identified imbalance `w_s = ½(θ_D−θ_L)` and `θ_T` under the
  tree-GBM prior (smoothing a data-determined coordinate is benign), and put an **explicit, interpretable**
  prior on the weak coordinate — the **excess turnover** `c_s = 2·min(r_D, r_L)` (the cancelling churn beyond
  net drift; vanishes when one rate is zero), e.g. `log c ~ GMRF on the species tree`. Penalty change only,
  not a likelihood change.
- **C. Structurally identifiable submodel** as a sensitivity bound: `c_s ≡ c₀` (global excess turnover),
  `ℓ_s = α·d_s` (fixed D/L ratio), or `c_s = 0` (minimal-churn/parsimony — mathematically clean, biologically
  strong; use as a bound, not a default).
- **D. Bayesian.** Report posterior sensitivity of turnover to the prior scale `π(c)`; if it moves materially
  with the prior, the data are not determining it.

Given the project's goal (fast, defensible DTL-rate estimation for reconciliation), **A + D** is the honest
headline and **B** is the concrete code improvement. Our existing CV-over-λ already gestures at D.

---

## 8. Decision rule (exact vs weak) and reproduce

Decisive diagnostics, in order of cost:
1. **Laplacian-null global-mode curvature** (test 1): `v_glob` is prior-immune; its data curvature is the
   prior-free turnover information. Nonzero ⇒ not exactly null. *(One HVP.)*
2. **Family scaling** (test 3): intrinsic Fisher info is linear in #families; a prior-lift / noise floor is
   not. *(One HVP per family count.)*
3. **Subspace split** (test 4): `λ_min(UᵀH_NLLU)` vs `λ_min(UᵀH_FU)` separates data from prior on the whole
   turnover subspace. *(S HVPs each.)*
4. (Not run) **KL simulation**: simulate two settings with equal net/pulled but different excess turnover
   `c`; per-family `KL` zero ⇒ exact equivalence, tiny positive ⇒ weak.

Script: `experiments/sanderson_cv/dl_identifiability.py` (build the model, load a converged θ, build
`H_data` via `make_lap(...,0.0)` and `H_F` via `make_lap(...,λ)` HVP operators, evaluate the quadratic forms
and the turnover-subspace `UᵀHU` spectrum). Run with `SADDLE_DTYPE=float64`, `PYTHONPATH=<worktree>`,
`GPUREC_PREPROCESS_PATH=...`. Converged points used: `runs/cv_archaea_n256_box_1e-4_16/refit_lam0.03_fp64_converged.pt`.

**One caveat to keep stated:** the GBM penalty `Σ‖θ_c−θ_p‖²` does *not* penalize a globally constant shift of
all `θ_D` (and all `θ_L`) — that mode is in the Laplacian null space. So a positive eigenvalue on a
*near-constant* turnover direction must come from data, while a positive eigenvalue on a *local-contrast*
turnover direction can be prior-induced. The measurements above use exactly this to separate the two.

# Receiver-weight (alpha) curvature consumers — joint PD certificate, Fisher/uncertainty, Newton on alpha (S9)

Status: IMPLEMENTED + verified. This is the CONSUMER side of the analytic joint `(theta, alpha)`
exact HVP (`receiver_weights_hvp_plan.md` / PR #4): the HVP exists and is FD-gated, and this layer
turns it into the things the HVP was *for* — a positive-definiteness certificate for the joint
minimum, the Fisher information / standard errors of the receiver weights, and Newton steps on
`alpha`.

Code: `gpurec/optim/receiver_curvature.py`. Gate: `gpurec/optim/_verify_s9_curvature.py`.
`newton_lanczos(with_receiver=True)` now delegates here (it previously raised `NotImplementedError`).

---

## 0. The one invariant everything hangs on: the receiver gauge

`alpha` (the per-species receiver logits, `R^S`) enters the model ONLY through a full unpinned
`log_softmax`:

    receiver_log_probs = log_softmax(alpha) / ln2          (extract_parameters.py:37-38, :81)

so the recipient distribution is `w = softmax(alpha)` (NATURAL softmax: `2^{receiver_log_probs} =
softmax(alpha)`) and the NLL is EXACTLY invariant under `alpha -> alpha + c·1`. Consequences for the
joint Hessian `H` of `z = [theta.reshape(-1); alpha]`:

- `1_S^T (dNLL/dalpha) = 0` (already true of the wired gradient, PR #3),
- `H_aa · 1_S = 0`, `H_ta · 1_S = 0`, `1_S^T H_at = 0` — **the all-ones receiver mode `[0; 1_S]` is
  an EXACT zero eigenvalue / null direction of `H`.**

Any consumer that ignores this sees a singular operator: Lanczos returns `lam_min = 0`, the Newton
system has a null direction, the Fisher information is rank-deficient. Every consumer here works in
the GAUGE-FIXED subspace via the block projector

    P_z = blockdiag( I_theta , I_S − (1/S) 1_S 1_S^T )     (identity on theta, mean-subtract on alpha)

`proj_z` / `proj_alpha` in `receiver_curvature.py`. The gauge-projected operator is

    A_z(v) = P_z · H(P_z v)                                 (make_gauge_operator)

symmetric (verified: `aᵀ A_z b = bᵀ A_z a`), with `A_z [0; 1_S] = 0` exactly (the input projection
annihilates the gauge null), and equal to the true reduced Hessian on the gauge-fixed subspace.

---

## 1. PD certificate — `certify_joint_min`

Gauge-projected Lanczos for the smallest reduced-Hessian eigenpair. **The subtlety:** `A_z` has a
genuine zero eigenvalue (the gauge null), and finite-precision Lanczos seeking the MINIMUM is drawn
to it — roundoff leaks the `1_S` mode back into the Krylov basis even from a `P_z`-projected start,
and `0 < lam_min_reduced` wins. The fix is **spectral deflation**: shift the null direction's
eigenvalue up to `C = 2·lam_max` via `A_z(v) + C·(v − P_z v)` (= `C` on `e_null`, unchanged on the
gauge-fixed subspace). The smallest eigenvalue of the deflated operator IS the reduced-Hessian
minimum, and its Ritz vector comes out gauge-fixed (so the Ritz residual, measured vs the UNSHIFTED
`A_z`, is the true reduced residual).

Returns `lam_min_gauge` (PD iff `> 0`), `ritz_resid`, `leak` (raw-HVP gauge leak
`|1ᵀ(H v_min)_a|/√S` — at the truncation floor confirms the operator is genuinely gauge-respecting,
not just force-projected), `gauge_comp` (`‖v_min − P_z v_min‖`, ~0 confirms the eigenvector is
gauge-fixed), `pd`, `v_min`. `lam_min_gauge > 0` with small `ritz_resid` ⟹ **certified gauge-fixed
joint `(theta, alpha)` minimum.**

## 2. Fisher information / uncertainty — `receiver_information`

At a local min the observed Fisher information of `z` is the joint Hessian `H`; the MLE covariance is
its gauge-fixed inverse `(P_z H P_z)⁺`. The alpha-MARGINAL covariance is computed Schur-complement-
correct (it accounts for the theta coupling, because each column is a FULL joint solve):

    Sigma_aa[:, j] = [ (P_z H P_z)⁺ · P_z e_{alpha_j} ]_alpha        (CG, one solve per species j)

solved by `cg_solve` on the gauge-projected operator (no deflation needed — the RHS is in the
gauge-fixed range, where `A_z` is PD at a certified min, so CG never excites the null). Returns:

- `Sigma_aa` — gauge-fixed marginal covariance (symmetric, mean-zero rows/cols),
- `se_alpha = sqrt(diag(Sigma_aa))` — per-species s.e. of the receiver logits,
- `se_w` — delta-method s.e. of the recipient probabilities `w = softmax(alpha)`, via
  `Sigma_w = J Sigma_aa Jᵀ`, `J = diag(w) − w wᵀ` (the softmax Jacobian; `J 1 = 0` so `Sigma_w` is
  automatically gauge-consistent). NO `ln2` factor — `w` is the natural softmax.

`species=` subsets which receiver coords to profile (each solve ≈ `cg_iters` joint HVP applies; the
full marginal is `S` solves). **This is the receiver-weight identifiability readout for the paper:
large `se_w[i]` ⟺ the data does not pin which species `i` receives transfers.**

## 3. Newton on (theta, alpha) — `newton_joint`

Gauge-projected LM-damped Newton on `z`. The step solves `P_z (H + penalty + lam_damp I) P_z dz =
−P_z g_z` with `cg_witness` (negative-curvature self-correction bumps `lam_damp`); steps are
globalized by Armijo backtracking on the joint forward loss `F = NLL + penalties(theta)`; after each
accepted step the alpha block is re-centered to the gauge slice. Optional ridge (`lam`/`theta_ref`)
and GBM tree-Laplacian (`lam_tree`/`sp_parent`) penalties act on the THETA block only (the receiver
block is penalty-free, matching `make_value_and_grad(optimize_receiver=True)`); their closed-form
Hessian is added to the operator so the Newton model is exact. The joint analytic HVP is rebuilt at
each accepted point (theta fixed across a point's CG iterations → the per-point cache amortizes).
Requires a NON-uniform `alpha0` (at a uniform base the receiver paths are dead). Run in fp64.
`newton_lanczos(static, theta0, receiver_weights, with_receiver=True, alpha0=...)` delegates here and
returns `(theta, alpha, history)`.

---

## 4. Verification (`_verify_s9_curvature.py`)

**A — synthetic, dense ground truth (CPU, machine precision).** Inject an exact gauge null into a
random symmetric matrix `M = P_z B P_z` (`B` SPD ⟹ the gauge-fixed reduced block is PD with one
exact-zero gauge eigenvalue). Versus dense `eigh` / `pinv`:

| consumer | result |
|---|---|
| `certify_joint_min` lam_min vs `eig(M)[1]` | rel_err **3.5e-16**, Ritz **1.8e-15**, PD ✓ |
| gauge null `‖A_z[0;1_S]‖` | **0.0** (exact) |
| `receiver_information` Sigma_aa vs `pinv(M)` alpha-block | rel **2.1e-13**, symmetry **0.0**, CG resid **8e-13** |

This validates the consumer linear algebra (deflated projected Lanczos + projected CG + Schur
marginal) independent of the model.

**B — live hogenom-8 (S=1331, p=5324), real analytic joint HVP, fp64 converged solver.** From a
30-step joint first-order warmup:

| check | result |
|---|---|
| `newton_joint` (Newton on α) | F 275.95 → 258.25, ‖P g‖ 2.50 → **0.385** (6.5×), monotone, escapes neg-curv via the witness ✓ |
| `certify_joint_min` | Ritz resid **2.8e-5**, gauge leak **5.3e-18**, gauge_comp **1.9e-10**; lam_min **−0.031** (NOT PD) |
| `receiver_information` (ridge 1.0) | CG converges 8 iters, resid **6e-8**, cross-cov symmetry **1e-11**, `se_alpha=1.00` ✓ |

The operator certificates (Ritz, leak, gauge-fixed eigenvector) are perfect on the real HVP. The
NEGATIVE `lam_min` and `se_alpha = 1.00 = 1/√ridge` are the SCIENTIFIC readout, not a defect: **8
families cannot identify 1331 receiver logits** (most species never receive a transfer), so the
gauge-fixed `H_aa` is near-singular with shallow negative curvature — the documented non-identifiable
landscape. The Fisher correctly reports "no information from the data" (the s.e. is set entirely by
the ridge). A meaningful identifiability readout needs many families (full archaea: 5446) where most
species actually receive transfers; the machinery is the same.

**C — primates (S=25) dense cross-check.** Skipped: the primates gene trees use Ensembl IDs that need
a `treerecs_mapping.link` gene→species map, which `GeneReconModel` does not ingest. The dense
ground-truth role is fully covered by part A (consumer correctness vs dense `eigh`/`pinv` to machine
precision); the live part B covers real-operator wiring.

Run: `python -m gates._verify_s9_curvature` (`--synthetic` = A only, no GPU; `--live` = B only).

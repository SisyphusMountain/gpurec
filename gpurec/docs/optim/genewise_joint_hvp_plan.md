# Genewise analytic HVP: per-family DTL `[G,3]` + per-family origination `[G,S]`

**Status:** design / plan (not yet implemented). 2026-07-06.
**Depends on:** the committed survival/receiver normalizer fix (`b6faad9b`) — the origination head's
`1/survival²` curvature is only trustworthy on the hardened forward-adjoint.

## 1. Goal

Provide an **analytic** Hessian / HVP for genewise fits over the joint parameter
`z_g = (θ_g ∈ ℝ³, ω_g ∈ ℝ^S)` per gene family `g ∈ {1..G}`:

- `θ` — per-family DTL logits, shape `[G, 3]` (the existing genewise rate tensor).
- `ω` — per-family origination logits, shape `[G, S]` (NEW; today origination is a single global `[S]`).

so genewise can run **Newton / Newton-CG** over DTL **and** per-family origination together, and so the
convergence certificate uses an exact (not finite-difference) curvature.

Non-goals (separate work): global receiver weights `α [S]` or origination `ω [S]` (the *arrowhead* case,
needs multi-batch accumulation + a `2S` Schur solve); per-family receiver weights `[G,S]` (α enters the
fixed points, so it is not a cheap head like ω — different cost profile).

## 2. Why this is the clean case: full block-diagonality

Both `θ_g` and `ω_g` are **per-family**, and family `g`'s loss depends only on `(θ_g, ω_g)`. There is **no
cross-family coupling**, so the joint Hessian is **block-diagonal** with `G` independent blocks, each
`(3+S)×(3+S)`:

```
family g:   [ H_θθ  (3×3) | H_θω  (3×S) ]
            [ H_ωθ  (S×3) | H_ωω  (S×S) ]
```

Consequences:
- **Newton is per-family and embarrassingly parallel** — no global reduction, no arrowhead, and (unlike the
  global-α/ω case) **no multi-batch accumulation** is required. `θ` stays per-batch/per-family independent.
- The `3` broadcast probes (a tangent `u[g] = e_j` for all families at once) recover column `j` of **every**
  family's block simultaneously — because the Hessian is block-diagonal. (This is exactly the property the
  failed reuse gate exposed the *lack of* for the θ-projection; see §4.)

## 3. Why the `(3+S)` block is cheap: origination is a *head* parameter

`ω_g` enters **only** the NLL aggregation, never the E / Pi-wave fixed-point solve
(`origination_grad_from_root_rows`, `solver.py:184-199`, and the weighted NLL `nll_vector_from_root_rows`,
`solver.py:158-169`):

```
NLL_g = -( logsumexp2(root_rows_g + log_softmax2(ω_g)) - log2(survival_g) )
survival_g = 1 - Σ_s softmax2(ω_g)_s · 2^{E_g,s}          # weighted survival (now cancellation-free)
```

That splits the block by cost:

| sub-block | derivative through | cost |
|-----------|--------------------|------|
| **H_θθ** `3×3` | E & Pi fixed points | **3 θ forward-over-reverse sweeps** (same as θ-only genewise) |
| **H_θω** `3×S` | θ-tangent of `(root_rows,E)` × head cross-partial | **falls out of the same 3 sweeps** (no extra) |
| **H_ωω** `S×S` | head only (`root_rows_g, E_g` held fixed) | **closed-form, no sweeps**; **diagonal + low-rank** |

`H_ωω` structure (both pieces are `ω`→softmax curvature):
- origination-prior term `log2 Σ_s 2^{root_rows_s} p_s` (`p=softmax2(ω)`) = `LSE(ω+r) − LSE(ω)`, a
  difference of two log-partition Hessians ⇒ `(diag(q)−qqᵀ) − (diag(p)−ppᵀ)` (`q` = reweighted softmax);
  diagonal + rank-2.
- survival term `−log2(1 − Σ_s p_s 2^{E_s})` → softmax-Jacobian (`diag(p)−p pᵀ`) scaled by `1/survival`,
  plus a rank-1 `∝ 1/survival²`. The `1/survival²` is now accurate because survival is computed
  cancellation-free (`survival_from_E`, committed in `b6faad9b`).
- **Sum = diagonal + low-rank (rank O(1))** ⇒ inverts in `O(S·rank)` via Woodbury, not `O(S³)`.

**Net cost:** the whole `[G,3]+[G,S]` analytic Hessian ≈ **3 broadcast θ sweeps + a cheap per-family
diag+low-rank ω head**. The large `S` of origination costs **no extra sweeps** — the payoff of it being a
head parameter. A joint HVP probe `[u_θ; u_ω]` is likewise **~one sweep**: `u_θ` drives the fixed-point
tangent, `u_ω` only touches the head.

## 4. What the feasibility gate already told us

Ran the existing HVP oracle (`_verify_hvp.py` pattern: analytic vs fp64 central-diff + symmetry) on a
2-identical-family fp64 converged model:

- **specieswise `(S,3)`: PASS** (`rel ~5e-7`) — the forward-over-reverse machinery and the SO/tangent
  kernels are sound; the committed normalizer edit to `hvp_exact.py` did not regress it.
- **genewise `(F,3)`: FAIL** — analytic HVP finite but `~400×` too large **and non-symmetric**
  (`rel_asym ~5e2`). Non-symmetric ⇒ it is not `H·u` for any Hessian ⇒ wrong operator, not a scale bug.

Diagnosis: `make_exact_hvp`'s **parameter-projection heads** assume `θ[s] ↔ species s`. Genewise `θ[g]`
broadcasts one 3-vector across all `S` species, so the θ↔rates forward-tangent/reverse-sum is mishandled.
The **core `C×S` propagation kernels are correct and reused unchanged**; only the parameter seed/projection
is wrong. This is the single thing Phase 0 fixes.

## 5. Components to build

The expensive `C×S` propagation kernels (`wave_tangent`, `e_step_tangent`, `wave_so`, `dts_so`,
`e_step_so`) are **reused as-is**. New work:

### P0 — Genewise θ seed/projection  *(critical path; also delivers the θ-only certificate)*
- **Seed (forward):** `d(extract_parameters_genewise)/dθ_g · u_θ`. `θ_g [3] → (log_pS,log_pD,log_pL,log_pT)`
  via `log_softmax([0, θ_g])` (`extract_parameters.py:30-38`), broadcast to `S` species. Push `u_θ [G,3]`
  through this per-family log-softmax Jacobian to per-species rate tangents.
- **Projection (reverse):** **sum** the per-species rate-cotangents `[G, S, ·]` back into `[G, 3]` (the
  transpose of the broadcast). The current `(S,3)` path keeps them per-species — that is the ~400×/asym bug.
- Deliverable: genewise **θ-only** block-diagonal `H_θθ` (`G × 3×3`) via 3 broadcast probes; passes the gate.
  This is Part 2 (the analytic certificate) on its own.

### P1 — Per-family origination in the model + NLL
- Model: origination parameter `[G, S]` (today global `[S]`, `model.py:79`). Per-family selection into each
  family's aggregation (mirror `theta_for_static`'s `index_select`, `_execution.py:18-19`).
- NLL: the weighted branch already exists (`nll_vector_from_root_rows`, `solver.py:165-169`); feed each
  family its own `ω_g` row. Gradient: `origination_grad_from_root_rows` (`solver.py:184-199`), per-family.

### P2 — Per-family ω head Hessian + θω coupling  ⇒ full `(3+S)` block
- `H_ωω` closed-form per family from `(root_rows_g, E_g)` (already computed by the forward). Adapt
  `_head_seed_tangents` (`hvp_exact.py:81-104`), which already does the origination head double-backward for a
  **global** ω, to be **per-family `[G,S]`** (do not sum to `[S]`). Expose the diag+low-rank form (or apply
  it matrix-free through the autograd head).
- `H_θω` = θ-sweep JVP of `(root_rows, E)` contracted with the head's `∂²/∂(root_rows,E)∂ω` — reuse the 3 θ
  sweeps from P0.

### P3 — Assembly + per-family Newton
- Form per-family `(3+S)` blocks, or expose a matrix-free per-family joint HVP `hvp(u; active⊆{θ,ω})`.
- **Per-family Newton** via Schur + Woodbury: `H_ωω = D + low-rank` inverts in `O(S·rank)`; reduce to a `3×3`
  θ system `(H_θθ − H_θω H_ωω⁻¹ H_ωθ) δ_θ = −(g_θ − H_θω H_ωω⁻¹ g_ω)`, back-substitute `δ_ω`. All families in
  parallel. (Or per-family Newton-CG if not forming blocks.)
- Wire into `genewise_fit.py`: replace the FD certificate Hessian (`:264-270`) with the analytic block; the
  inner-loop FD step (`:229-236`) may stay FD or switch — decide after measuring.

### P4 — Optimize (optional)
- Batch the width-3 direction axis into the tangent/SO kernels ⇒ the whole `H_θθ`/`H_θω` in **one** batched
  forward-over-reverse instead of 3.
- Exploit diag+low-rank `H_ωω` in the solve (Woodbury) rather than dense.

## 6. Interfaces (proposed)

```python
# core, per single genewise static (block-diagonal across its families):
genewise_hvp(static, theta, omega, sv, *, active=("theta","omega")) -> callable  # hvp(u_flat) -> H u
genewise_hessian_blocks(static, theta, omega, sv) -> dict(
    H_tt=[F,3,3], H_to=[F,3,S], H_oo_diag=[F,S], H_oo_lowrank=[F,S,r])  # structured, not dense S×S
newton_step_genewise(blocks, g_theta, g_omega, mu) -> (dtheta[F,3], domega[F,S])  # Schur + Woodbury
```

## 7. Verification

- **Gate (oracle):** extend `_verify_hvp.py` to the genewise `(3+S)` block — analytic `hvp(u)` vs fp64
  central-difference of the joint value-and-grad, for broadcast `e_j` (θ and ω) and random directions, plus
  the symmetry check `uᵀHw == wᵀHu`. Acceptance: `rel < 5e-4`, symmetric, on a ≥2-family fp64 converged model.
  The FAIL→PASS transition on this gate is the P0/P2 acceptance criterion.
- **Golden test** in the suite (fp32-vs-fp64 on a small fixture), alongside `tests/test_survival_normalizer.py`.
- **End-to-end:** genewise fit with active origination completes with finite curvature and a PD certificate.

## 8. Risks / open questions

- **Identifiability (modeling):** `[G,S]` origination = `S` params/family from one family's likelihood ⇒
  very likely under-determined (the low-rank `H_ωω` is the symptom). Almost certainly needs a **prior /
  regularizer** on `ω_g`, which also conditions the Newton block. This is a modeling decision to settle
  before turning it on in a real fit; the HVP/Newton machinery is agnostic to it.
- **P0 is the critical path** and is shared with the θ-only certificate — build and gate it first, in
  isolation, before the ω pieces.
- **Head double-backward must use the hardened survival** — satisfied by `b6faad9b` (survival + weighted
  survival are cancellation-free); the `1/survival²` in `H_ωω` is now accurate.
- **Truncation consistency:** the HVP must match the primal forward's `pi_iters` truncation
  (`tangent_self_iters`, `hvp_exact.py:122-143`) or the gate will show a small bias unrelated to the port.

## 9. Phasing summary

| Phase | Deliverable | Gate |
|-------|-------------|------|
| P0 | genewise θ seed/projection → `H_θθ` block-diagonal `[G,3×3]` (the analytic certificate) | θ gate PASS |
| P1 | per-family origination `[G,S]` in model + weighted NLL | grad matches FD |
| P2 | per-family `H_ωω` (diag+low-rank) + `H_θω` ⇒ full `(3+S)` block | joint gate PASS |
| P3 | per-family Newton (Schur+Woodbury) wired into fit/cert | PD cert, finite e2e |
| P4 | batch θ directions to 1 sweep; Woodbury solve | perf, no accuracy change |

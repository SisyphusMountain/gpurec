# Genewise joint analytic HVP: DTL `θ[G,3]` + origination `ω[G,S]` + receiver weights `α[S]`

**Status:** P0 (θ) done + committed (`a6322953`, 2026-07-06). P1–P4 (ω, α arrowhead, Newton) designed
here. Supersedes the earlier θ+ω-only version of this doc (α was previously a non-goal; it is now in
scope as the global arrowhead).

## 1. Goal

One analytic (forward-over-reverse) HVP over the **joint** parameter

```
z = [ θ (G×3)  |  ω (G×S)  |  α (S) ]        dim = 3G + GS + S   (parameter groups)
# FLAT HVP / gate-vector order is [θ; α; ω] (α before ω) — matches the established
# origination_curvature.py convention (z=[theta;alpha;omega]); newton_step_joint is order-agnostic.
```

for genewise fits, so genewise can run **Newton** over any subset of {per-family DTL `θ`, per-family
origination `ω`, global receiver weights `α`}.

- `θ` — per-family DTL logits `[G,3]` (existing genewise rates).
- `ω` — per-family origination logits `[G,S]` (NEW; today origination is a single global `[S]`).
- `α` — global receiver (transfer-recipient) weights `[S]` (existing; the current code's `receiver_log_probs`).

**Design axioms:**
- The HVP is a **single operator** `hvp(u) = H·u`. The Hessian's block structure is used **only** by
  the Newton *solve*, never to compute `H·u` (see §4).
- **Prior-agnostic.** Any regularizer/prior on `ω` (or damping) is the caller's — added as a diagonal
  on `H_ωω` or via the Newton damping `μ`. We build no prior.

## 2. What P0 actually was (correcting the earlier diagnosis)

P0 (genewise `H_θθ`, block-diagonal `[G,3,3]`) is green and committed. The **earlier** version of this
plan mis-diagnosed the P0 bug as a wrong "θ[s]↔species s" seed/projection. That was wrong. The seed
(`param_jvp`) and reverse projection were already correct for genewise. The real bugs were two
**per-family/per-species plumbing** errors:

1. **Head contraction** (`hvp_exact.py`): mixed the already-per-family `[G,1]` DTS cotangent
   (`acc["grad_log_pS/pD"]`) with the per-species `[G,S]` e-step cotangent (`base_p[.]`). The `[G,1]`
   then broadcast over `S` and the head contraction `(pS_hp[G,1]·cot).sum()` summed the species axis →
   an `S×` overcount. Fixed by summing the e-step term to `[G,1]` first, gated on `static.genewise`.
2. **`dts_backward_so`** (`dts_so.py`): the split SO kernel writes `log_pS/pD` cotangents per-species
   via `d_grad_p*_ptr + item*S + s` (a `[rows,S]` layout). Genewise passes a species-reduced `[G,1]`
   buffer → out-of-bounds writes (only `s=0/1` land, the rest corrupt adjacent pool memory, poisoning
   `d_grad_mt` too). Fixed with a `[rows,S]` scratch reduced back to `[G,1]`.

Neither was cross-term math; both were reductions. **Lesson for P1–P4:** the risk in the joint sweep is
per-family/per-species *reductions*, not derivations. Every new gate exists to catch that class.

## 3. The joint Hessian is an arrowhead

`ω` never enters the fixed-point solve (it is a *head* parameter — §5); `θ` and `α` do. `θ,ω` are
per-family; `α` is global. So `H` is **arrowhead**:

```
          θ_g (3)     ω_g (S)   │   α (S)
θ_g   [  H_θθ,g      H_θω,g    │  H_θα,g  ]   ┐ per-family block B_g,
ω_g   [  H_ωθ,g      H_ωω,g    │  H_ωα,g  ]   ┘ block-diagonal in g   (each (3+S)²)
──────────────────────────────────────────────
α     [ Σ_g H_αθ,g  Σ_g H_αω,g │  H_αα    ]   ← global arrow (dense S×S), sums over families
```

- **Core:** `G` independent blocks `B_g = [[H_θθ,g, H_θω,g],[H_ωθ,g, H_ωω,g]]`, each `(3+S)×(3+S)`.
- **Arrow:** global `α` with dense `H_αα` and couplings `H_zα,g = [H_θα,g; H_ωα,g]`.

## 4. Principle: one sweep computes `H·u`; blocks are only for the solve

`H` is the Jacobian of the gradient map `g(z)`. `H·u` is the directional derivative of that map:
linearize the **entire** gradient computation along `u`. That computation has a shared middle —
`θ,α → (E, Pi, root_rows) → NLL → adjoint → g`. A cross term like `H_θα` **is** `∂(g_θ)/∂α` routed
through `dE, dPi`; it lives inside that shared intermediate and has no independent existence.

Therefore:
- **One** forward-over-reverse sweep seeded with `u = [u_θ; u_α; u_ω]` returns
  `dg = [dg_θ; dg_ω; dg_α] = H·u`, with **all** cross terms — `dg_θ = H_θθu_θ + H_θωu_ω + H_θαu_α`,
  automatically, because every input direction flows through the same `E/Pi/root/adjoint`.
- **Isolated per-block operators, summed, cannot produce cross terms** (they'd give block-diagonal `H`).
  So we build the joint operator, not blocks.
- The blocks are materialized **only** for the Newton solve (§7), as a cheap way to invert the `H` we
  already have — never to compute `H·u`.

`hvp(u_vec)` in `hvp_exact.py` already realizes this for `θ+α` (returns `[out_θ; out_α]` from one
sweep, validated by `_verify_hvp_recv`). The joint work is to make `ω` a proper per-family in/out of the
same operator.

## 5. Per-parameter treatment

| param | enters fixed point? | how its Hessian rows/cols are produced | status |
|-------|--------------------|-----------------------------------------|--------|
| `θ[G,3]` | yes | 3 broadcast θ-tangent sweeps (forward-over-reverse) | ✅ P0 done |
| `ω[G,S]` | **no** (head only) | autograd double-backward over `head(root_rows, E, ω)` | ⚠️ head exists but ω is global `[S]` |
| `α[S]` | yes | α-tangent sweep (already emits `out_col`) | ✅ exists, validated **with specieswise θ** |

- **ω is head-only.** It touches only `NLL = head(root_rows(θ,α), E(θ,α), ω)`
  (`nll_vector_from_root_rows`). `_head_seed_tangents` already forms `⟨∇NLL, [t_root; dE; u_ω]⟩` and
  double-backwards it — `Hv_om` already contains `H_ωω u_ω + H_ωθ(·) + H_ωα(·)`. The **only** change is
  to keep `ω` per-family `[G,S]` instead of summing to `[S]`. Then all three ω-blocks appear on their own.
- **α×genewise-θ is untested.** The α path is validated only with specieswise θ. Given P0's genewise
  reduction bugs, the genewise θ×α coupling needs its own gate before we trust `H_θα`.

## 6. `H_ωω` structure — diagonal + low-rank (for Woodbury)

`H_ωω,g` is the head Hessian of two softmax/log-partition terms in `ω_g`:
- origination-prior `LSE(ω+r) − LSE(ω)` ⇒ `(diag(q)−qqᵀ) − (diag(p)−ppᵀ)` (`p=softmax2(ω)`,
  `q`=reweighted) — **diagonal + rank-2**.
- survival `−log2(1 − Σ_s p_s 2^{E_s})` ⇒ softmax-Jacobian `(diag(p)−ppᵀ)` scaled by `1/survival`,
  plus a rank-1 `∝ 1/survival²` (survival is cancellation-free since `b6faad9b`) — **diagonal + rank-1**.

Sum = **diagonal + low-rank (rank O(1))** ⇒ inverts in `O(S·rank)` via Woodbury, not `O(S³)`. The
caller's ω prior (if any) adds to the diagonal — still diag+low-rank.

## 7. Arrowhead Newton solve

Solve `H δ = −g` for `δ = [δ_θ; δ_ω; δ_α]`:
1. Per family, invert `B_g = [[H_θθ,g, H_θω,g],[·, H_ωω,g]]` using Woodbury on `H_ωω,g` (diag+low-rank);
   `H_θθ,g` is `3×3`. All families in parallel.
2. **Schur-complement** the core out → a **dense `S×S`** system in `α`:
   `(H_αα − Σ_g H_αz,g B_g⁻¹ H_zα,g) δ_α = −(g_α − Σ_g H_αz,g B_g⁻¹ g_z,g)`. Solve (dense `S×S`).
3. Back-substitute `δ_z,g = B_g⁻¹(−g_z,g − H_zα,g δ_α)` per family.

Conditioning/PD is the caller's `μ` (added to `H_ωω`/`H_αα`) or their ω prior. A **matrix-free
Newton-CG** on the full `hvp` is the cross-check oracle for this structured solve (they must agree).

## 8. Batching

Scope this on the **single collated batch** (all families up to `family_chunk_size`, matching what the
θ HVP reaches today; both HVP gates already require `len(batch_statics)==1`). Note: the gradient
aggregates over batches (`make_value_and_grad` loops), but the exact HVP is single-batch for θ too — so
multi-batch is not α-specific.

**Multi-batch is a uniform follow-on, not a special case:** `θ,ω` are block-diagonal → concatenate
across batches; `α` is the only cross-batch coupling → accumulate its Schur pieces (`α`-row/col +
`H_αα` + `Σ_g H_αz B⁻¹ H_zα`) over batches before the dense `S×S` solve. Same code path for all three.

## 9. Prior — out of scope

`ω[G,S]` is `S` params/family from one family's likelihood ⇒ under-determined (the low-rank `H_ωω` is
the symptom). A prior/regularizer is a **modeling** decision left to the caller; the machinery is
agnostic and only requires that whatever the caller adds is diagonal-on-`H_ωω` (keeps §6/§7 valid).

## 10. Verification

- **Genewise joint gate** (`tests/test_genewise_hvp.py`): analytic `hvp([u_θ; u_ω; u_α])` vs fp64 FD of
  the joint value-and-grad, for broadcast `e_j` (θ), broadcast/random `e_k` (ω), random `u_α`, and mixed
  directions; plus the symmetry check `uᵀHw == wᵀHu`. Accept `rel < 5e-4`, `rel_asym < 5e-3`, on a
  ≥2-family fp64 converged model. Extends the P0 θ gate.
- **α×genewise-θ** gets its own directions in that gate (the untested coupling, §5).
- **Newton cross-check**: structured Schur/Woodbury `δ` vs matrix-free Newton-CG on the same `hvp`.
- **Golden** fp32-vs-fp64 blocks test alongside `tests/test_survival_normalizer.py`.
- **Truncation:** `tangent_self_iters == solver_options.pi_iters` (else a bias unrelated to the port).

## 11. Phasing

| Phase | Deliverable | Gate |
|-------|-------------|------|
| P0 | genewise `H_θθ` block-diagonal `[G,3,3]` | ✅ θ gate PASS (done) |
| P1 | per-family origination `ω[G,S]` in model + weighted NLL + grad | grad vs FD |
| P2 | `ω` per-family in the HVP head ⇒ `H_ωω, H_ωθ, H_ωα` (auto) | joint θ+ω gate PASS |
| P3 | genewise θ×α gate (verify/fix the existing α path under genewise θ) ⇒ full `H·u` | joint θ+ω+α gate PASS |
| P4 | arrowhead `newton_step` (Schur+Woodbury+dense `S×S`) + wire into genewise fit + PD cert | Newton vs CG; e2e finite/PD |
| (P5) | multi-batch α accumulation (uniform outer loop) | joint gate on >1 batch |

## 12. Interfaces (proposed)

```python
# one joint HVP operator (single collated batch); u = [u_θ(3G); u_α(S); u_ω(GS)] (any tail omitted ⇒ 0)
# NOTE: flat order is [θ; α; ω] (α before ω) — matches the established origination_curvature.py convention.
make_exact_hvp(static, theta, alpha, sv, *, omega=..., ...) -> hvp(u_flat) -> H u
# structured curvature for the Newton solve (materialized only here, not for H·u):
genewise_hessian_blocks(static, theta, omega, alpha, sv) -> dict(
    H_tt=[G,3,3], H_to=[G,3,S], H_oo_diag=[G,S], H_oo_lr=[G,S,r],   # per-family core
    H_aa=..., H_za=[G,3+S,S])                                        # global arrow
newton_step_joint(blocks, g_theta[G,3], g_omega[G,S], g_alpha[S], mu) -> (dθ[G,3], dω[G,S], dα[S])
```

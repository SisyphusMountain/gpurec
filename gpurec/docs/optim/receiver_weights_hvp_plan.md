# Receiver-weight (alpha) exact (theta, alpha) HVP — implementation-grade plan

Status: IMPLEMENTED. This historical plan is retained for the design rationale and the
step-by-step validation strategy. It describes the second-order extension of the analytic
exact-Hessian HVP from theta-only to the joint variable
`z = [theta.reshape(-1); alpha]` (length `3S + S = 4S`), where `alpha` is the vector of
receiver logits in `R^S`.

All file:line anchors refer to the pre-implementation `receiver-weights-hvp` worktree as read on
2026-06-21. Live interface names use the current equation-aligned vocabulary; some illustrative
local names and all line numbers remain historical and are not expected to match the current
sources.

---

## 0. The variable, the gauge, and the one invariant everything hangs on

- `theta` is `[F,3]` (genewise) or `[S,3]` (specieswise); its flat length is `theta.numel()`.
  `alpha` is ALWAYS `R^S` with `S = receiver_weights.numel()` — keep S EXPLICIT, never derive
  it from `len(theta)` (genewise has `F != S`).
- The joint HVP operates on `u = [u_theta (3S or 3F); u_alpha (S)]`, returns `H u` of the same
  shape. The four Hessian blocks are `H_tt` (already implemented, `hvp_exact.py`), `H_ta`,
  `H_at`, `H_aa`.
- **GAUGE (the load-bearing fact).** `alpha` enters the model ONLY through a full unpinned
  `log_softmax`: `receiver_log_probs = log_softmax(alpha)/ln2`
  (`extract_parameters.py:37-38`, `:81`). Therefore NLL is EXACTLY invariant under
  `alpha -> alpha + c*1`, so:
  - `1^T (dNLL/dalpha) == 0` by construction (already true of the wired gradient), and
  - the alpha–alpha Hessian block `H_aa` has an EXACT zero eigenvalue along `1_S`, and
    `H_ta 1_S = 0`, `1_S^T H_at = 0`.
- **DECISION (carried from the gradient run, applied identically here).** Keep alpha symmetric in
  `R^S`. Handle the gauge by PROJECTING OUT the all-ones mode with
  `P = I - (1/S) 11^T` applied to the ALPHA SUB-BLOCK only. Define the block projector
  `P_z = blockdiag(I_{3S}, P)` (identity on theta, mean-subtract on the alpha block).
- **CORRECTION (1) — apply P_z to BOTH sides of EVERY HVP comparison and to the CG/Lanczos
  operator.** Because `H_aa` has an exact null vector `1_S`:
  - The raw analytic HVP and the raw FD HVP each carry an UNCONTROLLED component along
    `1_S` (analytic: should be ~0 up to truncation; FD: same, but FD noise lands there too).
    Comparing raw-vs-raw can FALSE-PASS (a real bug hidden entirely in the null space cancels
    in `1_S`) or FALSE-FAIL (FD null-space noise inflates the error).
  - FIX: every gate computes `rel = |P_z(Ha - Hf)| / |P_z Hf|`, i.e. project BOTH `Ha` and
    `Hf` with `P_z` before differencing AND in the denominator. Separately, ASSERT the
    null-space leakage `|(1/sqrt S) 1_S^T (H u)_alpha|` is at the truncation floor (it
    certifies the gauge invariance numerically — both `Ha` and `Hf` must satisfy it).
  - The CG / Lanczos operator used for the PD cert MUST be `P_z H P_z` (project the input
    direction, apply H, project the output). Otherwise the `1_S` zero eigenvalue makes the
    bare operator singular and the PD cert is meaningless (a "lam_min = 0" that is the gauge,
    not the data).

Helpers (build once, reuse in every gate and in the operator):
```
def proj_alpha(g_alpha):           # P g_alpha
    return g_alpha - g_alpha.mean()
def proj_z(u, theta_numel):        # P_z u
    return torch.cat([u[:theta_numel], proj_alpha(u[theta_numel:])])
```

---

## Map of what already exists (theta-only) and what the alpha block needs

The theta-only exact HVP is forward-over-reverse:
1. `build_point_cache` (`hvp_exact.py:47`) runs the production backward ONCE
   (`ggn.vjp_root_to_theta`), caching per-wave `(v_k, dts_r, active_mask)` and the E-adjoint `wE`.
2. `make_exact_hvp` (`hvp_exact.py:78`) builds, per `hvp(u)`: a tangent FORWARD sweep
   (`jvp_root_scores`, `forward_tangent.py`) + a tangent ADJOINT sweep (the `for wave` loop,
   `hvp_exact.py:228-334`) through the SO kernels (`wave_backward_so`, `dts_backward_so`,
   `e_step_backward_so`) + a smooth-head autograd term (`hvp_exact.py:387-394`).

Before this plan was implemented, every one of these paths ran with
`use_receiver_weights=False` and seeded no alpha tangent. Receiver-log-probability cotangent
accumulators existed, but the receiver paths inside the SO and tangent kernels were inactive and
there was no `d_receiver_log_probs` input.

The alpha-block work was therefore to: (a) make the SO and tangent kernels carry the
receiver-log-probability tangent and emit its cotangent, (b) seed the forward tangent with the
softmax Jacobian applied to `u_alpha`, (c) collect the alpha row of `H u` from the head autograd,
and (d) gate every step under `P_z`.

---

## CRITICAL FD-pitfall (inherited from the gradient run; applies to EVERY gate here)

The existing harnesses instantiate `receiver_weights = torch.zeros(S)`
(`_verify_hvp.py:46`, `:60`). That is the DEGENERATE UNIFORM point where
`receiver_weights_are_uniform(...) == True` (`solver.py:11-13`) =>
`use_receiver_weights = False` everywhere (`_execution.py:48`, `solver.py:27`,
`ggn.py:58`) => the entire alpha code path is DEAD, and the softmax Jacobian
`J = diag(w) - w w^T` is at its symmetric center. A gate at `rw=zeros` CERTIFIES NOTHING about
the alpha block.

**CORRECTION (2): re-point `_verify_hvp.py` to a NON-UNIFORM base alpha BEFORE any HVP gate.**
In both `_static_theta_rw_from_live` (`_verify_hvp.py:37-47`) and `_static_theta_rw_from_capture`
(`:50-61`), replace `rw = torch.zeros(S)` with a seeded non-uniform base:
```
g = torch.Generator(device=device).manual_seed(seed)
rw = 0.2 * torch.randn(S, generator=g, device=device, dtype=torch.float64)
rw = rw - rw.mean()                      # land on the gauge slice (optional, cosmetic)
from gpurec.core.inference.solver import receiver_weights_are_uniform
assert not receiver_weights_are_uniform(rw)       # alpha paths are LIVE
```
And assert `valid_mass` is bounded away from 0 at the base AND at `base ± eps*u_alpha` for every
FD direction (`extract_parameters.py:60-65` returns `-inf` when `valid_mass <= 0`, which would
silently poison the FD):
```
from gpurec.core.parameters.extract_parameters import (
    receiver_log_probs_from_weights, receiver_valid_log_normalizer)
def _valid_mass(alpha):
    rlp = receiver_log_probs_from_weights(alpha)
    norm = receiver_valid_log_normalizer(rlp, sp_parent, mad)   # -log2(valid_mass)
    return 1.0 - torch.exp2(rlp).cumsum... # equivalently 2**(-norm); assert finite & > 1e-3
assert torch.isfinite(norm).all() and float((2.0**(-norm)).min()) > 1e-3
```
(use `sp_parent = static.species_helpers["sp_parent"]`, `mad =
int(static.species_helpers["max_ancestor_depth"])`). This check runs on the base alpha and on
`base ± eps * u_alpha / |u_alpha|` for every gate direction.

---

## Step ordering, dependencies, file anchors, per-step FD gate

Steps S3..S8. Each step ends with its OWN FD gate at `4S` (theta AND alpha directions),
all under `P_z`, on the NON-UNIFORM base — **validate the endgame, not just the start**
(the alpha coupling is dominated by the `receiver_norm -> max_transfer` term that only appears
once the forward solve has run, so a start-only gate misses it).

### S3 — Forward-tangent seed: alpha -> rate coupling (FOUNDATION; do FIRST)

Deps: none (but every later step's FD gate needs S3's tangent forward to be correct).

**CORRECTION (6): replace `param_jvp_uniform` with a JVP of
`extract_parameters_weighted_receivers`.** Today `param_jvp_uniform`
(`forward_tangent.py:31-42`) JVPs `extract_parameters_UNIFORM`, which DROPS `receiver_norm`
entirely (uniform path, `extract_parameters.py:25-34`) => ZERO alpha->rate sensitivity. The
DOMINANT alpha coupling is `alpha -> receiver_log_probs -> receiver_valid_log_normalizer
(receiver_norm) -> max_transfer` (`extract_parameters.py:81-92`). Concretely:

- Add `param_jvp_weighted(static, theta, alpha, u_theta, u_alpha)` that does
  `torch.func.jvp(f, (theta, alpha), (u_theta, u_alpha))` where
  `f = lambda th, al: extract_parameters_weighted_receivers(th, al, sh,
  specieswise=..., genewise=..., uniform_fast=True)`. It returns tangents
  `(dlog_pS, dlog_pD, dlog_pL, dmax_transfer, dreceiver_log_probs)`. Note `uniform_fast=True`
  keeps the `- log2(S)` shift (`extract_parameters.py:91-92`) so it matches the primal
  forward's `max_transfer`; the JVP differentiates through `receiver_norm`, which is the
  coupling that was missing.
- The new tangent output `dreceiver_log_probs` (the softmax-Jacobian applied to
  `u_alpha`, i.e. `dreceiver_log_probs = (diag(w) - w w^T) u_alpha / ln2` in log2-space — autograd computes it,
  do NOT hand-roll) is the **alpha tangent SEED**. Thread it IDENTICALLY into the three tangent
  consumers, ALL of which currently force `use_receiver_weights=False`:
  - `e_tangent_fixed_point` (`e_step_tangent.py:174`, called at `forward_tangent.py:72`): pass
    `use_receiver_weights=True` and add a `dreceiver_log_probs` argument (new tangent input to the E-step tangent
    fixed point; the E-step depends on `receiver_log_probs`).
  - `compute_wave_step_tangent` / `compute_wave_step_tangent_selfloop`
    (`wave_tangent.py:392` / `:346`, called at `forward_tangent.py:138` / `:180`): pass
    `use_receiver_weights=True`, pass the live
    `receiver_log_probs = sv["receiver_log_probs"]` (already passed) and
    the new `dreceiver_log_probs`; the wave-step tangent's `p_prime`/`anc` terms gain a `+ ln2 * p' * dreceiver_log_probs`
    contribution (mirrors the SO-kernel change in S5).
  - `compute_dts_tangent` (`dts_tangent.py:96`, called at `forward_tangent.py:161`): pass
    `use_receiver_weights=True` and `dreceiver_log_probs`; the dts cross-wave tangent's `p_prime_c` terms gain the
    same `+ ln2 * p' * dreceiver_log_probs`.
- `_wave_tangent_constants` (`forward_tangent.py:68-96`) and `jvp_root_scores`
  (`forward_tangent.py:99`) grow an `alpha`/`u_alpha` argument; `jvp_root_scores`'s `full` dict
  gains `dreceiver_log_probs` (= `dreceiver_log_probs`) so the adjoint sweep (S4/S5) can consume it.

HARDEST PART of S3: getting the `dreceiver_log_probs` seed to enter the e-step tangent FIXED POINT consistently
with the wave tangent. The e-step and wave-step share `receiver_log_probs`; a `dreceiver_log_probs` that is
threaded into the wave but not the e-step (or vice versa) produces a tangent that is internally
inconsistent and the FD gate will fail by O(1), not O(eps). Thread it into BOTH or NEITHER.

**S3 FD gate.** Directional-derivative check of the FORWARD tangent ONLY (no adjoint yet):
for a seeded `u = [u_theta; u_alpha]`, compare `jvp_root_scores(...).reshape(-1)` against the
central FD of `forward_solve(...)`'s `root_rows` (the `pi_wave` at `root_ids`) at
`z ± eps*u/|u|`, fp64, converged solver. Run with `u_alpha != 0` (the new path) AND
`u_alpha = 0` (must reproduce the OLD theta-only tangent bit-for-bit — a regression guard).
Pass: `rel <= 5e-4` on the projected difference. ASSERT non-uniform + `valid_mass > 1e-3` first.

### S4 — Adjoint-sweep plumbing: thread `dreceiver_log_probs` and collect the alpha receiver-log-probability cotangent

Deps: S3 (needs `full["dreceiver_log_probs"]`).

In `make_exact_hvp` (`hvp_exact.py:200-335`):
- Read `u = u_vec[:theta_numel].reshape(theta_shape)` and `u_alpha = u_vec[theta_numel:]`
  (the function currently does `u = u_vec.reshape(S,3)`, `hvp_exact.py:201` — generalize to the
  joint split; keep theta_shape explicit, do NOT assume `[S,3]`).
- Pass `alpha`/`u_alpha` into `jvp_root_scores` (S3 signature) and pull
  `dreceiver_log_probs = full["dreceiver_log_probs"]` out alongside `dpS_m`/`dpD_m`/... (`hvp_exact.py:207-210`).
- `d_grad_receiver_log_probs` already exists (`hvp_exact.py:224`) as the receiver-log-probability cotangent accumulator and is already
  passed to `solve_reconciliation_wave_vjp` (`:274`) and `accumulate_transfer_complement_vjp_from_donor_adjoint`
  (`:320`) as `grad_receiver_log_probs` — but with `use_receiver_weights=False`, so those kernels
  do NOT scatter into it. Flip those two to `use_receiver_weights=True` (derived; see S7) so the
  receiver-log-probability cotangent is collected.
- Thread `dreceiver_log_probs` into the THREE SO kernels (S4/S5 wiring): `wave_backward_so` (`:253`),
  `dts_backward_so` (`:324`), and the E-side `e_step_backward_so` calls (`:347`, `:354`, `:377`).

**S4 FD gate.** None standalone — S4 is pure plumbing; its correctness is exercised by the S5/S6
kernel gates and the S8 end-to-end gate. (Do a smoke run that the loop executes with the new
args and `d_grad_receiver_log_probs.abs().max() > 0`.)

### S5 — `wave_backward_so`: add the alpha receiver-log-probability cotangent OUTPUT (HARDEST KERNEL)

Deps: S4 (caller passes `dreceiver_log_probs`, expects a `d_grad_receiver_log_probs` back).

This is THE hardest part of the whole effort. **CORRECTION (5):** `_reconciliation_vjp_directional_derivative_kernel`
(`wave_so.py:36-243`) has NO `d_grad_receiver_log_probs` output slot at all — it only writes event-VJP derivatives, `d_out`,
and scatters into `d_rhs`/`d_out` (child rows). The alpha block needs a brand-new receiver-log-probability cotangent
scatter, PLUS the two missing tangent terms where `col` enters as a variable:

1. **Add the missing `+ LN2 * p_prime * dreceiver_log_probs` at the row `p_prime` (`wave_so.py:85`).** Today
   `dp_prime = LN2 * p_prime * dpi_w` (`wave_so.py:85`) treats `col` as frozen. When alpha is a
   variable, `p_prime = exp2(colw + pi_w - rm)` (`wave_so.py:82`, the `USE_RECEIVER_WEIGHTS` branch)
   has an extra dependence: `dp_prime = LN2 * p_prime * (dpi_w + dcol_w)`. Load `dreceiver_log_probs` for the
   row states and add the `+ LN2 * p_prime * dcol_w` term.
2. **Add the missing `+ LN2 * pa * dcol_a` in the ancestor walk (`wave_so.py:103`).** Today
   `danc += LN2 * pa * dpi_a` (`wave_so.py:103`) — when alpha is a variable, the ancestor
   `pa = exp2(col_a + pi_a - rm)` (`wave_so.py:99`) gains `danc += LN2 * pa * (dpi_a + dcol_a)`;
   load `dreceiver_log_probs` at the ancestor index `cur` (mirror the `Pi` ancestor load at `:95`).
3. **Add a NEW receiver-log-probability cotangent scatter.** The transpose of the `col` dependence: every place
   `p_prime`/`pa` enters a contraction with `v`, the `col` partial scatters
   `LN2 * p_prime * (that-contraction)` into `d_grad_receiver_log_probs[s]` (and the ancestor analogue into
   `d_grad_receiver_log_probs[cur]`). Specifically the self block `d_self` (`wave_so.py:230`) and the pibar
   routing (`u_d`/`sub`, `:209-227`) each carry a `col` partial; add a new `d_grad_col_ptr`
   kernel argument and `tl.atomic_add(d_grad_col_ptr + row-or-ancestor-index, contrib)` for the
   col-derivative of each `p_prime`/`pa`-weighted term. Add `d_grad_receiver_log_probs` to the
   `wave_backward_so` Python wrapper outputs (`wave_so.py:246-293`) and to its named event-VJP outputs
   return (`:293`), and accept the new `dreceiver_log_probs` input.

HARDEST-PART WARNINGS for S5:
- The pibar tree routing (`pibar_u_coeff`, `sub`/`dsub`, `wave_so.py:209-227`) ALSO depends on
  `col` through `p_prime` in `inv_denom = 1/(row_sum - anc)` (`:106-108`). The col tangent of
  `row_sum`/`anc` (`drow_sum`/`danc`) feeds `ddenom` (`:109`) and hence `d_pibar_u_coeff`
  (`:210`). So adding `dreceiver_log_probs` to `dp_prime`/`danc` AUTOMATICALLY propagates into the pibar block —
  do NOT double-count by adding a separate pibar col term; just make sure `dp_prime`/`danc`
  carry `dreceiver_log_probs` (items 1–2) and verify the pibar tangent picks it up via `drow_sum`/`danc`.
- The receiver-log-probability cotangent scatter (item 3) and the existing `d_out` child scatters (`:242-243`) write
  to DIFFERENT buffers but share the ancestor/child index math — keep the `tl.debug_barrier()`
  ordering (`:218`, `:225`, `:238`) so the subtree sums are complete before the scatter reads
  them.

**S5 FD gate (PER-KERNEL).** Use `_fd_hessian_hvp` at `4S` but isolate the wave-SO contribution:
run a SINGLE-WAVE harness (one wave with splits + one leaf wave) and compare the analytic
`(d_Av, d_event_vjps, d_grad_receiver_log_probs)` against `torch.autograd.functional`-style central FD of the
first-order `solve_reconciliation_wave_vjp` outputs w.r.t. `(theta, alpha)` along a seeded
`u`, fp64. Project the alpha part of every compared quantity with `P` (the `1_S` null mode of
`d_grad_receiver_log_probs`). Pass `rel <= 5e-4`. Regression guard: `u_alpha = 0` must reproduce the current
`wave_backward_so` outputs bit-for-bit (the existing wave-SO gate must still PASS).

### S6 — `dts_backward_so`: add `dreceiver_log_probs` input + col tangent terms

Deps: S4 (caller threads `dreceiver_log_probs`), S5 done first (same `+ ln2 p' dreceiver_log_probs` pattern, simpler here).

**CORRECTION (4): `dts_backward_so` has NO `dreceiver_log_probs` input today (`dts_so.py:233-303`).** Add a
`dreceiver_log_probs` argument and thread it so `+ LN2 * p_prime * dreceiver_log_probs` lands in BOTH the `d_rhs` scatter AND
the `d_grad_receiver_log_probs` scatter at the tree kernel's `:229`/`:230`:
- `_transfer_subtree_vjp_directional_derivative_kernel` (`dts_so.py:144-230`): today `dp_prime = LN2 * p_prime * dpi_val`
  (`dts_so.py:225`) with `col` frozen. Add the `USE_RECEIVER_WEIGHTS` branch's col tangent:
  `dp_prime = LN2 * p_prime * (dpi_val + dcol_val)` where `dcol_val` is `dreceiver_log_probs` at `s_offs`.
  The two scatters at `:229` (`d_rhs`) and `:230` (`d_grad_receiver_log_probs`) then both correctly carry the
  col-derivative — the contrib `dp_prime*(A - sub) + p'*(dA - dsub)` (`:228`) automatically
  includes it once `dp_prime` does.
- Pass `dreceiver_log_probs` from the wrapper (`dts_backward_so`, `dts_so.py:233`) down to the tree-kernel
  launch (`:288-303`) and flip `USE_RECEIVER_WEIGHTS` per S7. The split kernel
  (`_gene_split_event_vjp_directional_derivative_kernel`, `:28-141`) does NOT touch `col` (its event weights are
  `exp2(d_k - pi_p)` with no col term), so NO change there.

HARDEST PART of S6: `dreceiver_log_probs` is a per-STATE `[S]` tangent, but the tree kernel indexes it at
`s_offs` (child rows) — confirm the indexing matches the `receiver_log_probs` load at `dts_so.py:220`
(`receiver_log_probability = tl.load(receiver_log_probs_ptr + s_offs)`), so
`dreceiver_log_probs` loads at the same `s_offs`.

**S6 FD gate (PER-KERNEL).** Same single-wave harness as S5 but exercise a wave WITH splits;
FD of `accumulate_gene_split_event_vjp` + the tree VJP w.r.t. `(theta, alpha)`. Project alpha
with `P`. Pass `rel <= 5e-4`. Regression: `u_alpha = 0` reproduces current `dts_backward_so`.

### S7 — `use_receiver_weights` derivation (kill the False hardcodes)

Deps: none structurally, but must be flipped before S5/S6/S8 gates pass.

**CORRECTION (7): `ggn.py:58` hardcodes `use_receiver_weights = False`** (and the HVP loop
mirrors it with `use_receiver_weights=False` / `use_receiver_weights=False` at `hvp_exact.py:259`,
`:274`, `:320`, `:333`, `:348`, `:355`, `:378`). Derive it from the base alpha's non-uniformity,
exactly as production does (`solver.py:27`, `_execution.py:48`):
```
from gpurec.core.inference.solver import receiver_weights_are_uniform
use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)
```
Thread this single boolean through `vjp_root_to_theta` (`ggn.py:41`), `build_point_cache`
(`hvp_exact.py:47`), and every kernel call in the HVP loop. CONSEQUENCE: when
`use_receiver_weights=True`, `extract_parameters_weighted_receivers` is called with
`uniform_fast=use_receiver_weights==False` — re-check the `uniform_fast` flag at
`ggn.py:259-260` (`uniform_fast=not use_receiver_weights`) and `hvp_exact.py:189`
(`uniform_fast=True` is HARDCODED — change to `uniform_fast=not use_receiver_weights`), else the
head re-introduces the `- log2(S)` shift inconsistently and the alpha-block FD gate fails by a
constant.

**S7 FD gate.** A first-order regression: with the non-uniform base, the production gradient
`grad_receiver` from `stream_batches` (already wired in S0–S2) must equal the central FD of the
loss w.r.t. alpha, projected with `P`, and the cached-backward
`grad_receiver_log_probs` from
`vjp_root_to_theta` must match it. This is the `_verify_recv_grad.py`-style gate; it certifies
the receiver paths are live before any HVP gate runs.

### S8 — Head autograd: grad the head scalar w.r.t. (theta_req, receiver_weights_req) + assemble `H u`

Deps: S3–S7 (needs `d_cot_receiver_log_probs` correct out of the adjoint sweep and `dreceiver_log_probs` correct out of the
forward tangent).

**CORRECTION (8): grad the head scalar w.r.t. BOTH `theta_req` AND `receiver_weights_req`, not `theta_req`
only.** Today:
- `receiver_weights_req` is built (`hvp_exact.py:184`) but NEVER used as a grad target.
- `phi1`'s grad `g1` is taken w.r.t. `theta_req` only (`hvp_exact.py:197`).
- the final `out` grad is w.r.t. `theta_req` only (`hvp_exact.py:393`).

Change BOTH `torch.autograd.grad` calls to differentiate w.r.t. `(theta_req, receiver_weights_req)`:
- `phi1` (`:195-197`) already includes the `(receiver_log_probs_h * cot_receiver_log_probs).sum()` term (`:196`), so adding
  `receiver_weights_req` as a grad target yields `g1_theta, g1_receiver_weights`. (`cot_receiver_log_probs` = the primal receiver-log-probability cotangent,
  `:181`.)
- `phi2` (`:389-390`) already includes `(receiver_log_probs_h * d_cot_receiver_log_probs).sum()` (`:390`), so the final grad
  w.r.t. `(theta_req, receiver_weights_req)` of `(g1_theta * u_theta).sum() + (g1_receiver_weights * u_alpha).sum() + phi2`
  gives `(out_theta, out_receiver_weights)`. Assemble `H u = cat([out_theta.reshape(-1), out_receiver_weights])`.

**CORRECTION (8, second half — DO NOT DOUBLE-COUNT THE SOFTMAX HESSIAN.** The alpha–alpha
softmax-curvature (`d^2 receiver_log_probs / d alpha^2`, i.e. the curvature of
`log_softmax(alpha)`) belongs in the HEAD autograd EXACTLY ONCE — it is produced by
differentiating `receiver_log_probs_h = receiver_log_probs(receiver_weights_req)` (a function of `receiver_weights_req` via
`receiver_log_probs_from_weights`, `extract_parameters.py:81`) in the `phi1`/`phi2` head grads
above. The KERNELS carry ONLY the receiver-log-probability cotangent LINEAR in `dreceiver_log_probs` (the `+ ln2 p' dreceiver_log_probs` terms of
S4–S6) — they must NOT also apply a softmax-Jacobian/Hessian, or `H_aa` double-counts the
softmax curvature. Concretely: the SO kernels receive `dreceiver_log_probs` (already the softmax-Jacobian times
`u_alpha`, computed once in S3) and emit `d_cot_receiver_log_probs` in `receiver_log_probs`-space; the head's
backward through `receiver_log_probs_h(receiver_weights_req)` maps that back to alpha-space and adds the second-order
softmax term ONCE. Keep `receiver_valid_log_normalizer`'s alpha dependence
(`extract_parameters.py:82-86`) inside the head graph (it is, via `receiver_weights_req`), so the
`receiver_norm -> max_transfer` curvature is captured by `max_transfer_h(receiver_weights_req)` here, not in kernels.

**S8 FD gate (END-TO-END, the real PD-relevant gate).** Mirror `_verify_hvp.run` but:
1. Base alpha NON-UNIFORM (S2 / correction 2) with the `valid_mass` asserts.
2. `vg = make_value_and_grad([static], rw, theta_shape=..., optimize_receiver=True)` so the FD
   side `_fd_hessian_hvp` perturbs the JOINT `z = [theta; alpha]` (length `4S`). The FD already
   reuses forward+backward; no kernel needed.
3. `x = cat([theta.reshape(-1), alpha])`, directions `u` in `R^{4S}` (seeded), including PURE
   theta (`u_alpha=0`, regression vs current 3S gate), PURE alpha (`u_theta=0`, exercises
   `H_aa`+`H_at`), and MIXED.
4. **Project BOTH sides with `P_z` (correction 1):** `rel = |P_z(Ha - Hf)| / |P_z Hf|`.
   Separately assert null-space leakage `|mean((Hu)_alpha)| <= 5e-4 * |Hu|` for BOTH `Ha` and
   `Hf` (the gauge-invariance numerical certificate).
5. Symmetry under `P_z`: `u^T P_z H P_z w == w^T P_z H P_z u` to `rel_asym <= 5e-3` (block
   symmetry across `H_ta` vs `H_at` is the real test of S4–S8 consistency).
Pass: `rel <= 5e-4` on every direction. Run `n_families=8` live fp64 first (cheap), THEN the
666x80 capture on the A100 — and validate the ENDGAME (a near-minimum alpha, where `H_aa` is
near-singular off the gauge), not just the init.

### (later, OUT OF SCOPE for this doc) S9 — PD cert / CG on `P_z H P_z`

The Newton-CG / Lanczos consumer (`newton_cg.py`) must wrap the joint HVP in `P_z` (project in,
apply, project out) so the `1_S` zero eigenvalue is removed and `lam_min` measures the DATA+prior
curvature on the gauge slice, not the gauge. This is the actual PD certification and is gated by
S8 being green; it is a separate sub-effort.

---

## Dependency graph (topo order)

```
S3 (forward tangent: dreceiver_log_probs seed via jvp of weighted-receivers; thread into e/wave/dts tangents)
 └─> S4 (adjoint plumbing: pass dreceiver_log_probs into SO kernels; collect d_grad_receiver_log_probs)
      ├─> S5 (wave_backward_so: NEW d_grad_receiver_log_probs output + missing +ln2 p' dreceiver_log_probs @ :85/:103)  [HARDEST]
      ├─> S6 (dts_backward_so: NEW dreceiver_log_probs input + +ln2 p' dreceiver_log_probs @ :229/:230)
      └─> S7 (use_receiver_weights derived; kill ggn.py:58 + uniform_fast hardcode @ hvp_exact:189)
           └─> S8 (head grad wrt (theta_req,receiver_weights_req); softmax-Hessian ONCE in head; assemble Hu)
                └─> S9 (P_z H P_z for CG/Lanczos PD cert — out of scope here)
```
S2 (re-point `_verify_hvp.py` to non-uniform base + valid_mass asserts) is a PREREQUISITE for the
S3..S8 gates and lands first.

---

## Invariants to re-assert at EVERY gate (checklist)

- [ ] base alpha NON-uniform: `not receiver_weights_are_uniform(rw)`.
- [ ] `valid_mass > 1e-3` (finite `receiver_norm`) at base AND at `base ± eps*u_alpha/|u_alpha|`.
- [ ] `use_receiver_weights` is DERIVED (no `False` hardcode reachable on the gate path).
- [ ] FD comparison projected with `P_z` on BOTH `Ha` and `Hf` AND in the denominator.
- [ ] null-space leakage `|mean((Hu)_alpha)|` at the truncation floor for BOTH sides.
- [ ] `u_alpha = 0` regression: reproduces the current theta-only HVP bit-for-bit.
- [ ] gate validates a near-minimum (endgame) alpha, not only the init.

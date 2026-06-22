# Dtype-aware, fail-safe solver tolerances

**PR:** `fix/solver-tolerance-robustness` · **Date:** 2026-06-23

This change removes the *arbitrary, dtype-fragile* numerical tolerances flagged as
**arbitrary-risky** in [`numerical_constants_audit.md`](numerical_constants_audit.md)
and replaces them with **dtype-derived** values plus **fail-safe** convergence
handling. It is scoped to the core E-adjoint / GGN linear solve (`_bicgstab`) and
its configuration; it is purely additive in behaviour for well-posed solves and
turns one class of spurious crash into a graceful return.

## The problem in one sentence

The historical defaults `bicgstab_tol = 1e-7` and `bicgstab_breakdown_tol = 1e-30`
were **hardcoded to fp32 machine epsilon**, but `1e-7` is **0.84× fp32 eps**
(`2⁻²³ = 1.19e-7`) — i.e. *below* the residual the iteration can actually reach in
single precision — so the solve would stagnate at ~`1.1·eps` and then **raise**,
even though it had effectively converged. This is the `N=128` CV crash:

```
RuntimeError: E-adjoint BiCGSTAB solve failed after 17 iterations
              (relative residual 1.297e-07)
```

`1.297e-7` *is* a converged fp32 residual. The solver crashed on success.

## Design principle

> A tolerance must be expressed relative to the precision it runs in, and a
> linear solve must never raise on an iterate that has reached the
> finite-precision floor.

Two ideas, applied consistently:

1. **Dtype-relative, not hardcoded.** Every tolerance is derived from
   `torch.finfo(working_dtype).eps`. This mirrors the patterns the codebase
   *already* gets right — `forward_tangent._default_tol` (`1e-6` fp32 / `1e-12`
   fp64) and `batched_lbfgs`'s `cubic_interpolate` (`eps = torch.finfo(x.dtype).eps`).
2. **Fail-safe, not fail-pedantic.** The solver returns the best iterate whenever
   it reaches the working-precision floor, and only raises on *genuine*
   non-convergence — but it still raises loudly then (no silent garbage).

## What changed

### 1. `_bicgstab` relative-residual target — `1e-7` → dtype-matched (`api/_implicit_grad.py`)

`tol=None` (the new default) resolves to `_bicgstab_rel_tol_default(dtype)`:

| dtype | target | = × eps | why |
|---|---|---|---|
| fp32 | `1e-6` | 8.4 × eps | above the ~`1.1·eps` stagnation floor; far below the ~`2e-4` downstream gradient atomic-noise floor → **zero** accuracy cost |
| fp64 | `1e-12` | 4.5e3 × eps | tight, but trivially reachable in double precision |

These are the **same values** `forward_tangent._default_tol` uses for the sibling
tangent solve, so the whole optimiser stack now speaks one tolerance convention.

A caller may still pass an explicit `tol` (e.g. `_verify_hvp` passes `1e-10` for
its fp64 gate). If a caller passes a value **below the dtype floor**
`4·eps` (unreachable), it is **clamped up with a `RuntimeWarning`** rather than
guaranteeing a stagnation-raise:

```
bicgstab tol=1.00e-12 is below the torch.float32 finite-precision residual
floor 4.77e-07; clamping to the floor. A tighter relative residual is
unreachable in this precision -- use fp64.
```

### 2. `breakdown_tol` — absolute `1e-30` → **relative** Krylov-breakdown test

`1e-30` is an absolute floor four orders below fp32's smallest *useful* magnitude.
In fp32 it either never trips (so the graceful-breakdown branch was dead code), or
it trips on the small-but-valid inner products that occur *near convergence* — and
the old code then **raised** on that essentially-converged iterate. (The `N=128`
crash broke out at iter 17 and raised.)

The breakdown checks are now **dimensionless relative tests** against the operand
norms, so they fire only on *true* loss-of-orthogonality / a near-null direction:

| quantity | old (absolute) | new (relative, `bd = eps`) | meaning when it fires |
|---|---|---|---|
| `ρ = ⟨r̂,r⟩` | `|ρ| ≤ 1e-30` | `|ρ| ≤ bd·‖r̂‖·‖r‖` | `r̂ ⟂ r` to machine precision (classic BiCGSTAB breakdown) |
| `⟨r̂,v⟩` | `≤ 1e-30` | `≤ bd·‖r̂‖·‖v‖` | search direction orthogonal to the shadow residual |
| `‖t‖² = ‖As‖²` | `≤ 1e-30` | `≤ (bd·‖s‖)²` | `As ≈ 0` with `s ≠ 0` (near-null direction); also guards `ω = ⟨t,s⟩/‖t‖²` |
| `ω` | `|ω| ≤ 1e-30` | `|ω| ≤ bd·|α|` | `ω` collapsed relative to `α` (guards the `α/ω` in the next `β`) |

`breakdown_tol=None` ⇒ `bd = eps`. An explicit value is reinterpreted as the
relative factor (legacy `1e-30` ⇒ "essentially never trip", which is now safe
because non-convergence is handled by the graceful-exit below rather than by a
mid-iteration raise).

### 3. Graceful exit — return the best iterate at the floor, raise only on real failure

The solver now tracks the **best** `(x, relative-residual)` it has seen. On loop
exit (max_iter **or** a relative breakdown):

- if `best_res ≤ max(target, default(dtype))` → **return** `best_x` (converged to
  working precision);
- else → **raise** `RuntimeError` with a diagnostic that distinguishes
  `broke down` vs `hit max_iter`, reports `best_res`, the target, and the dtype,
  and states the likely cause (singular / too ill-conditioned for this precision).

So a solve that reaches the floor never crashes, and a solve that genuinely fails
still fails *loudly* — we never silently return garbage.

### 4. `SolverOptions.bicgstab_tol` / `bicgstab_breakdown_tol`: `float` → `Optional[float] = None`

`None` = "use the dtype-relative auto value". `validate()` accepts `None` and still
rejects non-positive floats. The plumbing (`_implicit_grad`, `ggn`, `hvp_exact`,
`map_cv`) threads `None` through instead of the old hardcoded `1e-7` / `1e-30`; one
stray `float(so.bicgstab_tol)` (which would have crashed on `None`) was removed.
`_verify_hvp`'s explicit fp64 `1e-10` is **kept** — it is correctly motivated and
sits comfortably above the fp64 floor.

### 5. `hvp_exact` `tangent_self_iters` fallback `16` → `64` + warning

A separate arbitrary-risky item: when no `tangent_self_iters` / env / `pi_iters`
is available, the HVP self-loop fell back to **16**, which the repo memory documents
as *non-convergent* on the representative fixture (`+33` NLL, ~`2.6×` gradient bias)
— so an ad-hoc caller silently got a wrong Hessian. It now falls back to the
validated floor of **64** and emits a `RuntimeWarning`.

## Why these specific numbers are *not* arbitrary

- `eps = torch.finfo(dtype).eps` — the unit roundoff, the only principled scale for
  a finite-precision residual.
- target multiplier (fp32 `1e-6 ≈ 8·eps`, fp64 `1e-12`): **matched** to the values
  the codebase already uses for the sibling tangent solve (`forward_tangent._default_tol`),
  so the change introduces *no new convention*. Both sit above the eps floor and
  below the measured ~`2e-4` gradient atomic-noise floor (so tightening further buys
  nothing; loosening to here costs nothing).
- floor multiplier `4·eps`: a small constant > 1 guaranteeing the target is
  reachable. The empirically observed fp32 stagnation is ~`1.1·eps`; `4·eps` clears
  it with margin.
- `bd = eps` for the relative breakdown: a loss of orthogonality *to machine
  precision* is the textbook BiCGSTAB breakdown condition.

## Verification

**Unit (`/tmp/test_bicgstab_robust.py`, CPU, fp32+fp64):**

| case | result |
|---|---|
| (a) ill-conditioned fp32 @ `tol=1e-7` (the old crash mode) | **returns** `rel_res=6.1e-7` (was: `RuntimeError`) |
| (b) well-conditioned fp64 @ `tol=1e-12` | `rel_res=5.6e-13`, `‖x−x*‖/‖x*‖=1.4e-10` |
| (c) fp32 @ `tol=1e-12` (sub-floor) | warns + clamps, `rel_res=4.5e-7` |
| (d) unconverged (`max_iter=2`, `tol=1e-14`) | **raises** (fail-loud preserved) |
| (e) defaults dtype-matched & above floor | fp32 `1e-6`>`4.8e-7`, fp64 `1e-12`>`8.9e-16` |

**End-to-end:** the `hogenom N=128` 5-fold CV — which **crashes** on the current
library at exactly the `1.297e-7` breakdown — completes on this branch. (Run with
`PYTHONPATH` pointed at this worktree; see the PR description for the exact command
and result.)

## Deliberately *not* changed (documented follow-ups)

These audit items are real but out of this PR's scope (different worktree / wider
blast radius / better as their own change):

- `cg.py` Steihaug residual `tol` is **absolute**, not relative — a latent
  scale-mismatch with this now-relative bicgstab path. Touches every Newton caller;
  separate PR.
- Receiver-weight Fisher endgame (`receiver_curvature.py`, the joint driver): the
  `lam_min > 0.0` PD gate with no Ritz-noise margin, the `ridge = 0 if pd` logic,
  and `FISHER_CG_TOL` with no convergence assertion (the `se_w` non-convergence root
  cause). Lives in the `receiver-weights-hvp` worktree; folded into that branch.
- The `bnorm = max(‖b‖, 1.0)` floor turns the relative target absolute when
  `‖b‖ < 1`; left as-is pending a measurement of the typical adjoint-RHS magnitude.

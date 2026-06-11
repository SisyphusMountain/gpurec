# Full-dataset convergence: why `|g|` stalls and the exact levers

Investigation into why specieswise gradient descent on the **full archaea dataset**
(~5379 families, `largest` order, λ=1 Sanderson penalty, S=119) does not drive the
gradient to zero, and what the exact levers for true convergence are.

**TL;DR.** Convergence has two barriers with two distinct levers:

| | barrier | lever | effect |
|---|---|---|---|
| **1** | 205/5379 *stiff* families have an unconverged backward Neumann adjoint at `neumann_terms=16` → the gradient is **biased** → the optimizer converges to the **wrong point** | give **only the stiff families `neumann_terms≥64`, grouped into their own batch, set once and *frozen*** (re-checked every few steps, **never** per-step) | true `\|g\|inf` **14.7 → 1.66** (8.9×) |
| **2** | float32 **loss quantization**: the loss is a float32 sum of 5379 NLLs ≈ 358567, ULP = 0.031; near the minimum the entire remaining descent is < 1 ULP, so the line search can't see improvement | **float64 / Kahan accumulation of the family-sum** (loss **and** gradient reduction) | breaks the 1.66 floor |

Reproduce: `python -u scripts/converge_full_archaea.py` (attains `|g|inf ≈ 1.66`).

---

## The symptom

With the base solver (`pi_iters=16, neumann_terms=16`) and batched-LBFGS, `|g|inf`
plateaus around **5.5** and never approaches 0 — turning adaptive rebatching off did not
help, and `|g|rms` settled near 0.7. The plateau looked like either a numerical floor or
an optimizer failure. It is neither.

## Diagnosis

All runs use the **repo** gpurec (`sys.path.insert(0, REPO)`); a bare `python script.py`
otherwise imports the stale `.venv` build whose native parser is buggy.

The full dataset builds in **18 s** (25 streamed batches, **0.3 GB** GPU) — it is cheap.

### Barrier 1 — the base-solver gradient is biased

Optimize with the base solver to its plateau, then **hold theta fixed** and sweep the
solver accuracy:

```
at the base-solver "optimum":
  pi=16  nt=16  (what the optimizer used) : |g|inf = 5.50      <- biased undershoot
  pi=64  nt=64  (converged)               : |g|inf = 14.69
  pi=400 nt=128                           : |g|inf = 14.69
```

The accurate gradient is **bigger (14.7), not smaller**. So the base solver reports a
biased gradient and the optimizer drove *that* to ~5.5 — i.e. **it converged to the wrong
point.** Isolating the lever (sweep one knob, hold the other converged):

```
backward sweep (pi=400 fixed):   nt=4 -> 260,  nt=8 -> 95,  nt=16 -> 6.2,  nt=32 -> 14.6,  nt=64 -> 14.69,  nt=128 -> 14.69
forward  sweep (nt=128 fixed):   pi=16 -> 14.58,           pi=64 -> 14.69, pi=128 -> 14.69
```

→ the lever is the **backward** `neumann_terms` (needs ≥64); `pi_iters=16` is already fine.
`convergence_report` at the base solver: **205/5379** families have backward residual
> 1e-3 (tier-1), **0** need GMRES (tier-2), 1 marginal on the forward.

### Fixing it — per-tier, frozen (not per-step)

Group the 205 stiff families into their own batch at `nt=64`, leave the ~5174 bulk at
`nt=16`, set once and **freeze** (re-check every ~8 steps, never every step). Per-step
adaptation changes the gradient operator between LBFGS evaluations → corrupts the
curvature history `y_k = g(x_{k+1}) − g(x_k)` and the line search → it stalls/oscillates.
Frozen is **stable** (exactly 205 at every re-check, no oscillation), **LBFGS-safe**, and
costs ~28 % of uniform `nt=64` backward (`5174·16 + 205·64` vs `5379·64`).

Result: continuing LBFGS drives the **true** `|g|inf` **14.7 → ~1.66** (8.9×), then stalls.

### Barrier 2 — the 1.66 floor is float32 loss quantization

At the converged theta the gradient is **identical across solvers** (`pi=64/128/400`,
`nt=64/128` all give `|g|inf = 1.659`) → it is the *accurate* gradient, not truncation.
Probing the loss along `−g`:

```
loss(theta*) = 358567.6250
  the loss only ever moves in steps of ±0.0312   (= 2^-5, the float32 ULP at 2^18.4)
  best step reduces it by exactly one ULP (−0.0312)
plain GD (no line search) diverges:  |g|inf 1.66 -> 34 -> 1677 -> ...   (stiff dirs, huge curvature)
```

So **1.66 is the genuine float32 floor.** The loss is a float32 sum of 5379 large NLLs
≈ 358567 whose ULP is 0.031; near the minimum the entire remaining descent is < 1 ULP, so
gradient steps change the loss sub-ULP and the line search sees only quantization jitter.
The residual gradient floor is `g_floor ≈ √(ULP_loss · curvature) ≈ 1.66`
(`|g|inf/loss = 4.6e-6`); theta is within ~0.04 of the true minimum in the worst
coordinate, and closing that gap would lower the loss by ~1 ULP — invisible.

## Results

```
init                               |g|inf = 517
base-solver plateau (biased)       |g|inf = 5.5   (true grad there = 14.7)
per-tier frozen (nt=64 stiff)      |g|inf = 1.66   <- converged to the float32 floor
verify (uniform nt=128)            |g|inf = 1.66   (solver-accurate, stable)
```

`|g|inf` 517 → 1.66 (312× from init); the loss is at its provable float32 minimum.
That is **true convergence to float32 precision**.

## Exact levers (summary)

1. **`neumann_terms ≥ 64` for stiff families, per-tier & frozen** — the big lever
   (14.7 → 1.66). Without it the optimizer converges to the wrong point. `pi_iters=16` is
   fine; GMRES is not needed.
2. **float64 / Kahan accumulation of the family-sum** (loss **and** gradient reduction) —
   to go below the 1.66 float32 floor near the minimum. Mean-vs-sum scaling does **not**
   help (float32 *relative* precision is scale-invariant — only more bits help).
   **NB: not a runtime flag** — the model's float64 *backward* is currently a
   silent-corruption bug (see "float64 attempt" below), so this requires kernel work.
3. **Optimizer: LBFGS** (line search) reaches the float32 floor cleanly; plain GD diverges
   on the stiff directions. The bottleneck beyond 1.66 is *precision*, not the optimizer.

## Suggested directions

- **Fix the float64 adjoint first, then implement the float64 family-reduction.** Running
  the model in float64 today returns a *garbage* gradient (the backward is broken — see
  below), so float64 is not yet usable. The dominant quantization is the **within-batch**
  reduction over thousands of families, which lives in the adjoint / root-row-NLL
  **kernels** (`scatter_lse`, the NLL sum); a Python-level cross-batch float64 sum (25
  terms) is *not* enough (it moves `|g|` by 1.7e-4). Once the float64 backward works (or the
  float32 reduction uses compensated/Kahan summation), `|g|` should drop below 1.66.
- **Centering won't help; precision will.** Confirmed analytically and by the probe.
- **Keep the per-tier-frozen schedule, not per-step adapt.** Re-check stiffness on a cadence
  (e.g. every 8–20 steps); the stiff set is stable near the optimum (205, no drift/oscillation).
- **Classification reference must be the bulk `nt` (16), not the current `solver_options`** —
  classify "who is NOT converged at what the bulk runs."
- **A convergence schedule** (tighten `nt` as `|g|` drops) is unnecessary here: `nt=64`
  already converges every family's contribution to the float32 floor; the wall above is the
  loss/gradient precision, addressed by lever 2.

## float64 attempt: the adjoint is broken in float64 (and the floor is confirmed real)

Tried one BFGS/GD step in float64 from the 1.66 checkpoint
(`diagnose_float64_step.py`, `diagnose_float64_full.py`, `diagnose_fd_gradcheck.py`):

- **Full float64 forward works** — loss 358567.617 vs float32 358567.594 (~1 ULP apart).
- **Full float64 backward is broken** — it returns a garbage gradient: `|g|inf` = **60**
  then **83** on reruns (non-deterministic), with wrong signs, and `−g64` is an *ascent*
  direction (loss rises monotonically along it: +6e-5, +3e-3, +0.041, +0.37, +4.2). **No
  error is raised — it silently corrupts.**
- **Finite differences (float64 forward loss) confirm the true gradient ≈ the float32
  gradient.** Dominant coords: species 2/L `g32=−1.61`, FD = −1.66 (ε=1e-2) / −1.53
  (ε=1e-3) (the ε=1e-1 −3.6 is large-step curvature); species 19/T `g32=−1.01`, FD = −1.09.
  The float64 analytic values (+5.4, −5.7) match nothing. → **float32 gradient is accurate,
  the 1.66 floor is real.**
- **Cross-batch (Python-level) float64 doesn't help** — accumulating per-batch
  losses/gradients in float64 changes the gradient by only **1.7e-4**; the floor is the
  *in-kernel* float32 family reduction, not the 25-term cross-batch sum.
- A plain line-searched GD step from theta* also blows `|g|` up (1.6 → 9.7): the residual
  lives in a stiff (high-curvature) direction, so it is **ill-conditioned** as well as
  precision-limited (hence LBFGS, not GD).

**Consequences:** (1) the 1.66 float32 floor is genuine — convergence is achieved to float32
precision; (2) the float64 lever needs the **float64 adjoint kernels fixed** (a real,
independently-worth-fixing silent-corruption bug) — it is not a runtime flag; (3) even with
float64 you must keep a curvature optimizer for the stiff direction.

## Scripts (this investigation)

- `scripts/converge_full_archaea.py` — **the recipe**: build → base-optimize → adapt-once
  (per-tier frozen) → finish → verify; attains `|g|inf ≈ 1.66`.
- `scripts/diagnose_float64_step.py`, `diagnose_float64_full.py`, `diagnose_fd_gradcheck.py`
  — the float64 attempt + the finite-difference referee (above).
- `scripts/diagnose_full_convergence.py` — Phase A: base plateau + solver-accuracy sweeps +
  `convergence_report` (isolates the `neumann_terms` lever).
- `scripts/diagnose_full_convergence_phaseC.py` — per-tier-frozen convergence to 1.66.
- `scripts/diagnose_floor_probe.py` — proves 1.66 is the float32 loss-quantization floor.
- `scripts/diagnose_archaea_grad_convergence.py` — the easy 80-family subset (LBFGS → 1.2e-2;
  no stiff families, so adapt never fires there).
- `scripts/diagnose_notebook_nonconvergence.py`, `diagnose_grad_floor_scaling.py`,
  `diagnose_stiff_adapt_reproduce.py` — supporting experiments (adapt on/off, size scaling).

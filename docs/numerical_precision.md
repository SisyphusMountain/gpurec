# Numerical precision and how far convergence can be pushed

How floating-point precision limits gradient descent on the **full archaea dataset**
(~5379 families, `largest` order, λ=1 Sanderson penalty, specieswise, S=119), the
**precision ladder** that governs the achievable `|g|`, the **E-step backward race bug**
that had to be fixed to use float64 at all, and **what each recipe actually reached**.

This is the precision sequel to [`full_dataset_convergence.md`](full_dataset_convergence.md)
(which covers the gradient-bias lever — stiff-family `neumann_terms≥64`, per-tier frozen —
that gets `|g|inf` from 14.7 to the 1.66 float32 floor). Start there for Barrier 1; this
document is Barrier 2 (precision) and everything past the float32 floor.

## TL;DR

| stage | precision | `\|g\|inf` | loss | cost |
|---|---|---|---|---|
| init | fp32 | 517 | — | — |
| fp32, stiff `nt≥64` per-tier frozen | fp32 everywhere | **1.66** | 358567.63 | fp32 speed |
| **mixed**: fp64 master θ + fp64 batch accumulation, **fp32 kernels** | fp64 reduction | **~0.59** (best) | 358564.58 | ~fp32 speed |
| **fp64 polish**: full float64 kernels | fp64 everywhere | **~0.43** (best) | 358564.57 | ~1/64 speed |

- `|g|inf` went **517 → 1.66 → 0.43** (~1200× from init; ~4× below the float32 floor).
- The loss dropped **358567.63 → 358564.57**, a **real ~3.06** reduction ≈ **98 float32 ULPs**
  of genuine descent that float32 *could not see* (its loss ULP at this magnitude is 0.031).
- The residual `|g|inf ≈ 0.43` / `|g|rms ≈ 0.1` is a **broad, shallow ridge** (the penalized
  likelihood is nearly flat across 168+ rate coordinates) plus float32 kernel/gradient noise —
  **not** a sharp minimum we are still far from.
- Using float64 at all required fixing a **silent-corruption race in the E-step backward
  kernel** (commit `f71a38dae`).

## 1. The float32 floor is *loss-value quantization*, not the optimizer or the solver

At the converged θ the gradient is **identical across solver settings**
(`pi=64/128/400`, `nt=64/128` all give `|g|inf = 1.659`) → it is the *accurate* gradient.
Probing the loss along `−g` (`scripts/diagnose_floor_probe.py`):

```
loss(theta*) = 358567.6250
  the loss only ever moves in steps of ±0.0312  (= 2^-5, the float32 ULP at 2^18.4)
  best step reduces it by exactly one ULP (−0.0312)
```

The loss is a single float32 scalar ≈ 358567 whose ULP is **0.031**. Near the minimum the
entire remaining descent is **< 1 ULP**, so a strong-Wolfe line search sees only
quantization jitter and collapses `alpha → 0`. The residual gradient floor is
`g_floor ≈ √(ULP_loss · curvature) ≈ 1.66`. That is **true convergence to float32
precision** — but not to zero.

## 2. The precision ladder — where the bits actually matter

There are **three** distinct precision walls, hit in this order as you add bits:

**(a) Single-scalar loss ULP = 0.031** — the dominant wall.
The line search compares two float32 loss scalars; any improvement below 0.031 vanishes.
This is a property of *storing the loss value* in float32, independent of how accurately it
is computed — so **Kahan-summing into a float32 result does not help**; the loss must be
*carried* in float64.
Crucially, the **in-kernel float32 family sum is already accurate**: per-batch float64 ==
per-family float64 to **1.7e-5**, so accumulating just the **25 per-batch loss scalars** in
float64 captures essentially all the precision. That alone drops the loss-comparison noise
from 0.031 to **~1.7e-3** (~18× finer) — **at full float32 kernel speed** (this is the
*mixed* recipe). A Python-level cross-batch float64 sum is enough *for the loss*; it moves
the gradient by only 1.7e-4, because the gradient's residual lives elsewhere (see below).

**(b) Float32 kernel-recompute noise ≈ 1.7e-3** — the wall after (a).
Once the loss is carried in float64, the next limit is the run-to-run/representation noise
of the per-batch loss and gradient **recomputed in float32 kernels**. Only **float64
kernels** go below this.

**(c) Float32 gradient floor ≈ 1e-2 and the genuine shallow ridge.**
The minimum is broad: at the float32 floor `|g|rms ≈ 0.23` is spread across 168+
coordinates, and even in float64 `|g|inf` settles around ~0.4 with `|g|rms ≈ 0.1`. Part of
that is the float32 gradient accumulation floor; part is **geometry** — the penalized
likelihood is nearly flat in many rate directions, so `|g|` does not go to ~0 regardless of
precision.

> Scaling the objective (mean vs sum) does **not** help any of these — float32 *relative*
> precision is scale-invariant. Only more bits help.

## 3. The blocker: a silent lost-update race in the E-step backward kernel (`f71a38dae`)

Before float64 could be used, the float64 **backward** returned garbage:

- `|g|inf` = **1431 / 60 / 83** across reruns (non-deterministic), wrong signs, and `−g64`
  was an *ascent* direction (loss rose monotonically along it). **No error was raised.**
- Root cause: `_stage_extinction_and_transfer_complement_vjp_kernel` issued plain `tl.store` to `grad_E` /
  `excluded_u`, then `tl.atomic_add` into **overlapping** rows (a state's species-children
  `c1`/`c2` and `sp_parent` ancestors are other states handled by **other warps of the same
  CTA**) with **no barrier**. A warp's `atomic_add` could land *before* another warp's
  initializing `store`, which then overwrote it → a dropped gradient contribution.
- **Latent in float32**: the committed E fixture is shape `(1, 119)` = a single program, so
  warps stay lockstep and the window rarely opens. **float64 surfaces it** — its ~1/64
  throughput stretches inter-warp skew.
- Fix: a `tl.debug_barrier()` between the stores and the atomic_adds (`e_step.py`, +7 lines)
  so all initializing stores complete before any atomic accumulation begins.
- After the fix: float64 `|g|inf = 1.6103` is **deterministic** (run-to-run spread
  1431 → **1.76e-12**) and matches float32's 1.6146 at the checkpoint. The gradient is now
  correct.

This is an independently real correctness bug (it could silently corrupt any sufficiently
parallel backward), not just a float64 curiosity.

## 4. Demonstration that float64 resolves the descent float32 quantizes away

`scripts/diagnose_fp64_lineprobe.py` — line-probe along `−g` at the 1.66 checkpoint, reading
the loss in float64:

```
fp64 loss(theta*) = 358567.6166868689     |g|inf = 1.6103
   alpha      fp64 dloss    fp32 dloss
   3.0e-04     -0.005768     +0.0000     <- fp32 sees nothing (sub-ULP)
   1.0e-03     -0.017389     -0.0312     <- fp32 can't tell these
   2.0e-03     -0.029521     -0.0312     <-  three steps apart;
   3.0e-03     -0.036394     -0.0312     <-  all read as "-1 ULP"
   5.0e-03     -0.034357     -0.0312
   8.0e-03     +0.008186     +0.0000
```

float64 sees a smooth descent (true step minimum near α≈3e-3, Δloss = −0.036); float32
quantizes it into ±0.031 lumps and cannot distinguish the optimal step. Iterating this in
float64 descends below 1.66.

## 5. The three recipes and what each reached

**Mixed precision** — `scripts/converge_full_archaea_mixed.py`
(float64 master θ, float64 accumulation of per-batch loss/grad, **float32 kernels**; cast
θ→fp32 before each batch solve). 30 LBFGS steps from the 1.66 checkpoint:

```
checkpoint:  loss=358567.6335  |g|inf=1.6147
result:      loss=358564.5834  |g|inf=1.9998   (best seen 0.5915)   dloss=-3.05   (1018 s)
```

The loss descended by **3.05 ≈ 98 float32 ULPs** — real descent float32 could not see —
at essentially full float32 speed. `|g|inf` is noisy (it bounces 0.59↔2.0 over the shallow
ridge); the best iterate is ~0.59. This is the **best-value recipe**: it removes the
dominant single-scalar ULP wall without paying for float64 kernels.

**Full float64 polish** — `scripts/converge_full_archaea_fp64.py` (everything float64). 8
steps from the mixed checkpoint:

```
start (fp64):  loss=358564.5794  |g|inf=2.0014
best (step 4): loss=358564.5707  |g|inf=0.4347     |g|inf: 2.0014 -> 0.4347   (1342 s)
```

Full float64 kernels go a little deeper than mixed (0.59 → **0.43**) but at ~1/64 throughput
(8 steps ≈ 22 min). Use only as a short final polish.

**fp32-warm → fp64 two-phase** — `scripts/converge_full_archaea_fp32warm_fp64.py`
(build the L-BFGS curvature history cheaply in float32, cast it to float64, continue). Sound
idea, but a caveat we observed: the float32 warmup **builds 0 curvature pairs** because the
float32 line search stalls at the floor (no accepted steps → no `(s,y)` pairs). The phase-B
float64 descent still works (loss 358567.62 → 358567.14, `|g|` bouncing) but inherits no
curvature. To make the warmup useful it must run on the **float64-carried loss** too (i.e.
warm up under the mixed objective, not the quantized float32 one).

## 6. What we reached (bottom line)

- `|g|inf`: **517 → 1.66 (float32 floor) → ~0.43 (float64)** — ~1200× from init, ~4× below
  the float32 floor.
- loss: **358567.63 → 358564.57**, a genuine **~3.06** reduction (≈98 float32 ULPs) that
  float32 cannot resolve.
- The remaining `|g|inf ≈ 0.43`, `|g|rms ≈ 0.1` is a **broad shallow ridge** (flat
  penalized-likelihood directions) plus float32 kernel/gradient noise — not distance to a
  sharp optimum. For rate inference this is converged well past any practical need
  (`|g|/loss ≈ 1.2e-6`).

## 7. Recommended recipe

1. **Routine fits — float32.** Stiff-family `neumann_terms≥64` (per-tier frozen) → `|g|inf
   ≈ 1.66`, loss at its float32 minimum. Fully converged for practical purposes.
2. **Deeper polish — mixed precision (best value).** float64 master θ + float64 accumulation
   of the **per-batch** loss/grad, **float32 kernels**. Full float32 speed, removes the
   single-scalar ULP wall, reaches `|g|inf ≈ 0.5–0.6`. This is the cheap lever the
   investigation pointed to, now confirmed.
3. **Deepest — full float64 kernels.** Reaches `|g|inf ≈ 0.43` but at ~1/64 throughput; a
   short final polish with a curvature optimizer (LBFGS), ideally reusing curvature warmed
   under the float64-carried loss.

Do **not**: rely on mean-vs-sum scaling (relative precision is scale-invariant), Kahan-sum
into a float32 loss (the value must be *carried* in float64), use per-step adapt (it breaks
LBFGS), or use plain GD (it diverges on the stiff direction — `1.6 → 9.7` in one step).

## 8. Scripts

- `scripts/converge_full_archaea.py` — the float32 recipe (→ 1.66).
- `scripts/converge_full_archaea_mixed.py` — float64 reduction + float32 kernels (→ ~0.59).
- `scripts/converge_full_archaea_fp64.py` — full float64 (→ ~0.43).
- `scripts/converge_full_archaea_fp32warm_fp64.py` — fp32-curvature → fp64 two-phase.
- `scripts/diagnose_floor_probe.py` — proves 1.66 is the float32 loss-quantization floor.
- `scripts/diagnose_fp64_lineprobe.py` — proves float64 resolves the sub-ULP descent.
- `scripts/diagnose_float64_step.py`, `diagnose_float64_full.py`, `diagnose_fd_gradcheck.py`
  — the float64 attempt + finite-difference referee that pinned the bug.
- `docs/backward_atomics_profiling.md` — profiling of the backward atomics path.

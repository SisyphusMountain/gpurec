# Fresh-pair reseed: matched 200-family result

## Outcome

Rebuilding the endpoint complete-information seed from the already-paid EM1->EM2 gradient pair is
a safe, reproducible optimization, but it is not dramatic. It reduces shared-EM-inclusive gradient
work by 5.8% for native coordinates and 10.0% for hierarchical coordinates relative to their own
matched controls. After reseeding, native and hierarchy are effectively tied; the coordinate change
does not provide an additional material advantage.

No 500-family or H100 promotion was made because the stated objective is roughly 2x, while this
change projects to only a single-digit percentage reduction in the full 396 s run.

## Experiment definition

All four arms start from the same saved, point-consistent public-API EM2 artifact, use the same
family order and 200,740 clade budget, and charge the same two EM reverse passes. Likelihood,
float32 model/float64 accumulation, rate box `1e-6..2`, projected native-gradient tolerance `1e-3`,
pi/Neumann tiers, BFGS, Hessian refresh 15, trust settings, freezing, replanning, and certification
are unchanged.

The two reseed arms make one source-adapted change inside the first obligatory Newton evaluation:

1. consume the just-measured `g2` at the EM2 endpoint;
2. rebuild `Ic(theta2; N1)` from the final EM count buffer;
3. form the latest paid pair `s=theta2-theta1`, `y=g2-g1` (or consistently transformed `phi` pair);
4. scalar-calibrate with `(s'y)/(s'Ic s)` when positive and finite;
5. apply the ordinary full dense BFGS update to that pair;
6. take the existing native or hierarchical trust/bound step unchanged.

There is no extra gradient, no target optimum, and no fitted-family oracle. The adapter compiles
from asserted production-source sites and does not edit production files. CPU tests reproduce the
independently generated candidate seeds bit-for-bit.

## Results

| arm | continuation wall | shared-EM-inclusive clade equivalents | Newton steps | builds / replans | certified | fresh audit NLL |
|---|---:|---:|---:|---:|---:|---:|
| native old | 29.949 s | 13.5925 | 21 | 10 / 9 | 200/200 | 613261.799441 |
| native reseed | 27.107 s | 12.8101 | 17 | 8 / 7 | 200/200 | 613261.784221 |
| hierarchy old | 31.079 s | 14.1598 | 21 | 8 / 7 | 200/200 | 613261.797938 |
| hierarchy reseed | 27.062 s | 12.7439 | 18 | 7 / 6 | 200/200 | 613261.783095 |

Relative to matched controls:

- native reseed: -5.76% total gradient-clade work and -9.49% continuation wall;
- hierarchy reseed: -10.00% work and -12.93% wall;
- hierarchy reseed versus native reseed: -0.52% work and -0.17% wall, too small to distinguish.

Wall results are single ordered local runs, so work is the more robust comparison. The common
prototype artifact generation cost is 5.929 s and is included only in the driver's separately
reported prototype total; it cancels between arms. The driver charges all resident model clades in
every production gradient call. Analytic Hessian cost is separately timed and all arms used one
round.

All fits passed the original certificate. The common pi64/Neumann64 forward audit is outside fit
timing. Relative to native old, native reseed is 0.015220 bits lower and hierarchical reseed is
0.016346 bits lower. No family changes by 0.01 bits; maximum absolute family change is about
0.00136 bits. Native and hierarchical reseeds differ by only 0.001126 bits total and 0.001175 bits
maximum per family.

## Why this does not approach 2x

The best reseeded arm reduces total shared-EM-inclusive work by 6.24% versus the matched native
baseline. Even applying a similar reduction directly to the full run's roughly 356 s gradient/count
phase would save only about 22 s from a 396 s run. Reaching 200 s without changing the 40 s
non-gradient remainder requires removing about 55% of the gradient/count phase.

Reusing an unchanged-point gradient after a same-tier replan is another real but bounded idea. The
old native trace contains 1,857,617 such clades: 10.71% of continuation gradients or 9.14% after
including shared EM. An optimistic projection removes 32.9 s from the 307 s H100 gradient phase,
for a total near 363.4 s (8.3% faster), not 2x. Repeated-point NLLs were bit-identical, but gradient
differences from batch-order/pruning noise reached 0.00151, so near-threshold verification would be
required. This optimization was not combined with reseeding.

## BFGS and line-search ceiling

The hierarchical complete-information matrix is diagonal only before updating. The reseed applies
a full dense rank-two BFGS update immediately. Every later evaluated chord—including a rejected
trial before rollback—updates the same dense 3x3 matrix, and exact observed-Hessian refreshes are
dense. Therefore the complete-data factorization does not justify three independent observed-NLL
line searches, and reducing 3x3 linear algebra cannot matter beside likelihood passes.

In the old traces, rejection-containing evaluations covered 1.260 M rejected-family clades for
native and 1.367 M for hierarchy. The latter was almost entirely the already-fixed disastrous first
proposal (1.263 M). Rejected gradients are not fully wasted because their secants update BFGS before
rollback. The common-model fresh-seed first probe leaves only 9 native and 3 hierarchical rejects,
owning 0.323% and 0.113% of initial clades.

A loss-only forward costs about 0.31 of a full gradient on the current 200-family profile. With the
existing API, applying Armijo screening to every proposal then repeats the accepted candidates'
forward work inside the subsequent full-gradient call, adding roughly 31% of a likelihood pass per
trial; it must eliminate more than 0.31 later full-gradient calls per trial merely to break even.
After the reseed, first-step screening would pay that cost to protect a tiny rejected clade share.

That 31% is not an unconditional algorithmic tax. A new split forward/reverse API could retain the
accepted candidates' forward state, forward only a rejected subset at its shortened step, then run
the needed reverse passes without recomputing accepted forwards. This requires persistent batched
state, subset reverse masking/packing, and a benchmark of synchronization and memory overhead. It
also gives up the current BFGS secants from rejected points unless their reverse pass is still run.
It could be valuable when most clades reject, as in the obsolete old hierarchical first proposal;
the fresh seed reduces that case to a 0.113% rejected-clade share, where subset machinery has very
little first-step ceiling. A Wolfe test needs trial directional derivatives and likewise loses the
simple cheap-forward premise. A fixed-count surrogate search is another EM-like surrogate step,
not a certificate of observed-NLL descent.

The reseeded continuation was not step-traced, so no claim is made about later rejection counts.
The available evidence does not support dense BFGS replacement or line search as a dramatic route.

## Files and reproduction

- `fresh_seed.py`: native/hierarchical complete information and latest-pair reseeder.
- `reseed_adapter.py`: fail-closed in-memory production/hierarchy source adapters.
- `run_reseed_continuations.py`: fixed four-arm driver and common fresh audit.
- `test_reseed.py`: seed identity and source-integrity tests (3 passed).
- `reseed200.pt`, `reseed200.json`, `reseed200.log`: full results and audit.
- `smoke5.*`: five-family certified safety gate.
- `analyze_rebuild_reuse.py`, `rebuild_reuse_ceiling.json`: independent reuse ceiling.

Reproduce CPU checks:

```bash
PYTHONPATH=experiments/coleman_sol_20260906/dramatic/optimizer:. \
  .venv/bin/python -m pytest -q \
  experiments/coleman_sol_20260906/dramatic/optimizer/test_reseed.py
```

Reproduce the GPU experiment (exclusive local GPU required):

```bash
RAYON_NUM_THREADS=32 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=experiments/coleman_sol_20260906/dramatic/optimizer:. \
  .venv/bin/python \
  experiments/coleman_sol_20260906/dramatic/optimizer/run_reseed_continuations.py \
  --limit 200 --max-iter 200 --certify --audit \
  --out experiments/coleman_sol_20260906/dramatic/optimizer/reseed200.pt
```

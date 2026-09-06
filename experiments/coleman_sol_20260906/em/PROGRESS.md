# EM acceleration campaign

All scripts and artifacts created by this experiment agent are confined to this directory.
Production integration was performed and reviewed separately by the coordinating agents after the
experimental gates passed.

## Recovered evidence (2026-09-06)

- The four expected event counts, including the positive ghost-family contribution from survival
  conditioning, were validated against the production theta gradient and direct finite differences.
  All four were positive for all 500 sampled families at the common start, post-Adam point, and
  fitted point. A count pass took 5.7--6.6 s versus 5.4 s for the plain pass.
- `I_c - H` was PSD for every sampled family at all three points. The missing-information fraction
  (median / p90) was 1.72 / 2.49 at the common start, 1.04 / 1.49 post-Adam, and 0.837 / 0.915 at
  the fitted point. Long plain EM is therefore ruled out as an end game.
- One exact M-step from the common start reduced median distance to the saved optimum from 5.45 to
  1.90 log2 units; three production Adam passes reached 2.48. The recovered 20-pass run confirmed
  strong first-pass NLL progress but a slow tail. SQUAREM had 1,385 rejected extrapolations and only
  36% of families frozen after 20 passes.
- The recovered M-step used irreversible active-set pinning and reported three KKT violations at
  the fitted point. `mstep.py` enumerates all 27 active sets, which is exact and cheap in 3D.

## Current campaign

1. Validate `mstep.py` on the saved 500-family count tensors and randomized boundary cases.
2. Run production-faithful 200-family fits for the production baseline and 1--3 EM warm steps.
   Seed the existing BFGS phase with complete information and, where available without extra passes,
   BFGS updates from the EM secant pairs.
3. Compare wall time, actual count/gradient/Hessian phase time, clade-weighted pass cost, final NLL,
   and the unchanged projected-gradient certificate. Promote only a clear win to 500 families.

No fitted target-family optimum is used as a warm-start input; saved optima are evaluation-only.

## First 200-family production fits

All runs used the current `GpurecConfig.genewise_reference()` solver (float32 model, float64
accumulator), pi/Neumann 16 in the exact solver, the original `[1e-6,2]` box, projected-gradient
`1e-3` certificate, pruning `1e-6`, and the unchanged BFGS/Newton recipe after warm-up.

| run | prototype wall incl. parse/warm build | algorithm wall | actual gradient calls | full-clade equivalents | steps | NLL bits | certificate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adam3 baseline B | 44.02 s | 43.42 s | 38 | 16.977 | 25 | 613262.118661 | 200/200 |
| EM1 + complete information at old point | 37.69 s | 36.23 s | 30 | 13.919 | 22 | 613261.799116 | 200/200 |
| EM2 + scaled endpoint information + secant BFGS | 35.82 s | 34.44 s | 33 | 13.553 | 21 | 613261.800711 | 200/200 |

EM2 is 18.6% faster end to end and uses 20.2% less clade-weighted gradient work than the
instrumented baseline. Its prototype still pays a redundant warm-model build; an integrated path
would reuse the fit's resident parse/model.

The lower aggregate NLL is a basin change rather than uniform numerical improvement. Against
baseline B, both EM candidates materially change the same two families: COG0014_1 is about
0.302 bits worse while COG0057_2 is about 0.637 bits better. Every other per-family difference is
under 0.0017 bits. Both methods meet the identical gradient certificate; the 500-family test must
report all per-family changes above 0.01 bits before promotion.

## Promoted 500-family result

The exact-intermediate count hook was first checked on 20 families at the start and EM1 point.
It cost 1.003x / 1.004x an ordinary pass, returned identical NLLs, agreed with independently
evaluated production gradients to median relative L2 `8.0e-8` / `3.25e-7`, and returned positive
counts for every family.

| run | actual prototype wall | algorithm wall | actual gradient calls | full-clade equivalents | Hessians | steps | NLL bits | certificate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adam3 baseline | 108.33 s | 106.94 s | 50 | 17.273 | 2 | 35 | 1618461.966967 | 500/500 |
| EM2 hybrid | 91.51 s | 88.46 s | 35 | 13.660 | 1 | 23 | 1618461.994534 | 500/500 |

The prototype, despite redundantly building a separate warm model, is 15.5% faster end to end.
Kernel/optimizer wall is 17.3% lower and measured clade-weighted gradient work is 20.9% lower.
The aggregate NLL is 0.0276 bits worse. Four families differ by more than 0.01 bits, consistent
with alternative certified basins: COG0014_1 `+0.30143`, COG0057_2 `-0.63742`, COG0128_1
`-0.08839`, and COG0266_1 `+0.43374` bits (candidate minus baseline). Full theta pairs are in
`comparison_500_em2.json`.

## Integrated-path gate

The production opt-in path was then implemented by the coordinating agents: event counts are
returned through the model/streaming API from the existing intermediate VJP, and
`fit_genewise(warmup_method="em")` performs the two M-steps in the already-built model.

- Five-family smoke: 2.3 s, NLL 29914.400 bits, 5/5 certified.
- 200 families: 34.542 s driver wall, split into 4.572 s EM, 25.851 s Newton gradients,
  1.089 s Hessian, and 1.506 s rebuild; 21 steps, one Hessian, 200/200 certified at
  `|Pg|max=9.982e-4`, NLL 613261.798226 bits.
- The log reconstructs 29 Newton passes / 11.55 resident-model clade equivalents plus two full EM
  passes = 13.55, reproducing the prototype's 13.553. Median per-family theta discrepancy from the
  separate-model prototype is `6.2e-6`; aggregate NLL is 0.0025 bits lower.

This clears the full 5,124-family H100 promotion gate. No relaxed tolerance, altered pruning, or
target-optimum warm-start information is involved.

## One bounded EM3 follow-up

At the coordinator's request, one 500-family EM3 run tested whether a third full-population pass
earns its cost. It did, modestly but beyond the timing/work noise:

| run | actual prototype wall | algorithm wall | full-clade equivalents | Newton gradient time | steps / replans | NLL bits | certificate |
|---|---:|---:|---:|---:|---:|---:|---:|
| EM2 | 91.506 s | 88.463 s | 13.660 | 69.24 s | 23 / 9 | 1618461.994534 | 500/500 |
| EM3 | 87.206 s | 84.191 s | 13.299 | 61.20 s | 19 / 7 | 1618461.952767 | 500/500 |

EM3 is 4.7% faster in actual prototype wall and uses 2.6% less clade-weighted work. Its third count
pass costs 5.76 s, but 149 families are converged by Newton iteration 6 and the iteration-8 re-plan
leaves 175 families, versus 286 for EM2; this saves about 8.0 s in the Newton-gradient phase. EM3's
NLL is 0.0418 bits better than EM2, with no individual family differing by more than 0.01 bits.
Against Adam3 it selects the same four materially different basins already listed for EM2.

Conclusion: EM3 merits an integrated/full follow-up, although the already-running paired H100
promotion remains EM2 and should establish the primary full-data result first.

The integrated EM3 500-family gate then reproduced the prototype: 81.422 s driver wall,
16.863 s EM, 58.118 s Newton gradients, 1.156 s Hessian, 2.016 s rebuild, 19 steps, one Hessian,
and 500/500 certified at `|Pg|max=9.975e-4`. The production ledger measured 32 gradient/count calls
over 52,760,628 clades, or 13.3429 full-population equivalents. NLL was 1618461.956860 bits,
0.0041 bits from the prototype. Median per-family theta discrepancy was `5.72e-6`; only three
families exceeded 0.01 log2 units (maximum 1.01 in a flat coordinate). This clears EM3 for paired
full H100 runs.

## Completed full H100 campaign

All runs used all 5,124 families (including COG3676_X), the same H100 NVL, unchanged production
likelihood/precision/pruning/rate box, and the original `1e-3` projected-gradient freeze
certificate. All runs certified 5,124/5,124 with no premature drops.

| method | wall A / B | mean wall | clade equivalents A / B | mean work | fit steps A / B | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| warmed Adam | 512.718 s | 512.718 s | 17.0406 | 17.0406 | 48 | 39.74 |
| EM2 | 396.305 / 403.470 s | 399.888 s | 13.4031 / 13.3977 | 13.4004 | 28 / 49 | 23.59 / 23.57 |
| EM3 | 398.290 / 396.401 s | 397.345 s | 13.4659 / 13.4601 | 13.4630 | 65 / 65 | 23.42 / 23.02 |

Relative to warmed Adam, EM2 reduces mean wall time by 22.01% and measured clade-weighted work by
21.36%; EM3 reduces them by 22.50% and 20.99%. The fastest run is EM2A at 396.305 s, more than two
minutes below the historical approximately 520 s target and effectively tied with EM3B at
396.401 s. EM3's 2.54 s (0.64%) mean edge is smaller than the 7.17 s EM2 paired spread and comes
with 0.47% more clade-weighted work plus one mandatory full-population count pass. It is not a
clear speed win over EM2.

The common-model forward audits show repeatable aggregate objective improvements. EM2A and EM2B
are respectively 1.634510 and 1.630892 bits below Adam; the two EM2 fits differ by only 0.003619
bits total, at most 0.001310 bits for any family, and none by more than 0.01 bits. EM3 is about
1.764 bits below Adam; its two fits differ by 0.001206 bits total, at most 0.001742 per family, and
none by more than 0.01 bits. Neither result is per-family dominance: EM2 has 13 regressions and 12
improvements above 0.01 bits (largest `+1.4897` / `-2.3865`), while EM3 has 14 and 12. Both select
the same 25 material basin changes; EM3 additionally changes COG2352_1 by `+0.25155` bits and is
about 0.129 bits better in aggregate than EM2 through diffuse sub-0.01 changes.

Fresh cold gradient checks are inherently unstable under the existing float32/pruned evaluation:
the identical Adam theta certifies anywhere from 5,049 to 5,058 families across passes, and its
maximum projected gradient changes by as much as 0.001515 on repetition. EM2 gives 5,048--5,053
and EM3 5,052--5,055 in the same checks. Therefore the fair claim is **all 5,124 pass the unchanged
production freeze-time certificate**, not that any saved theta passes a fresh 5,124/5,124 common
gradient audit. See `QUALITY_REVIEW.md` for the independent interpretation.

Final recommendation: retain both `em_steps=2` and `em_steps=3`, with **EM2 as the default**. EM2
is simpler, uses less measured work, and is indistinguishable in speed at `n=2`. EM3 remains an
explicit option when its modest aggregate-NLL advantage is preferred despite the extra pass and
additional COG2352_1 basin regression. Long plain EM, SQUAREM, and hierarchical scaling remain
negative routes for this workload.

## Shared EM2 artifact for the reopened hybrid question

The earlier post-Adam hierarchy test does not by itself rule out changing coordinates after EM2.
`hybrid_shared_200_v2.pt` therefore captures exactly the shared warm trajectory needed for a fair
coupled-bound experiment, without running either downstream fit. Its V1 source
`hybrid_shared_200.pt` was generated through the
production count API from the common start on the first 200 families at `clade_budget=200740`:

- two calls over 1,491,100 resident clades each (exactly 2.0 full-clade equivalents);
- synchronized pass times 2.491 s and 2.110 s;
- NLL 739585.676965 then 620398.031349 bits;
- raw exact boxed-M-step theta1/theta2, float32-evaluated native theta0/theta1/theta2, native
  `g0/g1`, positive `N0/N1`, and per-family NLLs.

The artifact also contains the independently derived hierarchy quantities. With
V2 corrects V1's sub-float32 inconsistency: every `z`, Jacobian/gradient pullback, secant, and
endpoint information now uses the same float32-evaluated native points, while raw M-step outputs
remain separately available. With `z=(log2((D+T)/(1+L)),log2(T/D),log2(L))`, the accompanying derivation supplies
`J=dtheta/dz`; the artifact stores `g_z=J^Tg`, the latest
`z1-z0` / `g_z1-g_z0` secant, direct diagonal `I_c,z(theta2;N1)`, and a scalar-calibrated
safeguarded-BFGS seed. No native curvature matrix is reused. All 200 scalar and BFGS curvature
guards pass; the resulting minimum eigenvalue is `7.37e-4`.

For a matched native control, V2 stores a seed built from the same evaluated theta0/theta1 pair and
theta2 endpoint with the production free-at-both mask. It separately stores the faithful inline-EM
legacy seed built with raw CPU M-step theta1/theta2. Their relative Frobenius difference has
median/p90/max `2.36e-8 / 4.21e-8 / 9.76e-8` (absolute max `6.44e-6`); the transformed V2 seed
differs from its inconsistent V1 predecessor by at most `1.15e-7` relatively.

CPU/autograd validation is recorded in `hybrid_validation_v2.json`. Analytic Jacobian and gradient
errors are `1.11e-16` and `5.68e-14`; direct and chain-rule fixed-count Hessians agree with exact
autograd to at most `2.84e-13` on the real endpoint. Seven synthetic true boxed-M-step endpoints
exercise nine lower/upper active coordinates and agree to `1.11e-13`. The fixed-count Hessian is
diagonal in `z` everywhere: nonstationarity changes diagonal entries through the nonlinear
chain-rule correction but cannot create cross-terms for this particular factorized map. Omitting
that correction at the synthetic boundary endpoints changes a diagonal entry by as much as 69.18.
See `HYBRID_MATH_REVIEW.md` and the two reproducible scripts for the full contract and derivation.

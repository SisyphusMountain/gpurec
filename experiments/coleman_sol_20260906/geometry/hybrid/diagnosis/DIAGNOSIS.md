# Why the hierarchical hybrid did not reduce work

## Scope

This is a diagnostic replay, not a new optimizer. `traced_adapter.py` compiles native and
hierarchical-native continuations in memory from production source SHA-256
`7d398d3e596f150c7eb5fc36f5eccce3b94b7dd14561c337bbbf5d7363409cd5`. Exact, fail-closed callback
sites copy small optimizer state to CPU; they do not replace a production model method or alter the
likelihood, step, bounds, acceptance, rebuild, or certificate logic. Trace wall time is invalid for
performance comparison because the callbacks synchronize CUDA. Both serial traces started from the
same saved 200-family EM2 endpoint, took 21 proposals, and returned the fit's 200/200 certificate.

The raw state is `trace200.pt`; `trace200_complete.pt` adds the deterministically derivable phi and
phi-gradient to every native and hierarchical evaluation/proposal without another model call.
`analyze_trace.py` produces `trace200_analysis.json`, including every
pass and each family's last live pass. Counts below are from this diagnostic realization, whose
freeze decisions vary slightly with the existing FP32 gradient noise near `1e-3`.

## Main finding: a threshold interaction dominates the net difference

The hierarchy did not simply make slower geometric progress. Its accounting splits as follows:

| continuation gradient work | native clades | hierarchical clades | hierarchical − native |
|---|---:|---:|---:|
| live, changed point | 15,456,675 | 16,475,928 | +1,019,253 |
| live, same-theta remeasurement after rebuild | 1,857,617 | 575,524 | −1,282,093 |
| settled but still resident | 28,636 | 684,372 | +655,736 |
| total resident | 17,342,928 | 17,735,824 | **+392,896** |

Thus the hierarchical path did spend 6.6% more clade work evaluating genuinely changed live
points. Fewer rebuilds avoided 1.282 million clades of same-theta remeasurement, more than offsetting
that by itself. The remaining discontinuous scheduler cost then reversed the result.

At gradient pass 9 / Newton iteration 8, native froze 85 families owning 499,797 of 1,491,100
clades (33.5%), crossed `rebuild_frac=0.25`, and replanned immediately. Hierarchical froze 54
families owning 338,674 clades (22.7%), narrowly missed the gate, and evaluated two more full
resident passes. Those already-settled rows alone cost `2 * 338,674 = 677,348` clades, or 0.4543
full-clade equivalents. A later small episode added 7,024 clades. Native's only resident-settled
episode cost 28,636 clades. The differential dead work is 655,736 clades (0.4398 equivalents),
larger than the final 392,896-clade work gap.

An idealized, free “remove every settled row immediately” lower bound is 17,314,292 live clades for
native and 17,051,452 for hierarchy, a 262,840-clade (0.1763-equivalent) hierarchical advantage.
This is not an attainable timing prediction: real immediate removal pays another build and
same-theta remeasurement, precisely the categories separated above. It demonstrates that the fixed
25% rebuild discontinuity, not native-box handling, determines the sign of the measured work gap in
this realization. Clade-weighted median last-live pass was 11 for both arms; p90 was 15 native and
13 hierarchical, so the hierarchy did finish the large-clade distribution earlier despite its
larger changed-point total.

## The first hierarchical proposal is the geometric failure

The aggregate event totals hide a sharp first-step failure. Both endpoint seeds are calibrated with
the already-paid `theta0 -> theta1` / `g0 -> g1` EM pair and installed at `theta2`. The first Newton
gradient `g2` is then evaluated, but the adapter skips an ambient BFGS update because it has no
previous continuation point; consequently the first proposal does not incorporate the now-available
`theta1 -> theta2`, `g1 -> g2` endpoint secant.

| first proposal / following evaluation | native | hierarchical-native |
|---|---:|---:|
| radius-capped families / clades | 56 / 31,036 | 147 / 988,315 |
| rejected families / clades | 16 / 21,970 | 132 / 1,263,026 |
| next-gradient clade-weighted mean norm | 48.11 | 117.92 |
| aggregate actual NLL decrease, bits | +1065.66 | **−1441.47** |
| aggregate predicted decrease, bits | 2248.73 | 3239.91 |
| median actual/predicted ratio | +0.402 | **−0.389** |

The rejection machinery correctly rolls those families back and the hierarchy subsequently
recovers, but it spends one large-clade evaluation discovering that its lagged endpoint seed is
bad. This is the strongest evidence that better initialization/step control could help. A
point-consistent latest endpoint secant applied after measuring `g2` is a much narrower hypothesis
than replacing the bound solver; it must be tested for native and hierarchical coordinates from the
same paid information. No optimizer change is made in this diagnosis.

A subsequent common-model, forward-only probe validates that narrow hypothesis:

| first-step candidate | actual gain bits | existing-rule rejects | gain retained after rollback |
|---|---:|---:|---:|
| old native | +1065.66 | 16 | 1076.65 |
| old hierarchy | −1441.47 | 132 | 147.18 |
| latest-pair reseed, native | +1502.30 | 9 | 1531.29 |
| latest-pair reseed, hierarchy | +1466.51 | **3** | 1480.49 |
| carry latest pair into old native seed | −260.77 | 52 | 765.99 |
| carry latest pair into old hierarchy seed | +655.02 | 18 | 949.32 |
| old hierarchy, initial radius 0.5 | +1283.57 | **0** | 1283.57 |

The latest-pair candidate rebuilds and rescales `I_c(theta2;N1)` from `theta1 -> theta2` and
`g1 -> g2`; it is not merely another BFGS carry. That distinction is decisive here: carrying the
new pair into the old seed is materially worse in both coordinate systems, while a fresh reseed
turns the hierarchical opening into a strong step under the unchanged radius-two controller. The
radius-0.5 sensitivity independently shows that the old seed mainly failed through an over-large
opening step, but leaves more gain on the table than the reseed.

The candidate construction uses `g1` at the actual float32-evaluated `theta1`, each arm's already
paid traced `g2` at the common `theta2`, and the already-paid positive `N1` counts. `N1` defines the
same fixed-count EM surrogate curvature used by the current endpoint seed; obtaining posterior
counts at `theta2` would require an additional pass and is not assumed. Native uses its ordinary
free-at-both mask and hierarchy its ambient three-coordinate update, followed by their existing
mask, radius, feasible retraction/clamp, and prediction formulas. Independent gradients differ by
at most 0.001053 from FP32 evaluation noise. CPU/GPU eigensolver rounding makes the reconstructed
old proposal differ from its recorded GPU direction by at most about `4.4e-5`, which does not alter
the probe conclusion.

These are seven forward evaluations from one common model, not continuation fits. They demonstrate
first-step objective quality, not end-to-end work, certification, or runtime improvement. Testing
the reseed in the optimizer would be a separate authorized experiment.

## Bounds are not the missing win

The step-only hierarchical working set added just 11 coordinate faces over the whole run—3 loss
and 8 transfer, zero duplication—on 541 clades total (0.00036 full equivalents). No nonconverged
zero-direction, zero-ray, near-zero-ray, boundary-zero, or model-rounded zero step occurred.

Potential full-ray native-box crossings involved 49 hierarchical family-step events but only 6,891
clades, versus 28 events / 22,886 clades for native's component clamp. Even treating every such
event as avoidable would be orders of magnitude short of the 392,896-clade gap. A releasing
active-set QP would be mathematically cleaner than monotone face addition, but this trace supplies
no evidence that it would materially accelerate these 200 families. It remains relevant as a
robustness fix only if a future zero/near-zero diagnostic appears.

The cohort check strengthens this conclusion: only 28 of 200 families ever touched a native bound
in either fit, while 172 always remained interior. Those always-interior families account for
993,152 of the hierarchy's 1,019,253 extra changed-point clades (97.4%). The five largest penalties
are interior families 16, 169, 83, 101, and 177, each requiring three or four additional changed
evaluations except the other reported two-step cases. Bound handling cannot directly explain their
trajectories.

## Step control is active and could plausibly be improved, but is not indicted

The native-physical trust machinery constrains the hierarchical geometry more often:

| clade-weighted event | native | hierarchical |
|---|---:|---:|
| live proposal clades | 13,947,302 | 14,984,828 |
| radius-capped clades | 1,398,443 (10.0%) | 2,660,630 (17.8%) |
| any directional eigen trust-floor clades | 1,324,063 (9.5%) | 2,003,485 (13.4%) |
| pending-ratio clades rejected | 1,260,293 / 7,930,120 (15.9%) | 1,366,614 / 7,756,297 (17.6%) |
| radius-shrink clades | 1,596,830 | 1,526,644 |
| radius-grow clades | 476,005 | 1,398,346 |

For trust-limited positive eigenvalues, the clade-weighted adjustment multiplier was median/p90
1.73/3.51 native and 1.45/2.33 hierarchical. Over all positive eigenvalues even p90 remained 1.0;
negative curvature was only seven family-events in either arm. This is widespread mild step
limiting, not wholesale convexification of the hierarchical matrix.

BFGS curvature guards also are secondary. Excluding zero secants, native rejected 6 updates owning
21,036 clades and hierarchy rejected 16 owning 83,516 clades; the interior-only difference is about
62,000 clades, far below the 1.019-million changed-point penalty. Most zero secants are expected
same-theta remeasurements or rollback/rebuild state, not failed curvature observations.

The hierarchy's applied-ray fraction has clade-weighted p10 0.775 and minimum 0.0449, versus native
approximately 1.0 and 0.617. That makes a true constrained trust-region subproblem or a better
nonlinear physical-radius model a plausible future target. However, current acceptance evidence
does not show a bad model: hierarchical actual/predicted ratio p10/median/p90 was
−0.616/0.890/1.444, better than native's −0.948/0.695/1.278; it also shrank on slightly fewer clades
and grew on far more. More aggressive steps might reduce changed-point work, but could equally
increase the already slightly higher rejected share. The trace supports a focused experiment, not
a claim of recoverable speedup.

## Answer

The hybrid had no net gain for two distinct reasons:

1. its lagged first endpoint seed produced a severely over-optimistic first proposal, after which
   132 families owning 1.263 million clades were rejected; recovery then made the changed-point path
   cost about 1.02 million more clades, 97.4% of it on always-interior families;
2. although it avoided 1.28 million clades of rebuild remeasurement, an unlucky 22.7%-versus-25%
   rebuild threshold miss charged 0.66 million additional settled-row clades relative to native.

A better bound solver is very unlikely to change the outcome on this trace. Better step control is
plausible—first testing the latest paid endpoint secant, and only then a correct pullback-metric
trust-region subproblem. Ratios after the failed opening recover well, so a broad claim that the
quadratic model is generally poor would be unsupported. The clearest systems opportunity is to make replanning less
discontinuous or cheaper; testing that would change shared scheduler policy and must be evaluated
for native and hierarchical arms alike, rather than credited as a coordinate-only improvement.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/run_trace.py \
  --out experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200.pt \
  --summary experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200_summary.json

PYTHONPATH=. .venv/bin/python \
  experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/analyze_trace.py \
  --trace experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200.pt \
  --out experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200_analysis.json

PYTHONPATH=. .venv/bin/python \
  experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/cohort_analysis.py \
  --trace experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200.pt \
  --out experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200_cohorts.json
```

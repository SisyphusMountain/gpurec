# Why the EM2 + hierarchical continuation did not win

This report separates measurements from mathematical counterfactuals. The paired 200-family result
is the only end-to-end evidence; exact-metric calculations below optimize the saved local quadratic
and do not assert that the true likelihood would improve by the same amount.

## Conclusion

The negative result is not primarily a native-bound failure, and the current `native` trust metric
is not mathematically exact but is also not the main observed loss. The dominant actionable issue
is the first post-EM seed/step: the hierarchical seed is calibrated to the old EM0->EM1 pair, is
installed at the EM2 endpoint, and the normal first BFGS carry is suppressed. Therefore the already
paid-for EM1->EM2 / `g1->g2` pair is not used before the first proposal. That proposal is much too
aggressive for most families.

A better initial seed or first-step safeguard is plausible. A more exact physical trust solve has
localized upside and is worth at most a second-stage test. A more elaborate bound QP is unlikely to
recover the bulk work on this dataset.

## Measured evidence

### First-step model failure

The shared trace starts both arms at the identical float32 EM2 endpoint and records every actual
gradient, proposal, trust judgment, and resident-family clade charge.

| first proposal | median native step norm | predicted decrease sum | actual decrease sum | rejected | radius shrunk |
|---|---:|---:|---:|---:|---:|
| native | 1.151 | 2248.7 bits | +1065.7 bits | 16/200 | 48/200 |
| hierarchical/native metric | 2.000 | 3239.9 bits | **-1441.5 bits** | **132/200** | **148/200** |

For hierarchy, 143 first trials increased NLL; rejected rows alone lost 1588.6 bits. Rollback keeps
the endpoint and the rejected trial still supplies a BFGS secant, so the algorithm recovers: total
judged-step rejections are 156 for hierarchy versus 145 for native. Nevertheless, the first full
gradient evaluation is spent at bad points for most hierarchical families and their radii shrink.

The paired runs then show about 3.4% more gradient-clade work for hierarchical/native metric. The
work decomposition is important:

- changed-point evaluations: hierarchy +1,019,253 clades;
- same-point evaluations after builds: hierarchy -1,282,093 clades;
- frozen-but-resident work: hierarchy +655,736 clades;
- net: hierarchy +392,896 clades.

Thus the result combines genuine extra continuation evaluations with a batching threshold effect.
At one check hierarchy had frozen about 23.5% of resident clades, just below the 25% replan trigger,
and paid two more full-model passes; native crossed the trigger. Hierarchy also finished some
heavier rows earlier, which explains its fewer builds and why one wall-time ordering looked good
despite more total clade work. Wall time alone is not a clean geometry diagnostic here.

### Bounds are not the bulk cause

Across both traces, only 28 families ever touched a native bound. Of hierarchy's additional
changed-point clades, 993,152 of 1,019,253 (97.4%) came from 172 families interior in both arms.
The union of bound families accounts for only 26,101 extra changed-point clades. The five
clade-heaviest slower families identified from the trace were all interior and needed two to four
extra changed-point calls.

Within the hierarchical trace there were 2,216 live proposals, 121 rows on an authoritative KKT
face, 50 full proposed rays that crossed the box, and only 11 step-only working-face promotions.
An exact-full-metric direction, retaining only authoritative faces as equalities, still violated
the tangent cone for all 11 promoted rows. There was no observed unnecessary promotion caused
solely by the diagonal metric approximation. No zero/near-zero ray or nonconverged zero-direction
stall occurred.

### Endpoint quality is equivalent

Both hierarchical/native repeats reached the same basin as native to audit noise: total fresh NLL
differed by about -0.0025 to -0.0029 bits, maximum family difference was about 0.00136 bits, and no
family differed by 0.01 bits. This is a speed/work diagnosis, not a quality rejection.

## Mathematical diagnosis

### The current `native` metric is not an exact physical trust solver

Let `d` be a hierarchical-coordinate step and `J=d theta/d phi`. The linearized physical constraint
is

```
d' G d <= r^2,       G = J'J.
```

The exact convex trust step (after a generalized-eigenvalue curvature floor) has

```
(H + lambda G) d = -g
```

with one nonnegative multiplier chosen from a scalar root. The adapter instead diagonalizes `H`,
uses only `||J v_i|| = sqrt((V'GV)_ii)` for each curvature eigenvector, independently limits those
components, and finally scales the combined nonlinear ray. It omits the off-diagonal entries of
`V'GV`; even with a diagonal metric, three separate per-axis caps are not the same as one ellipsoid.

At the shared EM2 endpoint, `G` has median condition number 6.95 (maximum 8.33). In the seed-Hessian
eigenbasis, the maximum normalized off-diagonal is 0.719 median and the off-diagonal/diagonal
Frobenius ratio is 0.653 median. The omission is mathematically material.

Actual saved-step CPU counterfactuals tell a more restrained practical story. On 2,043 live rows
with no current face and with neither proposal crossing the box, an exact generalized-eigenvalue
floor plus exact full-`G` trust solve gives:

- current/exact quadratic-gain ratio: median essentially 1.0, p10 0.981, minimum 0.698;
- 38 rows below 90% of exact model gain and four below 75%;
- step 0 median 0.988 (minimum 0.930), step 1 median 0.978 (minimum 0.705);
- after the early trust-active steps, per-step medians are essentially 1.0.

The clade-heaviest slower families do not line up with the worst metric cases: their minimum gain
ratios are 0.959, 0.973, 0.977, 0.986, and approximately 1.0. So exact `G` could improve a few early
directions, but the trace does not support it as the main explanation for the extra calls.

The chart's nonlinearity in the radius is modest compared with the seed failure. In an endpoint
proxy calculation, the nonlinear native displacement is 1.020 times its linearized norm at the
median (maximum 1.059); the existing CPU64 retraction already corrects this by ray scaling.

### Direct complete information mostly cancels back to native locally

The fixed-count complete likelihood factorizes in the event-tree logits, so its Hessian is diagonal
in hierarchy coordinates. But at an interior fixed-count M-step endpoint its complete gradient is
zero, so the nonlinear Hessian correction vanishes and

```
I_complete,phi = J' I_complete,theta J.
```

Numerically, the direct hierarchical matrix and native complete-information pullback agree to
2.7e-9 relative Frobenius error at the median (9.3e-9 maximum). All 200 shared endpoints are
interior. Thus diagonalization is excellent algebra, but it does not create new local information;
an exact local Newton/trust method is coordinate invariant.

The actual seeds do not cancel after calibration and BFGS. The EM0->EM1 hierarchy chord is very
long (norm 5.41 median, 9.03 maximum). A scalar-calibrated direct-information prediction has median
secant-vector residual 0.587 and p90 1.054. BFGS enforces that one stale secant to numerical zero,
but changes the scaled information by 0.420 relative Frobenius norm at the median. The resulting
hierarchical seed differs from the pullback of the native seed by 0.520 median; even after optimal
scalar rescaling, median shape difference is 0.436. BFGS over a nonlinear chord is not chart
invariant, so the arms genuinely start with different approximate curvature—but the hierarchy arm
is trusting one distant direction rather than receiving extra observed-information content.

Most importantly, the first Newton gradient `g2` makes a newer EM1->EM2 secant available for free.
The current initialization does not fold it before proposing a step. CPU-generated candidates that
recalibrate endpoint complete information to this newest pair reduce median first-step norm from
1.151 to 0.800 in native and from 2.000 to 0.819 in hierarchy. Simply carrying the newest pair into
the old seed gives medians 1.248 and 0.988.

A common-model, forward-only evaluation of those saved candidates strongly confirms the narrow
first-step diagnosis:

| proposal | total actual gain | rejected by existing rule | retained gain after rollback rule | median actual/predicted |
|---|---:|---:|---:|---:|
| old native | +1065.66 bits | 16 | +1076.65 bits | 0.395 |
| old hierarchy | **-1441.47 bits** | **132** | +147.18 bits | -0.475 |
| newest-pair native | +1502.30 bits | 9 | +1531.29 bits | 0.924 |
| newest-pair hierarchy | +1466.51 bits | 3 | +1480.49 bits | 0.870 |
| old seed + ordinary carry, native | -260.77 bits | 52 | +765.99 bits | 0.459 |
| old seed + ordinary carry, hierarchy | +655.02 bits | 18 | +949.32 bits | 0.778 |
| old hierarchy seed, radius 0.5 | +1283.57 bits | 0 | +1283.57 bits | 0.804 |

This distinguishes full contemporaneous recalibration from merely not discarding the latest pair:
ordinary carry helps hierarchy but is clearly weaker, and is actively poor for native. A smaller
first radius is a robust fallback, but the latest-pair recalibration gets more retained gain while
remaining well calibrated. The newest native and hierarchy proposals are close; this evidence
supports a general EM endpoint-seed correction more strongly than a hierarchy-specific advantage.
The probes are not full fits and demonstrate neither fewer later passes nor runtime savings.

## Ranked low-cost alternatives

1. **Use the contemporaneous EM1->EM2 secant before the first proposal.** At the already required
   first endpoint gradient, rebuild endpoint `Ic(N1)`, scalar-calibrate it with `s=theta2-theta1`,
   `y=g2-g1`, and BFGS-fold that pair. Compare the same change in native and hierarchy; if both win,
   it is an EM warm-start improvement rather than evidence for hierarchical coordinates. The
   common-model probe strongly supports recalibration for both coordinates. Keeping the old seed
   and performing the ordinary first BFGS carry is not an adequate substitute.

2. **Safeguard only the first hierarchical radius.** A radius of 0.5 is an inexpensive diagnostic
   for the observed overshoot. A fixed small radius may slow families whose old model is good, so a
   production version should key the radius to newest-pair consistency or predicted model error,
   not impose 0.5 forever. This changes no likelihood or convergence tolerance.

3. **Solve the true linearized physical trust problem.** Whiten the free tangent by `G=J'J`, floor
   generalized curvature eigenvalues, and solve the one-dimensional trust-multiplier root. In three
   dimensions this is cheap and principled. Trace evidence predicts localized early improvement,
   not a wholesale recovery; test it after fixing the stale first seed.

4. **Use an exact tiny active-set bound solve only if future traces implicate bounds.** Enumerate at
   most eight current-face subsets, solve each equality-tangent trust problem, and check primal and
   multiplier signs. If a step hits a new face away from the current point, re-solve along that face
   instead of globally truncating all coordinates. This is mathematically cleaner than monotone
   face additions, but current bound work is too small to prioritize it.

5. **Treat replan scheduling separately from geometry.** Report changed-point, same-point rebuild,
   and frozen-resident clades, not only total resident work. A threshold just missed by one arm can
   reverse wall time without saying much about local convergence. Any replan change needs a paired
   native control.

## Diagnostics that would justify each change

- New seed / first radius: the common-model first-step probe is positive; the missing evidence is
  an end-to-end paired run measuring later changed-point and resident clades, builds, certification,
  and fresh endpoint quality.
- Exact `G`: trust-binding frequency, full-`G` versus approximate direction angle and quadratic
  gain, then actual NLL for the affected rows. A full run is warranted only if model gains translate
  to fewer changed-point clades.
- Bounds: box-hit versus radius-hit reason, promoted/released face counts, ray fraction, and clade
  work of ever-bound families. Nonconverged zero/near-zero steps would be a blocker; none occurred.
- Batching: live/moved, same-point-after-build, and frozen-resident clades per pass, plus the clade
  fraction at each replan decision.

## Reproducible artifacts

- `endpoint_geometry.json`: endpoint metric and seed calculations.
- `trace_geometry.json`: actual saved-step exact-`G` counterfactuals and working-face counts.
- `latest_secant_first_step.pt` and JSON sibling: old traced endpoints, newest-pair recalibrated
  native/hierarchy candidates, ordinary-carry controls, and old-hierarchy radius-0.5 candidate.
- `../../hybrid/diagnosis/first_step_probe.json`: read-only common-model evaluation of those
  candidates (outside fit timing and not an end-to-end run).
- `analyze_endpoint_geometry.py`, `analyze_trace_geometry.py`, and
  `build_latest_secant_candidates.py`: CPU-only generators; no production source is modified.

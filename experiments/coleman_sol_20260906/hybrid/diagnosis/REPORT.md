# Why the hierarchical hybrid lost the performance gate

## Conclusion

The strongest actionable problem is the **lagged first curvature seed and the
resulting oversized opening step**, not the hierarchical bound solver. Fresh
common-model likelihood probes substantially improve that opening by rebuilding
the seed with a gradient pair that is already available. They improve native
coordinates too, so they do not establish an intrinsic hierarchical advantage.
No production optimizer was changed, and no new end-to-end speedup is claimed.

This diagnosis reused the same two GPT-5.6 Sol workers. One recorded matched
trajectories and disaggregated work; the other independently analyzed metric
geometry and built CPU-only first-step counterfactuals. The coordinator checked
accounting independently and evaluated all counterfactuals on one fresh model.

## 1. The loss is mostly gradient work, not coordinate-algebra overhead

In the earlier order-balanced benchmark, hierarchy cost 1.318 s more on average.
The Newton-gradient phase accounted for 1.184 s of that difference. The small
matrix operations and CPU retraction therefore are not the main explanation.
This is a phase comparison, not an isolated microbenchmark of geometry overhead.

A new instrumented replay preserves the algorithms but synchronizes small state
tensors to CPU. Its **wall time is not a benchmark**; its work ledger is useful.
Both replayed fits passed their existing 200/200 freeze-time certificate.

| Continuation gradient work | Native clades | Hierarchical clades | Difference |
| --- | ---: | ---: | ---: |
| Live family, new parameter point | 15,456,675 | 16,475,928 | +1,019,253 |
| Live family, same point re-evaluated after rebuild | 1,857,617 | 575,524 | -1,282,093 |
| Already frozen, still resident | 28,636 | 684,372 | +655,736 |
| Total | 17,342,928 | 17,735,824 | +392,896 |

The identical two EM passes cancel in these differences; Hessian probes are
separate, and both fits used one scheduled exact refresh. New-point work includes
the first continuation evaluation and rejected trial points, not only accepted
steps. Same-point means exact equality with the previous evaluated parameters,
not that independent FP32 gradients are bit-identical.

There is a real 6.6% increase in new-point work. There is also a substantial
scheduling interaction: at iteration 8 hierarchy freezes families holding
22.7% of resident clades, missing the 25% rebuild threshold. Two more passes
over those frozen rows cost 677,348 clades. Native crosses the threshold then.
However, simply rebuilding earlier is not free: hierarchy's fewer rebuilds
already avoided 1.282 million clades of unchanged-point remeasurement. The
three categories must be considered together.

## 2. The first hierarchical step is severely over-optimistic

The EM trajectory is theta0 -> theta1 -> theta2. The original seed uses the
gradient pair at theta0/theta1 and installs the resulting matrix at theta2.
The first continuation evaluates g2, but its first BFGS carry is suppressed
because no previous continuation state has been installed. Thus the now-paid
theta1/theta2 gradient pair is unused before proposing the first step.

The older hierarchical secant has median length about 5.4; the fresh secant is
about 0.72. Applying a scalar calibration and a rank-two correction learned
across that much longer nonlinear coordinate chord need not give good local
curvature at theta2.

The observed opening reflects this mismatch:

- Hierarchy radius-caps 147/200 initial proposals, versus 56 native.
- It then rejects 132/200 proposals, covering 84.7% of dataset clades, versus
  16/200 and 1.47% for native.
- Its proposed points increase aggregate NLL by 1,441 bits despite a predicted
  improvement of about 3,240 bits. Rejection correctly restores those families;
  only about 147 bits of improvement survive the existing rejection rule.
- Native retains about 1,077 bits of improvement after the same rule.

Hierarchy recovers afterward, and the later quadratic predictions are much
better. The evidence does not say that every hierarchical step model is bad.

Diagonal complete-history curvature alone does not prevent this problem. It
is the Hessian of a frozen-count surrogate, not the observed likelihood. At
these interior EM M-step endpoints, it is essentially the congruence transform
of the corresponding native complete information. The candidate's substantive
difference comes from nonlinear secant calibration and subsequent step control,
not from acquiring previously unavailable curvature information. Moreover, the
native optimizer already solves full 3x3 systems cheaply.

## 3. Direct one-step probes support recalibration, not merely another BFGS update

We rebuilt complete information from the same already-paid counts N1 at theta2,
then used s=z2-z1 and y=J(z2)^T g2-J(z1)^T g1 for hierarchical scalar calibration
and safeguarded BFGS. Native received the equivalent native-coordinate treatment.
No posterior counts at theta2, additional gradient, or fitted optimum was used.
N1 remains the fixed-count surrogate source, not a claim to have current E-step
counts. The first continuation gradient g2 is already required by either fit.

All candidate points were evaluated on one fresh model, with the original
likelihood and box. These are **counterfactual first-step probes**, not fits.

| First-step construction | Rejected under existing rule | NLL gain retained after rejection, bits |
| --- | ---: | ---: |
| Original native | 16 | 1,076.65 |
| Original hierarchical | 132 | 147.18 |
| Fresh-pair recalibration, native | 9 | 1,531.29 |
| Fresh-pair recalibration, hierarchical | 3 | 1,480.49 |
| Old native seed plus fresh BFGS carry only | 52 | 765.99 |
| Old hierarchical seed plus fresh BFGS carry only | 18 | 949.32 |
| Original hierarchy, initial radius 0.5 instead of 2 | 0 | 1,283.57 |

Rebuilding/rescaling the seed is materially different from merely applying
another BFGS correction to the old one. The latter worsens the native opening.
Reducing the hierarchical radius also helps, corroborating an over-aggressive
opening, but obtains less immediate improvement than fresh-pair recalibration.
The refreshed hierarchical median physical step is 0.819, versus 2.0 originally.

This is strong evidence for testing improved initialization/initial step control.
It is not evidence that the new hierarchy beats a similarly improved native fit:
the refreshed native first step actually retains slightly more NLL improvement.
Only a paired continuation can determine eventual work, quality, and runtime.

## 4. A better bound solver is low priority for this dataset

Only 28 families touch a native bound in either traced fit. Families that remain
interior in both account for 993,152 of 1,019,253 extra new-point clades (97.4%).
The hierarchical working set adds only 11 coordinate faces, on family-step
events totaling 541 clades. Potential full-proposal box crossings total only
6,891 clades. There are no zero-step or near-zero-ray stalls.

A solver with multiplier-based releases, or enumeration of active subsets,
would be cleaner than monotone face addition. Projected steps followed by
free-subspace minimization are established approaches; see the original
[bound-constrained quasi-Newton paper](https://users.iems.northwestern.edu/~nocedal/PDFfiles/limited.pdf)
and [bound-constrained trust-region discussion](https://www.numerical.rl.ac.uk/media/people/nick-gould/ConnGoulToin96_mp.pdf).
But that mathematical improvement has very little direct exposure in this
trace. It is a robustness/other-dataset priority, not the best performance lead.
The rare-event counts do not constitute a universal upper bound on future
trajectory savings, but the always-interior cohort makes the conclusion strong.

## 5. The trust metric is approximate, but fixing it is not the first target

The correct *linearized* physical metric is G=J^T J. On an interior positive
curvature model, a physical trust-region minimizer satisfies

    (H + lambda G) d = -g,  lambda >= 0,
    d^T G d <= radius^2,
    lambda (d^T G d - radius^2) = 0.

On an active face, apply this system in its tangent subspace. Nonlinear endpoint
feasibility/radius still requires a consistent retraction or acceptance check.

The current implementation uses eigen-direction scales ||J v_i|| rather than
solving that coupled system. Metric off-diagonals are not small, so this is
indeed a heuristic, not the exact physical trust-region subproblem.

Nevertheless, offline solves on 2,043 recorded live interior/no-box-crossing
steps show limited practical room: the current/exact-G predicted-gain ratio
is essentially 1 at the median; only 38 steps are below 0.90 and four below
0.75. For the first step the median ratio is 0.988. Most of the heavy families
that require extra evaluations already have near-optimal steps for their
current quadratic model. Solving an inaccurate opening quadratic more exactly
does not repair its poor agreement with the actual likelihood.

## Recommended order of work

1. Test fresh-pair **recalibration** at the first continuation evaluation for
   both native and hierarchical coordinates, leaving the remaining algorithm
   unchanged. This is the strongest hypothesis and needs no extra gradient.
2. Separately assess a conservative initial radius or a model-agreement-based
   initialization of the radius; do not infer that larger or exact steps help.
3. Investigate reuse of unchanged-point gradients across rebuilds and a
   cost-aware rebuild trigger. These benefit both coordinate systems; they
   should not be credited as a coordinate-only speedup. Reuse needs validation
   against batch/pruning arithmetic and certificate semantics.
4. Only then consider the full-G trust solve or a releasing bound solver if
   new traces identify material remaining losses there.

The prior approximately 400-second full-Coleman result is unchanged. No fresh
seed was installed in production, and the first-step probes do not predict a
specific full-H100 runtime.

## Artifacts and checks

- [Step trace and event diagnosis](../../geometry/hybrid/diagnosis/DIAGNOSIS.md).
- [Independent geometry/metric review](../../em/hybrid_diagnosis/REPORT.md).
- [Exact work decomposition](work_breakdown.json), generated by
  [analyze_work.py](analyze_work.py).
- [Fresh-model first-step probe](first_step_probe.json), with per-family NLL
  vectors in the adjacent `.pt`, generated by [probe_first_steps.py](probe_first_steps.py).
- [Candidate construction](../../em/hybrid_diagnosis/build_latest_secant_candidates.py)
  was independently reviewed for own-point gradients, count source, masks,
  trust control, and prediction. CPU/GPU small-matrix rounding differs slightly;
  these are not claims of bit-identical GPU replay. Early NLL differences above
  are far larger than those numerical differences.

# Independent review of the post-EM hierarchical adapter

Date: 2026-09-06

## Gate decision

Approved for the bounded 200-family experiment. I found no remaining correctness blocker after
the step-only bound working set was added. The adapter is experiment-only, fail-closed against the
current production source, and the native control still calls the unmodified production function.

The qualification is that the bound solver is a conservative working-set heuristic, not an exact
active-set QP solver: faces are added monotonically during one step and are not released until the
next optimizer iteration. This can overconstrain a coupled step and make the hierarchy arm slower
than a fully solved tangent-cone subproblem, but it does not make the returned step infeasible. A
nonconverged zero direction or zero/near-zero feasible ray should therefore stop interpretation of
a screen as evidence against the coordinates and trigger either multiplier-based releases or
enumeration of the at most eight face subsets.

The completed 200-family screen has no such event in either hierarchy arm: zero ray, near-zero
ray, boundary-zero ray, nonconverged zero direction, and actual-native-zero counts are all zero.
The minimum ray fraction was 0.0448 for the native metric and 0.0002249 for the coordinate metric;
the latter is above the diagnostic's `1e-6` near-zero threshold but shows that a bound materially
truncated at least one step.

## Checks performed

- All 12 CPU geometry tests pass. They cover round trips, Jacobians and Hessians against
  autograd, tangent projection and face curvature, float32-effective bounds, trust caps, and the
  deterministic coupled lower-face failure in both metrics at exact and near-bound points.
- The source adapter compiles both private variants from production source SHA-256
  `7d398d3e596f150c7eb5fc36f5eccce3b94b7dd14561c337bbbf5d7363409cd5`; this equals the live
  source hash. Every edit site is asserted to occur exactly once.
- The V2 artifact is required by schema. Every arm starts at the same float32 `theta2` endpoint.
  Native receives the point-consistent native seed; hierarchy receives the direct hierarchical
  seed. Thus the intended coordinate/curvature continuation is the experimental difference, not
  a fitted target-family warm start or a different endpoint.

## Production-loop parity

- Native projected-gradient stopping, native bound certification, freezing, deferral, replan
  thresholds, verification, and final certification are untouched.
- Ambient hierarchical BFGS uses actual `(phi, J^T g)` pairs at their evaluated points. On a
  rejected step, `theta`, NLL, native gradient, `phi`, and hierarchical gradient all roll back to
  the previous accepted point. As in production, the valid secant from the evaluated rejected
  point is incorporated before rollback.
- Exact and targeted native Hessians are transformed with the full nonlinear chain rule at the
  point and gradient at which they were measured. The production refresh schedule and
  `since_exact` bookkeeping remain unchanged.
- KKT-blocking faces remain the authoritative mask for projected-gradient stopping and
  certification. The additional working faces affect only the current step model and retraction.
  The CPU64 retraction uses the model dtype's effective native bounds, preserves held native
  coordinates exactly, and returns an in-box endpoint. Its tangent displacement is used in the
  face quadratic prediction; the actual retracted coordinate chord is used by the next BFGS pair.
- The working-set repair addresses the original blocker: a native face with an inward gradient can
  nevertheless receive an outward coupled Newton direction. The old global ray then collapsed all
  three coordinates to `alpha=0`. The revised loop adds that directional face, resolves in its
  tangent, and retains feasible descent in the other directions. Monotone additions reset on every
  optimizer iteration. Lack of releases is the performance/optimality qualification above, not a
  feasibility or certification change.

## Work and timing accounting

- Production's continuation ledger charges every family resident in each model gradient call,
  including frozen rows awaiting a replan. The driver adds two shared EM calls, each charged the
  artifact's 1,491,100 clades for the 200-family run.
- `wall_seconds` is continuation-only. The common artifact records 5.929 s total generation time
  (including 4.601 s for its two EM reverse passes); add that common cost for an absolute
  prototype end-to-end wall time. It cancels in the three-arm continuation comparison.
- Hessian probes are intentionally excluded from `gradient_work`, matching production's ledger.
  Their number and measured seconds must accompany clade-equivalent comparisons. All three screen
  arms took one Hessian refresh.
- Each arm's parse/build and verification/replan activity is included in its measured wall and
  total `n_builds`. Different freeze/replan histories are legitimate algorithmic work differences,
  not missing charges. A fresh common-model audit remains preferable for likelihood comparisons,
  because freeze-time certificate NLLs can differ at float32 noise scale.

## Observed 200-family gate result

All arms certified 200/200 with the original native projected-gradient tolerance. Continuation
wall / shared-EM-inclusive gradient work / builds were:

| arm | wall (s) | full-clade equivalents | builds | NLL (bits) |
|---|---:|---:|---:|---:|
| native | 26.025 | 13.5207 | 10 | 613261.801129 |
| hierarchy, native metric | 25.433 | 13.9118 | 6 | 613261.798351 |
| hierarchy, coordinate metric | 26.402 | 14.0257 | 8 | 613261.813913 |

The hierarchy/native-metric arm is about 2.3% faster in this single continuation run despite about
2.9% more gradient-clade work, plausibly because it performed four fewer builds. This is a valid
gate pass, not yet a decisive speed claim; paired repeats and a common fresh likelihood/gradient
audit are needed. The coordinate metric is not favored by this screen.

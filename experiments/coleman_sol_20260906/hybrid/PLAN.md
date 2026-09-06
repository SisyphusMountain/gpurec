# Follow-up: EM endpoint plus hierarchical geometry

Status: bounded screen complete; see [RESULTS.md](RESULTS.md). The tested
implementation did not pass the performance gate for 500/full-H100 promotion.

The user's distinction is correct: the previous hierarchical trials started
after Adam and did not test the hybrid. The earlier negative conclusion applies
only to those trials, not to post-EM hierarchical optimization.

Reuse the same two GPT-5.6 Sol workers. No new optimizer variants or production
changes are accepted without the matched experiment below.

## Matched design

- Save one EM2 trajectory from the standard common start: theta0, theta1,
  theta2, native gradients g0/g1, latest counts N1, NLLs and charged work.
- Derivative pairs use the actual FP32-evaluated points (stored as FP64 for
  coordinate algebra), not the unrounded CPU M-step endpoints. Retain the raw
  endpoints and original seed for a numerical-difference check. Compare the
  native adapter against unmodified continuation with the identical saved seed.
- Start both continuations at exactly the same model-dtype theta2.
- Native control: use the evaluated-point-consistent native EM2 curvature seed
  with the unmodified production BFGS/Newton callable. The legacy raw-CPU-point
  seed is retained for comparison (maximum relative difference 9.76e-8).
- Hierarchical arm: z=(u,v,w), with u=log2((D+T)/(1+L)), v=log2(T/D), w=log2(L).
  Build diagonal complete-history curvature directly at z2 using N1; calibrate
  with s=z1-z0 and y=J(z1)^T g1-J(z0)^T g0, then safeguarded ambient BFGS.
  Do not transplant the native BFGS matrix.
- Preserve likelihood, precision, pruning, native rate box, native projected
  gradient threshold 1e-3, Hessian refresh every 15 Newton steps, rejection
  policy and clade-based replanning. Charge shared EM work to both arms.
- Exact hierarchical Hessians must include the gradient-times-map-curvature
  term: H_z=J^T H_theta J+sum_i g_theta_i Hess_z(theta_i).

## Coupled bounds

Native outward-active bounds remain authoritative. Their hierarchical tangent
normals are the corresponding rows of J. With tangent projector P, correct the
ambient curvature by subtracting sum_active g_theta_i Hess_z(theta_i) before
forming the tangent step model. This is the Hessian of the fixed-native-face
retraction, not an assumption that z has independent coordinate bounds.

Map a tangent proposal back to theta while explicitly holding current active
native coordinates fixed. Truncate its ray to a feasible endpoint if a new box
face or the original native-displacement trust budget would be violated. Only
roundoff is clipped. Predict reduction in the scaled tangent proposal; use the
actual lifted z endpoint chord for ambient BFGS secants. These are different
objects on a curved active face.

Independent review exposed a necessary working-set repair: a current bound
whose gradient points inward can nevertheless receive an outward coupled
Newton direction. Truncating that whole ray would discard feasible motion in
the other coordinates. Before retraction, add such directionally violated
current faces to a step-only working set and recompute the tangent model (at
most three added coordinates). Hold these faces in the retraction as well.
This mask must never replace the original native projected-gradient mask for
stopping or certification. Record zero-direction and near-zero-ray diagnostics
to distinguish optimizer behavior from an implementation stall.

Two explicitly named trust metrics are screened: the primary native metric
uses physical eigen-direction scales ||Jv|| and the original native displacement
radius; a coordinate-metric sensitivity arm uses unit scales and a tangent
coordinate norm cap. Both preserve the native rate box, numerical radius
settings, ratio policy, objective, and stopping rule. They are different trust
metrics, not purportedly identical algorithms.

`check_bound_geometry.py` independently checks the projected Hessian against
autodiff of the actual fixed-face retraction on all eight active masks.

## Gates and ownership

1. EM worker: shared 200-family endpoint artifact and independent transformed
   count/gradient/curvature checks. GPU lease only for artifact generation.
2. Geometry worker: experiment-only production-continuation adapter, CPU tests,
   then a tiny CUDA smoke. No approximate hand-written replay without scheduled
   exact Hessians. No production source edits.
3. Coordinator: independent constrained-geometry validation, adapter/control
   review, sequential 200-family comparison, then 500/full-H100 promotion only
   if work, likelihood, feasibility and convergence justify it.

No claim that a negative result excludes every post-EM reparameterization.

## Validation before the 200-family screen

- Shared V2 artifact SHA256:
  `67ade854b115bab698f092e8588283a64720c1aa81a0a29a382984ef2c378b78`.
- Independent 256-case face/retraction autodiff test: maximum curvature
  discrepancy 4.0e-15.
- Twelve adapter CPU tests, including the deterministic coupled-bound stall
  repair, pass.
- Five-family, 20-pass refresh smoke: all three arms execute one exact Hessian
  refresh at step 15, with finite parameters/curvature and identical charged
  gradient-clade work. This is a functional smoke, not a timing comparison.
- `audit_shared.py` performs a fresh common-model evaluation of every saved arm
  and repeats native parameters to measure gradient arithmetic variation;
  audit time is separate from fit timing.

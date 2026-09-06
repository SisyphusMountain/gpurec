# Post-EM hierarchical-coordinate adapter design

## Question and fair comparison

The earlier negative coordinate experiment started after Adam, so it does not answer whether the
hierarchical event tree helps after the much better EM2 warm start. The bounded experiment should
compare two continuations from one logically identical EM2 endpoint:

- same `theta0 -> theta1 -> theta2`, count passes, NLL values, and charged EM work;
- native continuation seeded by its current scaled endpoint complete information;
- hierarchical continuation seeded independently in `(u,v,w)` as described below;
- same model likelihood, `[1e-6,2]^3` native rate box, trust/acceptance policy, Hessian refresh 15,
  pruning, freezing/replanning, and native projected-gradient tolerance `1e-3`.

A replayed shared artifact is acceptable for the first post-warm diagnostic only if both arms are
charged the identical recorded EM2 work. Any promoted end-to-end result must execute and pay for
EM2 itself. No fitted optimum or fitted-optimum Hessian is an input.

## Coordinate algebra

Let `F(x)=log2(1+2**x)` and

```text
u = log2((D+T)/(1+L)),   v = log2(T/D),   w = log2(L).
```

The inverse used for every model evaluation is

```text
theta_D = u + F(w) - F(v)
theta_L = w
theta_T = u + F(w) + v - F(v).
```

With `t=2**v/(1+2**v)` and `l=2**w/(1+2**w)`, the Jacobian
`J=d theta/d phi`, with theta rows `(D,L,T)` and phi columns `(u,v,w)`, is

```text
J = [[1, -t,   l],
     [0,  0,   1],
     [1, 1-t,  l]].
```

Thus a native gradient becomes `g_phi=J.T@g_theta`. For an exact native Hessian at the same point,
the full chain rule is

```text
H_phi = J.T @ H_theta @ J
        + diag(0,
               -ln(2) t(1-t) (g_D+g_T),
               +ln(2) l(1-l) (g_D+g_T)).
```

The gradient-dependent diagonal term is required away from stationarity. Omitting it would make an
exact native Hessian refresh inexact in the new coordinates.

For positive ghost-augmented counts `(N_S,N_D,N_L,N_T)`, define `N_DT=N_D+N_T`,
`N_SL=N_S+N_L`, `N=N_DT+N_SL`, and `b=2**u/(1+2**u)`. The fixed-count complete NLL separates into
three binomials, giving the exact diagonal complete information

```text
I_c_phi = ln(2) diag(N b(1-b), N_DT t(1-t), N_SL l(1-l)).
```

At the shared EM2 endpoint, build this matrix from `counts1` and `phi2`. Transform the already-paid
native gradients at their own points,

```text
q0 = J(phi0).T g_theta0,   q1 = J(phi1).T g_theta1,
s = phi1-phi0,             y = q1-q0,
```

then scale `I_c_phi(phi2;counts1)` by `(s.y)/(s.I_c_phi.s)` when numerator and denominator are
positive and finite and apply the safeguarded hierarchical BFGS secant. Do not transform or reuse
the native endpoint matrix: the candidate being tested is the diagonal hierarchical complete
geometry plus a hierarchical-gradient secant.

## Native box as a curved feasible set

An independent `(u,v,w)` box is wrong: the three native rate faces are coupled curved surfaces in
phi. Keep the authoritative feasible set

```text
C = {phi : lo <= theta(phi)_D,theta(phi)_L,theta(phi)_T <= hi}.
```

At a current point, identify the same blocking faces as production in native variables:

```text
fixed_k = (theta_k at hi and g_theta_k < 0)
       or (theta_k at lo and g_theta_k > 0).
```

If `A` contains the corresponding rows of `J`, an active-set trial direction satisfies `A d=0`.
Let `P` be the Euclidean orthogonal projector onto `null(A)` (only eight native active masks exist).
The trial retraction below holds each blocking native theta coordinate constant. Its exact local
pullback curvature is the Lagrangian curvature

```text
B_corr = B_phi - sum_(k fixed) g_theta_k Hess_phi(theta_k).
```

Indeed, differentiating `theta_k(R(phi,sd))=theta_k(phi)` twice gives
`grad(theta_k).R''=-d.T Hess(theta_k)d`, which is precisely the subtracted term. With no blocking
face this reduces to the full ambient transformed curvature. As in the native control, first
convexify the full `B_corr`, with eigen-direction scales measured in native units as `||J v||`, and
only then reduce the solve to `range(P)`. Projecting before the eigendecomposition would change a
free direction through its coupling to a fixed one and would not preserve the native control's
ordering. Adding `I-P` makes the projected 3x3 solve nonsingular without changing its zero-normal
solution. With all three coordinates blocked, the step is zero. This treats an outward-active
inequality as its local equality face while leaving inward motion and inactive constraints
available.

The gradient-sign mask alone is insufficient for a coupled step: a face can have an inward native
gradient while off-diagonal curvature makes `Jd` point outward. The step therefore has a secondary
working set, distinct from the authoritative KKT/projected-gradient mask. Starting from the latter,
it promotes any currently touched lower face with `(Jd)_k < 0` or upper face with `(Jd)_k > 0`,
rebuilds the Lagrangian-corrected tangent model, and repeats for at most the three native faces.
Only this step and its retraction use the augmented mask; stopping, freezing, and certification do
not. Zero/near-zero applied steps with native `|Pg| >= tol` remain explicit diagnostics because
monotone face promotion can conservatively overconstrain a rare coupled corner.

The primary matched-budget arm uses native physical scaling: an eigenvector `v` has scale `||Jv||`,
the curvature floor is `mu ||Jv||^2`, its directional trust term is `|g_v| ||Jv||/r`, and the
retracted endpoint obeys `||theta_new-theta|| <= r`. A single coordinate-metric sensitivity arm is
also legitimate: unit eigen scales and `||d||_phi <= r`, while retaining the same native feasible
box and retraction. The adapter records which metric was used; native is the default.

Every trial is made exactly feasible by a ray-endpoint retraction. For a proposed tangent `d`, test
the nonlinear endpoint `theta(phi+alpha*d)`, explicitly overwrite the currently blocking native
coordinates with their current face values, and require both the original native box and the
selected trust metric. If `alpha=1` fails, 32 CPU-float64 bisections select a feasible endpoint.
Only endpoint feasibility is claimed because a nonlinear phi ray need not cross native faces
monotonically. The final clamp corrects roundoff only, at the bounds represented in model dtype.

```text
theta_trial(alpha) = phi_to_theta(phi+alpha*d)
theta_trial(alpha)[fixed] = theta(phi)[fixed]
R(phi,d) = theta_to_phi(theta_trial(alpha_feasible)).
```

This is the identity for an interior non-crossing step. It scales the tangent proposal instead of
applying a general native clamp/projection. Under the primary native metric, the trust radius caps
the actual native endpoint displacement; under the sensitivity metric it caps the scaled tangent
norm in phi. Because the corrected tangent model
is the Hessian of the retracted objective with respect to that initial tangent variable, predicted
decrease must use the applied `alpha*d`, not the lifted chord `phi_new-phi` (whose second-order normal component would
double-count face curvature). The actual lifted chord is used for the ambient secant update. The
next ordinary model pass supplies actual NLL; the existing ratio update and actual-NLL
rejection/rollback remain unchanged.

Convergence is never tested with `|g_phi|`. Decode to native theta, retain the ordinary native
gradient, and call the existing `project_rate_gradient_`; freezing and certification still require
`max|Pg_theta| < 1e-3`.

## Curvature update and active faces

Keep `B_phi` as an ambient approximation to the smooth observed objective `f(theta(phi))`. Use
ordinary safeguarded BFGS with the actual lifted endpoint chord `s=phi1-phi0` and transformed
gradients `y=gphi1-gphi0`, each evaluated at its own point. A clamped optimizer path does not make
the ambient objective nonsmooth, so these remain genuine phi-space secants; do not replace them by
component masks for `u`, `v`, or `w`. Apply the known active-face/Lagrangian correction only while
constructing the constrained step model. Exact refreshes reset accumulated approximation error.

On refresh 15, call the existing `_analytic_hessian_blocks` at decoded native theta and transform
each refreshed block with the exact chain rule above, using the native gradient measured at that
same evaluated point. Continue using the same eigenvalue convexification and refresh schedule in the
reduced tangent solve.

## Experiment-only adapter API

Create `hybrid/hierarchical_adapter.py` with pure tensor operations:

```python
class HierarchicalAdapter:
    encode(theta) -> phi
    decode(phi) -> theta
    jacobian(phi) -> J
    gradient(phi, g_theta) -> g_phi
    hessian(phi, g_theta, H_theta) -> H_phi
    complete_information(phi, counts) -> B_phi
    blocking(theta, g_theta, lo, hi, eps) -> mask
    tangent_projector(phi, blocking_mask) -> P
    step_curvature(phi, g_theta, B_phi, blocking_mask) -> (P, B_face, g_face)
    retract(phi, theta, delta, blocking_mask, lo, hi) -> (phi_new, theta_new, chord, hit_face)
    carry_bfgs(B, phi0, q0, phi1, q1) -> B_new
```

Add CPU tests in `hybrid/test_hierarchical_adapter.py` for round-trip/Jacobian/Hessian finite
differences, the diagonal complete-information identity, feasibility of random retractions, tangent
constraints `J_active@P=0`, the Lagrangian face-Hessian identity against second differences of the
actual retraction, native projected-gradient stopping, and ambient BFGS secants across face steps.

The current production optimizer has no coordinate seam and its warm-up/continuation are one
function. Do not monkeypatch global model methods. Before copying the large engine, the smallest
clean experimental seam would be a private continuation function receiving an `OptimizationState`
(`theta2`, EM trace/counts/gradients, curvature source, timers/work ledger) plus the adapter above.
If core refactoring is not authorized, make an experiment-only snapshot of the continuation and
record the exact source hash it mirrors. The required substitutions in that continuation are:

1. model calls decode phi to theta; keep both `g_theta` and `g_phi`;
2. best/final theta, NLL, bounds, native `Pg`, freeze, replan, and certificate logic stay native;
3. previous-state/BFGS pairs use actual phi chords; trust radii and predicted decrease use the
   initial tangent proposal of the retraction;
4. step construction uses the projected Lagrangian curvature and retraction above;
5. exact Hessian refreshes use `_analytic_hessian_blocks` followed by the full chain rule;
6. rollback restores both native theta and phi state; rejected evaluated pairs are treated exactly
   as in production;
7. all existing gradient/Hessian/build/count ledgers are retained, including resident frozen clades.

The first GPU gate should be a 20-family identity/trajectory smoke, followed by one bounded
200-family post-EM2 comparison. Stop if hierarchical charged work is already worse with no NLL or
tail advantage; promote only on unchanged certification and no new large-family failure.

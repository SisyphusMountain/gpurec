# Mathematical review of the integrated EM warm-up

This note audits the opt-in event-count API and the two- or three-step bounded EM warm-up as implemented on
6 September 2026. It distinguishes exact identities of the latent model from guarantees that are
only approximate in the pruned, finite-precision solver.

## Why the returned adjoints are ghost-augmented counts

Write `lambda_k = log2(p_k)` for `k in (S,D,L,T)`. For an observed-family history `h`, its
rate-dependent weight is proportional to

```text
2 ** sum_k n_k(h) lambda_k.
```

Let `A_y` be the total weight of histories compatible with the observed family, let `e` be the
extinction probability of an unobserved family, and let `q = 1-e` be survival probability. The
conditioned likelihood is `L_y=A_y/q`. The geometric identity

```text
A_y / q = A_y sum_(m>=0) e**m
```

augments one observed family with `M` independent extinct ghost families. In the normalized latent
posterior, `P(M=m)=q e**m`, so `E[M]=e/q`; conditional on being a ghost, a history has the usual
history distribution conditional on extinction. Therefore

```text
N_k = E_y[n_k] + (e/q) E_ext[n_k] >= 0.
```

Differentiating the negative base-2 log likelihood while temporarily treating the four `lambda_k`
as independent gives

```text
a_k := d ell / d lambda_k = -N_k.
```

Equivalently, the numerator contributes `-E_y[n_k]`. Since production represents survival as
`q=1-e`, differentiating `+log2(q)` contributes `-(e/q) E_ext[n_k]`, exactly the ghost term.

The implementation follows this calculation. The explicit `log2(q)` derivative is included in
`q_E`; the implicit E fixed-point adjoint is solved once; and the final parameter VJP is then asked
for adjoints of `log_pS_r`, `log_pD_r`, `log_pL_r`, and `mt_r` in the same call that returns the
ordinary theta gradient. Thus extinction and survival normalization are not omitted. Transfer has
the form `mt = log_pT +` a theta-independent recipient term, so summing the `mt` adjoint over its
species dimension is the adjoint of the one family-level `log_pT`. Negating the four adjoints gives
the buffer order `(S,D,L,T)`.

Finally, `lambda` is the four-way softmax of logits `(0,theta_D,theta_L,theta_T)`. Its Jacobian folds
the independent-log-probability adjoints into

```text
g_k = a_k - p_k sum_j a_j = p_k N_total - N_k,  k in (D,L,T).
```

This identity is both the mathematical check on the count interpretation and the test used by the
campaign. It also shows why the speciation count is needed even though speciation has no fitted
logit.

## Exact bounded M-step and its monotonicity

For fixed positive-total expected counts, the rate-dependent complete-data NLL in bits is

```text
R(theta | N) = N_total log2(1 + sum_k 2**theta_k) - sum_k N_k theta_k,
```

where the last sum is over `D,L,T`. Its gradient and Hessian are

```text
grad R = N_total p - (N_D,N_L,N_T),
H_R = ln(2) N_total [diag(p) - p p^T].
```

On finite positive rate bounds all four event probabilities are positive, so this Hessian is
positive definite in the three fitted logits. The surrogate therefore has a unique box-constrained
minimum.

For an active set `B` pinned at log-rate bounds `b_k`, define

```text
c = 1 + sum_(k in B) 2**b_k,
A = N_S + sum_(k in B) N_k.
```

Every free stationary coordinate is

```text
theta_k = log2(N_k c / A).
```

The implementation enumerates all `3**3=27` lower/free/upper states. It retains a state only when
its free solution lies in the box and the complete-log-likelihood score
`s_k=N_k-N_total p_k` obeys the KKT signs: `s_k<=0` at a lower bound, `s_k=0` when free, and
`s_k>=0` at an upper bound. These conditions are necessary and sufficient because the maximized
complete log likelihood is concave. Enumeration avoids irreversible-pinning errors caused by the
softmax coupling.

In exact arithmetic the selected endpoint cannot worsen the fixed-count surrogate: the old theta
is feasible and the M-step is its global optimizer. With exact posterior ghost counts, the standard
EM lower-bound argument then gives

```text
log L_y(theta_next) - log L_y(theta_old)
    >= Q(theta_next | theta_old) - Q(theta_old | theta_old) >= 0.
```

The statement remains true with a box because the current theta belongs to the same feasible box.

## Complete-information curvature seed

Holding the E-step counts fixed, the exact complete-data information in base-2 log-rate coordinates
is

```text
I_c(theta;N) = ln(2) N_total [diag(p(theta)) - p(theta)p(theta)^T].
```

For `K` warm-up steps (`K` is two or three), the integrated path uses the latest count set
`N_(K-1)` and evaluates this matrix at its M-step endpoint `theta_K`. This is exactly the Hessian of
the latest complete-data NLL surrogate at its minimum. It then calibrates the matrix along the
already-paid latest observed-gradient secant `s=theta_(K-1)-theta_(K-2)`,
`y=g_(K-1)-g_(K-2)` by `(s.y)/(s.I_c.s)` when both quantities are positive and finite, and applies
the safeguarded BFGS correction on coordinates free at both secant endpoints.

The unscaled `I_c` is exact for the fixed-count surrogate, but the final seed is deliberately an
inexpensive approximation to observed curvature. Marginalizing histories subtracts missing
information (a posterior score covariance), so `I_c` is not the observed Hessian. The scalar uses a
secant from the preceding EM transition, while `I_c` uses the newer counts, and the BFGS correction
only constrains that one direction. At an active solution only tangent/free directions govern the
local constrained problem. Consequently the seed is neither an exact Hessian nor a proven global
majorizer or descent guarantee; later acceptance, projection, convexification, and scheduled exact
Hessian refreshes remain authoritative.

## Numerical evidence boundary

The exact EM likelihood-monotonicity proof assumes exact posterior counts and exact evaluation of
the same likelihood. Production instead uses finite E/E-adjoint tolerances, prunes small adjoint
rows, stores the reconciliation state and extracted adjoints in the model dtype (FP32 in the
campaign), and rounds the double-precision CPU M-step endpoint back to model dtype. A float64 output
buffer preserves the extracted FP32 values; it does not retroactively widen the E-step or transfer
reduction. The active-set calculation itself is CPU float64 with a small KKT tolerance.

Therefore the code solves the surrogate defined by the extracted counts to CPU floating-point/KKT tolerance, but a strict
decrease of the separately evaluated FP32 observed NLL on every warm-up step is not a formal
software guarantee. There is no endpoint acceptance test. Monotonic observed NLL should be reported
as an empirical check, while the final unchanged projected-gradient certificate and matched NLL
audit determine fit validity.

## Audit-script review

`audit_fits.py` has the important basics right: it scores both theta tensors through one fresh model,
does not update warm starts, applies the production rate-box projection, records per-family NLL and
projected-gradient vectors, and keeps audit time outside either optimizer's timing.

Suggested hardening before treating its JSON as a durable audit record:

Coordinator follow-through: full-path/shape/finiteness validation, same-theta repeat,
nonnegative extrema, repeated-candidate auditing, and environment/solver metadata
were added. The EM3 job hashes every changed source file and the driver. The
16/16 audit budgets remain explicitly pinned to this campaign's recorded recipe.

1. Compare the complete normalized path lists, not only basenames; also validate theta shape and
   finiteness. Equal basenames from different directories can currently pass the order check.
2. Measure a same-theta repeat (ideally in reversed A/B order) and report its NLL/gradient noise.
   This separates real sub-millibit aggregate changes from FP32 atomic and reduction order effects.
3. Clamp headline `max_regression_bits` and `max_improvement_bits` at zero. As written, an all-better
   candidate reports a negative maximum regression, and conversely for an all-worse candidate.
4. Audit both repeated EM fits, or at least compare `em_full_a` with `em_full_b`, so optimizer
   repeatability/basin changes are not conflated with the Adam-versus-EM difference.
5. Record the audit solver options, resolved clade budget/batch plan, dtype, Torch/CUDA versions, and
   a complete source snapshot identifier. The paired job's current hash list omits count threading
   and driver files such as `_execution.py`, `model.py`, `dtl_fit.py`, and `run_genewise.py`.
6. The hard-coded `pi_iters=16, neumann_terms=16` matches this snapshot's one-tier exact production
   path, which explicitly makes the certificate use the same budgets. Deriving audit settings from
   the recorded run recipe would be safer than preserving that coincidence as the solver evolves.
7. If claiming more than the production pruned certificate, run and label the existing
   `--pruning-threshold 0` audit separately. The default `1e-6` result is intentionally a matched
   production certificate, not an unpruned mathematical stationarity proof.

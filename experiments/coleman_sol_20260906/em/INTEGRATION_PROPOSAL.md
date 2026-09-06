# Minimal production integration proposal

The measured candidates are two or three exact EM M-steps followed by the existing BFGS/Newton fit. It does
not change the rate box, solver settings, pruning threshold, trust policy, projected-gradient
tolerance, freezing, certification, or likelihood.

## Reverse-pass interface

Add an opt-in `need_event_counts` path through the genewise gradient call. In the existing final
`torch.autograd.grad` of `_e_adjoint_and_theta_vjp`, request gradients for the intermediate
`log_pS_r`, `log_pD_r`, `log_pL_r`, and `mt_r` tensors alongside `theta_req`. Autograd still returns
today's ordinary theta gradient, while these additional outputs are the free-event adjoints. Sum
the `mt_r` adjoint across recipient species to obtain the family log-pT direction. The resulting
`a[F,4]` includes the extinction and survival-normalization terms because it uses the same solved E
adjoint `wE` as today's theta VJP. Return both:

```python
counts = -a                                  # positive S,D,L,T counts
gradient = existing_theta_gradient           # bit-identical production path
```

This neither replaces nor duplicates today's theta-leaf E-step/VJP; it merely exposes gradients of
intermediates already present in its graph. Keep the normal requested-input tuple exactly unchanged
when counts are not requested. Return counts per batch through the streaming accumulator; do not use
a global sink or transfer each batch to CPU as the experiment hook does.

Validation gates before enabling it in a recipe:

- same-run folded versus ordinary theta gradient at the common start and first EM point, against
  the existing float32 same-code noise band;
- all four counts finite and nonnegative, including families with appreciable ghost contribution;
- one ordinary gradient versus count-gradient timing on 200 and 500 families.

## Warm-up in `fit_genewise`

Add an explicit warm-up choice rather than overloading `adam_steps`, for example
`warmup_method="adam"|"em"` plus `em_steps=2|3`; retain the old default until full-data validation
decides. For `em`:

1. At each evaluated `theta[j]`, request `(nll[j], g[j], N[j])` in one normal model pass and apply
   the exact boxed M-step to obtain `theta[j+1]`.
2. Reset the curvature to the fixed-count complete information at that M-step's endpoint,
   `B = ln(2) N[j]_total (diag(p(theta[j+1])) - p(theta[j+1])p(theta[j+1])^T)`.
3. From the second evaluation onward, calibrate that endpoint matrix in the latest available EM
   secant direction: `s=theta[j]-theta[j-1]`, `y=g[j]-g[j-1]`, and
   `scale=(s.y)/(s.B.s)` for positive finite numerator and denominator, otherwise one. Then apply
   the existing safeguarded `_bfgs_update(B,s,y,free)`. Resetting to endpoint information before
   this update is intentional: after three EM steps the final seed uses the latest `theta1→theta2`
   secant, not the much larger common-start pair.
4. After `em_steps` evaluations/M-steps, enter the existing Newton loop at `theta[em_steps]` with
   the final per-family curvature. The first Newton pass evaluates this endpoint; no warm-pass
   result is recomputed.

The box M-step should use either the validated 27-active-set solution (`mstep.py`) or the equivalent
monotone scalar solve. Irreversible pinning must not be used: it caused three KKT failures in the
recovered 500-family fitted-point test.

## Accounting and rollback

Expose warm count seconds and count passes next to `adam_seconds`. Count actual model clades for
each pass; a family frozen but still resident until re-plan still costs work. Preserve the old Adam
path as a one-key rollback. Do not add long EM, SQUAREM, or relaxed convergence settings: the
recovered measurements rule those out.

## Local choice between two and three steps

On 500 families, integrated EM3 took 81.422 s and 13.3429 full-clade equivalents versus the EM2
prototype's 91.506 s and 13.6601 equivalents (the integrated EM2 200-family gate reproduced its
prototype exactly). EM3's third count pass cost about 5.76 s locally but enabled a large early
freeze/re-plan and saved about 8 s of Newton-gradient time. It preserved 500/500 certification and
improved aggregate NLL by 0.0418 bits versus EM2, with no individual per-family difference above
0.01 bits. Therefore both values remain explicit for the paired H100 decision, with EM3 the local
leader; no further step-count sweep is warranted.

# EM2-to-hierarchical shared artifact and curvature derivation

## Artifact contract

`hybrid_shared_200_v2.pt` is the point-consistent artifact for downstream adapters. It derives from
the preserved GPU-generated `hybrid_shared_200.pt` V1 artifact, which contains the first 200
Coleman families from the common start evaluated
through the production `GeneReconModel.genewise_loss_vector_and_grad(..., event_counts_out=...)`
API with the reference float32 model / float64 accumulator and `clade_budget=200740`. It contains:

- native float32-evaluated `theta0`, `theta1`, and `theta2`, plus the raw exact-float64 boxed-M-step
  `theta1` and `theta2` separately;
- native observed gradients `g0` and `g1`, positive event counts `N0` and final `N1`, and both
  per-family NLL vectors;
- parse, model-build, and synchronized pass timings, family clade counts, and a two-call ledger;
- derived `z0,z1,z2`, `g_z0,g_z1`, the latest transformed secant, direct `I_c,z(theta2;N1)`, and
  its scalar-calibrated safeguarded-BFGS seed.

The artifact uses no fit and no fitted-target optimum. `g1` and the second NLL/count pass are
evaluated at the production float32 cast of raw exact `theta1`. V2 consistently uses that evaluated
point for `z1`, `J(z1)`, `g_z1`, and the transformed secant; it likewise uses the float32 endpoint
that a downstream optimizer receives for `z2` and its direct information. The raw-to-evaluated
discrepancies are at most `4.70e-7` and `4.74e-7` log2 units for theta1 and theta2.

Reproduce it with:

```bash
.venv/bin/python experiments/coleman_sol_20260906/em/generate_hybrid_shared.py \
  --species data/external/benchmarks/large_dataset_capacity/datasets/coleman/Section01.SpeciesTree/ReferenceTree.nwk \
  --families /tmp/claude-1000/-home-enzo-Documents-git-gpurec-gpurec/8392fbe3-5570-4d60-9656-16e4db97a7a9/scratchpad/families_all_local.txt \
  --limit 200 --clade-budget 200740 \
  --out experiments/coleman_sol_20260906/em/hybrid_shared_200.pt

.venv/bin/python experiments/coleman_sol_20260906/em/derive_hybrid_shared_v2.py \
  --input experiments/coleman_sol_20260906/em/hybrid_shared_200.pt \
  --output experiments/coleman_sol_20260906/em/hybrid_shared_200_v2.pt
```

## Coordinate derivatives

For native `theta=(d,l,t)=(log2 D,log2 L,log2 T)`, define

`z=(u,v,w)=(log2((D+T)/(1+L)), log2(T/D), log2(L))`.

With `F(x)=log2(1+2^x)`, the inverse map is

`d=u+F(w)-F(v)`, `l=w`, `t=u+F(w)+v-F(v)`.

Writing `q=sigmoid(ln(2)v)` and `r=sigmoid(ln(2)w)`, the requested Jacobian, with native rows and
hierarchical columns, is

```text
J = d theta / d z = [[1, -q,   r],
                     [0,  0,   1],
                     [1, 1-q,  r]].
```

Consequently `g_z=J^T g_theta`. For counts `(NS,ND,NL,NT)`, define `N=N_S+N_D+N_L+N_T`,
`N_DT=N_D+N_T`, `N_SL=N_S+N_L`, and `b=sigmoid(ln(2)u)`. The fixed-count complete NLL factorizes:

`Q(z)=N F(u)-N_DT u + N_DT F(v)-N_T v + N_SL F(w)-N_L w + constant`.

Its complete information at the true second EM endpoint is therefore constructed directly as

`I_c,z(theta2;N1)=ln(2) diag(N b(1-b), N_DT q(1-q), N_SL r(1-r))`.

No native-coordinate curvature is transformed or reused. Starting from this direct diagonal,
the seed uses only the latest transformed pair

`s=z1-z0`, `y=g_z1-g_z0`, `scale=(s.y)/(s.I_c,z.s)`

when both products are positive and finite, followed by the standard safeguarded BFGS update.
All 200 scalar calibrations and BFGS folds pass their curvature guards; scale min/median/p90/max is
`0.263/0.430/0.655/5.985`, and the final matrices have minimum eigenvalue `7.37e-4`.

V2 also provides a native-coordinate control constructed with the same evaluated theta0/theta1
pair, evaluated theta2 endpoint, latest native gradient pair, and the production free-at-both mask.
A separate field reproduces the current inline EM implementation faithfully, where CPU state keeps
the raw float64 M-step theta1/theta2 for its secant and endpoint information before copying the seed
and theta2 to float32. The corrected and legacy seeds differ by relative Frobenius median/p90/max
`2.36e-8 / 4.21e-8 / 9.76e-8` (absolute max `6.44e-6`). Thus the float64 seed discrepancy is less
than one float32 epsilon relatively, although the casts are not bit-identical (cast relative maximum
`1.33e-7`). This separation lets a native control and transformed
adapter share exactly the same evaluated endpoint/pair without pretending that the inline legacy
rounding convention is mathematically identical.

## Exact Hessian and boundary resolution

The direct fixed-count Hessian is diagonal at every finite `z`, not only at a stationary point.
Transforming the native Hessian requires the full nonlinear chain rule

`H_z=J^T H_theta J + sum_k g_theta,k Hessian_z(theta_k)`.

For this map the raw pullback is already diagonal by the same factorization, and the second term is
also diagonal:

`diag(0, -ln(2)q(1-q)(g_D+g_T), +ln(2)r(1-r)(g_D+g_T))`.

Therefore nonstationarity at a native-box constrained M-step endpoint can change the diagonal
relative to the raw pullback `J^T H_theta J`, but it cannot introduce cross-terms. Seven synthetic
true boxed-M-step endpoints exercise lower and upper active coordinates. Their raw-pullback
off-diagonal maximum is numerical noise, while omitting the diagonal correction changes an entry
by as much as about 69.2.

`validate_hybrid_shared.py` compares the analytic Jacobian and gradient pullback with autograd and
compares both the direct diagonal and full native-Hessian chain rule with the exact autograd Hessian
of `Q(z)`, for the real 200 endpoints and synthetic boundary endpoints. It also independently
reconstructs both native seeds. V2 results are stored in `hybrid_validation_v2.json`; the earlier
`hybrid_validation.json` remains the record for V1.

```bash
.venv/bin/python experiments/coleman_sol_20260906/em/validate_hybrid_shared.py \
  --artifact experiments/coleman_sol_20260906/em/hybrid_shared_200_v2.pt \
  --output experiments/coleman_sol_20260906/em/hybrid_validation_v2.json
```

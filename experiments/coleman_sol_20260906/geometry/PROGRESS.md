# Hierarchical event-coordinate experiment

## Scope and invariants

- Work is isolated to this directory; no production source is edited.
- The likelihood, pruning, warm-up, rate bounds, and convergence certificate remain unchanged.
- A family is converged only when the original-theta projected gradient has max norm below `1e-3`.
- Fitted optima may be used to score an experiment, never to construct its seed or step.
- Exact Hessians are evaluation data in the CPU screen and are not a proposed per-round input.
- GPU/H100 runs are deferred until the coordinator assigns a device.

## Coordinates and identities

Let

`u=log2((D+T)/(1+L))`, `v=log2(T/D)`, `w=log2(L)`, and
`F(x)=log2(1+2^x)`. Then

`theta_D=u+F(w)-F(v)`, `theta_L=w`, and
`theta_T=u+F(w)+v-F(v)`.

Writing `b=sigmoid(ln(2)u)`, `t=sigmoid(ln(2)v)`, and
`l=sigmoid(ln(2)w)` gives the binary event tree

`p_D=b(1-t)`, `p_T=bt`, `p_S=(1-b)(1-l)`, `p_L=(1-b)l`.

For fixed positive expected counts in `S,D,L,T` order, the complete NLL is

`Q=Ntot F(u)-NDT u + NDT F(v)-NT v + NSL F(w)-NL w + const`.

Its gradient is `(Ntot*b-NDT, NDT*t-NT, NSL*l-NL)` and its Hessian is exactly

`ln(2) diag(Ntot*b(1-b), NDT*t(1-t), NSL*l(1-l))`.

The inverse Jacobian (rows `D,L,T`, columns `u,v,w`) is

```
[[1, -t,   l],
 [0,  0,   1],
 [1, 1-t,  l]].
```

For the observed NLL, with theta gradient `g` and theta Hessian `H`, the exact transformed
Hessian is `J' H J` plus a diagonal map-curvature correction

`diag(0, -ln(2)t(1-t)(g_D+g_T), +ln(2)l(1-l)(g_D+g_T))`.

## CPU screen

`geometry_cpu.py` performs four checks using only saved CPU tensors:

1. inverse-map, Jacobian, and Hessian-transform checks against PyTorch differentiation;
2. count-derived gradient against the stored production gradient;
3. diagonal complete information against the transformed multinomial information;
4. observed geometry and inexpensive curvature seeds at the post-Adam point.

The seed candidates use no fitted-optimum information: raw complete information, a scalar
warm-up-secant calibration, a single native BFGS fold of each, a BB/BFGS control, and the
existing production curvature pulled into the new coordinates. Exact observed curvature is
included only as an oracle evaluation control.

The CPU step maps every proposal back to theta, projects against the actual theta bounds, and
applies the trust radius in theta space. It does not invent a rectangular `(u,v,w)` box. Its
trial distance is diagnostic only; actual NLL evaluation and acceptance are required on GPU.

### Result on the recovered 500-family tensors

The analytic round-trip, Jacobian, and Hessian checks have maximum absolute errors
`1.8e-15`, `1.1e-16`, and `8.9e-16`. Transforming the ordinary multinomial complete
information with the count-derived gradient agrees with the diagonal formula to `4.6e-13`.
At post-Adam, the count-derived hierarchical gradient agrees with the stored production
gradient to median relative error `4.0e-7` (maximum absolute difference `0.0024`, from the
independent float32 gradient evaluations).

Observed-Hessian indefiniteness at start/post-Adam/optimum is `82.2% / 9.0% / 0.2%` in the
hierarchy, versus the recovered log-rate result of `82% / 64% / 0.2%`. This is a useful
off-optimum reshaping, but the transformed Hessian's median relative drift from post-Adam to
the optimum is still `1.06`.

The post-Adam median max-coordinate distance to the saved optimum is `2.48`. Radius-two trial
points from non-oracle seeds score as follows. These are only geometric evaluations; no trial
NLL was available on CPU.

| seed | median / p90 distance | clade-weighted median | clades below distance 1 |
|---|---:|---:|---:|
| complete diagonal | 1.77 / 8.57 | 1.20 | 35.9% |
| scalar-secant complete diagonal | **1.48 / 8.13** | **1.00** | **50.5%** |
| complete diagonal + endpoint BFGS | 1.70 / 8.87 | 1.15 | 28.7% |
| scaled diagonal + endpoint BFGS | 1.62 / 8.73 | 1.03 | 48.8% |
| native BB + endpoint BFGS | 2.15 / 8.97 | 1.43 | 31.1% |
| pulled-back production curvature | 1.86 / 8.49 | 1.27 | 31.3% |
| exact local observed Hessian diagnostic | 2.45 / 8.49 | 1.92 | 8.4% |

All 500 warm-up secants gave a positive scalar calibration; its median was `0.343` and p90
`0.575`. This supports testing the scaled diagonal, but its actual-NLL result—not its distance
or Hessian resemblance—is decisive. In particular, the exact local Hessian diagnostic is not
an upper bound and performed badly under the nonlinear map, projection, and trust rule.

Reproduce with:

```bash
.venv/bin/python experiments/coleman_sol_20260906/geometry/geometry_cpu.py \
  --curvature /tmp/claude-1000/-home-enzo-Documents-git-gpurec-gpurec/8392fbe3-5570-4d60-9656-16e4db97a7a9/scratchpad/curv/s1_500.pt \
  --gradients /tmp/claude-1000/-home-enzo-Documents-git-gpurec-gpurec/8392fbe3-5570-4d60-9656-16e4db97a7a9/scratchpad/reparam/grads_500.pt \
  --counts /tmp/claude-1000/-home-enzo-Documents-git-gpurec-gpurec/8392fbe3-5570-4d60-9656-16e4db97a7a9/scratchpad/em/counts_500.pt \
  --radius 2 \
  --output experiments/coleman_sol_20260906/geometry/cpu_results.json
```

## GPU gate and fair comparison

Only candidates that improve the CPU direction/seed diagnostics without oracle data proceed.
The primary GPU comparison will start from the same post-warm-up theta and gradient as native
log-rate BFGS. Each round will count actual clade-weighted gradient passes, including rejected
trial evaluations and line-search evaluations. The hierarchy will use native gradient pairs and
the cheap count diagonal; it will not pull a nonlinear-coordinate seed built in log-rate space.
The final report will include NLL, theta projected-gradient certification, failures, and the
largest-family tail, rather than family-count convergence alone.

`gpu_replay.py` is a portable current-source driver. It imports only the sibling experiment's
portable count hook, not recovered `/tmp` scripts or tensors. It performs a shared production-
matched three-step Adam warm-up, times an ordinary and a count-producing post-Adam pass, and
then independently replays `log2`, `hier_raw`, and `hier_scaled`. Every evaluated model's actual
clades are charged, including frozen families before a rebuild and rejected trials. The hierarchy
projects coupled proposals through the original theta box and accepts them only through measured
NLL. A whole-population theta projected-gradient pass closes each run.

Example (run only after a GPU is assigned):

```bash
.venv/bin/python experiments/coleman_sol_20260906/geometry/gpu_replay.py \
  --species data/external/benchmarks/large_dataset_capacity/datasets/coleman/species_tree.nwk \
  --families FAMILY_LIST.txt --limit 200 --clade-budget 0 --rounds 20 \
  --runs log2,hier_raw,hier_scaled --rebuild-fraction 0.25 --trust 2 --trust-max 8 \
  --hessian-refresh 15 --hessian-pass-equivalent 7.7 \
  --output experiments/coleman_sol_20260906/geometry/replay_200.pt
```

### Local RTX 4090 result: first 200 Coleman families

The driver was first smoke-tested on 20 families, then run on the first 200 families with the
current source. The ordinary and count-producing post-Adam calls returned identical NLLs; their
maximum theta-gradient disagreement was `0.0021` in the independent float32 evaluations. The
count pass cost the same as the ordinary pass within run noise (`2.105 s` versus `2.111 s` in
the first 200 run). Recoverable caching-allocator OOM warnings occurred while choosing the
automatic clade budget, but no evaluated batch or family failed.

The bounded 16-round comparison was:

| run | charged passes incl. whole-pop certificate | gradient seconds | certified | NLL bits |
|---|---:|---:|---:|---:|
| native log-rate BB/BFGS replay | 16.725 | 37.05 | 139/200 | 613262.490 |
| raw hierarchical complete diagonal | 18.247 | 40.37 | 151/200 | 613265.249 |
| scalar-scaled hierarchical diagonal | **16.093** | **35.65** | **170/200** | 613267.165 |

Thus raw complete information is dominated. Scaling improves the approach and clade-weighted
freezing, but the 16-round endpoints are not comparable solutions because none is fully
certified. The current production baseline, run independently to completion on these families,
uses 16.977 gradient-clade equivalents, certifies 200/200, and has NLL 613262.119 bits.

The scaled hierarchy was then rerun for a bounded 40 rounds with stricter nonlinear-step
acceptance: every measured rise over `0.005` bits is rejected and shrinks the radius, regardless
of whether the local model predicted enough change for the ordinary trust-ratio test. It reached
188/200 frozen and 186/200 certified, at 16.100 charged passes and 33.54 gradient seconds. The
remaining 12 families own only 554 of 1,491,100 clades (`0.037%`), but the theta projected
gradient remains 1.90 and the NLL is 613263.485 bits, 1.367 above the converged production
baseline. Six families account for more than 0.05 bits each; the worst is COG0019_1 (+0.649 bits,
theta `[0.053, 1, 0.571]` versus the baseline `[-0.663, 1, 0.510]`). Several other failures sit
on the coupled `L=T=2` corner. This is a genuine certification/tail failure, not merely a median
distance issue.

The replay lacks production's one-time exact refresh at step 15. Portable support for a refresh
has been added but not GPU-run: at the scaled run's step-15 active set it would be charged roughly
`7.7 * 0.070 = 0.54` whole-population pass equivalents before subsequent work. Even an optimistic
total near 16.5 passes would be only a few percent below the 16.98-pass baseline and remains
unproven, whereas the independent two-EM-step experiment already reached 200/200 certification,
NLL 613261.801, and 13.553 pass equivalents. Consequently the coordinate-only candidate is not
promoted to 500 families, H100, or an EM combination. Its cheap approach gain does not justify
the less robust projected-bound tail.

The first bounded comparison is retained as `replay_200.pt`; the strict longer replay is
`replay_200_scaled_full.pt`. Both `.json` companions contain the per-evaluation accounting.

## Opt-in production count API

After the coordinate route was closed, the coordinator authorized a narrow integration in
`gpurec/api/_implicit_grad.py`, `gpurec/api/_execution.py`, and `gpurec/api/model.py`. The public
method now accepts

```python
event_counts = torch.empty((F, 4), device=theta.device, dtype=torch.float64)
loss, gradient, receiver_gradient = model.genewise_loss_vector_and_grad(
    theta=theta, need_grad=True, event_counts_out=event_counts,
)
```

The return arity remains three. The opt-in buffer is filled in `S,D,L,T` order with positive
posterior expected counts, including survival-conditioning ghosts. It must have shape `[F,4]`,
be on theta's device, use float32 or float64, be at least as wide as theta, and is only valid when
`need_grad=True`.

Implementation: the final existing parameter E-step asks the same `autograd.grad` call for the
ordinary theta/receiver gradients and, only when requested, the intermediate adjoints of
`log_pS_r`, `log_pD_r`, `log_pL_r`, and `mt_r`. The transfer count is the negative sum of the
`mt_r` adjoint over recipients. Multi-batch execution uses a batch-local `[Fb,4]` output and
scatters it back with each static's family indices. No E solve, Pi sweep, hook, global sink, or
return value is added. With `event_counts_out=None`, the original requested-input tuple passed to
autograd is unchanged.

Focused tests are in `tests/test_genewise_event_counts.py`. The CPU validation and scatter tests
pass, as does the tiny CUDA integration test: ordinary/count-enabled NLL and gradients agree,
counts are finite and nonnegative, and folding the four counts through the event softmax recovers
the returned theta gradient.

## Known pitfalls

- Original independent rate bounds become coupled nonlinear constraints in `(u,v,w)`.
- A projected trial requires actual NLL acceptance; quadratic prediction alone is insufficient.
- Complete information is positive but can substantially exceed observed information because of
  missing reconciliation histories. Scaling or secant correction may be essential.
- Large-clade families have the highest missing-information fractions and dominate wall time.
- A line search can cost a full pruning/gradient pass; it is not free even when only NLL is used.
- At an interior stationary point the hierarchy cannot change the generalized missing-information
  eigenvalues; this local EM rate is coordinate invariant. Away from stationarity both complete
  and observed curvature acquire the inverse-map curvature term, so the generalized spectrum can
  change. At an active-bound solution the comparison must be restricted to the feasible tangent
  space. The possible practical win remains cheap diagonal scaling and better alignment, not
  removal of statistical non-convexity.

## Post-EM2 hybrid reopening

The earlier closure applied only to the handwritten post-Adam replay. A production-continuation
adapter was therefore tested from a shared point-consistent EM2 endpoint. It uses ambient true-phi
BFGS secants, exact gradient/Hessian pullbacks, coupled native-face Lagrangian curvature, a
step-only working set for coupled outward directions, feasible endpoint ray bisection, native
projected-gradient stopping, and either native-physical or coordinate trust metric. The unchanged
native continuation is the control.

On 200 families all arms reached the fit's 200/200 certificate and the native-metric hierarchy had
no zero/near-zero ray-stall diagnostics, but it cost 13.91 and 14.09 gradient-clade equivalents in
two runs versus native's 13.52 and 13.56, inclusive of the common two EM passes. A reverse-order
run removed the hierarchy's initially apparent wall-time advantage. Two-run means were 25.46 s and
13.5399 equivalents for native versus 26.78 s and 14.0009 for the hierarchy. Fresh common-model audits put
the native-metric hierarchical endpoint 0.0029 bits below native with no per-family change above
0.01 bits in the first run and 0.0025 bits below in the reverse run; the coordinate-metric endpoint
was 0.0143 bits worse with one 0.0187-bit regression.
The candidate is therefore robust but not work-saving at this gate and is not promoted. Full
design, tests, results, raw tensors, ledgers, logs, and audit are in `geometry/hybrid/RESULTS.md`.

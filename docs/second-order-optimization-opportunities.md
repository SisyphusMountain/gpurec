# Second-Order and Pseudo-Second-Order Optimization Opportunities

Date: 2026-05-10.

Scope: global and uniform-mode DTL-rate optimization, with notes for genewise
and specieswise extensions. This document focuses on optimizer-level changes:
how to use curvature information, how expensive each option is likely to be,
and what would be required to compute second-order derivatives of the likelihood
in practice.

Related files:

- `gpurec/workflow/optimize.py`
- `gpurec/api/model.py`
- `gpurec/api/autograd.py`
- `gpurec/api/uniform_chunked.py`
- `gpurec/optimization/implicit_grad.py`
- `gpurec/optimization/batched_lbfgs.py`
- `profiling/bench_uniform_forward_backward_pipeline.py`
- `docs/lean-fast-path.md`
- `docs/hogenom-ccp-performance-log.md`

## Executive Summary

The most practical opportunity is not exact second-order autodiff. It is a
small custom global optimizer that keeps the current L-BFGS/BFGS curvature model
but moves line-search probes to the existing no-grad likelihood path.

Current PyTorch `LBFGS` uses a closure that runs full forward plus backward for
every strong-Wolfe line-search evaluation. In the uniform global path, a
loss-only probe can use the resident `GeneReconModel` no-grad forward path
(`model()` under `torch.no_grad()` or `full_loss_for_theta(theta)` when an
explicit trial tensor is needed) or `UniformChunkedReconModel.nll()`. These
paths avoid saving full backward state and use the root-row-only likelihood
where possible. On the measured 100-family fp32 polish phase, the recorded
totals were about `0.090 s` forward and `0.889 s` backward over 7 evaluations;
the loss-only part is roughly an order of magnitude cheaper than the gradient
part. On the full 1000-family chunked pipeline, loss-only forward is still about
`2.34 s`, but a gradient evaluation is about `8.68 s`, so loss-only probes are
still only about `27%` of a full gradient evaluation.

Recommended implementation order:

1. Add a global `BFGS` or `LBFGS` variant with Armijo/backtracking probes that
   call a no-grad loss closure. This is pseudo-second-order and very achievable.
2. Benchmark a damped 3x3 finite-difference Newton step using one gradient
   evaluation plus loss-only finite differences for the Hessian. This is easy to
   prototype but may be noisy in fp32 near the optimum.
3. Add a finite-difference gradient Hessian path as a diagnostic baseline, not
   as the first production candidate.
4. Treat exact double backward / exact Hessian-vector products as a research
   project. The current optimized path is intentionally `once_differentiable`,
   uses `torch.no_grad()` around the implicit-gradient machinery, and calls
   custom Triton/CUDA kernels whose backward formulas are not themselves
   differentiated.

For the current three-parameter global problem, exact Hessians are unlikely to
beat a good pseudo-second-order optimizer by much. The global problem is already
tiny in parameter dimension, and L-BFGS reaches the 100-family target in about
5 gradient evaluations. The larger opportunity is reducing wasted backward calls
during line search and stopping earlier when the NLL/rates have already landed.

## Current Optimizer Shape

The production workflow optimizer loop in `gpurec/workflow/optimize.py` builds
standard PyTorch `Adam`, `Adagrad`, or `LBFGS` optimizers over
`GeneReconModel.theta`, plus the retained production routes `hessian-sgd` for
genewise batches and `adagrad-restarts` for specieswise runs. `BatchedLBFGS`
remains available through explicit `optimizer=batched-lbfgs` for row-wise
genewise polishing, but `optimizer=auto` resolves genewise runs to
`hessian-sgd`, specieswise runs to `adagrad-restarts`, and global runs to
`adam`. The scalar optimizer closure:

- clamps `model.theta`;
- runs `loss = model()`;
- runs `loss.backward()`;
- records closure counts, diagnostics, and gradient history.

Every PyTorch `LBFGS` closure call is therefore a full gradient evaluation. That
is appropriate for strong Wolfe because the curvature condition needs the trial
gradient, but it is expensive for backtracking steps whose only job is to reject
a bad step length.

The no-grad inference path already exists:

- `GeneReconModel.forward()` switches to `_evaluate_static_state(...,
  need_grad=False)` when gradients are disabled or the active theta tensor does
  not require gradients;
- `GeneReconModel.full_loss_for_theta(theta)` streams all resident batches with
  `need_grad=False` for explicit trial tensors;
- both resident paths avoid saving `Pi/Pibar` state for backward.

The chunked global model also has a no-grad loss path:

- `UniformChunkedReconModel.nll()` calls `_evaluate_chunked_uniform(...,
  need_grad=False)`;
- the differentiable chunked forward computes the gradient during forward when
  grad is needed, then `backward()` only returns the cached gradient.

The genewise optimizer already has the useful pattern. `BatchedLBFGS` accepts a
`loss_closure` for cheaper row-wise Armijo probes and also supports a
vectorized row-wise strong-Wolfe search, while the workflow feeds it
full-dataset per-family losses and gradients from
`GeneReconModel.full_genewise_nll_and_grad()`. Current integration coverage for
the genewise optimizer lives in `tests/integration/test_gene_recon_model.py`;
avoid relying on historical line numbers in this note when changing that test.

## Current Differentiability Limits

Exact second-order gradients through the current production path are blocked by
design:

- `_GeneReconFunction.backward` is decorated with
  `@torch.autograd.function.once_differentiable`
  in `gpurec/api/autograd.py`.
- `implicit_grad_loglik_vjp_wave` is decorated with `@torch.no_grad()`
  in `gpurec/optimization/implicit_grad.py`.
- `Pi_wave_backward` is also `@torch.no_grad()` in `gpurec/core/backward.py`.
- `UniformChunkedReconModel` computes `(loss, grad)` inside a no-grad custom
  forward and its custom backward simply returns the cached gradient
  in `gpurec/api/uniform_chunked.py`.

This planning note intentionally names source files but avoids exact source line
numbers because the optimization internals move frequently during the audit.

As a result, `torch.autograd.functional.hessian`, `torch.func.hessian`,
`loss.backward(create_graph=True)`, and double backward through `model()` are not
expected to work on the optimized path. That is not a bug in the optimizer; the
path was built to expose a fast first-order analytical gradient, not a
differentiable program for the gradient computation.

Exact second-order support would require new math and new kernels:

- tangent or JVP propagation through the `E` fixed point;
- tangent propagation through each `Pi` wave or through the fixed unrolled Pi
  iterations;
- differentiated adjoint solves for the existing implicit gradient;
- second-order contributions for the uniform Pibar ancestor terms, DTS terms,
  root likelihood, and parameter extraction;
- a way to keep all of that memory-bounded under chunking.

That is a large project and should not be the first attempt.

## Cost Model

Recorded timings from existing docs/logs:

| Workload | Gradient path | Loss-only forward | Gradient eval | Notes |
|---|---:|---:|---:|---|
| `test_trees_100`, fp32 helper | PyTorch LBFGS | about `0.013 s` per eval in one polish phase | about `0.140 s` per eval | From fixed-one bf16 handoff fp32 phase: `0.089793 s` forward, `0.889257 s` backward over 7 evals. |
| first 100 of `test_trees_1000`, fp32 helper | PyTorch LBFGS | about `0.236 s` per eval | about `0.881 s` per eval | From fp32 phase totals: `3.061723 s` forward, `8.380231 s` backward over 13 evals. |
| full 1000-family chunked pipeline | chunked forward/backward | `2.344 s` | `8.685 s` | Historical chunked full-pipeline profiling notes: forward `2343.716 ms`, backward `6341.228 ms`. |

Approximate second-order costs for three global parameters:

| Method | Cost per Hessian refresh | 100-family equivalent | first-100-of-1000 equivalent | full 1000 equivalent |
|---|---:|---:|---:|---:|
| Gradient finite-difference Hessian | `6` gradient evals | about `0.84 s` | about `5.3 s` | about `52 s` |
| Loss finite-difference Hessian | `1` gradient + `18` loss evals | about `0.37 s` | about `5.1 s` | about `50.9 s` |
| HVP by gradient finite difference | `2` gradient evals per direction | about `0.28 s` | about `1.76 s` | about `17.4 s` |
| BFGS/L-BFGS Armijo with loss-only probes | `1` gradient per accepted step + cheap probes | depends on probes | likely best wall-time candidate | likely best wall-time candidate |

The finite-difference Hessian costs are acceptable as benchmark prototypes on
100-family workloads. They become expensive on the full 1000-family workload
unless a Hessian refresh replaces many full gradient evaluations.

## Proposal 1: Global BFGS with Loss-Only Armijo

Status: highest priority.

This is pseudo-second-order, not exact Newton. It still uses curvature pairs
`s_k = theta_{k+1} - theta_k` and `y_k = grad_{k+1} - grad_k`, but line-search
probes evaluate only the loss. The accepted point then gets one gradient
evaluation to update the BFGS state.

Implementation sketch:

```python
def objective_and_grad(theta):
    set_theta(theta)
    model.zero_grad(set_to_none=True)
    loss = model()
    loss.backward()
    return float(loss), model.theta.grad.detach().clone()

@torch.no_grad()
def objective_only(theta):
    saved_warm_state = snapshot_warm_state(model)
    try:
        set_theta(theta)
        if isinstance(model, UniformChunkedReconModel):
            chunked = model
            return float(chunked.nll())
        return float(model.full_loss_for_theta(theta))
    finally:
        restore_warm_state(model, saved_warm_state)
```

For a 3D global problem, use dense BFGS rather than limited-memory storage:

```text
H_inv starts as scalar * I
p = -H_inv @ g
line search: Armijo using objective_only(theta + alpha * p)
accepted: compute objective_and_grad once
BFGS update if y.T @ s is positive and large enough
```

Expected benefit:

- On 100-family runs, this can save most line-search backward calls. If PyTorch
  L-BFGS uses 7-12 full gradient closures, a custom BFGS path that uses 4-6
  gradient calls plus cheap probes could plausibly save `20-50%` wall time.
- On the full 1000-family chunked path, every avoided backward saves about
  `6.34 s`; the line-search probes still cost about `2.34 s` each, so the
  benefit depends on keeping probes low. Armijo backtracking is attractive
  because it usually needs only a few loss probes.

Risks:

- Armijo-only search does not enforce the strong-Wolfe curvature condition, so a
  bad step can produce poor `s/y` curvature. Mitigation: skip BFGS updates when
  `y.T @ s <= 1e-10 * ||s|| * ||y||`, damp the BFGS update, and fall back to
  steepest descent if the direction is not descent.
- The no-grad path updates `warm_E`. Loss probes must save/restore or disable
  `warm_E`, otherwise probe order can leak into the objective and into finite
  difference symmetry.
- Need bound handling in log-rate space. Projection is fine for Armijo, but
  curvature updates should use the actual projected step.

Benchmark target:

- Match the fp32 helper's 100-family final NLL and rate tolerance.
- Beat helper hit time and/or final optimizer time.
- On first 100 of `test_trees_1000`, beat the `13`-eval fp32 baseline without
  worse final NLL.

## Proposal 2: Damped 3x3 Finite-Difference Newton

Status: useful benchmark prototype; not first production candidate.

Because global mode has only three parameters, the full Hessian is small. We can
estimate it without double backward and solve a damped Newton step:

```text
(H + lambda I) p = -g
theta_trial = project(theta + alpha p)
```

Use eigenvalue damping:

```python
H = 0.5 * (H + H.T)
eigval, Q = torch.linalg.eigh(H)
eigval = torch.clamp(eigval, min=min_curvature)
p = -Q @ ((Q.T @ g) / eigval)
```

Loss finite-difference Hessian:

```text
H_ii = (f(theta + h_i e_i) - 2 f(theta) + f(theta - h_i e_i)) / h_i^2

H_ij = (
    f(theta + h_i e_i + h_j e_j)
  - f(theta + h_i e_i - h_j e_j)
  - f(theta - h_i e_i + h_j e_j)
  + f(theta - h_i e_i - h_j e_j)
) / (4 h_i h_j)
```

Cost is one gradient evaluation plus `18` loss-only evaluations for three
parameters. On the 100-family benchmark this is cheap enough to test. On full
1000-family chunked training, one Hessian refresh is roughly equivalent to
`5.8` gradient evaluations, so it only wins if it replaces a large number of
LBFGS closures.

Step-size policy:

- Work in log2-rate `theta`.
- Start with `h` in `{0.01, 0.03, 0.05}` log2 units.
- Near fp32 convergence, prefer larger `h` or switch to gradient-difference
  Hessians because scalar second differences can be dominated by fp32 noise.
- Always compare against a gradient-difference Hessian on small smoke problems.

Expected benefit:

- Could reduce iteration count to 2-4 accepted Newton-like steps if the objective
  is locally close to quadratic.
- May not beat BFGS Armijo, because the Hessian refresh itself is expensive on
  larger workloads.

Risks:

- fp32 loss finite differences can be noisy near the optimum.
- The Hessian can be indefinite away from the optimum.
- Bounds can invalidate symmetric finite differences when a parameter is near
  the lower floor.
- Warm-start state must be controlled for every probe.

## Proposal 3: Gradient-Difference Hessian or HVP

Status: good diagnostic; slower than loss-only Hessian in small global mode.

Full Hessian by gradient finite differences:

```text
H[:, i] = (g(theta + h_i e_i) - g(theta - h_i e_i)) / (2 h_i)
```

Hessian-vector product:

```text
H v = (g(theta + h v) - g(theta - h v)) / (2 h)
```

This avoids scalar second-difference noise but costs full backward passes. For
three global parameters, a full Hessian costs 6 gradient evaluations. That is
often too close to the whole current L-BFGS budget to be attractive.

Where it is still useful:

- validating the loss finite-difference Hessian;
- measuring Hessian conditioning and eigenvalues near the optimum;
- testing whether a Newton step would have helped before investing in exact
  second-order code;
- building HVPs for truncated Newton in higher-dimensional specieswise mode.

For global mode, full Hessian is simpler than HVP-CG because the parameter space
is only 3D.

## Proposal 4: Genewise Block Newton

Status: interesting but likely secondary.

In genewise mode, each family has its own 3-vector of rates and the objective is
block-separable by family. The Hessian is block diagonal with one 3x3 block per
family.

Important batching observation:

- We do not need `6 * G` gradient calls to finite-difference every block.
- Perturbing all families in coordinate `D`, then all in `L`, then all in `T`,
  gives all per-family Hessian columns in `6` batched gradient evaluations,
  because each row's gradient is independent of other rows.

This makes finite-difference block Newton feasible as a benchmark:

```text
for coord in D,L,T:
    theta_plus[:, coord] += h
    theta_minus[:, coord] -= h
    g_plus = grad_per_family(theta_plus)
    g_minus = grad_per_family(theta_minus)
    H_blocks[:, :, coord] = (g_plus - g_minus) / (2h)

for family g:
    solve damped 3x3 block
```

However, the existing `BatchedLBFGS` already supports per-row curvature and
loss-only Armijo probes. It is probably the stronger near-term production path.
Block Newton is worth testing if many families need many BFGS iterations or if
per-family convergence remains uneven.

## Proposal 5: Empirical Fisher or Gauss-Newton Approximation

Status: lower priority.

A positive semidefinite curvature approximation could be built from per-family
score vectors:

```text
F = sum_g grad_g grad_g.T
```

This can act as a preconditioner or a trust-region metric for global rates. It
is not the true Hessian of the marginal reconciliation likelihood, especially
because the likelihood includes fixed-point dependencies through `E` and `Pi`.

Potential use:

- stabilize global steps when Hessian estimates are indefinite;
- initialize dense BFGS `H_inv`;
- create a diagonal or low-rank preconditioner for specieswise mode.

Main issue:

- The global optimized path currently accumulates one global gradient. Per-family
  score extraction would need a genewise-style or chunked per-family gradient
  path. That is extra work and likely not needed for three global parameters.

## Proposal 6: Exact Implicit Hessian-Vector Products

Status: not recommended until first-order kernel work is exhausted.

For a fixed point

```text
x = F(x, theta)
L = L(x, theta)
A = I - F_x
```

the first-order adjoint solves

```text
A.T lambda = L_x
grad_theta = L_theta + F_theta.T lambda
```

An exact Hessian-vector product in direction `v` needs at least:

```text
A x_dot = F_theta v
```

and a differentiated adjoint solve for `lambda_dot`, which includes
second-derivative contractions such as:

```text
d(A.T)/dtheta[v] * lambda
d(A.T)/dx[x_dot] * lambda
d(L_x)/dtheta[v]
d(L_x)/dx[x_dot]
```

Then:

```text
H v = d/dtheta [L_theta + F_theta.T lambda] applied to v
```

In this codebase, `x` is not a single small vector. It includes the shared `E`
fixed point and a large chunked `Pi/Pibar` wave state. The hot operations are
custom uniform wave kernels, DTS kernels, and Pibar ancestor kernels. Exact HVP
support would therefore need second-order versions of the same memory-intensive
logic that already dominates first-order runtime.

Expected cost:

- likely at least one extra forward sensitivity pass and one extra adjoint-like
  pass per HVP;
- probably similar to or more expensive than a gradient evaluation;
- additional saved state or recomputation pressure under chunking.

Given that global mode has only three parameters and the current first-order
optimizer is already fast, this is not a good near-term tradeoff.

## Practical Benchmark Plan

Add a dedicated global-parameter optimizer benchmark harness or extend the
current profiling utilities with these strategies:

```text
bfgs-armijo-loss-only
fd-newton-loss-hessian
fd-newton-gradient-hessian
fd-hvp-newton-cg
```

Minimum metrics:

- gradient evaluations;
- loss-only evaluations;
- total optimizer time;
- hit evaluation/time for the existing NLL and rate target;
- final NLL, rates, gradient infinity norm;
- per-phase forward/backward timings;
- Hessian eigenvalues and damping applied;
- number of rejected line-search probes;
- number of skipped BFGS updates due to poor curvature;
- whether `warm_E` was saved/restored or disabled for probes.

Historical command sketch for the missing global-parameter benchmark harness:

```bash
python path/to/global_parameter_optimization_benchmark.py \
  --dataset tests/data/test_trees_100 \
  --strategies recommended-fp32,bfgs-armijo-loss-only,fd-newton-loss-hessian,fd-newton-gradient-hessian \
  --init-rate 0.05 \
  --no-print-evals
```

```bash
python path/to/global_parameter_optimization_benchmark.py \
  --dataset tests/data/test_trees_1000 \
  --max-families 100 \
  --allow-missing-target \
  --strategies recommended-fp32,bfgs-armijo-loss-only,fd-newton-loss-hessian \
  --init-rate 0.05 \
  --no-print-evals
```

For the full chunked 1000-family model, benchmark through
`UniformChunkedReconModel` rather than resident `GeneReconModel`, because the
existing docs show the resident 1000-family full-state path is memory-sensitive.

Promotion criteria:

- `bfgs-armijo-loss-only`: promote if it matches final NLL/rates and reduces
  optimizer wall time by at least `15%` on both 100-family and first-100-of-1000
  runs.
- `fd-newton-loss-hessian`: keep only if it beats BFGS Armijo on wall time or
  materially reduces full 1000-family gradient calls.
- `fd-newton-gradient-hessian`: keep as a diagnostic unless it unexpectedly beats
  the loss-Hessian variant.
- Exact HVP/double-backward: do not start until the finite-difference benchmark
  proves that better curvature can reduce total full-pipeline evaluations enough
  to justify implementation.

## Implementation Details to Get Right

Warm starts:

- `UniformChunkedReconModel.nll()` and resident no-grad probes such as
  `GeneReconModel.full_loss_for_theta(theta)` can mutate warm `E` state.
- Finite differences need symmetric, deterministic probes. Save and restore
  `warm_E` for every probe, or set it to `None` for every probe in the Hessian
  builder.
- For line search, decide whether to keep probe warm-start updates. The safer
  benchmark should save/restore during probes and only update warm state at the
  accepted point.

Bounds:

- Work in `theta = log2(rate)`.
- Clamp or project to `log2(min_rate)`.
- If a symmetric Hessian probe would cross the floor, shrink `h` for that
  coordinate or skip finite-difference Newton at that point.

Precision:

- fp32 finite-difference Hessians need fairly large log2 steps.
- fp64 is useful for Hessian diagnostics and final polish, but full fp64 backward
  may not fit on 24 GB for large workloads.
- bf16 should not be used for Hessian estimation; scalar NLL quantization already
  made bf16 handoff thresholds unreliable.

Stopping:

- Keep the existing NLL/rate-based success criteria.
- Do not rely only on `grad_inf` near the fp32 floor; prior docs show fp32
  gradients can jitter even after the rates and NLL have landed.

## Recommendation

Implement and benchmark `bfgs-armijo-loss-only` first. It is the most achievable
pseudo-second-order improvement because it reuses the existing gradient, the
existing no-grad likelihood, and a curvature model the project already trusts.

Then prototype `fd-newton-loss-hessian` as a benchmark-only strategy. It will
answer the core question: does better curvature reduce enough full gradient
evaluations to matter? If the answer is no, exact second-order kernels should be
deprioritized. If the answer is yes on full 1000-family chunked optimization,
then we can consider a more serious HVP path.

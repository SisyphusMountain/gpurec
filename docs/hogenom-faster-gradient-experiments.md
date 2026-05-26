# HOGENOM Faster Gradient Experiments

This note tracks isolated experiments for accelerating HOGENOM optimization
through faster implicit gradients.  It combines the fixed-point acceleration
ideas in `docs/faster_gradients.md` with the measured Pi-adjoint warmstart
result from the gradient-convergence benchmark.

## Baseline Observation

The current HOGENOM gradient path spends many repeated VJP iterations in the
Pi adjoint Neumann/self-loop solve.  Starting that Pi adjoint solve from the
previous accepted step's solved `v_Pi` substantially reduces the iteration
count needed to match a high-budget reference.

Measured against a 128-term reference:

| checkpoint | method | 10% rel L2 | 1% rel L2 | 0.1% rel L2 | 0.01% rel L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| +300 NLL | cold Neumann | 24 | 32 | 48 | 64 |
| +300 NLL | warm Pi adjoint | 2 | 16 | 24 | 32 |
| +120 NLL | cold Neumann | 32 | 48 | 64 | 96 |
| +120 NLL | warm Pi adjoint | 1 | 16 | 24 | 48 |

This isolates the important point: the useful warm state is the Pi adjoint,
not the E-adjoint solve.  Near convergence, warmstarted Pi gradients can use
roughly one half to one third of the Neumann/VJP iterations for the same
relative-gradient target.

## Active Worktrees

| worktree | branch | proposal | intended isolation |
| --- | --- | --- | --- |
| `agent-worktrees/pi-warm-cache` | `agent/pi-warm-cache` | Runtime Pi-adjoint cache | Model/autograd cache behavior only, no optimizer rollout |
| `agent-worktrees/pi-budget-policy` | `agent/pi-budget-policy` | Lower warmstarted budgets | Benchmark summarization only, no production behavior |
| `agent-worktrees/pi-anderson-prototype` | `agent/pi-anderson-prototype` | Fixed-point relaxation prototype | Kernel/implicit-gradient option only, no optimizer rollout |

## Suggestions And Practical Requirements

### 1. Make Pi-Adjoint Warmstart A First-Class Optimizer Feature

This is the highest-confidence path.  The measured win comes from reusing the
previous step's solved Pi adjoint, so the production optimization loop should
carry a `v_Pi` cache across accepted iterates.

Using it in practice requires:

* store one solved `v_Pi` tensor per resident batch static state,
* clear it when batch layout, dtype/device, theta shape, pruning policy, or
  solver settings change,
* keep lazy-prefetched statics consistent with the current cache setting,
* distinguish line-search trial gradients from accepted optimizer steps,
* update the next-step cache only from the gradient chosen for an accepted
  iterate.

The important optimizer detail is that failed line-search trials must not poison
the accepted-step warmstart.  A safe implementation can let trial gradients read
the last accepted `v_Pi`, but should stage their solved `v_Pi` separately and
commit it only when the trial is accepted.

### 2. Use Smaller Warmstarted Pi Budgets With Validation

The warmstart result changes the budget question: once the cache is available,
ordinary optimizer steps should not use the same Neumann count as cold
gradients.

Using it in practice requires a solver-budget policy:

* bootstrap with a cold high-budget gradient,
* use warm Pi with 16 or 24 Neumann terms for ordinary near-optimum steps,
* periodically validate with 48, 64, or 128 terms,
* escalate when line-search decisions are marginal or projected-gradient
  convergence checks become small.

For the measured checkpoints, 16 warm terms reach the 1% relative-gradient
target where cold gradients require 32 to 48 terms.  24 warm terms reach the
0.1% target where cold gradients require 48 to 64 terms.  Wall-clock speedups
were smaller than the term-count reductions because forward work, E-adjoint
work, and Python/model overhead remain.

### 3. Tune Warmstarted Richardson Relaxation Before Full Anderson

`docs/faster_gradients.md` suggests Krylov/Anderson-style methods because plain
Neumann is stationary Richardson iteration.  A bounded first step is to tune a
relaxation parameter for the warmstarted Pi fixed-point update:

```text
v_next = v + alpha * ((rhs + J^T v) - v)
```

This tests whether damped or over-relaxed Richardson improves the warmstarted
Pi solve before attempting full Anderson or GMRES.  A real Anderson/GMRES
implementation would require per-wave residual storage, small least-squares or
Krylov basis management, GPU reductions, and careful memory accounting.

Using relaxation in practice requires exposing `alpha` through the workflow
configuration, keeping the default at `alpha=1.0`, and benchmarking a small
grid per dataset/checkpoint.  A local HOGENOM smoke at the +300 checkpoint found
`alpha=1.25` better than `alpha=1.0` for warmstarted relative L2 error at
1, 4, and 16 terms, while damping below 1.0 worsened that smoke.  That is
promising but not enough to make 1.25 a default.

### 4. Add A Residual-Based Pi Stopping Rule

`docs/faster_gradients.md` also points out that truncating only by iteration
count is a weak stopping rule.  The fixed-point residual,

```text
r = rhs + J^T v - v
```

is the natural local signal for the Pi adjoint solve.

Using this in practice requires computing or estimating a residual norm from
the same warmstarted Pi update path.  The implementation needs cheap GPU
reductions, solver stats per batch/wave, and a calibration step mapping Pi
residual to final theta-gradient error.  This should be paired with the budget
policy: run a small minimum number of terms, check residual every few terms,
and escalate only when the residual is not small enough.

### 5. Defer GMRES / Anderson Until The Cheap Wins Are Integrated

GMRES or small-memory Anderson is the theoretically cleaner replacement for the
plain Neumann sum.  It should be considered if warmstart plus budget tuning
still leaves gradient evaluation as the dominant end-to-end cost.

Using it in practice requires a matrix-free operator for
`u -> u - J^T u`, storage for several `[C, S]` Pi-adjoint vectors, dot-product
reductions, residual monitoring, and careful interaction with wave batching.
The memory and synchronization cost could erase the VJP-count win unless the
plain iteration is still very slow, so this should be benchmarked after the
warmstarted baseline is in place.

### 6. Treat Primal Contraction Changes As Higher Risk

The adjoint convergence rate depends on the contraction of the fixed-point map.
Changing the primal fixed-point formulation could therefore help, but it is
higher risk for HOGENOM because it can change the forward solution or the
effective objective unless the formulation is exactly equivalent.

Using this in practice would require objective-equivalence tests, forward-loss
checks against the current kernels, and a separate end-to-end validation suite.
It is not the next implementation target.

## Budget Evidence

Measured rows from the gradient-convergence benchmark:

| checkpoint | target rel L2 | cold terms | warm terms | cold time | warm time | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| +300 NLL | 10% | 24 | 2 | 2.79s | 1.99s | 1.40x |
| +300 NLL | 1% | 32 | 16 | 3.07s | 2.49s | 1.23x |
| +300 NLL | 0.1% | 48 | 24 | 3.61s | 2.75s | 1.31x |
| +300 NLL | 0.01% | 64 | 32 | 4.19s | 3.07s | 1.37x |
| +120 NLL | 10% | 32 | 1 | 3.07s | 1.98s | 1.55x |
| +120 NLL | 1% | 48 | 16 | 3.64s | 2.53s | 1.44x |
| +120 NLL | 0.1% | 64 | 24 | 4.23s | 2.79s | 1.52x |
| +120 NLL | 0.01% | 96 | 48 | 5.39s | 3.62s | 1.49x |

The practical interpretation is that warm Pi cuts the Pi iteration budget by
about 2x for high-accuracy gradients near the optimum, and much more for coarse
10% gradients.  End-to-end speedup will be lower than the term reduction until
the non-Pi parts of the gradient path are also reduced or amortized.

## Results Log

| proposal | worker commit | supervisor result | targeted evidence | conclusion |
| --- | --- | --- | --- | --- |
| Runtime Pi-adjoint cache | `d8f6d68`, fix `2fc949a`, current production follow-ups | First review rejected stale lazy-prefetch statics; second review approved; the API bridge now owns an opt-in cache field, staged updates, stale-layout discard, solver telemetry, a Hessian-conditioned genewise accepted-step commit boundary, and a workflow flag for staged end-to-end validation without changing defaults. | Worker tests: `test_pi_adjoint_warmstart_cache.py`, targeted workflow tests, `test_implicit_grad_solver.py`, compileall, `git diff --check`. Production follow-ups added CPU bridge/cache tests, runtime-cache clearing guards, workflow commit/discard tests, and config/CLI coverage. | Accept as a lower-level cache mechanism. Production routes still keep it disabled until warmstarted gradient budgets are validated end to end. |
| Warmstarted budget policy summary | `e3ce916` | Approved, no findings | Unit test `test_hogenom_gradient_policy_summary.py`: 2 passed. Smoke benchmark produced a policy recommendation. Baseline benchmark rows above quantify useful budgets. | Accept as benchmark/reporting support. Use it to choose 16/24-term warm budgets plus periodic higher-budget validation. |
| Warmstarted fixed-point relaxation | `d0271bb`, fix `afcd7d3` | First review requested kernel coverage and bool validation; final review approved | Unit/core tests plus CUDA kernel regression: 14 passed locally. Supervisor rerun: bool/forwarding tests 10 passed, CUDA kernel test 1 passed, diff check clean. HOGENOM smoke: `alpha=1.25` improved relative L2 at terms 1, 4, 16 versus `alpha=1.0`. | Accept as an experimental knob. Keep default `alpha=1.0`; benchmark alpha grid before enabling in end-to-end optimization. |

## Recommended Integration Order

1. Validate the opt-in Pi-adjoint cache in end-to-end Hessian-SGD runs using
   `hessian_sgd_pi_adjoint_warmstart=true` or
   `--hessian-sgd-pi-adjoint-warmstart`. The API bridge can stage solved
   adjoints before commit, clears stale layout state, and the Hessian-conditioned
   genewise workflow commits only after the accepted current-theta gradient, so
   the next work is budget validation rather than cache ownership.
2. Add warm/cold gradient budgets to HOGENOM optimization config, starting with
   warm Pi terms of 16 or 24 and periodic 48/64/128-term validation.
3. Add residual logging for the Pi adjoint solve so budget escalation is based
   on convergence, not only on a fixed iteration count.
4. Expose `pi_fixed_point_relaxation` as an experimental config and benchmark
   alpha values on several near-optimum checkpoints.
5. Revisit Anderson or GMRES only after the warm cache and budget policy are
   measured in an end-to-end optimizer run.

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

## Proposal Requirements

### Runtime Pi-Adjoint Cache

Using this in practice requires storing one solved `v_Pi` tensor per resident
batch static state and invalidating it whenever the batch layout, dtype/device,
theta shape, pruning policy, or solver settings change.  For production
optimization, the cache must eventually be tied to accepted optimizer steps:
line-search trial gradients should not overwrite the cache used for the next
accepted iterate.

The current worktree experiment deliberately tests only the lower-level cache
mechanics: pass previous `v_Pi` into the next gradient call and update the cache
with the newly solved `v_Pi`.

### Lower Warmstarted Budgets

Using this in practice requires a solver-budget policy, not a new mathematical
kernel.  A plausible first production policy is:

* bootstrap with a cold high-budget gradient,
* use warm Pi with 16 or 24 Neumann iterations for ordinary optimizer steps,
* periodically validate with 48, 64, or 128 terms,
* escalate when line-search decisions are marginal or projected-gradient
  convergence checks become small.

The isolated experiment summarizes term/time reductions from the checkpoint
gradient-convergence benchmark rather than running end-to-end optimization.

### Fixed-Point Relaxation / Anderson Family

`docs/faster_gradients.md` suggests Krylov/Anderson-style methods because plain
Neumann is stationary Richardson iteration.  A bounded first experiment is to
add a relaxation parameter for the warmstarted Pi fixed-point update:

```text
v_next = v + alpha * ((rhs + J^T v) - v)
```

This tests whether damped or over-relaxed Richardson improves the warmstarted
Pi solve before attempting full Anderson or GMRES.  A real Anderson/GMRES
implementation would require per-wave residual storage, small least-squares or
Krylov basis management, GPU reductions, and careful memory accounting.

## Results Log

Results will be appended here after worker implementation and supervisor
review.  Each row should cite the worktree commit and the exact targeted test
or benchmark used.

| proposal | worker commit | supervisor result | targeted evidence | conclusion |
| --- | --- | --- | --- | --- |

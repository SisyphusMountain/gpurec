# Efficient GMRES for Gradient Self-Loop Solves

## Scope

This note describes how to implement GMRES efficiently for the retained
gradient self-loop solve. The goal is not to reduce the number of outer
optimization steps directly. The goal is to obtain an accurate gradient in less
time by reducing the expensive wave-local backward self-loop operator
applications.

The relevant solve appears during the backward pass. For each retained wave, we
need the adjoint `v` satisfying:

```text
v = rhs + J^T v
```

Equivalently:

```text
(I - J^T) v = rhs
A v = rhs
```

where:

- `rhs` is the wave-local adjoint source term.
- `J^T` is the wave-local self-loop transpose-Jacobian action.
- `A = I - J^T`.
- `v` is the wave-local adjoint used to accumulate parameter gradients.

The current Neumann method approximates:

```text
v ~= rhs + J^T rhs + (J^T)^2 rhs + ... + (J^T)^k rhs
```

This is simple and GPU-friendly, but it spends a fixed number of expensive
self-loop applications per wave. GMRES should replace this fixed polynomial by
an adaptive Krylov solve over the same linear operator.

## What Is Expensive

The important unit of cost is one application of the wave-local self-loop
backward operator:

```text
x -> J^T x
```

In the retained path, this is implemented by the Triton
`_wave_backward_uniform_2d_jt_kernel`. This kernel does the tree reductions and
self-loop adjoint propagation for one trial vector. It is the cost that scales
with the family, number of clades, species tree size, and wave structure.

For Neumann with `k` terms and `Waves` wave-local solves, the expensive work is
fixed:

```text
total backward iterations = k * Waves
```

For GMRES with maximum Krylov dimension `m`, the expensive work is adaptive:

```text
total backward iterations = sum(actual GMRES iterations used by each wave)
```

Each GMRES iteration requires one application of:

```text
A x = x - J^T x
```

The identity part is cheap. The expensive part is still exactly one `J^T x`.
The least-squares minimization inside GMRES does not require extra backward
operator applications. It uses the already stored Krylov basis and the small
Hessenberg matrix built during Arnoldi iteration.

## Why Matrix-Free GMRES Fits This Case

We should not form `J^T` or `A` as explicit matrices. The useful primitive we
already have is a fast matrix-vector product:

```text
apply_j(x) = J^T x
```

GMRES only needs this primitive. At iteration `j`, it applies `A` to one basis
vector:

```text
w = A q_j = q_j - J^T q_j
```

It then orthogonalizes `w` against the previous basis vectors. The resulting
small problem is:

```text
min_y || beta e_1 - H_j y ||
```

The approximate solution is:

```text
v_j = Q_j y
```

where `Q_j` contains the Krylov basis vectors and `H_j` is the small upper
Hessenberg matrix. This is exactly the structure we want: many operations on
small dense objects plus one expensive self-loop application per GMRES step.

## Production Algorithm

The production algorithm should be wave-local and matrix-free:

```text
for each retained backward wave:
    precompute wave coefficients for J^T
    define apply_A(x) = x - apply_J(x)
    choose initial guess v0
    compute r0 = rhs - apply_A(v0)
    if ||r0|| / ||rhs|| <= tol:
        accept v0
    else:
        solve A delta = r0 with restarted/adaptive GMRES
        v = v0 + delta
    accumulate parameter gradients from v
```

For a zero initial guess:

```text
v0 = 0
r0 = rhs
```

For a warm start from the previous optimizer step:

```text
v0 = previous wave-local v
r0 = rhs_new - A_new v0
```

The warm-started form is important. If `theta` changes between optimizer steps,
the previous `v` is not an exact solution for the new system. GMRES should solve
for a correction `delta`, not blindly continue from stale Neumann terms.

## Recommended Solver Policy

Use `gmres_max_iter` as a maximum, not as a fixed iteration count. Stop each
wave when the relative residual is small enough:

```text
||rhs - A v|| / max(||rhs||, eps) <= gmres_tol
```

Initial values worth testing:

```text
gmres_max_iter = 8, 12, 16
gmres_tol = 1e-6, 1e-8, 1e-10
```

The best production default may not be pure GMRES for every wave. A hybrid
policy is likely more efficient:

```text
run a small number of cheap Neumann/fixed-point steps
estimate the residual or residual decrease
if the wave is easy:
    stop or finish with Neumann
else:
    switch to GMRES using the current v as v0
```

This matters because GMRES has more orchestration overhead than Neumann. On easy
waves, a few Neumann terms may be faster than building a Krylov basis.

## Warm Starts Across Optimizer Steps

Warm starts should store the wave-local self-loop adjoint solution `v`, not the
final parameter gradient.

The parameter gradient is already the output of the whole backward pass. It is
not the state of the linear solve. The reusable state is:

```text
previous_v[family_or_batch, wave] ~= solution of (I - J(theta_old)^T) v = rhs(theta_old)
```

On the next optimizer step, use:

```text
A_new = I - J(theta_new)^T
r0 = rhs_new - A_new previous_v
```

Then:

```text
if ||r0|| / ||rhs_new|| <= tol:
    use previous_v
else:
    GMRES solves A_new delta = r0
    v_new = previous_v + delta
```

This costs one extra `A_new previous_v` application per warm-started wave, but
it can save many GMRES iterations when optimizer steps are small.

Warm starts should be disabled or ignored when:

- the wave shape changes;
- the active row mask changes in an incompatible way;
- the dtype or device changes;
- the previous residual under the new `theta` is too large;
- the optimizer accepted a very large parameter step.

The cache should be optional and bounded. It may be too expensive to store every
wave for very large batches. If memory is limiting, cache only hard waves or the
families known to require many Neumann iterations.

## Preconditioning

A cheap preconditioner should be tested before considering anything more
complex. The most natural candidate is a diagonal/Jacobi preconditioner derived
from the local diagonal contribution already computed for the self-loop:

```text
M ~= diag(I - J^T)
M_inv ~= 1 / (1 - diag_wt)
```

Then solve a right-preconditioned system:

```text
A M_inv z = rhs
v = M_inv z
```

or a left-preconditioned system:

```text
M_inv A v = M_inv rhs
```

Right preconditioning is often cleaner for preserving the true residual check:

```text
||rhs - A v||
```

The preconditioner must be guarded numerically:

```text
denom = clamp_abs(1 - diag_wt, min_abs=epsilon)
M_inv = 1 / denom
```

This is cheap because `diag_wt` already exists in the retained backward
precompute. The implementation cost is one elementwise multiply around
`apply_A`.

More complicated preconditioners should wait until profiling shows the diagonal
one is insufficient. The self-loop operator has tree-structured reductions and
receiver corrections, so an exact or block preconditioner may become expensive
quickly.

## GPU Implementation Details

The experimental GMRES path is useful to validate the math, but a production
implementation should remove Python-side overhead.

Recommended implementation details:

- Preallocate Krylov basis storage with shape `[m + 1, W, S]`.
- Preallocate the Hessenberg matrix with shape `[m + 1, m]`.
- Reuse existing scratch buffers for `J^T q_j`, `A q_j`, residuals, and
  corrected solutions.
- Avoid allocating a new `term_out` tensor inside every GMRES iteration.
- Avoid Python lists of basis tensors.
- Avoid calling `torch.linalg.lstsq` at every iteration.
- Use incremental QR with Givens rotations to update the residual estimate.
- Only materialize the final linear combination `Q y` once, after convergence
  or after hitting `gmres_max_iter`.
- Keep residual statistics on device as long as possible.
- Minimize CPU synchronization. Converting norms to Python floats every
  iteration can dominate small waves.

The ideal per-wave loop is:

```text
beta = norm(r0)
q_0 = r0 / beta

for j in 0 .. gmres_max_iter - 1:
    w = apply_A(q_j)

    for i in 0 .. j:
        h_ij = dot(q_i, w)
        w = w - h_ij q_i

    h_{j+1,j} = norm(w)
    q_{j+1} = w / h_{j+1,j}

    update QR factors for H
    update residual estimate

    if residual <= tol:
        break

solve small triangular system for y
v = v0 + Q_j y
```

For numerical stability, use modified Gram-Schmidt with optional
reorthogonalization. The Krylov dimension is expected to be small, so
reorthogonalization may be acceptable if it avoids rare accuracy failures.

## Concrete Implementation Plan

The next implementation should not start as one monolithic Triton kernel.
GMRES has wave-local control flow, repeated matrix-vector products, reductions,
small QR state, and basis storage. The fastest way to get a valid benchmark is
to keep the expensive self-loop operator in Triton, but orchestrate the Krylov
algorithm with GPU-resident tensors first.

The staged plan is:

1. Keep the existing Triton self-loop operator as the matvec.

   ```text
   apply_A(x) = x - apply_J(x)
   ```

   `apply_J` should continue to be the retained self-loop backward Triton
   kernel. GMRES should not materialize `J^T` or `A`.

2. Remove CPU least-squares from the timing path.

   A CPU float64 solve of the small Hessenberg problem is useful only as a
   debugging fallback. It introduces synchronization and transfer overhead per
   wave, so any wall-clock benchmark that includes it is not a valid GMRES
   performance measurement.

3. Implement fixed-dimension, GPU-resident GMRES.

   Start with fixed `m` iterations and no early stopping inside the loop. This
   avoids Python scalar reads from device tensors while we validate the math.
   The loop should keep all basis vectors, Hessenberg coefficients, Givens
   rotations, residual estimates, and final coefficients on GPU.

4. Use incremental QR with Givens rotations instead of `lstsq`.

   At each Arnoldi step:

   ```text
   append one Hessenberg column
   apply prior Givens rotations
   compute one new Givens rotation
   update the residual scalar
   ```

   This gives the GMRES residual estimate without solving a fresh dense least
   squares problem every iteration.

5. Use modified Gram-Schmidt with one reorthogonalization pass.

   For `m <= 16` or `m <= 32`, a second orthogonalization pass is cheap
   relative to unstable gradients. The first correctness target is finite,
   reproducible gradients in float32 without a CPU fallback.

6. Compare gradient quality before optimizing wall time.

   For fixed saved forward states and selected hard families, test:

   ```text
   Neumann: 16, 32, 64
   GMRES fixed m: 4, 8, 12, 16, 24, 32
   ```

   Measure:

   ```text
   gradient error vs reference
   J^T applications
   GMRES residuals
   wall time
   ```

   Only after GMRES gives a better gradient per `J^T` application should we
   tune the optimizer-level benchmark.

7. Move hot vector operations to Triton incrementally.

   Once the GPU-resident PyTorch version is correct, specialize only the
   bottlenecks:

   - wave-local dot products and norms;
   - axpy updates for orthogonalization;
   - basis scaling;
   - final `Q y` combination;
   - optional `apply_A` output mode in the self-loop kernel.

   This keeps the algorithm inspectable while avoiding a large fused kernel
   that is hard to debug.

8. Add early stopping after the fixed-iteration version is stable.

   Early stopping is useful, but Python `.item()` checks each iteration can
   dominate small waves. Prefer either:

   - fixed `m` for benchmarking candidate dimensions; or
   - device-side residual flags collected after the wave solve.

9. Add warm starts and preconditioning last.

   Warm starts and diagonal preconditioning can reduce iterations, but they
   make correctness harder to reason about. They should be layered on after
   zero-start GMRES is stable and benchmarked.

The immediate deliverable should be a GPU-resident fixed-`m` GMRES path that
never leaves the GPU and records enough diagnostics to compare against Neumann
on the exact same forward state. That gives a fair answer to whether GMRES
reduces the expensive mathematical work before we spend time on deeper Triton
specialization.

## Progress Notes

### 2026-06-05 Apply-A and Fixed-M Prototype

Implemented and tested:

- The retained self-loop Triton matvec can now write `A x = x - J^T x`
  directly for GMRES. This removes a PyTorch subtraction kernel and the dummy
  GMRES accumulation buffer/zeroing from each matvec.
- The benchmark driver can run either adaptive `gmres` or fixed-iteration
  `gmres_fixed`.
- `gmres_fixed` performs exactly `m` Krylov steps per wave and solves the small
  least-squares problem once at the end of the wave. It deliberately trades a
  few more self-loop matvecs for fewer Python/PyTorch orchestration costs.
- GMRES masks inactive rows before constructing the Krylov basis and the
  Apply-A kernel writes zeros for inactive rows. This avoids full-tensor Arnoldi
  reductions reading uninitialized inactive scratch.
- A Python-level incremental Givens version was tested and rejected as the
  default. It removed repeated cuSOLVER calls, but introduced enough scalar GPU
  operations that the hard-family runtime regressed.

Current hard-family result on `CLU_000680_20_4_C`:

| Solver | Total backward iterations | Backward-only time | Relative L2 gradient error |
|---|---:|---:|---:|
| adaptive GMRES max-10 before Apply-A | `598` | `0.265298 s` | `6.589763e-06` |
| adaptive GMRES max-10 with Apply-A | `598` | `0.263357 s` | `6.589763e-06` |
| fixed GMRES m=10 with Apply-A | `680` | `0.208197 s` | `6.588901e-06` |
| fixed GMRES m=12 with Apply-A | `816` | `0.247367 s` | `1.074764e-07` |

The useful lesson is that minimizing VJP count alone is not enough for the
current Python prototype. Adaptive GMRES uses fewer matvecs, but fixed-m is
faster because it removes per-iteration least-squares solves and residual
checks. The next optimization should reduce the dot/norm/AXPY launch count,
because those now dominate the fixed-m profile.

Next concrete implementation step:

```text
Move modified Gram-Schmidt dot products, norm, and basis update for one GMRES
step into Triton kernels that operate on the preallocated basis tensor.
```

The first target should still keep wave control flow in Python, but one Arnoldi
step should use a small number of GPU kernels instead of many PyTorch
reductions and elementwise kernels. Only after that should we revisit
incremental Givens in Triton.

## Fusing `A = I - J^T`

The current self-loop kernel naturally computes:

```text
term_out = J^T term_in
```

GMRES needs:

```text
term_out = term_in - J^T term_in
```

The production path should add a kernel mode or wrapper that writes `A x`
directly. This avoids:

- a separate PyTorch subtraction kernel;
- an extra temporary tensor;
- unnecessary memory traffic.

A minimal design is a new constexpr mode:

```text
OP_MODE = APPLY_J | APPLY_A | FIXED_POINT_UPDATE
```

with behavior:

```text
APPLY_J:
    out = J^T in

APPLY_A:
    out = in - J^T in

FIXED_POINT_UPDATE:
    v = rhs + J^T v
```

This keeps the existing Neumann path intact while giving GMRES a direct
matrix-free `A` application.

## Batched and Wave-Level Considerations

The retained path processes one wave-local system at a time. Each wave has
shape `[W, S]`, where `W` is the number of rows in the wave and `S` is the
species tree size.

GMRES convergence can vary by wave. Therefore the solver should record
per-wave:

- initial residual norm;
- final residual norm;
- iteration count;
- whether it hit `gmres_max_iter`;
- whether a warm start was accepted;
- whether it fell back to Neumann or zero-start GMRES.

This data is not just diagnostics. It should drive an adaptive policy:

```text
if a wave often converges in <= 2 iterations:
    prefer warm start or short Neumann

if a wave often hits gmres_max_iter:
    increase max_iter, add preconditioning, or inspect conditioning

if many waves have tiny rhs:
    skip or loosen relative checks with an absolute tolerance
```

## Accuracy Policy

The solver tolerance should be chosen based on gradient accuracy, not only
linear residual accuracy.

Recommended validation procedure:

1. Choose hard representative families.
2. Compute reference gradients with a very high Neumann budget or a strict
   GMRES tolerance.
3. Compare candidate solvers by relative gradient error:

```text
||g_candidate - g_reference|| / ||g_reference||
```

4. Also record the linear residuals per wave:

```text
||rhs - A v|| / ||rhs||
```

5. Tune `gmres_tol`, `gmres_max_iter`, warm-start policy, and preconditioning
   until the gradient error is acceptable.

A good production target is probably not `1e-12` wave residuals. The useful
target is the weakest tolerance that leaves the optimizer trajectory unchanged
or statistically equivalent for the benchmark workload.

## Measuring Success

The main metric should be total expensive operator applications, then wall time.

Record:

- total `J^T` applications per full backward pass;
- mean, median, max GMRES iterations per wave;
- number of waves hitting `gmres_max_iter`;
- wall-clock backward time;
- end-to-end optimizer time to convergence;
- peak GPU memory;
- final objective and final gradient norm;
- gradient error against a reference on selected families.

For Neumann:

```text
operator_applications = neumann_terms * number_of_waves
```

For zero-start GMRES:

```text
operator_applications = sum(gmres_iterations_per_wave)
```

For warm-start GMRES:

```text
operator_applications =
    one residual application per warm-started wave
    + sum(gmres_correction_iterations_per_wave)
```

The warm-start residual application should be counted. It is a real `A v0`
application and includes one expensive `J^T v0`.

## Relationship To Pi Iterations

The gradient self-loop solve is linear once the forward quantities and current
`theta` are fixed. That is why GMRES is the right tool for:

```text
(I - J^T) v = rhs
```

The Pi fixed-point iteration is generally nonlinear. Plain GMRES does not apply
directly unless we linearize a specific system. If we want to accelerate the
nonlinear Pi iteration, the closer analogue is Anderson acceleration or another
nonlinear fixed-point accelerator.

Therefore, the recommended first implementation target is the gradient
self-loop solve, not the forward Pi iteration.

## Proposed Implementation Milestones

### Milestone 1: Production-Ready Zero-Start GMRES

Implement a solver option for the retained backward path:

```text
self_loop_solver = "gmres"
gmres_max_iter
gmres_tol
gmres_restart
```

At this stage:

- use zero initial guess;
- use preallocated basis/Hessenberg buffers;
- use direct `apply_A`;
- use incremental QR residual tracking;
- report per-wave iteration stats.

Success criterion:

```text
same gradient accuracy as Neumann with fewer J^T applications
```

### Milestone 2: Hybrid Neumann/GMRES

Add a policy that starts cheaply and escalates only when needed:

```text
neumann_warmup_terms = 2 or 4
switch_to_gmres_if residual_ratio > threshold
```

Success criterion:

```text
lower wall time than pure GMRES and pure high-budget Neumann
```

### Milestone 3: Warm Starts Across Optimizer Steps

Store previous wave-local `v` values and use correction GMRES:

```text
r0 = rhs_new - A_new previous_v
A_new delta = r0
v_new = previous_v + delta
```

Success criterion:

```text
fewer total J^T applications during end-to-end optimization
```

### Milestone 4: Diagonal Preconditioning

Add optional diagonal preconditioning using the existing diagonal self-loop
weights:

```text
M_inv ~= 1 / (1 - diag_wt)
```

Success criterion:

```text
fewer GMRES iterations on hard waves without increasing wall time
```

## Risks And Failure Modes

GMRES can reduce mathematical iterations but still fail to improve wall time if
implementation overhead is too high. The main risks are:

- too many small PyTorch operations per wave;
- repeated allocations inside the Krylov loop;
- CPU synchronization from residual checks;
- `torch.linalg.lstsq` overhead for every iteration;
- excessive memory use from storing basis vectors;
- loss of orthogonality in fp32;
- poor warm starts after large optimizer steps;
- relative residual checks becoming unstable when `rhs` is tiny.

Mitigations:

- preallocate all solver buffers;
- use incremental QR/Givens rotations;
- keep convergence checks on device where possible;
- cap Krylov dimension at small values first;
- optionally reorthogonalize;
- combine relative and absolute residual tolerances;
- fall back to Neumann or zero-start GMRES when warm starts are poor.

## Profile Update

An Nsight Systems profile on HOGENOM family `CLU_000680_20_4_C` confirms that
the current prototype overhead is mostly orchestration, not the retained
self-loop matvec.

Profile:

```text
benchmarks/large_dataset_capacity/output/nsys_gmres_profile_20260605/
benchmarks/large_dataset_capacity/output/nsys_gmres_profile_20260605_rerun/
GMRES max iterations: 10
total self-loop applications: 598
backward-only captured range: 349.275 ms
```

Backward-only totals:

```text
summed GPU kernels:             206.167 ms
CUDA API time:                  141.521 ms
GPU synchronization activity:    10.152 ms
actual J^T matvec kernel:         8.165 ms across 598 launches
cuSOLVER QR from lstsq:          19.697 ms across 598 launches
PyTorch GMRES dot reductions:    50.716 ms across 4100 launches
```

A clean rerun using
`benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py` captures
only `implicit_grad_loglik_vjp_wave` after preparing the forward state outside
the profiler range. It gives the same ordering:

```text
summed GPU kernels:             206.528 ms
CUDA API time:                 ~141.5 ms
GPU synchronization activity:    10.421 ms
actual J^T matvec kernel:         8.134 ms across 598 launches
cuSOLVER QR from lstsq:          19.707 ms across 598 launches
PyTorch GMRES dot reductions:    50.760 ms across 4100 launches
```

This means a Triton rewrite of only `_wave_backward_uniform_2d_jt_kernel` would
not address the main cost. The next implementation should first remove the
Python/PyTorch GMRES loop overhead: preallocate buffers, fuse or batch dot/norm
work where practical, and replace repeated `torch.linalg.lstsq` calls with
incremental GPU-resident QR/Givens updates.

## CGS2 Fixed-M Update

Commit `01a0faa` implements the first lower-overhead Arnoldi increment for
`gmres_fixed`. It keeps the existing Apply-A Triton matvec and final
`torch.linalg.lstsq`, but replaces the per-basis-vector modified
Gram-Schmidt loop with two batched classical Gram-Schmidt passes using dense
matrix-vector operations. This targets the profiler-dominant dot/update launch
overhead without changing the number of `J^T` applications.

Hard-family result on `CLU_000680_20_4_C`, `m=10`:

```text
previous fixed MGS: 680 J^T applications, 0.208197 s
batched CGS2:       680 J^T applications, 0.160145 s
relative L2 error vs Neumann=512: 6.588901e-06
```

Nsys confirms the expected movement:

```text
GPU kernels:            187.624 ms / 22,024 launches -> 129.190 ms / 12,368 launches
PyTorch sum reductions:  58.178 ms / 4,365 launches  ->   4.618 ms / 557 launches
elementwise kernels:     23.063 ms / 14,923 launches ->   7.277 ms / 4,927 launches
cuBLAS GEMV/update:       0.105 ms / 64 launches     ->  11.268 ms / 4,212 launches
J^T matvec:               9.112 ms / 680 launches    ->   9.104 ms / 680 launches
```

This is still not the final GPU-resident GMRES design. The remaining work is to
move the batched CGS/Givens residual update into custom kernels or another
lower-overhead path, and then reintroduce adaptive stopping without returning
to per-iteration host synchronization.

## Adaptive CGS2 Update

Commit `1e71da7` applies the batched CGS2 Arnoldi path to adaptive `gmres` and
adds `gmres_check_interval`. The solver always checks after the first Krylov
step, then every configured interval, and always at `max_iter`. This avoids the
near-breakdown failure mode observed on a tiny public CUDA smoke test while
allowing fewer least-squares residual checks on hard waves.

Hard-family results on `CLU_000680_20_4_C`, max `m=10`:

```text
old adaptive MGS: 598 J^T applications, 598 checks, 0.263807 s
adaptive CGS2 i1: 598 J^T applications, 598 checks, 0.232383 s
adaptive CGS2 i4: 638 J^T applications, 251 checks, 0.187266 s
adaptive CGS2 i5: 670 J^T applications, 202 checks, 0.179272 s
fixed CGS2 m10:   680 J^T applications,  68 checks, 0.160145 s
```

All these max-10 GMRES variants have approximately the same relative L2
gradient error versus Neumann=512 on this family: about `6.59e-06`.

Nsys for adaptive CGS2 interval `4`:

```text
GPU kernels:            206.592 ms / 30,093 launches -> 137.375 ms / 15,362 launches
CUDA API:               142.419 ms / 69,317 calls    -> 101.383 ms / 34,754 calls
PyTorch sum reductions:  50.868 ms / 4,102 launches  ->   5.064 ms / 740 launches
QR/lstsq kernels:        26.721 ms / 2,324 launches  ->  11.928 ms / 1,004 launches
J^T matvec:               8.134 ms / 598 launches    ->   8.596 ms / 638 launches
```

The result is useful but not final. Coarse-check adaptive GMRES can reduce VJP
count versus fixed `m=10`, but fixed CGS2 is still faster because residual
checks remain expensive. The next meaningful step is a GPU-resident
small-Hessenberg residual update, likely via Givens rotations, without the
scalar-heavy Python/GPU interaction that made the earlier prototype slow.

## Triton Hessenberg Residual Update

Commit `198fa4b` adds a residual-only Triton Givens QR checker for adaptive
GMRES. It is deliberately narrower than full incremental GMRES:

- CGS2 Arnoldi is unchanged.
- checkpoint residual checks use the Triton kernel for `m <= 32`;
- CPU and larger-`m` cases fall back to `torch.linalg.lstsq`;
- the final solve for GMRES coefficients still uses one `torch.linalg.lstsq`
  per wave.

Hard-family result on `CLU_000680_20_4_C`, max `m=10`, check interval `3`:

```text
adaptive CGS2 I3 with Triton residual:
  619 J^T applications
  299 residual checks
  0.168223 s backward-only
  6.589550e-06 relative L2 gradient error vs Neumann=512

fixed CGS2 M10:
  680 J^T applications
  0.160145 s backward-only
  6.588901e-06 relative L2 gradient error vs Neumann=512
```

Nsys:

```text
GPU kernels:     old adaptive 206.592 ms / 30,093 launches
                 adaptive I3  127.934 ms / 11,935 launches
                 fixed M10    129.190 ms / 12,368 launches

QR/lstsq kernels: old adaptive 26.721 ms / 2,324 launches
                  adaptive I3   5.030 ms / 272 launches
                  fixed M10     5.457 ms / 272 launches

Triton residual kernel: adaptive I3 1.863 ms / 299 launches
```

This confirms that replacing checkpoint `lstsq` residual checks with a tiny
GPU-resident residual computation is the right direction. The remaining gap is
now small enough that the next decision should be driven by end-to-end
optimization experiments and/or replacing the final per-wave coefficient solve.

## Triton QR/Backsolve Update

Commit `8f5f50c` extends the small-Hessenberg Triton kernel to compute the final
GMRES coefficients for CUDA solves with `m <= 16`. This removes the last
per-wave `torch.linalg.lstsq` from the hot path while keeping the CPU and
larger-`m` fallback.

Hard-family result on `CLU_000680_20_4_C`, max/fixed `m=10`:

```text
adaptive CGS2 I3 + Triton QR:
  619 J^T applications
  299 residual checks
  0.162774 s backward-only
  6.589550e-06 relative L2 gradient error vs Neumann=512

fixed CGS2 M10 + Triton QR:
  680 J^T applications
  68 residual checks
  0.151533 s backward-only
  6.588901e-06 relative L2 gradient error vs Neumann=512
```

Nsys:

```text
GPU kernels:       old adaptive 206.592 ms / 30,093 launches
                   adaptive I3  122.871 ms / 10,847 launches
                   fixed M10    129.190 ms / 12,368 launches

QR/lstsq kernels:  old adaptive 26.721 ms / 2,324 launches
                   adaptive I3   0.000 ms / 0 launches
                   fixed M10     5.457 ms / 272 launches

Triton Hessenberg kernel:
                   adaptive I3   2.865 ms / 367 launches
```

The mathematical-work metric now favors adaptive GMRES (`619` vs `680` VJPs),
and the profiler kernel total favors adaptive as well. The backward-only wall
clock is still slightly better for fixed M10 on this single family, so the next
required proof is the user-facing one: end-to-end time to convergence on the
HOGENOM benchmark.

## Fresh Nsys/NCU Check

Code at `d87135bc` was profiled again with
`benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py` on the
same hard HOGENOM family, `CLU_000680_20_4_C`, using adaptive GMRES max `10`,
tolerance `1e-10`, and check interval `3`.

Artifacts:

```text
benchmarks/large_dataset_capacity/output/current_gmres_tritonqr_i3_profile_20260606_055419/
benchmarks/large_dataset_capacity/output/nsys_current_gmres_tritonqr_i3_profile_20260606_055419/
benchmarks/large_dataset_capacity/output/ncu_current_gmres_tritonqr_i3_hessenberg_basic_20260606_055419/
benchmarks/large_dataset_capacity/output/ncu_current_gmres_tritonqr_i3_jt_grid28_basic_20260606_055419/
```

Unprofiled backward-only result:

```text
elapsed:                    0.161840 s
waves:                      68
total J^T applications:     619
total GMRES checks:         299
mean iterations per wave:   9.103
max iterations per wave:    10
max relative residual:      3.849e-06
```

`nsys` captured the CUDA-profiler range around the backward solve:

```text
summed GPU kernels:         122.921 ms across 10,847 launches
CUDA API time:               87.269 ms across 24,671 calls
CUDA memcpy API time:        63.703 ms across 2,966 calls
GPU memcpy time:              3.309 ms across 2,966 copies
```

Top kernel buckets:

```text
_dts_ge2_stage1_kernel:                    25.924 ms /   66 launches
_wave_backward_uniform_2d_precompute:      24.916 ms /   68 launches
PyTorch reductions, grouped:               13.630 ms / 1192 launches
_dts_cross_backward_accum_kernel:           8.692 ms /   67 launches
_wave_backward_uniform_2d_jt_kernel:        8.333 ms /  619 launches
cuBLAS dot kernels:                         4.021 ms / 1238 launches
cuBLAS GEMV kernels:                        3.341 ms / 1084 launches
_gmres_hessenberg_residual_kernel:          2.866 ms /  367 launches
```

There are no cuSOLVER least-squares kernels left in this captured hot path. The
remaining GMRES overhead is therefore launch/control overhead plus PyTorch
reduction and cuBLAS dot/GEMV work, not dense CPU or cuSOLVER solves.

`ncu --set basic` on `_gmres_hessenberg_residual_kernel` sampled three launches:

```text
grid/block:                 1 block x 32 threads
duration:                   3.30 us, 5.34 us, 8.58 us
DRAM throughput:            0.26% - 0.27%
compute throughput:         0.10% - 0.22%
achieved occupancy:         2.08% - 2.10%
```

This kernel is intentionally tiny and underfilled. A direct Triton-to-Gluon
rewrite of only this one-block QR/backsolve kernel is unlikely to matter.

`ncu --set basic` on `_wave_backward_uniform_2d_jt_kernel` first sampled
one-block edge cases, so the representative sample skips to the most frequent
high-cost shape in the Nsys launch distribution:

```text
grid/block:                 28 blocks x 64 threads
duration:                   18.91 us - 19.10 us
registers/thread:           254
DRAM throughput:            8.15% - 8.24%
compute throughput:         4.40% - 4.49%
achieved occupancy:         4.15% - 4.23%
```

The matvec is register-limited and also underfills this GPU on the single hard
family. Gluon could be useful if it lets us reduce register pressure or fuse
GMRES bookkeeping with the matvec in a way that cuts launches and scalar host
checks. It should not be treated as a drop-in replacement for the current tiny
Hessenberg kernel.

## CUDA Host-Sync Reduction

The next low-risk optimization removed avoidable scalar device-to-host reads in
the CUDA GMRES path:

- the RHS norm is kept on device when the Triton Hessenberg path can handle the
  residual denominator;
- the post-check Arnoldi norm is no longer copied to the host for CUDA
  breakdown checks;
- the final residual is not copied when it is only needed for optional stats.

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_cuda_sync_reduction_20260606_060359/
benchmarks/large_dataset_capacity/output/nsys_gmres_cuda_sync_reduction_20260606_060359/
benchmarks/large_dataset_capacity/output/gmres_cuda_sync_reduction_correctness_20260606_060359/
```

Hard-family backward-only result on `CLU_000680_20_4_C`, adaptive GMRES max
`10`, tolerance `1e-10`, check interval `3`:

```text
before: 0.161840 s, 619 J^T applications, 299 residual checks
after:  0.149429 s, 619 J^T applications, 299 residual checks
```

The gradient was unchanged to the displayed precision:

```text
[-4.928546352910933, -2.3777557207255815, 0.8579805383195991]
```

`nsys` shows the expected movement in API and copy traffic while kernel work
stays effectively unchanged:

```text
GPU kernels:             122.921 ms / 10,847 launches -> 122.575 ms / 10,847 launches
cudaMemcpyAsync API:      63.703 ms /  2,966 calls    ->  50.495 ms /  2,500 calls
Device-to-Host copies:     0.621 ms /    747 copies   ->   0.290 ms /    349 copies
cudaStreamSynchronize:     0.536 ms /    747 calls    ->   0.331 ms /    349 calls
```

Correctness against Neumann-512 on the same family:

```text
Neumann32: 2176 J^T applications, rel L2 3.459027e-05
GMRES10:   619 J^T applications, rel L2 6.589550e-06
```

The full family loss-gradient harness reports similar elapsed times for
Neumann32 and GMRES10 because it includes the forward `E/Pi/Pibar` work around
the backward solve. The backward-only profile is the right evidence for this
particular overhead patch.

## End-to-End Benchmark Driver Knobs

The broader capacity benchmark driver now exposes the GMRES self-loop solver
controls needed for HOGENOM optimizer experiments:

```text
benchmarks/large_dataset_capacity/run_gpurec_benchmark.py
--self-loop-solver neumann|gmres|gmres_fixed
--gmres-max-iter N
--gmres-tol TOL
--gmres-check-interval N
```

`--gmres-max-iter` is an optional clarity alias for GMRES runs. If it is
omitted, the driver keeps using `--neumann-terms` as the effective Krylov
maximum, matching the lower-level `SolverOptions` field. The output JSON now
records the effective `solver_options` block, so a GMRES run is reproducible
without inferring which CLI value controlled the max iteration count.

Single-family HOGENOM smoke artifact:

```text
benchmarks/large_dataset_capacity/output/gmres_driver_smoke_20260606_061011/run.json
```

The smoke used `--self-loop-solver gmres --gmres-max-iter 2` and verified that
the driver constructed `solver_options.self_loop_solver = "gmres"` with
`solver_options.neumann_terms = 2`, then completed one optimizer step on the
single-family probe.

The driver now also records per-step self-loop backward work:

```text
self_loop_backward_pass_count
self_loop_waves_per_backward
self_loop_wave_solves
self_loop_backward_iterations
self_loop_mean_iterations_per_wave
gmres_total_checks
gmres_max_rel_res
```

For Neumann, `self_loop_backward_iterations` is computed from the number of
actual backward passes, the number of waves per backward pass, and the fixed
term count. For GMRES and `gmres_fixed`, it is the sum of per-wave Krylov
iterations recorded by the GMRES solver. This makes the optimizer benchmark
report the requested expensive-work metric directly.

Paired single-family accounting smoke:

```text
benchmarks/large_dataset_capacity/output/self_loop_accounting_smoke_20260606_061411/
```

Both runs used one Adam step on the same HOGENOM single-family probe with
`e_max_iter=4` and `pi_iters=2`:

```text
Neumann terms=2:    12 wave solves, 24 self-loop backward iterations
GMRES max_iter=2:   12 wave solves, 19 self-loop backward iterations, 19 checks
```

### Largest-10 End-to-End Timing Check

Artifact:

```text
benchmarks/large_dataset_capacity/output/gmres_vs_neumann_capacity_largest10_steps20_20260606_061809/
```

Settings:

```text
HOGENOM largest 10 families by gene-tree file size
one resident batch, clade_budget=500000
20 Adam steps, lr=0.01
e_max_iter=16, pi_iters=16
```

Results:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Train Time | Final Loss |
|---|---:|---:|---:|---:|
| Neumann16 | `27840` | n/a | `2.557 s` | `166606.4375` |
| Neumann32 | `55680` | n/a | `3.066 s` | `166606.4375` |
| Neumann64 | `111360` | n/a | `4.123 s` | `166606.4375` |
| GMRES10 I3 | `4443` | `3021` | `3.150 s` | `166606.4375` |
| GMRES4 I4 | `5160` | `2880` | `3.391 s` | `166606.4375` |
| GMRES10 I10 | `12000` | `2880` | `4.593 s` | `166606.4375` |

Interpretation:

- GMRES10 I3 cuts self-loop applications by `12.5x` versus Neumann32 and
  `25.1x` versus Neumann64.
- On this warmed 10-family end-to-end run, GMRES10 I3 is still slightly slower
  than Neumann32 (`3.150 s` vs `3.066 s`) and faster than Neumann64
  (`3.150 s` vs `4.123 s`).
- The identical displayed loss trajectory means this is a useful overhead
  measurement, not an optimizer-quality failure.
- Coarser residual checks (`I10`) and fixed small GMRES were not a solution:
  I10 spent too many Krylov iterations, and fixed-m runs hit an E-adjoint
  BiCGSTAB NaN on this setup.

This is the clearest current end-to-end evidence: GMRES is doing much less
mathematical work, but the remaining Python/PyTorch Arnoldi and residual-check
overhead still prevents a wall-time win against practical Neumann16/32 budgets
on this subset. The next implementation step should therefore move Arnoldi
vector algebra out of the PyTorch/cuBLAS loop or reduce residual-check/control
overhead further.

### Opt-In Triton Split-Arnoldi Prototype

I implemented an experimental split-Arnoldi path for adaptive CUDA float32
GMRES. It replaces the PyTorch/cuBLAS modified Gram-Schmidt bookkeeping with
four Triton kernels per Arnoldi step:

```text
_gmres_arnoldi_dot_partials_kernel
_gmres_arnoldi_reduce_project_dot_kernel
_gmres_arnoldi_reduce_project_norm_kernel
_gmres_arnoldi_reduce_norm_normalize_kernel
```

This path is deliberately opt-in:

```bash
GPUREC_GMRES_TRITON_ARNOLDI=1
```

The default remains the existing PyTorch CGS2 Arnoldi path. The Triton version
is mathematically useful but not yet production-safe as a default because cold
JIT compilation dominates short runs and because large waves can still fall
back to CGS2.

Validation:

```text
pytest -q tests/test_gmres_self_loop_solver.py tests/test_large_dataset_capacity_benchmark.py
20 passed
```

The CUDA float32 regression asserts that the Triton split path runs and matches
a dense solve within float32 tolerance.

Hard-family backward-only comparison on `CLU_000680_20_4_C`, adaptive GMRES10,
tolerance `1e-10`, check interval `3`, dtype `float32`:

| Backend | Elapsed | Waves | J^T Applications | GMRES Checks | Backend Counts |
|---|---:|---:|---:|---:|---|
| PyTorch CGS2 | `0.1068 s` | `68` | `619` | `299` | `torch_cgs2: 68` |
| Triton split Arnoldi | `0.0817 s` | `68` | `619` | `299` | `triton_split: 68` |

Artifacts:

```text
benchmarks/large_dataset_capacity/output/current_gmres_cgs2_float32_i3_profile_prep_20260606_063707/
benchmarks/large_dataset_capacity/output/current_gmres_triton_arnoldi_profile_prep_20260606_063333/
benchmarks/large_dataset_capacity/output/nsys_gmres_cgs2_float32_i3_profile_20260606_063735/
benchmarks/large_dataset_capacity/output/nsys_gmres_triton_arnoldi_float32_i3_profile_20260606_063459/
benchmarks/large_dataset_capacity/output/ncu_gmres_triton_arnoldi_float32_i3_reduce_project_dot_basic_20260606_063529/
benchmarks/large_dataset_capacity/output/ncu_gmres_triton_arnoldi_float32_i3_jt_basic_20260606_063625/
```

The `nsys` profile shows the Triton path cuts launch/copy overhead relative to
float32 CGS2 for this backward-only workload:

| Metric | CGS2 | Triton Split |
|---|---:|---:|
| summed GPU kernels | `34.830 ms`, `10753` launches | `24.751 ms`, `6718` launches |
| CUDA runtime API | `31.702 ms`, `23247` calls | `18.128 ms`, `10710` calls |
| device-to-device copies | `321.063 MB`, `2151` copies | `30.879 MB`, `226` copies |
| device-to-host copies | `349` copies | `349` copies |

Top Triton-split kernel buckets:

| Kernel Bucket | Time | Launches |
|---|---:|---:|
| `_wave_backward_uniform_2d_jt_kernel` | `5.987 ms` | `619` |
| `_gmres_arnoldi_reduce_project_dot_kernel` | `2.240 ms` | `619` |
| `_gmres_arnoldi_reduce_project_norm_kernel` | `1.688 ms` | `619` |
| `_gmres_arnoldi_dot_partials_kernel` | `1.334 ms` | `619` |
| `_gmres_arnoldi_reduce_norm_normalize_kernel` | `0.920 ms` | `619` |
| `_gmres_hessenberg_residual_kernel` | `0.763 ms` | `367` |

`ncu --set basic` on `_gmres_arnoldi_reduce_project_dot_kernel` sampled three
launches:

| Metric | Sampled Values |
|---|---:|
| grid/block | `47-52 x 256` |
| duration | `4.32 us` - `4.77 us` |
| registers/thread | `48` |
| DRAM throughput | `5.4%` - `14.8%` |
| SM throughput | `10.9%` - `13.0%` |
| active-warps occupancy | `16.2%` - `16.4%` |

`ncu --set basic` on `_wave_backward_uniform_2d_jt_kernel` sampled three
launches:

| Metric | Sampled Values |
|---|---:|
| grid/block | `18-20 x 64` |
| duration | `15.07 us` - `15.39 us` |
| registers/thread | `182` |
| DRAM throughput | `3.4%` - `3.8%` |
| SM throughput | `1.15%` - `1.28%` |
| active-warps occupancy | `4.14%` - `4.23%` |

The bottleneck remains underfilled small kernels and launch/control overhead,
not a dense solve and not a single throughput-saturating kernel.

Largest-10 opt-in run:

```text
benchmarks/large_dataset_capacity/output/gmres_triton_arnoldi_largest10_steps20_20260606_063130/
```

Same settings as the previous largest-10 comparison, with
`GPUREC_GMRES_TRITON_ARNOLDI=1` in effect at the time of the run.

| Metric | Value |
|---|---:|
| train seconds | `33.956 s` |
| wall seconds | `35.100 s` |
| step 1 seconds | `30.655 s` |
| mean step seconds, steps 2-20 | `0.174 s` |
| total self-loop backward iterations | `4769` |
| backend counts | `triton_split: 920`, `torch_cgs2: 820` |
| final loss | `166606.40625` |

The first step is dominated by Triton compilation. Warm steps without new JIT
specializations are near `0.124 s`, which is essentially the same as the
previous GMRES10 I3 and Neumann32 warmed steps. Some later steps still had
JIT-looking spikes (`0.626 s` and `0.540 s`). This is why the prototype is
opt-in: it improves one backward-only float32 hard-family solve, but it does
not yet improve the broader end-to-end benchmark.

#### Specialization Reduction Follow-Up

The first Triton prototype still specialized on values that are scalar runtime
inputs rather than true compile-time structure. I changed:

- `NUM_TILES` in the split-Arnoldi reduction kernels from `tl.constexpr` to a
  runtime scalar;
- `ITERS` in `_gmres_hessenberg_residual_kernel` from `tl.constexpr` to a
  runtime scalar;
- `do_not_specialize` for `N`, `J`, `NUM_TILES`, and residual `ITERS`;
- the opt-in Arnoldi reduction bucket to use at least `BLOCK_TILES=64`, so
  small waves do not compile separate 1/2/4/8/16/32-tile variants.

This does not change the GMRES math or the number of `J^T` applications. It
only reduces the number of Triton variants the opt-in path asks the compiler
to build.

One-family isolated-cache smoke, same largest HOGENOM family as above,
`steps=2`, adaptive GMRES10 I3:

| Variant | Wall Seconds | Step 1 | Step 2 | Triton Cache Files |
|---|---:|---:|---:|---:|
| original opt-in Triton split | `41.30` | `39.35` | `0.985` | `3668` |
| runtime `NUM_TILES` only | `26.02` | `24.97` | `0.075` | `1940` |
| runtime scalars + min `BLOCK_TILES=64` | `18.64` | `17.61` | `0.075` | `1121` |

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_triton_arnoldi_runtime_num_tiles_cold_smoke_20260606_064326/
benchmarks/large_dataset_capacity/output/gmres_triton_arnoldi_donotspecialize_cold_smoke_20260606_064801/
```

Largest-10 cold-cache opt-in run:

```text
benchmarks/large_dataset_capacity/output/gmres_triton_arnoldi_donotspecialize_largest10_steps20_20260606_064842/
```

| Metric | Before | After |
|---|---:|---:|
| wall seconds | `35.100` | `25.419` |
| train seconds | `33.956` | `24.278` |
| step 1 seconds | `30.655` | `21.931` |
| mean step seconds, steps 2-20 | `0.174` | `0.124` |
| late JIT-looking spikes | `0.626 s`, `0.540 s` | none observed |
| cache files | not isolated | `1512` |
| total self-loop backward iterations | `4769` | `4863` |
| backend counts | `triton_split: 920`, `torch_cgs2: 820` | same |

Largest-10 rerun with the same warmed Triton cache:

```text
benchmarks/large_dataset_capacity/output/gmres_triton_arnoldi_donotspecialize_largest10_steps20_warm_20260606_064937/
```

| Metric | Value |
|---|---:|
| train seconds | `3.071` |
| wall seconds | `4.214` |
| step 1 seconds | `0.732` |
| mean step seconds, steps 2-20 | `0.123` |
| total self-loop backward iterations | `4865` |
| backend counts | `triton_split: 920`, `torch_cgs2: 820` |
| final loss | `166606.625` |

This is a real overhead improvement, but it still does not prove the final
goal. The warmed opt-in GMRES10 run now roughly ties the earlier Neumann32
train time (`3.071 s` vs `3.066 s`) while using about `11.4x` fewer self-loop
applications (`4865` vs `55680`). It still does not beat Neumann16, and the
cold-cache run is still dominated by first-step compilation. The next useful
targets are therefore either prewarming unavoidable variants for optimizer
runs or attacking the remaining launch/control overhead, especially residual
checks and the large-wave CGS2 fallback.

#### GMRES10 I4 Timing And Correctness

After the specialization patch, I tested a coarser residual-check schedule:
GMRES max `10`, tolerance `1e-10`, check interval `4`, with the opt-in Triton
Arnoldi backend and a warm Triton cache.

Artifact:

```text
benchmarks/large_dataset_capacity/output/gmres_triton_max10_i4_largest10_steps20_20260606_065516/
```

Largest-10 HOGENOM, same short Adam setup:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Train Seconds | Wall Seconds | Mean Step 2-20 | Final Loss |
|---|---:|---:|---:|---:|---:|---:|
| Neumann32 baseline | `55680` | n/a | `3.066` | `4.219` | `0.1247` | `166606.4375` |
| GMRES10 I3 opt-in Triton | `4865` | about `3120` | `3.071` | `4.214` | `0.1231` | `166606.625` |
| GMRES10 I4 opt-in Triton | `5493` | `2971` | `3.006` | `4.169` | `0.1200` | `166606.46875` |

`GMRES10 I4` uses slightly more self-loop applications than I3, but fewer
residual checks and lower per-step overhead. On this warmed largest-10 run it
is the first measured GMRES setting that beats the Neumann32 train time while
using about `10.1x` fewer self-loop `J^T` applications.

Hard-family correctness against the high `Pi`/Neumann reference remained
better than Neumann32:

Artifact:

```text
benchmarks/large_dataset_capacity/output/gmres10_i4_correctness_hard_family_20260606_065537/
```

`CLU_000680_20_4_C`, reference Neumann512:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Relative L2 Error | Relative Inf Error | Gradient |
|---|---:|---:|---:|---:|---|
| Neumann32 | `2176` | n/a | `3.459027e-05` | `3.307442e-05` | `[-4.9283518950504375, -2.3778636587336126, 0.8579352123043589]` |
| GMRES10 I4 | `638` | `251` | `6.588813e-06` | `6.380493e-06` | `[-4.9285463491555435, -2.377755722398327, 0.8579805382449783]` |

This is progress toward the target state, not the final endpoint. GMRES10 I4
now beats Neumann32 on the warmed largest-10 timing, but it still does not beat
Neumann16 (`2.557 s` train time in the same benchmark), and the result still
depends on opt-in Triton plus a warm cache. The next work should therefore
focus on removing more residual-check/control overhead, prewarming unavoidable
variants in a reproducible way, or eliminating the large-wave CGS2 fallback.

#### Follow-Up Profiling And Rejected Knobs

A subagent review recommended a mixed-backend versus all-CGS2 `nsys` comparison
before attempting a hierarchical large-wave Triton Arnoldi implementation. I
ran that comparison on one warmed largest-10 backward pass with GMRES10 I4.

Artifact:

```text
benchmarks/large_dataset_capacity/output/nsys_largest10_backward_mixed_vs_cgs2_i4_20260606_070108/
```

The captured range starts after the forward pass and covers only
`loss.backward()`.

| Metric | Mixed Opt-In Triton | All CGS2 |
|---|---:|---:|
| backend counts | `triton_split: 46`, `torch_cgs2: 41` | `torch_cgs2: 87` |
| elapsed backward-only time | `0.109 s` | `0.114 s` |
| self-loop backward iterations | `273` | `258` |
| GMRES checks | `148` | `144` |
| summed GPU kernels | `52.240 ms`, `5592` launches | `52.603 ms`, `6222` launches |
| CUDA runtime API | `28.748 ms`, `10667` calls | `32.236 ms`, `12848` calls |
| device-to-device copies | `1910.540 MB`, `716` copies | `2157.969 MB`, `1125` copies |
| device-to-host copies | `154` copies | `150` copies |

The mixed opt-in Triton path cuts launches, runtime calls, and D2D copies, but
the total kernel time is essentially unchanged in this largest-10 backward
profile. This is not enough evidence to justify a medium-high-risk
hierarchical large-wave Arnoldi implementation yet; it suggests the current
fallback is only one part of the remaining Neumann16 gap.

I also tested coarser residual-check intervals beyond I4:

```text
benchmarks/large_dataset_capacity/output/gmres_triton_check_sweep_largest10_steps20_20260606_070219/
```

| Solver | Train Seconds | Self-Loop Backward Iterations | GMRES Checks | Mean Step 2-20 | Final Loss |
|---|---:|---:|---:|---:|---:|
| GMRES10 I4 opt-in Triton | `3.006` | `5493` | `2971` | `0.1200` | `166606.46875` |
| GMRES10 I5 opt-in Triton | `3.145` | `6539` | `2934` | `0.1264` | `166606.765625` |
| GMRES10 I6 opt-in Triton | `3.283` | `7608` | `2914` | `0.1335` | `166606.640625` |

I4 remains the best measured residual-check schedule in this set. I5 and I6
save few checks but add too many Krylov iterations.

Finally, I tested expanding opt-in Triton coverage by increasing the Arnoldi
tile width from `512` to `1024`. This moved largest-10 backend coverage from
`46/87` Triton waves to `80/87`, but it did not improve the benchmark:

```text
benchmarks/large_dataset_capacity/output/gmres_triton_block1024_i4_largest10_steps20_20260606_070354/
```

| Variant | Train Seconds | Self-Loop Backward Iterations | GMRES Checks | Mean Step 2-20 | Backend Counts | Final Loss |
|---|---:|---:|---:|---:|---|---:|
| `BLOCK_N=512`, GMRES10 I4 | `3.006` | `5493` | `2971` | `0.1200` | `triton_split: 920`, `torch_cgs2: 820` | `166606.46875` |
| `BLOCK_N=1024`, GMRES10 I4 | `3.023` | `6176` | `3166` | `0.1205` | `triton_split: 1600`, `torch_cgs2: 140` | `166608.734375` |

The broader Triton coverage changed the numerical/iteration trajectory enough
to increase work and worsen the displayed final loss. I reverted it and kept
`BLOCK_N=512`.

Implication for a Triton or Gluon rewrite: rewriting only the small
Hessenberg/residual kernel is not enough. The useful target is a larger fused
GMRES step or a lower-compilation, lower-launch Arnoldi implementation. Gluon
may help if it can express that fused step with less specialization overhead or
lower register pressure, but it should be judged against `nsys` launch counts
and end-to-end warmed/cold timings, not just one kernel's microseconds.

#### Corrected Genewise Ordering And Conservative Triton Coverage

A follow-up batch-gradient check found a separate correctness issue in the
single-batch `genewise` model path: it passed the full `theta` tensor directly
to the one-batch autograd function instead of indexing `theta` into the static
batch order first. The streamed multi-batch path already did the correct
`theta.index_select(...)` and scatter. This is now fixed in
`gpurec/api/model.py`, with a regression test that checks non-identity batch
order gradient scatter.

That fix forced a rerun of the largest-10 benchmark results. It also exposed a
real accuracy problem in the broad opt-in Triton Arnoldi coverage: with the
previous `1024` block-tile cap, adaptive GMRES10 I4 could look fast but produced
full-batch first-step gradients with milliscale relative error against
Neumann512. Disabling Triton Arnoldi, or capping it to at most `512` block
tiles, restored full-batch gradient agreement. The opt-in backend now uses the
conservative `512` cap.

Main corrected artifact:

```text
benchmarks/large_dataset_capacity/output/gmres_cap512_corrected_gradient_and_timing_20260606_072639/
```

Largest-10 fixed first-step gradient check, `E/Pi=16`, pruned adjoint path,
reference `Neumann512`:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Relative L2 Error | Relative Inf Error | Max Family Relative L2 | Backend Counts |
|---|---:|---:|---:|---:|---:|---|
| Neumann32 | `2784` | n/a | `9.098405e-08` | `1.234574e-07` | `2.090796e-07` | n/a |
| GMRES10 I4 tol `1e-10` | `261` | `145` | `1.440077e-07` | `1.851861e-07` | `1.944992e-07` | `triton_split: 28`, `torch_cgs2: 59` |
| GMRES10 I4 tol `1e-6` | `219` | `131` | `2.987016e-07` | `5.555583e-07` | `1.092594e-06` | `triton_split: 28`, `torch_cgs2: 59` |

Corrected largest-10 20-step Adam timing with the same cap-512 code path:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Train Seconds | Wall Seconds | Mean Step 2-20 | Final Loss |
|---|---:|---:|---:|---:|---:|---:|
| Neumann16 | `27840` | n/a | `2.560` | `3.718` | `0.0997` | `166606.4375` |
| Neumann32 | `55680` | n/a | `3.074` | `4.367` | `0.1251` | `166606.4375` |
| GMRES10 I4 tol `1e-10` | `5223` | `2901` | `3.053` | `4.190` | `0.1218` | `166606.4375` |
| GMRES10 I4 tol `1e-6` | `4380` | `2620` | `3.011` | `4.184` | `0.1196` | `166606.4375` |

The corrected result is therefore narrower but more defensible:

- GMRES10 I4 with the conservative Triton cap gives accurate full-batch
  gradients against `Neumann512` for this largest-10 first-step check.
- It cuts self-loop `J^T` applications by `6.4x` versus Neumann16 and `12.7x`
  versus Neumann32.
- It beats Neumann32 train time (`3.011 s` vs `3.074 s` for tol `1e-6`), but it
  still does not beat Neumann16 (`2.560 s`).

The all-CGS2 control was important:

```text
benchmarks/large_dataset_capacity/output/gmres_corrected_model_cgs2_gradient_check_largest10_e16pi16_pruned_20260606_072402/
```

With `GPUREC_GMRES_TRITON_ARNOLDI=0`, GMRES10 I4 tol `1e-10` matched
Neumann512 at `1.310482e-07` relative L2 and `1.851861e-07` relative inf. The
accuracy problem was therefore not GMRES itself; it was the too-broad opt-in
Triton split-Arnoldi coverage.

Additional `nsys`/`ncu` profiling on the hard family with GMRES10 I4 tol
`1e-6`:

```text
benchmarks/large_dataset_capacity/output/nsys_gmres10_i4_tol1e-6_hard_family_20260606_071815/
```

The `nsys` backward-only capture recorded `470` self-loop applications and
`190` GMRES checks. Top kernel buckets were:

| Kernel Bucket | Total Time | Launches |
|---|---:|---:|
| `_wave_backward_uniform_2d_jt_kernel` | `4.641 ms` | `470` |
| PyTorch reduction kernels | `1.949 ms` | `487` |
| `_gmres_arnoldi_reduce_project_dot_kernel` | `1.807 ms` | `470` |
| `_gmres_arnoldi_reduce_project_norm_kernel` | `1.297 ms` | `470` |
| `_gmres_arnoldi_dot_partials_kernel` | `1.039 ms` | `470` |
| `_gmres_hessenberg_residual_kernel` | `0.800 ms` | `258` |

`ncu --set basic` confirmed these are tiny, underfilled kernels rather than a
single saturated kernel:

| Kernel | Duration | Grid | Registers/Thread | Achieved Occupancy |
|---|---:|---:|---:|---:|
| `_wave_backward_uniform_2d_jt_kernel` | about `16.1 us` | `1` block | `168` | about `4.2%` |
| `_gmres_arnoldi_reduce_project_dot_kernel` | about `4.1 us` | `3` blocks | `48` | about `16.7%` |

This strengthens the implementation conclusion: the remaining wall-time gap is
mostly launch/control overhead around many small wave-local kernels. A Triton
or Gluon rewrite is only worth doing if it fuses more of each GMRES step or
reduces launch count without broadening numerically unsafe coverage.

#### Interval-1 Tolerance Tuning And Buffer Reuse

After the cap-512 correction, the full-batch gradient check had enough margin
to tune the residual tolerance and check interval. The best passing setting in
this sweep was GMRES max `10`, tolerance `7e-6`, and check interval `1`.

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_cap512_tol7e-6_check_interval_sweep_largest10_20260606_073256/
benchmarks/large_dataset_capacity/output/gmres_buffer_reuse_fixed_tol7e-6_i1_largest10_20260606_073856/
```

Largest-10 first-step gradient check against Neumann512 after the buffer-reuse
patch:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Relative L2 Error | Relative Inf Error | Max Family Relative L2 |
|---|---:|---:|---:|---:|---:|
| Neumann32 | `2784` | n/a | `1.504496e-07` | `1.851861e-07` | `2.569847e-07` |
| GMRES10 I1 tol `7e-6` | `140` | `140` | `1.778534e-06` | `3.518535e-06` | `3.520319e-06` |

Corrected largest-10 20-step timing, comparing against the previous cap-512
result and the fixed Neumann baselines:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Train Seconds | Wall Seconds | Mean Step 2-20 | Final Loss |
|---|---:|---:|---:|---:|---:|---:|
| Neumann16 | `27840` | n/a | `2.560` | `3.718` | `0.0997` | `166606.4375` |
| Neumann32 | `55680` | n/a | `3.074` | `4.367` | `0.1251` | `166606.4375` |
| GMRES10 I4 tol `1e-6`, cap-512 | `4380` | `2620` | `3.011` | `4.184` | `0.1196` | `166606.4375` |
| GMRES10 I1 tol `7e-6`, cap-512 | `2800` | `2800` | `2.757` | `3.916` | `0.1065` | `166606.453125` |

At this point, this was the best corrected largest-10 GMRES setting. It used
about `9.9x` fewer self-loop applications than Neumann16 and about `19.9x`
fewer than Neumann32. It was faster than Neumann32 by `0.317 s`, but still
slower than Neumann16 by `0.197 s` in this 20-step timing.

The implementation change behind the last row is intentionally small:

- the precompute kernel now zeroes inactive scratch rows for GMRES, so the
  already-materialized `v_k` buffer is a valid masked RHS;
- the GMRES matvec reuses `spec_buf` instead of allocating a separate
  `gmres_a_buf`;
- `_gmres_solve_wave_self_loop` accepts an optional output tensor, allowing the
  final solution to be written directly back into `v_k`.

This preserves the Krylov iteration decisions. The first-step GMRES count stayed
at `140` iterations and `140` checks before and after the buffer-reuse patch.

Nsight Systems profile for the patched hard-family backward-only run:

```text
benchmarks/large_dataset_capacity/output/nsys_gmres_buffer_reuse_i1_tol7e-6_hard_family_20260606_073926/
```

Compared with the pre-buffer-reuse interval-1 profile, the patched version kept
the same `347` hard-family self-loop applications and `347` checks, while
reducing:

| Metric | Before Buffer Reuse | After Buffer Reuse |
|---|---:|---:|
| `cudaLaunchKernel` calls | `2609` | `2473` |
| device-to-device copies | `226`, `30.879 MB` | `158`, `15.488 MB` |
| `cudaMalloc` calls | `12` | `11` |
| peak profiler memory | `0.0795 GiB` | `0.0770 GiB` |

The remaining overhead is now dominated by many residual checks and their
device-to-host scalar traffic: interval `1` used `397` D2H copies in the
profile. The next structural optimization should therefore batch or keep more
residual-stop decisions on device, or fuse the single-tile Arnoldi step to
reduce launches, rather than expanding Triton coverage beyond cap-512.

#### Max-3 Cap And Current Nsight Refresh

The buffer-reuse largest-10 run showed no wave needing more than three GMRES
steps, so I tested the same interval-1 tolerance setting with
`gmres_max_iter=3` instead of `10`.

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_cap512_max3_tol7e-6_i1_largest10_20260606_074905/
benchmarks/large_dataset_capacity/output/nsys_largest10_gmres10_i1_tol7e-6_buffer_reuse_20260606_075108/
benchmarks/large_dataset_capacity/output/ncu_gmres_buffer_reuse_i1_tol7e-6_hard_family_20260606_075108/
```

Largest-10 first-step gradient check against Neumann512:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Relative L2 Error | Relative Inf Error | Max Family Relative L2 |
|---|---:|---:|---:|---:|---:|
| Neumann32 | `2784` | n/a | `1.046496e-07` | `1.851861e-07` | `2.090796e-07` |
| GMRES3 I1 tol `7e-6` | `140` | `140` | `1.761937e-06` | `3.518536e-06` | `3.520320e-06` |

Largest-10 20-step timing:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Train Seconds | Wall Seconds | Mean Step 2-20 | Final Loss |
|---|---:|---:|---:|---:|---:|---:|
| GMRES10 I1 tol `7e-6`, cap-512 | `2800` | `2800` | `2.757` | `3.916` | `0.1065` | `166606.453125` |
| GMRES3 I1 tol `7e-6`, cap-512 | `2800` | `2800` | `2.776` | `3.966` | `0.1075` | `166606.4375` |

Reducing the maximum Krylov dimension did not reduce actual work: both runs
performed `2800` backward self-loop applications and `2800` residual checks.
It also did not reduce wall time. The max-10 interval-1/tol `7e-6` setting
therefore remained the best corrected largest-10 GMRES result before the
checked-y reuse patch below.

I also refreshed Nsight profiling on the current best largest-10 run. The
profiled run used the same `2800` GMRES self-loop applications and `2800`
checks; profiler wall time was `4.523 s`.

Key `nsys` summaries:

| Bucket | Count | Total Time |
|---|---:|---:|
| `cudaLaunchKernel` | `64570` | `200.651 ms` |
| `cuLaunchKernelEx` | `57285` | `102.338 ms` |
| `cudaStreamSynchronize` | `3679` | `147.345 ms` |
| `cudaMemcpyAsync` | `14619` | `248.989 ms` |
| device-to-device copies | `10940`, `24067.910 MB` | `33.398 ms` |
| device-to-host copies | `3065`, `0.012 MB` | `2.687 ms` |

Largest kernel buckets in that full benchmark profile:

| Kernel Bucket | Launches | Total Time |
|---|---:|---:|
| `_wave_step_kernel` | `26100` | `663.074 ms` |
| `_wave_backward_uniform_2d_precompute_kernel` | `1740` | `137.494 ms` |
| `_wave_backward_uniform_2d_jt_kernel` | `2800` | `71.874 ms` |
| `_gmres_hessenberg_residual_kernel` | `4540` | `14.792 ms` |
| `_gmres_arnoldi_reduce_project_dot_kernel` | `740` | `4.319 ms` |
| `_gmres_arnoldi_reduce_project_norm_kernel` | `740` | `2.710 ms` |
| `_gmres_arnoldi_dot_partials_kernel` | `740` | `2.226 ms` |

The refreshed `ncu --set basic` samples on the hard family confirm that the
main GMRES-specific kernels are tiny and underfilled:

| Kernel | Launch Shape | Duration | Registers/Thread | Achieved Occupancy | DRAM Throughput |
|---|---:|---:|---:|---:|---:|
| `_wave_backward_uniform_2d_jt_kernel` | `1 x 64` | `15.84-16.19 us` | `168` | `4.14-4.27%` | `0.41-0.42%` |
| `_gmres_arnoldi_reduce_project_dot_kernel` | `3 x 256` | `4.06 us` | `48` | `16.50-16.92%` | `0.58-0.85%` |
| `_gmres_hessenberg_residual_kernel` | `1 x 32` | `4.03-4.06 us` | `46` | `2.07-2.09%` | `0.43%` |

The hard family still hits the GMRES cap under this tolerance: the backward-only
`ncu` runs used `348` self-loop applications and `348` checks over `68` waves,
with maximum wave iterations `10`. This is different from the corrected
largest-10 batch, where the same tolerance needed at most three iterations per
wave. The profiling conclusion is unchanged: the remaining gap to Neumann16 is
launch/control/copy overhead around many small wave-local kernels, not a CPU
solve and not a single saturated CUDA kernel. Gluon is not available in the
current environment, so the practical GPU-kernel path remains Triton unless we
add Gluon as a deliberate dependency.

#### Reusing The Last Checked GMRES Solution

The `nsys` result above showed that each adaptive wave paid for residual checks
and then launched `_gmres_hessenberg_residual_kernel` again at wave exit to
materialize the small least-squares solution `y`. This was redundant: the final
adaptive check already has the current Hessenberg matrix. I changed the CUDA
residual check to optionally store `y`, and adaptive GMRES now reuses that
stored vector instead of launching the final solve kernel.

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_reuse_checked_y_tol7e-6_i1_largest10_20260606_075745/
benchmarks/large_dataset_capacity/output/nsys_largest10_gmres10_i1_tol7e-6_reuse_checked_y_20260606_075851/
benchmarks/large_dataset_capacity/output/ncu_gmres_reuse_checked_y_i1_tol7e-6_hard_family_20260606_075851/
```

Largest-10 first-step gradient check against Neumann512:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Relative L2 Error | Relative Inf Error | Max Family Relative L2 |
|---|---:|---:|---:|---:|---:|
| Neumann32 | `2784` | n/a | `8.936047e-08` | `1.234574e-07` | `1.893820e-07` |
| GMRES10 I1 tol `7e-6`, checked-y reuse | `140` | `140` | `1.730390e-06` | `3.395078e-06` | `3.398378e-06` |

Largest-10 20-step timing:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Train Seconds | Wall Seconds | Mean Step 2-20 | Final Loss |
|---|---:|---:|---:|---:|---:|---:|
| Neumann16 | `27840` | n/a | `2.560` | `3.718` | `0.0997` | `166606.4375` |
| Neumann32 | `55680` | n/a | `3.074` | `4.367` | `0.1251` | `166606.4375` |
| GMRES10 I1 tol `7e-6`, buffer reuse | `2800` | `2800` | `2.757` | `3.916` | `0.1065` | `166606.453125` |
| GMRES10 I1 tol `7e-6`, checked-y reuse | `2800` | `2800` | `2.731` | `3.910` | `0.1061` | `166606.4375` |

The VJP count and residual-check count are unchanged. The patch only removes
redundant per-wave post-check work. It improves train time by `0.026 s` versus
the previous buffer-reuse result. GMRES is now `0.343 s` faster than Neumann32
but remains `0.171 s` slower than Neumann16 on this 20-step largest-10 timing.

The follow-up `nsys` comparison confirmed the expected launch/copy reduction:

| Metric | Buffer Reuse | Checked-y Reuse |
|---|---:|---:|
| profiled wall time | `4.523 s` | `4.358 s` |
| `_gmres_hessenberg_residual_kernel` launches | `4540` | `2800` |
| `_gmres_hessenberg_residual_kernel` total time | `14.792 ms` | `10.998 ms` |
| `cudaLaunchKernel` calls | `64570` | `62830` |
| `cuLaunchKernelEx` calls | `57285` | `55545` |
| `cudaMemcpyAsync` calls | `14619` | `12879` |
| device-to-device copies | `10940` | `9200` |
| device-to-host copies | `3065` | `3065` |

The `ncu` sample on the hard family shows why this is only a small win:
storing `y` makes each residual kernel heavier, but still tiny. The checked-y
residual kernel used a `1 x 32` launch, took `5.31-5.34 us`, used `78`
registers/thread, and reached only `2.10-2.12%` achieved occupancy. The earlier
no-store residual sample was `4.03-4.06 us` with `46` registers/thread. Fewer
launches win, but the remaining D2H scalar residual checks are unchanged.

#### Reusing The Previous Check Schedule

The largest-10 checked-y run showed that every backward pass needed the same
GMRES work: `87` wave solves, `140` self-loop applications, and no wave used
more than three Krylov iterations. With `gmres_check_interval=1`, adaptive GMRES
still checked the residual after every GMRES iteration. That means a wave that
always converges in three iterations paid three scalar residual checks every
backward pass.

I added an opt-in schedule cache:

```text
SolverOptions(gmres_reuse_check_schedule=True)
--gmres-reuse-check-schedule
```

The schedule is stored on the batch static state and is keyed by the GMRES
parameters, adjoint pruning parameters, and wave layout. On the first backward
pass there is no schedule, so the behavior is identical to ordinary adaptive
GMRES. After each wave solve, the observed GMRES iteration count is recorded.
On the next backward pass, the same wave delays its first residual check until
that previous iteration count. If the check passes, the wave exits after one
check. If it fails, GMRES continues and checks again every
`gmres_check_interval` iterations until convergence or the maximum iteration
count. Therefore this is not a fixed-iteration shortcut and it does not accept a
wave without a residual check; it only avoids residual checks that were
predictably too early on the previous backward pass.

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_cached_check_schedule_warm_tol7e-6_i1_largest10_20260606_081504/
benchmarks/large_dataset_capacity/output/nsys_largest10_gmres10_i1_tol7e-6_cached_schedule_20260606_081504/
```

Largest-10 first-step gradient check against Neumann512, with one schedule
warmup backward before measuring the GMRES row:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Relative L2 Error | Relative Inf Error | Max Family Relative L2 |
|---|---:|---:|---:|---:|---:|
| Neumann32 | `2784` | n/a | `1.053618e-07` | `1.851861e-07` | `3.136194e-07` |
| GMRES10 I1 tol `7e-6`, checked-y + schedule | `140` | `87` | `1.718619e-06` | `3.395078e-06` | `3.398090e-06` |

Largest-10 20-step timing:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Train Seconds | Wall Seconds | Mean Step 2-20 | Final Loss |
|---|---:|---:|---:|---:|---:|---:|
| Neumann16 | `27840` | n/a | `2.560` | `3.718` | `0.0997` | `166606.4375` |
| Neumann32 | `55680` | n/a | `3.074` | `4.367` | `0.1251` | `166606.4375` |
| GMRES10 I1 tol `7e-6`, checked-y reuse | `2800` | `2800` | `2.731` | `3.910` | `0.1061` | `166606.4375` |
| GMRES10 I1 tol `7e-6`, checked-y + schedule | `2800` | `1793` | `2.689` | `3.968` | `0.1036` | `166606.453125` |

The first scheduled training step has no cached schedule yet, so it performed
`140` self-loop applications and `140` checks. Steps 2-20 used the cached
schedule and each performed `140` self-loop applications but only `87` checks.
The total is therefore:

```text
self-loop applications: 20 * 140 = 2800
GMRES checks:           140 + 19 * 87 = 1793
```

This does not reduce the expensive backward self-loop application count versus
the current best GMRES run. It reduces residual-check overhead. The benefit is
small but measurable in the warm training metric: `2.731 s` to `2.689 s` train
time, and `0.1061 s` to `0.1036 s` mean step time after step 1. It is still
slower than Neumann16 on this benchmark, but it remains much cheaper in the
mathematical work metric: `2800` self-loop backward applications instead of
`27840`.

The `nsys` profile confirms that the change removed the expected scalar
residual-check traffic:

| Metric | Checked-y Reuse | Checked-y + Schedule |
|---|---:|---:|
| profiled wall time | `4.358 s` | `4.430 s` |
| `_gmres_hessenberg_residual_kernel` launches | `2800` | `1793` |
| `_gmres_hessenberg_residual_kernel` total time | `10.998 ms` | `7.080 ms` |
| `cuLaunchKernelEx` calls | `55545` | `54538` |
| `cudaStreamSynchronize` calls | `3679` | `2672` |
| `cudaMemcpyAsync` calls | `12879` | `11872` |
| device-to-host copies | `3065`, `0.012 MB` | `2058`, `0.008 MB` |
| device-to-device copies | `9200`, `24067.903 MB` | `9200`, `24067.903 MB` |

The profiled wall time is noisier than the unprofiled train-time measurement,
but the event counts match the design exactly: the schedule removed `1007`
residual checks and therefore `1007` D2H scalar copies and stream syncs over the
20-step run. The existing `ncu` residual-kernel sample remains applicable
because this patch changes when the same kernel is launched, not the kernel
body. A Triton or Gluon rewrite should therefore be justified by fusing or
removing more of this small-kernel/check orchestration, not by tuning the
standalone residual kernel.

#### Diagonal Right Preconditioning

A subagent review ranked diagonal right preconditioning as the next best
candidate for reducing actual GMRES `A` applications. The retained precompute
kernel already stores the diagonal self-loop weight in `aw0`, so I added an
opt-in right preconditioner:

```text
SolverOptions(gmres_preconditioner="diagonal")
--gmres-preconditioner diagonal
--gmres-diagonal-preconditioner-floor 1e-4
```

The implemented solve is right-preconditioned GMRES:

```text
K z = A M z, then v = M z
M = 1 / clamp(1 - diag_wt, floor)
```

This preserves the checked residual for the original system because each
Arnoldi matvec applies `A(M q)`, and the final returned vector is scaled by
`M`. The option is default-off, and the cached check schedule key includes the
preconditioner settings so schedules do not cross solver variants.

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_diagonal_precond_tol7e-6_i1_largest10_20260606_083216/
benchmarks/large_dataset_capacity/output/gmres_diagonal_precond_tol7e-6_i1_largest10_20260606_083250/
benchmarks/large_dataset_capacity/output/nsys_largest10_gmres10_i1_tol7e-6_diagonal_precond_20260606_083427/
benchmarks/large_dataset_capacity/output/gmres_tol_sweep_i1_largest10_20260606_083337/
benchmarks/large_dataset_capacity/output/gmres_tol_fine_sweep_i1_largest10_20260606_083357/
```

Largest-10 gradient check against Neumann512, with one schedule warmup before
the measured GMRES row:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Relative L2 Error | Relative Inf Error | Max Family Relative L2 |
|---|---:|---:|---:|---:|---:|
| Neumann32 | `2784` | n/a | `9.692890e-08` | `1.234574e-07` | `2.090796e-07` |
| GMRES10 I1 tol `7e-6`, diagonal | `140` | `87` | `1.684517e-06` | `3.333350e-06` | `3.336457e-06` |

The preconditioner passes the gradient gate, but it does not reduce the
first-step self-loop application count. The 20-step timing is correspondingly
negative:

| Solver | Self-Loop Backward Iterations | GMRES Checks | Train Seconds | Wall Seconds | Mean Step 2-20 | Final Loss |
|---|---:|---:|---:|---:|---:|---:|
| GMRES10 I1 tol `7e-6`, checked-y + schedule | `2800` | `1793` | `2.689` | `3.968` | `0.1036` | `166606.453125` |
| GMRES10 I1 tol `7e-6`, diagonal + schedule | `2800` | `1793` | `2.785` | `3.948` | `0.1083` | `166606.4375` |

The `nsys` comparison explains the slowdown:

| Metric | Schedule Only | Diagonal + Schedule |
|---|---:|---:|
| `_wave_backward_uniform_2d_jt_kernel` launches | `2800` | `2800` |
| `_gmres_hessenberg_residual_kernel` launches | `1793` | `1793` |
| `cudaLaunchKernel` calls | `62830` | `72590` |
| `cudaMemcpyAsync` calls | `11872` | `13612` |
| device-to-device copies | `9200`, `24067.903 MB` | `10940`, `28915.511 MB` |
| reciprocal kernels | `0` | `1740` |
| clamp-scalar kernels | `3280` | `5020` |
| binary float elementwise kernels | `3060` | `7600` |

The current diagonal implementation adds scale construction and application
overhead without reducing Krylov iterations. It is useful as a correctness-tested
experimental hook, but it should not be used for the largest-10 production
setting unless a later fused implementation or different preconditioner actually
reduces self-loop applications.

I also swept looser non-preconditioned tolerances to check whether the current
`7e-6` point is too conservative:

| GMRES Tol | Self-Loop Backward Iterations | GMRES Checks | Relative L2 Error | Relative Inf Error | Max Family Relative L2 |
|---|---:|---:|---:|---:|---:|
| `8e-6` | `138` | `87` | `1.045550e-05` | `2.246925e-05` | `2.244268e-05` |
| `9e-6` | `138` | `87` | `1.051128e-05` | `2.259270e-05` | `2.256505e-05` |
| `1e-5` | `138` | `87` | `1.042568e-05` | `2.240752e-05` | `2.238117e-05` |
| `2e-5` | `137` | `87` | `1.597647e-05` | `3.321004e-05` | `3.318367e-05` |
| `5e-5` | `136` | `87` | `3.032280e-05` | `6.432130e-05` | `6.429629e-05` |

Those settings save only two to four self-loop applications on the first
backward pass and fail the `~1e-5` gradient-error gate, especially in relative
infinity norm. Therefore the current safe largest-10 setting remains
GMRES10/check-interval-1/tol `7e-6` with checked-y reuse and the cached check
schedule.

## Recommended First Experiment

Use the known hard HOGENOM family as the first target, then run the small
HOGENOM end-to-end benchmark.

For the family-level benchmark, compare:

```text
Neumann: 16, 32, 48, 64
GMRES zero-start: max_iter 8, 12, 16; tol 1e-6, 1e-8, 1e-10
GMRES + diagonal preconditioner: same grid
Hybrid Neumann warmup + GMRES: warmup 2, 4
```

For the optimizer benchmark, compare:

```text
baseline fixed Neumann
zero-start adaptive GMRES
warm-start adaptive GMRES
hybrid warm-start GMRES
```

Report both:

```text
time to convergence
total J^T applications until convergence
```

The second metric is essential because it isolates the mathematical work we are
trying to reduce. The first metric tells us whether the implementation is
efficient enough for the reduced work to matter.

## Bottom Line

The most efficient path is not to make GMRES more mathematically elaborate. It
is to make each GMRES step cost almost exactly one existing self-loop backward
application, with minimal overhead around it.

The implementation should therefore be:

- matrix-free;
- wave-local;
- adaptive;
- preallocated;
- residual-driven;
- warm-start capable;
- optionally diagonally preconditioned;
- instrumented in terms of total `J^T` applications.

That gives the right optimization target: accurate gradients with as few
expensive backward self-loop applications as possible, and with low enough
solver overhead that the lower iteration count becomes a wall-clock speedup.

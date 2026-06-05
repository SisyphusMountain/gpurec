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

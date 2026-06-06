# HOGENOM Neumann vs GMRES Family Experiment

## Family

This note records a focused gradient-adjoint experiment on HOGENOM family
`CLU_000680_20_4_C`.

- Family index in `alerax_hogenom_core_all_families.txt`: `2461`
- Gene-tree file:
  `benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core/hogenom/families/CLU_000680_20_4_C/gene_trees/ufboot1000.MFP.geneTree.newick`
- Single-family layout size: `2891` clades, `8918` splits, `109` gene leaves,
  `68` wave-backward self-loop solves
- Checkpoint:
  `benchmarks/large_dataset_capacity/output/full_hogenom_genewise_end2end_20260605_scheduled_golden_v6_lr/runs/full_hogenom_genewise_end2end_golden_float64_hsgd_schedule_fixed256/checkpoints/latest.pt`
- Checkpoint theta row:
  `[1.0, 0.83066342490394, -3.672977288738639]`
- Rates from that row:
  `D=2.0`, `L=1.7785030208916341`, `T=0.07840137198659322`

This family was selected because, at the optimized checkpoint row above, low
Neumann budgets produce materially different gradients. It is therefore a useful
small target for testing Krylov acceleration of the Pi/Pibar wave-adjoint
self-loop solve.

## Reproduction

Run:

```bash
python benchmarks/large_dataset_capacity/hogenom_gmres_neumann_family_experiment.py \
  --output-json /tmp/hogenom_gmres_neumann_family_experiment.json
```

The script builds a one-family `global` model, copies in the genewise checkpoint
theta row for family `2461`, disables adjoint pruning, uses `E=256` and
`Pi=256`, and compares all gradients to a `Neumann=512` reference.

The GMRES path uses the same retained Triton wave self-loop operator as the
Neumann path, but solves each wave-local system as:

```text
(I - J) v = rhs
```

where `J` is the wave-local self-loop transpose-Jacobian operator. This is wired
as a private experiment mode, not a public `SolverOptions` setting.

Reference gradient with `Neumann=512`:

```text
[-4.9285149028026485, -2.377774221053495, 0.8579814490324738]
```

## Total Backward Iteration Cost

The expensive inner work is the wave self-loop backward operator application.
This family has `68` wave-local self-loop solves in the backward pass.

For Neumann, the total number of backward iterations is fixed:

```text
total backward iterations = neumann_terms * 68
```

For GMRES, the `iterations` column below is the maximum per-wave Krylov
dimension. The `total backward iterations` column is the actual sum of
wave-local operator applications before residual convergence.

## What GMRES Replaces

For one wave, the current Neumann code applies the wave-local self-loop
transpose-Jacobian operator, called `J` here, a fixed number of times:

```text
v ~= rhs + J rhs + J^2 rhs + ... + J^N rhs
```

The implementation initializes `v` with `rhs`, then performs exactly
`neumann_terms` applications of `J` for every wave. With `68` waves in this
family, `Neumann=32` therefore costs:

```text
32 applications/wave * 68 waves = 2176 total backward iterations
```

The GMRES experiment keeps the same precomputed wave coefficients and the same
Triton kernel for applying `J`, but instead solves the equivalent linear system:

```text
(I - J) v = rhs
```

Each GMRES iteration applies `A = I - J` once to one Krylov basis vector. Since
the identity part is just a subtraction, the expensive part of each GMRES
iteration is one application of the same `J` operator used by Neumann. GMRES
then orthogonalizes the basis vectors and solves a small least-squares problem
to pick the best vector in the Krylov subspace. It stops per wave when the
relative residual is below `gmres_tol`, or when the configured maximum is
reached.

This is why the GMRES total is not generally divisible by the configured maximum
iteration count. For example, with `GMRES max=10`, not all `68` waves used all
`10` iterations. The run used `598` total `J` applications, or `8.794` on
average per wave.

## Neumann Ladder

| Neumann terms | Total backward iterations | Relative L2 error vs N=512 | Relative inf error | Gradient |
|---:|---:|---:|---:|---|
| 8 | `544` | `4.965785e-01` | `5.013882e-01` | `[-2.4574157638377585, -3.4080667270697065, 0.2274242501311065]` |
| 16 | `1088` | `6.644028e-02` | `6.448320e-02` | `[-4.610708477026534, -2.5451130503634722, 0.7778021398543274]` |
| 32 | `2176` | `3.459027e-05` | `3.307442e-05` | `[-4.9283518950504375, -2.3778636587336126, 0.8579352123043589]` |
| 40 | `2720` | `3.746673e-07` | `3.566373e-07` | `[-4.928513145110428, -2.377775187428435, 0.857980916689375]` |
| 48 | `3264` | `3.164838e-09` | `2.999291e-09` | `[-4.928514888020533, -2.3777742291809676, 0.8579814442643962]` |
| 56 | `3808` | `2.251959e-11` | `2.125639e-11` | `[-4.928514902697821, -2.3777742211109607, 0.8579814489966456]` |
| 64 | `4352` | `1.437890e-13` | `1.423676e-13` | `[-4.928514902801882, -2.377774221053808, 0.8579814490322385]` |

The practical threshold for this family is around `32` Neumann terms for
roughly `1e-4` relative gradient accuracy, and `64` terms for near-reference
accuracy.

## GMRES Ladder

| GMRES max iterations | Total backward iterations | Mean iterations/wave | Relative L2 error vs N=512 | Relative inf error | Gradient |
|---:|---:|---:|---:|---:|---|
| 2 | `136` | `2.000` | `1.683727e+00` | `1.670149e+00` | `[3.3028398233310052, -6.037227791363774, -1.5564875021638203]` |
| 4 | `272` | `4.000` | `5.596220e-01` | `5.727670e-01` | `[-2.1056242065927715, -3.464552715857137, 0.18083557831362018]` |
| 8 | `514` | `7.559` | `2.285848e-03` | `1.967333e-03` | `[-4.918818874496734, -2.3858691612808802, 0.857106183948154]` |
| 10 | `598` | `8.794` | `6.589763e-06` | `6.381409e-06` | `[-4.928546353672228, -2.3777557196675763, 0.8579805387352829]` |
| 12 | `649` | `9.544` | `1.079728e-07` | `1.012032e-07` | `[-4.928515401584035, -2.377773891507204, 0.8579814319958814]` |
| 14 | `661` | `9.721` | `2.178261e-10` | `1.739663e-10` | `[-4.928514903070442, -2.377774221859026, 0.8579814481750812]` |
| 16 | `662` | `9.735` | `1.778339e-09` | `1.448129e-09` | `[-4.928514895665459, -2.377774227786276, 0.8579814481622734]` |

GMRES reduces the number of required self-loop operator applications on this
family. Compared with Neumann, it reaches:

- roughly `1e-4` relative accuracy in `598` total backward iterations
  (`GMRES max=10`), while Neumann needs `2176` total iterations
  (`Neumann=32`);
- roughly `1e-8` relative accuracy in `661` total backward iterations
  (`GMRES max=14`), while Neumann needs `3264` total iterations
  (`Neumann=48`);
- near-reference accuracy in `661-662` total backward iterations, while the
  `Neumann=64` comparison costs `4352` total iterations.

## Timing Caveat

The current GMRES implementation is an experiment harness around the existing
Triton operator, not a fused production solver. In the run above, per-gradient
wall times were similar:

- Neumann `32`: `5.404 s`, `2176` total backward iterations
- Neumann `64`: `5.401 s`, `4352` total backward iterations
- GMRES `10`: `5.579 s`, `598` total backward iterations
- GMRES `14`: `5.580 s`, `661` total backward iterations

So the experiment shows fewer mathematical iterations, but not yet a wall-clock
speedup. The extra cost comes from Python-side GMRES orchestration and small
least-squares solves per wave. A production implementation would need to reduce
that overhead before the lower iteration count can translate into runtime.

## Nsight Systems Profile

Profile date: `2026-06-05`

Repository commit at profiling time: `5d63743`, with local GMRES experiment
changes in `gpurec/core/kernels/wave_backward.py`.

Artifacts:

```text
benchmarks/large_dataset_capacity/output/nsys_gmres_profile_20260605/
```

Two Nsight Systems captures were recorded for `GMRES max=10` on the same family:

- `gmres_family2461_max10`: one full loss-gradient evaluation after warmup.
- `gmres_family2461_max10_backward_only`: the forward `E/Pi/Pibar` state was
  computed before the capture, then only `implicit_grad_loglik_vjp_wave` was
  profiled with the GMRES self-loop monkeypatch.

The profiled GMRES backward pass used `68` wave-local solves and `598` total
self-loop operator applications:

```text
GMRES max iterations: 10
mean iterations/wave: 8.794
max iterations/wave: 10
max reported relative residual: 3.848858e-06
```

### Full Gradient Capture

The full gradient capture took `5.628501 s`. It is dominated by the forward
Pi/Pibar fixed-point iteration, not by the GMRES self-loop solve:

| Component | Total |
|---|---:|
| `_wave_step_kernel` | `5.229670 s` across `17,340` launches |
| summed GPU kernels | `5.470437 s` |
| CUDA API time | `5.156147 s` |
| GPU synchronization activity | `0.312686 s` |

The full-gradient profile is therefore mainly a reminder that the end-to-end
gradient call still pays for the forward `Pi=256` schedule.

### Backward-Only Capture

The backward-only capture took `349.275 ms`.

| Category | Total |
|---|---:|
| summed GPU kernels | `206.167 ms` |
| GPU memcpy + memset activity | `6.600 ms` |
| CUDA API time | `141.521 ms` |
| GPU synchronization activity | `10.152 ms` |
| estimated non-kernel wall gap | `136.508 ms` |

Top backward-only kernels:

| Kernel or operation | Total | Count | Note |
|---|---:|---:|---|
| PyTorch reduction kernels for GMRES dot products | `50.716 ms` | `4,100` | Arnoldi orthogonalization |
| `_dts_ge2_stage1_kernel` | `25.924 ms` | `66` | non-GMRES backward DTS work |
| `_wave_backward_uniform_2d_precompute_kernel` | `24.914 ms` | `68` | per-wave self-loop precompute |
| cuSOLVER `geqr2_smem` | `19.697 ms` | `598` | `torch.linalg.lstsq` QR work |
| PyTorch norm-reduction kernels | `10.435 ms` | `1,282` | GMRES norms/residual checks |
| `_wave_backward_uniform_2d_jt_kernel` | `8.165 ms` | `598` | the actual expensive `J^T` matvec |

Top CUDA API costs in the backward-only capture:

| API | Total | Count |
|---|---:|---:|
| `cudaMemcpyAsync` | `73.810 ms` | `6,841` |
| `cudaLaunchKernel` | `50.728 ms` | `28,318` |
| `cudaStreamSynchronize` | `10.460 ms` | `1,874` |

The main conclusion is that this prototype is not primarily sync-bound, and it
is not dominated by the retained Triton self-loop matvec. The actual
`_wave_backward_uniform_2d_jt_kernel` work is only about `8.2 ms` for all `598`
GMRES operator applications. The lost time is mostly orchestration: thousands of
small PyTorch reductions/elementwise kernels, repeated `torch.linalg.lstsq`
calls, many small async copies, and kernel launch overhead.

This confirms the implementation plan in
`docs/efficient_gmres_gradient_self_loop.md`: the next production step should
keep the self-loop matvec in Triton but move GMRES basis storage,
orthogonalization, residual tracking, and the small least-squares update into a
fixed-size GPU-resident implementation with incremental QR/Givens rotations.

### Backward-Only Rerun With Dedicated Driver

I also reran the backward-only profile with a dedicated driver:

```text
benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py
benchmarks/large_dataset_capacity/output/nsys_gmres_profile_20260605_rerun/
```

This driver warms once, computes the forward `E/Pi/Pibar` state outside the
Nsight capture range, then captures only the GMRES
`implicit_grad_loglik_vjp_wave` call. The profiled run again used `68`
wave-local solves and `598` GMRES self-loop applications.

The clean rerun gives the same diagnosis:

| Category | Total |
|---|---:|
| summed GPU kernels | `206.528 ms` |
| GPU memcpy + memset activity | `6.631 ms` |
| CUDA API time | about `141.5 ms` |
| GPU synchronization activity | `10.421 ms` |

Top grouped costs in the rerun:

| Operation | Total | Count |
|---|---:|---:|
| PyTorch GMRES dot-product reductions | `50.760 ms` | `4,100` |
| `_dts_ge2_stage1_kernel` | `25.923 ms` | `66` |
| `_wave_backward_uniform_2d_precompute_kernel` | `24.914 ms` | `68` |
| cuSOLVER `geqr2_smem` from `torch.linalg.lstsq` | `19.707 ms` | `598` |
| PyTorch norm reductions | `10.495 ms` | `1,282` |
| `_wave_backward_uniform_2d_jt_kernel` | `8.134 ms` | `598` |

The sync question is therefore fairly clear: synchronization is present
(`cudaStreamSynchronize` is about `10.7 ms` over `1,875` calls), but it is not
the dominant cost. The retained Triton `J^T` matvec is also not dominant. The
prototype spends most of its time in Python/PyTorch orchestration: reductions,
elementwise updates, small device-to-host convergence checks, kernel launches,
and a small cuSOLVER QR solve at each GMRES iteration.

### Fresh Nsight Rerun With Timing Control

Profile date: `2026-06-05 22:18 Europe/Paris`

Artifacts:

```text
benchmarks/large_dataset_capacity/output/nsys_gmres_profile_20260605_221800/
```

Command:

```bash
PYTHONPATH="$PWD" nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --export=sqlite \
  -o benchmarks/large_dataset_capacity/output/nsys_gmres_profile_20260605_221800/gmres_family2461_max10_backward_only \
  python -u benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py \
    --gmres-iters 10 \
    --warmup 1 \
    --output-json benchmarks/large_dataset_capacity/output/nsys_gmres_profile_20260605_221800/run_result.json
```

The run used `68` wave-local solves and `598` total GMRES self-loop operator
applications. CUDA tracing perturbed this API-heavy prototype heavily: the
same driver took `0.265298 s` without Nsys, but the instrumented run reported
`5.232288 s` inside the capture. The Nsys data below should therefore be used
for attribution, not as the wall-clock runtime.

Fresh Nsys totals:

| Category | Total | Count |
|---|---:|---:|
| summed GPU kernels | `206.592 ms` | `30,093` launches |
| CUDA API time under tracing | `142.419 ms` | `69,317` calls |
| GPU memcpy + memset activity | `6.681 ms` | `7,114` ops |
| `cudaStreamSynchronize` API time | `10.039 ms` | `1,875` calls |

Top kernel groups:

| Operation | Total | Count | Interpretation |
|---|---:|---:|---|
| PyTorch GMRES dot-product reductions | `50.859 ms` | `4,100` | Arnoldi orthogonalization |
| DTS backward kernels | `36.900 ms` | `264` | non-GMRES retained backward work |
| PyTorch elementwise/copy/fill kernels | `28.953 ms` | `20,973` | basis updates and tensor plumbing |
| `_wave_backward_uniform_2d_precompute_kernel` | `24.920 ms` | `68` | per-wave self-loop setup |
| cuSOLVER QR from `torch.linalg.lstsq` | `19.708 ms` | `598` | small least-squares solve |
| PyTorch norm reductions | `10.477 ms` | `1,282` | residual and basis norms |
| `_wave_backward_uniform_2d_jt_kernel` | `8.134 ms` | `598` | actual retained `J^T` matvec |

Top CUDA API groups under tracing:

| API group | Total | Count |
|---|---:|---:|
| `cudaMemcpyAsync` | `73.563 ms` | `6,842` |
| kernel launch APIs | `54.651 ms` | `30,093` |
| `cudaStreamSynchronize` | `10.039 ms` | `1,875` |

The answer to the sync-vs-kernel question is: not one long kernel, and not
primarily explicit synchronization. Synchronizations are numerous, mostly
paired with scalar device-to-host checks in the current Python GMRES loop, but
they are not the largest reported bucket. The largest GPU work is many small
PyTorch reduction and elementwise kernels from Arnoldi orthogonalization and
basis updates. The retained Triton `J^T` application itself is only about
`8.1 ms` across all `598` GMRES iterations.

## Apply-A Kernel and Fixed-M GMRES

Implementation date: `2026-06-05`

The next implementation step added two lower-overhead GMRES modes:

- `_wave_backward_uniform_2d_jt_kernel` can now write `A x = x - J^T x`
  directly for GMRES. This removes one PyTorch subtraction kernel per GMRES
  iteration and avoids the dummy `v_k` accumulation/zeroing used by the first
  prototype.
- `gmres_fixed` runs exactly the configured Krylov dimension per wave and solves
  the small least-squares problem once at the end of the wave. This is not the
  minimal-VJP adaptive policy, but it removes the per-iteration `lstsq` and
  residual synchronization overhead.
- GMRES now masks inactive rows before the Arnoldi solve and forces the
  `A x` kernel output to zero inactive rows. This is required because GMRES
  reductions operate over full wave tensors.

A Python-level incremental Givens prototype was also tested, but rejected as the
default because it was slower on this family: the scalar GPU operations added
more overhead than the removed per-iteration cuSOLVER work. The retained path is
therefore the simpler adaptive least-squares GMRES plus the separate fixed-m
benchmark mode.

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_applya_adaptive_20260605_222852/
benchmarks/large_dataset_capacity/output/gmres_applya_fixed_masksafe_20260605_224056/
benchmarks/large_dataset_capacity/output/gmres_applya_fixed_masksafe_ladder_20260605_224134/
benchmarks/large_dataset_capacity/output/nsys_gmres_fixed_masksafe_profile_20260605_224244/
```

Implementation commits:

```text
94b2be8 Add mask-safe fixed GMRES self-loop mode
e424419 Expose GMRES self-loop solver options
```

Backward-only timing on `CLU_000680_20_4_C`:

| Solver | Max/fixed m | Total backward iterations | Elapsed | Relative L2 error vs N=512 | Relative inf error |
|---|---:|---:|---:|---:|---:|
| GMRES adaptive, Apply-A kernel | 10 | `598` | `0.263357 s` | `6.589763e-06` | `6.381409e-06` |
| GMRES fixed, Apply-A kernel | 8 | `544` | `0.175573 s` | `2.285849e-03` | `1.967334e-03` |
| GMRES fixed, Apply-A kernel | 10 | `680` | `0.208197 s` | `6.588901e-06` | `6.380555e-06` |
| GMRES fixed, Apply-A kernel | 12 | `816` | `0.247367 s` | `1.074764e-07` | `1.004131e-07` |
| GMRES fixed, Apply-A kernel | 16 | `1088` | `0.343459 s` | `3.260914e-12` | `2.439352e-12` |

Compared with the previous adaptive GMRES max-10 backward-only baseline
(`0.265298 s`, `598` iterations), fixed `m=10` is about `22%` faster
(`0.208197 s`) at the same gradient accuracy, despite doing more self-loop
matvecs (`680` instead of `598`). The speedup comes from lower orchestration
overhead, not from fewer VJPs.

The fixed `m=12` point reaches roughly `1e-7` relative L2 gradient error in
`0.247367 s`, still far below the `Neumann=32` mathematical work
(`2176` backward iterations) and near the previous adaptive max-10 runtime.

Fresh Nsys fixed-m profile for `m=10`:

| Category | Adaptive max-10 before fixed mode | Fixed m=10 |
|---|---:|---:|
| summed GPU kernels | `206.592 ms` across `30,093` launches | `187.624 ms` across `22,024` launches |
| CUDA API time under tracing | `142.419 ms` across `69,317` calls | `111.976 ms` across `48,501` calls |
| cuSOLVER QR kernels | `19.708 ms` across `598` calls | `4.419 ms` across `68` calls |
| `cudaStreamSynchronize` API time | `10.039 ms` across `1,875` calls | `5.076 ms` across `254` calls |
| `_wave_backward_uniform_2d_jt_kernel` | `8.134 ms` across `598` calls | `9.112 ms` across `680` calls |

This confirms the fixed-m tradeoff: the actual retained `J^T` work increases,
but small QR solves, syncs, kernel launches, and API calls decrease enough to
win wall time on this family.

### Fixed-M CGS2 Arnoldi Update

Implementation commit:

```text
01a0faa Reduce fixed GMRES Arnoldi launch overhead
```

The first low-overhead follow-up keeps the same fixed-m GMRES mathematics and
the same `A x = x - J^T x` Apply-A kernel, but changes the Arnoldi
orthogonalization path for `gmres_fixed`. Instead of launching one PyTorch dot
reduction and one elementwise update per existing basis vector, each Arnoldi
step now uses two batched classical Gram-Schmidt passes:

```text
coeff  = Q_j w
w      = w - Q_j^T coeff
coeff2 = Q_j w
w      = w - Q_j^T coeff2
```

This is not the final Triton/Givens implementation, but it removes a large
fraction of the launch overhead for the real HOGENOM wave sizes, where a
single-block Triton MGS kernel would not cover the largest waves
(`162 x 1331 = 215,622` entries).

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_fixed_cgs2_20260606_225944/
benchmarks/large_dataset_capacity/output/gmres_fixed_cgs2_ladder_20260606_230014/
benchmarks/large_dataset_capacity/output/nsys_gmres_fixed_cgs2_profile_20260606_230112/
```

Backward-only timing for `gmres_fixed, m=10`:

| Implementation | Total Backward Iterations | Elapsed | Gradient |
|---|---:|---:|---|
| previous fixed MGS | `680` | `0.208197 s` | `[-4.928546349462736, -2.377755721941266, 0.8579805384960248]` |
| batched CGS2 | `680` | `0.160145 s` | `[-4.928546349462685, -2.377755721941347, 0.85798053849603]` |

Reference-error ladder against Neumann=512:

| Solver | Iterations | Total Backward Iterations | Relative L2 Error |
|---|---:|---:|---:|
| Neumann | 32 | `2176` | `3.459027e-05` |
| GMRES fixed CGS2 | 8 | `544` | `2.285849e-03` |
| GMRES fixed CGS2 | 10 | `680` | `6.588901e-06` |
| GMRES fixed CGS2 | 12 | `816` | `1.074765e-07` |

Fresh Nsys comparison for fixed `m=10`:

| Category | Previous Fixed MGS | Fixed CGS2 |
|---|---:|---:|
| summed GPU kernels | `187.624 ms`, `22,024` launches | `129.190 ms`, `12,368` launches |
| CUDA API time under tracing | `111.976 ms`, `48,501` calls | `87.620 ms`, `26,723` calls |
| PyTorch sum reductions | `58.178 ms`, `4,365` launches | `4.618 ms`, `557` launches |
| cuBLAS GEMV/dot/update kernels | `0.105 ms`, `64` launches | `11.268 ms`, `4,212` launches |
| elementwise kernels | `23.063 ms`, `14,923` launches | `7.277 ms`, `4,927` launches |
| `_wave_backward_uniform_2d_jt_kernel` | `9.112 ms`, `680` calls | `9.104 ms`, `680` calls |

The self-loop matvec cost is unchanged, as expected. The improvement comes from
collapsing many tiny PyTorch reductions and updates into fewer dense vector
operations. This validates the next production direction in
`docs/efficient_gmres_gradient_self_loop.md`: continue moving Arnoldi vector
algebra and residual tracking into lower-overhead GPU-resident kernels.

### Adaptive CGS2 With Coarse Residual Checks

Implementation commit:

```text
1e71da7 Use CGS2 Arnoldi for adaptive GMRES
```

Adaptive `gmres` now uses the same batched CGS2 Arnoldi vector algebra as
`gmres_fixed`. It also exposes `gmres_check_interval`, which always checks
after the first Krylov step and then checks every configured interval. This
keeps early convergence safe on near-breakdown waves while reducing repeated
least-squares residual checks.

Backward-only timing on `CLU_000680_20_4_C`, max `m=10`:

| Solver | Check Interval | Total Backward Iterations | Total GMRES Checks | Elapsed | Relative L2 Error vs N=512 |
|---|---:|---:|---:|---:|---:|
| old adaptive MGS | 1 | `598` | `598` | `0.263807 s` | `6.589763e-06` |
| adaptive CGS2 | 1 | `598` | `598` | `0.232383 s` | `6.589763e-06` |
| adaptive CGS2 | 4 | `638` | `251` | `0.187266 s` | `6.588813e-06` |
| adaptive CGS2 | 5 | `670` | `202` | `0.179272 s` | `6.588437e-06` |
| fixed CGS2 | fixed | `680` | `68` | `0.160145 s` | `6.588901e-06` |

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_adaptive_cgs2_i1_20260606_231216/
benchmarks/large_dataset_capacity/output/gmres_adaptive_cgs2_i4_20260606_231244/
benchmarks/large_dataset_capacity/output/gmres_adaptive_cgs2_i5_20260606_231304/
benchmarks/large_dataset_capacity/output/nsys_gmres_adaptive_cgs2_i4_profile_20260606_231402/
```

Nsys for interval `4` confirms that the overhead moved in the intended
direction:

| Category | Old Adaptive MGS | Adaptive CGS2 I4 | Fixed CGS2 M10 |
|---|---:|---:|---:|
| summed GPU kernels | `206.592 ms`, `30,093` launches | `137.375 ms`, `15,362` launches | `129.190 ms`, `12,368` launches |
| CUDA API time | `142.419 ms`, `69,317` calls | `101.383 ms`, `34,754` calls | `87.620 ms`, `26,723` calls |
| PyTorch sum reductions | `50.868 ms`, `4,102` launches | `5.064 ms`, `740` launches | `4.618 ms`, `557` launches |
| QR/lstsq kernels | `26.721 ms`, `2,324` launches | `11.928 ms`, `1,004` launches | `5.457 ms`, `272` launches |
| `_wave_backward_uniform_2d_jt_kernel` | `8.134 ms`, `598` calls | `8.596 ms`, `638` calls | `9.104 ms`, `680` calls |

This is now a real VJP/overhead tradeoff knob. Interval `4` uses fewer VJPs
than fixed `m=10` (`638` vs `680`) and is much faster than old adaptive GMRES,
but it still does not beat fixed CGS2 wall time because the remaining residual
checks are expensive. The next implementation step should replace checkpoint
`torch.linalg.lstsq` residual checks with a low-overhead GPU-resident Givens or
small-Hessenberg update.

### Triton Hessenberg Residual Checker

Implementation commit:

```text
198fa4b Use Triton residual checks for adaptive GMRES
```

Adaptive checkpoint residuals now use a tiny Triton Givens QR kernel over the
small Hessenberg matrix. This keeps CGS2 Arnoldi and the final per-wave
`torch.linalg.lstsq` solve for the GMRES coefficients, but removes repeated
checkpoint cuSOLVER least-squares solves.

Selected backward-only result on `CLU_000680_20_4_C`, max `m=10`, interval `3`:

| Solver | Total Backward Iterations | Total GMRES Checks | Elapsed | Relative L2 Error vs N=512 |
|---|---:|---:|---:|---:|
| adaptive CGS2 I3, Triton residual | `619` | `299` | `0.168223 s` | `6.589550e-06` |
| fixed CGS2 M10 | `680` | `68` | `0.160145 s` | `6.588901e-06` |

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_adaptive_cgs2_tritonres_guard_i3_20260606_232528/
benchmarks/large_dataset_capacity/output/nsys_gmres_adaptive_cgs2_tritonres_guard_i3_profile_20260606_232600/
```

Nsys comparison:

| Category | Old Adaptive MGS | Adaptive CGS2 I3 Triton Residual | Fixed CGS2 M10 |
|---|---:|---:|---:|
| summed GPU kernels | `206.592 ms`, `30,093` launches | `127.934 ms`, `11,935` launches | `129.190 ms`, `12,368` launches |
| CUDA API time | `142.419 ms`, `69,317` calls | `91.472 ms`, `27,119` calls | `87.620 ms`, `26,723` calls |
| `_wave_backward_uniform_2d_jt_kernel` | `8.134 ms`, `598` calls | `8.318 ms`, `619` calls | `9.104 ms`, `680` calls |
| `_gmres_hessenberg_residual_kernel` | none | `1.863 ms`, `299` calls | none |
| QR/lstsq kernels | `26.721 ms`, `2,324` launches | `5.030 ms`, `272` launches | `5.457 ms`, `272` launches |

This is the first point where adaptive GMRES both uses fewer VJPs than fixed
`m=10` and has lower summed GPU kernel time in the profiler. The measured
backward-only wall time is still close to fixed CGS2 rather than decisively
lower, so the next step is to remove the remaining per-wave final `lstsq` or
start testing whether this VJP reduction changes end-to-end optimizer time.

### Triton QR/Backsolve For Final Coefficients

Implementation commit:

```text
8f5f50c Solve small GMRES Hessenberg systems in Triton
```

The small-Hessenberg Triton kernel now also performs the final QR/backsolve for
`m <= 16`, removing the last per-wave `torch.linalg.lstsq` from the hot CUDA
path. Larger `m` and CPU execution still fall back to `torch.linalg.lstsq`.

Backward-only timing on `CLU_000680_20_4_C`, max/fixed `m=10`:

| Solver | Total Backward Iterations | Total GMRES Checks | Elapsed | Relative L2 Error vs N=512 |
|---|---:|---:|---:|---:|
| adaptive CGS2 I3 + Triton QR | `619` | `299` | `0.162774 s` | `6.589550e-06` |
| fixed CGS2 M10 + Triton QR | `680` | `68` | `0.151533 s` | `6.588901e-06` |

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_adaptive_cgs2_tritonqr_i3_20260606_233022/
benchmarks/large_dataset_capacity/output/gmres_fixed_cgs2_tritonqr_m10_20260606_233057/
benchmarks/large_dataset_capacity/output/nsys_gmres_adaptive_cgs2_tritonqr_i3_profile_20260606_233120/
```

Nsys comparison:

| Category | Old Adaptive MGS | Adaptive CGS2 I3 Triton QR | Previous Fixed CGS2 M10 |
|---|---:|---:|---:|
| summed GPU kernels | `206.592 ms`, `30,093` launches | `122.871 ms`, `10,847` launches | `129.190 ms`, `12,368` launches |
| CUDA API time | `142.419 ms`, `69,317` calls | `87.574 ms`, `24,671` calls | `87.620 ms`, `26,723` calls |
| `_wave_backward_uniform_2d_jt_kernel` | `8.134 ms`, `598` calls | `8.326 ms`, `619` calls | `9.104 ms`, `680` calls |
| `_gmres_hessenberg_residual_kernel` | none | `2.865 ms`, `367` calls | none |
| QR/lstsq kernels | `26.721 ms`, `2,324` launches | `0.000 ms`, `0` launches | `5.457 ms`, `272` launches |

This establishes the intended low-overhead adaptive behavior on the hard
family: fewer VJPs than fixed `m=10`, no checkpoint or final cuSOLVER solves,
and lower summed GPU kernel time than the previous fixed CGS2 profile. The
remaining validation is end-to-end: whether lower VJP count wins over the
remaining adaptive-check overhead during optimization.

### Current HEAD Nsys/NCU Rerun

Profiled code:

```text
d87135bc Document current GMRES profiler results
```

The same family and solver setting were rerun after the Triton QR work:

```text
script: benchmarks/large_dataset_capacity/profile_hogenom_gmres_backward.py
family: CLU_000680_20_4_C
GMRES max iterations: 10
GMRES tolerance: 1e-10
GMRES check interval: 3
self-loop solver: gmres
```

Artifacts:

```text
benchmarks/large_dataset_capacity/output/current_gmres_tritonqr_i3_profile_20260606_055419/
benchmarks/large_dataset_capacity/output/nsys_current_gmres_tritonqr_i3_profile_20260606_055419/
benchmarks/large_dataset_capacity/output/ncu_current_gmres_tritonqr_i3_hessenberg_basic_20260606_055419/
benchmarks/large_dataset_capacity/output/ncu_current_gmres_tritonqr_i3_jt_grid28_basic_20260606_055419/
```

Unprofiled result:

| Metric | Value |
|---|---:|
| elapsed backward-only time | `0.161840 s` |
| wave count | `68` |
| total backward iterations / `J^T` applications | `619` |
| total GMRES residual checks | `299` |
| mean iterations per wave | `9.103` |
| max iterations per wave | `10` |
| max relative residual | `3.849e-06` |

`nsys` captured only the CUDA-profiler range around the backward solve. The
wall-clock timer inside the profiled run is inflated by profiler overhead, so
the useful numbers are the CUDA summaries:

| Category | Total |
|---|---:|
| summed GPU kernels | `122.921 ms`, `10,847` launches |
| CUDA API time | `87.269 ms`, `24,671` calls |
| `cudaMemcpyAsync` API time | `63.703 ms`, `2,966` calls |
| GPU memcpy time | `3.309 ms`, `2,966` copies |

Top kernel buckets:

| Kernel bucket | Time | Launches |
|---|---:|---:|
| `_dts_ge2_stage1_kernel` | `25.924 ms` | `66` |
| `_wave_backward_uniform_2d_precompute_kernel` | `24.916 ms` | `68` |
| PyTorch reductions | `13.630 ms` | `1,192` |
| `_dts_cross_backward_accum_kernel` | `8.692 ms` | `67` |
| `_wave_backward_uniform_2d_jt_kernel` | `8.333 ms` | `619` |
| cuBLAS dot kernels | `4.021 ms` | `1,238` |
| cuBLAS GEMV kernels | `3.341 ms` | `1,084` |
| `_gmres_hessenberg_residual_kernel` | `2.866 ms` | `367` |

The capture contains no cuSOLVER least-squares work. The slow part that remains
is many small launches and scalar/reduction traffic around Arnoldi and residual
checking.

`ncu` on `_gmres_hessenberg_residual_kernel` shows that the Triton QR/residual
kernel is not a hardware-throughput problem:

| Metric | Sampled Values |
|---|---:|
| grid/block | `1 x 32` |
| duration | `3.30 us`, `5.34 us`, `8.58 us` |
| DRAM throughput | `0.26%` - `0.27%` |
| compute throughput | `0.10%` - `0.22%` |
| achieved occupancy | `2.08%` - `2.10%` |

`ncu` on `_wave_backward_uniform_2d_jt_kernel` shows that the repeated
self-loop matvec is register-limited and underfilled on this single family.
The first matching launches are one-block edge cases, so this run sampled the
most frequent high-cost shape from the Nsys launch distribution:

| Metric | Sampled Values |
|---|---:|
| grid/block | `28 x 64` |
| duration | `18.91 us` - `19.10 us` |
| registers/thread | `254` |
| DRAM throughput | `8.15%` - `8.24%` |
| compute throughput | `4.40%` - `4.49%` |
| achieved occupancy | `4.15%` - `4.23%` |

Implication: replacing only the one-block Triton Hessenberg kernel with Gluon
is unlikely to change end-to-end performance. Gluon is worth considering only
if it helps express a larger fused GMRES step, lowers register pressure in the
self-loop matvec, or removes the Python/PyTorch reduction and scalar-check
traffic that still dominates the adaptive path.

### CUDA Host-Sync Reduction

The next implementation step kept the GMRES mathematics and iteration counts
unchanged, but removed avoidable scalar host reads from the CUDA path:

- no RHS-norm Python float is needed when the Triton Hessenberg path is used;
- CUDA no longer copies the Arnoldi norm to the host only to check breakdown
  after a residual check;
- final residuals are not copied unless fixed-iteration stats require them.

Artifacts:

```text
benchmarks/large_dataset_capacity/output/gmres_cuda_sync_reduction_20260606_060359/
benchmarks/large_dataset_capacity/output/nsys_gmres_cuda_sync_reduction_20260606_060359/
benchmarks/large_dataset_capacity/output/gmres_cuda_sync_reduction_correctness_20260606_060359/
```

Backward-only result on `CLU_000680_20_4_C`, adaptive GMRES max `10`,
tolerance `1e-10`, check interval `3`:

| Metric | Before | After |
|---|---:|---:|
| elapsed backward-only time | `0.161840 s` | `0.149429 s` |
| total backward iterations / `J^T` applications | `619` | `619` |
| total GMRES residual checks | `299` | `299` |
| max relative residual | `3.849e-06` | `3.849e-06` |

Nsys confirms the patch reduces host synchronization and copy traffic:

| Category | Before | After |
|---|---:|---:|
| summed GPU kernels | `122.921 ms`, `10,847` launches | `122.575 ms`, `10,847` launches |
| `cudaMemcpyAsync` API time | `63.703 ms`, `2,966` calls | `50.495 ms`, `2,500` calls |
| Device-to-Host copies | `0.621 ms`, `747` copies | `0.290 ms`, `349` copies |
| `cudaStreamSynchronize` | `0.536 ms`, `747` calls | `0.331 ms`, `349` calls |

Correctness against the Neumann-512 reference remains in the same envelope:

| Solver | Total Backward Iterations | Relative L2 Error vs N=512 | Gradient |
|---|---:|---:|---|
| Neumann32 | `2176` | `3.459027e-05` | `[-4.9283518950504375, -2.3778636587336126, 0.8579352123043589]` |
| GMRES10 I3 | `619` | `6.589550e-06` | `[-4.928546352910933, -2.3777557207255815, 0.8579805383195991]` |

The full family harness still times the forward solve around the backward pass,
so it is not a pure measure of this patch. Its value here is correctness
against high Neumann/Pi; the backward-only profiler is the relevant speed
measurement.

### General Benchmark Driver GMRES Knobs

`benchmarks/large_dataset_capacity/run_gpurec_benchmark.py` now exposes the
same GMRES controls needed for broader HOGENOM optimizer comparisons:

```text
--self-loop-solver neumann|gmres|gmres_fixed
--gmres-max-iter N
--gmres-tol TOL
--gmres-check-interval N
```

The driver records the effective `solver_options` block in its JSON output.
`--gmres-max-iter` is optional; when supplied for `gmres` or `gmres_fixed`, it
sets the effective `SolverOptions.neumann_terms` value used as the Krylov
maximum.

Smoke artifact:

```text
benchmarks/large_dataset_capacity/output/gmres_driver_smoke_20260606_061011/run.json
```

That run used the HOGENOM single-family probe, `--self-loop-solver gmres`, and
`--gmres-max-iter 2`. It completed one optimizer step and recorded:

```text
solver_options.self_loop_solver = "gmres"
solver_options.neumann_terms = 2
```

The driver now records the total self-loop backward applications per step and
for the full run. This is the end-to-end counterpart to the hard-family
`total_backward_iterations` metric:

```text
self_loop_backward_iterations
self_loop_backward_pass_count
self_loop_wave_solves
self_loop_mean_iterations_per_wave
gmres_total_checks
gmres_max_rel_res
```

For Neumann, the count is exact from `backward_passes * waves * terms`. For
GMRES, it is the sum of actual per-wave Krylov iterations collected from the
solver stats.

Paired smoke artifact:

```text
benchmarks/large_dataset_capacity/output/self_loop_accounting_smoke_20260606_061411/
```

On the same one-step single-family HOGENOM probe:

| Solver | Wave Solves | Self-Loop Backward Iterations | GMRES Checks | Step Seconds |
|---|---:|---:|---:|---:|
| Neumann terms=2 | `12` | `24` | n/a | `1.123` |
| GMRES max=2 | `12` | `19` | `19` | `0.504` |

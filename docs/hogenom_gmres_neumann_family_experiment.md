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

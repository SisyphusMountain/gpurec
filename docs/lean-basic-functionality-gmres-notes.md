# `lean-basic-functionality` GMRES Notes

This branch is not a small follow-up. It is the branch that turns the retained wave-local backward solve into a GMRES-based path and then backs that change with tests, benchmarks, and profiler evidence.

## What Changed

### Solver surface

- `SolverOptions` now carries GMRES-specific controls such as `self_loop_solver="gmres"`, tolerance, check interval, trust/reuse toggles, warm-start cache knobs, and diagonal preconditioning settings.
- The autograd and execution paths now pass those options through to the implicit-gradient code.
- The batch-state layer now stores GMRES-specific schedule and solution caches, keyed by solver options and wave layout.
- `GeneReconModel.forward()` was made consistent for single-batch and batched paths so the same family-indexed parameter view is used everywhere.
- The retained kernel path in `wave_backward.py` gained an actual GMRES engine alongside the existing Neumann and fixed GMRES variants.

### Adaptive reuse

- The backward path now records GMRES metadata such as iterations, residual, backend choice, warm-start usage, trusted-check usage, and preconditioner selection.
- Those observations feed the next step’s schedule and cache reuse, so the backward pass is doing both solving and policy refinement.
- Warm starts are treated as wave-local solve state, not as final parameter gradients.

## What The Branch Was Trying To Prove

The branch was trying to answer three questions:

1. Can GMRES replace the retained Neumann-style self-loop solve without changing the final gradient?
2. Can it reduce the number of expensive `J^T` self-loop applications on difficult HOGENOM families?
3. Does that reduction actually translate into wall-clock savings, or is the prototype still overhead-bound?

The answer from the evidence is:

- Yes, the gradients are numerically valid.
- Yes, the expensive backward-operator count drops substantially on hard families.
- No, the current prototype is not yet faster in wall time, because Python-side orchestration and small dense least-squares work still dominate.

## Evidence Artifacts

### Design note

`docs/efficient_gmres_gradient_self_loop.md` is the solver design note. It explains why GMRES fits this case:

- the solve is wave-local and matrix-free;
- the expensive primitive is one `J^T` application;
- GMRES can reuse the same operator without forming matrices;
- warm starts and residual-based stopping are important for practical use.

### Family experiment

`docs/hogenom_gmres_neumann_family_experiment.md` is the strongest proof artifact. It compares GMRES against a high-Neumann reference on HOGENOM family `CLU_000680_20_4_C` and shows:

- GMRES reaches similar gradients with far fewer backward applications;
- the wall time stays similar because the prototype is still orchestration-heavy;
- the performance win is mathematical first, runtime second.

### Benchmarks and profiler output

The branch also includes benchmark scripts and captured outputs under `benchmarks/large_dataset_capacity/` that isolate:

- full HOGENOM gradient checks,
- self-loop overhead,
- GMRES warm-start and trusted-schedule behavior,
- Nsight Systems profiles of the GMRES path.

## Tests Added

- `tests/test_gmres_self_loop_solver.py` covers solver-option validation, dense CPU equivalence for the fixed GMRES solve, zero-RHS behavior, and Triton inactive-row masking.
- `tests/test_large_dataset_capacity_benchmark.py` covers benchmark-harness plumbing and GMRES option handling.
- `tests/test_public_api_integration.py` exercises the public API end to end, including CUDA execution and GMRES compatibility.

## Takeaway

The branch is best understood as:

- a real GMRES self-loop solver integration,
- plus the instrumentation needed to prove it is correct,
- plus the evidence needed to show where the current prototype still pays overhead.

In other words, `lean-basic-functionality` is the branch that makes GMRES real, then measures the gap between “correct” and “fast.”

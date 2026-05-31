# Workflow Run Contexts Refactor Plan - 2026-05-31

## Target

Reduce `gpurec/workflow/optimize.py` orchestration boilerplate by moving immutable run context construction into a private workflow helper.

## Scope

- Add a private `_run_contexts.py` module for building:
  - `_StepPlanningContext`
  - `_StepExecutionContext`
  - `_IterationArtifactsContext`
  - `_LoopPolicyContext`
- Return the derived solver warmup flags needed by `OptimizationRunner.run()`.
- Pass bound runner callbacks, especially `_make_optimizer` and `_active_fd_newton_step`, into the helper.

## Non-Goals

- Do not move mutable run state construction.
- Do not move adaptive rebatch construction or monkeypatchable constants.
- Do not move resume/checkpoint/finalization hooks.
- Do not extract the main iteration loop.
- Do not change optimizer restore or FD Newton algorithm behavior in this slice.

## Verification

- Compile and lint touched workflow modules and the saved plan.
- Run workflow tests covering planning, transitions, Hessian SGD/FD Newton, resume, finalization, and artifacts.
- Run the broad non-slow CPU unit gate and the existing genewise integration smoke.

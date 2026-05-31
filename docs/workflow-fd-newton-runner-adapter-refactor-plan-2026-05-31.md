# Workflow FD Newton Runner Adapter Refactor Plan - 2026-05-31

## Target

Reduce `gpurec/workflow/optimize.py` bloat by moving FD Newton runtime adapter construction into `gpurec/workflow/_fd_newton.py`.

## Scope

- Keep `OptimizationRunner._active_fd_newton_step` as the runner-level hook used by workflow step execution.
- Move the `_FDNewtonRuntime` wiring and `set_model_theta` adapter out of `optimize.py`.
- Store the runner in the adapter and resolve runner methods at call time so monkeypatches and subclass overrides are preserved.
- Keep `_FDNewtonHessianState` reachable from `gpurec.workflow.optimize` for existing private tests.

## Non-Goals

- Do not change the FD Newton algorithm.
- Do not move Hessian SGD step policy from `_step_execution.py`.
- Do not move Hessian-state reset/carry logic from `_transitions.py`.
- Do not alter workflow checkpoint, resume, or finalization behavior.

## Verification

- Compile and lint touched workflow modules and this saved plan.
- Run focused FD Newton/Hessian SGD tests, including the staged Pi-adjoint cache monkeypatch test.
- Run transition/policy helper tests.
- Run the broad non-slow CPU unit gate and the genewise integration smoke.

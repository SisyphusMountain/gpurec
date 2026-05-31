# Workflow Step Runtime Refactor Plan, 2026-05-31

## Target

`gpurec/workflow/_step_execution.py` still owned runtime helpers for theta
restore, active Adam warmup, nonfinite update handling, and solver cache
cleanup. Those helpers are runtime support for step execution, not phase branch
control flow.

## Scope

- Add `gpurec/workflow/_step_runtime.py` as a workflow-private helper module.
- Move `_NonfiniteParameterUpdate`, `_set_model_theta`,
  `_restore_theta_if_nonfinite_update`, `_active_adam_step`, and
  `_clear_solver_runtime_state_preserving_pi_cache` without changing behavior.
- Keep `_step_execution.py` as the compatibility import surface for the moved
  names.
- Do not unify this helper with `_evaluation._clear_solver_runtime_state_preserving_pi_cache`;
  that same-named helper has staged-cache-specific semantics and remains
  separate.
- Leave `execute_optimization_step` branches, closures, status handling, and
  metric assembly unchanged except for imports.

## Verification

- Compile `_step_execution.py`, `_step_runtime.py`, `optimize.py`, and focused
  workflow unit modules.
- Run ruff on the touched workflow modules and the focused workflow unit
  modules.
- Check the diff for whitespace errors.
- Run focused workflow tests around pi-adjoint cache handling, active Adam
  nonfinite warmup, first-order nonfinite restore, and FD-Newton/Hessian-SGD
  routes.

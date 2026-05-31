# Workflow FD Newton Hessian Refactor Plan - 2026-05-31

## Target

Reduce `gpurec/workflow/_fd_newton.py` by moving FD Newton Hessian state,
cache matching, finite-difference refresh, and BFGS update helpers into a
workflow-private sibling module.

## Moved Symbols

- Move `_FDNewtonHessianState` to `gpurec/workflow/_fd_newton_hessian.py`.
- Move `_fd_newton_state_matches` to `gpurec/workflow/_fd_newton_hessian.py`.
- Move `_bfgs_update_fd_newton_hessian` to
  `gpurec/workflow/_fd_newton_hessian.py`.
- Move `_refresh_fd_newton_hessian_state` to
  `gpurec/workflow/_fd_newton_hessian.py`.

## Compatibility

- Keep `_FDNewtonRuntime`, `_fd_newton_runtime_for_runner`,
  `active_fd_newton_step_for_runner`, and `active_fd_newton_step` in
  `_fd_newton.py`.
- Import and re-export the moved private Hessian names from `_fd_newton.py` so
  existing imports through `gpurec.workflow.optimize` and private tests keep
  resolving `_FDNewtonHessianState`.
- Keep `_fd_newton_hessian.py` independent from `_fd_newton.py` to avoid an
  import cycle; the moved helpers consume the runtime adapter structurally.

## Risks

- The main risk is a private import path break for callers that still import
  Hessian state through `_fd_newton.py`; the compatibility imports cover that.
- Another risk is changing FD Newton step semantics during extraction; this
  refactor keeps the active step algorithm in place and moves only helper
  definitions.

## Focused Gates

- `python -m compileall gpurec/workflow/_fd_newton.py gpurec/workflow/_fd_newton_hessian.py`
- `ruff check gpurec/workflow/_fd_newton.py gpurec/workflow/_fd_newton_hessian.py`
- `git diff --check`
- Focused FD Newton workflow tests when practical.

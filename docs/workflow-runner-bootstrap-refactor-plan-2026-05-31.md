# Workflow Runner Bootstrap Refactor Plan

## Goal

Reduce `gpurec/workflow/optimize.py` bloat by moving the pre-loop runner
bootstrap and state assembly into a private workflow helper while preserving
`OptimizationRunner` as the public and white-box compatibility surface.

## Scope

- Add `gpurec/workflow/_runner_bootstrap.py` for run setup, immutable contexts,
  adaptive rebatch state, mutable run state, checkpoint paths, loop policy
  state, transition ops/context, and the progress-row printer.
- Pass bound `OptimizationRunner` callbacks into the helper so subclasses and
  monkeypatches still control optimizer construction, status saving, checkpoint
  loading, and FD-Newton execution.
- Keep the optimization loop ordering in `optimize.py`, especially
  `self._record(row)` before status/checkpoint/progress handling.

## Verification

- Compile and Ruff-check changed workflow files.
- Run `git diff --check`.
- Run focused workflow tests for runner run/resume/checkpoint/status saving,
  L-BFGS-B, Hessian/FD-Newton, adaptive rebatch, and the hygiene source-order
  guard.

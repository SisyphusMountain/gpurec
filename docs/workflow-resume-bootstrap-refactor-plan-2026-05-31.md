# Workflow Resume Bootstrap Refactor Plan, 2026-05-31

## Target

`gpurec/workflow/optimize.py` still owned the full resume checkpoint
hydration block inside `OptimizationRunner.run()`, even though
`gpurec/workflow/_runtime_state.py` already owns checkpoint resume parsing and
progress validation.

## Scope

- Move initial resume checkpoint application into `_runtime_state.py`.
- Keep the public runner as the owner of model construction, optimizer planning,
  row construction, transitions, checkpoint writes, and finalization.
- Preserve load/validate/restore ordering, optimizer-state resume behavior,
  active-batch resume state, adaptive-rebatch restore, and dynamic Adagrad
  restart state handling.
- Do not change checkpoint schema, row schema, public exports, optimizer math,
  or transition priority.

## Verification

- Add a direct resume-application helper test in the existing optimization
  workflow test module.
- Run focused resume/transition workflow selectors that cover checkpoint load
  count, incompatible resume rejection, optimizer-state discard, batched-LBFGS
  resume, final latest resume, completed resume refresh, and LBFGSB fallback
  controls.
- Run the full CPU unit marker gate before commit.

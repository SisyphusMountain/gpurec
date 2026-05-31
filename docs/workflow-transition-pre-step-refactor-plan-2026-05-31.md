# Workflow Transition Pre-Step Refactor Plan, 2026-05-31

## Scope

Reduce `gpurec/workflow/_transitions.py` by moving the early/pre-step transition
executor into a private workflow helper while preserving the existing
`gpurec.workflow._transitions` import surface.

## Extraction

Add `gpurec/workflow/_transition_pre_step.py` with:

- `build_batch_transition_checkpoint_status()`
- `execute_iteration_transition()`

`_transitions.py` imports these names back from the helper so existing tests and
private callers that import them from `_transitions.py` keep working.

## Boundaries

- Leave `execute_iteration_full_transition()`, `apply_iteration_transition()`,
  `execute_step_status_transition()`, and
  `execute_iteration_post_step_transition()` in `_transitions.py`.
- Keep transition DTOs and `_classify_iteration_transition()` re-exported from
  `_transitions.py`.
- `_transition_pre_step.py` must not import `_transitions.py` or
  `workflow.optimize`; it should import only lower-level workflow helpers,
  types, and model/torch dependencies required by the moved executor.

## Preserved Behavior

- Pre-step action handling is unchanged for `adagrad_restart_terminal`,
  `adagrad_restart_advance`, `adaptive_rebatch`, `lbfgsb_loss_schedule`, `None`,
  `nonfinite_parameter_update`, `adaptive_rebatch_stop`,
  `projected_lbfgs_min_lr_reached`, and unexpected-action errors.
- Checkpoint status fields, planning-state replacements, resume-info clearing,
  optimizer/fd/hessian line-search state resets, adaptive-state updates,
  resident-batch replanning, cache invalidation, solver active-stage
  configuration, progress printing, and `save_status()` calls remain in the same
  order.

## Verification Gates

```bash
python -m compileall -q gpurec/workflow/_transitions.py gpurec/workflow/_transition_pre_step.py tests/unit/test_optimization_workflow.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py
python -m ruff check gpurec/workflow/_transitions.py gpurec/workflow/_transition_pre_step.py tests/unit/test_optimization_workflow.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py
git diff --check
python -m pytest -q tests/unit/test_optimization_workflow.py -k "transitions_reexports or next_batch or step_stopping or hessian_sgd_line_search or resume_state"
python -m pytest -q tests/unit/test_workflow.py -k "adagrad_restarts or adaptive_rebatch or lbfgsb_best_retry"
```

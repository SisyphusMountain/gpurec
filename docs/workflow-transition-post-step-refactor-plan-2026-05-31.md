# Workflow Post-Step Transition Refactor Plan

## Scope

- Move `execute_step_status_transition()` and
  `execute_iteration_post_step_transition()` from
  `gpurec.workflow._transitions` into
  `gpurec.workflow._transition_post_step`.
- Keep `execute_iteration_full_transition()` and
  `apply_iteration_transition()` in `gpurec.workflow._transitions`.

## Compatibility

- Re-export the moved helpers from `gpurec.workflow._transitions` so existing
  imports keep resolving to the same callable objects.
- Continue re-exporting transition DTOs, the transition classifier, and
  pre-step execution helpers from `_transitions`.
- Keep `_transition_post_step` independent from `_transitions`; it depends only
  on lower-level transition policy, pre-step, type, runtime, and solver helpers.

## Risks

- Post-step branches mutate shared workflow state, so the move must stay
  mechanical and preserve execution order.
- The full transition wrapper must still run pre-step handling first and call
  post-step handling only when no early continue or break decision was reached.
- Compatibility import checks should cover identity re-exports and ensure the
  new module does not import `_transitions`.

## Focused Gates

- `python -m compileall gpurec/workflow/_transitions.py gpurec/workflow/_transition_post_step.py`
- `ruff check gpurec/workflow/_transitions.py gpurec/workflow/_transition_post_step.py`
- `git diff --check`
- Focused workflow transition tests when practical:
  `pytest tests/unit/test_optimization_workflow.py -k "transition"`

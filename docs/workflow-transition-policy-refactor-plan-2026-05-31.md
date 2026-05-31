# Workflow Transition Policy Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/workflow/_transitions.py` by moving only the pure
`_classify_iteration_transition()` decision function into a private workflow
helper. Keep transition execution, checkpoint writes, optimizer resets, status
mutation, and post-step handling in `_transitions.py`.

Existing private imports from `gpurec.workflow._transitions` for
`_classify_iteration_transition()` must continue to work.

## Extraction

Add `gpurec/workflow/_transition_policy.py` with:

- `_classify_iteration_transition()`.

The helper imports only `IterationTransition` from `_transition_types.py`.
`_transitions.py` imports the classifier back as a compatibility alias and
continues to call it from the existing pre-step transition sites.

## Boundaries

- `_transition_policy.py` stays pure: no model imports, checkpoint writes,
  optimizer construction, runtime-state mutation, or tensor operations.
- `_transitions.py` remains the owner of executing transition effects.
- Do not export the helper from `gpurec.workflow.__all__` or top-level
  `gpurec`.

## Verification Gates

```bash
python -m compileall -q gpurec/workflow
python -m pytest -q tests/unit/test_optimization_workflow.py::test_transitions_reexports_workflow_transition_classifier
python -m pytest -q tests/unit/test_workflow.py -k "resume or checkpoint or lbfgsb or adagrad_restarts or hessian_sgd or batched_lbfgs or adaptive_rebatch"
python -m pytest -q tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_documents_workflow_submodule_ownership
ruff check gpurec/workflow/_transitions.py gpurec/workflow/_transition_policy.py tests/unit/test_optimization_workflow.py tests/unit/test_repository_hygiene.py
git diff --check
```

## Acceptance Criteria

- `_transitions.py` no longer owns the classifier implementation.
- `_transitions.py` and `_transition_policy.py` expose the same private
  classifier object for compatibility.
- Classification order and returned `IterationTransition` payloads are
  unchanged.

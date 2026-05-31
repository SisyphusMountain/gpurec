# Workflow Transition Types Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/workflow/_transitions.py` by moving transition data-transfer
objects into a private workflow helper. Keep transition execution,
classification, checkpoint writes, optimizer resets, and loop behavior
unchanged.

This is a representation-only extraction. Existing imports from
`gpurec.workflow._transitions` for the moved private DTOs must continue to work.

## Extraction

Add `gpurec/workflow/_transition_types.py` with:

- `IterationTransition`;
- `IterationTransitionExecution`;
- `IterationTransitionOps`;
- `IterationStatusTransitionExecution`;
- `IterationTransitionContext`; and
- `IterationTransitionInputs`.

`gpurec/workflow/_transitions.py` imports these classes back as compatibility
aliases, while `optimize.py` and `_run_state.py` can import the DTOs directly
from `_transition_types.py` to avoid depending on transition execution logic.

## Boundaries

- `_transition_types.py` contains only dataclasses, typing, and `__all__`.
- It must not import `gpurec.workflow._transitions`,
  `gpurec.workflow._run_state`, or `gpurec.workflow.optimize`.
- `IterationTransitionContext` remains mutable because `_run_state.py` updates
  it in place each iteration.
- `IterationTransitionOps` remains the only frozen transition DTO.
- Do not export these private DTOs from `gpurec.workflow.__all__` or top-level
  `gpurec`.

## Verification Gates

```bash
python -m compileall -q gpurec/workflow
python - <<'PY'
import importlib
tr = importlib.import_module("gpurec.workflow._transitions")
tt = importlib.import_module("gpurec.workflow._transition_types")
for name in ("IterationTransition", "IterationTransitionExecution", "IterationTransitionOps", "IterationStatusTransitionExecution", "IterationTransitionContext", "IterationTransitionInputs"):
    assert getattr(tr, name) is getattr(tt, name), name
PY
python -m pytest -q tests/unit/test_optimization_workflow.py::test_transitions_reexports_workflow_transition_type_classes
python -m pytest -q tests/unit/test_workflow.py -k "resume or checkpoint or lbfgsb or adagrad_restarts or hessian_sgd or batched_lbfgs or adaptive_rebatch"
python -m pytest -q tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_documents_workflow_submodule_ownership
ruff check gpurec/workflow/_transitions.py gpurec/workflow/_transition_types.py gpurec/workflow/optimize.py gpurec/workflow/_run_state.py tests/unit/test_optimization_workflow.py tests/unit/test_repository_hygiene.py
git diff --check
```

## Acceptance Criteria

- `_transitions.py` no longer owns transition DTO definitions.
- `_transitions.py` compatibility imports return the same class objects as
  `_transition_types.py`.
- No transition branch, status reason, checkpoint key, row key, or dataclass
  constructor field order changes.
- `_run_state.py` no longer imports transition execution code just to build
  transition context objects.

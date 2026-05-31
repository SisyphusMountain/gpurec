# Workflow Run-State Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/workflow/optimize.py` by moving optimizer run-state containers
and their transition/artifact glue into a private workflow helper. Keep
`OptimizationRunner.run()` and transition policy behavior in place for this
slice.

This extraction is intentionally narrow. It must preserve direct compatibility
imports from `gpurec.workflow.optimize` for the moved classes, while making
`gpurec/workflow/_run_state.py` the implementation owner.

## Extraction

Add `gpurec/workflow/_run_state.py` with:

- `ObjectiveState`;
- `BatchRunState`;
- `RestartRunState`;
- `LBFGSBRunState`; and
- `_OptimizationRunState`.

`gpurec/workflow/optimize.py` imports those names back so existing white-box
tests and downstream direct imports continue to work. The main run loop,
`_step_stopping_status()`, checkpoint helpers, optimizer factory calls,
transition policy modules, and row-building semantics stay unchanged.

## Boundaries

- `_run_state.py` is internal optimizer run-state plumbing and not a public
  workflow API surface.
- Private workflow helper modules must not import back from
  `gpurec.workflow.optimize`.
- `_run_state.py` may instantiate transition context and iteration-artifact
  dataclasses, but should keep annotation-only dependencies under
  `TYPE_CHECKING`.
- Do not export these state classes from `gpurec.workflow.__all__` or
  top-level `gpurec`.

## Verification Gates

```bash
python -m compileall -q gpurec/workflow
python - <<'PY'
import importlib
opt = importlib.import_module("gpurec.workflow.optimize")
rs = importlib.import_module("gpurec.workflow._run_state")
for name in ("ObjectiveState", "BatchRunState", "RestartRunState", "LBFGSBRunState", "_OptimizationRunState"):
    assert getattr(opt, name) is getattr(rs, name), name
PY
python -m pytest -q tests/unit/test_optimization_workflow.py tests/unit/test_workflow_batch_final_cache.py tests/unit/test_adaptive_iterations.py
python -m pytest -q tests/unit/test_workflow.py -k "resume or checkpoint or lbfgsb or adagrad_restarts or hessian_sgd or batched_lbfgs or adaptive_rebatch"
python -m pytest -q tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_documents_workflow_submodule_ownership
ruff check gpurec/workflow/optimize.py gpurec/workflow/_run_state.py tests/unit/test_optimization_workflow.py tests/unit/test_repository_hygiene.py
git diff --check
```

## Acceptance Criteria

- `optimize.py` no longer defines the run-state dataclasses.
- `from gpurec.workflow.optimize import ObjectiveState` and the sibling
  run-state imports still resolve to the `_run_state.py` definitions.
- Checkpoint/status keys and row fields remain byte-stable.
- No private workflow helper imports from `gpurec.workflow.optimize`.
- Existing workflow resume, checkpoint, LBFGSB, adagrad restart, Hessian-SGD,
  batched-LBFGS, adaptive-rebatch, and batch-final-cache gates pass.

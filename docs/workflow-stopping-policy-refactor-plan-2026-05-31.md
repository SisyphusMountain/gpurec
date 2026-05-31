# Workflow Stopping Policy Refactor Plan - 2026-05-31

## Scope

Move the optimizer loop's pure stop-status and active-batch patience helpers out
of `gpurec/workflow/optimize.py` without changing loop ordering, checkpoint
status shape, or active-batch convergence thresholds.

This extraction leaves `OptimizationRunner` responsible for:

- model construction and optimizer orchestration
- step execution and transition wiring
- checkpoint save/restore and final artifact publication
- history row recording

## Target Boundary

Add `gpurec/workflow/_stopping_policy.py` with:

- `_step_stopping_status`
- `_active_batch_patience`
- `_ACTIVE_BATCH_LBFGS_STALL_PATIENCE`

`gpurec.workflow.optimize` imports the private helpers back into the module so
existing private test imports continue to resolve while the policy body has a
single owner.

## Verification

Focused gates:

```bash
python -m compileall -q gpurec/workflow/optimize.py gpurec/workflow/_stopping_policy.py tests/unit/test_optimization_workflow.py tests/unit/test_repository_hygiene.py
ruff check gpurec/workflow/optimize.py gpurec/workflow/_stopping_policy.py tests/unit/test_optimization_workflow.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_optimization_workflow.py -k "step_stopping_status or active_batch_patience or reexports_stopping_policy_helpers"
python -m pytest -q tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_documents_workflow_submodule_ownership
```

CPU marker after focused gates:

```bash
python -m pytest -q -m "unit and not gpu"
```

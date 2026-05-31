# Batched L-BFGS History Refactor Plan - 2026-05-31

## Scope

Extract the row-wise two-loop direction and curvature-history bookkeeping from
`gpurec/optimization/batched_lbfgs.py` into a private support module.

The slice intentionally leaves these behaviors in `batched_lbfgs.py`:

- `BatchedLBFGS._strong_wolfe()`
- the row-wise Armijo loop
- `func_evals` and `max_eval` accounting
- accepted-step state keys and public optimizer exports

## Target Boundary

Add `gpurec/optimization/_batched_lbfgs_history.py` with:

- `_row_dot()`
- `BatchedLBFGSHistoryMixin._direction()`
- `BatchedLBFGSHistoryMixin._append_history()`

`BatchedLBFGS` inherits the mixin and continues to import `_row_dot()` for its
local line-search and step code.

## Verification

Focused gates:

```bash
python -m compileall -q gpurec/optimization tests/unit/test_batched_lbfgs.py tests/unit/test_repository_hygiene.py
ruff check gpurec/optimization/batched_lbfgs.py gpurec/optimization/_batched_lbfgs_history.py tests/unit/test_batched_lbfgs.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_batched_lbfgs.py
python -m pytest -q tests/unit/test_repository_hygiene.py::test_internal_optimization_helper_modules_document_support_boundary
```

Broader optimizer gate:

```bash
python -m pytest -q tests/unit/test_projected_lbfgs.py tests/unit/test_lbfgsb.py tests/unit/test_batched_lbfgs.py tests/unit/test_lbfgsb_schilling_conformance.py
```

CPU marker after commit:

```bash
python -m pytest -q -m "unit and not gpu"
```

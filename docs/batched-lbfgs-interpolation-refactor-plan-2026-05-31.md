# BatchedLBFGS Interpolation Refactor Plan - 2026-05-31

## Scope

Move the pure cubic line-search interpolation helpers out of
`gpurec/optimization/batched_lbfgs.py` without changing strong-Wolfe search
state, closure evaluation, evaluation accounting, parameter updates, or row
history behavior.

This extraction leaves `BatchedLBFGS` responsible for:

- line-search control flow and bracketing state
- trial evaluation and feasible-direction handling
- step acceptance, history updates, and optimizer state

## Target Boundary

Add `gpurec/optimization/_line_search_interpolation.py` with:

- `_clamp_tensor`
- `_cubic_interpolate`

`batched_lbfgs.py` imports `_cubic_interpolate` back into the module. The clamp
helper remains private to the interpolation module.

## Verification

Focused gates:

```bash
python -m compileall -q gpurec/optimization/batched_lbfgs.py gpurec/optimization/_line_search_interpolation.py tests/unit/test_optimization_helpers.py tests/unit/test_batched_lbfgs.py tests/unit/test_repository_hygiene.py
ruff check gpurec/optimization/batched_lbfgs.py gpurec/optimization/_line_search_interpolation.py tests/unit/test_optimization_helpers.py tests/unit/test_batched_lbfgs.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_optimization_helpers.py
python -m pytest -q tests/unit/test_batched_lbfgs.py::test_batched_lbfgs_strong_wolfe_matches_pytorch_per_row_search
python -m pytest -q tests/unit/test_batched_lbfgs.py
python -m pytest -q tests/unit/test_repository_hygiene.py::test_internal_optimization_helper_modules_document_support_boundary
```

CPU marker after focused gates:

```bash
python -m pytest -q -m "unit and not gpu"
```

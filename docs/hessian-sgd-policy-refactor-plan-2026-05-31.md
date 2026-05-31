# Hessian-SGD Policy Refactor Plan - 2026-05-31

## Scope

Centralize pure Hessian-SGD workflow threshold decisions while leaving all
runtime side effects in the existing optimizer loop and transition executor.

The extraction must not move:

- optimizer resets
- objective tracking mutation
- checkpoint or resume state updates
- skipped-full cache evaluation
- warmup Hessian tensor materialization

## Target Boundary

Add `gpurec/workflow/_hessian_sgd_policy.py` with:

- Hessian-SGD line-search and no-refresh threshold constants
- `HessianSGDLineSearchDecision`
- `hessian_sgd_line_search_decision()`
- `hessian_sgd_active_clade_count()`
- `hessian_sgd_should_skip_full_after_warmup()`
- `hessian_sgd_should_carry_warmup_hessian()`

`optimize.py` keeps the loop state mutation and delegates only the
low-acceptance line-search decision. `_transitions.py` keeps transition effects
and delegates only the skip/carry predicates.

## Verification

Focused gates:

```bash
python -m compileall -q gpurec/workflow tests/unit/test_hessian_sgd_policy.py tests/unit/test_repository_hygiene.py
ruff check gpurec/workflow/optimize.py gpurec/workflow/_transitions.py gpurec/workflow/_hessian_sgd_policy.py tests/unit/test_hessian_sgd_policy.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_hessian_sgd_policy.py
python -m pytest -q tests/unit/test_repository_hygiene.py::test_runtime_surface_plan_documents_workflow_submodule_ownership
```

Workflow parity gate:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "hessian_sgd_likelihood_plateau_skips_low_acceptance_line_search or hessian_sgd_low_acceptance_uses_line_search_before_plateau or hessian_sgd_large_batch_uses_long_refresh_until_line_search or hessian_sgd_large_batch_plateau_stops_before_line_search or hessian_sgd_warmup_plateau_promotes_to_full_solver or hessian_sgd_large_batch_warmup_plateau_skips_full_solver or hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full"
```

CPU marker after commit:

```bash
python -m pytest -q -m "unit and not gpu"
```

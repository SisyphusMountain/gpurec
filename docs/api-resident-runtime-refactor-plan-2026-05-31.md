# API Resident Runtime Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/api/model.py` by moving resident batch/cache lifecycle mechanics
into a private API helper module. Keep `GeneReconModel` as the public facade
and keep public method signatures and docstrings in `model.py`.

This slice is more stateful than the factory and genewise streaming slices, so
it should avoid constructor reshaping. Constructor setup may continue to assign
`_batch_specs`, `batch_metadata`, `_resident_cache`, `_static`, and
`_resident_common_state` directly.

## Candidate Extraction

Add `gpurec/api/_resident_runtime.py` with helper functions that take
`model` as the first argument:

- prefetch and cache lifecycle:
  `_shutdown_prefetch_executor_for_replan`, `_ensure_batch_static`,
  `_submit_prefetch`, `_schedule_prefetch`, legacy prefetch helpers, `clear`,
  `close`, and `_close_legacy_prefetch_executor`;
- active batch and theta addressing:
  `_active_static`, `_theta_for_batch_index`, `_active_theta`, `select_batch`,
  `activate_family`, and `next`;
- public resident utilities:
  `replan_resident_batches`, `cached_static_states`,
  `drop_cached_static_states`, and `materialize_batches`.

The model methods should become small delegates so existing direct calls such
as `GeneReconModel.select_batch(model, idx)` still work.

## Boundaries

- `gpurec/api/_resident_runtime.py` may import stdlib concurrency/context
  helpers, `torch`, and API-private modules.
- It must not import `gpurec.workflow`, `gpurec.optimization`, or
  `gpurec.api.model`.
- It must preserve legacy resident-cache compatibility branches until tests
  prove they can be removed in a separate behavior-changing commit.
- It must preserve public error text and batch-selection side effects.

## Verification Gates

```bash
python -m compileall -q gpurec/api
python -m pytest -q tests/unit/test_workflow.py -k "close_tolerates or clear_batched_resident or materialize_batches or select_batch or activate_family or replan_resident_batches or prefetch"
python -m pytest -q tests/unit/test_repository_hygiene.py -k "public_properties or full_batch_helpers or solver_reconfiguration_docs"
python -m pytest -q tests/integration/test_gene_recon_model.py::test_memory_safe_resident_batches_match_resident_and_slice
ruff check gpurec/api/model.py gpurec/api/_resident_runtime.py tests/unit/test_workflow.py
git diff --check
```

Run broader workflow/API gates if any lifecycle semantics change:

```bash
python -m pytest -q tests/unit/test_workflow.py tests/unit/test_model_no_grad_evaluator.py
python -m pytest -q tests/integration/test_gene_recon_model.py
```

## Acceptance Criteria

- `model.py` loses resident lifecycle implementation without losing docstrings
  for public properties and methods.
- Resident cache errors, duplicate/out-of-range validation, current-batch
  restoration, legacy prefetch behavior, and `close()` tolerance for partially
  initialized models are unchanged.
- `materialize_batches()` still returns a copy of `batch_metadata`.
- Package layering remains unchanged.

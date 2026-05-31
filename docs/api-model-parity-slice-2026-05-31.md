# API Model Parity Slice - 2026-05-31

Scope:
- `gpurec/api/model.py` compatibility for partially initialized and legacy-shaped resident prefetch/cache state.

Verification:
- `pytest tests/unit/test_workflow.py -q -k 'test_close_prevents_later_prefetch_restart or test_close_tolerates_partially_initialized_model or test_clear_batched_resident_does_not_materialize_missing_active_batch or test_clear_batched_resident_clears_existing_active_warm_state or test_close_shuts_down_executor_without_batch_lock'`
- Result: `5 passed, 723 deselected in 0.93s`

Integrated follow-up:
- The full workflow suite later passed with `728 passed`.

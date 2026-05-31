# API Resident Batch Mixin Refactor Plan - 2026-05-31

## Scope

Move `GeneReconModel` resident-batch lifecycle wrappers into a private API mixin
without changing public method resolution, construction behavior, likelihood
evaluation, or autograd ownership.

The extraction leaves these in `gpurec/api/model.py`:

- constructors and model initialization
- public metadata properties unrelated to resident lifecycle
- likelihood/loss methods
- export and rate-clamping helpers

## Target Boundary

Add `gpurec/api/_model_resident_batches.py` with
`_GeneReconModelResidentBatchMixin`, containing:

- resident static build/ensure/prefetch wrappers
- active-batch theta/static helpers
- full-batch streaming wrapper
- current batch and cached-static properties
- batch materialization and selection helpers
- lifecycle cleanup methods

`GeneReconModel` inherits the mixin after `_GeneReconModelControlsMixin`.

## Verification

Focused gates:

```bash
python -m compileall -q gpurec/api tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py
ruff check gpurec/api/model.py gpurec/api/_model_resident_batches.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_workflow.py -k "close_prevents_later_prefetch_restart or close_tolerates_partially_initialized_model or resident_batch_members_remain_inherited or current_batch_accessors_remain_inherited or resident_batch_runtime_wrappers_delegate or cached_static_states_property_delegates or stream_full_batches_wrapper_delegates or active_theta_public_wrapper_uses_private_selector or destructor_closes_and_suppresses_errors or clear_batched_resident or public_selectors_reject_nonintegral_indices or select_batch_rejects_out_of_range_indices or materialize_batches or activate_family_batch"
python -m pytest -q tests/unit/test_repository_hygiene.py -k "internal_api_helper_modules_document_support_boundary or project_readme_and_model_docstrings_document_full_batch_helpers or public_properties_and_batched_lbfgs_knobs_are_documented"
```

Streaming parity gate:

```bash
python -m pytest -q tests/unit/test_model_no_grad_evaluator.py -k "stream_full_batches or full_genewise_nll_and_grad or shared_no_grad_full_loss"
python -m pytest -q tests/unit/test_gradient_accumulator.py -k "stream_full_batches"
```

CPU marker after commit:

```bash
python -m pytest -q -m "unit and not gpu"
```

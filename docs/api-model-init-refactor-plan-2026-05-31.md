# API Model Init Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/api/model.py` by extracting `GeneReconModel.__init__`
normalization and static-state setup into API-private helpers. Keep the public
constructor signature, factory signatures, public method docstrings, and
private compatibility attributes unchanged.

This slice does not move likelihood, autograd, workflow, or optimization
behavior.

## Boundaries

- `gpurec/api/model.py` remains the public facade that owns `GeneReconModel`.
- `gpurec/api/_model_init.py` owns constructor validation, `SolverSettings`,
  `ModelBatchSettings`, default theta, origination prior preparation, resident
  common state, and legacy solver attribute mirroring.
- `gpurec/api/_resident_runtime.py` owns resident-batch and full-static
  initialization because it already owns resident cache lifecycle.
- API helpers must not import `gpurec.workflow`, `gpurec.optimization`, or
  `gpurec.api.model`.
- No changes to `core`, `workflow`, or `optimization` are needed.

## Invariants

- Validation order stays compatible: public solver controls reject bad values
  before CUDA/device validation, invalid dtype rejects before device validation,
  and mode/dataset flag mismatches keep their existing error text.
- `_settings`, `_fixed_iters_E`, `_max_iters_E`, `_tol_E`,
  `_fixed_iters_Pi`, `_neumann_terms`, `_adaptive_iters`,
  `_adaptive_neumann_terms`, `_convergence_check_interval`,
  `_e_logsumexp_tol`, `_pi_max_diff_tol`, `_gradient_change_tol`,
  `_gradient_change_rtol`, `_use_pruning`, `_pruning_threshold`,
  `_pi_adjoint_warmstart`, `_pi_adjoint_cache_update_mode`, and
  `_pi_fixed_point_relaxation` are still initialized for private callers.
- Public batching attributes (`family_chunk_size`, `clade_budget`,
  `batch_packing`, `small_family_max_leaves`, `lazy_preprocess`,
  `prefetch_batches`, and `shared_loss_batch_streams`) stay unchanged.
- Resident batch metadata, lazy prefetch behavior, warmup bracketing, and
  full-static metadata are unchanged.

## Verification Gates

```bash
python -m compileall -q gpurec/api
python -m pytest -q tests/unit/test_origination_prior.py::test_gene_recon_model_threads_prepared_origination_prior
python -m pytest -q tests/unit/test_workflow.py -k "gene_recon_init_rejects_invalid_dtype_before_device or gene_recon_init_rejects_invalid_solver_controls_before_device or gene_recon_init_preserves_defaults_and_private_solver_attrs or prefetch_batches_normalization"
python -m pytest -q tests/unit/test_validation.py
python -m pytest -q tests/unit/test_model_no_grad_evaluator.py -k "solver_kwargs_normalize_shared_loss_batch_streams or solver_kwargs_reject_invalid_shared_loss_batch_streams"
python -m pytest -q tests/unit/test_repository_hygiene.py -k "internal_api_helper_modules_document_support_boundary or model_static_state_evaluation"
ruff check gpurec/api/model.py gpurec/api/_model_init.py gpurec/api/_resident_runtime.py gpurec/api/_model_config.py tests/unit/test_origination_prior.py tests/unit/test_workflow.py
git diff --check
```

CUDA parity remains recommended when available:

```bash
python -m pytest -q tests/integration/test_gene_recon_model.py::test_memory_safe_resident_batches_match_resident_and_slice
```

## Acceptance Criteria

- `model.py` line count drops materially without creating a cross-layer helper.
- Constructor behavior and private compatibility attributes are preserved.
- Focused init and hygiene tests pass.
- The slice is committed as a small production-refactor step.

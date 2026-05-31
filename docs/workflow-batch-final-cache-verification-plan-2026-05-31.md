# Workflow Batch Final Cache Verification Plan - 2026-05-31

## Scope

This plan verifies the workflow slice that replaces the active-batch final-cache
tensor triple with a cache object. Production code was inspected but not edited
while producing this plan.

Current tensor triple:

- `batch_final_loss_cache`
- `batch_final_grad_cache`
- `batch_final_cache_ready`

Intended replacement:

- `gpurec/workflow/_batch_final_cache.py`
- `BatchFinalCache`
- `BatchFinalCache.create(model)`
- `BatchFinalCache.cache(model=..., loss_vec=..., active_indices=...)`
- `BatchFinalCache.invalidate(...)`
- `BatchFinalCache.cached_final_result()`

Primary production surfaces:

- `gpurec/workflow/optimize.py`: `_OptimizationRunState`, cache creation,
  run-loop writes, transition-context synchronization, finalization inputs.
- `gpurec/workflow/_transitions.py`: adaptive rebatch invalidation and
  Hessian-SGD warmup-skip cache writes.
- `gpurec/workflow/_finalization.py`: `cached_active_batches` final eval path.
- `gpurec/workflow/_step_execution.py`: `cacheable_active_batch_final_result`
  signal for `adam-fd-newton` and `hessian-sgd`.

The refactor should be a representation change. It must not change checkpoint
payload fields, public history row keys, summary schemas, artifact publication,
or optimizer behavior.

## Baseline Observations

Existing coverage exercises the Hessian-SGD cached final path, noncanonical
warmup skip, adaptive rebatch status, final artifact reuse, checkpoint cadence,
and staged artifact publication. One important gap remains: there is no focused
existing test that drives `batched-lbfgs` through every active batch in full
stage and asserts that finalization uses `cached_active_batches`.

Add that missing test before or during the cache-object extraction. The test can
use the existing fake active-batch model and either enough full-stage batch
transitions or a small scripted runner that marks both active batches ready.

## Gate 1: Cache Object Unit Contract

Recommended new selector:

```bash
python -m pytest -q tests/unit/test_workflow_batch_final_cache.py
```

Recommended tests:

- `test_batch_final_cache_allocates_model_shaped_buffers`
- `test_batch_final_cache_records_active_batch_loss_and_grad_by_global_index`
- `test_batch_final_cache_invalidates_only_selected_family_indices`
- `test_batch_final_cache_ready_payload_returns_detached_clones`

Invariants:

- `BatchFinalCache.create(model)` allocates loss shape `(n_families,)`, grad
  shape `model.theta.shape`, and ready shape `(n_families,)`.
- Loss and grad buffers match `model.theta.device`; loss and grad dtypes match
  `model.theta.dtype`; ready dtype is `torch.bool`.
- Recording copies only active global family indices returned by
  `active_batch_indices(model)`.
- Recording uses detached values and does not alias `loss_vec` or
  `model.theta.grad`.
- Readiness mirrors the legacy cache call: a family is marked ready when the
  active-batch result is accepted for caching. The current runtime path normally
  has a gradient, but the representation change does not alter legacy behavior
  if the gradient is unexpectedly absent.
- Empty active index sets are a no-op.
- Invalidation clears readiness only for the selected family indices and does
  not clear unrelated cached values.
- A ready payload returned to finalization is detached and cloned so finalization
  cannot mutate the cache while installing `model.theta.grad`.
- The cache object is internal runtime state. It must not be serialized into
  checkpoints or written to final artifacts.

## Gate 2: Batched-LBFGS Cached Final Eval

Existing selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_batched_lbfgs_mode_records_public_phase or optimization_runner_batched_lbfgs_advances_resident_batches"
```

Recommended missing selector:

```bash
python -m pytest -q tests/unit/test_workflow.py::test_optimization_runner_batched_lbfgs_uses_cached_final_eval_after_all_active_batches
```

Invariants:

- `batched-lbfgs` full-stage post-step results populate the cache for the
  current active batch.
- Warmup-stage `batched-lbfgs` rows do not mark a batch ready for finalization.
- A partially ready cache falls through to normal genewise final evaluation and
  leaves `optimizer/final_eval_source` absent.
- Once all family rows have been cached, finalization records
  `optimizer/final_eval_source == "cached_active_batches"`.
- The cached final row has `closure_evals == 0`.
- Final NLL is `cache.loss.sum()` and `model.theta.grad` is the cached gradient
  clone.
- `latest.pt["optimizer_phase"]` remains the last optimizer phase
  (`"batched-lbfgs"`) even when `latest.pt["last_row"]["optimizer/phase"]` is
  `"final_eval"`.
- Resident-batch advancement still preserves the public sequence asserted by
  `test_optimization_runner_batched_lbfgs_advances_resident_batches`:
  batch indices `[0, 0, 0, 0, 1]` and solver stages
  `["warmup", "warmup", "full", "full", "warmup"]`.
- Normal active-batch advancement clears transient runtime state without calling
  `drop_cached_static_states()`.

## Gate 3: Hessian-SGD Warmup Skip And Canonical Cache

Existing selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "hessian_sgd_large_batch_warmup_plateau_skips_full_solver or hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full"
```

Invariants:

- Large-batch Hessian-SGD warmup plateau may skip the full-stage row only after
  the warmup loss-stop condition is met.
- The skip path calls
  `solver.active_batch_result_is_canonical_full_solver(phase="hessian-sgd", solver_stage="full")`
  before caching anything as final.
- If the skipped full solver configuration is canonical, the transition
  temporarily configures active stage `"full"`, evaluates current-theta active
  loss and grad once, records the cache, clears runtime state, and finalization
  uses `cached_active_batches`.
- If the skipped full solver configuration is noncanonical, the cache remains
  incomplete and finalization performs the normal full genewise final eval.
- Noncanonical skip must keep `optimizer/final_eval_source` absent in the final
  row.
- Warmup-skip final status remains `converged` with reason
  `loss_change_patience` for the existing fake large-clade runner.
- Fixed Hessian state carryover and solver-stage reset behavior must stay
  independent from cache representation.

## Gate 4: Adaptive Rebatch Invalidation

Existing selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_adaptive_rebatch_replans_unconverged_families or optimization_runner_fd_newton_adaptive_rebatch_replans_unconverged_families or optimization_runner_hessian_sgd_adaptive_rebatch_replans_unconverged_families or optimization_runner_adaptive_rebatch_skips_tiny_active_batches"
```

Add the cache-object unit invalidation test from Gate 1.

Invariants:

- Adaptive rebatch invalidates cache readiness for replanned remaining-family
  indices only.
- Cached converged families that are not replanned remain eligible for final
  cached evaluation.
- Replanned families must be recomputed before `ready_all()` can become true.
- Rebatch still increments `batch_plan_generation`, persists
  `converged_family_indices`, and writes both fields into checkpoint status.
- Rebatch resets active batch state, local step, FD Hessian state,
  Hessian-SGD line-search state, optimizer state, and objective tracking as it
  did before the cache object.
- Tiny active batches still skip rebatch checks and must not invalidate cache
  readiness.

## Gate 5: Finalization Cached-Active-Batches Path

Recommended focused unit selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "cached_active_batches or final_eval_source"
```

If a direct finalization unit test is added:

```bash
python -m pytest -q tests/unit/test_workflow.py::test_finalize_optimization_uses_ready_batch_final_cache_object
```

Invariants:

- Finalization uses the cache only when `batchwise_active_optimizer` is true and
  the cache reports every family ready.
- Cached finalization sets `final_closure_evals = 0`.
- Cached finalization clones cached per-family loss and cached gradient before
  writing `model.theta.grad`.
- Cached finalization records `optimizer/final_eval_source` as
  `cached_active_batches`, plus normal grad, parameter, and solver stats.
- Finite checks still reject nonfinite cached loss or gradient and produce the
  existing `nonfinite_objective_or_gradient` failed status.
- Incomplete or absent cache falls through unchanged:
  `full_genewise_nll_and_grad` when available, otherwise scalar
  `evaluate_and_backward`.
- The memory fallback source `fallback_clade_budget` remains distinct from
  `cached_active_batches`.

## Gate 6: Checkpoint And Artifact Parity

Existing selectors:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_run_writes_outputs_with_fake_model or optimization_runner_reuses_final_genewise_vector_for_artifacts or optimization_runner_skips_per_family_artifact_after_failed_final_genewise_eval or optimization_runner_preserves_final_artifacts_when_staging_fails or optimization_runner_final_latest_resumes_at_next_optimizer_step or optimization_runner_completed_resume_only_refreshes_final_artifacts or optimization_runner_reports_latest_when_no_best_written_this_run"

python -m pytest -q tests/unit/test_workflow_artifacts.py tests/unit/test_artifacts_validator.py -k "workflow_artifacts or validate_checkpoint or cross_validate_summary_history_and_manifest_mismatch"

python -m pytest -q tests/unit/test_optimization_workflow.py -k "resume_state_from_payload_normalizes_checkpoint_metadata"
```

Invariants:

- Checkpoint top-level keys and schema remain unchanged.
- Checkpoints do not contain `BatchFinalCache` or its internal tensors.
- `optimizer_phase` in `latest.pt` and `best.pt` remains the last optimizer
  phase, not `"final_eval"`.
- `last_row` continues to store the final row, including
  `optimizer/final_eval_source` only when the final path actually uses a named
  source.
- `summary.json`, `run_manifest.json`, `history.jsonl`, `theta_final.pt`,
  `rates_final.tsv`, and `per_fam_likelihoods.tsv` remain byte-schema
  compatible.
- The final per-family likelihood artifact reuses the final vector when
  available and does not trigger an extra `full_nll_per_family()` call.
- Failed final genewise evaluation still skips the per-family likelihood
  artifact and writes failed summary/checkpoint status.
- Staged artifact publication and rollback remain independent from the cache
  representation.
- Resume metadata normalization remains unchanged for active batch fields,
  `converged_family_indices`, and `batch_plan_generation`.

## Commands Run For This Plan

Collection:

```bash
python -m pytest --collect-only -q tests/unit/test_workflow.py -k "optimization_runner_batched_lbfgs_mode_records_public_phase or optimization_runner_batched_lbfgs_advances_resident_batches or hessian_sgd_large_batch_warmup_plateau_skips_full_solver or hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full or optimization_runner_adaptive_rebatch_replans_unconverged_families or optimization_runner_fd_newton_adaptive_rebatch_replans_unconverged_families or optimization_runner_hessian_sgd_adaptive_rebatch_replans_unconverged_families or optimization_runner_adaptive_rebatch_skips_tiny_active_batches"
```

Result: `8/728 tests collected`.

```bash
python -m pytest --collect-only -q tests/unit/test_workflow.py -k "optimization_runner_run_writes_outputs_with_fake_model or optimization_runner_reuses_final_genewise_vector_for_artifacts or optimization_runner_skips_per_family_artifact_after_failed_final_genewise_eval or optimization_runner_preserves_final_artifacts_when_staging_fails or optimization_runner_final_latest_resumes_at_next_optimizer_step or optimization_runner_completed_resume_only_refreshes_final_artifacts or optimization_runner_reports_latest_when_no_best_written_this_run"
```

Result: `7/728 tests collected`.

```bash
python -m pytest --collect-only -q tests/unit/test_workflow_artifacts.py tests/unit/test_artifacts_validator.py -k "workflow_artifacts or validate_checkpoint or cross_validate_summary_history_and_manifest_mismatch"
```

Result: `7/15 tests collected`.

```bash
python -m pytest --collect-only -q tests/unit/test_optimization_workflow.py -k "resume_state_from_payload_normalizes_checkpoint_metadata"
```

Result: `1/41 tests collected`.

Focused execution:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_batched_lbfgs_mode_records_public_phase or optimization_runner_batched_lbfgs_advances_resident_batches or hessian_sgd_large_batch_warmup_plateau_skips_full_solver or hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full or optimization_runner_adaptive_rebatch_replans_unconverged_families or optimization_runner_fd_newton_adaptive_rebatch_replans_unconverged_families or optimization_runner_hessian_sgd_adaptive_rebatch_replans_unconverged_families or optimization_runner_adaptive_rebatch_skips_tiny_active_batches"
```

Result: `8 passed, 720 deselected`.

```bash
python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_run_writes_outputs_with_fake_model or optimization_runner_reuses_final_genewise_vector_for_artifacts or optimization_runner_skips_per_family_artifact_after_failed_final_genewise_eval or optimization_runner_preserves_final_artifacts_when_staging_fails or optimization_runner_final_latest_resumes_at_next_optimizer_step or optimization_runner_completed_resume_only_refreshes_final_artifacts or optimization_runner_reports_latest_when_no_best_written_this_run"
```

Result: `7 passed, 721 deselected`.

```bash
python -m pytest -q tests/unit/test_workflow_artifacts.py tests/unit/test_artifacts_validator.py -k "workflow_artifacts or validate_checkpoint or cross_validate_summary_history_and_manifest_mismatch"
```

Result: `7 passed, 8 deselected`.

```bash
python -m pytest -q tests/unit/test_optimization_workflow.py -k "resume_state_from_payload_normalizes_checkpoint_metadata"
```

Result: `1 passed, 40 deselected`.

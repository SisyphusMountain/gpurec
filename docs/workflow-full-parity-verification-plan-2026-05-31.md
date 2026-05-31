# Workflow Full Parity Verification Plan - 2026-05-31

## Scope

This plan is for the remaining `tests/unit/test_workflow.py` parity failures
after commit `0c93622` repaired the bounded workflow selector. The full file is
still expected to have 38 failures at the start of this slice.

Tester scope for this document is verification only. Do not edit production
code while producing this plan. If the worktree is dirty when running these
gates, capture `git status --short` before interpreting results. The focused
execution result below is a baseline observation from the start of the tester
pass; rerun it on the intended baseline if local production diffs are present.

Remaining groups:

- API model close/clear behavior.
- `OptimizationRunner` facade parity after helper extraction.
- Hessian-SGD active-step and transition behavior.
- Final iteration check helper behavior.
- Failed final genewise/scalar evaluation status and artifacts.

## Observed Fast Checks

HEAD at the start of this pass:

```bash
git rev-parse --short HEAD
```

Result: `0c93622`.

API close/clear focused execution:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "close_prevents_later_prefetch_restart or close_tolerates_partially_initialized_model or clear_batched_resident or close_shuts_down_executor_without_batch_lock"
```

Initial result: `5 failed, 723 deselected` in 1.35s. Failures were:

- `test_close_prevents_later_prefetch_restart`: `gpurec.api.model` did not
  expose patchable `ThreadPoolExecutor`.
- `test_close_tolerates_partially_initialized_model`: `close()` did not leave
  `_prefetch_closed` initialized on a partially constructed model.
- `test_clear_batched_resident_does_not_materialize_missing_active_batch`:
  `clear()` raised `resident cache is not initialized`.
- `test_clear_batched_resident_clears_existing_active_warm_state`: `clear()`
  raised `resident cache is not initialized`.
- `test_close_shuts_down_executor_without_batch_lock`: executor shutdown was
  not called when `_batch_lock` was missing.

Collection checks used for selector validation:

```bash
python -m pytest --collect-only -q tests/unit/test_workflow.py -k "close_prevents_later_prefetch_restart or close_tolerates_partially_initialized_model or clear_batched_resident or close_shuts_down_executor_without_batch_lock"
```

Result: `5/728 tests collected`.

```bash
python -m pytest --collect-only -q tests/unit/test_workflow.py -k "projected_gradient_uses_projection_mapping_near_rate_bound or optimization_runner_loss_probe_clears_transient_solver_state or optimizer_scalar_eval_rejects or optimization_runner_genewise_loss_probe_clears_transient_solver_state or genewise_loss_vector_probe_rejects_bad_full_vector_shape or active_genewise_loss_vector_probe_rejects_bad_local_shape or lbfgsb_resume_reapplies_current_fallback_controls or optimization_runner_reports_discarded_resume_optimizer_state or final_genewise_eval_falls_back_to_smaller_clade_budget or genewise_vector_eval_rejects_bad_gradient_shape or final_genewise_eval_does_not_fallback_for_non_memory_error"
```

Result: `12/728 tests collected`.

```bash
python -m pytest --collect-only -q tests/unit/test_workflow.py -k "optimization_runner_hessian_sgd_mode_records_public_phase or hessian_sgd_solver_warmup_keeps_full_e_budget or hessian_sgd_large_batch_warmup_uses_short_pi_neumann_schedule or specieswise_full_solver_stage_raises_e_budget or hessian_sgd_advances_batch_after_full_stage_plateau or hessian_sgd_normal_solver_controls_drive_full_stage or hessian_sgd_periodic_validation_budget_uses_high_budget_steps or hessian_sgd_advances_batch_after_best_likelihood_stall or hessian_sgd_likelihood_plateau_converges_with_nonzero_gradient or hessian_sgd_plateau_converges_without_refreshing_for_gradient or hessian_sgd_likelihood_plateau_does_not_refine_on_projected_gradient or hessian_sgd_likelihood_plateau_skips_low_acceptance_line_search or hessian_sgd_low_acceptance_uses_line_search_before_plateau or hessian_sgd_large_batch_uses_long_refresh_until_line_search or hessian_sgd_large_batch_plateau_stops_before_line_search or hessian_sgd_legacy_gradient_tolerance_is_ignored or hessian_sgd_warmup_plateau_promotes_to_full_solver or hessian_sgd_large_batch_warmup_plateau_skips_full_solver or hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full or optimization_runner_hessian_sgd_adaptive_rebatch_replans_unconverged_families or active_fd_newton_step_commits_staged_pi_adjoint_cache or hessian_sgd_reuses_fixed_hessian_between_refreshes or hessian_sgd_refreshes_fixed_hessian_after_configured_steps or hessian_sgd_refresh_override_forces_fixed_hessian_refresh"
```

Result: `24/728 tests collected`.

```bash
python -m pytest --collect-only -q tests/unit/test_workflow.py -k "final_iteration_check_keeps_static_layout_and_clears_runtime_state or final_iteration_check_runs_for_specieswise_mode or final_iteration_check_rejects_broadcastable_gradient_shape or final_iteration_check_skips_duplicate_when_baseline_is_check_iters or final_iteration_check_skipped_or_disabled_reports_reason or final_iteration_check_falls_back_to_smaller_clade_budget"
```

Result: `6/728 tests collected`.

```bash
python -m pytest --collect-only -q tests/unit/test_workflow.py -k "optimization_runner_lbfgs_rejects_nonfinite_post_step_evaluation or optimization_runner_marks_nonfinite_final_evaluation_failed or optimization_runner_skips_per_family_artifact_after_failed_final_genewise_eval or optimizer_scalar_eval_rejects_non_scalar_loss or optimizer_scalar_eval_rejects_bad_gradient_shape"
```

Result: `5/728 tests collected`.

## Gate 1: API Model Close/Clear

Focused selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "close_prevents_later_prefetch_restart or close_tolerates_partially_initialized_model or clear_batched_resident or close_shuts_down_executor_without_batch_lock"
```

Invariants:

- `gpurec.api.model.ThreadPoolExecutor` remains patchable by tests that exercise
  legacy resident prefetch behavior.
- `close()` is safe on `GeneReconModel.__new__(GeneReconModel)` instances and
  initializes `_prefetch_closed` and `_prefetch_executor`.
- `close()` shuts down an existing legacy prefetch executor with
  `wait=False, cancel_futures=True`, clears `_batch_futures`, and nulls
  `_prefetch_executor`.
- Closing prevents later `_schedule_prefetch()` calls from starting a new
  executor or recording new futures.
- Missing `_batch_lock` does not skip executor shutdown.
- `clear()` on legacy resident `_batch_statics` does not materialize a missing
  active batch.
- `clear()` on an existing active static clears only transient runtime state:
  `warm_E`, `pi_adjoint_cache`, and `pi_adjoint_pending_cache`.

## Gate 2: Runner Facade Parity

Focused selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "projected_gradient_uses_projection_mapping_near_rate_bound or optimization_runner_loss_probe_clears_transient_solver_state or optimizer_scalar_eval_rejects or optimization_runner_genewise_loss_probe_clears_transient_solver_state or genewise_loss_vector_probe_rejects_bad_full_vector_shape or active_genewise_loss_vector_probe_rejects_bad_local_shape or lbfgsb_resume_reapplies_current_fallback_controls or optimization_runner_reports_discarded_resume_optimizer_state or final_genewise_eval_falls_back_to_smaller_clade_budget or genewise_vector_eval_rejects_bad_gradient_shape or final_genewise_eval_does_not_fallback_for_non_memory_error"
```

These tests protect the private `OptimizationRunner` helper surface that older
workflow tests and subclassed fake runners still call directly. Keep these as
thin delegating facades when behavior has moved into split modules:

- `_final_iteration_check_iters()`
- `_configure_solver_stage(model, stage)`
- `_configure_active_solver_stage(model, stage)`
- `_make_optimizer(model, phase)`
- `_evaluate_and_backward(model)`
- `_evaluate_loss_only_probe(model)`
- `_evaluate_genewise_loss_vector_probe(model, active_batch=...)`
- `_evaluate_genewise_vector_and_grad(model)`
- `_evaluate_genewise_vector_and_grad_with_memory_fallback(model)`
- `_evaluate_active_genewise_vector_and_grad(model, solver_stage=...)`
- `_evaluate_active_genewise_vector_grad_at_current_theta(model, solver_stage=...)`
- `_active_batch_metrics(model, loss_vec=..., solver_stage=...)`
- `_active_fd_newton_step(...)`
- `_projected_grad_inf(model, lower_bound=..., upper_bound=...)`
- `_evaluate_final_iteration_check(...)`
- `_restore_optimizer_state(optimizer, state, current_phase=..., checkpoint_phase=...)`

Invariants:

- Facades preserve signatures used by tests, including keyword-only
  `current_phase` and `checkpoint_phase` for `_restore_optimizer_state`.
- Tests can monkeypatch runner methods and have extracted implementation paths
  call the patched runner method, not a module-level bypass.
- Scalar evaluation rejects non-scalar losses and broadcastable wrong-shaped
  gradients before writing invalid gradients into state.
- Genewise vector evaluation rejects incorrect loss vector and gradient shapes.
- Loss-only probes and genewise probes clear transient staged runtime state
  without calling `model.clear()` when staged Pi-adjoint cache semantics require
  preserving the accepted cache.
- `_restore_optimizer_state()` reports `missing`, `restored`, and `discarded`
  states without raising on incompatible optimizer payloads.
- Restored L-BFGS-B optimizers reapply current config runtime controls after
  loading checkpoint state.

## Gate 3: Hessian-SGD Step And Transition Behavior

Focused selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_hessian_sgd_mode_records_public_phase or hessian_sgd_solver_warmup_keeps_full_e_budget or hessian_sgd_large_batch_warmup_uses_short_pi_neumann_schedule or specieswise_full_solver_stage_raises_e_budget or hessian_sgd_advances_batch_after_full_stage_plateau or hessian_sgd_normal_solver_controls_drive_full_stage or hessian_sgd_periodic_validation_budget_uses_high_budget_steps or hessian_sgd_advances_batch_after_best_likelihood_stall or hessian_sgd_likelihood_plateau_converges_with_nonzero_gradient or hessian_sgd_plateau_converges_without_refreshing_for_gradient or hessian_sgd_likelihood_plateau_does_not_refine_on_projected_gradient or hessian_sgd_likelihood_plateau_skips_low_acceptance_line_search or hessian_sgd_low_acceptance_uses_line_search_before_plateau or hessian_sgd_large_batch_uses_long_refresh_until_line_search or hessian_sgd_large_batch_plateau_stops_before_line_search or hessian_sgd_legacy_gradient_tolerance_is_ignored or hessian_sgd_warmup_plateau_promotes_to_full_solver or hessian_sgd_large_batch_warmup_plateau_skips_full_solver or hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full or optimization_runner_hessian_sgd_adaptive_rebatch_replans_unconverged_families or active_fd_newton_step_commits_staged_pi_adjoint_cache or hessian_sgd_reuses_fixed_hessian_between_refreshes or hessian_sgd_refreshes_fixed_hessian_after_configured_steps or hessian_sgd_refresh_override_forces_fixed_hessian_refresh"
```

Invariants:

- Public rows use `optimizer/phase == "hessian-sgd"` and
  `optimizer/fd_newton_subphase == "hessian_sgd"` for Hessian-SGD steps.
- Active rows report active-batch scope, batch index, family count, first/last
  family, solver stage, projected gradient, and finite `theta_step_inf` when a
  step is applied.
- Solver warmup keeps the full E budget while using configured warmup Pi and
  Neumann budgets.
- Large active batches use the short warmup Pi/Neumann schedule and may skip a
  noncanonical full solver stage without caching it as canonical.
- Full-stage normal controls use `hessian_sgd_normal_fixed_iters_pi` and
  `hessian_sgd_normal_neumann_terms`.
- Periodic validation rows temporarily use validation Pi/Neumann budgets and
  restore normal budgets afterward.
- Fixed Hessian state is reused between refreshes when
  `update_hessian_with_bfgs=False`, with zero loss-only evaluations during
  no-line-search Hessian-SGD steps.
- Configured refresh intervals and refresh overrides force finite-difference
  Hessian refreshes with the expected evaluation counts.
- Plateau transitions do not refresh just to inspect projected gradient when
  loss patience has already converged.
- Low acceptance uses line search only after the configured plateau transition;
  large-batch plateau can stop before line search.
- Best-likelihood stall advances resident batches and strips stale polish
  metrics on the first row of the next batch.
- Adaptive rebatching replans unconverged families for Hessian-SGD and closes
  the fake model.
- `_active_fd_newton_step()` commits staged Pi-adjoint cache on accepted
  Hessian-SGD active steps while preserving inactive rows.

## Gate 4: Final Iteration Check Helpers

Focused selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "final_iteration_check_keeps_static_layout_and_clears_runtime_state or final_iteration_check_runs_for_specieswise_mode or final_iteration_check_rejects_broadcastable_gradient_shape or final_iteration_check_skips_duplicate_when_baseline_is_check_iters or final_iteration_check_skipped_or_disabled_reports_reason or final_iteration_check_falls_back_to_smaller_clade_budget"
```

Invariants:

- Final checks do not call `drop_cached_static_states()` for resident
  active-batch models.
- Runtime state is cleared by nulling `warm_E`, `pi_adjoint_cache`,
  `pi_adjoint_pending_cache`, and `last_solver_stats` while preserving static
  layout.
- Genewise final checks configure the final check Pi/Neumann budget, then
  restore the normal solver budget.
- Specieswise final checks configure both E and Pi final check budgets, then
  restore the prior specieswise solver controls.
- Memory retryable final checks rebuild a fallback model with half clade budget
  and `prefetch_batches=1`, report `fallback_clade_budget`, and close the
  fallback model.
- Non-memory final eval errors are not retried as memory fallback.
- Broadcastable wrong-shaped gradients fail with
  `optimizer/final_check_status == "failed"` and restore the baseline gradient.
- Baselines already evaluated at final check iters report status `baseline` and
  zero additional evals.
- Unsupported models report status `skipped` with
  `model_has_no_solver_iteration_controls`.
- `final_check_iters=0` reports status `disabled` with
  `final_check_iters_disabled`.
- `_final_check_summary_metrics()` maps status, source, reason, fallback
  budget, loss delta, and gradient deltas into summary/result fields.

## Gate 5: Failed Final Genewise/Scalar Evaluation Status

Focused selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_lbfgs_rejects_nonfinite_post_step_evaluation or optimization_runner_marks_nonfinite_final_evaluation_failed or optimization_runner_skips_per_family_artifact_after_failed_final_genewise_eval or optimizer_scalar_eval_rejects_non_scalar_loss or optimizer_scalar_eval_rejects_bad_gradient_shape"
```

Invariants:

- Nonfinite post-step scalar evaluation returns a failed result with
  `reason == "nonfinite_objective_or_gradient"` and records only the
  `final_eval` row for the failed finalization path.
- Nonfinite final scalar evaluation preserves the previous finite final NLL,
  sets `final_log_likelihood_bits is None`, and sanitizes summary gradient
  fields to `None`.
- Failed final rows carry
  `optimizer/final_eval_status == "failed"` and
  `optimizer/final_eval_reason == "nonfinite_objective_or_gradient"`.
- Failed final rows do not include stale `likelihood/data_nll_bits`.
- Checkpoints and `summary.json` carry failed status and reason.
- `sampling_checkpoint` is `None` on failed final evaluation.
- Genewise final vector failure calls `full_genewise_nll_and_grad` once and does
  not fall back to `full_nll_per_family()` for artifact writing.
- Stale `per_fam_likelihoods.tsv` is removed when final genewise evaluation
  fails; `summary.json`, `history.jsonl`, and `rates_final.tsv` still exist.
- Models are closed on every failed finalization path.

## Suggested Repair Order

1. Repair API close/clear first. It is independent and has the fastest
   execution gate.
2. Repair `OptimizationRunner` facade parity next. Many Hessian-SGD and final
   check failures can be facade fallout after workflow helper extraction.
3. Run the Hessian-SGD selector and split remaining failures into step metrics
   versus transition state.
4. Run final iteration check helpers before failed final-eval status tests so
   helper failures are not misdiagnosed as run-finalization failures.
5. After all focused gates pass, run:

```bash
python -m pytest -q tests/unit/test_workflow.py
```

Final pass criteria:

- All focused selectors pass.
- Full `tests/unit/test_workflow.py` passes.
- No production-facing test is skipped, xfailed, or renamed to bypass parity.
- Checkpoint, history, summary, and artifact assertions remain intact.

Main-agent follow-up: after integrating the API, runner facade, Hessian-SGD,
transition, and finalization repairs, `tests/unit/test_workflow.py` passed with
`728 passed`.

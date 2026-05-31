# Workflow Transition Ops Verification Plan - 2026-05-31

## Scope

This plan verifies the transition-callback bundling slice for
`gpurec/workflow/_transitions.py` and `gpurec/workflow/optimize.py`. The intended
change is to bundle transition callbacks into one explicit object while keeping
transition behavior, checkpoint payloads, history rows, summaries, and artifacts
unchanged.

Production code was inspected but not edited while producing this plan.

## Callback Bundle Contract

The bundle should contain only callback-style operations, not mutable per-step
runtime state such as optimizer, planning state, batch state, status, or current
phase.

Expected operation groups:

- Checkpoint/resume: `save_status`, `adaptive_checkpoint_status`,
  `load_checkpoint`, `validate_checkpoint_model_compatibility`,
  `restore_model_theta`, `resume_state_from_payload`.
- Optimizer/progress: `make_optimizer`, `restore_optimizer_state`,
  `print_progress_row`.
- Runtime/cache: `active_batch_indices`, `clear_cached_static_states_if_needed`,
  `clear_cached_solver_runtime_state`.
- Stable scalar config used by transitions: `fd_adam_warmup_steps`.

The `optimize.py` wiring must preserve the current bound call targets, especially
runner methods and monkeypatchable module functions. Do not replace bound runner
hooks with direct module calls unless the existing tests are updated for that
contract change.

## Behavior Risks

- Missing or swapped callbacks can silently alter transition branches:
  adaptive rebatch, resident-batch advancement, Hessian-SGD warmup promotion,
  warmup skip, L-BFGS-B retry, and terminal step stopping.
- `load_checkpoint` must keep `map_location="cpu"` and be used exactly where
  retry/resume logic expects it.
- `make_optimizer` must preserve the current runner-bound signature despite
  the local lambda ignoring its first `config` argument.
- Runtime clearing callbacks are distinct: solver runtime clearing must not be
  confused with dropping cached static states during adaptive replanning.
- The bundle must not freeze stale dynamic state. `sync_transition_context()`
  still needs to refresh planning, optimizer, Hessian, resume, cache, stage, and
  phase state each iteration.
- Checkpoint side effects must remain tied to the same branches and cadence:
  adaptive rebatch, next batch, solver warmup switch, adagrad phase advance,
  terminal adagrad status, and step stopping.
- Finalization must remain separate. Bundling transition callbacks must not
  change `final_eval`, cached active-batch final evaluation, staged artifact
  publication, or sampling checkpoint selection.

## Focused Verification Gates

### Gate 1: Collection

Use collection first to catch signature drift and selector typos before running
the heavier fake workflow scenarios:

```bash
python -m pytest --collect-only -q tests/unit/test_workflow.py -k "adaptive_rebatch or advances_batch or warmup_plateau or final_iteration_check or final_genewise_eval or final_latest_resumes or completed_resume or resume_loads_checkpoint_once or discards_resume_optimizer_state or batched_lbfgs_resume_restores_state"
python -m pytest --collect-only -q tests/unit/test_optimization_workflow.py -k "resume_state_from_payload"
python -m pytest --collect-only -q tests/unit/test_workflow_artifacts.py tests/unit/test_artifacts_validator.py -k "publish_staged_artifacts or validate_checkpoint or cross_validate_summary_history_and_manifest_mismatch"
```

Pass criteria: no import or collection errors; selector counts are nonzero.

### Gate 2: Adaptive Rebatch

```bash
python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_adaptive_rebatch_replans_unconverged_families or optimization_runner_fd_newton_adaptive_rebatch_replans_unconverged_families or optimization_runner_hessian_sgd_adaptive_rebatch_replans_unconverged_families or optimization_runner_adaptive_rebatch_skips_tiny_active_batches"
```

Invariants:

- Replanning still calls `replan_resident_batches()` with remaining-family
  indices and increments `batch_plan_generation`.
- Latest checkpoint status still carries `converged_family_indices` and
  `batch_plan_generation`.
- Adaptive rebatch resets optimizer, batch local step, FD Hessian state,
  Hessian-SGD line-search state, resume info, and objective tracking.
- Static-state dropping and batch-final-cache invalidation still use the correct
  callbacks.

### Gate 3: Batch Advancement And Hessian-SGD Warmup

```bash
python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_batched_lbfgs_advances_resident_batches or hessian_sgd_advances_batch_after_full_stage_plateau or hessian_sgd_advances_batch_after_best_likelihood_stall or hessian_sgd_warmup_plateau_promotes_to_full_solver or hessian_sgd_large_batch_warmup_plateau_skips_full_solver or hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full or hessian_sgd_low_acceptance_uses_line_search_before_plateau or hessian_sgd_large_batch_uses_long_refresh_until_line_search"
```

Invariants:

- Resident batch transitions preserve active batch indices, local-step resets,
  solver-stage sequencing, optimizer reset, and checkpoint status fields.
- Warmup-to-full promotion reconfigures active solver stage, clears resume info,
  and preserves/clears FD Hessian state according to the existing large-batch
  rules.
- Hessian-SGD line-search activation still resets tracking and does not advance
  a batch prematurely.
- Large-batch warmup skip caches only canonical full-solver active results and
  leaves noncanonical full-solver results uncached.

### Gate 4: Final Eval, Checkpoints, Resume

```bash
python -m pytest -q tests/unit/test_workflow.py -k "final_iteration_check_keeps_static_layout_and_clears_runtime_state or final_iteration_check_runs_for_specieswise_mode or final_iteration_check_rejects_broadcastable_gradient_shape or final_iteration_check_skips_duplicate_when_baseline_is_check_iters or final_iteration_check_skipped_or_disabled_reports_reason or final_iteration_check_falls_back_to_smaller_clade_budget or final_genewise_eval_falls_back_to_smaller_clade_budget or final_genewise_eval_does_not_fallback_for_non_memory_error or optimization_runner_marks_nonfinite_final_evaluation_failed or optimization_runner_batched_lbfgs_resume_restores_state or optimization_runner_final_latest_resumes_at_next_optimizer_step or optimization_runner_completed_resume_only_refreshes_final_artifacts or optimization_runner_reports_latest_when_no_best_written_this_run or optimization_runner_resume_loads_checkpoint_once or optimization_runner_discards_resume_optimizer_state_on_phase_mismatch"
python -m pytest -q tests/unit/test_optimization_workflow.py -k "resume_state_from_payload"
```

Invariants:

- Retry/resume paths use the bundled checkpoint loaders and restore hooks once,
  with the same status metadata and optimizer-state diagnostics as before.
- Final latest checkpoints still resume at the next optimizer step, and completed
  resumes refresh final artifacts only.
- Final evaluation fallback, final check status, nonfinite final evaluation
  failure, and cached final row behavior remain unchanged.

### Gate 5: Artifact And Validator Guard

```bash
python -m pytest -q tests/unit/test_workflow_artifacts.py tests/unit/test_artifacts_validator.py -k "publish_staged_artifacts or validate_checkpoint or cross_validate_summary_history_and_manifest_mismatch"
```

Invariants:

- Staged artifact publication remains atomic across transition-related failures.
- Checkpoint validation accepts valid checkpoints and rejects malformed payloads.
- Summary/history/manifest cross-validation remains unchanged.

## Review Checklist

- Only `_transitions.py` and `optimize.py` production code should change in the
  implementation slice.
- Public row keys, status reasons, checkpoint fields, and summary fields must be
  bit-for-bit compatible for the focused fake workflow scenarios above.
- New callback bundle types should stay internal to workflow implementation
  modules unless a later public API task explicitly exposes them.

## Local Baseline Note

Focused workflow selectors currently expose a wiring mismatch in the in-progress
bundle implementation: `_transitions.IterationTransitionContext` expects an
`ops` object, while active-batch workflow paths in `optimize.py` still pass
individual callback keywords such as `active_batch_indices`. The adaptive
rebatch selector fails before scenario assertions with this mismatch.

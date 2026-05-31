# Workflow Loop Policy Verification Plan - 2026-05-31

## Scope

This plan verifies a future extraction of the post-step optimizer policy block
from `OptimizationRunner.run()` into a private workflow helper. The extraction
target is the loop section after `execute_optimization_step()` that decides
effective loss tolerance, projected-gradient gating, restart phase completion,
L-BFGS-B loss-schedule and high-KKT behavior, best-objective bookkeeping, row
metadata inputs, step stopping, Hessian-SGD low-acceptance line-search activation,
and L-BFGS-B best-checkpoint retry eligibility.

Production code was inspected but not edited while producing this plan.

## Helper Boundary

Keep the helper private to `gpurec.workflow`, for example in an underscored module
or as an underscored function near the loop. It should not become a public
workflow export.

The helper should make policy decisions explicit and keep side effects narrow.
At minimum, it needs access to:

- Immutable config and loop inputs: `config`, `phase`, `step`, `metrics`,
  `delta`, active-objective flags, active family count, schedule specs, current
  optimizer, model metadata needed by Hessian-SGD, and best-checkpoint existence.
- Mutable state that currently changes in the block: `ObjectiveState`,
  `BatchRunState`, `RestartRunState`, `LBFGSBRunState`, optimizer param-group
  learning rate, and selected optimizer state entries.
- Return values currently consumed later in the loop:
  `loss_change_tol_bits`, `best_likelihood_min_delta_bits`,
  `row_best_nll`, `row_best_step`, `save_best_after_row`,
  `projected_lbfgs_backoff`, `projected_lbfgs_min_lr_reached`,
  `bounded_high_projected_plateau`, `adagrad_restart_terminal_status`,
  `adagrad_restart_phase_next_index`, `adagrad_restart_phase_next_start_step`,
  `lbfgsb_high_kkt_status`, `lbfgsb_loss_schedule_next_index`,
  `effective_loss_patience`, `step_status`, `hessian_sgd_activate_line_search`,
  and `can_lbfgsb_retry`.

Prefer returning a small internal result dataclass over spreading new local
variables through `run()`. If the helper mutates state directly, tests must prove
that each mutation still happens in the same order as today.

## Behavior Risks

- Ordering is the main risk. Projected-LBFGS backoff must happen before stable
  loss patience is updated, and that patience must be updated before Adagrad
  dynamic phase completion and L-BFGS-B loss-schedule advancement are evaluated.
- Active-batch optimizers scale `loss_change_tol` and
  `best_likelihood_min_delta` by active family count, not total family count.
- Projected-LBFGS LR backoff must suppress generic loss-stop handling until the
  configured minimum LR is reached. The helper must preserve the `lr_before`,
  `lr_after`, `lr_reduced`, `min_lr_reached`, `high_projected_grad`, and
  projected-gradient-gate metrics.
- L-BFGS-B uses the active loss-schedule tolerance and patience before generic
  stopping, but high-KKT stop is allowed only when the objective is stalled, the
  final loss phase is active, and the configured fallback-count floor is met.
- L-BFGS-B loss-schedule advancement can seed
  `consecutive_high_kkt_stalls` for the next iteration when
  `lbfgsb_loss_schedule_force_fallback` is enabled. Do not lose that optimizer
  state mutation.
- Adagrad restart dynamic phases use the just-updated stable-loss counter and
  phase-local step. The helper must preserve phase-complete metrics, next phase
  index/start-step fields, terminal status reasons, and the
  `adagrad-restarts-lbfgsb` transition into the L-BFGS-B tail.
- Best-row updates differ by scope. Active-batch rows update `BatchRunState`;
  global/specieswise rows update `ObjectiveState` and may write a best
  checkpoint after the row is finalized.
- Hessian-SGD and batched active optimizers cap patience with
  `_active_batch_patience()`. Hessian-SGD low-acceptance line search is evaluated
  after stop status and must not override a full-stage plateau stop.
- Checkpoint and resume parity depends on row fields generated from these policy
  decisions: stable counters, schedule next indexes, projected-gradient metrics,
  resume optimizer diagnostics, best/latest `next_step`, and optimizer phase.
- Finalization is outside the extraction. Final eval, staged artifact
  publication, sampling checkpoint selection, and cleanup-on-error should not
  move with this helper.

## Focused Verification Gates

### Gate 1: Collection

Run collection first to catch import errors, stale selector names, and helper
signature drift before executing the heavier fake workflow scenarios:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q tests/unit/test_workflow.py -k "projected_lbfgs_reduces_lr or projected_lbfgs_reports_min_lr or lbfgsb_can_stop_on_loss_plateau_without_projected_grad_gate or lbfgsb_loss_schedule_advances_before_stop or lbfgsb_high_kkt_waits_for_final_loss_phase or lbfgsb_can_stop_before_second_high_kkt_fallback or lbfgsb_can_stop_on_budget_exhausted_high_kkt_plateau or lbfgsb_high_kkt_waits_for_objective_plateau or adagrad_restarts_can_advance_flat_phases or adagrad_restarts_lbfgsb_dynamic_prefix_enters_tail or hessian_sgd_likelihood_plateau_converges_with_nonzero_gradient or hessian_sgd_plateau_converges_without_refreshing_for_gradient or hessian_sgd_likelihood_plateau_skips_low_acceptance_line_search or hessian_sgd_low_acceptance_uses_line_search_before_plateau or hessian_sgd_large_batch_plateau_stops_before_line_search or optimization_runner_batched_lbfgs_advances_resident_batches or optimization_runner_batched_lbfgs_resume_restores_state or final_latest_resumes_at_next_optimizer_step or completed_resume_only_refreshes_final_artifacts or resume_loads_checkpoint_once or discards_resume_optimizer_state_on_phase_mismatch"
CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q tests/unit/test_optimization_workflow.py -k "resume_state_from_payload or step_stopping_status"
```

Pass criteria: no import or collection errors, and both selectors collect
nonzero tests.

### Gate 2: Projected-LBFGS LR Backoff And Projected-Gradient Stop Gate

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs_reduces_lr or projected_lbfgs_reports_min_lr or lbfgsb_can_stop_on_loss_plateau_without_projected_grad_gate"
```

Invariants:

- Large projected gradients with stalled objective reduce projected-LBFGS LR
  instead of converging.
- Min-LR rows report `optimizer/projected_lbfgs_min_lr_reached=True` and then
  allow stop behavior according to the current gate.
- L-BFGS-B still records projected-gradient gate metrics without applying
  projected-LBFGS LR backoff.

### Gate 3: L-BFGS-B Loss Schedule And High-KKT Stop

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py -k "lbfgsb_loss_schedule_advances_before_stop or lbfgsb_high_kkt_waits_for_final_loss_phase or lbfgsb_can_stop_before_second_high_kkt_fallback or lbfgsb_can_stop_on_budget_exhausted_high_kkt_plateau or lbfgsb_high_kkt_waits_for_objective_plateau or lbfgsb_best_retry_reloads_checkpoint_once or lbfgsb_resume_reapplies_current_fallback_controls"
```

Invariants:

- The active loss-schedule phase controls loss tolerance and patience for the
  row, and schedule advancement is recorded before generic stop is applied.
- High-KKT stop waits for final loss phase, objective plateau, configured
  fallback count, and the stall/fallback or stall/budget-exhausted signal.
- Loss-schedule force-fallback preserves the optimizer-state mutation that
  primes `consecutive_high_kkt_stalls` for the next L-BFGS-B step.
- Best-checkpoint retry eligibility still depends on global scope, retry budget,
  available best checkpoint, and a known best step.

### Gate 4: Adagrad Restart Dynamic Phase Completion

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py -k "adagrad_restarts_can_advance_flat_phases or adagrad_restarts_lbfgsb_dynamic_prefix_enters_tail or adagrad_restarts_specieswise_uses_schedule or adagrad_restarts_accepts_split_solver_budgets"
```

Invariants:

- Phase completion by loss patience and by phase step cap preserve status
  reasons and next-phase metadata.
- `adagrad-restarts` stops with the existing schedule-complete or phase-patience
  reasons when no tail optimizer follows.
- `adagrad-restarts-lbfgsb` records the next phase as `lbfgsb` and starts that
  tail at `step + 1`.
- Split E/Pi phase budgets remain visible in rows and checkpoint metadata.

### Gate 5: Loss Patience For Hessian-SGD And Batched Optimizers

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py -k "hessian_sgd_likelihood_plateau_converges_with_nonzero_gradient or hessian_sgd_plateau_converges_without_refreshing_for_gradient or hessian_sgd_likelihood_plateau_skips_low_acceptance_line_search or hessian_sgd_low_acceptance_uses_line_search_before_plateau or hessian_sgd_large_batch_plateau_stops_before_line_search or hessian_sgd_advances_batch_after_full_stage_plateau or hessian_sgd_advances_batch_after_best_likelihood_stall or optimization_runner_batched_lbfgs_advances_resident_batches"
```

Invariants:

- Hessian-SGD uses loss patience for stop/advance decisions without reviving the
  legacy gradient-tolerance polish path.
- Low-acceptance line search activates only after the configured low-acceptance
  patience and does not override an already-detected plateau stop.
- Large-batch full-stage plateau avoids unnecessary line-search refresh.
- Batched active optimizers still use capped active-batch patience and reset
  local batch state when advancing resident batches.

### Gate 6: Resume And Checkpoint Parity

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_batched_lbfgs_resume_restores_state or optimization_runner_final_latest_resumes_at_next_optimizer_step or optimization_runner_completed_resume_only_refreshes_final_artifacts or optimization_runner_resume_loads_checkpoint_once or optimization_runner_discards_resume_optimizer_state_on_phase_mismatch or optimization_runner_reports_discarded_resume_optimizer_state"
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_optimization_workflow.py -k "resume_state_from_payload or step_stopping_status"
```

Invariants:

- Resume still loads the checkpoint once with `map_location="cpu"` and restores
  theta before optimizer-state restoration.
- Compatible active optimizer state is restored; missing or incompatible state is
  reported with the same row/checkpoint diagnostics.
- Final latest checkpoints resume at the next optimizer step, while completed
  resumes refresh final artifacts without emitting optimizer-step rows.
- `_step_stopping_status()` behavior remains unchanged for generic loss and best
  likelihood patience.

### Gate 7: Full Workflow And Artifact Guard

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow_artifacts.py tests/unit/test_artifacts_validator.py -k "publish_staged_artifacts or validate_checkpoint or cross_validate_summary_history_and_manifest_mismatch"
```

Invariants:

- The fake workflow suite remains unchanged across optimizer modes, final eval,
  checkpoint cadence, adaptive rebatch, and CLI-facing result summaries.
- Staged artifact publication and checkpoint validation still reject malformed
  payloads and keep summary/history/manifest validation stable.

### Gate 8: Hygiene

```bash
python -m py_compile gpurec/workflow/optimize.py tests/unit/test_workflow.py tests/unit/test_optimization_workflow.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_repository_hygiene.py -k "runtime_surface_plan_documents_workflow_submodule_ownership or optimization_workflow_call_graph_documents_current_cli_and_optimizers or project_readme_documents_workflow_optimizer_modes or project_readme_documents_completed_resume_status"
```

Invariants:

- The helper stays private and is not added to top-level `gpurec.workflow` or
  package exports.
- Repository docs that describe optimizer modes, completed resume, and workflow
  module ownership remain true after the extraction.
- No unrelated production files or generated artifacts are changed by the
  extraction.

## Review Checklist

- Production changes should be limited to workflow internals needed for this
  extraction and any tests that pin the moved policy contract.
- Row keys, status reasons, checkpoint fields, optimizer-state keys, and summary
  fields must stay byte-for-byte compatible in the focused scenarios above.
- Metrics should still be written before `build_iteration_artifacts()` so rows
  and checkpoints see the same values as before.
- `run_state.update_planning_state()` and transition context synchronization must
  continue to receive the post-policy state values.
- If the helper accepts mutable state, code review should trace every mutation
  back to the current block in `OptimizationRunner.run()` and confirm no
  transition or finalization behavior moved accidentally.

## Local Verification Notes

Run during this documentation-only pass:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q tests/unit/test_workflow.py -k "projected_lbfgs_reduces_lr or projected_lbfgs_reports_min_lr or lbfgsb_can_stop_on_loss_plateau_without_projected_grad_gate or lbfgsb_loss_schedule_advances_before_stop or lbfgsb_high_kkt_waits_for_final_loss_phase or lbfgsb_can_stop_before_second_high_kkt_fallback or lbfgsb_can_stop_on_budget_exhausted_high_kkt_plateau or lbfgsb_high_kkt_waits_for_objective_plateau or adagrad_restarts_can_advance_flat_phases or adagrad_restarts_lbfgsb_dynamic_prefix_enters_tail or hessian_sgd_likelihood_plateau_converges_with_nonzero_gradient or hessian_sgd_plateau_converges_without_refreshing_for_gradient or hessian_sgd_likelihood_plateau_skips_low_acceptance_line_search or hessian_sgd_low_acceptance_uses_line_search_before_plateau or hessian_sgd_large_batch_plateau_stops_before_line_search or optimization_runner_batched_lbfgs_advances_resident_batches or optimization_runner_batched_lbfgs_resume_restores_state or final_latest_resumes_at_next_optimizer_step or completed_resume_only_refreshes_final_artifacts or resume_loads_checkpoint_once or discards_resume_optimizer_state_on_phase_mismatch"
```

Result: 21 selected, 707 deselected; collection passed.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q tests/unit/test_optimization_workflow.py -k "resume_state_from_payload or step_stopping_status"
```

Result: 29 selected, 12 deselected; collection passed.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs_reduces_lr or projected_lbfgs_reports_min_lr or lbfgsb_can_stop_on_loss_plateau_without_projected_grad_gate or lbfgsb_loss_schedule_advances_before_stop or lbfgsb_high_kkt_waits_for_final_loss_phase or lbfgsb_can_stop_before_second_high_kkt_fallback or lbfgsb_can_stop_on_budget_exhausted_high_kkt_plateau or lbfgsb_high_kkt_waits_for_objective_plateau or adagrad_restarts_can_advance_flat_phases or adagrad_restarts_lbfgsb_dynamic_prefix_enters_tail or hessian_sgd_likelihood_plateau_converges_with_nonzero_gradient or hessian_sgd_plateau_converges_without_refreshing_for_gradient or hessian_sgd_likelihood_plateau_skips_low_acceptance_line_search or hessian_sgd_low_acceptance_uses_line_search_before_plateau or hessian_sgd_large_batch_plateau_stops_before_line_search or optimization_runner_batched_lbfgs_advances_resident_batches or optimization_runner_batched_lbfgs_resume_restores_state or final_latest_resumes_at_next_optimizer_step or completed_resume_only_refreshes_final_artifacts or resume_loads_checkpoint_once or discards_resume_optimizer_state_on_phase_mismatch"
```

Result: 21 passed, 707 deselected.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_optimization_workflow.py -k "resume_state_from_payload or step_stopping_status"
```

Result: 29 passed, 12 deselected.

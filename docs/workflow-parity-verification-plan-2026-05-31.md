# Workflow Parity Verification Plan - 2026-05-31

## Scope

This plan owns verification for restoring workflow parity after the workflow
optimizer refactor. The active failure surface is in
`tests/unit/test_workflow.py` around bounded specieswise optimizers,
batchwise `batched-lbfgs`, active-batch evaluation helpers, and resume-state
handoff.

Production code was inspected but not edited for this plan. The current split
points are:

- `gpurec/workflow/optimize.py`: run loop, checkpoint writes, public
  `OptimizationRunner` helper surface.
- `gpurec/workflow/_step_plan.py`: initial and per-step optimizer phase
  planning.
- `gpurec/workflow/_step_execution.py`: bounded optimizer and active-batch
  execution metrics.
- `gpurec/workflow/_evaluation.py`: scalar, genewise, and active-batch
  evaluation helpers.
- `gpurec/workflow/_rows.py`: history row and checkpoint-status assembly.
- `gpurec/workflow/_runtime_state.py`: resume-state parsing.

## Current Reproduction

Collection currently sees the broad selector and selects 22 tests:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest --collect-only -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
```

Current exact broad selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
```

Current result after the later concurrent workflow edits observed during this
tester pass: `5 failed, 17 passed, 706 deselected`.

Main-agent follow-up: after the final transition repair in this slice, the same
selector passed with `22 passed, 706 deselected`. The failure list below remains
as the historical narrow verification surface for the repair.

Current failures:

- `test_optimization_runner_lbfgsb_high_kkt_waits_for_final_loss_phase`
  records four `lbfgsb` rows, but row 3 has
  `optimizer/lbfgsb_high_kkt_stop_ready is False` instead of `True`.
- `test_optimization_runner_lbfgsb_best_retry_reloads_checkpoint_once`
  records 4 `lbfgsb` rows instead of 3.
- `test_optimization_runner_lbfgsb_can_stop_before_second_high_kkt_fallback`
  reports `optimizer/lbfgsb_fallback_used_count=0.0` on the first row.
- `test_optimization_runner_lbfgsb_can_stop_on_budget_exhausted_high_kkt_plateau`
  records 3 `lbfgsb` rows instead of 2.
- `test_optimization_runner_batched_lbfgs_advances_resident_batches`
  advances to batch 1, but row 4 keeps `optimizer/solver_stage == "full"`
  instead of resetting to `"warmup"`.

Earlier in this same pass, before the concurrent edits to
`gpurec/workflow/_step_plan.py` and `gpurec/workflow/optimize.py`, the same
selector reported `11 failed, 11 passed, 706 deselected`. The resolved failures
covered dynamic adagrad-to-`lbfgsb` entry, second-row deltas for bounded
optimizers, the active-batch helper location, and `batched-lbfgs` resume hook
arity. The gates below keep those contracts in scope so they do not regress.

The workflow import/hygiene selector below currently passes:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "import_gpurec_does_not_eagerly_import_workflow_or_backtracking or import_workflow_does_not_eagerly_import_heavy_workflow_modules or run_config_workflow_export_does_not_import_optimizer_or_sampling or workflow_config_submodule_import_does_not_import_optimizer_or_sampling or workflow_metadata_helper_import_does_not_import_heavy_modules or workflow_optimize_export_survives_child_module_import_order or top_level_workflow_export_survives_child_module_import_order"
```

Current result: `7 passed, 721 deselected`.

The touched workflow modules currently compile:

```bash
python -m py_compile gpurec/workflow/optimize.py gpurec/workflow/_step_plan.py gpurec/workflow/_step_execution.py gpurec/workflow/_evaluation.py gpurec/workflow/_rows.py gpurec/workflow/_runtime_state.py tests/unit/test_workflow.py
```

## Required Gates

### Gate 1: Focused Reproduction

Run the exact failing selector after each repair:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
```

When a narrower loop is needed, run the currently failing tests directly:

```bash
python -m pytest -q \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_high_kkt_waits_for_final_loss_phase \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_best_retry_reloads_checkpoint_once \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_can_stop_before_second_high_kkt_fallback \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_can_stop_on_budget_exhausted_high_kkt_plateau \
  tests/unit/test_workflow.py::test_optimization_runner_batched_lbfgs_advances_resident_batches
```

Pass criteria:

- The broad selector reports all 22 tests passing.
- The currently passing half of the selector remains passing.
- No test is skipped or deselected by renaming as a substitute for parity.

### Gate 2: Bounded Optimizer Workflow Rows

`projected-lbfgs`, `lbfgsb`, and `batched-lbfgs` rows must preserve row shape,
metric keys, value types, and checkpoint semantics.

Common bounded optimizer row invariants:

- `optimizer/phase` is the public phase string, not an internal helper name.
- `optimizer/eval_position` is `post_step` for bounded LBFGS rows.
- `closure_evals` is the sum of gradient and loss-only evaluations used by the
  optimizer step.
- `theta_step_inf` is finite and greater than zero for accepted updates.
- `delta_likelihood_bits` is `None` only on the first objective-bearing row
  after a reset; second comparable rows must carry a numeric delta.
- `loss_change_tol_bits`, `best_likelihood_min_delta_bits`,
  `stable_loss_steps`, `best_nll_bits`, and `best_step` remain populated as in
  the pre-refactor tests.
- `grad/projected_inf` is present for bounded specieswise optimizers.
- `optimizer_phase` in `checkpoints/latest.pt` stays at the last optimizer phase
  even when `last_row["optimizer/phase"] == "final_eval"`.

`projected-lbfgs` row invariants:

- First basic specieswise run records
  `optimizer/projected_lbfgs_grad_evals == 2.0`,
  `optimizer/projected_lbfgs_loss_evals == 1.0`,
  `optimizer/projected_lbfgs_accepted is True`, and positive
  `theta_step_inf`.
- Large projected-gradient rejection records
  `optimizer/projected_lbfgs_accepted is False`,
  `optimizer/projected_lbfgs_high_projected_grad is True`,
  `optimizer/projected_lbfgs_loss_stop_projected_grad_gate` from config,
  `optimizer/projected_lbfgs_lr_before`, `optimizer/projected_lbfgs_lr_after`,
  `optimizer/projected_lbfgs_lr_reduced`, and
  `optimizer/projected_lbfgs_min_lr_reached`.
- Backoff rows reset `stable_loss_steps` to `0` and do not terminate through
  loss patience until the projected-gradient backoff path is exhausted.

`lbfgsb` row invariants:

- Basic specieswise run records `optimizer/lbfgsb_grad_evals >= 1.0`,
  `optimizer/lbfgsb_loss_evals >= 1.0`,
  `optimizer/lbfgsb_accepted is True`, and
  `optimizer/lbfgsb_direction_kind` in
  `{"cauchy", "subspace", "projected_gradient"}`.
- Loss-stop gating rows record
  `optimizer/lbfgsb_high_projected_grad`,
  `optimizer/lbfgsb_blocked_loss_stop`, and
  `optimizer/lbfgsb_loss_stop_projected_grad_gate`.
- Fallback rows record `optimizer/lbfgsb_fallback_attempted`,
  `optimizer/lbfgsb_fallback_used`,
  `optimizer/lbfgsb_fallback_alpha`,
  `optimizer/lbfgsb_fallback_loss_evals`,
  `optimizer/lbfgsb_fallback_max_loss_evals` when configured,
  `optimizer/lbfgsb_fallback_budget_exhausted`,
  `optimizer/lbfgsb_fallback_reason`,
  `optimizer/lbfgsb_history_cleared_for_fallback`, and the cumulative
  `optimizer/lbfgsb_fallback_used_count`.
- High-KKT rows record `optimizer/lbfgsb_high_kkt_stall_count`,
  `optimizer/lbfgsb_high_kkt_stop_patience`,
  `optimizer/lbfgsb_high_kkt_stop_min_fallbacks`,
  `optimizer/lbfgsb_high_kkt_objective_stalled`,
  `optimizer/lbfgsb_high_kkt_final_loss_phase`, and
  `optimizer/lbfgsb_high_kkt_stop_ready`.
- Loss-schedule rows record `optimizer/lbfgsb_loss_schedule_index`,
  `optimizer/lbfgsb_loss_schedule_phases`,
  `optimizer/lbfgsb_loss_schedule_active_tol`,
  `optimizer/lbfgsb_loss_schedule_active_patience`,
  `optimizer/lbfgsb_loss_schedule_advance`, and on advance
  `optimizer/lbfgsb_loss_schedule_next_index`,
  `optimizer/lbfgsb_loss_schedule_next_tol`,
  `optimizer/lbfgsb_loss_schedule_next_patience`, plus
  `optimizer/lbfgsb_loss_schedule_force_fallback_next` when configured.
- Best-retry rows record `optimizer/lbfgsb_best_retry_count`,
  `optimizer/lbfgsb_best_retry_source_step`, and
  `resume_optimizer_state == "restored"` after the checkpoint reload.

`batched-lbfgs` row invariants:

- Basic genewise run records `optimizer/phase == "batched-lbfgs"`,
  `optimizer/objective_scope == "active_batch"`,
  `optimizer/batch_index`, `optimizer/batch_family_count`,
  `optimizer/batch_family_first`, `optimizer/batch_family_last`, and
  `optimizer/solver_stage`.
- Optimizer metrics include `optimizer/batched_lbfgs_grad_evals`,
  `optimizer/batched_lbfgs_loss_evals`,
  `optimizer/batched_lbfgs_reused_gradient`,
  `optimizer/batched_lbfgs_inner_iters`,
  `optimizer/batched_lbfgs_accepted_rows`,
  `optimizer/batched_lbfgs_accepted_fraction`, and alpha mean/max when the
  optimizer exposes alpha telemetry.
- Resident-batch advancement preserves the expected public sequence:
  batch indices `[0, 0, 0, 0, 1]` and solver stages
  `["warmup", "warmup", "full", "full", "warmup"]` for
  `test_optimization_runner_batched_lbfgs_advances_resident_batches`.
- `drop_cached_static_states()` is not used for normal active-batch advancement;
  runtime state is cleared by nulling `warm_E`, `pi_adjoint_cache`,
  `pi_adjoint_pending_cache`, and `last_solver_stats`.

### Gate 3: Resume-State Invariants

Resume parsing and checkpoint row assembly must preserve these fields:

- `_ResumeState.start_step` comes from checkpoint progress and drives the next
  row step number.
- `status.best_nll_bits`, `status.best_step`,
  `status.previous_objective`, and `status.stable_loss_steps` round-trip
  through checkpoint status and history row generation.
- `status.lbfgsb_fallback_used_count`,
  `status.lbfgsb_loss_schedule_index`, and
  `status.lbfgsb_best_retry_count` round-trip for `lbfgsb`.
- `status.active_batch_index`, `status.active_solver_stage`, and
  `status.active_batch_local_step` round-trip for active-batch optimizers.
- `status.adagrad_restart_dynamic_phase_index` and
  `status.adagrad_restart_dynamic_phase_start_step` round-trip for dynamic
  adagrad-restart schedules.
- When an adagrad-restart dynamic phase completes into the `lbfgsb` tail,
  checkpoint status advances the dynamic phase index to `len(schedule)`, sets
  the next start step, resets `previous_objective` to `None`, resets
  `stable_loss_steps` to `0`, and the next row phase is `lbfgsb`.
- When an `lbfgsb` loss-schedule phase advances, checkpoint status updates
  `lbfgsb_loss_schedule_index` to the next index and resets
  `stable_loss_steps` to `0`.
- Final evaluation writes `last_row["optimizer/phase"] == "final_eval"` but
  does not replace `optimizer_phase` with `final_eval`.

Optimizer-state restore invariants:

- `_restore_optimizer_state()` remains callable by direct tests with keyword
  phase arguments and by `_step_plan.prepare_initial_optimization_plan()` with
  the planner's restore hook contract.
- Missing optimizer state returns `{"resume_optimizer_state": "missing"}`.
- Matching phase returns `{"resume_optimizer_state": "restored"}` and refreshes
  current config runtime options, including `lbfgsb` fallback controls.
- Invalid or mismatched checkpoint phase returns `discarded` with the existing
  reason keys.
- `batched-lbfgs` resume from a one-step checkpoint writes a row at step `1`,
  records `resume_optimizer_state == "restored"`, leaves
  `optimizer_phase == "batched-lbfgs"`, and reports
  `result.steps_completed == 2`.

Focused resume commands:

```bash
python -m pytest -q \
  tests/unit/test_workflow.py::test_lbfgsb_resume_reapplies_current_fallback_controls \
  tests/unit/test_workflow.py::test_optimization_runner_batched_lbfgs_resume_restores_state \
  tests/unit/test_workflow.py::test_optimization_runner_reports_discarded_resume_optimizer_state \
  tests/unit/test_workflow.py::test_optimization_runner_discards_resume_optimizer_state_on_phase_mismatch
```

### Gate 4: Active-Batch Vector and Gradient Invariants

Active-batch evaluation is allowed to live behind `EvaluationOps`, but workflow
parity must keep the behavior contract stable.

Required vector and gradient behavior:

- Active-batch objective evaluation returns a full-length loss vector with shape
  `(model.n_families,)`.
- Active family indices receive local loss values; inactive family rows are
  exactly zero.
- `model.theta.grad` has the same shape as `model.theta`.
- Active gradient rows are preserved; inactive gradient rows are exactly zero.
- Metrics report `optimizer/objective_scope == "active_batch"`,
  `optimizer/batch_index == model.current_batch_index`,
  `optimizer/batch_family_count`, `optimizer/batch_family_first`,
  `optimizer/batch_family_last`, and the current `optimizer/solver_stage`.
- `likelihood/data_nll_bits` is the sum of the full loss vector, including zero
  inactive rows.
- `grad/*` metrics are computed after inactive rows have been zeroed.
- Active loss-only probes clear transient solver state before and after probing
  while preserving accepted staged Pi-adjoint cache values.
- Active-batch final caches copy only active loss and gradient rows and mark
  only those family indices ready.

Focused active-batch commands:

```bash
python -m pytest -q \
  tests/unit/test_workflow.py::test_batched_lbfgs_active_batch_closure_zeros_inactive_rows \
  tests/unit/test_workflow.py::test_optimization_runner_genewise_loss_probe_clears_transient_solver_state \
  tests/unit/test_workflow.py::test_active_genewise_loss_vector_probe_rejects_bad_local_shape \
  tests/unit/test_workflow.py::test_final_iteration_check_keeps_static_layout_and_clears_runtime_state
```

### Gate 5: Phase Advancement and Stop Criteria

The run loop must keep the old phase transition semantics:

- Dynamic `adagrad-restarts-lbfgsb` enters the `lbfgsb` tail immediately after
  the final adagrad dynamic phase completes by loss patience or phase cap.
- Entering the `lbfgsb` tail reconfigures the specieswise LBFGSB tail solver and
  creates a fresh optimizer for the `lbfgsb` phase.
- Bounded projected-gradient backoff rows do not increment `stable_loss_steps`.
- `lbfgsb` high-KKT stop is only ready when high-KKT signal, objective plateau,
  final loss-schedule phase, and fallback-count requirements are all satisfied.
- `lbfgsb` loss-schedule advancement happens before terminal loss-patience stop
  when another scheduled phase remains.
- `batched-lbfgs` active-batch plateau advances resident batches and resets
  local solver-stage state for the next active batch.

Focused phase commands:

```bash
python -m pytest -q \
  tests/unit/test_workflow.py::test_optimization_runner_adagrad_restarts_lbfgsb_continues_after_prefix \
  tests/unit/test_workflow.py::test_optimization_runner_adagrad_restarts_lbfgsb_dynamic_prefix_enters_tail \
  tests/unit/test_workflow.py::test_optimization_runner_projected_lbfgs_reduces_lr_instead_of_stopping_on_large_projected_gradient \
  tests/unit/test_workflow.py::test_optimization_runner_projected_lbfgs_reports_min_lr_with_large_projected_gradient \
  tests/unit/test_workflow.py::test_optimization_runner_batched_lbfgs_advances_resident_batches
```

### Gate 6: Hygiene, Imports, and Broader Workflow Checks

Run syntax and import checks after the focused failures pass:

```bash
python -m py_compile gpurec/workflow/optimize.py gpurec/workflow/_step_plan.py gpurec/workflow/_step_execution.py gpurec/workflow/_evaluation.py gpurec/workflow/_rows.py gpurec/workflow/_runtime_state.py tests/unit/test_workflow.py
python -m pytest -q tests/unit/test_workflow.py -k "import_gpurec_does_not_eagerly_import_workflow_or_backtracking or import_workflow_does_not_eagerly_import_heavy_workflow_modules or run_config_workflow_export_does_not_import_optimizer_or_sampling or workflow_config_submodule_import_does_not_import_optimizer_or_sampling or workflow_metadata_helper_import_does_not_import_heavy_modules or workflow_optimize_export_survives_child_module_import_order or top_level_workflow_export_survives_child_module_import_order"
```

Run the full workflow unit module before closing the repair:

```bash
python -m pytest -q tests/unit/test_workflow.py
```

If production changes touch public exports, repository hygiene, or CLI workflow
import boundaries, add:

```bash
python -m pytest -q tests/unit/test_repository_hygiene.py -k "workflow or optimizer or import"
python -m pytest -q tests/unit/test_cli_workflow.py -k "workflow_import or run_config or optimizer"
```

Pass criteria:

- Public lazy imports remain lazy.
- `gpurec.workflow` wildcard and top-level export tests remain unchanged.
- No workflow refactor helper becomes part of public `gpurec.workflow.__all__`
  unless intentionally documented and tested.
- Full `tests/unit/test_workflow.py` passes after focused parity is restored.

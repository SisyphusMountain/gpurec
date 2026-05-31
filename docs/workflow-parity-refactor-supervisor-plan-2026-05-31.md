# Workflow Parity Refactor Supervisor Plan, 2026-05-31

Scope: documentation-only supervisor plan for the next workflow repair and
consolidation slice. This pass inspected `gpurec/workflow/optimize.py`,
`gpurec/workflow/_step_plan.py`, `gpurec/workflow/_evaluation.py`,
`gpurec/workflow/_step_execution.py`, `gpurec/workflow/_transitions.py`, and
`tests/unit/test_workflow.py`. Do not edit production code in this supervisor
pass.

Concurrent-work note: while this plan was being prepared, uncommitted edits
appeared in `gpurec/workflow/optimize.py`, `gpurec/workflow/_step_plan.py`, and
later `gpurec/workflow/_transitions.py`. The first two add some runner
evaluation forwarding methods and keyword the initial resume optimizer-state
restore call. Treat those production edits as another agent's work; do not
overwrite them. The next implementation slice should keep any already-green
piece and complete the remaining parity repairs below.

## Current Reproduction

Clean `HEAD` was reported at 11 failures for:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
```

After the concurrent in-flight edits observed by this supervisor pass, the same
selector reported `9 failed, 13 passed, 706 deselected`. A later main-agent
follow-up completed the remaining transition repairs and brought the selector
to `22 passed, 706 deselected`. The two early resolved symptoms were the
removed runner active-batch evaluation helper and the positional restore
callback failure, but both still belong in the hard-gate plan because clean
`HEAD` had them.

Current remaining failures:

- `test_optimization_runner_adagrad_restarts_lbfgsb_dynamic_prefix_enters_tail`
  does not enter the `lbfgsb` tail by step 2.
- `test_optimization_runner_lbfgsb_can_stop_on_loss_plateau_without_projected_grad_gate`
  records `delta_likelihood_bits=None` on the second `lbfgsb` row.
- `test_optimization_runner_lbfgsb_loss_schedule_advances_before_stop` records
  5 `lbfgsb` rows instead of 4.
- `test_optimization_runner_lbfgsb_high_kkt_waits_for_final_loss_phase` records
  5 `lbfgsb` rows instead of 4.
- `test_optimization_runner_lbfgsb_best_retry_reloads_checkpoint_once` records
  4 `lbfgsb` rows instead of 3.
- `test_optimization_runner_lbfgsb_can_stop_before_second_high_kkt_fallback`
  reports `optimizer/lbfgsb_fallback_used_count=0.0` on the first row.
- `test_optimization_runner_lbfgsb_can_stop_on_budget_exhausted_high_kkt_plateau`
  records 3 `lbfgsb` rows instead of 2.
- `test_optimization_runner_projected_lbfgs_reduces_lr_instead_of_stopping_on_large_projected_gradient`
  records `delta_likelihood_bits=None` on the second projected-LBFGS row.
- `test_optimization_runner_batched_lbfgs_advances_resident_batches` keeps all
  rows on batch 0 instead of moving the final optimizer row to batch 1.

The common cause to audit first is stale `_StepPlanningState` tracking. The run
loop updates `ObjectiveState.previous_objective`,
`ObjectiveState.stable_loss_steps`, and `LBFGSBRunState.fallback_used_count`
while building the row, but no-op transition paths can return the older planning
state. On the next iteration, `_OptimizationRunState.apply_step_plan()` copies
that stale state back over the live state, which explains the repeated `delta is
None`, missing stable-loss progression, lost fallback count, and missing
active-batch advancement.

## Minimal Slice

### 1. Restore Runner Private API Parity

Keep `OptimizationRunner` as the compatibility facade for tests, subclasses, and
any internal callsites that predate the helper extraction. The methods should be
thin delegates only; do not move helper logic back into `optimize.py`.

Required facade surface for this slice:

- `runner._make_optimizer(model, phase)` delegates to
  `gpurec.workflow._optimizer_factory._make_optimizer(self.config, model, phase)`.
  Every optimizer creation site must go through this facade or an adapter around
  it so test subclasses that override `_make_optimizer()` still control
  scripted optimizers.
- `runner._restore_optimizer_state(optimizer, state, current_phase=None,
  checkpoint_phase=None)` accepts both legacy positional phase arguments and
  keyword phase arguments. It must keep the current return dictionaries for
  missing, discarded, and restored state.
- Preserve or add thin delegates for moved evaluation helpers still called by
  tests, especially `_evaluate_active_genewise_vector_and_grad()`,
  `_evaluate_active_genewise_vector_grad_at_current_theta()`,
  `_evaluate_loss_only_probe()`, `_evaluate_genewise_loss_vector_probe()`, and
  `_active_fd_newton_step()`.

Implementation shape:

- In `optimize.py`, add only facade methods and callback adapters.
- In `_step_plan.py`, remove the direct dependency on the module-level
  `_make_optimizer`; carry a runner-provided factory on `_StepPlanningContext`
  or pass it explicitly into `prepare_initial_optimization_plan()` and
  `select_step_optimization_plan()`.
- In `_transitions.py`, keep the existing `make_optimizer_fn` parameter if that
  is the smallest change, but pass an adapter from `optimize.py` that calls
  `self._make_optimizer(model, phase)`.

### 2. Normalize Restore Callback Signature

Use one restore callback contract everywhere. Prefer a small private `Protocol`
or type alias over drifting `Callable[...]` annotations:

```python
def __call__(
    optimizer: torch.optim.Optimizer,
    state: Any,
    current_phase: str | None = None,
    checkpoint_phase: Any = None,
) -> dict[str, Any]: ...
```

Call it with keywords at internal callsites for clarity:

```python
restore_optimizer_state(
    optimizer,
    resume_payload.get("optimizer_state"),
    current_phase=current_phase,
    checkpoint_phase=resume_payload.get("optimizer_phase"),
)
```

Parity requirements:

- Missing state returns `{"resume_optimizer_state": "missing"}`.
- Non-string checkpoint phase is discarded with
  `resume_optimizer_reason == "invalid_phase"`.
- Phase mismatch is discarded with both checkpoint and current phase recorded.
- Load exceptions from `RuntimeError`, `TypeError`, and `ValueError` are
  discarded with `resume_optimizer_error`.
- Restored bounded optimizers have current config runtime options re-applied,
  including `lbfgs_lr`, `lbfgs_max_iter`, `lbfgs_max_ls`,
  `lbfgsb_fallback_max_coordinates`, `lbfgsb_fallback_max_loss_evals`, and
  `lbfgsb_fallback_resolution_competition_factor`.

### 3. Fix Planning-State Tracking Once

Do not patch each failing optimizer branch separately. Add one runtime tracking
sync point after row bookkeeping and before transition execution returns control
to the next iteration.

Concrete target:

- Extend `_OptimizationRunState.update_planning_state()` or add a neighboring
  helper that refreshes the current `_StepPlanningState` from:
  `current_phase`, `optimizer`, `batch_state.active_index`,
  `batch_state.optimizer_batch_index`, `restart_state.active_phase_index`,
  `objective_state.previous_objective`, `objective_state.stable_loss_steps`, and
  `lbfgsb_state.fallback_used_count`.
- Call it after `objective_state.previous_objective`,
  `objective_state.stable_loss_steps`, and `lbfgsb_state.fallback_used_count`
  are updated, before `run_state.sync_transition_context(...)` and
  `apply_iteration_transition(...)`.
- Leave transition actions that intentionally reset tracking in `_transitions.py`
  intact: adagrad phase advance, adagrad-to-`lbfgsb` tail entry, active-batch
  switch, adaptive rebatch, and solver warmup-to-full switch.
- For `lbfgsb_loss_schedule`, reset `stable_loss_steps` to 0 for the next loss
  phase but keep `previous_objective` so the next comparable `lbfgsb` row can
  compute a numeric delta.

Expected effect:

- Dynamic adagrad restart phases can accumulate stable-loss rows and enter the
  `lbfgsb` tail.
- Projected-LBFGS backoff rows keep `previous_objective` for the next row while
  suppressing loss-patience stop.
- `lbfgsb` fallback-used counts persist into rows and checkpoints.
- Active-batch full-stage plateau can advance to the next resident batch.

### 4. Keep Consolidation Low Risk

The current failures are parity regressions, not a signal to split the state
machine further. For this slice, reduce bloat only where it removes duplicate
plumbing and keeps ownership clear:

- Use runner facade methods as compatibility shims instead of duplicated helper
  implementations in `optimize.py`.
- Replace repeated `replace(planning_state, previous_objective=...,
  stable_loss_steps=..., lbfgsb_fallback_used_count=...)` blocks with one helper
  that updates tracking fields from live run state.
- Remove direct optimizer-factory imports from `_step_plan.py` once factory
  injection is in place.
- Normalize restore callback annotations in `_step_plan.py` and
  `_transitions.py`.

Defer until the selector is green:

- Moving high-KKT policy, loss-schedule policy, or active-batch transition
  policy into new modules.
- Rewriting `execute_iteration_full_transition()` or adding another transition
  layer.
- Collapsing the transition result dataclasses. That may be a useful later
  cleanup, but it should not be mixed with parity repair.

## Invariants

Runner/API invariants:

- Existing test subclasses overriding `_make_optimizer(self, model, phase)` must
  affect initial plan creation, per-step phase changes, and `lbfgsb` best-retry
  optimizer recreation.
- Runner helper methods remain delegates; canonical logic stays in
  `_evaluation.py`, `_step_execution.py`, and `_optimizer_factory.py`.
- Resume state restore supports both positional legacy phase arguments and
  keyword phase arguments.

Objective tracking invariants:

- `delta_likelihood_bits is None` only on the first objective-bearing row after
  an intentional reset.
- Ordinary continuation rows carry numeric deltas and persist
  `previous_objective`.
- Projected-LBFGS LR backoff and min-LR rows reset `stable_loss_steps` to 0 but
  keep enough previous-objective state for the next row's delta.
- `lbfgsb_loss_schedule` advance resets `stable_loss_steps` for the next phase,
  advances `lbfgsb_loss_schedule_index`, and does not discard the comparable
  previous objective.
- `lbfgsb` high-KKT stop requires high-KKT signal, objective plateau, final loss
  schedule phase, and configured fallback-count threshold.
- `lbfgsb_state.fallback_used_count` increments before row/checkpoint artifact
  assembly and persists across ordinary rows.

Active-batch invariants:

- Active-batch genewise evaluation returns a full-family loss vector and zeroes
  inactive `theta.grad` rows.
- Normal active-batch plateau in full solver stage advances batches without
  dropping static states.
- The resident-batch sequence for the focused batched-LBFGS test remains batch
  indices `[0, 0, 0, 0, 1]` and solver stages
  `["warmup", "warmup", "full", "full", "warmup"]`.

Checkpoint/resume invariants:

- `latest.pt["optimizer_phase"]` remains the optimizer phase, even when
  `last_row["optimizer/phase"] == "final_eval"`.
- Resume from a `batched-lbfgs` checkpoint continues at the stored next step,
  records `resume_optimizer_state == "restored"`, and preserves
  `optimizer_phase == "batched-lbfgs"`.
- `lbfgsb` best-retry reloads the best checkpoint at most once when configured,
  records `optimizer/lbfgsb_best_retry_count`, records
  `optimizer/lbfgsb_best_retry_source_step`, and restores optimizer state.

## Required Tests

Run the broad selector after each meaningful repair:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
```

Pass criteria: all 22 selected tests pass, with no skips or test renames.

Use this direct failure list for the narrow loop:

```bash
python -m pytest -q \
  tests/unit/test_workflow.py::test_optimization_runner_adagrad_restarts_lbfgsb_dynamic_prefix_enters_tail \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_can_stop_on_loss_plateau_without_projected_grad_gate \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_loss_schedule_advances_before_stop \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_high_kkt_waits_for_final_loss_phase \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_best_retry_reloads_checkpoint_once \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_can_stop_before_second_high_kkt_fallback \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_can_stop_on_budget_exhausted_high_kkt_plateau \
  tests/unit/test_workflow.py::test_optimization_runner_projected_lbfgs_reduces_lr_instead_of_stopping_on_large_projected_gradient \
  tests/unit/test_workflow.py::test_batched_lbfgs_active_batch_closure_zeros_inactive_rows \
  tests/unit/test_workflow.py::test_optimization_runner_batched_lbfgs_advances_resident_batches \
  tests/unit/test_workflow.py::test_optimization_runner_batched_lbfgs_resume_restores_state
```

Keep these targeted parity checks in the final verification even if they pass
early due to concurrent work:

```bash
python -m pytest -q \
  tests/unit/test_workflow.py::test_optimization_runner_reports_discarded_resume_optimizer_state \
  tests/unit/test_workflow.py::test_lbfgsb_resume_reapplies_current_fallback_controls \
  tests/unit/test_workflow.py::test_batched_lbfgs_active_batch_closure_zeros_inactive_rows \
  tests/unit/test_workflow.py::test_optimization_runner_batched_lbfgs_resume_restores_state
```

Compile the touched workflow modules:

```bash
python -m py_compile \
  gpurec/workflow/optimize.py \
  gpurec/workflow/_step_plan.py \
  gpurec/workflow/_evaluation.py \
  gpurec/workflow/_step_execution.py \
  gpurec/workflow/_transitions.py \
  gpurec/workflow/_rows.py \
  tests/unit/test_workflow.py
```

## Stop Conditions

Stop the slice when the focused selector is green and the production diff shows
only compatibility shims, restore-signature normalization, optimizer-factory
injection, and the single planning-state tracking sync. Do not continue into a
larger transition rewrite in the same slice.

# Workflow Transition Wrapper Collapse Plan - 2026-05-31

## Scope

- Collapse the private `_execute_iteration_full_transition()` trampoline in
  `gpurec/workflow/_transitions.py`.
- Keep `execute_iteration_full_transition(context=..., inputs=...)` and
  `apply_iteration_transition(context=..., inputs=...)` unchanged for callers.
- Keep transition DTOs, checkpoint payloads, status dictionaries, optimizer
  state handling, history row fields, and transition priority unchanged.

## Invariants

- First-step transition classification still ignores post-step-only inputs:
  `lbfgsb_high_kkt_status=None`, `hessian_sgd_activate_line_search=False`, and
  `step_status=None`.
- Post-step transition logic runs only when the first-step transition did not
  request `continue_loop` or `break_loop`.
- Active-batch transitions must preserve `row_best_step`, checkpoint status
  fields, solver-stage resets, optimizer resets, and cache invalidation.
- LBFGSB best-retry must still restore the best checkpoint once, update
  `current_phase`, and report retry telemetry through `resume_info`.

## Implementation Steps

1. Move the private helper body into `execute_iteration_full_transition()` using
   direct `context.*` and `inputs.*` references.
2. Delete `_execute_iteration_full_transition()` and the redundant long
   argument mapping.
3. Add focused direct transition tests for next-batch and step-stopping effects
   so future edits cannot drop checkpoint/status fields while simplifying.
4. Run focused transition/workflow tests and the broad CPU unit marker.

## Verification Gates

- `python -m compileall -q gpurec/workflow/_transitions.py tests/unit/test_optimization_workflow.py tests/unit/test_workflow.py`
- `ruff check gpurec/workflow/_transitions.py tests/unit/test_optimization_workflow.py tests/unit/test_workflow.py`
- `rg -n "_execute_iteration_full_transition|execute_iteration_full_transition\\(" gpurec/workflow tests`
- `python -m pytest -q tests/unit/test_optimization_workflow.py -k "transitions_reexports or transition"`
- `python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_batched_lbfgs_advances_resident_batches or hessian_sgd_warmup_plateau_promotes_to_full_solver or hessian_sgd_large_batch_warmup_plateau_skips_full_solver or optimization_runner_adaptive_rebatch_replans_unconverged_families or optimization_runner_lbfgsb_best_retry_reloads_checkpoint_once or optimization_runner_final_latest_resumes_at_next_optimizer_step or optimization_runner_adagrad_restarts_can_advance_flat_phases"`
- `python -m pytest -q -m "unit and not gpu"`

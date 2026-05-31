# Workflow Transition Ops Supervisor Plan, 2026-05-31

Scope: documentation-only supervisor plan for the transition callback bundling
slice on branch `production` at `06157ee`. Do not edit production code in this
pass. The reviewed production files were `gpurec/workflow/_transitions.py` and
`gpurec/workflow/optimize.py`, with targeted workflow tests under
`tests/unit/test_workflow.py`, `tests/unit/test_optimization_workflow.py`,
`tests/unit/test_cli_workflow.py`, and `tests/unit/test_workflow_batch_final_cache.py`.

## Command Evidence

Commands run from `/home/enzo/Documents/git/gpurec/gpurec`:

```bash
git status --short --branch
rg -n "transition|callback|checkpoint|warmup|rebatch|resume|artifact|active batch|active_batch|next batch|next_batch|step_status|status" tests gpurec/workflow
rg --files tests gpurec/workflow docs
nl -ba gpurec/workflow/_transitions.py | sed -n '1,260p'
nl -ba gpurec/workflow/_transitions.py | sed -n '261,560p'
nl -ba gpurec/workflow/_transitions.py | sed -n '561,920p'
rg -n "execute_iteration_full_transition|execute_step_status_transition|apply_iteration_transition|TransitionContext|TransitionInputs|TransitionResult|StepTransitionResult|transition_callback|step_status|resume_info" gpurec/workflow/optimize.py gpurec/workflow/_transitions.py tests/unit/test_workflow.py tests/unit/test_optimization_workflow.py tests/unit/test_adaptive_iterations.py tests/unit/test_cli_workflow.py
nl -ba gpurec/workflow/_transitions.py | sed -n '921,1220p'
nl -ba gpurec/workflow/_transitions.py | sed -n '1221,1425p'
nl -ba gpurec/workflow/optimize.py | sed -n '1,540p'
nl -ba gpurec/workflow/optimize.py | sed -n '1560,1865p'
nl -ba gpurec/workflow/optimize.py | sed -n '540,980p'
nl -ba gpurec/workflow/optimize.py | sed -n '980,1565p'
rg -n "active_batch|batch_state|next_batch|warmup|adaptive_rebatch|rebatch|resume|latest_checkpoint|best_checkpoint|final|IterationTransition|step_status|lbfgsb_retry|hessian_sgd|batch_final_cache|checkpoint" tests/unit/test_workflow.py tests/unit/test_optimization_workflow.py tests/unit/test_adaptive_iterations.py tests/unit/test_cli_workflow.py tests/unit/test_workflow_batch_final_cache.py
rg -n "IterationTransition|execute_iteration|execute_step_status|apply_iteration_transition|step-status|next_batch|adaptive_rebatch|hessian_sgd|warmup|resume_info|checkpoint_status" tests -g '*.py'
nl -ba tests/unit/test_workflow.py | sed -n '9140,10195p'
nl -ba tests/unit/test_workflow.py | sed -n '10640,11245p'
nl -ba tests/unit/test_workflow.py | sed -n '11740,12435p'
nl -ba tests/unit/test_optimization_workflow.py | sed -n '540,720p'
nl -ba tests/unit/test_workflow.py | sed -n '10190,10285p'
nl -ba gpurec/workflow/_adaptive_rebatch.py | sed -n '1,320p'
nl -ba gpurec/workflow/_finalization.py | sed -n '1,380p'
nl -ba gpurec/workflow/_batch_final_cache.py | sed -n '1,180p'
nl -ba gpurec/workflow/_adaptive_rebatch.py | sed -n '320,430p'
nl -ba gpurec/workflow/_runtime_state.py | sed -n '1,260p'
nl -ba docs/optimizer-helper-refactor-supervisor-plan-2026-05-31.md | sed -n '1,220p'
nl -ba docs/workflow-batch-final-cache-supervisor-plan-2026-05-31.md | sed -n '1,220p'
git rev-parse --short HEAD
git diff --name-only
test -e docs/workflow-transition-ops-supervisor-plan-2026-05-31.md; printf '%s\n' $?
rg -n "execute_iteration_transition\(|execute_step_status_transition\(|execute_iteration_post_step_transition\(|_execute_iteration_full_transition\(|IterationTransitionContext\(|IterationTransitionInputs\(" gpurec/workflow tests/unit/test_workflow.py tests/unit/test_optimization_workflow.py
```

Observed results:

- `git status --short --branch` reported
  `## production...origin/production [ahead 90]` plus unrelated untracked scratch
  files and directories. Leave them untouched.
- `git diff --name-only` was empty before this documentation edit.
- `git rev-parse --short HEAD` returned `06157ee`.
- The target plan path did not exist before this edit (`test -e ...` returned
  `1`).
- The current transition call graph has one public workflow call site:
  `OptimizationRunner.run()` builds `IterationTransitionInputs` and calls
  `apply_iteration_transition()` from `gpurec/workflow/optimize.py`.
- Tests do not directly instantiate `IterationTransitionContext` or
  `IterationTransitionInputs`; the searched hits for transition execution are
  production-only. This makes the slice primarily a compile and behavior-parity
  exercise, not a test fixture migration.
- The prior batch-final-cache consolidation has already landed as
  `BatchFinalCache` in `gpurec/workflow/_batch_final_cache.py`. This slice should
  not rework that object.

## Recommendation

Introduce one private transition ops dataclass and thread that object through
transition context and private transition helpers instead of repeating callback
fields and arguments.

Recommended shape:

```python
@dataclass(frozen=True)
class TransitionOps:
    active_batch_indices: Callable[[GeneReconModel], torch.Tensor]
    clear_cached_static_states_if_needed: Callable[[GeneReconModel], None]
    clear_cached_solver_runtime_state: Callable[[GeneReconModel], None]
    load_checkpoint: Callable[[Path], dict[str, Any]]
    validate_checkpoint_model_compatibility: Callable[..., None]
    restore_model_theta: Callable[[GeneReconModel, dict[str, Any]], None]
    make_optimizer: Callable[[RunConfig, GeneReconModel, str], torch.optim.Optimizer]
    restore_optimizer_state: Callable[
        [torch.optim.Optimizer, Any, str | None, Any | None],
        dict[str, Any],
    ]
    resume_state_from_payload: Callable[[Path, dict[str, Any]], Any]
    save_status: Callable[[Path, Any], None]
    adaptive_checkpoint_status: Callable[[dict[str, Any]], dict[str, Any]]
    print_progress_row: Callable[..., None]
```

The exact class name can be `_TransitionOps` if kept fully module-private. Prefer
placing it in `gpurec/workflow/_transitions.py` next to
`IterationTransitionContext` for this slice. Moving it to a separate private
module is not necessary unless a later slice shares it outside transitions.

Keep `IterationTransitionInputs` as the per-iteration data object. It describes
facts produced by the current loop iteration. The new ops object should contain
only stable helper callbacks and no mutable run state.

## Current Transition Surface

`IterationTransitionContext` currently mixes four kinds of data:

- mutable run state: objective, batch, restart, LBFGSB, adaptive, planning,
  optimizer, Hessian-SGD flags, resume info, and `BatchFinalCache`;
- static route/config state: `RunConfig`, model, evaluation, solver,
  solver-stage booleans, loss schedules, current phase, checkpoint paths,
  checkpoint cadence, and logging cadence;
- callback-like ops: active-index lookup, cache clearing, checkpoint load/save,
  compatibility validation, theta restore, optimizer creation/restore, resume
  state decoding, checkpoint status enrichment, and progress printing;
- iteration data, currently kept separately in `IterationTransitionInputs`.

The callback-like ops are then expanded repeatedly through:

- `execute_iteration_full_transition()` into `_execute_iteration_full_transition()`;
- `_execute_iteration_full_transition()` into `execute_iteration_transition()`;
- `_execute_iteration_full_transition()` into
  `execute_iteration_post_step_transition()`;
- `execute_iteration_post_step_transition()` into
  `execute_step_status_transition()`.

Bundling only the ops removes argument duplication while keeping the existing
transition decision and state objects readable.

## File Plan

`gpurec/workflow/_transitions.py`

- Add the ops dataclass near `IterationTransitionContext`.
- Replace the callback fields on `IterationTransitionContext` with
  `ops: TransitionOps`.
- Change private helper signatures to accept `ops: TransitionOps` instead of
  individual callback arguments:
  `execute_iteration_transition()`, `execute_step_status_transition()`,
  `execute_iteration_post_step_transition()`, and
  `_execute_iteration_full_transition()`.
- Update call sites inside `_transitions.py` by replacing direct argument names
  with `ops.<name>`.
- Keep `apply_iteration_transition()` and `execute_iteration_full_transition()`
  signatures unchanged apart from their context internals.
- Preserve the existing public-ish compatibility of `IterationTransitionInputs`.
  Do not bundle iteration facts into ops.
- Consider adding a brief comment that ops are side-effect hooks, not state
  snapshots. This avoids future misuse where mutable per-step data is put into
  the ops object.

`gpurec/workflow/optimize.py`

- Construct the ops object once in `_OptimizationRunState.make_transition_context()`
  or just before constructing `IterationTransitionContext`.
- Keep callback implementations exactly as they are today:
  `load_checkpoint(..., map_location="cpu")`, `validate_checkpoint_model_compatibility`,
  `restore_model_theta`, `_resume_state_from_payload`, `_drop_cached_static_states_if_needed`,
  `_clear_cached_solver_runtime_state`, `adaptive_state.checkpoint_status`, and
  `_print_progress_row`.
- Avoid rebuilding ops in `sync_transition_context()`. Sync should continue to
  refresh mutable state only: planning state, optimizer, Hessian-SGD state,
  resume info, cache, solver-stage scope, and current phase.
- Keep the lambda signature for `make_optimizer` compatible with the transition
  layer even though the current lambda ignores its `config` argument.

Tests

- No test migration is expected for constructor changes because tests do not
  directly instantiate transition dataclasses.
- Add direct transition unit coverage only if the implementation exposes new
  behavior or accidentally makes ops mutable. Otherwise prefer workflow behavior
  gates below.

## Behavioral Invariants

General:

- The slice is internal only. No public config, CLI, checkpoint schema, result
  schema, history-row schema, or artifact filename changes.
- `IterationTransitionInputs` must still be the only object carrying per-row
  transition facts such as `step_status`, active-batch count, adaptive-rebatch
  pending indices, nonfinite-update flags, and LBFGSB status.
- `TransitionOps` must not capture changing values such as current phase,
  optimizer, resume info, planning state, active batch index, step, row, status,
  or checkpoint status.
- Preserve every existing `save_status()` argument value, especially `step`,
  `next_step`, `optimizer`, `row`, and `optimizer_phase`.
- Preserve all `resume_info` merge/reset behavior. The transition result is still
  the only source used by `_OptimizationRunState.apply_transition_result()`.

Step-status transitions:

- `_classify_iteration_transition()` decision order must not change. Nonfinite
  update, adaptive rebatch stop, Adagrad terminal/advance, adaptive rebatch,
  LBFGSB loss schedule, projected-LBFGS min LR, Hessian-SGD line-search, LBFGSB
  high-KKT retry/stop, active-batch step status, and global step status remain in
  the same priority order.
- Active-objective `step_status` advances to `next_batch` until the last active
  batch; only the last batch turns `step_status` into a terminal status.
- Non-active-objective `step_status` still allows an LBFGSB retry before
  terminal stop when `can_lbfgsb_retry` is true.
- `status_out` cleanup in `execute_iteration_transition()` still drops dicts that
  do not contain `"status"`.

Next-batch checkpointing:

- `next_batch` must reset the batch with
  `warmup=(global_solver_warmup or (active_objective_scope and solver.uses_warmup()))`.
- It must reset optimizer, FD-Newton Hessian state, Hessian-SGD line-search flags,
  objective tracking, and `adaptive_state.last_checked_converged_count`.
- With `checkpoint_every` enabled, it must save `latest.pt` with
  `active_batch_index`, `active_solver_stage`, `active_batch_local_step`,
  `previous_objective=None`, `stable_loss_steps=0`, `best_nll_bits=None`, and
  `best_step=None`.
- With `checkpoint_every` disabled, it must keep the existing behavior of
  returning `break_loop=True` after switching batch, because there is no durable
  next-batch checkpoint to resume from.
- After checkpointed next-batch transition, runtime solver cache clearing,
  `model.select_batch()`, and `solver.configure_active_stage()` must still occur
  before continuing.

Warmup skip and cached active batches:

- Warmup-to-full switching in `execute_iteration_post_step_transition()` must
  continue to suppress `step_status` only when the warmup plateau should promote
  to full solver.
- The large-batch Hessian-SGD skip-full path must cache a final active-batch
  result only when `solver.active_batch_result_is_canonical_full_solver()` is
  true.
- Noncanonical full-equivalent results must not populate `BatchFinalCache` and
  finalization must not emit `optimizer/final_eval_source =
  cached_active_batches`.
- The cached path still evaluates finite loss/grad before caching, clears the
  model afterward, and leaves `warmup_switch=False` so the terminal status can
  finish the run.
- `BatchFinalCache.cached_final_result()` must remain detached clone based; this
  slice should not change cache semantics.

Adaptive rebatch:

- Adaptive rebatch transition must increment `batch_plan_generation`, drop cached
  static states, call `model.replan_resident_batches(indices)`, reset to active
  batch `0`, reset the batch without solver warmup, then set
  `local_step = fd_adam_warmup_steps`.
- It must clear optimizer/Hessian-SGD runtime state, reset objective tracking,
  reset `last_checked_converged_count`, invalidate only replanned cache indices,
  reconfigure the active solver stage, and save a transition checkpoint when
  `checkpoint_every` is enabled.
- `adaptive_checkpoint_status()` must still enrich checkpoint status with
  `converged_family_indices` and `batch_plan_generation`.
- `restore_from_resume()` behavior remains outside this slice except for keeping
  its callback wiring unchanged.

Resume restore and LBFGSB retry:

- Initial resume must still load the checkpoint exactly once with
  `map_location="cpu"` before `prepare_initial_optimization_plan()`.
- Resume must restore theta only after progress and model-compatibility
  validation.
- Active batch index/stage/local step restored from checkpoint status must
  continue to validate against model batch count and solver warmup availability.
- Adaptive-rebatch resume must still restore `converged_family_indices` and
  `batch_plan_generation`, then replan remaining current-plan indices when needed.
- LBFGSB retry must still reload `best.pt`, validate compatibility, require
  `optimizer_phase == "lbfgsb"` for retry restoration, restore model theta,
  rebuild the optimizer, restore optimizer state, restore objective/LBFGSB
  counters, increment `lbfgsb_state.best_retry_count`, clear the model, and return
  resume info containing retry metadata.

Final artifacts:

- Finalization remains downstream of transitions. The ops bundle must not change
  `finalize_optimization()` inputs or the final checkpoint/artifact sequence.
- `finalize_optimization()` must still append the final row, save best when final
  evaluation improves, always save latest, clear `sampling_checkpoint` on failed
  final status, then publish final artifacts through the staged artifact writer.
- Final failed-evaluation behavior must remain: status `failed`,
  reason `nonfinite_objective_or_gradient`, no sampling checkpoint, and no
  per-family likelihood TSV.
- Existing stale-artifact protection must remain untouched: a staged write
  failure cannot partially overwrite prior final artifacts.

## Implementation Sequence

1. Add `TransitionOps` to `_transitions.py`, import no new third-party packages,
   and keep type hints aligned with the current callback signatures.
2. Replace callback fields on `IterationTransitionContext` with one `ops` field.
3. Update `OptimizationRunner.make_transition_context()` to create and pass
   `TransitionOps`; leave `sync_transition_context()` focused on mutable run
   state.
4. Update `execute_iteration_full_transition()` to pass `context.ops` into
   `_execute_iteration_full_transition()`.
5. Convert `_execute_iteration_full_transition()` parameters by removing the
   individual callbacks and using `ops` for its calls to
   `execute_iteration_transition()` and `execute_iteration_post_step_transition()`.
6. Convert `execute_iteration_transition()`,
   `execute_iteration_post_step_transition()`, and
   `execute_step_status_transition()` to accept `ops` and replace callback uses
   mechanically.
7. Run formatting and compile gates before behavioral tests. If a type checker is
   available in the branch workflow, run it after compile and before pytest.

## Verification Gates

Run gates in this order after the production-code slice:

```bash
python -m compileall -q gpurec/workflow
pytest -q tests/unit/test_workflow.py -k "batched_lbfgs_advances_resident_batches or hessian_sgd_advances_batch_after_full_stage_plateau or hessian_sgd_advances_batch_after_best_likelihood_stall"
pytest -q tests/unit/test_workflow.py -k "warmup_plateau_promotes_to_full_solver or warmup_plateau_skips_full_solver or warmup_skip_does_not_cache_noncanonical_full"
pytest -q tests/unit/test_workflow.py -k "adaptive_rebatch_replans_unconverged_families or adaptive_rebatch_skips_tiny_active_batches"
pytest -q tests/unit/test_workflow.py -k "batched_lbfgs_resume_restores_state or final_latest_resumes_at_next_optimizer_step or completed_resume_only_refreshes_final_artifacts or resume_loads_checkpoint_once or discards_resume_optimizer_state_on_phase_mismatch"
pytest -q tests/unit/test_workflow.py -k "nonfinite_parameter_update or nonfinite_final_evaluation or preserves_final_artifacts_when_staging_fails or reports_latest_when_no_best_written_this_run"
pytest -q tests/unit/test_optimization_workflow.py -k "resume_state_from_payload"
pytest -q tests/unit/test_workflow_batch_final_cache.py
pytest -q tests/unit/test_cli_workflow.py -k "checkpoint_info or resume_checkpoint"
pytest -q tests/unit/test_workflow.py
```

Expected search gates:

```bash
rg -n "load_checkpoint_fn|validate_checkpoint_model_compatibility: Callable|restore_model_theta_fn|make_optimizer_fn|restore_optimizer_state_fn|resume_state_from_payload_fn|clear_cached_solver_runtime_state: Callable|active_batch_indices: Callable|print_progress_row: Callable|save_status: Callable|adaptive_checkpoint_status: Callable" gpurec/workflow/_transitions.py gpurec/workflow/optimize.py
rg -n "TransitionOps|ops\." gpurec/workflow/_transitions.py gpurec/workflow/optimize.py
```

The first search should return no repeated callback-field/signature hits except
acceptable definitions inside the new ops dataclass. The second search should
show a single ops object constructed in `optimize.py` and threaded through
transition execution in `_transitions.py`.

## Review Checklist

- Diff touches only `_transitions.py` and `optimize.py` for production code unless
  a tiny private module is introduced deliberately.
- No changes to `RunConfig`, CLI parser, checkpoint save/load schema, result
  dataclasses, final artifact writers, or workflow tests are required for the
  basic slice.
- Every previous callback name has exactly one home in the ops dataclass.
- No mutable per-iteration value is hidden inside ops.
- All transition result assignments still flow through
  `_OptimizationRunState.apply_transition_result()`.
- All targeted gates pass before running the full workflow test file.

# Workflow Loop Policy Supervisor Plan, 2026-05-31

Scope: documentation-only supervisor plan for the workflow loop-policy
extraction slice on branch `production` at `1457a17`. Do not edit production
code in this pass. The reviewed production files were
`gpurec/workflow/optimize.py`, `gpurec/workflow/_rows.py`,
`gpurec/workflow/_transitions.py`, `gpurec/workflow/_runtime_state.py`,
`gpurec/workflow/_step_plan.py`, and `gpurec/workflow/_finalization.py`, with
targeted workflow tests under `tests/unit/test_workflow.py` and
`tests/unit/test_optimization_workflow.py`.

Concurrent-work note: `git status --short --branch` reported
`## production...origin/production [ahead 92]` plus unrelated untracked scratch
files and directories. Leave those untouched. `git diff --name-only` was empty
before this documentation edit, and the target plan path did not exist.

Post-write concurrent-work note: final verification later showed
`M gpurec/workflow/optimize.py`, `?? gpurec/workflow/_loop_policies.py`, and
`?? docs/workflow-loop-policy-verification-plan-2026-05-31.md`. Treat those as
concurrent implementation/verification work. Do not overwrite or rename them
just to match the suggested helper-module name below; audit and adapt the plan
around whichever private helper file already exists.

## Command Evidence

Commands run from `/home/enzo/Documents/git/gpurec/gpurec`:

```bash
git status --short --branch
rg --files -g 'optimize.py' -g '_rows.py' -g '_transitions.py' -g '*test*' -g 'tests/**'
rg -n "post-step|projected|LBFGS|L-BFGS|adagrad|best update|KKT|loss schedule|checkpoint|history" gpurec tests
nl -ba gpurec/workflow/optimize.py | sed -n '600,760p'
nl -ba gpurec/workflow/optimize.py | sed -n '780,900p'
nl -ba gpurec/workflow/optimize.py | sed -n '1210,1435p'
nl -ba gpurec/workflow/optimize.py | sed -n '1420,1515p'
nl -ba gpurec/workflow/optimize.py | sed -n '1500,1775p'
nl -ba gpurec/workflow/_rows.py | sed -n '1,260p'
nl -ba gpurec/workflow/_transitions.py | sed -n '1,320p'
nl -ba gpurec/workflow/_transitions.py | sed -n '320,680p'
nl -ba gpurec/workflow/_transitions.py | sed -n '680,1040p'
rg -n "projected_lbfgs|lbfgsb_loss_schedule|high_kkt|adagrad_restart_dynamic|adagrad_restart_phase|best_likelihood|stable_loss_steps|best_step|checkpoint_status" tests/unit/test_optimization_workflow.py tests/unit/test_workflow.py tests/unit/test_projected_lbfgs.py tests/unit/test_hogenom_multifidelity_adagrad.py tests/unit/test_lbfgsb.py tests/unit/test_workflow_artifacts.py tests/unit/test_cli_workflow.py
rg -n "build_iteration_artifacts|_classify_iteration_transition|execute_iteration|IterationTransition|lbfgsb_loss_schedule|adagrad_restart" tests gpurec | head -n 200
ls docs | sed -n '1,120p'
nl -ba tests/unit/test_workflow.py | sed -n '7700,7850p'
nl -ba tests/unit/test_workflow.py | sed -n '8250,8320p'
nl -ba tests/unit/test_workflow.py | sed -n '8290,8420p'
nl -ba tests/unit/test_workflow.py | sed -n '8416,8790p'
nl -ba tests/unit/test_workflow.py | sed -n '8782,8876p'
nl -ba tests/unit/test_workflow.py | sed -n '8870,8960p'
nl -ba tests/unit/test_optimization_workflow.py | sed -n '470,720p'
git rev-parse --short HEAD
git diff --name-only
test -e docs/workflow-loop-policy-supervisor-plan-2026-05-31.md; printf '%s\n' $?
rg -n "class ObjectiveState|def update_best|class LBFGSBRunState|class RestartRunState|make_iteration_artifacts|make_iteration_artifacts_state|make_iteration_artifacts_inputs|apply_transition_result" gpurec/workflow/optimize.py gpurec/workflow/_runtime_state.py
nl -ba gpurec/workflow/optimize.py | sed -n '160,270p'
nl -ba gpurec/workflow/optimize.py | sed -n '300,460p'
nl -ba gpurec/workflow/optimize.py | sed -n '460,520p'
rg -n "def _step_stopping_status|class _ResumeState|def _resume_state_from_payload|def checkpoint_status_dict|checkpoint_nonnegative_int|lbfgsb_loss_schedule_index|adagrad_restart_dynamic_phase_index" gpurec/workflow tests/unit/test_workflow.py tests/unit/test_optimization_workflow.py
nl -ba gpurec/workflow/_runtime_state.py | sed -n '1,180p'
nl -ba gpurec/workflow/_finalization.py | sed -n '180,260p'
nl -ba gpurec/workflow/_step_plan.py | sed -n '80,180p'
nl -ba gpurec/workflow/_step_plan.py | sed -n '230,310p'
rg -n "class AdagradRestartPhase|class LossStopPhase|def _adagrad_restart_phase_name|class _ActiveAdagradRestartPhase|def _continues_after_adagrad_restart_prefix" gpurec/workflow gpurec
nl -ba gpurec/workflow/_step_plan.py | sed -n '1,80p'
nl -ba gpurec/workflow/_phase.py | sed -n '1,120p'
nl -ba gpurec/workflow/config.py | sed -n '1,180p'
nl -ba docs/optimizer-helper-refactor-supervisor-plan-2026-05-31.md | sed -n '1,220p'
nl -ba docs/workflow-batch-final-cache-supervisor-plan-2026-05-31.md | sed -n '1,220p'
rg -n "class _WorkflowRejectingProjectedLBFGSRunner|class _WorkflowSpecieswiseAdagradRestartPlateauRunner|_optimizer_mode_history_rows|def _optimizer_mode_config" tests/unit/test_workflow.py
nl -ba tests/unit/test_workflow.py | sed -n '7200,7295p'
```

Observed results:

- `git status --short --branch` showed the branch as `production`, ahead of
  `origin/production`, with unrelated untracked scratch paths. No tracked
  production-code diffs were present before this documentation edit.
- `git rev-parse --short HEAD` returned `1457a17`.
- The first `rg --files` command was intentionally broad and surfaced the
  requested workflow files plus many test, data, and pycache paths; subsequent
  `rg` and `nl` reads narrowed the review to the loop-policy block and its
  targeted tests.
- The post-step policy block currently lives in `OptimizationRunner.run()` after
  metrics collection and before `build_iteration_artifacts()`, spanning the
  projected-gradient stop/backoff policy, stable-loss tracking, adagrad dynamic
  phase decisions, best update choice, LBFGSB high-KKT stop, and LBFGSB loss
  schedule advance.
- `_rows.py` constructs the history row and checkpoint status from prepared
  values. It resets checkpoint `stable_loss_steps` and `previous_objective` when
  adagrad dynamic phase advances, and resets checkpoint `stable_loss_steps` when
  the LBFGSB loss schedule advances.
- `_transitions.py` classifies and executes effects after the row is already
  built. It advances adagrad phases, applies LBFGSB loss-schedule transitions,
  handles projected-LBFGS min-LR termination, retries LBFGSB from the best
  checkpoint, and writes transition checkpoints.
- Existing regression tests cover adagrad dynamic phase advancement, adagrad to
  LBFGSB tail entry, LBFGSB loss-schedule advancement, high-KKT stop gates,
  best-retry checkpoint reload, projected-LBFGS LR backoff, projected-LBFGS
  min-LR termination, generic stopping status, and resume metadata validation.

## Recommendation

Extract the post-step policy decisions into one private helper module, using
small private result dataclasses. A singular name such as
`gpurec/workflow/_loop_policy.py` is acceptable for a fresh slice, but if
concurrent work already introduced `gpurec/workflow/_loop_policies.py`, keep that
name and review the implementation against the invariants below. Keep row
construction in `_rows.py`, transition classification and side effects in
`_transitions.py`, and optimizer execution in `optimize.py`.

The helper module should make decisions and prepare metric updates. It must not
save checkpoints, append history rows, switch solver stages, build final rows, or
call `apply_iteration_transition()`.

Recommended private helpers:

```python
@dataclass(frozen=True)
class _ProjectedLossPolicyDecision:
    backoff: bool
    min_lr_reached: bool
    bounded_high_projected_plateau: bool


def _apply_projected_loss_stop_policy(
    *,
    phase: str,
    optimizer: torch.optim.Optimizer | None,
    metrics: dict[str, Any],
    delta: float | None,
    loss_change_tol_bits: float,
    projected_grad_tol: float,
    loss_stop_projected_grad_gate: bool,
    projected_lbfgs_min_lr: float,
) -> _ProjectedLossPolicyDecision: ...


def _update_stable_loss_tracking(
    objective_state: ObjectiveState,
    *,
    objective: float,
    delta: float | None,
    loss_change_tol_bits: float,
    projected_backoff: bool,
    projected_min_lr_reached: bool,
    bounded_high_projected_plateau: bool,
) -> bool: ...


@dataclass(frozen=True)
class _AdagradRestartDynamicDecision:
    next_index: int | None
    next_start_step: int | None
    terminal_status: dict[str, str] | None


def _decide_adagrad_restart_dynamic_phase(
    *,
    enabled: bool,
    active_phase: _ActiveAdagradRestartPhase | None,
    phase_step: int | None,
    stable_loss_steps: int,
    loss_patience: int,
    specs: tuple[AdagradRestartPhase, ...],
    optimizer_name: str,
    step: int,
    metrics: dict[str, Any],
) -> _AdagradRestartDynamicDecision: ...


@dataclass(frozen=True)
class _BestUpdateDecision:
    best_nll: float | None
    best_step: int | None
    save_best_after_row: bool


def _update_best_after_objective(
    *,
    active_objective_scope: bool,
    objective_state: ObjectiveState,
    batch_state: BatchRunState,
    objective: float,
    step: int,
    best_likelihood_min_delta_bits: float,
) -> _BestUpdateDecision: ...


@dataclass(frozen=True)
class _LBFGSBHighKKTDecision:
    status: dict[str, str] | None
    stop_ready: bool


def _decide_lbfgsb_high_kkt_stop(
    *,
    phase: str,
    metrics: dict[str, Any],
    lbfgsb_state: LBFGSBRunState,
    loss_schedule: tuple[LossStopPhase, ...],
    loss_schedule_index: int,
    objective_plateau_this_row: bool,
    high_kkt_stop_patience: int,
    high_kkt_stop_min_fallbacks: int,
) -> _LBFGSBHighKKTDecision: ...


@dataclass(frozen=True)
class _LBFGSBLossScheduleDecision:
    next_index: int | None


def _decide_lbfgsb_loss_schedule_advance(
    *,
    phase: str,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    theta: torch.nn.Parameter,
    high_kkt_status: dict[str, str] | None,
    loss_schedule: tuple[LossStopPhase, ...],
    loss_schedule_index: int,
    stable_loss_steps: int,
    effective_loss_patience: int,
    force_fallback: bool,
) -> _LBFGSBLossScheduleDecision: ...
```

If importing `ObjectiveState`, `BatchRunState`, or `LBFGSBRunState` from
`optimize.py` would create an awkward cycle, keep the helpers in `optimize.py`
for the first slice and move them to `_loop_policy.py` later. The extraction
boundary matters more than the module move. If helpers stay in `optimize.py`,
place them near the existing state dataclasses and keep them module-private.

## File Plan

`gpurec/workflow/optimize.py`

- Replace the current inline block around lines 1239-1535 with helper calls in
  the same order:
  projected policy, stable-loss update, adagrad dynamic decision, best update,
  LBFGSB high-KKT decision, LBFGSB loss-schedule decision, row build.
- Keep local variable names that feed `_rows.py` and `_transitions.py`:
  `projected_lbfgs_backoff`, `projected_lbfgs_min_lr_reached`,
  `bounded_high_projected_plateau`, `objective_plateau_this_row`,
  `adagrad_restart_phase_next_index`,
  `adagrad_restart_phase_next_start_step`,
  `adagrad_restart_terminal_status`, `row_best_nll`, `row_best_step`,
  `save_best_after_row`, `lbfgsb_high_kkt_status`, and
  `lbfgsb_loss_schedule_next_index`.
- Do not change the position of `build_iteration_artifacts()`. Rows must still
  be constructed after all metrics and next-phase/next-schedule signals are
  prepared, and before first-order pending steps are applied.
- Do not change save timing. Pre-step optimizers still save a best checkpoint
  with `optimizer/step_applied = False` and `next_step = step` before the pending
  first-order update. Post-step optimizers still save the best checkpoint after
  transition handling when no transition consumed the row.
- Preserve the existing suppression of `_step_stopping_status()` when projected
  backoff, projected min-LR detection, or bounded high-projected-gradient plateau
  occurred.
- Keep `current_phase`, planning-state sync, transition inputs, finalization
  inputs, and optimizer construction unchanged.

`gpurec/workflow/_loop_policy.py` or `_loop_policies.py` if introduced

- Import only private workflow types and `torch`; do not export the helpers from
  package `__init__` files.
- Keep metric mutation explicit. Each helper may mutate the supplied `metrics`
  dict because the row currently records those same keys, but helper return
  values must carry all control-flow signals.
- Avoid checkpoint or transition imports. This module should not know about
  `IterationTransitionInputs`, `build_iteration_artifacts()`, checkpoint paths,
  or finalization.
- If type cycles appear, use structural parameters instead of importing
  optimize-state classes. For example, pass callable `update_best` owners or use
  small scalar fields rather than broad run-state objects.

`gpurec/workflow/_rows.py`

- No planned behavior change. It must continue to receive the same prepared
  signals and emit the same row and checkpoint-status schemas.
- Keep the checkpoint-status reset rules:
  LBFGSB schedule advance writes the next schedule index and checkpoint
  `stable_loss_steps = 0`; adagrad dynamic phase advance writes the next dynamic
  phase metadata and checkpoint `previous_objective = None` plus
  `stable_loss_steps = 0`.

`gpurec/workflow/_transitions.py`

- No planned behavior change. The extraction should not reorder
  `_classify_iteration_transition()`.
- `adagrad_restart_advance`, `lbfgsb_loss_schedule`,
  `projected_lbfgs_min_lr_reached`, `lbfgsb_retry`, and `step_stopping` must keep
  their current actions, optimizer reset behavior, checkpoint writes, and
  `resume_info` reset/merge behavior.

Tests

- Keep the current workflow-level tests as the primary guard. They cover the
  serialized history rows and checkpoint status, which direct helper tests alone
  would miss.
- Add small helper-level tests only for decision-matrix branches that are hard to
  hit through `OptimizationRunner.run()`. Do not replace the existing workflow
  tests with helper-only tests.

## Behavioral Invariants

General:

- No public config, CLI, result, checkpoint, history row, or artifact schema
  changes.
- Every metric key currently written in the inline block remains present under
  the same conditions and with the same value type. Existing boolean metrics stay
  booleans, and existing counter/index metrics stay floats where rows currently
  write floats.
- Helper return values must not skip row construction. Even terminal decisions
  such as projected min-LR and LBFGSB high-KKT stop still produce a history row
  before transition handling.
- `objective_state.previous_objective` is updated to the current objective after
  stable-loss tracking, before `build_iteration_artifacts()`, so checkpoint
  status and resume state keep the same value.
- `active_objective_scope` continues to select `batch_state.update_best()` and
  suppress best-checkpoint saving for active-batch rows. Non-active scope
  continues to use `objective_state.update_best()` and return the improved flag
  as `save_best_after_row`.

Projected-LBFGS and LBFGSB projected-gradient policy:

- The policy runs only when `phase in {"projected-lbfgs", "lbfgsb"}` and an
  optimizer exists.
- Missing `grad/projected_inf` still behaves as `inf`. Missing accepted metric
  still behaves as accepted.
- `plateau` remains `delta is not None and delta <= loss_change_tol_bits`.
- `high_projected_grad` remains
  `projected_inf_value > config.projected_grad_tol`.
- `bounded_high_projected_plateau` remains gated by
  `config.loss_stop_projected_grad_gate`, high projected gradient, and either a
  plateau or a rejected step.
- Projected-LBFGS LR backoff still happens on high projected gradient plus
  plateau or rejected step, independent of whether the loss-stop projected-grad
  gate is enabled.
- Backoff still uses accepted alpha when `0.0 < accepted_alpha < old_lr`,
  otherwise `old_lr * shrink`, and clamps to
  `config.projected_lbfgs_min_lr`.
- If the computed LR is lower than the old LR, mutate
  `optimizer.param_groups[0]["lr"]` and set
  `optimizer/projected_lbfgs_lr_reduced = True`. Otherwise set
  `optimizer/projected_lbfgs_min_lr_reached = True`.
- Projected backoff and projected min-LR detection both keep
  `stable_loss_steps` at zero for that row and suppress generic
  `_step_stopping_status()`. Projected min-LR still flows through the transition
  layer as `projected_lbfgs_min_lr_reached`.
- For LBFGSB, the helper must continue to write
  `optimizer/lbfgsb_projected_grad_tol`,
  `optimizer/lbfgsb_loss_stop_projected_grad_gate`,
  `optimizer/lbfgsb_high_projected_grad`, and
  `optimizer/lbfgsb_blocked_loss_stop`.

Stable-loss tracking:

- `objective_plateau_this_row` is true only when delta is within tolerance and
  projected-LBFGS did not back off or hit min LR.
- Increment `objective_state.stable_loss_steps` only when
  `objective_plateau_this_row` is true and
  `bounded_high_projected_plateau` is false.
- Reset `stable_loss_steps` to zero for non-plateau rows, projected backoff,
  projected min-LR rows, and bounded high-projected-gradient plateau rows.
- The stable-loss count used by adagrad dynamic phase, LBFGSB high-KKT, LBFGSB
  loss schedule, row construction, and `_step_stopping_status()` must be the
  post-update count from the same row.

Adagrad restart dynamic phase decisions:

- Dynamic decisions run only when dynamic restarts are enabled, an active
  adagrad phase exists, and `adagrad_restart_phase_step` is known.
- `phase_done_by_loss` remains
  `stable_loss_steps >= config.adagrad_restart_phase_loss_patience`.
- `phase_done_by_cap` remains
  `adagrad_restart_phase_step + 1 >= active_phase.phase.steps`.
- Loss patience keeps priority over phase cap when both are true, preserving
  `optimizer/adagrad_restart_phase_complete_reason = "loss_change_patience"`.
- Completed dynamic rows still write
  `optimizer/adagrad_restart_dynamic_phase = True`,
  `optimizer/adagrad_restart_phase_complete = True`, complete reason, and
  `optimizer/adagrad_restart_phase_loss_patience` as a float.
- Non-complete dynamic rows still write
  `optimizer/adagrad_restart_dynamic_phase = True` and
  `optimizer/adagrad_restart_phase_complete = False`.
- Last dynamic adagrad phase in `adagrad-restarts-lbfgsb` advances to next index
  `len(adagrad_restart_specs)` with `next_start_step = step + 1` and row metric
  `optimizer/adagrad_restart_next_phase = "lbfgsb"`.
- Last dynamic adagrad phase in plain `adagrad-restarts` returns terminal
  converged status with reason `adagrad_restart_phase_loss_patience` for
  loss-driven completion or `adagrad_restart_schedule_complete` for cap-driven
  completion.
- Any adagrad phase advance must still cause `_rows.py` to checkpoint the next
  dynamic phase index/start step and reset checkpoint objective tracking, and
  `_transitions.py` to reset optimizer, active phase, active optimizer batch, and
  resume info.

Best update decision:

- Active-objective rows use `BatchRunState.update_best()` and never request a
  best checkpoint from this decision.
- Non-active rows use `ObjectiveState.update_best()` and request a best
  checkpoint exactly when the objective is below
  `best_nll - best_likelihood_min_delta_bits` or no best exists.
- `best_likelihood_min_delta_bits` remains
  `config.best_likelihood_min_delta * active_family_count`.
- `best_nll_bits` and `best_step` written to rows and checkpoint status continue
  to come from the selected objective scope.

LBFGSB high-KKT stop:

- `lbfgsb_state.fallback_used_count` increments exactly once for each LBFGSB row
  with `optimizer/lbfgsb_fallback_used` true.
- Missing `optimizer/lbfgsb_high_kkt_stall_count` still behaves as zero.
- `high_kkt_stop_patience` is still read from
  `config.lbfgsb_high_kkt_stop_patience` and written to metrics as a float.
- `high_kkt_stop_signal` keeps the current two-part rule:
  stall count reaches `2` when configured patience is `0` or `1`, or reaches the
  configured patience otherwise; or stall count reaches configured patience and
  the row used a fallback or exhausted the fallback budget.
- `high_kkt_objective_stalled` remains exactly `objective_plateau_this_row`.
  Large objective improvements must not trigger high-KKT stop, even with high
  stall counts.
- High-KKT stop is ready only when patience is positive, the signal is true, the
  objective stalled, the active loss schedule is in its final phase, and
  `lbfgsb_state.fallback_used_count >= config.lbfgsb_high_kkt_stop_min_fallbacks`.
- With a nonfinal LBFGSB loss schedule, high-KKT stop must wait and allow the
  schedule-advance path to run first.
- The status for a ready high-KKT stop remains
  `{"status": "converged", "reason": "lbfgsb_high_kkt_tiny_progress_patience"}`.
- The row still records
  `optimizer/lbfgsb_fallback_used_count`,
  `optimizer/lbfgsb_high_kkt_stop_patience`,
  `optimizer/lbfgsb_high_kkt_stop_min_fallbacks`,
  `optimizer/lbfgsb_high_kkt_objective_stalled`,
  `optimizer/lbfgsb_high_kkt_final_loss_phase`, and
  `optimizer/lbfgsb_high_kkt_stop_ready`.
- Preserve the current broad metric behavior unless deliberately changed in a
  separate behavior PR: these LBFGSB high-KKT metrics are written by the current
  block even outside LBFGSB phases, with zero/default values.

LBFGSB loss schedule advance:

- Schedule advance runs only for `phase == "lbfgsb"`, a nonempty schedule, no
  high-KKT terminal status, nonzero effective loss patience, stable-loss count at
  least effective patience, and an available next schedule phase.
- Advance rows still write
  `optimizer/lbfgsb_loss_schedule_advance = True`,
  `optimizer/lbfgsb_loss_schedule_next_index`,
  `optimizer/lbfgsb_loss_schedule_next_tol`, and
  `optimizer/lbfgsb_loss_schedule_next_patience`.
- Non-advance LBFGSB rows with a schedule still write
  `optimizer/lbfgsb_loss_schedule_advance = False` and
  `optimizer/lbfgsb_loss_schedule_force_fallback_next = False`.
- When `config.lbfgsb_loss_schedule_force_fallback` is true and optimizer state
  for `model.theta` is a dict, set
  `consecutive_high_kkt_stalls` to at least `2`, write
  `optimizer/lbfgsb_loss_schedule_force_fallback_next = True`, and record the
  previous stall count.
- The transition layer still owns applying the next schedule index and resetting
  `objective_state.stable_loss_steps = 0`. The helper only returns
  `lbfgsb_loss_schedule_next_index`.

Rows, checkpoints, resume, and finalization:

- `build_iteration_artifacts()` must receive identical inputs before and after
  extraction. This is the highest-value gate because it preserves `history.jsonl`
  and checkpoint status.
- Checkpoint status must continue to serialize
  `previous_objective`, `stable_loss_steps`,
  `lbfgsb_fallback_used_count`, `lbfgsb_best_retry_count`,
  `lbfgsb_loss_schedule_index`, and adagrad dynamic phase fields.
- `_resume_state_from_payload()` must continue to normalize and validate those
  status fields without schema changes.
- Finalization must still write `lbfgsb_loss_schedule_index` into final status
  when a schedule is active, and preserve `stable_loss_steps` and LBFGSB counters.
- Best-retry handling in `_transitions.py` must reload objective tracking,
  fallback count, and schedule index from checkpoint status exactly as it does
  today.

## Implementation Sequence

1. Add private result dataclasses and helper functions, either in a new
   `gpurec/workflow/_loop_policy.py` module or in `optimize.py` if imports would
   otherwise cycle.
2. Move the projected-LBFGS/LBFGSB projected-gradient policy first. This is the
   riskiest small piece because it mutates optimizer LR and suppresses stopping.
   Keep its tests green before moving on.
3. Move stable-loss tracking as a narrow helper returning
   `objective_plateau_this_row`, then update `previous_objective` at the same
   point in the loop as today.
4. Move adagrad dynamic phase decisions. Preserve metric writes and next-phase
   variables, then verify dynamic prefix tests before touching LBFGSB logic.
5. Move the best update decision. Keep the two objective scopes explicit so
   active-batch behavior stays visible.
6. Move LBFGSB high-KKT stop. Preserve fallback count mutation and all row
   metrics, including default-valued metrics outside LBFGSB phases.
7. Move LBFGSB loss-schedule advance. Keep force-fallback optimizer-state
   mutation inside this helper and keep schedule application in `_transitions.py`.
8. Run the targeted gates below before any broad cleanup. Only after they pass
   should the implementer reduce local comments or rearrange imports.

## Verification Gates

Run gates in this order after the production-code extraction:

```bash
python -m compileall -q gpurec/workflow
pytest -q tests/unit/test_optimization_workflow.py -k "step_stopping_status or resume_state"
pytest -q tests/unit/test_workflow.py -k "adagrad_restarts_can_advance_flat_phases or adagrad_restarts_lbfgsb_continues_after_prefix or adagrad_restarts_lbfgsb_dynamic_prefix_enters_tail"
pytest -q tests/unit/test_workflow.py -k "projected_lbfgs_reduces_lr or projected_lbfgs_reports_min_lr or lbfgsb_can_stop_on_loss_plateau_without_projected_grad_gate"
pytest -q tests/unit/test_workflow.py -k "lbfgsb_loss_schedule_advances_before_stop or lbfgsb_high_kkt_waits_for_final_loss_phase or lbfgsb_best_retry_reloads_checkpoint_once"
pytest -q tests/unit/test_workflow.py -k "lbfgsb_can_stop_before_second_high_kkt_fallback or lbfgsb_can_stop_on_budget_exhausted_high_kkt_plateau or lbfgsb_high_kkt_waits_for_objective_plateau"
pytest -q tests/unit/test_workflow.py -k "checkpoint_status_defaults or checkpoint_nonnegative_int"
pytest -q tests/unit/test_workflow.py
pytest -q tests/unit/test_optimization_workflow.py tests/unit/test_cli_workflow.py tests/unit/test_workflow_artifacts.py
git diff --check
```

Expected post-extraction inspection gates:

```bash
rg -n "projected_lbfgs_backoff|bounded_high_projected_plateau|adagrad_restart_phase_next_index|lbfgsb_high_kkt_status|lbfgsb_loss_schedule_next_index" gpurec/workflow/optimize.py
rg -n "_loop_policy|_apply_projected_loss_stop_policy|_decide_adagrad_restart_dynamic_phase|_decide_lbfgsb_high_kkt_stop|_decide_lbfgsb_loss_schedule_advance" gpurec/workflow tests
rg -n "lbfgsb_loss_schedule_index|adagrad_restart_dynamic_phase_index|stable_loss_steps|previous_objective" gpurec/workflow/_rows.py gpurec/workflow/_transitions.py gpurec/workflow/_runtime_state.py gpurec/workflow/_finalization.py
```

Expected results:

- `optimize.py` still contains the orchestration variables and transition inputs,
  but no long inline policy blocks for the extracted decisions.
- The helper module or helper section contains private functions only. No public
  exports are added.
- `_rows.py`, `_transitions.py`, `_runtime_state.py`, and `_finalization.py`
  retain the same checkpoint and resume field names.
- The targeted tests continue to assert row fields such as
  `optimizer/projected_lbfgs_lr_reduced`,
  `optimizer/projected_lbfgs_min_lr_reached`,
  `optimizer/adagrad_restart_phase_complete`,
  `optimizer/adagrad_restart_next_phase`,
  `optimizer/lbfgsb_loss_schedule_advance`,
  `optimizer/lbfgsb_loss_schedule_index`,
  `optimizer/lbfgsb_high_kkt_stop_ready`, `stable_loss_steps`, and terminal
  result reasons.

## Review Checklist For The Implementer

- Does every helper have a single policy responsibility and an explicit return
  object for control-flow signals?
- Are metric mutations still visible in one place per policy, with exact key
  names preserved?
- Did any helper accidentally call checkpoint, transition, finalization, logging,
  model selection, or solver-stage APIs?
- Are `stable_loss_steps` and `previous_objective` updated before row building
  exactly as before?
- Are adagrad and LBFGSB schedule transitions still represented as next-index
  signals consumed by `_rows.py` and `_transitions.py`, rather than being applied
  early?
- Does projected-LBFGS backoff still prevent generic loss-stop convergence, and
  does min-LR still terminate through transition classification?
- Does LBFGSB high-KKT stop still wait for both objective plateau and final loss
  schedule phase?
- Do best checkpoints still use the same row, `step`, `next_step`, optimizer, and
  optimizer-phase values as before?
- Does resume from best checkpoint restore LBFGSB fallback count, retry count,
  and loss schedule index without helper involvement?

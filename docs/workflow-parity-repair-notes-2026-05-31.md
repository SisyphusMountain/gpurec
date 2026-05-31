# Workflow parity repair notes, 2026-05-31

## Scope and commands

Worker scope was limited to tests plus this note. I did not edit production workflow files.

Commands run:

```bash
git status --short
git rev-parse --short HEAD
rg -n "projected_lbfgs|lbfgsb|batched_lbfgs|_restore_optimizer_state|_evaluate_active_genewise_vector_and_grad|prepare_initial_optimization_plan" tests/unit/test_workflow.py gpurec docs
python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
python -m pytest -q tests/unit/test_workflow.py::test_batched_lbfgs_active_batch_closure_zeros_inactive_rows
git diff -- gpurec/workflow/optimize.py gpurec/workflow/_step_plan.py gpurec/workflow/_transitions.py
python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
python -m pytest -q tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_best_retry_reloads_checkpoint_once tests/unit/test_workflow.py::test_optimization_runner_batched_lbfgs_advances_resident_batches
python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
```

I also ran short `PYTHONPATH=. python - <<'PY' ...` diagnostic snippets to dump history rows for the dynamic-prefix, batched-LBFGS, and best-retry scenarios.

HEAD at start was `79cc3e9`. The first status check showed no tracked local changes, only unrelated untracked files. During this worker turn, concurrent production edits appeared in `gpurec/workflow/_step_plan.py`, `gpurec/workflow/optimize.py`, and `gpurec/workflow/_transitions.py`; I left them intact.

## Initial failures

Initial selector:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
```

Result: 11 failed, 11 passed, 706 deselected.

Exact failures from that run:

- `test_optimization_runner_adagrad_restarts_lbfgsb_dynamic_prefix_enters_tail`
- `test_optimization_runner_lbfgsb_can_stop_on_loss_plateau_without_projected_grad_gate`
- `test_optimization_runner_lbfgsb_loss_schedule_advances_before_stop`
- `test_optimization_runner_lbfgsb_high_kkt_waits_for_final_loss_phase`
- `test_optimization_runner_lbfgsb_best_retry_reloads_checkpoint_once`
- `test_optimization_runner_lbfgsb_can_stop_before_second_high_kkt_fallback`
- `test_optimization_runner_lbfgsb_can_stop_on_budget_exhausted_high_kkt_plateau`
- `test_optimization_runner_projected_lbfgs_reduces_lr_instead_of_stopping_on_large_projected_gradient`
- `test_batched_lbfgs_active_batch_closure_zeros_inactive_rows`
- `test_optimization_runner_batched_lbfgs_advances_resident_batches`
- `test_optimization_runner_batched_lbfgs_resume_restores_state`

The private API failures were:

- `test_batched_lbfgs_active_batch_closure_zeros_inactive_rows`: `OptimizationRunner` lacked `_evaluate_active_genewise_vector_and_grad`.
- `test_optimization_runner_batched_lbfgs_resume_restores_state`: `_step_plan.prepare_initial_optimization_plan` called `OptimizationRunner._restore_optimizer_state` with positional `current_phase` and `checkpoint_phase`, but the method accepts them keyword-only.

## Concurrent production repairs observed

Production edits from another worker were applied while this note was being prepared. The current worktree now includes:

- `gpurec/workflow/_step_plan.py`: `prepare_initial_optimization_plan` now calls `restore_optimizer_state(..., current_phase=..., checkpoint_phase=...)`.
- `gpurec/workflow/optimize.py`: `OptimizationRunner` now exposes `_evaluate_active_genewise_vector_and_grad` and `_evaluate_active_genewise_vector_grad_at_current_theta`.
- `gpurec/workflow/optimize.py` and `gpurec/workflow/_step_plan.py`: optimizer construction now goes through an injected runner hook, preserving `_make_optimizer` overrides in tests.
- `gpurec/workflow/optimize.py`: the planning state is refreshed from current objective and LBFGS-B counters before applying iteration transitions.
- `gpurec/workflow/_transitions.py`: high-KKT L-BFGS-B stop classification now allows a configured best-checkpoint retry first, and active-batch warmup transitions use the active optimizer warmup setting.

After those edits, the active-batch closure test passes:

```bash
python -m pytest -q tests/unit/test_workflow.py::test_batched_lbfgs_active_batch_closure_zeros_inactive_rows
```

An intermediate selector run reported 9 failures after only the first two private API repairs, then 2 failures after the planning and optimizer-hook repairs. After the latest concurrent transition patch, the full selector passes:

```bash
python -m pytest -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
```

Current result: 22 passed, 706 deselected.

## Recommended private API patches

For `_restore_optimizer_state`, the smallest production repair is the direction already taken: call it with keywords from `gpurec/workflow/_step_plan.py`. The type hint should also match the keyword call. A small `Protocol` or `Callable[..., dict[str, Any]]` would be more accurate than the current two-argument `Callable[[torch.optim.Optimizer, Any], dict[str, Any]]`.

For `_evaluate_active_genewise_vector_and_grad`, restore a thin `OptimizationRunner` proxy:

```python
def _evaluate_active_genewise_vector_and_grad(
    self,
    model: GeneReconModel,
    *,
    solver_stage: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    return self.evaluation.evaluate_active_genewise_vector_and_grad(
        model,
        solver_stage=solver_stage,
    )
```

The current worktree already has this shape.

## Row and metric failure cause

The row-count and metric failures were caused by transition-policy extraction and a private optimizer-factory hook regression, not by resume planning and not by stale expectations.

The common transition symptom before the concurrent planning-state repair was that ordinary rows did not carry updated objective tracking into the next planning snapshot. Actual histories at that point showed `delta_likelihood_bits` staying `None`, `previous_objective` staying `None`, and `stable_loss_steps` staying `0` across repeated rows. That prevented:

- dynamic `adagrad-restarts-lbfgsb` prefix completion from entering the `lbfgsb` tail;
- projected-LBFGS and L-BFGS-B plateau rows from reporting `delta_likelihood_bits == 0.0`;
- L-BFGS-B loss schedules and high-KKT stops from firing at the expected row;
- batched L-BFGS warmup/full and next-batch transitions from advancing.

Suspect functions at the failing point:

- `gpurec/workflow/_transitions.py::execute_iteration_post_step_transition`, especially the no-action return around the `step_status is None` branch.
- `gpurec/workflow/optimize.py::_OptimizationRunState.apply_step_plan`, which overwrites `objective_state.previous_objective` and `stable_loss_steps` from `_StepPlanningState` every iteration.
- `gpurec/workflow/_step_plan.py::select_step_optimization_plan`, which correctly returns the planning values it receives; the stale source is the transition result.

Recommended patch shape:

- Refresh the planning state before it can be consumed by the next iteration. The concurrent production patch does this in `OptimizationRunner.run()` before `apply_iteration_transition`.
- An equivalent repair would be to update the no-op post-step transition return with `planning_state=replace(planning_state, previous_objective=objective_state.previous_objective, stable_loss_steps=objective_state.stable_loss_steps, lbfgsb_fallback_used_count=lbfgsb_state.fallback_used_count, active_batch_index=batch_state.active_index, active_optimizer_batch_index=batch_state.optimizer_batch_index, active_adagrad_restart_phase_index=restart_state.active_phase_index)`.
- Audit transition returns that can continue without changing phase and make sure they carry the same current objective and LBFGS-B counters.

The scripted L-BFGS-B tests also revealed that optimizer construction no longer honored `OptimizationRunner._make_optimizer` overrides. Nested test runners define `_make_optimizer`, but planning used module-level `_make_optimizer` directly in `gpurec/workflow/_step_plan.py`, so those tests ran the real optimizer and missed scripted metrics such as `last_fallback_used`.

Suspect functions:

- `gpurec/workflow/_step_plan.py::prepare_initial_optimization_plan`
- `gpurec/workflow/_step_plan.py::select_step_optimization_plan`
- transition retry paths that receive `make_optimizer_fn`

Recommended patch shape, now reflected in the concurrent worktree:

- Restore `OptimizationRunner._make_optimizer(self, model, phase)` as the private factory hook.
- Pass that hook into step planning, for example via `_StepPlanningContext`, and replace direct `_make_optimizer(config, model, phase)` calls with the injected hook.
- In transition contexts, pass a wrapper around `self._make_optimizer` instead of the module-level factory so best-retry and phase-reset paths preserve the same hook.

## Resume planning assessment

Resume planning was implicated only by the initial `_restore_optimizer_state` signature mismatch. After the keyword-call repair, `test_optimization_runner_batched_lbfgs_resume_restores_state` passed in the full selector. The row-count and metric failures were transition state synchronization plus optimizer factory private-hook parity. With the current concurrent production changes, the requested selector passes.

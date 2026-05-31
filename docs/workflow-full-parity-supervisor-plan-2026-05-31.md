# Workflow Full Parity Supervisor Plan, 2026-05-31

Scope: documentation-only supervisor plan for the next full
`tests/unit/test_workflow.py` parity slice after commit `0c93622` on branch
`production`. This pass inspected workflow/API tests and implementation files,
but did not edit production code under `gpurec/api` or `gpurec/workflow`.

## Current Evidence

Commands run from `/home/enzo/Documents/git/gpurec/gpurec`:

```bash
git status --short --branch
git show --stat --oneline --decorate -1 0c93622
pytest -q tests/unit/test_workflow.py
rg -n "ThreadPoolExecutor|_prefetch_closed|def clear|def close|_prefetch|resident cache" gpurec/api/model.py gpurec/api/_resident_cache.py tests/unit/test_workflow.py
rg -n "_final_iteration_check_iters|_projected_grad_inf|_evaluate_and_backward|_evaluate_genewise_vector_and_grad|_evaluate_genewise_vector_and_grad_with_memory_fallback|_evaluate_final_iteration_check|_cache_active_batch_final_result|next_fd_state|next_fd_newton_hessian_state" gpurec/workflow tests/unit/test_workflow.py
```

Observed state:

- `HEAD` is `0c93622 (production) Repair bounded workflow parity state tracking`.
- The branch is ahead of `origin/production` and the worktree already has
  unrelated untracked files such as `.tmp_wave_backward_pipeline.dot`, `plots/`,
  several `tmp_verify_global*` directories, and script/readme artifacts.
- `pytest -q tests/unit/test_workflow.py` collected 728 tests and reported
  `38 failed, 690 passed in 10.53s`.
- The failures are now outside the bounded projected-LBFGS/L-BFGS-B/batched
  parity slice repaired by `0c93622`.

## Architecture Owners

### Owner A: API Resident Cache Compatibility

Failures: 5.

- `test_close_prevents_later_prefetch_restart`
- `test_close_tolerates_partially_initialized_model`
- `test_clear_batched_resident_does_not_materialize_missing_active_batch`
- `test_clear_batched_resident_clears_existing_active_warm_state`
- `test_close_shuts_down_executor_without_batch_lock`

Likely cause: resident-batch state moved from legacy private fields on
`GeneReconModel` (`_batch_statics`, `_batch_futures`, `_prefetch_executor`,
`_prefetch_closed`, `_batch_lock`) into `ResidentBatchCache`. The tests still
construct partially initialized models with `GeneReconModel.__new__()` and patch
`gpurec.api.model.ThreadPoolExecutor`, but the executor owner is now
`gpurec/api/_resident_cache.py`. `GeneReconModel.clear()` and `close()` now
require `_resident_cache` for batched resident models, so the old idempotent
partial-init behavior is gone.

Lowest-risk repair:

- Keep `ResidentBatchCache` as the only real cache owner. Do not restore a
  second long-lived cache state machine in `api/model.py`.
- Add a narrow compatibility path for partial-init/legacy private-field tests,
  or migrate those tests to construct a `ResidentBatchCache` directly plus one
  model-level smoke test for `close()` idempotence.
- If production compatibility shims are kept, centralize them behind one helper
  such as "legacy resident cache fields present" and make them no-ops for normal
  initialized models.
- Preserve these behavioral invariants: `close()` is idempotent before
  `__init__`; `close()` cancels pending prefetch futures without requiring
  `_batch_lock`; `clear()` must not materialize a missing active batch; clearing
  an existing active batch drops `warm_E` and Pi-adjoint runtime caches.

### Owner B: Workflow Runner Facade And Evaluation Delegates

Failures: 13 visible now, plus one latent hessian-SGD test dependency.

- Missing `_final_iteration_check_iters`.
- Missing `_projected_grad_inf`.
- Missing `_evaluate_and_backward`.
- Missing `_evaluate_genewise_vector_and_grad`.
- Missing `_evaluate_genewise_vector_and_grad_with_memory_fallback`.
- Missing `_evaluate_final_iteration_check`.
- Latent after hessian callback repair: tests call `runner._active_batch_metrics`.

Likely cause: logic was extracted into `EvaluationOps`,
`SolverStageController`, and `_finalization.py`, but `OptimizationRunner`
stopped exposing the private compatibility facade used by tests and subclasses.
This is an API-shape regression, not a reason to move logic back into
`optimize.py`.

Lowest-risk repair:

- Add thin `OptimizationRunner` delegates only:
  `_final_iteration_check_iters -> self.solver_stage.final_iteration_check_iters`.
  `_projected_grad_inf -> self.evaluation.projected_grad_inf`.
  `_evaluate_and_backward -> self.evaluation.evaluate_and_backward`.
  `_evaluate_genewise_vector_and_grad -> self.evaluation.evaluate_genewise_vector_and_grad`.
  `_evaluate_genewise_vector_and_grad_with_memory_fallback -> self.evaluation.evaluate_genewise_vector_and_grad_with_memory_fallback`.
  `_active_batch_metrics -> self.evaluation.active_batch_metrics`.
  `_evaluate_final_iteration_check -> _finalization._evaluate_final_iteration_check(...)`.
- Keep the helper implementations in their extracted modules. The runner should
  remain a compatibility facade and orchestration boundary, not a duplicate
  evaluation implementation.
- Where step execution needs overridable behavior, pass callbacks from the
  runner into context objects rather than importing module helpers directly.

### Owner C: Hessian-SGD Step Execution And Transition Contracts

Failures: 18.

Main symptoms:

- 14 tests crash with `UnboundLocalError: local variable 'next_fd_state'
  referenced before assignment` from `gpurec/workflow/_step_execution.py`.
- 2 large-batch hessian-SGD tests record no calls to their patched
  `runner._active_fd_newton_step`, because `execute_optimization_step()` calls
  the module helper directly.
- `test_hessian_sgd_large_batch_warmup_plateau_skips_full_solver` fails with
  `_cache_active_batch_final_result() takes 2 positional arguments but 6 were
  given`.
- `test_hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full`
  records one extra warmup row.

Likely causes:

- In `_step_execution.py`, the `_active_fd_newton_step(...)` call is nested
  under the large-clade refresh override condition. Normal hessian-SGD paths do
  not assign `next_fd_state`, then unconditionally copy it into
  `next_fd_newton_hessian_state`.
- Step execution bypasses the runner private hook. Existing test subclasses
  patch `runner._active_fd_newton_step`, but the extracted executor invokes the
  module-level helper.
- `_transitions.py` calls `cache_active_batch_final_result` positionally while
  `OptimizationRunner._cache_active_batch_final_result` is keyword-only after
  `model`.
- The large-clade warmup-skip transition is continuing one row too long or
  caching/evaluating a noncanonical full-stage result.

Lowest-risk repair:

- Fix the hessian-SGD branch shape first: compute the refresh-step override
  conditionally, but execute one FD-Newton/hessian-SGD step for every non-Adam
  warmup path. Initialize and return `next_fd_newton_hessian_state` from that
  call.
- Add an `active_fd_newton_step` callback to `_StepExecutionContext`, supplied
  by `OptimizationRunner._active_fd_newton_step`. This preserves subclass/test
  override parity without moving FD-Newton logic back into `optimize.py`.
- Normalize callback signatures with small type aliases or protocols. Internal
  callsites should use keywords for `_cache_active_batch_final_result`, or pass
  an adapter from `optimize.py` that accepts the transition module's positional
  shape.
- Repair the warmup-skip transition after the core hessian step runs. The
  expected behavior is two warmup rows, convergence by loss-change patience, and
  cached final active-batch output only when the cached result is canonical for
  final artifacts.

### Owner D: Finalization Failed-Eval Status

Failures: 2.

- `test_optimization_runner_marks_nonfinite_final_evaluation_failed`
- `test_optimization_runner_skips_per_family_artifact_after_failed_final_genewise_eval`

Likely cause: `_finalization.finalize_optimization()` detects
`final_eval_failed` and writes a failed final-eval row, but it builds
`final_status` from the incoming loop status. When the loop status is
`not_converged/max_steps`, the summary and `OptimizationResult` keep
`not_converged` instead of upgrading to
`failed/nonfinite_objective_or_gradient`.

Lowest-risk repair:

- When `final_eval_failed` is true, set the final status payload to
  `{"status": "failed", "reason": "nonfinite_objective_or_gradient"}` before
  checkpoint, summary, and artifact decisions.
- Keep `sampling_checkpoint=None`, skip per-family likelihood artifacts, remove
  stale per-family output, and write nullable summary metrics for failed final
  gradient/projected-gradient values. `OptimizationResult.final_grad_inf` may
  still map missing/nonfinite summary values to `math.inf` through
  `_result.py`.

## Repair Sequence

1. Restore runner facade delegates and callback injection.
   This is the smallest workflow-surface repair and removes the large set of
   `AttributeError` failures without reintroducing bloat. Include
   `_active_batch_metrics` before hessian fake-step tests start reaching it.

2. Fix hessian-SGD step execution.
   Repair the `next_fd_state` branch and route hessian-SGD through the runner
   callback. This should collapse most of the hessian group before touching
   transition policy.

3. Normalize transition callback contracts.
   Fix `_cache_active_batch_final_result` invocation and keep callback types in
   `_transitions.py`, `_step_execution.py`, and `optimize.py` aligned. Prefer
   adapters/protocols over broad `Callable[..., Any]` drift where practical.

4. Repair hessian-SGD warmup-skip policy.
   Once execution works, tune only the large-clade warmup skip and canonical
   cache behavior. Avoid rewriting the broader transition state machine.

5. Restore finalization failed-status propagation.
   This is isolated and should be done after the runner facade exists because
   nearby final-check tests depend on those delegates.

6. Address API resident-cache compatibility.
   Keep this isolated from workflow changes. Prefer tests aimed at
   `ResidentBatchCache` plus minimal model idempotence shims, or a single
   compatibility adapter if current tests must remain unchanged.

## Bloat Control Rules

- Do not move evaluation, FD-Newton, optimizer factory, or final-check logic
  back into `optimize.py`.
- Do not re-create a parallel resident-batch cache in `api/model.py`; use
  `ResidentBatchCache` as the owner and keep any legacy field handling narrow.
- Use runner facade methods as compatibility shims only.
- Use explicit context callbacks for override points that tests and subclasses
  already depend on: optimizer creation, FD-Newton stepping, final-batch cache,
  final-check/evaluation delegates.
- Defer larger refactors until `tests/unit/test_workflow.py` is green:
  transition dataclass consolidation, further policy-module extraction, and
  test-suite repartitioning.

## Verification Gates

Run gates in this order:

```bash
pytest -q tests/unit/test_workflow.py -k "adagrad_restarts_accepts_split_solver_budgets or projected_gradient_uses_projection_mapping or optimizer_scalar_eval or genewise_vector_eval or final_genewise_eval or final_iteration_check"
pytest -q tests/unit/test_workflow.py -k "hessian_sgd"
pytest -q tests/unit/test_workflow.py -k "nonfinite_final_evaluation or failed_final_genewise_eval"
pytest -q tests/unit/test_workflow.py -k "close_prevents_later_prefetch_restart or close_tolerates_partially_initialized_model or clear_batched_resident or close_shuts_down_executor"
pytest -q tests/unit/test_workflow.py -k "projected_lbfgs or lbfgsb or batched_lbfgs"
pytest -q tests/unit/test_workflow.py
```

Broader regression gates after the full workflow file is green:

```bash
pytest -q tests/unit/test_optimization_workflow.py tests/unit/test_cli_workflow.py tests/unit/test_workflow_artifacts.py
pytest -q tests/unit
```

Expected completion state: `tests/unit/test_workflow.py` reports 728 passed, the
previous bounded parity selector remains green, failed final evaluations produce
failed summaries/checkpoints, and workflow/API production files have less
duplicated logic than before this refactor slice.

Main-agent follow-up: the integrated repair later reached `728 passed` for
`tests/unit/test_workflow.py` and `268 passed` for
`tests/unit/test_optimization_workflow.py`,
`tests/unit/test_cli_workflow.py`, and
`tests/unit/test_workflow_artifacts.py`.

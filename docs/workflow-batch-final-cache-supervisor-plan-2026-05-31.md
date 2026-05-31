# Workflow Batch Final Cache Supervisor Plan, 2026-05-31

Scope: documentation-only supervisor plan for consolidating the active-batch
final-cache plumbing on branch `production` at `493dd15`. This pass inspected
the workflow implementation and tests, but did not edit production code under
`gpurec/workflow`.

## Command Evidence

Commands run from `/home/enzo/Documents/git/gpurec/gpurec`:

```bash
pwd
git status --short --branch
git rev-parse --short HEAD
git diff --name-only
rg -n "batch_final_loss_cache|batch_final_grad_cache|batch_final_cache_ready|_cache_active_batch_final_result|IterationTransitionContext" gpurec docs tests
rg -n "batch_final_loss_cache|batch_final_grad_cache|batch_final_cache_ready" gpurec/workflow/optimize.py gpurec/workflow/_transitions.py gpurec/workflow/_finalization.py
nl -ba gpurec/workflow/optimize.py | sed -n '250,470p'
nl -ba gpurec/workflow/optimize.py | sed -n '740,790p'
nl -ba gpurec/workflow/optimize.py | sed -n '1060,1165p'
nl -ba gpurec/workflow/optimize.py | sed -n '1210,1270p'
nl -ba gpurec/workflow/optimize.py | sed -n '1880,1920p'
nl -ba gpurec/workflow/_transitions.py | sed -n '240,405p'
nl -ba gpurec/workflow/_transitions.py | sed -n '470,585p'
nl -ba gpurec/workflow/_transitions.py | sed -n '900,1065p'
nl -ba gpurec/workflow/_transitions.py | sed -n '1245,1410p'
nl -ba gpurec/workflow/_finalization.py | sed -n '1,125p'
nl -ba tests/unit/test_workflow.py | sed -n '9960,10060p'
nl -ba gpurec/workflow/_step_execution.py | sed -n '90,125p'
nl -ba gpurec/workflow/_step_execution.py | sed -n '650,765p'
```

Observed state:

- `git status --short --branch` reports `## production...origin/production [ahead 89]`
  plus unrelated untracked files and directories.
- `git diff --name-only` was empty before this documentation edit.
- The three cache tensors appear in `gpurec/workflow/optimize.py`,
  `gpurec/workflow/_transitions.py`, and `gpurec/workflow/_finalization.py`.
- `_OptimizationRunState` owns `batch_final_loss_cache`,
  `batch_final_grad_cache`, and `batch_final_cache_ready`, then threads all
  three through `IterationTransitionContext` and finalization inputs.
- `_cache_active_batch_final_result` writes active-batch loss and gradient rows
  and marks readiness bits true; adaptive rebatch transition handling directly
  invalidates readiness bits for replanned indices.
- Finalization consumes the cache only when all three tensors are present and
  every readiness bit is true, then emits
  `optimizer/final_eval_source = cached_active_batches` with zero final closure
  evals.

## Consolidation Goal

Replace the three separately threaded active-batch final-cache tensors with one
small internal object, preserving behavior and keeping cache policy local. The
object should hide allocation, readiness checks, active-batch writes, adaptive
rebatch invalidation, and finalization extraction behind named methods.

Recommended shape:

```python
@dataclass
class _ActiveBatchFinalCache:
    loss: torch.Tensor
    grad: torch.Tensor
    ready: torch.Tensor

    @classmethod
    def allocate(cls, model: GeneReconModel) -> "_ActiveBatchFinalCache": ...
    def cache_active_result(self, model: GeneReconModel, active_idx: torch.Tensor, loss_vec: torch.Tensor) -> None: ...
    def invalidate(self, indices: Sequence[int] | torch.Tensor | None) -> None: ...
    def is_complete(self) -> bool: ...
    def final_result(self) -> tuple[torch.Tensor, torch.Tensor] | None: ...
```

Keep this object internal to workflow. Prefer placing it near current run-state
ownership in `gpurec/workflow/optimize.py` for the first pass unless reuse pressure
justifies a tiny private module such as `gpurec/workflow/_batch_final_cache.py`.
The slice is about reducing argument bloat, not creating a new public API.

## File Ownership

`gpurec/workflow/optimize.py`

- Replace `_OptimizationRunState.batch_final_loss_cache`,
  `.batch_final_grad_cache`, and `.batch_final_cache_ready` with one
  `batch_final_cache: _ActiveBatchFinalCache | None`.
- Replace the allocation block around active-batch optimizer setup with
  `_ActiveBatchFinalCache.allocate(model)`.
- Simplify `_cache_active_batch_final_result` so callers pass only
  `model` and `loss_vec`, or remove the runner helper and call
  `run_state.batch_final_cache.cache_active_result(...)` at the two current
  cache sites.
- Thread only `batch_final_cache` into transition context and finalization.

`gpurec/workflow/_transitions.py`

- Replace `IterationTransitionContext`'s three tensor fields with one optional
  cache object.
- Change `_execute_iteration_full_transition`,
  `execute_iteration_post_step_transition`, and `execute_iteration_transition`
  signatures so transition code receives the cache object, not separate tensors.
- Move adaptive rebatch readiness invalidation from direct
  `batch_final_cache_ready.index_fill_` mutation to `batch_final_cache.invalidate(...)`.
- Keep the skipped-full hessian-SGD cache path behavior unchanged: cache only
  canonical full-solver active-batch results.

`gpurec/workflow/_finalization.py`

- Replace `_FinalizationInputs`'s three tensor fields with
  `batch_final_cache: _ActiveBatchFinalCache | None`.
- Use `cache.final_result()` or equivalent to obtain detached clones only when
  the cache is complete.
- Preserve the current fallback path: incomplete or missing cache performs the
  normal final evaluation and should not emit `cached_active_batches`.

`tests/unit/test_workflow.py`

- Keep existing behavioral tests for cached active batches:
  `test_hessian_sgd_large_batch_warmup_plateau_skips_full_solver` expects the
  cached final-eval source, while
  `test_hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full`
  expects no cached source.
- Add or adjust focused coverage for adaptive rebatch invalidation if existing
  tests do not explicitly catch readiness reset through the new object.

## Behavioral Invariants

- Cache allocation remains limited to batchwise active optimizers; non-batchwise
  routes keep `batch_final_cache is None`.
- Loss cache shape remains `(model.n_families,)`, device and dtype matching
  `model.theta`; gradient cache remains `torch.empty_like(model.theta)`;
  readiness remains a boolean tensor on the theta device.
- `cache_active_result` writes only active-family rows from the current full or
  canonical full-equivalent active-batch result, using detached values.
- If `model.theta.grad is None`, readiness must not silently mark a batch
  complete unless that matches current behavior intentionally. The current code
  marks ready even when grad is missing, so changing this requires a separate
  behavior decision and tests.
- Adaptive rebatch invalidates readiness for replanned families and does not
  clear unrelated completed families.
- Finalization uses the cached path only when every readiness bit is true; any
  incomplete cache falls back to full final evaluation.
- The cached finalization path still clones detached tensors before assigning
  `model.theta.grad`, reports `optimizer/final_eval_source =
  cached_active_batches`, and records `final_closure_evals = 0`.
- Checkpoint status, optimizer state, solver-stage transitions, and history row
  semantics must not change.

## Implementation Sequence

1. Introduce `_ActiveBatchFinalCache` with allocation, write, invalidate,
   completeness, and final-result methods. Keep it private and dependency-light.
2. Change `_OptimizationRunState` to hold the object and update allocation,
   transition-context construction, context sync, loop cache sites, and
   finalization input creation.
3. Update transition dataclasses and helper signatures to accept the cache
   object. Remove separate loss/grad/ready parameters from internal transition
   calls in one mechanical pass.
4. Replace direct transition readiness mutation with `cache.invalidate(...)`.
5. Update finalization to consume `cache.final_result()` and preserve fallback
   behavior when the cache is missing or incomplete.
6. Run the targeted gates first, then the broader workflow tests.

## Verification Gates

Run gates in this order after the production-code slice:

```bash
python -m compileall -q gpurec/workflow
pytest -q tests/unit/test_workflow.py -k "cached_active_batches or warmup_skip_does_not_cache_noncanonical_full"
pytest -q tests/unit/test_workflow.py -k "hessian_sgd_large_batch_warmup"
pytest -q tests/unit/test_workflow.py -k "adaptive_rebatch"
pytest -q tests/unit/test_workflow.py -k "final_eval_source or final_genewise_eval or nonfinite_final_evaluation"
pytest -q tests/unit/test_workflow.py
pytest -q tests/unit/test_optimization_workflow.py tests/unit/test_cli_workflow.py tests/unit/test_workflow_artifacts.py
```

Expected completion state:

- `rg -n "batch_final_loss_cache|batch_final_grad_cache|batch_final_cache_ready" gpurec/workflow`
  returns no production-code hits, except acceptable compatibility notes if any
  were intentionally left in comments or tests.
- `rg -n "batch_final_cache" gpurec/workflow` shows one object threaded through
  run state, transitions, and finalization rather than three parallel tensors.
- The cached and noncached hessian-SGD warmup-skip tests keep their current
  expectations.
- No public config, checkpoint, result, or artifact schema changes are required.

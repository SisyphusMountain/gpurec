# Refactor Next Slice Plan, 2026-05-31

This is a documentation-only supervisor plan for the next refactor slice on the
dirty `production` worktree. Source code was not edited for this pass.
Future file paths named in this plan are optional proposed targets until a
slice creates them.

The current uncommitted split has already reduced the two original largest
files substantially, but the remaining large production groups are still:

| Group | Current size | Current shape |
| --- | ---: | --- |
| `gpurec/workflow/optimize.py` | 1,751 lines | public runner, loop orchestration, resume setup, status/metric decisions |
| `gpurec/workflow/_transitions.py` | 1,415 lines | extracted transition policy, but now a large policy hub |
| `gpurec/api/model.py` | 1,362 lines | public `GeneReconModel` facade plus constructor, resident-batch management, full-streaming methods |
| `gpurec/api/_batch_specs.py` | 535 lines | batch normalization, schedule stats, Python/Rust resident batch spec building |
| `gpurec/optimization/lbfgsb.py` | 1,593 lines | scalar bounded L-BFGS-B, line search, fallback competition, state telemetry |
| `gpurec/optimization/batched_lbfgs.py` | 993 lines | row-wise L-BFGS, duplicated projection/evaluation scaffolding, vectorized strong-Wolfe |
| `gpurec/optimization/projected_lbfgs.py` | 434 lines | simpler bounded projected L-BFGS with duplicated scalar helpers |

The target is code reduction through fewer duplicate mechanics and clearer
ownership boundaries, not code compression. Keep package layering intact:

- `core` remains the implementation namespace for preprocessing, scheduling,
  kernels, likelihood math, and tensor contracts.
- `api` may import `core`, but should not import `workflow` or
  `optimization`.
- `optimization` stays a pure PyTorch optimizer package and must not import
  `api`, `workflow`, or `core`.
- `workflow` owns product orchestration and may import `api` plus
  `optimization` through the existing optimizer factory boundary.

Do not edit `docs/optimization-workflow-call-graph.md` in this slice; the main
agent owns call graph updates.

## Highest-Priority Slice

Start with the optimization package. It has the best reduction-to-risk ratio:
there is obvious duplicated bounded-optimizer plumbing across three files, the
boundary is already independent, and the test suite already has focused
coverage for `LBFGSB`, `BatchedLBFGS`, `ProjectedLBFGS`, and Schilling
conformance data.

Do not start by splitting `workflow/_transitions.py` again. The current
workflow split has already moved thousands of lines out of `optimize.py`, and
another policy move before parity tests settle would risk spreading the same
state machine over more files. Treat workflow as the second slice after the
optimizer helper extraction is green.

## Slice 1: Shared Bounded Optimizer Primitives

### Goal

Reduce repeated code in:

- `gpurec/optimization/lbfgsb.py`
- `gpurec/optimization/batched_lbfgs.py`
- `gpurec/optimization/projected_lbfgs.py`

without changing optimizer public classes, state keys, accepted-step behavior,
or `gpurec.optimization.__all__`.

Expected net reduction: about 200-350 package lines after helper code is added.
The large win is not only line count: future fallback or bound fixes should
land in one place instead of three.

### Proposed Abstractions

Add `gpurec/optimization/_bounds.py`:

- `bound_for_flat(bound, flat, parameter_shape, *, broadcast_to_flat=False)`
- `bounds_for_flat(flat, lower_bound, upper_bound, parameter_shape, *, broadcast_to_flat=False)`
- `project_flat(flat, lower_bound, upper_bound, parameter_shape, *, broadcast_to_flat=False)`
- `projected_gradient(flat, grad, lower_bound, upper_bound, parameter_shape, *, broadcast_to_flat=False)`
- `feasible_direction(flat, direction, lower_bound, upper_bound, parameter_shape, *, broadcast_to_flat=False)`

The `broadcast_to_flat` option is important because the scalar optimizers
reshape bounds against the original parameter shape, while `BatchedLBFGS`
sometimes needs a `[B, N]` flat shape fallback. Preserve those semantics
explicitly instead of forcing one broadcast behavior.

Add `gpurec/optimization/_closures.py`:

- `scalar_loss_tensor(loss, owner)`
- `loss_vector_tensor(loss, batch_size, owner)`
- `flat_grad(param, flat_like, owner, *, row_batch_size=None)`
- `evaluate_scalar_with_grad(param, flat_like, closure, owner)`
- `evaluate_scalar_loss(closure, loss_closure, owner)`
- `evaluate_vector_with_grad(param, flat_like, closure, batch_size, owner)`
- `evaluate_vector_loss(closure, loss_closure, batch_size, owner)`

These helpers should keep the current error messages owner-specific
(`LBFGSB`, `ProjectedLBFGS`, `BatchedLBFGS`) and preserve sparse-gradient
densification plus complex-gradient rejection.

Add `gpurec/optimization/_armijo.py`:

- `armijo_accepts(trial_loss, loss, trial_gtd, c1)`
- `armijo_required_decrease(loss, trial_gtd, c1)`

Use this only for scalar Armijo behavior shared by `LBFGSB` and
`ProjectedLBFGS`. Keep `BatchedLBFGS._strong_wolfe()` in
`batched_lbfgs.py` for now; it is large but high-risk and already has a direct
PyTorch per-row parity test.

### Files To Touch

Production:

- Add `gpurec/optimization/_bounds.py`.
- Add `gpurec/optimization/_closures.py`.
- Add `gpurec/optimization/_armijo.py`.
- Update `gpurec/optimization/lbfgsb.py` to delegate bound, projection,
  scalar closure, gradient, and Armijo helpers.
- Update `gpurec/optimization/projected_lbfgs.py` the same way.
- Update `gpurec/optimization/batched_lbfgs.py` only for bound/projection,
  feasible-direction, vector closure, and gradient helpers.
- Leave `gpurec/optimization/__init__.py` unchanged unless import-time tests
  expose a need; new helpers are internal.

Tests:

- Add focused unit tests for `_bounds.py` and `_closures.py` only where behavior
  is not already covered by optimizer tests.
- Extend existing optimizer tests rather than creating broad new integration
  fixtures.

Do not touch:

- `gpurec/api/*`
- `gpurec/core/*`
- `gpurec/workflow/*`
- `docs/optimization-workflow-call-graph.md`

### Parity Tests

Run the focused optimizer suite before and after the extraction:

```bash
pytest tests/unit/test_projected_lbfgs.py \
  tests/unit/test_lbfgsb.py \
  tests/unit/test_batched_lbfgs.py \
  tests/unit/test_lbfgsb_schilling_conformance.py
```

Add or preserve cases for:

- scalar lower/upper bound broadcasting;
- tensor bounds shaped like the original parameter;
- tensor bounds shaped like the flattened parameter;
- batched bounds that broadcast to `[B, N]`;
- sparse gradient densification;
- complex gradient rejection;
- closure return-shape rejection for scalar and row-wise optimizers;
- `LBFGSB` fallback state keys:
  `last_fallback_attempted`, `last_fallback_used`,
  `last_fallback_reason`, `last_direction_kind`,
  `consecutive_high_kkt_stalls`;
- `BatchedLBFGS` strong-Wolfe parity against PyTorch's per-row
  `_strong_wolfe`;
- legacy state tolerance for older `LBFGSB` optimizer state dicts.

Then run import and hygiene smoke:

```bash
pytest tests/unit/test_dependency_inventory.py \
  tests/unit/test_repository_hygiene.py
python - <<'PY'
from gpurec.optimization import BatchedLBFGS, LBFGSB, ProjectedLBFGS
print(BatchedLBFGS.__name__, LBFGSB.__name__, ProjectedLBFGS.__name__)
PY
```

Acceptance criteria:

- No public optimizer class changes.
- No optimizer state-key changes.
- `LBFGSB` and `ProjectedLBFGS` make the same accept/reject decisions on the
  existing scalar tests.
- `BatchedLBFGS` keeps the same per-row losses, gradients, accepted alphas, and
  eval counters on existing tests.
- `optimization` helpers import only Python stdlib and `torch`.

## Slice 2: Workflow Loop Policy Consolidation

Start this only after Slice 1 is merged or backed out cleanly.

### Goal

Reduce `gpurec/workflow/optimize.py` without turning
`gpurec/workflow/_transitions.py` into an even larger catch-all. Keep
`OptimizationRunner` as the public runner implementation, but make the loop
delegate complete decisions instead of carrying many local booleans.

Expected net reduction: about 250-450 lines from `optimize.py` and
`_transitions.py` combined if the extracted policies replace duplicated
bookkeeping instead of adding wrapper layers.

### Proposed Abstractions

Add `gpurec/workflow/_loop_policies.py`:

- `ObjectiveStopPolicy`
- `BoundedOptimizerPlateauDecision`
- `LBFGSBLossScheduleDecision`
- `LBFGSBHighKKTDecision`
- `HessianSGDLineSearchDecision`
- `AdagradRestartPhaseDecision`

Each object should be a small dataclass-returning function, not a class
hierarchy. Inputs should be plain context dataclasses already owned by
`workflow`, and outputs should include:

- metrics to merge into the row;
- state mutations requested;
- terminal status, if any;
- whether the loop should block normal loss-stop handling.

Add `gpurec/workflow/_batch_final_cache.py`:

- `BatchFinalCache`
- `create_batch_final_cache(model)`
- `cache_active_batch_final_result(model, loss_vec, cache, active_batch_indices)`

This removes three parallel tensors and repeated `is not None` guards from
`OptimizationRunner.run()`, while preserving checkpoint payload fields.

Move `_clear_cuda_allocator_cache_if_needed`,
`_drop_cached_static_states_if_needed`, `_clear_cached_solver_runtime_state`,
`_cached_static_states`, `_commit_pi_adjoint_pending_caches`, and
`_discard_pi_adjoint_pending_caches` into a single workflow cache helper if
they are still duplicated between `optimize.py`, `_evaluation.py`, and
`_fd_newton.py`. Do not move them into `api`; they are workflow cache policy
around `GeneReconModel`, not public model behavior.

### Files To Touch

Production:

- `gpurec/workflow/optimize.py`
- `gpurec/workflow/_transitions.py`
- Add `gpurec/workflow/_loop_policies.py`
- Add `gpurec/workflow/_batch_final_cache.py`
- Possibly add `gpurec/workflow/_model_cache.py`
- Update `gpurec/workflow/_evaluation.py` and `_fd_newton.py` only to consume
  shared cache helpers if this deletes duplication.

Tests:

- Existing workflow tests under `tests/unit/test_workflow.py` and
  `tests/unit/test_optimization_workflow.py`.
- Focused unit tests for each policy function with fake metrics and state.

Do not touch:

- `gpurec/api/model.py`
- `gpurec/core/*`
- `gpurec/optimization/*`
- `docs/optimization-workflow-call-graph.md`

### Parity Tests

Run existing workflow unit coverage:

```bash
pytest tests/unit/test_optimization_workflow.py \
  tests/unit/test_workflow.py \
  tests/unit/test_cli_workflow.py \
  tests/unit/test_workflow_artifacts.py
```

Add targeted tests for:

- projected-LBFGS high projected-gradient plateau reduces LR before loss stop;
- projected-LBFGS min-LR state blocks normal loss stop exactly as before;
- `LBFGSB` loss-change schedule advances and forces fallback state;
- `LBFGSB` high-KKT stop requires final loss phase and minimum fallback count;
- dynamic adagrad-restart phase completion by loss patience and by phase cap;
- hessian-SGD low-acceptance line-search activation and large-batch
  no-refresh suppression;
- active-batch final cache writes loss/gradient only for active family rows;
- resume path preserves `active_batch_index`, `active_solver_stage`,
  restart phase metadata, and optimizer phase mismatch behavior.

Acceptance criteria:

- `OptimizationRunner.run()` remains readable as build, resume, plan, loop,
  finalize.
- `workflow/_transitions.py` drops or stays flat in line count; it must not
  grow as a dumping ground.
- Checkpoint schema and history row keys remain unchanged.
- `workflow` continues to import `api` and `optimization` only at orchestration
  boundaries.

## Slice 3: `GeneReconModel` Facade Slimming

Start this after workflow parity is stable. The API group has more public
surface risk than the optimizer group.

### Goal

Keep `GeneReconModel` as the public facade while moving constructor assembly
and resident-batch control into model-internal helpers. Do not change public
factory methods, properties, tensor shapes, or autograd behavior.

Expected net reduction: about 150-275 lines from `model.py` plus better
separation of construction, batch residency, and evaluation.

### Proposed Abstractions

Add `gpurec/api/_model_builders.py`:

- `ModelBuildInputs`
- `normalize_model_build_inputs(...)`
- `prepare_dataset_from_trees(...)`
- `prepare_dataset_from_alerax_families(...)`
- `expand_initial_theta(...)`

This removes duplicated `from_trees()` and `from_alerax_families()` plumbing:
mode normalization, dtype validation, CUDA warmup, solver kwargs
normalization, theta base expansion, and retained preprocess selection.

Add `gpurec/api/_resident_batches.py`:

- `ResidentBatchRuntime`
- `create_resident_batch_runtime(...)`
- `create_full_static_runtime(...)`
- `replan_runtime(...)`
- `active_static(runtime)`
- `active_theta(runtime, theta)`
- `select_batch(runtime, batch_index)`
- `activate_family(runtime, family_index)`

This wraps `_batch_specs`, `_resident_cache`, `_static`,
`_current_batch_index`, and batch metadata without exposing `workflow`.

Add `gpurec/api/_genewise_streaming.py`:

- `full_genewise_nll_and_grad(model, need_grad)`
- `full_nll_per_family(model)`

This is only a relocation if it reduces `model.py` and preserves the public
method as a thin delegate. Keep tensor validation in existing
`_tensor_validation.py`.

### Files To Touch

Production:

- `gpurec/api/model.py`
- Add `gpurec/api/_model_builders.py`
- Add `gpurec/api/_resident_batches.py`
- Possibly add `gpurec/api/_genewise_streaming.py`
- Update existing `gpurec/api/_batch_specs.py`, `_static_builder.py`,
  `_streaming.py`, `_resident_cache.py` only where duplicated code can be
  removed.

Tests:

- Existing integration and unit model tests.
- Add direct tests for the new internal runtime helpers if behavior is not
  already covered through `GeneReconModel`.

Do not touch:

- `gpurec/core/*` public/internal kernel behavior
- `gpurec/workflow/*`
- `gpurec/optimization/*`
- `docs/optimization-workflow-call-graph.md`

### Parity Tests

Run model-focused coverage:

```bash
pytest tests/integration/test_gene_recon_model.py \
  tests/integration/test_hogenom_alerax_input.py \
  tests/integration/test_uniform_chunked_model.py \
  tests/unit/test_model_no_grad_evaluator.py \
  tests/unit/test_alerax_family_input.py \
  tests/unit/test_origination_prior.py
```

Add or preserve cases for:

- `from_trees()` and `from_alerax_families()` produce identical theta,
  family names, species names, and batch metadata;
- default and explicit `theta_init_rates` in global, specieswise, and genewise
  modes;
- resident-batched mode matches single-static mode for full loss and gradient;
- `lazy_preprocess`, `prefetch_batches`, `family_chunk_size`, `clade_budget`,
  and `batch_packing` semantics;
- `replan_resident_batches()` duplicate/out-of-range validation and returned
  metadata;
- `select_batch()`, `activate_family()`, `next()`, `clear()`, `close()`, and
  cached static state lifecycle;
- `full_loss()`, `full_loss_for_theta()`, `nll_per_family()`,
  `full_genewise_nll_and_grad()`, `full_nll_per_family()`, and
  `reconciliation_state()` tensors and errors.

Acceptance criteria:

- `api/model.py` reads as public facade plus small delegates.
- New internal API helpers import `core` and existing `api` helpers only; they
  do not import `workflow` or `optimization`.
- Public docs and import examples stay valid.

## Commit Strategy

Use small commits that each pass focused tests:

1. `optimization: extract bounded optimizer primitives`
   - Add `_bounds.py`, `_closures.py`, `_armijo.py`.
   - Update `ProjectedLBFGS` first because it is the simplest scalar bounded
     optimizer.
   - Run `tests/unit/test_projected_lbfgs.py`.

2. `optimization: reuse bounded primitives in lbfgsb`
   - Update `LBFGSB`.
   - Run `tests/unit/test_lbfgsb.py` and Schilling conformance.

3. `optimization: reuse bounded primitives in batched lbfgs`
   - Update `BatchedLBFGS`.
   - Run `tests/unit/test_batched_lbfgs.py`.

4. `workflow: extract loop policy decisions`
   - Move one policy family at a time: projected-LBFGS plateau, LBFGSB schedule,
     LBFGSB high-KKT, adagrad restart, hessian-SGD line search.
   - Run workflow tests after each move.

5. `api: slim GeneReconModel construction/runtime`
   - Split constructor/factory prep first, resident batch runtime second,
     genewise full-streaming last.
   - Run model integration tests after each move.

Each commit message should include:

- net `wc -l` impact for touched production files;
- public behavior statement;
- focused test command and result;
- explicit note that package layering was preserved.

## Rollback Strategy

Keep every slice mechanically reversible:

- For new helper modules, the old method bodies are still visible in the
  previous commit. If parity fails, revert the helper-use commit and keep any
  standalone helper tests only if they exposed a real pre-existing bug.
- Do not mix behavior fixes with extraction commits. If a parity test finds an
  actual bug, land the bug fix first against the old structure, then retry the
  extraction.
- Do not reformat whole files during extraction. Use narrow diffs so `git
  revert <commit>` cleanly restores prior behavior.
- Preserve public imports and optimizer state keys until all downstream
  workflow tests pass.
- Stop and rollback the current slice if a refactor requires changes in a
  lower layer that violates ownership boundaries, such as `optimization`
  importing `api` or `workflow`, or `api` importing `workflow`.

## Supervisor Guidance For The Team

- Implementation subagent: start with Slice 1 and touch only the optimization
  package plus focused tests.
- Tester subagent: snapshot focused optimizer test output before the extraction
  and compare failure modes, not only pass/fail.
- Supervisor subagent: reject moves that merely relocate code into another
  large file, especially `workflow/_transitions.py`. Require net reduction and
  stable boundaries per commit.

The next code-producing agent should not update the workflow call graph in this
slice. That stays with the main agent after source changes are real and tested.

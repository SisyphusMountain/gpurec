# Optimizer Helper Verification Plan - 2026-05-31

## Scope

This plan defines parity and coverage gates for extracting shared helper code from:

- `gpurec/optimization/lbfgsb.py`
- `gpurec/optimization/projected_lbfgs.py`
- `gpurec/optimization/batched_lbfgs.py`

The intended extraction is bounds/projection, closure/loss/gradient handling, and scalar Armijo plumbing. Keep optimizer classes, public imports, optimizer state keys, accepted-step behavior, and `BatchedLBFGS._strong_wolfe()` behavior unchanged.

As inspected during this test pass, the shared helper modules are not yet present. Existing unit coverage is strongest for `LBFGSB` fallback behavior and `BatchedLBFGS` strong-Wolfe parity; it is weakest for helper-level broadcasting and closure/gradient edge cases.

## Current Coverage

`tests/unit/test_lbfgsb.py` currently covers:

- SciPy parity on boxed quadratic and bounded Rosenbrock objectives.
- L-BFGS-B internals: generalized Cauchy point kink handling and scalar-limited subspace steps.
- Projected-gradient sign/top-k/coordinate fallback helpers, including loss-eval budgets.
- Full `LBFGSB.step()` fallback recovery after failed candidate line search.
- Sign fallback recovery when the projected-gradient map is too long.
- Competition after tiny or resolution-limited projected-gradient accepts.
- Repeated high-KKT tiny-progress fallback trigger.
- Legacy optimizer parameter-group tolerance for missing fallback control keys.

`tests/unit/test_projected_lbfgs.py` currently covers:

- Rejected loss-only line-search probes restore the original parameter.
- Rejected steps leave `last_accepted=False`, `last_alpha=0.0`, `last_step_inf=0.0`, and `last_loss_evals == max_ls`.

`tests/unit/test_batched_lbfgs.py` currently covers:

- Independent row-wise convergence with different objective scales.
- Row-wise Armijo behavior, not global loss reduction.
- `line_search_fn="strong_wolfe"` parity against PyTorch's per-row `_strong_wolfe` for loss, gradient, alpha, and final parameter.
- Tight `max_eval` accounting with separate gradient and loss-only closure calls.
- Lower and upper bound projection, inactive outward gradients at bounds, and free-coordinate step scaling.
- Scalar loss rejection and unknown line-search rejection.

`tests/unit/test_lbfgsb_schilling_conformance.py` should remain in the focused suite even though it is not one of the three optimizer files. It protects Schilling spec-vector parity for `projgr`, `active`, `bmv`, `cauchy`, and `subsm`, which are exactly the L-BFGS-B primitives that helper extraction can disturb indirectly.

## Required Gates

### Gate 1: Collection and Imports

Run before and after the helper extraction:

```bash
python -m pytest -q tests/unit/test_lbfgsb.py tests/unit/test_projected_lbfgs.py tests/unit/test_batched_lbfgs.py --collect-only
python -m pytest -q tests/unit/test_lbfgsb_schilling_conformance.py --collect-only
python - <<'PY'
from gpurec.optimization import BatchedLBFGS, LBFGSB, ProjectedLBFGS
print(BatchedLBFGS.__name__, LBFGSB.__name__, ProjectedLBFGS.__name__)
PY
```

Pass criteria:

- No collection errors.
- Optimizer public imports still resolve from `gpurec.optimization`.
- New helper modules remain internal and do not require changes to public `__all__`.

### Gate 2: Existing Focused Optimizer Parity

Run the focused optimizer suite:

```bash
python -m pytest -q \
  tests/unit/test_projected_lbfgs.py \
  tests/unit/test_lbfgsb.py \
  tests/unit/test_batched_lbfgs.py \
  tests/unit/test_lbfgsb_schilling_conformance.py
```

Pass criteria:

- Existing pass/fail behavior is unchanged.
- Existing state assertions keep the same value types, not just same truthiness.
- `BatchedLBFGS` strong-Wolfe parity still matches PyTorch per row for accepted alpha, loss, gradient, and final parameter.

### Gate 3: State Key and Type Parity

The extraction must preserve these state contracts.

`LBFGSB` per-parameter state must keep:

- `old_dirs`, `old_stps`
- `last_loss`, `last_grad`, `last_projected_grad`
- `last_grad_evals`, `last_loss_evals`
- `last_accepted`, `last_alpha`, `last_step_inf`, `last_directional_derivative`
- `last_direction_kind`, `last_line_search_decrease`, `last_armijo_required_decrease`
- `last_fallback_attempted`, `last_fallback_used`, `last_fallback_alpha`
- `last_fallback_loss_evals`, `last_fallback_max_loss_evals`
- `last_fallback_budget_exhausted`, `last_fallback_reason`
- `last_high_kkt_stall_count`, `last_history_cleared_for_fallback`
- `consecutive_high_kkt_stalls` when high-KKT tiny progress is tracked
- `last_projected_gradient_fallback_alpha` after fallback use
- `n_iter`

`ProjectedLBFGS` per-parameter state must keep:

- `old_dirs`, `old_stps`, `ro`
- `last_loss`, `last_grad`, `last_projected_grad`
- `last_grad_evals`, `last_loss_evals`
- `last_accepted`, `last_alpha`, `last_step_inf`, `last_directional_derivative`
- `n_iter`

`BatchedLBFGS` per-parameter state must keep:

- `func_evals`, `n_iter`
- `old_dirs`, `old_stps`, `ro`, `H_diag`
- `last_loss`, `last_grad`, `last_projected_grad`
- `last_accepted`, `last_alpha`, `last_n_iter`

Recommended missing focused tests:

- Add `test_lbfgsb_state_keys_and_types_after_regular_step`.
- Add `test_projected_lbfgs_state_keys_and_types_after_regular_step`.
- Add `test_batched_lbfgs_state_keys_and_shapes_after_regular_step`.

These should assert the exact key set for the fields above, scalar-vs-vector tensor shapes, and Python `bool`/`int`/`float` where current code stores Python scalars.

### Gate 4: Closure Evaluation Counters and Loss-Only Probes

The helper extraction must preserve the split between gradient closures and loss-only probes:

- `LBFGSB` and `ProjectedLBFGS` start each `step()` with one gradient closure.
- Accepted scalar line searches perform loss-only probes, then one final gradient refresh.
- Rejected scalar line searches restore the original parameter and do not perform a final gradient refresh.
- `BatchedLBFGS` Armijo mode uses loss-only probes when `loss_closure` is provided.
- `BatchedLBFGS` strong-Wolfe mode uses gradient closures for trial evaluations and must continue to update `func_evals` once per vectorized closure, not once per row.

Existing coverage:

- `ProjectedLBFGS` rejected probes assert `last_loss_evals == 3`.
- `BatchedLBFGS` tight `max_eval=2` asserts `state["func_evals"] == 2` and call counts `{"grad": 1, "loss": 1}`.
- Workflow coverage checks projected-LBFGS loss-only probe metrics, but the optimizer unit suite should not rely on workflow tests for this helper extraction.

Recommended missing focused tests:

- Add `test_projected_lbfgs_loss_only_probe_counter_matches_closure_calls`.
- Add `test_lbfgsb_loss_only_probe_counter_matches_closure_calls`.
- Extend `test_batched_lbfgs_strong_wolfe_matches_pytorch_per_row_search` to assert `state["func_evals"]` and the number of gradient closure calls.
- Add a scalar helper test where `loss_closure` asserts `torch.is_grad_enabled() is False`.
- Add a no-`loss_closure` scalar line-search test proving `_evaluate_loss` falls back to `closure()` under grad mode without incrementing `last_grad_evals`.

Targeted commands once these tests exist:

```bash
python -m pytest -q \
  tests/unit/test_projected_lbfgs.py::test_projected_lbfgs_loss_only_probe_counter_matches_closure_calls \
  tests/unit/test_lbfgsb.py::test_lbfgsb_loss_only_probe_counter_matches_closure_calls \
  tests/unit/test_batched_lbfgs.py::test_batched_lbfgs_respects_max_eval_after_line_search \
  tests/unit/test_batched_lbfgs.py::test_batched_lbfgs_strong_wolfe_matches_pytorch_per_row_search
```

### Gate 5: Bound Broadcasting and Projection Semantics

This is the highest-risk helper extraction surface because current optimizer implementations are similar but not identical.

Required semantics to preserve:

- Scalar bounds work for all three optimizers.
- Tensor bounds matching flattened scalar optimizer shape work.
- Tensor bounds matching original parameter shape work.
- Bounds broadcastable to original parameter shape work for scalar optimizers.
- `BatchedLBFGS` additionally preserves broadcast-to-flat fallback for shapes such as `[B, 1]`, `[1, N]`, and `[B, N]` after flattening the row dimensions.
- `lower_bound > upper_bound` still raises `ValueError("lower_bound must be <= upper_bound")`.
- Bounds are detached and converted to the flat tensor's device and dtype.
- Projection and projected-gradient sign conventions remain unchanged at lower and upper bounds.

Existing coverage:

- `LBFGSB` uses tensor lower/upper bounds in SciPy parity and internal Cauchy/subspace tests.
- `ProjectedLBFGS` only has scalar lower/upper bounds in the rejected-probe test.
- `BatchedLBFGS` has scalar lower and upper bounds, plus active-bound projected-gradient behavior.

Recommended missing focused tests:

- Add `tests/unit/test_optimizer_bounds_helpers.py` for the extracted helper functions.
- Add optimizer-level smoke tests that exercise helper delegation through each class:
  - `test_lbfgsb_bound_broadcasting_flat_param_and_original_shape_match`
  - `test_projected_lbfgs_bound_broadcasting_matches_lbfgsb_for_scalar_shapes`
  - `test_batched_lbfgs_bound_broadcasting_to_flat_rows`
  - `test_batched_lbfgs_invalid_broadcast_raises_runtime_error`

Targeted commands once helper tests exist:

```bash
python -m pytest -q tests/unit/test_optimizer_bounds_helpers.py
python -m pytest -q \
  tests/unit/test_lbfgsb.py -k "bound or cauchy or subspace or scipy" \
  tests/unit/test_projected_lbfgs.py -k "bound or projected or restores" \
  tests/unit/test_batched_lbfgs.py -k "bound or projected or lower or upper"
```

### Gate 6: Sparse and Complex Gradient Handling

Required semantics to preserve:

- Sparse gradients are densified before flattening.
- Missing gradients become zeros with the same shape as the flattened parameter.
- Complex parameters are rejected by each optimizer constructor.
- Complex gradients, if encountered by a shared helper, raise owner-specific `TypeError` messages:
  - `LBFGSB only supports real-valued gradients`
  - `ProjectedLBFGS only supports real-valued gradients`
  - `BatchedLBFGS only supports real-valued gradients`

Existing coverage:

- No focused sparse-gradient densification tests were found for these optimizer tests.
- No focused complex parameter or complex gradient rejection tests were found for these optimizer tests.

Recommended missing focused tests:

- Add `tests/unit/test_optimizer_closure_helpers.py` for `flat_grad` behavior:
  - scalar dense parameter plus sparse COO grad returns dense 1-D grad.
  - batched dense parameter plus sparse COO grad returns dense `[B, N]` grad.
  - missing grad returns zeros for scalar and batched cases.
  - owner-specific error messages are preserved for complex gradients if the helper can be exercised directly.
- Add optimizer constructor tests for complex and sparse parameter rejection in `test_lbfgsb.py`, `test_projected_lbfgs.py`, and `test_batched_lbfgs.py`.

Targeted commands once helper tests exist:

```bash
python -m pytest -q tests/unit/test_optimizer_closure_helpers.py
python -m pytest -q \
  tests/unit/test_lbfgsb.py -k "sparse or complex" \
  tests/unit/test_projected_lbfgs.py -k "sparse or complex" \
  tests/unit/test_batched_lbfgs.py -k "sparse or complex"
```

### Gate 7: Scalar Armijo Parity

Only `LBFGSB` and `ProjectedLBFGS` should share scalar Armijo helpers. Do not route `BatchedLBFGS` strong-Wolfe behavior through scalar Armijo helpers.

Required semantics to preserve:

- Non-finite trial loss, baseline loss, or directional derivative rejects.
- Acceptance threshold remains `min(loss + c1 * trial_gtd, nextafter(loss, -inf))`.
- Required decrease remains `max(0.0, loss - threshold)`.
- Rejected line searches restore the pre-probe parameter.
- `LBFGSB` `_LineSearchResult` fields and fallback competition continue to see the same `decrease`, `loss_evals`, `next_alpha`, and `armijo_required_decrease`.

Existing coverage:

- `ProjectedLBFGS` rejected probes verify restore and `last_loss_evals`.
- `LBFGSB` fallback tests indirectly verify Armijo result fields and fallback competition.

Recommended missing focused tests:

- Add `tests/unit/test_optimizer_armijo_helpers.py`.
- Add direct tests for non-finite rejection and strict `nextafter` decrease behavior.
- Add an `LBFGSB` line-search test that asserts `last_armijo_required_decrease` for a known scalar quadratic probe.

Targeted commands once helper tests exist:

```bash
python -m pytest -q tests/unit/test_optimizer_armijo_helpers.py
python -m pytest -q \
  tests/unit/test_projected_lbfgs.py::test_projected_lbfgs_restores_parameter_when_line_search_rejects_all_probes \
  tests/unit/test_lbfgsb.py -k "fallback or armijo or line_search"
```

### Gate 8: LBFGSB Fallback Behavior

Required semantics to preserve:

- Failed candidate line search clears history and triggers projected-gradient fallback.
- Fallback direction-kind strings remain stable:
  - `projected_gradient_fallback`
  - `projected_gradient_sign_fallback`
  - `projected_gradient_top{k}_sign_fallback`
  - `projected_gradient_coord{rank}_sign_fallback`
- Fallback budget accounting remains in loss evaluations, not gradient evaluations.
- `last_fallback_loss_evals`, `last_fallback_max_loss_evals`, and `last_fallback_budget_exhausted` remain consistent.
- Adaptive fallback alpha reuses `last_projected_gradient_fallback_alpha`.
- Repeated high-KKT tiny-progress behavior still forces fallback after the same number of stalls.
- Legacy checkpoint state without new fallback controls still steps.

Existing coverage is good, but add one missing direct optimizer-level budget test:

- `test_lbfgsb_step_sets_fallback_budget_exhausted_when_max_loss_evals_reached`

Targeted commands:

```bash
python -m pytest -q tests/unit/test_lbfgsb.py -k "fallback or legacy or high_kkt"
python -m pytest -q \
  tests/unit/test_workflow.py::test_lbfgsb_resume_reapplies_current_fallback_controls \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_specieswise_records_kkt_metrics
```

### Gate 9: BatchedLBFGS Strong-Wolfe Parity

Required semantics to preserve:

- Per-row accepted alpha, loss, gradient, and final parameter match PyTorch per-row `_strong_wolfe` on the existing quartic test.
- Trial evaluations stay vectorized and masked: inactive rows must not overwrite active row results, and accepted rows must keep accepted values while other rows continue searching.
- `remaining_eval_budget` remains a vectorized closure budget.
- Bound projection inside strong-Wolfe uses the same projected candidate and feasible-direction logic as Armijo mode.
- `state["last_grad"]` stores the accepted trial gradient for strong-Wolfe mode without an extra refresh.

Existing coverage:

- The current strong-Wolfe parity test checks alpha, loss, gradient, and final theta.

Recommended missing focused tests:

- Extend the existing strong-Wolfe test to assert `state["func_evals"]` and gradient closure call count.
- Add `test_batched_lbfgs_strong_wolfe_respects_bounds_for_inactive_rows`.
- Add `test_batched_lbfgs_strong_wolfe_respects_tight_max_eval_budget`.

Targeted commands:

```bash
python -m pytest -q \
  tests/unit/test_batched_lbfgs.py::test_batched_lbfgs_strong_wolfe_matches_pytorch_per_row_search \
  tests/unit/test_batched_lbfgs.py -k "strong_wolfe or max_eval or bound"
```

## Recommended Command Sequence

Use this sequence for the refactor PR:

```bash
python -m pytest -q tests/unit/test_lbfgsb.py tests/unit/test_projected_lbfgs.py tests/unit/test_batched_lbfgs.py --collect-only
python -m pytest -q \
  tests/unit/test_projected_lbfgs.py \
  tests/unit/test_lbfgsb.py \
  tests/unit/test_batched_lbfgs.py \
  tests/unit/test_lbfgsb_schilling_conformance.py
python -m pytest -q \
  tests/unit/test_optimizer_bounds_helpers.py \
  tests/unit/test_optimizer_closure_helpers.py \
  tests/unit/test_optimizer_armijo_helpers.py
python -m pytest -q \
  tests/unit/test_workflow.py::test_optimization_runner_projected_lbfgs_specieswise_uses_loss_only_probes \
  tests/unit/test_workflow.py::test_optimization_runner_lbfgsb_specieswise_records_kkt_metrics \
  tests/unit/test_workflow.py::test_lbfgsb_resume_reapplies_current_fallback_controls
```

The helper test files in the third command are expected new focused tests. If the implementation chooses different filenames, keep the same coverage topics and update the command.

## Merge Criteria

Do not accept the helper extraction until:

- All existing focused optimizer tests pass.
- New helper tests cover bounds, closure/gradient handling, and scalar Armijo.
- State keys and value types are unchanged for all three optimizers.
- Closure counters match actual call counts for accepted, rejected, loss-only, and tight-budget paths.
- `BatchedLBFGS` strong-Wolfe parity remains direct against PyTorch per-row `_strong_wolfe`.
- `LBFGSB` fallback state, budget, direction-kind strings, and legacy-state tolerance remain unchanged.

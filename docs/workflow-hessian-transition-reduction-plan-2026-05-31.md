# Workflow Hessian/Transition Reduction Plan - 2026-05-31

## Scope

Reduce workflow orchestration bloat without changing public or private DTO
compatibility.

Implemented slice:

- Let `OptimizationRunner.run()` call
  `hessian_sgd_line_search_decision(...)` directly after step-status
  classification. The policy helper already owns Hessian-SGD eligibility,
  active-line-search, active-objective, and full-stage-plateau guards.
- Collapse repeated `IterationStatusTransitionExecution(...)` constructors in
  `_transitions.py` into local `execution(...)` builders inside the two status
  transition functions.
- Add direct transition coverage for Hessian-SGD line-search activation resets
  so the constructor collapse remains anchored to observable state.

## Compatibility Invariants

- `hessian_sgd_line_search_decision(...)` keeps its existing signature and
  behavior. Ineligible calls preserve the current low-acceptance counter and do
  not activate line search.
- `IterationStatusTransitionExecution` and related transition dataclasses remain
  defined in `_transition_types.py` and re-exported through `_transitions.py`.
- Transition side effects remain in the same branches: batch reset, optimizer
  reset, Hessian-state reset/carry, checkpoint save, resume-info replacement,
  and planning-state replacement are unchanged.
- The Hessian-SGD full-stage plateau stop still wins over low-acceptance
  line-search activation.

## Focused Verification

Run:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_hessian_sgd_policy.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_optimization_workflow.py::test_next_batch_transition_resets_active_batch_and_checkpoint_status tests/unit/test_optimization_workflow.py::test_step_stopping_transition_saves_active_checkpoint_fields tests/unit/test_optimization_workflow.py::test_hessian_sgd_line_search_transition_resets_active_state
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py -k "hessian_sgd_likelihood_plateau_skips_low_acceptance_line_search or hessian_sgd_low_acceptance_uses_line_search_before_plateau or hessian_sgd_large_batch_uses_long_refresh_until_line_search or hessian_sgd_large_batch_plateau_stops_before_line_search"
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py -k "hessian_sgd_warmup_plateau_promotes_to_full_solver or hessian_sgd_large_batch_warmup_plateau_skips_full_solver or hessian_sgd_large_batch_warmup_skip_does_not_cache_noncanonical_full or hessian_sgd_advances_batch_after_full_stage_plateau or hessian_sgd_advances_batch_after_best_likelihood_stall"
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit -m "not slow and not gpu"
```

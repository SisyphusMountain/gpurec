# LBFGSB Fallback Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/optimization/lbfgsb.py` by moving the projected-gradient fallback
direction builders, fallback searches, fallback competition, and budget helpers
into a private optimization helper. Keep `LBFGSB` as the optimizer facade and
preserve all existing private method names on optimizer instances.

This slice is an extraction only. It must not change the public optimizer
exports, constructor defaults, fallback direction strings, loss-evaluation
accounting, or optimizer state keys consumed by workflow metrics.

## Extraction

Add `gpurec/optimization/_lbfgsb_fallbacks.py` with
`LBFGSBFallbackMixin`. The mixin owns:

- projected-gradient fallback directions;
- top-k sign and coordinate sign fallback searches;
- loss-evaluation budget helpers;
- tiny-progress and loss-resolution checks;
- fallback competition; and
- adaptive projected-gradient fallback alpha.

`gpurec/optimization/lbfgsb.py` keeps `_LineSearchResult` defined in the
original module and inherits the mixin before the shared bound/closure/Armijo
mixins. The helper calls `_project_flat()` and `_backtracking_line_search()`
through `self` so instance monkeypatches and subclasses keep working.

## Boundaries

- `_lbfgsb_fallbacks.py` is private optimization support and is not exported
  from `gpurec.optimization`.
- The helper may import stdlib typing/math plus `torch`.
- It must not import `gpurec.api`, `gpurec.workflow`, or `gpurec.core`.
- It must not runtime-import `gpurec.optimization.lbfgsb`; `_LineSearchResult`
  may be referenced only for type checking to avoid circular imports.

## Verification Gates

```bash
python -m compileall -q gpurec/optimization
python -m pytest -q tests/unit/test_lbfgsb.py tests/unit/test_lbfgsb_schilling_conformance.py
python -m pytest -q tests/unit/test_optimization_helpers.py
python -m pytest -q tests/unit/test_repository_hygiene.py -k "internal_optimization_helper_modules_document_support_boundary"
ruff check gpurec/optimization/lbfgsb.py gpurec/optimization/_lbfgsb_fallbacks.py tests/unit/test_lbfgsb.py tests/unit/test_repository_hygiene.py
git diff --check
```

Run focused workflow LBFGSB gates if the step loop or state telemetry changes.

## Acceptance Criteria

- `LBFGSB` instances still expose the moved fallback private methods.
- `_LineSearchResult` remains importable from `gpurec.optimization.lbfgsb`.
- Fallback strings such as `projected_gradient_fallback`,
  `projected_gradient_sign_fallback`,
  `projected_gradient_top{k}_sign_fallback`, and
  `projected_gradient_coord{rank}_sign_fallback` are unchanged.
- State keys beginning with `last_fallback_`, `last_high_kkt_stall_count`,
  `last_history_cleared_for_fallback`, `consecutive_high_kkt_stalls`, and
  `last_projected_gradient_fallback_alpha` are unchanged.
- Focused fallback tests, Schilling conformance, helper tests, and layering
  hygiene pass.

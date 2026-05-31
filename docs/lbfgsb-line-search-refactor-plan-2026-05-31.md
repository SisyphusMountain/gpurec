# LBFGSB Line Search Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/optimization/lbfgsb.py` by moving the scalar L-BFGS-B Armijo
backtracking line search into a private optimization helper. Preserve the
optimizer facade, constructor defaults, step signature, state telemetry, and
all line-search arithmetic.

## Extraction

Add `gpurec/optimization/_lbfgsb_line_search.py` with:

- `_LineSearchResult`, preserving its historical
  `gpurec.optimization.lbfgsb` module identity for white-box tests; and
- `LBFGSBLineSearchMixin`, owning `_backtracking_line_search()`.

`gpurec/optimization/lbfgsb.py` imports and re-exports `_LineSearchResult`, then
adds the mixin to `LBFGSB` inheritance. The method remains available on
optimizer instances so monkeypatches of `optimizer._backtracking_line_search`
continue to work.

## Boundaries

- `_lbfgsb_line_search.py` is private optimization support and is not exported
  from `gpurec.optimization`.
- The helper does not runtime-import `gpurec.optimization.lbfgsb`, avoiding a
  circular dependency.
- The extraction must not change accepted/rejected trial handling, loss-eval
  accounting, Armijo threshold usage, projected bounds, or parameter restore
  behavior.

## Verification Gates

```bash
python -m compileall -q gpurec/optimization/lbfgsb.py gpurec/optimization/_lbfgsb_line_search.py
ruff check gpurec/optimization/lbfgsb.py gpurec/optimization/_lbfgsb_line_search.py
git diff --check
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_lbfgsb.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/unit/test_workflow.py -k "lbfgsb"
```

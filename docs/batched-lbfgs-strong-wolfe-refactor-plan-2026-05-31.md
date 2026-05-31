# Batched L-BFGS Strong-Wolfe Refactor Plan, 2026-05-31

## Scope

Extract the vectorized strong-Wolfe line-search machinery from
`gpurec/optimization/batched_lbfgs.py` into a private optimization helper mixin.
This is a mechanical code-motion slice: no line-search math, closure evaluation
semantics, parameter projection, or optimizer state keys should change.

## Move

Add `gpurec/optimization/_batched_lbfgs_strong_wolfe.py` with:

- `BatchedLBFGSStrongWolfeMixin._evaluate_trial_with_grad()`
- `BatchedLBFGSStrongWolfeMixin._strong_wolfe()`

`BatchedLBFGS` inherits the mixin and keeps `_strong_wolfe` callable on
optimizer instances.  The main `step()` loop remains in `batched_lbfgs.py`,
including Armijo search, `func_evals` accounting, accepted-step bookkeeping,
history updates, and final `last_*` state writes.

## Boundaries

- No changes to public `gpurec.optimization` exports.
- No imports from `api`, `core`, or `workflow` in the new helper.
- Do not merge this with scalar `LBFGSB` or `ProjectedLBFGS` Armijo helpers.
- Keep `_line_search_interpolation.py` as pure interpolation support and
  `_batched_lbfgs_history.py` as row-history support.

## Verification Gates

```bash
python -m compileall -q gpurec/optimization tests/unit/test_batched_lbfgs.py tests/unit/test_repository_hygiene.py
ruff check gpurec/optimization/batched_lbfgs.py gpurec/optimization/_batched_lbfgs_strong_wolfe.py tests/unit/test_batched_lbfgs.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_batched_lbfgs.py
python -m pytest -q tests/unit/test_optimization_helpers.py
python -m pytest -q tests/unit/test_repository_hygiene.py::test_internal_optimization_helper_modules_document_support_boundary
python -m pytest -q tests/unit/test_projected_lbfgs.py tests/unit/test_lbfgsb.py tests/unit/test_batched_lbfgs.py tests/unit/test_lbfgsb_schilling_conformance.py
python -m pytest -q tests/integration/test_gene_recon_model.py::test_batched_lbfgs_genewise_runs_one_polish_step
CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"
```

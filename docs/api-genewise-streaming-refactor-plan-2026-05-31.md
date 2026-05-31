# API Genewise Streaming Refactor Plan, 2026-05-31

## Scope

Move the implementation of `GeneReconModel.full_genewise_nll_and_grad()` and
`GeneReconModel.full_nll_per_family()` into a private API helper module while
leaving both public methods, signatures, decorators, and docstrings on
`GeneReconModel`.

This is deliberately narrower than resident-batch runtime extraction. It moves
the genewise full-streaming loop only; batch selection, active theta handling,
static cache ownership, and full scalar streaming stay where they are.

## Boundaries

- Add `gpurec/api/_genewise_streaming.py`.
- The helper may import `torch`, `_tensor_validation`, and
  `_uniform_evaluator`.
- The helper must not import `gpurec.workflow` or `gpurec.optimization`.
- Keep `gpurec/api/model.py` as the public facade.

## Test Notes

Existing unit tests that monkeypatch `_evaluate_static_state` need to patch the
module that owns the call:

- full-genewise tests should patch `gpurec.api._genewise_streaming`;
- full scalar streaming tests should patch `gpurec.api._streaming`;
- `model.py` keeps its `_evaluate_static_state` alias for existing hygiene
  checks and diagnostic imports, but it should not be the monkeypatch route for
  moved helpers.

## Verification Gates

```bash
python -m compileall -q gpurec/api
python -m pytest -q tests/unit/test_model_no_grad_evaluator.py tests/unit/test_gradient_accumulator.py
python -m pytest -q tests/unit/test_workflow.py -k "full_nll_per_family or full_genewise_nll_and_grad"
python -m pytest -q tests/unit/test_repository_hygiene.py -k "static_state_evaluation or full_batch_helpers"
python -m pytest -q tests/integration/test_gene_recon_model.py
ruff check gpurec/api/model.py gpurec/api/_genewise_streaming.py tests/unit/test_model_no_grad_evaluator.py tests/unit/test_gradient_accumulator.py
git diff --check
```

Acceptance criteria: public behavior and error text remain unchanged, active
batch selection is restored after batched genewise streaming, and package
layering remains unchanged.

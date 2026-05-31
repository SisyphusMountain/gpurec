# API Model Builder Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/api/model.py` by extracting only `GeneReconModel` factory
construction plumbing into a private API helper module. Keep the public
`GeneReconModel` constructor, `from_trees()`, and `from_alerax_families()`
signatures and docstrings in place.

This slice deliberately does not move resident-batch runtime methods,
full-batch streaming, or reconciliation/export behavior. Those areas carry
more state-machine risk and should remain unchanged until factory parity is
green.

## Boundaries

- `gpurec/api/_model_builders.py` may import `gpurec.core` and API internals.
- `gpurec/api/_model_builders.py` must not import `gpurec.workflow` or
  `gpurec.optimization`.
- `gpurec/api/model.py` remains the public facade and owns `GeneReconModel`.
- No changes to `core`, `optimization`, or `workflow` are needed for this
  slice.

## Candidate Extraction

Move duplicated work shared by `from_trees()` and `from_alerax_families()`:

- mode normalization and mode flag derivation;
- default objective gate;
- dtype validation;
- solver keyword normalization;
- `preprocess_cpu_cores` validation;
- CUDA device validation and warmup bracketing;
- retained `GeneDataset` construction;
- `theta_init_rates` conversion and expanded initial theta construction.

Keep AleRax-only family selection and parsing in the AleRax helper path, and
keep public factory methods as short delegates that pass through to
`GeneReconModel(...)`.

## Baseline Finding

Before new code edits, this command failed:

```bash
python -m pytest -q tests/unit/test_alerax_family_input.py tests/unit/test_origination_prior.py tests/unit/test_model_no_grad_evaluator.py
```

Result: 71 passed, 6 failed.

The failures are API-private test compatibility issues around helpers that were
already extracted from `model.py`:

- `api_model._build_static_state` is no longer available for monkeypatching;
- `api_model.evaluate_resident_static_state` is no longer available for
  monkeypatching;
- `api_model.solve_resident_e` is no longer available for monkeypatching;
- `_streaming.py` imports evaluator functions directly, so several
  `api_model` monkeypatches no longer intercept the streaming path.

The factory extraction should either preserve these compatibility aliases in a
narrow way or adjust the tests to patch the new authoritative helper modules.
Do not mix broader behavior changes into this refactor.

Resolution in this slice: tests were updated to patch `_model_builders`,
`_streaming`, and the current `model.py` static-builder alias instead of
reintroducing stale `model.py` compatibility aliases.

## Verification Gates

Minimum local gates:

```bash
python -m compileall -q gpurec/api
python -m pytest -q tests/unit/test_alerax_family_input.py tests/unit/test_origination_prior.py tests/unit/test_model_no_grad_evaluator.py
python -m pytest -q tests/unit/test_workflow.py -k "gene_recon_constructors or from_trees or from_alerax or adaptive_neumann_terms_mode or theta_init"
python -m pytest -q tests/unit/test_repository_hygiene.py -k "model_docstrings or static_state_evaluation or full_batch_helpers"
ruff check gpurec/api/model.py gpurec/api/_model_builders.py tests/unit/test_model_no_grad_evaluator.py tests/unit/test_origination_prior.py
git diff --check
```

GPU integration parity remains recommended when CUDA is available:

```bash
python -m pytest -q tests/integration/test_gene_recon_model.py tests/integration/test_hogenom_alerax_input.py
```

## Acceptance Criteria

- `GeneReconModel.from_trees()` and `GeneReconModel.from_alerax_families()` are
  thin delegates and preserve existing validation order.
- Public model construction behavior and tensor shapes are unchanged.
- `gpurec/api/model.py` line count drops without creating a larger catch-all
  module.
- Focused API/unit gates pass, including the baseline compatibility failures.
- Package layering remains unchanged.

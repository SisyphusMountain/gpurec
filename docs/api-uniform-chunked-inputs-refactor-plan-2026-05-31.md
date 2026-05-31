# API UniformChunked Inputs Refactor Plan, 2026-05-31

## Scope

Extract private dtype validation, solver-kwarg validation, and folder gene-path
selection helpers from `gpurec/api/uniform_chunked.py` into a dedicated private
input helper module.

## Move

Add `gpurec/api/_uniform_chunked_inputs.py` with:

- `_validate_uniform_dtype`
- `_normalize_uniform_solver_kwargs`
- `_selected_gene_paths`

`uniform_chunked.py` keeps aliases for these moved private helpers so existing
tests, profiling tools, and private imports continue to resolve.  `_as_auto_int`
stays in the facade for the existing private compatibility surface.

## Boundaries

- This slice does not change constructor validation order or accepted values.
- Direct `UniformChunkedReconModel` bf16 behavior remains unchanged.
- Folder selection still validates `start` and `max_families` before glob
  selection and keeps the `g.nwk` fallback for the default glob.
- No `workflow` or `optimization` imports are introduced into `gpurec.api`.

## Verification Gates

```bash
python -m compileall -q gpurec/api tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py
ruff check gpurec/api/uniform_chunked.py gpurec/api/_uniform_chunked_inputs.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_workflow.py -k "uniform_chunked or uniform_auto or family_chunk_size_normalization"
python -m pytest -q tests/unit/test_repository_hygiene.py -k "uniform_chunked"
python -m pytest -q tests/unit/test_bench_uniform_forward_backward_pipeline.py
python -m pytest -q tests/integration/test_uniform_chunked_model.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"
```

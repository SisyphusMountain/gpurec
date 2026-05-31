# API UniformChunked Evaluation Refactor Plan, 2026-05-31

## Scope

Extract evaluation-only internals from `gpurec/api/uniform_chunked.py` into the
private helper module `gpurec/api/_uniform_chunked_eval.py` without changing the
public `UniformChunkedReconModel` or `UniformChunkMetadata` API.

## Moved Names

The helper now owns:

- `_PI_BACKWARD_TENSOR_KEYS`
- `_PI_BACKWARD_COUNTER_KEYS`
- `_UniformChunkedEvaluation`
- `_UniformChunkedReadOnlyEvaluation`
- `_UniformChunkStatsRow`
- `_new_pi_backward_accumulator`
- `_time_cuda_ms`
- `_root_count_tensor`
- `_selected_chunks`
- `_e_adjoint_stats_fields`
- `_chunk_stats_row`
- `_require_chunked_gradient_dtype`
- `_evaluate_chunked_uniform_result`
- `_evaluate_chunked_uniform`
- `_evaluate_chunked_uniform_read_only`

`gpurec/api/uniform_chunked.py` imports these names back so existing private
imports such as `from gpurec.api.uniform_chunked import _selected_chunks`
continue to resolve.

## Boundaries

- `uniform_chunked.py` keeps the public facade, construction flow, metadata, and
  chunk/state containers: `_UniformChunkSpec`, `_UniformBuiltChunk`, and
  `_UniformChunkedState`.
- `_uniform_chunked_eval.py` treats state and built chunks as internal structural
  objects typed as `Any`, avoiding a reverse import from `uniform_chunked.py`.
- The new helper imports core tensor/evaluation utilities and keeps the existing
  `gpurec.optimization.implicit_grad` bridge used by chunked gradient
  evaluation.
- The helper does not import `gpurec.workflow`.
- `gpurec/api/__init__.py` public exports stay unchanged.

## Verification Gates

Run the focused checks for this slice:

```bash
python -m compileall -q gpurec/api
ruff check gpurec/api/uniform_chunked.py gpurec/api/_uniform_chunked_eval.py tests/unit/test_optimization_workflow.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_optimization_workflow.py -k "uniform_chunked"
python -m pytest -q tests/unit/test_workflow.py -k "uniform_chunk"
python -m pytest -q tests/unit/test_origination_prior.py -k "uniform"
python -m pytest -q tests/unit/test_repository_hygiene.py -k "uniform_chunked or runtime_surface"
python -m pytest -q tests/integration/test_uniform_chunked_model.py
```

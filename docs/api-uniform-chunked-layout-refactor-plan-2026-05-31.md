# API UniformChunked Layout Refactor Plan, 2026-05-31

## Scope

Extract private uniform chunk layout/state containers and retained-Rust payload
conversion helpers from `gpurec/api/uniform_chunked.py` into a dedicated
private module, without changing the public `UniformChunkedReconModel` or
`UniformChunkMetadata` API.

## Move

Add `gpurec/api/_uniform_chunked_layout.py` with:

- `_UniformChunkSpec`
- `_UniformBuiltChunk`
- `_UniformChunkedState`
- `_dtype_name_for_rust`
- `_move_wave_layout_to_device`
- `_built_chunks_from_rust`

Keep `UniformChunkMetadata` in `uniform_chunked.py` because it is public and
should keep its existing module identity.  Re-export the moved private names as
module aliases from `uniform_chunked.py` so existing private imports used by
tests and profiling helpers continue to resolve.

## Boundaries

- This slice is mechanical code motion only.
- No `core`, `optimization`, or `workflow` imports are introduced into the new
  helper.
- Constructor validation order, bf16 direct-API behavior, chunk metadata, and
  retained-Rust layout payload semantics remain unchanged.
- The public `__all__` remains `["UniformChunkMetadata",
  "UniformChunkedReconModel"]`.

## Verification Gates

```bash
python -m compileall -q gpurec/api
ruff check gpurec/api/uniform_chunked.py gpurec/api/_uniform_chunked_layout.py tests/unit/test_workflow.py tests/unit/test_repository_hygiene.py tests/unit/test_origination_prior.py tests/unit/test_optimization_workflow.py
python -m pytest -q tests/unit/test_workflow.py -k "uniform_chunked"
python -m pytest -q tests/unit/test_origination_prior.py -k "uniform_chunked"
python -m pytest -q tests/unit/test_optimization_workflow.py -k "uniform_chunked"
python -m pytest -q tests/unit/test_repository_hygiene.py -k "uniform_chunked or benchmark"
python -m pytest -q tests/unit/test_bench_uniform_forward_backward_pipeline.py
python -m pytest -q tests/integration/test_uniform_chunked_model.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q -m "unit and not gpu"
```

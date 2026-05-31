# Uniform Chunked Init Refactor Plan - 2026-05-31

## Scope

Extract `UniformChunkedReconModel` constructor setup from
`gpurec/api/uniform_chunked.py` into a private API helper while preserving the
public constructor, factory methods, `__all__`, and the existing private
compatibility aliases that tests import from `gpurec.api.uniform_chunked`.

The extraction target is constructor-only setup:

- public argument validation and normalization;
- CUDA device validation;
- Rust preprocessing and chunk-layout construction;
- memory-policy auto selection;
- origination-prior preparation;
- `_UniformChunkedState` and public model attribute registration.

Runtime evaluation methods, factory signatures, and public metadata stay in
`gpurec/api/uniform_chunked.py`.

## Compatibility Constraints

- `gpurec.api.uniform_chunked.GeneDataset` and
  `gpurec.api.uniform_chunked.require_cuda_device` remain module-level
  monkeypatch points. The private helper receives both as explicit
  dependencies instead of importing fresh copies.
- Factories keep their validation order: bad mode, dtype, solver controls,
  family selection, and theta init fail before CUDA checks or file/Rust IO.
- Direct construction keeps accepting `torch.bfloat16` at the uniform dtype
  boundary; workflow and CLI dtype exposure remain unchanged.
- Prepared origination state keeps identity guarantees:
  `model._origination_prior`, `model.origination_probs`,
  `model._state.origination_prior`, and `model._state.origination_probs` refer
  to the same prepared prior/tensor objects as before.
- Depth-first-fit scheduler counts, memory-policy auto selection,
  `build_chunked_layouts(...)` kwargs, CUDA synchronize behavior, and public
  attributes such as `family_chunk_size`, `max_wave_size`, `batch_packing`,
  `memory_policy`, `gene_trees`, `family_names`, and `species_tree` remain
  unchanged.

## Implementation Shape

- Add private `gpurec/api/_uniform_chunked_init.py`.
- Introduce `UniformChunkedInitDependencies` for injected compatibility
  dependencies.
- Introduce `UniformChunkedInitState` for prepared constructor state.
- Use `prepare_uniform_chunked_init(...)` for validation, preprocessing, memory
  planning, chunk layout construction, and origination preparation.
- Use `apply_uniform_chunked_init(...)` to register `theta`, buffers, private
  runtime state, and public constructor attributes on the model.
- Keep private aliases from `_uniform_chunked_layout`,
  `_uniform_chunked_inputs`, and `_uniform_chunked_eval` imported by
  `gpurec.api.uniform_chunked` and out of `__all__`.

## Verification Gates

Run focused static and behavioral checks after the extraction:

```bash
python -m compileall -q gpurec/api/uniform_chunked.py gpurec/api/_uniform_chunked_init.py
ruff check gpurec/api/uniform_chunked.py gpurec/api/_uniform_chunked_init.py
git diff --check -- gpurec/api/uniform_chunked.py gpurec/api/_uniform_chunked_init.py docs/uniform-chunked-init-refactor-plan-2026-05-31.md
```

Run focused unit coverage:

```bash
python -m pytest -q \
  tests/unit/test_workflow.py::test_uniform_chunked_rejects_bad_chunk_controls_before_device_or_io \
  tests/unit/test_workflow.py::test_uniform_chunked_factories_reject_invalid_dtype_before_device_or_io \
  tests/unit/test_workflow.py::test_bfloat16_is_direct_uniform_api_only \
  tests/unit/test_workflow.py::test_uniform_chunked_factories_reject_invalid_solver_controls_before_device_or_io \
  tests/unit/test_workflow.py::test_uniform_chunked_init_rejects_nonbool_controls_before_side_effects \
  tests/unit/test_workflow.py::test_uniform_chunked_factories_reject_nonbool_controls_before_device_or_io \
  tests/unit/test_workflow.py::test_uniform_chunked_alerax_constructor_validates_mode_before_io \
  tests/unit/test_workflow.py::test_uniform_chunked_from_folder_validates_selection_before_io \
  tests/unit/test_workflow.py::test_uniform_chunked_constructors_reject_bad_theta_init_before_io \
  tests/unit/test_workflow.py::test_uniform_chunked_constructors_reject_unavailable_cuda_before_io
python -m pytest -q tests/unit/test_origination_prior.py::test_uniform_chunked_model_threads_prepared_origination_prior
python -m pytest -q tests/unit/test_optimization_workflow.py -k uniform_chunked
python -m pytest -q \
  tests/unit/test_repository_hygiene.py::test_uniform_chunked_keeps_private_layout_aliases_unexported \
  tests/unit/test_repository_hygiene.py::test_uniform_chunked_keeps_private_input_aliases_unexported \
  tests/unit/test_repository_hygiene.py::test_uniform_chunked_keeps_private_evaluator_aliases_unexported
```

Run integration coverage when CUDA/data availability makes it feasible:

```bash
python -m pytest -q tests/integration/test_uniform_chunked_model.py
```

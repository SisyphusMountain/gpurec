# Refactor Proposal: Typed Config Models With Pydantic

## Goal

Introduce typed configuration models backed by Pydantic v2 so configuration
validation, CLI/config normalization, and model-construction option mapping have
one authoritative schema layer.

This is intended to reduce the amount of hand-written validation and conversion
logic currently spread across `gpurec.workflow.config`, `gpurec.cli`,
`gpurec.api._validation`, `gpurec.api.model`, and
`gpurec.api.uniform_chunked`.

## Non-goals for the first implementation PR

- Do not change the public `RunConfig` or `SamplingConfig` import names.
- Do not change supported JSON config field names.
- Do not change CLI precedence: explicit CLI flags continue to override
  `--config` values.
- Do not change output artifact names, summary keys, checkpoint metadata, or
  route-gate semantics.
- Do not migrate CUDA, Triton, likelihood, or optimizer internals.

## Proposed dependency

Add Pydantic v2 as a runtime dependency:

```toml
pydantic = ">=2,<3"
```

Configuration validation is part of the public runtime surface, so this should
be a normal dependency rather than a development extra.

## Proposed module layout

Add a new schema module, initially internal:

```text
gpurec/workflow/config_models.py
```

The first version should define:

```python
Mode = Literal["global", "specieswise", "genewise"]
OptimizerName = Literal[
    "adam",
    "adagrad",
    "projected-sgd",
    "lbfgs",
    "adam-lbfgs",
    "projected-lbfgs",
    "lbfgsb",
    "batched-lbfgs",
    "adam-fd-newton",
    "hessian-sgd",
    "adagrad-restarts",
    "adagrad-restarts-lbfgsb",
]

class BatchConfigModel(BaseModel):
    family_chunk_size: int | str | None = 0
    clade_budget: int | None = DEFAULT_CLADE_BUDGET
    batch_packing: str = "depth_first_fit"
    max_wave_size: int | None = 8192
    small_family_max_leaves: int = 0

class SolverConfigModel(BaseModel):
    fixed_iters_e: int | None = None
    max_iters_e: int = 2000
    tol_e: float = 1e-8
    fixed_iters_pi: int = 16
    neumann_terms: int = 16
    solver_warmup_iters: int = 4
    solver_warmup_loss_patience: int = 2
    adaptive_iters: bool = True
    adaptive_neumann_terms: bool = False
    final_check_iters: int = 32
    convergence_check_interval: int = 4
    e_logsumexp_tol: float = 1e-5
    pi_max_diff_tol: float = 1e-5
    gradient_change_tol: float = 1e-4
    gradient_change_rtol: float = 1e-4

class OptimizerConfigModel(BaseModel):
    optimizer: str = "auto"
    steps: int = 5000
    lr: float = 0.01
    # Existing optimizer-specific fields remain flat for compatibility in phase 1.

class RunConfigModel(BaseModel):
    species_tree: Path
    families_file: Path
    out_dir: Path
    mode: Mode = "genewise"
    device: str = ""
    dtype: str = "float32"

    # Phase 1 keeps the existing flat field shape.
    start: int = 0
    max_families: int | None = None
    preprocess_cpu_cores: int | None = None
    # Include current RunConfig fields with validators that preserve behavior.
```

Phase 1 should keep the external flat schema. Nested `solver`, `batching`, and
`optimizer` models can be introduced later as an internal view, then exposed only
if the API contract is updated.

## Migration plan

### Phase 1: Compatibility-preserving schema layer

1. Add `pydantic>=2,<3` to `pyproject.toml`.
2. Add `gpurec.workflow.config_models` with `RunConfigModel` and
   `SamplingConfigModel`.
3. Configure both models with `extra="forbid"`.
4. Move scalar normalization into Pydantic field validators:
   - mode normalization
   - optimizer normalization and `auto` resolution
   - dtype normalization
   - path expansion/resolution
   - integer, positive integer, positive even integer, and non-negative integer
     checks
   - finite float checks
   - schedule normalization for adagrad restarts and loss-stop schedules
5. Keep the existing `RunConfig` dataclass as the public object, but make
   `RunConfig.from_dict()` validate through `RunConfigModel` before constructing
   the dataclass.
6. Keep `RunConfig.to_dict()` output byte-for-byte compatible except for existing
   normalization behavior.

### Phase 2: Remove duplicated validation

After Phase 1 lands and tests pass, remove duplicated validation from:

- `_validate_json_scalar_types`
- redundant scalar checks in `RunConfig.__post_init__`
- repeated optimizer/mode/dtype checks that are fully covered by the model

The dataclass can remain as a compatibility wrapper until a later API-contract
revision.

### Phase 3: Typed model-construction options

Introduce internal typed views:

```python
class ModelConstructionOptions(BaseModel):
    mode: Mode
    solver: SolverConfigModel
    batching: BatchConfigModel
    lazy_preprocess: bool = True
    prefetch_batches: int | Literal["all"] = "all"
```

Use this object to build `GeneReconModel.from_alerax_families(...)` kwargs.
This replaces long ad-hoc keyword lists in workflow model construction.

## Error compatibility

Pydantic errors should not be exposed raw from the CLI. Convert validation
failures into the current `ValueError`/argparse error style so exit-code behavior
stays compatible:

- malformed usage remains exit code `2`
- runtime/config validation gates remain exit code `1` where currently expected
- `--json` inspection commands continue to emit stable JSON objects

## Test plan

Add or update unit tests for:

- unknown fields are rejected
- legacy fields are ignored where currently ignored
- required path fields are enforced
- JSON scalar type behavior remains strict
- mode and optimizer aliases normalize as before
- `optimizer="auto"` resolves by mode as before
- dtype aliases normalize as before
- adagrad restart schedule normalization remains unchanged
- invalid hessian/adagrad optimizer combinations still fail
- `RunConfig.to_dict()` preserves existing flat output shape
- CLI config-file values remain overridden by explicit flags

Run:

```bash
python -m ruff check gpurec tests/unit
python -m mypy --config-file=pyproject.toml --follow-imports=skip gpurec
python -m pytest -q tests/unit
```

## Acceptance criteria

- Existing public imports remain valid.
- Existing flat JSON `RunConfig` files remain valid.
- Existing CLI precedence remains unchanged.
- Existing workflow tests pass.
- New tests cover Pydantic-backed validation.
- The first PR introduces a schema boundary without changing optimizer/runtime
  behavior.

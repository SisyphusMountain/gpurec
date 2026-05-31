# Workflow Config IO Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/workflow/config.py` by moving JSON object loading, RunConfig path
resolution, legacy-field filtering constants, and JSON scalar type checks into
a private workflow helper while preserving the existing
`gpurec.workflow.config` import surface.

This is an IO/schema-location extraction. It must not change `RunConfig`
normalization, config-template behavior, CLI preflight validation, or checkpoint
config loading.

## Moved Logic

Add `gpurec/workflow/_config_io.py` for:

- `_JSON_INT_FIELDS`;
- `_JSON_FLOAT_FIELDS`;
- `_JSON_BOOL_FIELDS`;
- `_RUN_CONFIG_REQUIRED_PATH_FIELDS`;
- `_RUN_CONFIG_PATH_FIELDS`;
- `_RUN_CONFIG_LEGACY_FIELDS`;
- `_validate_json_scalar_types()`;
- `_resolve_run_config_path_fields()`;
- `_reject_json_constant()`;
- `load_json_object()`;
- `load_json_object_text()`;
- `load_run_config_data()`; and
- `load_run_config_text()`.

`gpurec/workflow/config.py` keeps facade imports or wrappers for the public
loader names used by CLI helpers and tests.

## Boundaries

- `_config_io.py` may import only stdlib JSON/path helpers and typing needed for
  schema checks. It must not import `gpurec.workflow.config`,
  `gpurec.workflow.optimize`, CLI modules, API modules, or `torch`.
- `config.py` remains the owner of `RunConfig`, `SamplingConfig`, runtime
  normalization, device/dtype parsing, and public facade names.
- `RunConfig.from_dict()` may continue to read the private field sets through
  facade imports, but unknown, legacy, missing-path, and scalar-type errors must
  remain unchanged.
- Do not add `_config_io.py` to `gpurec.workflow.__all__` or top-level
  `gpurec`.

## Verification Gates

```bash
python -m compileall -q gpurec/workflow
ruff check gpurec/workflow/config.py gpurec/workflow/_config_io.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_workflow.py -k "load_run_config or from_dict or unknown RunConfig or JSON or non_string_mode or workflow_config_submodule_import"
python -m pytest -q tests/unit/test_cli_workflow.py -k "validate_config or config_template or load_config"
python -m pytest -q tests/unit/test_validation.py tests/unit/test_repository_hygiene.py -k "RunConfig or workflow_submodule or run_config_reference"
git diff --check
```

## Acceptance Criteria

- Public loader imports from `gpurec.workflow.config` keep working.
- Relative paths still resolve against the supplied file/base directory.
- JSON constants, booleans, integer fields, path fields, unknown fields, and
  legacy fields keep the same validation behavior and messages.
- Workflow helper ownership docs and hygiene guards keep `_config_io.py`
  private and import-light.

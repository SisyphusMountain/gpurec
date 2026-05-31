# Workflow Route Defaults Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/workflow/config.py` by moving production route/default audit
implementation into a private workflow helper while preserving the existing
public import surface from `gpurec.workflow.config`.

This is a policy-location extraction. It must not change `RunConfig`
normalization, checkpoint route metadata, CLI strict-route behavior, or the
public helper names already imported by tests and downstream callers.

## Moved Logic

Add `gpurec/workflow/_route_defaults.py` for the implementation behind:

- `production_default_route_contract()`;
- `production_default_route_contract_fields()`;
- `production_default_optimizer_config_overrides()`;
- `production_default_optimizer_setting_mismatches_from_route()`;
- `production_default_route_mismatches_from_route()`;
- `production_default_optimizer_setting_mismatches()`; and
- `effective_route_metadata()`.

`gpurec/workflow/config.py` keeps wrappers or aliases for those names so
existing imports from `gpurec.workflow.config` continue to work.

## Boundaries

- `config.py` remains the owner of `RunConfig`, `SamplingConfig`, scalar field
  normalization, and public workflow config helpers. JSON loading is now
  classified separately under the private `_config_io.py` helper.
- `_route_defaults.py` remains private and must not be added to
  `gpurec.workflow.__all__` or top-level `gpurec`.
- Route-contract constants should have one source of truth; hygiene tests
  should accept that source in the helper rather than requiring literal storage
  in `config.py`.
- CLI helpers and checkpoint code continue importing public helpers from
  `gpurec.workflow.config`.

## Verification Gates

```bash
python -m compileall -q gpurec/workflow
ruff check gpurec/workflow/config.py gpurec/workflow/_route_defaults.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_workflow.py -k "production_route or production_default or effective_route_metadata or run_config_auto_optimizer or effective_final_check_iters"
python -m pytest -q tests/unit/test_cli_workflow.py -k "production_default_route or mode_default_optimizer or route_status"
python -m pytest -q tests/unit/test_repository_hygiene.py -k "effective_route_metadata or config_template_reuses_production_optimizer_profile_source or run_config_reference or output_artifact_reference or production_optimization_guide"
git diff --check
```

## Acceptance Criteria

- Public route/default helper objects remain importable from
  `gpurec.workflow.config`.
- Existing route dictionaries return identical missing/mismatch tuples.
- `effective_route_metadata()` emits the same keys and JSON-compatible values.
- Strict CLI route checks keep recomputing stale or incomplete route evidence.

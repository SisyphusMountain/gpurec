# Workflow Schedules Refactor Plan, 2026-05-31

## Scope

Slim `gpurec/workflow/config.py` by moving Adagrad restart and loss-stop
schedule parsing into a private workflow helper while keeping
`gpurec.workflow.config` as the public import surface.

This is a parser-location extraction. It must not change `RunConfig`
normalization, Adagrad restart phase semantics, loss-stop formatting, workflow
optimizer behavior, or CLI config-template defaults.

## Candidate Moved Logic

Add `gpurec/workflow/_schedules.py` for:

- `AdagradRestartPhase`;
- `LossStopPhase`;
- `adagrad_restart_schedule_specs()`;
- `adagrad_restart_schedule_total_steps()`;
- `_normalize_adagrad_restart_schedule()`;
- `DEFAULT_NORMALIZED_ADAGRAD_RESTART_SCHEDULE`;
- `DEFAULT_ADAGRAD_RESTART_TOTAL_STEPS`;
- `loss_stop_schedule_specs()`; and
- `_normalize_optional_loss_stop_schedule()`.

`gpurec/workflow/config.py` keeps compatibility imports or wrappers so existing
imports from `gpurec.workflow.config` continue to work.

## Boundaries

- `_schedules.py` may import small validation helpers and route-default
  constants, but must not import `gpurec.workflow.config`,
  `gpurec.workflow.optimize`, or runtime orchestration helpers.
- `config.py` remains the owner of `RunConfig`, device/dtype parsing, and
  public facade names. JSON/path loading is classified separately under the
  private `_config_io.py` helper.
- Schedule dataclasses keep their legacy public identity through
  `gpurec.workflow.config` for introspection and pickle compatibility.
- Workflow runtime modules can keep importing from `config.py` unless moving to
  `_schedules.py` clearly reduces dependency weight without changing public
  behavior.
- Do not add `_schedules.py` to `gpurec.workflow.__all__` or top-level
  `gpurec`.

## Verification Gates

```bash
python -m compileall -q gpurec/workflow
ruff check gpurec/workflow/config.py gpurec/workflow/_schedules.py tests/unit/test_repository_hygiene.py
python -m pytest -q tests/unit/test_workflow.py -k "adagrad_restart_schedule or loss_stop_schedule or adagrad_restarts or effective_final_check_iters"
python -m pytest -q tests/unit/test_examples.py
python -m pytest -q tests/unit/test_repository_hygiene.py -k "workflow_submodule or run_config_reference"
git diff --check
```

## Acceptance Criteria

- Existing schedule strings normalize to exactly the same canonical form.
- Existing malformed schedule strings raise the same user-facing errors.
- Public config imports for schedule dataclasses and parser functions remain
  available.
- `AdagradRestartPhase` and `LossStopPhase` keep their historical
  `gpurec.workflow.config` pickle path.
- Workflow helper ownership docs and hygiene guards keep `_schedules.py`
  private.

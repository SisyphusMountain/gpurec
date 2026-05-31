# Refactor verification report - 2026-05-31

Tester subagent pass on branch `production`, current uncommitted worktree.

## Main-agent follow-up

The initial failures below were fixed in the shared worktree by restoring the
`gpurec.cli` compatibility facade, adding the thin `OptimizationRunner`
solver-stage compatibility methods, migrating hygiene checks to the split
owners, and tightening subprocess timeout coverage.

Follow-up verification on the same date:

- `python -m compileall -q gpurec` passed.
- `python -m pytest -q tests/unit/test_cli_workflow.py tests/unit/test_optimization_workflow.py tests/unit/test_specieswise_uniform.py tests/unit/test_workflow_artifacts.py` passed: 273 tests.
- `python -m pytest -q tests/unit/test_repository_hygiene.py tests/unit/test_artifacts_validator.py tests/unit/test_dependency_inventory.py tests/unit/test_long_validation_runner.py` passed: 126 tests.
- `python -m pytest -q tests/unit/test_workflow.py -k "optimization_runner_preserves_final_artifacts_when_staging_fails"` passed: 1 selected test.
- `git diff --check` passed.

## Commands run

- `git status --short`
- `python -m pytest tests/unit/test_cli_workflow.py tests/unit/test_optimization_workflow.py tests/unit/test_specieswise_uniform.py tests/unit/test_workflow_artifacts.py tests/unit/test_repository_hygiene.py`
- `python -m py_compile gpurec/workflow/optimize.py gpurec/workflow/_runtime_state.py gpurec/workflow/_artifacts.py gpurec/workflow/_evaluation.py gpurec/workflow/_step_execution.py gpurec/workflow/_step_plan.py gpurec/api/model.py gpurec/api/_model_config.py gpurec/api/_model_types.py`
- `python -m py_compile gpurec/cli.py gpurec/_cli_commands.py gpurec/_cli_helpers.py`
- `python -m pytest tests/unit/test_optimization_workflow.py tests/unit/test_specieswise_uniform.py tests/unit/test_workflow_artifacts.py tests/unit/test_repository_hygiene.py`
- `python -m pytest tests/unit/test_workflow.py -k "production_default_optimizer_config_overrides or production_default_route_contract_fields or run_config_defaults_to_hessian_sgd_for_genewise_mode or run_config_auto_optimizer_uses_adagrad_restarts_for_specieswise_mode or accepts_adam_fd_newton_for_genewise_mode or accepts_hessian_sgd_for_genewise_mode"`
- `python -m pytest tests/kernels/test_wave_step_forward_kernel.py -q`
- A small import probe for `gpurec`, `gpurec.workflow.config`, `gpurec.workflow.optimize`, and `gpurec.workflow._solver_stage`.

## Test-harness update applied

`tests/unit/test_optimization_workflow.py` imported `_ResumeState` and
`_resume_state_from_payload` from `gpurec.workflow.optimize`. The refactor moved
those private helpers to `gpurec.workflow._runtime_state`. I updated that test
import only. The related resume-state tests then passed in the non-CLI focused
run.

## Historical failures and risks before follow-up fixes

### CLI module export drift blocks CLI/workflow test collection

`tests/unit/test_cli_workflow.py` currently fails collection because `gpurec.cli`
is now a thin wrapper exposing only `main`:

```text
ImportError: cannot import name '_run_config_cli_override_fields' from 'gpurec.cli'
```

`tests/unit/test_workflow.py` also fails collection for the same class of issue:

```text
ImportError: cannot import name '_sampling_config_from_args' from 'gpurec.cli'
```

This is production/test API-surface drift from the CLI split. Either the CLI
module should intentionally re-export the helper surface that tests and docs
still use, or the tests/docs should be migrated to the new owning modules
(`gpurec._cli_helpers` / `gpurec._cli_commands`) with an explicit ownership
decision.

### Non-CLI focused set: 149 passed, 14 failed

After the test-only import update, this command ran:

```text
python -m pytest tests/unit/test_optimization_workflow.py tests/unit/test_specieswise_uniform.py tests/unit/test_workflow_artifacts.py tests/unit/test_repository_hygiene.py
```

Result:

```text
14 failed, 149 passed
```

Production-path failures:

- `tests/unit/test_optimization_workflow.py::test_specieswise_solver_warmup_starts_below_full_pi_budget`
- `tests/unit/test_optimization_workflow.py::test_specieswise_solver_warmup_is_skipped_when_not_lower_budget`

Both fail because `OptimizationRunner` no longer has `_uses_solver_warmup`.
The refactor appears to have moved this behavior into `SolverStageController`
(`gpurec/workflow/_solver_stage.py`). This should be resolved by either
restoring a compatibility method on `OptimizationRunner` or updating tests to
target the new solver-stage owner.

Hygiene/doc-surface failures:

- shared numeric validation expectation moved out of `gpurec/api/model.py`
- subprocess timeout hygiene found existing offenders in other tests
- script ownership matrix does not recognize `generate_dependency_inventory.py`
- CLI production-route/config source checks still inspect `gpurec/cli.py`
- `_FINAL_ARTIFACT_FILES` moved from `gpurec.workflow.optimize` to `_artifacts`
- CLI route gate wording constants moved out of `gpurec/cli.py`
- model static-state evaluator wrapper source shape changed
- log-every docs check still inspects `gpurec/cli.py`
- small-species backward limitation docs conflict with current source text
- supported env flag docs are missing `GPUREC_TORCH_SEED`

### Kernel parity check passed

`python -m pytest tests/kernels/test_wave_step_forward_kernel.py -q` passed:

```text
10 passed
```

This covers the replacement forward-kernel test file, including shared,
genewise-scalar, and genewise-specieswise leaf-logp modes.

### Public workflow/config import probe passed

The direct import probe passed and showed:

- `default_optimizer_for_mode("genewise") == "hessian-sgd"`
- `default_optimizer_for_mode("specieswise") == "adagrad-restarts"`
- `production_default_optimizer_config_overrides("genewise")` is importable and includes the expected hessian-SGD defaults.

## Notes on concurrent worktree movement

The first focused pytest collection saw a transient syntax error in
`gpurec/_cli_helpers.py` at the summary-info status f-string. A later
`py_compile` and `ast.parse` pass over the CLI files succeeded without any edit
from this subagent, so the shared worktree likely changed during the verification
pass. The current blocking CLI issue is export drift, not syntax.

## Recommendation

Do not treat the current tree as parity-verified yet. The forward kernel parity
check and non-CLI specieswise uniform tests are green, but CLI/workflow
collection is blocked and production ownership decisions are still needed for
the split helper modules. The highest-priority fixes are:

1. Decide whether `gpurec.cli` remains a compatibility facade for private helper
   tests/docs, or migrate those tests/docs to the new `_cli_*` owners.
2. Decide whether `OptimizationRunner._uses_solver_warmup` remains a compatibility
   method, or update tests to target `SolverStageController.uses_warmup()`.
3. Re-export or migrate artifact constants such as `_FINAL_ARTIFACT_FILES`.
4. Re-run the focused pytest command and CLI workflow tests after those ownership
   decisions land.

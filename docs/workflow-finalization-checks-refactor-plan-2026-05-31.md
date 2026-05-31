# Workflow Finalization Checks Refactor Plan, 2026-05-31

## Goal

Move the final iteration check logic out of `gpurec/workflow/_finalization.py`
while keeping finalization behavior and existing white-box import surfaces
stable.

## Scope

- Add `gpurec/workflow/_finalization_checks.py` for
  `_evaluate_final_iteration_check` and the genewise memory fallback helper.
- Keep `gpurec.workflow._finalization` exporting the same helper names through
  compatibility shims used by `optimize.py` and existing tests.
- Preserve the `optimize._sync_artifact_hooks()` patch surface for
  `build_alerax_workflow_model` and CUDA cache clearing by syncing those globals
  from `_finalization` into the new helper module before delegation.
- Leave `finalize_optimization` behavior unchanged.

## Verification

- Compile `_finalization.py`, `_finalization_checks.py`, and `optimize.py`.
- Ruff-check the same workflow files.
- Run `git diff --check`.
- Run focused workflow tests covering final check status, genewise fallback
  clade budgets, final artifacts, and import/export compatibility.

# Workflow Run Setup Refactor Plan - 2026-05-31

## Target

Reduce `gpurec/workflow/optimize.py` bloat by extracting pure run setup derivation from `OptimizationRunner.run()` into a private helper module.

## Scope

- Add a private workflow setup dataclass/helper for config-derived schedules and route flags.
- Move the adagrad restart schedule derivation, LBFGSB loss schedule derivation, batchwise optimizer flags, and optimization stop-step calculation out of `optimize.py`.
- Keep `OptimizationRunner` as the owner of model construction, filesystem setup, checkpoint hook injection, mutable run state, transition wiring, and finalization.

## Non-Goals

- Do not extract the main optimization loop.
- Do not move checkpoint loading/saving, `load_checkpoint` lambdas, or `_sync_artifact_hooks()`.
- Do not change adaptive rebatch state construction or the monkeypatchable constants exposed through `gpurec.workflow.optimize`.
- Do not remove private compatibility re-exports from `optimize.py`.

## Verification

- Compile touched workflow modules.
- Run focused workflow tests covering run modes, finalization, resume, adaptive rebatch, LBFGSB, batched LBFGS, and Hessian SGD behavior.
- Run helper-adjacent workflow artifact/policy tests.
- Run the broad non-slow CPU unit gate and the existing genewise integration smoke.

# Workflow Step Metrics Refactor Plan, 2026-05-31

## Target

`gpurec/workflow/_step_execution.py` still mixed optimizer control flow with
row-level telemetry assembly for bounded L-BFGS, batched L-BFGS, Hessian-SGD,
and Adagrad restart phases.

## Scope

- Add a workflow-private helper for optimizer-step metric schema assembly.
- Keep all optimizer stepping, closures, nonfinite recovery, theta restoration,
  solver configuration, cache handling, and FD-Newton control flow in
  `_step_execution.py`.
- Preserve every existing metric key, optional-key condition, and value type.
- Document the new private helper in the runtime surface ownership table.

## Verification

- Direct helper tests lock key/type parity for projected optimizers,
  batched-LBFGS summaries, Hessian-SGD budget labels, cached loss metrics, and
  Adagrad restart fields.
- Focused workflow tests cover emitted history rows for projected-LBFGS,
  L-BFGS-B, batched-LBFGS, Hessian-SGD, and Adagrad restarts.
- The broad CPU unit marker remains the final parity gate before commit.

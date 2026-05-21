# Documentation Map

This directory contains current workflow notes, audit records, and historical
performance logs.  Treat the main `README.md` as the user entry point and this
file as the map for deeper context.

## Current Operating Notes

- `lean-fast-path.md`: retained benchmark path and performance command for the
  lean branch.
- `refactor-simplification-plan-2026-05-21.md`: current simplification backlog
  for reducing duplicated evaluation paths, mode branches, scheduler
  alternatives, env selectors, and compatibility surface while retaining
  global/uniform, specieswise, and genewise modes.
- `gradient-likelihood-refactor-plan-2026-05-21.md`: focused plan for
  unifying E/Pi/root-likelihood/gradient computation behind one typed evaluator.
- `runtime-surface-pruning-plan-2026-05-21.md`: plan for pruning public/internal
  runtime surface, scheduler helpers, C++ extension exports, env variables,
  scripts, profiling entry points, and test-only helpers.
- `repo-wide-audit-2026-05-21.md`: tracked-file audit of untested code,
  unnecessary complexity, documentation gaps, and deletion candidates.
- `professionalization-audit-progress.tex`: running audit log for repository
  cleanup and verification work.
- `release-readiness.md`: release blockers, clean-checkout hygiene, and
  packaging verification gates.
- `../examples/minimal-run-config.json`: source-checkout/source-archive CUDA
  config/parser fixture for the retained optimized path, not a CPU fallback and
  not an end-to-end optimizer smoke while Pi backward requires `S > 256`.
- `../configs/hogenom_ccp_wandb.yaml`: checkout-local HOGENOM Hydra/W&B
  experiment config, not a portable example.
- `../notebooks/README.md`: ownership note for tracked notebooks.  The
  notebooks themselves are checkout-local HOGENOM analysis artifacts, not
  portable examples.

## Validation Notes

- `fresh-alerax-backtracking-validation.md`: small fresh AleRax backtracking
  validation run.
- `fresh-alerax-backtracking-validation-10000.md`: larger 10,000-sample
  backtracking validation run.
- `stochastic-backtracking-progress.md`: implementation notes for stochastic
  backtracking and RecPhyloXML output.

## Historical Performance And Research Logs

- `hogenom-ccp-performance-log.md`: chronological HOGENOM CCP performance log.
- `lean-performance-path-regression.md`: regression investigation from the lean
  branch transition.
- `core-simplification-suggestions.md`: historical cleanup snapshot for core
  CUDA and scheduling code. Revalidate each item before treating it as active
  backlog.
- `alerax-scaledvalue-followup-results.md`: AleRax ScaledValue follow-up
  optimization results.
- `second-order-optimization-opportunities.md`: notes on second-order and
  pseudo-second-order optimization options.

Historical notes preserve experimental context and may describe paths or
branch states that are no longer the recommended workflow.  Prefer the main
README, the CLI help, and the current operating notes for user-facing guidance.

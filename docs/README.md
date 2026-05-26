# Documentation Map

This directory contains current workflow notes, audit records, and historical
performance logs.  Treat the main `README.md` as the user entry point and this
file as the map for deeper context.

## Current Operating Notes

- `lean-fast-path.md`: retained benchmark path and performance command for the
  lean branch.
- `production-optimization-guide.md`: production-facing guide for the
  likelihood objective, gradient route, solver fidelity controls, genewise
  `hessian-sgd` default, specieswise `adagrad-restarts` default, and
  HOGENOM/`test_trees_1000` validation gates.
- `optimization-workflow-call-graph.md`: current production call graph for
  `validate-config`, AleRax preprocessing, resident likelihood/gradient
  evaluation, optimizer phases, checkpoints, and final artifacts.
- `output-artifacts.md`: operator-facing reference for optimization,
  checkpoint, rate-table, per-family likelihood, and sampling output files.
- `troubleshooting.md`: operator-facing preflight and failure triage for JSON
  configs, AleRax family inputs, CUDA memory, optimization status, and sampling
  binary setup.
- `simplification-opportunity-index-2026-05-21.md`: direct inventory of
  removable or mergeable alternative paths, with source-file evidence, retained
  behavior, and deletion gates.
- `simplification-execution-log-2026-05-21.md`: commit-by-commit execution log
  for attempted simplification tasks, including proposal coverage and
  verification gates.
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
- `../examples/README.md`: source-checkout/source-archive CUDA flat JSON
  config/parser fixtures for the genewise `hessian-sgd` and specieswise
  `adagrad-restarts` production defaults. This example directory is not a CPU fallback
  and not an end-to-end optimizer smoke while Pi backward requires `S > 256`.
- `../configs/hogenom_ccp_wandb.yaml`: checkout-local HOGENOM Hydra/W&B
  experiment config, not a portable example.
- `../configs/README.md`: config ownership note separating installed flat JSON
  workflow configs from checkout-local Hydra/HOGENOM experiment inputs.
- `../notebooks/README.md`: ownership note for tracked notebooks.  The
  notebooks themselves are checkout-local HOGENOM analysis artifacts, not
  portable examples.
- `../profiling/README.md`: source-checkout profiling ownership note for
  supported benchmark entry points, local-data assumptions, and ignored
  artifact policy.

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

## Low-Level API Stability

The supported package entry points are the high-level classes and workflow
helpers exported from `gpurec`, `gpurec.api`, and `gpurec.workflow`.
`gpurec.core` is an implementation namespace for preprocessing, likelihood
kernels, scheduling, and white-box tests.  Direct imports from `gpurec.core`
are unstable unless a helper is explicitly documented as supported; tests may
use internals to guard behavior, but those imports are not public API evidence.
The current narrow exception is direct `GeneDataset(..., leaf_species_maps=...)`
construction for custom gene-leaf to species mapping when `from_trees` prefix
fallback and AleRax `mapping` entries are not sufficient.

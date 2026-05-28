# Documentation Map

This directory contains current workflow notes, audit records, and historical
performance logs.  Treat the main `README.md` as the user entry point and this
file as the map for deeper context.
This map separates stable user workflows from HOGENOM-only research scripts so
production operators can stay on supported surfaces.

## Current Operating Notes

- `lean-fast-path.md`: retained benchmark path and performance command for the
  lean branch.
- `production-optimization-guide.md`: production-facing guide for the
  likelihood objective, gradient route, solver fidelity controls, genewise
  `hessian-sgd` default, specieswise `adagrad-restarts` default, and
  HOGENOM/`test_trees_1000` validation gates.
- `run-config-reference.md`: maintained field-by-field `RunConfig` and CLI
  flag reference, including optimizer scoping and completed-checkpoint resume
  semantics.
- `platform-matrix.md`: explicit Linux/Python/PyTorch/Triton/GPU/rust-native
  support and installation matrix with recommended preflight commands.
- `api-contract.md`: public API and CLI contract for versioned user-visible
  behavior, including command set, import surface, config precedence, environment
  variables, output artifacts, exit behavior, and deprecation policy.
- `input-preparation.md`: bioinformatician-facing guide for species trees,
  AleRax `[FAMILIES]` records, gene-tree files, mapping files, JSON config path
  resolution, and `validate-config --check-preprocess` preflight validation.
- `production-likelihood-optimization-strategy.md`: production policy for
  choosing and validating genewise and specieswise likelihood optimization
  routes without overfitting to HOGENOM or `test_trees_1000`.
- `production-grade-slimdown-plan.md`: operating plan for keeping production
  quality work tied to active Python line-count reduction, smaller runtime
  surfaces, and centralized diagnostics instead of exception-handling sprawl.
- `file-size-reduction-analysis-2026-05-28.md`: file-by-file hotspot analysis
  from focused mini-agent reviews, with reduction ranges and verification gates.
- `known-limitations.md`: explicit production constraints for CUDA readiness,
  size-dependent backward gates, parser subset boundaries, and native artifact
  dependencies.
- `glossary.md`: stable definitions for D/T/L terms, optimization modes, route
  metadata, solver budgets, and checkpoint terminology used in user docs.
- `workflow-examples/`: a runnable, deterministic end-to-end mini dataset plus
  Snakemake, Nextflow, and Slurm workflow starter templates.
- `long-validation-workflow.md`: reproducible pre-release validation bundle over
  the public end-to-end dataset with report and threshold checks.
- `validation-envelope.md`: release-candidate acceptance envelope for public
  long-validation counts, status, runtime/NLL guardrails, and evidence capture.
- `optimization-workflow-call-graph.md`: current production call graph for
  `validate-config`, AleRax preprocessing, resident likelihood/gradient
  evaluation, optimizer phases, checkpoints, and final artifacts.
- `output-artifacts.md`: operator-facing reference for optimization,
  checkpoint, rate-table, per-family likelihood, and sampling output files.
- `troubleshooting.md`: operator-facing preflight and failure triage for JSON
  configs, AleRax family inputs, CUDA memory, optimization status, and sampling
  binary setup.
- `bioinformatics-quickstart.md`: first-run end-to-end path for new users:
  install, validate inputs, optimize, inspect outputs, and sample.
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
- `support-policy.md`: explicit production support scope, platform envelope,
  release/patch policy, and support evidence requirements.
- `versioning-policy.md`: semantic-versioning rules, compatibility commitments,
  and release-line policy for user-facing contract changes.
- `publication-checklist.md`: publication-facing checklist for citation,
  reproducibility artifacts, validation evidence, and reporting requirements.
- `release-notes.md`: release note template with migration notes and release
  limitation blocks.
- `../examples/README.md`: source-checkout/source-archive CUDA flat JSON
  config/parser fixtures for the genewise `hessian-sgd` and specieswise
  `adagrad-restarts` production defaults. This example directory is not a CPU fallback,
  is not an end-to-end optimizer smoke, and Pi backward requires `S > 256`.
- `../docs/workflow-examples/input-validation-fixtures/`: minimal public
  AleRax input fixtures for `validate-inputs`, including both valid and failing
  cases.
- `../Dockerfile`: minimal CUDA runtime/deployment image with native artifact
  build and environment defaults.
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

The supported package entry points are the high-level classes, workflow
helpers, backtracking helpers, and entropy helpers exported from `gpurec`,
`gpurec.api`, `gpurec.workflow`, `gpurec.backtracking`, and `gpurec.entropy`.
`gpurec.core` is an implementation namespace for preprocessing, likelihood
kernels, scheduling, and white-box tests.  Direct imports from `gpurec.core`
are unstable unless a helper is explicitly documented as supported; tests may
use internals to guard behavior, but those imports are not public API evidence.
The current narrow exception is direct `GeneDataset(..., leaf_species_maps=...)`
construction for custom gene-leaf to species mapping when `from_trees` prefix
fallback and AleRax `mapping` entries are not sufficient.
For a complete public-interface statement, see [`api-contract.md`](api-contract.md).

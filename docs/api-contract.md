# Public API And CLI Contract (v1)

This document defines the stable interfaces for external users of `gpurec`
(versioned as API contract v1).  The contract is composed of the supported
Python import surface, CLI surface, configuration schema, environment contract,
output files, and gate behaviors.  Anything not listed here is explicitly unstable.

## API Surface Scope

### Supported Python imports

`gpurec` exposes the following top-level names:

- `GeneReconModel`, `UniformChunkedReconModel`
- `UniformChunkMetadata`, `ActiveFamilyBatch`, `BatchMetadata`, `FamilyInput`,
  `ReconciliationState`
- `EVENT_KEYS`, `ensure_backtracking_available`, `export_backtracking_input`,
  `recphyloxml_event_counts`, `sample_backtracking_summaries`,
  `sample_recphyloxml`, `sample_recphyloxmls`, `sample_recphyloxmls_to_dir`
- `compute_reconciliation_entropy`, `reconciliation_entropy_from_payload`
- `RunConfig`, `SamplingConfig`, `OptimizationResult`, `OptimizationRunner`,
  `SamplingResult`, `SamplingRunner`, `optimize`, `sample`

The supported package-level modules are `gpurec.api`, `gpurec.workflow`,
`gpurec.backtracking`, and `gpurec.entropy`.  Their public surfaces are defined
by this contract and the module docstrings.

The following remains unstable:

- `gpurec.core` is an implementation namespace and not a supported import
  surface unless explicitly documented as a stable low-level helper.
- Underscored names and any behavior not described in this or an adjacent
  reference doc are considered unstable and may change without notice.

### Public workflow and model behavior guarantees

- Public workflow helpers are lazy-loaded from `gpurec.workflow` and include the
  listed `RunConfig`, `SamplingConfig`, `optimize`, and `sample` symbols.
- `optimize(...)` and `sample(...)` return typed result objects (`OptimizationResult`,
  `SamplingResult`) and raise Python exceptions for validation/runtime errors.
- `RunConfig.from_dict(...)` and `SamplingConfig.from_dict(...)` reject unknown
  fields and unsupported value shapes before model construction.
- CLI and Python surfaces share the same normalized field naming/alias rules:
  mode and optimizer names are case-normalized and underscore aliases for optimizer
  names are accepted (for example `hessian_sgd` and `adagrad_restarts`).

### Supported CLI contract

Supported commands are:

- `optimize`
- `validate-config`
- `validate-inputs`
- `sample`
- `run`
- `backtrack-check`
- `preprocess-check`
- `doctor`
- `checkpoint-info`
- `summary-info`
- `config-template`

No other command names are part of the public CLI contract.

Behavioral guarantees:

- `optimize` and `run` require at least a valid optimization config.
- `validate-config` and `validate-inputs` are CPU-safe preflight commands.
- `sample` reads a checkpoint and writes sampling artifacts.
- `run` combines optimization and sampling and exits before sampling when the
  optimization route gates require it.
- `backtrack-check`, `preprocess-check`, and `doctor` are lightweight readiness
  probes.
- `checkpoint-info` and `summary-info` inspect outputs without constructing
  likelihood models.
- `config-template` writes (or prints) a mode-specific template `RunConfig`.

## RunConfig Contract And CLI Precedence

Configuration accepts either:

1) JSON via `--config <path>`
2) stdin via `--config -`
3) explicit CLI flags in command arguments

Precedence is explicit CLI flags override values loaded from `--config`.

`--config` consumes a flat JSON object, not Hydra YAML. Relative paths follow:

- JSON config file paths resolve relative to that config file directory.
- stdin (`--config -`) and explicit CLI flags resolve relative to current working
  directory.

The resolved config is validated by `RunConfig.from_dict(...)` before any model
construction.

## Output Contracts

The following artifacts are stable output filenames when using supported workflow
commands:

- `run_config.json`
- `history.jsonl`
- `optimization_history.csv`
- `summary.json`
- `run_manifest.json`
- `rates_final.tsv`
- `per_fam_likelihoods.tsv` (genewise)
- `theta_final.pt`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `reconciliations/event_counts.tsv`
- `reconciliations/totalSpeciesEventCounts.txt`
- `reconciliations/totalTransfers.txt`
- `reconciliations/all/<family>_eventCounts_<seed>.txt`
- `reconciliations/all/<family>_sample_<seed>.xml`

Artifact semantics are in `docs/output-artifacts.md`, and route/gating evidence
fields are serialized in `summary.json`, `history.jsonl`, checkpoints,
`--summary-info`, and `--checkpoint-info` payloads.

## CLI output modes

`--json` mode is supported by:

- `validate-config`
- `validate-inputs`
- `backtrack-check`
- `preprocess-check`
- `doctor`
- `checkpoint-info`
- `summary-info`

JSON mode emits single JSON objects with stable keys for automated parsing.
Text mode remains the default for interactive use.

## Environment Surface

The following environment variables are part of the public runtime contract:

- `GPUREC_BACKTRACK_BIN`
- `GPUREC_BACKTRACK_NATIVE_LIB`
- `GPUREC_PREPROCESS_BIN`
- `GPUREC_PREPROCESS_NATIVE_LIB`
- `GPUREC_MEMORY_POLICY_FRACTION`
- `GPUREC_MEMORY_POLICY_RESERVE_GIB`
- `GPUREC_ALERAX_COMPAT`

No other `GPUREC_*` variables are documented as public support.

## Exit behavior

- Exit status `0` is used for successful command completion.
- Exit status `1` is used for runtime and route-validation failures, including:
  gate failures (`--require-...`), invalid checkpoint status for sample/checkpoint
  flow, non-converged requirements, and non-ok final-check requirements.
- Exit status `2` is used for CLI parse/config errors emitted by argparse or
  malformed command usage.

Status text remains machine-parseable (key-value fields) and JSON mode is the
required machine path for automation.

## Contract Stability, versioning, deprecation

Compatibility policy scope covers config fields, CLI flags, Python imports, and
output artifacts.

- This contract currently applies to the current `gpurec` release line and is
  versioned as API contract `v1`.
- Future contract additions are added by release notes and docs; removals or
  behavior changes are only introduced behind deprecation and release
  communication.
- Backward-incompatible behavior changes are announced explicitly in release
  notes, including migration steps and replacement call patterns.
- A feature can enter deprecation by adding deprecation warnings in docs and
  retaining old behavior for at least one full release cycle where practical.
- Release notes must include migration notes for removed or replaced supported
  behavior.

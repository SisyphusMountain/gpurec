# Bioinformatics Quickstart

This page is the shortest path for running `gpurec` end-to-end on public
AleRax-style inputs.

## Run Lifecycle

Follow this lifecycle in order: create config, validate, run, resume, inspect,
sample, archive.

## Install

## Installation Decision Tree

- If you are running from a source checkout or source archive with Rust/Cargo
  available: use `pip install .` and continue with `gpurec doctor`.
- If you are using a wheel-only environment: set
  `GPUREC_PREPROCESS_NATIVE_LIB` and `GPUREC_BACKTRACK_BIN`, then run
  `gpurec preprocess-check`, `gpurec backtrack-check`, and `gpurec doctor`.
- If you are deploying to cluster/container workflows: start from the project
  `Dockerfile` or your site CUDA base image, then run the same readiness checks.
- If your lab standardizes on conda/mamba: capture an explicit conda
  environment file and run the same readiness checks after environment solve.
- If you require fully offline installation: treat it as unsupported for the
  default production contract and follow the offline policy in
  `docs/platform-matrix.md`.

From a source checkout:

```bash
pip install .
```

For workflow usage with prebuilt wheels, confirm CUDA/PyTorch and set native
artifact paths (or build from source) before running any preprocessing or
optimization:

```bash
export GPUREC_PREPROCESS_NATIVE_LIB=/path/to/libgpurec_preprocess.so
export GPUREC_BACKTRACK_BIN=/path/to/gpurec-backtrack
```

Then verify availability without reading input files:

```bash
gpurec doctor
gpurec preprocess-check
gpurec backtrack-check
```

## Structured JSON Mode

For workflow-manager automation, use `--json` instead of parsing status lines:

```bash
gpurec doctor --json
gpurec validate-config --config run.json --json
gpurec summary-info --summary output_gpurec/summary.json --json
gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt --json
```

## Create Config

Start from a mode-specific template and edit the paths:

```bash
gpurec config-template --mode genewise --output run.json
```

## Validate

Prepare a rooted species tree and an AleRax `[FAMILIES]` file, then preflight
the config:

```bash
gpurec validate-config --config run.json --require-mode-default-optimizer
gpurec validate-config --config run.json --require-cuda-backward-ready --check-preprocess
```

Use `--require-production-default-route` instead of
`--require-mode-default-optimizer` only when you also require the shipped
HOGENOM/`test_trees_1000` objective, gradient, and resident-batch route.

## Run

Start from the same `run.json` used for preflight:

```bash
gpurec optimize --config run.json --require-mode-default-optimizer
```

Add `--require-final-check-ok` to make final high-fidelity likelihood/gradient
validation failures gate success explicitly.

## Resume

To continue an interrupted run, point the same config at a checkpoint:

```bash
gpurec optimize --config run.json --resume-from output_gpurec/checkpoints/latest.pt --require-mode-default-optimizer
```

## Inspect

Every optimization writes checkpoints, history, and summary metadata:

```bash
gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt
gpurec summary-info --summary output_gpurec/summary.json --require-converged
gpurec summary-info --summary output_gpurec/summary.json --require-final-check-ok
```

Add `--require-production-default-route` to preflight the same route metadata
if the workflow is part of a production release check.

## Sample

Use the output checkpoint for sampling-ready state:

```bash
gpurec sample --checkpoint output_gpurec/checkpoints/latest.pt --samples 50
```

## RNG Behavior

Set a sampling `--seed` and keep it in run records when reproducibility is
required.

Keep the output directory and RNG settings explicit for reproducibility.

## Archive

Archive the run directory artifacts needed for reproducibility and triage:

- `run_config.json`
- `run_manifest.json`
- `summary.json`
- `history.jsonl`
- `checkpoints/`
- sampled RecPhyloXML and event TSV outputs

## Next checks

Review parser and workflow limits before scaling:

- `docs/known-limitations.md` for constraints like CUDA readiness, `S > 256`,
  supported Newick subset, and native artifact requirements.
- `docs/input-preparation.md` for mapping and file-format details.
- `docs/run-config-reference.md` for every configuration field and CLI alias.

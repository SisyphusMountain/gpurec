# Bioinformatics Quickstart

This page is the shortest path for running `gpurec` end-to-end on public
AleRax-style inputs.

## Install

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

## Validate Inputs

1. Prepare a rooted species tree and an AleRax `[FAMILIES]` file.
2. Generate a mode-specific starter config and edit file paths:

```bash
gpurec config-template --mode genewise --output run.json
```

3. Preflight both static and parser/compute contracts:

```bash
gpurec validate-config --config run.json --require-mode-default-optimizer
gpurec validate-config --config run.json --require-cuda-backward-ready --check-preprocess
```

Use `--require-production-default-route` instead of
`--require-mode-default-optimizer` only when you also require the shipped
HOGENOM/`test_trees_1000` objective, gradient, and resident-batch route.

## Run Optimization

Start from the same `run.json` used for preflight:

```bash
gpurec optimize --config run.json --require-mode-default-optimizer
```

Add `--require-final-check-ok` to make final high-fidelity likelihood/gradient
validation failures gate success explicitly.

## Inspect Output

Every optimization writes checkpoints, history, and summary metadata:

```bash
gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt
gpurec summary-info --summary output_gpurec/summary.json --require-converged
gpurec summary-info --summary output_gpurec/summary.json --require-final-check-ok
```

Add `--require-production-default-route` to preflight the same route metadata
if the workflow is part of a production release check.

## Sample Reconciliations

Use the output checkpoint for sampling-ready state:

```bash
gpurec sample --checkpoint output_gpurec/checkpoints/latest.pt --samples 50
```

Keep the output directory and RNG settings explicit for reproducibility.

## Next checks

Review parser and workflow limits before scaling:

- `docs/known-limitations.md` for constraints like CUDA readiness, `S > 256`,
  supported Newick subset, and native artifact requirements.
- `docs/input-preparation.md` for mapping and file-format details.
- `docs/run-config-reference.md` for every configuration field and CLI alias.

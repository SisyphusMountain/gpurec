# Validation Envelope

This page defines the reproducible validation envelope for the public long
validation dataset in `docs/workflow-examples/end-to-end-tutorial/`.

These bounds are acceptance gates for release-candidate validation evidence and
are intentionally broad enough to avoid overfitting to one workstation.

## Dataset Identity

- Config: `docs/workflow-examples/end-to-end-tutorial/run.json`
- Species tree: `261` species (`S > 256` CUDA backward-ready gate)
- Families: `2`
- Sampling shape target: `families_sampled * samples_per_family`

## Required Outcome

- Convergence status must be explicitly recorded from `summary.status`.
- `summary.status == "converged"`
- `summary.reason` is present and documented in the validation report
- Sampling output shape must satisfy the XML-count identity.
- `reconciliations/summary.json` exists and has consistent XML count:
  `xml_files == families_sampled * samples_per_family`
- Output artifacts pass `scripts/validate_output_artifacts.py`

## Numeric/Runtime Envelope

Use `scripts/run_long_validation.py` with these default guardrails:

- `--min-families 2`
- `--min-species 261`
- `--max-elapsed-s 3600`
- `--observed-peak-memory-gib ...` and `--max-observed-peak-memory-gib ...`
  when release evidence includes an explicit peak memory envelope.
- `--max-final-nll-bits-abs 1000000000`

The generated report (`gpurec.long_validation_report.v1`) should capture runtime
envelope, peak memory evidence (when measured), and final NLL range bounds.
It is benchmark evidence, not a guaranteed runtime/performance contract for all
GPU models, drivers, or CUDA stacks.

## Publication Evidence

Before publishing a release candidate, archive:

1. The long-validation report JSON from `scripts/run_long_validation.py`.
2. `summary.json`, `history.jsonl`, `run_manifest.json`, and sampled
   reconciliation summary from the validated run directory.
3. Dependency and supply-chain evidence from
   `scripts/generate_dependency_inventory.py` and audit outputs listed in
   `docs/release-readiness.md`.

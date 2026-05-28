# Publication Checklist

Use this checklist before submitting results that depend on `gpurec`.

## Citation And Version Evidence

- Record the exact `gpurec` version/tag used for each result set.
- Include `CITATION.cff` metadata in project records.
- Record the command outputs from:
  - `gpurec doctor --json`
  - `gpurec summary-info --summary <out_dir>/summary.json --json`

## Reproducibility Artifacts

- Archive `run_config.json`, `run_manifest.json`, `summary.json`,
  `history.jsonl`, `checkpoints/`, and the selected sampling checkpoint.
- Archive checksums and provenance evidence for the publication artifact bundle.
- Archive native-toolchain provenance when applicable (Rust/Cargo versions,
  backtracking binary path/version, preprocessing native library path/version).
- Keep dependency inventory and vulnerability scan outputs generated during
  release-candidate checks.

## Validation Gates

- Confirm preflight succeeded (`validate-config --check-preprocess`,
  and where required `--require-cuda-backward-ready`).
- Confirm release-candidate long validation satisfied
  [`validation-envelope.md`](validation-envelope.md).
- Confirm output artifact sanity checks passed using
  `scripts/validate_output_artifacts.py`.

## Reporting Requirements

- Cite known limitations from [`known-limitations.md`](known-limitations.md) in
  manuscripts or internal reports when they affect interpretation.
- Include migration notes from [`release-notes.md`](release-notes.md) if results
  span multiple release tags.

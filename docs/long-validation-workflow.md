# Long Validation Workflow

This workflow is the maintainers' reproducible pre-release validation bundle for
the public end-to-end dataset under
`docs/workflow-examples/end-to-end-tutorial/`.

It is intended to provide externally reproducible evidence for:

- input parsing and preprocessing validation,
- CUDA optimization route execution,
- output artifact publication and structural sanity,
- checkpoint-based RecPhyloXML sampling,
- run metadata capture for later audit.

## Command

Run from repository root:

```bash
python scripts/run_long_validation.py \
  --config docs/workflow-examples/end-to-end-tutorial/run.json \
  --output-report dist/long-validation-report.json \
  --samples 2 \
  --seed 42 \
  --checkpoint-choice best \
  --min-families 2 \
  --min-species 261 \
  --max-elapsed-s 3600 \
  --observed-peak-memory-gib 20.5 \
  --max-observed-peak-memory-gib 24 \
  --max-final-nll-bits-abs 1000000000
```

The script runs:

1. `gpurec doctor --json`
2. `gpurec validate-config --check-preprocess --require-cuda-backward-ready`
3. `gpurec optimize --require-final-check-ok`
4. `gpurec summary-info --require-converged --require-final-check-ok`
5. `gpurec sample --checkpoint ...`
6. `scripts/validate_output_artifacts.py` over `summary.json`,
   `history.jsonl`, checkpoint, `run_manifest.json`, and TSV outputs.

## Report Artifact

The output report JSON (`gpurec.long_validation_report.v1`) captures:

- command list, return codes, elapsed time, and captured stdout/stderr,
- observed summary metrics (`families`, `species`, `final_nll_bits`, `elapsed_s`),
- optional observed peak memory (`observed_peak_memory_gib`) from external GPU
  telemetry such as `nvidia-smi`,
- sampling shape (`xml_files` vs expected `families_sampled * samples_per_family`),
- selected threshold settings used for the run.

Treat this report as benchmark evidence, not a hard performance guarantee across
all hardware.

For release-candidate acceptance bounds and evidence requirements, see
[validation-envelope.md](validation-envelope.md).

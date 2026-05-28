# Workflow Examples

These examples show the same end-to-end gpurec sequence in different runtime
contexts:

- `end-to-end-tutorial/`: tracked mini public dataset and a deterministic, ready-to-run `run.json`
- `snakemake/`: preflight → optimize → sample workflow
- `nextflow/`: equivalent process chain for Nextflow pipelines
- `slurm/`: simple scheduler entrypoint with checkpoint resume logic
- `input-validation-fixtures/`: tiny valid and invalid AleRax input fixtures for
  `validate-inputs` smoke checks with expected parsing and preprocessing failures.

Snakemake and Nextflow examples are production workflow-manager references that:

- fail fast on bad config during preflight
- resume from a checkpoint after interruptions
- reject non-converged outputs before sampling

From the repository root:

```bash
cd docs/workflow-examples/end-to-end-tutorial
python generate_dataset.py
```

Re-run the tutorial command sequence from that folder to reproduce the expected
workflow shape before adapting settings for production datasets.

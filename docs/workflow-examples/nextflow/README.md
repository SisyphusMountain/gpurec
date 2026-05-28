# Nextflow Example

From `docs/workflow-examples/nextflow` run:

```bash
nextflow run main.nf -resume
```

The process chain mirrors the same checkpoints used by the Snakemake example:

1. `validate` (`gpurec validate-config --check-preprocess` preflight)
2. `optimize` (gated production run with `--require-converged` and `--require-final-check-ok`)
3. `inspect` (`summary-info` and `checkpoint-info` hard gates with `--require-converged` and `--require-final-check-ok`)
4. `sample` (checkpoint-based sampling)

Tune sample count and random seed through Nextflow params:

```bash
nextflow run main.nf --sample_count 5 --sample_seed 7 -resume
```

# Nextflow Example

From `docs/workflow-examples/nextflow` run:

```bash
nextflow run main.nf -resume
```

The process chain mirrors the same checkpoints used by the Snakemake example:

1. `validate` (preflight)
2. `optimize` (gated production run)
3. `sample` (checkpoint-based sampling)

Tune sample count and random seed through Nextflow params:

```bash
nextflow run main.nf --sample_count 5 --sample_seed 7 -resume
```

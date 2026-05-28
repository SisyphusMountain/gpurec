# Snakemake Example

From `docs/workflow-examples/snakemake` run:

```bash
snakemake --cores 1
```

The workflow runs:

1. `gpurec validate-config` (hard gate for CUDA-ready backward and parser checks)
2. `gpurec optimize` with production-route gates
3. `gpurec sample` from the best checkpoint

Each step writes marker/output artifacts in `output_gpurec/` so retries can
reuse work already produced.

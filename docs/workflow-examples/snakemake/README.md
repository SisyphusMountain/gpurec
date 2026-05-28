# Snakemake Example

From `docs/workflow-examples/snakemake` run:

```bash
snakemake --cores 1
```

The workflow runs:

1. `gpurec validate-config` (hard gate for CUDA-ready backward and parser checks)
2. `gpurec optimize` with production-route and convergence gates
3. `gpurec summary-info` and `gpurec checkpoint-info` hard gates
4. `gpurec sample` from the best checkpoint

Each step writes marker/output artifacts in `output_gpurec/` so retries can
reuse work already produced.

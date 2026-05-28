# Snakemake Example

From `docs/workflow-examples/snakemake` run:

```bash
snakemake --cores 1
```

The workflow runs:

1. Fail fast on bad config with `gpurec validate-config --check-preprocess`
   (hard gate for CUDA-ready backward and parser checks)
2. `gpurec optimize` with production-route and convergence gates (`--require-converged` and `--require-final-check-ok`)
3. `gpurec summary-info` and `gpurec checkpoint-info` hard gates (`--require-converged` and `--require-final-check-ok`)
4. `gpurec sample` from the best checkpoint

This workflow is designed to reject non-converged outputs.

Each step writes marker/output artifacts in `output_gpurec/` so retries can
reuse work already produced.

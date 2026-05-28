# Slurm Example

From `docs/workflow-examples/slurm`, submit:

```bash
sbatch run-gpurec.sbatch
```

The script:

1. Runs `gpurec validate-config --check-preprocess` with the same hard preflight gates.
2. Runs `gpurec optimize`; resume from `output_gpurec/checkpoints/latest.pt`
   if present, with convergence/final-check gates enabled.
3. Runs `gpurec summary-info` and `gpurec checkpoint-info` hard gates.
4. Runs `gpurec sample` on the best checkpoint.

Update resource requests (`--gres`, `--cpus-per-task`, memory, queue) for your
cluster profile.

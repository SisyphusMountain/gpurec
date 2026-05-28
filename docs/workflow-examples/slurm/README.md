# Slurm Example

From `docs/workflow-examples/slurm`, submit:

```bash
sbatch run-gpurec.sbatch
```

The script:

1. Validate inputs by running `gpurec validate-config --check-preprocess`
   with the same hard preflight gates.
2. Runs `gpurec optimize`; resume from `output_gpurec/checkpoints/latest.pt`
   if present, with convergence/final-check gates enabled.
3. Runs `gpurec summary-info` and `gpurec checkpoint-info` hard gates.
4. Runs `gpurec sample` on the best checkpoint.

Update resource requests (`--gres`, `--cpus-per-task`, memory, queue) for your
cluster profile.

Document environment modules, CUDA visibility, and output paths in your Slurm
submission wrapper so runs are reproducible across cluster partitions.
Choose local scratch for hot intermediates and shared network storage for
retained outputs and publication bundles.
Document thread controls for preprocessing and PyTorch execution (for example
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `TORCH_NUM_THREADS`) in the Slurm
wrapper or module profile.

When asking for support, collect `run_config.json`, `summary.json`,
`history.jsonl`, and full stderr/stdout logs from the scheduler job.

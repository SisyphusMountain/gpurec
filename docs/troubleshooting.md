# Troubleshooting Production Runs

This guide is for operators running the supported AleRax-style `gpurec`
workflow. It focuses on failures that can happen before or after the CUDA
likelihood/gradient route starts.

## Preflight First

Run the cheap path/reference preflight before optimization:

```bash
gpurec validate-config --config run.json
```

This checks JSON fields, selected family records, mapping files, referenced gene
trees, and the resolved optimizer route. It does not construct the CUDA
likelihood model.

Then run the heavier CPU parser preflight:

```bash
gpurec validate-config --config run.json --check-preprocess
```

Use this when preparing a new dataset or after editing Newick trees or mapping
files. It runs the retained Rust parser on the selected families and catches
tree/mapping problems before a full optimization run. It also reports
`cuda_backward_ready`; if this is `false`, the species tree is too small for
the retained CUDA backward path (`S > 256`) even though parser validation
succeeded. Add `--require-cuda-backward-ready` to make that condition fail the
preflight command.

## Common Preflight Failures

| Symptom | Likely cause | Next action |
|---|---|---|
| `unknown RunConfig field` | A JSON config contains an obsolete or misspelled key. | Start from `gpurec config-template --mode genewise` or `--mode specieswise`, then copy only supported fields. |
| `missing gene-tree path` | A `[FAMILIES]` entry points at a gene tree that does not exist relative to the family file. | Fix the `starting_gene_tree` path or move the file next to the family list. |
| Mapping or species-name errors during `--check-preprocess` | Gene-tree leaves, mapping labels, or species-tree labels disagree. | Check the map file first, then verify every mapped species appears in the species tree. |
| `cuda_backward_ready=false` | CPU preprocessing succeeded, but the species tree does not satisfy the retained CUDA backward size gate. | Use a production species tree with more than 256 postorder species nodes before running `gpurec optimize`. |
| A CUDA error appears during `validate-config` | The command probably reached `optimize`, not preflight. | Re-run only `validate-config`; the default preflight does not build `GeneReconModel` or touch CUDA. |

## Optimization Status

Read `summary.json` first. It records `status`, `reason`, final objective and
gradient diagnostics, final-check likelihood/gradient deltas when available,
and the effective optimizer/batch/solver route that produced the run.
For a terminal view of the same fields, use:

```bash
gpurec summary-info --summary output/summary.json
```

Use `--require-converged` when a shell pipeline, Snakemake rule, or Nextflow
process should fail unless the run ended with `status=converged`.
For combined optimize-and-sample pipelines, use `gpurec run --require-converged`
to print the optimization status and stop before sampling unless optimization
converged.
Use `checkpoint-info` when the final summary is unavailable or you need to
inspect a resume target directly:

```bash
gpurec checkpoint-info --checkpoint output/checkpoints/latest.pt
```

| Status and reason | Meaning | Next action |
|---|---|---|
| `converged` | The workflow met its stopping condition. | Use `checkpoints/best.pt` for sampling unless you intentionally need the last checkpoint. |
| `not_converged` / `max_steps` | The run exhausted the configured step budget. | Inspect `history.jsonl`, `grad/projected_inf`, and `best_nll_bits`; increase `steps` and resume from `checkpoints/latest.pt` if the trajectory is still improving. |
| `failed` / `nonfinite_objective_or_gradient` | A mandatory objective/gradient evaluation became nonfinite. | Keep the failed artifacts for debugging, then retry from an earlier checkpoint with a smaller learning rate or a more conservative optimizer route. |
| `adagrad_restart_schedule_complete` | The specieswise restart ladder finished all scheduled phases. | Treat it like a completed multifidelity run and inspect the fixed128 final-check diagnostics. |

To continue a run:

```bash
gpurec optimize --config run.json --resume-from output/checkpoints/latest.pt
```

Set `steps` above the checkpoint `next_step`; otherwise resume only refreshes
final artifacts.

## CUDA Memory Or Runtime Failures

The optimized likelihood and gradient path currently requires CUDA. If model
construction or final evaluation runs out of memory:

- Lower `clade_budget` to build smaller resident batches.
- Set a positive `family_chunk_size` when one large all-family batch is too
  large.
- Keep `batch_packing=depth_first_fit` unless a dataset-specific benchmark
  supports another packing policy.
- Preserve the failed `summary.json`, `history.jsonl`, and checkpoint before
  retrying so the run can be compared.

Genewise final evaluation has smaller-clade fallback logic for retryable memory
errors. If every fallback fails, reduce `clade_budget` in the config and resume.

## Sampling Failures

Sampling uses the Rust backtracking binary. If `gpurec sample`, `gpurec run`,
or `gpurec backtrack-check` reports a missing binary, set
`GPUREC_BACKTRACK_BIN` or pass `--backtrack-binary`:

```bash
export GPUREC_BACKTRACK_BIN=/path/to/gpurec-backtrack
gpurec sample --checkpoint output/checkpoints/best.pt --backtrack-binary /path/to/gpurec-backtrack
```

Use `checkpoints/best.pt` when it exists. Do not pass `theta_final.pt` as a
sampling checkpoint; it is only a raw tensor export and does not contain
configuration, family ordering, species ordering, or optimizer state.

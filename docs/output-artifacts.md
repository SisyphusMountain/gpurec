# Output Artifact Reference

This page describes the files written by the supported `gpurec optimize`,
`gpurec run`, and `gpurec sample` workflows. It is meant as an operator-facing
contract for reading run results without having to inspect workflow source code.

Optimization writes final artifacts through a staged publish step. Existing
final artifacts are backed up, new files are moved into place, and backups are
restored if publication fails. Intermediate checkpoint files can exist before a
run finishes; final TSV, JSON, CSV, and tensor exports are refreshed together at
the end of the optimization phase.

## Optimization Artifacts

| Path | Written by | Contents | Notes |
|---|---|---|---|
| `history.jsonl` | `optimize`, optimization phase of `run` | One JSON object per recorded optimizer step plus the final evaluation row. | Strict JSON is used: non-finite diagnostics are serialized as `null`, not `NaN` or `Infinity`. |
| `optimization_history.csv` | `optimize`, optimization phase of `run` | CSV version of the in-memory history rows. | Useful for spreadsheets; `history.jsonl` preserves richer typing. |
| `summary.json` | `optimize`, optimization phase of `run` | Final status, reason, elapsed time, family/species/batch counts, effective optimizer/batch/solver route, best NLL metadata, final NLL, final gradient infinity norm, and projected-gradient infinity norm. | Check `status` and `reason` before treating rates as accepted. |
| `rates_final.tsv` | `optimize`, optimization phase of `run` | Final D/T/L rates, survival probability `pS`, and raw theta values. | Rows are `global`, species labels, or family labels depending on mode. |
| `per_fam_likelihoods.tsv` | Genewise `optimize` and genewise optimization phase of `run` | Final per-family NLL and log-likelihood in bits. | Genewise-only because rows are independent only in genewise mode. |
| `theta_final.pt` | `optimize`, optimization phase of `run` | Raw CPU tensor containing final theta values. | For inspection only; it does not carry config, family ordering, species ordering, or optimizer state. |
| `checkpoints/latest.pt` | `optimize`, optimization phase of `run` | Versioned checkpoint for resume. | Carries config metadata, theta, optimizer state when available, progress, status, last row, family names, and species names. |
| `checkpoints/best.pt` | `optimize`, optimization phase of `run` | Versioned checkpoint at the best accepted NLL. | Preferred checkpoint for downstream sampling when present. |

The primary objective is negative log-likelihood in bits:
`likelihood/data_nll_bits`. The corresponding log-likelihood is
`likelihood/log_likelihood_bits`. Gradient summaries use `grad/*`; projected
gradient summaries at bounds use `grad/projected_inf`. Solver diagnostics use
`solver/*`. Specieswise Adagrad restart runs also record
`optimizer/adagrad_restart_*` fields.

`theta_final.pt` is intentionally smaller than a checkpoint. Tooling that needs
to restore a model, sample reconciliations, or verify family/species ordering
should read `checkpoints/best.pt` or `checkpoints/latest.pt` with
`gpurec.workflow.checkpoint.load_checkpoint(...)`, then pass the stored
configuration through `RunConfig.from_dict(...)`.

## Sampling Artifacts

Sampling writes under `reconciliations/` in the sampling output directory. For
`gpurec run`, that is the same `out_dir` unless a sampling output override is
provided.

| Path | Contents |
|---|---|
| `reconciliations/summary.json` | Family range, number of sampled families, samples per family, XML file count, checkpoint path, and output directory. |
| `reconciliations/event_counts.tsv` | Per-family, per-sample counts for the supported RecPhyloXML event keys. |
| `reconciliations/totalSpeciesEventCounts.txt` | Species-level event summaries averaged over the configured samples. |
| `reconciliations/totalTransfers.txt` | Donor/recipient transfer counts averaged over the configured samples. |
| `reconciliations/all/*_sample_*.xml` | Individual RecPhyloXML reconciliations. |
| `reconciliations/all/*_eventCounts_*.txt` | AleRax-style event-count text files for each sample. |

Sampling outputs are also staged and published together. A failed sampling
publish restores the previous generated sampling outputs when possible.

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
| `summary.json` | `optimize`, optimization phase of `run` | Final status, reason, elapsed time, family/species/batch counts, objective/gradient/parameterization metadata, effective optimizer/batch/solver route, best NLL/log-likelihood metadata, final NLL/log-likelihood, final gradient infinity norm, and projected-gradient infinity norm. | Check `status` and `reason` before treating rates as accepted. |
| `rates_final.tsv` | `optimize`, optimization phase of `run` | Final D/T/L rates, survival probability `pS`, and raw theta values. | Rows are `global`, species labels, or family labels depending on mode. |
| `per_fam_likelihoods.tsv` | Genewise `optimize` and genewise optimization phase of `run` | Final per-family NLL and log-likelihood in bits. | Genewise-only because rows are independent only in genewise mode. |
| `theta_final.pt` | `optimize`, optimization phase of `run` | Raw CPU tensor containing final theta values. | For inspection only; it does not carry config, family ordering, species ordering, or optimizer state. |
| `checkpoints/latest.pt` | `optimize`, optimization phase of `run` | Versioned checkpoint for resume. | Carries config metadata, effective route metadata, theta, optimizer state when available, progress, status, last row, family names, and species names. |
| `checkpoints/best.pt` | `optimize`, optimization phase of `run` | Versioned checkpoint at the best accepted NLL. | Preferred checkpoint for downstream sampling when present. |

The primary objective is negative log-likelihood in bits:
`likelihood/data_nll_bits`. The corresponding log-likelihood is
`likelihood/log_likelihood_bits`. Gradient summaries use `grad/*`; projected
gradient summaries at bounds use `grad/projected_inf`. Solver diagnostics use
`solver/*`. Specieswise Adagrad restart runs also record
`optimizer/adagrad_restart_*` fields. `summary.json` and
`gpurec validate-config` expose the stable route metadata fields
`objective=negative_log_likelihood_bits`,
`gradient_route=implicit_first_order_adjoint`,
`rate_parameterization=base2_log_dlt_rates`, and
`production_default_basis=hogenom_and_test_trees_1000`, plus
optimizer-specific route fields such as the specieswise restart schedule and
genewise Hessian-SGD normal-stage solver overrides. For specieswise
`adagrad-restarts`, `adagrad_restart_total_steps` records the derived number of
scheduled Adagrad updates; the run stops when this schedule is complete even if
`steps` is larger. Route metadata also records `configured_steps`,
`optimizer_step_cap`, and `optimizer_step_cap_reason`, so downstream tools can
distinguish a normal configured step limit from a specieswise restart schedule
cap.

`summary.json` repeats the completed optimizer step as `steps_completed`, the
final objective as `final_nll_bits` and `final_log_likelihood_bits`, and the
best accepted objective as `best_nll_bits` and `best_log_likelihood_bits`.  If
the final likelihood/gradient validation fails, `final_log_likelihood_bits` is
`null`; inspect `status`, `reason`, and the final `history.jsonl` row before
using the rates.
The `final_check_iters` field records the effective solver iteration budget
used for the final high-fidelity likelihood/gradient validation; for
specieswise `adagrad-restarts`, this is the resolved
`adagrad_restart_final_check_iters` value.
When the final validation runs, `summary.json` also includes
`final_check_status`, `final_check_source`, `final_check_reason`,
`final_check_fallback_clade_budget`, `final_check_loss_abs_delta_bits`,
`final_check_grad_max_abs_delta`, and `final_check_grad_rel_inf_delta` so the
one-file summary carries the high-fidelity likelihood/gradient agreement check.
Nominal successful checks may omit `final_check_reason` and
`final_check_fallback_clade_budget`; skipped, disabled, failed, fallback, or
cache-drop recomputed checks carry a reason when the workflow can determine one.
The `gpurec optimize` status line and the optimization portion of `gpurec run`
print the resolved `mode` and `optimizer`, `steps_completed`, `elapsed_s`,
`best_step`, and the same final/best NLL and log-likelihood fields, plus
`final_grad_inf`, `final_projected_grad_inf`, and the final validation source,
reason, status, and fallback budget/loss/gradient delta fields, for quick
terminal triage.
Text and path values that contain whitespace or control characters are emitted
as JSON strings with spaces escaped as `\u0020` so each status line remains one
record.

`theta_final.pt` is intentionally smaller than a checkpoint. Tooling that needs
to restore a model, sample reconciliations, or verify family/species ordering
should read `checkpoints/best.pt` or `checkpoints/latest.pt` with
`gpurec.workflow.checkpoint.load_checkpoint(...)`, then pass the stored
configuration through `RunConfig.from_dict(...)`. Current checkpoints also
include `route_metadata`, matching the route fields in `summary.json`, so
inspection tools can identify the objective, gradient route, optimizer, batch
packing, solver budgets, restart schedule, and Hessian-SGD normal-stage
overrides without reconstructing a full `RunConfig`.
The same checkpoint status and route fields are available from the CLI:

```bash
gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt
```

## Sampling Artifacts

Sampling writes under `reconciliations/` in the sampling output directory. For
`gpurec run`, that is the same `out_dir` unless a sampling output override is
provided.
Standalone `gpurec sample` prints `sampled_families`, `samples`, `xml`, and
`out_dir` with the same status-line escaping rule.  The combined `gpurec run`
status line prints the optimization fields plus `sampled_families`, `samples`,
`xml`, `out_dir`, and `sample_out_dir` after sampling succeeds. If
optimization fails, `gpurec run` prints the optimization status fields and exits
without sampling fields.

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

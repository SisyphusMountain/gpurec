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
| `run_config.json` | `optimize`, optimization phase of `run` | Canonical flat JSON `RunConfig` snapshot written before model construction. | This normalized config snapshot includes resolved paths for reruns and audit records; it is not a resumable checkpoint. |
| `history.jsonl` | `optimize`, optimization phase of `run` | One JSON object per recorded optimizer step plus the final evaluation row. | Strict JSON is used: non-finite diagnostics are serialized as `null`, not `NaN` or `Infinity`. |
| `optimization_history.csv` | `optimize`, optimization phase of `run` | CSV version of the in-memory history rows. | Useful for spreadsheets; `history.jsonl` preserves richer typing. |
| `summary.json` | `optimize`, optimization phase of `run` | Final status, reason, elapsed time, selected sampling checkpoint, family/species/batch counts, objective/gradient/parameterization metadata, effective optimizer/batch/solver route, best NLL/log-likelihood metadata, final NLL/log-likelihood, final gradient infinity norm, and projected-gradient infinity norm. | Check `status` and `reason` before treating rates as accepted. |
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
`optimizer/adagrad_restart_*` fields. Warmstart-enabled Pi-adjoint gradients
also report `solver/pi_adjoint_residual_absmax_max`,
`solver/pi_adjoint_residual_relmax_max`, and
`solver/pi_adjoint_residual_checked_batches` when residual telemetry is
available. `summary.json` and
`gpurec validate-config` expose the stable route metadata fields
`objective=negative_log_likelihood_bits`,
`gradient_route=implicit_first_order_adjoint`,
`rate_parameterization=base2_log_dlt_rates`, and
`production_default_basis=hogenom_and_test_trees_1000`, plus
`mode_default_optimizer` and `uses_mode_default_optimizer` for auditing whether
a run used the production optimizer default for its sharing mode. They also
record `uses_production_default_optimizer_settings` and
`production_default_optimizer_setting_mismatches`, which distinguish a plain
optimizer-name match from the full shipped HOGENOM/`test_trees_1000`
optimizer-specific route. They additionally record
`uses_production_default_route` and `production_default_route_mismatches`, the
combined verdict used by `--require-production-default-route` for the objective,
gradient route, rate parameterization, production default basis, optimizer, and
optimizer-specific settings. Optimizer-specific route fields include the
specieswise restart schedule and genewise Hessian-SGD normal-stage solver
overrides. For specieswise
`adagrad-restarts`, `adagrad_restart_total_steps` records the derived number of
scheduled Adagrad updates; the run stops when this schedule is complete even if
`steps` is larger. Route metadata also records `configured_steps`,
`optimizer_step_cap`, and `optimizer_step_cap_reason`, so downstream tools can
distinguish a normal configured step limit from a specieswise restart schedule
cap.
The `OptimizationResult` returned by the Python API and the optimization status
line expose the same family/species/batch counts, `batch_packing`,
`family_chunk_size`, `clade_budget`, `fixed_iters_e`, `fixed_iters_pi`,
`neumann_terms`, route contract fields, configured/effective step cap,
`mode_default_optimizer`, `uses_mode_default_optimizer`,
`uses_production_default_optimizer_settings`,
`production_default_optimizer_setting_mismatches`,
`uses_production_default_route`, `production_default_route_mismatches`,
`final_check_iters`, and
optimizer-specific route fields for quick programmatic and terminal triage.
For genewise `hessian-sgd`, those fields are
`solver_warmup_iters`, `fd_adam_warmup_steps`, `fd_hessian_refresh_steps`,
`hessian_sgd_normal_fixed_iters_pi`, and
`hessian_sgd_normal_neumann_terms`, plus the experimental
`hessian_sgd_pi_adjoint_warmstart` flag, `pi_fixed_point_relaxation`, and
periodic validation controls
`hessian_sgd_validation_interval`,
`hessian_sgd_validation_fixed_iters_pi`, and
`hessian_sgd_validation_neumann_terms`. For specieswise `adagrad-restarts`,
they are `adagrad_restart_schedule`, `adagrad_restart_total_steps`, and
`adagrad_restart_final_check_iters`.

`summary.json` repeats the completed optimizer step as `steps_completed`, the
selected checkpoint path as `sampling_checkpoint`, the final objective as
`final_nll_bits` and `final_log_likelihood_bits`, and the best accepted
objective as `best_nll_bits` and `best_log_likelihood_bits`. Failed runs set
`sampling_checkpoint` and `final_log_likelihood_bits` to `null`; inspect
`status`, `reason`, and the final `history.jsonl` row before using the rates.
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
When the final gradient route emits E-adjoint solver diagnostics, the summary
also includes `final_solver_e_adjoint_failed_batches`,
`final_solver_e_adjoint_success_batches`, and
`final_solver_e_adjoint_rel_res_max` so nonconverged adjoint solves are visible
without scanning `history.jsonl`.
The `gpurec optimize` status line and the optimization portion of `gpurec run`
print the resolved `mode` and `optimizer`, `families`, `species`, `batches`,
base batch/solver route fields, route contract fields, configured and effective
step cap, `mode_default_optimizer`, `uses_mode_default_optimizer`,
`final_check_iters`, optimizer-specific route fields,
`steps_completed`, `elapsed_s`, `best_step`, `sampling_checkpoint`, and the
same final/best NLL and log-likelihood fields, plus `final_grad_inf`,
`final_projected_grad_inf`, and the final validation source, reason, status,
and fallback budget/loss/gradient delta fields. When available, it also prints
the final E-adjoint solver diagnostics
`final_solver_e_adjoint_failed_batches`,
`final_solver_e_adjoint_success_batches`, and
`final_solver_e_adjoint_rel_res_max` for quick terminal triage.
Add `--require-converged` to `gpurec optimize` when the command should print
the same status line and then exit nonzero unless the optimization status is
`converged`. Add `--require-final-check-ok` when it should also require
`final_check_status=ok` before returning success. Add
`--require-mode-default-optimizer` to `gpurec validate-config`,
`gpurec optimize`, or `gpurec run` when automation should fail unless the
resolved optimizer matches the production default for the selected mode. Add
`--require-production-default-route` when automation should also reject
stale likelihood/gradient route metadata or optimizer-specific setting
overrides reported by `production_default_route_mismatches`.
Text and path values that contain whitespace or control characters are emitted
as JSON strings with spaces escaped as `\u0020` so each status line remains one
record.
The same one-line summary view is available after a run without loading a
checkpoint or constructing the CUDA model:

```bash
gpurec summary-info --summary output_gpurec/summary.json
```

For older summaries that have `mode` and `optimizer` but predate
`mode_default_optimizer` and `uses_mode_default_optimizer`, `summary-info`
infers those audit fields before printing so the displayed line matches the
route evidence used by `--require-mode-default-optimizer`. Mode and optimizer
strings are normalized with the same `RunConfig` rules before these audit fields
are inferred, so legacy casing and underscore aliases do not fail the
default-route gates. If the summary is too old or incomplete to prove both
`mode` and `optimizer`, the gate fails with an incomplete-evidence error instead
of treating the route as accepted.
`summary-info` also infers the stricter production-route settings audit when
the summary carries the relevant optimizer-specific fields; otherwise
`--require-production-default-route` fails with an incomplete-evidence error.
When complete evidence is available, it prints
`uses_production_default_route` and `production_default_route_mismatches` with
the same verdict that drives the gate.

Add `--require-converged` when the command should print the same summary line
and then exit nonzero unless `summary.status` is `converged`. Add
`--require-final-check-ok` when downstream automation should also require
`summary.final_check_status` to be `ok`. Add
`--require-mode-default-optimizer` when downstream automation should reject
summaries that do not prove the mode default optimizer was used, or
`--require-production-default-route` when the likelihood/gradient route
metadata and optimizer-specific settings must also match the shipped production
route.

`theta_final.pt` is intentionally smaller than a checkpoint. Tooling that needs
to restore a model, sample reconciliations, or verify family/species ordering
should read `checkpoints/best.pt` or `checkpoints/latest.pt` with
`gpurec.workflow.checkpoint.load_checkpoint(...)`, then pass the stored
configuration through `RunConfig.from_dict(...)`. Current checkpoints also
include `route_metadata`, matching the route fields in `summary.json`, so
inspection tools can identify the objective, gradient route, optimizer, batch
packing, solver budgets, restart schedule, and Hessian-SGD normal-stage
overrides without reconstructing a full `RunConfig`. Resume and sampling
compatibility checks treat absent route metadata as a legacy checkpoint, but a
checkpoint that does carry `route_metadata` must include and match all current
non-mutable route fields before theta restoration.
The same checkpoint status and route fields are available from the CLI:

```bash
gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt
```

When the checkpoint's saved last row contains final validation metrics,
`checkpoint-info` also prints `last_final_check_status`,
`last_final_check_source`, `last_final_check_reason`,
`last_final_check_fallback_clade_budget`,
`last_final_check_loss_abs_delta_bits`,
`last_final_check_grad_max_abs_delta`, and
`last_final_check_grad_rel_inf_delta`. It also prints
`last_solver_e_adjoint_failed_batches`,
`last_solver_e_adjoint_success_batches`, and
`last_solver_e_adjoint_rel_res_max` when those metrics were recorded in the
checkpoint's last row. Add `--require-final-check-ok` when
automation should fail unless the checkpoint last row has
`optimizer/final_check_status=ok`; add `--require-mode-default-optimizer` when
the checkpoint route must use the production optimizer default for its mode, or
`--require-production-default-route` when the checkpoint route must also match
the shipped likelihood/gradient contract and optimizer-specific route. If
a legacy checkpoint has no `route_metadata`, `checkpoint-info` falls back to
recoverable config `mode` and `optimizer` fields; incomplete artifacts fail the
gate with an incomplete-evidence error.

## Sampling Artifacts

Sampling writes under `reconciliations/` in the sampling output directory. For
`gpurec run`, that is the same `out_dir` unless a sampling output override is
provided.
Standalone `gpurec sample` prints `sampled_families`, `samples`, `xml`, and
`out_dir` with the same status-line escaping rule.  The combined `gpurec run`
status line prints the optimization fields plus `sampled_families`, `samples`,
`xml`, `out_dir`, and `sample_out_dir` after sampling succeeds. If
optimization fails, `gpurec run` prints the optimization status fields and exits
without sampling fields. With `--require-converged`, `gpurec run` also prints
the optimization status fields and exits before sampling when the optimization
status is anything other than `converged`. With `--require-final-check-ok`, it
also exits before sampling unless `final_check_status=ok`. With
`--require-mode-default-optimizer`, it exits before optimization or sampling
unless the resolved optimizer is the production default for the selected mode.
With `--require-production-default-route`, it also rejects changed
optimizer-specific settings or stale likelihood/gradient route metadata before
optimization or sampling and reports the offending
`production_default_route_mismatches`.
The same `--require-mode-default-optimizer` flag is available on standalone
`gpurec sample`; it inspects the checkpoint route and exits before sampling if
the checkpoint cannot prove it used the production default optimizer. Standalone
`gpurec sample` also supports `--require-production-default-route` to require
the full shipped likelihood/gradient and optimizer-specific route before
sampling.

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

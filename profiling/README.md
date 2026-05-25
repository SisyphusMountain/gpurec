# Profiling Surface

The tracked profiling directory is source-checkout tooling, not installed
package API.  Profiling commands may require CUDA, local generated tree
fixtures, HOGENOM benchmark data, Nsight tools, or experiment-specific
environment variables.  Keep user-facing workflows in `gpurec` and
`gpurec.workflow`; use this directory for measured performance investigations.

## Supported Entrypoints

| Entrypoint | Owner | Inputs | Output contract |
| --- | --- | --- | --- |
| `bench_preprocess_rust.py` | Maintained preprocessing backend benchmark. | Source checkout with the Rust preprocessing crate, local AleRax-style family input such as `tests/data/hogenom_bench`, and explicit repeat/thread flags. | JSON timing summaries comparing Rust native adapter, Rust subprocess adapter, and Rust CLI output modes; use for local backend validation, not as a stable downstream schema. |
| `bench_resident_likelihood.py` | Maintained resident model likelihood benchmark. | CUDA, a generated tree dataset such as `tests/data/test_trees_1000`, and explicit resident batching, solver-budget, materialization, and repeat flags. | JSONL timing records for `GeneReconModel.full_loss_for_theta()` likelihood-only passes and optional `full_loss()` gradient passes; output is profiling evidence, not a stable workflow artifact schema. |
| `bench_uniform_forward_backward_pipeline.py` | Maintained full-pipeline benchmark for the lean branch. | CUDA, a source-checkout dataset such as generated `test_trees_*`, and explicit benchmark flags. | Human-readable timing lines and strict optimized-kernel verdicts; new output fields should be documented before downstream tooling depends on them. |
| `evaluate_hogenom_alerax_rates.py` | Checkout-local HOGENOM/AleRax validation helper. | Local untracked HOGENOM benchmark layout, AleRax output files, GPUREC checkpoints, and CUDA. | CSV-style likelihood comparisons for local validation only; it is not a general AleRax rate-file parser. |
| `profile_active_batch_step.py` | Checkout-local active resident-batch profiler. | CUDA, source-checkout species/family inputs, and explicit batch selector, solver, and profiling flags. | JSONL timing records for active-batch closures or one batched LBFGS step; output is profiling evidence, not a stable workflow artifact schema. |

New tracked profiling entrypoints should have a `--help` smoke or a focused unit
guard, document required local data and CUDA/Nsight assumptions, and state
whether their output is stable enough for other tools.

## Artifact Policy

Ignored profiling directories such as `profiling/ancestor_batching/`,
`profiling/bf16_backward_nsys/`, `profiling/bf16_handoff_prod/`,
`profiling/hogenom_ccp/`, and `profiling/specieswise_worker3/` are local
artifact workspaces.  They may contain Nsight reports, SQLite captures, CSV
summaries, JSONL sweeps, logs, plots, or temporary harness output.  Keep raw
artifacts ignored; move durable conclusions into `docs/` before deleting or
externalizing the local files.

Ignored `profiling/proposal2/` and `profiling/proposal8/` directories currently
contain only local Python bytecode cache from prototype runs.  They are not
source, fixtures, or retained benchmark results and can be deleted locally.

Before promoting any ignored artifact or prototype result into tracked source,
add a reproducible command, a small fixture or explicit local-data requirement,
and a test or documented manual verification gate.

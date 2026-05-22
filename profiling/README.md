# Profiling Surface

The tracked profiling directory is source-checkout tooling, not installed
package API.  Profiling commands may require CUDA, local generated tree
fixtures, HOGENOM benchmark data, Nsight tools, or experiment-specific
environment variables.  Keep user-facing workflows in `gpurec` and
`gpurec.workflow`; use this directory for measured performance investigations.

## Supported Entrypoints

| Entrypoint | Owner | Inputs | Output contract |
| --- | --- | --- | --- |
| `bench_preprocess_rust_vs_cpp.py` | Maintained preprocessing backend benchmark. | Source checkout with the Rust preprocessing crate, C++ preprocessing extension, local AleRax-style family input such as `tests/data/hogenom_bench`, and explicit repeat/thread flags. | JSON timing summaries comparing C++ pybind, Rust native adapter, Rust subprocess adapter, and Rust CLI output modes; use for local backend validation, not as a stable downstream schema. |
| `bench_uniform_forward_backward_pipeline.py` | Maintained full-pipeline benchmark for the lean branch. | CUDA, a source-checkout dataset such as generated `test_trees_*`, and explicit benchmark flags. | Human-readable timing lines and strict optimized-kernel verdicts; new output fields should be documented before downstream tooling depends on them. |
| `evaluate_hogenom_alerax_rates.py` | Checkout-local HOGENOM/AleRax validation helper. | Local untracked HOGENOM benchmark layout, AleRax output files, GPUREC checkpoints, and CUDA. | CSV-style likelihood comparisons for local validation only; it is not a general AleRax rate-file parser. |

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

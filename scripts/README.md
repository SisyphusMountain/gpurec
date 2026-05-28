# HOGENOM Experiment Scripts

The installed `gpurec` CLI is the supported workflow for general optimization,
checkpointing, and sampling:

```bash
gpurec optimize --species-tree S.tree --families-file families.txt --out-dir out --device cuda
```

The optimized workflow currently requires CUDA.  CPU-only checkouts can still
run config parsing, help, packaging, and unit-level hygiene checks, but not the
optimized likelihood workflow.

The scripts in this directory are checkout-local experiment launchers,
diagnostics, and compatibility helpers.  The large launchers
`hogenom_ccp_wandb_opt.py` and `fast_optimize_hogenom_ccp.py` are legacy
HOGENOM reproducers: they retain experiment-specific optimizer schedules,
plotting, W&B behavior, local path defaults, and reporting conventions.
Some older launchers, including the global-uniform and specieswise-uniform
scripts, are fixed-dataset reproducers with HOGENOM paths declared as module
constants rather than general path flags.

New production workflow behavior should go into `gpurec.workflow` and the
installed CLI first.  Mirror it into these legacy scripts only when a retained
HOGENOM experiment needs that behavior.
Legacy scripts must have explicit keep, migrate, or delete decisions in the
ownership matrix below.

## Ownership Matrix

| Path | Status | Ownership / next action |
| --- | --- | --- |
| `optimize_hogenom_ccp_wandb.py` | Compatibility wrapper. | Keep as the documented HOGENOM launcher entry point while `hogenom_ccp_wandb_opt.py` exists. |
| `hogenom_ccp_wandb_opt.py` | Legacy full HOGENOM W&B optimizer. | Retain for historical reproduction; migrate reusable behavior into `gpurec.workflow` before changing optimizer semantics here. |
| `optimize_hogenom_ccp_hydra.py` | Hydra adapter for the legacy W&B optimizer. | Keep only as a checkout-local adapter for `configs/hogenom_ccp_wandb.yaml`; the supported CLI remains flat JSON. |
| `fast_optimize_hogenom_ccp.py` | Legacy fast HOGENOM launcher. | Retain as a historical reproducer until its optimizer schedule is covered by CLI tests or deleted. |
| `optimize_hogenom_ccp_global_uniform.py` | Fixed-dataset global-uniform reproducer. | Checkout-local HOGENOM run with one shared D/T/L theta row, uniform origination, optional active-batch/full Adagrad warmup, LBFGS, and square/l1/huber/elastic-net/gaussian/beta-pS regularizers. Writes `global_optimization_history.csv`, `global_rate_distribution_history.csv`, `optimized_global_rates.csv`, `uniform_origination_distribution.csv`, and `run_config.json` under `output_gpurec_global_uniform_opt_max100`. Candidate for deletion or migration after this optimizer/reporting behavior is covered by supported workflow tests. |
| `optimize_hogenom_ccp_specieswise_uniform.py` | Fixed-dataset specieswise-uniform reproducer. | Checkout-local HOGENOM run with one D/T/L theta row per species, uniform origination, helper-owned optimizer schedules (`lbfgs`, `adagrad`, `adam`, minibatch variants, and two-phase Adagrad+LBFGS), and beta-pS/square-theta/gaussian-theta regularizers. Writes `specieswise_optimization_history.csv`, `specieswise_parameter_history.csv`, `optimized_specieswise_rates.csv`, `uniform_origination_distribution.csv`, and `run_config.json` under `output_gpurec_specieswise_uniform_opt_max100`. Candidate for deletion or migration once helper reuse and output expectations are covered by tests. |
| `hogenom_opt_helpers.py` | Shared helper for legacy uniform launchers. | Keep only while the fixed-dataset launchers remain. |
| `optimize_hogenom_penalty316_kkt.py` | One-off branch-scale penalty/KKT analysis. | Checkout-local branchscaled penalty-316.22776601683796 reproducer using CUDA, W&B disabled, 100 Adam warmup steps, Strong-Wolfe LBFGS, L1 KKT residual checks, timestamped output directories, and `latest_run.txt`; archive/delete or migrate once branchscaled KKT reporting is owned by the supported CLI, and add a tiny fixture guard before changing its loader or output schema. |
| `make_hogenom_branchscale_penalty_report.py` | One-off LaTeX report builder. | Stale relative to newer timestamped run directories. Expected layout: `penalty_*` child directories with `history.jsonl`, `branchscaled_node_rates_final.tsv`, optional `run_config.json`, and `tree_plots/rates_final.png`; timestamped launcher outputs must be copied or symlinked under `penalty_*` names before this historical script discovers them. Delete or migrate once branchscaled reporting is owned by the supported CLI. |
| `profile_hogenom_ccp_pass.py` | Internal checkout-local profiler. | HOGENOM-only CUDA/Nsight harness for one specieswise forward/backward pass. It hard-codes HOGENOM inputs and emits JSON lines for `config`, `model`, `warmup`, `measured`, and `summary` events while bracketing measured runs with CUDA profiler API calls and NVTX ranges. Keep only until `profiling/bench_uniform_forward_backward_pipeline.py` or another maintained benchmark owns the same scheduler/memory questions; keep it as local profiling, not as a portable benchmark or public API contract. |
| `benchmark_hogenom_gradient_convergence.py` | Checkout-local HOGENOM optimizer benchmark. | Measures Neumann-term gradient convergence near accepted HOGENOM specieswise checkpoints and compares cold Pi adjoint solves against LBFGSB-history Pi adjoint warm starts. Keep as internal benchmark evidence until gradient warmstart policy is promoted into `gpurec.workflow` or retired. |
| `benchmark_hogenom_specieswise_multifidelity_adagrad.py` | Checkout-local HOGENOM optimizer benchmark. | Reproduces the counts-free specieswise HOGENOM multifidelity Adagrad route from uniform 0.05 rates: fixed8 warmup, fixed16 bridge, fixed32 repair, and fixed128 validation. `--schedule-mode adaptive` promotes phases by higher-budget validation stalls and restores the best validated theta before each promotion. Writes history, summary, theta, and checkpoint artifacts under the requested output directory. Keep as benchmark evidence until this optimizer schedule is promoted into the supported workflow. |
| `benchmark_hogenom_specieswise_e2e.py` | Checkout-local HOGENOM route benchmark. | Summarizes the accepted HOGENOM specieswise optimization route from local run directories and pulse checkpoints, reporting time-to-target, per-stage solver settings, objectives, projected-gradient residuals, and which manual probe stages have unknown elapsed time. Supports replacing manual checkpoint stages with measured exact-delta replay directories, truncating/appending route stages for relaxed targets, consuming pulse-benchmark fixed-budget validation rows, and an effective resume-elapsed mode that charges each run only through the checkpoint consumed by the next stage. Keep as internal benchmark provenance until the route is promoted into a maintained workflow command. |
| `benchmark_hogenom_specieswise_pulses.py` | Checkout-local HOGENOM optimizer benchmark. | Benchmarks bounded HOGENOM specieswise pulse probes from an existing checkpoint, validates selected candidates at higher fixed solver budgets, and writes CSV/JSONL summaries plus candidate checkpoints. Top-k/pair probes default to scaled projected-gradient steps; single-coordinate probes default to absolute sign steps to match the historical micro-polish. With an empty candidate set it still records the base fixed-budget validation row, which is useful for timing stop-point checks. Keep as an internal benchmark harness until the pulse schedule is either promoted into `gpurec.workflow` or retired. |
| `replay_hogenom_specieswise_tail.py` | Checkout-local HOGENOM route replay. | Replays accepted HOGENOM specieswise pulse checkpoints by exact theta deltas, measures per-delta elapsed time, optionally validates the final checkpoint at fixed128, and writes history/checkpoint artifacts consumable by `benchmark_hogenom_specieswise_e2e.py`. Also has a dynamic pulse replay mode; historical top-k moves use scaled projected-gradient directions while coordinate micro-moves use absolute sign directions. Keep as benchmark evidence until the tail is either automated in the workflow or replaced by a faster route. |
| `compare_backtracking_alerax_events.py` | Checkout-local AleRax comparison helper. | Keep for validation runs that have AleRax output and local HOGENOM data. |
| `export_hogenom_rates_from_checkpoint.py` | HOGENOM checkpoint rate exporter. | Keep as a utility for local analysis; promote only if rate-export format becomes a supported CLI feature. |
| `visualize_hogenom_loss_landscape.py` | Checkout-local HOGENOM landscape visualizer. | HOGENOM research analysis that plots per-family D/L/T loss contours around supplied or locally optimized anchors, writing CSV summaries and PNG/PDF panels under `output_gpurec_loss_landscape`. Keep unless promoted into a supported diagnostics command. |
| `generate_dependency_inventory.py` | Release hygiene utility. | Keep as release-scoped supply-chain evidence generation and dependency-manifest snapshot command. |
| `validate_output_artifacts.py` | Artifact validation utility. | Keep as release and handoff QA for `summary.json`, `history.jsonl`, checkpoints, TSV outputs, and `run_manifest.json`. |
| `run_long_validation.py` | Release-candidate long validation runner. | Keep as a reproducible pre-release GPU workflow command over the public end-to-end dataset. It orchestrates `doctor`, `validate-config`, `optimize`, `summary-info`, and `sample`, writes a machine-readable report, and enforces configured runtime/NLL/count thresholds. |
| `plot_hogenom_rates.R` | Optional plotting helper. | Keep with the `hogenom` extra/documented plotting dependencies. |
| `check_release_metadata.py` | Release metadata gate. | Keep as release hygiene, not as part of the HOGENOM experiment surface. |

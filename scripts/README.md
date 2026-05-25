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
| `benchmark_hogenom_specieswise_multifidelity_adagrad.py` | Checkout-local HOGENOM optimizer benchmark. | Reproduces the counts-free specieswise HOGENOM multifidelity Adagrad route from uniform 0.05 rates: fixed8 warmup, fixed16 bridge, fixed32 repair, and fixed128 validation. Writes history, summary, theta, and checkpoint artifacts under the requested output directory. Keep as benchmark evidence until this optimizer schedule is promoted into the supported workflow. |
| `compare_backtracking_alerax_events.py` | Checkout-local AleRax comparison helper. | Keep for validation runs that have AleRax output and local HOGENOM data. |
| `export_hogenom_rates_from_checkpoint.py` | HOGENOM checkpoint rate exporter. | Keep as a utility for local analysis; promote only if rate-export format becomes a supported CLI feature. |
| `plot_hogenom_rates.R` | Optional plotting helper. | Keep with the `hogenom` extra/documented plotting dependencies. |
| `check_release_metadata.py` | Release metadata gate. | Keep as release hygiene, not as part of the HOGENOM experiment surface. |

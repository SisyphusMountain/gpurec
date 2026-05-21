# HOGENOM Experiment Scripts

The installed `gpurec` CLI is the supported workflow for general optimization,
checkpointing, and sampling:

```bash
gpurec optimize --species-tree S.tree --families-file families.txt --out-dir out
```

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
| `optimize_hogenom_ccp_global_uniform.py` | Fixed-dataset global-uniform reproducer. | Candidate for deletion or migration after any unique optimizer/reporting behavior is documented. |
| `optimize_hogenom_ccp_specieswise_uniform.py` | Fixed-dataset specieswise-uniform reproducer. | Candidate for deletion or migration after helper reuse and output expectations are documented. |
| `hogenom_opt_helpers.py` | Shared helper for legacy uniform launchers. | Keep only while the fixed-dataset launchers remain. |
| `optimize_hogenom_penalty316_kkt.py` | One-off branch-scale penalty/KKT analysis. | Archive or migrate with a tiny fixture test before changing its loader or output schema. |
| `make_hogenom_branchscale_penalty_report.py` | One-off LaTeX report builder. | Stale relative to newer timestamped run directories; document the expected sweep layout before updating or deleting. |
| `profile_hogenom_ccp_pass.py` | Internal checkout-local profiler. | Keep as a CUDA/HOGENOM profiler, not as a portable benchmark or public API contract. |
| `compare_backtracking_alerax_events.py` | Checkout-local AleRax comparison helper. | Keep for validation runs that have AleRax output and local HOGENOM data. |
| `export_hogenom_rates_from_checkpoint.py` | HOGENOM checkpoint rate exporter. | Keep as a utility for local analysis; promote only if rate-export format becomes a supported CLI feature. |
| `plot_hogenom_rates.R` | Optional plotting helper. | Keep with the `hogenom` extra/documented plotting dependencies. |
| `check_release_metadata.py` | Release metadata gate. | Keep as release hygiene, not as part of the HOGENOM experiment surface. |

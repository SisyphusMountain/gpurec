#!/usr/bin/env python3
"""Checkout-local Hydra adapter for the legacy HOGENOM W&B optimizer.

This launcher consumes ``configs/hogenom_ccp_wandb.yaml`` and forwards those
values into the legacy W&B optimizer. Supported production optimization uses
the installed ``gpurec optimize`` command with flat JSON ``RunConfig`` files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as exc:  # pragma: no cover - exercised by CLI users.
    raise SystemExit(
        "hydra-core is required for this launcher. Run with:\n"
        "  uv run --with hydra-core --with wandb python -u "
        "scripts/optimize_hogenom_ccp_hydra.py"
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hogenom_ccp_wandb_opt import config_from_args, parse_args, run  # noqa: E402


def _as_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _append_option(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    argv.extend([flag, _as_cli_value(value)])


def _argv_from_config(cfg: dict[str, Any]) -> list[str]:
    paths = cfg["paths"]
    runtime = cfg["runtime"]
    model = cfg["model"]
    solver = cfg["solver"]
    optimizer = cfg["optimizer"]
    bounds = cfg["bounds"]
    convergence = cfg["convergence"]
    regularization = cfg["regularization"]
    diagnostics = cfg["diagnostics"]
    logging_cfg = cfg["logging"]
    wandb = cfg["wandb"]

    argv: list[str] = []
    _append_option(argv, "--species-tree", paths.get("species_tree"))
    _append_option(argv, "--families-file", paths.get("families_file"))
    _append_option(argv, "--out-dir", paths.get("out_dir"))
    if not bool(paths.get("timestamped_out_dir", True)):
        argv.append("--no-timestamped-out-dir")
    _append_option(argv, "--resume-from", paths.get("resume_from"))

    _append_option(argv, "--device", runtime.get("device"))

    _append_option(argv, "--mode", model.get("mode"))
    _append_option(argv, "--max-families", model.get("max_families"))
    _append_option(argv, "--family-chunk-size", model.get("family_chunk_size"))
    _append_option(argv, "--clade-budget", model.get("clade_budget"))
    _append_option(argv, "--max-wave-size", model.get("max_wave_size"))

    _append_option(argv, "--max-iters-e", solver.get("max_iters_e"))
    _append_option(argv, "--max-iters-pi", solver.get("max_iters_pi"))
    _append_option(argv, "--max-neumann-terms", solver.get("max_neumann_terms"))
    _append_option(argv, "--convergence-check-interval", solver.get("convergence_check_interval"))
    _append_option(argv, "--e-logsumexp-tol", solver.get("e_logsumexp_tol"))
    _append_option(argv, "--pi-max-diff-tol", solver.get("pi_max_diff_tol"))
    _append_option(argv, "--gradient-change-tol", solver.get("gradient_change_tol"))
    _append_option(argv, "--gradient-change-rtol", solver.get("gradient_change_rtol"))
    _append_option(argv, "--solver-iteration-schedule", solver.get("iteration_schedule"))
    _append_option(argv, "--solver-budget-initial-iters", solver.get("budget_initial_iters"))
    _append_option(argv, "--solver-budget-increment", solver.get("budget_increment"))
    _append_option(argv, "--solver-budget-patience", solver.get("budget_patience"))
    _append_option(argv, "--solver-budget-step-interval", solver.get("budget_step_interval"))

    _append_option(argv, "--optimizer", optimizer.get("name"))
    _append_option(argv, "--adam-warmup-steps", optimizer.get("adam_warmup_steps"))
    _append_option(argv, "--steps", optimizer.get("steps"))
    _append_option(argv, "--lr", optimizer.get("lr"))
    _append_option(argv, "--lr-decay-every", optimizer.get("lr_decay_every"))
    _append_option(argv, "--lr-decay-factor", optimizer.get("lr_decay_factor"))
    _append_option(argv, "--lbfgs-lr", optimizer.get("lbfgs_lr"))
    _append_option(argv, "--lbfgs-history-size", optimizer.get("lbfgs_history_size"))
    _append_option(argv, "--lbfgs-line-search", optimizer.get("lbfgs_line_search"))

    _append_option(argv, "--min-rate", bounds.get("min_rate"))
    _append_option(argv, "--max-rate", bounds.get("max_rate"))

    _append_option(argv, "--grad-inf-tol", convergence.get("grad_inf_tol"))
    _append_option(argv, "--loss-change-tol", convergence.get("loss_change_tol"))
    _append_option(argv, "--loss-patience", convergence.get("loss_patience"))

    _append_option(argv, "--beta-ps-alpha", regularization.get("beta_ps_alpha"))
    _append_option(argv, "--beta-ps-beta", regularization.get("beta_ps_beta"))
    _append_option(argv, "--beta-prior-weight", regularization.get("beta_prior_weight"))
    _append_option(argv, "--branchscale-prior-weight", regularization.get("branchscale_prior_weight"))

    _append_option(argv, "--diagnostic-mode", diagnostics.get("mode"))

    _append_option(argv, "--log-every", logging_cfg.get("log_every"))
    _append_option(argv, "--plot-every", logging_cfg.get("plot_every"))
    _append_option(argv, "--checkpoint-every", logging_cfg.get("checkpoint_every"))

    _append_option(argv, "--wandb-project", wandb.get("project"))
    _append_option(argv, "--wandb-entity", wandb.get("entity"))
    _append_option(argv, "--wandb-run-name", wandb.get("run_name"))
    _append_option(argv, "--wandb-mode", wandb.get("mode"))
    return argv


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="hogenom_ccp_wandb",
)
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("expected a mapping Hydra config")
    args = parse_args(_argv_from_config(config_dict))
    run_config = config_from_args(args)
    if not run_config.species_tree.exists():
        raise FileNotFoundError(run_config.species_tree)
    if not run_config.families_file.exists():
        raise FileNotFoundError(run_config.families_file)
    run_config.out_dir.mkdir(parents=True, exist_ok=True)
    (run_config.out_dir / "hydra_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )
    run(run_config, args)


if __name__ == "__main__":
    main()

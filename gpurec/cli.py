from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gpurec.workflow import RunConfig, SamplingConfig, optimize, sample


_EXPECTED_WORKFLOW_ERRORS = (ValueError, OSError, RuntimeError)


def _chunk_size(value: str) -> int | None:
    text = value.strip().lower()
    if text in {"none", "null"}:
        return None
    if text in {"", "0", "all"}:
        return 0
    if text == "auto":
        raise argparse.ArgumentTypeError(
            "family chunk size 'auto' is not supported; use 0 for one resident "
            "batch or a positive integer"
        )
    try:
        size = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "family chunk size must be 0, all, none, or a positive integer"
        ) from exc
    if size < 0:
        raise argparse.ArgumentTypeError("family chunk size must be non-negative")
    return size


def _config_data(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError(
            "--config currently expects a flat JSON RunConfig file; "
            "Hydra-style YAML configs must be converted to JSON or passed as "
            "explicit CLI flags"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise ValueError(f"could not read config {path}: {detail}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config {path}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must contain a JSON object")
    return data


def _set_if_present(data: dict[str, Any], args: argparse.Namespace, name: str) -> None:
    value = getattr(args, name)
    if value is not None:
        data[name] = value


def _run_config_from_args(args: argparse.Namespace) -> RunConfig:
    data = _config_data(args.config)
    for name in (
        "species_tree",
        "families_file",
        "out_dir",
        "mode",
        "device",
        "dtype",
        "start",
        "max_families",
        "preprocess_cache",
        "refresh_preprocess_cache",
        "family_chunk_size",
        "clade_budget",
        "batch_packing",
        "max_wave_size",
        "fixed_iters_e",
        "max_iters_e",
        "tol_e",
        "fixed_iters_pi",
        "neumann_terms",
        "convergence_check_interval",
        "e_logsumexp_tol",
        "pi_max_diff_tol",
        "gradient_change_tol",
        "gradient_change_rtol",
        "theta_init_d",
        "theta_init_l",
        "theta_init_t",
        "min_rate",
        "max_rate",
        "optimizer",
        "steps",
        "lr",
        "adam_warmup_steps",
        "lbfgs_lr",
        "lbfgs_history_size",
        "lbfgs_max_iter",
        "lbfgs_line_search",
        "grad_inf_tol",
        "loss_change_tol",
        "loss_patience",
        "best_likelihood_patience",
        "best_likelihood_min_delta",
        "checkpoint_every",
        "log_every",
        "resume_from",
    ):
        _set_if_present(data, args, name)
    if args.adaptive_iters is not None:
        data["adaptive_iters"] = args.adaptive_iters
    missing = [name for name in ("species_tree", "families_file", "out_dir") if name not in data]
    if missing:
        raise ValueError(f"missing required optimize option(s): {', '.join(missing)}")
    return RunConfig.from_dict(data)


def _add_run_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="Flat JSON RunConfig file; explicit CLI flags override matching fields.",
    )
    parser.add_argument(
        "--species-tree",
        type=Path,
        help="Species tree Newick path. Required unless supplied by --config.",
    )
    parser.add_argument(
        "--families-file",
        type=Path,
        help="AleRax-style family list. Required unless supplied by --config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Output directory for checkpoints, logs, and rates. Required "
            "unless supplied by --config."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("genewise", "global", "specieswise"),
        help="Parameter sharing mode. Workflow default: genewise.",
    )
    parser.add_argument(
        "--device",
        help="Torch device for production optimization. Workflow default: cuda.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        help="Floating-point dtype. Workflow default: float32.",
    )
    parser.add_argument("--start", type=int, help="First family index to load.")
    parser.add_argument(
        "--max-families",
        type=int,
        help="Maximum number of families to load.",
    )
    parser.add_argument(
        "--preprocess-cache",
        type=Path,
        help="Directory for reusable preprocessing cache files.",
    )
    parser.add_argument(
        "--refresh-preprocess-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Regenerate preprocessing cache entries before optimization.",
    )
    parser.add_argument(
        "--family-chunk-size",
        type=_chunk_size,
        help="Families per resident batch; use 0/all/none for one resident batch.",
    )
    parser.add_argument(
        "--clade-budget",
        type=int,
        help="Clade budget for non-sequential resident-batch packing.",
    )
    parser.add_argument(
        "--batch-packing",
        choices=("sequential", "clade_first_fit", "depth_first_fit"),
        help="Resident-batch packing policy. Workflow default: depth_first_fit.",
    )
    parser.add_argument(
        "--max-wave-size",
        type=int,
        help="Maximum clades scheduled into one resident wave.",
    )
    parser.add_argument("--fixed-iters-e", type=int, help="Fixed E iterations per solve.")
    parser.add_argument("--max-iters-e", type=int, help="Maximum adaptive E iterations.")
    parser.add_argument("--tol-e", type=float, help="E fixed-point convergence tolerance.")
    parser.add_argument("--fixed-iters-pi", type=int, help="Fixed Pi iterations per solve.")
    parser.add_argument(
        "--neumann-terms",
        type=int,
        help="Terms for implicit-gradient Neumann series.",
    )
    parser.add_argument(
        "--adaptive-iters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable adaptive E/Pi solver iteration stopping.",
    )
    parser.add_argument(
        "--convergence-check-interval",
        type=int,
        help="Iteration interval for adaptive solver convergence checks.",
    )
    parser.add_argument(
        "--e-logsumexp-tol",
        type=float,
        help="E logsumexp convergence tolerance.",
    )
    parser.add_argument(
        "--pi-max-diff-tol",
        type=float,
        help="Pi max-difference convergence tolerance.",
    )
    parser.add_argument(
        "--gradient-change-tol",
        type=float,
        help="Absolute gradient-change tolerance.",
    )
    parser.add_argument(
        "--gradient-change-rtol",
        type=float,
        help="Relative gradient-change tolerance.",
    )
    parser.add_argument("--theta-init-d", type=float, help="Initial duplication rate.")
    parser.add_argument("--theta-init-l", type=float, help="Initial loss rate.")
    parser.add_argument("--theta-init-t", type=float, help="Initial transfer rate.")
    parser.add_argument("--min-rate", type=float, help="Minimum allowed D/L/T rate.")
    parser.add_argument("--max-rate", type=float, help="Maximum allowed D/L/T rate.")
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adagrad", "lbfgs", "adam-lbfgs"),
        help="Optimizer schedule. Workflow default: adam.",
    )
    parser.add_argument("--steps", type=int, help="Maximum optimization steps.")
    parser.add_argument("--lr", type=float, help="Adam/Adagrad learning rate.")
    parser.add_argument(
        "--adam-warmup-steps",
        type=int,
        help="Adam steps before LBFGS in adam-lbfgs mode.",
    )
    parser.add_argument("--lbfgs-lr", type=float, help="LBFGS learning rate.")
    parser.add_argument("--lbfgs-history-size", type=int, help="LBFGS history size.")
    parser.add_argument("--lbfgs-max-iter", type=int, help="LBFGS inner iterations per step.")
    parser.add_argument(
        "--lbfgs-line-search",
        choices=("none", "strong_wolfe"),
        help="LBFGS line-search mode.",
    )
    parser.add_argument(
        "--grad-inf-tol",
        type=float,
        help="Stop when gradient infinity norm is below this value.",
    )
    parser.add_argument(
        "--loss-change-tol",
        type=float,
        help="Loss-change stopping tolerance.",
    )
    parser.add_argument(
        "--loss-patience",
        type=int,
        help="Consecutive small-loss-change steps before stopping.",
    )
    parser.add_argument(
        "--best-likelihood-patience",
        type=int,
        help="Steps without best-likelihood improvement before stopping.",
    )
    parser.add_argument(
        "--best-likelihood-min-delta",
        type=float,
        help="Minimum best-likelihood improvement to reset patience.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="Checkpoint interval in optimization steps; 0 disables periodic checkpoints.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        help="History logging interval in optimization steps.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume optimization state from an existing checkpoint.",
    )


def _add_sampling_args(
    parser: argparse.ArgumentParser,
    *,
    checkpoint_required: bool,
    include_checkpoint: bool = True,
) -> None:
    if include_checkpoint:
        parser.add_argument(
            "--checkpoint",
            type=Path,
            required=checkpoint_required,
            help="Optimization checkpoint to sample.",
        )
    parser.add_argument(
        "--sample-out-dir",
        "--sampling-out-dir",
        dest="sample_out_dir",
        type=Path,
        help="Sampling output directory. Defaults under the checkpoint run directory.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples per selected family.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for backtracking samples.",
    )
    parser.add_argument(
        "--family-start",
        type=int,
        default=0,
        help="First family index to sample.",
    )
    parser.add_argument(
        "--sample-max-families",
        dest="sample_max_families",
        type=int,
        help="Maximum number of families to sample.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=100_000,
        help="Maximum events per sampled reconciliation.",
    )
    parser.add_argument(
        "--backtrack-binary",
        type=Path,
        help="Rust backtracking binary; otherwise GPUREC_BACKTRACK_BIN or cargo fallback is used.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpurec",
        description="Optimize D/T/L reconciliation likelihoods and sample RecPhyloXML scenarios.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    optimize_parser = sub.add_parser(
        "optimize",
        help="Optimize D/T/L likelihood parameters.",
        description="Optimize D/T/L likelihood parameters from AleRax-style family inputs.",
    )
    _add_run_config_args(optimize_parser)

    sample_parser = sub.add_parser(
        "sample",
        help="Sample RecPhyloXML scenarios from a checkpoint.",
        description="Sample RecPhyloXML scenarios from a gpurec optimization checkpoint.",
    )
    _add_sampling_args(sample_parser, checkpoint_required=True)

    run_parser = sub.add_parser(
        "run",
        help="Optimize, then sample from the best checkpoint.",
        description="Run optimization, then sample from the best or latest checkpoint it produced.",
    )
    _add_run_config_args(run_parser)
    _add_sampling_args(run_parser, checkpoint_required=False, include_checkpoint=False)
    run_parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "optimize":
        try:
            config = _run_config_from_args(args)
        except ValueError as exc:
            parser.error(str(exc))
        result = optimize(config)
        print(
            f"status={result.status} reason={result.reason} "
            f"final_nll_bits={result.final_nll_bits:.6f} out_dir={result.out_dir}",
            flush=True,
        )
        return
    if args.command == "sample":
        try:
            sampling_config = SamplingConfig(
                checkpoint=args.checkpoint,
                out_dir=args.sample_out_dir,
                samples=args.samples,
                seed=args.seed,
                family_start=args.family_start,
                max_families=args.sample_max_families,
                max_events=args.max_events,
                backtrack_binary=args.backtrack_binary,
            )
            result = sample(sampling_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            parser.error(str(exc))
        print(
            f"sampled families={result.families_sampled} "
            f"samples={result.samples_per_family} xml={result.xml_files} "
            f"out_dir={result.out_dir}",
            flush=True,
        )
        return
    if args.command == "run":
        if args.checkpoint is not None:
            parser.error(
                "gpurec run samples from the checkpoint produced by this optimization; "
                "use gpurec sample --checkpoint to sample an existing checkpoint, or "
                "--resume-from to resume optimization"
            )
        try:
            run_config = _run_config_from_args(args)
        except ValueError as exc:
            parser.error(str(exc))
        opt_result = optimize(run_config)
        checkpoint = run_config.out_dir / "checkpoints" / "best.pt"
        if not checkpoint.exists():
            checkpoint = run_config.out_dir / "checkpoints" / "latest.pt"
        try:
            sampling_config = SamplingConfig(
                checkpoint=checkpoint,
                out_dir=args.sample_out_dir,
                samples=args.samples,
                seed=args.seed,
                family_start=args.family_start,
                max_families=args.sample_max_families,
                max_events=args.max_events,
                backtrack_binary=args.backtrack_binary,
            )
            sampling_result = sample(sampling_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            parser.error(str(exc))
        print(
            f"status={opt_result.status} reason={opt_result.reason} "
            f"sampled_families={sampling_result.families_sampled} "
            f"out_dir={run_config.out_dir}",
            flush=True,
        )
        return
    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()

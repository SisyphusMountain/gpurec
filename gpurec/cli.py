from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from gpurec.core.batch_planning import (
    normalize_batch_packing,
    normalize_family_chunk_size,
)


_EXPECTED_WORKFLOW_ERRORS = (ValueError, OSError, RuntimeError)
_RAW_THETA_CHECKPOINT_ERROR = "must contain a dictionary payload"


def _run_config_cli_override_fields() -> tuple[str, ...]:
    from dataclasses import fields

    from gpurec.workflow.config import RunConfig

    return tuple(field.name for field in fields(RunConfig))


def __getattr__(name: str) -> Any:
    if name == "_RUN_CONFIG_CLI_OVERRIDE_FIELDS":
        return _run_config_cli_override_fields()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _sampling_error_message(exc: BaseException) -> str:
    message = str(exc)
    if _RAW_THETA_CHECKPOINT_ERROR in message:
        return (
            f"{message}; --checkpoint must point to an optimization checkpoint "
            "such as checkpoints/best.pt or checkpoints/latest.pt, not "
            "theta_final.pt"
        )
    return message


def _exit_runtime_error(parser: argparse.ArgumentParser, message: str) -> None:
    parser.exit(status=1, message=f"error: {message}\n")


def optimize(config: Any) -> Any:
    from gpurec.workflow.optimize import optimize as _optimize

    return _optimize(config)


def sample(config: Any) -> Any:
    from gpurec.workflow.sampling import sample as _sample

    return _sample(config)


def _chunk_size(value: str) -> int:
    try:
        return int(normalize_family_chunk_size(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _dtype_name(value: str) -> str:
    from gpurec.workflow.config import dtype_name_from_name

    try:
        return dtype_name_from_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _batch_packing(value: str) -> str:
    try:
        return normalize_batch_packing(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _config_data(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError(
            "--config currently expects a flat JSON RunConfig file; "
            "Hydra-style YAML configs must be converted to JSON or passed as "
            "explicit CLI flags"
        )
    from gpurec.workflow.config import load_run_config_data

    return load_run_config_data(path)


def _set_if_present(data: dict[str, Any], args: argparse.Namespace, name: str) -> None:
    value = getattr(args, name)
    if value is not None:
        data[name] = value


def _validate_run_config_input_paths(config: RunConfig) -> None:
    for option, path in (
        ("--species-tree", config.species_tree),
        ("--families-file", config.families_file),
    ):
        if not path.is_file():
            raise ValueError(f"{option} path does not exist or is not a file: {path}")
    if config.resume_from is not None and not config.resume_from.is_file():
        raise ValueError(
            "--resume-from path does not exist or is not a file: "
            f"{config.resume_from}"
        )


def _validate_sampling_checkpoint_path(checkpoint: Path) -> None:
    path = checkpoint.expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"--checkpoint path does not exist or is not a file: {path}"
        )


def _run_config_from_args(args: argparse.Namespace) -> RunConfig:
    data = _config_data(args.config)
    from gpurec.workflow.config import RunConfig

    for name in _run_config_cli_override_fields():
        _set_if_present(data, args, name)
    missing = [
        name
        for name in ("species_tree", "families_file", "out_dir")
        if data.get(name) is None
    ]
    if missing:
        raise ValueError(f"missing required optimize option(s): {', '.join(missing)}")
    config = RunConfig.from_dict(data)
    _validate_run_config_input_paths(config)
    return config


def _sampling_config_from_args(
    args: argparse.Namespace,
    checkpoint: Path,
) -> SamplingConfig:
    from gpurec.workflow.config import SamplingConfig

    return SamplingConfig(
        checkpoint=checkpoint,
        out_dir=args.sample_out_dir,
        samples=args.samples,
        seed=args.seed,
        family_start=args.family_start,
        max_families=args.sample_max_families,
        max_events=args.max_events,
        backtrack_binary=args.backtrack_binary,
    )


def _ensure_backtracking_available(backtrack_binary: Path | None) -> None:
    from gpurec.backtracking import ensure_backtracking_available

    ensure_backtracking_available(backtrack_binary)


def _validate_run_sampling_args(args: argparse.Namespace, run_config: RunConfig) -> None:
    _sampling_config_from_args(
        args,
        run_config.out_dir / "checkpoints" / "sampling-argument-validation.pt",
    )


def _add_run_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Flat JSON RunConfig file; relative config paths resolve from the "
            "config file, and explicit CLI flags override matching fields."
        ),
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
        type=_dtype_name,
        metavar="{float32,float64}",
        help=(
            "Floating-point dtype; aliases include fp32/single and "
            "fp64/double. Workflow default: float32."
        ),
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
        help=(
            "Families per resident batch; use 0/all/none/null for one "
            "resident batch."
        ),
    )
    parser.add_argument(
        "--clade-budget",
        type=int,
        help="Clade budget for non-sequential resident-batch packing.",
    )
    parser.add_argument(
        "--batch-packing",
        type=_batch_packing,
        metavar="{sequential,clade_first_fit,depth_first_fit}",
        help=(
            "Resident-batch packing policy; aliases include "
            "contiguous/input_order, ffd/clade_ffd, and "
            "depth_ffd/wave_first_fit. Workflow default: depth_first_fit."
        ),
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


def _add_backtrack_binary_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backtrack-binary",
        type=Path,
        help=(
            "Rust backtracking binary. Installed sampling requires this or "
            "GPUREC_BACKTRACK_BIN; source trees can fall back to cargo when "
            "a Rust toolchain is present."
        ),
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
            help=(
                "Optimization checkpoint to sample, usually checkpoints/best.pt "
                "or checkpoints/latest.pt; theta_final.pt is only a raw tensor "
                "export."
            ),
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
    _add_backtrack_binary_arg(parser)


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
    optimize_parser.set_defaults(_command_parser=optimize_parser)

    sample_parser = sub.add_parser(
        "sample",
        help="Sample RecPhyloXML scenarios from a checkpoint.",
        description="Sample RecPhyloXML scenarios from a gpurec optimization checkpoint.",
    )
    _add_sampling_args(sample_parser, checkpoint_required=True)
    sample_parser.set_defaults(_command_parser=sample_parser)

    run_parser = sub.add_parser(
        "run",
        help="Optimize, then sample from the best checkpoint.",
        description="Run optimization, then sample from the best or latest checkpoint it produced.",
    )
    _add_run_config_args(run_parser)
    _add_sampling_args(run_parser, checkpoint_required=False, include_checkpoint=False)
    run_parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    run_parser.set_defaults(_command_parser=run_parser)

    backtrack_check_parser = sub.add_parser(
        "backtrack-check",
        help="Check Rust backtracking command availability.",
        description=(
            "Validate the Rust backtracking binary or source-tree cargo fallback "
            "by running --help without loading a checkpoint."
        ),
    )
    _add_backtrack_binary_arg(backtrack_check_parser)
    backtrack_check_parser.set_defaults(_command_parser=backtrack_check_parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command_parser = getattr(args, "_command_parser", parser)
    if args.command == "optimize":
        try:
            config = _run_config_from_args(args)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        try:
            result = optimize(config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, str(exc))
        print(
            f"status={result.status} reason={result.reason} "
            f"final_nll_bits={result.final_nll_bits:.6f} out_dir={result.out_dir}",
            flush=True,
        )
        if result.status == "failed":
            command_parser.exit(status=1)
        return
    if args.command == "sample":
        try:
            sampling_config = _sampling_config_from_args(args, args.checkpoint)
            _validate_sampling_checkpoint_path(sampling_config.checkpoint)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(_sampling_error_message(exc))
        try:
            result = sample(sampling_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, _sampling_error_message(exc))
        print(
            f"sampled families={result.families_sampled} "
            f"samples={result.samples_per_family} xml={result.xml_files} "
            f"out_dir={result.out_dir}",
            flush=True,
        )
        return
    if args.command == "backtrack-check":
        try:
            _ensure_backtracking_available(args.backtrack_binary)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, str(exc))
        print("backtracking_available=true", flush=True)
        return
    if args.command == "run":
        if args.checkpoint is not None:
            command_parser.error(
                "gpurec run samples from the checkpoint produced by this optimization; "
                "use gpurec sample --checkpoint to sample an existing checkpoint, or "
                "--resume-from to resume optimization"
            )
        try:
            run_config = _run_config_from_args(args)
            _validate_run_sampling_args(args, run_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        try:
            _ensure_backtracking_available(args.backtrack_binary)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, str(exc))
        try:
            opt_result = optimize(run_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, str(exc))
        if opt_result.status == "failed":
            command_parser.exit(
                status=1,
                message=(
                    "optimization failed; refusing to sample from a failed run "
                    f"({opt_result.reason})"
                    "\n"
                ),
            )
        checkpoint = getattr(opt_result, "sampling_checkpoint", None)
        if checkpoint is None:
            checkpoint = run_config.out_dir / "checkpoints" / "best.pt"
            if not checkpoint.exists():
                checkpoint = run_config.out_dir / "checkpoints" / "latest.pt"
        else:
            checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            _exit_runtime_error(
                command_parser,
                "optimization completed but no sampling checkpoint was found "
                f"at {checkpoint}",
            )
        try:
            sampling_config = _sampling_config_from_args(args, checkpoint)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(_sampling_error_message(exc))
        try:
            sampling_result = sample(sampling_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, _sampling_error_message(exc))
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

from __future__ import annotations

import argparse
import json
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
    value = getattr(args, name, None)
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


def _validate_run_config_family_references(config: RunConfig) -> dict[str, int]:
    from gpurec.core.model import parse_alerax_family_file

    family_names, tree_paths, leaf_species_maps = parse_alerax_family_file(
        config.families_file,
        start=config.start,
        max_families=config.max_families,
    )
    missing: list[tuple[str, Path]] = []
    gene_tree_files = 0
    for family, paths in zip(family_names, tree_paths):
        for raw_path in paths:
            gene_tree_files += 1
            path = Path(raw_path)
            if not path.is_file():
                missing.append((family, path))
    if missing:
        preview = "; ".join(
            f"{family}: {path}" for family, path in missing[:5]
        )
        suffix = "" if len(missing) <= 5 else f"; ... {len(missing) - 5} more"
        raise ValueError(
            "AleRax family file references missing gene-tree path(s): "
            f"{preview}{suffix}"
        )
    return {
        "families": len(family_names),
        "gene_tree_files": gene_tree_files,
        "mapped_families": sum(1 for mapping in leaf_species_maps if mapping),
    }


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


def _preflight_run_config(config: RunConfig) -> dict[str, int]:
    return _validate_run_config_family_references(config)


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


def _config_template_data(args: argparse.Namespace) -> dict[str, Any]:
    from gpurec.workflow.config import DEFAULT_ADAGRAD_RESTART_SCHEDULE

    data: dict[str, Any] = {
        "species_tree": str(args.species_tree),
        "families_file": str(args.families_file),
        "out_dir": str(args.out_dir),
        "mode": args.mode,
        "device": args.device,
        "dtype": "float32",
        "optimizer": "auto",
        "family_chunk_size": 0,
        "batch_packing": "depth_first_fit",
        "clade_budget": 500_000,
        "fixed_iters_pi": 16,
        "neumann_terms": 16,
        "steps": 5000,
        "log_every": 1,
        "checkpoint_every": 1,
    }
    if args.mode == "genewise":
        data.update(
            {
                "fd_adam_warmup_steps": 3,
                "fd_hessian_refresh_steps": 16,
            }
        )
    elif args.mode == "specieswise":
        data.update(
            {
                "adagrad_restart_schedule": DEFAULT_ADAGRAD_RESTART_SCHEDULE,
                "adagrad_restart_final_check_iters": 128,
            }
        )
    return data


def _write_config_template(args: argparse.Namespace) -> Path | None:
    text = json.dumps(_config_template_data(args), indent=2) + "\n"
    output = args.output
    if output is None:
        print(text, end="", flush=True)
        return None
    output = output.expanduser().resolve()
    if output.exists() and not args.force:
        raise ValueError(
            f"output config already exists: {output}; use --force to overwrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


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
        "--preprocess-cpu-cores",
        type=int,
        help=(
            "Worker thread count for CPU preprocessing. Workflow default uses "
            "Rust preprocessing's runtime default."
        ),
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
    parser.add_argument(
        "--small-family-max-leaves",
        type=int,
        help=(
            "Plan families with at most this many leaves before larger "
            "families; use 0 to disable. Workflow default: 0."
        ),
    )
    parser.add_argument(
        "--adaptive-rebatch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable adaptive resident-batch rebuilding for supported genewise runs.",
    )
    parser.add_argument(
        "--adaptive-rebatch-fraction",
        type=float,
        help="Fraction threshold used by adaptive resident-batch rebuilding.",
    )
    parser.add_argument(
        "--adaptive-rebatch-check-interval",
        type=int,
        help="Step interval for adaptive resident-batch rebuilding checks.",
    )
    parser.add_argument(
        "--adaptive-rebatch-min-remaining-families",
        type=int,
        help="Minimum remaining families before adaptive rebatching can run.",
    )
    parser.add_argument(
        "--fixed-iters-e",
        type=int,
        help=(
            "Fixed E iterations per solve. In specieswise mode, fixed Pi "
            "budgets above 16 force E to be at least the Pi budget."
        ),
    )
    parser.add_argument("--max-iters-e", type=int, help="Maximum adaptive E iterations.")
    parser.add_argument("--tol-e", type=float, help="E fixed-point convergence tolerance.")
    parser.add_argument("--fixed-iters-pi", type=int, help="Fixed Pi iterations per solve.")
    parser.add_argument(
        "--neumann-terms",
        type=int,
        help="Terms for implicit-gradient Neumann series.",
    )
    parser.add_argument(
        "--solver-warmup-iters",
        type=int,
        help=(
            "Initial fixed solver budget for supported genewise active-batch "
            "optimizers and specieswise runs whose full Pi budget is larger; "
            "hessian-sgd keeps E at --fixed-iters-e and uses this only for "
            "Pi/Neumann. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--final-check-iters",
        type=int,
        help=(
            "Final validation solver budget used only to compare the final loss "
            "and gradient against the configured full budget. Specieswise mode "
            "also uses this for fixed E iterations; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--solver-warmup-grad-inf-tol",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--solver-warmup-loss-patience",
        type=int,
        help=(
            "Switch genewise batched-LBFGS from warmup to full solvers after "
            "this many flat warmup steps."
        ),
    )
    parser.add_argument(
        "--adaptive-iters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable adaptive E/Pi solver iteration stopping.",
    )
    parser.add_argument(
        "--adaptive-neumann-terms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable legacy adaptive backward Neumann gradient-convergence "
            "checks. Default is disabled."
        ),
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
    parser.add_argument(
        "--min-rate",
        type=float,
        help="Minimum allowed D/L/T rate; defaults to 2^-30.",
    )
    parser.add_argument(
        "--max-rate",
        type=float,
        help="Maximum allowed D/L/T rate; defaults to 2.",
    )
    parser.add_argument(
        "--optimizer",
        choices=(
            "auto",
            "adam",
            "adagrad",
            "projected-sgd",
            "lbfgs",
            "adam-lbfgs",
            "projected-lbfgs",
            "lbfgsb",
            "batched-lbfgs",
            "adam-fd-newton",
            "hessian-sgd",
            "adagrad-restarts",
        ),
        help=(
            "Optimizer schedule. auto uses hessian-sgd for genewise mode, "
            "adagrad-restarts for specieswise mode, and adam otherwise."
        ),
    )
    parser.add_argument("--steps", type=int, help="Maximum optimization steps.")
    parser.add_argument(
        "--lr",
        type=float,
        help="Adam/Adagrad learning rate or hessian-sgd preconditioned step scale.",
    )
    parser.add_argument(
        "--adam-warmup-steps",
        type=int,
        help="Adam steps before LBFGS in adam-lbfgs mode.",
    )
    parser.add_argument(
        "--fd-adam-warmup-steps",
        type=int,
        help="Adam steps per resident batch before finite-difference Newton updates.",
    )
    parser.add_argument(
        "--fd-hessian-refresh-steps",
        type=int,
        help="Newton steps between full finite-difference Hessian refreshes.",
    )
    parser.add_argument(
        "--hessian-sgd-normal-fixed-iters-pi",
        type=int,
        help=(
            "Optional Pi iteration budget for hessian-sgd full-stage steps."
        ),
    )
    parser.add_argument(
        "--hessian-sgd-normal-neumann-terms",
        type=int,
        help=(
            "Optional Neumann iteration budget for hessian-sgd full-stage steps."
        ),
    )
    parser.add_argument(
        "--adagrad-restart-schedule",
        help=(
            "Specieswise adagrad-restarts phase schedule as "
            "budget:lr:steps entries, for example 8:1.0:60,16:0.5:35."
        ),
    )
    parser.add_argument(
        "--adagrad-restart-final-check-iters",
        type=int,
        help=(
            "Final specieswise validation budget for adagrad-restarts; "
            "workflow default: 128."
        ),
    )
    parser.add_argument("--lbfgs-lr", type=float, help="LBFGS learning rate.")
    parser.add_argument("--lbfgs-history-size", type=int, help="LBFGS history size.")
    parser.add_argument("--lbfgs-max-iter", type=int, help="LBFGS inner iterations per step.")
    parser.add_argument("--lbfgs-max-ls", type=int, help="Batched LBFGS line-search probes.")
    parser.add_argument(
        "--lbfgs-line-search",
        choices=("none", "strong_wolfe"),
        help="LBFGS line-search mode.",
    )
    parser.add_argument(
        "--fd-hessian-epsilon",
        type=float,
        help="Finite-difference epsilon for Hessian-conditioned genewise probes.",
    )
    parser.add_argument(
        "--fd-newton-damping",
        type=float,
        help="Diagonal damping added to finite-difference Hessians.",
    )
    parser.add_argument(
        "--grad-inf-tol",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--loss-change-tol",
        type=float,
        help=(
            "Loss-change stopping tolerance; genewise active-batch optimizers "
            "apply this per active family."
        ),
    )
    parser.add_argument(
        "--projected-grad-tol",
        type=float,
        help=(
            "Projected-gradient infinity-norm tolerance for projected optimizers; "
            "projected-lbfgs/lbfgsb keep optimizing instead of stopping "
            "while this is exceeded."
        ),
    )
    parser.add_argument(
        "--projected-lbfgs-min-lr",
        type=float,
        help="Minimum projected-lbfgs base learning rate after automatic backoff.",
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
        help="Console progress print interval in optimization steps; history is recorded every step.",
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

    validate_parser = sub.add_parser(
        "validate-config",
        help="Validate an optimization config without CUDA.",
        description=(
            "Validate a flat RunConfig JSON file or equivalent CLI flags, "
            "including AleRax family references, without constructing the "
            "CUDA likelihood model."
        ),
    )
    _add_run_config_args(validate_parser)
    validate_parser.set_defaults(_command_parser=validate_parser)

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

    template_parser = sub.add_parser(
        "config-template",
        help="Print or write a flat JSON RunConfig template.",
        description=(
            "Print or write a flat JSON RunConfig template for installed "
            "production workflows. The mode keeps optimizer=auto so genewise "
            "uses hessian-sgd and specieswise uses adagrad-restarts."
        ),
    )
    template_parser.add_argument(
        "--mode",
        choices=("genewise", "specieswise", "global"),
        default="genewise",
        help="Template parameter-sharing mode. Default: genewise.",
    )
    template_parser.add_argument(
        "--species-tree",
        default="S.tree",
        help="Species tree path to place in the template.",
    )
    template_parser.add_argument(
        "--families-file",
        default="families.txt",
        help="AleRax [FAMILIES] path to place in the template.",
    )
    template_parser.add_argument(
        "--out-dir",
        default="output_gpurec",
        help="Output directory to place in the template.",
    )
    template_parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to place in the template. Default: cuda.",
    )
    template_parser.add_argument(
        "--output",
        type=Path,
        help="Write the template to this path instead of stdout.",
    )
    template_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --output if it already exists.",
    )
    template_parser.set_defaults(_command_parser=template_parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command_parser = getattr(args, "_command_parser", parser)
    if args.command == "config-template":
        try:
            output = _write_config_template(args)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        if output is not None:
            print(f"config_template={output}", flush=True)
        return
    if args.command == "optimize":
        try:
            config = _run_config_from_args(args)
            _preflight_run_config(config)
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
    if args.command == "validate-config":
        try:
            config = _run_config_from_args(args)
            summary = _preflight_run_config(config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        print(
            "valid_config=true "
            f"mode={config.mode} optimizer={config.optimizer} "
            f"families={summary['families']} "
            f"gene_tree_files={summary['gene_tree_files']} "
            f"mapped_families={summary['mapped_families']} "
            f"device={config.device} out_dir={config.out_dir}",
            flush=True,
        )
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
            _preflight_run_config(run_config)
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

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gpurec.workflow import RunConfig, SamplingConfig, optimize, sample


def _chunk_size(value: str) -> int | str | None:
    text = value.strip().lower()
    if text in {"none", "null"}:
        return None
    if text in {"auto"}:
        return "auto"
    return int(text)


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
        data = json.loads(path.read_text(encoding="utf-8"))
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
        raise SystemExit(f"missing required optimize option(s): {', '.join(missing)}")
    return RunConfig.from_dict(data)


def _add_run_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--species-tree", type=Path)
    parser.add_argument("--families-file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--mode", choices=("genewise", "global", "specieswise"))
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--start", type=int)
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--preprocess-cache", type=Path)
    parser.add_argument(
        "--refresh-preprocess-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--family-chunk-size", type=_chunk_size)
    parser.add_argument("--clade-budget", type=int)
    parser.add_argument(
        "--batch-packing",
        choices=("sequential", "clade_first_fit", "depth_first_fit"),
    )
    parser.add_argument("--max-wave-size", type=int)
    parser.add_argument("--fixed-iters-e", type=int)
    parser.add_argument("--max-iters-e", type=int)
    parser.add_argument("--tol-e", type=float)
    parser.add_argument("--fixed-iters-pi", type=int)
    parser.add_argument("--neumann-terms", type=int)
    parser.add_argument("--adaptive-iters", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--convergence-check-interval", type=int)
    parser.add_argument("--e-logsumexp-tol", type=float)
    parser.add_argument("--pi-max-diff-tol", type=float)
    parser.add_argument("--gradient-change-tol", type=float)
    parser.add_argument("--gradient-change-rtol", type=float)
    parser.add_argument("--theta-init-d", type=float)
    parser.add_argument("--theta-init-l", type=float)
    parser.add_argument("--theta-init-t", type=float)
    parser.add_argument("--min-rate", type=float)
    parser.add_argument("--max-rate", type=float)
    parser.add_argument("--optimizer", choices=("adam", "adagrad", "lbfgs", "adam-lbfgs"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--adam-warmup-steps", type=int)
    parser.add_argument("--lbfgs-lr", type=float)
    parser.add_argument("--lbfgs-history-size", type=int)
    parser.add_argument("--lbfgs-max-iter", type=int)
    parser.add_argument("--lbfgs-line-search", choices=("none", "strong_wolfe"))
    parser.add_argument("--grad-inf-tol", type=float)
    parser.add_argument("--loss-change-tol", type=float)
    parser.add_argument("--loss-patience", type=int)
    parser.add_argument("--best-likelihood-patience", type=int)
    parser.add_argument("--best-likelihood-min-delta", type=float)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--resume-from", type=Path)


def _add_sampling_args(parser: argparse.ArgumentParser, *, checkpoint_required: bool) -> None:
    parser.add_argument("--checkpoint", type=Path, required=checkpoint_required)
    parser.add_argument("--sample-out-dir", "--sampling-out-dir", dest="sample_out_dir", type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family-start", type=int, default=0)
    parser.add_argument("--sample-max-families", dest="sample_max_families", type=int)
    parser.add_argument("--max-events", type=int, default=100_000)
    parser.add_argument("--backtrack-binary", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpurec")
    sub = parser.add_subparsers(dest="command", required=True)

    optimize_parser = sub.add_parser("optimize", help="Optimize D/T/L likelihood parameters.")
    _add_run_config_args(optimize_parser)

    sample_parser = sub.add_parser("sample", help="Sample RecPhyloXML scenarios from a checkpoint.")
    _add_sampling_args(sample_parser, checkpoint_required=True)

    run_parser = sub.add_parser("run", help="Optimize, then sample from the best checkpoint.")
    _add_run_config_args(run_parser)
    _add_sampling_args(run_parser, checkpoint_required=False)
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
        result = sample(
            SamplingConfig(
                checkpoint=args.checkpoint,
                out_dir=args.sample_out_dir,
                samples=args.samples,
                seed=args.seed,
                family_start=args.family_start,
                max_families=args.sample_max_families,
                max_events=args.max_events,
                backtrack_binary=args.backtrack_binary,
            )
        )
        print(
            f"sampled families={result.families_sampled} "
            f"samples={result.samples_per_family} xml={result.xml_files} "
            f"out_dir={result.out_dir}",
            flush=True,
        )
        return
    if args.command == "run":
        try:
            run_config = _run_config_from_args(args)
        except ValueError as exc:
            parser.error(str(exc))
        opt_result = optimize(run_config)
        checkpoint = args.checkpoint or (run_config.out_dir / "checkpoints" / "best.pt")
        if not checkpoint.exists():
            checkpoint = run_config.out_dir / "checkpoints" / "latest.pt"
        sampling_result = sample(
            SamplingConfig(
                checkpoint=checkpoint,
                out_dir=args.sample_out_dir,
                samples=args.samples,
                seed=args.seed,
                family_start=args.family_start,
                max_families=args.sample_max_families,
                max_events=args.max_events,
                backtrack_binary=args.backtrack_binary,
            )
        )
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

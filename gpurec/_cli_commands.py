from __future__ import annotations

from gpurec._cli_helpers import *  # noqa: F401,F403

def _add_run_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "relative config paths resolve from the config file. "
            "Use '-' as --config to read JSON from stdin; in that mode, "
            "relative paths resolve from stdin's current directory. "
            "Explicit CLI flags override matching fields. "
            "Flat JSON RunConfig file may also be provided."
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
        type=_mode_name,
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
        help=(
            "Clade budget for non-sequential resident-batch packing. "
            "Workflow default: 315000."
        ),
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
            "Switch genewise active-batch optimizers from warmup to full "
            "solvers after this many flat warmup steps."
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
            "Disabled compatibility flag; enabling it is rejected because the "
            "adaptive Neumann path is not part of the supported production "
            "optimization route."
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
        type=_optimizer_name,
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
            "adagrad-restarts-lbfgsb",
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
        help="Adam steps per resident batch before Hessian-conditioned genewise updates.",
    )
    parser.add_argument(
        "--fd-hessian-refresh-steps",
        type=int,
        help="Hessian-conditioned genewise steps between full finite-difference Hessian refreshes.",
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
        "--hessian-sgd-pi-adjoint-warmstart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the experimental staged Pi-adjoint warm-start cache for "
            "genewise hessian-sgd runs. Workflow default: disabled."
        ),
    )
    parser.add_argument(
        "--pi-fixed-point-relaxation",
        type=float,
        help=(
            "Experimental Pi-adjoint fixed-point relaxation factor for "
            "warm-started genewise hessian-sgd runs. Workflow default: 1.0."
        ),
    )
    parser.add_argument(
        "--hessian-sgd-validation-interval",
        type=int,
        help=(
            "Full-stage hessian-sgd cadence for high-budget validation "
            "gradient steps; 0 disables periodic validation."
        ),
    )
    parser.add_argument(
        "--hessian-sgd-validation-fixed-iters-pi",
        type=int,
        help=(
            "Optional Pi iteration budget for periodic hessian-sgd validation "
            "gradient steps."
        ),
    )
    parser.add_argument(
        "--hessian-sgd-validation-neumann-terms",
        type=int,
        help=(
            "Optional Neumann budget for periodic hessian-sgd validation "
            "gradient steps."
        ),
    )
    parser.add_argument(
        "--adagrad-restart-schedule",
        help=(
            "Specieswise adagrad-restarts phase schedule as "
            "budget:lr:steps or E/Pi[/Neumann]:lr:steps entries, for "
            "example 8/4:1.0:60,16:0.5:35. Also controls the Adagrad "
            "prefix of adagrad-restarts-lbfgsb."
        ),
    )
    parser.add_argument(
        "--adagrad-restart-final-check-iters",
        type=int,
        help=(
            "Final specieswise validation budget for adagrad-restarts and "
            "adagrad-restarts-lbfgsb; "
            "workflow default: 128."
        ),
    )
    parser.add_argument(
        "--adagrad-restart-phase-loss-patience",
        type=int,
        help=(
            "For specieswise adagrad-restarts, advance to the next restart "
            "phase after this many flat-loss steps; 0 keeps fixed phase lengths. "
            "The same rule controls the Adagrad prefix of "
            "adagrad-restarts-lbfgsb."
        ),
    )
    parser.add_argument("--lbfgs-lr", type=float, help="LBFGS learning rate.")
    parser.add_argument("--lbfgs-history-size", type=int, help="LBFGS history size.")
    parser.add_argument("--lbfgs-max-iter", type=int, help="LBFGS inner iterations per step.")
    parser.add_argument("--lbfgs-max-ls", type=int, help="Batched LBFGS line-search probes.")
    parser.add_argument(
        "--lbfgsb-high-kkt-stop-patience",
        type=int,
        help=(
            "For lbfgsb, stop after this many consecutive high-KKT "
            "tiny-progress rows; 0 disables the stop."
        ),
    )
    parser.add_argument(
        "--lbfgsb-high-kkt-stop-min-fallbacks",
        type=int,
        help=(
            "Minimum accepted lbfgsb fallback rows before the high-KKT stop can "
            "trigger."
        ),
    )
    parser.add_argument(
        "--lbfgsb-fallback-max-coordinates",
        type=int,
        help=(
            "Maximum coordinate sign-fallback candidates for lbfgsb fallback "
            "competition; 0 disables coordinate fallback."
        ),
    )
    parser.add_argument(
        "--lbfgsb-fallback-max-loss-evals",
        type=int,
        help=(
            "Optional per-step loss-only evaluation budget for lbfgsb fallback "
            "line searches and fallback competition."
        ),
    )
    parser.add_argument(
        "--lbfgsb-fallback-resolution-competition-factor",
        type=float,
        help=(
            "For lbfgsb fallback competition, also challenge accepted fallback "
            "moves whose decrease is at most this multiple of the fp loss "
            "resolution; 0 keeps only the ordinary tiny-progress trigger."
        ),
    )
    parser.add_argument(
        "--lbfgsb-best-retry-attempts",
        type=int,
        help=(
            "For lbfgsb, reload the best checkpoint this many times when a "
            "terminal plateau is reached, preserving serialized LBFGS-B state."
        ),
    )
    parser.add_argument(
        "--lbfgsb-loss-change-tol-schedule",
        help=(
            "Optional lbfgsb loss-stop schedule as "
            "loss_change_tol:loss_patience entries, for example 0.25:2,0.1:2."
        ),
    )
    parser.add_argument(
        "--lbfgsb-loss-schedule-force-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When an lbfgsb loss-stop schedule advances, force the next row to "
            "start from the projected-gradient fallback."
        ),
    )
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
        "--loss-stop-projected-grad-gate",
        dest="loss_stop_projected_grad_gate",
        action="store_true",
        default=None,
        help=(
            "Require projected-lbfgs/lbfgsb to pass --projected-grad-tol before "
            "loss-change patience can stop the run."
        ),
    )
    parser.add_argument(
        "--no-loss-stop-projected-grad-gate",
        dest="loss_stop_projected_grad_gate",
        action="store_false",
        default=None,
        help=(
            "Allow loss-change patience to stop projected-lbfgs/lbfgsb even when "
            "the projected-gradient diagnostic is still above tolerance."
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


def _add_preprocess_native_lib_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preprocess-native-lib",
        type=Path,
        help=(
            "Native Rust preprocessing extension. Installed workflow "
            "preprocessing requires this or GPUREC_PREPROCESS_NATIVE_LIB; source "
            "trees can fall back to Cargo when a Rust toolchain is present."
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
    _add_require_mode_default_optimizer_arg(optimize_parser)
    _add_require_production_default_route_arg(optimize_parser)
    optimize_parser.add_argument(
        "--require-converged",
        action="store_true",
        help=(
            "After printing the optimization status, exit with status 1 unless "
            "the status is converged."
        ),
    )
    optimize_parser.add_argument(
        "--require-final-check-ok",
        action="store_true",
        help=(
            "After printing the optimization status, exit with status 1 unless "
            "final_check_status is ok."
        ),
    )
    optimize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config, routes, and CPU preprocessing only; print "
            "estimated workflow readiness/counts without running optimization."
        ),
    )
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
    _add_json_output_arg(validate_parser)
    _add_require_mode_default_optimizer_arg(validate_parser)
    _add_require_production_default_route_arg(validate_parser)
    validate_parser.add_argument(
        "--check-preprocess",
        action="store_true",
        help=(
            "Also run CPU preprocessing with the retained Rust parser to check "
            "selected Newick trees and leaf/species mappings, then report "
            "whether the species-node count passes the CUDA backward S > 256 gate."
        ),
    )
    validate_parser.add_argument(
        "--require-cuda-backward-ready",
        action="store_true",
        help=(
            "With --check-preprocess, fail unless the species-node count passes "
            "the retained CUDA backward S > 256 gate."
        ),
    )
    validate_parser.add_argument(
        "--explain-config",
        action="store_true",
        help=(
            "Include effective-config defaults and route/optimizer resolution "
            "details to explain why selected defaults were chosen."
        ),
    )
    validate_parser.set_defaults(_command_parser=validate_parser)

    validate_inputs_parser = sub.add_parser(
        "validate-inputs",
        help="Validate AleRax input files and references without CUDA.",
        description=(
            "Validate species tree and AleRax family declarations, optionally "+
            "running CPU preprocessing to validate Newick parsing and mapping "
            "coverage."
        ),
    )
    validate_inputs_parser.add_argument(
        "--species-tree",
        type=Path,
        required=True,
        help="Species-tree Newick path.",
    )
    validate_inputs_parser.add_argument(
        "--families-file",
        type=Path,
        required=True,
        help="AleRax [FAMILIES] file path.",
    )
    validate_inputs_parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First family index to validate.",
    )
    validate_inputs_parser.add_argument(
        "--max-families",
        type=int,
        help="Maximum number of families to validate.",
    )
    validate_inputs_parser.add_argument(
        "--mode",
        type=_mode_name,
        choices=("genewise", "global", "specieswise"),
        default="genewise",
        help=(
            "Parameter-sharing mode used during preprocessing. Workflow "
            "default: genewise."
        ),
    )
    validate_inputs_parser.add_argument(
        "--preprocess-cpu-cores",
        type=int,
        help=(
            "Worker thread count for CPU preprocessing. Workflow default uses "
            "Rust preprocessing's runtime default."
        ),
    )
    _add_json_output_arg(validate_inputs_parser)
    validate_inputs_parser.add_argument(
        "--check-preprocess",
        action="store_true",
        help=(
            "Also run CPU preprocessing with the retained Rust parser to check "
            "selected Newick trees and leaf/species mappings, then report "
            "whether the species-node count passes the CUDA backward S > 256 gate."
        ),
    )
    validate_inputs_parser.add_argument(
        "--require-cuda-backward-ready",
        action="store_true",
        help=(
            "With --check-preprocess, fail unless the species-node count "
            "passes the retained CUDA backward S > 256 gate."
        ),
    )
    validate_inputs_parser.set_defaults(_command_parser=validate_inputs_parser)

    sample_parser = sub.add_parser(
        "sample",
        help="Sample RecPhyloXML scenarios from a checkpoint.",
        description="Sample RecPhyloXML scenarios from a gpurec optimization checkpoint.",
    )
    _add_sampling_args(sample_parser, checkpoint_required=True)
    _add_require_mode_default_optimizer_arg(sample_parser)
    _add_require_production_default_route_arg(sample_parser)
    sample_parser.set_defaults(_command_parser=sample_parser)

    run_parser = sub.add_parser(
        "run",
        help="Optimize, then sample from the best checkpoint.",
        description="Run optimization, then sample from the best or latest checkpoint it produced.",
    )
    _add_run_config_args(run_parser)
    _add_sampling_args(run_parser, checkpoint_required=False, include_checkpoint=False)
    _add_require_mode_default_optimizer_arg(run_parser)
    _add_require_production_default_route_arg(run_parser)
    run_parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    run_parser.add_argument(
        "--require-converged",
        action="store_true",
        help=(
            "After optimization, print the optimization status and exit before "
            "sampling unless the status is converged."
        ),
    )
    run_parser.add_argument(
        "--require-final-check-ok",
        action="store_true",
        help=(
            "After optimization, print the optimization status and exit before "
            "sampling unless final_check_status is ok."
        ),
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config, routes, sampling args, and CPU preprocessing "
            "only; print estimated workflow readiness/counts without running "
            "optimization or sampling."
        ),
    )
    run_parser.set_defaults(_command_parser=run_parser)

    backtrack_check_parser = sub.add_parser(
        "backtrack-check",
        help="Check Rust backtracking command availability.",
        description=(
            "Validate the Rust backtracking binary or source-tree cargo fallback "
            "by running --help without loading a checkpoint."
        ),
    )
    _add_json_output_arg(backtrack_check_parser)
    _add_backtrack_binary_arg(backtrack_check_parser)
    backtrack_check_parser.set_defaults(_command_parser=backtrack_check_parser)

    preprocess_check_parser = sub.add_parser(
        "preprocess-check",
        help="Check Rust preprocessing native extension availability.",
        description=(
            "Validate the Rust preprocessing native extension or source-tree "
            "Cargo build fallback without reading dataset files."
        ),
    )
    _add_json_output_arg(preprocess_check_parser)
    _add_preprocess_native_lib_arg(preprocess_check_parser)
    preprocess_check_parser.set_defaults(_command_parser=preprocess_check_parser)

    doctor_parser = sub.add_parser(
        "doctor",
        help="Print workflow readiness checks before running optimization.",
        description=(
            "Collect installation and runtime readiness for Python runtime, "
            "PyTorch, Triton, native preprocessing, backtracking binary, "
            "and a writable output directory."
        ),
    )
    _add_json_output_arg(doctor_parser)
    _add_preprocess_native_lib_arg(doctor_parser)
    _add_backtrack_binary_arg(doctor_parser)
    doctor_parser.add_argument(
        "--out-dir",
        type=Path,
        help="Directory to probe for writable tempfile checks when validating out-dir readiness.",
    )
    doctor_parser.set_defaults(_command_parser=doctor_parser)

    checkpoint_info_parser = sub.add_parser(
        "checkpoint-info",
        help="Print optimization checkpoint status and route metadata.",
        description=(
            "Safely inspect a gpurec optimization checkpoint without building "
            "the CUDA likelihood model."
        ),
    )
    _add_json_output_arg(checkpoint_info_parser)
    checkpoint_info_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Optimization checkpoint to inspect, usually checkpoints/best.pt or latest.pt.",
    )
    checkpoint_info_parser.add_argument(
        "--require-final-check-ok",
        action="store_true",
        help=(
            "Exit with status 1 after printing checkpoint info unless the "
            "checkpoint last row has optimizer/final_check_status ok."
        ),
    )
    _add_require_mode_default_optimizer_arg(checkpoint_info_parser)
    _add_require_production_default_route_arg(checkpoint_info_parser)
    checkpoint_info_parser.set_defaults(_command_parser=checkpoint_info_parser)

    summary_info_parser = sub.add_parser(
        "summary-info",
        help="Print optimization summary status and route metadata.",
        description=(
            "Inspect a gpurec optimization summary.json file without building "
            "the CUDA likelihood model."
        ),
    )
    _add_json_output_arg(summary_info_parser)
    summary_info_parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Optimization summary.json file to inspect.",
    )
    summary_info_parser.add_argument(
        "--require-converged",
        action="store_true",
        help=(
            "Exit with status 1 after printing the summary unless "
            "summary.status is converged."
        ),
    )
    summary_info_parser.add_argument(
        "--require-final-check-ok",
        action="store_true",
        help=(
            "Exit with status 1 after printing the summary unless "
            "summary.final_check_status is ok."
        ),
    )
    _add_require_mode_default_optimizer_arg(summary_info_parser)
    _add_require_production_default_route_arg(summary_info_parser)
    summary_info_parser.set_defaults(_command_parser=summary_info_parser)

    template_parser = sub.add_parser(
        "config-template",
        help="Print or write a flat JSON RunConfig template.",
        description=(
            "Print or write a flat JSON RunConfig template. Genewise and "
            "specieswise templates are production-route starters; global "
            "remains a mode-default Adam diagnostic outside "
            "--require-production-default-route."
        ),
    )
    template_parser.add_argument(
        "--mode",
        type=_mode_name,
        choices=("genewise", "specieswise", "global"),
        default="genewise",
        help=(
            "Template parameter-sharing mode. Genewise/specieswise are "
            "production-route starters; global is a diagnostic Adam template. "
            "Default: genewise."
        ),
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
    invocation_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    command_parser = getattr(args, "_command_parser", parser)
    if args.command == "config-template":
        try:
            output = _write_config_template(args)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        if output is not None:
            print(_optional_text("config_template", output), flush=True)
        return
    if args.command == "optimize":
        try:
            prepared = _prepare_run_config_command(
                args,
                command_parser,
                check_preprocess=args.dry_run,
            )
            config = prepared.config
            route_metadata = prepared.route_metadata
            summary = prepared.summary
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        route_metadata_for_report = route_metadata
        if args.dry_run:
            print(
                _workflow_dry_run_text(
                    command="optimize",
                    config=config,
                    summary=summary,
                    route_metadata=route_metadata_for_report,
                ),
                flush=True,
            )
            return
        try:
            result = _run_optimize_command(config, invocation_argv)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _suggested_exit_error(
                command_parser, exc, _SUGGEST_RUN_OPTIMIZE_FAILURE
            )
        print(
            f"{_optimization_result_text(result)} "
            f"{_optional_text('out_dir', result.out_dir)}",
            flush=True,
        )
        if result.status == "failed":
            command_parser.exit(status=1)
        if args.require_converged and result.status != "converged":
            command_parser.exit(
                status=1,
                message=(
                    _with_suggestion(
                        "optimization status is "
                        f"{result.status!r}; expected 'converged'",
                        "inspect summary/checkpoint diagnostics and resume with higher steps or adjusted optimizer settings",
                    )
                    + "\n"
                ),
            )
        if args.require_final_check_ok:
            _exit_unless_final_check_ok(
                command_parser,
                getattr(result, "final_check_status", None),
                subject="optimization",
            )
        return
    if args.command == "validate-config":
        if args.require_cuda_backward_ready and not args.check_preprocess:
            _suggested_command_error(
                command_parser,
                "--require-cuda-backward-ready requires --check-preprocess",
                _SUGGEST_REQUIRE_CUDA_BACKWARD_READY,
            )
        try:
            prepared = _prepare_run_config_command(
                args,
                command_parser,
                check_preprocess=args.check_preprocess,
            )
            raw_config_data = prepared.raw_config_data
            config = prepared.config
            route_metadata_for_report = prepared.route_metadata
            summary = prepared.summary
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        explanation = (
            _run_config_explanation(
                config,
                raw_config_data=raw_config_data,
                route_metadata=route_metadata_for_report,
            )
            if args.explain_config
            else None
        )
        preprocess_text = ""
        if args.check_preprocess:
            cuda_backward_ready = (
                "true" if summary["cuda_backward_ready"] else "false"
            )
            cuda_backward_reason = _optional_text(
                "cuda_backward_ready_reason",
                summary.get("cuda_backward_ready_reason"),
            )
            preprocess_text = (
                f" preprocess_checked=true"
                f" preprocessed_families={summary['preprocessed_families']}"
                f" preprocessed_species_nodes={summary['preprocessed_species_nodes']}"
                f" cuda_backward_ready={cuda_backward_ready}"
                f" {cuda_backward_reason}"
            )
            if (
                args.require_cuda_backward_ready
                and not summary["cuda_backward_ready"]
            ):
                command_parser.error(
                    _with_suggestion(
                        "cuda_backward_ready=false "
                        f"{cuda_backward_reason}; retained CUDA backward requires "
                        "more than 256 postorder species nodes",
                        _SUGGEST_VALIDATE_CUDA_BACKWARD,
                    )
                )
        if args.json:
            payload: dict[str, Any] = {
                "valid_config": True,
                "mode": config.mode,
                "optimizer": config.optimizer,
                "families": summary["families"],
                "gene_tree_files": summary["gene_tree_files"],
                "mapped_families": summary["mapped_families"],
                "device": config.device,
                "out_dir": config.out_dir,
            }
            if args.check_preprocess:
                payload["preprocess_checked"] = True
                payload["preprocessed_families"] = summary[
                    "preprocessed_families"
                ]
                payload["preprocessed_species_nodes"] = summary[
                    "preprocessed_species_nodes"
                ]
                payload["cuda_backward_ready"] = summary["cuda_backward_ready"]
                payload["cuda_backward_ready_reason"] = summary[
                    "cuda_backward_ready_reason"
                ]
            else:
                payload["preprocess_checked"] = False
            payload["route"] = _ensure_json_ready(route_metadata_for_report)
            if explanation is not None:
                payload["explain_config"] = explanation
            print(json.dumps(_ensure_json_ready(payload), indent=2), flush=True)
        else:
            explain_text = ""
            if explanation is not None:
                explain_text = (
                    " explain_config=true "
                    f"optimizer_source={explanation['optimizer_resolution']['source']} "
                    f"default_fields={len(explanation['inferred_default_fields'])}"
                )
            print(
                "valid_config=true "
                f"mode={config.mode} optimizer={config.optimizer} "
                f"families={summary['families']} "
                f"gene_tree_files={summary['gene_tree_files']} "
                f"mapped_families={summary['mapped_families']} "
                f"{_validate_config_route_text(config, route_metadata=route_metadata_for_report)} "
                f"device={config.device} {_optional_text('out_dir', config.out_dir)}"
                f"{preprocess_text}{explain_text}",
                flush=True,
            )
        return
    if args.command == "validate-inputs":
        if args.require_cuda_backward_ready and not args.check_preprocess:
            _suggested_command_error(
                command_parser,
                "--require-cuda-backward-ready requires --check-preprocess",
                _SUGGEST_REQUIRE_CUDA_BACKWARD_READY,
            )
        config = SimpleNamespace(
            species_tree=args.species_tree.expanduser().resolve(),
            families_file=args.families_file.expanduser().resolve(),
            start=args.start,
            max_families=args.max_families,
            mode=args.mode,
            preprocess_cpu_cores=args.preprocess_cpu_cores,
        )
        try:
            summary = _summarize_alerax_family_inputs(config)
            if args.check_preprocess:
                summary = _validate_run_config_preprocess_inputs(config, summary)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        preprocess_text = ""
        if args.check_preprocess:
            if args.require_cuda_backward_ready and not summary["valid_inputs"]:
                command_parser.error(
                    _with_suggestion(
                        "input validation failed; fix input issues before checking "
                        "CUDA backward readiness",
                        "fix input issues, then rerun validate-inputs --json "
                        "before enforcing the CUDA backward gate",
                    )
                )
            preprocess_text = (
                f" preprocess_checked={summary.get('preprocess_checked', False)}"
                f" preprocessed_families={summary.get('preprocessed_families', 0)}"
                f" preprocessed_species_nodes={summary.get('preprocessed_species_nodes', 0)}"
                f" cuda_backward_ready={summary.get('cuda_backward_ready', False)}"
                f" {_optional_text('cuda_backward_ready_reason', summary.get('cuda_backward_ready_reason'))}"
            )
            if args.require_cuda_backward_ready and (
                not summary.get("cuda_backward_ready", False)
            ):
                reason = (
                    summary.get("cuda_backward_ready_reason")
                    or summary.get("preprocess_error")
                )
                command_parser.error(
                    _with_suggestion(
                        "cuda_backward_ready=false "
                        f"{_optional_text('cuda_backward_ready_reason', reason)}; "
                        "retained CUDA backward requires more than 256 postorder species nodes",
                        _SUGGEST_VALIDATE_CUDA_BACKWARD,
                    )
                )
        if args.json:
            print(json.dumps(_ensure_json_ready(summary), indent=2), flush=True)
        else:
            print(
                f"valid_inputs={str(summary['valid_inputs']).lower()} "
                f"families={summary['families']} "
                f"gene_tree_files={summary['gene_tree_files']} "
                f"mapped_families={summary['mapped_families']} "
                f"issues={len(summary['issues'])} "
                f"mode={config.mode}"
                f"{preprocess_text}",
                flush=True,
            )
        if not summary["valid_inputs"]:
            command_parser.exit(status=1)
        return
    if args.command == "sample":
        try:
            sampling_config = _sampling_config_from_args(args, args.checkpoint)
            _validate_sampling_checkpoint_path(sampling_config.checkpoint)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(_sampling_error_message(exc))
        if args.require_mode_default_optimizer or args.require_production_default_route:
            try:
                _load_checkpoint_for_route_gates(
                    command_parser,
                    sampling_config.checkpoint,
                    require_mode_default_optimizer=args.require_mode_default_optimizer,
                    require_production_default_route=args.require_production_default_route,
                )
            except _EXPECTED_WORKFLOW_ERRORS as exc:
                _exit_runtime_error(command_parser, _sampling_error_message(exc))
        try:
            result = sample(sampling_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, _sampling_error_message(exc))
        print(
            f"sampled_families={result.families_sampled} "
            f"samples={result.samples_per_family} xml={result.xml_files} "
            f"{_optional_text('out_dir', result.out_dir)}",
            flush=True,
        )
        return
    if args.command == "checkpoint-info":
        try:
            checkpoint = args.checkpoint.expanduser().resolve()
            _validate_sampling_checkpoint_path(checkpoint)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        try:
            (
                payload,
                route_for_report,
                route_source_for_report,
                checkpoint_route_evidence,
            ) = _load_checkpoint_for_route_gates(
                command_parser,
                checkpoint,
                require_mode_default_optimizer=args.require_mode_default_optimizer,
                require_production_default_route=args.require_production_default_route,
                apply_gates=False,
            )
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _exit_runtime_error(command_parser, _sampling_error_message(exc))
        status = payload.get("status")
        if not isinstance(status, dict):
            status = {}
        last_row = payload.get("last_row")
        if not isinstance(last_row, dict):
            last_row = {}
        config_data = payload.get("config")
        if not isinstance(config_data, dict):
            config_data = {}
        family_names = payload.get("family_names")
        species_names = payload.get("species_names")
        if args.json:
            print(
                json.dumps(
                    _ensure_json_ready(
                        {
                            "checkpoint": checkpoint,
                            "version": payload.get("version"),
                            "step": payload.get("step"),
                            "next_step": payload.get("next_step"),
                            "status": {
                                "status": status.get("status"),
                                "reason": status.get("reason"),
                            },
                            "mode": route_for_report.get(
                                "mode", config_data.get("mode")
                            ),
                            "optimizer": route_for_report.get(
                                "optimizer", config_data.get("optimizer")
                            ),
                            "route": route_for_report,
                            "route_metadata_source": route_source_for_report,
                            "optimizer_phase": payload.get("optimizer_phase"),
                            "last_phase": last_row.get("optimizer/phase"),
                            "families": None
                            if not isinstance(family_names, list)
                            else len(family_names),
                            "species": None
                            if not isinstance(species_names, list)
                            else len(species_names),
                            "best_step": status.get("best_step"),
                            "best_nll_bits": status.get("best_nll_bits"),
                            "last_nll_bits": last_row.get("likelihood/data_nll_bits"),
                            "last_log_likelihood_bits": last_row.get(
                                "likelihood/log_likelihood_bits"
                            ),
                            "last_grad_inf": last_row.get("grad/inf"),
                            "last_projected_grad_inf": last_row.get(
                                "grad/projected_inf"
                            ),
                            "last_final_check_iters": last_row.get(
                                "optimizer/final_check_iters"
                            ),
                            "last_final_check_status": last_row.get(
                                "optimizer/final_check_status"
                            ),
                            "last_final_check_source": last_row.get(
                                "optimizer/final_check_source"
                            ),
                            "last_final_check_reason": last_row.get(
                                "optimizer/final_check_reason"
                            ),
                            "last_final_check_fallback_clade_budget": last_row.get(
                                "optimizer/final_check_fallback_clade_budget"
                            ),
                            "last_final_check_loss_abs_delta_bits": last_row.get(
                                "optimizer/final_check_loss_abs_delta_bits"
                            ),
                            "last_final_check_grad_max_abs_delta": last_row.get(
                                "optimizer/final_check_grad_max_abs_delta"
                            ),
                            "last_final_check_grad_rel_inf_delta": last_row.get(
                                "optimizer/final_check_grad_rel_inf_delta"
                            ),
                            "last_solver_e_adjoint_failed_batches": last_row.get(
                                "solver/e_adjoint_failed_batches"
                            ),
                            "last_solver_e_adjoint_success_batches": last_row.get(
                                "solver/e_adjoint_success_batches"
                            ),
                            "last_solver_e_adjoint_rel_res_max": last_row.get(
                                "solver/e_adjoint_rel_res_max"
                            ),
                        }
                    ),
                    indent=2,
                ),
                flush=True,
            )
        else:
            print(
                _checkpoint_info_text(
                    checkpoint,
                    payload,
                    route_metadata=(route_for_report, route_source_for_report),
                    production_route_evidence=checkpoint_route_evidence,
                ),
                flush=True,
            )
        _apply_checkpoint_route_gates(
            command_parser,
            route_for_report,
            subject="checkpoint",
            require_mode_default_optimizer=args.require_mode_default_optimizer,
            require_production_default_route=args.require_production_default_route,
            evidence=checkpoint_route_evidence,
        )
        if args.require_final_check_ok:
            _exit_unless_final_check_ok(
                command_parser,
                _checkpoint_final_check_status(payload),
                subject="checkpoint",
            )
        return
    if args.command == "summary-info":
        try:
            summary = args.summary.expanduser().resolve()
            _validate_summary_path(summary)
            from gpurec.workflow.config import load_json_object

            payload = load_json_object(summary, description="summary")
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        summary_route = _summary_info_route_metadata(payload)
        production_route_evidence = (
            _production_default_route_evidence(summary_route)
            if args.require_production_default_route
            else None
        )
        audited_payload = (
            {
                **payload,
                **_route_with_production_default_evidence_fields(
                    production_route_evidence[0],
                    production_route_evidence[1],
                    production_route_evidence[2],
                ),
            }
            if production_route_evidence is not None
            else None
        )
        gate_payload = (
            audited_payload if audited_payload is not None else summary_route
        )
        if args.json:
            print(
                json.dumps(
                    _ensure_json_ready(
                        _summary_info_payload(
                            summary,
                            payload,
                            summary_route,
                            audited_payload=audited_payload,
                        )
                    ),
                    indent=2,
                ),
                flush=True,
            )
        else:
            print(
                _summary_info_text(summary, payload, audited_payload=audited_payload),
                flush=True,
            )
        if args.require_mode_default_optimizer:
            _exit_unless_mode_default_optimizer(
                command_parser,
                gate_payload,
                subject="summary",
                audited_route=audited_payload,
            )
        if args.require_production_default_route:
            _exit_unless_production_default_route(
                command_parser,
                gate_payload,
                subject="summary",
                production_route_evidence=production_route_evidence,
            )
        if args.require_converged and payload.get("status") != "converged":
            command_parser.exit(
                status=1,
                message=(
                    _with_suggestion(
                        "summary status is "
                        f"{payload.get('status')!r}; expected 'converged'",
                        "review summary status/reason and resume optimization before enforcing converged-only gates",
                    )
                    + "\n"
                ),
            )
        if args.require_final_check_ok:
            _exit_unless_final_check_ok(
                command_parser,
                payload.get("final_check_status"),
                subject="summary",
            )
        return
    if args.command == "backtrack-check":
        from gpurec import __version__ as package_version

        backtrack_payload = _doctor_backtracking_readiness(
            args.backtrack_binary,
            package_version=package_version,
        )
        if not backtrack_payload.get("ok"):
            _suggested_exit_error(
                command_parser,
                str(backtrack_payload.get("error")),
                "install or point to a compatible backtracking artifact via --backtrack-binary/GPUREC_BACKTRACK_BIN, then rerun backtrack-check",
            )
        payload = {
            "backtracking_available": True,
            "backtrack_binary": (
                str(args.backtrack_binary)
                if args.backtrack_binary is not None
                else None
            ),
            "expected_version": backtrack_payload.get("expected_version"),
            "package_version": backtrack_payload.get("package_version"),
            "version_compatible": backtrack_payload.get("version_compatible"),
        }
        if backtrack_payload.get("path") is not None:
            payload["backtrack_binary"] = backtrack_payload.get("path")
        _emit_readiness_check(
            command_parser,
            payload=payload,
            json_output=args.json,
            text_success="backtracking_available=true",
            suggestion="install or point to a compatible backtracking artifact via --backtrack-binary/GPUREC_BACKTRACK_BIN, then rerun backtrack-check",
        )
        return
    if args.command == "preprocess-check":
        from gpurec import __version__ as package_version

        preprocess_payload = _doctor_preprocessing_readiness(
            args.preprocess_native_lib,
            package_version=package_version,
        )
        if not preprocess_payload.get("ok"):
            _suggested_exit_error(
                command_parser,
                str(preprocess_payload.get("error")),
                "install or point to a compatible preprocessing native library via --preprocess-native-lib/GPUREC_PREPROCESS_NATIVE_LIB, then rerun preprocess-check",
            )
        preprocess_native_lib = preprocess_payload.get("path")
        payload = {
            "preprocessing_available": True,
            "preprocess_native_lib": str(preprocess_native_lib),
            "expected_version": preprocess_payload.get("expected_version"),
            "package_version": preprocess_payload.get("package_version"),
            "version_compatible": preprocess_payload.get("version_compatible"),
        }
        _emit_readiness_check(
            command_parser,
            payload=payload,
            json_output=args.json,
            text_success=(
                "preprocessing_available=true "
                f"{_optional_text('preprocess_native_lib', preprocess_native_lib)}"
            ),
            suggestion="install or point to a compatible preprocessing native library via --preprocess-native-lib/GPUREC_PREPROCESS_NATIVE_LIB, then rerun preprocess-check",
        )
        return

    if args.command == "doctor":
        report = _doctor_readiness_report(
            args.out_dir,
            args.preprocess_native_lib,
            args.backtrack_binary,
        )
        if args.json:
            print(json.dumps(_ensure_json_ready(report), indent=2), flush=True)
        else:
            print(_doctor_readiness_text(report), flush=True)
        if not report["ready"]:
            command_parser.exit(status=1)
        return
    if args.command == "run":
        if args.checkpoint is not None:
            command_parser.error(
                _with_suggestion(
                    "gpurec run samples from the checkpoint produced by this optimization; "
                    "use gpurec sample --checkpoint to sample an existing checkpoint, or "
                    "--resume-from to resume optimization",
                    "remove --checkpoint from run; use run for optimize+sample, sample --checkpoint for sampling-only, or optimize --resume-from for resume-only",
                )
            )
        try:
            prepared = _prepare_run_config_command(
                args,
                command_parser,
                check_preprocess=args.dry_run,
            )
            run_config = prepared.config
            summary = prepared.summary
            route_metadata = prepared.route_metadata
            _validate_run_sampling_args(args, run_config)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            command_parser.error(str(exc))
        if args.dry_run:
            print(
                _workflow_dry_run_text(
                    command="run",
                    config=run_config,
                    summary=summary,
                    route_metadata=route_metadata,
                ),
                flush=True,
            )
            return
        try:
            _ensure_backtracking_available(args.backtrack_binary)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _suggested_exit_error(
                command_parser,
                exc,
                _SUGGEST_RUN_BACKTRACK_PREP,
            )
        try:
            opt_result = _run_optimize_command(run_config, invocation_argv)
        except _EXPECTED_WORKFLOW_ERRORS as exc:
            _suggested_exit_error(
                command_parser,
                exc,
                _SUGGEST_RUN_OPTIMIZE_PREP,
            )
        if opt_result.status == "failed":
            print(
                f"{_optimization_result_text(opt_result)} "
                f"{_optional_text('out_dir', run_config.out_dir)}",
                flush=True,
            )
            command_parser.exit(
                status=1,
                message=(
                    _with_suggestion(
                        "optimization failed; refusing to sample from a failed run "
                        f"({opt_result.reason})",
                        "inspect summary/checkpoint diagnostics, fix the failure cause, then resume or rerun optimize before sampling",
                    )
                    + "\n"
                ),
            )
        if args.require_converged and opt_result.status != "converged":
            print(
                f"{_optimization_result_text(opt_result)} "
                f"{_optional_text('out_dir', run_config.out_dir)}",
                flush=True,
            )
            command_parser.exit(
                status=1,
                message=(
                    _with_suggestion(
                        "optimization status is "
                        f"{opt_result.status!r}; expected 'converged'; "
                        "refusing to sample",
                        "resume or rerun optimization until converged before invoking run-level sampling gates",
                    )
                    + "\n"
                ),
            )
        if args.require_final_check_ok and getattr(
            opt_result,
            "final_check_status",
            None,
        ) != "ok":
            print(
                f"{_optimization_result_text(opt_result)} "
                f"{_optional_text('out_dir', run_config.out_dir)}",
                flush=True,
            )
            _exit_unless_final_check_ok(
                command_parser,
                getattr(opt_result, "final_check_status", None),
                subject="optimization",
                action="refusing to sample",
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
                _with_suggestion(
                    "optimization completed but no sampling checkpoint was found "
                    f"at {checkpoint}",
                    _SUGGEST_MISSING_SAMPLE_CHECKPOINT,
                ),
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
            f"{_optimization_result_text(opt_result)} "
            f"sampled_families={sampling_result.families_sampled} "
            f"samples={sampling_result.samples_per_family} "
            f"xml={sampling_result.xml_files} "
            f"{_optional_text('out_dir', run_config.out_dir)}",
            f"{_optional_text('sample_out_dir', sampling_result.out_dir)}",
            flush=True,
        )
        return
    parser.error(f"unknown command {args.command!r}")


__all__ = [name for name in globals().keys() if not name.startswith("__")]


if __name__ == "__main__":
    main()

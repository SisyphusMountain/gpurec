from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hogenom_opt_helpers import (
    DatasetConfig,
    RegularizationConfig,
    build_model,
    evaluate_full,
    load_species_names,
    run_training,
    uniform_origination_probs,
    write_outputs,
)


# Dataset / model constants. Edit here if you want a different HOGENOM run.
HOGENOM_DIR = REPO / "tests" / "data" / "HOGENOM" / "hogenom"
FAMILIES_FILE = HOGENOM_DIR / "hogenom_families.local.txt"
ALERAX_OUTPUT = HOGENOM_DIR / "output_alerax_corrected"
INFERRED_SPECIES_TREE = ALERAX_OUTPUT / "species_trees" / "inferred_species_tree.newick"
SPECIES_TREE = (
    INFERRED_SPECIES_TREE
    if INFERRED_SPECIES_TREE.exists()
    else HOGENOM_DIR / "hogenom_S.tree"
)
PREPROCESS_CACHE = HOGENOM_DIR / "output_gpurec_ccp_reconciliation" / "preprocess_cache"
OUT_DIR = HOGENOM_DIR / "output_gpurec_specieswise_uniform_opt_max100"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

FAMILY_CHUNK_SIZE = 25
MAX_FAMILIES = None
FIXED_ITERS_E = 6
FIXED_ITERS_PI = 6
NEUMANN_TERMS = 6
USE_PRUNING = True

INITIAL_RATES = (0.05, 0.05, 0.05)
MIN_RATE = 1e-10
MAX_RATE = 100.0

# Regularizer defaults. The CLI only chooses the regularizer and its weight.
THETA_PRIOR_CENTER = math.log2(0.05)
THETA_PRIOR_STD = 0.5
BETA_PS_ALPHA = 4.0
BETA_PS_BETA = 1.0

OPTIMIZERS = (
    "lbfgs",
    "adagrad",
    "adam",
    "minibatch-adagrad",
    "adagrad-lbfgs",
    "minibatch-adagrad-lbfgs",
)
REGULARIZERS = ("none", "beta-ps", "square-theta", "gaussian-theta")
TWO_PHASE_OPTIMIZERS = ("adagrad-lbfgs", "minibatch-adagrad-lbfgs")


def parse_csv_values(text: str, cast):
    return tuple(cast(part.strip()) for part in text.split(",") if part.strip())


def default_steps(optimizer: str) -> str:
    if optimizer in TWO_PHASE_OPTIMIZERS:
        return "5,50"
    if optimizer.startswith("minibatch-"):
        return "5"
    return "50"


def default_lr(optimizer: str) -> str:
    if optimizer == "lbfgs":
        return "0.1"
    if optimizer in ("adam", "adamw", "minibatch-adam", "minibatch-adamw"):
        return "0.01"
    return "1.0,0.1" if optimizer in TWO_PHASE_OPTIMIZERS else "1.0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize specieswise rates on HOGENOM CCP with uniform origination."
    )
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="lbfgs")
    parser.add_argument("--regularization", choices=REGULARIZERS, default="none")
    parser.add_argument(
        "--steps",
        default=None,
        help="Number of steps. Use warmup,lbfgs for two-phase optimizers.",
    )
    parser.add_argument(
        "--lr",
        default=None,
        help="Learning rate. Use warmup,lbfgs for two-phase optimizers.",
    )
    parser.add_argument("--regularization-weight", type=float, default=1.0)
    args = parser.parse_args(argv)

    args.steps = parse_csv_values(args.steps or default_steps(args.optimizer), int)
    args.lr = parse_csv_values(args.lr or default_lr(args.optimizer), float)
    expected = 2 if args.optimizer in TWO_PHASE_OPTIMIZERS else 1
    if len(args.steps) != expected or len(args.lr) != expected:
        raise SystemExit(
            f"{args.optimizer} expects {expected} step value(s) and {expected} lr value(s)"
        )
    if any(step < 0 for step in args.steps):
        raise SystemExit("--steps values must be non-negative")
    if any(lr <= 0.0 for lr in args.lr):
        raise SystemExit("--lr values must be positive")
    if args.regularization_weight < 0.0:
        raise SystemExit("--regularization-weight must be non-negative")
    return args


def dataset_config() -> DatasetConfig:
    return DatasetConfig(
        species_tree=SPECIES_TREE,
        families_file=FAMILIES_FILE,
        preprocess_cache=PREPROCESS_CACHE,
        out_dir=OUT_DIR,
        device=DEVICE,
        dtype=DTYPE,
        max_families=MAX_FAMILIES,
        family_chunk_size=FAMILY_CHUNK_SIZE,
        fixed_iters_E=FIXED_ITERS_E,
        fixed_iters_Pi=FIXED_ITERS_PI,
        neumann_terms=NEUMANN_TERMS,
        use_pruning=USE_PRUNING,
        initial_rates=INITIAL_RATES,
        min_rate=MIN_RATE,
        max_rate=MAX_RATE,
    )


def regularization_config(args: argparse.Namespace) -> RegularizationConfig:
    return RegularizationConfig(
        kind=args.regularization,
        weight=args.regularization_weight,
        theta_center=THETA_PRIOR_CENTER,
        theta_std=THETA_PRIOR_STD,
        beta_ps_alpha=BETA_PS_ALPHA,
        beta_ps_beta=BETA_PS_BETA,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data = dataset_config()
    reg = regularization_config(args)

    print("species_tree", data.species_tree, flush=True)
    print("families_file", data.families_file, flush=True)
    print("device", data.device, flush=True)
    print("output_dir", data.out_dir, flush=True)
    print("optimizer", args.optimizer, "steps", args.steps, "lr", args.lr, flush=True)
    print("regularization", reg, flush=True)
    print("max_rate", data.max_rate, flush=True)
    print("origination uniform", flush=True)

    if not data.species_tree.exists():
        raise FileNotFoundError(data.species_tree)
    if not data.families_file.exists():
        raise FileNotFoundError(data.families_file)

    species_names = load_species_names(data.species_tree)
    origination_probs, origination_probs_cpu = uniform_origination_probs(
        len(species_names),
        device=data.device,
        dtype=data.dtype,
    )

    build_t0 = time.perf_counter()
    model = build_model(data, origination_probs)
    print(f"build_s={time.perf_counter() - build_t0:.3f}", flush=True)
    print("families", sum(meta.family_count for meta in model.batch_metadata), flush=True)
    print("species", model.n_species, flush=True)
    print("batches", len(model.batch_metadata), flush=True)

    warm_t0 = time.perf_counter()
    warm = evaluate_full(model, reg)
    model.theta.grad = None
    model.clear()
    print(
        f"initial_full_eval_s={time.perf_counter() - warm_t0:.3f} "
        f"data_nll_bits={warm['data_nll_bits']:.6f} "
        f"regularization_penalty_bits={warm['regularization_penalty_bits']:.6f} "
        f"objective_bits={warm['objective_bits']:.6f} "
        f"grad_inf={warm['grad_inf']:.6g}",
        flush=True,
    )

    try:
        history, distribution_history = run_training(
            model=model,
            optimizer_name=args.optimizer,
            steps=args.steps,
            lrs=args.lr,
            regularization=reg,
            dataset_config=data,
        )
    except KeyboardInterrupt:
        print("interrupted; writing partial results", flush=True)
    finally:
        write_outputs(
            model=model,
            species_names=species_names,
            origination_probs_cpu=origination_probs_cpu,
            history=locals().get("history", []),
            distribution_history=locals().get("distribution_history", []),
            dataset_config=data,
            regularization=reg,
            optimizer_name=args.optimizer,
            steps=args.steps,
            lrs=args.lr,
        )
        model.close()


if __name__ == "__main__":
    main()

"""Checkout-local global-uniform HOGENOM optimizer reproducer.

This fixed-dataset launcher hard-codes the local HOGENOM CCP input layout,
optimizes one shared/global D/T/L theta row with uniform origination, and can run
optional active-batch/full-Adagrad warmup before Torch LBFGS.  It retains
historical regularizer experiments (square, l1, huber, elastic-net, gaussian,
and beta-pS) and writes CSV/JSON outputs under
``output_gpurec_global_uniform_opt_max100``.  Migrate the unique optimizer or
reporting behavior into the supported workflow/CLI before deleting the script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpurec import GeneReconModel


HOGENOM_DIR = REPO / "tests" / "data" / "HOGENOM" / "hogenom"
FAMILIES_FILE = HOGENOM_DIR / "hogenom_families.local.txt"
ALERAX_OUTPUT = HOGENOM_DIR / "output_alerax_corrected"
INFERRED_SPECIES_TREE = ALERAX_OUTPUT / "species_trees" / "inferred_species_tree.newick"
SPECIES_TREE = (
    INFERRED_SPECIES_TREE
    if INFERRED_SPECIES_TREE.exists()
    else HOGENOM_DIR / "hogenom_S.tree"
)
OUT_DIR = HOGENOM_DIR / "output_gpurec_global_uniform_opt_max100"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

MAX_FAMILIES = None
FAMILY_CHUNK_SIZE = 25
LAZY_PREPROCESS = True
PREFETCH_BATCHES = "all"

FIXED_ITERS_E = 6
FIXED_ITERS_PI = 6
NEUMANN_TERMS = 6
USE_PRUNING = True

INITIAL_RATES = (0.05, 0.05, 0.05)
MIN_RATE = 1e-10
MAX_RATE = 100.0

TORCH_LBFGS_LR = 1.0
TORCH_LBFGS_MAX_ITER_PER_STEP = 1
TORCH_LBFGS_MAX_EVAL = 20
TORCH_LBFGS_HISTORY_SIZE = 10
TORCH_LBFGS_LINE_SEARCH_FN = "strong_wolfe"

ADAGRAD_STEPS = 0
ADAGRAD_LR = 1.0
ADAGRAD_EPS = 1e-10

MINIBATCH_EPOCHS = 0
MINIBATCH_OPTIMIZER = "adagrad"
MINIBATCH_LR = 1.0
MINIBATCH_WEIGHT_DECAY = 0.0
MINIBATCH_FULL_EVAL_EVERY = 1

WARMUP_FULL_OBJECTIVE = True
CONVERGENCE_GRAD_INF = 1e-5
CONVERGENCE_LOSS_CHANGE = 1e-6
CONVERGENCE_PATIENCE = 3
MIN_CONVERGENCE_STEPS = 2
LOG_EVERY = 1

REGULARIZATION = "none"
REGULARIZATION_WEIGHT = 1.0
REGULARIZATION_CENTER = 0.0
REGULARIZATION_STD = 0.5
REGULARIZATION_HUBER_DELTA = 1.0
REGULARIZATION_ELASTICNET_L1_RATIO = 0.5
REGULARIZATION_PS_BETA_ALPHA = 4.0
REGULARIZATION_PS_BETA_BETA = 1.0
REGULARIZATION_PS_REFERENCE_MIN = 0.2

RATE_ORDER = (("D", 0), ("T", 2), ("L", 1))
RATE_QUANTILES = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])
REGULARIZATION_CHOICES = (
    "none",
    "square",
    "l1",
    "huber",
    "elastic-net",
    "gaussian",
    "beta-ps",
)
MINIBATCH_OPTIMIZER_CHOICES = ("sgd", "adagrad", "adam", "adamw")
LN2 = math.log(2.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize one shared/global D/T/L rate vector on the HOGENOM CCP dataset with "
            "uniform origination probabilities."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Checkout-local contract: hard-coded HOGENOM inputs, one global "
            "D/T/L theta row, uniform origination, optional mini-batch/full "
            "Adagrad warmup plus LBFGS, and CSV/JSON outputs under "
            "output_gpurec_global_uniform_opt_max100. Migrate this behavior to "
            "gpurec.workflow before deleting the script."
        ),
    )
    parser.add_argument(
        "--regularization",
        choices=REGULARIZATION_CHOICES,
        default=REGULARIZATION,
        help=(
            "Penalty applied to theta log2 D/T/L parameters or derived p_S. "
            "Default keeps the run unregularized."
        ),
    )
    parser.add_argument(
        "--regularization-weight",
        type=float,
        default=REGULARIZATION_WEIGHT,
        help=(
            "Multiplier for the selected penalty. For square this gives "
            "weight * sum((theta - center)^2)."
        ),
    )
    parser.add_argument(
        "--regularization-center",
        type=float,
        default=REGULARIZATION_CENTER,
        help="Center used by square, l1, huber, elastic-net, and gaussian penalties.",
    )
    parser.add_argument(
        "--regularization-std",
        type=float,
        default=REGULARIZATION_STD,
        help=(
            "Std used by gaussian penalty: "
            "0.5 * weight * sum(((theta - center) / std)^2)."
        ),
    )
    parser.add_argument(
        "--regularization-huber-delta",
        type=float,
        default=REGULARIZATION_HUBER_DELTA,
        help="Huber transition point used by --regularization huber.",
    )
    parser.add_argument(
        "--regularization-elasticnet-l1-ratio",
        type=float,
        default=REGULARIZATION_ELASTICNET_L1_RATIO,
        help="L1 ratio in [0, 1] used by --regularization elastic-net.",
    )
    parser.add_argument(
        "--regularization-ps-beta-alpha",
        type=float,
        default=REGULARIZATION_PS_BETA_ALPHA,
        help=(
            "Alpha parameter for --regularization beta-ps. The default "
            "Beta(4, 1) has 99.84%% prior mass above p_S=0.2."
        ),
    )
    parser.add_argument(
        "--regularization-ps-beta-beta",
        type=float,
        default=REGULARIZATION_PS_BETA_BETA,
        help="Beta parameter for --regularization beta-ps.",
    )
    parser.add_argument(
        "--regularization-ps-reference-min",
        type=float,
        default=REGULARIZATION_PS_REFERENCE_MIN,
        help="Reference p_S lower bound recorded in the run config.",
    )
    parser.add_argument(
        "--adagrad-steps",
        type=int,
        default=ADAGRAD_STEPS,
        help="Number of full-objective Adagrad warmup steps before starting LBFGS.",
    )
    parser.add_argument(
        "--adagrad-lr",
        type=float,
        default=ADAGRAD_LR,
        help="Learning rate used for the optional Adagrad warmup.",
    )
    parser.add_argument(
        "--adagrad-eps",
        type=float,
        default=ADAGRAD_EPS,
        help="Numerical epsilon passed to torch.optim.Adagrad.",
    )
    parser.add_argument(
        "--minibatch-epochs",
        type=int,
        default=MINIBATCH_EPOCHS,
        help=(
            "Number of active-batch stochastic optimization epochs before "
            "full-objective Adagrad/LBFGS."
        ),
    )
    parser.add_argument(
        "--minibatch-optimizer",
        choices=MINIBATCH_OPTIMIZER_CHOICES,
        default=MINIBATCH_OPTIMIZER,
        help="Optimizer used during the optional active-batch phase.",
    )
    parser.add_argument(
        "--minibatch-lr",
        type=float,
        default=MINIBATCH_LR,
        help="Learning rate for the optional active-batch optimizer.",
    )
    parser.add_argument(
        "--minibatch-weight-decay",
        type=float,
        default=MINIBATCH_WEIGHT_DECAY,
        help="Weight decay passed to the optional active-batch optimizer.",
    )
    parser.add_argument(
        "--minibatch-full-eval-every",
        type=int,
        default=MINIBATCH_FULL_EVAL_EVERY,
        help=(
            "Evaluate exact full objective every N mini-batch epochs. Use 0 "
            "to disable exact epoch-end evaluation."
        ),
    )
    parser.add_argument(
        "--lbfgs-lr",
        type=float,
        default=TORCH_LBFGS_LR,
        help="Learning rate passed to torch.optim.LBFGS.",
    )
    parser.add_argument(
        "--lbfgs-max-iter-per-step",
        type=int,
        default=TORCH_LBFGS_MAX_ITER_PER_STEP,
        help=(
            "Internal LBFGS iterations per outer optimizer.step call. The script "
            "logs once per outer step."
        ),
    )
    parser.add_argument(
        "--lbfgs-max-eval",
        type=int,
        default=TORCH_LBFGS_MAX_EVAL,
        help="Maximum closure evaluations per LBFGS optimizer.step call.",
    )
    parser.add_argument(
        "--lbfgs-history-size",
        type=int,
        default=TORCH_LBFGS_HISTORY_SIZE,
        help="History size passed to torch.optim.LBFGS.",
    )
    parser.add_argument(
        "--lbfgs-line-search-fn",
        choices=("strong_wolfe", "none"),
        default=TORCH_LBFGS_LINE_SEARCH_FN,
        help="Line search passed to torch.optim.LBFGS.",
    )
    parser.add_argument(
        "--skip-lbfgs",
        action="store_true",
        help="Stop after the optional mini-batch/full-Adagrad phases.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.regularization_weight < 0.0:
        raise ValueError("--regularization-weight must be non-negative")
    if args.regularization_std <= 0.0:
        raise ValueError("--regularization-std must be positive")
    if args.regularization_huber_delta <= 0.0:
        raise ValueError("--regularization-huber-delta must be positive")
    if not 0.0 <= args.regularization_elasticnet_l1_ratio <= 1.0:
        raise ValueError("--regularization-elasticnet-l1-ratio must be in [0, 1]")
    if args.regularization_ps_beta_alpha <= 0.0:
        raise ValueError("--regularization-ps-beta-alpha must be positive")
    if args.regularization_ps_beta_beta <= 0.0:
        raise ValueError("--regularization-ps-beta-beta must be positive")
    if not 0.0 < args.regularization_ps_reference_min < 1.0:
        raise ValueError("--regularization-ps-reference-min must be in (0, 1)")
    if args.adagrad_steps < 0:
        raise ValueError("--adagrad-steps must be non-negative")
    if args.adagrad_lr <= 0.0:
        raise ValueError("--adagrad-lr must be positive")
    if args.adagrad_eps <= 0.0:
        raise ValueError("--adagrad-eps must be positive")
    if args.minibatch_epochs < 0:
        raise ValueError("--minibatch-epochs must be non-negative")
    if args.minibatch_lr <= 0.0:
        raise ValueError("--minibatch-lr must be positive")
    if args.minibatch_weight_decay < 0.0:
        raise ValueError("--minibatch-weight-decay must be non-negative")
    if args.minibatch_full_eval_every < 0:
        raise ValueError("--minibatch-full-eval-every must be non-negative")
    if args.lbfgs_lr <= 0.0:
        raise ValueError("--lbfgs-lr must be positive")
    if args.lbfgs_max_iter_per_step <= 0:
        raise ValueError("--lbfgs-max-iter-per-step must be positive")
    if args.lbfgs_max_eval <= 0:
        raise ValueError("--lbfgs-max-eval must be positive")
    if args.lbfgs_history_size <= 0:
        raise ValueError("--lbfgs-history-size must be positive")


def regularization_config(args: argparse.Namespace) -> dict[str, float | str]:
    return {
        "regularization": args.regularization,
        "regularization_weight": args.regularization_weight,
        "regularization_center": args.regularization_center,
        "regularization_std": args.regularization_std,
        "regularization_huber_delta": args.regularization_huber_delta,
        "regularization_elasticnet_l1_ratio": (
            args.regularization_elasticnet_l1_ratio
        ),
        "regularization_ps_beta_alpha": args.regularization_ps_beta_alpha,
        "regularization_ps_beta_beta": args.regularization_ps_beta_beta,
        "regularization_ps_reference_min": args.regularization_ps_reference_min,
    }


def optimization_config(args: argparse.Namespace) -> dict[str, float | int | str | bool]:
    return {
        "minibatch_epochs": args.minibatch_epochs,
        "minibatch_optimizer": args.minibatch_optimizer,
        "minibatch_lr": args.minibatch_lr,
        "minibatch_weight_decay": args.minibatch_weight_decay,
        "minibatch_full_eval_every": args.minibatch_full_eval_every,
        "adagrad_steps": args.adagrad_steps,
        "adagrad_lr": args.adagrad_lr,
        "adagrad_eps": args.adagrad_eps,
        "lbfgs_lr": args.lbfgs_lr,
        "lbfgs_max_iter_per_step": args.lbfgs_max_iter_per_step,
        "lbfgs_max_eval": args.lbfgs_max_eval,
        "lbfgs_history_size": args.lbfgs_history_size,
        "lbfgs_line_search_fn": args.lbfgs_line_search_fn,
        "skip_lbfgs": bool(args.skip_lbfgs),
    }


def lbfgs_line_search_fn(args: argparse.Namespace) -> str | None:
    if args.lbfgs_line_search_fn == "none":
        return None
    return args.lbfgs_line_search_fn


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_uniform_origination_probs(n_species: int) -> tuple[torch.Tensor, torch.Tensor]:
    probs_cpu = torch.full((n_species,), 1.0 / n_species, dtype=DTYPE)
    probs = probs_cpu.to(device=DEVICE, dtype=DTYPE)
    return probs, probs_cpu


def theta_event_log_probs(theta: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.log_softmax(theta_event_natural_logits(theta), dim=-1)


def theta_event_natural_logits(theta: torch.Tensor) -> torch.Tensor:
    zeros = theta.new_zeros((*theta.shape[:-1], 1))
    return torch.cat((zeros, theta), dim=-1) * LN2


def regularization_penalty(
    theta: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    kind = args.regularization
    weight = float(args.regularization_weight)
    if kind == "none" or weight == 0.0:
        return theta.new_zeros(())

    shifted = theta - theta.new_tensor(float(args.regularization_center))
    if kind == "square":
        penalty = shifted.square().sum()
    elif kind == "l1":
        penalty = shifted.abs().sum()
    elif kind == "huber":
        delta = theta.new_tensor(float(args.regularization_huber_delta))
        abs_shifted = shifted.abs()
        quadratic = torch.minimum(abs_shifted, delta)
        linear = abs_shifted - quadratic
        penalty = (0.5 * quadratic.square() + delta * linear).sum()
    elif kind == "elastic-net":
        l1_ratio = float(args.regularization_elasticnet_l1_ratio)
        penalty = (
            (1.0 - l1_ratio) * shifted.square().sum()
            + l1_ratio * shifted.abs().sum()
        )
    elif kind == "gaussian":
        std = theta.new_tensor(float(args.regularization_std))
        penalty = 0.5 * (shifted / std).square().sum()
    elif kind == "beta-ps":
        natural_logits = theta_event_natural_logits(theta)
        log_probs = torch.nn.functional.log_softmax(natural_logits, dim=-1)
        log_pS = log_probs[..., 0]
        log_one_minus_pS = torch.logsumexp(
            natural_logits[..., 1:],
            dim=-1,
        ) - torch.logsumexp(natural_logits, dim=-1)
        alpha = float(args.regularization_ps_beta_alpha)
        beta = float(args.regularization_ps_beta_beta)
        penalty = -(
            (alpha - 1.0) * log_pS
            + (beta - 1.0) * log_one_minus_pS
        ).sum() / LN2
    else:
        raise ValueError(f"unknown regularization penalty: {kind}")
    return penalty * weight


def objective_terms(
    model: GeneReconModel,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data_loss = model.full_loss()
    penalty = regularization_penalty(model.theta, args)
    objective = data_loss + penalty
    return data_loss, penalty, objective


def active_batch_objective_terms(
    model: GeneReconModel,
    args: argparse.Namespace,
    *,
    penalty_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data_loss = model()
    penalty = regularization_penalty(model.theta, args) * penalty_scale
    objective = data_loss + penalty
    return data_loss, penalty, objective


def evaluate_loss_and_grad(
    model: GeneReconModel,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if optimizer is None:
        model.theta.grad = None
    else:
        optimizer.zero_grad(set_to_none=True)
    data_loss, penalty, objective = objective_terms(model, args)
    objective.backward()
    synchronize()
    if model.theta.grad is None:
        raise RuntimeError("missing model theta gradient")
    return (
        data_loss.detach(),
        penalty.detach(),
        objective.detach(),
        model.theta.grad.detach(),
    )


def make_minibatch_optimizer(
    model: GeneReconModel,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    params = [model.theta]
    kwargs = {
        "lr": args.minibatch_lr,
        "weight_decay": args.minibatch_weight_decay,
    }
    if args.minibatch_optimizer == "sgd":
        return torch.optim.SGD(params, **kwargs)
    if args.minibatch_optimizer == "adagrad":
        return torch.optim.Adagrad(params, eps=args.adagrad_eps, **kwargs)
    if args.minibatch_optimizer == "adam":
        return torch.optim.Adam(params, **kwargs)
    if args.minibatch_optimizer == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    raise ValueError(f"unknown mini-batch optimizer: {args.minibatch_optimizer}")


def activate_batch(model: GeneReconModel, batch_idx: int) -> None:
    model.select_batch(batch_idx)


def rate_distribution_rows(
    theta: torch.Tensor,
    iteration: int,
) -> list[dict[str, float | int | str]]:
    theta_rows = theta.detach()
    if theta_rows.ndim == 1:
        theta_rows = theta_rows.unsqueeze(0)
    rates = torch.exp2(theta_rows).cpu()
    rows: list[dict[str, float | int | str]] = []
    for rate_name, column in RATE_ORDER:
        values = rates[:, column]
        quantiles = torch.quantile(values, RATE_QUANTILES.to(dtype=values.dtype))
        rows.append(
            {
                "iteration": iteration,
                "rate": rate_name,
                "min": float(quantiles[0]),
                "p05": float(quantiles[1]),
                "median": float(quantiles[2]),
                "p95": float(quantiles[3]),
                "max": float(quantiles[4]),
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
            }
        )
    pS_values = torch.exp(theta_event_log_probs(theta_rows)[..., 0]).cpu()
    pS_quantiles = torch.quantile(
        pS_values,
        RATE_QUANTILES.to(dtype=pS_values.dtype),
    )
    rows.append(
        {
            "iteration": iteration,
            "rate": "pS",
            "min": float(pS_quantiles[0]),
            "p05": float(pS_quantiles[1]),
            "median": float(pS_quantiles[2]),
            "p95": float(pS_quantiles[3]),
            "max": float(pS_quantiles[4]),
            "mean": float(pS_values.mean()),
            "std": float(pS_values.std(unbiased=False)),
        }
    )
    return rows


def format_rate_distribution(rows: list[dict[str, float | int | str]]) -> str:
    parts = []
    for row in rows:
        parts.append(
            f"{row['rate']}[min={row['min']:.3g} p05={row['p05']:.3g} "
            f"med={row['median']:.3g} p95={row['p95']:.3g} "
            f"max={row['max']:.3g}]"
        )
    return " ".join(parts)


def run_minibatch_phase(
    *,
    model: GeneReconModel,
    args: argparse.Namespace,
    history: list[dict[str, float | int | bool | str]],
    rate_history: list[dict[str, float | int | str]],
    start_step: int,
) -> int:
    if args.minibatch_epochs <= 0:
        return start_step

    optimizer = make_minibatch_optimizer(model, args)
    num_batches = len(model.batch_metadata)
    penalty_scale = 1.0 / float(num_batches)
    step = start_step
    phase_iteration = 0

    print(
        f"minibatch_start epochs={args.minibatch_epochs} "
        f"optimizer={args.minibatch_optimizer} lr={args.minibatch_lr} "
        f"batches={num_batches} penalty_scale={penalty_scale:.6g}",
        flush=True,
    )

    for epoch in range(args.minibatch_epochs):
        epoch_started = time.perf_counter()
        epoch_data_nll_bits = 0.0
        epoch_penalty_bits = 0.0
        epoch_objective_bits = 0.0

        for batch_idx in range(num_batches):
            if batch_idx == 0:
                activate_batch(model, 0)
            else:
                model.next()
            meta = model.current_batch_metadata
            iteration_started = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            data_loss, penalty, objective = active_batch_objective_terms(
                model,
                args,
                penalty_scale=penalty_scale,
            )
            objective.backward()
            synchronize()
            if model.theta.grad is None:
                raise RuntimeError("missing model theta gradient")

            data_nll_bits = float(data_loss.detach().cpu())
            regularization_penalty_bits = float(penalty.detach().cpu())
            objective_bits = float(objective.detach().cpu())
            grad = model.theta.grad.detach()
            grad_inf = float(grad.abs().amax().cpu())
            grad_norm = float(torch.linalg.vector_norm(grad).cpu())

            theta_before = model.theta.detach().clone()
            optimizer_step_started = time.perf_counter()
            optimizer.step()
            synchronize()
            optimizer_step_s = time.perf_counter() - optimizer_step_started
            model.clamp_theta_(min_rate=MIN_RATE, max_rate=MAX_RATE)
            theta_step_inf = float(
                (model.theta.detach() - theta_before).abs().amax().cpu()
            )
            accepted_step = theta_step_inf > 0.0
            iteration_s = time.perf_counter() - iteration_started
            log_likelihood_bits = -data_nll_bits

            epoch_data_nll_bits += data_nll_bits
            epoch_penalty_bits += regularization_penalty_bits
            epoch_objective_bits += objective_bits

            row = {
                "phase": "minibatch",
                "iteration": step,
                "phase_iteration": phase_iteration,
                "epoch": epoch,
                "batch_index": batch_idx,
                "batch_family_count": int(meta.family_count),
                "batch_clade_count": int(meta.clade_count),
                "data_nll_bits": data_nll_bits,
                "regularization_penalty_bits": regularization_penalty_bits,
                "objective_bits": objective_bits,
                "log_likelihood_bits": log_likelihood_bits,
                "delta_nll_bits": float("nan"),
                "delta_objective_bits": float("nan"),
                "abs_delta_nll_bits": float("nan"),
                "abs_delta_objective_bits": float("nan"),
                "grad_inf": grad_inf,
                "grad_norm": grad_norm,
                "iteration_s": iteration_s,
                "optimizer_step_s": optimizer_step_s,
                "minibatch_step_s": optimizer_step_s,
                "adagrad_step_s": float("nan"),
                "lbfgs_step_s": float("nan"),
                "closure_time_s": iteration_s,
                "closure_mean_s": iteration_s,
                "final_eval_s": 0.0,
                "theta_step_inf": theta_step_inf,
                "accepted_step": accepted_step,
                "closure_evals": 1,
                "lbfgs_total_iter": 0,
                "lbfgs_total_func_evals": 0,
            }
            history.append(row)

            rate_rows = rate_distribution_rows(model.theta, step)
            rate_history.extend(rate_rows)

            if step % LOG_EVERY == 0:
                print(
                    f"phase=minibatch iter={step:04d} "
                    f"epoch={epoch:04d} batch={batch_idx:04d}/{num_batches} "
                    f"families={meta.family_count} clades={meta.clade_count} "
                    f"data_nll_bits={data_nll_bits:.6f} "
                    f"regularization_penalty_bits={regularization_penalty_bits:.6f} "
                    f"objective_bits={objective_bits:.6f} "
                    f"loglik_bits={log_likelihood_bits:.6f} "
                    f"grad_inf={grad_inf:.6g} grad_norm={grad_norm:.6g} "
                    f"iteration_s={iteration_s:.3f} "
                    f"minibatch_step_s={optimizer_step_s:.3f} "
                    f"theta_step_inf={theta_step_inf:.3g} "
                    f"accepted_step={accepted_step}",
                    flush=True,
                )
                print("  distributions " + format_rate_distribution(rate_rows), flush=True)

            model.clear()
            step += 1
            phase_iteration += 1

        epoch_s = time.perf_counter() - epoch_started
        print(
            f"minibatch_epoch={epoch:04d} epoch_s={epoch_s:.3f} "
            f"sum_batch_data_nll_bits={epoch_data_nll_bits:.6f} "
            f"sum_batch_regularization_penalty_bits={epoch_penalty_bits:.6f} "
            f"sum_batch_objective_bits={epoch_objective_bits:.6f}",
            flush=True,
        )

        if (
            args.minibatch_full_eval_every > 0
            and (epoch + 1) % args.minibatch_full_eval_every == 0
        ):
            full_eval_started = time.perf_counter()
            full_data_loss, full_penalty, full_objective, full_grad = (
                evaluate_loss_and_grad(model, args)
            )
            full_eval_s = time.perf_counter() - full_eval_started
            print(
                f"minibatch_epoch_full_eval={epoch:04d} "
                f"full_eval_s={full_eval_s:.3f} "
                f"data_nll_bits={float(full_data_loss.detach().cpu()):.6f} "
                f"regularization_penalty_bits={float(full_penalty.detach().cpu()):.6f} "
                f"objective_bits={float(full_objective.detach().cpu()):.6f} "
                f"grad_inf={float(full_grad.abs().amax().cpu()):.6g}",
                flush=True,
            )
            model.theta.grad = None
            model.clear()

    activate_batch(model, 0)
    return step


def write_outputs(
    *,
    model: GeneReconModel,
    species_names: list[str],
    origination_probs_cpu: torch.Tensor,
    history: list[dict[str, float | int | bool | str]],
    rate_history: list[dict[str, float | int | str]],
    convergence_reason: str | None,
    args: argparse.Namespace,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_path = OUT_DIR / "global_optimization_history.csv"
    history_df.to_csv(history_path, index=False)

    rate_history_df = pd.DataFrame(rate_history)
    rate_history_path = OUT_DIR / "global_rate_distribution_history.csv"
    rate_history_df.to_csv(rate_history_path, index=False)

    with torch.no_grad():
        theta = model.theta.detach().cpu().numpy()
        rates = torch.exp2(model.theta.detach()).cpu().numpy()
        event_probs = torch.exp(theta_event_log_probs(model.theta.detach())).cpu().numpy()

    rates_df = pd.DataFrame(
        [
            {
                "parameter_scope": "global",
                "duplication_theta_log2": theta[0],
                "loss_theta_log2": theta[1],
                "transfer_theta_log2": theta[2],
                "duplication_rate": rates[0],
                "loss_rate": rates[1],
                "transfer_rate": rates[2],
                "p_S": event_probs[0],
                "p_D": event_probs[1],
                "p_L": event_probs[2],
                "p_T": event_probs[3],
            }
        ]
    )
    rates_path = OUT_DIR / "optimized_global_rates.csv"
    rates_df.to_csv(rates_path, index=False)

    origination_df = pd.DataFrame(
        {
            "species_index": np.arange(len(species_names)),
            "species": species_names,
            "origination_prob": origination_probs_cpu.numpy(),
        }
    )
    origination_path = OUT_DIR / "uniform_origination_distribution.csv"
    origination_df.to_csv(origination_path, index=False)

    config = {
        "species_tree": str(SPECIES_TREE),
        "families_file": str(FAMILIES_FILE),
        "max_families": MAX_FAMILIES,
        "family_chunk_size": FAMILY_CHUNK_SIZE,
        "fixed_iters_E": FIXED_ITERS_E,
        "fixed_iters_Pi": FIXED_ITERS_PI,
        "neumann_terms": NEUMANN_TERMS,
        "use_pruning": USE_PRUNING,
        "initial_rates": INITIAL_RATES,
        "min_rate": MIN_RATE,
        "max_rate": MAX_RATE,
        "parameter_mode": "global",
        "origination_mode": "uniform",
        "dtl_prior": None,
        **regularization_config(args),
        "optimizer": "+".join(
            [
                *(["active-batch torch.optim." + args.minibatch_optimizer]
                  if args.minibatch_epochs > 0 else []),
                *(["torch.optim.Adagrad"] if args.adagrad_steps > 0 else []),
                *(["torch.optim.LBFGS"] if not args.skip_lbfgs else []),
            ]
        )
        or "none",
        **optimization_config(args),
        "torch_lbfgs_lr": args.lbfgs_lr,
        "torch_lbfgs_max_iter_per_step": args.lbfgs_max_iter_per_step,
        "torch_lbfgs_max_eval": args.lbfgs_max_eval,
        "torch_lbfgs_history_size": args.lbfgs_history_size,
        "torch_lbfgs_line_search_fn": args.lbfgs_line_search_fn,
        "warmup_full_objective": WARMUP_FULL_OBJECTIVE,
        "convergence_grad_inf": CONVERGENCE_GRAD_INF,
        "convergence_loss_change": CONVERGENCE_LOSS_CHANGE,
        "convergence_patience": CONVERGENCE_PATIENCE,
        "min_convergence_steps": MIN_CONVERGENCE_STEPS,
        "convergence_reason": convergence_reason,
        "final_data_nll_bits": (
            None if history_df.empty else float(history_df.iloc[-1]["data_nll_bits"])
        ),
        "final_regularization_penalty_bits": (
            None
            if history_df.empty
            else float(history_df.iloc[-1]["regularization_penalty_bits"])
        ),
        "final_objective_bits": (
            None if history_df.empty else float(history_df.iloc[-1]["objective_bits"])
        ),
        "final_log_likelihood_bits": (
            None
            if history_df.empty
            else float(history_df.iloc[-1]["log_likelihood_bits"])
        ),
        "final_grad_inf": (
            None if history_df.empty else float(history_df.iloc[-1]["grad_inf"])
        ),
    }
    config_path = OUT_DIR / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    print("wrote", history_path)
    print("wrote", rate_history_path)
    print("wrote", rates_path)
    print("wrote", origination_path)
    print("wrote", config_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SPECIES_TREE.exists():
        raise FileNotFoundError(SPECIES_TREE)
    if not FAMILIES_FILE.exists():
        raise FileNotFoundError(FAMILIES_FILE)

    print("species_tree", SPECIES_TREE, flush=True)
    print("families_file", FAMILIES_FILE, flush=True)
    print("device", DEVICE, flush=True)
    print("output_dir", OUT_DIR, flush=True)
    print("parameter_mode", "global", flush=True)
    print("max_rate", MAX_RATE, flush=True)
    print("origination", "uniform", flush=True)
    print("dtl_prior", None, flush=True)
    print("regularization", regularization_config(args), flush=True)
    print("optimization", optimization_config(args), flush=True)

    build_t0 = time.perf_counter()
    model = GeneReconModel.from_alerax_families(
        str(SPECIES_TREE),
        FAMILIES_FILE,
        mode="global",
        start=0,
        max_families=MAX_FAMILIES,
        device=DEVICE,
        dtype=DTYPE,
        theta_init_rates=INITIAL_RATES,
        fixed_iters_E=FIXED_ITERS_E,
        fixed_iters_Pi=FIXED_ITERS_PI,
        neumann_terms=NEUMANN_TERMS,
        use_pruning=USE_PRUNING,
        family_chunk_size=FAMILY_CHUNK_SIZE,
        lazy_preprocess=LAZY_PREPROCESS,
        prefetch_batches=PREFETCH_BATCHES,
    )
    species_names = model.species_names
    _, origination_probs_cpu = make_uniform_origination_probs(model.n_species)
    synchronize()

    print(f"build_s={time.perf_counter() - build_t0:.3f}", flush=True)
    print("families", sum(meta.family_count for meta in model.batch_metadata), flush=True)
    print("species", model.n_species, flush=True)
    print("theta_shape", tuple(model.theta.shape), flush=True)
    print("batches", len(model.batch_metadata), flush=True)

    if WARMUP_FULL_OBJECTIVE:
        warm_t0 = time.perf_counter()
        warm_data_loss, warm_penalty, warm_objective, warm_grad = evaluate_loss_and_grad(
            model,
            args,
        )
        warm_nll_bits = float(warm_data_loss.detach().cpu())
        warm_penalty_bits = float(warm_penalty.detach().cpu())
        warm_objective_bits = float(warm_objective.detach().cpu())
        warm_grad_inf = float(warm_grad.abs().amax().cpu())
        print(
            f"warmup_full_objective_s={time.perf_counter() - warm_t0:.3f} "
            f"data_nll_bits={warm_nll_bits:.6f} "
            f"regularization_penalty_bits={warm_penalty_bits:.6f} "
            f"objective_bits={warm_objective_bits:.6f} "
            f"loglik_bits={-warm_nll_bits:.6f} grad_inf={warm_grad_inf:.6g}",
            flush=True,
        )
        model.theta.grad = None
        model.clear()

    history: list[dict[str, float | int | bool | str]] = []
    rate_history: list[dict[str, float | int | str]] = []
    previous_nll: float | None = None
    previous_objective: float | None = None
    stable_loss_steps = 0
    convergence_reason: str | None = None
    step = 0

    try:
        step = run_minibatch_phase(
            model=model,
            args=args,
            history=history,
            rate_history=rate_history,
            start_step=step,
        )
        previous_nll = None
        previous_objective = None
        stable_loss_steps = 0

        if args.adagrad_steps > 0:
            adagrad_optimizer = torch.optim.Adagrad(
                [model.theta],
                lr=args.adagrad_lr,
                eps=args.adagrad_eps,
            )
            initial_eval_started = time.perf_counter()
            current_data_loss, current_penalty, current_objective, current_grad = (
                evaluate_loss_and_grad(model, args, adagrad_optimizer)
            )
            initial_eval_s = time.perf_counter() - initial_eval_started
            previous_nll = float(current_data_loss.detach().cpu())
            previous_objective = float(current_objective.detach().cpu())
            print(
                f"adagrad_initial_eval_s={initial_eval_s:.3f} "
                f"data_nll_bits={previous_nll:.6f} "
                f"regularization_penalty_bits={float(current_penalty.detach().cpu()):.6f} "
                f"objective_bits={previous_objective:.6f} "
                f"grad_inf={float(current_grad.abs().amax().cpu()):.6g}",
                flush=True,
            )

            for adagrad_step in range(args.adagrad_steps):
                iteration_started = time.perf_counter()
                theta_before = model.theta.detach().clone()
                adagrad_started = time.perf_counter()
                adagrad_optimizer.step()
                synchronize()
                adagrad_step_s = time.perf_counter() - adagrad_started
                model.clamp_theta_(min_rate=MIN_RATE, max_rate=MAX_RATE)
                theta_step_inf = float(
                    (model.theta.detach() - theta_before).abs().amax().cpu()
                )
                accepted_step = theta_step_inf > 0.0

                final_eval_started = time.perf_counter()
                final_data_loss, final_penalty, final_objective, final_grad = (
                    evaluate_loss_and_grad(model, args, adagrad_optimizer)
                )
                final_eval_s = time.perf_counter() - final_eval_started
                data_nll_bits = float(final_data_loss.detach().cpu())
                regularization_penalty_bits = float(final_penalty.detach().cpu())
                objective_bits = float(final_objective.detach().cpu())
                log_likelihood_bits = -data_nll_bits
                grad_inf = float(final_grad.abs().amax().cpu())
                grad_norm = float(torch.linalg.vector_norm(final_grad).cpu())
                delta_nll_bits = previous_nll - data_nll_bits
                delta_objective_bits = previous_objective - objective_bits
                abs_delta_nll_bits = abs(delta_nll_bits)
                abs_delta_objective_bits = abs(delta_objective_bits)

                iteration_s = time.perf_counter() - iteration_started
                row = {
                    "phase": "adagrad",
                    "iteration": step,
                    "phase_iteration": adagrad_step,
                    "data_nll_bits": data_nll_bits,
                    "regularization_penalty_bits": regularization_penalty_bits,
                    "objective_bits": objective_bits,
                    "log_likelihood_bits": log_likelihood_bits,
                    "delta_nll_bits": delta_nll_bits,
                    "delta_objective_bits": delta_objective_bits,
                    "abs_delta_nll_bits": abs_delta_nll_bits,
                    "abs_delta_objective_bits": abs_delta_objective_bits,
                    "grad_inf": grad_inf,
                    "grad_norm": grad_norm,
                    "iteration_s": iteration_s,
                    "optimizer_step_s": adagrad_step_s,
                    "adagrad_step_s": adagrad_step_s,
                    "lbfgs_step_s": float("nan"),
                    "closure_time_s": final_eval_s,
                    "closure_mean_s": final_eval_s,
                    "final_eval_s": final_eval_s,
                    "theta_step_inf": theta_step_inf,
                    "accepted_step": accepted_step,
                    "closure_evals": 1,
                    "lbfgs_total_iter": 0,
                    "lbfgs_total_func_evals": 0,
                }
                history.append(row)

                rate_rows = rate_distribution_rows(model.theta, step)
                rate_history.extend(rate_rows)

                if step % LOG_EVERY == 0:
                    print(
                        f"phase=adagrad iter={step:04d} "
                        f"phase_iter={adagrad_step:04d} "
                        f"data_nll_bits={data_nll_bits:.6f} "
                        f"regularization_penalty_bits={regularization_penalty_bits:.6f} "
                        f"objective_bits={objective_bits:.6f} "
                        f"loglik_bits={log_likelihood_bits:.6f} "
                        f"delta_nll_bits={delta_nll_bits:.6g} "
                        f"delta_objective_bits={delta_objective_bits:.6g} "
                        f"grad_inf={grad_inf:.6g} grad_norm={grad_norm:.6g} "
                        f"iteration_s={iteration_s:.3f} "
                        f"adagrad_step_s={adagrad_step_s:.3f} "
                        f"theta_step_inf={theta_step_inf:.3g} "
                        f"accepted_step={accepted_step} final_eval_s={final_eval_s:.3f}",
                        flush=True,
                    )
                    print("  distributions " + format_rate_distribution(rate_rows), flush=True)

                previous_nll = data_nll_bits
                previous_objective = objective_bits
                model.clear()
                step += 1

            stable_loss_steps = 0
            previous_nll = None
            previous_objective = None

        if args.skip_lbfgs:
            convergence_reason = "lbfgs_skipped"
        else:
            optimizer = torch.optim.LBFGS(
                [model.theta],
                lr=args.lbfgs_lr,
                max_iter=args.lbfgs_max_iter_per_step,
                max_eval=args.lbfgs_max_eval,
                tolerance_grad=CONVERGENCE_GRAD_INF,
                tolerance_change=CONVERGENCE_LOSS_CHANGE,
                history_size=args.lbfgs_history_size,
                line_search_fn=lbfgs_line_search_fn(args),
            )

            lbfgs_phase_step = 0
            while True:
                iteration_started = time.perf_counter()
                closure_metrics = {"count": 0, "time_s": 0.0}

                def closure() -> torch.Tensor:
                    closure_started = time.perf_counter()
                    closure_metrics["count"] += 1
                    optimizer.zero_grad(set_to_none=True)
                    _, _, objective = objective_terms(model, args)
                    objective.backward()
                    synchronize()
                    closure_metrics["time_s"] += time.perf_counter() - closure_started
                    return objective

                theta_before = model.theta.detach().clone()
                lbfgs_started = time.perf_counter()
                optimizer.step(closure)
                lbfgs_step_s = time.perf_counter() - lbfgs_started
                model.clamp_theta_(min_rate=MIN_RATE, max_rate=MAX_RATE)
                theta_step_inf = float(
                    (model.theta.detach() - theta_before).abs().amax().cpu()
                )
                accepted_step = theta_step_inf > 0.0

                final_eval_started = time.perf_counter()
                final_data_loss, final_penalty, final_objective, final_grad = (
                    evaluate_loss_and_grad(model, args, optimizer)
                )
                final_eval_s = time.perf_counter() - final_eval_started
                data_nll_bits = float(final_data_loss.detach().cpu())
                regularization_penalty_bits = float(final_penalty.detach().cpu())
                objective_bits = float(final_objective.detach().cpu())
                log_likelihood_bits = -data_nll_bits
                grad_inf = float(final_grad.abs().amax().cpu())
                grad_norm = float(torch.linalg.vector_norm(final_grad).cpu())
                delta_nll_bits = (
                    None if previous_nll is None else previous_nll - data_nll_bits
                )
                delta_objective_bits = (
                    None
                    if previous_objective is None
                    else previous_objective - objective_bits
                )
                abs_delta_nll_bits = (
                    None if delta_nll_bits is None else abs(delta_nll_bits)
                )
                abs_delta_objective_bits = (
                    None
                    if delta_objective_bits is None
                    else abs(delta_objective_bits)
                )
                if (
                    abs_delta_objective_bits is not None
                    and abs_delta_objective_bits <= CONVERGENCE_LOSS_CHANGE
                ):
                    stable_loss_steps += 1
                else:
                    stable_loss_steps = 0

                state = optimizer.state[model.theta]
                iteration_s = time.perf_counter() - iteration_started
                closure_evals = int(closure_metrics["count"])
                closure_time_s = float(closure_metrics["time_s"])
                closure_mean_s = (
                    closure_time_s / closure_evals if closure_evals else float("nan")
                )

                row = {
                    "phase": "lbfgs",
                    "iteration": step,
                    "phase_iteration": lbfgs_phase_step,
                    "data_nll_bits": data_nll_bits,
                    "regularization_penalty_bits": regularization_penalty_bits,
                    "objective_bits": objective_bits,
                    "log_likelihood_bits": log_likelihood_bits,
                    "delta_nll_bits": (
                        float("nan") if delta_nll_bits is None else delta_nll_bits
                    ),
                    "delta_objective_bits": (
                        float("nan")
                        if delta_objective_bits is None
                        else delta_objective_bits
                    ),
                    "abs_delta_nll_bits": (
                        float("nan")
                        if abs_delta_nll_bits is None
                        else abs_delta_nll_bits
                    ),
                    "abs_delta_objective_bits": (
                        float("nan")
                        if abs_delta_objective_bits is None
                        else abs_delta_objective_bits
                    ),
                    "grad_inf": grad_inf,
                    "grad_norm": grad_norm,
                    "iteration_s": iteration_s,
                    "optimizer_step_s": lbfgs_step_s,
                    "adagrad_step_s": float("nan"),
                    "lbfgs_step_s": lbfgs_step_s,
                    "closure_time_s": closure_time_s,
                    "closure_mean_s": closure_mean_s,
                    "final_eval_s": final_eval_s,
                    "theta_step_inf": theta_step_inf,
                    "accepted_step": accepted_step,
                    "closure_evals": closure_evals,
                    "lbfgs_total_iter": int(state.get("n_iter", 0)),
                    "lbfgs_total_func_evals": int(state.get("func_evals", 0)),
                }
                history.append(row)

                rate_rows = rate_distribution_rows(model.theta, step)
                rate_history.extend(rate_rows)

                if step % LOG_EVERY == 0:
                    print(
                        f"phase=lbfgs iter={step:04d} "
                        f"phase_iter={lbfgs_phase_step:04d} "
                        f"data_nll_bits={data_nll_bits:.6f} "
                        f"regularization_penalty_bits={regularization_penalty_bits:.6f} "
                        f"objective_bits={objective_bits:.6f} "
                        f"loglik_bits={log_likelihood_bits:.6f} "
                        f"delta_nll_bits={float('nan') if delta_nll_bits is None else delta_nll_bits:.6g} "
                        f"delta_objective_bits={float('nan') if delta_objective_bits is None else delta_objective_bits:.6g} "
                        f"grad_inf={grad_inf:.6g} grad_norm={grad_norm:.6g} "
                        f"iteration_s={iteration_s:.3f} lbfgs_step_s={lbfgs_step_s:.3f} "
                        f"theta_step_inf={theta_step_inf:.3g} accepted_step={accepted_step} "
                        f"closure_evals={closure_evals} closure_mean_s={closure_mean_s:.3f} "
                        f"final_eval_s={final_eval_s:.3f}",
                        flush=True,
                    )
                    print("  distributions " + format_rate_distribution(rate_rows), flush=True)

                converged_grad = grad_inf <= CONVERGENCE_GRAD_INF
                converged_change = (
                    lbfgs_phase_step >= MIN_CONVERGENCE_STEPS
                    and stable_loss_steps >= CONVERGENCE_PATIENCE
                )
                if converged_grad:
                    convergence_reason = "grad_inf"
                elif converged_change:
                    convergence_reason = "loss_change_patience"

                previous_nll = data_nll_bits
                previous_objective = objective_bits
                model.clear()
                if convergence_reason is not None:
                    break
                step += 1
                lbfgs_phase_step += 1
    except KeyboardInterrupt:
        convergence_reason = "interrupted"
        print("interrupted; writing partial results", flush=True)
    finally:
        write_outputs(
            model=model,
            species_names=species_names,
            origination_probs_cpu=origination_probs_cpu,
            history=history,
            rate_history=rate_history,
            convergence_reason=convergence_reason,
            args=args,
        )
        model.close()

    print("convergence_reason", convergence_reason, flush=True)


if __name__ == "__main__":
    main()

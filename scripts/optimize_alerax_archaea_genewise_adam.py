#!/usr/bin/env python3
"""Optimize AleRax Archaea DTL rates, then sample reconciliations."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from gpurec import (
    GeneReconModel,
    SolverOptions,
    clamp_log_rate_,
    project_rate_gradient_,
    sample_reconciliations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "tests/data/alerax_archaea_davin2017"
DEFAULT_OUTPUT = REPO_ROOT / "output/alerax_archaea_genewise_adam/result.json"
OPTIMIZER_DEFAULT_LR = {
    "adam": 0.03,
    "rmsprop": 0.01,
    "sgd": 0.01,
    "rprop": 0.01,
    "adadelta": 1.0,
    "adafactor": 0.01,
    "adamax": 0.03,
}
SELF_LOOP_SOLVERS = {"neumann", "gmres"}


def parse_schedule(raw: str, *, default_self_loop_solver: str = "neumann") -> list[dict[str, Any]]:
    default_self_loop_solver = str(default_self_loop_solver).strip().lower()
    if default_self_loop_solver not in SELF_LOOP_SOLVERS:
        raise ValueError("default self-loop solver must be one of: neumann, gmres")
    phases: list[dict[str, Any]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in (3, 4, 5, 6):
            raise ValueError(
                "schedule entries must be formatted as "
                "STEPS:PI_ITERS:NEUMANN_TERMS[:DTYPE[:SELF_LOOP_SOLVER[:OPTIMIZER_LR]]]"
            )
        steps = int(parts[0])
        pi_iters = int(parts[1])
        neumann_terms = int(parts[2])
        dtype_name = parts[3] if len(parts) >= 4 else "float32"
        self_loop_solver = parts[4].strip().lower() if len(parts) == 5 else default_self_loop_solver
        phase_lr = float(parts[5]) if len(parts) == 6 else None
        if len(parts) == 6:
            self_loop_solver = parts[4].strip().lower()
        if steps < 1:
            raise ValueError("schedule steps must be positive")
        if pi_iters < 2 or pi_iters % 2 != 0:
            raise ValueError("scheduled pi_iters must be an even integer at least 2")
        if neumann_terms < 0:
            raise ValueError("scheduled neumann_terms must be non-negative")
        if dtype_name not in {"float32", "float64"}:
            raise ValueError("scheduled dtype must be float32 or float64")
        if self_loop_solver not in SELF_LOOP_SOLVERS:
            raise ValueError("scheduled self-loop solver must be one of: neumann, gmres")
        if phase_lr is not None and phase_lr <= 0.0:
            raise ValueError("scheduled optimizer LR must be positive")
        phases.append(
            {
                "steps": steps,
                "pi_iters": pi_iters,
                "neumann_terms": neumann_terms,
                "dtype": dtype_name,
                "self_loop_solver": self_loop_solver,
                "optimizer_lr": phase_lr,
            }
        )
    if not phases:
        raise ValueError("empty solver schedule")
    return phases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--species-tree", type=Path, default=None)
    parser.add_argument("--family-dir", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("global", "specieswise", "genewise"),
        default="genewise",
        help="Rate-parameter sharing mode.",
    )
    parser.add_argument("--max-families", type=int, default=16)
    parser.add_argument("--min-leaves", type=int, default=4)
    parser.add_argument(
        "--recursive-family-search",
        action="store_true",
        help="Search for .ale files recursively under --family-dir.",
    )
    parser.add_argument(
        "--family-order",
        choices=("smallest", "largest", "lexicographic"),
        default="smallest",
        help="Ordering before applying --max-families.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--theta-init-rate", type=float, default=0.05)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=2.0)
    parser.add_argument(
        "--unbounded-unprojected",
        action="store_true",
        help="Disable rate-bound gradient projection and post-step theta clamping.",
    )
    parser.add_argument("--clip-grad-norm", type=float, default=200.0)
    parser.add_argument(
        "--optimizer",
        choices=tuple(OPTIMIZER_DEFAULT_LR),
        default="adam",
        help="Optimizer for theta parameters.",
    )
    parser.add_argument(
        "--optimizer-lr",
        type=float,
        default=None,
        help="Override optimizer learning rate; defaults are optimizer-specific.",
    )
    parser.add_argument(
        "--adam-lr",
        type=float,
        default=None,
        help="Deprecated alias for --optimizer-lr.",
    )
    parser.add_argument("--optimizer-momentum", type=float, default=0.0)
    parser.add_argument("--rmsprop-alpha", type=float, default=0.99)
    parser.add_argument("--adadelta-rho", type=float, default=0.9)
    parser.add_argument(
        "--optimizer-eps",
        type=float,
        default=None,
        help="Override optimizer epsilon; defaults are optimizer-specific.",
    )
    parser.add_argument(
        "--preserve-optimizer-state-across-phases",
        action="store_true",
        help="Do not reset optimizer accumulator statistics at schedule phase boundaries.",
    )
    parser.add_argument(
        "--lr-ramp-steps",
        type=int,
        default=0,
        help="Number of initial steps in each post-transition phase used to ramp LR up to the phase target.",
    )
    parser.add_argument(
        "--lr-ramp-start-factor",
        type=float,
        default=0.25,
        help="Initial LR as a fraction of the phase target when --lr-ramp-steps is positive.",
    )
    parser.add_argument(
        "--lr-ramp-first-phase",
        action="store_true",
        help="Also apply LR ramping to the first schedule phase.",
    )
    parser.add_argument(
        "--hierarchical-eb",
        action="store_true",
        help="Add a learned hierarchical normal prior over row-wise log2 rates.",
    )
    parser.add_argument(
        "--prior-initial-sigma",
        type=float,
        default=2.0,
        help="Initial population standard deviation in log2-rate units.",
    )
    parser.add_argument(
        "--prior-min-sigma",
        type=float,
        default=0.1,
        help="Lower bound for learned population standard deviations in log2-rate units.",
    )
    parser.add_argument(
        "--prior-mu-sigma",
        type=float,
        default=10.0,
        help="Weak normal hyperprior standard deviation for population means.",
    )
    parser.add_argument(
        "--prior-log-sigma-sigma",
        type=float,
        default=0.05,
        help="Normal hyperprior standard deviation for log population sigmas.",
    )
    parser.add_argument(
        "--schedule",
        default="80:4:4:float32,80:8:8:float32,80:12:12:float32,80:16:16:float64",
        help=(
            "Comma-separated "
            "STEPS:PI_ITERS:NEUMANN_TERMS[:DTYPE[:SELF_LOOP_SOLVER[:OPTIMIZER_LR]]] phases."
        ),
    )
    parser.add_argument("--tail-window", type=int, default=25)
    parser.add_argument("--tail-slope-tol", type=float, default=0.02)
    parser.add_argument("--min-improvement-bits", type=float, default=1.0)
    parser.add_argument("--max-projected-grad-norm", type=float, default=0.05)
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="Write output JSON instead of failing when convergence criteria are not met.",
    )
    parser.add_argument("--family-chunk-size", type=int, default=16)
    parser.add_argument(
        "--clade-budget",
        type=int,
        default=120_000,
        help="Maximum clades per packed batch; use 0 to disable budget packing.",
    )
    parser.add_argument("--batch-packing", default="depth_first_fit")
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--self-loop-solver", choices=("neumann", "gmres"), default="neumann")
    parser.add_argument("--e-max-iter", type=int, default=2000)
    parser.add_argument("--e-tol", type=float, default=1e-8)
    parser.add_argument("--backtrack-families", type=int, default=3)
    parser.add_argument("--backtrack-samples", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def count_ale_leaves(ale_path: Path) -> int:
    in_leaf_section = False
    count = 0
    with ale_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if in_leaf_section and line != "#leaf-id":
                    break
                in_leaf_section = line == "#leaf-id"
                continue
            if in_leaf_section:
                count += 1
    return count


def select_families(
    family_dir: Path,
    *,
    max_families: int,
    family_order: str,
    min_leaves: int,
    recursive: bool,
) -> tuple[list[Path], dict[str, int]]:
    candidates = list((family_dir.rglob if recursive else family_dir.glob)("*.ale"))
    if not candidates:
        raise FileNotFoundError(f"no .ale files found in {family_dir}")
    leaf_counts = {path: count_ale_leaves(path) for path in candidates}
    families = [path for path in candidates if leaf_counts[path] >= min_leaves]
    if not families:
        raise FileNotFoundError(f"no .ale files with at least {min_leaves} leaves found in {family_dir}")
    if family_order == "smallest":
        families.sort(key=lambda path: (leaf_counts[path], path.stat().st_size, path.name))
    elif family_order == "largest":
        families.sort(key=lambda path: (-leaf_counts[path], -path.stat().st_size, path.name))
    else:
        families.sort(key=lambda path: path.name)
    if max_families > 0:
        families = families[:max_families]
    return families, {
        "candidate_families": len(candidates),
        "excluded_below_min_leaves": sum(1 for count in leaf_counts.values() if count < min_leaves),
        "eligible_families": sum(1 for count in leaf_counts.values() if count >= min_leaves),
        "selected_families": len(families),
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def theta_stats(theta: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        rates = torch.exp2(theta.detach())
        return {
            "theta_min": float(theta.detach().min().cpu()),
            "theta_max": float(theta.detach().max().cpu()),
            "rate_min": float(rates.min().cpu()),
            "rate_max": float(rates.max().cpu()),
            "rate_mean": float(rates.mean().cpu()),
        }


def torch_dtype(dtype_name: str) -> torch.dtype:
    return torch.float64 if dtype_name == "float64" else torch.float32


def optimizer_lr(args: argparse.Namespace) -> float:
    if args.optimizer_lr is not None and args.adam_lr is not None:
        raise ValueError("use only one of --optimizer-lr and --adam-lr")
    if args.optimizer_lr is not None:
        return float(args.optimizer_lr)
    if args.adam_lr is not None:
        return float(args.adam_lr)
    return OPTIMIZER_DEFAULT_LR[str(args.optimizer)]


def optimizer_eps(args: argparse.Namespace, optimizer: str) -> float:
    if args.optimizer_eps is not None:
        return float(args.optimizer_eps)
    if optimizer == "adadelta":
        return 1e-6
    if optimizer == "adafactor":
        return 1e-3
    return 1e-8


def make_optimizer(params: list[torch.Tensor], args: argparse.Namespace, lr: float) -> torch.optim.Optimizer:
    optimizer = str(args.optimizer)
    eps = optimizer_eps(args, optimizer)
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, eps=eps)
    if optimizer == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=lr,
            alpha=args.rmsprop_alpha,
            eps=eps,
            momentum=args.optimizer_momentum,
        )
    if optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=args.optimizer_momentum)
    if optimizer == "rprop":
        return torch.optim.Rprop(params, lr=lr)
    if optimizer == "adadelta":
        return torch.optim.Adadelta(params, lr=lr, rho=args.adadelta_rho, eps=eps)
    if optimizer == "adafactor":
        return torch.optim.Adafactor(params, lr=lr, eps=(None, eps))
    if optimizer == "adamax":
        return torch.optim.Adamax(params, lr=lr, eps=eps)
    raise ValueError(f"unsupported optimizer {optimizer!r}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def ramped_lr(target_lr: float, *, phase_step: int, phase_index: int, args: argparse.Namespace) -> float:
    if args.lr_ramp_steps <= 0:
        return target_lr
    if phase_index == 0 and not args.lr_ramp_first_phase:
        return target_lr
    ramp_steps = int(args.lr_ramp_steps)
    if phase_step >= ramp_steps:
        return target_lr
    if ramp_steps == 1:
        return target_lr
    start_lr = target_lr * float(args.lr_ramp_start_factor)
    fraction = float(phase_step) / float(ramp_steps - 1)
    return start_lr + (target_lr - start_lr) * fraction


def optimizer_options(args: argparse.Namespace, lr: float) -> dict[str, Any]:
    options: dict[str, Any] = {
        "name": str(args.optimizer),
        "lr": lr,
        "eps": optimizer_eps(args, str(args.optimizer)),
    }
    if args.optimizer in {"rmsprop", "sgd"}:
        options["momentum"] = args.optimizer_momentum
    if args.optimizer == "rmsprop":
        options["alpha"] = args.rmsprop_alpha
    if args.optimizer == "adadelta":
        options["rho"] = args.adadelta_rho
    return options


def inverse_softplus(value: float) -> float:
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def initialize_hierarchical_prior(
    args: argparse.Namespace,
    theta: torch.Tensor,
) -> dict[str, Any] | None:
    if not args.hierarchical_eb:
        return None
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError("hierarchical EB prior requires row-wise theta with shape (rows, 3)")
    if args.prior_min_sigma <= 0.0:
        raise ValueError("--prior-min-sigma must be positive")
    if args.prior_initial_sigma <= args.prior_min_sigma:
        raise ValueError("--prior-initial-sigma must be greater than --prior-min-sigma")
    if args.prior_mu_sigma <= 0.0 or args.prior_log_sigma_sigma <= 0.0:
        raise ValueError("prior hyperprior standard deviations must be positive")

    mu0_value = math.log2(args.theta_init_rate)
    raw_sigma_value = inverse_softplus(args.prior_initial_sigma - args.prior_min_sigma)
    return {
        "mu": torch.nn.Parameter(torch.full((3,), mu0_value, dtype=theta.dtype, device=theta.device)),
        "raw_sigma": torch.nn.Parameter(torch.full((3,), raw_sigma_value, dtype=theta.dtype, device=theta.device)),
        "mu0_value": mu0_value,
        "log_sigma0_value": math.log(args.prior_initial_sigma),
        "row_count": int(theta.shape[0]),
    }


def move_hierarchical_prior_(
    prior: dict[str, Any] | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if prior is None:
        return
    prior["mu"] = torch.nn.Parameter(prior["mu"].detach().to(device=device, dtype=dtype))
    prior["raw_sigma"] = torch.nn.Parameter(prior["raw_sigma"].detach().to(device=device, dtype=dtype))


def hierarchical_prior_parameters(prior: dict[str, Any] | None) -> list[torch.Tensor]:
    if prior is None:
        return []
    return [prior["mu"], prior["raw_sigma"]]


def hierarchical_sigma(prior: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    return args.prior_min_sigma + torch.nn.functional.softplus(prior["raw_sigma"])


def hierarchical_prior_bits(
    theta: torch.Tensor,
    prior: dict[str, Any] | None,
    args: argparse.Namespace,
) -> torch.Tensor:
    if prior is None:
        return theta.new_zeros(())
    sigma = hierarchical_sigma(prior, args)
    centered = (theta - prior["mu"].unsqueeze(0)) / sigma.unsqueeze(0)
    nll_nats = 0.5 * torch.sum(centered * centered) + theta.shape[0] * torch.sum(torch.log(sigma))

    mu0 = theta.new_tensor(prior["mu0_value"])
    log_sigma0 = theta.new_tensor(prior["log_sigma0_value"])
    nll_nats = nll_nats + 0.5 * torch.sum(((prior["mu"] - mu0) / args.prior_mu_sigma) ** 2)
    nll_nats = nll_nats + 0.5 * torch.sum(
        ((torch.log(sigma) - log_sigma0) / args.prior_log_sigma_sigma) ** 2
    )
    return nll_nats / math.log(2.0)


def hierarchical_prior_stats(prior: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    if prior is None:
        return {}
    with torch.no_grad():
        sigma = hierarchical_sigma(prior, args)
        return {
            "prior_mu": [float(value) for value in prior["mu"].detach().cpu()],
            "prior_sigma": [float(value) for value in sigma.detach().cpu()],
        }


def event_summary(sample: list[Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for event in sample:
        if not event:
            continue
        name = str(event[0])
        counts[name] = counts.get(name, 0) + 1
    return {
        "event_count": len(sample),
        "event_type_counts": counts,
        "first_events": [list(event) for event in sample[:8]],
    }


def fit_line_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = torch.arange(n, dtype=torch.float64)
    ys = torch.tensor(values, dtype=torch.float64)
    x_centered = xs - xs.mean()
    denom = torch.sum(x_centered * x_centered)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.sum(x_centered * (ys - ys.mean())) / denom)


def assert_converged(
    history: list[dict[str, Any]],
    *,
    tail_window: int,
    tail_slope_tol: float,
    min_improvement_bits: float,
    max_projected_grad_norm: float,
) -> dict[str, float]:
    losses = [float(row["loss_bits"]) for row in history]
    if len(losses) < max(2, tail_window):
        raise RuntimeError("not enough history rows to assess convergence")
    initial = losses[0]
    final = losses[-1]
    best = min(losses)
    improvement = initial - final
    tail = losses[-int(tail_window) :]
    tail_slope = fit_line_slope(tail)
    tail_span = max(tail) - min(tail)
    final_projected_grad_norm = float(history[-1]["projected_grad_norm"])
    if final > best + max(1e-4, abs(best) * 1e-6):
        raise RuntimeError(f"final loss {final:.6f} is not near best loss {best:.6f}")
    if improvement < min_improvement_bits:
        raise RuntimeError(
            f"loss improvement {improvement:.6f} bits is below required {min_improvement_bits:.6f}"
        )
    if abs(tail_slope) > tail_slope_tol:
        raise RuntimeError(
            f"tail loss slope {tail_slope:.6g} bits/step exceeds tolerance {tail_slope_tol:.6g}"
        )
    if final_projected_grad_norm > max_projected_grad_norm:
        raise RuntimeError(
            f"final projected gradient norm {final_projected_grad_norm:.6g} "
            f"exceeds tolerance {max_projected_grad_norm:.6g}"
        )
    return {
        "initial_loss_bits": initial,
        "final_loss_bits": final,
        "best_loss_bits": best,
        "improvement_bits": improvement,
        "tail_slope_bits_per_step": tail_slope,
        "tail_span_bits": tail_span,
        "final_projected_grad_norm": final_projected_grad_norm,
    }


def summarize_history(history: list[dict[str, Any]], *, tail_window: int) -> dict[str, float]:
    losses = [float(row["loss_bits"]) for row in history]
    initial = losses[0]
    final = losses[-1]
    best = min(losses)
    tail = losses[-int(tail_window) :]
    return {
        "initial_loss_bits": initial,
        "final_loss_bits": final,
        "best_loss_bits": best,
        "improvement_bits": initial - final,
        "tail_slope_bits_per_step": fit_line_slope(tail),
        "tail_span_bits": max(tail) - min(tail),
        "final_projected_grad_norm": float(history[-1]["projected_grad_norm"]),
    }


def main() -> None:
    args = parse_args()
    if args.lr_ramp_steps < 0:
        raise ValueError("--lr-ramp-steps must be non-negative")
    if not (0.0 < args.lr_ramp_start_factor <= 1.0):
        raise ValueError("--lr-ramp-start-factor must be in (0, 1]")
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    use_rate_bounds = not bool(args.unbounded_unprojected)
    species_tree = args.species_tree or args.data_root / "species_reference/reference_species_tree.newick"
    family_dir = args.family_dir or args.data_root / "ale_gene_tree_distributions/main_families_ge4seq"
    clade_budget = None if args.clade_budget == 0 else args.clade_budget
    lr = optimizer_lr(args)
    ale_families, selection = select_families(
        family_dir,
        max_families=args.max_families,
        family_order=args.family_order,
        min_leaves=args.min_leaves,
        recursive=args.recursive_family_search,
    )
    schedule = parse_schedule(args.schedule, default_self_loop_solver=args.self_loop_solver)
    solver_options = SolverOptions(
        e_init=-1000.0,
        e_max_iter=args.e_max_iter,
        e_tol=args.e_tol,
        pi_iters=int(schedule[0]["pi_iters"]),
        neumann_terms=int(schedule[0]["neumann_terms"]),
        self_loop_solver=str(schedule[0]["self_loop_solver"]),
    )
    model = GeneReconModel(
        species_tree,
        ale_families,
        mode=args.mode,
        device=device,
        family_chunk_size=args.family_chunk_size,
        clade_budget=clade_budget,
        batch_packing=args.batch_packing,
        max_wave_size=args.max_wave_size,
        solver_options=solver_options,
    )
    current_dtype_name = str(schedule[0]["dtype"])
    model.to(dtype=torch_dtype(current_dtype_name))
    model.receiver_weights.requires_grad_(False)
    with torch.no_grad():
        model.theta.fill_(math.log2(args.theta_init_rate))
        if use_rate_bounds:
            clamp_log_rate_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
        model.clear_warm_starts()

    hierarchical_prior = initialize_hierarchical_prior(args, model.theta)
    optimizer_params = [model.theta, *hierarchical_prior_parameters(hierarchical_prior)]
    current_lr = float(schedule[0]["optimizer_lr"] or lr)
    optimizer = make_optimizer(optimizer_params, args, current_lr)
    history: list[dict[str, Any]] = []
    global_step = 0
    t0 = time.perf_counter()
    for phase_index, phase in enumerate(schedule):
        phase_steps = int(phase["steps"])
        phase_dtype_name = str(phase["dtype"])
        phase_target_lr = float(phase["optimizer_lr"] or lr)
        model.configure_solver(
            pi_iters=int(phase["pi_iters"]),
            neumann_terms=int(phase["neumann_terms"]),
            self_loop_solver=str(phase["self_loop_solver"]),
        )
        optimizer_reset = False
        if phase_dtype_name != current_dtype_name:
            model.to(dtype=torch_dtype(phase_dtype_name))
            current_dtype_name = phase_dtype_name
            move_hierarchical_prior_(
                hierarchical_prior,
                dtype=torch_dtype(phase_dtype_name),
                device=model.theta.device,
            )
            optimizer_params = [model.theta, *hierarchical_prior_parameters(hierarchical_prior)]
            optimizer = make_optimizer(optimizer_params, args, phase_target_lr)
            current_lr = phase_target_lr
            optimizer_reset = True
        elif phase_index > 0 and not args.preserve_optimizer_state_across_phases:
            optimizer = make_optimizer(optimizer_params, args, phase_target_lr)
            current_lr = phase_target_lr
            optimizer_reset = True
        elif phase_target_lr != current_lr:
            set_optimizer_lr(optimizer, phase_target_lr)
            current_lr = phase_target_lr
        model.clear_warm_starts()
        for phase_step in range(phase_steps):
            step_lr = ramped_lr(phase_target_lr, phase_step=phase_step, phase_index=phase_index, args=args)
            if step_lr != current_lr:
                set_optimizer_lr(optimizer, step_lr)
                current_lr = step_lr
            optimizer.zero_grad(set_to_none=True)
            sync(device)
            step_t0 = time.perf_counter()
            data_loss = model()
            prior_loss = hierarchical_prior_bits(model.theta, hierarchical_prior, args)
            loss = data_loss + prior_loss
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"non-finite loss at step {global_step}: total={loss} data={data_loss} prior={prior_loss}"
                )
            loss.backward()
            if model.theta.grad is None or not torch.isfinite(model.theta.grad).all().item():
                raise FloatingPointError(f"non-finite theta gradient at step {global_step}")
            raw_grad_norm = float(model.theta.grad.detach().norm().cpu())
            if use_rate_bounds:
                project_rate_gradient_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
            projected_grad_norm = float(model.theta.grad.detach().norm().cpu())
            if args.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_([model.theta], args.clip_grad_norm)
            clipped_grad_norm = float(model.theta.grad.detach().norm().cpu())
            optimizer.step()
            if use_rate_bounds:
                clamp_log_rate_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
            sync(device)

            row = {
                "step": global_step,
                "phase": phase_index,
                "phase_step": phase_step,
                "optimizer": str(args.optimizer),
                "optimizer_lr": current_lr,
                "optimizer_target_lr": phase_target_lr,
                "optimizer_reset_at_phase_start": optimizer_reset,
                "pi_iters": int(phase["pi_iters"]),
                "neumann_terms": int(phase["neumann_terms"]),
                "self_loop_solver": str(phase["self_loop_solver"]),
                "dtype": phase_dtype_name,
                "loss_bits": float(loss.detach().cpu()),
                "data_loss_bits": float(data_loss.detach().cpu()),
                "prior_loss_bits": float(prior_loss.detach().cpu()),
                "raw_grad_norm": raw_grad_norm,
                "projected_grad_norm": projected_grad_norm,
                "clipped_grad_norm": clipped_grad_norm,
                "step_s": time.perf_counter() - step_t0,
                **theta_stats(model.theta),
                **hierarchical_prior_stats(hierarchical_prior, args),
            }
            history.append(row)
            if global_step == 0 or (global_step + 1) % 20 == 0:
                print(
                    f"step={global_step + 1:04d} phase={phase_index} "
                    f"pi={phase['pi_iters']} neumann={phase['neumann_terms']} "
                    f"solver={phase['self_loop_solver']} dtype={phase_dtype_name} "
                    f"opt={args.optimizer} lr={current_lr:g} "
                    f"loss={row['loss_bits']:.6f} data={row['data_loss_bits']:.6f} "
                    f"prior={row['prior_loss_bits']:.6f} grad={raw_grad_norm:.3g} "
                    f"rate=[{row['rate_min']:.3g},{row['rate_max']:.3g}] "
                    f"step_s={row['step_s']:.3f}",
                    flush=True,
                )
            global_step += 1

    convergence_error = None
    try:
        convergence = assert_converged(
            history,
            tail_window=args.tail_window,
            tail_slope_tol=args.tail_slope_tol,
            min_improvement_bits=args.min_improvement_bits,
            max_projected_grad_norm=args.max_projected_grad_norm,
        )
        convergence["converged"] = True
    except RuntimeError as exc:
        if not args.allow_unconverged:
            raise
        convergence_error = str(exc)
        convergence = summarize_history(history, tail_window=args.tail_window)
        convergence["converged"] = False

    backtracking: list[dict[str, Any]] = []
    max_backtrack_families = min(args.backtrack_families, len(ale_families))
    for family_index in range(max_backtrack_families):
        for sample_index in range(args.backtrack_samples):
            seed = args.seed + 1009 * family_index + sample_index
            sample = sample_reconciliations(model, family_index=family_index, seed=seed)
            backtracking.append(
                {
                    "family_index": family_index,
                    "family_path": str(ale_families[family_index]),
                    "seed": seed,
                    **event_summary(sample),
                }
            )

    payload = {
        "species_tree": str(species_tree),
        "family_dir": str(family_dir),
        "families": [str(path) for path in ale_families],
        "family_input": "ALE complete CCP files",
        "min_leaves": args.min_leaves,
        "recursive_family_search": args.recursive_family_search,
        "selection": selection,
        "mode": args.mode,
        "device": str(device),
        "optimizer": optimizer_options(args, lr),
        "optimizer_phase_controls": {
            "reset_optimizer_state_across_phases": not bool(args.preserve_optimizer_state_across_phases),
            "lr_ramp_steps": args.lr_ramp_steps,
            "lr_ramp_start_factor": args.lr_ramp_start_factor,
            "lr_ramp_first_phase": bool(args.lr_ramp_first_phase),
        },
        "use_rate_bounds": use_rate_bounds,
        "hierarchical_eb": bool(args.hierarchical_eb),
        "hierarchical_prior": {
            "initial_sigma": args.prior_initial_sigma,
            "min_sigma": args.prior_min_sigma,
            "mu_sigma": args.prior_mu_sigma,
            "log_sigma_sigma": args.prior_log_sigma_sigma,
            "row_count": hierarchical_prior["row_count"],
            **hierarchical_prior_stats(hierarchical_prior, args),
        }
        if hierarchical_prior is not None
        else None,
        "clip_grad_norm": args.clip_grad_norm,
        "clade_budget": clade_budget,
        "schedule": schedule,
        "solver_options": {
            "e_max_iter": solver_options.e_max_iter,
            "e_tol": solver_options.e_tol,
            "self_loop_solver": solver_options.self_loop_solver,
        },
        "convergence": convergence,
        "convergence_error": convergence_error,
        "elapsed_s": time.perf_counter() - t0,
        "history": history,
        "backtracking": backtracking,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    status = "converged" if convergence.get("converged") else "unconverged"
    print(
        f"{status} final_loss={convergence['final_loss_bits']:.6f} "
        f"improvement={convergence['improvement_bits']:.6f} "
        f"tail_slope={convergence['tail_slope_bits_per_step']:.6g}",
        flush=True,
    )
    if convergence_error:
        print(f"convergence_error={convergence_error}", flush=True)
    print(f"backtracking_samples={len(backtracking)} output_json={args.output_json}", flush=True)


if __name__ == "__main__":
    main()

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
    parser.add_argument(
        "--init-json",
        type=Path,
        default=None,
        help="Initialize theta from a previous output JSON.",
    )
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
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
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
        "--lr-decay-steps",
        type=int,
        default=0,
        help="Number of initial steps in each phase used to decay LR from the schedule value.",
    )
    parser.add_argument(
        "--lr-decay-end-factor",
        type=float,
        default=1.0,
        help="Final LR as a fraction of the schedule value when --lr-decay-steps is positive.",
    )
    parser.add_argument(
        "--penalty-lambda",
        type=float,
        default=1.0,
        help=(
            "Sanderson roughness-penalty smoothing parameter. The objective is "
            "NLL + lambda * Phi(theta), where Phi penalizes squared parent-child "
            "differences in log2-rate across the species tree (specieswise mode)."
        ),
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Choose --penalty-lambda by k-fold cross-validation over gene families first.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of family folds for cross-validation.",
    )
    parser.add_argument(
        "--cv-lambda-grid",
        default="0.1,0.3,1.0,3.0,10.0,30.0,100.0",
        help="Comma-separated lambda grid scanned by cross-validation.",
    )
    parser.add_argument(
        "--cv-steps",
        type=int,
        default=60,
        help="Optimization steps per (fold, lambda) fit during cross-validation.",
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


def theta_values(theta: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        detached = theta.detach().cpu()
        return {
            "theta": detached.tolist(),
            "rates": torch.exp2(detached).tolist(),
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
        return torch.optim.Adam(params, lr=lr, betas=(args.adam_beta1, args.adam_beta2), eps=eps)
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
        return torch.optim.Adamax(params, lr=lr, betas=(args.adam_beta1, args.adam_beta2), eps=eps)
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


def decayed_lr(start_lr: float, *, phase_step: int, args: argparse.Namespace) -> float:
    if args.lr_decay_steps <= 0:
        return start_lr
    decay_steps = int(args.lr_decay_steps)
    end_lr = start_lr * float(args.lr_decay_end_factor)
    if phase_step >= decay_steps:
        return end_lr
    if decay_steps == 1:
        return end_lr
    fraction = float(phase_step) / float(decay_steps - 1)
    return start_lr + (end_lr - start_lr) * fraction


def scheduled_step_lr(target_lr: float, *, phase_step: int, phase_index: int, args: argparse.Namespace) -> float:
    if args.lr_decay_steps > 0:
        return decayed_lr(target_lr, phase_step=phase_step, args=args)
    return ramped_lr(target_lr, phase_step=phase_step, phase_index=phase_index, args=args)


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
    if args.optimizer in {"adam", "adamax"}:
        options["betas"] = [args.adam_beta1, args.adam_beta2]
    return options


def load_initial_state_(
    args: argparse.Namespace,
    *,
    model: GeneReconModel,
) -> None:
    if args.init_json is None:
        return
    payload = json.loads(args.init_json.read_text(encoding="utf-8"))
    if payload.get("mode") != args.mode:
        raise ValueError(f"--init-json mode {payload.get('mode')!r} does not match requested mode {args.mode!r}")
    final_theta = payload.get("final_theta")
    if not final_theta or "theta" not in final_theta:
        raise ValueError("--init-json does not contain final_theta.theta")
    theta = torch.as_tensor(final_theta["theta"], dtype=model.theta.dtype, device=model.theta.device)
    if tuple(theta.shape) != tuple(model.theta.shape):
        raise ValueError(f"--init-json theta shape {tuple(theta.shape)} does not match model theta {tuple(model.theta.shape)}")
    with torch.no_grad():
        model.theta.copy_(theta)


def roughness_penalty(theta: torch.Tensor, species_helpers: dict[str, Any]) -> torch.Tensor:
    """Sanderson (2002) roughness penalty over the species tree, summed across D/T/L.

    ``theta`` is the specieswise ``[S, 3]`` log2-rate tensor. Penalizes squared
    parent-child differences in log2-rate for every node except the root and the
    root's children, plus the variance of the root children's rates (eq. 3).
    Assumes a binary species tree (the root has exactly two children).
    """
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError("roughness_penalty requires specieswise theta of shape (S, 3)")
    sp_parent = species_helpers["sp_parent"].to(theta.device).long()
    sp_child1 = species_helpers["sp_child1"].to(theta.device).long()
    sp_child2 = species_helpers["sp_child2"].to(theta.device).long()
    S = theta.shape[0]
    idx = torch.arange(S, device=theta.device)
    root = idx[sp_parent < 0]
    root_children = torch.cat([sp_child1[root], sp_child2[root]])
    rc_mask = torch.zeros(S, dtype=torch.bool, device=theta.device)
    rc_mask[root_children] = True
    use = (sp_parent >= 0) & (~rc_mask)
    parent = sp_parent.clamp_min(0)
    diff = theta[use] - theta[parent[use]]
    phi = (diff * diff).sum()
    phi = phi + theta[root_children].var(dim=0, unbiased=False).sum()
    return phi


def kfold_family_indices(n: int, folds: int, *, seed: int = 0) -> list[list[int]]:
    """Round-robin partition of ``range(n)`` into ``folds`` shuffled folds."""
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).tolist()
    return [perm[fold::folds] for fold in range(folds)]


def cross_validate_lambda(
    families: list[Path],
    lambda_grid: list[float],
    *,
    folds: int,
    build_model,
    fit_theta,
    eval_nll,
    seed: int = 0,
    log=print,
) -> tuple[float, dict[float, float]]:
    """k-fold cross-validation over gene families to choose the smoothing lambda.

    For each fold a train and a validation specieswise model are built via
    ``build_model(families)``. For each lambda (ascending, warm-starting theta
    across lambdas) ``fit_theta(model_train, lam, warm_start)`` returns a fitted
    theta, scored on the held-out families with ``eval_nll(model_val, theta)``.
    The lambda minimizing the summed held-out NLL is returned with the full curve.
    """
    fold_members = kfold_family_indices(len(families), folds, seed=seed)
    grid = sorted(float(lam) for lam in lambda_grid)
    cv_scores: dict[float, float] = {lam: 0.0 for lam in grid}
    for fold, val_idx in enumerate(fold_members):
        val_set = set(val_idx)
        train_families = [fam for i, fam in enumerate(families) if i not in val_set]
        val_families = [families[i] for i in val_idx]
        if not train_families or not val_families:
            continue
        model_train = build_model(train_families)
        model_val = build_model(val_families)
        warm_start = None
        for lam in grid:
            warm_start = fit_theta(model_train, lam, warm_start)
            score = eval_nll(model_val, warm_start)
            cv_scores[lam] += score
            log(
                f"cv fold={fold + 1}/{folds} lambda={lam:g} "
                f"held_out_nll={score:.4f}",
                flush=True,
            )
    best_lambda = min(cv_scores, key=cv_scores.get)
    return best_lambda, cv_scores


def gradient_norm(parameters: list[torch.Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all().item():
            return float("nan")
        total += float(torch.sum(grad * grad).cpu())
    return math.sqrt(total)


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
    final_joint_grad_norm = float(history[-1].get("joint_grad_norm", final_projected_grad_norm))
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
    if final_joint_grad_norm > max_projected_grad_norm:
        raise RuntimeError(
            f"final joint gradient norm {final_joint_grad_norm:.6g} "
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
        "final_joint_grad_norm": final_joint_grad_norm,
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
        "final_joint_grad_norm": float(history[-1].get("joint_grad_norm", history[-1]["projected_grad_norm"])),
    }


def main() -> None:
    args = parse_args()
    if args.lr_ramp_steps < 0:
        raise ValueError("--lr-ramp-steps must be non-negative")
    if not (0.0 < args.lr_ramp_start_factor <= 1.0):
        raise ValueError("--lr-ramp-start-factor must be in (0, 1]")
    if args.lr_decay_steps < 0:
        raise ValueError("--lr-decay-steps must be non-negative")
    if not (0.0 < args.lr_decay_end_factor <= 1.0):
        raise ValueError("--lr-decay-end-factor must be in (0, 1]")
    if args.lr_ramp_steps > 0 and args.lr_decay_steps > 0:
        raise ValueError("use only one of --lr-ramp-steps and --lr-decay-steps")
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

    load_initial_state_(args, model=model)

    penalty_active = args.mode == "specieswise"
    if not penalty_active and float(args.penalty_lambda) != 0.0:
        print(
            f"warning: roughness penalty requires --mode specieswise; "
            f"ignoring --penalty-lambda {args.penalty_lambda:g} in mode {args.mode!r}",
            flush=True,
        )

    penalty_lambda = float(args.penalty_lambda)
    cv_result = None
    if args.cv:
        if not penalty_active:
            raise ValueError("--cv with the roughness penalty requires --mode specieswise")
        lambda_grid = [float(x) for x in str(args.cv_lambda_grid).split(",") if x.strip()]

        def _cv_build_model(fams: list[Path]) -> GeneReconModel:
            sub = GeneReconModel(
                species_tree,
                fams,
                mode=args.mode,
                device=device,
                family_chunk_size=args.family_chunk_size,
                clade_budget=clade_budget,
                batch_packing=args.batch_packing,
                max_wave_size=args.max_wave_size,
                solver_options=solver_options,
            )
            sub.to(dtype=torch_dtype(str(schedule[0]["dtype"])))
            sub.receiver_weights.requires_grad_(False)
            return sub

        def _cv_fit_theta(sub: GeneReconModel, lam: float, warm_start: torch.Tensor | None) -> torch.Tensor:
            with torch.no_grad():
                if warm_start is not None:
                    sub.theta.copy_(warm_start.to(sub.theta))
                else:
                    sub.theta.fill_(math.log2(args.theta_init_rate))
                sub.clear_warm_starts()
            opt = make_optimizer([sub.theta], args, lr)
            for _ in range(int(args.cv_steps)):
                opt.zero_grad(set_to_none=True)
                loss = sub() + lam * roughness_penalty(sub.theta, sub.species_helpers)
                loss.backward()
                opt.step()
            return sub.theta.detach().clone()

        def _cv_eval_nll(sub: GeneReconModel, theta: torch.Tensor) -> float:
            with torch.no_grad():
                return float(sub(theta=theta.to(sub.theta)).detach().cpu())

        best_lambda, cv_scores = cross_validate_lambda(
            ale_families,
            lambda_grid,
            folds=int(args.cv_folds),
            build_model=_cv_build_model,
            fit_theta=_cv_fit_theta,
            eval_nll=_cv_eval_nll,
            seed=args.seed,
        )
        penalty_lambda = float(best_lambda)
        cv_result = {
            "lambda_grid": lambda_grid,
            "cv_scores": {str(k): v for k, v in cv_scores.items()},
            "best_lambda": best_lambda,
            "folds": int(args.cv_folds),
            "cv_steps": int(args.cv_steps),
        }
        print(f"cv selected penalty_lambda={penalty_lambda:g}", flush=True)

    optimizer_params = [model.theta]
    current_lr = float(schedule[0]["optimizer_lr"] or lr)
    optimizer = make_optimizer(optimizer_params, args, current_lr)
    history: list[dict[str, Any]] = []
    global_step = 0
    t0 = time.perf_counter()

    def loss_backward(
        *,
        step_label: int,
        collect_metrics: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        optimizer.zero_grad(set_to_none=True)
        data_loss = model()
        if penalty_active:
            penalty_loss = penalty_lambda * roughness_penalty(model.theta, model.species_helpers)
        else:
            penalty_loss = data_loss.new_zeros(())
        loss = data_loss + penalty_loss
        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"non-finite loss at step {step_label}: total={loss} data={data_loss} penalty={penalty_loss}"
            )
        loss.backward()
        if model.theta.grad is None or not torch.isfinite(model.theta.grad).all().item():
            raise FloatingPointError(f"non-finite theta gradient at step {step_label}")

        metrics: dict[str, float] = {}
        if collect_metrics:
            metrics["raw_grad_norm"] = float(model.theta.grad.detach().norm().cpu())
            metrics["theta_grad_max"] = float(model.theta.grad.detach().abs().max().cpu())
        if use_rate_bounds:
            project_rate_gradient_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
        if collect_metrics:
            metrics["projected_grad_norm"] = float(model.theta.grad.detach().norm().cpu())
        if args.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([model.theta], args.clip_grad_norm)
        if collect_metrics:
            metrics["clipped_grad_norm"] = float(model.theta.grad.detach().norm().cpu())
        return loss, data_loss, penalty_loss, metrics

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
            optimizer_params = [model.theta]
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
            step_lr = scheduled_step_lr(
                phase_target_lr,
                phase_step=phase_step,
                phase_index=phase_index,
                args=args,
            )
            if step_lr != current_lr:
                set_optimizer_lr(optimizer, step_lr)
                current_lr = step_lr
            sync(device)
            step_t0 = time.perf_counter()
            loss, data_loss, penalty_loss, grad_metrics = loss_backward(
                step_label=global_step,
                collect_metrics=True,
            )
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
                "penalty_bits": float(penalty_loss.detach().cpu()),
                "penalty_lambda": penalty_lambda,
                **grad_metrics,
                "step_s": time.perf_counter() - step_t0,
                **theta_stats(model.theta),
            }
            history.append(row)
            if global_step == 0 or (global_step + 1) % 20 == 0:
                print(
                    f"step={global_step + 1:04d} phase={phase_index} "
                    f"pi={phase['pi_iters']} neumann={phase['neumann_terms']} "
                    f"solver={phase['self_loop_solver']} dtype={phase_dtype_name} "
                    f"opt={args.optimizer} lr={current_lr:g} "
                    f"loss={row['loss_bits']:.6f} data={row['data_loss_bits']:.6f} "
                    f"penalty={row['penalty_bits']:.6f} "
                    f"|g_theta|inf={row['theta_grad_max']:.3g} "
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
            "lr_decay_steps": args.lr_decay_steps,
            "lr_decay_end_factor": args.lr_decay_end_factor,
        },
        "use_rate_bounds": use_rate_bounds,
        "penalty": {
            "method": "sanderson_roughness",
            "lambda": penalty_lambda,
            "active": penalty_active,
            "cross_validation": cv_result,
        },
        "clip_grad_norm": args.clip_grad_norm,
        "clade_budget": clade_budget,
        "schedule": schedule,
        "solver_options": {
            "e_max_iter": solver_options.e_max_iter,
            "e_tol": solver_options.e_tol,
            "self_loop_solver": solver_options.self_loop_solver,
        },
        "final_theta": theta_values(model.theta),
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

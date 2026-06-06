#!/usr/bin/env python3
"""Run a gpurec capacity benchmark on a species tree plus gene-tree files."""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import time
from pathlib import Path
from typing import Any

import torch

from gpurec import GeneReconModel, SolverOptions, clamp_log_rate_, project_rate_gradient_
from gpurec.core.kernels import wave_backward as wave_backward_module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--species-tree", required=True, type=Path)
    parser.add_argument("--gene-tree-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.treefile")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--select", choices=("sorted", "largest", "random"), default="sorted")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizer", choices=("adam", "fd-diag-hessian-sgd"), default="adam")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hessian-eps", type=float, default=1e-3)
    parser.add_argument("--hessian-damping", type=float, default=1e-2)
    parser.add_argument("--hessian-max-step", type=float, default=1.0)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=1e-2)
    parser.add_argument("--convergence-window", type=int, default=20)
    parser.add_argument("--rel-loss-tol", type=float, default=1e-5)
    parser.add_argument("--projected-grad-tol", type=float, default=1e-4)
    parser.add_argument("--projected-grad-rel-tol", type=float, default=1e-5)
    parser.add_argument("--family-chunk-size", type=int, default=25)
    parser.add_argument("--clade-budget", type=int, default=315_000)
    parser.add_argument("--batch-packing", default="depth_first_fit")
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--e-max-iter", type=int, default=2000)
    parser.add_argument("--e-tol", type=float, default=1e-8)
    parser.add_argument("--pi-iters", type=int, default=6)
    parser.add_argument("--neumann-terms", type=int, default=3)
    parser.add_argument("--self-loop-solver", choices=("neumann", "gmres", "gmres_fixed"), default="neumann")
    parser.add_argument(
        "--gmres-max-iter",
        type=int,
        default=None,
        help=(
            "Maximum GMRES Krylov dimension. If omitted, --neumann-terms is "
            "used for both Neumann terms and GMRES max iterations."
        ),
    )
    parser.add_argument("--gmres-tol", type=float, default=1e-10)
    parser.add_argument("--gmres-check-interval", type=int, default=1)
    parser.add_argument(
        "--gmres-reuse-check-schedule",
        action="store_true",
        help=(
            "Reuse each wave's previous adaptive GMRES iteration count as the "
            "minimum first residual-check iteration on the next backward pass."
        ),
    )
    parser.add_argument("--bicgstab-max-iter", type=int, default=500)
    parser.add_argument("--bicgstab-tol", type=float, default=1e-7)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args(argv)
    if args.gmres_max_iter is not None:
        if args.gmres_max_iter < 1:
            parser.error("--gmres-max-iter must be at least 1")
        if args.self_loop_solver == "neumann":
            parser.error("--gmres-max-iter requires --self-loop-solver gmres or gmres_fixed")
    return args


def solver_options_from_args(args: argparse.Namespace) -> SolverOptions:
    neumann_terms = int(args.neumann_terms)
    if args.self_loop_solver in ("gmres", "gmres_fixed") and args.gmres_max_iter is not None:
        neumann_terms = int(args.gmres_max_iter)
    return SolverOptions(
        e_max_iter=args.e_max_iter,
        e_tol=args.e_tol,
        pi_iters=args.pi_iters,
        neumann_terms=neumann_terms,
        self_loop_solver=args.self_loop_solver,
        gmres_tol=args.gmres_tol,
        gmres_check_interval=args.gmres_check_interval,
        gmres_reuse_check_schedule=args.gmres_reuse_check_schedule,
        bicgstab_max_iter=args.bicgstab_max_iter,
        bicgstab_tol=args.bicgstab_tol,
    )


def choose_gene_trees(args: argparse.Namespace) -> list[Path]:
    globber = args.gene_tree_dir.rglob if args.recursive else args.gene_tree_dir.glob
    files = sorted(globber(args.pattern))
    if args.select == "largest":
        files = sorted(files, key=lambda path: path.stat().st_size, reverse=True)
    elif args.select == "random":
        rng = random.Random(args.seed)
        rng.shuffle(files)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"no gene-tree files matched {args.gene_tree_dir / args.pattern}")
    return files


def finite_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_memory(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def projected_grad_stats(model: GeneReconModel, min_rate: float, max_rate: float) -> dict[str, float | None]:
    grad = model.theta.grad
    if grad is None:
        return {"max_abs": None, "l2": None}
    projected = projected_grad(model, min_rate, max_rate)
    return {
        "max_abs": finite_float(float(projected.abs().max().item())),
        "l2": finite_float(float(torch.linalg.vector_norm(projected).item())),
    }


def projected_grad(model: GeneReconModel, min_rate: float, max_rate: float) -> torch.Tensor:
    grad = model.theta.grad
    if grad is None:
        raise RuntimeError("missing gradient to project")
    projected = grad.detach().clone()
    project_rate_gradient_(model.theta, projected, min_rate=min_rate, max_rate=max_rate)
    return projected


def self_loop_wave_count(model: GeneReconModel) -> int:
    return int(sum(len(static.wave_layout["wave_metas"]) for static in model.batch_statics))


class SelfLoopBackwardRecorder:
    """Collect self-loop backward work across one optimizer step."""

    def __init__(self, model: GeneReconModel) -> None:
        self.model = model
        self.solver_options = model.solver_options
        self.backward_pass_count = 0
        self._old_gmres_stats = None
        self._gmres_stats: list[dict[str, float | int | str]] | None = None

    def __enter__(self) -> "SelfLoopBackwardRecorder":
        if self.solver_options.self_loop_solver in ("gmres", "gmres_fixed"):
            self._old_gmres_stats = wave_backward_module._GMRES_SELF_LOOP_STATS
            self._gmres_stats = []
            wave_backward_module._GMRES_SELF_LOOP_STATS = self._gmres_stats
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._gmres_stats is not None:
            wave_backward_module._GMRES_SELF_LOOP_STATS = self._old_gmres_stats

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_pass_count += 1
        loss.backward()

    def summary(self) -> dict[str, float | int | None | str]:
        solver = self.solver_options.self_loop_solver
        waves_per_backward = self_loop_wave_count(self.model)
        if solver in ("gmres", "gmres_fixed"):
            gmres_stats = self._gmres_stats or []
            per_wave_iterations = [int(row["iterations"]) for row in gmres_stats]
            per_wave_checks = [int(row.get("check_count", 0)) for row in gmres_stats]
            rel_res = [float(row.get("rel_res", 0.0)) for row in gmres_stats]
            arnoldi_backend_counts: dict[str, int] = {}
            for row in gmres_stats:
                backend = str(row.get("arnoldi_backend", "unknown"))
                arnoldi_backend_counts[backend] = arnoldi_backend_counts.get(backend, 0) + 1
            wave_solves = len(per_wave_iterations)
            total_iterations = int(sum(per_wave_iterations))
            return {
                "self_loop_solver": solver,
                "self_loop_backward_pass_count": int(self.backward_pass_count),
                "self_loop_waves_per_backward": int(waves_per_backward),
                "self_loop_wave_solves": int(wave_solves),
                "self_loop_backward_iterations": total_iterations,
                "self_loop_mean_iterations_per_wave": (
                    total_iterations / wave_solves if wave_solves else None
                ),
                "self_loop_max_iterations_per_wave": max(per_wave_iterations, default=0),
                "gmres_total_checks": int(sum(per_wave_checks)),
                "gmres_mean_checks_per_wave": (
                    sum(per_wave_checks) / wave_solves if wave_solves else None
                ),
                "gmres_max_checks_per_wave": max(per_wave_checks, default=0),
                "gmres_max_rel_res": max(rel_res, default=None),
                "gmres_arnoldi_backend_counts": arnoldi_backend_counts,
            }

        wave_solves = int(self.backward_pass_count) * int(waves_per_backward)
        total_iterations = int(wave_solves) * int(self.solver_options.neumann_terms)
        return {
            "self_loop_solver": solver,
            "self_loop_backward_pass_count": int(self.backward_pass_count),
            "self_loop_waves_per_backward": int(waves_per_backward),
            "self_loop_wave_solves": int(wave_solves),
            "self_loop_backward_iterations": int(total_iterations),
            "self_loop_mean_iterations_per_wave": (
                float(self.solver_options.neumann_terms) if wave_solves else None
            ),
            "self_loop_max_iterations_per_wave": int(self.solver_options.neumann_terms),
            "gmres_total_checks": None,
            "gmres_mean_checks_per_wave": None,
            "gmres_max_checks_per_wave": None,
            "gmres_max_rel_res": None,
            "gmres_arnoldi_backend_counts": None,
        }


def fd_diag_hessian_sgd_step(
    model: GeneReconModel,
    loss: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    backward_recorder: SelfLoopBackwardRecorder,
) -> dict[str, float | None]:
    """Take one damped finite-difference diagonal Hessian-SGD-like step.

    This is intentionally local to the benchmark harness. The gpurec kernels do
    not expose an exact Hessian API, so we estimate diagonal curvature by
    comparing projected gradients at theta and at one small simultaneous
    per-coordinate perturbation.
    """
    backward_recorder.backward(loss)
    base_grad = projected_grad(model, args.min_rate, args.max_rate)
    theta0 = model.theta.detach().clone()
    direction = torch.sign(base_grad)
    perturb = args.hessian_eps * direction
    active = perturb != 0

    stats: dict[str, float | None] = {
        "projected_grad_max_abs": finite_float(float(base_grad.abs().max().item())),
        "projected_grad_l2": finite_float(float(torch.linalg.vector_norm(base_grad).item())),
        "hessian_fd_loss": None,
        "hessian_diag_min_abs": None,
        "hessian_diag_median_abs": None,
        "hessian_diag_max_abs": None,
        "update_l2": 0.0,
        "update_max_abs": 0.0,
    }
    if not bool(active.any().item()):
        return stats

    with torch.no_grad():
        model.theta.copy_(theta0 + perturb)
        clamp_log_rate_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
        actual_delta = model.theta.detach() - theta0

    model.theta.grad = None
    fd_loss = model()
    if not torch.isfinite(fd_loss).item():
        with torch.no_grad():
            model.theta.copy_(theta0)
        return stats
    backward_recorder.backward(fd_loss)
    fd_grad = projected_grad(model, args.min_rate, args.max_rate)
    stats["hessian_fd_loss"] = finite_float(float(fd_loss.detach().item()))

    with torch.no_grad():
        model.theta.copy_(theta0)
        valid = actual_delta != 0
        diag = torch.full_like(base_grad, args.hessian_damping)
        diag[valid] = (fd_grad[valid] - base_grad[valid]) / actual_delta[valid]
        denom = diag.abs().clamp_min(args.hessian_damping)
        update = -args.lr * base_grad / denom
        update = update.clamp(min=-args.hessian_max_step, max=args.hessian_max_step)
        model.theta.add_(update)
        clamp_log_rate_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)

    abs_diag = diag[valid].abs()
    if bool(abs_diag.numel()):
        stats["hessian_diag_min_abs"] = finite_float(float(abs_diag.min().item()))
        stats["hessian_diag_median_abs"] = finite_float(float(abs_diag.median().item()))
        stats["hessian_diag_max_abs"] = finite_float(float(abs_diag.max().item()))
    stats["update_l2"] = finite_float(float(torch.linalg.vector_norm(update).item()))
    stats["update_max_abs"] = finite_float(float(update.abs().max().item()))
    model.theta.grad = None
    cuda_sync(device)
    return stats


def dataset_stats(model: GeneReconModel, gene_trees: list[Path]) -> dict[str, Any]:
    clade_counts = [int(family["C"]) for family in model.families]
    split_counts = [int(family["N_splits"]) for family in model.families]
    leaf_counts = [len(family["leaf_row_index"]) for family in model.families]
    species_nodes = int(model.species_helpers["S"])
    leaves_if_binary = (species_nodes + 1) // 2
    return {
        "species_nodes": species_nodes,
        "species_leaves_if_binary": leaves_if_binary,
        "gene_family_count": len(model.families),
        "gene_tree_files": [str(path) for path in gene_trees],
        "total_clades": int(sum(clade_counts)),
        "max_clades_per_family": max(clade_counts),
        "total_splits": int(sum(split_counts)),
        "max_splits_per_family": max(split_counts),
        "total_gene_leaves": int(sum(leaf_counts)),
        "max_gene_leaves_per_family": max(leaf_counts),
        "batch_count": len(model.family_batches),
        "batch_sizes": [len(batch) for batch in model.family_batches],
    }


def convergence_status(
    history: list[dict[str, Any]],
    *,
    window: int,
    rel_loss_tol: float,
    projected_grad_tol: float,
    projected_grad_rel_tol: float,
) -> tuple[bool, dict[str, Any]]:
    if not history:
        return False, {"reason": "no history"}
    last_grad = history[-1].get("projected_grad_max_abs")
    last_loss = history[-1].get("loss")
    grad_ok = last_grad is not None and last_grad <= projected_grad_tol
    rel_grad = None
    rel_grad_ok = False
    if last_grad is not None and last_loss is not None:
        rel_grad = last_grad / max(1.0, abs(float(last_loss)))
        rel_grad_ok = rel_grad <= projected_grad_rel_tol
    if len(history) <= window:
        return False, {
            "reason": "not enough steps",
            "projected_grad_ok": grad_ok,
            "projected_grad_rel_ok": rel_grad_ok,
            "projected_grad_max_abs": last_grad,
            "projected_grad_relative": rel_grad,
        }
    window_losses = [entry["loss"] for entry in history[-window - 1 :]]
    if any(loss is None for loss in window_losses):
        return False, {"reason": "non-finite loss in convergence window"}
    start = float(window_losses[0])
    end = float(window_losses[-1])
    rel_improvement = abs(start - end) / max(1.0, abs(start))
    loss_ok = rel_improvement <= rel_loss_tol
    return loss_ok and (grad_ok or rel_grad_ok), {
        "window": window,
        "relative_loss_change": rel_improvement,
        "relative_loss_tol": rel_loss_tol,
        "loss_ok": loss_ok,
        "projected_grad_max_abs": last_grad,
        "projected_grad_tol": projected_grad_tol,
        "projected_grad_ok": grad_ok,
        "projected_grad_relative": rel_grad,
        "projected_grad_rel_tol": projected_grad_rel_tol,
        "projected_grad_rel_ok": rel_grad_ok,
    }


def main() -> None:
    args = parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    gene_trees = choose_gene_trees(args)
    device = torch.device(args.device)

    run: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "host": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "started_at_unix": time.time(),
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        run["cuda_device_name"] = torch.cuda.get_device_name(device)

    solver_options = solver_options_from_args(args)
    run["solver_options"] = dict(vars(solver_options))

    t0 = time.perf_counter()
    model = GeneReconModel(
        args.species_tree,
        gene_trees,
        mode="genewise",
        device=device,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        batch_packing=args.batch_packing,
        max_wave_size=args.max_wave_size,
        solver_options=solver_options,
    )
    cuda_sync(device)
    run["model_init_seconds"] = time.perf_counter() - t0
    run["dataset"] = dataset_stats(model, gene_trees)
    run["history"] = []
    model.receiver_weights.requires_grad_(False)

    if args.eval_only:
        with torch.no_grad():
            t_eval = time.perf_counter()
            loss = model()
            cuda_sync(device)
        run["eval_seconds"] = time.perf_counter() - t_eval
        run["eval_loss"] = finite_float(float(loss.detach().item()))
        run["converged"] = False
        run["convergence"] = {"reason": "eval only"}
    else:
        optimizer = torch.optim.Adam([model.theta], lr=args.lr) if args.optimizer == "adam" else None
        converged = False
        convergence: dict[str, Any] = {"reason": "not evaluated"}
        train_t0 = time.perf_counter()
        for step in range(1, args.steps + 1):
            step_t0 = time.perf_counter()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.theta.grad = None
            loss = model()
            if not torch.isfinite(loss).item():
                run["history"].append({"step": step, "loss": None, "step_seconds": time.perf_counter() - step_t0})
                convergence = {"reason": "non-finite loss"}
                break
            with SelfLoopBackwardRecorder(model) as backward_recorder:
                if optimizer is None:
                    grad_stats = fd_diag_hessian_sgd_step(model, loss, args, device, backward_recorder)
                else:
                    backward_recorder.backward(loss)
                    grad_stats = projected_grad_stats(model, args.min_rate, args.max_rate)
                    project_rate_gradient_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
                    optimizer.step()
                    clamp_log_rate_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
                    cuda_sync(device)
                self_loop_stats = backward_recorder.summary()
            entry = {
                "step": step,
                "loss": finite_float(float(loss.detach().item())),
                "step_seconds": time.perf_counter() - step_t0,
            }
            entry.update(self_loop_stats)
            if args.optimizer == "adam":
                entry["projected_grad_max_abs"] = grad_stats["max_abs"]
                entry["projected_grad_l2"] = grad_stats["l2"]
            else:
                entry.update(grad_stats)
            run["history"].append(entry)
            converged, convergence = convergence_status(
                run["history"],
                window=args.convergence_window,
                rel_loss_tol=args.rel_loss_tol,
                projected_grad_tol=args.projected_grad_tol,
                projected_grad_rel_tol=args.projected_grad_rel_tol,
            )
            if converged:
                break
        run["train_seconds"] = time.perf_counter() - train_t0
        run["converged"] = converged
        run["convergence"] = convergence
        run["self_loop_backward_iterations"] = int(
            sum(int(entry.get("self_loop_backward_iterations") or 0) for entry in run["history"])
        )
        run["self_loop_backward_pass_count"] = int(
            sum(int(entry.get("self_loop_backward_pass_count") or 0) for entry in run["history"])
        )

    run["finished_at_unix"] = time.time()
    run["wall_seconds"] = run["finished_at_unix"] - run["started_at_unix"]
    run["cuda_memory"] = cuda_memory(device)
    output.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset_summary = dict(run["dataset"])
    dataset_summary.pop("gene_tree_files", None)
    print(json.dumps({
        "output": str(output),
        "dataset": dataset_summary,
        "converged": run["converged"],
        "convergence": run["convergence"],
        "wall_seconds": run["wall_seconds"],
        "cuda_memory": run["cuda_memory"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

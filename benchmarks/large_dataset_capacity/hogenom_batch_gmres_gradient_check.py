#!/usr/bin/env python3
"""Compare GMRES gradients with a high-Neumann reference on a HOGENOM batch."""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.large_dataset_capacity.hogenom_gmres_neumann_family_experiment import (
    DEFAULT_SPECIES_TREE,
)
from benchmarks.large_dataset_capacity.run_gpurec_benchmark import (
    SelfLoopBackwardRecorder,
    choose_gene_trees,
    dataset_stats,
)
from gpurec import GeneReconModel, SolverOptions


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENE_TREE_DIR = REPO_ROOT / "benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core/hogenom/families"


def _csv_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _csv_floats(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-tree", type=Path, default=DEFAULT_SPECIES_TREE)
    parser.add_argument("--gene-tree-dir", type=Path, default=DEFAULT_GENE_TREE_DIR)
    parser.add_argument("--pattern", default="ufboot1000.MFP.geneTree.newick")
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--select", choices=("sorted", "largest", "random"), default="largest")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("global", "specieswise", "genewise"), default="genewise")
    parser.add_argument("--family-chunk-size", type=int, default=0)
    parser.add_argument("--clade-budget", type=int, default=500_000)
    parser.add_argument("--batch-packing", default="depth_first_fit")
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--e-iters", type=int, default=256)
    parser.add_argument("--e-tol", type=float, default=1e-8)
    parser.add_argument("--pi-iters", type=int, default=256)
    parser.add_argument("--reference-neumann", type=int, default=512)
    parser.add_argument("--neumann-terms", type=_csv_ints, default=_csv_ints("32"))
    parser.add_argument("--gmres-solver", choices=("gmres", "gmres_fixed"), default="gmres")
    parser.add_argument("--gmres-iters", type=int, default=10)
    parser.add_argument("--gmres-tols", type=_csv_floats, default=_csv_floats("1e-8,1e-7,1e-6"))
    parser.add_argument("--gmres-check-interval", type=int, default=4)
    parser.add_argument("--bicgstab-max-iter", type=int, default=1000)
    parser.add_argument("--bicgstab-tol", type=float, default=1e-8)
    parser.add_argument("--use-adjoint-pruning", action="store_true")
    parser.add_argument("--adjoint-pruning-threshold", type=float, default=1e-6)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _family_name(path: Path) -> str:
    for parent in path.parents:
        if parent.parent.name == "families":
            return parent.name
    return path.parent.name


def _solver_options(args: argparse.Namespace, *, solver: str, iterations: int, gmres_tol: float) -> SolverOptions:
    return SolverOptions(
        e_max_iter=int(args.e_iters),
        e_tol=float(args.e_tol),
        pi_iters=int(args.pi_iters),
        neumann_terms=int(iterations),
        self_loop_solver=solver,
        gmres_tol=float(gmres_tol),
        gmres_check_interval=int(args.gmres_check_interval),
        bicgstab_max_iter=int(args.bicgstab_max_iter),
        bicgstab_tol=float(args.bicgstab_tol),
        adjoint_pruning_threshold=float(args.adjoint_pruning_threshold) if args.use_adjoint_pruning else 0.0,
        use_adjoint_pruning=bool(args.use_adjoint_pruning),
        pibar_side_threshold=0.0,
    )


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_memory(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _run_gradient(
    model: GeneReconModel,
    args: argparse.Namespace,
    *,
    solver: str,
    iterations: int,
    gmres_tol: float,
) -> dict[str, Any]:
    device = model.theta.device
    model.solver_options = _solver_options(args, solver=solver, iterations=iterations, gmres_tol=gmres_tol)
    model.clear_warm_starts()
    model.theta.grad = None
    model.receiver_weights.grad = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    loss = model()
    with SelfLoopBackwardRecorder(model) as recorder:
        recorder.backward(loss)
        _cuda_sync(device)
        self_loop = recorder.summary()
    elapsed_s = time.perf_counter() - start

    grad = model.theta.grad.detach().cpu().double().clone()
    return {
        "solver": solver,
        "iterations": int(iterations),
        "gmres_tol": None if solver == "neumann" else float(gmres_tol),
        "gmres_check_interval": None if solver == "neumann" else int(args.gmres_check_interval),
        "loss": float(loss.detach().cpu()),
        "elapsed_s": elapsed_s,
        "gradient": grad.reshape(-1).tolist(),
        "gradient_shape": list(grad.shape),
        "self_loop": self_loop,
        "cuda_memory": _cuda_memory(device),
    }


def _annotate_errors(row: dict[str, Any], reference_gradient: torch.Tensor) -> None:
    if "gradient" not in row:
        return
    grad = torch.tensor(row["gradient"], dtype=torch.float64).reshape_as(reference_gradient)
    delta = grad - reference_gradient
    ref_norm = max(float(torch.linalg.vector_norm(reference_gradient)), 1e-30)
    ref_inf = max(float(reference_gradient.abs().max()), 1e-30)
    row["rel_l2_error"] = float(torch.linalg.vector_norm(delta) / ref_norm)
    row["rel_inf_error"] = float(delta.abs().max() / ref_inf)
    row["abs_l2_delta"] = float(torch.linalg.vector_norm(delta))

    if grad.ndim == 2:
        family_errors = []
        for idx in range(int(grad.shape[0])):
            ref_i = reference_gradient[idx]
            delta_i = delta[idx]
            ref_i_norm = max(float(torch.linalg.vector_norm(ref_i)), 1e-30)
            ref_i_inf = max(float(ref_i.abs().max()), 1e-30)
            family_errors.append(
                {
                    "family_index": idx,
                    "rel_l2_error": float(torch.linalg.vector_norm(delta_i) / ref_i_norm),
                    "rel_inf_error": float(delta_i.abs().max() / ref_i_inf),
                    "abs_l2_delta": float(torch.linalg.vector_norm(delta_i)),
                }
            )
        row["family_errors"] = family_errors
        row["max_family_rel_l2_error"] = max((item["rel_l2_error"] for item in family_errors), default=math.nan)
        row["max_family_rel_inf_error"] = max((item["rel_inf_error"] for item in family_errors), default=math.nan)


def _candidate_error(
    *,
    solver: str,
    iterations: int,
    gmres_tol: float | None,
    args: argparse.Namespace,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "solver": solver,
        "iterations": int(iterations),
        "gmres_tol": gmres_tol,
        "gmres_check_interval": None if solver == "neumann" else int(args.gmres_check_interval),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def main() -> None:
    args = parse_args()
    output = args.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    gene_trees = choose_gene_trees(args)

    model = GeneReconModel(
        args.species_tree,
        gene_trees,
        mode=args.mode,
        device=device,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        batch_packing=args.batch_packing,
        max_wave_size=args.max_wave_size,
        solver_options=_solver_options(args, solver="neumann", iterations=args.reference_neumann, gmres_tol=1.0),
    )
    model.receiver_weights.requires_grad_(False)

    reference = _run_gradient(
        model,
        args,
        solver="neumann",
        iterations=args.reference_neumann,
        gmres_tol=1.0,
    )
    reference_gradient = torch.tensor(reference["gradient"], dtype=torch.float64).reshape(reference["gradient_shape"])

    rows: list[dict[str, Any]] = []
    for iterations in args.neumann_terms:
        try:
            row = _run_gradient(model, args, solver="neumann", iterations=iterations, gmres_tol=1.0)
        except Exception as exc:  # noqa: BLE001 - benchmark output should preserve failures.
            row = _candidate_error(
                solver="neumann",
                iterations=iterations,
                gmres_tol=None,
                args=args,
                exc=exc,
            )
        _annotate_errors(row, reference_gradient)
        rows.append(row)
    for tol in args.gmres_tols:
        try:
            row = _run_gradient(model, args, solver=args.gmres_solver, iterations=args.gmres_iters, gmres_tol=tol)
        except Exception as exc:  # noqa: BLE001 - benchmark output should preserve failures.
            row = _candidate_error(
                solver=args.gmres_solver,
                iterations=args.gmres_iters,
                gmres_tol=tol,
                args=args,
                exc=exc,
            )
        _annotate_errors(row, reference_gradient)
        rows.append(row)

    result = {
        "host": socket.gethostname(),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "family_names": [_family_name(path) for path in gene_trees],
        "gene_tree_files": [str(path) for path in gene_trees],
        "batch_family_indices": [list(static.family_indices) for static in model.batch_statics],
        "dataset": dataset_stats(model, gene_trees),
        "reference": reference,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"output\t{output}")
    print("solver\titerations\tgmres_tol\tself_loop_iterations\tgmres_checks\trel_l2\trel_inf\tmax_family_l2\tmax_family_inf\tloss\telapsed_s")
    for row in rows:
        if "error" in row:
            print(
                f"{row['solver']}\t{row['iterations']}\t{row['gmres_tol']}\t"
                f"ERROR\tERROR\tERROR\tERROR\tERROR\tERROR\t{row['error_type']}: {row['error']}\t"
            )
            continue
        self_loop = row["self_loop"]
        print(
            f"{row['solver']}\t{row['iterations']}\t{row['gmres_tol']}\t"
            f"{self_loop['self_loop_backward_iterations']}\t{self_loop['gmres_total_checks']}\t"
            f"{row['rel_l2_error']:.6e}\t{row['rel_inf_error']:.6e}\t"
            f"{row.get('max_family_rel_l2_error', math.nan):.6e}\t"
            f"{row.get('max_family_rel_inf_error', math.nan):.6e}\t"
            f"{row['loss']:.6f}\t{row['elapsed_s']:.3f}"
        )


if __name__ == "__main__":
    main()

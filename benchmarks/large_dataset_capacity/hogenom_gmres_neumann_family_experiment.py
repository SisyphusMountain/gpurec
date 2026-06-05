#!/usr/bin/env python3
"""Compare Neumann and GMRES wave-adjoint solves on one HOGENOM family."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.api._execution import evaluate_static_loss_grad
from gpurec.core.kernels import wave_backward as wave_backward_module


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPECIES_TREE = REPO_ROOT / (
    "benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core/hogenom/"
    "runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_"
    "--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick"
)
DEFAULT_FAMILIES_FILE = (
    REPO_ROOT / "benchmarks/large_dataset_capacity/generated/alerax_hogenom_core_all_families.txt"
)
DEFAULT_CHECKPOINT = REPO_ROOT / (
    "benchmarks/large_dataset_capacity/output/"
    "full_hogenom_genewise_end2end_20260605_scheduled_golden_v6_lr/runs/"
    "full_hogenom_genewise_end2end_golden_float64_hsgd_schedule_fixed256/"
    "checkpoints/latest.pt"
)
DEFAULT_FAMILY_INDEX = 2461
DEFAULT_FAMILY_NAME = "CLU_000680_20_4_C"


def _csv_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-tree", type=Path, default=DEFAULT_SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=DEFAULT_FAMILIES_FILE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--family-index", type=int, default=DEFAULT_FAMILY_INDEX)
    parser.add_argument("--family-name", default=DEFAULT_FAMILY_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--e-iters", type=int, default=256)
    parser.add_argument("--pi-iters", type=int, default=256)
    parser.add_argument("--reference-neumann", type=int, default=512)
    parser.add_argument("--neumann-terms", type=_csv_ints, default=_csv_ints("8,16,32,64"))
    parser.add_argument("--gmres-iters", type=_csv_ints, default=_csv_ints("2,4,8,16,32"))
    parser.add_argument("--gmres-tol", type=float, default=1e-10)
    parser.add_argument("--gmres-check-interval", type=int, default=1)
    parser.add_argument(
        "--gmres-solver",
        choices=("gmres", "gmres_fixed"),
        default="gmres",
    )
    parser.add_argument("--clade-budget", type=int, default=250_000)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_family_entries(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            current = {"name": line[2:].strip()}
            entries.append(current)
        elif line.startswith("starting_gene_tree") and current is not None:
            current["tree"] = line.split("=", 1)[1].strip()
    return entries


def family_tree_path(families_file: Path, family_name: str) -> Path:
    for entry in load_family_entries(families_file):
        if entry.get("name") == family_name:
            tree = Path(entry["tree"])
            if not tree.is_absolute():
                tree = (families_file.parent / tree).resolve()
            return tree
    raise ValueError(f"family {family_name!r} not found in {families_file}")


def solver_options(args: argparse.Namespace, neumann_terms: int) -> SolverOptions:
    return SolverOptions(
        e_max_iter=args.e_iters,
        e_tol=1e-10,
        pi_iters=args.pi_iters,
        neumann_terms=neumann_terms,
        bicgstab_max_iter=1000,
        bicgstab_tol=1e-10,
        adjoint_pruning_threshold=0.0,
        use_adjoint_pruning=False,
        pibar_side_threshold=0.0,
    )


def build_model(args: argparse.Namespace, tree_path: Path) -> GeneReconModel:
    return GeneReconModel(
        args.species_tree,
        [tree_path],
        mode="global",
        device=args.device,
        family_chunk_size=1,
        clade_budget=args.clade_budget,
        max_wave_size=args.max_wave_size,
        solver_options=solver_options(args, args.reference_neumann),
    )


def run_gradient(
    model: GeneReconModel,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    solver: str,
    iterations: int,
    gmres_tol: float,
    gmres_check_interval: int,
) -> dict[str, Any]:
    model.configure_solver(
        neumann_terms=iterations,
        self_loop_solver=solver,
        gmres_tol=gmres_tol,
        gmres_check_interval=gmres_check_interval,
    )
    model.clear_warm_starts()
    if theta.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(theta.device)

    wave_count = len(model.batch_statics[0].wave_layout["wave_metas"])
    original_gmres_stats = wave_backward_module._GMRES_SELF_LOOP_STATS
    gmres_stats: list[dict[str, float | int]] | None = None

    if solver in ("gmres", "gmres_fixed"):
        gmres_stats = []
        wave_backward_module._GMRES_SELF_LOOP_STATS = gmres_stats
    elif solver != "neumann":
        raise ValueError(f"unsupported solver {solver!r}")

    try:
        start = time.perf_counter()
        loss, grad_theta, _ = evaluate_static_loss_grad(
            model.batch_statics[0],
            theta,
            receiver_weights,
            need_grad=True,
        )
        if theta.device.type == "cuda":
            torch.cuda.synchronize(theta.device)
        elapsed = time.perf_counter() - start
    finally:
        wave_backward_module._GMRES_SELF_LOOP_STATS = original_gmres_stats

    peak_gb = None
    if theta.device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(theta.device) / 1024**3

    if gmres_stats is None:
        total_backward_iterations = int(iterations) * wave_count
        per_wave_iterations = None
    else:
        per_wave_iterations = [int(item["iterations"]) for item in gmres_stats]
        total_backward_iterations = int(sum(per_wave_iterations))
        per_wave_checks = [int(item.get("check_count", 0)) for item in gmres_stats]

    return {
        "solver": solver,
        "iterations": int(iterations),
        "wave_count": wave_count,
        "total_backward_iterations": total_backward_iterations,
        "per_wave_iterations": per_wave_iterations,
        "total_gmres_checks": None if gmres_stats is None else int(sum(per_wave_checks)),
        "per_wave_gmres_checks": None if gmres_stats is None else per_wave_checks,
        "max_gmres_checks": None if gmres_stats is None else max(per_wave_checks, default=0),
        "mean_gmres_checks": (
            None
            if gmres_stats is None
            else sum(per_wave_checks) / max(1, len(per_wave_checks))
        ),
        "max_wave_iterations": None if per_wave_iterations is None else max(per_wave_iterations, default=0),
        "mean_wave_iterations": (
            None
            if per_wave_iterations is None
            else total_backward_iterations / max(1, len(per_wave_iterations))
        ),
        "loss": float(loss.detach().cpu()),
        "gradient": grad_theta.detach().cpu().double().reshape(-1).tolist(),
        "elapsed_s": elapsed,
        "peak_gb": peak_gb,
    }


def annotate_errors(row: dict[str, Any], reference_gradient: torch.Tensor) -> dict[str, Any]:
    grad = torch.tensor(row["gradient"], dtype=torch.float64)
    delta = grad - reference_gradient
    ref_norm = max(float(torch.linalg.vector_norm(reference_gradient)), 1e-30)
    ref_inf = max(float(reference_gradient.abs().max()), 1e-30)
    row["rel_l2_error"] = float(torch.linalg.vector_norm(delta) / ref_norm)
    row["rel_inf_error"] = float(delta.abs().max() / ref_inf)
    row["abs_l2_delta"] = float(torch.linalg.vector_norm(delta))
    return row


def main() -> None:
    args = parse_args()
    tree_path = family_tree_path(args.families_file, args.family_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    theta_row = checkpoint["theta"][args.family_index].detach().double()

    model = build_model(args, tree_path)
    theta = theta_row.to(device=args.device).contiguous()
    receiver_weights = torch.zeros(
        (int(model.species_helpers["S"]),),
        dtype=theta.dtype,
        device=theta.device,
    )
    family = model.families[0]

    reference = run_gradient(
        model,
        theta,
        receiver_weights,
        solver="neumann",
        iterations=args.reference_neumann,
        gmres_tol=args.gmres_tol,
        gmres_check_interval=args.gmres_check_interval,
    )
    reference_gradient = torch.tensor(reference["gradient"], dtype=torch.float64)

    rows: list[dict[str, Any]] = []
    for iterations in args.neumann_terms:
        rows.append(
            annotate_errors(
                run_gradient(
                    model,
                    theta,
                    receiver_weights,
                    solver="neumann",
                    iterations=iterations,
                    gmres_tol=args.gmres_tol,
                    gmres_check_interval=args.gmres_check_interval,
                ),
                reference_gradient,
            )
        )
    for iterations in args.gmres_iters:
        rows.append(
            annotate_errors(
                run_gradient(
                    model,
                    theta,
                    receiver_weights,
                    solver=args.gmres_solver,
                    iterations=iterations,
                    gmres_tol=args.gmres_tol,
                    gmres_check_interval=args.gmres_check_interval,
                ),
                reference_gradient,
            )
        )

    result = {
        "family_index": args.family_index,
        "family_name": args.family_name,
        "family_tree": str(tree_path),
        "checkpoint": str(args.checkpoint),
        "theta": theta_row.tolist(),
        "rates": torch.exp2(theta_row).tolist(),
        "family_stats": {
            "clades": int(family["C"]),
            "splits": int(family["N_splits"]),
            "leaves": len(family["leaf_row_index"]),
            "wave_count": len(model.batch_statics[0].wave_layout["wave_metas"]),
            "total_wave_rows": sum(
                int(meta["W"]) for meta in model.batch_statics[0].wave_layout["wave_metas"]
            ),
        },
        "reference": reference,
        "rows": rows,
    }

    print(
        json.dumps(
            {
                "family": result["family_name"],
                "family_index": result["family_index"],
                "rates": result["rates"],
                "family_stats": result["family_stats"],
                "reference_gradient": reference["gradient"],
            },
            indent=2,
        )
    )
    print(
        "solver\titerations\ttotal_backward_iterations\tmean_wave_iterations\t"
        "total_gmres_checks\tmean_gmres_checks\t"
        "rel_l2_error\trel_inf_error\tgradient\telapsed_s\tpeak_gb"
    )
    for row in rows:
        peak = "" if row["peak_gb"] is None else f"{row['peak_gb']:.3f}"
        mean_wave_iterations = (
            ""
            if row["mean_wave_iterations"] is None
            else f"{row['mean_wave_iterations']:.3f}"
        )
        total_gmres_checks = "" if row["total_gmres_checks"] is None else str(row["total_gmres_checks"])
        mean_gmres_checks = "" if row["mean_gmres_checks"] is None else f"{row['mean_gmres_checks']:.3f}"
        print(
            f"{row['solver']}\t{row['iterations']}\t{row['total_backward_iterations']}\t"
            f"{mean_wave_iterations}\t{total_gmres_checks}\t{mean_gmres_checks}\t"
            f"{row['rel_l2_error']:.6e}\t"
            f"{row['rel_inf_error']:.6e}\t{row['gradient']}\t"
            f"{row['elapsed_s']:.3f}\t{peak}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")

    del model
    if theta.device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

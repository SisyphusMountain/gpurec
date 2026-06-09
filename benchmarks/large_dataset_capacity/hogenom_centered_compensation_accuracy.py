#!/usr/bin/env python3
"""Compare centered Pi compensation dtypes on selected HOGENOM families."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.large_dataset_capacity.hogenom_gmres_neumann_family_experiment import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FAMILIES_FILE,
    DEFAULT_SPECIES_TREE,
    family_tree_path,
)
from gpurec import GeneReconModel, SolverOptions
from gpurec.api._execution import evaluate_static_loss_grad


DEFAULT_OUTPUT_JSON = REPO_ROOT / (
    "benchmarks/large_dataset_capacity/output/"
    "hogenom_centered_production_forward_20260607/centered_compensation_accuracy.json"
)


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
    parser.add_argument("--family-indices", type=_csv_ints, default=_csv_ints("799,8173"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--e-iters", type=int, default=256)
    parser.add_argument("--pi-iters", type=int, default=256)
    parser.add_argument("--neumann-terms", type=int, default=8)
    parser.add_argument("--clade-budget", type=int, default=250_000)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def solver_options(args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(
        e_max_iter=int(args.e_iters),
        e_tol=1e-10,
        pi_iters=int(args.pi_iters),
        neumann_terms=int(args.neumann_terms),
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
        solver_options=solver_options(args),
    )


@contextmanager
def temporary_env(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_variant(
    args: argparse.Namespace,
    *,
    tree_path: Path,
    theta_row: torch.Tensor,
    dtype: torch.dtype,
    centered: bool,
    compensation_dtype: str | None,
    variant: str,
    extra_env: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    env = {
        "GPUREC_CENTERED_PI_FORWARD": "1" if centered else None,
        "GPUREC_CENTERED_PI_COMPENSATION_DTYPE": compensation_dtype if centered else None,
    }
    if extra_env:
        env.update(extra_env)
    with temporary_env(env):
        model = build_model(args, tree_path)
        theta = theta_row.to(device=args.device, dtype=dtype).contiguous()
        receiver_weights = torch.zeros(
            (int(model.species_helpers["S"]),),
            dtype=dtype,
            device=theta.device,
        )
        if theta.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(theta.device)
        start = time.perf_counter()
        loss, grad, _ = evaluate_static_loss_grad(
            model.batch_statics[0],
            theta,
            receiver_weights,
            need_grad=True,
        )
        cuda_sync(theta.device)
        elapsed_s = time.perf_counter() - start
        peak_gb = None
        if theta.device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(theta.device) / 1024**3
        row = {
            "variant": variant,
            "dtype": str(dtype).replace("torch.", ""),
            "centered": bool(centered),
            "compensation_dtype": compensation_dtype,
            "extra_env": extra_env or {},
            "loss": float(loss.detach().cpu()),
            "gradient": [float(x) for x in grad.detach().cpu().double().reshape(-1).tolist()],
            "elapsed_s": elapsed_s,
            "peak_gb": peak_gb,
        }
        del model, theta, receiver_weights, loss, grad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return row


def add_errors(row: dict[str, Any], reference: dict[str, Any]) -> None:
    grad = torch.tensor(row["gradient"], dtype=torch.float64)
    ref = torch.tensor(reference["gradient"], dtype=torch.float64)
    diff = grad - ref
    row["grad_rel_l2_vs_fp64"] = float(
        torch.linalg.vector_norm(diff) / torch.clamp(torch.linalg.vector_norm(ref), min=1e-30)
    )
    row["grad_abs_l2_vs_fp64"] = float(torch.linalg.vector_norm(diff))
    row["loss_abs_delta_vs_fp64"] = abs(float(row["loss"]) - float(reference["loss"]))


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    family_names = list(checkpoint["family_names"])
    theta_cpu = checkpoint["theta"].detach().cpu().double()

    rows: list[dict[str, Any]] = []
    for family_index in args.family_indices:
        family_name = family_names[family_index]
        tree_path = family_tree_path(args.families_file, family_name)
        theta_row = theta_cpu[family_index]
        family_rows: list[dict[str, Any]] = []
        reference = run_variant(
            args,
            tree_path=tree_path,
            theta_row=theta_row,
            dtype=torch.float64,
            centered=False,
            compensation_dtype=None,
            variant="current_abs_fp64_reference",
        )
        family_rows.append(reference)
        variants = [
            ("current_abs_fp32", torch.float32, False, None, None),
            ("centered_comp_fp64", torch.float32, True, "fp64", None),
            ("centered_comp_fp32_integer", torch.float32, True, "fp32", None),
            (
                "centered_comp_fp32_integer_param_store",
                torch.float32,
                True,
                "fp32",
                {"GPUREC_SELF_LOOP_PARAM_STORE_ACCUM": "1"},
            ),
            (
                "centered_comp_fp32_integer_direct_source_theta",
                torch.float32,
                True,
                "fp32",
                {"GPUREC_DIRECT_THETA_SOURCE_KERNEL": "1"},
            ),
        ]
        for variant, dtype, centered, compensation_dtype, extra_env in variants:
            row = run_variant(
                args,
                tree_path=tree_path,
                theta_row=theta_row,
                dtype=dtype,
                centered=centered,
                compensation_dtype=compensation_dtype,
                variant=variant,
                extra_env=extra_env,
            )
            add_errors(row, reference)
            family_rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "variant_done",
                        "family_index": family_index,
                        "family_name": family_name,
                        "variant": variant,
                        "grad_rel_l2_vs_fp64": row["grad_rel_l2_vs_fp64"],
                        "loss_abs_delta_vs_fp64": row["loss_abs_delta_vs_fp64"],
                        "elapsed_s": row["elapsed_s"],
                    }
                ),
                flush=True,
            )
        rows.append(
            {
                "family_index": family_index,
                "family_name": family_name,
                "family_tree": str(tree_path),
                "theta": [float(x) for x in theta_row.tolist()],
                "variants": family_rows,
            }
        )

    result = {
        "benchmark": "hogenom_centered_compensation_accuracy",
        "checkpoint": str(args.checkpoint),
        "species_tree": str(args.species_tree),
        "families_file": str(args.families_file),
        "family_indices": args.family_indices,
        "e_iters": int(args.e_iters),
        "pi_iters": int(args.pi_iters),
        "neumann_terms": int(args.neumann_terms),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "done", "output_json": str(args.output_json)}), flush=True)


if __name__ == "__main__":
    main()

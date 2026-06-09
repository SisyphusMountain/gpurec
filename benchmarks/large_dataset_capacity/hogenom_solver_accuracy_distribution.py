#!/usr/bin/env python3
"""All-family HOGENOM gradient-error distributions against an fp64 reference."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.large_dataset_capacity.hogenom_neumann_gradient_screen import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    load_family_entries,
    resolve_tree_path,
)
from gpurec import GeneReconModel, SolverOptions  # noqa: E402
from gpurec.api._execution import evaluate_static_loss_grad, theta_for_static  # noqa: E402


DEFAULT_OUTPUT_JSON = REPO_ROOT / (
    "benchmarks/large_dataset_capacity/output/"
    "hogenom_solver_accuracy_distribution_20260608/distribution.json"
)
DEFAULT_OUTPUT_TSV = DEFAULT_OUTPUT_JSON.with_suffix(".tsv")

CONTROLLED_ENV_KEYS = (
    "GPUREC_CENTERED_PI_FORWARD",
    "GPUREC_CENTERED_PI_COMPENSATION_DTYPE",
    "GPUREC_CENTERED_PI_FUSED_RECENTER",
    "GPUREC_CENTERED_PI_INTEGER_COMPENSATION",
    "GPUREC_CENTERED_PI_ACCELERATED_SELF_LOOP",
    "GPUREC_CENTERED_PI_ALLOW_FP64",
    "GPUREC_CENTERED_BACKWARD_COMPENSATED_EXP_ARGS",
    "GPUREC_DIRECT_THETA_VJP",
    "GPUREC_DIRECT_THETA_SOURCE_KERNEL",
    "GPUREC_DIRECT_THETA_SOURCE_PARTIALS",
    "GPUREC_DIRECT_THETA_SOURCE_PARTIALS_FP64_REDUCE",
    "GPUREC_PARAM_HIGH_LOW",
    "GPUREC_PARAM_HIGH_LOW_FORWARD",
    "GPUREC_PARAM_HIGH_LOW_E_SOLVE",
    "GPUREC_PARAM_HIGH_LOW_REPLACE_FORWARD_HIGH",
)


@dataclass(frozen=True)
class GradientCase:
    label: str
    solver: str
    iterations: int
    dtype_name: str
    env: dict[str, str | None]


def _csv_ints(value: str) -> list[int]:
    if str(value).strip().lower() in {"", "none", "null", "-"}:
        return []
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if any(item < 0 for item in out):
        raise argparse.ArgumentTypeError("iteration counts must be non-negative")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--species-tree", type=Path, default=None)
    parser.add_argument("--families-file", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--family-chunk-size", type=int, default=0)
    parser.add_argument("--clade-budget", type=int, default=100_000)
    parser.add_argument("--batch-packing", default="depth_first_fit")
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--e-iters", type=int, default=256)
    parser.add_argument("--pi-iters", type=int, default=256)
    parser.add_argument("--reference-neumann", type=int, default=512)
    parser.add_argument("--neumann-terms", type=_csv_ints, default=_csv_ints("16,32,64,128"))
    parser.add_argument("--gmres-terms", type=_csv_ints, default=_csv_ints("12,20,32"))
    parser.add_argument("--centered-neumann-terms", type=_csv_ints, default=_csv_ints("32,64,128"))
    parser.add_argument("--centered-gmres-terms", type=_csv_ints, default=_csv_ints(""))
    parser.add_argument("--fp64-neumann-terms", type=_csv_ints, default=_csv_ints(""))
    parser.add_argument("--fp64-gmres-terms", type=_csv_ints, default=_csv_ints(""))
    parser.add_argument("--centered-fp64-neumann-terms", type=_csv_ints, default=_csv_ints(""))
    parser.add_argument("--centered-fp64-gmres-terms", type=_csv_ints, default=_csv_ints(""))
    parser.add_argument("--gmres-tol", type=float, default=1e-10)
    parser.add_argument("--gmres-check-interval", type=int, default=1)
    parser.add_argument("--gmres-preconditioner", choices=("none", "diagonal"), default="none")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--family-indices", type=_csv_ints, default=None)
    parser.add_argument("--family-indices-file", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-tsv", type=Path, default=DEFAULT_OUTPUT_TSV)
    parser.add_argument(
        "--reference-gradient-pt",
        type=Path,
        default=None,
        help="Path for saving/loading the fp64 reference gradient tensor.",
    )
    parser.add_argument(
        "--reuse-reference-gradient",
        action="store_true",
        help="Load --reference-gradient-pt instead of recomputing the fp64 reference.",
    )
    parser.add_argument(
        "--save-case-gradients",
        action="store_true",
        help="Save one tensor per case next to the reference tensor.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if str(name) == "float64":
        return torch.float64
    if str(name) == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def temporary_env(env: dict[str, str | None]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in CONTROLLED_ENV_KEYS}
    try:
        for key in CONTROLLED_ENV_KEYS:
            value = env.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def solver_options(args: argparse.Namespace, *, solver: str, iterations: int) -> SolverOptions:
    return SolverOptions(
        e_max_iter=int(args.e_iters),
        e_tol=1e-10,
        pi_iters=int(args.pi_iters),
        neumann_terms=int(iterations),
        self_loop_solver=str(solver),
        gmres_tol=float(args.gmres_tol),
        gmres_check_interval=int(args.gmres_check_interval),
        gmres_preconditioner=str(args.gmres_preconditioner),
        bicgstab_max_iter=1000,
        bicgstab_tol=1e-10,
        adjoint_pruning_threshold=0.0,
        use_adjoint_pruning=False,
        pibar_side_threshold=0.0,
    )


def build_model(
    args: argparse.Namespace,
    *,
    species_tree: Path,
    gene_trees: list[Path],
    device: torch.device,
    dtype: torch.dtype,
    solver: str,
    iterations: int,
) -> GeneReconModel:
    model = GeneReconModel(
        species_tree,
        gene_trees,
        mode="genewise",
        device=device,
        family_chunk_size=int(args.family_chunk_size),
        clade_budget=int(args.clade_budget),
        batch_packing=str(args.batch_packing),
        max_wave_size=int(args.max_wave_size),
        solver_options=solver_options(args, solver=solver, iterations=iterations),
    )
    model.to(dtype=dtype)
    model.receiver_weights.requires_grad_(False)
    return model


def run_batched_gradient(
    model: GeneReconModel,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    case: GradientCase,
    progress_every: int,
) -> dict[str, Any]:
    model.configure_solver(neumann_terms=int(case.iterations), self_loop_solver=str(case.solver))
    model.clear_warm_starts()
    device = theta.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    grad_total = torch.zeros_like(theta)
    loss_total = torch.zeros((), dtype=theta.dtype, device=device)
    batch_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with temporary_env(case.env):
        for batch_index, static in enumerate(model.batch_statics):
            batch_start = time.perf_counter()
            theta_batch = theta_for_static(static, theta, genewise=model.genewise)
            loss_i, grad_i, grad_receiver_i = evaluate_static_loss_grad(
                static,
                theta_batch,
                receiver_weights,
                need_grad=True,
            )
            if grad_i is None or grad_receiver_i is None:
                raise RuntimeError("missing gradient from static evaluation")
            loss_total = loss_total + loss_i.to(device=device, dtype=theta.dtype)
            grad_total.index_add_(
                0,
                static.family_index_tensor,
                grad_i.to(device=device, dtype=theta.dtype),
            )
            static.warm_E = None
            del theta_batch, loss_i, grad_i, grad_receiver_i
            if device.type == "cuda":
                torch.cuda.empty_cache()
            cuda_sync(device)
            row = {
                "batch_index": batch_index,
                "families": int(static.family_index_tensor.numel()),
                "first_family_index": int(static.family_index_tensor[0].detach().cpu()),
                "last_family_index": int(static.family_index_tensor[-1].detach().cpu()),
                "elapsed_s": time.perf_counter() - batch_start,
            }
            batch_rows.append(row)
            if progress_every > 0 and (
                batch_index == 0
                or (batch_index + 1) % progress_every == 0
                or batch_index + 1 == len(model.batch_statics)
            ):
                print(
                    json.dumps(
                        {
                            "event": "batch_done",
                            "label": case.label,
                            "solver": case.solver,
                            "iterations": int(case.iterations),
                            "batch": batch_index + 1,
                            "batches": len(model.batch_statics),
                            **row,
                        }
                    ),
                    flush=True,
                )

    cuda_sync(device)
    elapsed = time.perf_counter() - start
    peak_gb = None
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    return {
        "label": case.label,
        "solver": case.solver,
        "iterations": int(case.iterations),
        "dtype": case.dtype_name,
        "env": case.env,
        "loss": float(loss_total.detach().cpu()),
        "gradient": grad_total.detach().cpu().double(),
        "elapsed_s": elapsed,
        "peak_gb": peak_gb,
        "batch_rows": batch_rows,
    }


def family_stats(model: GeneReconModel, index: int) -> dict[str, int]:
    family = model.families[index]
    return {
        "clades": int(family["C"]),
        "splits": int(family["N_splits"]),
        "leaves": int(len(family["leaf_row_index"])),
    }


def quantile_summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().cpu().double().reshape(-1)
    finite = values[torch.isfinite(values)]
    out: dict[str, Any] = {
        "count": int(values.numel()),
        "finite_count": int(finite.numel()),
    }
    if int(finite.numel()) == 0:
        return out
    probs = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0], dtype=torch.float64)
    quantiles = torch.quantile(finite, probs)
    names = ("min", "p25", "p50", "p75", "p90", "p95", "p99", "p999", "max")
    out.update({name: float(value) for name, value in zip(names, quantiles.tolist())})
    out["mean"] = float(finite.mean())
    out["threshold_counts"] = {
        str(threshold): int((finite <= threshold).sum())
        for threshold in (1e-2, 1e-3, 1e-4, 1e-5)
    }
    out["threshold_fractions"] = {
        str(threshold): float((finite <= threshold).double().mean())
        for threshold in (1e-2, 1e-3, 1e-4, 1e-5)
    }
    return out


def compare_case(
    *,
    names: list[str],
    original_indices: list[int],
    theta_cpu: torch.Tensor,
    reference_gradient: torch.Tensor,
    run: dict[str, Any],
    model: GeneReconModel,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gradient = run["gradient"]
    delta = gradient - reference_gradient
    ref_norm = torch.linalg.vector_norm(reference_gradient, dim=1).clamp_min(1e-12)
    grad_norm = torch.linalg.vector_norm(gradient, dim=1)
    abs_l2 = torch.linalg.vector_norm(delta, dim=1)
    abs_inf = delta.abs().amax(dim=1)
    ref_inf = reference_gradient.abs().amax(dim=1).clamp_min(1e-12)
    rel_l2 = abs_l2 / ref_norm
    rel_inf = abs_inf / ref_inf
    rates = torch.exp2(theta_cpu.cpu().double())

    rows: list[dict[str, Any]] = []
    for idx in range(reference_gradient.shape[0]):
        row = {
            "label": run["label"],
            "solver": run["solver"],
            "iterations": int(run["iterations"]),
            "dtype": run["dtype"],
            "family_index": int(original_indices[idx]),
            "family_name": names[idx],
            "rel_l2": float(rel_l2[idx]),
            "rel_inf": float(rel_inf[idx]),
            "abs_l2": float(abs_l2[idx]),
            "abs_inf": float(abs_inf[idx]),
            "ref_l2": float(ref_norm[idx]),
            "grad_l2": float(grad_norm[idx]),
            "theta_D": float(theta_cpu[idx, 0]),
            "theta_T": float(theta_cpu[idx, 1]),
            "theta_L": float(theta_cpu[idx, 2]),
            "D": float(rates[idx, 0]),
            "T": float(rates[idx, 1]),
            "L": float(rates[idx, 2]),
            "ref_grad_D": float(reference_gradient[idx, 0]),
            "ref_grad_T": float(reference_gradient[idx, 1]),
            "ref_grad_L": float(reference_gradient[idx, 2]),
            "grad_D": float(gradient[idx, 0]),
            "grad_T": float(gradient[idx, 1]),
            "grad_L": float(gradient[idx, 2]),
        }
        row.update(family_stats(model, idx))
        rows.append(row)

    order_rel = torch.argsort(rel_l2, descending=True)
    order_abs = torch.argsort(abs_l2, descending=True)
    summary = {
        "label": run["label"],
        "solver": run["solver"],
        "iterations": int(run["iterations"]),
        "dtype": run["dtype"],
        "env": run["env"],
        "loss": run["loss"],
        "elapsed_s": run["elapsed_s"],
        "peak_gb": run["peak_gb"],
        "global_rel_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference_gradient).clamp_min(1e-12)),
        "global_abs_l2": float(torch.linalg.vector_norm(delta)),
        "global_ref_l2": float(torch.linalg.vector_norm(reference_gradient)),
        "rel_l2_distribution": quantile_summary(rel_l2),
        "rel_inf_distribution": quantile_summary(rel_inf),
        "abs_l2_distribution": quantile_summary(abs_l2),
        "top_by_rel_l2": [rows[int(idx)] for idx in order_rel[:top_k].tolist()],
        "top_by_abs_l2": [rows[int(idx)] for idx in order_abs[:top_k].tolist()],
    }
    rows.sort(key=lambda item: item["family_index"])
    return summary, rows


def write_tsv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tsv_fieldnames()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()


def append_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = tsv_fieldnames()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def tsv_fieldnames() -> list[str]:
    return [
        "label",
        "solver",
        "iterations",
        "dtype",
        "family_index",
        "family_name",
        "rel_l2",
        "rel_inf",
        "abs_l2",
        "abs_inf",
        "ref_l2",
        "grad_l2",
        "clades",
        "splits",
        "leaves",
        "D",
        "T",
        "L",
        "theta_D",
        "theta_T",
        "theta_L",
        "ref_grad_D",
        "ref_grad_T",
        "ref_grad_L",
        "grad_D",
        "grad_T",
        "grad_L",
    ]


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")


def controlled_env(**updates: str | None) -> dict[str, str | None]:
    env = {key: None for key in CONTROLLED_ENV_KEYS}
    env.update(updates)
    return env


def make_cases(args: argparse.Namespace) -> list[GradientCase]:
    cases: list[GradientCase] = []
    base_fp32 = controlled_env(GPUREC_DIRECT_THETA_VJP="1")
    for term in args.neumann_terms:
        cases.append(
            GradientCase(
                label="abs_fp32_direct_theta",
                solver="neumann",
                iterations=int(term),
                dtype_name="float32",
                env=base_fp32,
            )
        )
    for term in args.gmres_terms:
        cases.append(
            GradientCase(
                label="abs_fp32_direct_theta",
                solver="gmres",
                iterations=int(term),
                dtype_name="float32",
                env=base_fp32,
            )
        )
    centered_fp32 = controlled_env(
        GPUREC_CENTERED_PI_FORWARD="1",
        GPUREC_CENTERED_PI_COMPENSATION_DTYPE="fp32",
        GPUREC_DIRECT_THETA_VJP="1",
    )
    for term in args.centered_neumann_terms:
        cases.append(
            GradientCase(
                label="centered_fp32_comp_pi_direct_theta",
                solver="neumann",
                iterations=int(term),
                dtype_name="float32",
                env=centered_fp32,
            )
        )
    centered_accel_fp32 = controlled_env(
        GPUREC_CENTERED_PI_FORWARD="1",
        GPUREC_CENTERED_PI_COMPENSATION_DTYPE="fp32",
        GPUREC_CENTERED_PI_ACCELERATED_SELF_LOOP="1",
        GPUREC_DIRECT_THETA_VJP="1",
    )
    for term in args.centered_gmres_terms:
        cases.append(
            GradientCase(
                label="centered_fp32_comp_pi_direct_theta",
                solver="gmres",
                iterations=int(term),
                dtype_name="float32",
                env=centered_accel_fp32,
            )
        )
    base_fp64 = controlled_env(GPUREC_DIRECT_THETA_VJP="1")
    for term in args.fp64_neumann_terms:
        cases.append(
            GradientCase(
                label="abs_fp64_direct_theta",
                solver="neumann",
                iterations=int(term),
                dtype_name="float64",
                env=base_fp64,
            )
        )
    for term in args.fp64_gmres_terms:
        cases.append(
            GradientCase(
                label="abs_fp64_direct_theta",
                solver="gmres",
                iterations=int(term),
                dtype_name="float64",
                env=base_fp64,
            )
        )
    centered_fp64 = controlled_env(
        GPUREC_CENTERED_PI_FORWARD="1",
        GPUREC_CENTERED_PI_COMPENSATION_DTYPE="fp64",
        GPUREC_CENTERED_PI_ALLOW_FP64="1",
        GPUREC_DIRECT_THETA_VJP="1",
    )
    for term in args.centered_fp64_neumann_terms:
        cases.append(
            GradientCase(
                label="centered_fp64_comp_pi_direct_theta",
                solver="neumann",
                iterations=int(term),
                dtype_name="float64",
                env=centered_fp64,
            )
        )
    centered_accel_fp64 = controlled_env(
        GPUREC_CENTERED_PI_FORWARD="1",
        GPUREC_CENTERED_PI_COMPENSATION_DTYPE="fp64",
        GPUREC_CENTERED_PI_ACCELERATED_SELF_LOOP="1",
        GPUREC_CENTERED_PI_ALLOW_FP64="1",
        GPUREC_DIRECT_THETA_VJP="1",
    )
    for term in args.centered_fp64_gmres_terms:
        cases.append(
            GradientCase(
                label="centered_fp64_comp_pi_direct_theta",
                solver="gmres",
                iterations=int(term),
                dtype_name="float64",
                env=centered_accel_fp64,
            )
        )
    return cases


def reference_gradient_path(args: argparse.Namespace) -> Path:
    if args.reference_gradient_pt is not None:
        return args.reference_gradient_pt
    return args.output_json.with_name(
        f"{args.output_json.stem}_fp64_neumann{int(args.reference_neumann)}_gradient.pt"
    )


def _family_indices_from_file(path: Path) -> list[int]:
    out: list[int] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split()[0].rstrip(",")
        if first.lower() in {"family_index", "index"}:
            continue
        out.append(int(first))
    return out


def load_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, Path, list[int], list[str], torch.Tensor, list[Path]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config") or {}
    species_tree = args.species_tree or Path(config["species_tree"])
    families_file = args.families_file or Path(config["families_file"])
    checkpoint_names = list(checkpoint["family_names"])
    checkpoint_theta = checkpoint["theta"].detach().cpu().double()
    if args.family_indices is not None and args.family_indices_file is not None:
        raise ValueError("use only one of --family-indices or --family-indices-file")
    if args.family_indices_file is not None:
        original_indices = _family_indices_from_file(args.family_indices_file)
    elif args.family_indices is not None:
        original_indices = list(args.family_indices)
    else:
        original_indices = list(range(len(checkpoint_names)))
    if args.limit is not None:
        original_indices = original_indices[: args.limit]
    if not original_indices:
        raise ValueError("selected family set is empty")
    if min(original_indices) < 0 or max(original_indices) >= len(checkpoint_names):
        raise ValueError(
            f"family index out of range for checkpoint with {len(checkpoint_names)} families"
        )
    names = [checkpoint_names[idx] for idx in original_indices]
    theta_cpu = checkpoint_theta[original_indices].contiguous()

    entries = load_family_entries(families_file)
    if len(entries) < len(checkpoint_names):
        raise RuntimeError(
            f"families file has only {len(entries)} entries for {len(checkpoint_names)} checkpoint rows"
        )
    for original_idx, name in zip(original_indices, names):
        if entries[original_idx].get("name") != name:
            raise RuntimeError(
                "family order mismatch at "
                f"{original_idx}: {entries[original_idx].get('name')} != {name}"
            )
    gene_trees = [resolve_tree_path(families_file, entries[idx]["tree"]) for idx in original_indices]
    return checkpoint, species_tree, families_file, original_indices, names, theta_cpu, gene_trees


def run_reference(
    args: argparse.Namespace,
    *,
    species_tree: Path,
    gene_trees: list[Path],
    original_indices: list[int],
    theta_cpu: torch.Tensor,
    device: torch.device,
    path: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.reuse_reference_gradient:
        payload = torch.load(path, map_location="cpu")
        gradient = payload["gradient"].detach().cpu().double()
        if int(gradient.shape[0]) != int(theta_cpu.shape[0]):
            index = torch.tensor(original_indices, dtype=torch.long)
            if int(gradient.shape[0]) <= int(index.max()):
                raise ValueError(
                    "saved reference gradient does not have enough rows for selected "
                    f"family index {int(index.max())}"
                )
            gradient = gradient.index_select(0, index).contiguous()
        return gradient, dict(payload.get("metadata") or {})

    case = GradientCase(
        label="fp64_neumann_reference",
        solver="neumann",
        iterations=int(args.reference_neumann),
        dtype_name="float64",
        env=controlled_env(GPUREC_DIRECT_THETA_VJP="1"),
    )
    model = build_model(
        args,
        species_tree=species_tree,
        gene_trees=gene_trees,
        device=device,
        dtype=torch.float64,
        solver=case.solver,
        iterations=case.iterations,
    )
    theta = theta_cpu.to(device=device, dtype=torch.float64).contiguous()
    receiver_weights = torch.zeros((int(model.species_helpers["S"]),), dtype=torch.float64, device=device)
    print(
        json.dumps(
            {
                "event": "reference_model_built",
                "batches": len(model.batch_statics),
                "families": len(model.families),
            }
        ),
        flush=True,
    )
    run = run_batched_gradient(model, theta, receiver_weights, case=case, progress_every=args.progress_every)
    metadata = {key: value for key, value in run.items() if key != "gradient"}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gradient": run["gradient"], "metadata": metadata}, path)
    del model, theta, receiver_weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run["gradient"], metadata


def main() -> None:
    args = parse_args()
    checkpoint, species_tree, families_file, original_indices, names, theta_cpu, gene_trees = load_inputs(args)
    config = checkpoint.get("config") or {}
    device = torch.device(args.device or config.get("device") or "cuda")
    cases = make_cases(args)
    ref_path = reference_gradient_path(args)

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "species_tree": str(species_tree),
        "families_file": str(families_file),
        "family_count": len(names),
        "family_indices": original_indices,
        "reference_neumann": int(args.reference_neumann),
        "e_iters": int(args.e_iters),
        "pi_iters": int(args.pi_iters),
        "device": str(device),
        "family_chunk_size": int(args.family_chunk_size),
        "clade_budget": int(args.clade_budget),
        "batch_packing": str(args.batch_packing),
        "max_wave_size": int(args.max_wave_size),
        "reference_gradient_pt": str(ref_path),
        "cases_requested": [
            {
                "label": case.label,
                "solver": case.solver,
                "iterations": int(case.iterations),
                "dtype": case.dtype_name,
            }
            for case in cases
        ],
        "reference": None,
        "results": [],
    }
    write_summary(args.output_json, summary)
    write_tsv_header(args.output_tsv)

    print(
        json.dumps(
            {
                "event": "start",
                "families": len(names),
                "reference_neumann": int(args.reference_neumann),
                "cases": summary["cases_requested"],
                "output_json": str(args.output_json),
                "output_tsv": str(args.output_tsv),
                "reference_gradient_pt": str(ref_path),
            }
        ),
        flush=True,
    )

    reference_gradient, reference_metadata = run_reference(
        args,
        species_tree=species_tree,
        gene_trees=gene_trees,
        original_indices=original_indices,
        theta_cpu=theta_cpu,
        device=device,
        path=ref_path,
    )
    summary["reference"] = reference_metadata
    write_summary(args.output_json, summary)

    runtime_by_dtype: dict[str, tuple[GeneReconModel, torch.Tensor, torch.Tensor]] = {}

    def runtime_for_dtype(dtype_name: str) -> tuple[GeneReconModel, torch.Tensor, torch.Tensor]:
        runtime = runtime_by_dtype.get(dtype_name)
        if runtime is not None:
            return runtime
        dtype = dtype_from_name(dtype_name)
        model = build_model(
            args,
            species_tree=species_tree,
            gene_trees=gene_trees,
            device=device,
            dtype=dtype,
            solver="neumann",
            iterations=1,
        )
        theta = theta_cpu.to(device=device, dtype=dtype).contiguous()
        receiver_weights = torch.zeros((int(model.species_helpers["S"]),), dtype=dtype, device=device)
        runtime = (model, theta, receiver_weights)
        runtime_by_dtype[dtype_name] = runtime
        print(
            json.dumps(
                {
                    "event": "model_built",
                    "dtype": dtype_name,
                    "batches": len(model.batch_statics),
                    "families": len(model.families),
                }
            ),
            flush=True,
        )
        return runtime

    for case in cases:
        print(
            json.dumps(
                {
                    "event": "case_start",
                    "label": case.label,
                    "solver": case.solver,
                    "iterations": int(case.iterations),
                }
            ),
            flush=True,
        )
        try:
            model, theta, receiver_weights = runtime_for_dtype(case.dtype_name)
            run = run_batched_gradient(
                model,
                theta,
                receiver_weights,
                case=case,
                progress_every=args.progress_every,
            )
            if args.save_case_gradients:
                grad_path = ref_path.with_name(
                    f"{ref_path.stem}_{case.label}_{case.solver}{int(case.iterations)}.pt"
                )
                torch.save({"gradient": run["gradient"], "metadata": {k: v for k, v in run.items() if k != "gradient"}}, grad_path)
                run["gradient_pt"] = str(grad_path)
            case_summary, rows = compare_case(
                names=names,
                original_indices=original_indices,
                theta_cpu=theta_cpu,
                reference_gradient=reference_gradient,
                run=run,
                model=model,
                top_k=int(args.top_k),
            )
            append_tsv(args.output_tsv, rows)
        except Exception as exc:  # pragma: no cover - diagnostic script path
            if device.type == "cuda":
                torch.cuda.empty_cache()
            case_summary = {
                "label": case.label,
                "solver": case.solver,
                "iterations": int(case.iterations),
                "dtype": case.dtype_name,
                "env": case.env,
                "error": repr(exc),
            }
        summary["results"].append(case_summary)
        write_summary(args.output_json, summary)
        print(
            json.dumps(
                {
                    "event": "case_done",
                    "label": case.label,
                    "solver": case.solver,
                    "iterations": int(case.iterations),
                    "global_rel_l2": case_summary.get("global_rel_l2"),
                    "rel_l2_p50": (case_summary.get("rel_l2_distribution") or {}).get("p50"),
                    "rel_l2_p95": (case_summary.get("rel_l2_distribution") or {}).get("p95"),
                    "rel_l2_p99": (case_summary.get("rel_l2_distribution") or {}).get("p99"),
                    "rel_l2_max": (case_summary.get("rel_l2_distribution") or {}).get("max"),
                    "error": case_summary.get("error"),
                }
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "done",
                "output_json": str(args.output_json),
                "output_tsv": str(args.output_tsv),
                "reference_gradient_pt": str(ref_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

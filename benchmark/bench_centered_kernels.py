#!/usr/bin/env python3
"""Benchmark absolute and centered Pi representations on live HOGENOM data.

The output is a machine-readable JSON record. Generated files belong under the
ignored ``output/`` directory; they are evidence for a particular machine and
must not be committed as source.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.core.inference.logspace import log2_survival, logsumexp2
from gpurec.core.inference.solver import nll_from_root_rows, solve_resident_e_pi


REPO_ROOT = Path(__file__).resolve().parents[1]
_HOGENOM_ROOT_ENV = os.environ.get("GPUREC_HOGENOM_ROOT")
DEFAULT_HOGENOM_ROOT = Path(_HOGENOM_ROOT_ENV) if _HOGENOM_ROOT_ENV else None
DEFAULT_FAMILIES_FILE = REPO_ROOT / "experiments/sanderson_cv/families_1055.txt"
DEFAULT_OUTPUT = REPO_ROOT / "output/centered_kernels_benchmark.json"


def _csv(value: str) -> list[str]:
    result = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hogenom-root", type=Path, default=DEFAULT_HOGENOM_ROOT)
    parser.add_argument("--families-file", type=Path, default=DEFAULT_FAMILIES_FILE)
    parser.add_argument("--families", type=int, default=1, help="prefix length from --families-file")
    parser.add_argument("--mode", choices=("global", "specieswise", "genewise"), default="global")
    parser.add_argument(
        "--representations", type=_csv, default=["absolute", "centered"],
        help="comma-separated absolute,centered",
    )
    parser.add_argument("--reference-fp64", action="store_true")
    parser.add_argument(
        "--paired-alternating",
        action="store_true",
        help="also keep fp32 absolute/centered models resident and alternate timed evaluations",
    )
    parser.add_argument(
        "--cuda-profiler-capture",
        action="store_true",
        help="wrap one post-warmup forward in cudaProfilerStart/Stop for nsys capture-range",
    )
    parser.add_argument(
        "--head-control",
        action="store_true",
        help="for one resident unweighted batch, record fp32/fp64 likelihood-head controls",
    )
    parser.add_argument("--weighted", action="store_true", help="use nonuniform receiver/origination weights")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--optimizer-steps", type=int, default=5)
    parser.add_argument("--optimizer-lr", type=float, default=1e-4)
    parser.add_argument("--e-max-iter", type=int, default=128)
    parser.add_argument("--e-tol", type=float, default=1e-8)
    parser.add_argument("--pi-iters", type=int, default=64)
    parser.add_argument("--neumann-terms", type=int, default=64)
    parser.add_argument("--family-chunk-size", type=int, default=300)
    parser.add_argument("--clade-budget", type=int, default=315_000)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.hogenom_root is None:
        parser.error("set GPUREC_HOGENOM_ROOT or pass --hogenom-root")
    if args.families < 1:
        parser.error("--families must be at least 1")
    if args.warmups < 0 or args.repeats < 1 or args.optimizer_steps < 0:
        parser.error("warmups/optimizer-steps must be nonnegative and repeats positive")
    if args.head_control and args.weighted:
        parser.error("--head-control currently requires uniform origination weights")
    invalid = sorted(set(args.representations) - {"absolute", "centered"})
    if invalid:
        parser.error(f"unknown representations: {', '.join(invalid)}")
    return args


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _git(command: list[str]) -> str:
    completed = subprocess.run(
        ["git", *command], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip()


def _driver_version() -> str | None:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip().splitlines()[0] if completed.returncode == 0 else None


def environment_record() -> dict[str, Any]:
    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = None
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
            "driver": _driver_version(),
        }
    return {
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "git_revision": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton_version,
        "gpu": gpu,
    }


def dataset_paths(args: argparse.Namespace) -> tuple[Path, list[Path], list[str]]:
    species_tree = args.hogenom_root / (
        "runs/MFP/true_start_ufboot1000/"
        "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/"
        "species_trees/starting_species_tree.newick"
    )
    family_names = [
        line.strip()
        for line in args.families_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.families]
    gene_trees = [
        args.hogenom_root / "families" / name / "gene_trees/ufboot1000.MFP.geneTree.newick"
        for name in family_names
    ]
    missing = [path for path in [species_tree, *gene_trees] if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing HOGENOM inputs:\n" + "\n".join(map(str, missing[:10])))
    return species_tree, gene_trees, family_names


def build_model(
    args: argparse.Namespace,
    species_tree: Path,
    gene_trees: list[Path],
    *,
    representation: str,
    dtype: torch.dtype,
) -> GeneReconModel:
    options = SolverOptions(
        e_max_iter=args.e_max_iter,
        e_tol=args.e_tol,
        pi_iters=args.pi_iters,
        pi_representation=representation,
        neumann_terms=args.neumann_terms,
    )
    model = GeneReconModel(
        species_tree,
        gene_trees,
        mode=args.mode,
        device="cuda",
        dtype=dtype,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        max_wave_size=args.max_wave_size,
        solver_options=options,
    )
    with torch.no_grad():
        model.theta.zero_()
        model.receiver_weights.zero_()
        model.origination_weights.zero_()
        if args.weighted:
            model.receiver_weights.copy_(
                torch.linspace(-1.25, 1.75, model.receiver_weights.numel(), device="cuda", dtype=dtype)
            )
            # This also handles genewise [G,S] origination weights.
            model.origination_weights.copy_(
                torch.linspace(
                    0.9, -1.1, model.origination_weights.numel(), device="cuda", dtype=dtype
                ).reshape_as(model.origination_weights)
            )
    return model


def _gradient_vector(model: GeneReconModel) -> torch.Tensor:
    parts = [
        parameter.grad.detach().reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not parts:
        raise RuntimeError("model backward produced no parameter gradients")
    return torch.cat(parts).double().cpu()


def _forward(model: GeneReconModel) -> float:
    with torch.no_grad():
        loss = model()
    _sync()
    return float(loss.detach().cpu())


def _loss_grad(model: GeneReconModel) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    loss = model()
    loss.backward()
    _sync()
    return float(loss.detach().cpu()), _gradient_vector(model)


def _head_control(model: GeneReconModel) -> tuple[dict[str, float], torch.Tensor]:
    """Evaluate identical solved roots through fp32 and fp64 likelihood heads."""
    if len(model.batch_statics) != 1:
        raise ValueError("--head-control requires a single resident batch")
    static = model.batch_statics[0]
    theta = model._theta_for_static(static, model.theta)
    solved = solve_resident_e_pi(
        static, theta, model.receiver_weights, warm_start_E=None
    )
    E, root_rows = solved[0], solved[4]

    def uniform_loss(dtype: torch.dtype) -> float:
        root = root_rows.to(dtype)
        extinction = E.to(dtype)
        loss = -(
            logsumexp2(root, dim=-1)
            - math.log2(root.shape[-1])
            - log2_survival(extinction)
        ).sum()
        return float(loss.detach().cpu())

    result = {
        "production_loss": float(nll_from_root_rows(root_rows, E).detach().cpu()),
        "fp32_head_loss": uniform_loss(torch.float32),
        "fp64_head_loss": uniform_loss(torch.float64),
    }
    return result, root_rows.detach().double().cpu()


def _summary(samples: list[float]) -> dict[str, Any]:
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def run_variant(
    args: argparse.Namespace,
    species_tree: Path,
    gene_trees: list[Path],
    *,
    name: str,
    representation: str,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    build_start = time.perf_counter()
    model = build_model(
        args, species_tree, gene_trees, representation=representation, dtype=dtype
    )
    _sync()
    result: dict[str, Any] = {
        "name": name,
        "representation": representation,
        "dtype": str(dtype).removeprefix("torch."),
        "build_s": time.perf_counter() - build_start,
        "n_batches": len(model.batch_statics),
        "species_nodes": int(model.species_helpers["S"]),
        "clades": sum(int(family["C"]) for family in model.families),
    }

    # Warm both paths so compilation and first-touch allocation do not enter
    # steady-state timings. A centered consumer that is intentionally guarded is
    # recorded instead of hiding the unsupported phase.
    gradient_supported = True
    gradient_error = None
    for _ in range(args.warmups):
        _forward(model)
    try:
        for _ in range(args.warmups):
            _loss_grad(model)
    except RuntimeError as exc:
        gradient_supported = False
        gradient_error = str(exc)

    if args.cuda_profiler_capture:
        torch.cuda.cudart().cudaProfilerStart()
        _forward(model)
        torch.cuda.cudart().cudaProfilerStop()

    torch.cuda.reset_peak_memory_stats()
    forward_ms: list[float] = []
    forward_losses: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        forward_losses.append(_forward(model))
        forward_ms.append((time.perf_counter() - start) * 1e3)
    result.update(
        forward_ms=_summary(forward_ms),
        forward_loss=forward_losses[-1],
        forward_loss_range=max(forward_losses) - min(forward_losses),
        peak_forward_bytes=torch.cuda.max_memory_allocated(),
    )

    representative_gradient = None
    if gradient_supported:
        torch.cuda.reset_peak_memory_stats()
        loss_grad_ms: list[float] = []
        gradient_losses: list[float] = []
        gradients: list[torch.Tensor] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            loss, gradient = _loss_grad(model)
            loss_grad_ms.append((time.perf_counter() - start) * 1e3)
            gradient_losses.append(loss)
            gradients.append(gradient)
        representative_gradient = gradients[-1]
        first = gradients[0]
        repeat_delta = [float((gradient - first).abs().max()) for gradient in gradients[1:]]
        result.update(
            loss_grad_ms=_summary(loss_grad_ms),
            gradient_loss=gradient_losses[-1],
            gradient_loss_range=max(gradient_losses) - min(gradient_losses),
            gradient_l2=float(representative_gradient.norm()),
            gradient_inf=float(representative_gradient.abs().max()),
            gradient_repeat_max_abs_delta=max(repeat_delta, default=0.0),
            peak_loss_grad_bytes=torch.cuda.max_memory_allocated(),
        )
    else:
        result["gradient_error"] = gradient_error

    if gradient_supported and args.optimizer_steps:
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optimizer_lr)
        trajectory = []
        for step in range(args.optimizer_steps):
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            loss = model()
            loss.backward()
            _sync()
            trajectory.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "gradient_inf": max(
                        float(parameter.grad.detach().abs().max().cpu())
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ),
                    "wall_ms": (time.perf_counter() - start) * 1e3,
                }
            )
            optimizer.step()
        result["optimizer"] = {
            "name": "SGD",
            "lr": args.optimizer_lr,
            "trajectory": trajectory,
        }
        model.load_state_dict(initial)

    controlled_root = None
    if args.head_control:
        result["head_control"], controlled_root = _head_control(model)

    del model
    torch.cuda.empty_cache()
    return result, representative_gradient, controlled_root


def run_paired_alternating(
    args: argparse.Namespace,
    species_tree: Path,
    gene_trees: list[Path],
) -> dict[str, Any]:
    """Alternate baseline/candidate evaluations to expose short-term GPU drift.

    This deliberately keeps two models resident and is therefore intended for
    subsets where the extra static memory is practical. Per-variant peak memory
    still comes from the sequential runs above.
    """

    names = ("absolute_fp32", "centered_fp32")
    models = {
        name: build_model(
            args,
            species_tree,
            gene_trees,
            representation=name.removesuffix("_fp32"),
            dtype=torch.float32,
        )
        for name in names
    }
    for _ in range(args.warmups):
        for name in names:
            _forward(models[name])
            _loss_grad(models[name])

    samples = {
        name: {"forward_ms": [], "loss_grad_ms": [], "loss": [], "gradient_loss": []}
        for name in names
    }
    for repeat in range(args.repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in order:
            start = time.perf_counter()
            loss = _forward(models[name])
            samples[name]["forward_ms"].append((time.perf_counter() - start) * 1e3)
            samples[name]["loss"].append(loss)
        # Reverse the operation's starting variant as well, so neither model
        # systematically follows the same preceding kernel mix.
        grad_order = tuple(reversed(order))
        for name in grad_order:
            start = time.perf_counter()
            loss, _gradient = _loss_grad(models[name])
            samples[name]["loss_grad_ms"].append((time.perf_counter() - start) * 1e3)
            samples[name]["gradient_loss"].append(loss)

    forward_ratios = [
        centered / absolute - 1.0
        for absolute, centered in zip(
            samples["absolute_fp32"]["forward_ms"],
            samples["centered_fp32"]["forward_ms"],
        )
    ]
    gradient_ratios = [
        centered / absolute - 1.0
        for absolute, centered in zip(
            samples["absolute_fp32"]["loss_grad_ms"],
            samples["centered_fp32"]["loss_grad_ms"],
        )
    ]
    result = {
        "ordering": "repeat parity alternates first variant; gradient order reverses forward order",
        "variants": {
            name: {
                "forward_ms": _summary(values["forward_ms"]),
                "loss_grad_ms": _summary(values["loss_grad_ms"]),
                "loss": values["loss"],
                "gradient_loss": values["gradient_loss"],
            }
            for name, values in samples.items()
        },
        "centered_over_absolute_forward_fraction": _summary(forward_ratios),
        "centered_over_absolute_loss_grad_fraction": _summary(gradient_ratios),
    }
    del models
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    species_tree, gene_trees, family_names = dataset_paths(args)
    variants: list[tuple[str, str, torch.dtype]] = [
        (f"{representation}_fp32", representation, torch.float32)
        for representation in args.representations
    ]
    if args.reference_fp64:
        variants.append(("absolute_fp64_reference", "absolute", torch.float64))

    rows: list[dict[str, Any]] = []
    gradients: dict[str, torch.Tensor] = {}
    controlled_roots: dict[str, torch.Tensor] = {}
    for name, representation, dtype in variants:
        print(json.dumps({"event": "variant_start", "name": name}), flush=True)
        row, gradient, controlled_root = run_variant(
            args,
            species_tree,
            gene_trees,
            name=name,
            representation=representation,
            dtype=dtype,
        )
        rows.append(row)
        if gradient is not None:
            gradients[name] = gradient
        if controlled_root is not None:
            controlled_roots[name] = controlled_root
        print(json.dumps({"event": "variant_done", "name": name}), flush=True)

    reference = next((row for row in rows if row["name"] == "absolute_fp64_reference"), None)
    if reference is not None:
        reference_gradient = gradients.get(reference["name"])
        for row in rows:
            row["loss_abs_error_vs_fp64"] = abs(
                float(row["forward_loss"]) - float(reference["forward_loss"])
            )
            gradient = gradients.get(row["name"])
            if gradient is not None and reference_gradient is not None:
                delta = gradient - reference_gradient
                row["gradient_abs_l2_error_vs_fp64"] = float(delta.norm())
                row["gradient_rel_l2_error_vs_fp64"] = float(
                    delta.norm() / reference_gradient.norm().clamp_min(1e-30)
                )
                row["gradient_abs_inf_error_vs_fp64"] = float(delta.abs().max())

    paired_alternating = None
    if args.paired_alternating:
        if not {"absolute", "centered"}.issubset(set(args.representations)):
            raise ValueError("--paired-alternating requires absolute and centered representations")
        paired_alternating = run_paired_alternating(args, species_tree, gene_trees)

    head_control_comparison = None
    if args.head_control:
        absolute_root = controlled_roots.get("absolute_fp32")
        centered_root = controlled_roots.get("centered_fp32")
        if absolute_root is not None and centered_root is not None:
            delta = absolute_root - centered_root
            head_control_comparison = {
                "absolute_vs_centered_root_max_abs": float(delta.abs().max()),
                "absolute_vs_centered_root_l2": float(delta.norm()),
                "absolute_vs_centered_root_nonzero_count": int(torch.count_nonzero(delta)),
            }
            fp64_root = controlled_roots.get("absolute_fp64_reference")
            if fp64_root is not None:
                for name, root in (
                    ("absolute_fp32", absolute_root),
                    ("centered_fp32", centered_root),
                ):
                    root_delta = root - fp64_root
                    head_control_comparison[f"{name}_root_max_abs_error_vs_fp64"] = float(
                        root_delta.abs().max()
                    )
                    head_control_comparison[f"{name}_root_l2_error_vs_fp64"] = float(
                        root_delta.norm()
                    )

    output = {
        "benchmark": "centered_kernels",
        "environment": environment_record(),
        "method": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timing": "host perf_counter with CUDA synchronization after each evaluation",
            "ordering": "variants sequential; compilation excluded; samples retained in full",
            "theta_initialization": "zero log2-rate logits",
            "weighted_receiver_and_origination": args.weighted,
        },
        "dataset": {
            "hogenom_root": str(args.hogenom_root),
            "species_tree": str(species_tree),
            "families_file": str(args.families_file),
            "family_names": family_names,
        },
        "solver": {
            "mode": args.mode,
            "e_max_iter": args.e_max_iter,
            "e_tol": args.e_tol,
            "pi_iters": args.pi_iters,
            "neumann_terms": args.neumann_terms,
            "family_chunk_size": args.family_chunk_size,
            "clade_budget": args.clade_budget,
            "max_wave_size": args.max_wave_size,
        },
        "variants": rows,
        "paired_alternating": paired_alternating,
        "head_control_comparison": head_control_comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()

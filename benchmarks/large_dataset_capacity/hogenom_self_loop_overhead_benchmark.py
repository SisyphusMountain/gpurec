#!/usr/bin/env python3
"""Time HOGENOM self-loop backward work for Neumann and GMRES.

This benchmark isolates the cost that matters for GMRES overhead: the implicit
backward phase after the forward loss has already been built.  It reports both
wall time and self-loop application counts, so a GMRES run can be compared to
Neumann in terms of time per expensive backward application.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.large_dataset_capacity.hogenom_gmres_neumann_family_experiment import DEFAULT_SPECIES_TREE
from benchmarks.large_dataset_capacity.run_gpurec_benchmark import (
    SelfLoopBackwardRecorder,
    choose_gene_trees,
    dataset_stats,
)
from gpurec import GeneReconModel, SolverOptions


DEFAULT_GENE_TREE_DIR = REPO_ROOT / "benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core/hogenom/families"


@dataclass(frozen=True)
class SolverCase:
    name: str
    solver: str
    iterations: int
    gmres_tol: float | None = None
    gmres_check_interval: int = 1
    reuse_check_schedule: bool = False
    trust_check_schedule: bool = False
    trusted_schedule_validation_interval: int = 20
    trusted_schedule_safety_margin: int = 0
    reuse_solution: bool = False
    preconditioner: str = "none"


@dataclass
class BackwardGraphReplay:
    graph: torch.cuda.CUDAGraph
    stream: torch.cuda.Stream
    static_loss: torch.Tensor
    reference_self_loop: dict[str, Any]


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


def _git_status_short() -> list[str] | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in output.splitlines() if line]


def _parse_case(text: str) -> SolverCase:
    raw_parts = [part.strip() for part in text.split(":") if part.strip()]
    if len(raw_parts) < 2:
        raise argparse.ArgumentTypeError(
            "cases must look like neumann:16, gmres_fixed:16, or gmres:16:7e-6"
        )
    solver = raw_parts[0].lower()
    if solver not in {"neumann", "gmres", "gmres_fixed"}:
        raise argparse.ArgumentTypeError(f"unsupported solver {solver!r}")
    iterations = int(raw_parts[1])
    if iterations < 1:
        raise argparse.ArgumentTypeError("case iteration count must be positive")
    gmres_tol = float(raw_parts[2]) if solver in {"gmres", "gmres_fixed"} and len(raw_parts) >= 3 else None

    options = {part.lower() for part in raw_parts[3:]}
    valid_options = {"reuse", "trust", "solution", "diagonal"}
    unknown = sorted(options - valid_options)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown case options: {unknown}")

    suffix = []
    if gmres_tol is not None:
        suffix.append(f"tol{gmres_tol:g}")
    suffix.extend(sorted(options))
    name = "_".join([solver, str(iterations), *suffix])
    return SolverCase(
        name=name,
        solver=solver,
        iterations=iterations,
        gmres_tol=gmres_tol,
        reuse_check_schedule="reuse" in options or "trust" in options,
        trust_check_schedule="trust" in options,
        reuse_solution="solution" in options,
        preconditioner="diagonal" if "diagonal" in options else "none",
    )


def _parse_cases(text: str) -> list[SolverCase]:
    cases = [_parse_case(part) for part in text.split(",") if part.strip()]
    if not cases:
        raise argparse.ArgumentTypeError("at least one solver case is required")
    return cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--e-max-iter", type=int, default=16)
    parser.add_argument("--e-tol", type=float, default=1e-8)
    parser.add_argument("--pi-iters", type=int, default=16)
    parser.add_argument("--bicgstab-max-iter", type=int, default=500)
    parser.add_argument("--bicgstab-tol", type=float, default=1e-7)
    parser.add_argument("--gmres-check-interval", type=int, default=1)
    parser.add_argument("--gmres-trusted-schedule-validation-interval", type=int, default=0)
    parser.add_argument("--gmres-trusted-schedule-safety-margin", type=int, default=0)
    parser.add_argument("--gmres-solution-cache-min-iterations", type=int, default=2)
    parser.add_argument("--gmres-diagonal-preconditioner-floor", type=float, default=1e-4)
    parser.add_argument(
        "--cases",
        type=_parse_cases,
        default=_parse_cases("neumann:16,gmres:10:7e-6,gmres:10:7e-6:reuse:trust"),
        help=(
            "Comma-separated cases. Format: solver:iterations[:tol][:options]. "
            "Options for GMRES: reuse, trust, solution, diagonal."
        ),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument(
        "--cuda-graph-backward-replay",
        action="store_true",
        help=(
            "After warmups, capture a backward-only CUDA graph and time graph.replay() "
            "instead of launching the Python backward path for each repeat. This is "
            "intended for launch-overhead experiments; checked adaptive GMRES may be "
            "uncapturable because it reads residuals back to the CPU."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record case failures and continue with the remaining cases.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    return args


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


def _solver_options(args: argparse.Namespace, case: SolverCase) -> SolverOptions:
    return SolverOptions(
        e_max_iter=int(args.e_max_iter),
        e_tol=float(args.e_tol),
        pi_iters=int(args.pi_iters),
        neumann_terms=int(case.iterations),
        self_loop_solver=case.solver,
        gmres_tol=float(case.gmres_tol if case.gmres_tol is not None else 1e-10),
        gmres_check_interval=int(args.gmres_check_interval),
        gmres_reuse_check_schedule=bool(case.reuse_check_schedule),
        gmres_trust_check_schedule=bool(case.trust_check_schedule),
        gmres_trusted_schedule_validation_interval=int(args.gmres_trusted_schedule_validation_interval),
        gmres_trusted_schedule_safety_margin=int(args.gmres_trusted_schedule_safety_margin),
        gmres_reuse_solution=bool(case.reuse_solution),
        gmres_solution_cache_min_iterations=int(args.gmres_solution_cache_min_iterations),
        gmres_preconditioner=case.preconditioner,
        gmres_diagonal_preconditioner_floor=float(args.gmres_diagonal_preconditioner_floor),
        bicgstab_max_iter=int(args.bicgstab_max_iter),
        bicgstab_tol=float(args.bicgstab_tol),
    )


def _configure_model(model: GeneReconModel, args: argparse.Namespace, case: SolverCase) -> None:
    model.solver_options = _solver_options(args, case)
    model.clear_warm_starts()


def _run_one_backward(model: GeneReconModel, device: torch.device, *, nvtx_name: str | None = None) -> dict[str, Any]:
    model.theta.grad = None
    model.receiver_weights.grad = None
    forward_start = time.perf_counter()
    loss = model()
    _cuda_sync(device)
    forward_seconds = time.perf_counter() - forward_start

    if device.type == "cuda" and nvtx_name:
        torch.cuda.nvtx.range_push(nvtx_name)
    backward_start = time.perf_counter()
    with SelfLoopBackwardRecorder(model) as recorder:
        recorder.backward(loss)
        _cuda_sync(device)
        self_loop = recorder.summary()
    backward_seconds = time.perf_counter() - backward_start
    if device.type == "cuda" and nvtx_name:
        torch.cuda.nvtx.range_pop()

    applications = int(self_loop.get("self_loop_backward_iterations") or 0)
    return {
        "loss": float(loss.detach().cpu()),
        "forward_seconds": float(forward_seconds),
        "backward_seconds": float(backward_seconds),
        "backward_ms_per_self_loop_application": (
            1000.0 * backward_seconds / applications if applications else None
        ),
        "self_loop": self_loop,
        "cuda_memory": _cuda_memory(device),
    }


def _capture_backward_graph(
    model: GeneReconModel,
    device: torch.device,
    *,
    reference_self_loop: dict[str, Any],
) -> BackwardGraphReplay:
    if device.type != "cuda":
        raise RuntimeError("CUDA graph replay requires a CUDA device")
    current_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(current_stream)
    if model.theta.grad is None:
        model.theta.grad = torch.zeros_like(model.theta)
    else:
        model.theta.grad.zero_()
    with torch.cuda.stream(capture_stream):
        model.theta.grad.zero_()
        static_loss = model()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        graph,
        stream=capture_stream,
        capture_error_mode="relaxed",
    ):
        model.theta.grad.zero_()
        static_loss.backward(retain_graph=True)
    _cuda_sync(device)
    return BackwardGraphReplay(
        graph=graph,
        stream=capture_stream,
        static_loss=static_loss,
        reference_self_loop=dict(reference_self_loop),
    )


def _run_backward_graph_replay(
    replay: BackwardGraphReplay,
    device: torch.device,
    *,
    nvtx_name: str | None = None,
) -> dict[str, Any]:
    if device.type == "cuda" and nvtx_name:
        torch.cuda.nvtx.range_push(nvtx_name)
    start = time.perf_counter()
    replay.graph.replay()
    _cuda_sync(device)
    elapsed = time.perf_counter() - start
    if device.type == "cuda" and nvtx_name:
        torch.cuda.nvtx.range_pop()

    applications = int(replay.reference_self_loop.get("self_loop_backward_iterations") or 0)
    return {
        "loss": float(replay.static_loss.detach().cpu()),
        "forward_seconds": 0.0,
        "backward_seconds": float(elapsed),
        "backward_ms_per_self_loop_application": (
            1000.0 * elapsed / applications if applications else None
        ),
        "self_loop": dict(replay.reference_self_loop),
        "cuda_memory": _cuda_memory(device),
        "cuda_graph_backward_replay": True,
    }


def _summarize_repeats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("forward_seconds", "backward_seconds", "backward_ms_per_self_loop_application"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            out[key] = None
            continue
        out[key] = {
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    if rows:
        out["last_self_loop"] = rows[-1]["self_loop"]
        out["last_loss"] = rows[-1]["loss"]
        out["cuda_graph_backward_replay"] = bool(rows[-1].get("cuda_graph_backward_replay", False))
    return out


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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
        solver_options=_solver_options(args, args.cases[0]),
    )
    model.receiver_weights.requires_grad_(False)
    _cuda_sync(device)

    git_status_short = _git_status_short()
    result: dict[str, Any] = {
        "benchmark": "hogenom_self_loop_overhead",
        "git_commit": _git_commit(),
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "host": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("GPUREC_GMRES") or key.startswith("CUDA")
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "cases"},
        "cases": [],
        "dataset": dataset_stats(model, gene_trees),
    }

    for case in args.cases:
        _configure_model(model, args, case)
        warmup_rows = []
        repeat_rows = []
        try:
            for warmup_idx in range(int(args.warmups)):
                warmup_rows.append(
                    _run_one_backward(
                        model,
                        device,
                        nvtx_name=f"warmup:{case.name}:{warmup_idx + 1}" if args.cuda_profiler_range else None,
                    )
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            replay: BackwardGraphReplay | None = None
            if args.cuda_graph_backward_replay:
                if not warmup_rows:
                    warmup_rows.append(
                        _run_one_backward(
                            model,
                            device,
                            nvtx_name=(
                                f"graph_reference:{case.name}"
                                if args.cuda_profiler_range
                                else None
                            ),
                        )
                    )
                replay = _capture_backward_graph(
                    model,
                    device,
                    reference_self_loop=warmup_rows[-1]["self_loop"],
                )
            if args.cuda_profiler_range and device.type == "cuda":
                torch.cuda.cudart().cudaProfilerStart()
            try:
                for repeat_idx in range(int(args.repeat)):
                    nvtx_name = (
                        f"measure:{case.name}:{repeat_idx + 1}"
                        if args.cuda_profiler_range
                        else None
                    )
                    if replay is None:
                        repeat_rows.append(
                            _run_one_backward(
                                model,
                                device,
                                nvtx_name=nvtx_name,
                            )
                        )
                    else:
                        repeat_rows.append(
                            _run_backward_graph_replay(
                                replay,
                                device,
                                nvtx_name=nvtx_name,
                            )
                        )
            finally:
                if args.cuda_profiler_range and device.type == "cuda":
                    torch.cuda.cudart().cudaProfilerStop()

            case_result = {
                "name": case.name,
                "case": case.__dict__,
                "solver_options": dict(vars(_solver_options(args, case))),
                "warmups": warmup_rows,
                "repeats": repeat_rows,
                "summary": _summarize_repeats(repeat_rows),
                "cuda_graph_backward_replay": bool(args.cuda_graph_backward_replay),
                "cuda_memory_after_case": _cuda_memory(device),
            }
        except Exception as exc:
            if not args.continue_on_error:
                raise
            case_result = {
                "name": case.name,
                "case": case.__dict__,
                "solver_options": dict(vars(_solver_options(args, case))),
                "warmups": warmup_rows,
                "repeats": repeat_rows,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "cuda_memory_after_case": _cuda_memory(device),
            }
        result["cases"].append(case_result)

    result["finished_at_unix"] = time.time()
    return result


def main() -> None:
    args = parse_args()
    output = args.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

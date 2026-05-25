#!/usr/bin/env python3
"""Benchmark resident GeneReconModel likelihood passes on generated tree data.

This source-checkout profiler targets datasets laid out like
``tests/data/test_trees_1000`` with ``sp.nwk`` and ``g_*.nwk`` files. It uses
the public ``GeneReconModel.from_trees`` API, materializes resident batches, and
times full-dataset likelihood-only or likelihood+gradient passes.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpurec import GeneReconModel  # noqa: E402


def _parse_dtype(value: str) -> torch.dtype:
    text = value.strip().lower()
    if text in ("float32", "fp32", "single"):
        return torch.float32
    if text in ("float64", "fp64", "double"):
        return torch.float64
    raise argparse.ArgumentTypeError("dtype must be float32/fp32 or float64/fp64")


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "0", "none", "null"):
        return None
    return int(text)


def _parse_fixed_iters(value: str) -> list[int]:
    budgets: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        budget = int(text)
        if budget <= 0 or budget % 2:
            raise argparse.ArgumentTypeError("fixed iteration budgets must be positive even integers")
        budgets.append(budget)
    if not budgets:
        raise argparse.ArgumentTypeError("at least one fixed iteration budget is required")
    return budgets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("tests/data/test_trees_1000"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--fams", type=int, default=1000)
    parser.add_argument(
        "--mode",
        choices=("global", "specieswise", "genewise"),
        default="specieswise",
    )
    parser.add_argument("--dtype", type=_parse_dtype, default=_parse_dtype("float32"))
    parser.add_argument("--fixed-iters", type=_parse_fixed_iters, default=_parse_fixed_iters("6,8"))
    parser.add_argument("--theta-rate", type=float, default=0.05)
    parser.add_argument("--family-chunk-size", type=int, default=300)
    parser.add_argument("--clade-budget", type=_parse_optional_int, default=315_000)
    parser.add_argument(
        "--batch-packing",
        choices=("sequential", "clade_first_fit", "depth_first_fit"),
        default="depth_first_fit",
    )
    parser.add_argument("--max-wave-size", type=_parse_optional_int, default=8192)
    parser.add_argument(
        "--measure",
        choices=("loss-only", "loss-grad", "both"),
        default="loss-only",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--prefetch-batches",
        choices=("all", "none"),
        default="all",
    )
    return parser


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _selected_genes(dataset: Path, *, start: int, fams: int) -> list[Path]:
    genes = sorted(dataset.glob("g_*.nwk"))
    if fams <= 0:
        raise ValueError("--fams must be positive")
    selected = genes[start : start + fams]
    if len(selected) != fams:
        raise FileNotFoundError(
            f"requested {fams} gene trees from offset {start}, found {len(selected)}"
        )
    return selected


def _clear_solver_runtime_state(model: GeneReconModel) -> None:
    model.theta.grad = None
    for static in model.cached_static_states:
        if hasattr(static, "warm_E"):
            static.warm_E = None
        if hasattr(static, "last_solver_stats"):
            static.last_solver_stats = None


def _configure_budget(model: GeneReconModel, budget: int) -> None:
    model.configure_solver_iterations(
        fixed_iters_E=budget,
        fixed_iters_Pi=budget,
        neumann_terms=budget,
        adaptive_neumann_terms=False,
    )
    _clear_solver_runtime_state(model)


def _time_loss_only(model: GeneReconModel) -> tuple[float, float]:
    _synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        loss = model.full_loss_for_theta(model.theta.detach())
    _synchronize()
    return time.perf_counter() - started, float(loss.detach().cpu())


def _time_loss_grad(model: GeneReconModel) -> tuple[float, float, float]:
    _synchronize()
    started = time.perf_counter()
    loss = model.full_loss()
    loss.backward()
    _synchronize()
    grad_inf = float(model.theta.grad.detach().abs().amax().cpu())
    return time.perf_counter() - started, float(loss.detach().cpu()), grad_inf


def _timed_rows(
    model: GeneReconModel,
    *,
    budget: int,
    measure: str,
    warmups: int,
    reps: int,
) -> list[dict[str, Any]]:
    if warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if reps <= 0:
        raise ValueError("--reps must be positive")
    rows: list[dict[str, Any]] = []
    for warmup in range(warmups):
        _configure_budget(model, budget)
        if measure == "loss-only":
            elapsed_s, loss = _time_loss_only(model)
            row = {"event": "warmup", "idx": warmup, "measure": measure, "elapsed_s": elapsed_s, "loss_bits": loss}
        else:
            elapsed_s, loss, grad_inf = _time_loss_grad(model)
            row = {
                "event": "warmup",
                "idx": warmup,
                "measure": measure,
                "elapsed_s": elapsed_s,
                "loss_bits": loss,
                "grad_inf": grad_inf,
            }
        print(json.dumps(row, sort_keys=True), flush=True)

    for rep in range(reps):
        _configure_budget(model, budget)
        if measure == "loss-only":
            elapsed_s, loss = _time_loss_only(model)
            row = {"event": "measured", "idx": rep, "measure": measure, "elapsed_s": elapsed_s, "loss_bits": loss}
        else:
            elapsed_s, loss, grad_inf = _time_loss_grad(model)
            row = {
                "event": "measured",
                "idx": rep,
                "measure": measure,
                "elapsed_s": elapsed_s,
                "loss_bits": loss,
                "grad_inf": grad_inf,
            }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def _summary_row(
    *,
    model: GeneReconModel,
    budget: int,
    measure: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    times = [float(row["elapsed_s"]) for row in rows]
    last = rows[-1]
    row: dict[str, Any] = {
        "event": "summary",
        "measure": measure,
        "budget": budget,
        "reps": len(rows),
        "median_s": statistics.median(times),
        "mean_s": statistics.mean(times),
        "min_s": min(times),
        "loss_bits": float(last["loss_bits"]),
        "families": int(model.n_families),
        "species": int(model.n_species),
        "batches": len(model.batch_metadata),
        "max_batch_clades": max(meta.clade_count for meta in model.batch_metadata),
        "max_batch_waves": max(meta.wave_count for meta in model.batch_metadata),
        "max_batch_wave_size": max(meta.max_wave_size for meta in model.batch_metadata),
    }
    if "grad_inf" in last:
        row["grad_inf"] = float(last["grad_inf"])
    if torch.cuda.is_available():
        row["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        row["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return row


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for resident likelihood benchmarking")
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    dataset = args.dataset
    species_tree = dataset / "sp.nwk"
    if not species_tree.is_file():
        raise FileNotFoundError(species_tree)
    genes = _selected_genes(dataset, start=args.start, fams=args.fams)

    torch.cuda.empty_cache()
    gc.collect()
    started = time.perf_counter()
    model = GeneReconModel.from_trees(
        species_tree=str(species_tree),
        gene_trees=[str(path) for path in genes],
        mode=args.mode,
        device="cuda",
        dtype=args.dtype,
        theta_init_rates=(args.theta_rate, args.theta_rate, args.theta_rate),
        fixed_iters_E=args.fixed_iters[0],
        fixed_iters_Pi=args.fixed_iters[0],
        neumann_terms=args.fixed_iters[0],
        adaptive_iters=False,
        adaptive_neumann_terms=False,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        batch_packing=args.batch_packing,
        max_wave_size=args.max_wave_size,
        lazy_preprocess=True,
        prefetch_batches=args.prefetch_batches,
    )
    model.materialize_batches()
    _synchronize()
    print(
        json.dumps(
            {
                "event": "model",
                "dataset": str(dataset),
                "mode": args.mode,
                "dtype": str(args.dtype).replace("torch.", ""),
                "build_s": time.perf_counter() - started,
                "families": int(model.n_families),
                "species": int(model.n_species),
                "batches": len(model.batch_metadata),
                "family_chunk_size": args.family_chunk_size,
                "clade_budget": args.clade_budget,
                "batch_packing": args.batch_packing,
                "max_wave_size": args.max_wave_size,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    measures = ("loss-only", "loss-grad") if args.measure == "both" else (args.measure,)
    for budget in args.fixed_iters:
        for measure in measures:
            rows = _timed_rows(
                model,
                budget=budget,
                measure=measure,
                warmups=args.warmups,
                reps=args.reps,
            )
            print(
                json.dumps(
                    _summary_row(model=model, budget=budget, measure=measure, rows=rows),
                    sort_keys=True,
                ),
                flush=True,
            )

    close = getattr(model, "close", None)
    if callable(close):
        close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Profile one active resident batch through the genewise optimizer path.

This checkout-local harness is intentionally close to
``gpurec.workflow.optimize``: in genewise mode it evaluates only the selected
resident batch, scatters the active per-family loss vector back to the full
genewise shape, and zeros inactive gradients before BatchedLBFGS sees them.
It emits JSON lines so shell scripts can aggregate timings without parsing
human-oriented text.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpurec import GeneReconModel  # noqa: E402
from gpurec.optimization import BatchedLBFGS  # noqa: E402


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_profiler_start(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        synchronize()
        torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        synchronize()
        torch.cuda.cudart().cudaProfilerStop()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype {name!r}")


def optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return None if parsed == 0 else parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile active-batch genewise closure or one BatchedLBFGS step."
        ),
    )
    parser.add_argument("--species-tree", type=Path, required=True)
    parser.add_argument("--families-file", type=Path, required=True)
    parser.add_argument("--mode", choices=("genewise", "specieswise", "global"), default="genewise")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--theta-path", type=Path, default=None)
    parser.add_argument("--theta-rate", type=float, default=None)
    parser.add_argument("--fixed-iters-e", type=optional_positive_int, default=None)
    parser.add_argument("--fixed-iters-pi", type=int, default=64)
    parser.add_argument("--neumann-terms", type=int, default=64)
    parser.add_argument(
        "--adaptive-iters",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--convergence-check-interval", type=int, default=4)
    parser.add_argument("--family-chunk-size", type=int, default=0)
    parser.add_argument("--clade-budget", type=int, default=None)
    parser.add_argument(
        "--batch-packing",
        choices=("sequential", "clade_first_fit", "depth_first_fit"),
        default="depth_first_fit",
    )
    parser.add_argument("--max-wave-size", type=optional_positive_int, default=8192)
    parser.add_argument("--max-dts-partial-rows", type=optional_positive_int, default=None)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument(
        "--batch-selector",
        choices=("index", "largest-clades", "largest-splits", "largest-waves"),
        default="index",
    )
    parser.add_argument(
        "--operation",
        choices=("closure", "lbfgs-step", "list-batches"),
        default="closure",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--profile-runs", type=int, default=1)
    parser.add_argument(
        "--cuda-profiler-api",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--evict-inactive-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop already-built non-selected batch statics before measuring.",
    )
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-max-iter", type=int, default=1)
    parser.add_argument("--lbfgs-history-size", type=int, default=20)
    parser.add_argument("--lbfgs-max-ls", type=int, default=8)
    parser.add_argument(
        "--lbfgs-line-search",
        choices=("armijo", "strong_wolfe"),
        default="armijo",
    )
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=None)
    return parser.parse_args()


def batch_row(model: GeneReconModel, batch_index: int) -> dict[str, Any]:
    meta = model.batch_metadata[batch_index]
    families = tuple(int(idx) for idx in meta.family_indices)
    row: dict[str, Any] = {
        "batch_index": int(batch_index),
        "family_count": int(meta.family_count),
        "clade_count": int(meta.clade_count),
        "split_count": int(meta.split_count),
        "wave_count": int(meta.wave_count),
        "max_wave_size": int(meta.max_wave_size),
        "first_family": int(min(families)) if families else None,
        "last_family": int(max(families)) if families else None,
    }
    return row


def choose_batch(model: GeneReconModel, args: argparse.Namespace) -> int:
    if args.batch_selector == "index":
        idx = int(args.batch_index)
    elif args.batch_selector == "largest-clades":
        idx = max(
            range(len(model.batch_metadata)),
            key=lambda i: model.batch_metadata[i].clade_count,
        )
    elif args.batch_selector == "largest-splits":
        idx = max(
            range(len(model.batch_metadata)),
            key=lambda i: model.batch_metadata[i].split_count,
        )
    else:
        idx = max(
            range(len(model.batch_metadata)),
            key=lambda i: model.batch_metadata[i].wave_count,
        )
    if idx < 0 or idx >= len(model.batch_metadata):
        raise ValueError(f"batch index {idx} outside [0, {len(model.batch_metadata)})")
    return idx


def load_theta(model: GeneReconModel, path: Path) -> None:
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        for key in ("theta", "theta_final", "model_theta"):
            value = loaded.get(key)
            if torch.is_tensor(value):
                loaded = value
                break
    if not torch.is_tensor(loaded):
        raise TypeError(f"{path} did not contain a theta tensor")
    theta = loaded.to(device=model.theta.device, dtype=model.theta.dtype)
    if tuple(theta.shape) != tuple(model.theta.shape):
        raise ValueError(
            f"theta shape mismatch: file has {tuple(theta.shape)}, "
            f"model expects {tuple(model.theta.shape)}"
        )
    with torch.no_grad():
        model.theta.copy_(theta)


def build_model(args: argparse.Namespace) -> GeneReconModel:
    theta_init_rates = None
    if args.theta_rate is not None:
        theta_init_rates = (args.theta_rate, args.theta_rate, args.theta_rate)
    model = GeneReconModel.from_alerax_families(
        str(args.species_tree),
        args.families_file,
        mode=args.mode,
        start=args.start,
        max_families=args.max_families,
        device=args.device,
        dtype=dtype_from_name(args.dtype),
        theta_init_rates=theta_init_rates,
        fixed_iters_E=args.fixed_iters_e,
        fixed_iters_Pi=args.fixed_iters_pi,
        neumann_terms=args.neumann_terms,
        adaptive_iters=args.adaptive_iters,
        convergence_check_interval=args.convergence_check_interval,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        batch_packing=args.batch_packing,
        max_wave_size=args.max_wave_size,
        max_dts_partial_rows=args.max_dts_partial_rows,
        lazy_preprocess=True,
        prefetch_batches=0,
    )
    if args.theta_path is not None:
        load_theta(model, args.theta_path)
    return model


def evict_inactive_batches(model: GeneReconModel, selected_batch: int) -> None:
    statics = getattr(model, "_batch_statics", None)
    if not isinstance(statics, list):
        return
    for idx in range(len(statics)):
        if idx != selected_batch:
            statics[idx] = None
    futures = getattr(model, "_batch_futures", None)
    if isinstance(futures, dict):
        futures.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def active_batch_indices(model: GeneReconModel) -> torch.Tensor:
    return torch.as_tensor(
        tuple(int(idx) for idx in model.current_batch_metadata.family_indices),
        dtype=torch.long,
        device=model.theta.device,
    )


def full_vector_from_active(model: GeneReconModel, active_values: torch.Tensor) -> torch.Tensor:
    idx = active_batch_indices(model)
    values = active_values.detach().reshape(-1).to(
        device=model.theta.device,
        dtype=model.theta.dtype,
    )
    if values.numel() != idx.numel():
        raise RuntimeError(
            f"active objective returned {values.numel()} values for {idx.numel()} rows"
        )
    full = torch.zeros(
        (int(model.n_families),),
        device=model.theta.device,
        dtype=model.theta.dtype,
    )
    full.index_copy_(0, idx, values)
    return full


def zero_inactive_grad(model: GeneReconModel) -> None:
    grad = model.theta.grad
    if grad is None:
        raise RuntimeError("active genewise closure did not produce theta.grad")
    idx = active_batch_indices(model)
    mask = torch.zeros((int(model.n_families),), device=grad.device, dtype=torch.bool)
    mask.index_fill_(0, idx, True)
    filtered = grad.detach().clone()
    filtered[~mask] = 0
    model.theta.grad = filtered


def grad_stats(model: GeneReconModel) -> dict[str, float]:
    grad = model.theta.grad
    if grad is None:
        return {"grad_inf": math.nan, "grad_norm": math.nan}
    grad = grad.detach()
    return {
        "grad_inf": float(grad.abs().amax().cpu()),
        "grad_norm": float(torch.linalg.vector_norm(grad).cpu()),
    }


def memory_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024.0**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024.0**3),
    }


def event_elapsed_seconds(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return start.elapsed_time(end) / 1000.0


def run_closure(model: GeneReconModel, batch_index: int) -> dict[str, Any]:
    model.select_batch(batch_index)
    model.theta.grad = None
    model.clear()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    synchronize()

    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    with torch.enable_grad(), nvtx_range(f"gpurec_profile.closure.forward.batch_{batch_index:04d}"):
        fwd_start.record()
        if model._mode == "genewise":
            loss_vec = model.nll_per_family()
            loss = loss_vec.sum()
        else:
            loss = model()
        fwd_end.record()
    with torch.enable_grad(), nvtx_range(f"gpurec_profile.closure.backward.batch_{batch_index:04d}"):
        loss.backward()
        if model._mode == "genewise":
            zero_inactive_grad(model)
        bwd_end.record()
    synchronize()
    wall_end = time.perf_counter()

    row: dict[str, Any] = {
        "operation": "closure",
        "loss_bits": float(loss.detach().cpu()),
        "forward_s": event_elapsed_seconds(fwd_start, fwd_end),
        "backward_s": event_elapsed_seconds(fwd_end, bwd_end),
        "forward_backward_s": event_elapsed_seconds(fwd_start, bwd_end),
        "wall_s": wall_end - wall_start,
        **batch_row(model, batch_index),
        **grad_stats(model),
        **memory_stats(),
    }
    stats = model.solver_stat_records()
    if stats:
        last = stats[-1]
        for key in (
            "E_iters",
            "Pi_iters",
            "Pi_wave_count",
            "E_converged",
            "Pi_converged",
        ):
            if key in last:
                row[f"solver/{key}"] = last[key]
    return row


def run_lbfgs_step(model: GeneReconModel, batch_index: int, args: argparse.Namespace) -> dict[str, Any]:
    if model._mode != "genewise":
        raise ValueError("lbfgs-step operation requires --mode genewise")
    model.select_batch(batch_index)
    model.clear()
    optimizer = BatchedLBFGS(
        [model.theta],
        lr=args.lbfgs_lr,
        max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history_size,
        max_ls=args.lbfgs_max_ls,
        line_search_fn=args.lbfgs_line_search,
        lower_bound=math.log2(args.min_rate),
        upper_bound=None if args.max_rate is None else math.log2(args.max_rate),
    )
    counts = {"grad_evals": 0, "loss_evals": 0}

    def closure() -> torch.Tensor:
        counts["grad_evals"] += 1
        with torch.no_grad():
            model.clamp_theta_(args.min_rate, args.max_rate)
        optimizer.zero_grad(set_to_none=True)
        with nvtx_range(
            f"gpurec_profile.lbfgs_step.grad_eval_{counts['grad_evals']:03d}.batch_{batch_index:04d}"
        ):
            local_loss_vec = model.nll_per_family()
            local_loss_vec.sum().backward()
        zero_inactive_grad(model)
        return full_vector_from_active(model, local_loss_vec)

    def loss_closure() -> torch.Tensor:
        counts["loss_evals"] += 1
        with torch.no_grad():
            model.clamp_theta_(args.min_rate, args.max_rate)
            with nvtx_range(
                f"gpurec_profile.lbfgs_step.loss_eval_{counts['loss_evals']:03d}.batch_{batch_index:04d}"
            ):
                local_loss_vec = model.nll_per_family()
            return full_vector_from_active(model, local_loss_vec)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    wall_start = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.enable_grad(), nvtx_range(f"gpurec_profile.lbfgs_step.batch_{batch_index:04d}"):
        start_event.record()
        loss_vec = optimizer.step(closure, loss_closure=loss_closure)
        end_event.record()
    synchronize()
    wall_end = time.perf_counter()

    state = optimizer.state.get(model.theta, {})
    row: dict[str, Any] = {
        "operation": "lbfgs-step",
        "loss_bits": float(loss_vec.detach().sum().cpu()),
        "step_s": event_elapsed_seconds(start_event, end_event),
        "wall_s": wall_end - wall_start,
        "grad_evals": counts["grad_evals"],
        "loss_evals": counts["loss_evals"],
        "closure_evals": counts["grad_evals"] + counts["loss_evals"],
        "optimizer/n_iter": int(state.get("last_n_iter", state.get("n_iter", 0))),
        **batch_row(model, batch_index),
        **grad_stats(model),
        **memory_stats(),
    }
    for key in ("last_accepted", "last_alpha"):
        value = state.get(key)
        if torch.is_tensor(value):
            active = value.index_select(0, active_batch_indices(model))
            if active.dtype == torch.bool:
                row[f"optimizer/{key}_active_true"] = int(active.sum().cpu())
                row[f"optimizer/{key}_active_count"] = int(active.numel())
            else:
                row[f"optimizer/{key}_active_min"] = float(active.min().cpu())
                row[f"optimizer/{key}_active_max"] = float(active.max().cpu())
                row[f"optimizer/{key}_active_mean"] = float(active.mean().cpu())
    return row


def print_json(row: dict[str, Any]) -> None:
    print(json.dumps(row, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not args.species_tree.exists():
        raise FileNotFoundError(args.species_tree)
    if not args.families_file.exists():
        raise FileNotFoundError(args.families_file)

    print_json(
        {
            "event": "config",
            "species_tree": str(args.species_tree),
            "families_file": str(args.families_file),
            "mode": args.mode,
            "operation": args.operation,
            "device": args.device,
            "dtype": args.dtype,
            "fixed_iters_e": args.fixed_iters_e,
            "fixed_iters_pi": args.fixed_iters_pi,
            "neumann_terms": args.neumann_terms,
            "adaptive_iters": args.adaptive_iters,
            "family_chunk_size": args.family_chunk_size,
            "clade_budget": args.clade_budget,
            "batch_packing": args.batch_packing,
            "max_wave_size": args.max_wave_size,
            "max_dts_partial_rows": args.max_dts_partial_rows,
            "theta_path": None if args.theta_path is None else str(args.theta_path),
        }
    )

    build_start = time.perf_counter()
    with nvtx_range("gpurec_profile.build_model"):
        model = build_model(args)
    synchronize()
    build_s = time.perf_counter() - build_start

    print_json(
        {
            "event": "model",
            "build_s": build_s,
            "families": int(model.n_families),
            "species": int(model.n_species),
            "batches": len(model.batch_metadata),
            "theta_shape": tuple(int(v) for v in model.theta.shape),
        }
    )

    if args.operation == "list-batches":
        for idx in range(len(model.batch_metadata)):
            print_json({"event": "batch", **batch_row(model, idx)})
        model.close()
        return

    selected_batch = choose_batch(model, args)
    model.select_batch(selected_batch)
    if args.evict_inactive_batches:
        evict_inactive_batches(model, selected_batch)
    print_json({"event": "selected_batch", **batch_row(model, selected_batch)})

    run = (
        (lambda: run_closure(model, selected_batch))
        if args.operation == "closure"
        else (lambda: run_lbfgs_step(model, selected_batch, args))
    )

    for warmup_idx in range(args.warmup_runs):
        row = run()
        print_json({"event": "warmup", "idx": warmup_idx, **row})

    measured: list[dict[str, Any]] = []
    cuda_profiler_start(args.cuda_profiler_api)
    try:
        for profile_idx in range(args.profile_runs):
            row = run()
            measured.append(row)
            print_json({"event": "measured", "idx": profile_idx, **row})
    finally:
        cuda_profiler_stop(args.cuda_profiler_api)
        model.close()

    if measured:
        time_key = "forward_backward_s" if args.operation == "closure" else "step_s"
        print_json(
            {
                "event": "summary",
                "operation": args.operation,
                "profile_runs": len(measured),
                f"median_{time_key}": float(
                    torch.median(torch.tensor([float(row[time_key]) for row in measured]))
                ),
                "max_peak_allocated_gib": max(
                    float(row.get("peak_allocated_gib", math.nan)) for row in measured
                ),
                "max_peak_reserved_gib": max(
                    float(row.get("peak_reserved_gib", math.nan)) for row in measured
                ),
            }
        )


if __name__ == "__main__":
    main()

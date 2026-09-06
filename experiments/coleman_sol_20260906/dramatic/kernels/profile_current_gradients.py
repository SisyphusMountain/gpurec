"""Profile two current exact/exact 200-family gradients at saved, paid rate points.

This is deliberately an experiment-only driver.  It does not replace a model method or edit
production source.  Small, scoped wrappers put CUDA events around the production forward and
reverse calls and retain the already-computed adjoint masks long enough to count pruning after
the profiled call.  All module globals are restored in ``finally``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.profiler import ProfilerActivity, profile

from gpurec.api.model import GeneReconModel
from gpurec.config import GpurecConfig


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _kernel_rows(profiler) -> list[dict]:
    # ``key_averages().device_time_total`` includes a CPU operator's CUDA children and therefore
    # double-counts them beside the CUDA events themselves.  Aggregate only raw CUDA events.
    aggregate: dict[str, list[float | int]] = {}
    for event in profiler.events():
        if getattr(event.device_type, "name", "") != "CUDA":
            continue
        elapsed = float(event.time_range.elapsed_us())
        item = aggregate.setdefault(event.name, [0.0, 0])
        item[0] += elapsed
        item[1] += 1
    rows = [
        {
            "key": name,
            "count": int(value[1]),
            "device_time_us": float(value[0]),
            "self_device_time_us": float(value[0]),
        }
        for name, value in aggregate.items()
    ]
    rows.sort(key=lambda row: row["device_time_us"], reverse=True)
    return rows


@contextmanager
def _timed_production_regions():
    """Add asynchronous timings and retain masks without changing the computation."""
    import gpurec.api._execution as execution
    import gpurec.api._implicit_grad as implicit

    original_forward = execution.solve_resident_e_pi
    original_reverse = execution.implicit_grad_loglik_vjp_wave
    original_mask = implicit.compute_active_adjoint_row_mask
    state = {"forward_events": [], "reverse_events": [], "masks": []}

    def timed_forward(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_forward(*args, **kwargs)
        end.record()
        state["forward_events"].append((start, end))
        return result

    def timed_reverse(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_reverse(*args, **kwargs)
        end.record()
        state["reverse_events"].append((start, end))
        return result

    def retained_mask(*args, **kwargs):
        result = original_mask(*args, **kwargs)
        state["masks"].append(result)
        return result

    execution.solve_resident_e_pi = timed_forward
    execution.implicit_grad_loglik_vjp_wave = timed_reverse
    implicit.compute_active_adjoint_row_mask = retained_mask
    try:
        yield state
    finally:
        execution.solve_resident_e_pi = original_forward
        execution.implicit_grad_loglik_vjp_wave = original_reverse
        implicit.compute_active_adjoint_row_mask = original_mask


def _profile_one(model, theta, label: str, output: Path) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    total_start, total_end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    wall_start = time.perf_counter()
    with _timed_production_regions() as regions:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            total_start.record()
            loss, gradient, receiver_gradient = model.genewise_loss_vector_and_grad(
                theta=theta, need_grad=True
            )
            total_end.record()
            torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start

    if receiver_gradient is not None:
        raise RuntimeError("rate-fit profile unexpectedly computed a receiver gradient")
    masks = regions["masks"]
    active_rows = sum(int(mask.sum().item()) for mask in masks)
    total_rows = sum(int(mask.numel()) for mask in masks)
    forward_ms = sum(float(a.elapsed_time(b)) for a, b in regions["forward_events"])
    reverse_ms = sum(float(a.elapsed_time(b)) for a, b in regions["reverse_events"])
    total_ms = float(total_start.elapsed_time(total_end))
    kernels = _kernel_rows(prof)
    chrome = output / f"{label}.chrome.json"
    prof.export_chrome_trace(str(chrome))
    pi_states = [static.pi_forward_state for static in model.batch_statics]
    wide_rows = sum(int(state.wide_row_total) for state in pi_states if state is not None)
    return {
        "label": label,
        "total_cuda_event_ms": total_ms,
        "forward_cuda_event_ms": forward_ms,
        "reverse_cuda_event_ms": reverse_ms,
        "other_cuda_event_ms": total_ms - forward_ms - reverse_ms,
        "wall_profiled_seconds": wall_seconds,
        "loss_bits": float(loss.double().sum()),
        "gradient_abs_max": float(gradient.abs().max()),
        "adjoint_wave_masks": len(masks),
        "adjoint_rows_total": total_rows,
        "adjoint_rows_active": active_rows,
        "adjoint_rows_pruned": total_rows - active_rows,
        "adjoint_active_fraction": active_rows / total_rows,
        "forward_wide_fallback_rows": wide_rows,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "profiler_kernel_time_us_sum": sum(row["device_time_us"] for row in kernels),
        "kernels": kernels,
        "chrome_trace": str(chrome),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared",
        default="experiments/coleman_sol_20260906/em/hybrid_shared_200_v2.pt",
    )
    parser.add_argument(
        "--late",
        default="experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200_complete.pt",
    )
    parser.add_argument("--output", default=str(HERE))
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    if args.warmups < 1 or args.warmups > 3:
        raise ValueError("warmups must be in [1, 3]")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shared = torch.load(args.shared, map_location="cpu", weights_only=False)
    late = torch.load(args.late, map_location="cpu", weights_only=False)
    if shared.get("schema") != "gpurec.em_hybrid_shared.v2" or len(shared["paths"]) != 200:
        raise ValueError("expected the point-consistent 200-family EM2 artifact")
    if not str(late.get("schema", "")).startswith("gpurec.hybrid_diagnostic_trace.v1"):
        raise ValueError("expected the completed paired diagnostic trace")
    if late["paths"] != shared["paths"]:
        raise ValueError("late theta and EM2 artifact do not describe the same families")

    config = GpurecConfig.genewise_reference()
    solver = replace(config.solver, pi_iters=16, neumann_terms=16)
    model = GeneReconModel(
        shared["metadata"]["species"],
        shared["paths"],
        mode="genewise",
        device="cuda",
        dtype="float32",
        config=config,
        solver_options=solver,
        clade_budget=int(shared["metadata"]["clade_budget"]),
    )
    model.receiver_weights.requires_grad_(False)
    points = {
        "em2_endpoint": shared["theta_native"]["theta2"].to(
            device="cuda", dtype=torch.float32
        ),
        "late_native_final": late["arms"]["native"]["result"]["theta"].to(
            device="cuda", dtype=torch.float32
        ),
    }
    if any(tuple(theta.shape) != (200, 3) for theta in points.values()):
        raise ValueError("profile points must have shape [200, 3]")

    # Compile every current path and remove first-use allocation effects before either of the
    # exactly two profiled gradients.  Warmups are intentionally ordinary, uninstrumented calls.
    for theta in points.values():
        for _ in range(args.warmups):
            model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
            torch.cuda.synchronize()

    report = {
        "schema": "gpurec.current_exact_gradient_profile.v1",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "model_dtype": str(model.theta.dtype),
        "accumulator_dtype": str(model.accumulator_dtype),
        "families": len(shared["paths"]),
        "species": int(model.species_helpers["S"]),
        "family_clades": int(shared["family_clades"].sum()),
        "clade_budget": int(shared["metadata"]["clade_budget"]),
        "batches": len(model.batch_statics),
        "waves": sum(len(static.wave_layout["wave_metas"]) for static in model.batch_statics),
        "forward_gene_split_cache": bool(model.forward_gene_split_ok),
        "solver": dict(shared["metadata"]["solver"]),
        "inputs": {"shared": args.shared, "late": args.late},
        "source_sha256": {
            str(path.relative_to(REPO)): _sha256(path)
            for path in (
                REPO / "gpurec/core/inference/forward.py",
                REPO / "gpurec/core/kernels/pi_forward.py",
                REPO / "gpurec/api/_implicit_grad.py",
                REPO / "gpurec/core/kernels/wave_backward.py",
                REPO / "gpurec/core/kernels/wave_backward_kernels.py",
            )
        },
        "profiles": [],
    }
    for label, theta in points.items():
        result = _profile_one(model, theta, label, output)
        report["profiles"].append(result)
        print(
            f"[{label}] total={result['total_cuda_event_ms']:.1f} ms "
            f"forward={result['forward_cuda_event_ms']:.1f} ms "
            f"reverse={result['reverse_cuda_event_ms']:.1f} ms "
            f"active={result['adjoint_rows_active']:,}/{result['adjoint_rows_total']:,}",
            flush=True,
        )
        for row in result["kernels"][:15]:
            print(
                f"  {100 * row['device_time_us'] / result['profiler_kernel_time_us_sum']:5.1f}% "
                f"{row['device_time_us'] / 1e3:8.1f} ms {row['count']:6d} {row['key'][:100]}",
                flush=True,
            )
    destination = output / "current_gradient_profiles.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

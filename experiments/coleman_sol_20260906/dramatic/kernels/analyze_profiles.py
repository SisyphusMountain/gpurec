"""Reduce the two profiler traces to non-overlapping CUDA work and Amdahl gates."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
PROFILE_JSON = HERE / "current_gradient_profiles.json"


def _gpu_events(path: Path):
    trace = json.loads(path.read_text())["traceEvents"]
    return [
        event
        for event in trace
        if event.get("ph") == "X"
        and event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]


def _union_time_us(events) -> tuple[float, float]:
    intervals = sorted((float(event["ts"]), float(event["ts"] + event["dur"])) for event in events)
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    busy = sum(end - start for start, end in merged)
    span = merged[-1][1] - merged[0][0]
    return busy, span


def _aggregate(events) -> list[dict]:
    aggregate = defaultdict(lambda: [0.0, 0])
    for event in events:
        aggregate[event["name"]][0] += float(event["dur"])
        aggregate[event["name"]][1] += 1
    rows = [
        {"name": name, "time_ms": value[0] / 1000.0, "launches": value[1]}
        for name, value in aggregate.items()
    ]
    return sorted(rows, key=lambda row: row["time_ms"], reverse=True)


def _wave_shape(events, kernel: str) -> dict:
    selected = [event for event in events if event["name"] == kernel]
    widths = torch.tensor([event["args"]["grid"][0] for event in selected], dtype=torch.float64)
    durations = torch.tensor([event["dur"] for event in selected], dtype=torch.float64)
    design = torch.stack((torch.ones_like(widths), widths), dim=1)
    coefficient = torch.linalg.lstsq(design, durations.unsqueeze(1)).solution.flatten()
    fit = design @ coefficient
    r2 = 1.0 - ((durations - fit) ** 2).sum() / ((durations - durations.mean()) ** 2).sum()
    bins = []
    for low, high in ((1, 8), (9, 32), (33, 128), (129, 512), (513, 4096), (4097, 10**9)):
        chosen = (widths >= low) & (widths <= high)
        if bool(chosen.any()):
            bins.append(
                {
                    "low": low,
                    "high": high,
                    "launches": int(chosen.sum()),
                    "time_ms": float(durations[chosen].sum() / 1000.0),
                    "median_us": float(durations[chosen].median()),
                }
            )
    return {
        "launches": len(selected),
        "rows_sum": int(widths.sum()),
        "rows_median": float(widths.median()),
        "rows_mean": float(widths.mean()),
        "rows_max": int(widths.max()),
        "fit_intercept_us": float(coefficient[0]),
        "fit_us_per_row": float(coefficient[1]),
        "fit_r2": float(r2),
        "bins": bins,
    }


def main() -> None:
    raw = json.loads(PROFILE_JSON.read_text())
    output = {"schema": "gpurec.current_exact_gradient_profile_analysis.v1", "profiles": []}
    for profile_row in raw["profiles"]:
        events = _gpu_events(Path(profile_row["chrome_trace"]))
        busy_us, span_us = _union_time_us(events)
        kernels = _aggregate(events)
        by_name = {row["name"]: row for row in kernels}
        dominant_reverse_names = (
            "_solve_reconciliation_self_loop_transpose_row_kernel",
            "_accumulate_gene_split_event_vjp_kernel",
            "_accumulate_transfer_subtree_vjp_kernel",
            "_accumulate_reconciliation_event_vjp_kernel",
        )
        dominant_reverse_ms = sum(by_name[name]["time_ms"] for name in dominant_reverse_names)
        total_ms = profile_row["total_cuda_event_ms"]
        forward_ms = profile_row["forward_cuda_event_ms"]
        reverse_ms = profile_row["reverse_cuda_event_ms"]
        target_2x_ms = total_ms / 2.0
        outside_regions_ms = total_ms - forward_ms - reverse_ms
        output["profiles"].append(
            {
                "label": profile_row["label"],
                "total_cuda_event_ms": total_ms,
                "forward_cuda_event_ms": forward_ms,
                "reverse_cuda_event_ms": reverse_ms,
                "forward_fraction": forward_ms / total_ms,
                "reverse_fraction": reverse_ms / total_ms,
                "gpu_busy_ms": busy_us / 1000.0,
                "gpu_span_ms": span_us / 1000.0,
                "gpu_idle_fraction": (span_us - busy_us) / span_us,
                "active_fraction": profile_row["adjoint_active_fraction"],
                "wide_rows": profile_row["forward_wide_fallback_rows"],
                "top_cuda_events": kernels[:24],
                "exact_forward_wave_shape": _wave_shape(events, "_exact_tree_pi_self_loop_kernel"),
                "exact_reverse_wave_shape": _wave_shape(
                    events, "_solve_reconciliation_self_loop_transpose_row_kernel"
                ),
                "amdahl": {
                    "max_speedup_if_reverse_free": total_ms / (total_ms - reverse_ms),
                    "reverse_speedup_required_for_2x": reverse_ms
                    / (target_2x_ms - (total_ms - reverse_ms)),
                    "dominant_four_reverse_ms": dominant_reverse_ms,
                    "max_speedup_if_dominant_four_free": total_ms / (total_ms - dominant_reverse_ms),
                    "dominant_four_speedup_required_for_2x": dominant_reverse_ms
                    / (target_2x_ms - (total_ms - dominant_reverse_ms)),
                    "semiring_forward_multiple_for_2x": (target_2x_ms - outside_regions_ms)
                    / forward_ms,
                    "semiring_forward_multiple_for_2x_with_20pct_fewer_passes": (
                        (0.625 * total_ms - outside_regions_ms) / forward_ms
                    ),
                },
            }
        )
    destination = HERE / "current_gradient_analysis.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()

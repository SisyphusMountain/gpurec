from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


RATE_COLUMNS = (("D", 0), ("L", 1), ("T", 2))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else out


def tensor_stats(prefix: str, tensor: torch.Tensor | None) -> dict[str, float]:
    if tensor is None:
        return {}
    flat = tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    if flat.numel() == 0:
        return {}
    abs_flat = flat.abs()
    return {
        f"{prefix}/min": float(flat.min()),
        f"{prefix}/max": float(flat.max()),
        f"{prefix}/median": float(torch.quantile(flat, 0.5)),
        f"{prefix}/mean": float(flat.mean()),
        f"{prefix}/abs_min": float(abs_flat.min()),
        f"{prefix}/abs_max": float(abs_flat.max()),
        f"{prefix}/abs_mean": float(abs_flat.mean()),
        f"{prefix}/norm": float(torch.linalg.vector_norm(flat)),
        f"{prefix}/inf": float(abs_flat.max()),
    }


def rates_and_survival_probability(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return D/T/L rate weights and the model's normalized survival probability."""
    theta_cpu = theta.detach().reshape(-1, 3).to(device="cpu", dtype=torch.float64)
    rates = torch.exp2(theta_cpu)
    p_s = 1.0 / (1.0 + rates.sum(dim=1))
    return rates, p_s


def parameter_stats(theta: torch.Tensor) -> dict[str, float]:
    theta_cpu = theta.detach().reshape(-1, 3).to(device="cpu", dtype=torch.float64)
    rates, p_s = rates_and_survival_probability(theta_cpu)
    out: dict[str, float] = {}
    for name, col in RATE_COLUMNS:
        out.update(tensor_stats(f"theta/{name}", theta_cpu[:, col]))
        out.update(tensor_stats(f"rate/{name}", rates[:, col]))
    out.update(tensor_stats("pS", p_s))
    return out


def solver_stats(model: Any) -> dict[str, float]:
    stats = model.solver_stat_records() if hasattr(model, "solver_stat_records") else []
    if not stats:
        return {}
    e_iterations = [int(row.get("E_iterations", 0)) for row in stats]
    pi_caps = [int(row.get("Pi_max_iterations", 0)) for row in stats]
    pi_wave_iterations = [
        int(value)
        for row in stats
        for value in row.get("Pi_wave_iterations", []) or []
    ]
    neumann = [int(row.get("Neumann_terms", 0)) for row in stats if "Neumann_terms" in row]
    grad_converged = [
        bool(row.get("Gradient_converged"))
        for row in stats
        if "Gradient_converged" in row
    ]
    pi_wave_count = sum(int(row.get("Pi_wave_count", 0)) for row in stats)
    pi_converged = sum(int(row.get("Pi_converged_waves", 0)) for row in stats)
    out = {
        "solver/batches_with_stats": float(len(stats)),
        "solver/e_iterations_max": float(max(e_iterations, default=0)),
        "solver/e_iterations_mean": float(sum(e_iterations) / max(len(e_iterations), 1)),
        "solver/pi_iteration_limit_max": float(max(pi_caps, default=0)),
        "solver/pi_wave_count": float(pi_wave_count),
        "solver/pi_converged_waves": float(pi_converged),
        "solver/neumann_terms_max": float(max(neumann, default=0)),
    }
    if pi_wave_iterations:
        out["solver/pi_iterations_max"] = float(max(pi_wave_iterations))
        out["solver/pi_iterations_mean"] = float(
            sum(pi_wave_iterations) / len(pi_wave_iterations)
        )
    if grad_converged:
        out["solver/gradient_converged_batches"] = float(sum(grad_converged))
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

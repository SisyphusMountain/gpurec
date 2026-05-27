"""Internal workflow diagnostic serialization and summary-stat helpers.

Optimization and sampling use this module for strict JSON/CSV writes and
likelihood, gradient, parameter, and solver summaries. User-facing consumers
should prefer `summary.json`, `history.jsonl`, and the CLI inspection commands.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


RATE_COLUMNS = (("D", 0), ("L", 1), ("T", 2))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def json_dumps_strict(value: Any, **kwargs: Any) -> str:
    """Serialize diagnostics as standards-compliant JSON after sanitization.

    The "strict" part refers to the emitted JSON grammar: Python-only NaN and
    Infinity tokens are never written.  Non-finite floats are intentionally
    replaced with JSON null before dumping so diagnostics can be parsed by
    strict JSON readers while optimization code still keeps numeric values in
    memory.
    """
    return json.dumps(_json_safe(value), allow_nan=False, **kwargs)


def write_json_strict(
    path: Path,
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json_dumps_strict(value, indent=indent, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


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


def solver_stats(model: Any) -> dict[str, float | int]:
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
    e_adjoint_iterations = [
        int(row.get("E_adjoint_iterations", 0))
        for row in stats
        if "E_adjoint_iterations" in row
    ]
    e_adjoint_success = [
        bool(row.get("E_adjoint_success"))
        for row in stats
        if "E_adjoint_success" in row
    ]
    pi_adjoint_warmstart_enabled = [
        bool(row.get("Pi_adjoint_warmstart_enabled"))
        for row in stats
        if "Pi_adjoint_warmstart_enabled" in row
    ]
    pi_adjoint_warmstart_used = [
        bool(row.get("Pi_adjoint_warmstart_used"))
        for row in stats
        if "Pi_adjoint_warmstart_used" in row
    ]
    pi_adjoint_residual_absmax = []
    pi_adjoint_residual_relmax = []
    pi_adjoint_residual_wave_count = []
    e_adjoint_rel_res = []
    for row in stats:
        if "E_adjoint_rel_res" in row:
            value = float(row.get("E_adjoint_rel_res", math.nan))
            if math.isfinite(value):
                e_adjoint_rel_res.append(value)
        if "Pi_adjoint_residual_absmax" in row:
            value = float(row.get("Pi_adjoint_residual_absmax", math.nan))
            if math.isfinite(value):
                pi_adjoint_residual_absmax.append(value)
        if "Pi_adjoint_residual_relmax" in row:
            value = float(row.get("Pi_adjoint_residual_relmax", math.nan))
            if math.isfinite(value):
                pi_adjoint_residual_relmax.append(value)
        if "Pi_adjoint_residual_wave_count" in row:
            pi_adjoint_residual_wave_count.append(
                int(row.get("Pi_adjoint_residual_wave_count", 0))
            )
    pi_wave_count = sum(int(row.get("Pi_wave_count", 0)) for row in stats)
    pi_converged = sum(int(row.get("Pi_converged_waves", 0)) for row in stats)
    out = {
        "solver/batches_with_stats": len(stats),
        "solver/e_iterations_max": float(max(e_iterations, default=0)),
        "solver/e_iterations_mean": float(sum(e_iterations) / max(len(e_iterations), 1)),
        "solver/pi_iteration_limit_max": float(max(pi_caps, default=0)),
        "solver/pi_wave_count": pi_wave_count,
        "solver/pi_converged_waves": pi_converged,
        "solver/neumann_terms_max": float(max(neumann, default=0)),
    }
    if pi_wave_iterations:
        out["solver/pi_iterations_max"] = float(max(pi_wave_iterations))
        out["solver/pi_iterations_mean"] = float(
            sum(pi_wave_iterations) / len(pi_wave_iterations)
        )
    if grad_converged:
        out["solver/gradient_converged_batches"] = int(sum(grad_converged))
    if e_adjoint_iterations:
        out["solver/e_adjoint_iterations_max"] = float(max(e_adjoint_iterations))
        out["solver/e_adjoint_iterations_mean"] = float(
            sum(e_adjoint_iterations) / len(e_adjoint_iterations)
        )
    if e_adjoint_rel_res:
        out["solver/e_adjoint_rel_res_max"] = float(max(e_adjoint_rel_res))
        out["solver/e_adjoint_rel_res_mean"] = float(
            sum(e_adjoint_rel_res) / len(e_adjoint_rel_res)
        )
    if e_adjoint_success:
        success_count = sum(e_adjoint_success)
        out["solver/e_adjoint_success_batches"] = int(success_count)
        out["solver/e_adjoint_failed_batches"] = int(
            len(e_adjoint_success) - success_count
        )
    if pi_adjoint_warmstart_enabled:
        out["solver/pi_adjoint_warmstart_enabled_batches"] = int(
            sum(pi_adjoint_warmstart_enabled)
        )
    if pi_adjoint_warmstart_used:
        out["solver/pi_adjoint_warmstart_used_batches"] = int(
            sum(pi_adjoint_warmstart_used)
        )
    if pi_adjoint_residual_absmax:
        out["solver/pi_adjoint_residual_absmax_max"] = float(
            max(pi_adjoint_residual_absmax)
        )
        out["solver/pi_adjoint_residual_absmax_mean"] = float(
            sum(pi_adjoint_residual_absmax)
            / len(pi_adjoint_residual_absmax)
        )
    if pi_adjoint_residual_relmax:
        out["solver/pi_adjoint_residual_relmax_max"] = float(
            max(pi_adjoint_residual_relmax)
        )
        out["solver/pi_adjoint_residual_relmax_mean"] = float(
            sum(pi_adjoint_residual_relmax)
            / len(pi_adjoint_residual_relmax)
        )
    if pi_adjoint_residual_wave_count:
        out["solver/pi_adjoint_residual_checked_batches"] = int(
            len(pi_adjoint_residual_wave_count)
        )
        out["solver/pi_adjoint_residual_wave_count"] = int(
            sum(pi_adjoint_residual_wave_count)
        )
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json_dumps_strict(row, sort_keys=True) + "\n")


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

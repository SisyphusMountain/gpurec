"""Checkout-local HOGENOM gradient convergence benchmark.

The benchmark measures how quickly the current implicit-gradient path
approaches a high-Neumann reference near the HOGENOM optimum.  When the
checkpoint contains LBFGSB history, it also reconstructs the previous optimizer
theta and uses its Pi adjoint as the initial guess for the current Pi
self-loop fixed-point solve.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpurec.api.autograd import evaluate_resident_gradient_forward  # noqa: E402
from gpurec.optimization.implicit_grad import (  # noqa: E402
    implicit_grad_loglik_vjp_wave,
)
from gpurec.workflow.checkpoint import (  # noqa: E402
    load_checkpoint,
    restore_model_theta,
    validate_checkpoint_model_compatibility,
)
from gpurec.workflow.config import RunConfig  # noqa: E402
from gpurec.workflow.model_factory import build_alerax_workflow_model  # noqa: E402


HOGENOM_DIR = REPO / "tests" / "data" / "HOGENOM" / "hogenom"
DEFAULT_SPECIES_TREE = HOGENOM_DIR / "hogenom_S.tree"
DEFAULT_FAMILIES_FILE = HOGENOM_DIR / "hogenom_families.local.txt"
DEFAULT_CHECKPOINTS = (
    (
        "plus300",
        Path(
            "/tmp/gpurec_hogenom_specieswise_lbfgsb_high32_from_init_continue/"
            "checkpoints/latest.pt"
        ),
    ),
    (
        "plus120",
        Path(
            "/tmp/gpurec_hogenom_specieswise_lbfgsb_high32_from_init_continue2/"
            "checkpoints/latest.pt"
        ),
    ),
    (
        "near_optimum",
        Path(
            "/tmp/gpurec_hogenom_specieswise_lbfgsb_high32_from_init_continue4/"
            "checkpoints/best.pt"
        ),
    ),
)


@dataclass(frozen=True)
class GradientEvaluation:
    loss_bits: float
    grad: torch.Tensor
    pi_adjoint_by_batch: list[torch.Tensor]
    stats_by_batch: list[dict[str, Any]]
    elapsed_s: float


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one comma-separated integer")
    return sorted({item for item in items if item > 0})


def _parse_float_list(value: str) -> list[float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one comma-separated float")
    for item in items:
        if not math.isfinite(item) or item < 0.0:
            raise ValueError("thresholds must be finite non-negative values")
    return sorted(set(items), reverse=True)


def _parse_checkpoint_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    if not values:
        return [(label, path) for label, path in DEFAULT_CHECKPOINTS]

    specs: list[tuple[str, Path]] = []
    for raw in values:
        if "=" in raw:
            label, path_text = raw.split("=", 1)
            label = label.strip()
            if not label:
                raise ValueError(f"checkpoint label is empty in {raw!r}")
        else:
            path_text = raw
            label = Path(raw).stem
        specs.append((label, Path(path_text).expanduser()))
    return specs


def _checkpoint_step(payload: dict[str, Any]) -> int:
    next_step = payload.get("next_step", payload.get("step", 0))
    if isinstance(next_step, bool) or not isinstance(next_step, int):
        raise RuntimeError("checkpoint next_step must be an integer")
    return int(next_step)


def _run_config_from_args(
    args: argparse.Namespace,
    *,
    first_checkpoint: Path,
    checkpoint_step: int,
    max_terms: int,
) -> RunConfig:
    return RunConfig(
        species_tree=args.species_tree,
        families_file=args.families_file,
        out_dir=args.out_dir,
        mode="specieswise",
        device=args.device,
        dtype=args.dtype,
        optimizer="projected-sgd",
        steps=max(1, checkpoint_step),
        lr=args.lr,
        fixed_iters_e=args.forward_iters,
        fixed_iters_pi=args.forward_iters,
        neumann_terms=max_terms,
        adaptive_iters=False,
        adaptive_neumann_terms=False,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        batch_packing=args.batch_packing,
        max_wave_size=args.max_wave_size,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
        checkpoint_every=0,
        loss_patience=0,
        best_likelihood_patience=0,
        final_check_iters=0,
        resume_from=first_checkpoint,
    )


def _origination_probs_for_static(static: Any) -> torch.Tensor | None:
    prior = getattr(static, "origination_prior", None)
    if prior is None:
        return getattr(static, "origination_probs", None)
    return prior.probs


def _clear_runtime_state(model: Any) -> None:
    model.theta.grad = None
    for static in model.cached_static_states:
        static.warm_E = None
        static.last_solver_stats = None


def _set_model_theta(model: Any, theta: torch.Tensor) -> torch.Tensor:
    theta_device = theta.to(device=model.theta.device, dtype=model.theta.dtype)
    with torch.no_grad():
        model.theta.copy_(theta_device)
        model.theta.grad = None
    model.clear()
    return model.theta.detach().clone()


def _evaluate_gradient(
    model: Any,
    theta: torch.Tensor,
    *,
    terms: int,
    fixed_forward_iters: int,
    pi_adjoint_warm_by_batch: list[torch.Tensor] | None = None,
    capture_pi_adjoint: bool = False,
) -> GradientEvaluation:
    if model.mode != "specieswise":
        raise RuntimeError("this benchmark currently supports specieswise mode only")
    model.configure_solver_iterations(
        fixed_iters_E=fixed_forward_iters,
        fixed_iters_Pi=fixed_forward_iters,
        neumann_terms=terms,
        adaptive_neumann_terms=False,
    )
    theta_device = _set_model_theta(model, theta)
    metadata = model.materialize_batches()
    statics = model.cached_static_states
    if len(statics) != len(metadata):
        raise RuntimeError("cached static state count does not match batch metadata")
    if pi_adjoint_warm_by_batch is not None and len(pi_adjoint_warm_by_batch) != len(statics):
        raise RuntimeError("warm Pi-adjoint count does not match batch count")

    _clear_runtime_state(model)
    total_loss = torch.zeros((), device=theta_device.device, dtype=theta_device.dtype)
    total_grad = torch.zeros_like(theta_device)
    pi_states: list[torch.Tensor] = []
    stats_rows: list[dict[str, Any]] = []

    _synchronize()
    started = time.perf_counter()
    for batch_idx, static in enumerate(statics):
        batch_started = time.perf_counter()
        gradient_forward = evaluate_resident_gradient_forward(
            static,
            theta_device,
            warm_start_E=None,
        )
        solve = gradient_forward.solve
        warm = (
            None
            if pi_adjoint_warm_by_batch is None
            else pi_adjoint_warm_by_batch[batch_idx]
        )
        grad_theta, stats, aux = implicit_grad_loglik_vjp_wave(
            static.wave_layout,
            static.species_helpers,
            Pi_star_wave=solve.pi_out["Pi_wave_ordered"],
            Pibar_star_wave=solve.pi_out["Pibar_wave_ordered"],
            E_star=solve.e_out["E"],
            Ebar=solve.e_out["E_bar"],
            E_s1=solve.e_out["E_s1"],
            E_s2=solve.e_out["E_s2"],
            log_pS=solve.log_p_s,
            log_pD=solve.log_p_d,
            log_pL=solve.log_p_l,
            max_transfer_mat=solve.max_transfer,
            root_clade_ids_perm=static.wave_layout["root_clade_ids"],
            theta=solve.theta,
            unnorm_row_max=static.unnorm_row_max,
            specieswise=static.specieswise,
            device=static.device,
            dtype=static.dtype,
            neumann_terms=terms,
            use_pruning=static.use_pruning,
            pruning_threshold=static.pruning_threshold,
            ancestors_T=static.ancestors_T,
            family_idx=static.wave_layout["family_idx"] if static.genewise else None,
            uniform_pibar_row_max=solve.pi_out.get("uniform_pibar_row_max"),
            origination_probs=_origination_probs_for_static(static),
            origination_probs_prepared=True,
            genewise=static.genewise,
            pi_adjoint_initial_guess=warm,
            return_aux=True,
        )
        total_loss = total_loss + gradient_forward.loss_vec.sum().detach().to(
            device=total_loss.device,
            dtype=total_loss.dtype,
        )
        total_grad = total_grad + grad_theta.detach().to(
            device=total_grad.device,
            dtype=total_grad.dtype,
        )
        if capture_pi_adjoint:
            pi_states.append(aux["pi_adjoint"].detach().clone())
        _synchronize()
        forward_stats = dict(static.last_solver_stats or {})
        stats_rows.append(
            {
                "batch_index": int(batch_idx),
                "terms": int(terms),
                "batch_elapsed_s": time.perf_counter() - batch_started,
                "E_iterations": forward_stats.get("E_iterations"),
                "Pi_max_iterations": forward_stats.get("Pi_max_iterations"),
                "Pi_wave_count": forward_stats.get("Pi_wave_count"),
                "method": stats.method,
                "E_adjoint_iterations": int(stats.iters),
                "E_adjoint_rel_res": float(stats.rel_res),
                "E_adjoint_success": bool(stats.success),
                "Pi_adjoint_warm_start": bool(aux.get("used_pi_initial_guess", False)),
            }
        )

    _synchronize()
    return GradientEvaluation(
        loss_bits=float(total_loss.detach().cpu()),
        grad=total_grad.detach().clone(),
        pi_adjoint_by_batch=pi_states,
        stats_by_batch=stats_rows,
        elapsed_s=time.perf_counter() - started,
    )


def _project_theta(theta: torch.Tensor, *, lower: float, upper: float) -> torch.Tensor:
    return torch.clamp(theta, min=lower, max=upper)


def _projected_gradient(
    theta: torch.Tensor,
    grad: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    return theta - _project_theta(theta - grad, lower=lower, upper=upper)


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_flat = left.reshape(-1).to(dtype=torch.float64)
    right_flat = right.reshape(-1).to(dtype=torch.float64)
    left_norm = torch.linalg.vector_norm(left_flat)
    right_norm = torch.linalg.vector_norm(right_flat)
    denom = left_norm * right_norm
    if _safe_float(denom) == 0.0:
        return None
    return _safe_float(torch.dot(left_flat, right_flat) / denom)


def _gradient_metrics(
    *,
    theta: torch.Tensor,
    grad: torch.Tensor,
    reference_grad: torch.Tensor,
    lower: float,
    upper: float,
) -> dict[str, float]:
    grad_cpu = grad.detach().cpu().to(dtype=torch.float64)
    ref_cpu = reference_grad.detach().cpu().to(dtype=torch.float64)
    theta_cpu = theta.detach().cpu().to(dtype=torch.float64)
    diff = grad_cpu - ref_cpu
    grad_norm = torch.linalg.vector_norm(grad_cpu)
    ref_norm = torch.linalg.vector_norm(ref_cpu)
    diff_norm = torch.linalg.vector_norm(diff)
    grad_inf = grad_cpu.abs().amax() if grad_cpu.numel() else grad_cpu.new_zeros(())
    ref_inf = ref_cpu.abs().amax() if ref_cpu.numel() else ref_cpu.new_zeros(())
    diff_inf = diff.abs().amax() if diff.numel() else diff.new_zeros(())
    projected = _projected_gradient(theta_cpu, grad_cpu, lower=lower, upper=upper)
    ref_projected = _projected_gradient(theta_cpu, ref_cpu, lower=lower, upper=upper)
    projected_diff = projected - ref_projected
    projected_norm = torch.linalg.vector_norm(projected)
    ref_projected_norm = torch.linalg.vector_norm(ref_projected)
    projected_diff_norm = torch.linalg.vector_norm(projected_diff)
    ref_direction = torch.dot(ref_cpu.reshape(-1), -ref_projected.reshape(-1))
    grad_direction = torch.dot(grad_cpu.reshape(-1), -ref_projected.reshape(-1))
    return {
        "grad_inf": _safe_float(grad_inf),
        "grad_l2": _safe_float(grad_norm),
        "reference_grad_inf": _safe_float(ref_inf),
        "reference_grad_l2": _safe_float(ref_norm),
        "error_max_abs": _safe_float(diff_inf),
        "error_l2": _safe_float(diff_norm),
        "error_rel_linf": _safe_float(diff_inf / max(_safe_float(ref_inf), 1.0)),
        "error_rel_l2": _safe_float(diff_norm / max(_safe_float(ref_norm), 1.0)),
        "cosine": _cosine(grad_cpu, ref_cpu),
        "projected_grad_inf": _safe_float(
            projected.abs().amax() if projected.numel() else projected.new_zeros(())
        ),
        "reference_projected_grad_inf": _safe_float(
            ref_projected.abs().amax()
            if ref_projected.numel()
            else ref_projected.new_zeros(())
        ),
        "projected_error_l2": _safe_float(projected_diff_norm),
        "projected_error_rel_l2": _safe_float(
            projected_diff_norm / max(_safe_float(ref_projected_norm), 1.0)
        ),
        "projected_cosine": _cosine(projected, ref_projected),
        "reference_directional_derivative": _safe_float(ref_direction),
        "directional_derivative": _safe_float(grad_direction),
        "directional_derivative_rel_error": _safe_float(
            torch.abs(grad_direction - ref_direction)
            / max(abs(_safe_float(ref_direction)), 1.0)
        ),
        "projected_grad_l2": _safe_float(projected_norm),
        "reference_projected_grad_l2": _safe_float(ref_projected_norm),
    }


def _aggregate_stats(stats_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats_rows:
        return {}
    adjoint_iters = [
        int(row["E_adjoint_iterations"])
        for row in stats_rows
        if row.get("E_adjoint_iterations") is not None
    ]
    adjoint_res = [
        float(row["E_adjoint_rel_res"])
        for row in stats_rows
        if row.get("E_adjoint_rel_res") is not None
    ]
    forward_iters = [
        int(row["E_iterations"])
        for row in stats_rows
        if row.get("E_iterations") is not None
    ]
    return {
        "batch_count": len(stats_rows),
        "E_iterations_mean": (
            sum(forward_iters) / len(forward_iters) if forward_iters else None
        ),
        "E_iterations_max": max(forward_iters) if forward_iters else None,
        "E_adjoint_iterations_mean": (
            sum(adjoint_iters) / len(adjoint_iters) if adjoint_iters else None
        ),
        "E_adjoint_iterations_max": max(adjoint_iters) if adjoint_iters else None,
        "E_adjoint_rel_res_max": max(adjoint_res) if adjoint_res else None,
        "E_adjoint_success_all": all(
            bool(row.get("E_adjoint_success", False)) for row in stats_rows
        ),
        "Pi_adjoint_warm_start": all(
            bool(row.get("Pi_adjoint_warm_start", False)) for row in stats_rows
        ),
    }


def _row_for_eval(
    *,
    checkpoint_label: str,
    checkpoint_path: Path,
    checkpoint_step: int,
    method: str,
    terms: int,
    evaluation: GradientEvaluation,
    reference: GradientEvaluation,
    theta: torch.Tensor,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_step),
        "method": method,
        "terms": int(terms),
        "loss_bits": evaluation.loss_bits,
        "reference_loss_bits": reference.loss_bits,
        "elapsed_s": evaluation.elapsed_s,
    }
    row.update(
        _gradient_metrics(
            theta=theta,
            grad=evaluation.grad,
            reference_grad=reference.grad,
            lower=lower,
            upper=upper,
        )
    )
    row.update(_aggregate_stats(evaluation.stats_by_batch))
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _optimizer_state_entry(payload: dict[str, Any]) -> dict[str, Any] | None:
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        return None
    state = optimizer_state.get("state")
    if not isinstance(state, dict) or not state:
        return None
    first_key = sorted(state, key=lambda item: str(item))[0]
    entry = state.get(first_key)
    return entry if isinstance(entry, dict) else None


def _previous_theta_from_lbfgsb_history(
    payload: dict[str, Any],
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    theta = payload.get("theta")
    if not torch.is_tensor(theta):
        return None, {"available": False, "reason": "checkpoint theta is missing"}
    entry = _optimizer_state_entry(payload)
    if entry is None:
        return None, {"available": False, "reason": "optimizer state is missing"}
    old_dirs = entry.get("old_dirs")
    if not isinstance(old_dirs, list) or not old_dirs:
        return None, {"available": False, "reason": "LBFGSB old_dirs is empty"}
    last_step = old_dirs[-1]
    if not torch.is_tensor(last_step) or last_step.numel() != theta.numel():
        return None, {"available": False, "reason": "last LBFGSB step has wrong shape"}
    previous_flat = theta.reshape(-1) - last_step.to(dtype=theta.dtype).reshape(-1)
    return previous_flat.reshape_as(theta).detach().clone(), {
        "available": True,
        "source": "lbfgsb_history",
        "step_inf": float(last_step.detach().abs().amax().cpu()),
        "n_iter": entry.get("n_iter"),
        "last_alpha": entry.get("last_alpha"),
        "last_direction_kind": entry.get("last_direction_kind"),
    }


def _previous_gradient_from_lbfgsb_history(
    payload: dict[str, Any],
) -> torch.Tensor | None:
    theta = payload.get("theta")
    if not torch.is_tensor(theta):
        return None
    entry = _optimizer_state_entry(payload)
    if entry is None:
        return None
    last_grad = entry.get("last_grad")
    old_stps = entry.get("old_stps")
    if (
        not torch.is_tensor(last_grad)
        or not isinstance(old_stps, list)
        or not old_stps
        or not torch.is_tensor(old_stps[-1])
    ):
        return None
    last_y = old_stps[-1]
    if last_grad.numel() != theta.numel() or last_y.numel() != theta.numel():
        return None
    previous = last_grad.reshape(-1) - last_y.reshape(-1).to(dtype=last_grad.dtype)
    return previous.reshape_as(theta).detach().clone()


def _first_terms_by_threshold(
    rows: list[dict[str, Any]],
    *,
    checkpoint_label: str,
    method: str,
    metric: str,
    thresholds: list[float],
) -> dict[str, int | None]:
    selected = [
        row
        for row in rows
        if row.get("checkpoint_label") == checkpoint_label
        and row.get("method") == method
    ]
    selected.sort(key=lambda row: int(row["terms"]))
    out: dict[str, int | None] = {}
    for threshold in thresholds:
        first = None
        for row in selected:
            value = row.get(metric)
            if value is not None and float(value) <= threshold:
                first = int(row["terms"])
                break
        out[str(threshold)] = first
    return out


def _append_batch_rows(
    batch_rows: list[dict[str, Any]],
    *,
    checkpoint_label: str,
    method: str,
    evaluation: GradientEvaluation,
) -> None:
    for row in evaluation.stats_by_batch:
        out = dict(row)
        out["checkpoint_label"] = checkpoint_label
        out["method"] = method
        batch_rows.append(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=None)
    parser.add_argument("--species-tree", type=Path, default=DEFAULT_SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=DEFAULT_FAMILIES_FILE)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/gpurec_gradient_convergence_study"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--forward-iters", type=int, default=32)
    parser.add_argument("--reference-terms", type=int, default=128)
    parser.add_argument("--terms", default="1,2,4,8,16,24,32,48,64,96,128")
    parser.add_argument("--thresholds", default="0.1,0.01,0.001,0.0001")
    parser.add_argument("--family-chunk-size", type=int, default=300)
    parser.add_argument("--clade-budget", type=int, default=315_000)
    parser.add_argument("--batch-packing", default="depth_first_fit")
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-limit", type=int, default=0)
    parser.add_argument("--no-warmstart", action="store_true")
    args = parser.parse_args()

    checkpoint_specs = _parse_checkpoint_specs(args.checkpoint)
    if args.checkpoint_limit > 0:
        checkpoint_specs = checkpoint_specs[: args.checkpoint_limit]
    for _label, checkpoint in checkpoint_specs:
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint not found: {checkpoint}")

    terms_schedule = _parse_int_list(args.terms)
    if args.reference_terms not in terms_schedule:
        terms_schedule.append(args.reference_terms)
        terms_schedule = sorted(set(terms_schedule))
    thresholds = _parse_float_list(args.thresholds)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    first_payload = load_checkpoint(checkpoint_specs[0][1], map_location="cpu")
    first_step = _checkpoint_step(first_payload)
    config = _run_config_from_args(
        args,
        first_checkpoint=checkpoint_specs[0][1],
        checkpoint_step=first_step,
        max_terms=max(max(terms_schedule), args.reference_terms),
    )
    lower = math.log2(config.min_rate)
    upper = math.log2(config.max_rate)

    model = build_alerax_workflow_model(config)
    rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    try:
        for checkpoint_label, checkpoint_path in checkpoint_specs:
            payload = load_checkpoint(checkpoint_path, map_location="cpu")
            validate_checkpoint_model_compatibility(
                path=checkpoint_path,
                config=config,
                model=model,
                payload=payload,
            )
            restore_model_theta(model, payload)
            theta = payload["theta"].detach().clone()
            checkpoint_step = _checkpoint_step(payload)
            previous_theta, previous_meta = _previous_theta_from_lbfgsb_history(payload)

            warm_pi_states = None
            previous_gradient_metrics = None
            if previous_theta is not None and not args.no_warmstart:
                previous_eval = _evaluate_gradient(
                    model,
                    previous_theta,
                    terms=args.reference_terms,
                    fixed_forward_iters=args.forward_iters,
                    capture_pi_adjoint=True,
                )
                warm_pi_states = previous_eval.pi_adjoint_by_batch
                _append_batch_rows(
                    batch_rows,
                    checkpoint_label=checkpoint_label,
                    method="previous_reference",
                    evaluation=previous_eval,
                )
                previous_history_grad = _previous_gradient_from_lbfgsb_history(payload)
                if previous_history_grad is not None:
                    previous_gradient_metrics = _gradient_metrics(
                        theta=previous_theta,
                        grad=previous_history_grad,
                        reference_grad=previous_eval.grad.cpu(),
                        lower=lower,
                        upper=upper,
                    )

            reference = _evaluate_gradient(
                model,
                theta,
                terms=args.reference_terms,
                fixed_forward_iters=args.forward_iters,
            )
            _append_batch_rows(
                batch_rows,
                checkpoint_label=checkpoint_label,
                method="reference",
                evaluation=reference,
            )
            print(
                json.dumps(
                    {
                        "checkpoint": checkpoint_label,
                        "stage": "reference",
                        "terms": args.reference_terms,
                        "elapsed_s": reference.elapsed_s,
                        "loss_bits": reference.loss_bits,
                    },
                    allow_nan=False,
                    sort_keys=True,
                ),
                flush=True,
            )

            for terms in terms_schedule:
                if terms == args.reference_terms:
                    cold_eval = reference
                else:
                    cold_eval = _evaluate_gradient(
                        model,
                        theta,
                        terms=terms,
                        fixed_forward_iters=args.forward_iters,
                    )
                cold_row = _row_for_eval(
                    checkpoint_label=checkpoint_label,
                    checkpoint_path=checkpoint_path,
                    checkpoint_step=checkpoint_step,
                    method="cold",
                    terms=terms,
                    evaluation=cold_eval,
                    reference=reference,
                    theta=theta,
                    lower=lower,
                    upper=upper,
                )
                rows.append(cold_row)
                _append_batch_rows(
                    batch_rows,
                    checkpoint_label=checkpoint_label,
                    method="cold",
                    evaluation=cold_eval,
                )
                print(
                    json.dumps(
                        {
                            "checkpoint": checkpoint_label,
                            "method": "cold",
                            "terms": terms,
                            "elapsed_s": cold_eval.elapsed_s,
                            "error_rel_l2": cold_row["error_rel_l2"],
                            "projected_error_rel_l2": cold_row["projected_error_rel_l2"],
                        },
                        allow_nan=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )

                if warm_pi_states is None:
                    continue
                warm_eval = _evaluate_gradient(
                    model,
                    theta,
                    terms=terms,
                    fixed_forward_iters=args.forward_iters,
                    pi_adjoint_warm_by_batch=warm_pi_states,
                )
                warm_row = _row_for_eval(
                    checkpoint_label=checkpoint_label,
                    checkpoint_path=checkpoint_path,
                    checkpoint_step=checkpoint_step,
                    method="warm_pi_adjoint",
                    terms=terms,
                    evaluation=warm_eval,
                    reference=reference,
                    theta=theta,
                    lower=lower,
                    upper=upper,
                )
                rows.append(warm_row)
                _append_batch_rows(
                    batch_rows,
                    checkpoint_label=checkpoint_label,
                    method="warm_pi_adjoint",
                    evaluation=warm_eval,
                )
                print(
                    json.dumps(
                        {
                            "checkpoint": checkpoint_label,
                            "method": "warm_pi_adjoint",
                            "terms": terms,
                            "elapsed_s": warm_eval.elapsed_s,
                            "error_rel_l2": warm_row["error_rel_l2"],
                            "projected_error_rel_l2": warm_row["projected_error_rel_l2"],
                        },
                        allow_nan=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )

            checkpoint_summaries.append(
                {
                    "label": checkpoint_label,
                    "checkpoint": str(checkpoint_path),
                    "step": checkpoint_step,
                    "reference_terms": args.reference_terms,
                    "reference_loss_bits": reference.loss_bits,
                    "reference_elapsed_s": reference.elapsed_s,
                    "previous_step": previous_meta,
                    "previous_history_gradient_metrics": previous_gradient_metrics,
                    "first_cold_error_rel_l2_terms": _first_terms_by_threshold(
                        rows,
                        checkpoint_label=checkpoint_label,
                        method="cold",
                        metric="error_rel_l2",
                        thresholds=thresholds,
                    ),
                    "first_warm_error_rel_l2_terms": _first_terms_by_threshold(
                        rows,
                        checkpoint_label=checkpoint_label,
                        method="warm_pi_adjoint",
                        metric="error_rel_l2",
                        thresholds=thresholds,
                    ),
                    "first_cold_projected_error_rel_l2_terms": _first_terms_by_threshold(
                        rows,
                        checkpoint_label=checkpoint_label,
                        method="cold",
                        metric="projected_error_rel_l2",
                        thresholds=thresholds,
                    ),
                    "first_warm_projected_error_rel_l2_terms": _first_terms_by_threshold(
                        rows,
                        checkpoint_label=checkpoint_label,
                        method="warm_pi_adjoint",
                        metric="projected_error_rel_l2",
                        thresholds=thresholds,
                    ),
                }
            )

        _write_csv(args.out_dir / "gradient_convergence.csv", rows)
        _write_jsonl(args.out_dir / "gradient_convergence.jsonl", rows)
        _write_jsonl(args.out_dir / "batch_solver_stats.jsonl", batch_rows)
        summary = {
            "config": {
                "species_tree": str(config.species_tree),
                "families_file": str(config.families_file),
                "device": config.device,
                "dtype": config.dtype,
                "forward_iters": args.forward_iters,
                "reference_terms": args.reference_terms,
                "terms": terms_schedule,
                "thresholds": thresholds,
            },
            "checkpoints": checkpoint_summaries,
            "outputs": {
                "rows_csv": str(args.out_dir / "gradient_convergence.csv"),
                "rows_jsonl": str(args.out_dir / "gradient_convergence.jsonl"),
                "batch_jsonl": str(args.out_dir / "batch_solver_stats.jsonl"),
            },
        }
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, allow_nan=False, sort_keys=True), flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()

"""Checkout-local HOGENOM specieswise pulse benchmark.

This script benchmarks short projected-gradient pulse searches from an existing
specieswise HOGENOM checkpoint.  It is intended for local optimization research:
the HOGENOM data and many historical checkpoints live outside the package
distribution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Literal

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpurec.workflow.checkpoint import (  # noqa: E402
    load_checkpoint,
    restore_model_theta,
    save_checkpoint,
    validate_checkpoint_model_compatibility,
)
from gpurec.workflow.config import RunConfig  # noqa: E402
from gpurec.workflow.model_factory import build_alerax_workflow_model  # noqa: E402


HOGENOM_DIR = REPO / "tests" / "data" / "HOGENOM" / "hogenom"
DEFAULT_SPECIES_TREE = HOGENOM_DIR / "hogenom_S.tree"
DEFAULT_FAMILIES_FILE = HOGENOM_DIR / "hogenom_families.local.txt"
DEFAULT_CHECKPOINT = Path(
    "/tmp/gpurec_hogenom_specieswise_coord3141_micro_from_coord3147/"
    "candidate_coord3141.pt"
)
PARAMETER_NAMES = ("D", "L", "T")


@dataclass(frozen=True)
class Evaluation:
    loss_bits: float
    grad_inf: float | None
    projected_grad_inf: float | None
    elapsed_s: float
    grad: torch.Tensor | None
    projected_grad: torch.Tensor | None


@dataclass(frozen=True)
class Candidate:
    label: str
    kind: str
    alpha: float
    indices: tuple[int, ...]
    theta: torch.Tensor
    direction: Literal["projected-gradient", "sign"]


def _parse_int_list(value: str | None) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_float_list(value: str) -> tuple[float, ...]:
    items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("expected at least one comma-separated float")
    for item in items:
        if not math.isfinite(item) or item <= 0.0:
            raise ValueError("pulse step sizes must be positive finite floats")
    return items


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clear_solver_runtime_state(model: Any) -> None:
    model.theta.grad = None
    for static in model.cached_static_states:
        if hasattr(static, "warm_E"):
            static.warm_E = None
        if hasattr(static, "last_solver_stats"):
            static.last_solver_stats = None


def _project_theta(theta: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.clamp(theta, min=lower, max=upper)


def _projected_gradient(
    theta: torch.Tensor,
    grad: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    return theta - _project_theta(theta - grad, lower, upper)


def _set_model_theta(model: Any, theta: torch.Tensor, *, lower: float, upper: float) -> None:
    with torch.no_grad():
        model.theta.copy_(
            _project_theta(theta.to(device=model.theta.device, dtype=model.theta.dtype), lower, upper)
        )
        model.theta.grad = None


def _evaluate(
    model: Any,
    theta: torch.Tensor,
    *,
    lower: float,
    upper: float,
    need_grad: bool,
) -> Evaluation:
    _set_model_theta(model, theta, lower=lower, upper=upper)
    _clear_solver_runtime_state(model)
    _synchronize()
    started = time.perf_counter()
    if need_grad:
        model.theta.grad = None
        loss = model.full_loss()
        loss.backward()
        if model.theta.grad is None:
            raise RuntimeError("missing gradient after full_loss backward")
        grad = model.theta.grad.detach().clone()
        projected = _projected_gradient(
            model.theta.detach(),
            grad,
            lower=lower,
            upper=upper,
        )
        grad_inf = float(grad.detach().abs().amax().cpu()) if grad.numel() else 0.0
        projected_inf = (
            float(projected.detach().abs().amax().cpu()) if projected.numel() else 0.0
        )
    else:
        with torch.no_grad():
            loss = model.full_loss()
        grad = None
        projected = None
        grad_inf = None
        projected_inf = None
    _synchronize()
    elapsed = time.perf_counter() - started
    return Evaluation(
        loss_bits=float(loss.detach().cpu()),
        grad_inf=grad_inf,
        projected_grad_inf=projected_inf,
        elapsed_s=elapsed,
        grad=grad,
        projected_grad=projected,
    )


def _flat_label(index: int) -> str:
    row = int(index) // 3
    column = int(index) % 3
    return f"{row}:{PARAMETER_NAMES[column]}"


def _candidate_label(kind: str, alpha: float, indices: tuple[int, ...]) -> str:
    if len(indices) > 8:
        head = "-".join(_flat_label(index).replace(":", "") for index in indices[:5])
        tail = "-".join(_flat_label(index).replace(":", "") for index in indices[-2:])
        checksum = sum((offset + 1) * int(index) for offset, index in enumerate(indices))
        index_part = f"n{len(indices)}_{head}_tail{tail}_c{checksum}"
    else:
        index_part = "-".join(_flat_label(index).replace(":", "") for index in indices)
    alpha_part = f"{alpha:.8g}".replace("-", "m").replace(".", "p")
    return f"{kind}_a{alpha_part}_{index_part}"


def _unique_indices(indices: list[int], limit: int) -> tuple[int, ...]:
    out: list[int] = []
    for index in indices:
        if index < 0 or index >= limit:
            raise ValueError(f"flat coordinate {index} outside 0..{limit - 1}")
        if index not in out:
            out.append(index)
    return tuple(out)


def _candidate_from_indices(
    *,
    base_theta_flat: torch.Tensor,
    projected_grad_flat: torch.Tensor,
    lower: float,
    upper: float,
    kind: str,
    alpha: float,
    indices: tuple[int, ...],
    direction_mode: Literal["projected-gradient", "sign"],
) -> Candidate | None:
    direction = torch.zeros_like(base_theta_flat)
    if direction_mode == "projected-gradient":
        direction[list(indices)] = -projected_grad_flat[list(indices)]
    else:
        direction[list(indices)] = -projected_grad_flat[list(indices)].sign()
    if not bool((direction != 0).any().detach().cpu()):
        return None
    trial = _project_theta(base_theta_flat + alpha * direction, lower, upper)
    if not bool((trial != base_theta_flat).any().detach().cpu()):
        return None
    label = _candidate_label(kind, alpha, indices)
    return Candidate(
        label=label,
        kind=kind,
        alpha=float(alpha),
        indices=tuple(int(index) for index in indices),
        theta=trial.reshape_as(base_theta_flat),
        direction=direction_mode,
    )


def _build_candidates(
    *,
    base_theta: torch.Tensor,
    projected_grad: torch.Tensor,
    lower: float,
    upper: float,
    topk_sizes: tuple[int, ...],
    alphas: tuple[float, ...],
    coordinate_indices: tuple[int, ...],
    coordinate_top: int,
    pair_top: int,
    pulse_direction: Literal["projected-gradient", "sign"],
    coordinate_direction: Literal["projected-gradient", "sign"],
) -> list[Candidate]:
    theta_flat = base_theta.detach().reshape(-1)
    projected_flat = projected_grad.detach().reshape(-1)
    values = projected_flat.abs()
    order = torch.topk(values, min(int(values.numel()), max(coordinate_top, pair_top, max(topk_sizes, default=0)))).indices
    ordered = [int(index) for index in order.detach().cpu().tolist()]
    if coordinate_indices:
        coordinate_pool = _unique_indices(list(coordinate_indices), int(values.numel()))
    else:
        coordinate_pool = tuple(ordered[:coordinate_top])
    pair_pool = tuple(ordered[:pair_top])

    candidates: list[Candidate] = []
    seen: set[tuple[str, float, tuple[int, ...]]] = set()

    def add(kind: str, alpha: float, indices: tuple[int, ...]) -> None:
        key = (kind, float(alpha), tuple(indices))
        if key in seen:
            return
        seen.add(key)
        candidate = _candidate_from_indices(
            base_theta_flat=theta_flat,
            projected_grad_flat=projected_flat,
            lower=lower,
            upper=upper,
            kind=kind,
            alpha=alpha,
            indices=indices,
            direction_mode=coordinate_direction if kind == "coord" else pulse_direction,
        )
        if candidate is not None:
            candidates.append(candidate)

    for alpha in alphas:
        for size in topk_sizes:
            if size <= 0:
                continue
            indices = tuple(ordered[: min(size, len(ordered))])
            if indices:
                add(f"top{len(indices)}", alpha, indices)
        for index in coordinate_pool:
            add("coord", alpha, (index,))
        for left, right in itertools.combinations(pair_pool, 2):
            add("pair", alpha, (left, right))
    return candidates


def _row_for_result(
    candidate: Candidate,
    *,
    stage: str,
    evaluation: Evaluation,
    base_loss_bits: float,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "label": candidate.label,
        "kind": candidate.kind,
        "alpha": candidate.alpha,
        "indices": " ".join(str(index) for index in candidate.indices),
        "coordinates": " ".join(_flat_label(index) for index in candidate.indices),
        "pulse_direction": candidate.direction,
        "loss_bits": evaluation.loss_bits,
        "delta_bits": evaluation.loss_bits - base_loss_bits,
        "grad_inf": evaluation.grad_inf,
        "projected_grad_inf": evaluation.projected_grad_inf,
        "elapsed_s": evaluation.elapsed_s,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _record_row(
    rows: list[dict[str, Any]],
    *,
    live_path: Path,
    row: dict[str, Any],
    progress_index: int,
    progress_every: int,
) -> int:
    rows.append(row)
    _append_jsonl(live_path, row)
    next_index = progress_index + 1
    if progress_every > 0 and next_index % progress_every == 0:
        print(
            json.dumps(
                {
                    "progress_rows": next_index,
                    "stage": row.get("stage"),
                    "label": row.get("label"),
                    "loss_bits": row.get("loss_bits"),
                    "projected_grad_inf": row.get("projected_grad_inf"),
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return next_index


def _run_config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        species_tree=args.species_tree,
        families_file=args.families_file,
        out_dir=args.out_dir,
        mode="specieswise",
        device=args.device,
        dtype=args.dtype,
        optimizer="projected-sgd",
        steps=args.checkpoint_step,
        lr=args.lr,
        fixed_iters_e=args.probe_iters,
        fixed_iters_pi=args.probe_iters,
        neumann_terms=args.probe_iters,
        adaptive_iters=args.adaptive_iters,
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
        resume_from=args.checkpoint,
    )


def _checkpoint_step(payload: dict[str, Any]) -> int:
    next_step = payload.get("next_step", payload.get("step", 0))
    if isinstance(next_step, bool) or not isinstance(next_step, int):
        raise RuntimeError("checkpoint next_step must be an integer")
    return int(next_step)


def _save_candidate_checkpoint(
    *,
    path: Path,
    config: RunConfig,
    model: Any,
    candidate: Candidate,
    evaluation: Evaluation,
    step: int,
    status_reason: str,
) -> None:
    _set_model_theta(
        model,
        candidate.theta.reshape_as(model.theta),
        lower=math.log2(config.min_rate),
        upper=math.log2(config.max_rate),
    )
    status = {
        "status": "not_converged",
        "reason": status_reason,
        "best_nll_bits": evaluation.loss_bits,
        "best_step": step,
        "previous_objective": evaluation.loss_bits,
        "stable_loss_steps": 0,
        "probe": candidate.kind,
        "probe_label": candidate.label,
        "probe_direction": candidate.direction,
    }
    row = {
        "step": step,
        "optimizer/phase": candidate.kind,
        "optimizer/eval_position": "pulse_probe",
        "optimizer/step_applied": True,
        "optimizer/pulse_direction": candidate.direction,
        "likelihood/data_nll_bits": evaluation.loss_bits,
        "grad/inf": evaluation.grad_inf,
        "grad/projected_inf": evaluation.projected_grad_inf,
    }
    save_checkpoint(
        path,
        config=config,
        model=model,
        optimizer=None,
        step=step,
        next_step=step,
        status=status,
        row=row,
        optimizer_phase=candidate.kind,
    )


def _select_for_gradient(
    rows: list[tuple[Candidate, Evaluation]],
    *,
    base_loss_bits: float,
    loss_window: float,
    limit: int,
) -> list[tuple[Candidate, Evaluation]]:
    ordered = sorted(rows, key=lambda item: (item[1].loss_bits, item[0].kind, item[0].label))
    threshold = base_loss_bits + loss_window
    eligible = [item for item in ordered if item[1].loss_bits <= threshold]
    if not eligible:
        eligible = ordered

    def kind_rank(kind: str) -> tuple[int, str]:
        if kind == "pair":
            return (0, kind)
        if kind == "coord":
            return (1, kind)
        if kind.startswith("top"):
            return (2, kind)
        return (3, kind)

    by_kind: dict[str, list[tuple[Candidate, Evaluation]]] = {}
    for item in eligible:
        by_kind.setdefault(item[0].kind, []).append(item)

    selected: list[tuple[Candidate, Evaluation]] = []
    while len(selected) < limit and by_kind:
        progressed = False
        for kind in sorted(list(by_kind), key=kind_rank):
            items = by_kind[kind]
            if not items:
                by_kind.pop(kind, None)
                continue
            selected.append(items.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
            if not items:
                by_kind.pop(kind, None)
        if not progressed:
            break
    for item in ordered:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    return selected


def _select_best(
    rows: list[tuple[Candidate, Evaluation]],
    *,
    objective_quantum: float,
) -> tuple[Candidate, Evaluation] | None:
    if not rows:
        return None
    best_loss = min(item[1].loss_bits for item in rows)
    threshold = best_loss + max(0.0, objective_quantum)
    near_best = [item for item in rows if item[1].loss_bits <= threshold]
    return min(
        near_best,
        key=lambda item: (
            math.inf
            if item[1].projected_grad_inf is None
            else item[1].projected_grad_inf,
            item[1].loss_bits,
            item[0].label,
        ),
    )


def _beats_baseline(
    evaluation: Evaluation,
    baseline: Evaluation,
    *,
    objective_quantum: float,
) -> bool:
    if evaluation.loss_bits < baseline.loss_bits:
        return True
    if evaluation.projected_grad_inf is None or baseline.projected_grad_inf is None:
        return False
    if evaluation.loss_bits <= baseline.loss_bits + max(0.0, objective_quantum):
        return evaluation.projected_grad_inf < baseline.projected_grad_inf
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--species-tree", type=Path, default=DEFAULT_SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=DEFAULT_FAMILIES_FILE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--probe-iters", type=int, default=64)
    parser.add_argument("--validate-iters", type=int, default=128)
    parser.add_argument("--family-chunk-size", type=int, default=300)
    parser.add_argument("--clade-budget", type=int, default=315_000)
    parser.add_argument("--batch-packing", default="depth_first_fit")
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adaptive-iters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--topk-sizes", default="1,2,3,5,10,20,50")
    parser.add_argument("--alphas", default="5e-5,1e-4,1.5e-4,2e-4,5e-4,1e-3,5e-3,1e-2")
    parser.add_argument("--coordinate-indices", default=None)
    parser.add_argument("--coordinate-top", type=int, default=8)
    parser.add_argument("--pair-top", type=int, default=6)
    parser.add_argument(
        "--pulse-direction",
        choices=("projected-gradient", "sign"),
        default="projected-gradient",
        help=(
            "Use scaled projected-gradient steps or unit sign steps on selected "
            "top-k and pair coordinates."
        ),
    )
    parser.add_argument(
        "--coordinate-direction",
        choices=("projected-gradient", "sign"),
        default="sign",
        help=(
            "Use scaled projected-gradient steps or absolute sign steps for "
            "single-coordinate probes."
        ),
    )
    parser.add_argument("--gradient-candidates", type=int, default=20)
    parser.add_argument("--validate-candidates", type=int, default=3)
    parser.add_argument("--loss-window-bits", type=float, default=0.125)
    parser.add_argument("--objective-quantum-bits", type=float, default=0.0625)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    args.checkpoint_step = _checkpoint_step(payload)
    config = _run_config_from_args(args)
    topk_sizes = _parse_int_list(args.topk_sizes)
    alphas = _parse_float_list(args.alphas)
    coordinate_indices = _parse_int_list(args.coordinate_indices)
    lower = math.log2(config.min_rate)
    upper = math.log2(config.max_rate)

    model = build_alerax_workflow_model(config)
    try:
        validate_checkpoint_model_compatibility(
            path=args.checkpoint,
            config=config,
            model=model,
            payload=payload,
        )
        restore_model_theta(model, payload)
        base_theta = model.theta.detach().clone()
        base_eval = _evaluate(
            model,
            base_theta,
            lower=lower,
            upper=upper,
            need_grad=True,
        )
        if base_eval.projected_grad is None:
            raise RuntimeError("base evaluation did not produce projected gradient")

        candidates = _build_candidates(
            base_theta=base_theta,
            projected_grad=base_eval.projected_grad,
            lower=lower,
            upper=upper,
            topk_sizes=topk_sizes,
            alphas=alphas,
            coordinate_indices=coordinate_indices,
            coordinate_top=args.coordinate_top,
            pair_top=args.pair_top,
            pulse_direction=args.pulse_direction,
            coordinate_direction=args.coordinate_direction,
        )

        rows: list[dict[str, Any]] = []
        live_path = args.out_dir / "pulse_benchmark.live.jsonl"
        live_path.write_text("", encoding="utf-8")
        progress_index = 0
        progress_index = _record_row(
            rows,
            live_path=live_path,
            row={
                "stage": "base_probe",
                "label": "base",
                "kind": "base",
                "alpha": 0.0,
                "indices": "",
                "coordinates": "",
                "pulse_direction": args.pulse_direction,
                "coordinate_direction": args.coordinate_direction,
                "loss_bits": base_eval.loss_bits,
                "delta_bits": 0.0,
                "grad_inf": base_eval.grad_inf,
                "projected_grad_inf": base_eval.projected_grad_inf,
                "elapsed_s": base_eval.elapsed_s,
            },
            progress_index=progress_index,
            progress_every=args.progress_every,
        )
        loss_only_results: list[tuple[Candidate, Evaluation]] = []
        for candidate in candidates:
            evaluation = _evaluate(
                model,
                candidate.theta.reshape_as(model.theta),
                lower=lower,
                upper=upper,
                need_grad=False,
            )
            loss_only_results.append((candidate, evaluation))
            progress_index = _record_row(
                rows,
                live_path=live_path,
                row=_row_for_result(
                    candidate,
                    stage="probe_loss",
                    evaluation=evaluation,
                    base_loss_bits=base_eval.loss_bits,
                ),
                progress_index=progress_index,
                progress_every=args.progress_every,
            )

        gradient_inputs = _select_for_gradient(
            loss_only_results,
            base_loss_bits=base_eval.loss_bits,
            loss_window=args.loss_window_bits,
            limit=args.gradient_candidates,
        )
        gradient_results: list[tuple[Candidate, Evaluation]] = []
        for candidate, _loss_only in gradient_inputs:
            evaluation = _evaluate(
                model,
                candidate.theta.reshape_as(model.theta),
                lower=lower,
                upper=upper,
                need_grad=True,
            )
            gradient_results.append((candidate, evaluation))
            progress_index = _record_row(
                rows,
                live_path=live_path,
                row=_row_for_result(
                    candidate,
                    stage="probe_grad",
                    evaluation=evaluation,
                    base_loss_bits=base_eval.loss_bits,
                ),
                progress_index=progress_index,
                progress_every=args.progress_every,
            )

        validated_results: list[tuple[Candidate, Evaluation]] = []
        validate_base: Evaluation | None = None
        best_probe = _select_best(
            gradient_results,
            objective_quantum=args.objective_quantum_bits,
        )
        if args.validate_iters > 0:
            model.configure_solver_iterations(
                fixed_iters_E=args.validate_iters,
                fixed_iters_Pi=args.validate_iters,
                neumann_terms=args.validate_iters,
            )
            validate_base = _evaluate(
                model,
                base_theta,
                lower=lower,
                upper=upper,
                need_grad=True,
            )
            progress_index = _record_row(
                rows,
                live_path=live_path,
                row={
                    "stage": "base_validate",
                    "label": "base",
                    "kind": "base",
                    "alpha": 0.0,
                    "indices": "",
                    "coordinates": "",
                    "pulse_direction": args.pulse_direction,
                    "coordinate_direction": args.coordinate_direction,
                    "loss_bits": validate_base.loss_bits,
                    "delta_bits": 0.0,
                    "grad_inf": validate_base.grad_inf,
                    "projected_grad_inf": validate_base.projected_grad_inf,
                    "elapsed_s": validate_base.elapsed_s,
                },
                progress_index=progress_index,
                progress_every=args.progress_every,
            )
        if validate_base is not None and gradient_results:
            validate_inputs = sorted(
                gradient_results,
                key=lambda item: (
                    item[1].loss_bits,
                    math.inf
                    if item[1].projected_grad_inf is None
                    else item[1].projected_grad_inf,
                    item[0].label,
                ),
            )[: max(1, args.validate_candidates)]
            for candidate, _probe_eval in validate_inputs:
                evaluation = _evaluate(
                    model,
                    candidate.theta.reshape_as(model.theta),
                    lower=lower,
                    upper=upper,
                    need_grad=True,
                )
                validated_results.append((candidate, evaluation))
                progress_index = _record_row(
                    rows,
                    live_path=live_path,
                    row=_row_for_result(
                        candidate,
                        stage="validate_grad",
                        evaluation=evaluation,
                        base_loss_bits=validate_base.loss_bits,
                    ),
                    progress_index=progress_index,
                    progress_every=args.progress_every,
                )
        best_validated = _select_best(
            validated_results,
            objective_quantum=args.objective_quantum_bits,
        )
        best = best_validated or best_probe
        baseline_for_best = (
            validate_base
            if validate_base is not None and validated_results
            else base_eval
        )
        if best is not None and not _beats_baseline(
            best[1],
            baseline_for_best,
            objective_quantum=args.objective_quantum_bits,
        ):
            best = None

        if best is not None:
            best_candidate, best_eval = best
            _save_candidate_checkpoint(
                path=args.out_dir / f"{best_candidate.label}.pt",
                config=config,
                model=model,
                candidate=best_candidate,
                evaluation=best_eval,
                step=args.checkpoint_step,
                status_reason="hogenom_specieswise_pulse_probe",
            )

        _write_csv(args.out_dir / "pulse_benchmark.csv", rows)
        _write_jsonl(args.out_dir / "pulse_benchmark.jsonl", rows)
        summary = {
            "checkpoint": str(args.checkpoint),
            "probe_iters": args.probe_iters,
            "validate_iters": args.validate_iters,
            "pulse_direction": args.pulse_direction,
            "coordinate_direction": args.coordinate_direction,
            "candidate_count": len(candidates),
            "gradient_candidate_count": len(gradient_results),
            "validated_candidate_count": len(validated_results),
            "base_probe": {
                "loss_bits": base_eval.loss_bits,
                "grad_inf": base_eval.grad_inf,
                "projected_grad_inf": base_eval.projected_grad_inf,
                "elapsed_s": base_eval.elapsed_s,
            },
            "base_validate": None
            if validate_base is None
            else {
                "loss_bits": validate_base.loss_bits,
                "grad_inf": validate_base.grad_inf,
                "projected_grad_inf": validate_base.projected_grad_inf,
                "elapsed_s": validate_base.elapsed_s,
            },
            "best_probe": None
            if best_probe is None
            else {
                "label": best_probe[0].label,
                "loss_bits": best_probe[1].loss_bits,
                "projected_grad_inf": best_probe[1].projected_grad_inf,
            },
            "best_validated": None
            if best_validated is None
            else {
                "label": best_validated[0].label,
                "loss_bits": best_validated[1].loss_bits,
                "projected_grad_inf": best_validated[1].projected_grad_inf,
            },
            "saved_checkpoint": None
            if best is None
            else str(args.out_dir / f"{best[0].label}.pt"),
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

"""Replay the accepted HOGENOM specieswise pulse tail with measured timing.

The best current fixed128 HOGENOM specieswise checkpoint was produced by a
short sequence of manual projected-gradient pulses after a projected-SGD repair
checkpoint.  By default this script replays the accepted checkpoint deltas
exactly, records per-delta evaluation wall time, and writes workflow-style
history/checkpoint artifacts so the end-to-end route benchmark can account for
the tail without unknown elapsed stages.  A dynamic top-k pulse mode is also
kept for diagnostics, but it is not expected to reproduce the historical tail
because the historical probes were selected from branch-specific searches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
DEFAULT_START_CHECKPOINT = Path(
    "/tmp/gpurec_hogenom_specieswise_projectedsgd_high64_lr1e4_repair_from_objective829/"
    "checkpoints/latest.pt"
)
PARAMETER_NAMES = ("D", "L", "T")


@dataclass(frozen=True)
class EvalResult:
    loss_bits: float
    grad_inf: float
    projected_grad_inf: float
    elapsed_s: float
    grad: torch.Tensor
    projected_grad: torch.Tensor


@dataclass(frozen=True)
class PulseOp:
    label: str
    kind: Literal["topk", "coord"]
    alpha: float
    k: int | None = None
    index: int | None = None


@dataclass(frozen=True)
class DeltaOp:
    label: str
    checkpoint: Path


DEFAULT_TAIL: tuple[PulseOp, ...] = (
    PulseOp("topk_tradeoff_800_a0p005", "topk", 0.005, k=800),
    PulseOp("kkt_topk_100_a0p001", "topk", 0.001, k=100),
    PulseOp("frontier_top2_a0p02", "topk", 0.02, k=2),
    PulseOp("frontier_objective_top20_a0p03", "topk", 0.03, k=20),
    PulseOp("greedy_repair_top8_a0p008", "topk", 0.008, k=8),
    PulseOp("greedy_repair_top2_a0p03", "topk", 0.03, k=2),
    PulseOp("greedy_repair_top2_a0p02", "topk", 0.02, k=2),
    PulseOp("greedy_repair_top2_a0p005", "topk", 0.005, k=2),
    PulseOp("coord3147_micro_a0p0002", "coord", 0.0002, index=3147),
    PulseOp("coord3141_micro_a0p00015", "coord", 0.00015, index=3141),
)

DEFAULT_DELTA_CHAIN: tuple[DeltaOp, ...] = (
    DeltaOp(
        "topk_tradeoff_step879",
        Path("/tmp/gpurec_hogenom_specieswise_topk_probe_from_step879/candidate_tradeoff.pt"),
    ),
    DeltaOp(
        "kkt_topk_step879",
        Path("/tmp/gpurec_hogenom_specieswise_kkt_probe_from_topk879/candidate_kkt.pt"),
    ),
    DeltaOp(
        "frontier_top2_step879",
        Path(
            "/tmp/gpurec_hogenom_specieswise_frontier_grad_probe_from_kkt879/"
            "candidate_frontier.pt"
        ),
    ),
    DeltaOp(
        "frontier_objective_step879",
        Path(
            "/tmp/gpurec_hogenom_specieswise_frontier2_objective_candidate/"
            "candidate_objective.pt"
        ),
    ),
    DeltaOp(
        "greedy_objective_step879",
        Path(
            "/tmp/gpurec_hogenom_specieswise_greedy_frontier_from_objective875/"
            "candidate_greedy_objective.pt"
        ),
    ),
    DeltaOp(
        "greedy_objective2_step879",
        Path(
            "/tmp/gpurec_hogenom_specieswise_greedy_frontier2_from_objective875/"
            "candidate_greedy_objective.pt"
        ),
    ),
    DeltaOp(
        "coord3147_micro_step879",
        Path(
            "/tmp/gpurec_hogenom_specieswise_coord3147_micro_from_objective875_cycle2/"
            "candidate_coord3147.pt"
        ),
    ),
    DeltaOp(
        "coord3141_micro_step879",
        Path(
            "/tmp/gpurec_hogenom_specieswise_coord3141_micro_from_coord3147/"
            "candidate_coord3141.pt"
        ),
    ),
)


def _checkpoint_step(payload: dict[str, Any]) -> int:
    value = payload.get("next_step", payload.get("step", 0))
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("checkpoint next_step must be an integer")
    return int(value)


def _run_config_from_args(args: argparse.Namespace, checkpoint_step: int) -> RunConfig:
    return RunConfig(
        species_tree=args.species_tree,
        families_file=args.families_file,
        out_dir=args.out_dir,
        mode="specieswise",
        device=args.device,
        dtype=args.dtype,
        optimizer="projected-sgd",
        steps=checkpoint_step,
        lr=1e-4,
        fixed_iters_e=args.probe_iters,
        fixed_iters_pi=args.probe_iters,
        neumann_terms=args.probe_iters,
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
        resume_from=args.start_checkpoint,
    )


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


def _evaluate_current(model: Any, *, lower: float, upper: float) -> EvalResult:
    _clear_solver_runtime_state(model)
    _synchronize()
    started = time.perf_counter()
    model.theta.grad = None
    loss = model.full_loss()
    loss.backward()
    _synchronize()
    elapsed_s = time.perf_counter() - started
    if model.theta.grad is None:
        raise RuntimeError("missing gradient after full_loss backward")
    grad = model.theta.grad.detach().clone()
    projected_grad = _projected_gradient(
        model.theta.detach(),
        grad,
        lower=lower,
        upper=upper,
    )
    return EvalResult(
        loss_bits=float(loss.detach().cpu()),
        grad_inf=float(grad.detach().abs().amax().cpu()) if grad.numel() else 0.0,
        projected_grad_inf=(
            float(projected_grad.detach().abs().amax().cpu())
            if projected_grad.numel()
            else 0.0
        ),
        elapsed_s=elapsed_s,
        grad=grad,
        projected_grad=projected_grad,
    )


def _flat_label(index: int) -> str:
    return f"{int(index) // 3}:{PARAMETER_NAMES[int(index) % 3]}"


def _apply_pulse(
    model: Any,
    op: PulseOp,
    projected_grad: torch.Tensor,
    *,
    lower: float,
    upper: float,
    pulse_direction: Literal["projected-gradient", "sign"],
    coordinate_direction: Literal["projected-gradient", "sign"],
) -> tuple[float, str]:
    theta_flat = model.theta.detach().reshape(-1)
    projected_flat = projected_grad.detach().reshape(-1)
    direction = torch.zeros_like(theta_flat)
    if op.kind == "topk":
        if op.k is None or op.k <= 0:
            raise RuntimeError(f"invalid top-k pulse {op}")
        count = min(int(op.k), int(projected_flat.numel()))
        indices = torch.topk(projected_flat.abs(), count).indices
        if pulse_direction == "projected-gradient":
            direction[indices] = -projected_flat[indices]
        else:
            direction[indices] = -projected_flat[indices].sign()
        coordinate_text = " ".join(_flat_label(int(index)) for index in indices[:8])
        if count > 8:
            coordinate_text += f" ... n={count}"
    elif op.kind == "coord":
        if op.index is None:
            raise RuntimeError(f"invalid coordinate pulse {op}")
        index = int(op.index)
        if index < 0 or index >= int(projected_flat.numel()):
            raise RuntimeError(f"coordinate {index} outside theta range")
        if coordinate_direction == "projected-gradient":
            direction[index] = -projected_flat[index]
        else:
            direction[index] = -projected_flat[index].sign()
        coordinate_text = _flat_label(index)
    else:
        raise RuntimeError(f"unknown pulse kind {op.kind!r}")
    delta = float(op.alpha) * direction
    step_inf = float(delta.detach().abs().amax().cpu()) if delta.numel() else 0.0
    with torch.no_grad():
        model.theta.copy_(_project_theta(theta_flat + delta, lower, upper).reshape_as(model.theta))
        model.theta.grad = None
    return step_inf, coordinate_text


def _apply_checkpoint_delta(
    model: Any,
    target_theta: torch.Tensor,
) -> tuple[float, float, int, str]:
    current = model.theta.detach()
    target = target_theta.to(device=current.device, dtype=current.dtype)
    if tuple(target.shape) != tuple(current.shape):
        raise RuntimeError(
            f"target theta shape {tuple(target.shape)} does not match model "
            f"shape {tuple(current.shape)}"
        )
    delta = target - current
    delta_flat = delta.detach().reshape(-1)
    abs_delta = delta_flat.abs()
    nonzero = int((abs_delta > 0).sum().detach().cpu())
    step_inf = float(abs_delta.amax().detach().cpu()) if abs_delta.numel() else 0.0
    step_l2 = (
        float(torch.linalg.vector_norm(delta_flat).detach().cpu())
        if delta_flat.numel()
        else 0.0
    )
    if nonzero:
        count = min(nonzero, 8)
        indices = torch.topk(abs_delta, count).indices
        coordinate_text = " ".join(_flat_label(int(index)) for index in indices)
        if nonzero > 8:
            coordinate_text += f" ... n={nonzero}"
    else:
        coordinate_text = ""
    with torch.no_grad():
        model.theta.copy_(target)
        model.theta.grad = None
    return step_inf, step_l2, nonzero, coordinate_text


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _save_replay_checkpoint(
    *,
    path: Path,
    config: RunConfig,
    model: Any,
    row: dict[str, Any],
    checkpoint_step: int,
    elapsed_s: float,
) -> None:
    status = {
        "status": "not_converged",
        "reason": "hogenom_specieswise_tail_replay",
        "best_nll_bits": row["likelihood/data_nll_bits"],
        "best_step": checkpoint_step,
        "previous_objective": row["likelihood/data_nll_bits"],
        "stable_loss_steps": 0,
        "elapsed_s": elapsed_s,
    }
    save_checkpoint(
        path,
        config=config,
        model=model,
        optimizer=None,
        step=checkpoint_step,
        next_step=checkpoint_step,
        status=status,
        row=row,
        optimizer_phase=str(row["optimizer/phase"]),
    )


def _parse_delta_op(value: str) -> DeltaOp:
    if "=" in value:
        label, path = value.split("=", 1)
        if not label:
            raise ValueError("delta checkpoint label cannot be empty")
        return DeltaOp(label=label, checkpoint=Path(path))
    path = Path(value)
    return DeltaOp(label=path.stem, checkpoint=path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-checkpoint", type=Path, default=DEFAULT_START_CHECKPOINT)
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
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("checkpoint-delta", "dynamic-pulses"),
        default="checkpoint-delta",
        help=(
            "checkpoint-delta copies the accepted target checkpoint tensors in "
            "order; dynamic-pulses recomputes top-k/coordinate pulses from the "
            "fresh projected gradient at each replay step."
        ),
    )
    parser.add_argument(
        "--pulse-direction",
        choices=("projected-gradient", "sign"),
        default="projected-gradient",
        help=(
            "Top-k direction used in dynamic-pulses mode. Historical HOGENOM "
            "top-k pulses used projected-gradient scaling, not unit sign steps."
        ),
    )
    parser.add_argument(
        "--coordinate-direction",
        choices=("projected-gradient", "sign"),
        default="sign",
        help=(
            "Coordinate direction used in dynamic-pulses mode. Historical "
            "coordinate micro probes used absolute sign steps."
        ),
    )
    parser.add_argument(
        "--delta-checkpoint",
        action="append",
        default=None,
        help=(
            "Checkpoint delta stage as label=/path or /path. May be repeated. "
            "Defaults to the accepted post-step879 checkpoint chain."
        ),
    )
    args = parser.parse_args()

    if not args.start_checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.start_checkpoint}")
    delta_ops = (
        [_parse_delta_op(value) for value in args.delta_checkpoint]
        if args.delta_checkpoint is not None
        else list(DEFAULT_DELTA_CHAIN)
    )
    if args.mode == "checkpoint-delta":
        for op in delta_ops:
            if not op.checkpoint.is_file():
                raise SystemExit(f"delta checkpoint not found: {op.checkpoint}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")

    payload = load_checkpoint(args.start_checkpoint, map_location="cpu")
    checkpoint_step = _checkpoint_step(payload)
    config = _run_config_from_args(args, checkpoint_step)
    config.write_json(args.out_dir / "run_config.json")
    lower = math.log2(config.min_rate)
    upper = math.log2(config.max_rate)

    model = build_alerax_workflow_model(config)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    try:
        validate_checkpoint_model_compatibility(
            path=args.start_checkpoint,
            config=config,
            model=model,
            payload=payload,
        )
        restore_model_theta(model, payload)
        current_eval = _evaluate_current(model, lower=lower, upper=upper)
        base_row = {
            "step": checkpoint_step,
            "optimizer/phase": "tail_replay_base",
            "optimizer/eval_position": "base",
            "optimizer/step_applied": False,
            "optimizer/tail_coordinates": "",
            "theta_step_inf": 0.0,
            "step_s": current_eval.elapsed_s,
            "likelihood/data_nll_bits": current_eval.loss_bits,
            "likelihood/log_likelihood_bits": -current_eval.loss_bits,
            "grad/inf": current_eval.grad_inf,
            "grad/projected_inf": current_eval.projected_grad_inf,
        }
        rows.append(base_row)
        _append_jsonl(history_path, base_row)
        print(json.dumps(base_row, allow_nan=False, sort_keys=True), flush=True)

        if args.mode == "checkpoint-delta":
            for op in delta_ops:
                step_started = time.perf_counter()
                target_payload = load_checkpoint(op.checkpoint, map_location="cpu")
                validate_checkpoint_model_compatibility(
                    path=op.checkpoint,
                    config=config,
                    model=model,
                    payload=target_payload,
                )
                step_inf, step_l2, nonzero, coordinate_text = _apply_checkpoint_delta(
                    model,
                    target_payload["theta"],
                )
                current_eval = _evaluate_current(model, lower=lower, upper=upper)
                step_s = time.perf_counter() - step_started
                row = {
                    "step": checkpoint_step,
                    "optimizer/phase": op.label,
                    "optimizer/eval_position": "tail_replay",
                    "optimizer/replay_mode": args.mode,
                    "optimizer/step_applied": True,
                    "optimizer/tail_kind": "checkpoint_delta",
                    "optimizer/target_checkpoint": str(op.checkpoint),
                    "optimizer/tail_coordinates": coordinate_text,
                    "optimizer/delta_nonzero": nonzero,
                    "theta_step_inf": step_inf,
                    "theta_step_l2": step_l2,
                    "step_s": step_s,
                    "likelihood/data_nll_bits": current_eval.loss_bits,
                    "likelihood/log_likelihood_bits": -current_eval.loss_bits,
                    "grad/inf": current_eval.grad_inf,
                    "grad/projected_inf": current_eval.projected_grad_inf,
                }
                rows.append(row)
                _append_jsonl(history_path, row)
                _save_replay_checkpoint(
                    path=checkpoint_dir / f"{op.label}.pt",
                    config=config,
                    model=model,
                    row=row,
                    checkpoint_step=checkpoint_step,
                    elapsed_s=time.perf_counter() - started,
                )
                print(json.dumps(row, allow_nan=False, sort_keys=True), flush=True)
        else:
            for op in DEFAULT_TAIL:
                step_started = time.perf_counter()
                step_inf, coordinate_text = _apply_pulse(
                    model,
                    op,
                    current_eval.projected_grad,
                    lower=lower,
                    upper=upper,
                    pulse_direction=args.pulse_direction,
                    coordinate_direction=args.coordinate_direction,
                )
                current_eval = _evaluate_current(model, lower=lower, upper=upper)
                step_s = time.perf_counter() - step_started
                row = {
                    "step": checkpoint_step,
                    "optimizer/phase": op.label,
                    "optimizer/eval_position": "tail_replay",
                    "optimizer/replay_mode": args.mode,
                    "optimizer/pulse_direction": args.pulse_direction,
                    "optimizer/coordinate_direction": args.coordinate_direction,
                    "optimizer/step_applied": True,
                    "optimizer/tail_kind": op.kind,
                    "optimizer/tail_k": op.k,
                    "optimizer/tail_index": op.index,
                    "optimizer/tail_alpha": op.alpha,
                    "optimizer/tail_coordinates": coordinate_text,
                    "theta_step_inf": step_inf,
                    "step_s": step_s,
                    "likelihood/data_nll_bits": current_eval.loss_bits,
                    "likelihood/log_likelihood_bits": -current_eval.loss_bits,
                    "grad/inf": current_eval.grad_inf,
                    "grad/projected_inf": current_eval.projected_grad_inf,
                }
                rows.append(row)
                _append_jsonl(history_path, row)
                _save_replay_checkpoint(
                    path=checkpoint_dir / f"{op.label}.pt",
                    config=config,
                    model=model,
                    row=row,
                    checkpoint_step=checkpoint_step,
                    elapsed_s=time.perf_counter() - started,
                )
                print(json.dumps(row, allow_nan=False, sort_keys=True), flush=True)

        final_eval = current_eval
        validation_row: dict[str, Any] | None = None
        if not args.skip_validation and args.validate_iters > 0:
            model.configure_solver_iterations(
                fixed_iters_E=args.validate_iters,
                fixed_iters_Pi=args.validate_iters,
                neumann_terms=args.validate_iters,
            )
            validation_eval = _evaluate_current(model, lower=lower, upper=upper)
            validation_row = {
                "step": checkpoint_step,
                "optimizer/phase": "fixed128_validation",
                "optimizer/eval_position": "final",
                "optimizer/step_applied": False,
                "optimizer/final_check_iters": args.validate_iters,
                "optimizer/tail_coordinates": "",
                "theta_step_inf": 0.0,
                "step_s": validation_eval.elapsed_s,
                "likelihood/data_nll_bits": validation_eval.loss_bits,
                "likelihood/log_likelihood_bits": -validation_eval.loss_bits,
                "grad/inf": validation_eval.grad_inf,
                "grad/projected_inf": validation_eval.projected_grad_inf,
            }
            rows.append(validation_row)
            _append_jsonl(history_path, validation_row)
            final_eval = validation_eval
            print(json.dumps(validation_row, allow_nan=False, sort_keys=True), flush=True)

        elapsed_s = time.perf_counter() - started
        final_row = validation_row if validation_row is not None else rows[-1]
        _save_replay_checkpoint(
            path=checkpoint_dir / "latest.pt",
            config=config,
            model=model,
            row=final_row,
            checkpoint_step=checkpoint_step,
            elapsed_s=elapsed_s,
        )
        _save_replay_checkpoint(
            path=checkpoint_dir / "best.pt",
            config=config,
            model=model,
            row=final_row,
            checkpoint_step=checkpoint_step,
            elapsed_s=elapsed_s,
        )
        summary = {
            "status": "not_converged",
            "reason": "hogenom_specieswise_tail_replay",
            "families": int(model.n_families),
            "species": int(model.n_species),
            "batches": len(model.batch_metadata),
            "elapsed_s": elapsed_s,
            "replay_mode": args.mode,
            "pulse_direction": (
                args.pulse_direction if args.mode == "dynamic-pulses" else None
            ),
            "coordinate_direction": (
                args.coordinate_direction if args.mode == "dynamic-pulses" else None
            ),
            "probe_iters": args.probe_iters,
            "validate_iters": (0 if args.skip_validation else args.validate_iters),
            "tail_stage_count": (
                len(delta_ops) if args.mode == "checkpoint-delta" else len(DEFAULT_TAIL)
            ),
            "final_nll_bits": final_eval.loss_bits,
            "final_grad_inf": final_eval.grad_inf,
            "final_projected_grad_inf": final_eval.projected_grad_inf,
            "fixed64_nll_bits": rows[-2]["likelihood/data_nll_bits"]
            if validation_row is not None
            else final_eval.loss_bits,
            "fixed64_projected_grad_inf": rows[-2]["grad/projected_inf"]
            if validation_row is not None
            else final_eval.projected_grad_inf,
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

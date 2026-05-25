"""Counts-free multifidelity Adagrad route for HOGENOM specieswise rates.

The route starts from uniform 0.05 D/L/T rates on every species branch and uses
only gpurec full-objective gradients. By default fixed mode replays the fixed
phase lengths from the first successful route:

1. fixed8 Adagrad warmup,
2. fixed16 Adagrad bridge,
3. fixed32 Adagrad repair,
4. fixed128 loss-only validation.

With ``--schedule-mode adaptive`` the same budget ladder is promoted by
higher-budget validation stalls instead of fixed phase lengths, restoring the
best validated theta before each promotion.  Adaptive mode prepends a fixed4
phase by default before the fixed8 phase.

It is checkout-local benchmarking glue for the bundled HOGENOM data, not a
general workflow optimizer.
"""

from __future__ import annotations

import argparse
import csv
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

from gpurec.workflow.checkpoint import save_checkpoint  # noqa: E402
from gpurec.workflow.config import RunConfig  # noqa: E402
from gpurec.workflow.model_factory import build_alerax_workflow_model  # noqa: E402


HOGENOM_DIR = REPO / "tests" / "data" / "HOGENOM" / "hogenom"
DEFAULT_SPECIES_TREE = HOGENOM_DIR / "hogenom_S.tree"
DEFAULT_FAMILIES_FILE = HOGENOM_DIR / "hogenom_families.local.txt"


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


def _configure_budget(model: Any, budget: int) -> None:
    model.configure_solver_iterations(
        fixed_iters_E=budget,
        fixed_iters_Pi=budget,
        neumann_terms=budget,
        adaptive_neumann_terms=False,
    )


def _loss_only(model: Any, budget: int) -> tuple[float, float]:
    _configure_budget(model, budget)
    _clear_solver_runtime_state(model)
    _synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        loss = model.full_loss_for_theta(model.theta.detach())
    _synchronize()
    return float(loss.detach().cpu()), time.perf_counter() - started


def _restore_theta(
    model: Any,
    theta: torch.Tensor,
    *,
    lower_bound: float,
    upper_bound: float,
) -> None:
    with torch.no_grad():
        model.theta.copy_(
            theta.to(device=model.theta.device, dtype=model.theta.dtype).clamp_(
                lower_bound,
                upper_bound,
            )
        )
        model.theta.grad = None


def _grad_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    budget: int,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float, float]:
    _configure_budget(model, budget)
    _clear_solver_runtime_state(model)
    _synchronize()
    started = time.perf_counter()
    loss = model.full_loss()
    loss.backward()
    if model.theta.grad is None:
        raise RuntimeError("missing theta gradient")
    grad_inf = float(model.theta.grad.detach().abs().amax().cpu())
    _synchronize()
    elapsed = time.perf_counter() - started
    optimizer.step()
    with torch.no_grad():
        model.theta.clamp_(lower_bound, upper_bound)
    return float(loss.detach().cpu()), grad_inf, elapsed


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_phase(
    *,
    model: Any,
    phase: str,
    budget: int,
    lr: float,
    steps: int,
    lower_bound: float,
    upper_bound: float,
    wall_start: float,
    first_global_step: int,
    history_jsonl: Path,
    rows: list[dict[str, Any]],
) -> int:
    optimizer = torch.optim.Adagrad([model.theta], lr=lr)
    global_step = first_global_step
    for phase_step in range(steps):
        pre_loss, grad_inf, step_s = _grad_step(
            model,
            optimizer,
            budget=budget,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        row = {
            "global_step": global_step,
            "phase": phase,
            "phase_step": phase_step,
            "budget": budget,
            "lr": lr,
            "pre_step_loss_bits": pre_loss,
            "grad_inf": grad_inf,
            "step_s": step_s,
            "wall_s": time.perf_counter() - wall_start,
        }
        rows.append(row)
        _write_jsonl(history_jsonl, row)
        global_step += 1
    return global_step


def _run_adaptive_phase(
    *,
    model: Any,
    phase: str,
    budget: int,
    validation_budget: int,
    lr: float,
    max_steps: int,
    check_interval: int,
    min_checks: int,
    patience: int,
    min_delta: float,
    min_improvement_per_second: float,
    target_nll: float | None,
    lower_bound: float,
    upper_bound: float,
    wall_start: float,
    max_wall_s: float,
    final_validation_reserve_s: float,
    first_global_step: int,
    history_jsonl: Path,
    rows: list[dict[str, Any]],
) -> tuple[int, dict[str, Any], bool]:
    optimizer = torch.optim.Adagrad([model.theta], lr=lr)
    global_step = first_global_step
    checks = 0
    stale_checks = 0
    previous_validation_loss: float | None = None
    previous_validation_wall: float | None = None
    best_validation_loss = math.inf
    best_validation_step = -1
    best_theta = model.theta.detach().clone()
    stop_reason = "max_steps"
    reached_target = False

    for phase_step in range(max_steps):
        if time.perf_counter() - wall_start >= max_wall_s - final_validation_reserve_s:
            stop_reason = "wall_budget"
            break
        pre_loss, grad_inf, step_s = _grad_step(
            model,
            optimizer,
            budget=budget,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        row = {
            "global_step": global_step,
            "phase": phase,
            "phase_step": phase_step,
            "budget": budget,
            "validation_budget": validation_budget,
            "lr": lr,
            "pre_step_loss_bits": pre_loss,
            "grad_inf": grad_inf,
            "step_s": step_s,
            "wall_s": time.perf_counter() - wall_start,
        }
        rows.append(row)
        _write_jsonl(history_jsonl, row)
        global_step += 1

        if (phase_step + 1) % check_interval != 0:
            continue

        validation_loss, validation_s = _loss_only(model, validation_budget)
        validation_wall = time.perf_counter() - wall_start
        checks += 1
        improvement = None
        improvement_per_second = None
        if previous_validation_loss is not None and previous_validation_wall is not None:
            improvement = previous_validation_loss - validation_loss
            elapsed = max(validation_wall - previous_validation_wall, 1e-9)
            improvement_per_second = improvement / elapsed
            if (
                checks >= min_checks
                and (
                    improvement < min_delta
                    or improvement_per_second < min_improvement_per_second
                )
            ):
                stale_checks += 1
            else:
                stale_checks = 0
        previous_validation_loss = validation_loss
        previous_validation_wall = validation_wall

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_step = phase_step
            best_theta = model.theta.detach().clone()

        validation_row = {
            "global_step": global_step,
            "phase": phase,
            "phase_step": phase_step + 1,
            "budget": budget,
            "validation_budget": validation_budget,
            "lr": lr,
            "validation_loss_bits": validation_loss,
            "validation_s": validation_s,
            "validation_improvement_bits": improvement,
            "validation_improvement_bits_per_s": improvement_per_second,
            "stale_checks": stale_checks,
            "best_validation_loss_bits": best_validation_loss,
            "best_validation_phase_step": best_validation_step,
            "wall_s": validation_wall,
        }
        rows.append(validation_row)
        _write_jsonl(history_jsonl, validation_row)

        if target_nll is not None and validation_loss <= target_nll:
            reached_target = True
            stop_reason = "target_nll"
            break
        if stale_checks >= patience:
            stop_reason = "validation_stall"
            break
    else:
        phase_step = max_steps - 1

    if math.isfinite(best_validation_loss):
        _restore_theta(
            model,
            best_theta,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        restore_row = {
            "global_step": global_step,
            "phase": phase,
            "phase_step": best_validation_step,
            "budget": budget,
            "validation_budget": validation_budget,
            "lr": lr,
            "restored_best_validation": True,
            "best_validation_loss_bits": best_validation_loss,
            "stop_reason": stop_reason,
            "wall_s": time.perf_counter() - wall_start,
        }
        rows.append(restore_row)
        _write_jsonl(history_jsonl, restore_row)
    else:
        best_validation_loss, validation_s = _loss_only(model, validation_budget)
        best_validation_step = int(locals().get("phase_step", -1))
        validation_row = {
            "global_step": global_step,
            "phase": phase,
            "phase_step": best_validation_step,
            "budget": budget,
            "validation_budget": validation_budget,
            "lr": lr,
            "validation_loss_bits": best_validation_loss,
            "validation_s": validation_s,
            "stop_reason": stop_reason,
            "wall_s": time.perf_counter() - wall_start,
        }
        rows.append(validation_row)
        _write_jsonl(history_jsonl, validation_row)

    summary = {
        "phase": phase,
        "budget": budget,
        "validation_budget": validation_budget,
        "lr": lr,
        "steps": global_step - first_global_step,
        "checks": checks,
        "stop_reason": stop_reason,
        "best_validation_loss_bits": best_validation_loss,
        "best_validation_phase_step": best_validation_step,
    }
    return global_step, summary, reached_target


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-tree", type=Path, default=DEFAULT_SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=DEFAULT_FAMILIES_FILE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/gpurec_hogenom_multifidelity_adagrad_route"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--family-chunk-size", type=int, default=300)
    parser.add_argument("--clade-budget", type=int, default=315_000)
    parser.add_argument("--batch-packing", default="depth_first_fit")
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=100.0)
    parser.add_argument("--theta-init-d", type=float, default=0.05)
    parser.add_argument("--theta-init-l", type=float, default=0.05)
    parser.add_argument("--theta-init-t", type=float, default=0.05)
    parser.add_argument(
        "--schedule-mode",
        choices=("fixed", "adaptive"),
        default="fixed",
    )
    parser.add_argument("--warmup-budget", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--warmup-lr", type=float, default=1.0)
    parser.add_argument("--bridge-budget", type=int, default=16)
    parser.add_argument("--bridge-steps", type=int, default=35)
    parser.add_argument("--bridge-lr", type=float, default=0.5)
    parser.add_argument("--repair-budget", type=int, default=32)
    parser.add_argument("--repair-steps", type=int, default=30)
    parser.add_argument("--repair-lr", type=float, default=0.5)
    parser.add_argument("--validation-budget", type=int, default=128)
    parser.add_argument(
        "--adaptive-initial-budget",
        type=int,
        default=4,
        help=(
            "Optional first adaptive solver budget before the fixed8 warmup; "
            "use 0 to start directly at --warmup-budget."
        ),
    )
    parser.add_argument(
        "--adaptive-initial-validation-budget",
        type=int,
        default=16,
        help="Validation budget used to decide when the initial adaptive phase stalls.",
    )
    parser.add_argument("--adaptive-warmup-validation-budget", type=int, default=32)
    parser.add_argument("--adaptive-bridge-validation-budget", type=int, default=32)
    parser.add_argument("--adaptive-max-steps-per-phase", type=int, default=256)
    parser.add_argument("--adaptive-check-interval", type=int, default=10)
    parser.add_argument("--adaptive-repair-check-interval", type=int, default=10)
    parser.add_argument("--adaptive-min-checks", type=int, default=2)
    parser.add_argument("--adaptive-patience", type=int, default=1)
    parser.add_argument("--adaptive-min-delta", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-min-improvement-per-second",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--adaptive-repair-min-improvement-per-second",
        type=float,
        default=0.0,
    )
    parser.add_argument("--adaptive-target-nll", type=float, default=None)
    parser.add_argument("--adaptive-max-wall-s", type=float, default=300.0)
    parser.add_argument(
        "--adaptive-final-validation-reserve-s",
        type=float,
        default=8.0,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_jsonl = args.out_dir / "history.jsonl"
    if history_jsonl.exists():
        history_jsonl.unlink()

    wall_start = time.perf_counter()
    config = RunConfig(
        species_tree=args.species_tree,
        families_file=args.families_file,
        out_dir=args.out_dir,
        mode="specieswise",
        device=args.device,
        dtype=args.dtype,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        batch_packing=args.batch_packing,
        max_wave_size=args.max_wave_size,
        fixed_iters_e=args.warmup_budget,
        fixed_iters_pi=args.warmup_budget,
        neumann_terms=args.warmup_budget,
        adaptive_iters=False,
        adaptive_neumann_terms=False,
        final_check_iters=0,
        theta_init_d=args.theta_init_d,
        theta_init_l=args.theta_init_l,
        theta_init_t=args.theta_init_t,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
        optimizer="adagrad",
        steps=1,
        checkpoint_every=0,
    )
    model = build_alerax_workflow_model(config)
    model.materialize_batches()

    lower_bound = math.log2(args.min_rate)
    upper_bound = math.log2(args.max_rate)
    rows: list[dict[str, Any]] = []
    global_step = 0
    fixed_phases = (
        (
            f"fixed{args.warmup_budget}_warmup",
            args.warmup_budget,
            args.warmup_lr,
            args.warmup_steps,
        ),
        (
            f"fixed{args.bridge_budget}_bridge",
            args.bridge_budget,
            args.bridge_lr,
            args.bridge_steps,
        ),
        (
            f"fixed{args.repair_budget}_repair",
            args.repair_budget,
            args.repair_lr,
            args.repair_steps,
        ),
    )
    actual_schedule: list[dict[str, Any]] = []
    if args.schedule_mode == "fixed":
        for phase, budget, lr, steps in fixed_phases:
            global_step = _run_phase(
                model=model,
                phase=phase,
                budget=budget,
                lr=lr,
                steps=steps,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                wall_start=wall_start,
                first_global_step=global_step,
                history_jsonl=history_jsonl,
                rows=rows,
            )
            phase_loss, phase_loss_s = _loss_only(model, budget)
            row = {
                "global_step": global_step,
                "phase": phase,
                "phase_step": steps,
                "budget": budget,
                "lr": lr,
                "post_phase_loss_bits": phase_loss,
                "loss_only_s": phase_loss_s,
                "wall_s": time.perf_counter() - wall_start,
            }
            rows.append(row)
            _write_jsonl(history_jsonl, row)
            actual_schedule.append(
                {"phase": phase, "budget": budget, "lr": lr, "steps": steps}
            )
    else:
        adaptive_phases_list: list[
            tuple[str, int, int, float, int, float, float | None]
        ] = []
        if args.adaptive_initial_budget > 0:
            adaptive_phases_list.append(
                (
                    f"fixed{args.adaptive_initial_budget}_initial",
                    args.adaptive_initial_budget,
                    args.adaptive_initial_validation_budget,
                    args.warmup_lr,
                    args.adaptive_check_interval,
                    args.adaptive_min_improvement_per_second,
                    None,
                )
            )
        adaptive_phases_list.extend(
            [
                (
                    f"fixed{args.warmup_budget}_warmup",
                    args.warmup_budget,
                    args.adaptive_warmup_validation_budget,
                    args.warmup_lr,
                    args.adaptive_check_interval,
                    args.adaptive_min_improvement_per_second,
                    None,
                ),
                (
                    f"fixed{args.bridge_budget}_bridge",
                    args.bridge_budget,
                    args.adaptive_bridge_validation_budget,
                    args.bridge_lr,
                    args.adaptive_check_interval,
                    args.adaptive_min_improvement_per_second,
                    None,
                ),
                (
                    f"fixed{args.repair_budget}_repair",
                    args.repair_budget,
                    args.validation_budget,
                    args.repair_lr,
                    args.adaptive_repair_check_interval,
                    args.adaptive_repair_min_improvement_per_second,
                    args.adaptive_target_nll,
                ),
            ]
        )
        for (
            phase,
            budget,
            validation_budget,
            lr,
            check_interval,
            min_improvement_per_second,
            target_nll,
        ) in adaptive_phases_list:
            global_step, phase_summary, reached_target = _run_adaptive_phase(
                model=model,
                phase=phase,
                budget=budget,
                validation_budget=validation_budget,
                lr=lr,
                max_steps=args.adaptive_max_steps_per_phase,
                check_interval=check_interval,
                min_checks=args.adaptive_min_checks,
                patience=args.adaptive_patience,
                min_delta=args.adaptive_min_delta,
                min_improvement_per_second=min_improvement_per_second,
                target_nll=target_nll,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                wall_start=wall_start,
                max_wall_s=args.adaptive_max_wall_s,
                final_validation_reserve_s=args.adaptive_final_validation_reserve_s,
                first_global_step=global_step,
                history_jsonl=history_jsonl,
                rows=rows,
            )
            actual_schedule.append(phase_summary)
            if reached_target:
                break
            if (
                time.perf_counter() - wall_start
                >= args.adaptive_max_wall_s - args.adaptive_final_validation_reserve_s
            ):
                break

    validation_loss, validation_s = _loss_only(model, args.validation_budget)
    final_wall_s = time.perf_counter() - wall_start
    validation_row = {
        "global_step": global_step,
        "phase": "fixed128_validation",
        "phase_step": 0,
        "budget": args.validation_budget,
        "loss_bits": validation_loss,
        "loss_only_s": validation_s,
        "wall_s": final_wall_s,
    }
    rows.append(validation_row)
    _write_jsonl(history_jsonl, validation_row)

    _write_csv(args.out_dir / "optimization_history.csv", rows)
    torch.save(model.theta.detach().cpu(), args.out_dir / "theta_final.pt")
    save_checkpoint(
        args.out_dir / "checkpoints" / "final.pt",
        config=config,
        model=model,
        optimizer=None,
        step=global_step,
        status={
            "status": "complete",
            "reason": "fixed128_validation",
            "wall_s": final_wall_s,
            "validation_loss_bits": validation_loss,
        },
        row=validation_row,
        next_step=global_step,
        optimizer_phase="fixed128_validation",
    )
    summary = {
        "status": "complete",
        "families": int(model.n_families),
        "species": int(model.n_species),
        "batches": len(model.batch_metadata),
        "wall_s": final_wall_s,
        "schedule_mode": args.schedule_mode,
        "validation_budget": args.validation_budget,
        "validation_loss_bits": validation_loss,
        "validation_s": validation_s,
        "schedule": actual_schedule,
        "adaptive_controls": (
            None
            if args.schedule_mode != "adaptive"
            else {
                "check_interval": args.adaptive_check_interval,
                "repair_check_interval": args.adaptive_repair_check_interval,
                "initial_budget": args.adaptive_initial_budget,
                "initial_validation_budget": (
                    args.adaptive_initial_validation_budget
                    if args.adaptive_initial_budget > 0
                    else None
                ),
                "min_checks": args.adaptive_min_checks,
                "patience": args.adaptive_patience,
                "min_delta": args.adaptive_min_delta,
                "min_improvement_per_second": (
                    args.adaptive_min_improvement_per_second
                ),
                "repair_min_improvement_per_second": (
                    args.adaptive_repair_min_improvement_per_second
                ),
                "target_nll": args.adaptive_target_nll,
                "max_wall_s": args.adaptive_max_wall_s,
                "final_validation_reserve_s": (
                    args.adaptive_final_validation_reserve_s
                ),
            }
        ),
        "counts_or_alerax_event_summaries_used": False,
        "theta_init_rates": [args.theta_init_d, args.theta_init_l, args.theta_init_t],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

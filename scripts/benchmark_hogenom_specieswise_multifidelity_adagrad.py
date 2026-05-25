"""Counts-free multifidelity Adagrad route for HOGENOM specieswise rates.

The route starts from uniform 0.05 D/L/T rates on every species branch and uses
only gpurec full-objective gradients:

1. fixed8 Adagrad warmup,
2. fixed16 Adagrad bridge,
3. fixed32 Adagrad repair,
4. fixed128 loss-only validation.

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
    phases = (
        ("fixed8_warmup", args.warmup_budget, args.warmup_lr, args.warmup_steps),
        ("fixed16_bridge", args.bridge_budget, args.bridge_lr, args.bridge_steps),
        ("fixed32_repair", args.repair_budget, args.repair_lr, args.repair_steps),
    )
    for phase, budget, lr, steps in phases:
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
        "validation_budget": args.validation_budget,
        "validation_loss_bits": validation_loss,
        "validation_s": validation_s,
        "schedule": [
            {"phase": phase, "budget": budget, "lr": lr, "steps": steps}
            for phase, budget, lr, steps in phases
        ],
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

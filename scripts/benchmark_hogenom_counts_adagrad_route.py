#!/usr/bin/env python3
"""Benchmark the HOGENOM uniform-start counts-guided Adagrad route.

This script is intentionally narrow: it reproduces the fastest clean-start
route found during HOGENOM specieswise optimization experiments while
recording the required uniform 0.05 specieswise starting point before applying
the counts-guided optimizer jump.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from gpurec.workflow import RunConfig
from gpurec.workflow.checkpoint import load_checkpoint, save_checkpoint
from gpurec.workflow.diagnostics import write_json_strict
from gpurec.workflow.model_factory import build_alerax_workflow_model
from gpurec.workflow.optimize import OptimizationRunner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIES_TREE = ROOT / "tests/data/HOGENOM/hogenom/hogenom_S.tree"
DEFAULT_FAMILIES_FILE = ROOT / "tests/data/HOGENOM/hogenom/hogenom_families.local.txt"
DEFAULT_COUNTS_FILE = (
    ROOT
    / "tests/data/HOGENOM/hogenom/output_alerax_corrected/reconciliations/"
    / "totalSpeciesEventCounts.txt"
)
TARGET_NLL_BITS = 526_822.875
REFERENCE_BEST_NLL_BITS = 526_789.625
REFERENCE_PLUS_10_NLL_BITS = REFERENCE_BEST_NLL_BITS + 10.0


def _load_species_count_rates(path: Path) -> dict[str, tuple[float, float, float]]:
    rates: dict[str, tuple[float, float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            label = str(row["species_label"]).strip()
            copies = float(row["copies"])
            duplications = float(row["duplications"])
            losses = float(row["losses"])
            transfers = float(row["transfers"])
            denom = copies + 0.1
            rates[label] = (
                max(1e-5, 2.0 * (duplications + 0.1) / denom),
                max(1e-5, 2.0 * (losses + 0.1) / denom),
                max(1e-5, 0.5 * (transfers + 0.1) / denom),
            )
    return rates


def _counts_theta_for_model(
    model: Any,
    config: RunConfig,
    counts_file: Path,
) -> tuple[torch.Tensor, int]:
    rates_by_species = _load_species_count_rates(counts_file)
    theta_rows: list[list[float]] = []
    missing = 0
    for species in model.species_names[: int(model.theta.shape[0])]:
        rates = rates_by_species.get(species)
        if rates is None:
            missing += 1
            rates = config.theta_init_rates
        theta_rows.append([math.log2(float(value)) for value in rates])
    theta = torch.as_tensor(
        theta_rows,
        device=model.theta.device,
        dtype=model.theta.dtype,
    )
    return theta, missing


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clear_solver_runtime_state(model: Any) -> None:
    model.theta.grad = None
    for static in getattr(model, "cached_static_states", []) or []:
        if hasattr(static, "warm_E"):
            static.warm_E = None
        if hasattr(static, "last_solver_stats"):
            static.last_solver_stats = None


def _projected_gradient(
    theta: torch.Tensor,
    grad: torch.Tensor,
    *,
    min_rate: float,
    max_rate: float,
) -> torch.Tensor:
    lower = math.log2(min_rate)
    upper = math.log2(max_rate)
    return theta - torch.clamp(theta - grad, min=lower, max=upper)


def _evaluate_model(model: Any, config: RunConfig) -> dict[str, float]:
    _clear_solver_runtime_state(model)
    _synchronize()
    started = time.perf_counter()
    loss = model.full_loss()
    loss.backward()
    _synchronize()
    elapsed_s = time.perf_counter() - started
    if model.theta.grad is None:
        raise RuntimeError("missing gradient after full_loss backward")
    grad = model.theta.grad.detach()
    projected = _projected_gradient(
        model.theta.detach(),
        grad,
        min_rate=config.min_rate,
        max_rate=config.max_rate,
    )
    return {
        "loss_bits": float(loss.detach().cpu()),
        "grad_inf": float(grad.detach().abs().amax().cpu()) if grad.numel() else 0.0,
        "projected_grad_inf": (
            float(projected.detach().abs().amax().cpu()) if projected.numel() else 0.0
        ),
        "elapsed_s": elapsed_s,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _write_uniform_start_counts_checkpoint(
    *,
    config: RunConfig,
    counts_file: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.out_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    config.write_json(config.out_dir / "run_config.json")

    model = build_alerax_workflow_model(config)
    try:
        uniform_rate = torch.as_tensor(
            config.theta_init_rates,
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        uniform_theta = torch.log2(uniform_rate).reshape(1, 3)
        theta_rows = model.theta.detach().reshape(-1, 3)
        max_uniform_theta_delta = float(
            (theta_rows - uniform_theta).detach().abs().amax().cpu()
        )
        initial_eval = _evaluate_model(model, config)
        initial_row = {
            "step": 0,
            "optimizer/phase": "uniform_start",
            "optimizer/eval_position": "initial",
            "optimizer/step_applied": False,
            "likelihood/data_nll_bits": initial_eval["loss_bits"],
            "likelihood/log_likelihood_bits": -initial_eval["loss_bits"],
            "grad/inf": initial_eval["grad_inf"],
            "grad/projected_inf": initial_eval["projected_grad_inf"],
            "theta_step_inf": 0.0,
            "step_s": initial_eval["elapsed_s"],
        }
        _append_jsonl(history_path, initial_row)
        save_checkpoint(
            checkpoint_dir / "uniform_start.pt",
            config=config,
            model=model,
            optimizer=None,
            step=0,
            next_step=0,
            status={
                "status": "initialized",
                "reason": "uniform_0p05_start",
                "elapsed_s": time.perf_counter() - started,
                "best_nll_bits": initial_eval["loss_bits"],
                "best_step": 0,
                "previous_objective": initial_eval["loss_bits"],
                "stable_loss_steps": 0,
            },
            row=initial_row,
            optimizer_phase="uniform_start",
        )

        counts_theta, missing_species = _counts_theta_for_model(
            model,
            config,
            counts_file,
        )
        before_counts = model.theta.detach().clone()
        with torch.no_grad():
            model.theta.copy_(counts_theta)
            model.clamp_theta_(config.min_rate, config.max_rate)
            model.theta.grad = None
        model.clear()
        theta_step_inf = float(
            (model.theta.detach() - before_counts).detach().abs().amax().cpu()
        )
        counts_eval = _evaluate_model(model, config)
        counts_row = {
            "step": 0,
            "optimizer/phase": "counts_guided_initialization",
            "optimizer/eval_position": "post_step",
            "optimizer/step_applied": True,
            "optimizer/counts_file": str(counts_file),
            "optimizer/missing_count_species": missing_species,
            "likelihood/data_nll_bits": counts_eval["loss_bits"],
            "likelihood/log_likelihood_bits": -counts_eval["loss_bits"],
            "grad/inf": counts_eval["grad_inf"],
            "grad/projected_inf": counts_eval["projected_grad_inf"],
            "theta_step_inf": theta_step_inf,
            "step_s": counts_eval["elapsed_s"],
        }
        _append_jsonl(history_path, counts_row)
        elapsed_s = time.perf_counter() - started
        counts_checkpoint = checkpoint_dir / "counts_guided_start.pt"
        save_checkpoint(
            counts_checkpoint,
            config=config,
            model=model,
            optimizer=None,
            step=0,
            next_step=0,
            status={
                "status": "initialized",
                "reason": "counts_guided_step_from_uniform_0p05",
                "elapsed_s": elapsed_s,
                "best_nll_bits": counts_eval["loss_bits"],
                "best_step": 0,
                "previous_objective": counts_eval["loss_bits"],
                "stable_loss_steps": 0,
                "resume_optimizer_state": "reset_after_counts_guided_step",
            },
            row=counts_row,
            optimizer_phase="counts_guided_initialization",
        )
        summary = {
            "out_dir": str(config.out_dir),
            "status": "initialized",
            "reason": "counts_guided_step_from_uniform_0p05",
            "wall_s": elapsed_s,
            "summary_elapsed_s": elapsed_s,
            "uniform_theta_max_abs_delta": max_uniform_theta_delta,
            "uniform_theta_init_rates": list(config.theta_init_rates),
            "initial_nll_bits": initial_eval["loss_bits"],
            "initial_projected_grad_inf": initial_eval["projected_grad_inf"],
            "counts_guided_nll_bits": counts_eval["loss_bits"],
            "counts_guided_projected_grad_inf": counts_eval["projected_grad_inf"],
            "counts_guided_theta_step_inf": theta_step_inf,
            "counts_guided_checkpoint": str(counts_checkpoint),
            "missing_count_species": missing_species,
            "final_nll_bits": counts_eval["loss_bits"],
            "best_nll_bits": counts_eval["loss_bits"],
            "final_check_loss_bits": None,
            "final_step": 0,
            "final_phase": "counts_guided_initialization",
            "resume_from": None,
        }
        write_json_strict(config.out_dir / "summary.json", summary)
        return summary
    finally:
        model.close()


def _reset_checkpoint(
    src: Path,
    dst: Path,
    *,
    clear_best: bool,
) -> Path:
    payload = dict(load_checkpoint(src))
    payload["optimizer_state"] = None
    payload["optimizer_phase"] = None
    status = dict(payload.get("status") or {})
    status["previous_objective"] = None
    status["stable_loss_steps"] = 0
    if clear_best:
        status["best_nll_bits"] = None
        status["best_step"] = None
    payload["status"] = status
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(dst)
    return dst


def _config(
    *,
    species_tree: Path,
    families_file: Path,
    out_dir: Path,
    resume_from: Path | None,
    optimizer: str,
    steps: int,
    lr: float,
    fixed_iters: int,
    final_check_iters: int,
    log_every: int,
    checkpoint_every: int,
    device: str,
) -> RunConfig:
    return RunConfig(
        species_tree=species_tree,
        families_file=families_file,
        out_dir=out_dir,
        mode="specieswise",
        device=device,
        dtype="float32",
        family_chunk_size=300,
        clade_budget=315_000,
        batch_packing="depth_first_fit",
        fixed_iters_e=fixed_iters,
        fixed_iters_pi=fixed_iters,
        neumann_terms=fixed_iters,
        final_check_iters=final_check_iters,
        adaptive_iters=False,
        adaptive_neumann_terms=False,
        min_rate=1e-10,
        max_rate=100.0,
        optimizer=optimizer,
        steps=steps,
        lr=lr,
        loss_patience=0,
        best_likelihood_patience=0,
        checkpoint_every=checkpoint_every,
        log_every=log_every,
        resume_from=resume_from,
    )


def _run_stage(
    *,
    config: RunConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    runner = OptimizationRunner(config)
    result = runner.run()
    wall_s = time.perf_counter() - started
    summary_path = config.out_dir / "summary.json"
    summary = load_json(summary_path)
    history_rows = load_history(config.out_dir / "history.jsonl")
    final_row = history_rows[-1] if history_rows else {}
    return {
        "out_dir": str(config.out_dir),
        "status": result.status,
        "reason": result.reason,
        "wall_s": wall_s,
        "summary_elapsed_s": summary.get("elapsed_s"),
        "final_nll_bits": summary.get("final_nll_bits"),
        "best_nll_bits": summary.get("best_nll_bits"),
        "final_check_loss_bits": final_row.get("optimizer/final_check_loss_bits"),
        "final_step": final_row.get("step"),
        "final_phase": final_row.get("optimizer/phase"),
        "resume_from": str(config.resume_from) if config.resume_from else None,
    }


def load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: Path) -> list[dict[str, Any]]:
    import json

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-tree", type=Path, default=DEFAULT_SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=DEFAULT_FAMILIES_FILE)
    parser.add_argument("--counts-file", type=Path, default=DEFAULT_COUNTS_FILE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/gpurec_hogenom_counts_adagrad_route_clean"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists() and not args.keep_existing:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    route_started = time.perf_counter()
    stages: list[dict[str, Any]] = []

    stage0_dir = out_dir / "stage0_uniform_start_counts_guided"
    stage0_config = _config(
        species_tree=args.species_tree,
        families_file=args.families_file,
        out_dir=stage0_dir,
        resume_from=None,
        optimizer="adagrad",
        steps=1,
        lr=1.0,
        fixed_iters=16,
        final_check_iters=0,
        log_every=1,
        checkpoint_every=0,
        device=args.device,
    )
    stage0 = _write_uniform_start_counts_checkpoint(
        config=stage0_config,
        counts_file=args.counts_file,
    )
    stages.append(stage0)
    counts_start = Path(stage0["counts_guided_checkpoint"])

    stage1_dir = out_dir / "stage1_counts_adagrad16_lr1_to40"
    stage1_config = _config(
        species_tree=args.species_tree,
        families_file=args.families_file,
        out_dir=stage1_dir,
        resume_from=counts_start,
        optimizer="adagrad",
        steps=40,
        lr=1.0,
        fixed_iters=16,
        final_check_iters=0,
        log_every=10,
        checkpoint_every=20,
        device=args.device,
    )
    stages.append(_run_stage(config=stage1_config))

    stage1_reset = _reset_checkpoint(
        stage1_dir / "checkpoints/latest.pt",
        out_dir / "checkpoints/stage1_step40_reset.pt",
        clear_best=False,
    )
    stage2_dir = out_dir / "stage2_reset_adagrad16_lr0p5_to100"
    stage2_config = _config(
        species_tree=args.species_tree,
        families_file=args.families_file,
        out_dir=stage2_dir,
        resume_from=stage1_reset,
        optimizer="adagrad",
        steps=100,
        lr=0.5,
        fixed_iters=16,
        final_check_iters=0,
        log_every=10,
        checkpoint_every=20,
        device=args.device,
    )
    stages.append(_run_stage(config=stage2_config))

    stage2_reset = _reset_checkpoint(
        stage2_dir / "checkpoints/latest.pt",
        out_dir / "checkpoints/stage2_step100_reset.pt",
        clear_best=True,
    )
    stage3_dir = out_dir / "stage3_reset_adagrad32_lr0p2_to110_e128"
    stage3_config = _config(
        species_tree=args.species_tree,
        families_file=args.families_file,
        out_dir=stage3_dir,
        resume_from=stage2_reset,
        optimizer="adagrad",
        steps=110,
        lr=0.2,
        fixed_iters=32,
        final_check_iters=128,
        log_every=5,
        checkpoint_every=10,
        device=args.device,
    )
    stages.append(_run_stage(config=stage3_config))

    total_wall_s = time.perf_counter() - route_started
    final_stage = stages[-1]
    final_check = final_stage.get("final_check_loss_bits")
    final_nll = final_check if final_check is not None else final_stage["final_nll_bits"]
    target_nll = min(TARGET_NLL_BITS, REFERENCE_PLUS_10_NLL_BITS)
    summary = {
        "status": "accepted" if float(final_nll) <= target_nll else "not_accepted",
        "target_nll_bits": target_nll,
        "legacy_target_nll_bits": TARGET_NLL_BITS,
        "reference_best_nll_bits": REFERENCE_BEST_NLL_BITS,
        "reference_plus_10_nll_bits": REFERENCE_PLUS_10_NLL_BITS,
        "final_nll_bits": final_nll,
        "final_fixed32_nll_bits": final_stage["final_nll_bits"],
        "total_wall_s": total_wall_s,
        "stage_wall_s_sum": sum(float(stage["wall_s"]) for stage in stages),
        "stages": stages,
    }
    write_json_strict(out_dir / "route_summary.json", summary)
    print((out_dir / "route_summary.json").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()

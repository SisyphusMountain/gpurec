#!/usr/bin/env python3
"""Bounded L2-penalty CV pilot for HOGENOM specieswise optimization.

This script intentionally keeps the regularization experiment outside the
core gpurec implementation.  It trains specieswise log2-rate parameters on a
small deterministic family split, evaluates raw held-out NLL, and writes both
machine-readable results and the markdown report requested for the pilot.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from gpurec import GeneReconModel, SolverOptions, clamp_log_rate_, project_rate_gradient_


REPO = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO / "benchmarks/large_dataset_capacity/reports/hogenom_penalty_cv_subagent.md"
DEFAULT_OUTPUT_ROOT = REPO / "benchmarks/large_dataset_capacity/output/hogenom_penalty_cv_pilot"


@dataclass(frozen=True)
class TrialConfig:
    penalty_lambda: float
    steps: int
    lr: float
    clip_grad_norm: float
    pi_iters: int
    neumann_terms: int
    self_loop_solver: str
    min_rate: float
    max_rate: float
    theta_init_rate: float
    l2_target_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-tree", type=Path, default=REPO / "tests/data/hogenom_S.tree")
    parser.add_argument("--gene-tree-dir", type=Path, default=REPO / "tests/data/hogenom_trees")
    parser.add_argument("--max-families", type=int, default=80)
    parser.add_argument("--val-families", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--penalties", default="0,0.001,0.01,0.1,1.0")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--clip-grad-norm", type=float, default=500.0)
    parser.add_argument("--pi-iters", type=int, default=16)
    parser.add_argument("--neumann-terms", type=int, default=16)
    parser.add_argument("--self-loop-solver", default="gmres")
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=2.0)
    parser.add_argument("--theta-init-rate", type=float, default=0.05)
    parser.add_argument("--l2-target-rate", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--family-chunk-size", type=int, default=0)
    parser.add_argument("--clade-budget", type=int, default=315_000)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def parse_penalties(raw: str) -> list[float]:
    penalties = [float(part) for part in raw.split(",") if part.strip()]
    if not penalties:
        raise ValueError("penalty grid is empty")
    return penalties


def select_families(gene_tree_dir: Path, *, max_families: int, val_families: int, seed: int):
    all_families = sorted(gene_tree_dir.glob("*.trees"))
    if not all_families:
        raise FileNotFoundError(f"no *.trees files found in {gene_tree_dir}")
    if max_families <= val_families:
        raise ValueError("--max-families must be greater than --val-families")

    rng = random.Random(seed)
    shuffled = list(all_families)
    rng.shuffle(shuffled)
    selected = shuffled[:max_families]
    val = selected[:val_families]
    train = selected[val_families:]
    return all_families, train, val


def build_model(args: argparse.Namespace, families: list[Path]) -> GeneReconModel:
    solver_options = SolverOptions(
        e_init=-1000.0,
        e_max_iter=2000,
        e_tol=1e-8,
        pi_iters=args.pi_iters,
        neumann_terms=args.neumann_terms,
        self_loop_solver=args.self_loop_solver,
        bicgstab_max_iter=500,
        bicgstab_tol=1e-7,
        bicgstab_breakdown_tol=1e-30,
        adjoint_pruning_threshold=1e-6,
        use_adjoint_pruning=True,
        pibar_side_threshold=0.0,
    )
    model = GeneReconModel(
        args.species_tree,
        families,
        mode="specieswise",
        device=args.device,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        batch_packing="depth_first_fit",
        max_wave_size=args.max_wave_size,
        solver_options=solver_options,
    )
    model.receiver_weights.requires_grad_(False)
    return model


@torch.no_grad()
def initialize_theta_(model: GeneReconModel, rate: float, *, min_rate: float, max_rate: float) -> None:
    model.theta.fill_(math.log2(rate))
    clamp_log_rate_(model.theta, min_rate=min_rate, max_rate=max_rate)
    model.clear_warm_starts()


@torch.no_grad()
def copy_theta_(dst: GeneReconModel, src: GeneReconModel, *, min_rate: float, max_rate: float) -> None:
    dst.theta.copy_(src.theta.detach())
    clamp_log_rate_(dst.theta, min_rate=min_rate, max_rate=max_rate)
    dst.clear_warm_starts()


def sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate_raw_nll(model: GeneReconModel, device: str) -> float:
    model.zero_grad(set_to_none=True)
    sync_if_cuda(device)
    loss = model()
    sync_if_cuda(device)
    return float(loss.detach().cpu())


def theta_stats(model: GeneReconModel, *, min_rate: float, max_rate: float) -> dict[str, float | int | bool]:
    with torch.no_grad():
        theta = model.theta.detach()
        rates = torch.pow(torch.tensor(2.0, device=theta.device, dtype=theta.dtype), theta)
        lower = math.log2(min_rate)
        upper = math.log2(max_rate)
        return {
            "theta_min": float(theta.min().cpu()),
            "theta_max": float(theta.max().cpu()),
            "theta_mean": float(theta.mean().cpu()),
            "theta_std": float(theta.std(unbiased=False).cpu()),
            "rate_min": float(rates.min().cpu()),
            "rate_max": float(rates.max().cpu()),
            "rate_mean": float(rates.mean().cpu()),
            "active_lower": int((theta <= lower + 1e-6).sum().cpu()),
            "active_upper": int((theta >= upper - 1e-6).sum().cpu()),
            "param_count": int(theta.numel()),
            "finite": bool(torch.isfinite(theta).all().cpu()),
        }


def run_trial(
    args: argparse.Namespace,
    train_model: GeneReconModel,
    val_model: GeneReconModel,
    penalty_lambda: float,
) -> dict:
    config = TrialConfig(
        penalty_lambda=penalty_lambda,
        steps=args.steps,
        lr=args.lr,
        clip_grad_norm=args.clip_grad_norm,
        pi_iters=args.pi_iters,
        neumann_terms=args.neumann_terms,
        self_loop_solver=args.self_loop_solver,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
        theta_init_rate=args.theta_init_rate,
        l2_target_rate=args.l2_target_rate,
    )
    target = torch.full_like(train_model.theta, math.log2(args.l2_target_rate))
    initialize_theta_(train_model, args.theta_init_rate, min_rate=args.min_rate, max_rate=args.max_rate)
    copy_theta_(val_model, train_model, min_rate=args.min_rate, max_rate=args.max_rate)
    optimizer = torch.optim.Adam([train_model.theta], lr=args.lr)

    history = []
    trial_t0 = time.perf_counter()
    initial_val = evaluate_raw_nll(val_model, args.device)
    for step in range(args.steps + 1):
        train_model.zero_grad(set_to_none=True)
        sync_if_cuda(args.device)
        step_t0 = time.perf_counter()
        raw_loss = train_model()
        penalty = 0.5 * penalty_lambda * torch.sum((train_model.theta - target) ** 2)
        objective = raw_loss + penalty
        objective.backward()
        grad_norm_before_projection = float(train_model.theta.grad.detach().norm().cpu())
        project_rate_gradient_(train_model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
        projected_grad_norm = float(train_model.theta.grad.detach().norm().cpu())
        torch.nn.utils.clip_grad_norm_([train_model.theta], args.clip_grad_norm)
        clipped_grad_norm = float(train_model.theta.grad.detach().norm().cpu())

        if step < args.steps:
            optimizer.step()
            clamp_log_rate_(train_model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
        sync_if_cuda(args.device)
        step_s = time.perf_counter() - step_t0

        copy_theta_(val_model, train_model, min_rate=args.min_rate, max_rate=args.max_rate)
        val_t0 = time.perf_counter()
        val_raw_nll = evaluate_raw_nll(val_model, args.device)
        val_s = time.perf_counter() - val_t0
        stats = theta_stats(train_model, min_rate=args.min_rate, max_rate=args.max_rate)
        row = {
            "step": step,
            "train_raw_nll_bits": float(raw_loss.detach().cpu()),
            "train_penalty_bits": float(penalty.detach().cpu()),
            "train_objective_bits": float(objective.detach().cpu()),
            "val_raw_nll_bits": val_raw_nll,
            "grad_norm_before_projection": grad_norm_before_projection,
            "projected_grad_norm": projected_grad_norm,
            "clipped_grad_norm": clipped_grad_norm,
            "step_s": step_s,
            "val_s": val_s,
            **stats,
        }
        history.append(row)

    elapsed_s = time.perf_counter() - trial_t0
    final = history[-1]
    stable = bool(final["finite"] and final["rate_min"] >= args.min_rate * 0.999 and final["rate_max"] <= args.max_rate * 1.001)
    return {
        "config": asdict(config),
        "initial_val_raw_nll_bits": initial_val,
        "final": final,
        "history": history,
        "elapsed_s": elapsed_s,
        "stable": stable,
        "validation_nll_per_family": final["val_raw_nll_bits"] / len(val_model.families),
        "train_nll_per_family": final["train_raw_nll_bits"] / len(train_model.families),
    }


def select_recommendation(trials: list[dict]) -> dict:
    stable_trials = [trial for trial in trials if trial["stable"]]
    candidates = stable_trials or trials
    return min(candidates, key=lambda trial: (trial["validation_nll_per_family"], trial["config"]["penalty_lambda"]))


def write_report(path: Path, payload: dict, command: str) -> None:
    selected = payload["selected"]
    rows = []
    for trial in payload["trials"]:
        final = trial["final"]
        rows.append(
            "| {lam:g} | {val:.6f} | {train:.6f} | {rate_min:.3e} | {rate_max:.3e} | "
            "{tmin:.3f} | {tmax:.3f} | {stable} | {elapsed:.2f} |".format(
                lam=trial["config"]["penalty_lambda"],
                val=trial["validation_nll_per_family"],
                train=trial["train_nll_per_family"],
                rate_min=final["rate_min"],
                rate_max=final["rate_max"],
                tmin=final["theta_min"],
                tmax=final["theta_max"],
                stable=trial["stable"],
                elapsed=trial["elapsed_s"],
            )
        )

    split = payload["split"]
    settings = payload["settings"]
    report = f"""# HOGENOM Specieswise L2 Penalty CV Pilot

## Scope

This is a bounded pilot, not the full 1055-family production optimization.  It uses the local HOGENOM fixture under `tests/data` with {split["selected_families"]} deterministically sampled families from {split["available_families"]} available families.

## Data Split

- Seed: `{split["seed"]}`
- Train families: `{split["train_families"]}`
- Validation families: `{split["val_families"]}`
- Split method: shuffle sorted `*.trees` with Python `random.Random(seed)`, then take validation first and training second from the bounded sample.
- Species tree: `{payload["species_tree"]}`
- Gene tree directory: `{payload["gene_tree_dir"]}`

## Penalty Grid

L2 penalty on specieswise log2-rate parameters:

```text
objective = train_raw_nll_bits + 0.5 * lambda * sum((theta - log2({settings["l2_target_rate"]}))^2)
```

Grid: `{settings["penalties"]}`

## Optimizer Settings

- Mode: `specieswise`
- Optimizer: `Adam`
- Steps per penalty: `{settings["steps"]}`
- Learning rate: `{settings["lr"]}`
- Gradient clipping: `{settings["clip_grad_norm"]}`
- Rate bounds: `[{settings["min_rate"]}, {settings["max_rate"]}]`
- Initial D/T/L rates: `{settings["theta_init_rate"]}`
- Solver: `{settings["self_loop_solver"]}`
- Forward Pi iterations: `{settings["pi_iters"]}`
- Backward/self-loop iterations: `{settings["neumann_terms"]}`
- Device: `{settings["device"]}`

## Validation Objective

The validation objective is raw held-out negative log likelihood in bits, evaluated on the validation families after copying the trained specieswise theta into a separate validation model.  The L2 penalty is not included in validation.

## Results

| lambda | val NLL/family | train NLL/family | rate min | rate max | theta min | theta max | stable | runtime s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
{chr(10).join(rows)}

## Selected Penalty Recommendation

Recommend `lambda={selected["config"]["penalty_lambda"]:g}` for the next larger specieswise run from this pilot.  It had the best stable held-out raw NLL per validation family: `{selected["validation_nll_per_family"]:.6f}` bits/family.

## Theta And Rate Stability

All trials were considered stable when theta stayed finite and natural rates stayed within the configured bounds.  Final selected stats:

```json
{json.dumps(selected["final"], indent=2)}
```

## Runtime

- Total elapsed: `{payload["elapsed_s"]:.2f}` s
- Output JSON: `{payload["results_json"]}`

## Commands Run

```bash
{command}
```

## Blockers

- Full 1055-family specieswise CV was not run in this bounded pilot because multiplying the full dataset by the penalty grid would be substantially more expensive.
- No GBM/tree prior was exercised; no ready script-level GBM prior hook was found during the bounded inspection, so this pilot focused on L2 as requested.
- This script uses the current worktree implementation of GMRES; it does not modify or independently validate core GMRES internals.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if not args.species_tree.exists():
        raise FileNotFoundError(args.species_tree)
    if not args.gene_tree_dir.is_dir():
        raise FileNotFoundError(args.gene_tree_dir)

    penalties = parse_penalties(args.penalties)
    all_families, train_families, val_families = select_families(
        args.gene_tree_dir,
        max_families=args.max_families,
        val_families=args.val_families,
        seed=args.seed,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    torch.set_float32_matmul_precision("high")
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    build_t0 = time.perf_counter()
    train_model = build_model(args, train_families)
    val_model = build_model(args, val_families)
    sync_if_cuda(args.device)
    build_s = time.perf_counter() - build_t0

    trials = []
    total_t0 = time.perf_counter()
    for penalty_lambda in penalties:
        trial = run_trial(args, train_model, val_model, penalty_lambda)
        trials.append(trial)
        trial_path = run_dir / f"trial_lambda_{penalty_lambda:g}.json"
        trial_path.write_text(json.dumps(trial, indent=2) + "\n")
        print(
            f"lambda={penalty_lambda:g} val_nll_per_family={trial['validation_nll_per_family']:.6f} "
            f"stable={trial['stable']} elapsed_s={trial['elapsed_s']:.2f}",
            flush=True,
        )

    selected = select_recommendation(trials)
    elapsed_s = time.perf_counter() - total_t0
    results_json = run_dir / "results.json"
    payload = {
        "species_tree": str(args.species_tree),
        "gene_tree_dir": str(args.gene_tree_dir),
        "split": {
            "seed": args.seed,
            "available_families": len(all_families),
            "selected_families": len(train_families) + len(val_families),
            "train_families": len(train_families),
            "val_families": len(val_families),
            "train_names": [path.name for path in train_families],
            "val_names": [path.name for path in val_families],
        },
        "settings": {
            "penalties": penalties,
            "steps": args.steps,
            "lr": args.lr,
            "clip_grad_norm": args.clip_grad_norm,
            "pi_iters": args.pi_iters,
            "neumann_terms": args.neumann_terms,
            "self_loop_solver": args.self_loop_solver,
            "min_rate": args.min_rate,
            "max_rate": args.max_rate,
            "theta_init_rate": args.theta_init_rate,
            "l2_target_rate": args.l2_target_rate,
            "device": args.device,
            "family_chunk_size": args.family_chunk_size,
            "clade_budget": args.clade_budget,
            "max_wave_size": args.max_wave_size,
            "build_s": build_s,
        },
        "trials": trials,
        "selected": selected,
        "elapsed_s": elapsed_s,
        "results_json": str(results_json),
    }
    results_json.write_text(json.dumps(payload, indent=2) + "\n")
    command = "python " + " ".join([str(Path(__file__)), *(__import__("sys").argv[1:])])
    write_report(args.report, payload, command)
    print(f"selected_lambda={selected['config']['penalty_lambda']:g}")
    print(f"results_json={results_json}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()

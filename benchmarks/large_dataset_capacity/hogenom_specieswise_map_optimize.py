#!/usr/bin/env python3
"""Run HOGENOM specieswise MAP optimization with solver controls."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from gpurec import GeneReconModel, SolverOptions, clamp_log_rate_, project_rate_gradient_


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO / "benchmarks/large_dataset_capacity/output/hogenom_specieswise_map"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-tree", type=Path, default=REPO / "tests/data/hogenom_S.tree")
    parser.add_argument("--gene-tree-dir", type=Path, default=REPO / "tests/data/hogenom_trees")
    parser.add_argument("--max-families", type=int, default=0, help="0 means all families.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--penalty-lambda", type=float, default=0.01)
    parser.add_argument("--penalty-lambdas", default=None, help="Optional comma-separated D,T,L L2 strengths.")
    parser.add_argument("--tree-penalty-lambdas", default=None, help="Optional comma-separated D,T,L unit-branch Brownian edge strengths.")
    parser.add_argument("--root-penalty-lambdas", default=None, help="Optional comma-separated D,T,L root-anchor strengths.")
    parser.add_argument("--l2-target-rate", type=float, default=0.05)
    parser.add_argument("--theta-init-rate", type=float, default=0.05)
    parser.add_argument("--init-theta", type=Path, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=500.0)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=2.0)
    parser.add_argument("--stop-rate-max", type=float, default=0.0, help="Optional early stop when recorded rate_max reaches this value; 0 disables.")
    parser.add_argument("--pi-iters", type=int, default=16)
    parser.add_argument("--neumann-terms", type=int, default=16)
    parser.add_argument("--self-loop-solver", default="gmres")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--family-chunk-size", type=int, default=300)
    parser.add_argument("--clade-budget", type=int, default=315_000)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summarize-run", type=Path, default=None, help="Write summary/report for an existing run dir.")
    return parser.parse_args()


def sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def theta_stats(theta: torch.Tensor, *, min_rate: float, max_rate: float) -> dict:
    with torch.no_grad():
        rates = torch.pow(torch.tensor(2.0, dtype=theta.dtype, device=theta.device), theta)
        lower = math.log2(min_rate)
        upper = math.log2(max_rate)
        return {
            "theta_min": float(theta.min().detach().cpu()),
            "theta_max": float(theta.max().detach().cpu()),
            "theta_mean": float(theta.mean().detach().cpu()),
            "theta_std": float(theta.std(unbiased=False).detach().cpu()),
            "rate_min": float(rates.min().detach().cpu()),
            "rate_max": float(rates.max().detach().cpu()),
            "rate_mean": float(rates.mean().detach().cpu()),
            "active_lower": int((theta <= lower + 1e-6).sum().detach().cpu()),
            "active_upper": int((theta >= upper - 1e-6).sum().detach().cpu()),
            "finite": bool(torch.isfinite(theta).all().detach().cpu()),
        }


def write_report(run_dir: Path, payload: dict) -> None:
    history = payload["history"]
    first = history[0]
    last = history[-1]
    report = f"""# HOGENOM Specieswise MAP Optimization

## Scope

This run optimizes specieswise HOGENOM parameters with the configured MAP penalty and solver settings.

## Data

- Species tree: `{payload["species_tree"]}`
- Gene tree directory: `{payload["gene_tree_dir"]}`
- Families: `{payload["families"]}`
- Mode: `specieswise`

## Solver And Objective

- Forward Pi iterations: `{payload["settings"]["pi_iters"]}`
- Backward self-loop solver: `{payload["settings"]["self_loop_solver"]}`
- Backward/self-loop iterations: `{payload["settings"]["neumann_terms"]}`
- Objective: `{payload["objective_description"]}`

## Optimizer

- Optimizer: `Adam`
- Steps completed: `{last["step"]}`
- Learning rate: `{payload["settings"]["lr"]}`
- Gradient clipping: `{payload["settings"]["clip_grad_norm"]}`
- Rate bounds: `[{payload["settings"]["min_rate"]}, {payload["settings"]["max_rate"]}]`

## Results

- Initial raw NLL: `{first["raw_nll_bits"]:.6f}` bits
- Final raw NLL: `{last["raw_nll_bits"]:.6f}` bits
- Final objective: `{last["objective_bits"]:.6f}` bits
- Final penalty: `{last["penalty_bits"]:.6f}` bits
- Final gradient norm: `{last["grad_norm"]:.6g}`
- Final rate range: `{last["rate_min"]:.6g}` to `{last["rate_max"]:.6g}`
- Total elapsed: `{payload["elapsed_s"]:.2f}` s

## Artifacts

- History JSONL: `{payload["history_jsonl"]}`
- Summary JSON: `{payload["summary_json"]}`
- Final theta: `{payload["theta_final"]}`
"""
    (run_dir / "README.md").write_text(report, encoding="utf-8")


def jsonable_settings(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def parse_penalty_lambdas(args: argparse.Namespace, *, device, dtype) -> torch.Tensor:
    values = _parse_event_values(
        args.penalty_lambdas,
        default=[float(args.penalty_lambda)] * 3,
        option_name="--penalty-lambdas",
    )
    return torch.tensor(values, device=device, dtype=dtype)


def parse_optional_event_lambdas(raw, *, device, dtype, option_name: str) -> torch.Tensor | None:
    if raw is None:
        return None
    values = _parse_event_values(raw, default=None, option_name=option_name)
    return torch.tensor(values, device=device, dtype=dtype)


def _parse_event_values(raw, *, default, option_name: str) -> list[float]:
    if raw is None:
        if default is None:
            raise ValueError(f"{option_name} is required")
        return list(default)
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError(f"{option_name} must contain exactly three comma-separated values for D,T,L")
    return values


def penalty_description(args: argparse.Namespace) -> str:
    parts = []
    if args.penalty_lambdas is None:
        parts.append(
            f"raw_nll_bits + 0.5 * {args.penalty_lambda} * "
            f"sum((theta - log2({args.l2_target_rate}))^2)"
        )
    else:
        parts.append(
            "raw_nll_bits + 0.5 * sum(lambda_event * "
            f"(theta - log2({args.l2_target_rate}))^2), lambda_event(D,T,L)={args.penalty_lambdas}"
        )
    if args.tree_penalty_lambdas is not None:
        parts.append(
            "0.5 * sum(lambda_tree_event * (theta_child - theta_parent)^2), "
            f"lambda_tree_event(D,T,L)={args.tree_penalty_lambdas}"
        )
    if args.root_penalty_lambdas is not None:
        parts.append(
            "0.5 * sum(lambda_root_event * (theta_root - "
            f"log2({args.l2_target_rate}))^2), lambda_root_event(D,T,L)={args.root_penalty_lambdas}"
        )
    return " + ".join(parts)


def tree_prior_penalty(
    theta: torch.Tensor,
    species_helpers: dict,
    *,
    tree_lambdas: torch.Tensor | None,
    root_lambdas: torch.Tensor | None,
    target: torch.Tensor,
) -> torch.Tensor:
    penalty = theta.new_zeros(())
    parent = species_helpers["sp_parent"].to(device=theta.device, dtype=torch.long)
    edge_mask = parent >= 0
    if tree_lambdas is not None and bool(edge_mask.any().detach().cpu()):
        child_theta = theta[edge_mask]
        parent_theta = theta[parent[edge_mask]]
        penalty = penalty + 0.5 * torch.sum(tree_lambdas * (child_theta - parent_theta) ** 2)
    if root_lambdas is not None:
        root_mask = parent < 0
        root_theta = theta[root_mask]
        penalty = penalty + 0.5 * torch.sum(root_lambdas * (root_theta - target[: root_theta.shape[-1]]) ** 2)
    return penalty


def penalty_parts(
    theta: torch.Tensor,
    species_helpers: dict,
    *,
    l2_lambdas: torch.Tensor,
    tree_lambdas: torch.Tensor | None,
    root_lambdas: torch.Tensor | None,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    l2 = 0.5 * torch.sum(l2_lambdas * (theta - target) ** 2)
    tree = tree_prior_penalty(
        theta,
        species_helpers,
        tree_lambdas=tree_lambdas,
        root_lambdas=None,
        target=target,
    )
    root = tree_prior_penalty(
        theta,
        species_helpers,
        tree_lambdas=None,
        root_lambdas=root_lambdas,
        target=target,
    )
    return l2 + tree + root, l2, tree, root



def summarize_existing_run(args: argparse.Namespace) -> None:
    run_dir = args.summarize_run
    if run_dir is None:
        raise ValueError("missing --summarize-run")
    history_jsonl = run_dir / "history.jsonl"
    theta_final = run_dir / "theta_final.pt"
    summary_json = run_dir / "summary.json"
    history = [
        json.loads(line)
        for line in history_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gene_trees = sorted(args.gene_tree_dir.glob("*.trees"))
    if args.max_families > 0:
        gene_trees = gene_trees[: args.max_families]
    payload = {
        "species_tree": str(args.species_tree),
        "gene_tree_dir": str(args.gene_tree_dir),
        "families": len(gene_trees),
        "settings": jsonable_settings(args),
        "history": history,
        "objective_description": penalty_description(args),
        "elapsed_s": sum(float(row.get("step_s", 0.0)) for row in history),
        "history_jsonl": str(history_jsonl),
        "summary_json": str(summary_json),
        "theta_final": str(theta_final),
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(run_dir, payload)
    print(f"summary_json={summary_json}")
    print(f"report={run_dir / 'README.md'}")


def main() -> None:
    args = parse_args()
    if args.summarize_run is not None:
        summarize_existing_run(args)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    gene_trees = sorted(args.gene_tree_dir.glob("*.trees"))
    if args.max_families > 0:
        gene_trees = gene_trees[: args.max_families]
    if not gene_trees:
        raise FileNotFoundError(f"no *.trees files found in {args.gene_tree_dir}")

    run_dir = args.output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    history_jsonl = run_dir / "history.jsonl"
    summary_json = run_dir / "summary.json"
    theta_final = run_dir / "theta_final.pt"

    solver_options = SolverOptions(
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
        gene_trees,
        mode="specieswise",
        device=args.device,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        batch_packing="depth_first_fit",
        max_wave_size=args.max_wave_size,
        solver_options=solver_options,
    )
    model.receiver_weights.requires_grad_(False)
    with torch.no_grad():
        if args.init_theta is not None:
            theta0 = torch.load(args.init_theta, map_location="cpu")
            if tuple(theta0.shape) != tuple(model.theta.shape):
                raise ValueError(
                    f"init theta shape {tuple(theta0.shape)} does not match model theta shape {tuple(model.theta.shape)}"
                )
            model.theta.copy_(theta0.to(device=model.theta.device, dtype=model.theta.dtype))
        else:
            model.theta.fill_(math.log2(args.theta_init_rate))
        clamp_log_rate_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
        model.clear_warm_starts()

    target = torch.full_like(model.theta, math.log2(args.l2_target_rate))
    penalty_lambdas = parse_penalty_lambdas(args, device=model.theta.device, dtype=model.theta.dtype)
    tree_penalty_lambdas = parse_optional_event_lambdas(
        args.tree_penalty_lambdas,
        device=model.theta.device,
        dtype=model.theta.dtype,
        option_name="--tree-penalty-lambdas",
    )
    root_penalty_lambdas = parse_optional_event_lambdas(
        args.root_penalty_lambdas,
        device=model.theta.device,
        dtype=model.theta.dtype,
        option_name="--root-penalty-lambdas",
    )
    optimizer = torch.optim.Adam([model.theta], lr=args.lr)
    history = []
    t_run = time.perf_counter()
    with history_jsonl.open("w", encoding="utf-8") as handle:
        for step in range(args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            sync(args.device)
            t_step = time.perf_counter()
            raw_nll = model()
            penalty, l2_penalty, tree_penalty, root_penalty = penalty_parts(
                model.theta,
                model.species_helpers,
                l2_lambdas=penalty_lambdas,
                tree_lambdas=tree_penalty_lambdas,
                root_lambdas=root_penalty_lambdas,
                target=target,
            )
            objective = raw_nll + penalty
            objective.backward()
            grad_norm = float(model.theta.grad.detach().norm().cpu())
            project_rate_gradient_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
            projected_grad_norm = float(model.theta.grad.detach().norm().cpu())
            torch.nn.utils.clip_grad_norm_([model.theta], args.clip_grad_norm)
            clipped_grad_norm = float(model.theta.grad.detach().norm().cpu())
            if step < args.steps:
                optimizer.step()
                clamp_log_rate_(model.theta, min_rate=args.min_rate, max_rate=args.max_rate)
            sync(args.device)
            row = {
                "step": step,
                "raw_nll_bits": float(raw_nll.detach().cpu()),
                "penalty_bits": float(penalty.detach().cpu()),
                "l2_penalty_bits": float(l2_penalty.detach().cpu()),
                "tree_penalty_bits": float(tree_penalty.detach().cpu()),
                "root_penalty_bits": float(root_penalty.detach().cpu()),
                "objective_bits": float(objective.detach().cpu()),
                "grad_norm": grad_norm,
                "projected_grad_norm": projected_grad_norm,
                "clipped_grad_norm": clipped_grad_norm,
                "step_s": time.perf_counter() - t_step,
                **theta_stats(model.theta, min_rate=args.min_rate, max_rate=args.max_rate),
            }
            history.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if step == 0 or step == args.steps or (args.print_every > 0 and step % args.print_every == 0):
                print(
                    f"step={step} raw_nll={row['raw_nll_bits']:.6f} "
                    f"objective={row['objective_bits']:.6f} grad={row['grad_norm']:.3g} "
                    f"rate=[{row['rate_min']:.3e},{row['rate_max']:.3e}] step_s={row['step_s']:.3f}",
                    flush=True,
                )
            if args.stop_rate_max > 0.0 and row["rate_max"] >= float(args.stop_rate_max):
                print(
                    f"early_stop=rate_max threshold={args.stop_rate_max:.6g} "
                    f"step={step} rate_max={row['rate_max']:.6g}",
                    flush=True,
                )
                break

    torch.save(model.theta.detach().cpu(), theta_final)
    payload = {
        "species_tree": str(args.species_tree),
        "gene_tree_dir": str(args.gene_tree_dir),
        "families": len(gene_trees),
        "settings": jsonable_settings(args),
        "history": history,
        "objective_description": penalty_description(args),
        "elapsed_s": time.perf_counter() - t_run,
        "history_jsonl": str(history_jsonl),
        "summary_json": str(summary_json),
        "theta_final": str(theta_final),
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(run_dir, payload)
    print(f"summary_json={summary_json}")
    print(f"report={run_dir / 'README.md'}")


if __name__ == "__main__":
    main()

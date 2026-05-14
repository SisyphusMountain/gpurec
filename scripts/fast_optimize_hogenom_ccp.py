from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpurec import GeneReconModel  # noqa: E402
from gpurec.core.preprocess_cpp import _load_extension  # noqa: E402


HOGENOM_DIR = REPO / "tests" / "data" / "HOGENOM" / "hogenom"
FAMILIES_FILE = HOGENOM_DIR / "hogenom_families.local.txt"
ALERAX_OUTPUT = HOGENOM_DIR / "output_alerax_corrected"
INFERRED_SPECIES_TREE = ALERAX_OUTPUT / "species_trees" / "inferred_species_tree.newick"
SPECIES_TREE = (
    INFERRED_SPECIES_TREE
    if INFERRED_SPECIES_TREE.exists()
    else HOGENOM_DIR / "hogenom_S.tree"
)
PREPROCESS_CACHE = HOGENOM_DIR / "output_gpurec_ccp_reconciliation" / "preprocess_cache"
OUT_DIR = HOGENOM_DIR / "output_gpurec_fast_ccp_opt"

LN2 = math.log(2.0)
RATE_QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_species_names(species_tree: Path) -> list[str]:
    ext = _load_extension()
    raw = ext.preprocess_multiple_families(
        str(species_tree),
        {},
        include_species_matrices=False,
    )
    return [str(x) for x in raw["species"]["names"]]


def uniform_origination(n_species: int, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((n_species,), 1.0 / n_species, device=device, dtype=dtype)


def theta_logits(theta: torch.Tensor) -> torch.Tensor:
    theta2 = theta.reshape(-1, 3)
    zeros = theta2.new_zeros((theta2.shape[0], 1))
    return torch.cat((zeros, theta2), dim=1) * LN2


def pS_values(theta: torch.Tensor) -> torch.Tensor:
    return torch.softmax(theta_logits(theta), dim=1)[:, 0]


def regularization(theta: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.regularization == "none" or args.regularization_weight == 0.0:
        return theta.new_zeros(())

    theta2 = theta.reshape(-1, 3)
    if args.regularization == "square-theta":
        center = theta2.new_tensor(args.regularization_center)
        penalty = (theta2 - center).square().sum()
    elif args.regularization == "gaussian-theta":
        center = theta2.new_tensor(args.regularization_center)
        std = theta2.new_tensor(args.regularization_std)
        penalty = 0.5 * ((theta2 - center) / std).square().sum()
    elif args.regularization == "beta-ps":
        logits = theta_logits(theta)
        log_probs = torch.log_softmax(logits, dim=1)
        log_pS = log_probs[:, 0] / LN2
        log_not_pS = torch.logsumexp(logits[:, 1:], dim=1) / LN2 - torch.logsumexp(
            logits,
            dim=1,
        ) / LN2
        penalty = -(
            (args.beta_ps_alpha - 1.0) * log_pS
            + (args.beta_ps_beta - 1.0) * log_not_pS
        ).sum()
    else:
        raise ValueError(f"unknown regularization {args.regularization!r}")

    return penalty * args.regularization_weight


def rate_summary(theta: torch.Tensor) -> dict[str, dict[str, float]]:
    theta2 = theta.detach().reshape(-1, 3)
    rates = torch.exp2(theta2).float().cpu()
    pS = pS_values(theta.detach()).float().cpu()
    out: dict[str, dict[str, float]] = {}
    for name, column in (("D", 0), ("T", 2), ("L", 1)):
        values = rates[:, column]
        qs = torch.quantile(values, torch.tensor(RATE_QUANTILES))
        out[name] = {
            "min": float(qs[0]),
            "p05": float(qs[1]),
            "median": float(qs[2]),
            "p95": float(qs[3]),
            "max": float(qs[4]),
            "mean": float(values.mean()),
        }
    qs = torch.quantile(pS, torch.tensor(RATE_QUANTILES))
    out["pS"] = {
        "min": float(qs[0]),
        "p05": float(qs[1]),
        "median": float(qs[2]),
        "p95": float(qs[3]),
        "max": float(qs[4]),
        "mean": float(pS.mean()),
    }
    return out


def format_rate_summary(summary: dict[str, dict[str, float]]) -> str:
    return " ".join(
        f"{name}[min={vals['min']:.3g} med={vals['median']:.3g} "
        f"p95={vals['p95']:.3g} max={vals['max']:.3g}]"
        for name, vals in summary.items()
    )


def evaluate(
    model: GeneReconModel,
    args: argparse.Namespace,
    *,
    zero_grad: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if zero_grad:
        model.theta.grad = None
    synchronize()
    start = time.perf_counter()
    data_loss = model.full_loss()
    penalty = regularization(model.theta, args)
    objective = data_loss + penalty
    objective.backward()
    synchronize()
    elapsed = time.perf_counter() - start
    if model.theta.grad is None:
        raise RuntimeError("missing theta gradient")
    grad = model.theta.grad.detach()
    metrics = {
        "data_nll_bits": float(data_loss.detach().cpu()),
        "regularization_bits": float(penalty.detach().cpu()),
        "objective_bits": float(objective.detach().cpu()),
        "log_likelihood_bits": float(-data_loss.detach().cpu()),
        "grad_inf": float(grad.abs().amax().cpu()),
        "grad_norm": float(torch.linalg.vector_norm(grad).cpu()),
        "eval_s": elapsed,
    }
    return objective, metrics


def clamp_rates(model: GeneReconModel, args: argparse.Namespace) -> None:
    model.clamp_theta_(min_rate=args.min_rate, max_rate=args.max_rate)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def log_row(row: dict[str, Any], summary: dict[str, dict[str, float]]) -> None:
    delta = row.get("delta_objective_bits")
    delta_text = "nan" if delta is None else f"{float(delta):.6g}"
    print(
        f"phase={row['phase']} iter={row['iteration']:04d} "
        f"objective_bits={row['objective_bits']:.6f} "
        f"data_nll_bits={row['data_nll_bits']:.6f} "
        f"delta_objective_bits={delta_text} "
        f"grad_inf={row['grad_inf']:.6g} grad_norm={row['grad_norm']:.6g} "
        f"theta_step_inf={row['theta_step_inf']:.3g} "
        f"eval_s={row['eval_s']:.3f} step_s={row['step_s']:.3f} "
        f"closure_evals={row.get('closure_evals', 1)}",
        flush=True,
    )
    print("  " + format_rate_summary(summary), flush=True)


def converged(
    row: dict[str, Any],
    *,
    stable_loss_steps: int,
    args: argparse.Namespace,
) -> bool:
    if row["grad_inf"] <= args.grad_inf_tol:
        return True
    if row["theta_step_inf"] <= args.theta_step_tol and row["iteration"] >= args.min_steps:
        return True
    if (
        stable_loss_steps >= args.loss_patience
        and row["iteration"] >= args.min_steps
    ):
        return True
    return False


def first_order_step(
    model: GeneReconModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> tuple[dict[str, float], float]:
    theta_before = model.theta.detach().clone()
    _objective, metrics = evaluate(model, args)
    optimizer.step()
    synchronize()
    clamp_rates(model, args)
    theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
    model.clear()
    return metrics, theta_step


def lbfgs_step(
    model: GeneReconModel,
    optimizer: torch.optim.LBFGS,
    args: argparse.Namespace,
) -> tuple[dict[str, float], float, int]:
    theta_before = model.theta.detach().clone()
    closure_evals = 0
    last_metrics: dict[str, float] | None = None

    def closure() -> torch.Tensor:
        nonlocal closure_evals, last_metrics
        closure_evals += 1
        optimizer.zero_grad(set_to_none=True)
        objective, metrics = evaluate(model, args, zero_grad=False)
        last_metrics = metrics
        return objective

    optimizer.step(closure)
    synchronize()
    clamp_rates(model, args)
    theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
    model.clear()
    if last_metrics is None:
        raise RuntimeError("LBFGS closure was not called")
    return last_metrics, theta_step, closure_evals


def build_model(args: argparse.Namespace, origination_probs: torch.Tensor) -> GeneReconModel:
    return GeneReconModel.from_alerax_families(
        str(args.species_tree),
        args.families_file,
        mode=args.mode,
        start=0,
        max_families=args.max_families,
        device=args.device,
        dtype=torch.float32,
        theta_init_rates=(args.init_d, args.init_l, args.init_t),
        preprocess_cache_dir=args.preprocess_cache,
        fixed_iters_E=args.fixed_iters_e,
        fixed_iters_Pi=args.fixed_iters_pi,
        neumann_terms=args.neumann_terms,
        use_pruning=True,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        batch_packing=args.batch_packing,
        max_wave_size=args.max_wave_size,
        lazy_preprocess=True,
        prefetch_batches="all",
        origination_probs=origination_probs,
    )


def prepare_all_batches(model: GeneReconModel) -> None:
    for idx in range(len(model.batch_metadata)):
        model._ensure_batch_static(idx)
    model._current_batch_index = 0
    model.clear()


def write_outputs(
    model: GeneReconModel,
    species_names: list[str],
    out_dir: Path,
    args: argparse.Namespace,
    final_metrics: dict[str, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.theta.detach().cpu(), out_dir / "theta_final.pt")
    with (out_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2, sort_keys=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    theta = model.theta.detach().reshape(-1, 3).cpu()
    rates = torch.exp2(theta)
    ps = pS_values(theta).cpu()
    names = species_names if args.mode == "specieswise" else ["global"]
    with (out_dir / "rates_final.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row", "name", "D", "T", "L", "pS", "theta_D", "theta_T", "theta_L"])
        for i, name in enumerate(names):
            writer.writerow(
                [
                    i,
                    name,
                    float(rates[i, 0]),
                    float(rates[i, 2]),
                    float(rates[i, 1]),
                    float(ps[i]),
                    float(theta[i, 0]),
                    float(theta[i, 2]),
                    float(theta[i, 1]),
                ]
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast exact optimization of the HOGENOM CCP likelihood."
    )
    parser.add_argument("--optimizer", choices=("adagrad", "adam", "lbfgs", "adagrad-lbfgs"), default="adagrad")
    parser.add_argument("--steps", type=int, default=200, help="Maximum optimization steps after any warmup phase.")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Adagrad warmup steps for --optimizer adagrad-lbfgs.")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate for Adagrad/Adam or LBFGS if no separate LBFGS lr is given.")
    parser.add_argument("--lbfgs-lr", type=float, default=0.1)
    parser.add_argument("--lbfgs-history-size", type=int, default=10)
    parser.add_argument("--lbfgs-line-search", choices=("none", "strong_wolfe"), default="strong_wolfe")
    parser.add_argument("--grad-inf-tol", type=float, default=1e-3)
    parser.add_argument("--loss-change-tol", type=float, default=1e-5)
    parser.add_argument("--theta-step-tol", type=float, default=1e-6)
    parser.add_argument("--loss-patience", type=int, default=5)
    parser.add_argument("--min-steps", type=int, default=5)

    parser.add_argument("--mode", choices=("global", "specieswise"), default="specieswise")
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--species-tree", type=Path, default=SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=FAMILIES_FILE)
    parser.add_argument("--preprocess-cache", type=Path, default=PREPROCESS_CACHE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)

    parser.add_argument("--family-chunk-size", type=int, default=0)
    parser.add_argument("--clade-budget", type=int, default=305000)
    parser.add_argument("--batch-packing", choices=("sequential", "clade_first_fit", "depth_first_fit"), default="depth_first_fit")
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--fixed-iters-e", type=int, default=6)
    parser.add_argument("--fixed-iters-pi", type=int, default=6)
    parser.add_argument("--neumann-terms", type=int, default=6)

    parser.add_argument("--init-d", type=float, default=0.05)
    parser.add_argument("--init-l", type=float, default=0.05)
    parser.add_argument("--init-t", type=float, default=0.05)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=100.0)

    parser.add_argument("--regularization", choices=("none", "square-theta", "gaussian-theta", "beta-ps"), default="none")
    parser.add_argument("--regularization-weight", type=float, default=1.0)
    parser.add_argument("--regularization-center", type=float, default=0.0)
    parser.add_argument("--regularization-std", type=float, default=0.5)
    parser.add_argument("--beta-ps-alpha", type=float, default=4.0)
    parser.add_argument("--beta-ps-beta", type=float, default=1.0)
    parser.add_argument("--prepare-all-batches", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def run_phase(
    *,
    model: GeneReconModel,
    args: argparse.Namespace,
    phase: str,
    optimizer: torch.optim.Optimizer,
    start_iteration: int,
    max_steps: int,
    history_path: Path,
) -> tuple[int, dict[str, float]]:
    previous_objective: float | None = None
    stable_loss_steps = 0
    final_metrics: dict[str, float] = {}

    for local_step in range(max_steps):
        step_start = time.perf_counter()
        if isinstance(optimizer, torch.optim.LBFGS):
            metrics, theta_step, closure_evals = lbfgs_step(model, optimizer, args)
        else:
            metrics, theta_step = first_order_step(model, optimizer, args)
            closure_evals = 1
        step_s = time.perf_counter() - step_start

        objective = metrics["objective_bits"]
        delta = None if previous_objective is None else previous_objective - objective
        if delta is not None and abs(delta) <= args.loss_change_tol:
            stable_loss_steps += 1
        else:
            stable_loss_steps = 0
        previous_objective = objective

        row: dict[str, Any] = {
            "phase": phase,
            "iteration": start_iteration + local_step,
            "phase_iteration": local_step,
            "theta_step_inf": theta_step,
            "step_s": step_s,
            "closure_evals": closure_evals,
            "delta_objective_bits": delta,
            **metrics,
        }
        summary = rate_summary(model.theta)
        row["rates"] = summary
        append_jsonl(history_path, row)
        log_row(row, summary)
        final_metrics = metrics

        if converged(row, stable_loss_steps=stable_loss_steps, args=args):
            row = {
                "event": "converged",
                "phase": phase,
                "iteration": start_iteration + local_step,
                "stable_loss_steps": stable_loss_steps,
            }
            append_jsonl(history_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
            return start_iteration + local_step + 1, final_metrics

    return start_iteration + max_steps, final_metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.fixed_iters_pi % 2 != 0:
        raise ValueError("--fixed-iters-pi must be even")
    if args.max_rate <= args.min_rate:
        raise ValueError("--max-rate must be greater than --min-rate")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    print("fast_hogenom_ccp_optimizer", flush=True)
    print(f"species_tree={args.species_tree}", flush=True)
    print(f"families_file={args.families_file}", flush=True)
    print(
        "layout="
        f"chunk_size={args.family_chunk_size} clade_budget={args.clade_budget} "
        f"packing={args.batch_packing} max_wave_size={args.max_wave_size}",
        flush=True,
    )
    print(
        f"optimizer={args.optimizer} lr={args.lr} lbfgs_lr={args.lbfgs_lr} "
        f"max_rate={args.max_rate} origination=uniform",
        flush=True,
    )

    species_names = load_species_names(args.species_tree)
    origination_probs = uniform_origination(
        len(species_names),
        device=args.device,
        dtype=torch.float32,
    )

    build_start = time.perf_counter()
    model = build_model(args, origination_probs)
    if args.prepare_all_batches:
        prepare_all_batches(model)
    synchronize()
    print(
        f"build_s={time.perf_counter() - build_start:.3f} "
        f"families={sum(m.family_count for m in model.batch_metadata)} "
        f"species={model.n_species} batches={len(model.batch_metadata)} "
        f"waves={sum(m.wave_count for m in model.batch_metadata)}",
        flush=True,
    )

    try:
        print("warmup_exact_full_eval", flush=True)
        _, initial = evaluate(model, args)
        model.theta.grad = None
        model.clear()
        print(
            f"initial objective_bits={initial['objective_bits']:.6f} "
            f"data_nll_bits={initial['data_nll_bits']:.6f} "
            f"grad_inf={initial['grad_inf']:.6g} eval_s={initial['eval_s']:.3f}",
            flush=True,
        )

        iteration = 0
        final_metrics = initial
        if args.optimizer == "adagrad-lbfgs":
            warm_optimizer = torch.optim.Adagrad(
                [model.theta],
                lr=args.lr,
                eps=1e-10,
            )
            iteration, final_metrics = run_phase(
                model=model,
                args=args,
                phase="adagrad",
                optimizer=warm_optimizer,
                start_iteration=iteration,
                max_steps=args.warmup_steps,
                history_path=history_path,
            )
            lbfgs_optimizer = torch.optim.LBFGS(
                [model.theta],
                lr=args.lbfgs_lr,
                max_iter=1,
                max_eval=20,
                history_size=args.lbfgs_history_size,
                line_search_fn=None if args.lbfgs_line_search == "none" else args.lbfgs_line_search,
            )
            iteration, final_metrics = run_phase(
                model=model,
                args=args,
                phase="lbfgs",
                optimizer=lbfgs_optimizer,
                start_iteration=iteration,
                max_steps=args.steps,
                history_path=history_path,
            )
        else:
            if args.optimizer == "adagrad":
                optimizer: torch.optim.Optimizer = torch.optim.Adagrad(
                    [model.theta],
                    lr=args.lr,
                    eps=1e-10,
                )
            elif args.optimizer == "adam":
                optimizer = torch.optim.Adam([model.theta], lr=args.lr)
            else:
                optimizer = torch.optim.LBFGS(
                    [model.theta],
                    lr=args.lbfgs_lr,
                    max_iter=1,
                    max_eval=20,
                    history_size=args.lbfgs_history_size,
                    line_search_fn=None if args.lbfgs_line_search == "none" else args.lbfgs_line_search,
                )
            iteration, final_metrics = run_phase(
                model=model,
                args=args,
                phase=args.optimizer,
                optimizer=optimizer,
                start_iteration=iteration,
                max_steps=args.steps,
                history_path=history_path,
            )

        print("final_exact_full_eval", flush=True)
        _, final_metrics = evaluate(model, args)
        model.theta.grad = None
        model.clear()
        final_metrics["total_iterations"] = float(iteration)
        print(
            f"final objective_bits={final_metrics['objective_bits']:.6f} "
            f"data_nll_bits={final_metrics['data_nll_bits']:.6f} "
            f"grad_inf={final_metrics['grad_inf']:.6g} eval_s={final_metrics['eval_s']:.3f}",
            flush=True,
        )
        write_outputs(model, species_names, args.out_dir, args, final_metrics)
        print(f"wrote_outputs={args.out_dir}", flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()

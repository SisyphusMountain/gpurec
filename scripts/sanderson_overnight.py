#!/usr/bin/env python3
"""Overnight Sanderson penalized-likelihood sweep for AleRax Archaea DTL rates.

Fits specieswise DTL rates by penalized likelihood (Sanderson 2002):
``NLL(theta) + lambda * Phi(theta)`` where ``Phi`` is the roughness penalty over
the species tree. Sweeps ``lambda`` on a log grid, runs k-fold cross-validation
over gene families to score each ``lambda``, and writes figures + statistics:

  * fig1_cross_validation.png  -- CV score vs log lambda (Sanderson fig. 1)
  * fig2_coefficient_of_variation.png -- CoV of rates across nodes vs log lambda (fig. 2)
  * fig3_branch_rates.png      -- selected species-branch rates vs log lambda (fig. 3)
  * fig_loss_convergence.png   -- loss and |g_theta|inf vs step, per lambda
  * fig_rate_histograms.png    -- rate distributions (D/T/L) at the CV-best lambda
  * results.json               -- machine-readable everything (incrementally saved)
  * summary.md                 -- human-readable tables

Run it (use the repo on PYTHONPATH so the repo's native parser is used):

    PYTHONPATH=$(pwd) nohup python scripts/sanderson_overnight.py \
        --outdir output/sanderson_overnight > overnight.log 2>&1 &

Use --smoke for a fast end-to-end check on a tiny subset.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Use the repo copy of gpurec (its native .ale parser matches this dataset),
# mirroring what the notebook does. Running `python scripts/foo.py` would
# otherwise put scripts/ on sys.path[0] and pick up a site-packages gpurec.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import importlib.util

# Reuse the optimization helpers (select_families, parse_schedule, optimizer
# helpers, roughness_penalty, cross_validate_lambda, theta_stats, ...).
_AO_PATH = REPO_ROOT / "scripts/optimize_alerax_archaea_genewise_adam.py"
_spec = importlib.util.spec_from_file_location("archaea_opt", _AO_PATH)
ao = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ao)

from gpurec import GeneReconModel, SolverOptions


DEFAULT_DATA_ROOT = REPO_ROOT / "tests/data/alerax_archaea_davin2017"
DEFAULT_LAMBDA_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "output/sanderson_overnight")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260609)

    # Family selection for the per-lambda full fits (figs 2/3 + stats).
    p.add_argument("--max-families", type=int, default=0, help="0 = all eligible families.")
    p.add_argument("--min-leaves", type=int, default=4)
    p.add_argument("--family-order", default="smallest", choices=("smallest", "largest", "lexicographic"))

    # Lambda grid (shared by CV and the per-lambda fits).
    p.add_argument("--lambda-grid", default=",".join(str(x) for x in DEFAULT_LAMBDA_GRID),
                   help="Comma-separated lambda values.")

    # Per-lambda fit schedule and optimizer.
    p.add_argument("--schedule", default="80:16:16:float32:neumann:0.1,160:16:16:float32:neumann:0.03",
                   help="STEPS:PI:NEUMANN:DTYPE:SOLVER:LR phases (comma-separated).")
    p.add_argument("--theta-init-rate", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=200.0)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)

    # Cross-validation.
    p.add_argument("--cv", dest="cv", action="store_true", default=True)
    p.add_argument("--no-cv", dest="cv", action="store_false")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-max-families", type=int, default=0, help="Families used for CV (0 = all = whole dataset).")
    p.add_argument("--cv-steps", type=int, default=80, help="Adam steps per (fold, lambda) CV fit.")
    p.add_argument("--cv-dtype", default="float32", choices=("float32", "float64"))

    # Packing / solver.
    p.add_argument("--family-chunk-size", type=int, default=0)
    p.add_argument("--clade-budget", type=int, default=300_000, help="0 disables.")
    p.add_argument("--batch-packing", default="depth_first_fit")
    p.add_argument("--max-wave-size", type=int, default=8192)
    p.add_argument("--e-max-iter", type=int, default=2000)
    p.add_argument("--e-tol", type=float, default=1e-8)

    p.add_argument("--branch-rate-topk", type=int, default=6,
                   help="Number of high-variance species branches to draw in fig. 3.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny end-to-end run for validation (overrides sizes).")
    args = p.parse_args()

    if args.smoke:
        args.max_families = 24
        args.lambda_grid = "0.3,3.0,30.0"
        args.schedule = "8:16:16:float32:neumann:0.1"
        args.cv_folds = 2
        args.cv_max_families = 12
        args.cv_steps = 8
        args.branch_rate_topk = 4
    return args


# --------------------------------------------------------------------------- #
# Logging / IO
# --------------------------------------------------------------------------- #
class Tee:
    def __init__(self, path: Path):
        self.handle = open(path, "a", encoding="utf-8")

    def __call__(self, *parts: Any, **_kwargs: Any) -> None:
        msg = " ".join(str(p) for p in parts)
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()


def optimizer_namespace(args: argparse.Namespace) -> SimpleNamespace:
    """Minimal namespace consumed by ao.make_optimizer / scheduled_step_lr."""
    return SimpleNamespace(
        optimizer="adam", optimizer_lr=None, adam_lr=None,
        adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
        rmsprop_alpha=0.99, adadelta_rho=0.9, optimizer_momentum=0.0, optimizer_eps=None,
        lr_ramp_steps=0, lr_ramp_start_factor=0.25, lr_ramp_first_phase=False,
        lr_decay_steps=0, lr_decay_end_factor=1.0,
    )


# --------------------------------------------------------------------------- #
# Model construction / fitting
# --------------------------------------------------------------------------- #
def build_model(args: argparse.Namespace, families: list[Path], schedule: list[dict], dtype_name: str) -> GeneReconModel:
    solver = SolverOptions(
        e_max_iter=args.e_max_iter, e_tol=args.e_tol,
        pi_iters=int(schedule[0]["pi_iters"]), neumann_terms=int(schedule[0]["neumann_terms"]),
        self_loop_solver=str(schedule[0]["self_loop_solver"]),
    )
    model = GeneReconModel(
        args.species_tree, families, mode="specieswise", device=args.device,
        family_chunk_size=args.family_chunk_size,
        clade_budget=None if args.clade_budget == 0 else args.clade_budget,
        batch_packing=args.batch_packing, max_wave_size=args.max_wave_size,
        solver_options=solver,
    )
    model.to(dtype=ao.torch_dtype(dtype_name))
    model.receiver_weights.requires_grad_(False)
    return model


def fit_lambda(model: GeneReconModel, lam: float, schedule: list[dict], args: argparse.Namespace,
               opt_args: SimpleNamespace, *, init_theta: torch.Tensor | None, device: torch.device,
               log) -> dict[str, Any]:
    """Fit theta at a fixed lambda over the schedule; record convergence history."""
    with torch.no_grad():
        if init_theta is not None:
            model.theta.copy_(init_theta.to(model.theta))
        else:
            model.theta.fill_(math.log2(args.theta_init_rate))
        model.clear_warm_starts()

    history: list[dict[str, Any]] = []
    current_dtype = str(schedule[0]["dtype"])
    global_step = 0
    for phase_index, phase in enumerate(schedule):
        if str(phase["dtype"]) != current_dtype:
            model.to(dtype=ao.torch_dtype(str(phase["dtype"])))
            current_dtype = str(phase["dtype"])
        model.configure_solver(
            pi_iters=int(phase["pi_iters"]), neumann_terms=int(phase["neumann_terms"]),
            self_loop_solver=str(phase["self_loop_solver"]),
        )
        target_lr = float(phase["optimizer_lr"] or ao.optimizer_lr(opt_args))
        optimizer = ao.make_optimizer([model.theta], opt_args, target_lr)
        model.clear_warm_starts()
        for phase_step in range(int(phase["steps"])):
            step_lr = ao.scheduled_step_lr(target_lr, phase_step=phase_step, phase_index=phase_index, args=opt_args)
            ao.set_optimizer_lr(optimizer, step_lr)
            ao.sync(device)
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            data_loss = model()
            penalty = lam * ao.roughness_penalty(model.theta, model.species_helpers)
            loss = data_loss + penalty
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"non-finite loss at lambda={lam} step={global_step}")
            loss.backward()
            grad = model.theta.grad.detach()
            theta_grad_max = float(grad.abs().max().cpu())
            theta_grad_rms = float(grad.norm().cpu()) / math.sqrt(grad.numel())
            if args.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_([model.theta], args.clip_grad_norm)
            optimizer.step()
            ao.sync(device)
            prev = history[-1]["loss_bits"] if history else float("nan")
            row = {
                "step": global_step, "phase": phase_index, "dtype": current_dtype,
                "lr": step_lr,
                "loss_bits": float(loss.detach().cpu()),
                "data_loss_bits": float(data_loss.detach().cpu()),
                "penalty_bits": float(penalty.detach().cpu()),
                "loss_delta": float(loss.detach().cpu()) - prev,
                "theta_grad_max": theta_grad_max,
                "theta_grad_rms": theta_grad_rms,
                "step_s": time.perf_counter() - t0,
                **ao.theta_stats(model.theta),
            }
            history.append(row)
            global_step += 1

    with torch.no_grad():
        theta = model.theta.detach().float().cpu()
        rates = torch.exp2(theta)
    tail = [r["loss_bits"] for r in history][-25:]
    return {
        "lambda": lam,
        "history": history,
        "theta": theta.tolist(),
        "rates": rates.tolist(),
        "final_loss_bits": history[-1]["loss_bits"],
        "final_data_loss_bits": history[-1]["data_loss_bits"],
        "final_penalty_bits": history[-1]["penalty_bits"],
        "final_theta_grad_max": history[-1]["theta_grad_max"],
        "tail_slope_bits_per_step": ao.fit_line_slope(tail),
        "rate_stats": rate_stats(rates.numpy()),
    }


def rate_stats(rates: np.ndarray) -> dict[str, Any]:
    """Per-rate-type (D/T/L) and overall statistics across species nodes."""
    names = ["D", "T", "L"]
    out: dict[str, Any] = {}
    for d, name in enumerate(names):
        col = rates[:, d]
        mean = float(col.mean())
        out[name] = {
            "mean": mean, "std": float(col.std()), "min": float(col.min()),
            "max": float(col.max()), "median": float(np.median(col)),
            "cov": float(col.std() / mean) if mean > 0 else float("nan"),
            "fold_variation": float(col.max() / col.min()) if col.min() > 0 else float("inf"),
        }
    return out


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def run_cross_validation(args: argparse.Namespace, families: list[Path], grid: list[float],
                         schedule: list[dict], opt_args: SimpleNamespace, device: torch.device, log) -> tuple[float, dict[float, float]]:
    cv_families = families if args.cv_max_families in (0, None) else families[: args.cv_max_families]
    log(f"cross-validation: {len(cv_families)} families, {args.cv_folds} folds, grid={grid}")

    def _build(fams: list[Path]) -> GeneReconModel:
        return build_model(args, fams, schedule, args.cv_dtype)

    def _fit(sub: GeneReconModel, lam: float, warm: torch.Tensor | None) -> torch.Tensor:
        with torch.no_grad():
            if warm is not None:
                sub.theta.copy_(warm.to(sub.theta))
            else:
                sub.theta.fill_(math.log2(args.theta_init_rate))
            sub.clear_warm_starts()
        opt = ao.make_optimizer([sub.theta], opt_args, ao.optimizer_lr(opt_args))
        for _ in range(int(args.cv_steps)):
            opt.zero_grad(set_to_none=True)
            loss = sub() + lam * ao.roughness_penalty(sub.theta, sub.species_helpers)
            loss.backward()
            if args.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_([sub.theta], args.clip_grad_norm)
            opt.step()
        return sub.theta.detach().clone()

    def _eval(sub: GeneReconModel, theta: torch.Tensor) -> float:
        with torch.no_grad():
            return float(sub(theta=theta.to(sub.theta)).detach().cpu())

    best, scores = ao.cross_validate_lambda(
        list(cv_families), grid, folds=args.cv_folds,
        build_model=_build, fit_theta=_fit, eval_nll=_eval, seed=args.seed, log=log,
    )
    log(f"cross-validation chose lambda = {best:g}")
    return best, scores


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_cv(scores: dict[float, float], best: float | None, path: Path) -> None:
    lams = sorted(scores)
    ys = [scores[l] for l in lams]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lams, ys, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("smoothing parameter lambda (log)")
    ax.set_ylabel("cross-validation score (held-out NLL, bits)")
    ax.set_title("Cross-validation (Sanderson fig. 1)")
    if best is not None:
        ax.axvline(best, color="k", ls="--", lw=0.8, label=f"optimal lambda = {best:g}")
        ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cov(per_lambda: list[dict], best: float | None, path: Path) -> None:
    lams = [r["lambda"] for r in per_lambda]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ["D", "T", "L"]:
        cov = [r["rate_stats"][name]["cov"] for r in per_lambda]
        ax.plot(lams, cov, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("smoothing parameter lambda (log)")
    ax.set_ylabel("coefficient of variation of rates across nodes")
    ax.set_title("Rate variation vs smoothing (Sanderson fig. 2)")
    if best is not None:
        ax.axvline(best, color="k", ls="--", lw=0.8, label=f"optimal lambda = {best:g}")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_branch_rates(per_lambda: list[dict], best: float | None, topk: int, path: Path) -> None:
    lams = np.array([r["lambda"] for r in per_lambda], dtype=float)
    rates = np.array([r["rates"] for r in per_lambda], dtype=float)  # [L, S, 3]
    names = ["D", "T", "L"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for d, (name, ax) in enumerate(zip(names, axes)):
        col = rates[:, :, d]                       # [L, S]
        var_over_lambda = col.var(axis=0)          # [S]
        nodes = np.argsort(var_over_lambda)[::-1][:topk]
        for node in nodes:
            ax.plot(lams, col[:, node], marker=".", lw=1, label=f"node {int(node)}")
        ax.set_xscale("log")
        ax.set_xlabel("lambda (log)")
        ax.set_ylabel(f"{name} rate (exp2 theta)")
        ax.set_title(f"{name}: {topk} most variable branches")
        if best is not None:
            ax.axvline(best, color="k", ls="--", lw=0.8)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Per-branch rate vs smoothing (Sanderson fig. 3)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_convergence(per_lambda: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in per_lambda:
        steps = [h["step"] for h in r["history"]]
        axes[0].plot(steps, [h["loss_bits"] for h in r["history"]], lw=1, label=f"lambda={r['lambda']:g}")
        axes[1].plot(steps, [h["theta_grad_max"] for h in r["history"]], lw=1, label=f"lambda={r['lambda']:g}")
    axes[0].set_title("Penalized objective vs step")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (bits)")
    axes[1].set_title("theta gradient sup-norm vs step")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("|g_theta|inf")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_rate_histograms(best_result: dict, path: Path) -> None:
    rates = np.array(best_result["rates"], dtype=float)
    names = ["D", "T", "L"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for d, (name, ax) in enumerate(zip(names, axes)):
        col = rates[:, d]
        ax.hist(col, bins=40)
        ax.set_title(f"{name} rate (lambda={best_result['lambda']:g})\n"
                     f"mean={col.mean():.4g} cv={col.std()/col.mean():.3g}")
        ax.set_xlabel("rate (exp2 theta)")
        ax.set_ylabel("species nodes")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_summary_md(args, grid, best, cv_scores, per_lambda, path: Path) -> None:
    lines = ["# Sanderson penalized-likelihood sweep", ""]
    lines.append(f"- families (per-lambda fits): {args.n_families}")
    lines.append(f"- species nodes (theta rows): {args.n_species}")
    lines.append(f"- lambda grid: {grid}")
    lines.append(f"- cross-validation best lambda: **{best}**" if best is not None else "- cross-validation: skipped")
    lines.append("")
    if cv_scores:
        lines += ["## Cross-validation score (held-out NLL, bits)", "", "| lambda | CV score |", "|---:|---:|"]
        for lam in sorted(cv_scores):
            lines.append(f"| {lam:g} | {cv_scores[lam]:.4f} |")
        lines.append("")
    lines += ["## Per-lambda fit", "",
              "| lambda | final loss | data NLL | penalty | tail slope | final \\|g_theta\\|inf | CoV(D) | CoV(T) | CoV(L) | rate min | rate max |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_lambda:
        rs = r["rate_stats"]
        rmin = min(rs[n]["min"] for n in "DTL")
        rmax = max(rs[n]["max"] for n in "DTL")
        lines.append(
            f"| {r['lambda']:g} | {r['final_loss_bits']:.3f} | {r['final_data_loss_bits']:.3f} | "
            f"{r['final_penalty_bits']:.3f} | {r['tail_slope_bits_per_step']:.4g} | {r['final_theta_grad_max']:.3g} | "
            f"{rs['D']['cov']:.3g} | {rs['T']['cov']:.3g} | {rs['L']['cov']:.3g} | {rmin:.4g} | {rmax:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    args.species_tree = args.data_root / "species_reference/reference_species_tree.newick"
    family_dir = args.data_root / "ale_gene_tree_distributions/main_families_ge4seq"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    run_dir = args.outdir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Tee(run_dir / "run.log")
    log(f"output dir: {run_dir}")

    grid = sorted(float(x) for x in str(args.lambda_grid).split(",") if x.strip())
    schedule = ao.parse_schedule(args.schedule)
    opt_args = optimizer_namespace(args)

    families, selection = ao.select_families(
        family_dir, max_families=args.max_families, family_order=args.family_order,
        min_leaves=args.min_leaves, recursive=False,
    )
    log(f"selected {len(families)} families: {selection}")
    args.n_families = len(families)

    results: dict[str, Any] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "selection": selection, "lambda_grid": grid, "schedule": schedule,
        "cv_scores": None, "best_lambda": None, "per_lambda": [], "errors": [],
    }

    def save_results() -> None:
        (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    t_start = time.perf_counter()

    # ----- Cross-validation (Sanderson fig. 1) -----
    best_lambda = None
    cv_scores: dict[float, float] = {}
    if args.cv:
        try:
            best_lambda, cv_scores = run_cross_validation(args, families, grid, schedule, opt_args, device, log)
            results["best_lambda"] = best_lambda
            results["cv_scores"] = {str(k): v for k, v in cv_scores.items()}
            save_results()
            plot_cv(cv_scores, best_lambda, run_dir / "fig1_cross_validation.png")
        except Exception as exc:  # keep going so the sweep still runs overnight
            log(f"ERROR in cross-validation: {exc}")
            results["errors"].append({"phase": "cv", "error": str(exc), "trace": traceback.format_exc()})
            save_results()

    # ----- Per-lambda full fits (figs 2/3, convergence, stats) -----
    warm: torch.Tensor | None = None
    for lam in grid:
        log(f"=== fitting lambda = {lam:g} ===")
        try:
            model = build_model(args, families, schedule, str(schedule[0]["dtype"]))
            args.n_species = int(model.theta.shape[0])
            t0 = time.perf_counter()
            result = fit_lambda(model, lam, schedule, args, opt_args, init_theta=warm, device=device, log=log)
            warm = torch.tensor(result["theta"])  # warm-start next lambda (ascending)
            result["elapsed_s"] = time.perf_counter() - t0
            results["per_lambda"].append(result)
            rs = result["rate_stats"]
            log(f"lambda={lam:g} done in {result['elapsed_s']:.1f}s "
                f"final_loss={result['final_loss_bits']:.3f} data={result['final_data_loss_bits']:.3f} "
                f"penalty={result['final_penalty_bits']:.3f} tail_slope={result['tail_slope_bits_per_step']:.4g} "
                f"CoV(D/T/L)=({rs['D']['cov']:.3g},{rs['T']['cov']:.3g},{rs['L']['cov']:.3g})")
            save_results()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            log(f"ERROR fitting lambda={lam:g}: {exc}")
            results["errors"].append({"phase": f"fit_lambda={lam}", "error": str(exc), "trace": traceback.format_exc()})
            save_results()

    # ----- Figures & summary -----
    per_lambda = results["per_lambda"]
    if per_lambda:
        args.n_species = int(np.array(per_lambda[0]["rates"]).shape[0])
        try:
            plot_cov(per_lambda, best_lambda, run_dir / "fig2_coefficient_of_variation.png")
            plot_branch_rates(per_lambda, best_lambda, args.branch_rate_topk, run_dir / "fig3_branch_rates.png")
            plot_convergence(per_lambda, run_dir / "fig_loss_convergence.png")
            best_result = None
            if best_lambda is not None:
                best_result = min(per_lambda, key=lambda r: abs(r["lambda"] - best_lambda))
            best_result = best_result or min(per_lambda, key=lambda r: r["final_loss_bits"])
            plot_rate_histograms(best_result, run_dir / "fig_rate_histograms.png")
            write_summary_md(args, grid, best_lambda, cv_scores, per_lambda, run_dir / "summary.md")
        except Exception as exc:
            log(f"ERROR while plotting: {exc}")
            results["errors"].append({"phase": "plot", "error": str(exc), "trace": traceback.format_exc()})

    results["elapsed_s"] = time.perf_counter() - t_start
    save_results()
    log(f"DONE in {results['elapsed_s']:.1f}s. Figures + summary.md + results.json in {run_dir}")


if __name__ == "__main__":
    main()

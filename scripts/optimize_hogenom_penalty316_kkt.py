#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hogenom_ccp_wandb_opt as opt


DEFAULT_PENALTY = 316.22776601683796


def penalty_slug(value: float) -> str:
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def build_model(config: opt.RunConfig, dtype: torch.dtype) -> opt.GeneReconModel:
    device = torch.device(config.device)
    species_count = opt.count_species_nodes(config.species_tree)
    origination_probs = torch.full(
        (species_count,),
        1.0 / species_count,
        device=device,
        dtype=dtype,
    )
    return opt.GeneReconModel.from_alerax_families(
        str(config.species_tree),
        config.families_file,
        mode=opt.internal_parameter_mode(config),
        start=0,
        max_families=config.max_families,
        device=device,
        dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        preprocess_cache_dir=config.preprocess_cache,
        fixed_iters_E=None,
        max_iters_E=config.max_iters_e,
        fixed_iters_Pi=config.max_iters_pi,
        neumann_terms=config.max_neumann_terms,
        family_chunk_size=config.family_chunk_size,
        clade_budget=config.clade_budget,
        batch_packing="depth_first_fit",
        max_wave_size=config.max_wave_size,
        lazy_preprocess=True,
        prefetch_batches="all",
        adaptive_iters=True,
        convergence_check_interval=config.convergence_check_interval,
        e_logsumexp_tol=config.e_logsumexp_tol,
        pi_max_diff_tol=config.pi_max_diff_tol,
        gradient_change_tol=config.gradient_change_tol,
        gradient_change_rtol=config.gradient_change_rtol,
        origination_probs=origination_probs,
    )


def tensor_inf(tensor: torch.Tensor) -> float:
    return float(tensor.detach().abs().amax().cpu())


def kkt_metrics(
    model: opt.GeneReconModel,
    config: opt.RunConfig,
    branch: opt.BranchScaledParameters,
    *,
    branch_zero_tol: float = 1e-6,
) -> dict[str, float]:
    branch.branch_log_l.requires_grad_(True)
    for param in (branch.shared_theta, branch.branch_log_l):
        param.grad = None
    _objective, metrics = opt.evaluate_and_backward(
        model,
        config,
        branch,
        [branch.shared_theta, branch.branch_log_l],
    )
    shared_inf = tensor_inf(branch.shared_theta.grad)
    branch_total_grad = branch.branch_log_l.grad.detach()
    branch_l = torch.exp(branch.branch_log_l.detach())
    smooth_prior_grad = (
        config.branchscale_prior_weight
        * torch.sign(branch_l - 1.0)
        * branch_l
    )
    branch_data_grad = branch_total_grad - smooth_prior_grad
    zero_mask = (branch_l - 1.0).abs() <= branch_zero_tol
    branch_residual = branch_total_grad.abs()
    branch_residual = torch.where(
        zero_mask,
        torch.clamp(branch_data_grad.abs() - config.branchscale_prior_weight, min=0.0),
        branch_residual,
    )
    metrics.update(
        {
            "kkt/shared_grad_inf": shared_inf,
            "kkt/branch_subgradient_residual_inf": tensor_inf(branch_residual),
            "kkt/branch_data_grad_inf": tensor_inf(branch_data_grad),
            "kkt/branch_total_grad_inf": tensor_inf(branch_total_grad),
            "kkt/branch_at_kink": float(zero_mask.sum().detach().cpu()),
            "kkt/branch_off_kink": float((~zero_mask).sum().detach().cpu()),
            "kkt/residual_inf": max(shared_inf, tensor_inf(branch_residual)),
        }
    )
    return metrics


def write_row(history_path: Path, row: dict[str, object]) -> None:
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Branchscale local-minimum run: 100 Adam steps, then Strong-Wolfe "
            "LBFGS with L1 KKT checks."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/data/HOGENOM/hogenom/output_gpurec_penalty316_adam100_lbfgs_kkt"),
    )
    parser.add_argument("--adam-steps", type=int, default=100)
    parser.add_argument("--lbfgs-steps", type=int, default=50)
    parser.add_argument("--penalty", type=float, default=DEFAULT_PENALTY)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--final-branch-mode",
        choices=("shared-at-one", "full"),
        default="shared-at-one",
        help=(
            "Use shared-at-one for penalties whose optimum is at l_e=1; use full "
            "to include branch multipliers in the Strong-Wolfe LBFGS phase."
        ),
    )
    parser.add_argument("--branch-zero-tol", type=float, default=1e-6)
    parser.add_argument(
        "--branch-snap-tol",
        type=float,
        default=0.0,
        help="In full branch mode, snap |log_l_e| below this tolerance exactly to the L1 kink.",
    )
    parser.add_argument("--lbfgs-max-iter", type=int, default=100)
    parser.add_argument("--lbfgs-max-eval", type=int, default=125)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument("--final-pi-iters", type=int, default=256)
    parser.add_argument("--final-neumann-terms", type=int, default=256)
    parser.add_argument("--final-pi-max-diff-tol", type=float, default=1e-7)
    parser.add_argument("--final-gradient-change-tol", type=float, default=1e-7)
    parser.add_argument("--objective-patience", type=int, default=5)
    parser.add_argument("--objective-min-delta", type=float, default=0.0)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"{stamp}_branchscaled_penalty{penalty_slug(args.penalty)}_kkt"
    out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "latest_run.txt").write_text(str(out_dir) + "\n", encoding="utf-8")
    history_path = out_dir / "history.jsonl"

    base_args = opt.parse_args(
        [
            "--out-dir",
            str(out_dir),
            "--no-timestamped-out-dir",
            "--device",
            "cuda",
            "--mode",
            "branchscaled",
            "--optimizer",
            "adam-lbfgs",
            "--adam-warmup-steps",
            str(args.adam_steps),
            "--steps",
            str(args.adam_steps + args.lbfgs_steps),
            "--lr",
            "0.01",
            "--lr-decay-every",
            "100",
            "--lr-decay-factor",
            "0.5",
            "--lbfgs-lr",
            "1.0",
            "--lbfgs-history-size",
            "50",
            "--lbfgs-max-iter",
            "100",
            "--lbfgs-max-eval",
            "125",
            "--lbfgs-line-search",
            "strong_wolfe",
            "--max-iters-pi",
            "64",
            "--max-neumann-terms",
            "64",
            "--solver-iteration-schedule",
            "adaptive",
            "--family-chunk-size",
            "0",
            "--clade-budget",
            "305000",
            "--max-wave-size",
            "8192",
            "--beta-prior-weight",
            "0",
            "--branchscale-prior-weight",
            str(args.penalty),
            "--loss-patience",
            "0",
            "--best-likelihood-patience",
            "0",
            "--grad-inf-tol",
            str(args.tol),
            "--wandb-mode",
            "disabled",
        ]
    )
    config = opt.config_from_args(base_args)
    (out_dir / "run_config.json").write_text(
        json.dumps(opt._jsonable(opt.asdict(config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    root = opt.parse_newick(config.species_tree)
    layout_cache = opt.tree_layout(root)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    model = build_model(config, dtype)
    labels = opt.species_labels(model)
    branch = opt.make_branchscaled_parameters(model)
    all_params = opt.trainable_parameters(model, branch)
    adam = torch.optim.Adam(all_params, lr=config.lr)

    print(f"out_dir={out_dir}", flush=True)
    started = time.perf_counter()

    def save_checkpoint(path: Path, step: int, stop_reason: str, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "step": step,
                "stop_reason": stop_reason,
                "shared_theta": branch.shared_theta.detach().cpu(),
                "branch_log_l": branch.branch_log_l.detach().cpu(),
                "branch_l": torch.exp(branch.branch_log_l.detach()).cpu(),
                "elapsed_s": time.perf_counter() - started,
                "penalty": args.penalty,
                "final_branch_mode": args.final_branch_mode,
                "metrics": dict(metrics),
            },
            path,
        )

    previous_objective: float | None = None
    for step in range(args.adam_steps):
        step_t0 = time.perf_counter()
        opt.apply_solver_iteration_settings(
            model,
            opt.solver_iteration_settings(config, step, None),
        )
        opt.set_optimizer_lr(adam, opt.scheduled_lr(config, step))
        objective, metrics = opt.evaluate_and_backward(model, config, branch, all_params)
        adam.step()
        opt.clamp_parameters_(config, model, branch)
        model.clear()
        objective_bits = metrics["objective/bits"]
        delta = None if previous_objective is None else previous_objective - objective_bits
        previous_objective = objective_bits
        row = {
            "step": step,
            "phase": "adam",
            "delta_objective_bits": delta,
            "step_s": time.perf_counter() - step_t0,
            **metrics,
            **opt.branchscale_stats(branch),
        }
        write_row(history_path, row)
        if step % 10 == 0 or step == args.adam_steps - 1:
            print(
                f"step={step:04d} phase=adam objective={objective_bits:.6f} "
                f"grad_inf={metrics['grad/inf']:.6g} "
                f"l_max={row['branchscale/l_max']:.6g} step_s={row['step_s']:.3f}",
                flush=True,
            )

    final_solver_settings = opt.SolverIterationSettings(
        phase="final_kkt",
        pi_iters=args.final_pi_iters,
        neumann_terms=args.final_neumann_terms,
        pi_max_diff_tol=args.final_pi_max_diff_tol,
        gradient_change_tol=args.final_gradient_change_tol,
    )
    opt.apply_solver_iteration_settings(model, final_solver_settings)
    model.clear()
    with torch.no_grad():
        if args.final_branch_mode == "shared-at-one":
            branch.branch_log_l.zero_()
        model.theta.copy_(opt.effective_theta(model, branch))
    model.clear()
    if args.final_branch_mode == "shared-at-one":
        print("snapped branch multipliers to l_e=1 for the L1 KKT kink", flush=True)
    else:
        print("keeping Adam branch multipliers trainable for full LBFGS", flush=True)
    kkt = kkt_metrics(model, config, branch, branch_zero_tol=args.branch_zero_tol)
    previous_objective = kkt["objective/bits"]
    write_row(
        history_path,
        {
            "step": args.adam_steps,
            "phase": (
                "l1_kink_projection"
                if args.final_branch_mode == "shared-at-one"
                else "post_adam_kkt"
            ),
            "final_branch_mode": args.final_branch_mode,
            **kkt,
            **opt.branchscale_stats(branch),
        },
    )
    print(
        f"step={args.adam_steps:04d} phase="
        f"{'l1_kink_projection' if args.final_branch_mode == 'shared-at-one' else 'post_adam_kkt'} "
        f"objective={kkt['objective/bits']:.6f} "
        f"shared_grad_inf={kkt['kkt/shared_grad_inf']:.6g} "
        f"branch_kkt_inf={kkt['kkt/branch_subgradient_residual_inf']:.6g}",
        flush=True,
    )

    best_step = args.adam_steps
    best_objective = kkt["objective/bits"]
    best_kkt = kkt["kkt/residual_inf"]
    best_metrics = dict(kkt)
    best_shared_theta = branch.shared_theta.detach().clone()
    best_branch_log_l = branch.branch_log_l.detach().clone()
    no_improvement_steps = 0
    checkpoint_best = out_dir / "checkpoint_best.pt"
    save_checkpoint(checkpoint_best, best_step, "best_so_far", best_metrics)

    branch.branch_log_l.requires_grad_(args.final_branch_mode == "full")
    lbfgs_params = (
        [branch.shared_theta]
        if args.final_branch_mode == "shared-at-one"
        else [branch.shared_theta, branch.branch_log_l]
    )
    lbfgs = torch.optim.LBFGS(
        lbfgs_params,
        lr=1.0,
        max_iter=args.lbfgs_max_iter,
        max_eval=args.lbfgs_max_eval,
        history_size=args.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )

    stop_reason = "max_lbfgs_steps"
    final_step = args.adam_steps
    lbfgs_phase = (
        "lbfgs_shared_strong_wolfe"
        if args.final_branch_mode == "shared-at-one"
        else "lbfgs_full_strong_wolfe"
    )
    for outer in range(args.lbfgs_steps):
        step = args.adam_steps + outer + 1
        final_step = step
        step_t0 = time.perf_counter()
        closure_evals = 0
        opt.apply_solver_iteration_settings(model, final_solver_settings)

        def closure() -> torch.Tensor:
            nonlocal closure_evals
            closure_evals += 1
            lbfgs.zero_grad(set_to_none=True)
            objective_i, _metrics_i = opt.evaluate_and_backward(
                model,
                config,
                branch,
                lbfgs_params,
            )
            return objective_i

        lbfgs.step(closure)
        opt.clamp_parameters_(config, model, branch)
        with torch.no_grad():
            if args.final_branch_mode == "shared-at-one":
                branch.branch_log_l.zero_()
            elif args.branch_snap_tol > 0.0:
                branch.branch_log_l.masked_fill_(
                    branch.branch_log_l.abs() <= args.branch_snap_tol,
                    0.0,
                )
            model.theta.copy_(opt.effective_theta(model, branch))
        model.clear()
        kkt = kkt_metrics(model, config, branch, branch_zero_tol=args.branch_zero_tol)
        delta = previous_objective - kkt["objective/bits"]
        previous_objective = kkt["objective/bits"]
        row = {
            "step": step,
            "phase": lbfgs_phase,
            "final_branch_mode": args.final_branch_mode,
            "closure_evals": closure_evals,
            "delta_objective_bits": delta,
            "step_s": time.perf_counter() - step_t0,
            **kkt,
            **opt.branchscale_stats(branch),
        }
        write_row(history_path, row)
        objective = kkt["objective/bits"]
        residual = kkt["kkt/residual_inf"]
        objective_improved = objective < best_objective
        objective_improved_for_patience = objective < best_objective - args.objective_min_delta
        kkt_tie_improved = (
            objective <= best_objective + args.objective_min_delta
            and residual < best_kkt
        )
        if objective_improved or kkt_tie_improved:
            best_step = step
            best_objective = objective
            best_kkt = residual
            best_metrics = dict(kkt)
            best_shared_theta = branch.shared_theta.detach().clone()
            best_branch_log_l = branch.branch_log_l.detach().clone()
            save_checkpoint(checkpoint_best, best_step, "best_so_far", best_metrics)
        if objective_improved_for_patience or kkt_tie_improved:
            no_improvement_steps = 0
        else:
            no_improvement_steps += 1
        print(
            f"step={step:04d} phase={lbfgs_phase} "
            f"objective={kkt['objective/bits']:.6f} delta={delta:.6g} "
            f"shared_grad_inf={kkt['kkt/shared_grad_inf']:.6g} "
            f"branch_kkt_inf={kkt['kkt/branch_subgradient_residual_inf']:.6g} "
            f"kkt_inf={kkt['kkt/residual_inf']:.6g} "
            f"best_objective={best_objective:.6f} "
            f"no_improve={no_improvement_steps}/{args.objective_patience} "
            f"closures={closure_evals} step_s={row['step_s']:.3f}",
            flush=True,
        )
        branch.branch_log_l.requires_grad_(args.final_branch_mode == "full")
        if kkt["kkt/residual_inf"] <= args.tol:
            stop_reason = "kkt_tolerance"
            break
        if no_improvement_steps >= args.objective_patience:
            stop_reason = "objective_patience"
            break

    with torch.no_grad():
        branch.shared_theta.copy_(best_shared_theta)
        branch.branch_log_l.copy_(best_branch_log_l)
        model.theta.copy_(opt.effective_theta(model, branch))
    model.clear()
    final_rates = out_dir / "branchscaled_node_rates_final.tsv"
    opt.write_rate_table(final_rates, model, labels, branch)
    final_plot = out_dir / "tree_plots" / "rates_final.png"
    opt.plot_tree_rates(
        root=root,
        layout_cache=layout_cache,
        rate_by_label=opt.current_rate_by_label(model, labels, branch),
        out_path=final_plot,
        title=f"HOGENOM CCP rates final (branchscaled penalty {args.penalty:g} KKT)",
    )
    final_metrics = dict(best_metrics)
    final_metrics["best_step"] = float(best_step)
    final_metrics["best_objective_bits"] = best_objective
    final_metrics["best_kkt_residual_inf"] = best_kkt
    final_metrics["objective_patience"] = float(args.objective_patience)
    final_metrics["objective_min_delta"] = float(args.objective_min_delta)
    save_checkpoint(out_dir / "checkpoint_final.pt", best_step, stop_reason, final_metrics)
    print(f"stopped reason={stop_reason}", flush=True)
    print(f"best_step {best_step}", flush=True)
    print(f"best_objective_bits {best_objective:.6f}", flush=True)
    print(f"best_kkt_residual_inf {best_kkt:.6g}", flush=True)
    print(f"history {history_path}", flush=True)
    print(f"best_checkpoint {checkpoint_best}", flush=True)
    print(f"final_rates {final_rates}", flush=True)
    print(f"final_plot {final_plot}", flush=True)


if __name__ == "__main__":
    main()

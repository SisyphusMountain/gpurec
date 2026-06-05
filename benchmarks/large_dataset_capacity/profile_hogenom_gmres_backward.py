#!/usr/bin/env python3
"""Profile one HOGENOM GMRES implicit backward pass with Nsight Systems."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.large_dataset_capacity.hogenom_gmres_neumann_family_experiment import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FAMILIES_FILE,
    DEFAULT_FAMILY_INDEX,
    DEFAULT_FAMILY_NAME,
    DEFAULT_SPECIES_TREE,
    build_model,
    family_tree_path,
)
from gpurec.api import _implicit_grad as implicit_grad_module
from gpurec.core.inference.solver import nll_from_root_rows, receiver_weights_are_uniform, solve_resident_e_pi
from gpurec.core.kernels import wave_backward as wave_backward_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-tree", type=Path, default=DEFAULT_SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=DEFAULT_FAMILIES_FILE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--family-index", type=int, default=DEFAULT_FAMILY_INDEX)
    parser.add_argument("--family-name", default=DEFAULT_FAMILY_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--e-iters", type=int, default=256)
    parser.add_argument("--pi-iters", type=int, default=256)
    parser.add_argument("--reference-neumann", type=int, default=512)
    parser.add_argument("--gmres-iters", type=int, default=10)
    parser.add_argument("--gmres-tol", type=float, default=1e-10)
    parser.add_argument("--gmres-check-interval", type=int, default=1)
    parser.add_argument(
        "--self-loop-solver",
        choices=("gmres", "gmres_fixed"),
        default="gmres",
    )
    parser.add_argument("--clade-budget", type=int, default=250_000)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _prepare_forward_state(
    model,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
) -> dict[str, Any]:
    model.clear_warm_starts()
    static = model.batch_statics[0]
    (
        E,
        E_s1,
        E_s2,
        Ebar,
        root_rows,
        pi_wave,
        pibar_wave,
        pibar_row_max,
        log_pS,
        log_pD,
        log_pL,
        max_transfer_vec,
        receiver_log_probs,
    ) = solve_resident_e_pi(static, theta, receiver_weights, warm_start_E=static.warm_E)
    loss = nll_from_root_rows(root_rows, E).detach()
    use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)
    return {
        "static": static,
        "E": E,
        "E_s1": E_s1,
        "E_s2": E_s2,
        "Ebar": Ebar,
        "pi_wave": pi_wave,
        "pibar_wave": pibar_wave,
        "pibar_row_max": pibar_row_max,
        "log_pS": log_pS,
        "log_pD": log_pD,
        "log_pL": log_pL,
        "max_transfer_vec": max_transfer_vec,
        "receiver_log_probs": receiver_log_probs,
        "loss": loss,
        "use_receiver_weights": use_receiver_weights,
    }


def _run_gmres_backward_from_state(
    state: dict[str, Any],
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    gmres_iters: int,
    gmres_tol: float,
    gmres_check_interval: int,
    self_loop_solver: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    static = state["static"]
    static.solver_options.neumann_terms = int(gmres_iters)

    original_gmres_stats = wave_backward_module._GMRES_SELF_LOOP_STATS
    gmres_stats: list[dict[str, float | int]] = []

    wave_backward_module._GMRES_SELF_LOOP_STATS = gmres_stats
    try:
        grad_theta, grad_receiver = implicit_grad_module.implicit_grad_loglik_vjp_wave(
            static.wave_layout,
            static.species_helpers,
            Pi_star_wave=state["pi_wave"],
            Pibar_star_wave=state["pibar_wave"],
            E_star=state["E"],
            Ebar=state["Ebar"],
            E_s1=state["E_s1"],
            E_s2=state["E_s2"],
            log_pS=state["log_pS"],
            log_pD=state["log_pD"],
            log_pL=state["log_pL"],
            max_transfer_mat=state["max_transfer_vec"],
            receiver_log_probs=state["receiver_log_probs"],
            use_receiver_weights=state["use_receiver_weights"],
            theta=theta,
            receiver_weights=receiver_weights,
            family_idx=static.rate_family_idx,
            uniform_pibar_row_max=state["pibar_row_max"],
            specieswise=static.specieswise,
            genewise=static.genewise,
            neumann_terms=int(gmres_iters),
            self_loop_solver=self_loop_solver,
            gmres_tol=gmres_tol,
            gmres_check_interval=gmres_check_interval,
            bicgstab_max_iter=static.solver_options.bicgstab_max_iter,
            bicgstab_tol=static.solver_options.bicgstab_tol,
            bicgstab_breakdown_tol=static.solver_options.bicgstab_breakdown_tol,
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
        )
    finally:
        wave_backward_module._GMRES_SELF_LOOP_STATS = original_gmres_stats

    per_wave_iterations = [int(row["iterations"]) for row in gmres_stats]
    per_wave_checks = [int(row.get("check_count", 0)) for row in gmres_stats]
    stats = {
        "wave_count": len(gmres_stats),
        "total_backward_iterations": int(sum(per_wave_iterations)),
        "total_gmres_checks": int(sum(per_wave_checks)),
        "mean_wave_iterations": float(sum(per_wave_iterations) / max(1, len(per_wave_iterations))),
        "mean_gmres_checks": float(sum(per_wave_checks) / max(1, len(per_wave_checks))),
        "max_wave_iterations": max(per_wave_iterations, default=0),
        "max_gmres_checks": max(per_wave_checks, default=0),
        "max_rel_res": max((float(row["rel_res"]) for row in gmres_stats), default=0.0),
        "loss": float(state["loss"].detach().cpu()),
    }
    return grad_theta.detach(), grad_receiver.detach(), stats


def _run_gmres_backward(
    model,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    gmres_iters: int,
    gmres_tol: float,
    gmres_check_interval: int,
    self_loop_solver: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    model.configure_solver(neumann_terms=int(gmres_iters))
    state = _prepare_forward_state(model, theta, receiver_weights)
    return _run_gmres_backward_from_state(
        state,
        theta,
        receiver_weights,
        gmres_iters=gmres_iters,
        gmres_tol=gmres_tol,
        gmres_check_interval=gmres_check_interval,
        self_loop_solver=self_loop_solver,
    )


def _cuda_profiler_start() -> None:
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop() -> None:
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    args = parse_args()
    tree_path = family_tree_path(args.families_file, args.family_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    theta_row = checkpoint["theta"][args.family_index].detach().double()

    model = build_model(args, tree_path)
    theta = theta_row.to(device=args.device).contiguous()
    receiver_weights = torch.zeros(
        (int(model.species_helpers["S"]),),
        dtype=theta.dtype,
        device=theta.device,
    )

    for _ in range(max(0, int(args.warmup))):
        _run_gmres_backward(
            model,
            theta,
            receiver_weights,
            gmres_iters=args.gmres_iters,
            gmres_tol=args.gmres_tol,
            gmres_check_interval=args.gmres_check_interval,
            self_loop_solver=args.self_loop_solver,
        )
    if theta.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(theta.device)

    state = _prepare_forward_state(model, theta, receiver_weights)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("gmres_backward_only")
    _cuda_profiler_start()
    start = time.perf_counter()
    grad_theta, grad_receiver, stats = _run_gmres_backward_from_state(
        state,
        theta,
        receiver_weights,
        gmres_iters=args.gmres_iters,
        gmres_tol=args.gmres_tol,
        gmres_check_interval=args.gmres_check_interval,
        self_loop_solver=args.self_loop_solver,
    )
    _cuda_profiler_stop()
    elapsed_s = time.perf_counter() - start
    torch.cuda.nvtx.range_pop()

    peak_gb = None
    if theta.device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(theta.device) / 1024**3

    result = {
        "family_index": args.family_index,
        "family_name": args.family_name,
        "family_tree": str(tree_path),
        "checkpoint": str(args.checkpoint),
        "gmres_iters": int(args.gmres_iters),
        "gmres_tol": float(args.gmres_tol),
        "gmres_check_interval": int(args.gmres_check_interval),
        "self_loop_solver": args.self_loop_solver,
        "warmup": int(args.warmup),
        "elapsed_s": elapsed_s,
        "peak_gb": peak_gb,
        "gradient": grad_theta.detach().cpu().double().reshape(-1).tolist(),
        "receiver_grad_norm": float(torch.linalg.vector_norm(grad_receiver).detach().cpu()),
        **stats,
    }
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")

    del model
    if theta.device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

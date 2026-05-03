#!/usr/bin/env python3
"""Uniform backward benchmark helper for Proposal 8 profiling.

This intentionally lives outside core code. It times one warmed backward pass
for the existing global/uniform workload and prints static wave/family shape
metrics useful for evaluating family/chunk-aware pruning.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

import torch

from gpurec.api.autograd import _extract_parameters
from gpurec import GeneReconModel
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood


DEFAULT_FLAGS = {
    "GPUREC_KERNELIZED_BACKWARD_DTS": "1",
    "GPUREC_FUSED_DTS_BACKWARD_ACCUM": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL": "tree",
    "GPUREC_FUSED_UNIFORM_BACKWARD": "1",
    "GPUREC_UNIFORM_PINGPONG": "1",
    "GPUREC_BACKWARD_LEAF_INDEX": "1",
    "GPUREC_FUSED_WAVE_PARAM_ACCUM": "1",
    "GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES": "1",
    "GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS": "1",
    "GPUREC_DTS_GRAD_MT_TWO_STAGE": "1",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.getenv("DATASET", "tests/data/test_trees_1000"))
    parser.add_argument("--fams", type=int, default=int(os.getenv("FAMS", "50")))
    parser.add_argument("--start", type=int, default=int(os.getenv("FAMILY_START", "0")))
    parser.add_argument("--reps", type=int, default=int(os.getenv("REPS", "9")))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "5")))
    parser.add_argument("--max-wave-size", default=os.getenv("MAX_WAVE_SIZE", "32768"))
    parser.add_argument("--max-root-wave-size", type=int, default=None)
    parser.add_argument("--pruning-threshold", type=float, default=float(os.getenv("PRUNING_THRESHOLD", "1e-6")))
    parser.add_argument("--use-pruning", action=argparse.BooleanOptionalAction, default=os.getenv("USE_PRUNING", "1") != "0")
    parser.add_argument("--stats-only", action="store_true", default=os.getenv("STATS_ONLY", "0") != "0")
    parser.add_argument("--diag-only", action="store_true", default=os.getenv("DIAG_ONLY", "0") != "0")
    parser.add_argument("--diag-chunk-rows", type=int, default=int(os.getenv("DIAG_CHUNK_ROWS", "256")))
    parser.add_argument("--profile-cuda-api", action="store_true", default=os.getenv("PROFILE_CUDA_API", "0") != "0")
    parser.add_argument("--cuda-graph", action="store_true", default=os.getenv("CUDA_GRAPH", "0") != "0")
    parser.add_argument(
        "--cuda-graph-target",
        choices=("model", "pi_backward"),
        default=os.getenv("CUDA_GRAPH_TARGET", "model"),
        help="Capture target for --cuda-graph.",
    )
    parser.add_argument(
        "--graph-fixed-schedule-mode",
        choices=("no_cpu", "device", "existing"),
        default=os.getenv("CUDA_GRAPH_FIXED_SCHEDULE_MODE", "no_cpu"),
        help=(
            "Fixed-schedule pruning mode to use for CUDA graph capture. "
            "'no_cpu' sets GPUREC_BACKWARD_NO_CPU_PRUNING=1, 'device' sets "
            "GPUREC_DEVICE_PRUNING=1, and 'existing' requires one of them to "
            "already be set."
        ),
    )
    parser.add_argument(
        "--cuda-graph-check",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("CUDA_GRAPH_CHECK", "1") != "0",
        help="Compare one normal fixed-schedule backward against one graph replay after capture.",
    )
    parser.add_argument("--cache-dir", default=os.getenv("PREPROCESS_CACHE_DIR", "/tmp/gpurec_preprocess_cache"))
    args = parser.parse_args()
    mws = str(args.max_wave_size).strip().lower()
    args.max_wave_size = None if mws in ("", "0", "none", "null") else int(mws)
    return args


def _configure_cuda_graph_env(args: argparse.Namespace) -> None:
    if not args.cuda_graph:
        return
    if args.graph_fixed_schedule_mode == "no_cpu":
        os.environ["GPUREC_BACKWARD_NO_CPU_PRUNING"] = "1"
    elif args.graph_fixed_schedule_mode == "device":
        os.environ["GPUREC_DEVICE_PRUNING"] = "1"


def _validate_cuda_graph_env(args: argparse.Namespace) -> None:
    if not args.cuda_graph:
        return
    no_cpu = os.environ.get("GPUREC_BACKWARD_NO_CPU_PRUNING", "0") != "0"
    device = os.environ.get("GPUREC_DEVICE_PRUNING", "0") != "0"
    if not (no_cpu or device):
        raise RuntimeError(
            "CUDA graph mode needs a fixed backward schedule. Set "
            "GPUREC_BACKWARD_NO_CPU_PRUNING=1, GPUREC_DEVICE_PRUNING=1, or use "
            "--graph-fixed-schedule-mode=no_cpu/device."
        )


def _cuda_event_elapsed(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _grad_snapshot(model: GeneReconModel) -> list[torch.Tensor | None]:
    return [
        None if p.grad is None else p.grad.detach().clone()
        for p in model.parameters()
    ]


def _max_grad_abs_diff(
    lhs: list[torch.Tensor | None],
    rhs: list[torch.Tensor | None],
) -> float:
    max_diff = 0.0
    for a, b in zip(lhs, rhs):
        if a is None or b is None:
            max_diff = max(max_diff, 0.0 if a is b else float("inf"))
            continue
        max_diff = max(max_diff, float((a - b).abs().max().detach().cpu()))
    return max_diff


def _max_tensor_dict_abs_diff(
    lhs: dict,
    rhs: dict,
    *,
    keys: tuple[str, ...],
) -> float:
    max_diff = 0.0
    for key in keys:
        a = lhs.get(key)
        b = rhs.get(key)
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        max_diff = max(max_diff, float((a - b).abs().max().detach().cpu()))
    return max_diff


def _selected_genes(root: Path, start: int, fams: int) -> list[str]:
    genes = sorted(root.glob("g_*.nwk"))
    stop = start + fams
    if stop > len(genes):
        raise ValueError(f"requested families [{start}:{stop}], but only found {len(genes)} genes in {root}")
    return [str(p) for p in genes[start:stop]]


def _family_runs(values: torch.Tensor) -> int:
    if values.numel() == 0:
        return 0
    return int(1 + (values[1:] != values[:-1]).sum().item())


def _print_wave_shape(model: GeneReconModel) -> None:
    wl = model.static.wave_layout
    metas = wl["wave_metas"]
    family_idx = wl.get("family_idx")
    if family_idx is not None:
        family_idx = family_idx.detach().cpu()

    split_rows = sum(int(m["sl"].numel()) if m.get("has_splits", False) else 0 for m in metas)
    leaves = int(model.static.wave_layout["leaf_row_index"].numel())
    roots = int(model.static.root_clade_ids.numel())
    max_w = max((int(m["W"]) for m in metas), default=0)

    print(
        "shape",
        "S", model.n_species,
        "G", model.n_families,
        "C", int(model.static.Pi_shape[0]) if hasattr(model.static, "Pi_shape") else int(sum(m["W"] for m in metas)),
        "waves", len(metas),
        "maxW", max_w,
        "split_rows", split_rows,
        "leaves", leaves,
        "roots", roots,
    )

    rows = []
    for k, meta in enumerate(metas):
        ws = int(meta["start"])
        we = int(meta["end"])
        w = int(meta["W"])
        n_splits = int(meta["sl"].numel()) if meta.get("has_splits", False) else 0
        fanout = (n_splits / w) if w else 0.0
        fams = runs = split_fams = 0
        if family_idx is not None:
            fi = family_idx[ws:we]
            fams = int(torch.unique(fi).numel())
            runs = _family_runs(fi)
            if n_splits:
                parent_fi = family_idx[ws + meta["reduce_idx"].detach().cpu()]
                split_fams = int(torch.unique(parent_fi).numel())
        rows.append((k, ws, w, n_splits, fanout, fams, runs, split_fams))

    print("top_wave_rows k start W split_rows fanout families family_runs split_families")
    for row in sorted(rows, key=lambda r: r[2], reverse=True)[:12]:
        print("%d %d %d %d %.3f %d %d %d" % row)

    print("top_split_rows k start W split_rows fanout families family_runs split_families")
    for row in sorted(rows, key=lambda r: r[3], reverse=True)[:12]:
        print("%d %d %d %d %.3f %d %d %d" % row)


@torch.no_grad()
def _prepare_pi_backward_inputs(model: GeneReconModel) -> tuple[dict, torch.Tensor]:
    static = model.static
    theta = model.theta.detach()
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = _extract_parameters(
        theta, static
    )

    E_out = E_fixed_point(
        species_helpers=static.species_helpers,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        transfer_mat=transfer_mat,
        max_transfer_mat=max_transfer_vec,
        max_iters=static.max_iters_E,
        tolerance=static.tol_E,
        warm_start_E=static.warm_E,
        dtype=static.dtype,
        device=static.device,
        pibar_mode=static.pibar_mode,
        ancestors_T=static.ancestors_T,
    )
    pi_out = Pi_wave_forward(
        wave_layout=static.wave_layout,
        species_helpers=static.species_helpers,
        E=E_out["E"],
        Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"],
        E_s2=E_out["E_s2"],
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        transfer_mat=transfer_mat,
        max_transfer_mat=max_transfer_vec,
        device=static.device,
        dtype=static.dtype,
        local_iters=static.max_iters_Pi,
        local_tolerance=static.tol_Pi,
        fixed_iters=static.fixed_iters_Pi,
        pibar_mode=static.pibar_mode,
        return_original=False,
        family_idx=(
            static.wave_layout.get("family_idx") if static.genewise else None
        ),
    )
    nll = compute_log_likelihood(
        pi_out["Pi_wave_ordered"],
        E_out["E"],
        static.wave_layout["root_clade_ids"],
    ).sum()
    kwargs = {
        "wave_layout": static.wave_layout,
        "Pi_star_wave": pi_out["Pi_wave_ordered"],
        "Pibar_star_wave": pi_out["Pibar_wave_ordered"],
        "E": E_out["E"],
        "Ebar": E_out["E_bar"],
        "E_s1": E_out["E_s1"],
        "E_s2": E_out["E_s2"],
        "log_pS": log_pS,
        "log_pD": log_pD,
        "log_pL": log_pL,
        "max_transfer_mat": max_transfer_vec,
        "species_helpers": static.species_helpers,
        "root_clade_ids_perm": static.wave_layout["root_clade_ids"],
        "device": static.device,
        "dtype": static.dtype,
        "neumann_terms": static.neumann_terms,
        "use_pruning": static.use_pruning,
        "pruning_threshold": static.pruning_threshold,
        "pibar_mode": static.pibar_mode,
        "transfer_mat": transfer_mat,
        "ancestors_T": static.ancestors_T,
        "uniform_pibar_row_max": pi_out.get("uniform_pibar_row_max"),
    }
    return kwargs, nll


@torch.no_grad()
def _run_pi_backward_diag(model: GeneReconModel, args: argparse.Namespace) -> None:
    kwargs, nll = _prepare_pi_backward_inputs(model)

    old_diag = os.environ.get("GPUREC_BACKWARD_FAMILY_CHUNK_DIAG")
    old_rows = os.environ.get("GPUREC_BACKWARD_FAMILY_CHUNK_ROWS")
    os.environ["GPUREC_BACKWARD_FAMILY_CHUNK_DIAG"] = "1"
    os.environ["GPUREC_BACKWARD_FAMILY_CHUNK_ROWS"] = str(args.diag_chunk_rows)
    try:
        pi_bwd = Pi_wave_backward(**kwargs)
    finally:
        if old_diag is None:
            os.environ.pop("GPUREC_BACKWARD_FAMILY_CHUNK_DIAG", None)
        else:
            os.environ["GPUREC_BACKWARD_FAMILY_CHUNK_DIAG"] = old_diag
        if old_rows is None:
            os.environ.pop("GPUREC_BACKWARD_FAMILY_CHUNK_ROWS", None)
        else:
            os.environ["GPUREC_BACKWARD_FAMILY_CHUNK_ROWS"] = old_rows

    torch.cuda.synchronize()
    print("diag_loss", f"{float(nll.detach().cpu()):.8f}")
    diag = pi_bwd.get("family_chunk_pruning_diag", {})
    print("family_chunk_pruning_diag")
    for key in sorted(diag):
        print(key, diag[key])


def _run_cuda_graph_model_bench(model: GeneReconModel, args: argparse.Namespace) -> None:
    _validate_cuda_graph_env(args)

    normal_times = []
    normal_peaks = []
    for _ in range(args.warmups):
        model.zero_grad(set_to_none=False)
        loss = model()
        loss.backward()
        torch.cuda.synchronize()

    baseline_loss = None
    baseline_grads = None
    for _ in range(args.reps):
        model.zero_grad(set_to_none=False)
        loss = model()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStart()

        def backward_only() -> None:
            loss.backward()

        normal_times.append(_cuda_event_elapsed(backward_only))
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStop()
        normal_peaks.append(torch.cuda.max_memory_allocated() / 1e9)
        if baseline_loss is None:
            baseline_loss = loss.detach().clone()
            baseline_grads = _grad_snapshot(model)

    assert baseline_loss is not None and baseline_grads is not None
    del loss

    try:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            model.zero_grad(set_to_none=False)
            warmup_loss = model()
            warmup_loss.backward()
        torch.cuda.current_stream().wait_stream(side_stream)
        del warmup_loss

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            model.zero_grad(set_to_none=False)
            static_loss = model()
            static_loss.backward()
    except Exception as exc:
        torch.cuda.synchronize()
        print("cuda_graph_capture_failed", "target", "model", type(exc).__name__, str(exc))
        print(
            "cuda_graph_attempt",
            "side-stream warmup, deleted live eager loss, then captured "
            "zero_grad(set_to_none=False); model(); loss.backward() with "
            "fixed-schedule env",
        )
        print(
            "normal_backward_ms",
            "mean", f"{statistics.mean(normal_times):.3f}",
            "median", f"{statistics.median(normal_times):.3f}",
            "min", f"{min(normal_times):.3f}",
            "times", ",".join(f"{t:.3f}" for t in normal_times),
        )
        print("normal_peak_alloc_gb", f"{max(normal_peaks):.3f}")
        return

    graph.replay()
    torch.cuda.synchronize()

    graph_loss = static_loss.detach().clone()
    graph_grads = _grad_snapshot(model)
    loss_abs_diff = float((graph_loss - baseline_loss).abs().detach().cpu())
    grad_abs_diff = _max_grad_abs_diff(baseline_grads, graph_grads)

    replay_times = []
    replay_peaks = []
    for _ in range(args.reps):
        torch.cuda.reset_peak_memory_stats()
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStart()
        replay_times.append(_cuda_event_elapsed(graph.replay))
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStop()
        replay_peaks.append(torch.cuda.max_memory_allocated() / 1e9)

    print("loss", float(graph_loss.detach().cpu()))
    print(
        "normal_backward_ms",
        "mean", f"{statistics.mean(normal_times):.3f}",
        "median", f"{statistics.median(normal_times):.3f}",
        "min", f"{min(normal_times):.3f}",
        "times", ",".join(f"{t:.3f}" for t in normal_times),
    )
    print(
        "cuda_graph_replay_ms",
        "mean", f"{statistics.mean(replay_times):.3f}",
        "median", f"{statistics.median(replay_times):.3f}",
        "min", f"{min(replay_times):.3f}",
        "times", ",".join(f"{t:.3f}" for t in replay_times),
    )
    if args.cuda_graph_check:
        print(
            "cuda_graph_check",
            "loss_abs_diff", f"{loss_abs_diff:.8e}",
            "max_grad_abs_diff", f"{grad_abs_diff:.8e}",
        )
    print("normal_peak_alloc_gb", f"{max(normal_peaks):.3f}")
    print("cuda_graph_peak_alloc_gb", f"{max(replay_peaks):.3f}")


@torch.no_grad()
def _run_cuda_graph_pi_backward_bench(
    model: GeneReconModel,
    args: argparse.Namespace,
) -> None:
    _validate_cuda_graph_env(args)
    kwargs, nll = _prepare_pi_backward_inputs(model)
    torch.cuda.synchronize()

    normal_times = []
    normal_peaks = []
    baseline_bwd = None
    for _ in range(args.warmups):
        Pi_wave_backward(**kwargs)
        torch.cuda.synchronize()

    for _ in range(args.reps):
        torch.cuda.reset_peak_memory_stats()
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStart()
        holder = {}

        def direct_backward() -> None:
            holder["out"] = Pi_wave_backward(**kwargs)

        normal_times.append(_cuda_event_elapsed(direct_backward))
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStop()
        normal_peaks.append(torch.cuda.max_memory_allocated() / 1e9)
        if baseline_bwd is None:
            baseline_bwd = {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in holder["out"].items()
            }

    assert baseline_bwd is not None

    try:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            Pi_wave_backward(**kwargs)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            static_bwd = Pi_wave_backward(**kwargs)
    except Exception as exc:
        torch.cuda.synchronize()
        print("cuda_graph_capture_failed", "target", "pi_backward", type(exc).__name__, str(exc))
        print(
            "cuda_graph_attempt",
            "computed E_fixed_point/Pi_wave_forward outside capture, then "
            "captured Pi_wave_backward(**static_kwargs) with fixed-schedule env",
        )
        print("loss", float(nll.detach().cpu()))
        print(
            "normal_pi_backward_ms",
            "mean", f"{statistics.mean(normal_times):.3f}",
            "median", f"{statistics.median(normal_times):.3f}",
            "min", f"{min(normal_times):.3f}",
            "times", ",".join(f"{t:.3f}" for t in normal_times),
        )
        print("normal_peak_alloc_gb", f"{max(normal_peaks):.3f}")
        return

    graph.replay()
    torch.cuda.synchronize()

    tensor_keys = (
        "v_Pi",
        "grad_E",
        "grad_Ebar",
        "grad_E_s1",
        "grad_E_s2",
        "grad_log_pD",
        "grad_log_pS",
        "grad_max_transfer_mat",
        "grad_transfer_mat",
    )
    graph_bwd = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in static_bwd.items()
    }
    output_abs_diff = _max_tensor_dict_abs_diff(
        baseline_bwd,
        graph_bwd,
        keys=tensor_keys,
    )

    replay_times = []
    replay_peaks = []
    for _ in range(args.reps):
        torch.cuda.reset_peak_memory_stats()
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStart()
        replay_times.append(_cuda_event_elapsed(graph.replay))
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStop()
        replay_peaks.append(torch.cuda.max_memory_allocated() / 1e9)

    print("loss", float(nll.detach().cpu()))
    print(
        "normal_pi_backward_ms",
        "mean", f"{statistics.mean(normal_times):.3f}",
        "median", f"{statistics.median(normal_times):.3f}",
        "min", f"{min(normal_times):.3f}",
        "times", ",".join(f"{t:.3f}" for t in normal_times),
    )
    print(
        "cuda_graph_pi_backward_replay_ms",
        "mean", f"{statistics.mean(replay_times):.3f}",
        "median", f"{statistics.median(replay_times):.3f}",
        "min", f"{min(replay_times):.3f}",
        "times", ",".join(f"{t:.3f}" for t in replay_times),
    )
    if args.cuda_graph_check:
        print("cuda_graph_check", "max_output_abs_diff", f"{output_abs_diff:.8e}")
    print("normal_peak_alloc_gb", f"{max(normal_peaks):.3f}")
    print("cuda_graph_peak_alloc_gb", f"{max(replay_peaks):.3f}")


def _run_cuda_graph_bench(model: GeneReconModel, args: argparse.Namespace) -> None:
    if args.cuda_graph_target == "model":
        _run_cuda_graph_model_bench(model, args)
    elif args.cuda_graph_target == "pi_backward":
        _run_cuda_graph_pi_backward_bench(model, args)
    else:
        raise ValueError(f"unknown cuda graph target: {args.cuda_graph_target}")


def main() -> None:
    args = _parse_args()
    for key, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)
    _configure_cuda_graph_env(args)

    root = Path(args.dataset)
    genes = _selected_genes(root, args.start, args.fams)

    torch.cuda.empty_cache()
    gc.collect()

    t0 = time.perf_counter()
    model = GeneReconModel.from_trees(
        species_tree=str(root / "sp.nwk"),
        gene_trees=genes,
        mode="global",
        pibar_mode="uniform",
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_Pi=6,
        neumann_terms=3,
        use_pruning=args.use_pruning,
        pruning_threshold=args.pruning_threshold,
        preprocess_cache_dir=args.cache_dir,
        max_wave_size=args.max_wave_size,
        max_root_wave_size=args.max_root_wave_size,
    )
    torch.cuda.synchronize()

    print("build_s", f"{time.perf_counter() - t0:.3f}")
    print("dataset", root, "family_range", f"{args.start}:{args.start + args.fams}")
    print("use_pruning", args.use_pruning, "pruning_threshold", args.pruning_threshold)
    print("env_flags")
    for key in sorted(k for k in os.environ if k.startswith("GPUREC_")):
        print(key, os.environ[key])
    _print_wave_shape(model)

    if args.stats_only:
        return
    if args.diag_only:
        _run_pi_backward_diag(model, args)
        return
    if args.cuda_graph:
        _run_cuda_graph_bench(model, args)
        return

    for _ in range(args.warmups):
        model.zero_grad(set_to_none=True)
        loss = model()
        loss.backward()
        torch.cuda.synchronize()

    times = []
    peaks = []
    for _ in range(args.reps):
        model.zero_grad(set_to_none=True)
        loss = model()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStart()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStop()
        times.append(start.elapsed_time(end))
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)

    print("loss", float(loss.detach().cpu()))
    print(
        "backward_ms",
        "mean", f"{statistics.mean(times):.3f}",
        "median", f"{statistics.median(times):.3f}",
        "min", f"{min(times):.3f}",
        "times", ",".join(f"{t:.3f}" for t in times),
    )
    print("peak_alloc_gb", f"{max(peaks):.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Genewise uniform backward benchmark helper for Proposal 2 profiling.

This intentionally lives outside core code. It times one warmed backward pass
for the current genewise/uniform workload and prints static wave/family shape
metrics useful for evaluating the backward self-loop path.
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
    "GPUREC_FORWARD_LEAF_INDEX": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL": "tiled",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY": "1",
    "GPUREC_FORWARD_TOPOLOGY_INT32": "1",
    "GPUREC_KERNELIZED_BACKWARD_DTS": "1",
    "GPUREC_FUSED_DTS_BACKWARD_ACCUM": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL": "tree",
    "GPUREC_FUSED_UNIFORM_BACKWARD": "1",
    "GPUREC_UNIFORM_PINGPONG": "1",
    "GPUREC_BACKWARD_LEAF_INDEX": "1",
    "GPUREC_FUSED_WAVE_PARAM_ACCUM": "1",
    "GPUREC_DTS_PIBAR_UD_FUSION": "1",
    "GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES": "1",
    "GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS": "1",
    "GPUREC_DTS_GRAD_MT_TWO_STAGE": "1",
    "GPUREC_FORWARD_FAMILY_INDEXED_CONSTS": "1",
    "GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS": "1",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.getenv("DATASET", "tests/data/test_trees_1000"))
    parser.add_argument("--fams", type=int, default=int(os.getenv("FAMS", "50")))
    parser.add_argument("--start", type=int, default=int(os.getenv("FAMILY_START", "0")))
    parser.add_argument("--reps", type=int, default=int(os.getenv("REPS", "9")))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "5")))
    parser.add_argument(
        "--family-chunk-size",
        type=int,
        default=int(os.getenv("FAMILY_CHUNK_SIZE", "0")),
        help=(
            "Explicit autograd resident chunk size. 0 keeps the legacy "
            "single-model benchmark; 50 and 100 are the Proposal 5 first "
            "policies for test_trees_1000."
        ),
    )
    parser.add_argument("--max-wave-size", default=os.getenv("MAX_WAVE_SIZE", "32768"))
    parser.add_argument("--max-root-wave-size", type=int, default=None)
    parser.add_argument("--pruning-threshold", type=float, default=float(os.getenv("PRUNING_THRESHOLD", "1e-6")))
    parser.add_argument("--use-pruning", action=argparse.BooleanOptionalAction, default=os.getenv("USE_PRUNING", "1") != "0")
    parser.add_argument(
        "--backward-path",
        choices=("optimized-genewise", "generic-self-loop-fallback"),
        default=os.getenv("BACKWARD_PATH", "optimized-genewise"),
        help=(
            "Which genewise backward self-loop path to benchmark. "
            "'optimized-genewise' enables the family-indexed fused path and "
            "fails if the old generic self-loop is reached; "
            "'generic-self-loop-fallback' disables the fused self-loop for a "
            "reference run."
        ),
    )
    parser.add_argument("--stats-only", action="store_true", default=os.getenv("STATS_ONLY", "0") != "0")
    parser.add_argument("--diag-only", action="store_true", default=os.getenv("DIAG_ONLY", "0") != "0")
    parser.add_argument("--diag-chunk-rows", type=int, default=int(os.getenv("DIAG_CHUNK_ROWS", "256")))
    parser.add_argument("--profile-cuda-api", action="store_true", default=os.getenv("PROFILE_CUDA_API", "0") != "0")
    parser.add_argument("--cuda-graph", action="store_true", default=os.getenv("CUDA_GRAPH", "0") != "0")
    parser.add_argument(
        "--strict-optimized-kernels",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("STRICT_OPTIMIZED_KERNELS", "1") != "0",
        help=(
            "When benchmarking --backward-path=optimized-genewise, fail if "
            "Pi_wave_backward reaches generic self-loop, DTS, or Pibar VJP "
            "fallbacks."
        ),
    )
    parser.add_argument(
        "--empty-cache-between-chunks",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("EMPTY_CACHE_BETWEEN_CHUNKS", "1") != "0",
        help="Release the CUDA caching allocator between resident chunks.",
    )
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
    parser.add_argument(
        "--cuda-graph-profile-phase",
        choices=("both", "normal", "replay"),
        default=os.getenv("CUDA_GRAPH_PROFILE_PHASE", "both"),
        help=(
            "When PROFILE_CUDA_API=1 in CUDA graph mode, choose whether the "
            "cudaProfilerStart/Stop range wraps normal fixed-schedule timing, "
            "graph replay timing, or both."
        ),
    )
    parser.add_argument("--cache-dir", default=os.getenv("PREPROCESS_CACHE_DIR", "/tmp/gpurec_preprocess_cache"))
    args = parser.parse_args()
    mws = str(args.max_wave_size).strip().lower()
    args.max_wave_size = None if mws in ("", "0", "none", "null") else int(mws)
    return args


def _configure_backward_path_env(args: argparse.Namespace) -> None:
    if args.backward_path == "optimized-genewise":
        os.environ["GPUREC_FUSED_GENEWISE_BACKWARD"] = "1"
        os.environ["GPUREC_FUSED_GENEWISE_BACKWARD_SELF_LOOP"] = "1"
        os.environ["GPUREC_FUSED_UNIFORM_BACKWARD"] = "1"
        os.environ["GPUREC_BACKWARD_LEAF_INDEX"] = "1"
        os.environ["GPUREC_FORWARD_FAMILY_INDEXED_CONSTS"] = "1"
        os.environ["GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS"] = "1"
        os.environ["GPUREC_KERNELIZED_BACKWARD_DTS"] = "1"
        os.environ["GPUREC_FUSED_DTS_BACKWARD_ACCUM"] = "1"
        os.environ["GPUREC_FUSED_CROSS_PIBAR_VJP"] = "1"
        os.environ["GPUREC_DTS_PIBAR_UD_FUSION"] = "1"
        os.environ.setdefault("GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL", "tree")
        os.environ["GPUREC_REQUIRE_OPTIMIZED_GENEWISE_BACKWARD"] = (
            "1" if args.strict_optimized_kernels else "0"
        )
    elif args.backward_path == "generic-self-loop-fallback":
        os.environ["GPUREC_FUSED_GENEWISE_BACKWARD_SELF_LOOP"] = "0"
        os.environ["GPUREC_FUSED_UNIFORM_BACKWARD"] = "0"
        os.environ["GPUREC_REQUIRE_OPTIMIZED_GENEWISE_BACKWARD"] = "0"
    else:
        raise ValueError(f"unknown backward path: {args.backward_path}")


def _install_optimized_backward_guard(args: argparse.Namespace) -> None:
    if args.backward_path != "optimized-genewise":
        return

    from gpurec.core import backward as backward_core

    def blocked_generic_self_loop(*_args, **_kwargs):
        raise RuntimeError(
            "--backward-path=optimized-genewise reached the old generic "
            "self-loop path. The optimized genewise fused backward path is "
            "not active for this run."
        )

    backward_core._self_loop_vjp_precompute = blocked_generic_self_loop
    backward_core._gmres_self_loop_solve = blocked_generic_self_loop


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


def _profile_phase_enabled(args: argparse.Namespace, phase: str) -> bool:
    return bool(
        args.profile_cuda_api
        and args.cuda_graph_profile_phase in ("both", phase)
    )


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


def _tensor_max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.numel() == 0:
        return 0.0
    chunk_elems = int(os.getenv("CUDA_GRAPH_CHECK_CHUNK_ELEMS", str(16 * 1024 * 1024)))
    if chunk_elems <= 0 or a.numel() <= chunk_elems:
        return float((a - b).abs().max().detach().cpu())
    if not (a.is_contiguous() and b.is_contiguous()):
        return float((a - b).abs().max().detach().cpu())
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    scratch = torch.empty((min(chunk_elems, a.numel()),), device=a.device, dtype=a.dtype)
    max_diff = 0.0
    for start in range(0, a.numel(), chunk_elems):
        stop = min(start + chunk_elems, a.numel())
        span = stop - start
        tmp = scratch[:span]
        torch.sub(a_flat[start:stop], b_flat[start:stop], out=tmp)
        tmp.abs_()
        max_diff = max(max_diff, float(tmp.max().detach().cpu()))
    return max_diff


def _tensor_max_abs(a: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    chunk_elems = int(os.getenv("CUDA_GRAPH_CHECK_CHUNK_ELEMS", str(16 * 1024 * 1024)))
    if chunk_elems <= 0 or a.numel() <= chunk_elems or not a.is_contiguous():
        return float(a.abs().max().detach().cpu())
    a_flat = a.view(-1)
    scratch = torch.empty((min(chunk_elems, a.numel()),), device=a.device, dtype=a.dtype)
    max_abs = 0.0
    for start in range(0, a.numel(), chunk_elems):
        stop = min(start + chunk_elems, a.numel())
        span = stop - start
        tmp = scratch[:span]
        torch.abs(a_flat[start:stop], out=tmp)
        max_abs = max(max_abs, float(tmp.max().detach().cpu()))
    return max_abs


def _max_tensor_dict_abs_diff(
    lhs: dict,
    rhs: dict,
    *,
    keys: tuple[str, ...],
) -> tuple[float, str, float]:
    max_diff = 0.0
    max_key = ""
    max_ref = 0.0
    for key in keys:
        a = lhs.get(key)
        b = rhs.get(key)
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        diff = _tensor_max_abs_diff(a, b)
        if diff >= max_diff:
            max_diff = diff
            max_key = key
            max_ref = _tensor_max_abs(a)
    denom = max(max_ref, 1.0)
    return max_diff, max_key, max_diff / denom


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


def _wave_family_locality(model: GeneReconModel) -> tuple[list[dict[str, int | float]], dict[str, int | float]]:
    wl = model.static.wave_layout
    metas = wl["wave_metas"]
    family_idx = wl.get("family_idx")
    if family_idx is not None:
        family_idx = family_idx.detach().cpu()

    rows: list[dict[str, int | float]] = []
    totals: dict[str, int | float] = {
        "rows": 0,
        "family_runs": 0,
        "family_touches": 0,
        "extra_family_runs": 0,
        "interleaved_waves": 0,
        "split_rows": 0,
        "split_family_touches": 0,
        "locality_preserved": 1,
        "run_per_row": 0.0,
        "runs_per_family_touch": 0.0,
        "split_family_touch_per_split_row": 0.0,
    }

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

        extra_runs = max(0, runs - fams)
        rows.append({
            "k": k,
            "start": ws,
            "W": w,
            "split_rows": n_splits,
            "fanout": fanout,
            "families": fams,
            "family_runs": runs,
            "split_families": split_fams,
            "extra_family_runs": extra_runs,
        })

        totals["rows"] = int(totals["rows"]) + w
        totals["family_runs"] = int(totals["family_runs"]) + runs
        totals["family_touches"] = int(totals["family_touches"]) + fams
        totals["extra_family_runs"] = int(totals["extra_family_runs"]) + extra_runs
        totals["interleaved_waves"] = int(totals["interleaved_waves"]) + int(extra_runs > 0)
        totals["split_rows"] = int(totals["split_rows"]) + n_splits
        totals["split_family_touches"] = int(totals["split_family_touches"]) + split_fams

    total_rows = int(totals["rows"])
    family_touches = int(totals["family_touches"])
    split_rows = int(totals["split_rows"])
    totals["run_per_row"] = (int(totals["family_runs"]) / total_rows) if total_rows else 0.0
    totals["runs_per_family_touch"] = (
        int(totals["family_runs"]) / family_touches
    ) if family_touches else 0.0
    totals["split_family_touch_per_split_row"] = (
        int(totals["split_family_touches"]) / split_rows
    ) if split_rows else 0.0
    totals["locality_preserved"] = int(int(totals["extra_family_runs"]) == 0)

    return rows, totals


def _chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= total:
        return [(0, total)]
    return [
        (start, min(start + chunk_size, total))
        for start in range(0, total, chunk_size)
    ]


def _wave_shape_stats(model: GeneReconModel) -> dict[str, int]:
    wl = model.static.wave_layout
    metas = wl["wave_metas"]
    split_rows = sum(
        int(m["sl"].numel()) if m.get("has_splits", False) else 0
        for m in metas
    )
    _, locality = _wave_family_locality(model)
    return {
        "S": int(model.n_species),
        "G": int(model.n_families),
        "C": int(model.static.Pi_shape[0]) if hasattr(model.static, "Pi_shape") else int(sum(m["W"] for m in metas)),
        "waves": len(metas),
        "maxW": max((int(m["W"]) for m in metas), default=0),
        "split_rows": split_rows,
        "leaves": int(wl["leaf_row_index"].numel()),
        "roots": int(model.static.root_clade_ids.numel()),
        "family_runs": int(locality["family_runs"]),
        "family_touches": int(locality["family_touches"]),
        "extra_family_runs": int(locality["extra_family_runs"]),
        "interleaved_waves": int(locality["interleaved_waves"]),
        "locality_preserved": int(locality["locality_preserved"]),
    }


def _print_wave_shape(model: GeneReconModel) -> None:
    wl = model.static.wave_layout
    metas = wl["wave_metas"]

    stats = _wave_shape_stats(model)
    print(
        "shape",
        "S", stats["S"],
        "G", stats["G"],
        "C", stats["C"],
        "waves", stats["waves"],
        "maxW", stats["maxW"],
        "split_rows", stats["split_rows"],
        "leaves", stats["leaves"],
        "roots", stats["roots"],
    )

    rows, locality = _wave_family_locality(model)
    print(
        "family_locality_summary",
        "rows", int(locality["rows"]),
        "family_runs", int(locality["family_runs"]),
        "family_touches", int(locality["family_touches"]),
        "extra_family_runs", int(locality["extra_family_runs"]),
        "interleaved_waves", int(locality["interleaved_waves"]),
        "locality_preserved", int(locality["locality_preserved"]),
        "run_per_row", f"{float(locality['run_per_row']):.8f}",
        "runs_per_family_touch", f"{float(locality['runs_per_family_touch']):.8f}",
        "split_rows", int(locality["split_rows"]),
        "split_family_touches", int(locality["split_family_touches"]),
        "split_family_touch_per_split_row", f"{float(locality['split_family_touch_per_split_row']):.8f}",
    )

    print("top_wave_rows k start W split_rows fanout families family_runs split_families")
    for row in sorted(rows, key=lambda r: int(r["W"]), reverse=True)[:12]:
        print(
            "%d %d %d %d %.3f %d %d %d" % (
                int(row["k"]),
                int(row["start"]),
                int(row["W"]),
                int(row["split_rows"]),
                float(row["fanout"]),
                int(row["families"]),
                int(row["family_runs"]),
                int(row["split_families"]),
            )
        )

    print("top_split_rows k start W split_rows fanout families family_runs split_families")
    for row in sorted(rows, key=lambda r: int(r["split_rows"]), reverse=True)[:12]:
        print(
            "%d %d %d %d %.3f %d %d %d" % (
                int(row["k"]),
                int(row["start"]),
                int(row["W"]),
                int(row["split_rows"]),
                float(row["fanout"]),
                int(row["families"]),
                int(row["family_runs"]),
                int(row["split_families"]),
            )
        )


def _print_active_path_flags(args: argparse.Namespace, model: GeneReconModel) -> None:
    wl = model.static.wave_layout
    print(
        "active_path_flags",
        "mode", "genewise",
        "pibar_mode", model.static.pibar_mode,
        "fixed_iters_Pi", model.static.fixed_iters_Pi,
        "S", model.n_species,
        "family_idx", int(wl.get("family_idx") is not None),
        "leaf_index", os.environ.get("GPUREC_BACKWARD_LEAF_INDEX", "unset"),
        "forward_family_consts", os.environ.get("GPUREC_FORWARD_FAMILY_INDEXED_CONSTS", "unset"),
        "forward_family_dts_params", os.environ.get("GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS", "unset"),
        "fused_genewise_backward", os.environ.get("GPUREC_FUSED_GENEWISE_BACKWARD", "unset"),
        "fused_genewise_self_loop", os.environ.get("GPUREC_FUSED_GENEWISE_BACKWARD_SELF_LOOP", "unset"),
        "fused_uniform_backward", os.environ.get("GPUREC_FUSED_UNIFORM_BACKWARD", "unset"),
        "kernelized_backward_dts", os.environ.get("GPUREC_KERNELIZED_BACKWARD_DTS", "unset"),
        "fused_dts_backward_accum", os.environ.get("GPUREC_FUSED_DTS_BACKWARD_ACCUM", "unset"),
        "dts_pibar_ud_fusion", os.environ.get("GPUREC_DTS_PIBAR_UD_FUSION", "unset"),
        "fused_cross_pibar_vjp", os.environ.get("GPUREC_FUSED_CROSS_PIBAR_VJP", "unset"),
        "cross_pibar_impl", os.environ.get("GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL", "unset"),
        "strict_optimized_kernels", int(args.strict_optimized_kernels),
        "require_optimized_guard", os.environ.get("GPUREC_REQUIRE_OPTIMIZED_GENEWISE_BACKWARD", "unset"),
    )


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
    root_clade_ids = static.wave_layout["root_clade_ids"]
    if torch.is_tensor(root_clade_ids):
        root_clade_ids = [int(r) for r in root_clade_ids.detach().cpu().tolist()]
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
        "root_clade_ids_perm": root_clade_ids,
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
        if _profile_phase_enabled(args, "normal"):
            torch.cuda.cudart().cudaProfilerStart()

        def backward_only() -> None:
            loss.backward()

        normal_times.append(_cuda_event_elapsed(backward_only))
        if _profile_phase_enabled(args, "normal"):
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
        if _profile_phase_enabled(args, "replay"):
            torch.cuda.cudart().cudaProfilerStart()
        replay_times.append(_cuda_event_elapsed(graph.replay))
        if _profile_phase_enabled(args, "replay"):
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
        if _profile_phase_enabled(args, "normal"):
            torch.cuda.cudart().cudaProfilerStart()
        holder = {}

        def direct_backward() -> None:
            holder["out"] = Pi_wave_backward(**kwargs)

        normal_times.append(_cuda_event_elapsed(direct_backward))
        if _profile_phase_enabled(args, "normal"):
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
    output_abs_diff, output_abs_diff_key, output_rel_diff = _max_tensor_dict_abs_diff(
        baseline_bwd,
        graph_bwd,
        keys=tensor_keys,
    )

    replay_times = []
    replay_peaks = []
    for _ in range(args.reps):
        torch.cuda.reset_peak_memory_stats()
        if _profile_phase_enabled(args, "replay"):
            torch.cuda.cudart().cudaProfilerStart()
        replay_times.append(_cuda_event_elapsed(graph.replay))
        if _profile_phase_enabled(args, "replay"):
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
        print(
            "cuda_graph_check",
            "max_output_abs_diff", f"{output_abs_diff:.8e}",
            "key", output_abs_diff_key,
            "rel_to_key_max", f"{output_rel_diff:.8e}",
        )
    print("normal_peak_alloc_gb", f"{max(normal_peaks):.3f}")
    print("cuda_graph_peak_alloc_gb", f"{max(replay_peaks):.3f}")


def _run_cuda_graph_bench(model: GeneReconModel, args: argparse.Namespace) -> None:
    if args.cuda_graph_target == "model":
        _run_cuda_graph_model_bench(model, args)
    elif args.cuda_graph_target == "pi_backward":
        _run_cuda_graph_pi_backward_bench(model, args)
    else:
        raise ValueError(f"unknown cuda graph target: {args.cuda_graph_target}")


def _make_genewise_model(
    args: argparse.Namespace,
    root: Path,
    genes: list[str],
) -> GeneReconModel:
    return GeneReconModel.from_trees(
        species_tree=str(root / "sp.nwk"),
        gene_trees=genes,
        mode="genewise",
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


def _time_cuda_event(fn) -> tuple[float, object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _run_chunked_autograd_bench(
    args: argparse.Namespace,
    root: Path,
    genes: list[str],
) -> None:
    if args.cuda_graph:
        raise ValueError("--cuda-graph is only supported by the legacy single-model benchmark")

    ranges = _chunk_ranges(len(genes), args.family_chunk_size)
    print(
        "chunked_autograd_policy",
        "dataset", root,
        "family_range", f"{args.start}:{args.start + args.fams}",
        "families", len(genes),
        "chunks", len(ranges),
        "family_chunk_size", args.family_chunk_size,
        "backward_path", args.backward_path,
        "use_pruning", args.use_pruning,
        "pruning_threshold", args.pruning_threshold,
        "max_wave_size", args.max_wave_size if args.max_wave_size is not None else "none",
        "max_root_wave_size", args.max_root_wave_size if args.max_root_wave_size is not None else "none",
    )
    print("env_flags")
    for key in sorted(k for k in os.environ if k.startswith("GPUREC_")):
        print(key, os.environ[key])

    first_model = _make_genewise_model(args, root, genes[ranges[0][0]:ranges[0][1]])
    _print_active_path_flags(args, first_model)
    print(
        "chunk_table idx family_start family_stop families S C waves maxW "
        "split_rows leaves roots family_runs family_touches extra_family_runs "
        "interleaved_waves locality_preserved run_per_row runs_per_family_touch"
    )
    first_stats = _wave_shape_stats(first_model)
    _, first_locality = _wave_family_locality(first_model)
    print(
        "chunk_shape",
        0,
        args.start + ranges[0][0],
        args.start + ranges[0][1],
        first_stats["G"],
        first_stats["S"],
        first_stats["C"],
        first_stats["waves"],
        first_stats["maxW"],
        first_stats["split_rows"],
        first_stats["leaves"],
        first_stats["roots"],
        first_stats["family_runs"],
        first_stats["family_touches"],
        first_stats["extra_family_runs"],
        first_stats["interleaved_waves"],
        first_stats["locality_preserved"],
        f"{float(first_locality['run_per_row']):.8f}",
        f"{float(first_locality['runs_per_family_touch']):.8f}",
    )
    del first_model
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    if args.stats_only:
        for chunk_idx, (lo, hi) in enumerate(ranges[1:], start=1):
            model = _make_genewise_model(args, root, genes[lo:hi])
            stats = _wave_shape_stats(model)
            _, locality = _wave_family_locality(model)
            print(
                "chunk_shape",
                chunk_idx,
                args.start + lo,
                args.start + hi,
                stats["G"],
                stats["S"],
                stats["C"],
                stats["waves"],
                stats["maxW"],
                stats["split_rows"],
                stats["leaves"],
                stats["roots"],
                stats["family_runs"],
                stats["family_touches"],
                stats["extra_family_runs"],
                stats["interleaved_waves"],
                stats["locality_preserved"],
                f"{float(locality['run_per_row']):.8f}",
                f"{float(locality['runs_per_family_touch']):.8f}",
            )
            del model
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        return

    def run_pass(*, timed: bool) -> dict[str, object]:
        chunk_rows = []
        total_forward_ms = 0.0
        total_backward_ms = 0.0
        total_loss = 0.0
        max_forward_peak_gb = 0.0
        max_backward_peak_gb = 0.0
        grad_chunks = []
        build_s_total = 0.0

        for chunk_idx, (lo, hi) in enumerate(ranges):
            t0 = time.perf_counter()
            model = _make_genewise_model(args, root, genes[lo:hi])
            torch.cuda.synchronize()
            build_s = time.perf_counter() - t0
            build_s_total += build_s
            stats = _wave_shape_stats(model)
            _, locality = _wave_family_locality(model)
            if timed and chunk_idx > 0:
                print(
                    "chunk_shape",
                    chunk_idx,
                    args.start + lo,
                    args.start + hi,
                    stats["G"],
                    stats["S"],
                    stats["C"],
                    stats["waves"],
                    stats["maxW"],
                    stats["split_rows"],
                    stats["leaves"],
                    stats["roots"],
                    stats["family_runs"],
                    stats["family_touches"],
                    stats["extra_family_runs"],
                    stats["interleaved_waves"],
                    stats["locality_preserved"],
                    f"{float(locality['run_per_row']):.8f}",
                    f"{float(locality['runs_per_family_touch']):.8f}",
                )

            model.zero_grad(set_to_none=True)
            model.static.warm_E = None
            torch.cuda.reset_peak_memory_stats()
            forward_ms, loss = _time_cuda_event(model)
            forward_peak_gb = torch.cuda.max_memory_allocated() / 1e9

            backward_ms, _ = _time_cuda_event(loss.backward)
            backward_peak_gb = torch.cuda.max_memory_allocated() / 1e9

            loss_value = float(loss.detach().cpu())
            total_loss += loss_value
            if model.theta.grad is None:
                raise RuntimeError("theta.grad was not populated for chunked autograd")
            grad_chunk = model.theta.grad.detach().cpu().clone()
            if not torch.isfinite(grad_chunk).all():
                raise FloatingPointError(f"non-finite theta gradient in chunk {chunk_idx}")
            grad_chunks.append(grad_chunk)

            total_forward_ms += forward_ms
            total_backward_ms += backward_ms
            max_forward_peak_gb = max(max_forward_peak_gb, forward_peak_gb)
            max_backward_peak_gb = max(max_backward_peak_gb, backward_peak_gb)
            chunk_rows.append({
                "idx": chunk_idx,
                "lo": args.start + lo,
                "hi": args.start + hi,
                "families": hi - lo,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "forward_peak_gb": forward_peak_gb,
                "backward_peak_gb": backward_peak_gb,
                "loss": loss_value,
                "build_s": build_s,
                "run_per_row": float(locality["run_per_row"]),
                "runs_per_family_touch": float(locality["runs_per_family_touch"]),
                **stats,
            })

            del loss, model
            if args.empty_cache_between_chunks:
                gc.collect()
                torch.cuda.empty_cache()

        grad = torch.cat(grad_chunks, dim=0) if grad_chunks else torch.empty(0)
        return {
            "chunks": chunk_rows,
            "forward_ms": total_forward_ms,
            "backward_ms": total_backward_ms,
            "loss": total_loss,
            "forward_peak_gb": max_forward_peak_gb,
            "backward_peak_gb": max_backward_peak_gb,
            "grad_absmax": float(grad.abs().max()) if grad.numel() else 0.0,
            "build_s": build_s_total,
        }

    for _ in range(args.warmups):
        run_pass(timed=False)

    reps = []
    for rep in range(args.reps):
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStart()
        result = run_pass(timed=True)
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStop()
        reps.append(result)
        print(
            "chunked_rep",
            rep,
            "forward_ms", f"{result['forward_ms']:.3f}",
            "backward_ms", f"{result['backward_ms']:.3f}",
            "total_ms", f"{(result['forward_ms'] + result['backward_ms']):.3f}",
            "loss", f"{result['loss']:.8f}",
            "max_forward_peak_gb", f"{result['forward_peak_gb']:.3f}",
            "max_backward_peak_gb", f"{result['backward_peak_gb']:.3f}",
            "grad_absmax", f"{result['grad_absmax']:.8e}",
            "build_s", f"{result['build_s']:.3f}",
        )
        print(
            "chunk_timing_table rep idx family_start family_stop families C waves "
            "maxW split_rows family_runs family_touches extra_family_runs "
            "interleaved_waves locality_preserved run_per_row "
            "runs_per_family_touch forward_ms backward_ms forward_peak_gb "
            "backward_peak_gb loss build_s"
        )
        for row in result["chunks"]:
            print(
                "chunk_timing",
                rep,
                row["idx"],
                row["lo"],
                row["hi"],
                row["families"],
                row["C"],
                row["waves"],
                row["maxW"],
                row["split_rows"],
                row["family_runs"],
                row["family_touches"],
                row["extra_family_runs"],
                row["interleaved_waves"],
                row["locality_preserved"],
                f"{row['run_per_row']:.8f}",
                f"{row['runs_per_family_touch']:.8f}",
                f"{row['forward_ms']:.3f}",
                f"{row['backward_ms']:.3f}",
                f"{row['forward_peak_gb']:.3f}",
                f"{row['backward_peak_gb']:.3f}",
                f"{row['loss']:.8f}",
                f"{row['build_s']:.3f}",
            )

    forward_times = [float(r["forward_ms"]) for r in reps]
    backward_times = [float(r["backward_ms"]) for r in reps]
    total_times = [f + b for f, b in zip(forward_times, backward_times)]
    print(
        "chunked_summary",
        "reps", len(reps),
        "chunks", len(ranges),
        "family_chunk_size", args.family_chunk_size,
        "forward_median_ms", f"{statistics.median(forward_times):.3f}",
        "forward_mean_ms", f"{statistics.mean(forward_times):.3f}",
        "backward_median_ms", f"{statistics.median(backward_times):.3f}",
        "backward_mean_ms", f"{statistics.mean(backward_times):.3f}",
        "total_median_ms", f"{statistics.median(total_times):.3f}",
        "total_mean_ms", f"{statistics.mean(total_times):.3f}",
        "max_backward_peak_gb", f"{max(float(r['backward_peak_gb']) for r in reps):.3f}",
        "loss_last", f"{float(reps[-1]['loss']):.8f}",
    )


def main() -> None:
    args = _parse_args()
    for key, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)
    _configure_backward_path_env(args)
    _configure_cuda_graph_env(args)

    root = Path(args.dataset)
    genes = _selected_genes(root, args.start, args.fams)

    torch.cuda.empty_cache()
    gc.collect()

    if args.family_chunk_size > 0:
        _install_optimized_backward_guard(args)
        _run_chunked_autograd_bench(args, root, genes)
        return

    t0 = time.perf_counter()
    model = _make_genewise_model(args, root, genes)
    torch.cuda.synchronize()

    print("build_s", f"{time.perf_counter() - t0:.3f}")
    print("dataset", root, "family_range", f"{args.start}:{args.start + args.fams}")
    print("backward_path", args.backward_path)
    print("use_pruning", args.use_pruning, "pruning_threshold", args.pruning_threshold)
    print("env_flags")
    for key in sorted(k for k in os.environ if k.startswith("GPUREC_")):
        print(key, os.environ[key])
    _print_active_path_flags(args, model)
    _print_wave_shape(model)

    if args.stats_only:
        return
    _install_optimized_backward_guard(args)
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

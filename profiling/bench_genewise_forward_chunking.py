#!/usr/bin/env python3
"""Genewise uniform forward profiling harness.

This entrypoint times the current optimized genewise/uniform inference path:
``GeneReconModel`` in no-grad mode, root-row likelihood output, family-indexed
forward constants, family-indexed DTS parameters, and no backward saved
tensors.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

import torch

from gpurec import GeneReconModel


DEFAULT_FLAGS = {
    "GPUREC_FORWARD_LEAF_INDEX": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL": "tiled",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY": "1",
    "GPUREC_FORWARD_TOPOLOGY_INT32": "1",
    "GPUREC_FORWARD_FAMILY_INDEXED_CONSTS": "1",
    "GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS": "1",
    "GPUREC_UNIFORM_PINGPONG": "1",
}

STRICT_FORWARD_FLAGS = (
    "GPUREC_FORWARD_LEAF_INDEX",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY",
    "GPUREC_FORWARD_FAMILY_INDEXED_CONSTS",
    "GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS",
    "GPUREC_UNIFORM_PINGPONG",
)


def _parse_optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "0", "none", "null"):
        return None
    return int(text)


def _parse_dtype(value: str) -> torch.dtype:
    text = value.strip().lower()
    if text in ("float32", "fp32", "f32"):
        return torch.float32
    if text in ("float64", "fp64", "f64", "double"):
        return torch.float64
    raise ValueError(f"unsupported dtype {value!r}; expected fp32 or fp64")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.getenv("DATASET", "tests/data/test_trees_1000"))
    parser.add_argument(
        "--family-start",
        "--start",
        dest="family_start",
        type=int,
        default=int(os.getenv("FAMILY_START", "0")),
    )
    parser.add_argument("--family-stop", type=int, default=None)
    parser.add_argument("--fams", type=int, default=int(os.getenv("FAMS", "50")))
    parser.add_argument("--family-chunk-size", type=int, default=int(os.getenv("FAMILY_CHUNK_SIZE", "150")))
    parser.add_argument("--max-wave-size", default=os.getenv("MAX_WAVE_SIZE", "32768"))
    parser.add_argument("--max-root-wave-size", default=os.getenv("MAX_ROOT_WAVE_SIZE", ""))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "3")))
    parser.add_argument("--reps", type=int, default=int(os.getenv("REPS", "7")))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "fp32"))
    parser.add_argument("--device", default=os.getenv("DEVICE", "cuda"))
    parser.add_argument("--stats-only", action="store_true", default=os.getenv("STATS_ONLY", "0") != "0")
    parser.add_argument(
        "--strict-optimized-kernels",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("STRICT_OPTIMIZED_KERNELS", "1") != "0",
        help="Fail if the run is not using the optimized genewise uniform inference path.",
    )
    parser.add_argument("--cache-dir", default=os.getenv("PREPROCESS_CACHE_DIR", "/tmp/gpurec_preprocess_cache"))
    parser.add_argument(
        "--empty-cache-between-chunks",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("EMPTY_CACHE_BETWEEN_CHUNKS", "1") != "0",
    )
    args = parser.parse_args()
    args.max_wave_size = _parse_optional_int(args.max_wave_size)
    args.max_root_wave_size = _parse_optional_int(args.max_root_wave_size)
    args.dtype = _parse_dtype(args.dtype)
    if args.family_stop is not None:
        if args.family_stop <= args.family_start:
            raise ValueError("--family-stop must be greater than --family-start")
        args.fams = args.family_stop - args.family_start
    if args.fams <= 0:
        raise ValueError("--fams must be positive")
    if args.family_chunk_size < 0:
        raise ValueError("--family-chunk-size must be non-negative")
    return args


def _selected_genes(root: Path, start: int, fams: int) -> list[str]:
    genes = sorted(root.glob("g_*.nwk"))
    stop = start + fams
    if stop > len(genes):
        raise ValueError(f"requested families [{start}:{stop}], but only found {len(genes)} genes in {root}")
    return [str(p) for p in genes[start:stop]]


def _chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= total:
        return [(0, total)]
    return [
        (start, min(start + chunk_size, total))
        for start in range(0, total, chunk_size)
    ]


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
        "C": int(sum(int(m["W"]) for m in metas)),
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
        device=args.device,
        dtype=args.dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_Pi=6,
        use_pruning=False,
        preprocess_cache_dir=args.cache_dir,
        max_wave_size=args.max_wave_size,
        max_root_wave_size=args.max_root_wave_size,
    )


def _optimized_forward_reasons(args: argparse.Namespace, model: GeneReconModel | None = None) -> list[str]:
    reasons: list[str] = []
    device = torch.device(args.device)
    if device.type != "cuda":
        reasons.append(f"device={device.type}, expected cuda")
    for key in STRICT_FORWARD_FLAGS:
        if os.environ.get(key, "0") != "1":
            reasons.append(f"{key}={os.environ.get(key, 'unset')}, expected 1")
    if os.environ.get("GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL", "unset") != "tiled":
        reasons.append(
            "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL="
            f"{os.environ.get('GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL', 'unset')}, expected tiled"
        )
    if model is not None:
        if not bool(model.static.genewise):
            reasons.append("mode is not genewise")
        if model.static.pibar_mode != "uniform":
            reasons.append(f"pibar_mode={model.static.pibar_mode}, expected uniform")
        if model.static.fixed_iters_Pi != 6:
            reasons.append(f"fixed_iters_Pi={model.static.fixed_iters_Pi}, expected 6")
        if model.static.wave_layout.get("family_idx") is None:
            reasons.append("family_idx missing from wave layout")
    return reasons


def _handle_optimized_status(args: argparse.Namespace, model: GeneReconModel | None = None) -> bool:
    reasons = _optimized_forward_reasons(args, model)
    optimized = not reasons
    print(
        "optimized_forward_status",
        "optimized", int(optimized),
        "strict", int(args.strict_optimized_kernels),
        "reason", "none" if optimized else ";".join(reasons),
    )
    if reasons and args.strict_optimized_kernels:
        raise RuntimeError("non-optimized genewise forward harness configuration: " + "; ".join(reasons))
    return optimized


def _print_active_path_flags(args: argparse.Namespace, model: GeneReconModel) -> None:
    wl = model.static.wave_layout
    print(
        "active_path_flags",
        "mode", "genewise",
        "pibar_mode", model.static.pibar_mode,
        "fixed_iters_Pi", model.static.fixed_iters_Pi,
        "S", model.n_species,
        "root_row_output", 1,
        "return_root_rows", 1,
        "need_pibar", 0,
        "full_saved_tensors", 0,
        "backward_saved_tensors", 0,
        "forward_only_no_grad", 1,
        "family_idx", int(wl.get("family_idx") is not None),
        "leaf_index", os.environ.get("GPUREC_FORWARD_LEAF_INDEX", "unset"),
        "parent_reduced_dts", os.environ.get("GPUREC_FORWARD_PARENT_REDUCED_DTS", "unset"),
        "parent_reduced_dts_impl", os.environ.get("GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL", "unset"),
        "parent_reduced_dts_ge2_only", os.environ.get("GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY", "unset"),
        "family_indexed_constants", os.environ.get("GPUREC_FORWARD_FAMILY_INDEXED_CONSTS", "unset"),
        "family_indexed_dts_params", os.environ.get("GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS", "unset"),
        "uniform_pingpong", os.environ.get("GPUREC_UNIFORM_PINGPONG", "unset"),
        "fused_uniform_backward", os.environ.get("GPUREC_FUSED_UNIFORM_BACKWARD", "unset"),
        "generic_pytorch_fallback", 0 if not _optimized_forward_reasons(args, model) else 1,
        "strict_optimized_kernels", int(args.strict_optimized_kernels),
    )


def _time_forward(model: GeneReconModel) -> tuple[float, torch.Tensor]:
    model.static.warm_E = None
    if model.static.device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            out = model()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end), out

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model()
    return (time.perf_counter() - t0) * 1000.0, out


def _memory_peak_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e9


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _print_chunk_table_header() -> None:
    print(
        "chunk_table idx family_start family_stop families S C waves maxW "
        "split_rows leaves roots family_runs family_touches extra_family_runs "
        "interleaved_waves locality_preserved run_per_row runs_per_family_touch"
    )


def _print_chunk_shape(
    idx: int,
    family_start: int,
    family_stop: int,
    model: GeneReconModel,
) -> None:
    stats = _wave_shape_stats(model)
    _, locality = _wave_family_locality(model)
    print(
        "chunk_shape",
        idx,
        family_start,
        family_stop,
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


def _run_resident(args: argparse.Namespace, root: Path, genes: list[str]) -> None:
    t0 = time.perf_counter()
    model = _make_genewise_model(args, root, genes)
    _sync(model.static.device)
    build_s = time.perf_counter() - t0

    print("build_s", f"{build_s:.3f}")
    _print_active_path_flags(args, model)
    _handle_optimized_status(args, model)
    _print_wave_shape(model)
    if args.stats_only:
        return

    for _ in range(args.warmups):
        ms, out = _time_forward(model)
        del ms, out

    times: list[float] = []
    peaks: list[float] = []
    nll_value = float("nan")
    for rep in range(args.reps):
        _reset_peak_memory(model.static.device)
        ms, out = _time_forward(model)
        nll_value = float(out.detach().cpu())
        peak_gb = _memory_peak_gb(model.static.device)
        times.append(ms)
        peaks.append(peak_gb)
        print(
            "resident_rep",
            rep,
            "forward_ms", f"{ms:.3f}",
            "peak_gb", f"{peak_gb:.3f}",
            "nll", f"{nll_value:.8f}",
        )
        del out

    print(
        "resident_summary",
        "reps", len(times),
        "forward_median_ms", f"{statistics.median(times):.3f}",
        "forward_mean_ms", f"{statistics.mean(times):.3f}",
        "forward_min_ms", f"{min(times):.3f}",
        "peak_gb", f"{max(peaks):.3f}",
        "nll_last", f"{nll_value:.8f}",
    )


def _run_chunked(args: argparse.Namespace, root: Path, genes: list[str]) -> None:
    ranges = _chunk_ranges(len(genes), args.family_chunk_size)
    print(
        "chunked_forward_policy",
        "dataset", root,
        "family_range", f"{args.family_start}:{args.family_start + args.fams}",
        "families", len(genes),
        "chunks", len(ranges),
        "family_chunk_size", args.family_chunk_size,
        "max_wave_size", args.max_wave_size if args.max_wave_size is not None else "none",
        "max_root_wave_size", args.max_root_wave_size if args.max_root_wave_size is not None else "none",
    )

    first_model = _make_genewise_model(args, root, genes[ranges[0][0]:ranges[0][1]])
    _print_active_path_flags(args, first_model)
    _handle_optimized_status(args, first_model)
    _print_chunk_table_header()
    _print_chunk_shape(0, args.family_start + ranges[0][0], args.family_start + ranges[0][1], first_model)
    del first_model
    gc.collect()
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()

    if args.stats_only:
        for chunk_idx, (lo, hi) in enumerate(ranges[1:], start=1):
            model = _make_genewise_model(args, root, genes[lo:hi])
            _print_chunk_shape(chunk_idx, args.family_start + lo, args.family_start + hi, model)
            del model
            gc.collect()
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
        return

    def run_pass(*, timed: bool) -> dict[str, object]:
        total_forward_ms = 0.0
        total_loss = 0.0
        max_peak_gb = 0.0
        build_s_total = 0.0
        chunk_rows = []
        for chunk_idx, (lo, hi) in enumerate(ranges):
            t0 = time.perf_counter()
            model = _make_genewise_model(args, root, genes[lo:hi])
            _sync(model.static.device)
            build_s = time.perf_counter() - t0
            build_s_total += build_s
            if timed and chunk_idx > 0:
                _print_chunk_shape(chunk_idx, args.family_start + lo, args.family_start + hi, model)

            _reset_peak_memory(model.static.device)
            forward_ms, out = _time_forward(model)
            peak_gb = _memory_peak_gb(model.static.device)
            loss_value = float(out.detach().cpu())
            total_forward_ms += forward_ms
            total_loss += loss_value
            max_peak_gb = max(max_peak_gb, peak_gb)
            stats = _wave_shape_stats(model)
            _, locality = _wave_family_locality(model)
            chunk_rows.append({
                "idx": chunk_idx,
                "lo": args.family_start + lo,
                "hi": args.family_start + hi,
                "families": hi - lo,
                "forward_ms": forward_ms,
                "peak_gb": peak_gb,
                "loss": loss_value,
                "build_s": build_s,
                "run_per_row": float(locality["run_per_row"]),
                "runs_per_family_touch": float(locality["runs_per_family_touch"]),
                **stats,
            })
            del out, model
            gc.collect()
            if args.empty_cache_between_chunks and torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()

        return {
            "chunks": chunk_rows,
            "forward_ms": total_forward_ms,
            "loss": total_loss,
            "peak_gb": max_peak_gb,
            "build_s": build_s_total,
        }

    for _ in range(args.warmups):
        run_pass(timed=False)

    reps = []
    for rep in range(args.reps):
        result = run_pass(timed=True)
        reps.append(result)
        print(
            "chunked_rep",
            rep,
            "forward_ms", f"{float(result['forward_ms']):.3f}",
            "loss", f"{float(result['loss']):.8f}",
            "peak_gb", f"{float(result['peak_gb']):.3f}",
            "build_s", f"{float(result['build_s']):.3f}",
        )
        print(
            "chunk_timing_table rep idx family_start family_stop families C waves "
            "maxW split_rows family_runs family_touches extra_family_runs "
            "interleaved_waves locality_preserved run_per_row "
            "runs_per_family_touch forward_ms peak_gb loss build_s"
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
                f"{row['peak_gb']:.3f}",
                f"{row['loss']:.8f}",
                f"{row['build_s']:.3f}",
            )

    forward_times = [float(r["forward_ms"]) for r in reps]
    print(
        "chunked_summary",
        "reps", len(reps),
        "chunks", len(ranges),
        "family_chunk_size", args.family_chunk_size,
        "forward_median_ms", f"{statistics.median(forward_times):.3f}",
        "forward_mean_ms", f"{statistics.mean(forward_times):.3f}",
        "forward_min_ms", f"{min(forward_times):.3f}",
        "peak_gb", f"{max(float(r['peak_gb']) for r in reps):.3f}",
        "loss_last", f"{float(reps[-1]['loss']):.8f}",
    )


def main() -> None:
    args = _parse_args()
    for key, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)

    root = Path(args.dataset)
    genes = _selected_genes(root, args.family_start, args.fams)

    print(
        "policy",
        "dataset", root,
        "family_range", f"{args.family_start}:{args.family_start + args.fams}",
        "families", len(genes),
        "family_chunk_size", args.family_chunk_size,
        "device", args.device,
        "dtype", str(args.dtype).replace("torch.", ""),
        "mode", "genewise",
        "pibar_mode", "uniform",
        "fixed_iters_Pi", 6,
        "root_row_output", 1,
        "full_saved_tensors", 0,
        "backward_saved_tensors", 0,
        "stats_only", int(args.stats_only),
        "strict_optimized_kernels", int(args.strict_optimized_kernels),
    )
    print("env_flags")
    for key in sorted(k for k in os.environ if k.startswith("GPUREC_")):
        print(key, os.environ[key])

    _handle_optimized_status(args, None)

    if args.family_chunk_size > 0:
        _run_chunked(args, root, genes)
    else:
        _run_resident(args, root, genes)


if __name__ == "__main__":
    main()

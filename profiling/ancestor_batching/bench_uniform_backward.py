#!/usr/bin/env python3
"""CUDA-event benchmark for uniform backward ancestor-batching experiments."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gpurec import GeneReconModel


DEFAULT_FLAGS = {
    "GPUREC_FORWARD_LEAF_INDEX": "1",
    "GPUREC_UNIFORM_PINGPONG": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL": "tiled",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS": "64",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY": "1",
    "GPUREC_FORWARD_TOPOLOGY_INT32": "1",
    "GPUREC_FORWARD_DTS_OVERLAP_MODE": "off",
    "GPUREC_KERNELIZED_ACTIVE_MASK": "1",
    "GPUREC_KERNELIZED_BACKWARD_DTS": "1",
    "GPUREC_FUSED_DTS_BACKWARD_ACCUM": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL": "tree",
    "GPUREC_FUSED_UNIFORM_BACKWARD": "1",
    "GPUREC_BACKWARD_LEAF_INDEX": "1",
    "GPUREC_FUSED_WAVE_PARAM_ACCUM": "1",
    "GPUREC_DTS_PIBAR_UD_FUSION": "1",
    "GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES": "1",
    "GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS": "1",
    "GPUREC_DTS_GRAD_MT_TWO_STAGE": "1",
    "GPUREC_BACKWARD_PARENT_REDUCED_DTS": "tiled",
    "GPUREC_BACKWARD_PARENT_REDUCED_DTS_TILE_SPLITS": "64",
}


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip()


def _parse_dtype(value: str) -> torch.dtype:
    text = value.strip().lower()
    if text in ("fp32", "float32"):
        return torch.float32
    if text in ("bf16", "bfloat16"):
        return torch.bfloat16
    if text in ("fp64", "float64"):
        return torch.float64
    raise argparse.ArgumentTypeError("dtype must be fp32, bf16, or fp64")


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "0", "none", "null"):
        return None
    return int(text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.getenv("DATASET", "tests/data/test_trees_1000"))
    parser.add_argument("--start", type=int, default=int(os.getenv("FAMILY_START", "0")))
    parser.add_argument("--fams", type=int, default=int(os.getenv("FAMS", "50")))
    parser.add_argument("--variant-label", default=os.getenv("VARIANT_LABEL", "manual"))
    parser.add_argument("--reps", type=int, default=int(os.getenv("REPS", "9")))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "5")))
    parser.add_argument("--dtype", type=_parse_dtype, default=_parse_dtype(os.getenv("DTYPE", "fp32")))
    parser.add_argument("--cache-dir", default=os.getenv("PREPROCESS_CACHE_DIR", "/tmp/gpurec_ancestor_batching_cache"))
    parser.add_argument("--max-wave-size", default=os.getenv("MAX_WAVE_SIZE", "8192"))
    parser.add_argument("--max-root-wave-size", default=os.getenv("MAX_ROOT_WAVE_SIZE", ""))
    parser.add_argument("--fixed-iters-Pi", type=int, default=int(os.getenv("FIXED_ITERS_PI", "6")))
    parser.add_argument("--neumann-terms", type=int, default=int(os.getenv("NEUMANN_TERMS", "3")))
    parser.add_argument("--theta-rate", type=float, default=float(os.getenv("THETA_RATE", "0.05")))
    parser.add_argument("--pruning-threshold", type=float, default=float(os.getenv("PRUNING_THRESHOLD", "1e-6")))
    parser.add_argument("--use-pruning", action=argparse.BooleanOptionalAction, default=os.getenv("USE_PRUNING", "1") != "0")
    parser.add_argument("--profile-cuda-api", action="store_true", default=os.getenv("PROFILE_CUDA_API", "0") != "0")
    parser.add_argument("--stats-only", action="store_true", default=os.getenv("STATS_ONLY", "0") != "0")
    parser.add_argument("--no-default-flags", action="store_true")
    args = parser.parse_args()
    args.max_wave_size = _parse_optional_int(args.max_wave_size)
    args.max_root_wave_size = _parse_optional_int(args.max_root_wave_size)
    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    return args


def _selected_genes(root: Path, start: int, fams: int) -> list[str]:
    genes = sorted(root.glob("g_*.nwk"))
    stop = start + fams
    if stop > len(genes):
        raise ValueError(f"requested families [{start}:{stop}], but found {len(genes)} in {root}")
    return [str(path) for path in genes[start:stop]]


def _family_runs(values: torch.Tensor) -> int:
    if values.numel() == 0:
        return 0
    return int(1 + (values[1:] != values[:-1]).sum().item())


def _shape_summary(model: GeneReconModel) -> dict[str, Any]:
    wl = model.static.wave_layout
    metas = wl["wave_metas"]
    family_idx = wl.get("family_idx")
    if torch.is_tensor(family_idx):
        family_idx = family_idx.detach().cpu()

    rows = []
    split_rows = 0
    for k, meta in enumerate(metas):
        ws = int(meta["start"])
        we = int(meta["end"])
        w = int(meta["W"])
        n_splits = int(meta["sl"].numel()) if meta.get("has_splits", False) else 0
        split_rows += n_splits
        fams = family_runs = split_fams = 0
        if family_idx is not None:
            fi = family_idx[ws:we]
            fams = int(torch.unique(fi).numel())
            family_runs = _family_runs(fi)
            if n_splits:
                parent_fi = family_idx[ws + meta["reduce_idx"].detach().cpu()]
                split_fams = int(torch.unique(parent_fi).numel())
        rows.append(
            {
                "wave": k,
                "start": ws,
                "rows": w,
                "split_rows": n_splits,
                "fanout": (n_splits / w) if w else 0.0,
                "families": fams,
                "family_runs": family_runs,
                "split_families": split_fams,
            }
        )

    top_wave_rows = sorted(rows, key=lambda row: row["rows"], reverse=True)[:12]
    top_split_rows = sorted(rows, key=lambda row: row["split_rows"], reverse=True)[:12]
    return {
        "type": "shape",
        "S": int(model.n_species),
        "G": int(model.n_families),
        "C": int(model.static.Pi_shape[0]) if hasattr(model.static, "Pi_shape") else int(sum(int(m["W"]) for m in metas)),
        "waves": len(metas),
        "max_wave_rows": max((int(m["W"]) for m in metas), default=0),
        "split_rows": split_rows,
        "leaves": int(wl["leaf_row_index"].numel()),
        "roots": int(model.static.root_clade_ids.numel()),
        "top_wave_rows": top_wave_rows,
        "top_split_rows": top_split_rows,
    }


def _time_pass(model: GeneReconModel, *, profile_cuda_api: bool) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()

    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)

    forward_start.record()
    loss = model()
    forward_end.record()
    torch.cuda.synchronize()

    if profile_cuda_api:
        torch.cuda.cudart().cudaProfilerStart()
    backward_start.record()
    loss.backward()
    backward_end.record()
    torch.cuda.synchronize()
    if profile_cuda_api:
        torch.cuda.cudart().cudaProfilerStop()

    grad = model.theta.grad.detach().float().cpu()
    return {
        "forward_ms": float(forward_start.elapsed_time(forward_end)),
        "backward_ms": float(backward_start.elapsed_time(backward_end)),
        "total_ms": float(forward_start.elapsed_time(forward_end) + backward_start.elapsed_time(backward_end)),
        "peak_gib": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
        "loss": float(loss.detach().float().cpu()),
        "grad": [float(x) for x in grad.tolist()],
        "grad_norm": float(torch.linalg.vector_norm(grad).item()),
        "grad_finite": bool(torch.isfinite(grad).all().item()),
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    args = _parse_args()
    if not args.no_default_flags:
        for key, value in DEFAULT_FLAGS.items():
            os.environ.setdefault(key, value)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    root = Path(args.dataset)
    genes = _selected_genes(root, args.start, args.fams)

    torch.cuda.empty_cache()
    gc.collect()

    _json_print(
        {
            "type": "config",
            "variant": args.variant_label,
            "commit": _git_commit(),
            "dataset": str(root),
            "family_range": f"{args.start}:{args.start + args.fams}",
            "families": len(genes),
            "dtype": str(args.dtype).replace("torch.", ""),
            "device": torch.cuda.get_device_name(),
            "cuda_capability": ".".join(str(x) for x in torch.cuda.get_device_capability()),
            "fixed_iters_Pi": args.fixed_iters_Pi,
            "neumann_terms": args.neumann_terms,
            "max_wave_size": args.max_wave_size,
            "max_root_wave_size": args.max_root_wave_size,
            "use_pruning": args.use_pruning,
            "pruning_threshold": args.pruning_threshold,
            "theta_rate": args.theta_rate,
            "env_flags": {key: os.environ[key] for key in sorted(os.environ) if key.startswith("GPUREC_")},
        }
    )

    t0 = time.perf_counter()
    model = GeneReconModel.from_trees(
        species_tree=str(root / "sp.nwk"),
        gene_trees=genes,
        mode="global",
        pibar_mode="uniform",
        device="cuda",
        dtype=args.dtype,
        theta_init_rates=(args.theta_rate, args.theta_rate, args.theta_rate),
        preprocess_cache_dir=args.cache_dir,
        fixed_iters_Pi=args.fixed_iters_Pi,
        max_wave_size=args.max_wave_size,
        max_root_wave_size=args.max_root_wave_size,
        neumann_terms=args.neumann_terms,
        use_pruning=args.use_pruning,
        pruning_threshold=args.pruning_threshold,
    )
    torch.cuda.synchronize()

    _json_print({"type": "build", "build_s": time.perf_counter() - t0})
    _json_print(_shape_summary(model))

    if args.stats_only:
        return

    for warmup in range(args.warmups):
        result = _time_pass(model, profile_cuda_api=False)
        _json_print({"type": "warmup", "warmup": warmup, **result})

    reps: list[dict[str, Any]] = []
    for rep in range(args.reps):
        result = _time_pass(model, profile_cuda_api=args.profile_cuda_api)
        reps.append(result)
        _json_print({"type": "rep", "rep": rep, **result})

    last = reps[-1]
    _json_print(
        {
            "type": "summary",
            "variant": args.variant_label,
            "families": len(genes),
            "reps": len(reps),
            "warmups": args.warmups,
            "forward_ms": _summary([float(row["forward_ms"]) for row in reps]),
            "backward_ms": _summary([float(row["backward_ms"]) for row in reps]),
            "total_ms": _summary([float(row["total_ms"]) for row in reps]),
            "peak_gib": max(float(row["peak_gib"]) for row in reps),
            "loss_last": last["loss"],
            "grad_last": last["grad"],
            "grad_norm_last": last["grad_norm"],
            "grad_finite": int(all(bool(row["grad_finite"]) for row in reps)),
        }
    )


if __name__ == "__main__":
    main()

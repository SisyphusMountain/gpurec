#!/usr/bin/env python3
"""Run timing, Nsight Systems, and Nsight Compute ancestor-batching profiles."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().with_name("bench_uniform_backward.py")


@dataclasses.dataclass(frozen=True)
class Variant:
    name: str
    proposal: str
    description: str
    env: dict[str, str]
    required_flags: tuple[str, ...] = ()
    required_markers: tuple[str, ...] = ()
    ncu_kernel_regex: str | None = None


VARIANTS = [
    Variant(
        name="baseline",
        proposal="baseline",
        description="optimized global/uniform backward with Proposal 0 disabled",
        env={"GPUREC_SELF_LOOP_2D_TRITON": "0"},
        ncu_kernel_regex="_wave_backward_uniform_kernel",
    ),
    Variant(
        name="proposal0_2d_triton_bw1",
        proposal="0",
        description="2D Triton self-loop, BLOCK_W=1",
        env={"GPUREC_SELF_LOOP_2D_TRITON": "1", "GPUREC_SELF_LOOP_2D_BLOCK_W": "1"},
        required_flags=("GPUREC_SELF_LOOP_2D_TRITON",),
        required_markers=("_wave_backward_uniform_2d_prototype",),
        ncu_kernel_regex="_wave_backward_uniform_2d",
    ),
    Variant(
        name="proposal0_2d_triton_bw2",
        proposal="0",
        description="2D Triton self-loop, BLOCK_W=2",
        env={"GPUREC_SELF_LOOP_2D_TRITON": "1", "GPUREC_SELF_LOOP_2D_BLOCK_W": "2"},
        required_flags=("GPUREC_SELF_LOOP_2D_TRITON",),
        required_markers=("_wave_backward_uniform_2d_prototype",),
        ncu_kernel_regex="_wave_backward_uniform_2d",
    ),
    Variant(
        name="proposal1_tree_staged_w4_s256",
        proposal="1",
        description="staged tree-DP self-loop, tile W=4, S=256",
        env={
            "GPUREC_SELF_LOOP_TREE_STAGED": "1",
            "GPUREC_SELF_LOOP_TREE_TILE_W": "4",
            "GPUREC_SELF_LOOP_TREE_TILE_S": "256",
        },
        required_flags=("GPUREC_SELF_LOOP_TREE_STAGED",),
        required_markers=("_wave_backward_uniform_tree_staged_prototype",),
        ncu_kernel_regex="tree",
    ),
    Variant(
        name="proposal2_tree_transposed_w64",
        proposal="2",
        description="species-major transposed tree scratch, tile W=64",
        env={"GPUREC_SELF_LOOP_TREE_TRANSPOSED": "1", "GPUREC_SELF_LOOP_TREE_TILE_W": "64"},
        required_flags=("GPUREC_SELF_LOOP_TREE_TRANSPOSED",),
        ncu_kernel_regex="tree",
    ),
    Variant(
        name="proposal3_hybrid_tree_threshold4096",
        proposal="3",
        description="hybrid tree router threshold W>=4096",
        env={
            "GPUREC_SELF_LOOP_TREE_STAGED": "1",
            "GPUREC_SELF_LOOP_TREE_MIN_W": "4096",
            "GPUREC_SELF_LOOP_TREE_SPLIT_WAVES": "0",
        },
        required_flags=("GPUREC_SELF_LOOP_TREE_STAGED", "GPUREC_SELF_LOOP_TREE_MIN_W"),
        ncu_kernel_regex="tree",
    ),
    Variant(
        name="proposal4_forward_path_prefix",
        proposal="4",
        description="forward uniform path-prefix prototype",
        env={"GPUREC_FORWARD_UNIFORM_PATH_PREFIX": "1", "GPUREC_FORWARD_UNIFORM_PATH_TILE_W": "4"},
        required_flags=("GPUREC_FORWARD_UNIFORM_PATH_PREFIX",),
        ncu_kernel_regex="_wave_backward_uniform_kernel",
    ),
    Variant(
        name="proposal5_cuda_nosplit_self",
        proposal="5",
        description="NVRTC CUDA no-split self-loop, self correction",
        env={
            "GPUREC_CUDA_SELF_LOOP_NOSPLIT": "1",
            "GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION": "self",
        },
        required_flags=("GPUREC_CUDA_SELF_LOOP_NOSPLIT",),
        ncu_kernel_regex="gpurec_wave_backward_nosplit_uniform_fp32",
    ),
    Variant(
        name="proposal5_cuda_nosplit_tree",
        proposal="5",
        description="NVRTC CUDA no-split self-loop, tree correction",
        env={
            "GPUREC_CUDA_SELF_LOOP_NOSPLIT": "1",
            "GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION": "tree",
        },
        required_flags=("GPUREC_CUDA_SELF_LOOP_NOSPLIT",),
        ncu_kernel_regex="gpurec_wave_backward_nosplit_uniform_fp32",
    ),
]

SELECTED_NCU_METRICS = {
    "Duration",
    "Memory Throughput",
    "DRAM Throughput",
    "L1/TEX Cache Throughput",
    "L2 Cache Throughput",
    "Compute (SM) Throughput",
    "SM Busy",
    "L1/TEX Hit Rate",
    "L2 Hit Rate",
    "Local Memory Spilling Requests",
    "Local Memory Spilling Request Overhead",
    "Block Size",
    "Grid Size",
    "Registers Per Thread",
    "Dynamic Shared Memory Per Block",
    "Theoretical Occupancy",
    "Achieved Occupancy",
    "Achieved Active Warps Per SM",
    "Waves Per SM",
    "Branch Efficiency",
    "Avg. Divergent Branches",
    "Issue Slots Busy",
}


def _parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_csv_text(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/data/test_trees_1000")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--fams", default="10,50,100")
    parser.add_argument("--variants", default="available")
    parser.add_argument("--phases", default="timing,nsys,ncu")
    parser.add_argument("--artifact-root", default="profiling/ancestor_batching/artifacts")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--cache-dir", default="/tmp/gpurec_ancestor_batching_cache")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--nsys-fams", type=int, default=50)
    parser.add_argument("--nsys-warmups", type=int, default=2)
    parser.add_argument("--ncu-fams", type=int, default=50)
    parser.add_argument("--ncu-warmups", type=int, default=2)
    parser.add_argument("--ncu-set", default="detailed")
    parser.add_argument("--top-kernels", type=int, default=30)
    parser.add_argument("--force-all-proposals", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _source_contains(flag: str) -> bool:
    for path in (REPO_ROOT / "gpurec").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            if flag in path.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def _available_variants(args: argparse.Namespace) -> tuple[list[Variant], list[dict[str, Any]]]:
    by_name = {variant.name: variant for variant in VARIANTS}
    skipped: list[dict[str, Any]] = []
    if args.variants == "available":
        requested = [variant.name for variant in VARIANTS]
    else:
        requested = _parse_csv_text(args.variants)

    selected = []
    for name in requested:
        if name not in by_name:
            raise ValueError(f"unknown variant {name!r}; known: {', '.join(sorted(by_name))}")
        variant = by_name[name]
        missing_flags = [flag for flag in variant.required_flags if not _source_contains(flag)]
        missing_markers = [marker for marker in variant.required_markers if not _source_contains(marker)]
        if (missing_flags or missing_markers) and not args.force_all_proposals:
            skipped.append(
                {
                    "variant": variant.name,
                    "proposal": variant.proposal,
                    "description": variant.description,
                    "reason": "required GPUREC flags or implementation markers are not present in gpurec/",
                    "missing_flags": missing_flags,
                    "missing_markers": missing_markers,
                }
            )
            continue
        selected.append(variant)
    return selected, skipped


def _command_string(cmd: list[str], env_delta: dict[str, str], stdout_path: Path | None = None) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env_delta.items()))
    command = shlex.join(cmd)
    if prefix:
        command = f"{prefix} {command}"
    if stdout_path is not None:
        command = f"{command} > {shlex.quote(str(stdout_path))} 2>&1"
    return command


def _run(
    cmd: list[str],
    *,
    env_delta: dict[str, str],
    stdout_path: Path | None,
    command_log,
    dry_run: bool,
    continue_on_error: bool,
    stderr_path: Path | None = None,
) -> int:
    command_log.write(_command_string(cmd, env_delta, stdout_path) + "\n\n")
    command_log.flush()
    if dry_run:
        return 0
    env = os.environ.copy()
    env.update(env_delta)
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        if stderr_path is None:
            with stdout_path.open("w", encoding="utf-8") as out:
                proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        else:
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
                proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=out, stderr=err)
    else:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if proc.returncode != 0 and not continue_on_error:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return int(proc.returncode)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _last_record(rows: Iterable[dict[str, Any]], record_type: str) -> dict[str, Any] | None:
    found = None
    for row in rows:
        if row.get("type") == record_type:
            found = row
    return found


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _timing_cmd(args: argparse.Namespace, variant: Variant, fams: int) -> list[str]:
    return [
        sys.executable,
        str(BENCH),
        "--dataset",
        args.dataset,
        "--start",
        str(args.start),
        "--fams",
        str(fams),
        "--variant-label",
        variant.name,
        "--dtype",
        args.dtype,
        "--cache-dir",
        args.cache_dir,
        "--reps",
        str(args.reps),
        "--warmups",
        str(args.warmups),
    ]


def _profile_cmd(args: argparse.Namespace, variant: Variant, fams: int, warmups: int) -> list[str]:
    return [
        sys.executable,
        str(BENCH),
        "--dataset",
        args.dataset,
        "--start",
        str(args.start),
        "--fams",
        str(fams),
        "--variant-label",
        variant.name,
        "--dtype",
        args.dtype,
        "--cache-dir",
        args.cache_dir,
        "--reps",
        "1",
        "--warmups",
        str(warmups),
        "--profile-cuda-api",
    ]


def _run_timings(args: argparse.Namespace, run_dir: Path, variants: list[Variant], command_log) -> list[dict[str, Any]]:
    timing_dir = run_dir / "timing"
    summaries = []
    for fams in _parse_csv_ints(args.fams):
        for variant in variants:
            log_path = timing_dir / f"{variant.name}_fams{fams}.jsonl"
            rc = _run(
                _timing_cmd(args, variant, fams),
                env_delta=variant.env,
                stdout_path=log_path,
                command_log=command_log,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )
            rows = _read_jsonl(log_path)
            config = _last_record(rows, "config") or {}
            shape = _last_record(rows, "shape") or {}
            summary = _last_record(rows, "summary") or {}
            summaries.append(
                {
                    "variant": variant.name,
                    "proposal": variant.proposal,
                    "families": fams,
                    "returncode": rc,
                    "log": str(log_path),
                    "commit": config.get("commit", ""),
                    "S": shape.get("S", ""),
                    "C": shape.get("C", ""),
                    "waves": shape.get("waves", ""),
                    "max_wave_rows": shape.get("max_wave_rows", ""),
                    "split_rows": shape.get("split_rows", ""),
                    "forward_median_ms": (summary.get("forward_ms") or {}).get("median", ""),
                    "forward_mean_ms": (summary.get("forward_ms") or {}).get("mean", ""),
                    "backward_median_ms": (summary.get("backward_ms") or {}).get("median", ""),
                    "backward_mean_ms": (summary.get("backward_ms") or {}).get("mean", ""),
                    "backward_min_ms": (summary.get("backward_ms") or {}).get("min", ""),
                    "total_median_ms": (summary.get("total_ms") or {}).get("median", ""),
                    "peak_gib": summary.get("peak_gib", ""),
                    "loss_last": summary.get("loss_last", ""),
                    "grad_last": json.dumps(summary.get("grad_last", [])),
                    "grad_norm_last": summary.get("grad_norm_last", ""),
                    "grad_finite": summary.get("grad_finite", ""),
                }
            )
    _write_csv(
        run_dir / "timing_summary.csv",
        summaries,
        [
            "variant",
            "proposal",
            "families",
            "returncode",
            "log",
            "commit",
            "S",
            "C",
            "waves",
            "max_wave_rows",
            "split_rows",
            "forward_median_ms",
            "forward_mean_ms",
            "backward_median_ms",
            "backward_mean_ms",
            "backward_min_ms",
            "total_median_ms",
            "peak_gib",
            "loss_last",
            "grad_last",
            "grad_norm_last",
            "grad_finite",
        ],
    )
    _write_parity(run_dir / "parity_summary.csv", summaries)
    return summaries


def _write_parity(path: Path, summaries: list[dict[str, Any]]) -> None:
    baselines = {int(row["families"]): row for row in summaries if row["variant"] == "baseline"}
    rows = []
    for row in summaries:
        fams = int(row["families"])
        base = baselines.get(fams)
        if base is None or row["variant"] == "baseline":
            continue
        try:
            grad = json.loads(str(row["grad_last"]))
            base_grad = json.loads(str(base["grad_last"]))
            loss = float(row["loss_last"])
            base_loss = float(base["loss_last"])
        except Exception:
            continue
        grad_diffs = [abs(float(a) - float(b)) for a, b in zip(grad, base_grad)]
        grad_abs_diff = max(grad_diffs, default=float("nan"))
        base_grad_inf = max([abs(float(x)) for x in base_grad] + [1.0])
        rows.append(
            {
                "variant": row["variant"],
                "families": fams,
                "baseline_variant": "baseline",
                "loss_abs_diff": abs(loss - base_loss),
                "grad_max_abs_diff": grad_abs_diff,
                "grad_rel_to_baseline_inf": grad_abs_diff / base_grad_inf,
                "variant_grad_finite": row.get("grad_finite", ""),
                "baseline_grad_finite": base.get("grad_finite", ""),
            }
        )
    _write_csv(
        path,
        rows,
        [
            "variant",
            "families",
            "baseline_variant",
            "loss_abs_diff",
            "grad_max_abs_diff",
            "grad_rel_to_baseline_inf",
            "variant_grad_finite",
            "baseline_grad_finite",
        ],
    )


def _export_nsys_sqlite(rep_path: Path, sqlite_path: Path, args: argparse.Namespace, command_log) -> int:
    cmd = [
        "nsys",
        "export",
        "--type",
        "sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_path),
        str(rep_path),
    ]
    return _run(
        cmd,
        env_delta={},
        stdout_path=sqlite_path.with_suffix(".export.log"),
        command_log=command_log,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


def _kernel_rows(sqlite_path: Path) -> list[dict[str, Any]]:
    if not sqlite_path.exists():
        return []
    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            select
                k.start,
                k.end,
                (k.end - k.start) / 1000000.0 as duration_ms,
                s.value as name,
                k.contextId as context_id,
                k.streamId as stream_id,
                k.gridX,
                k.gridY,
                k.gridZ,
                k.blockX,
                k.blockY,
                k.blockZ,
                k.registersPerThread,
                k.staticSharedMemory,
                k.dynamicSharedMemory
            from CUPTI_ACTIVITY_KIND_KERNEL k
            join StringIds s on k.demangledName = s.id
            order by k.start
            """
        ).fetchall()
    finally:
        con.close()
    return [dict(row) for row in rows]


def _summarize_nsys(sqlite_path: Path, out_prefix: Path, target_regex: str | None, top_k: int) -> dict[str, Any] | None:
    rows = _kernel_rows(sqlite_path)
    if not rows:
        return None
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = buckets.setdefault(
            row["name"],
            {
                "kernel": row["name"],
                "launches": 0,
                "total_ms": 0.0,
                "min_ms": None,
                "max_ms": 0.0,
            },
        )
        dur = float(row["duration_ms"])
        bucket["launches"] += 1
        bucket["total_ms"] += dur
        bucket["min_ms"] = dur if bucket["min_ms"] is None else min(float(bucket["min_ms"]), dur)
        bucket["max_ms"] = max(float(bucket["max_ms"]), dur)
    bucket_rows = sorted(buckets.values(), key=lambda row: float(row["total_ms"]), reverse=True)
    for row in bucket_rows:
        row["mean_ms"] = float(row["total_ms"]) / max(int(row["launches"]), 1)
    _write_csv(
        out_prefix.with_name(out_prefix.name + "_kernel_buckets.csv"),
        bucket_rows[:top_k],
        ["kernel", "launches", "total_ms", "mean_ms", "min_ms", "max_ms"],
    )

    launch_rows = sorted(rows, key=lambda row: float(row["duration_ms"]), reverse=True)[:top_k]
    _write_csv(
        out_prefix.with_name(out_prefix.name + "_kernel_launches.csv"),
        launch_rows,
        [
            "name",
            "duration_ms",
            "gridX",
            "gridY",
            "gridZ",
            "blockX",
            "blockY",
            "blockZ",
            "registersPerThread",
            "dynamicSharedMemory",
            "context_id",
            "stream_id",
        ],
    )

    if target_regex is None:
        return None
    pattern = re.compile(target_regex)
    matches = []
    invocation = 0
    for row in rows:
        if pattern.search(str(row["name"])):
            invocation += 1
            item = dict(row)
            item["invocation"] = invocation
            matches.append(item)
    if not matches:
        return None
    representative = max(matches, key=lambda row: float(row["duration_ms"]))
    representative["kernel_id"] = f"::regex:{target_regex}:{representative['invocation']}"
    _write_csv(
        out_prefix.with_name(out_prefix.name + "_representative_kernel.csv"),
        [representative],
        [
            "name",
            "invocation",
            "kernel_id",
            "duration_ms",
            "gridX",
            "gridY",
            "gridZ",
            "blockX",
            "blockY",
            "blockZ",
            "registersPerThread",
            "dynamicSharedMemory",
            "context_id",
            "stream_id",
        ],
    )
    return representative


def _run_nsys(args: argparse.Namespace, run_dir: Path, variants: list[Variant], command_log) -> dict[str, dict[str, Any]]:
    nsys_dir = run_dir / "nsys"
    representatives: dict[str, dict[str, Any]] = {}
    for variant in variants:
        out_base = nsys_dir / f"{variant.name}_fams{args.nsys_fams}"
        rep_path = out_base.with_suffix(".nsys-rep")
        log_path = out_base.with_suffix(".nsys.log")
        cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "-o",
            str(out_base),
            *_profile_cmd(args, variant, args.nsys_fams, args.nsys_warmups),
        ]
        _run(
            cmd,
            env_delta=variant.env,
            stdout_path=log_path,
            command_log=command_log,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        sqlite_path = out_base.with_suffix(".sqlite")
        _export_nsys_sqlite(rep_path, sqlite_path, args, command_log)
        representative = _summarize_nsys(sqlite_path, out_base, variant.ncu_kernel_regex, args.top_kernels)
        if representative is not None:
            representatives[variant.name] = representative
    return representatives


def _ncu_cmd(args: argparse.Namespace, variant: Variant, kernel_id: str, out_base: Path) -> list[str]:
    return [
        "ncu",
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--set",
        args.ncu_set,
        "--kernel-id",
        kernel_id,
        "--launch-count",
        "1",
        "--csv",
        "--page",
        "raw",
        "-f",
        "-o",
        str(out_base),
        *_profile_cmd(args, variant, args.ncu_fams, args.ncu_warmups),
    ]


def _parse_ncu_csv(path: Path, variant: str, kernel_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = next((i for i, line in enumerate(lines) if '"Metric Name"' in line or "Metric Name" in line), None)
    if header_idx is None:
        return []
    reader = csv.DictReader(lines[header_idx:])
    rows = []
    for row in reader:
        metric = row.get("Metric Name", "")
        if metric not in SELECTED_NCU_METRICS:
            continue
        rows.append(
            {
                "variant": variant,
                "kernel_id": kernel_id,
                "kernel_name": row.get("Kernel Name", ""),
                "section": row.get("Section Name", ""),
                "metric": metric,
                "unit": row.get("Metric Unit", ""),
                "value": row.get("Metric Value", ""),
                "grid": row.get("Grid Size", ""),
                "block": row.get("Block Size", ""),
            }
        )
    return rows


def _run_ncu(
    args: argparse.Namespace,
    run_dir: Path,
    variants: list[Variant],
    representatives: dict[str, dict[str, Any]],
    command_log,
) -> list[dict[str, Any]]:
    ncu_dir = run_dir / "ncu"
    summaries = []
    for variant in variants:
        representative = representatives.get(variant.name)
        if representative is None:
            continue
        kernel_id = str(representative["kernel_id"])
        out_base = ncu_dir / f"{variant.name}_fams{args.ncu_fams}"
        csv_path = out_base.with_suffix(".csv")
        log_path = out_base.with_suffix(".ncu.stderr.log")
        _run(
            _ncu_cmd(args, variant, kernel_id, out_base),
            env_delta=variant.env,
            stdout_path=csv_path,
            stderr_path=log_path,
            command_log=command_log,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        summaries.extend(_parse_ncu_csv(csv_path, variant.name, kernel_id))
    _write_csv(
        run_dir / "ncu_summary.csv",
        summaries,
        ["variant", "kernel_id", "kernel_name", "section", "metric", "unit", "value", "grid", "block"],
    )
    return summaries


def main() -> None:
    args = _parse_args()
    phases = set(_parse_csv_text(args.phases))
    unknown = phases - {"timing", "nsys", "ncu"}
    if unknown:
        raise ValueError(f"unknown phases: {', '.join(sorted(unknown))}")

    variants, skipped = _available_variants(args)
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / args.artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "skipped_variants.json").write_text(json.dumps(skipped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "selected_variants.json").write_text(
        json.dumps([dataclasses.asdict(variant) for variant in variants], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (run_dir / "commands.sh").open("w", encoding="utf-8") as command_log:
        command_log.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        command_log.write(f"# Generated from {Path(__file__).relative_to(REPO_ROOT)}\n")
        command_log.write(f"# Run directory: {run_dir}\n\n")
        timing_summaries: list[dict[str, Any]] = []
        representatives: dict[str, dict[str, Any]] = {}
        if "timing" in phases:
            timing_summaries = _run_timings(args, run_dir, variants, command_log)
        if "nsys" in phases:
            representatives = _run_nsys(args, run_dir, variants, command_log)
        if "ncu" in phases:
            if not representatives:
                rep_path = run_dir / "nsys_representatives_required.json"
                rep_path.write_text(
                    json.dumps({"error": "run nsys phase first so --kernel-id can use representative invocations"}, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                _run_ncu(args, run_dir, variants, representatives, command_log)

    print(json.dumps({"run_dir": str(run_dir), "variants": [v.name for v in variants], "skipped": skipped}, sort_keys=True))


if __name__ == "__main__":
    main()

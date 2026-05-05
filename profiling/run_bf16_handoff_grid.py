#!/usr/bin/env python3
"""Small grid runner for bf16-to-fp32 handoff benchmarks.

This intentionally shells out to ``bench_global_parameter_optimization.py`` so
each candidate starts with a fresh CUDA peak-memory counter and process state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "profiling" / "bench_global_parameter_optimization.py"


def _csv_floats(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/tmp/gpurec_paramopt_cache")
    parser.add_argument("--artifact-dir", default="profiling/bf16_handoff_grid")
    parser.add_argument("--rate-thresholds", type=_csv_floats, default=[1e-2, 5e-3, 1e-3])
    parser.add_argument("--nll-thresholds", type=_csv_floats, default=[1e-1, 1e-2, 1e-3])
    parser.add_argument("--criteria", choices=("any", "all"), default="any")
    parser.add_argument("--bf16-fixed-steps", type=int, default=1)
    parser.add_argument("--bf16-min-steps", type=int, default=2)
    parser.add_argument("--bf16-max-steps", type=int, default=8)
    parser.add_argument("--fp32-polish-steps", type=int, default=4)
    parser.add_argument("--maxfun", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--bench-arg",
        action="append",
        default=[],
        help="Extra argument to append to every benchmark command; repeat for multiple args.",
    )
    return parser.parse_args()


def _dataset_specs() -> list[dict[str, object]]:
    return [
        {
            "label": "test_trees_100",
            "dataset": "tests/data/test_trees_100",
            "extra": [],
        },
        {
            "label": "test_trees_1000_first100",
            "dataset": "tests/data/test_trees_1000",
            "extra": ["--max-families", "100", "--allow-missing-target"],
        },
    ]


def _base_cmd(args: argparse.Namespace, spec: dict[str, object], strategy: str) -> list[str]:
    return [
        sys.executable,
        str(BENCH),
        "--dataset",
        str(spec["dataset"]),
        "--cache-dir",
        args.cache_dir,
        "--strategies",
        strategy,
        "--bf16-start-steps",
        str(args.bf16_fixed_steps),
        "--bf16-threshold-min-steps",
        str(args.bf16_min_steps),
        "--bf16-threshold-max-steps",
        str(args.bf16_max_steps),
        "--fp32-polish-steps",
        str(args.fp32_polish_steps),
        "--maxfun",
        str(args.maxfun),
        "--bf16-switch-criteria",
        args.criteria,
        "--no-print-evals",
        *list(spec["extra"]),
        *args.bench_arg,
    ]


def main() -> None:
    args = _parse_args()
    artifact_dir = (REPO_ROOT / args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for spec in _dataset_specs():
        fixed_cmd = _base_cmd(
            args,
            spec,
            "bf16-resident-fixed-fp32-polish",
        )
        fixed_log = artifact_dir / f"{spec['label']}_fixed_bf16{args.bf16_fixed_steps}.log"
        runs.append({"label": spec["label"], "kind": "fixed", "cmd": fixed_cmd, "log": str(fixed_log)})

        for rate_threshold in args.rate_thresholds:
            for nll_threshold in args.nll_thresholds:
                cmd = _base_cmd(
                    args,
                    spec,
                    "bf16-resident-threshold-fp32-polish",
                )
                cmd.extend(
                    [
                        "--bf16-switch-rate-rtol",
                        f"{rate_threshold:.8g}",
                        "--bf16-switch-nll-abs-tol",
                        f"{nll_threshold:.8g}",
                    ]
                )
                log = artifact_dir / (
                    f"{spec['label']}_rate{rate_threshold:.0e}_nll{nll_threshold:.0e}_{args.criteria}.log"
                )
                runs.append(
                    {
                        "label": spec["label"],
                        "kind": "threshold",
                        "rate_threshold": rate_threshold,
                        "nll_threshold": nll_threshold,
                        "criteria": args.criteria,
                        "cmd": cmd,
                        "log": str(log),
                    }
                )

    for run in runs:
        print("grid_command", "log", run["log"], "cmd", " ".join(run["cmd"]), flush=True)
        if args.dry_run:
            run["returncode"] = None
            continue
        start = time.perf_counter()
        with Path(run["log"]).open("w") as out:
            proc = subprocess.run(run["cmd"], cwd=REPO_ROOT, stdout=out, stderr=subprocess.STDOUT, text=True)
        run["returncode"] = proc.returncode
        run["elapsed_s"] = time.perf_counter() - start
        print(
            "grid_result",
            "log", run["log"],
            "returncode", proc.returncode,
            "elapsed_s", f"{run['elapsed_s']:.3f}",
            flush=True,
        )
        if proc.returncode != 0:
            break

    manifest = artifact_dir / "manifest.json"
    manifest.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    print("grid_manifest", manifest, flush=True)


if __name__ == "__main__":
    main()

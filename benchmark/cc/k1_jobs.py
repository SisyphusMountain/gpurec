"""Submit the fused-linear-self-loop verification jobs from an agent checkout.

Usage (on the cluster login node, with benchmark/cc/env.sh already sourced):
    python benchmark/cc/k1_jobs.py smoke|verify|fit40 [--compare N] [--time N] [--minutes M]

Wraps benchmark/cc/sbatch_h100.sh so the job command is assembled here rather than
through several layers of shell quoting.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

JOBS = {
    "smoke": (
        "k1_smoke",
        "benchmark/cc/test_linear_forward.py",
        dict(compare=8, time=8, reps=2, minutes=20, dtype="float32"),
    ),
    "fp64": (
        "k1_fp64",
        "benchmark/cc/test_linear_forward.py",
        dict(compare=8, time=8, reps=2, minutes=20, dtype="float64"),
    ),
    "verify": (
        "k1_verify",
        "benchmark/cc/test_linear_forward.py",
        dict(compare=100, time=500, reps=3, minutes=30, dtype="float32"),
    ),
    "fit40": ("k1_smoke40", "benchmark/cc/run_genewise.py", dict(minutes=30)),
    # 40-family end-to-end fit plus the float64 control that shows the residual log-vs-linear
    # disagreement is model-dtype rounding of the row frame, not an arithmetic difference.
    "fit40_fp64": ("k1_fit40_fp64", None, dict(minutes=30)),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=sorted(JOBS))
    parser.add_argument("--compare", type=int, default=None)
    parser.add_argument("--time", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--minutes", type=int, default=None)
    args = parser.parse_args()

    repo = os.environ["CC_REPO"]
    name, script, spec = JOBS[args.job]
    minutes = args.minutes if args.minutes is not None else spec["minutes"]
    fit_command = (
        "$CC_PY -u benchmark/cc/run_genewise.py --species $CC_SPECIES "
        "--families $CC_FAMILIES --limit 40 --out-dir $CC_RUNS/results --tag k1_smoke40"
    )
    fp64_command = (
        "$CC_PY -u benchmark/cc/test_linear_forward.py --species $CC_SPECIES "
        "--families $CC_FAMILIES --limit-compare 8 --limit-time 8 --clade-budget 315000 "
        "--pi-iters 16 --neumann-terms 16 --theta -6.0 --window 60 --reps 2 --dtype float64 "
        "--fused-blocks 256"
    )
    if args.job == "fit40":
        command = fit_command
    elif args.job == "fit40_fp64":
        command = f"{fit_command}; {fp64_command}"
    else:
        compare = args.compare if args.compare is not None else spec["compare"]
        limit_time = args.time if args.time is not None else spec["time"]
        reps = args.reps if args.reps is not None else spec["reps"]
        command = (
            f"$CC_PY -u {script} --species $CC_SPECIES --families $CC_FAMILIES "
            f"--limit-compare {compare} --limit-time {limit_time} --clade-budget 315000 "
            f"--pi-iters 16 --neumann-terms 16 --theta -6.0 --window 60 --reps {reps} "
            f"--dtype {spec['dtype']} --fused-blocks 256,512,1024,2048"
        )
    result = subprocess.run(
        ["bash", f"{repo}/benchmark/cc/sbatch_h100.sh", name, f"00:{minutes:02d}:00", command],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

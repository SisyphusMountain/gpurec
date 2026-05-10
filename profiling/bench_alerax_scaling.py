#!/usr/bin/env python3
"""Benchmark AleRax reconciliation time as a function of gene-family count.

Generates a families_subsample.txt containing the first N families from the
test_trees_1000 dataset and times AleRax on that subsample.

Usage:
    python profiling/bench_alerax_scaling.py
    python profiling/bench_alerax_scaling.py --n-families 10 50 100 250 500 1000
    python profiling/bench_alerax_scaling.py --n-families 100 --n-mpi 4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "tests" / "data" / "test_trees_1000"
ALERAX_BINARY = shutil.which("alerax") or str(
    PROJECT_ROOT / "extra" / "AleRax_modified" / "build" / "bin" / "alerax"
)
MPI_BINARY = shutil.which("mpiexec") or shutil.which("mpirun")


def parse_families(families_path: Path) -> list[tuple[str, str]]:
    """Parse families.txt and return list of (family_name, gene_tree_filename)."""
    families = []
    current_name = None
    with open(families_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("- "):
                current_name = line[2:]
            elif line.startswith("gene_tree = ") and current_name is not None:
                gene_tree = line[len("gene_tree = "):]
                families.append((current_name, gene_tree))
                current_name = None
    return families


def write_subsample(families: list[tuple[str, str]], n: int, out_path: Path) -> None:
    subset = families[:n]
    lines = ["[FAMILIES]"]
    for name, gene_tree in subset:
        lines.append(f"- {name}")
        lines.append(f"gene_tree = {gene_tree}")
    out_path.write_text("\n".join(lines) + "\n")


def run_alerax(data_dir: Path, families_file: Path, output_dir: Path, n_mpi: int) -> float:
    """Run AleRax and return wall-clock time in seconds."""
    if not Path(ALERAX_BINARY).exists():
        raise FileNotFoundError(f"AleRax binary not found: {ALERAX_BINARY}")
    if MPI_BINARY is None:
        raise FileNotFoundError("mpiexec/mpirun not found on PATH")

    cmd = [
        MPI_BINARY, "-np", str(n_mpi),
        ALERAX_BINARY,
        "-s", "sp.nwk",
        "-f", str(families_file),
        "-p", str(output_dir),
        "--model-parametrization", "GLOBAL",
        "--gene-tree-samples", "0",
    ]

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
        cwd=str(data_dir),
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print("STDOUT:", result.stdout[-3000:] if result.stdout else "(empty)")
        print("STDERR:", result.stderr[-3000:] if result.stderr else "(empty)")
        raise RuntimeError(f"AleRax failed (return code {result.returncode})")

    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AleRax scaling with family count")
    parser.add_argument(
        "--n-families", type=int, nargs="+",
        default=[10, 50, 100, 250, 500, 1000],
        metavar="N",
        help="Number of families to subsample (default: 10 50 100 250 500 1000)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Path to dataset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--n-mpi", type=int, default=24,
        help="Number of MPI processes (default: 24)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    families_path = data_dir / "families.txt"
    if not families_path.exists():
        print(f"Error: {families_path} not found", file=sys.stderr)
        sys.exit(1)

    all_families = parse_families(families_path)
    max_available = len(all_families)
    print(f"Dataset: {data_dir}")
    print(f"Total families available: {max_available}")
    print(f"MPI processes: {args.n_mpi}")
    print()

    results: list[tuple[int, float]] = []

    with tempfile.TemporaryDirectory(prefix="alerax_bench_") as tmpdir:
        subsample_file = Path(tmpdir) / "families_subsample.txt"

        for n in args.n_families:
            if n > max_available:
                print(f"Skipping n={n}: only {max_available} families available")
                continue

            write_subsample(all_families, n, subsample_file)
            output_dir = Path(tmpdir) / f"output_{n}"
            shutil.rmtree(output_dir, ignore_errors=True)

            print(f"Running AleRax on {n} families...", flush=True)
            try:
                elapsed = run_alerax(data_dir, subsample_file, output_dir, args.n_mpi)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"  FAILED: {e}")
                continue

            per_family = elapsed / n
            results.append((n, elapsed))
            print(f"  {elapsed:8.2f}s total  |  {per_family:.3f}s/family")

    print()
    print(f"{'N families':>12}  {'total (s)':>12}  {'s/family':>10}")
    print("-" * 38)
    for n, t in results:
        print(f"{n:>12}  {t:>12.2f}  {t/n:>10.3f}")


if __name__ == "__main__":
    main()

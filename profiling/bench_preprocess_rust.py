"""Benchmark Rust preprocessing adapters on AleRax-style family inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from gpurec.core.model import parse_alerax_family_file
from gpurec.core.preprocess_rust import (
    RustPreprocessExtension,
    RustPreprocessSubprocessExtension,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "tests" / "data" / "hogenom_bench"
RUST_MANIFEST = ROOT / "crates" / "gpurec-preprocess" / "Cargo.toml"
RUST_BINARY = ROOT / "crates" / "gpurec-preprocess" / "target" / "release" / "gpurec-preprocess"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--species", type=Path, default=None)
    parser.add_argument("--families-file", type=Path, default=None)
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--build-release", action="store_true")
    parser.add_argument("--include-species-matrices", action="store_true")
    parser.add_argument("--skip-adapter", action="store_true")
    return parser


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _time_repeated(fn: Callable[[], Any], repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return times


def _build_release() -> None:
    base_cmd = [
        "cargo",
        "build",
        "--release",
        "--locked",
    ]
    for extra in ([], ["--features", "python-extension"]):
        subprocess.run(
            [
                *base_cmd,
                *extra,
                "--manifest-path",
                str(RUST_MANIFEST),
            ],
            cwd=ROOT,
            check=True,
        )


def _load_inputs(args: argparse.Namespace):
    data_dir = args.data_dir.resolve()
    species = (args.species or data_dir / "sp.nwk").resolve()
    families_file = (args.families_file or data_dir / "families.txt").resolve()
    names, tree_paths, leaf_maps = parse_alerax_family_file(
        families_file,
        max_families=args.max_families,
    )
    families = {name: paths for name, paths in zip(names, tree_paths)}
    leaf_species_maps = {
        name: mapping
        for name, mapping in zip(names, leaf_maps)
        if mapping
    }
    return species, families, leaf_species_maps


def _rust_cli_call(binary: Path, request_path: Path, *flags: str) -> bytes:
    result = subprocess.run(
        [str(binary), *flags, str(request_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.max_families is not None and args.max_families <= 0:
        raise ValueError("--max-families must be positive when provided")
    if args.build_release:
        _build_release()
    if not RUST_BINARY.exists():
        raise RuntimeError(
            f"Rust release binary not found at {RUST_BINARY}; pass --build-release first"
        )

    species, families, leaf_species_maps = _load_inputs(args)
    request = {
        "species_path": str(species),
        "families": families,
        "leaf_species_maps": leaf_species_maps,
        "include_species_matrices": args.include_species_matrices,
        "num_threads": args.threads,
    }

    rust_native_ext = RustPreprocessExtension()
    rust_subprocess_ext = RustPreprocessSubprocessExtension(preprocess_binary=RUST_BINARY)
    results: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="gpurec-preprocess-bench-") as tmp:
        request_path = Path(tmp) / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")

        _rust_cli_call(RUST_BINARY, request_path, "--discard-output")

        discard_times = _time_repeated(
            lambda: _rust_cli_call(RUST_BINARY, request_path, "--discard-output"),
            args.repeats,
        )
        binary_sizes: list[int] = []
        binary_times = _time_repeated(
            lambda: binary_sizes.append(
                len(_rust_cli_call(RUST_BINARY, request_path, "--binary-output"))
            ),
            args.repeats,
        )
        results["rust_discard_output"] = {
            "times_s": discard_times,
            "median_s": _median(discard_times),
        }
        results["rust_binary_output"] = {
            "times_s": binary_times,
            "median_s": _median(binary_times),
            "bytes": binary_sizes[-1] if binary_sizes else 0,
        }

        if not args.skip_adapter:
            rust_native_ext.preprocess_multiple_families(
                str(species),
                families,
                leaf_species_maps=leaf_species_maps,
                include_species_matrices=args.include_species_matrices,
                num_threads=args.threads,
            )
            native_times = _time_repeated(
                lambda: rust_native_ext.preprocess_multiple_families(
                    str(species),
                    families,
                    leaf_species_maps=leaf_species_maps,
                    include_species_matrices=args.include_species_matrices,
                    num_threads=args.threads,
                ),
                args.repeats,
            )
            results["rust_native_adapter"] = {
                "times_s": native_times,
                "median_s": _median(native_times),
            }

            rust_subprocess_ext.preprocess_multiple_families(
                str(species),
                families,
                leaf_species_maps=leaf_species_maps,
                include_species_matrices=args.include_species_matrices,
                num_threads=args.threads,
            )
            subprocess_times = _time_repeated(
                lambda: rust_subprocess_ext.preprocess_multiple_families(
                    str(species),
                    families,
                    leaf_species_maps=leaf_species_maps,
                    include_species_matrices=args.include_species_matrices,
                    num_threads=args.threads,
                ),
                args.repeats,
            )
            results["rust_subprocess_adapter"] = {
                "times_s": subprocess_times,
                "median_s": _median(subprocess_times),
            }

    print(
        json.dumps(
            {
                "config": {
                    "species": str(species),
                    "families": len(families),
                    "threads": args.threads,
                    "repeats": args.repeats,
                    "include_species_matrices": args.include_species_matrices,
                },
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

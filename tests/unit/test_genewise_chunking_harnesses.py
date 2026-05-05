"""Subprocess coverage for the Proposal 7 genewise profiling CLIs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
except Exception:  # pragma: no cover - import availability is environment-specific
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "tests" / "data" / "test_trees_1000"
FORWARD_HARNESS = REPO_ROOT / "profiling" / "bench_genewise_forward_chunking.py"
BACKWARD_HARNESS = REPO_ROOT / "profiling" / "bench_genewise_backward_chunking.py"


def _require_cuda_data_and_harness(script: Path) -> None:
    if torch is None or not torch.cuda.is_available():
        pytest.skip("CUDA required for genewise profiling harness smoke tests")
    if not DATASET.exists() or not (DATASET / "sp.nwk").exists():
        pytest.skip(f"dataset not found: {DATASET}")
    if len(list(DATASET.glob("g_*.nwk"))) < 2:
        pytest.skip("test_trees_1000 needs at least two gene trees")
    if not script.exists():
        pytest.skip(f"Proposal 7 harness is not present yet: {script}")


def _base_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PREPROCESS_CACHE_DIR"] = str(tmp_path / "preprocess_cache")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not env.get("PYTHONPATH")
        else str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    return env


def _run_subprocess(
    script: Path,
    args: list[str],
    *,
    tmp_path: Path,
    env_overrides: dict[str, str] | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    env = _base_env(tmp_path)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _help_text(script: Path, tmp_path: Path) -> str:
    result = _run_subprocess(script, ["--help"], tmp_path=tmp_path, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def _stats_args(script: Path, tmp_path: Path, *, backward: bool) -> list[str]:
    help_out = _help_text(script, tmp_path)
    args = [
        "--dataset",
        str(DATASET),
        "--fams",
        "2",
        "--stats-only",
    ]
    if "--family-chunk-size" in help_out:
        args += ["--family-chunk-size", "1"]
    if "--reps" in help_out:
        args += ["--reps", "1"]
    if "--warmups" in help_out:
        args += ["--warmups", "0"]
    if "--strict-optimized-kernels" in help_out:
        args.append("--strict-optimized-kernels")
    if backward and "--backward-path" in help_out:
        args += ["--backward-path", "optimized-genewise"]
    if not backward and "--root-rows" in help_out:
        args.append("--root-rows")
    return args


def _assert_contains_any(output: str, choices: tuple[str, ...]) -> None:
    lowered = output.lower()
    if not any(choice.lower() in lowered for choice in choices):
        joined = "\n  ".join(choices)
        pytest.fail(f"expected one of:\n  {joined}\n\noutput:\n{output}")


def _assert_positive_contract(output: str) -> None:
    _assert_contains_any(output, ("mode=genewise", "mode genewise"))
    _assert_contains_any(output, ("pibar_mode=uniform", "pibar_mode uniform"))
    _assert_contains_any(output, ("active_path_flags", "active path flags"))
    _assert_contains_any(
        output,
        (
            "strict_optimized_verdict pass",
            "strict_optimized=1",
            "strict_optimized_kernels 1",
            "strict optimized verdict pass",
        ),
    )
    _assert_contains_any(output, ("root-row", "root_rows", "root_row"))
    _assert_contains_any(
        output,
        (
            "optimized_path_verdict verdict optimized",
            "generic_fallback 0",
            "generic_pyTorch_fallback 0",
            "generic_pytorch_fallback 0",
            "generic_pytorch_fallback=0",
            "no_generic_fallback",
            "fallback_verdict pass",
        ),
    )
    _assert_contains_any(output, ("family_locality_summary", "locality_preserved"))
    _assert_contains_any(output, ("family_runs", "family runs"))


@pytest.mark.parametrize(
    ("script", "backward"),
    (
        (FORWARD_HARNESS, False),
        (BACKWARD_HARNESS, True),
    ),
)
def test_genewise_chunking_harness_stats_only_contract(
    script: Path,
    backward: bool,
    tmp_path: Path,
) -> None:
    _require_cuda_data_and_harness(script)

    result = _run_subprocess(
        script,
        _stats_args(script, tmp_path, backward=backward),
        tmp_path=tmp_path,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    _assert_positive_contract(output)


def test_genewise_backward_harness_reports_nonoptimized_when_strict_disabled(
    tmp_path: Path,
) -> None:
    _require_cuda_data_and_harness(BACKWARD_HARNESS)

    help_out = _help_text(BACKWARD_HARNESS, tmp_path)
    if "--no-strict-optimized-kernels" not in help_out:
        pytest.skip("backward harness has no supported strict-disable CLI knob")

    args = _stats_args(BACKWARD_HARNESS, tmp_path, backward=True)
    args = [
        "--no-strict-optimized-kernels" if arg == "--strict-optimized-kernels" else arg
        for arg in args
    ]
    result = _run_subprocess(BACKWARD_HARNESS, args, tmp_path=tmp_path)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        _assert_contains_any(output, ("strict", "optimized", "fallback"))
        return
    _assert_contains_any(
        output,
        (
            "strict_optimized_verdict fail",
            "strict_optimized=0",
            "strict_optimized_kernels 0",
            "non_optimized",
            "non-optimized",
            "require_optimized_guard 0",
        ),
    )

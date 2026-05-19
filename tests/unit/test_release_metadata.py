from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = ROOT / "scripts" / "check_release_metadata.py"


def _has_module_level_gpu_marker(text: str) -> bool:
    return (
        re.search(
            r"^pytestmark\s*=\s*(?:pytest\.mark\.gpu|\[[^\]]*pytest\.mark\.gpu)",
            text,
            flags=re.M | re.S,
        )
        is not None
    )


def test_release_metadata_check_reports_only_current_license_blockers():
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing top-level LICENSE file" in result.stdout
    assert "license metadata" in result.stdout
    assert "license classifier" in result.stdout
    assert "authors" not in result.stdout
    assert "classifier(s)" not in result.stdout
    assert "project.urls" not in result.stdout
    assert "Traceback" not in result.stderr


def test_release_metadata_check_accepts_complete_metadata_fixture(tmp_path: Path):
    (tmp_path / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# fixture\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "fixture"
version = "0.0.0"
description = "fixture"
readme = "README.md"
requires-python = ">=3.10,<3.13"
license = { file = "LICENSE" }
authors = [{ name = "Fixture Maintainer" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

[project.urls]
Repository = "https://example.invalid/repo"
Issues = "https://example.invalid/issues"
Documentation = "https://example.invalid/docs"
""".lstrip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "release metadata check passed" in result.stdout
    assert result.stderr == ""


def test_cpu_ci_builds_and_smokes_release_artifacts():
    workflow = (ROOT / ".github" / "workflows" / "cpu-unit.yml").read_text(
        encoding="utf-8"
    )

    assert "\n  package:\n" in workflow
    for required in (
        'python -m pip install -e ".[release]"',
        "python -m build",
        "python -m twine check dist/*",
        "tarfile.open",
        "zipfile.ZipFile",
        "gpurec/core/cpp/preprocess.cpp",
        "gpurec/core/cpp/clade_utils.hpp",
        "wheel missing required package data",
        "python -m pip install --no-deps dist/*.whl",
        "gpurec --help",
        "python -m gpurec.cli --help",
    ):
        assert required in workflow


@pytest.mark.parametrize(
    "command",
    (
        ("gpurec", "--help"),
        (sys.executable, "-m", "gpurec.cli", "--help"),
    ),
)
def test_cli_help_smokes_are_quiet_on_cpu(command: tuple[str, ...]):
    if command[0] == "gpurec" and shutil.which("gpurec") is None:
        pytest.skip("gpurec console script is not installed")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "usage: gpurec" in result.stdout
    assert result.stderr == ""


def test_cpu_ci_matrix_covers_declared_python_versions():
    workflow = (ROOT / ".github" / "workflows" / "cpu-unit.yml").read_text(
        encoding="utf-8"
    )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    supported_versions = ("3.10", "3.11", "3.12")

    assert 'python-version: ["3.10", "3.11", "3.12"]' in workflow
    assert "python-version: ${{ matrix.python-version }}" in workflow
    assert 'requires-python = ">=3.10,<3.13"' in pyproject
    for version in supported_versions:
        assert f"Programming Language :: Python :: {version}" in pyproject


def test_rust_backtracking_uses_pinned_git_dependency():
    manifest = (ROOT / "crates" / "gpurec-backtrack" / "Cargo.toml").read_text(
        encoding="utf-8"
    )
    lockfile = (ROOT / "crates" / "gpurec-backtrack" / "Cargo.lock").read_text(
        encoding="utf-8"
    )

    assert (
        'rustree = { git = "https://github.com/SisyphusMountain/rustree.git"'
        in manifest
    )
    assert 'rev = "e3a58478f0e57c80af04c730acade639d8e9015e"' in manifest
    assert 'path = "../../rustree"' not in manifest
    assert "git+https://github.com/SisyphusMountain/rustree.git" in lockfile


def test_cpu_ci_runs_rust_backtracking_gate():
    workflow = (ROOT / ".github" / "workflows" / "cpu-unit.yml").read_text(
        encoding="utf-8"
    )

    assert "\n  rust-backtrack:\n" in workflow
    assert "rustup default stable" in workflow
    assert "actions/setup-python@v5" in workflow
    assert "python -m pip install pytest" in workflow
    assert (
        "cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml"
        in workflow
    )
    assert (
        "cargo run --locked --quiet --manifest-path "
        "crates/gpurec-backtrack/Cargo.toml -- --help"
    ) in workflow
    assert "pytest -q tests/integration/test_rust_backtracking_fixture.py" in workflow
    assert 'pytest -q -m "integration and not gpu"' not in workflow


def test_tests_readme_explicit_cpu_unit_paths_match_marker_gate():
    readme = (ROOT / "tests" / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"The explicit equivalent is useful.*?```bash\n(?P<block>.*?)```",
        readme,
        flags=re.S,
    )

    assert match is not None
    documented_paths = {
        line.strip().rstrip(" \\")
        for line in match.group("block").splitlines()
        if line.strip().startswith("tests/unit/")
    }
    cpu_unit_modules = {
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "tests" / "unit").glob("test_*.py"))
        if not _has_module_level_gpu_marker(path.read_text(encoding="utf-8"))
    }

    assert documented_paths == cpu_unit_modules


def test_release_readiness_orders_clean_checkout_before_build():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    assert guide.index("git clean -Xdn") < guide.index("python -m build")
    assert "stale `build/`, `dist/`, or `*.egg-info/`" in guide
    assert "gpurec --help" in guide
    assert "python -m gpurec.cli --help" in guide
    assert 'pytest -q -m "integration and not gpu"' in guide

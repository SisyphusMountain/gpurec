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


def _write_complete_release_metadata_fixture(
    root: Path,
    *,
    readme_line: str = 'readme = "README.md"',
    create_readme: bool = True,
) -> None:
    (root / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    if create_readme:
        (root / "README.md").write_text("# fixture\n", encoding="utf-8")
    readme_block = f"{readme_line}\n" if readme_line else ""
    (root / "pyproject.toml").write_text(
        f"""
[project]
name = "fixture"
version = "0.0.0"
description = "fixture"
{readme_block}requires-python = ">=3.10,<3.13"
license = {{ file = "LICENSE" }}
authors = [{{ name = "Fixture Maintainer" }}]
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
    assert "readme" not in result.stdout.lower()
    assert "Traceback" not in result.stderr


def test_release_metadata_check_accepts_complete_metadata_fixture(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path)

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "release metadata check passed" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_readme_metadata(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path, readme_line="")

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "must declare readme metadata" in result.stdout
    assert "license" not in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_declared_readme_file(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        readme_line='readme = "MISSING.md"',
        create_readme=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "declared readme file does not exist: MISSING.md" in result.stdout
    assert "license" not in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_accepts_readme_table_fixture(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        readme_line='readme = { file = "README.md" }',
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
        "required_sdist = required_wheel +",
        "examples/minimal-run-config.json",
        "examples/tiny/families.txt",
        "examples/tiny/gene.map",
        "examples/tiny/gene.nwk",
        "examples/tiny/species.nwk",
        "crates/gpurec-backtrack/Cargo.toml",
        "crates/gpurec-backtrack/Cargo.lock",
        "crates/gpurec-backtrack/src/lib.rs",
        "crates/gpurec-backtrack/src/main.rs",
        "forbidden_sdist_prefixes",
        "crates/gpurec-backtrack/target/",
        "forbidden_wheel_prefixes",
        "gpurec/core/cpp/preprocess.cpp",
        "gpurec/core/cpp/clade_utils.hpp",
        "sdist missing required source files",
        "sdist includes forbidden paths",
        "wheel missing required package data",
        "wheel includes forbidden paths",
        "Smoke Rust crate from source archive",
        "SDIST_UNPACK_DIR",
        'cargo test --locked --manifest-path "$root/crates/gpurec-backtrack/Cargo.toml"',
        (
            'cargo run --locked --quiet --manifest-path "$root/crates/gpurec-backtrack/'
            'Cargo.toml" -- --help'
        ),
        "python -m pip install --no-deps dist/*.whl",
        "smoke_dir=$(mktemp -d)",
        'cd "$smoke_dir"',
        "gpurec --help",
        "python -m gpurec.cli --help",
        "import gpurec",
        'Path(os.environ["GITHUB_WORKSPACE"]).resolve()',
        "package_path.is_relative_to(workspace)",
        '"site-packages" not in package_path.parts',
        '"dist-packages" not in package_path.parts',
        "imported gpurec from checkout",
    ):
        assert required in workflow
    assert workflow.index("python -m pip install --no-deps dist/*.whl") < workflow.index(
        'cd "$smoke_dir"'
    )
    assert workflow.index('cd "$smoke_dir"') < workflow.index("gpurec --help")


def test_manifest_includes_documented_examples_in_source_archive():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "recursive-include gpurec/core/cpp *.cpp *.hpp" in manifest
    assert "recursive-include examples" in manifest
    for pattern in ("*.json", "*.map", "*.nwk", "*.txt"):
        assert pattern in manifest


def test_manifest_includes_rust_backtracking_crate_in_source_archive_only():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "include crates/gpurec-backtrack/Cargo.toml" in manifest
    assert "include crates/gpurec-backtrack/Cargo.lock" in manifest
    assert "recursive-include crates/gpurec-backtrack/src *.rs" in manifest
    assert "prune crates/gpurec-backtrack/target" in manifest


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


def test_runtime_dependencies_include_cpp_extension_build_backend():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r"^dependencies\s*=\s*\[(?P<block>.*?)^\]",
        pyproject,
        flags=re.M | re.S,
    )

    assert match is not None
    dependencies = {
        line.strip().rstrip(",").strip('"').split(";", 1)[0]
        for line in match.group("block").splitlines()
        if line.strip().startswith('"')
    }
    dependency_names = {
        re.split(r"[\[<>=!~ ]", dependency, maxsplit=1)[0].lower().replace("_", "-")
        for dependency in dependencies
    }

    assert "ninja" in dependency_names


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
    assert 'pytest -q -m "integration and not gpu"' in workflow
    assert "pytest -q tests/integration/test_rust_backtracking_fixture.py" not in workflow


def test_stochastic_backtracking_notes_use_current_rust_commands():
    notes = (ROOT / "docs" / "stochastic-backtracking-progress.md").read_text(
        encoding="utf-8"
    )

    assert "pinned\n  `rustree` git dependency" in notes
    assert "local\n  `rustree` checkout" not in notes
    assert "cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml" in notes
    assert (
        "cargo build --locked --release --manifest-path "
        "crates/gpurec-backtrack/Cargo.toml"
    ) in notes
    assert (
        "cargo run --locked --quiet --manifest-path "
        "crates/gpurec-backtrack/Cargo.toml -- --help"
    ) in notes
    assert "tests/integration/test_rust_backtracking_fixture.py" in notes
    assert "tests/integration/test_stochastic_backtracking.py" not in notes


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


def test_tests_readme_backtracking_binary_smoke_is_reproducible():
    readme = (ROOT / "tests" / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"Backtracking smoke should prefer.*?```bash\n(?P<block>.*?)```",
        readme,
        flags=re.S,
    )

    assert match is not None
    block = match.group("block")
    assert (
        "cargo build --locked --release --manifest-path "
        "crates/gpurec-backtrack/Cargo.toml"
    ) in block
    assert (
        "GPUREC_BACKTRACK_BIN=crates/gpurec-backtrack/target/release/"
        "gpurec-backtrack"
    ) in block
    assert (
        "tests/integration/test_stochastic_backtracking.py::"
        "test_rust_stochastic_backtracking_exports_recphyloxml"
    ) in block


def test_release_readiness_orders_clean_checkout_before_build():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    preview_command = "git clean -Xdn -- build dist '*.egg-info'"
    clean_command = "git clean -Xdf -- build dist '*.egg-info'"

    assert guide.index(preview_command) < guide.index(clean_command)
    assert guide.index(clean_command) < guide.index("python -m build")
    assert "stale `build/`, `dist/`, or `*.egg-info/`" in guide
    assert "gpurec --help" in guide
    assert "python -m gpurec.cli --help" in guide
    assert 'pytest -q -m "integration and not gpu"' in guide


def test_release_readiness_scopes_ignored_clean_commands():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    assert re.search(r"(?m)^git clean -Xdn$", guide) is None
    assert re.search(r"(?m)^git clean -Xdf$", guide) is None
    assert "git clean -Xdn -- build dist '*.egg-info'" in guide
    assert "git clean -Xdf -- build dist '*.egg-info'" in guide


def test_final_theta_artifact_is_documented_as_export_only():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    for text in (readme, guide):
        assert "theta_final.pt" in text
        assert "raw tensor export" in text
        assert "does not carry run configuration" in text
        assert "family ordering" in text
        assert "species ordering" in text
        assert "checkpoints/best.pt" in text
        assert "checkpoints/latest.pt" in text


def test_readme_scopes_example_config_to_source_artifacts():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "For a source checkout or source archive" in readme
    assert "examples/minimal-run-config.json" in readme
    assert "Installed wheels do not install the `examples/` directory" in readme
    assert '"species_tree": "S.tree"' in readme
    assert '"families_file": "families.txt"' in readme


def test_readme_documents_installed_sampling_binary_setup():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    for text in (readme, guide):
        normalized = " ".join(text.split())
        assert "Wheels currently do not ship" in normalized
        assert "prebuilt binary" in normalized
    assert "### Sampling Binary Setup" in readme
    assert (
        "cargo build --locked --release --manifest-path "
        "crates/gpurec-backtrack/Cargo.toml"
    ) in readme
    assert "GPUREC_BACKTRACK_BIN" in readme
    assert "--backtrack-binary" in readme
    assert "fallback works from a source checkout or unpacked\nsource archive" in readme
    assert "unpacked-source-archive `cargo run`\n  fallback" in guide

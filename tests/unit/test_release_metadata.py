from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import gpurec

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = ROOT / "scripts" / "check_release_metadata.py"
SUBPROCESS_TIMEOUT = 60


def _load_check_release_metadata_module():
    spec = importlib.util.spec_from_file_location(
        "check_release_metadata_under_test",
        CHECK_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load release metadata checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cpu_ci_workflow() -> dict:
    text = (ROOT / ".github" / "workflows" / "cpu-unit.yml").read_text(
        encoding="utf-8"
    )
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise AssertionError("CPU CI workflow must parse as a YAML mapping")
    return loaded


def _workflow_step(job: dict, name: str) -> dict:
    for step in job.get("steps", []):
        if step.get("name") == name:
            return step
    raise AssertionError(f"workflow job is missing step {name!r}")


def _step_run(job: dict, name: str) -> str:
    step = _workflow_step(job, name)
    run = step.get("run")
    assert isinstance(run, str), f"workflow step {name!r} must have run script"
    return run


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
    license_line: str = 'license = { file = "LICENSE" }',
    create_readme: bool = True,
    urls_block: str | None = None,
) -> None:
    (root / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    if create_readme:
        (root / "README.md").write_text("# fixture\n", encoding="utf-8")
    readme_block = f"{readme_line}\n" if readme_line else ""
    if urls_block is None:
        urls_block = """
[project.urls]
Repository = "https://example.invalid/repo"
Issues = "https://example.invalid/issues"
Documentation = "https://example.invalid/docs"
""".lstrip()
    (root / "pyproject.toml").write_text(
        f"""
[project]
name = "fixture"
version = "0.0.0"
description = "fixture"
{readme_block}requires-python = ">=3.10,<3.13"
{license_line}
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

{urls_block}
""".lstrip(),
        encoding="utf-8",
    )


def test_release_metadata_check_reports_only_current_license_blockers():
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
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


def test_top_level_package_version_matches_project_metadata():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M)

    assert match is not None
    assert gpurec.__version__ == match.group(1)


def test_release_metadata_check_accepts_complete_metadata_fixture(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path)

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
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
        timeout=SUBPROCESS_TIMEOUT,
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
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "declared readme file does not exist: MISSING.md" in result.stdout
    assert "license" not in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("repository_line", "message"),
    [
        ('Repository = ""', "Repository must be an http(s) URL"),
        ('Repository = "not-a-url"', "Repository must be an http(s) URL"),
        (
            'Repository = "ftp://example.invalid/repo"',
            "Repository must be an http(s) URL",
        ),
    ],
)
def test_release_metadata_check_requires_usable_project_url_values(
    tmp_path: Path,
    repository_line: str,
    message: str,
):
    _write_complete_release_metadata_fixture(
        tmp_path,
        urls_block=f"""
[project.urls]
{repository_line}
Issues = "https://example.invalid/issues"
Documentation = "https://example.invalid/docs"
""".lstrip(),
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert message in result.stdout
    assert "license" not in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_project_urls_table(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        urls_block='urls = "https://example.invalid/repo"',
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "[project.urls] must be a table" in result.stdout
    assert "license" not in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_declared_license_file(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        license_line='license = { file = "MISSING-LICENSE" }',
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "declared license file does not exist: MISSING-LICENSE" in result.stdout
    assert "missing top-level LICENSE file" not in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_accepts_license_text_fixture(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        license_line='license = { text = "fixture license text" }',
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0
    assert "release metadata check passed" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_rejects_empty_license_text(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        license_line='license = { text = "" }',
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "license.text must be nonempty" in result.stdout
    assert "missing top-level LICENSE file" not in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_rejects_license_directory(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        license_line='license = { text = "fixture license text" }',
    )
    (tmp_path / "LICENSE").unlink()
    (tmp_path / "LICENSE").mkdir()

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing top-level LICENSE file" in result.stdout
    assert "license metadata" not in result.stdout
    assert "license classifier" not in result.stdout
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
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0
    assert "release metadata check passed" in result.stdout
    assert result.stderr == ""


def test_minimal_pyproject_parser_extracts_release_metadata_subset():
    checker = _load_check_release_metadata_module()
    text = """
[build-system]
requires = ["setuptools>=68.0", "wheel"]

[project]
name = "fixture"
readme = { file = "README.md" }
license = { text = "fixture license text" }
authors = [{ name = "Fixture Maintainer" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
]

[project.urls]
Repository = "https://example.invalid/repo"
Issues = "https://example.invalid/issues"
Documentation = "https://example.invalid/docs"

[project.scripts]
gpurec = "gpurec.cli:main"
""".lstrip()

    parsed = checker._parse_minimal_pyproject(text)

    project = parsed["project"]
    assert project["readme"] == {"file": "README.md"}
    assert project["license"] == {"text": "fixture license text"}
    assert project["authors"] == '[{ name = "Fixture Maintainer" }]'
    assert project["classifiers"] == [
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ]
    assert project["urls"] == {
        "Repository": "https://example.invalid/repo",
        "Issues": "https://example.invalid/issues",
        "Documentation": "https://example.invalid/docs",
    }
    assert "gpurec" not in project


def test_minimal_pyproject_parser_supports_current_project_release_fields():
    checker = _load_check_release_metadata_module()
    parsed = checker._parse_minimal_pyproject(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    project = parsed["project"]
    assert project["readme"] == "README.md"
    assert project["authors"] == '[{ name = "SisyphusMountain" }]'
    assert project["urls"]["Repository"] == "https://github.com/SisyphusMountain/gpurec"
    assert project["urls"]["Issues"] == (
        "https://github.com/SisyphusMountain/gpurec/issues"
    )
    assert project["urls"]["Documentation"] == (
        "https://github.com/SisyphusMountain/gpurec#readme"
    )
    for required in (
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ):
        assert required in project["classifiers"]
    assert "license" not in project


def test_cpu_ci_builds_and_smokes_release_artifacts():
    workflow = _load_cpu_ci_workflow()
    package = workflow["jobs"]["package"]

    assert package["strategy"]["matrix"]["python-version"] == ["3.10", "3.12"]
    assert _workflow_step(package, "Set up Python")["with"][
        "python-version"
    ] == "${{ matrix.python-version }}"

    install = _step_run(package, "Install package, runtime, and release dependencies")
    assert 'python -m pip install -e ".[release]"' in install

    build = _step_run(package, "Build source and wheel artifacts")
    assert "rm -rf dist" in build
    assert "python -m build" in build

    assert _step_run(package, "Check artifact metadata") == "python -m twine check dist/*"

    artifact_check = _step_run(package, "Check artifact package data")
    for required in (
        "import json",
        "tarfile.open",
        "zipfile.ZipFile",
        "required_sdist = required_wheel +",
        "docs/README.md",
        "docs/input-preparation.md",
        "docs/lean-fast-path.md",
        "docs/optimization-workflow-call-graph.md",
        "docs/output-artifacts.md",
        "docs/production-optimization-guide.md",
        "docs/professionalization-audit-progress.tex",
        "docs/release-readiness.md",
        "docs/troubleshooting.md",
        "examples/README.md",
        "examples/minimal-run-config.json",
        "examples/specieswise-adagrad-restarts-config.json",
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
        "forbidden_wheel_names",
        "gpurec-backtrack.exe",
        "wheel includes Rust backtracking binaries",
        "crates/gpurec-preprocess/Cargo.toml",
        "crates/gpurec-preprocess/Cargo.lock",
        "crates/gpurec-preprocess/src/lib.rs",
        "crates/gpurec-preprocess/src/main.rs",
        "examples/",
        "example_configs",
        "configs.append",
        "json.load",
        'for field in ("species_tree", "families_file")',
        "example config targets missing from sdist",
        "sdist missing required source files",
        "sdist includes forbidden paths",
        "wheel missing required package data",
        "wheel includes forbidden paths",
    ):
        assert required in artifact_check

    assert _workflow_step(
        package,
        "Smoke source archive Rust crates and examples",
    )["name"] == (
        "Smoke source archive Rust crates and examples"
    )
    rust_smoke = _step_run(package, "Smoke source archive Rust crates and examples")
    for required in (
        "SDIST_UNPACK_DIR",
        'cargo test --locked --manifest-path "$root/crates/gpurec-backtrack/Cargo.toml"',
        (
            'cargo run --locked --quiet --manifest-path "$root/crates/gpurec-backtrack/'
            'Cargo.toml" -- --help'
        ),
        'cargo test --locked --manifest-path "$root/crates/gpurec-preprocess/Cargo.toml"',
        (
            'cargo run --locked --quiet --manifest-path "$root/crates/gpurec-preprocess/'
            'Cargo.toml" -- --help'
        ),
        'cd "$root"',
        (
            "python -m gpurec.cli validate-config --config "
            "examples/minimal-run-config.json --check-preprocess"
        ),
        "optimizer=hessian-sgd",
        "hessian_sgd_normal_fixed_iters_pi=full",
        "cuda_backward_ready=false",
        (
            "python -m gpurec.cli validate-config --config "
            "examples/specieswise-adagrad-restarts-config.json --check-preprocess"
        ),
        "optimizer=adagrad-restarts",
        "adagrad_restart_schedule=8:1:60,16:0.5:35,32:0.5:30",
        "adagrad_restart_total_steps=125",
        "optimizer_step_cap=125",
        "optimizer_step_cap_reason=adagrad_restart_schedule",
        "cuda_backward_ready=false",
        "preprocess_checked=true",
    ):
        assert required in rust_smoke

    wheel_smoke = _step_run(package, "Install built wheel and smoke CLI")
    for required in (
        "python -m pip install --no-deps dist/*.whl",
        "smoke_dir=$(mktemp -d)",
        'cd "$smoke_dir"',
        "gpurec --help",
        "python -m gpurec.cli --help",
        "gpurec config-template --help",
        "gpurec config-template --mode genewise",
        "genewise-config-template.json",
        '"optimizer": "auto"',
        '"mode": "genewise"',
        '"solver_warmup_iters": 4',
        '"hessian_sgd_normal_fixed_iters_pi": null',
        "gpurec config-template --mode specieswise",
        "specieswise-config-template.json",
        '"mode": "specieswise"',
        '"adagrad_restart_schedule": "8:1.0:60,16:0.5:35,32:0.5:30"',
        '"adagrad_restart_final_check_iters": 128',
        "gpurec optimize --help",
        "optimize-help.txt",
        'grep -q -- "--require-final-check-ok" optimize-help.txt',
        "gpurec validate-config --help",
        "validate-config-help.txt",
        "--require-cuda-backward-ready",
        "gpurec summary-info --help",
        "summary-info-help.txt",
        "--summary",
        "--require-converged",
        'grep -q -- "--require-final-check-ok" summary-info-help.txt',
        "summary.json",
        "gpurec checkpoint-info --help",
        "checkpoint-info-help.txt",
        'grep -q -- "--require-final-check-ok" checkpoint-info-help.txt',
        "gpurec sample --help",
        "gpurec run --help",
        "run-help.txt",
        'grep -q -- "--require-final-check-ok" run-help.txt',
        "gpurec backtrack-check --help",
        "--backtrack-binary",
        "GPUREC_BACKTRACK_BIN",
        "checkpoints/best.pt",
        "checkpoints/latest.pt",
        "backtrack-check.txt",
        "unset GPUREC_BACKTRACK_BIN",
        'test "$status" -eq 1',
        "import gpurec",
        'Path(os.environ["GITHUB_WORKSPACE"]).resolve()',
        "package_path.is_relative_to(workspace)",
        '"site-packages" not in package_path.parts',
        '"dist-packages" not in package_path.parts',
        "imported gpurec from checkout",
        "import gpurec.workflow as workflow",
        "for name in gpurec.__all__:",
        "top-level exports missing from dir(gpurec)",
        "for name in workflow.__all__:",
        "getattr(workflow, name)",
        "workflow exports missing from dir(gpurec.workflow)",
        "exec(\"from gpurec.workflow import *\", namespace)",
        "workflow wildcard mismatch",
        "workflow export missing from gpurec.__all__",
        "top-level workflow export mismatch",
        "exports_ok",
    ):
        assert required in wheel_smoke
    assert wheel_smoke.index(
        "python -m pip install --no-deps dist/*.whl"
    ) < wheel_smoke.index(
        'cd "$smoke_dir"'
    )
    assert wheel_smoke.index('cd "$smoke_dir"') < wheel_smoke.index("gpurec --help")


def test_cpu_ci_workflow_uses_minimal_permissions_and_concurrency():
    workflow = _load_cpu_ci_workflow()

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {
        "group": "${{ github.workflow }}-${{ github.ref }}",
        "cancel-in-progress": True,
    }


def test_manifest_includes_documented_examples_in_source_archive():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "recursive-include examples" in manifest
    for pattern in ("*.json", "*.map", "*.nwk", "*.txt", "*.md"):
        assert pattern in manifest


def test_manifest_includes_current_docs_in_source_archive():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "recursive-include docs *.md *.tex" in manifest


def test_manifest_includes_rust_backtracking_crate_in_source_archive_only():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "include crates/gpurec-backtrack/Cargo.toml" in manifest
    assert "include crates/gpurec-backtrack/Cargo.lock" in manifest
    assert "recursive-include crates/gpurec-backtrack/src *.rs" in manifest
    assert "prune crates/gpurec-backtrack/target" in manifest


def test_manifest_includes_rust_preprocess_crate_in_source_archive_only():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "include crates/gpurec-preprocess/Cargo.toml" in manifest
    assert "include crates/gpurec-preprocess/Cargo.lock" in manifest
    assert "recursive-include crates/gpurec-preprocess/src *.rs" in manifest
    assert "prune crates/gpurec-preprocess/target" in manifest


@pytest.mark.parametrize(
    "command",
    (
        ("gpurec", "--help"),
        (sys.executable, "-m", "gpurec.cli", "--help"),
    ),
)
def test_cli_help_smokes_are_quiet_on_cpu(command: tuple[str, ...]):
    if command[0] == "gpurec":
        script = shutil.which("gpurec")
        if script is None:
            pytest.skip("gpurec console script is not installed")
        try:
            script_text = Path(script).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            script_text = ""
        if "gpurec.cli.reconcile" in script_text:
            pytest.skip("gpurec console script points at stale pre-audit entry point")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0
    assert "usage: gpurec" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("code", "expected_stdout"),
    (
        (
            "\n".join(
                (
                    "import gpurec",
                    "for name in gpurec.__all__:",
                    "    getattr(gpurec, name)",
                    "print('exports_ok')",
                )
            ),
            "exports_ok",
        ),
        (
            "\n".join(
                (
                    "import gpurec.workflow as workflow",
                    "for name in workflow.__all__:",
                    "    getattr(workflow, name)",
                    "print('workflow_exports_ok')",
                )
            ),
            "workflow_exports_ok",
        ),
        (
            "\n".join(
                (
                    "import gpurec",
                    "import gpurec.workflow as workflow",
                    "for name in workflow.__all__:",
                    "    if name not in gpurec.__all__:",
                    "        raise SystemExit(f'workflow export missing from gpurec.__all__: {name}')",
                    "    if getattr(gpurec, name) is not getattr(workflow, name):",
                    "        raise SystemExit(f'top-level workflow export mismatch: {name}')",
                    "print('workflow_identity_ok')",
                )
            ),
            "workflow_identity_ok",
        ),
        (
            "\n".join(
                (
                    "import gpurec",
                    "namespace = {}",
                    "exec('from gpurec import *', namespace)",
                    "exported = {name for name in namespace if not name.startswith('__')}",
                    "if exported != set(gpurec.__all__):",
                    "    raise SystemExit(f'wildcard mismatch: {sorted(exported ^ set(gpurec.__all__))}')",
                    "print('wildcard_ok')",
                )
            ),
            "wildcard_ok",
        ),
        (
            "\n".join(
                (
                    "import gpurec.workflow as workflow",
                    "namespace = {}",
                    "exec('from gpurec.workflow import *', namespace)",
                    "exported = {name for name in namespace if not name.startswith('__')}",
                    "if exported != set(workflow.__all__):",
                    "    raise SystemExit(f'workflow wildcard mismatch: {sorted(exported ^ set(workflow.__all__))}')",
                    "print('workflow_wildcard_ok')",
                )
            ),
            "workflow_wildcard_ok",
        ),
    ),
)
def test_public_import_smokes_are_quiet_on_cpu(
    code: str,
    expected_stdout: str,
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == expected_stdout
    assert result.stderr == ""


def test_cpu_ci_matrix_covers_declared_python_versions():
    workflow = _load_cpu_ci_workflow()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    supported_versions = ("3.10", "3.11", "3.12")
    unit = workflow["jobs"]["unit"]

    assert unit["strategy"]["matrix"]["python-version"] == list(supported_versions)
    assert _workflow_step(unit, "Set up Python")["with"][
        "python-version"
    ] == "${{ matrix.python-version }}"
    assert pyproject["project"]["requires-python"] == ">=3.10,<3.13"
    classifiers = pyproject["project"]["classifiers"]
    for version in supported_versions:
        assert f"Programming Language :: Python :: {version}" in classifiers


def test_dev_extra_installs_tomli_for_python310_toml_tests():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]

    assert any(
        dependency.startswith("tomli;")
        and "python_version" in dependency
        and "< '3.11'" in dependency
        for dependency in dev_dependencies
    )


def test_readme_install_docs_match_declared_python_range():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requires_python = re.search(
        r'^requires-python\s*=\s*"([^"]+)"',
        pyproject,
        flags=re.M,
    )
    supported_versions = tuple(
        re.findall(r"Programming Language :: Python :: (\d+\.\d+)", pyproject)
    )

    assert requires_python is not None
    assert supported_versions == ("3.10", "3.11", "3.12")
    assert f'`requires-python = "{requires_python.group(1)}"`' in readme
    assert f"Python {supported_versions[0]}-{supported_versions[-1]}" in readme


def test_runtime_dependencies_do_not_include_removed_extension_build_backend():
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

    assert "ninja" not in dependency_names


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
    workflow = _load_cpu_ci_workflow()
    rust = workflow["jobs"]["rust-backtrack"]

    assert _step_run(rust, "Set up Rust") == "rustup default stable"
    assert _workflow_step(rust, "Set up Python")["uses"] == "actions/setup-python@v5"
    assert _workflow_step(rust, "Set up Python")["with"]["python-version"] == "3.10"

    install = _step_run(rust, "Install Python contract-test dependency")
    assert "python -m pip install --upgrade pip" in install
    assert "python -m pip install pytest" in install
    assert (
        _step_run(rust, "Run Rust backtracking tests")
        == "cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml"
    )
    assert (
        _step_run(rust, "Smoke Rust backtracking CLI")
        ==
        "cargo run --locked --quiet --manifest-path "
        "crates/gpurec-backtrack/Cargo.toml -- --help"
    )
    assert (
        _step_run(rust, "Run Rust backtracking fixture")
        == "pytest -q tests/integration/test_rust_backtracking_fixture.py"
    )
    all_run_scripts = "\n".join(
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if isinstance(step.get("run"), str)
    )
    assert 'pytest -q -m "integration and not gpu"' not in all_run_scripts


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
    assert "`--max-events` support" in notes
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
        r"Backtracking fixture smokes should use.*?```bash\n(?P<block>.*?)```",
        readme,
        flags=re.S,
    )

    assert match is not None
    block = match.group("block")
    assert (
        "tests/integration/test_rust_backtracking_fixture.py::"
        "test_rust_backtracking_cli_reads_json_fixture_and_writes_recphyloxml"
    ) in block
    assert "test_stochastic_backtracking.py" not in block


def test_release_readiness_orders_clean_checkout_before_build():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    preview_command = "git clean -Xdn -- build dist '*.egg-info'"
    clean_command = "git clean -Xdf -- build dist '*.egg-info'"

    assert "Python 3.10 and 3.12" in guide
    assert guide.index(preview_command) < guide.index(clean_command)
    assert guide.index(clean_command) < guide.index("python -m build")
    assert "stale `build/`, `dist/`, or `*.egg-info/`" in guide
    assert "gpurec --help" in guide
    assert "python -m gpurec.cli --help" in guide
    assert "pytest -q tests/integration/test_rust_backtracking_fixture.py" in guide


def test_release_readiness_preserves_license_no_publish_blocker():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    for token in (
        "Required Before Redistribution",
        "Choose and add a project license",
        "top-level `LICENSE` file",
        "matching `pyproject.toml` license metadata",
        "license classifier",
        "currently expected to fail",
        "Do not bypass it for redistribution",
        "Do not publish artifacts until the license",
    ):
        assert token in guide


def test_release_readiness_smokes_top_level_exports():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    assert "import gpurec.workflow as workflow" in guide
    assert "for name in gpurec.__all__" in guide
    assert "getattr(gpurec, name)" in guide
    assert "for name in workflow.__all__" in guide
    assert "getattr(workflow, name)" in guide
    assert "exec(\"from gpurec.workflow import *\", namespace)" in guide
    assert "workflow wildcard mismatch" in guide
    assert "top-level workflow export mismatch" in guide


def test_release_readiness_documents_installed_wheel_smoke():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    for expected in (
        "python -m pip install --no-deps dist/*.whl",
        "smoke_dir=$(mktemp -d)",
        'cd "$smoke_dir"',
        "gpurec config-template --mode genewise",
        "gpurec config-template --mode specieswise",
        "gpurec optimize --help",
        "optimization convergence and final-check gates",
        "final-check gates",
        "gpurec validate-config --help",
        "--require-cuda-backward-ready",
        "gpurec summary-info --help",
        "--require-converged",
        "--require-final-check-ok",
        "gpurec checkpoint-info --help",
        "checkpoint final-check gate",
        "gpurec sample --help",
        "gpurec run --help",
        "pre-sampling convergence and final-check gates",
        "gpurec backtrack-check --help",
        "package_path = Path(gpurec.__file__).resolve()",
        "package_path.is_relative_to(repo_root)",
        "imported gpurec from checkout",
        "site-packages",
        "dist-packages",
        "for name in gpurec.__all__",
        "for name in workflow.__all__",
        "top-level exports missing from dir(gpurec)",
        "workflow exports missing from dir(gpurec.workflow)",
        "workflow wildcard mismatch",
        "exports_ok",
    ):
        assert expected in guide


def test_release_readiness_documents_source_archive_preprocess_smoke():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    for expected in (
        "source-archive `validate-config --check-preprocess`",
        "examples/minimal-run-config.json",
        "examples/specieswise-adagrad-restarts-config.json",
        "cuda_backward_ready=false",
        "--require-cuda-backward-ready",
    ):
        assert expected in guide


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
    examples_readme = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    workflow = _load_cpu_ci_workflow()
    artifact_check = _step_run(
        workflow["jobs"]["package"],
        "Check artifact package data",
    )

    assert "For a source checkout or source archive" in readme
    assert "checked JSON configs and a tiny\nAleRax-style fixture" in readme
    assert "examples/minimal-run-config.json" in readme
    assert "examples/specieswise-adagrad-restarts-config.json" in readme
    assert (
        "gpurec validate-config --config examples/minimal-run-config.json "
        "--check-preprocess"
    ) in readme
    assert "gpurec optimize --config examples/minimal-run-config.json" not in readme
    assert "examples/README.md" in readme
    assert "gpurec config-template --mode genewise --output run.json" in readme
    assert (
        "gpurec config-template --mode specieswise --output specieswise-run.json"
        in readme
    )
    assert '"optimizer": "auto"' in readme
    assert "`hessian-sgd` and `mode=specieswise` resolves to `adagrad-restarts`" in readme
    assert "source-tree config/parser fixture" in readme
    assert 'sets `"device": "cuda"`' in readme
    assert "not a CPU fallback" in readme
    assert "S > 256" in readme
    assert "not an end-to-end optimizer smoke" in readme
    assert "Installed wheels do not install the `examples/` directory" in readme
    assert "Installed wheels intentionally do not install this directory" in examples_readme
    assert '"species_tree": "S.tree"' in readme
    assert '"families_file": "families.txt"' in readme
    assert '"examples/"' in artifact_check
    assert "example_configs" in artifact_check
    assert "json.load" in artifact_check
    assert "example config targets missing from sdist" in artifact_check


def test_readme_documents_installed_sampling_binary_setup():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    normalized_guide = " ".join(guide.split())

    for text in (readme, guide):
        normalized = " ".join(text.split())
        assert "Wheels currently do not ship" in normalized
    assert "prebuilt binary" in normalized
    assert "### Sampling Binary Setup" in readme
    assert "`gpurec sample` and the sampling phase of `gpurec run`" in readme
    assert (
        "`config-template`, `validate-config`,\n  `optimize`, "
        "`summary-info`, `checkpoint-info`, `sample`, `run`, and\n  "
        "`backtrack-check` commands"
    ) in readme
    assert "gpurec validate-config --config examples/minimal-run-config.json" in readme
    assert "--check-preprocess" in readme
    assert "retained Rust parser to run on\nCPU" in readme
    assert "CPU-safe path/reference preflight" in readme
    assert "In a wheel-only\ninstall" in readme
    assert "For a source checkout or unpacked source archive" in readme
    assert (
        "cargo build --locked --release --manifest-path "
        "crates/gpurec-backtrack/Cargo.toml"
    ) in readme
    assert "GPUREC_BACKTRACK_BIN" in readme
    assert "--backtrack-binary" in readme
    assert "gpurec backtrack-check" in readme
    assert "The same `GPUREC_BACKTRACK_BIN` environment variable" in readme
    assert "fallback works from a source checkout or unpacked\nsource archive" in readme
    assert "unpacked-source-archive `cargo run` fallback" in normalized_guide
    assert (
        "installed `gpurec config-template --help`"
        in normalized_guide
    )
    assert (
        "`gpurec config-template --mode specieswise`, and "
        "`gpurec validate-config --help`"
        in normalized_guide
    )
    assert "`gpurec summary-info --help`" in normalized_guide
    assert "`gpurec checkpoint-info --help`" in normalized_guide
    assert (
        "installed `gpurec sample --help`, `gpurec run --help`, and "
        "`gpurec backtrack-check`"
        in normalized_guide
    )
    assert "gpurec backtrack-check --help" in guide

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
    create_changelog: bool = True,
    create_citation: bool = True,
    create_dockerfile: bool = True,
    create_release_notes: bool = True,
    create_support_policy: bool = True,
    create_versioning_policy: bool = True,
    create_publication_checklist: bool = True,
    create_platform_matrix: bool = True,
    create_api_contract: bool = True,
    create_known_limitations: bool = True,
    create_bioinformatics_quickstart: bool = True,
    create_input_preparation: bool = True,
    create_output_artifacts: bool = True,
    create_long_validation_workflow: bool = True,
    create_validation_envelope: bool = True,
    create_troubleshooting: bool = True,
    create_docs_readme: bool = True,
    create_production_optimization_guide: bool = True,
    create_glossary: bool = True,
    create_workflow_examples: bool = True,
    create_optimization_workflow_call_graph: bool = True,
    create_lean_fast_path: bool = True,
    create_professionalization_audit_progress: bool = True,
    create_dependency_inventory_script: bool = True,
    create_release_readiness: bool = True,
    urls_block: str | None = None,
    scripts_block: str | None = None,
    project_extra: str = "",
) -> None:
    (root / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    if create_readme:
        (root / "README.md").write_text(
            "\n".join(
                [
                    "# fixture",
                    "",
                    "Short user path: install, validate inputs, run optimization,",
                    "inspect output, sample reconciliations.",
                    "",
                    "CLI exit codes are stable for workflow managers:",
                    "- `0`: command completed successfully.",
                    "- `1`: command ran but failed a runtime or validation gate",
                    "- `2`: CLI usage or argument parsing error from `argparse`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_changelog:
        (root / "CHANGELOG.md").write_text(
            "# Changelog\n\n## 0.0.0 - 2026-01-01\n\n- fixture\n",
            encoding="utf-8",
        )
    if create_citation:
        (root / "CITATION.cff").write_text(
            "\n".join(
                [
                    "cff-version: 1.2.0",
                    'title: "fixture"',
                    'repository-code: "https://example.invalid/repo"',
                    'version: "0.0.0"',
                    "preferred-citation:",
                    '  title: "fixture"',
                    '  type: software',
                    '  version: "0.0.0"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_release_notes:
        (root / "docs" / "release-notes.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "release-notes.md").write_text(
            "\n".join(
                [
                    "# Release Notes",
                    "",
                    "## 0.0.0 - 2026-01-01",
                    "",
                    "- Added",
                    "- Dependency and Python/Torch/CUDA support updates",
                    "- Known limitations",
                    "- Migration notes",
                    "- Release artifact notes",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_release_readiness:
        (root / "docs" / "release-readiness.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "release-readiness.md").write_text(
            "\n".join(
                [
                    "# Release Readiness",
                    "",
                    "python scripts/check_release_metadata.py",
                    "scripts/run_long_validation.py",
                    "validation-envelope.md",
                    "gpurec doctor",
                    "Quick PR checks",
                    "Nightly checks",
                    "Release-candidate checks",
                    "Final publication checks",
                    "Checksums and provenance evidence are required.",
                    "sha256sum dist/* > dist/SHA256SUMS",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_dockerfile:
        (root / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    if create_support_policy:
        (root / "docs" / "support-policy.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "support-policy.md").write_text(
            "\n".join(
                [
                    "# Support Policy",
                    "",
                    "production support scope applies to documented surfaces.",
                    "The latest release tag is the primary supported line.",
                    "Support window covers Python, PyTorch, CUDA, and native artifact versions.",
                    "Release and patch policy: older tags may receive backports.",
                    "Support evidence includes summary.json and run_manifest.json.",
                    "Support evidence includes checkpoint metadata when applicable.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_versioning_policy:
        (root / "docs" / "versioning-policy.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "versioning-policy.md").write_text(
            "\n".join(
                [
                    "# Versioning Policy",
                    "",
                    "This project follows semantic versioning.",
                    "Version format is MAJOR.MINOR.PATCH.",
                    "The latest release tag is the primary supported line.",
                    "Backports to older tags are best-effort.",
                    "Version consistency checks include pyproject.toml,",
                    "gpurec.__version__, and release notes heading alignment.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_publication_checklist:
        (root / "docs" / "publication-checklist.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "publication-checklist.md").write_text(
            "\n".join(
                [
                    "# Publication Checklist",
                    "",
                    "Reference CITATION.cff for software citation metadata.",
                    "Archive run_manifest.json and summary.json for reproducibility.",
                    "Record gpurec doctor --json and gpurec summary-info --summary output_gpurec/summary.json --json.",
                    "Archive history.jsonl and checkpoints/ for rerun audit trails.",
                    "Archive checksums and provenance evidence with publication artifacts.",
                    "Report known-limitations.md caveats and release-notes.md migration notes.",
                    "Run scripts/validate_output_artifacts.py before publication.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_platform_matrix:
        (root / "docs" / "platform-matrix.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "platform-matrix.md").write_text(
            "\n".join(
                [
                    "# Platform Matrix",
                    "",
                    "Primary supported configuration uses Linux x86_64,",
                    "Python 3.10-3.12, PyTorch + Triton, CUDA-capable NVIDIA GPU runtime, and",
                    "source-built native preprocessing/backtracking artifacts with Rust/Cargo compiler toolchain.",
                    "",
                    "## Offline Installation Policy",
                    "",
                    "Offline installation is not currently supported as a production guarantee.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_api_contract:
        (root / "docs" / "api-contract.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "api-contract.md").write_text(
            "\n".join(
                [
                    "# API Contract",
                    "",
                    "## CLI output modes",
                    "",
                    "`--json` mode is supported by validate-config, doctor,",
                    "checkpoint-info, and summary-info.",
                    "JSON mode emits single JSON objects with stable keys.",
                    "JSON mode is the required machine path for automation.",
                    "Compatibility policy covers config fields, CLI flags,",
                    "Python imports, and output artifacts.",
                    "Deprecation warnings and migration notes are required",
                    "before removing supported behavior.",
                    "Exit status `0` indicates success.",
                    "Exit status `1` indicates runtime and route-validation failures.",
                    "Exit status `2` indicates CLI parse/config errors.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_known_limitations:
        (root / "docs" / "known-limitations.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "known-limitations.md").write_text(
            "\n".join(
                [
                    "# Known Limitations",
                    "",
                    "CUDA-only production route with S > 256 gate.",
                    "Parser Newick subset limits are explicit: unsupported quoted labels",
                    "and embedded delimiters, nested comments, NHX/BEAST metadata,",
                    "unary species nodes, and non-binary species trees.",
                    "Wheel installs may require external native artifacts.",
                    "bf16 remains experimental.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_bioinformatics_quickstart:
        (root / "docs" / "bioinformatics-quickstart.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "bioinformatics-quickstart.md").write_text(
            "\n".join(
                [
                    "# Bioinformatics Quickstart",
                    "",
                    "Create config, validate, run, resume, inspect, sample, archive.",
                    "Installation decision tree for source checkout or source archive,",
                    "wheel-only environment, cluster/container workflows, and",
                    "offline installation policy.",
                    "Run gpurec preprocess-check and gpurec backtrack-check",
                    "as installation verification commands.",
                    "Structured JSON mode includes gpurec doctor --json,",
                    "gpurec validate-config --config run.json --json,",
                    "gpurec summary-info --summary output_gpurec/summary.json --json,",
                    "and gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt --json.",
                    "RNG behavior keeps a sampling seed for reproducibility.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_input_preparation:
        (root / "docs" / "input-preparation.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "input-preparation.md").write_text(
            "\n".join(
                [
                    "# Input Preparation",
                    "",
                    "Use --max-families to sample the first `N` families.",
                    "Use preprocess outputs as a memory estimate and tune",
                    "clade_budget plus family_chunk_size for large runs.",
                    "Family-file guidance covers multiple families, multiple trees per family,",
                    "and mapping files.",
                    "Conversion guidance covers Treerecs, GeneRax, AleRax,",
                    "OrthoFinder, and gene -> species TSV mappings.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_output_artifacts:
        (root / "docs" / "output-artifacts.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "output-artifacts.md").write_text(
            "\n".join(
                [
                    "# Output Artifacts",
                    "",
                    "Output artifacts follow stable schemas or compatibility rules.",
                    "Example output snippets for summary.json, rates_final.tsv,",
                    "per_fam_likelihoods.tsv, and a RecPhyloXML output snippet.",
                    "Run directory structure uses output_gpurec/, checkpoints/,",
                    "and reconciliations/.",
                    "Input/output flow uses validate-config --check-preprocess,",
                    "gpurec optimize, gpurec sample, and reconciliations/*.xml.",
                    "run_manifest.json records package version, native artifact",
                    "metadata, PyTorch version, CUDA availability, GPU name,",
                    "command line invocation, config hash, random seed fields,",
                    "selected route metadata, and evidence to reproduce or audit runs.",
                    "theta_final.pt is for inspection only, and a checkpoint is required",
                    "for resume, route checks, or sampling.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_long_validation_workflow:
        (root / "docs" / "long-validation-workflow.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "long-validation-workflow.md").write_text(
            "\n".join(
                [
                    "# Long Validation Workflow",
                    "",
                    "Use this report as benchmark evidence, not a hard performance guarantee",
                    "and not a guaranteed performance contract.",
                    "gpurec doctor --json",
                    "gpurec validate-config --check-preprocess --require-cuda-backward-ready",
                    "gpurec optimize --require-final-check-ok",
                    "gpurec summary-info --require-converged --require-final-check-ok",
                    "gpurec sample --checkpoint",
                    "scripts/validate_output_artifacts.py",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_validation_envelope:
        (root / "docs" / "validation-envelope.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "validation-envelope.md").write_text(
            "\n".join(
                [
                    "# Validation Envelope",
                    "",
                    "Runtime envelope, peak memory evidence, final NLL range, and benchmark evidence scope.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_troubleshooting:
        (root / "docs" / "troubleshooting.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "troubleshooting.md").write_text(
            "\n".join(
                [
                    "# Troubleshooting",
                    "",
                    "Organized by symptom for operator triage.",
                    "Retryable runtime failures vs input contract failures.",
                    "Use likely cause and next action columns in issue triage tables.",
                    "Authoritative files: summary.json, history.jsonl, checkpoints/latest.pt.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_docs_readme:
        (root / "docs" / "README.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "README.md").write_text(
            "\n".join(
                [
                    "# Documentation Map",
                    "",
                    "This map separates stable user workflows from HOGENOM-only research scripts.",
                    "Use CLI help as primary user-facing command reference.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_production_optimization_guide:
        (root / "docs" / "production-optimization-guide.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "production-optimization-guide.md").write_text(
            "\n".join(
                [
                    "# Production Optimization Guide",
                    "",
                    "Recommended defaults by user goal: exploratory run,",
                    "production genewise run, production specieswise run,",
                    "diagnostics-only global run.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_glossary:
        (root / "docs" / "glossary.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "glossary.md").write_text(
            "\n".join(
                [
                    "# Glossary",
                    "",
                    "`D` `T` `L` `DTL` `CCP` `specieswise` `genewise` `global`",
                    "`RecPhyloXML` `NLL` `route` `solver budget` `checkpoint`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_workflow_examples:
        (root / "docs" / "workflow-examples" / "README.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "workflow-examples" / "README.md").write_text(
            "\n".join(
                [
                    "# Workflow Examples",
                    "",
                    "Tracked mini public dataset with deterministic workflow fixtures.",
                    "Snakemake and Nextflow references fail fast on bad config,",
                    "run gpurec validate-config --check-preprocess in preflight,",
                    "resume from a checkpoint, and reject non-converged outputs.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (root / "docs" / "workflow-examples" / "end-to-end-tutorial").mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "workflow-examples" / "end-to-end-tutorial" / "README.md").write_text(
            "\n".join(
                [
                    "# End-to-End Tutorial",
                    "",
                    "First successful run tutorial uses only public commands.",
                    "Tracked or downloadable dataset that writes outputs and samples RecPhyloXML.",
                    "gpurec validate-config",
                    "--check-preprocess",
                    "--require-cuda-backward-ready",
                    "gpurec optimize",
                    "gpurec optimize --resume-from output_gpurec/checkpoints/latest.pt",
                    "gpurec sample",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (root / "docs" / "workflow-examples" / "end-to-end-tutorial" / "run.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (root / "docs" / "workflow-examples" / "end-to-end-tutorial" / "generate_dataset.py").write_text(
            "print('fixture')\n", encoding="utf-8"
        )
        (root / "docs" / "workflow-examples" / "input-validation-fixtures").mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "workflow-examples" / "input-validation-fixtures" / "README.md").write_text(
            "\n".join(
                [
                    "# Input Validation Fixtures",
                    "",
                    "validate-inputs checks run without constructing a CUDA model.",
                    "Issue entries include file path, family name, affected label,",
                    "expected format, and next action.",
                    "Structured reports cover every family with missing mapping,",
                    "duplicate family name, duplicate species mappings,",
                    "rejected tree, and species coverage.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (root / "docs" / "workflow-examples" / "snakemake").mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "workflow-examples" / "snakemake" / "README.md").write_text(
            "\n".join(
                [
                    "# Snakemake Example",
                    "",
                    "gpurec validate-config --check-preprocess",
                    "--require-converged",
                    "--require-final-check-ok",
                    "gpurec sample",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (root / "docs" / "workflow-examples" / "nextflow").mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "workflow-examples" / "nextflow" / "README.md").write_text(
            "\n".join(
                [
                    "# Nextflow Example",
                    "",
                    "nextflow run main.nf -resume",
                    "gpurec validate-config --check-preprocess",
                    "--require-converged",
                    "--require-final-check-ok",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (root / "docs" / "workflow-examples" / "slurm").mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "workflow-examples" / "slurm" / "README.md").write_text(
            "\n".join(
                [
                    "# Slurm Example",
                    "",
                    "gpurec validate-config --check-preprocess",
                    "gpurec optimize",
                    "output_gpurec/checkpoints/latest.pt",
                    "gpurec sample",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if create_optimization_workflow_call_graph:
        (root / "docs" / "optimization-workflow-call-graph.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "optimization-workflow-call-graph.md").write_text(
            "# Optimization Workflow Call Graph\n", encoding="utf-8"
        )
    if create_lean_fast_path:
        (root / "docs" / "lean-fast-path.md").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "lean-fast-path.md").write_text(
            "# Lean Fast Path\n", encoding="utf-8"
        )
    if create_professionalization_audit_progress:
        (root / "docs" / "professionalization-audit-progress.tex").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "docs" / "professionalization-audit-progress.tex").write_text(
            "% professionalization audit fixture\n", encoding="utf-8"
        )
    if create_dependency_inventory_script:
        (root / "scripts").mkdir(parents=True, exist_ok=True)
        (root / "scripts" / "generate_dependency_inventory.py").write_text(
            "#!/usr/bin/env python3\nprint('fixture')\n",
            encoding="utf-8",
        )
    readme_block = f"{readme_line}\n" if readme_line else ""
    if urls_block is None:
        urls_block = """
[project.urls]
Repository = "https://example.invalid/repo"
Issues = "https://example.invalid/issues"
Documentation = "https://example.invalid/docs"
""".lstrip()
    if scripts_block is None:
        scripts_block = """
[project.scripts]
gpurec = "gpurec.cli:main"
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
{project_extra}classifiers = [
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
{scripts_block}
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

    assert result.returncode == 0
    assert "release metadata check passed" in result.stdout
    assert result.stderr == ""
    assert "Traceback" not in result.stderr


def test_required_release_artifacts_exist_in_repository():
    checker = _load_check_release_metadata_module()
    missing = [
        artifact
        for artifact in checker.REQUIRED_RELEASE_ARTIFACTS
        if not (ROOT / artifact).is_file()
    ]
    assert missing == []


def test_required_release_artifacts_contract_is_normalized_and_stable():
    checker = _load_check_release_metadata_module()
    artifacts = list(checker.REQUIRED_RELEASE_ARTIFACTS)

    assert artifacts == sorted(artifacts)
    assert len(artifacts) == len(set(artifacts))
    for artifact in artifacts:
        assert artifact == artifact.strip()
        assert artifact
        assert not artifact.startswith("/")
        assert "\\" not in artifact
        assert ".." not in Path(artifact).parts


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


def test_release_metadata_check_requires_governance_artifacts(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path, create_changelog=False)

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: CHANGELOG.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_support_policy(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path, create_support_policy=False)

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/support-policy.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_support_policy_scope_statements(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "support-policy.md").write_text(
        "# Support Policy\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must describe production support scope" in result.stdout
    assert "must describe latest release tag support" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_support_policy_support_window_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "support-policy.md").write_text(
        "\n".join(
            [
                "# Support Policy",
                "",
                "production support scope applies to documented surfaces.",
                "The latest release tag is the primary supported line.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document support-window phrase: support window" in result.stdout
    assert "must document support-window phrase: python" in result.stdout
    assert "must document support-window phrase: pytorch" in result.stdout
    assert "must document support-window phrase: cuda" in result.stdout
    assert "must document support-window phrase: native artifact" in result.stdout
    assert "must document support-window phrase: release and patch policy" in result.stdout
    assert "must document support-window phrase: older tags may receive backports" in result.stdout
    assert "must document support-window phrase: summary.json" in result.stdout
    assert "must document support-window phrase: run_manifest.json" in result.stdout
    assert "must document support-window phrase: checkpoint metadata" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_versioning_policy(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path, create_versioning_policy=False)

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/versioning-policy.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_versioning_policy_semver_statements(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "versioning-policy.md").write_text(
        "# Versioning Policy\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must state semantic versioning policy" in result.stdout
    assert "must describe MAJOR.MINOR.PATCH semantics" in result.stdout
    assert "must define latest release tag support line" in result.stdout
    assert "must define backport support expectations" in result.stdout
    assert "must document version-consistency phrase: pyproject.toml" in result.stdout
    assert "must document version-consistency phrase: gpurec.__version__" in result.stdout
    assert "must document version-consistency phrase: release notes heading" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_publication_checklist(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path, create_publication_checklist=False)

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "missing required release artifact: docs/publication-checklist.md"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_publication_checklist_citation_reference(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "publication-checklist.md").write_text(
        "Archive run_manifest.json and summary.json.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must mention CITATION.cff metadata" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_publication_checklist_artifact_references(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "publication-checklist.md").write_text(
        "Reference CITATION.cff for citation metadata.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must mention run_manifest.json" in result.stdout
    assert "must mention summary.json" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_publication_checklist_json_command_references(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "publication-checklist.md").write_text(
        "\n".join(
            [
                "# Publication Checklist",
                "",
                "Reference CITATION.cff for software citation metadata.",
                "Archive run_manifest.json and summary.json for reproducibility.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must mention gpurec doctor --json" in result.stdout
    assert (
        "must mention gpurec summary-info --summary ... --json" in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_publication_checklist_history_checkpoint_references(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "publication-checklist.md").write_text(
        "\n".join(
            [
                "# Publication Checklist",
                "",
                "Reference CITATION.cff for software citation metadata.",
                "Archive run_manifest.json and summary.json for reproducibility.",
                "Record gpurec doctor --json and gpurec summary-info --summary output_gpurec/summary.json --json.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must mention history.jsonl" in result.stdout
    assert "must mention checkpoints/ archive guidance" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_publication_checklist_reporting_references(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "publication-checklist.md").write_text(
        "\n".join(
            [
                "# Publication Checklist",
                "",
                "Reference CITATION.cff for software citation metadata.",
                "Archive run_manifest.json and summary.json for reproducibility.",
                "Record gpurec doctor --json and gpurec summary-info --summary output_gpurec/summary.json --json.",
                "Archive history.jsonl and checkpoints/ for rerun audit trails.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must mention known-limitations.md reporting guidance" in result.stdout
    assert "must mention release-notes.md migration notes guidance" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_publication_checklist_artifact_validator_reference(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "publication-checklist.md").write_text(
        "\n".join(
            [
                "# Publication Checklist",
                "",
                "Reference CITATION.cff for software citation metadata.",
                "Archive run_manifest.json and summary.json for reproducibility.",
                "Record gpurec doctor --json and gpurec summary-info --summary output_gpurec/summary.json --json.",
                "Archive history.jsonl and checkpoints/ for rerun audit trails.",
                "Report known-limitations.md caveats and release-notes.md migration notes.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must mention scripts/validate_output_artifacts.py gate" in result.stdout
    assert "must mention checksums evidence guidance" in result.stdout
    assert "must mention provenance evidence guidance" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_platform_matrix(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path, create_platform_matrix=False)

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/platform-matrix.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_platform_matrix_offline_policy_section(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "platform-matrix.md").write_text(
        "# Production Platform Matrix\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must include an 'Offline Installation Policy' section" in result.stdout
    assert "must explicitly state current offline-installation support policy" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_platform_matrix_offline_policy_statement(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "platform-matrix.md").write_text(
        "## Offline Installation Policy\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must explicitly state current offline-installation support policy" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_platform_matrix_primary_configuration(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "platform-matrix.md").write_text(
        "\n".join(
            [
                "# Platform Matrix",
                "",
                "## Offline Installation Policy",
                "",
                "Offline installation is not currently supported as a production guarantee.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document the primary supported configuration" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_platform_matrix_core_terms(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "platform-matrix.md").write_text(
        "\n".join(
            [
                "# Platform Matrix",
                "",
                "Primary supported configuration is documented.",
                "",
                "## Offline Installation Policy",
                "",
                "Offline installation is not currently supported as a production guarantee.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document matrix term: python" in result.stdout
    assert "must document matrix term: pytorch" in result.stdout
    assert "must document matrix term: cuda" in result.stdout
    assert "must document matrix term: triton" in result.stdout
    assert "must document matrix term: gpu" in result.stdout
    assert "must document matrix term: rust" in result.stdout
    assert "must document matrix term: compiler" in result.stdout
    assert "must document matrix term: cargo" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_api_contract(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path, create_api_contract=False)

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/api-contract.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_known_limitations(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path, create_known_limitations=False
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/known-limitations.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_known_limitations_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "known-limitations.md").write_text(
        "# Known Limitations\n\nGeneral constraints.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document limitation phrase: cuda" in result.stdout
    assert "must document limitation phrase: s > 256" in result.stdout
    assert "must document limitation phrase: newick subset" in result.stdout
    assert "must document limitation phrase: quoted labels" in result.stdout
    assert "must document limitation phrase: embedded delimiters" in result.stdout
    assert "must document limitation phrase: nested comments" in result.stdout
    assert "must document limitation phrase: nhx/beast metadata" in result.stdout
    assert "must document limitation phrase: unary species nodes" in result.stdout
    assert "must document limitation phrase: non-binary species trees" in result.stdout
    assert "must document limitation phrase: wheel" in result.stdout
    assert "must document limitation phrase: external" in result.stdout
    assert "must document limitation phrase: bf16" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_bioinformatics_quickstart(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_bioinformatics_quickstart=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "missing required release artifact: docs/bioinformatics-quickstart.md"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_quickstart_lifecycle_stages(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "bioinformatics-quickstart.md").write_text(
        "# Bioinformatics Quickstart\n\nInstall only.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document lifecycle stage: create config" in result.stdout
    assert "must document lifecycle stage: validate" in result.stdout
    assert "must document lifecycle stage: run" in result.stdout
    assert "must document lifecycle stage: resume" in result.stdout
    assert "must document lifecycle stage: inspect" in result.stdout
    assert "must document lifecycle stage: sample" in result.stdout
    assert "must document lifecycle stage: archive" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_quickstart_installation_decision_tree_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "bioinformatics-quickstart.md").write_text(
        "# Bioinformatics Quickstart\n\nCreate config, validate, run, resume, inspect, sample, archive.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document installation-decision phrase: installation decision tree" in result.stdout
    assert "must document installation-decision phrase: source checkout or source archive" in result.stdout
    assert "must document installation-decision phrase: wheel-only environment" in result.stdout
    assert "must document installation-decision phrase: cluster/container workflows" in result.stdout
    assert "must document installation-decision phrase: offline installation" in result.stdout
    assert "must document installation-decision phrase: gpurec preprocess-check" in result.stdout
    assert "must document installation-decision phrase: gpurec backtrack-check" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_quickstart_json_mode_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "bioinformatics-quickstart.md").write_text(
        "\n".join(
            [
                "# Bioinformatics Quickstart",
                "",
                "Create config, validate, run, resume, inspect, sample, archive.",
                "Installation decision tree for source checkout or source archive.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document json-mode phrase: structured json mode" in result.stdout
    assert "must document json-mode phrase: gpurec doctor --json" in result.stdout
    assert (
        "must document json-mode phrase: gpurec validate-config --config run.json --json"
        in result.stdout
    )
    assert (
        "must document json-mode phrase: gpurec summary-info --summary output_gpurec/summary.json --json"
        in result.stdout
    )
    assert (
        "must document json-mode phrase: gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt --json"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_quickstart_rng_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "bioinformatics-quickstart.md").write_text(
        "\n".join(
            [
                "# Bioinformatics Quickstart",
                "",
                "Create config, validate, run, resume, inspect, sample, archive.",
                "Installation decision tree for source checkout or source archive,",
                "wheel-only environment, cluster/container workflows, and",
                "offline installation policy.",
                "Structured JSON mode includes gpurec doctor --json.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document rng phrase: rng behavior" in result.stdout
    assert "must document rng phrase: seed" in result.stdout
    assert "must document rng phrase: reproducibility" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_input_preparation(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_input_preparation=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/input-preparation.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_input_preparation_large_dataset_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "input-preparation.md").write_text(
        "# Input Preparation\n\nBasic notes.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document large-dataset phrase: max-families" in result.stdout
    assert "must document large-dataset phrase: sample the first `n` families" in result.stdout
    assert "must document large-dataset phrase: memory estimate" in result.stdout
    assert "must document large-dataset phrase: clade_budget" in result.stdout
    assert "must document large-dataset phrase: family_chunk_size" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_input_preparation_conversion_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "input-preparation.md").write_text(
        "# Input Preparation\n\nBasic mapping notes.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document conversion phrase: treerecs" in result.stdout
    assert "must document conversion phrase: generax" in result.stdout
    assert "must document conversion phrase: alerax" in result.stdout
    assert "must document conversion phrase: orthofinder" in result.stdout
    assert "must document conversion phrase: gene -> species" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_input_preparation_family_file_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "input-preparation.md").write_text(
        "# Input Preparation\n\nBasic family-file notes.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document family-file phrase: multiple families" in result.stdout
    assert "must document family-file phrase: multiple trees per family" in result.stdout
    assert "must document family-file phrase: mapping files" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_output_artifacts_doc(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_output_artifacts=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/output-artifacts.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_output_artifact_snippet_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "output-artifacts.md").write_text(
        "# Output Artifacts\n\nContracts.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document output snippet phrase: example output snippets" in result.stdout
    assert "must document output snippet phrase: summary.json" in result.stdout
    assert "must document output snippet phrase: rates_final.tsv" in result.stdout
    assert "must document output snippet phrase: per_fam_likelihoods.tsv" in result.stdout
    assert "must document output snippet phrase: recphyloxml output snippet" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_output_artifact_directory_structure_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "output-artifacts.md").write_text(
        "# Output Artifacts\n\nExample output snippets.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document directory-structure phrase: run directory structure" in result.stdout
    assert "must document directory-structure phrase: output_gpurec/" in result.stdout
    assert "must document directory-structure phrase: checkpoints/" in result.stdout
    assert "must document directory-structure phrase: reconciliations/" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_output_artifact_flow_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "output-artifacts.md").write_text(
        "# Output Artifacts\n\nRun directory structure.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document flow phrase: input/output flow" in result.stdout
    assert "must document flow phrase: validate-config --check-preprocess" in result.stdout
    assert "must document flow phrase: gpurec optimize" in result.stdout
    assert "must document flow phrase: gpurec sample" in result.stdout
    assert "must document flow phrase: reconciliations/*.xml" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_output_artifact_run_manifest_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "output-artifacts.md").write_text(
        "\n".join(
            [
                "# Output Artifacts",
                "",
                "Example output snippets.",
                "Run directory structure.",
                "Input/output flow.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document run-manifest phrase: run_manifest.json" in result.stdout
    assert "must document run-manifest phrase: package version" in result.stdout
    assert "must document run-manifest phrase: native artifact" in result.stdout
    assert "must document run-manifest phrase: pytorch version" in result.stdout
    assert "must document run-manifest phrase: cuda availability" in result.stdout
    assert "must document run-manifest phrase: gpu name" in result.stdout
    assert "must document run-manifest phrase: command line invocation" in result.stdout
    assert "must document run-manifest phrase: config hash" in result.stdout
    assert "must document run-manifest phrase: random seed" in result.stdout
    assert "must document run-manifest phrase: selected route" in result.stdout
    assert "must document run-manifest phrase: reproduce or audit" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_output_artifact_theta_checkpoint_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "output-artifacts.md").write_text(
        "\n".join(
            [
                "# Output Artifacts",
                "",
                "Example output snippets.",
                "Run directory structure.",
                "Input/output flow.",
                "run_manifest.json contract text.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document theta/checkpoint phrase: theta_final.pt" in result.stdout
    assert "must document theta/checkpoint phrase: for inspection only" in result.stdout
    assert "must document theta/checkpoint phrase: checkpoint is required" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_output_artifact_schema_compatibility_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "output-artifacts.md").write_text(
        "\n".join(
            [
                "# Output Artifacts",
                "",
                "Example output snippets.",
                "Run directory structure.",
                "Input/output flow.",
                "run_manifest.json contract text.",
                "theta_final.pt is for inspection only.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document schema-compatibility phrase: stable schemas" in result.stdout
    assert (
        "must document schema-compatibility phrase: compatibility rules"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_long_validation_workflow(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_long_validation_workflow=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "missing required release artifact: docs/long-validation-workflow.md"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_long_validation_evidence_scope_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "long-validation-workflow.md").write_text(
        "# Long Validation Workflow\n\nRelease run steps.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document evidence-scope phrase: benchmark evidence" in result.stdout
    assert (
        "must document evidence-scope phrase: not a hard performance guarantee"
        in result.stdout
    )
    assert (
        "must document evidence-scope phrase: not a guaranteed performance contract"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_long_validation_command_sequence_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "long-validation-workflow.md").write_text(
        "\n".join(
            [
                "# Long Validation Workflow",
                "",
                "Use this report as benchmark evidence, not a hard performance guarantee.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document command-sequence phrase: gpurec doctor --json" in result.stdout
    assert (
        "must document command-sequence phrase: gpurec validate-config --check-preprocess --require-cuda-backward-ready"
        in result.stdout
    )
    assert (
        "must document command-sequence phrase: gpurec optimize --require-final-check-ok"
        in result.stdout
    )
    assert (
        "must document command-sequence phrase: gpurec summary-info --require-converged --require-final-check-ok"
        in result.stdout
    )
    assert "must document command-sequence phrase: gpurec sample --checkpoint" in result.stdout
    assert (
        "must document command-sequence phrase: scripts/validate_output_artifacts.py"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_validation_envelope(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_validation_envelope=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/validation-envelope.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_validation_envelope_evidence_terms(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "validation-envelope.md").write_text(
        "# Validation Envelope\n\nEvidence.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document validation evidence term: runtime envelope" in result.stdout
    assert "must document validation evidence term: peak memory" in result.stdout
    assert "must document validation evidence term: final nll" in result.stdout
    assert "must document validation evidence term: benchmark evidence" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_troubleshooting_doc(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_troubleshooting=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/troubleshooting.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_troubleshooting_recovery_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "troubleshooting.md").write_text(
        "# Troubleshooting\n\nGeneral notes.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document failure-recovery phrase: by symptom" in result.stdout
    assert "must document failure-recovery phrase: retryable runtime failures" in result.stdout
    assert "must document failure-recovery phrase: input contract failures" in result.stdout
    assert "must document failure-recovery phrase: likely cause" in result.stdout
    assert "must document failure-recovery phrase: next action" in result.stdout
    assert "must document failure-recovery phrase: authoritative files" in result.stdout
    assert "must document failure-recovery phrase: summary.json" in result.stdout
    assert "must document failure-recovery phrase: history.jsonl" in result.stdout
    assert "must document failure-recovery phrase: checkpoints/latest.pt" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_docs_readme(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_docs_readme=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/README.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_production_optimization_guide(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_production_optimization_guide=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "missing required release artifact: docs/production-optimization-guide.md"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_user_goal_defaults_in_optimization_guide(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "production-optimization-guide.md").write_text(
        "# Production Optimization Guide\n\nDefaults.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document user-goal default: exploratory run" in result.stdout
    assert "must document user-goal default: production genewise run" in result.stdout
    assert (
        "must document user-goal default: production specieswise run" in result.stdout
    )
    assert (
        "must document user-goal default: diagnostics-only global run"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_glossary(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_glossary=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/glossary.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_glossary_core_terms(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "glossary.md").write_text(
        "# Glossary\n\nTerms list.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document glossary term: `d`" in result.stdout
    assert "must document glossary term: `t`" in result.stdout
    assert "must document glossary term: `l`" in result.stdout
    assert "must document glossary term: `dtl`" in result.stdout
    assert "must document glossary term: `ccp`" in result.stdout
    assert "must document glossary term: `specieswise`" in result.stdout
    assert "must document glossary term: `genewise`" in result.stdout
    assert "must document glossary term: `global`" in result.stdout
    assert "must document glossary term: `recphyloxml`" in result.stdout
    assert "must document glossary term: `nll`" in result.stdout
    assert "must document glossary term: `route`" in result.stdout
    assert "must document glossary term: `solver budget`" in result.stdout
    assert "must document glossary term: `checkpoint`" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_workflow_examples_readme(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_workflow_examples=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/workflow-examples/README.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_workflow_examples_run_config(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "workflow-examples" / "end-to-end-tutorial" / "run.json").unlink()

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "missing required release artifact: "
        "docs/workflow-examples/end-to-end-tutorial/run.json"
    ) in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_workflow_examples_dataset_generator(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (
        tmp_path
        / "docs"
        / "workflow-examples"
        / "end-to-end-tutorial"
        / "generate_dataset.py"
    ).unlink()

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "missing required release artifact: "
        "docs/workflow-examples/end-to-end-tutorial/generate_dataset.py"
    ) in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_input_validation_fixture_issue_shape_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (
        tmp_path
        / "docs"
        / "workflow-examples"
        / "input-validation-fixtures"
        / "README.md"
    ).write_text(
        "# Input Validation Fixtures\n\nFixture overview.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document issue-shape phrase: file path" in result.stdout
    assert "must document issue-shape phrase: family name" in result.stdout
    assert "must document issue-shape phrase: affected label" in result.stdout
    assert "must document issue-shape phrase: expected format" in result.stdout
    assert "must document issue-shape phrase: next action" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_input_validation_fixture_category_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (
        tmp_path
        / "docs"
        / "workflow-examples"
        / "input-validation-fixtures"
        / "README.md"
    ).write_text(
        "\n".join(
            [
                "# Input Validation Fixtures",
                "",
                "Issue entries include file path, family name, affected label,",
                "expected format, and next action.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document category phrase: every family" in result.stdout
    assert "must document category phrase: missing mapping" in result.stdout
    assert "must document category phrase: duplicate family name" in result.stdout
    assert "must document category phrase: duplicate species mappings" in result.stdout
    assert "must document category phrase: rejected tree" in result.stdout
    assert "must document category phrase: species coverage" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_input_validation_fixture_cpu_safe_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (
        tmp_path
        / "docs"
        / "workflow-examples"
        / "input-validation-fixtures"
        / "README.md"
    ).write_text(
        "# Input Validation Fixtures\n\nCategory coverage only.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document cpu-safe phrase: without constructing a cuda model" in result.stdout
    assert "must document cpu-safe phrase: validate-inputs" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_end_to_end_tutorial_public_command_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "workflow-examples" / "end-to-end-tutorial" / "README.md").write_text(
        "# End-to-End Tutorial\n\nRun this workflow.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document tutorial phrase: first successful run tutorial" in result.stdout
    assert "must document tutorial phrase: uses only public commands" in result.stdout
    assert "must document tutorial phrase: tracked or downloadable dataset" in result.stdout
    assert "must document tutorial phrase: writes outputs" in result.stdout
    assert "must document tutorial phrase: samples recphyloxml" in result.stdout
    assert "must document tutorial phrase: gpurec validate-config" in result.stdout
    assert "must document tutorial phrase: --check-preprocess" in result.stdout
    assert "must document tutorial phrase: --require-cuda-backward-ready" in result.stdout
    assert "must document tutorial phrase: gpurec optimize" in result.stdout
    assert "must document tutorial phrase: --resume-from" in result.stdout
    assert "must document tutorial phrase: output_gpurec/checkpoints/latest.pt" in result.stdout
    assert "must document tutorial phrase: gpurec sample" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_slurm_lifecycle_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "workflow-examples" / "slurm" / "README.md").write_text(
        "# Slurm Example\n\nsbatch run-gpurec.sbatch\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document lifecycle phrase: gpurec validate-config" in result.stdout
    assert "must document lifecycle phrase: --check-preprocess" in result.stdout
    assert "must document lifecycle phrase: gpurec optimize" in result.stdout
    assert (
        "must document lifecycle phrase: output_gpurec/checkpoints/latest.pt"
        in result.stdout
    )
    assert "must document lifecycle phrase: gpurec sample" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_snakemake_gate_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "workflow-examples" / "snakemake" / "README.md").write_text(
        "# Snakemake Example\n\nsnakemake --cores 1\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document gate phrase: gpurec validate-config" in result.stdout
    assert "must document gate phrase: --check-preprocess" in result.stdout
    assert "must document gate phrase: --require-converged" in result.stdout
    assert "must document gate phrase: --require-final-check-ok" in result.stdout
    assert "must document gate phrase: gpurec sample" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_nextflow_gate_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "workflow-examples" / "nextflow" / "README.md").write_text(
        "# Nextflow Example\n\nnextflow run main.nf\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document gate phrase: nextflow run main.nf -resume" in result.stdout
    assert "must document gate phrase: gpurec validate-config" in result.stdout
    assert "must document gate phrase: --check-preprocess" in result.stdout
    assert "must document gate phrase: --require-converged" in result.stdout
    assert "must document gate phrase: --require-final-check-ok" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_workflow_examples_overview_gate_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "workflow-examples" / "README.md").write_text(
        "# Workflow Examples\n\nOverview only.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document acceptance-gate phrase: snakemake" in result.stdout
    assert "must document acceptance-gate phrase: nextflow" in result.stdout
    assert "must document acceptance-gate phrase: fail fast" in result.stdout
    assert "must document acceptance-gate phrase: --check-preprocess" in result.stdout
    assert (
        "must document acceptance-gate phrase: resume from a checkpoint"
        in result.stdout
    )
    assert (
        "must document acceptance-gate phrase: reject non-converged outputs"
        in result.stdout
    )
    assert "must document acceptance-gate phrase: tracked mini public dataset" in result.stdout
    assert "must document acceptance-gate phrase: deterministic" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_docs_map_scope_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "README.md").write_text(
        "# Documentation Map\n\nGeneral docs index.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document scope phrase: stable user workflows" in result.stdout
    assert "must document scope phrase: hogenom-only research scripts" in result.stdout
    assert "must document scope phrase: cli help" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_readme_short_user_path_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# fixture",
                "",
                "CLI exit codes are stable for workflow managers:",
                "- `0`: command completed successfully.",
                "- `1`: command ran but failed a runtime or validation gate",
                "- `2`: CLI usage or argument parsing error from `argparse`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document short-path phrase: install" in result.stdout
    assert "must document short-path phrase: validate inputs" in result.stdout
    assert "must document short-path phrase: run optimization" in result.stdout
    assert "must document short-path phrase: inspect output" in result.stdout
    assert "must document short-path phrase: sample reconciliations" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_api_contract_json_output_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "api-contract.md").write_text(
        "# API Contract\n\nMinimal contract.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document json-output contract phrase: cli output modes" in result.stdout
    assert "must document json-output contract phrase: --json" in result.stdout
    assert "must document json-output contract phrase: validate-config" in result.stdout
    assert "must document json-output contract phrase: doctor" in result.stdout
    assert "must document json-output contract phrase: checkpoint-info" in result.stdout
    assert "must document json-output contract phrase: summary-info" in result.stdout
    assert (
        "must document json-output contract phrase: json mode emits single json objects with stable keys"
        in result.stdout
    )
    assert (
        "must document json-output contract phrase: required machine path for automation"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_api_contract_compatibility_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "api-contract.md").write_text(
        "\n".join(
            [
                "# API Contract",
                "",
                "## CLI output modes",
                "",
                "`--json` mode is supported by validate-config, doctor,",
                "checkpoint-info, and summary-info.",
                "JSON mode emits single JSON objects with stable keys.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document compatibility phrase: compatibility policy" in result.stdout
    assert "must document compatibility phrase: config fields" in result.stdout
    assert "must document compatibility phrase: cli flags" in result.stdout
    assert "must document compatibility phrase: python imports" in result.stdout
    assert "must document compatibility phrase: output artifacts" in result.stdout
    assert "must document compatibility phrase: deprecation warnings" in result.stdout
    assert "must document compatibility phrase: migration notes" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_api_contract_exit_code_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "api-contract.md").write_text(
        "\n".join(
            [
                "# API Contract",
                "",
                "## CLI output modes",
                "",
                "`--json` mode is supported by validate-config, doctor,",
                "checkpoint-info, and summary-info.",
                "JSON mode emits single JSON objects with stable keys.",
                "Compatibility policy covers config fields, CLI flags,",
                "Python imports, and output artifacts.",
                "Deprecation warnings and migration notes are required",
                "before removing supported behavior.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document exit-code phrase: exit status `0`" in result.stdout
    assert "must document exit-code phrase: exit status `1`" in result.stdout
    assert "must document exit-code phrase: exit status `2`" in result.stdout
    assert (
        "must document exit-code phrase: runtime and route-validation failures"
        in result.stdout
    )
    assert "must document exit-code phrase: cli parse/config errors" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_optimization_workflow_call_graph(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_optimization_workflow_call_graph=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "missing required release artifact: "
        "docs/optimization-workflow-call-graph.md"
    ) in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_lean_fast_path_doc(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_lean_fast_path=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "missing required release artifact: docs/lean-fast-path.md" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_professionalization_audit_progress(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_professionalization_audit_progress=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "missing required release artifact: "
        "docs/professionalization-audit-progress.tex"
    ) in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_dependency_inventory_script(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(
        tmp_path,
        create_dependency_inventory_script=False,
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "missing required release artifact: scripts/generate_dependency_inventory.py"
        in result.stdout
    )
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


def test_release_metadata_check_requires_readme_cli_exit_code_policy(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "README.md").write_text("# fixture\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "README.md must document CLI exit-code policy" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_release_readiness_gate_phrases(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "release-readiness.md").write_text(
        "# Release Readiness\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must document release gate phrase: python scripts/check_release_metadata.py" in result.stdout
    assert "must document release gate phrase: scripts/run_long_validation.py" in result.stdout
    assert "must document release gate phrase: validation-envelope.md" in result.stdout
    assert "must document release gate phrase: gpurec doctor" in result.stdout
    assert "must document release gate phrase: quick pr checks" in result.stdout
    assert "must document release gate phrase: nightly checks" in result.stdout
    assert "must document release gate phrase: release-candidate checks" in result.stdout
    assert "must document release gate phrase: final publication checks" in result.stdout
    assert "must document release gate phrase: checksums" in result.stdout
    assert "must document release gate phrase: provenance" in result.stdout
    assert "must document release gate phrase: sha256sum dist/* > dist/sha256sums" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_changelog_current_version(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## 9.9.9\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "CHANGELOG.md must mention current pyproject version" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_release_notes_current_version(tmp_path: Path):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "release-notes.md").write_text(
        "# Release Notes\n\n## 9.9.9\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "docs/release-notes.md must mention current pyproject version" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_release_notes_known_limitations_section(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "release-notes.md").write_text(
        "# Release Notes\n\n## 0.0.0 - 2026-01-01\n\n- Migration notes\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must include a 'Known limitations' section" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_release_notes_migration_notes_section(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "release-notes.md").write_text(
        "# Release Notes\n\n## 0.0.0 - 2026-01-01\n\n- Known limitations\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must include a 'Migration notes' section" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_release_notes_release_artifact_notes_section(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "release-notes.md").write_text(
        "# Release Notes\n\n## 0.0.0 - 2026-01-01\n\n- Known limitations\n- Migration notes\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "must include a 'Release artifact notes' section" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_release_notes_dependency_support_guidance(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "docs" / "release-notes.md").write_text(
        "\n".join(
            [
                "# Release Notes",
                "",
                "## 0.0.0 - 2026-01-01",
                "",
                "- Known limitations",
                "- Migration notes",
                "- Release artifact notes",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "must include dependency/python/torch/cuda support-update guidance"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_citation_top_level_version_match(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        citation.read_text(encoding="utf-8").replace('"0.0.0"', '"9.9.9"', 1),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "CITATION.cff top-level version must match pyproject version" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_citation_top_level_version_field(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        "\n".join(
            [
                "cff-version: 1.2.0",
                'title: "fixture"',
                "preferred-citation:",
                '  title: "fixture"',
                '  type: software',
                '  version: "0.0.0"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "CITATION.cff must declare a top-level version" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_citation_cff_version_field(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "CITATION.cff").write_text(
        "\n".join(
            [
                'title: "fixture"',
                'repository-code: "https://example.invalid/repo"',
                'version: "0.0.0"',
                "preferred-citation:",
                '  title: "fixture"',
                '  type: software',
                '  version: "0.0.0"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "CITATION.cff must declare cff-version" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_citation_repository_code_field(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    (tmp_path / "CITATION.cff").write_text(
        "\n".join(
            [
                "cff-version: 1.2.0",
                'title: "fixture"',
                'version: "0.0.0"',
                "preferred-citation:",
                '  title: "fixture"',
                '  type: software',
                '  version: "0.0.0"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "CITATION.cff must declare repository-code" in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_citation_repository_url_match(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        citation.read_text(encoding="utf-8").replace(
            'repository-code: "https://example.invalid/repo"',
            'repository-code: "https://example.invalid/other"',
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "CITATION.cff repository-code must match "
        "pyproject [project.urls].Repository"
    ) in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_citation_preferred_version_match(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        citation.read_text(encoding="utf-8").replace('  version: "0.0.0"', '  version: "9.9.9"', 1),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert (
        "CITATION.cff preferred-citation version must match pyproject version"
        in result.stdout
    )
    assert result.stderr == ""


def test_release_metadata_check_requires_citation_preferred_version_field(
    tmp_path: Path,
):
    _write_complete_release_metadata_fixture(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        "\n".join(
            [
                "cff-version: 1.2.0",
                'title: "fixture"',
                'version: "0.0.0"',
                "preferred-citation:",
                '  title: "fixture"',
                "  type: software",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "CITATION.cff preferred-citation must declare a version" in result.stdout
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


def test_release_metadata_check_requires_project_scripts_table(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        scripts_block="",
        project_extra='scripts = "gpurec.cli:main"\n',
    )

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 1
    assert "[project.scripts] must be a table" in result.stdout
    assert "license" not in result.stdout
    assert result.stderr == ""


def test_release_metadata_check_requires_gpurec_console_script(tmp_path: Path):
    _write_complete_release_metadata_fixture(
        tmp_path,
        scripts_block="""
[project.scripts]
gpurec = "gpurec.cli.reconcile:main"
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
    assert "[project.scripts] gpurec must be 'gpurec.cli:main'" in result.stdout
    assert "gpurec.cli.reconcile" not in result.stdout
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
    assert project["scripts"] == {"gpurec": "gpurec.cli:main"}


def test_minimal_pyproject_parser_supports_current_project_release_fields():
    checker = _load_check_release_metadata_module()
    parsed = checker._parse_minimal_pyproject(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    project = parsed["project"]
    assert project["readme"] == "README.md"
    assert project["license"] == {"file": "LICENSE"}
    assert project["authors"] == '[{ name = "SisyphusMountain" }]'
    assert project["urls"]["Repository"] == "https://github.com/SisyphusMountain/gpurec"
    assert project["urls"]["Issues"] == (
        "https://github.com/SisyphusMountain/gpurec/issues"
    )
    assert project["urls"]["Documentation"] == (
        "https://github.com/SisyphusMountain/gpurec#readme"
    )
    assert project["scripts"]["gpurec"] == "gpurec.cli:main"
    for required in (
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ):
        assert required in project["classifiers"]


def test_cpu_ci_builds_and_smokes_release_artifacts():
    workflow = _load_cpu_ci_workflow()
    package = workflow["jobs"]["package"]

    assert package["strategy"]["matrix"]["python-version"] == ["3.10", "3.12"]
    assert _workflow_step(package, "Set up Python")["with"][
        "python-version"
    ] == "${{ matrix.python-version }}"

    install = _step_run(package, "Install package, runtime, and release dependencies")
    assert 'python -m pip install -e ".[release]"' in install

    assert _step_run(package, "Run release metadata checker") == (
        "python scripts/check_release_metadata.py"
    )

    build = _step_run(package, "Build source and wheel artifacts")
    assert "rm -rf dist" in build
    assert "python -m build" in build

    assert _step_run(package, "Check artifact metadata") == "python -m twine check dist/*"
    checksums = _step_run(package, "Generate artifact checksums")
    assert "sha256sum dist/* > dist/SHA256SUMS" in checksums
    assert "test -s dist/SHA256SUMS" in checksums
    assert "ls dist/*.whl dist/*.tar.gz >/dev/null" in checksums
    assert "grep -Eq '\\.whl$' dist/SHA256SUMS" in checksums
    assert "grep -Eq '\\.tar\\.gz$' dist/SHA256SUMS" in checksums

    artifact_check = _step_run(package, "Check artifact package data")
    for required in (
        "import json",
        "tarfile.open",
        "zipfile.ZipFile",
        "required_sdist = required_wheel +",
        "LICENSE",
        "CHANGELOG.md",
        "CITATION.cff",
        "Dockerfile",
        "docs/README.md",
        "docs/input-preparation.md",
        "docs/api-contract.md",
        "docs/known-limitations.md",
        "docs/bioinformatics-quickstart.md",
        "docs/lean-fast-path.md",
        "docs/optimization-workflow-call-graph.md",
        "docs/output-artifacts.md",
        "docs/platform-matrix.md",
        "docs/production-optimization-guide.md",
        "docs/professionalization-audit-progress.tex",
        "docs/release-readiness.md",
        "docs/release-notes.md",
        "docs/long-validation-workflow.md",
        "docs/validation-envelope.md",
        "docs/support-policy.md",
        "docs/versioning-policy.md",
        "docs/publication-checklist.md",
        "docs/troubleshooting.md",
        "docs/glossary.md",
        "docs/workflow-examples/README.md",
        "docs/workflow-examples/end-to-end-tutorial/README.md",
        "docs/workflow-examples/end-to-end-tutorial/run.json",
        "docs/workflow-examples/end-to-end-tutorial/generate_dataset.py",
        "docs/workflow-examples/input-validation-fixtures/README.md",
        "docs/workflow-examples/snakemake/README.md",
        "docs/workflow-examples/nextflow/README.md",
        "docs/workflow-examples/slurm/README.md",
        "scripts/generate_dependency_inventory.py",
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


def test_cpu_ci_runs_release_metadata_checker_before_build():
    workflow = _load_cpu_ci_workflow()
    steps = workflow["jobs"]["package"]["steps"]
    names = [step.get("name") for step in steps]

    assert "Run release metadata checker" in names
    assert "Build source and wheel artifacts" in names
    assert names.index("Run release metadata checker") < names.index(
        "Build source and wheel artifacts"
    )


def test_cpu_sdist_required_artifacts_cover_release_metadata_required_artifacts():
    workflow = _load_cpu_ci_workflow()
    package = workflow["jobs"]["package"]
    artifact_check = _step_run(package, "Check artifact package data")
    checker = _load_check_release_metadata_module()

    for artifact_path in checker.REQUIRED_RELEASE_ARTIFACTS:
        assert (
            artifact_path in artifact_check
        ), f"required release artifact missing from sdist gate: {artifact_path}"

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
            "examples/minimal-run-config.json --require-mode-default-optimizer "
            "--require-production-default-route --check-preprocess"
        ),
        "optimizer=hessian-sgd",
        "uses_mode_default_optimizer=true",
        "uses_production_default_route=true",
        "production_default_route_mismatches=none",
        "hessian_sgd_normal_fixed_iters_pi=full",
        "hessian_sgd_pi_adjoint_warmstart=false",
        "pi_fixed_point_relaxation=1.000000",
        "hessian_sgd_validation_interval=0",
        "hessian_sgd_validation_fixed_iters_pi=configured",
        "hessian_sgd_validation_neumann_terms=configured",
        "cuda_backward_ready=false",
        "preprocess_checked=true",
        (
            "python -m gpurec.cli validate-config --config "
            "examples/minimal-run-config.json --require-mode-default-optimizer "
            "--require-production-default-route --check-preprocess "
            "--require-cuda-backward-ready"
        ),
        "genewise-cuda-ready.out",
        "genewise-cuda-ready.err",
        "genewise_example_cuda_status=$?",
        'test "$genewise_example_cuda_status" -eq 2',
        "cuda_backward_ready=false cuda_backward_ready_reason=requires_s_gt_256",
        "test ! -s genewise-cuda-ready.out",
        (
            "python -m gpurec.cli validate-config --config "
            "examples/specieswise-adagrad-restarts-config.json "
            "--require-mode-default-optimizer --require-production-default-route "
            "--check-preprocess"
        ),
        "optimizer=adagrad-restarts",
        "adagrad_restart_schedule=8:1:60,16:0.5:35,32:0.5:30",
        "adagrad_restart_total_steps=125",
        "optimizer_step_cap=125",
        "optimizer_step_cap_reason=adagrad_restart_schedule",
        "cuda_backward_ready=false",
        "preprocess_checked=true",
        (
            "python -m gpurec.cli validate-config --config "
            "examples/specieswise-adagrad-restarts-config.json "
            "--require-mode-default-optimizer --require-production-default-route "
            "--check-preprocess --require-cuda-backward-ready"
        ),
        "specieswise-cuda-ready.out",
        "specieswise-cuda-ready.err",
        "specieswise_example_cuda_status=$?",
        'test "$specieswise_example_cuda_status" -eq 2',
        "test ! -s specieswise-cuda-ready.out",
    ):
        assert required in rust_smoke

    wheel_smoke = _step_run(package, "Install built wheel and smoke CLI")
    for required in (
        "python -m pip install --no-deps dist/*.whl",
        "smoke_dir=$(mktemp -d)",
        'cd "$smoke_dir"',
        "gpurec --help",
        "python -m gpurec.cli --help",
        "gpurec preprocess-check --help | tee preprocess-check-help.txt",
        'grep -q -- "--preprocess-native-lib" preprocess-check-help.txt',
        'grep -q -- "GPUREC_PREPROCESS_NATIVE_LIB" preprocess-check-help.txt',
        "unset GPUREC_PREPROCESS_NATIVE_LIB",
        "gpurec preprocess-check > preprocess-check-missing.txt 2>&1",
        "preprocess_check_status=$?",
        'test "$preprocess_check_status" -eq 1',
        'grep -q -- "GPUREC_PREPROCESS_NATIVE_LIB" preprocess-check-missing.txt',
        'grep -q -- "--preprocess-native-lib" preprocess-check-missing.txt',
        "gpurec config-template --help | tee config-template-help.txt",
        'grep -q -- "mode-default Adam" config-template-help.txt',
        "gpurec config-template --mode genewise",
        "genewise-config-template.json",
        '"optimizer": "auto"',
        '"mode": "genewise"',
        '"solver_warmup_iters": 4',
        '"hessian_sgd_normal_fixed_iters_pi": null',
        '"hessian_sgd_pi_adjoint_warmstart": false',
        '"pi_fixed_point_relaxation": 1.0',
        '"hessian_sgd_validation_interval": 0',
        '"hessian_sgd_validation_fixed_iters_pi": null',
        '"hessian_sgd_validation_neumann_terms": null',
        "gpurec config-template --mode specieswise",
        "specieswise-config-template.json",
        '"mode": "specieswise"',
        '"adagrad_restart_schedule": "8:1.0:60,16:0.5:35,32:0.5:30"',
        '"adagrad_restart_final_check_iters": 128',
        "gpurec config-template --mode global",
        "global-config-template.json",
        '"mode": "global"',
        "generated-genewise-run.json",
        "generated-genewise-validate.txt",
        "generated-genewise-cuda-ready.out",
        "generated-genewise-cuda-ready.err",
        "generated-specieswise-run.json",
        "generated-specieswise-validate.txt",
        "generated-specieswise-cuda-ready.out",
        "generated-specieswise-cuda-ready.err",
        "generated-global-run.json",
        "generated-global-mode-default-validate.txt",
        "generated-global-production-route.out",
        "generated-global-production-route.err",
        (
            "cargo build --release --locked --features python-extension "
            '--manifest-path "$GITHUB_WORKSPACE/crates/gpurec-preprocess/Cargo.toml"'
        ),
        (
            'export GPUREC_PREPROCESS_NATIVE_LIB="$GITHUB_WORKSPACE/crates/'
            'gpurec-preprocess/target/release/libgpurec_preprocess.so"'
        ),
        "gpurec preprocess-check | tee preprocess-check.txt",
        "preprocessing_available=true",
        "preprocess_native_lib=",
        "$GITHUB_WORKSPACE/examples/tiny/species.nwk",
        "$GITHUB_WORKSPACE/examples/tiny/families.txt",
        "gpurec validate-config --config generated-genewise-run.json "
        "--require-mode-default-optimizer --require-production-default-route "
        "--check-preprocess",
        "gpurec validate-config --config generated-genewise-run.json "
        "--require-mode-default-optimizer --require-production-default-route "
        "--check-preprocess --require-cuda-backward-ready",
        "gpurec validate-config --config generated-specieswise-run.json "
        "--require-mode-default-optimizer --require-production-default-route "
        "--check-preprocess",
        "gpurec validate-config --config generated-specieswise-run.json "
        "--require-mode-default-optimizer --require-production-default-route "
        "--check-preprocess --require-cuda-backward-ready",
        "gpurec validate-config --config generated-global-run.json "
        "--require-mode-default-optimizer",
        "gpurec validate-config --config generated-global-run.json "
        "--require-production-default-route",
        "optimizer=hessian-sgd",
        "optimizer=adagrad-restarts",
        "optimizer=adam",
        "uses_mode_default_optimizer=true",
        "uses_production_default_route=true",
        "uses_production_default_route=false",
        "production_default_route_mismatches=none",
        "production_default_route_mismatches=mode",
        "cuda_backward_ready=false",
        "preprocess_checked=true",
        "genewise_cuda_status=$?",
        'test "$genewise_cuda_status" -eq 2',
        "specieswise_cuda_status=$?",
        'test "$specieswise_cuda_status" -eq 2',
        "cuda_backward_ready=false cuda_backward_ready_reason=requires_s_gt_256",
        "test ! -s generated-genewise-cuda-ready.out",
        "test ! -s generated-specieswise-cuda-ready.out",
        "global_status=$?",
        'test "$global_status" -eq 2',
        "config production default route fields differ for mode 'global': mode",
        "test ! -s generated-global-production-route.out",
        "gpurec optimize --help",
        "optimize-help.txt",
        'grep -q -- "--require-final-check-ok" optimize-help.txt',
        'grep -q -- "--require-mode-default-optimizer" optimize-help.txt',
        'grep -q -- "--require-production-default-route" optimize-help.txt',
        'grep -q -- "likelihood/gradient route" optimize-help.txt',
        'grep -q -- "rate parameterization" optimize-help.txt',
        'grep -q -- "optimizer route" optimize-help.txt',
        "gpurec validate-config --help",
        "validate-config-help.txt",
        "--require-cuda-backward-ready",
        'grep -q -- "--require-mode-default-optimizer" validate-config-help.txt',
        'grep -q -- "--require-production-default-route" validate-config-help.txt',
        "gpurec summary-info --help",
        "summary-info-help.txt",
        "--summary",
        "--require-converged",
        'grep -q -- "--require-final-check-ok" summary-info-help.txt',
        'grep -q -- "--require-mode-default-optimizer" summary-info-help.txt',
        'grep -q -- "--require-production-default-route" summary-info-help.txt',
        "summary.json",
        "gpurec checkpoint-info --help",
        "checkpoint-info-help.txt",
        'grep -q -- "--require-final-check-ok" checkpoint-info-help.txt',
        'grep -q -- "--require-mode-default-optimizer" checkpoint-info-help.txt',
        'grep -q -- "--require-production-default-route" checkpoint-info-help.txt',
        "gpurec sample --help",
        "gpurec run --help",
        "run-help.txt",
        'grep -q -- "--require-final-check-ok" run-help.txt',
        'grep -q -- "--require-mode-default-optimizer" sample-help.txt',
        'grep -q -- "--require-production-default-route" sample-help.txt',
        'grep -q -- "--require-mode-default-optimizer" run-help.txt',
        'grep -q -- "--require-production-default-route" run-help.txt',
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


def test_release_readiness_documents_resolved_license_readiness():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    for token in (
        "Required Before Redistribution",
        "top-level `LICENSE`",
        "pyproject.toml` license metadata",
        "license classifier",
        "`gpurec = \"gpurec.cli:main\"` console-script entry point",
        "check should pass for redistribution",
        "Do not publish artifacts until the license and all command-surface checks",
    ):
        assert token in guide
    assert "Decide the Rust backtracking binary distribution model" not in guide


def test_release_readiness_governance_list_matches_metadata_checker_core_artifacts():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    checker = _load_check_release_metadata_module()

    governance_core = {
        "CHANGELOG.md",
        "CITATION.cff",
        "Dockerfile",
        "docs/release-notes.md",
        "docs/support-policy.md",
        "docs/versioning-policy.md",
        "docs/publication-checklist.md",
    }
    assert governance_core <= set(checker.REQUIRED_RELEASE_ARTIFACTS)
    for artifact in sorted(governance_core):
        assert artifact in guide


def test_platform_matrix_documents_offline_installation_policy():
    matrix_doc = (ROOT / "docs" / "platform-matrix.md").read_text(encoding="utf-8")

    for token in (
        "Offline Installation Policy",
        "Offline installation is not currently supported as a production guarantee.",
        "pinned git dependency",
        "pre-populated cargo/git cache mirror",
        "gpurec doctor",
        "preprocess-check",
        "backtrack-check",
    ):
        assert token in matrix_doc


def test_release_readiness_documents_sampling_binary_distribution_contract():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    for token in (
        "Sampling Binary Distribution Contract",
        "Source-only installation is the supported production path.",
        "`gpurec sample`, the sampling phase of `gpurec run`, and",
        "`gpurec backtrack-check` must fail with actionable diagnostics",
        "Source archives include `crates/gpurec-backtrack/`",
        "locked\n  `cargo run` fallback",
        "requires Rust/Cargo",
        "pinned `rustree` git dependency",
        "`GPUREC_BACKTRACK_NATIVE_LIB`",
        "does not replace the CLI binary requirement",
    ):
        assert token in guide


def test_release_readiness_documents_preprocess_native_distribution_contract():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    for token in (
        "Preprocessing Native Extension Contract",
        "Source-only installation is the supported production path.",
        "`gpurec validate-config --check-preprocess`, `gpurec optimize`, `gpurec run`,",
        "`gpurec preprocess-check` must fail with actionable diagnostics",
        "Source archives include `crates/gpurec-preprocess/`",
        "Cargo build fallback for the native extension",
        "requires Rust/Cargo",
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
    normalized = " ".join(guide.split())

    for expected in (
        "python -m pip install --no-deps dist/*.whl",
        "smoke_dir=$(mktemp -d)",
        'cd "$smoke_dir"',
        "gpurec preprocess-check --help",
        "gpurec preprocess-check > preprocess-check-missing.txt 2>&1",
        "preprocess_check_status=$?",
        'test "$preprocess_check_status" -eq 1',
        "GPUREC_PREPROCESS_NATIVE_LIB",
        "--preprocess-native-lib",
        "gpurec config-template --mode genewise",
        "gpurec config-template --mode specieswise",
        "gpurec config-template --mode global",
        "cargo build --release --locked --features python-extension",
        '--manifest-path "$repo_root/crates/gpurec-preprocess/Cargo.toml"',
        (
            'export GPUREC_PREPROCESS_NATIVE_LIB="$repo_root/crates/'
            'gpurec-preprocess/target/release/libgpurec_preprocess.so"'
        ),
        "gpurec preprocess-check",
        "generated-genewise-run.json",
        "generated-specieswise-run.json",
        "generated-global-run.json",
        "gpurec validate-config --config generated-genewise-run.json",
        "gpurec validate-config --config generated-specieswise-run.json",
        "gpurec validate-config --config generated-global-run.json",
        "generated-genewise-cuda-ready.err",
        "generated-specieswise-cuda-ready.err",
        "cuda_backward_ready_reason=requires_s_gt_256",
        "no stdout",
        'test "$genewise_cuda_status" -eq 2',
        'test "$specieswise_cuda_status" -eq 2',
        "test ! -s generated-genewise-cuda-ready.out",
        "test ! -s generated-specieswise-cuda-ready.out",
        "generated-global-production-route.err",
        "config production default route fields differ for mode 'global': mode",
        'test "$global_status" -eq 2',
        "test ! -s generated-global-production-route.out",
        "gpurec optimize --help",
        "optimization convergence, final-check, mode-default optimizer, and production-route gates",
        "objective, likelihood/gradient route, rate parameterization",
        "validates both with `gpurec validate-config --require-mode-default-optimizer --require-production-default-route --check-preprocess`",
        "preprocess_checked=true",
        "cuda_backward_ready=false",
        "mode-default `adam` diagnostic and fails `--require-production-default-route` with a `mode` mismatch",
        "source-checkout `crates/gpurec-preprocess` PyO3 extension",
        "`GPUREC_PREPROCESS_NATIVE_LIB`, running `gpurec preprocess-check`, and only then calling installed-wheel",
        "compatible prebuilt native preprocessing extension",
        "mode-default optimizer gates",
        "gpurec validate-config --help",
        "--require-cuda-backward-ready",
        "--require-mode-default-optimizer",
        "--require-production-default-route",
        "gpurec summary-info --help",
        "--require-converged",
        "--require-final-check-ok",
        "gpurec checkpoint-info --help",
        "checkpoint final-check gate",
        "gpurec sample --help",
        "direct sampling mode-default optimizer and production-route gating",
        "gpurec run --help",
        "pre-sampling convergence, final-check, mode-default optimizer, and production-route gates",
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
        assert expected in normalized
    for stale in (
        "default-optimizer gates",
        "default-optimizer audit gate",
        "direct sampling default-optimizer",
        "final-check, default-optimizer",
    ):
        assert stale not in guide


def test_release_readiness_documents_source_archive_preprocess_smoke():
    guide = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")

    for expected in (
        "source-archive\n`validate-config --require-mode-default-optimizer",
        "--require-production-default-route --check-preprocess`",
        "examples/minimal-run-config.json",
        "examples/specieswise-adagrad-restarts-config.json",
        "cuda_backward_ready=false",
        "cuda_backward_ready_reason=requires_s_gt_256",
        "--require-cuda-backward-ready",
        "source-archive examples hard-fail\n`--require-cuda-backward-ready` with empty stdout",
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
        "--require-mode-default-optimizer --require-production-default-route "
        "--check-preprocess"
    ) in readme
    assert (
        "gpurec validate-config --config "
        "examples/specieswise-adagrad-restarts-config.json "
        "--require-mode-default-optimizer --require-production-default-route "
        "--check-preprocess"
    ) in readme
    assert "`cuda_backward_ready=false`" in readme
    assert "`cuda_backward_ready_reason=requires_s_gt_256`" in readme
    assert "`--require-cuda-backward-ready` intentionally exits nonzero" in readme
    assert "gpurec optimize --config examples/minimal-run-config.json" not in readme
    assert "examples/README.md" in readme
    assert "gpurec config-template --mode genewise --output run.json" in readme
    assert (
        "gpurec config-template --mode specieswise --output specieswise-run.json"
        in readme
    )
    assert (
        "gpurec config-template --mode global --output global-diagnostic-run.json"
        in readme
    )
    assert '"optimizer": "auto"' in readme
    assert "`hessian-sgd`, `mode=specieswise` resolves to `adagrad-restarts`" in readme
    assert "`mode=global` resolves to `adam`" in readme
    assert "shared-rate\ndiagnostics" in readme
    assert "will not pass `--require-production-default-route`" in readme
    assert "mode-default `adam` optimizer" in readme
    assert "source-tree config/parser fixture" in readme
    assert 'sets `"device": "cuda"`' in readme
    assert "not a CPU fallback" in readme
    assert "S > 256" in readme
    assert "not an end-to-end optimizer smoke" in readme
    assert "Installed wheels do not install the `examples/` directory" in readme
    assert "global is a shared-rate diagnostic" in readme
    assert "fails the strict production-route gate" in readme
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
    normalized_readme = " ".join(readme.split())
    normalized_guide = " ".join(guide.split())

    assert "source-based installation only for production use" in normalized_readme
    assert (
        "Offline installation is not currently supported as a production guarantee"
        in readme
    )
    assert "Source-only installation is the supported production path." in guide
    assert "Offline installation policy is documented and current in" in guide
    assert "docs/platform-matrix.md" in guide
    assert "`gpurec sample` and the sampling phase of `gpurec run`" in readme
    assert "gpurec validate-config --config examples/minimal-run-config.json" in readme
    assert "--check-preprocess" in readme
    assert "retained Rust parser to run on\nCPU" in readme
    assert "CPU-safe path/reference preflight" in readme
    assert "Workflow preprocessing is implemented by\nthe native Rust" in readme
    assert "Use `gpurec preprocess-check` and `gpurec backtrack-check`" in readme
    assert "CLI exit codes are stable for workflow managers" in readme
    assert "- `0`: command completed successfully." in readme
    assert "- `1`: command ran but failed a runtime or validation gate" in readme
    assert "- `2`: CLI usage or argument parsing error from `argparse`." in readme
    assert "### Preprocessing Native Extension Setup" in readme
    assert "cargo build --locked --release --manifest-path crates/gpurec-backtrack/Cargo.toml" in readme
    assert "GPUREC_PREPROCESS_NATIVE_LIB" in normalized_guide
    assert "gpurec preprocess-check" in readme
    assert "--preprocess-native-lib" in readme
    assert (
        "`GPUREC_PREPROCESS_BIN` is reserved for the subprocess adapter"
        in " ".join(readme.split())
    )
    assert "For a source checkout or unpacked source archive" in readme
    assert "docs/versioning-policy.md" in readme
    assert "docs/publication-checklist.md" in readme
    assert (
        "cargo build --locked --release --manifest-path "
        "crates/gpurec-backtrack/Cargo.toml"
    ) in readme
    assert "GPUREC_BACKTRACK_BIN" in readme
    assert "GPUREC_BACKTRACK_NATIVE_LIB" in readme
    assert "--backtrack-binary" in readme
    assert "gpurec backtrack-check" in readme
    assert 'backend="native"' in readme
    assert "The same `GPUREC_BACKTRACK_BIN` environment variable" in readme
    assert "fallback works from a source checkout or unpacked\nsource archive" in readme
    assert "locked `cargo run` fallback" in normalized_guide
    assert (
        "installed `gpurec config-template --help`"
        in normalized_guide
    )
    assert (
        "`gpurec config-template --mode specieswise`, and "
        "`gpurec validate-config --help`"
        not in normalized_guide
    )
    assert (
        "`gpurec config-template --mode specieswise`, "
        "`gpurec config-template --mode global`, and "
        "`gpurec validate-config --help`"
        in normalized_guide
    )
    assert "`gpurec summary-info --help`" in normalized_guide
    assert "`gpurec checkpoint-info --help`" in normalized_guide
    assert "`gpurec preprocess-check --help`" in normalized_guide
    assert "running `gpurec preprocess-check`, and only then calling installed-wheel" in normalized_guide
    assert (
        "installed `gpurec sample --help`, `gpurec run --help`, and "
        "`gpurec backtrack-check`"
        in normalized_guide
    )
    assert "gpurec backtrack-check --help" in guide

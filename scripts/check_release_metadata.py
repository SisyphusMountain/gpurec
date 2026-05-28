#!/usr/bin/env python3
"""Check release metadata fields before building public artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any


REQUIRED_CLASSIFIERS = {
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
}
REQUIRED_URLS = {"Repository", "Issues", "Documentation"}
REQUIRED_CONSOLE_SCRIPTS = {"gpurec": "gpurec.cli:main"}
_URL_PATTERN = re.compile(r"^https?://\S+$")
REQUIRED_RELEASE_ARTIFACTS = (
    "CHANGELOG.md",
    "CITATION.cff",
    "Dockerfile",
    "LICENSE",
    "docs/README.md",
    "docs/api-contract.md",
    "docs/bioinformatics-quickstart.md",
    "docs/glossary.md",
    "docs/input-preparation.md",
    "docs/known-limitations.md",
    "docs/lean-fast-path.md",
    "docs/long-validation-workflow.md",
    "docs/optimization-workflow-call-graph.md",
    "docs/output-artifacts.md",
    "docs/platform-matrix.md",
    "docs/production-optimization-guide.md",
    "docs/professionalization-audit-progress.tex",
    "docs/publication-checklist.md",
    "docs/release-notes.md",
    "docs/support-policy.md",
    "docs/troubleshooting.md",
    "docs/validation-envelope.md",
    "docs/versioning-policy.md",
    "docs/workflow-examples/README.md",
    "docs/workflow-examples/end-to-end-tutorial/README.md",
    "docs/workflow-examples/end-to-end-tutorial/generate_dataset.py",
    "docs/workflow-examples/end-to-end-tutorial/run.json",
    "docs/workflow-examples/input-validation-fixtures/README.md",
    "docs/workflow-examples/nextflow/README.md",
    "docs/workflow-examples/slurm/README.md",
    "docs/workflow-examples/snakemake/README.md",
    "scripts/generate_dependency_inventory.py",
)


def _readme_metadata_issues(project: dict[str, Any], root: Path) -> list[str]:
    readme = project.get("readme")
    if not readme:
        return ["pyproject.toml [project] must declare readme metadata"]
    if isinstance(readme, str):
        if not (root / readme).is_file():
            return [f"declared readme file does not exist: {readme}"]
        return []
    if isinstance(readme, dict):
        if "file" in readme:
            readme_file = readme["file"]
            if not isinstance(readme_file, str) or not readme_file:
                return ["pyproject.toml [project] readme.file must be a path string"]
            if not (root / readme_file).is_file():
                return [f"declared readme file does not exist: {readme_file}"]
            return []
        if "text" in readme:
            if not isinstance(readme["text"], str) or not readme["text"].strip():
                return ["pyproject.toml [project] readme.text must be nonempty"]
            return []
    return ["pyproject.toml [project] readme must be a file path or table"]


def _license_metadata_issues(project: dict[str, Any], root: Path) -> list[str]:
    license_metadata = project.get("license")
    if not license_metadata:
        return ["pyproject.toml [project] must declare license metadata"]
    if isinstance(license_metadata, str):
        if license_metadata.strip():
            return []
        return ["pyproject.toml [project] license must be nonempty"]
    if isinstance(license_metadata, dict):
        if "file" in license_metadata:
            license_file = license_metadata["file"]
            if not isinstance(license_file, str) or not license_file:
                return ["pyproject.toml [project] license.file must be a path string"]
            if not (root / license_file).is_file():
                return [f"declared license file does not exist: {license_file}"]
            return []
        if "text" in license_metadata:
            if (
                not isinstance(license_metadata["text"], str)
                or not license_metadata["text"].strip()
            ):
                return ["pyproject.toml [project] license.text must be nonempty"]
            return []
    return [
        "pyproject.toml [project] license must be a string, file table, or text table"
    ]


def _release_artifact_issues(root: Path) -> list[str]:
    issues: list[str] = []
    for name in REQUIRED_RELEASE_ARTIFACTS:
        if not (root / name).is_file():
            issues.append(f"missing required release artifact: {name}")
    return issues


def _citation_metadata_issues(project: dict[str, Any], root: Path) -> list[str]:
    citation = root / "CITATION.cff"
    if not citation.is_file():
        return []

    project_version = project.get("version")
    if not isinstance(project_version, str) or not project_version.strip():
        return []

    top_level_version: str | None = None
    has_cff_version = False
    has_title = False
    has_repository_code = False
    citation_repository_code: str | None = None
    preferred_version: str | None = None
    in_preferred = False
    preferred_indent = 0
    for raw_line in citation.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if stripped.startswith("preferred-citation:"):
            in_preferred = True
            preferred_indent = indent
            continue
        if in_preferred and indent <= preferred_indent:
            in_preferred = False
        if stripped.startswith("cff-version:"):
            has_cff_version = True
        if stripped.startswith("title:"):
            has_title = True
        if stripped.startswith("repository-code:"):
            has_repository_code = True
            citation_repository_code = stripped.split(":", 1)[1].strip().strip('"').strip("'")
        if stripped.startswith("version:"):
            value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if in_preferred and preferred_version is None:
                preferred_version = value
            elif top_level_version is None:
                top_level_version = value

    issues: list[str] = []
    if not has_cff_version:
        issues.append("CITATION.cff must declare cff-version")
    if not has_title:
        issues.append("CITATION.cff must declare title")
    if not has_repository_code:
        issues.append("CITATION.cff must declare repository-code")
    if top_level_version is None:
        issues.append("CITATION.cff must declare a top-level version")
    elif top_level_version != project_version:
        issues.append(
            "CITATION.cff top-level version must match pyproject version "
            f"({top_level_version!r} != {project_version!r})"
        )

    if preferred_version is None:
        issues.append("CITATION.cff preferred-citation must declare a version")
    elif preferred_version != project_version:
        issues.append(
            "CITATION.cff preferred-citation version must match pyproject version "
            f"({preferred_version!r} != {project_version!r})"
        )

    project_urls = project.get("urls")
    if isinstance(project_urls, dict):
        repository_url = project_urls.get("Repository")
        if isinstance(repository_url, str) and citation_repository_code is not None:
            if citation_repository_code != repository_url:
                issues.append(
                    "CITATION.cff repository-code must match "
                    "pyproject [project.urls].Repository "
                    f"({citation_repository_code!r} != {repository_url!r})"
                )

    return issues


def _release_notes_version_issues(project: dict[str, Any], root: Path) -> list[str]:
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        return []
    version = version.strip()

    issues: list[str] = []
    changelog = root / "CHANGELOG.md"
    if changelog.is_file():
        text = changelog.read_text(encoding="utf-8")
        if version not in text:
            issues.append(
                "CHANGELOG.md must mention current pyproject version "
                f"{version!r}"
            )

    release_notes = root / "docs" / "release-notes.md"
    if release_notes.is_file():
        text = release_notes.read_text(encoding="utf-8")
        if version not in text:
            issues.append(
                "docs/release-notes.md must mention current pyproject version "
                f"{version!r}"
            )
        lower_text = text.lower()
        if "known limitations" not in lower_text:
            issues.append(
                "docs/release-notes.md must include a 'Known limitations' section"
            )
        if "migration notes" not in lower_text:
            issues.append(
                "docs/release-notes.md must include a 'Migration notes' section"
            )
        if "release artifact notes" not in lower_text:
            issues.append(
                "docs/release-notes.md must include a 'Release artifact notes' section"
            )
        if "dependency and python/torch/cuda support updates" not in lower_text:
            issues.append(
                "docs/release-notes.md must include dependency/python/torch/cuda support-update guidance"
            )

    return issues


def _policy_document_issues(root: Path) -> list[str]:
    issues: list[str] = []

    support_policy = root / "docs" / "support-policy.md"
    if support_policy.is_file():
        text = support_policy.read_text(encoding="utf-8").lower()
        if "production" not in text:
            issues.append(
                "docs/support-policy.md must describe production support scope"
            )
        if "latest release tag" not in text:
            issues.append(
                "docs/support-policy.md must describe latest release tag support"
            )
        support_window_phrases = (
            "support window",
            "python",
            "pytorch",
            "cuda",
            "native artifact",
            "release and patch policy",
            "older tags may receive backports",
            "summary.json",
            "run_manifest.json",
            "checkpoint metadata",
            "gpurec doctor --json",
        )
        for phrase in support_window_phrases:
            if phrase not in text:
                issues.append(
                    "docs/support-policy.md must document support-window phrase: "
                    + phrase
                )

    versioning_policy = root / "docs" / "versioning-policy.md"
    if versioning_policy.is_file():
        text = versioning_policy.read_text(encoding="utf-8").lower()
        if "semantic versioning" not in text:
            issues.append(
                "docs/versioning-policy.md must state semantic versioning policy"
            )
        if "major.minor.patch" not in text:
            issues.append(
                "docs/versioning-policy.md must describe MAJOR.MINOR.PATCH semantics"
            )
        if "latest release tag" not in text:
            issues.append(
                "docs/versioning-policy.md must define latest release tag support line"
            )
        if "backports" not in text or "best-effort" not in text:
            issues.append(
                "docs/versioning-policy.md must define backport support expectations"
            )
        version_consistency_phrases = (
            "pyproject.toml",
            "gpurec.__version__",
            "release notes heading",
        )
        for phrase in version_consistency_phrases:
            if phrase not in text:
                issues.append(
                    "docs/versioning-policy.md must document version-consistency phrase: "
                    + phrase
                )

    return issues


def _publication_checklist_issues(root: Path) -> list[str]:
    publication = root / "docs" / "publication-checklist.md"
    if not publication.is_file():
        return []

    text = publication.read_text(encoding="utf-8").lower()
    issues: list[str] = []
    if "citation.cff" not in text:
        issues.append(
            "docs/publication-checklist.md must mention CITATION.cff metadata"
        )
    if "run_manifest.json" not in text:
        issues.append(
            "docs/publication-checklist.md must mention run_manifest.json"
        )
    if "summary.json" not in text:
        issues.append(
            "docs/publication-checklist.md must mention summary.json"
        )
    if "gpurec doctor --json" not in text:
        issues.append(
            "docs/publication-checklist.md must mention gpurec doctor --json"
        )
    if "gpurec summary-info --summary" not in text or "--json" not in text:
        issues.append(
            "docs/publication-checklist.md must mention gpurec summary-info --summary ... --json"
        )
    if "history.jsonl" not in text:
        issues.append(
            "docs/publication-checklist.md must mention history.jsonl"
        )
    if "checkpoints/" not in text:
        issues.append(
            "docs/publication-checklist.md must mention checkpoints/ archive guidance"
        )
    if "known-limitations.md" not in text:
        issues.append(
            "docs/publication-checklist.md must mention known-limitations.md reporting guidance"
        )
    if "release-notes.md" not in text or "migration notes" not in text:
        issues.append(
            "docs/publication-checklist.md must mention release-notes.md migration notes guidance"
        )
    if "scripts/validate_output_artifacts.py" not in text:
        issues.append(
            "docs/publication-checklist.md must mention scripts/validate_output_artifacts.py gate"
        )
    if "checksums" not in text:
        issues.append(
            "docs/publication-checklist.md must mention checksums evidence guidance"
        )
    if "provenance" not in text:
        issues.append(
            "docs/publication-checklist.md must mention provenance evidence guidance"
        )
    return issues


def _platform_matrix_issues(root: Path) -> list[str]:
    matrix = root / "docs" / "platform-matrix.md"
    if not matrix.is_file():
        return []

    text = matrix.read_text(encoding="utf-8").lower()
    issues: list[str] = []
    if "offline installation policy" not in text:
        issues.append(
            "docs/platform-matrix.md must include an 'Offline Installation Policy' section"
        )
    if "offline installation is not currently supported as a production guarantee" not in text:
        issues.append(
            "docs/platform-matrix.md must explicitly state current offline-installation support policy"
        )
    if "primary supported configuration" not in text:
        issues.append(
            "docs/platform-matrix.md must document the primary supported configuration"
        )
    matrix_terms = (
        "python",
        "pytorch",
        "cuda",
        "triton",
        "gpu",
        "rust",
        "compiler",
        "cargo",
    )
    for term in matrix_terms:
        if term not in text:
            issues.append(
                "docs/platform-matrix.md must document matrix term: " + term
            )
    return issues


def _readme_cli_exit_code_issues(root: Path) -> list[str]:
    readme = root / "README.md"
    if not readme.is_file():
        return []

    text = readme.read_text(encoding="utf-8")
    required_phrases = (
        "CLI exit codes are stable for workflow managers",
        "`0`: command completed successfully.",
        "`1`: command ran but failed a runtime or validation gate",
        "`2`: CLI usage or argument parsing error from `argparse`.",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "README.md must document CLI exit-code policy including: "
                + phrase
            )
    return issues


def _readme_short_user_path_issues(root: Path) -> list[str]:
    readme = root / "README.md"
    if not readme.is_file():
        return []

    text = readme.read_text(encoding="utf-8").lower()
    required_phrases = (
        "install",
        "validate inputs",
        "run optimization",
        "inspect output",
        "sample reconciliations",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append("README.md must document short-path phrase: " + phrase)
    return issues


def _release_readiness_issues(root: Path) -> list[str]:
    readiness = root / "docs" / "release-readiness.md"
    if not readiness.is_file():
        return []

    text = readiness.read_text(encoding="utf-8").lower()
    required_phrases = (
        "python scripts/check_release_metadata.py",
        "validation-envelope.md",
        "scripts/run_long_validation.py",
        "gpurec doctor",
        "writable output directory",
        "quick pr checks",
        "nightly checks",
        "release-candidate checks",
        "final publication checks",
        "checksums",
        "provenance",
        "binary provenance",
        "sha256sum dist/* > dist/sha256sums",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/release-readiness.md must document release gate phrase: "
                + phrase
            )
    return issues


def _validation_envelope_issues(root: Path) -> list[str]:
    envelope = root / "docs" / "validation-envelope.md"
    if not envelope.is_file():
        return []

    text = envelope.read_text(encoding="utf-8").lower()
    required_phrases = (
        "runtime envelope",
        "peak memory",
        "final nll",
        "benchmark evidence",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/validation-envelope.md must document validation evidence term: "
                + phrase
            )
    return issues


def _long_validation_evidence_scope_issues(root: Path) -> list[str]:
    guide = root / "docs" / "long-validation-workflow.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "benchmark evidence",
        "not a hard performance guarantee",
        "not a guaranteed performance contract",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/long-validation-workflow.md must document evidence-scope phrase: "
                + phrase
            )
    return issues


def _long_validation_command_sequence_issues(root: Path) -> list[str]:
    guide = root / "docs" / "long-validation-workflow.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "gpurec doctor --json",
        "gpurec validate-config --check-preprocess --require-cuda-backward-ready",
        "gpurec optimize --require-final-check-ok",
        "gpurec summary-info --require-converged --require-final-check-ok",
        "gpurec sample --checkpoint",
        "scripts/validate_output_artifacts.py",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/long-validation-workflow.md must document command-sequence phrase: "
                + phrase
            )
    return issues


def _troubleshooting_recovery_issues(root: Path) -> list[str]:
    guide = root / "docs" / "troubleshooting.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "by symptom",
        "retryable runtime failures",
        "input contract failures",
        "likely cause",
        "next action",
        "authoritative files",
        "summary.json",
        "history.jsonl",
        "checkpoints/latest.pt",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/troubleshooting.md must document failure-recovery phrase: "
                + phrase
            )
    return issues


def _input_preparation_large_dataset_issues(root: Path) -> list[str]:
    guide = root / "docs" / "input-preparation.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "max-families",
        "sample the first `n` families",
        "memory estimate",
        "clade_budget",
        "family_chunk_size",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/input-preparation.md must document large-dataset phrase: "
                + phrase
            )
    return issues


def _input_preparation_conversion_issues(root: Path) -> list[str]:
    guide = root / "docs" / "input-preparation.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "treerecs",
        "generax",
        "alerax",
        "orthofinder",
        "gene -> species",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/input-preparation.md must document conversion phrase: "
                + phrase
            )
    return issues


def _input_preparation_family_file_shape_issues(root: Path) -> list[str]:
    guide = root / "docs" / "input-preparation.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "multiple families",
        "multiple trees per family",
        "mapping files",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/input-preparation.md must document family-file phrase: "
                + phrase
            )
    return issues


def _output_artifact_snippet_issues(root: Path) -> list[str]:
    guide = root / "docs" / "output-artifacts.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "example output snippets",
        "summary.json",
        "rates_final.tsv",
        "per_fam_likelihoods.tsv",
        "recphyloxml output snippet",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/output-artifacts.md must document output snippet phrase: "
                + phrase
            )
    return issues


def _output_artifact_directory_structure_issues(root: Path) -> list[str]:
    guide = root / "docs" / "output-artifacts.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "run directory structure",
        "output_gpurec/",
        "checkpoints/",
        "reconciliations/",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/output-artifacts.md must document directory-structure phrase: "
                + phrase
            )
    return issues


def _output_artifact_flow_issues(root: Path) -> list[str]:
    guide = root / "docs" / "output-artifacts.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "input/output flow",
        "validate-config --check-preprocess",
        "gpurec optimize",
        "gpurec sample",
        "reconciliations/*.xml",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/output-artifacts.md must document flow phrase: "
                + phrase
            )
    return issues


def _output_artifact_run_manifest_contract_issues(root: Path) -> list[str]:
    guide = root / "docs" / "output-artifacts.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "run_manifest.json",
        "package version",
        "native artifact",
        "pytorch version",
        "cuda availability",
        "gpu name",
        "command line invocation",
        "config hash",
        "random seed",
        "selected route",
        "reproduce or audit",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/output-artifacts.md must document run-manifest phrase: "
                + phrase
            )
    return issues


def _output_artifact_theta_checkpoint_boundary_issues(root: Path) -> list[str]:
    guide = root / "docs" / "output-artifacts.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "theta_final.pt",
        "for inspection only",
        "checkpoint is required",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/output-artifacts.md must document theta/checkpoint phrase: "
                + phrase
            )
    return issues


def _output_artifact_schema_compatibility_issues(root: Path) -> list[str]:
    guide = root / "docs" / "output-artifacts.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "stable schemas",
        "compatibility rules",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/output-artifacts.md must document schema-compatibility phrase: "
                + phrase
            )
    return issues


def _quickstart_lifecycle_issues(root: Path) -> list[str]:
    quickstart = root / "docs" / "bioinformatics-quickstart.md"
    if not quickstart.is_file():
        return []

    text = quickstart.read_text(encoding="utf-8").lower()
    required_phrases = (
        "create config",
        "validate",
        "run",
        "resume",
        "inspect",
        "sample",
        "archive",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/bioinformatics-quickstart.md must document lifecycle stage: "
                + phrase
            )
    return issues


def _quickstart_installation_decision_tree_issues(root: Path) -> list[str]:
    quickstart = root / "docs" / "bioinformatics-quickstart.md"
    if not quickstart.is_file():
        return []

    text = quickstart.read_text(encoding="utf-8").lower()
    required_phrases = (
        "installation decision tree",
        "source checkout or source archive",
        "wheel-only environment",
        "cluster/container workflows",
        "offline installation",
        "gpurec preprocess-check",
        "gpurec backtrack-check",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/bioinformatics-quickstart.md must document installation-decision phrase: "
                + phrase
            )
    return issues


def _quickstart_json_mode_issues(root: Path) -> list[str]:
    quickstart = root / "docs" / "bioinformatics-quickstart.md"
    if not quickstart.is_file():
        return []

    text = quickstart.read_text(encoding="utf-8").lower()
    required_phrases = (
        "structured json mode",
        "gpurec doctor --json",
        "gpurec validate-config --config run.json --json",
        "gpurec summary-info --summary output_gpurec/summary.json --json",
        "gpurec checkpoint-info --checkpoint output_gpurec/checkpoints/latest.pt --json",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/bioinformatics-quickstart.md must document json-mode phrase: "
                + phrase
            )
    return issues


def _quickstart_rng_behavior_issues(root: Path) -> list[str]:
    quickstart = root / "docs" / "bioinformatics-quickstart.md"
    if not quickstart.is_file():
        return []

    text = quickstart.read_text(encoding="utf-8").lower()
    required_phrases = (
        "rng behavior",
        "seed",
        "reproducibility",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/bioinformatics-quickstart.md must document rng phrase: "
                + phrase
            )
    return issues


def _known_limitations_issues(root: Path) -> list[str]:
    guide = root / "docs" / "known-limitations.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "cuda",
        "s > 256",
        "newick subset",
        "quoted labels",
        "embedded delimiters",
        "nested comments",
        "nhx/beast metadata",
        "unary species nodes",
        "non-binary species trees",
        "wheel",
        "external",
        "bf16",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/known-limitations.md must document limitation phrase: "
                + phrase
            )
    return issues


def _end_to_end_tutorial_public_command_issues(root: Path) -> list[str]:
    tutorial = root / "docs" / "workflow-examples" / "end-to-end-tutorial" / "README.md"
    if not tutorial.is_file():
        return []

    text = tutorial.read_text(encoding="utf-8").lower()
    required_phrases = (
        "first successful run tutorial",
        "uses only public commands",
        "tracked or downloadable dataset",
        "writes outputs",
        "samples recphyloxml",
        "gpurec validate-config",
        "--check-preprocess",
        "--require-cuda-backward-ready",
        "gpurec optimize",
        "--resume-from",
        "output_gpurec/checkpoints/latest.pt",
        "gpurec sample",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/workflow-examples/end-to-end-tutorial/README.md must document tutorial phrase: "
                + phrase
            )
    return issues


def _slurm_example_lifecycle_issues(root: Path) -> list[str]:
    guide = root / "docs" / "workflow-examples" / "slurm" / "README.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "gpurec validate-config",
        "--check-preprocess",
        "gpurec optimize",
        "output_gpurec/checkpoints/latest.pt",
        "gpurec sample",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/workflow-examples/slurm/README.md must document lifecycle phrase: "
                + phrase
            )
    return issues


def _snakemake_example_gate_issues(root: Path) -> list[str]:
    guide = root / "docs" / "workflow-examples" / "snakemake" / "README.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "gpurec validate-config",
        "--check-preprocess",
        "--require-converged",
        "--require-final-check-ok",
        "gpurec sample",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/workflow-examples/snakemake/README.md must document gate phrase: "
                + phrase
            )
    return issues


def _nextflow_example_gate_issues(root: Path) -> list[str]:
    guide = root / "docs" / "workflow-examples" / "nextflow" / "README.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "nextflow run main.nf -resume",
        "gpurec validate-config",
        "--check-preprocess",
        "--require-converged",
        "--require-final-check-ok",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/workflow-examples/nextflow/README.md must document gate phrase: "
                + phrase
            )
    return issues


def _workflow_examples_overview_gate_issues(root: Path) -> list[str]:
    guide = root / "docs" / "workflow-examples" / "README.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "snakemake",
        "nextflow",
        "fail fast",
        "--check-preprocess",
        "resume from a checkpoint",
        "reject non-converged outputs",
        "tracked mini public dataset",
        "deterministic",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/workflow-examples/README.md must document acceptance-gate phrase: "
                + phrase
            )
    return issues


def _api_contract_json_mode_issues(root: Path) -> list[str]:
    guide = root / "docs" / "api-contract.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "cli output modes",
        "--json",
        "validate-config",
        "doctor",
        "checkpoint-info",
        "summary-info",
        "json mode emits single json objects with stable keys",
        "required machine path for automation",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/api-contract.md must document json-output contract phrase: "
                + phrase
            )
    return issues


def _api_contract_compatibility_policy_issues(root: Path) -> list[str]:
    guide = root / "docs" / "api-contract.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "compatibility policy",
        "config fields",
        "cli flags",
        "python imports",
        "output artifacts",
        "deprecation warnings",
        "migration notes",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/api-contract.md must document compatibility phrase: " + phrase
            )
    return issues


def _api_contract_exit_code_issues(root: Path) -> list[str]:
    guide = root / "docs" / "api-contract.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "exit status `0`",
        "exit status `1`",
        "exit status `2`",
        "runtime and route-validation failures",
        "cli parse/config errors",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/api-contract.md must document exit-code phrase: " + phrase
            )
    return issues


def _input_validation_fixture_issue_shape_issues(root: Path) -> list[str]:
    guide = root / "docs" / "workflow-examples" / "input-validation-fixtures" / "README.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "file path",
        "family name",
        "affected label",
        "expected format",
        "next action",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/workflow-examples/input-validation-fixtures/README.md must document issue-shape phrase: "
                + phrase
            )
    return issues


def _input_validation_fixture_category_issues(root: Path) -> list[str]:
    guide = root / "docs" / "workflow-examples" / "input-validation-fixtures" / "README.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "every family",
        "missing mapping",
        "duplicate family name",
        "duplicate species mappings",
        "rejected tree",
        "species coverage",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/workflow-examples/input-validation-fixtures/README.md must document category phrase: "
                + phrase
            )
    return issues


def _input_validation_fixture_cpu_safe_issues(root: Path) -> list[str]:
    guide = root / "docs" / "workflow-examples" / "input-validation-fixtures" / "README.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "without constructing a cuda model",
        "validate-inputs",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/workflow-examples/input-validation-fixtures/README.md must document cpu-safe phrase: "
                + phrase
            )
    return issues


def _docs_map_user_vs_research_scope_issues(root: Path) -> list[str]:
    guide = root / "docs" / "README.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "stable user workflows",
        "hogenom-only research scripts",
        "cli help",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append("docs/README.md must document scope phrase: " + phrase)
    return issues


def _glossary_core_term_issues(root: Path) -> list[str]:
    guide = root / "docs" / "glossary.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "`d`",
        "`t`",
        "`l`",
        "`dtl`",
        "`ccp`",
        "`specieswise`",
        "`genewise`",
        "`global`",
        "`recphyloxml`",
        "`nll`",
        "`route`",
        "`solver budget`",
        "`checkpoint`",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append("docs/glossary.md must document glossary term: " + phrase)
    return issues


def _optimization_guide_goal_defaults_issues(root: Path) -> list[str]:
    guide = root / "docs" / "production-optimization-guide.md"
    if not guide.is_file():
        return []

    text = guide.read_text(encoding="utf-8").lower()
    required_phrases = (
        "exploratory run",
        "production genewise run",
        "production specieswise run",
        "diagnostics-only global run",
    )
    issues: list[str] = []
    for phrase in required_phrases:
        if phrase not in text:
            issues.append(
                "docs/production-optimization-guide.md must document user-goal default: "
                + phrase
            )
    return issues


def _url_metadata_issues(project: dict[str, Any]) -> list[str]:
    urls = project.get("urls") or {}
    if not isinstance(urls, dict):
        return ["pyproject.toml [project.urls] must be a table"]
    missing_urls = sorted(REQUIRED_URLS - set(urls))
    issues: list[str] = []
    if missing_urls:
        issues.append(
            "pyproject.toml [project.urls] is missing: " + ", ".join(missing_urls)
        )
    for key in sorted(REQUIRED_URLS & set(urls)):
        value = urls[key]
        if not isinstance(value, str) or not _URL_PATTERN.match(value):
            issues.append(
                f"pyproject.toml [project.urls] {key} must be an http(s) URL"
            )
    return issues


def _script_metadata_issues(project: dict[str, Any]) -> list[str]:
    scripts = project.get("scripts") or {}
    if not isinstance(scripts, dict):
        return ["pyproject.toml [project.scripts] must be a table"]
    issues: list[str] = []
    for name, expected in sorted(REQUIRED_CONSOLE_SCRIPTS.items()):
        value = scripts.get(name)
        if value != expected:
            issues.append(
                f"pyproject.toml [project.scripts] {name} must be {expected!r}"
            )
    return issues


def _load_toml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _parse_minimal_pyproject(text)
    return tomllib.loads(text)


def _parse_minimal_pyproject(text: str) -> dict[str, Any]:
    """Parse the small TOML subset used by this repository's pyproject."""
    data: dict[str, Any] = {"project": {}}
    table: str | None = None
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            table = line.strip("[]")
            if table == "project.urls":
                data["project"].setdefault("urls", {})
            elif table == "project.scripts":
                data["project"].setdefault("scripts", {})
            continue
        if (
            table not in {"project", "project.urls", "project.scripts"}
            or "=" not in line
        ):
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if value == "[":
            values: list[Any] = []
            while index < len(lines):
                item = lines[index].strip().rstrip(",")
                index += 1
                if item == "]":
                    break
                if item:
                    values.append(_parse_toml_value(item))
            _set_project_value(data, table, key, values)
            continue
        _set_project_value(data, table, key, _parse_toml_value(value.rstrip(",")))
    return data


def _parse_toml_value(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("{") and value.endswith("}"):
        return {
            key: parsed.strip('"')
            for key, parsed in re.findall(r"([A-Za-z0-9_-]+)\s*=\s*(\"[^\"]*\")", value)
        }
    return value


def _set_project_value(
    data: dict[str, Any], table: str | None, key: str, value: Any
) -> None:
    project = data["project"]
    if table == "project.urls":
        project.setdefault("urls", {})[key] = value
    elif table == "project.scripts":
        project.setdefault("scripts", {})[key] = value
    else:
        project[key] = value


def release_metadata_issues(root: Path) -> list[str]:
    project_root = root.resolve()
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return [f"missing {pyproject}"]
    data = _load_toml(pyproject)
    project = data.get("project")
    if not isinstance(project, dict):
        return ["pyproject.toml is missing a [project] table"]

    issues: list[str] = []
    if not project.get("authors"):
        issues.append("pyproject.toml [project] must declare authors")
    issues.extend(_readme_metadata_issues(project, project_root))

    classifiers = set(project.get("classifiers") or [])
    missing_classifiers = sorted(REQUIRED_CLASSIFIERS - classifiers)
    if missing_classifiers:
        issues.append(
            "pyproject.toml [project] is missing classifier(s): "
            + ", ".join(missing_classifiers)
        )

    issues.extend(_url_metadata_issues(project))
    issues.extend(_script_metadata_issues(project))
    issues.extend(_release_artifact_issues(project_root))
    issues.extend(_citation_metadata_issues(project, project_root))
    issues.extend(_release_notes_version_issues(project, project_root))
    issues.extend(_policy_document_issues(project_root))
    issues.extend(_publication_checklist_issues(project_root))
    issues.extend(_platform_matrix_issues(project_root))
    issues.extend(_readme_cli_exit_code_issues(project_root))
    issues.extend(_readme_short_user_path_issues(project_root))
    issues.extend(_release_readiness_issues(project_root))
    issues.extend(_validation_envelope_issues(project_root))
    issues.extend(_long_validation_evidence_scope_issues(project_root))
    issues.extend(_long_validation_command_sequence_issues(project_root))
    issues.extend(_troubleshooting_recovery_issues(project_root))
    issues.extend(_input_preparation_large_dataset_issues(project_root))
    issues.extend(_input_preparation_conversion_issues(project_root))
    issues.extend(_input_preparation_family_file_shape_issues(project_root))
    issues.extend(_output_artifact_snippet_issues(project_root))
    issues.extend(_output_artifact_directory_structure_issues(project_root))
    issues.extend(_output_artifact_flow_issues(project_root))
    issues.extend(_output_artifact_run_manifest_contract_issues(project_root))
    issues.extend(_output_artifact_theta_checkpoint_boundary_issues(project_root))
    issues.extend(_output_artifact_schema_compatibility_issues(project_root))
    issues.extend(_quickstart_lifecycle_issues(project_root))
    issues.extend(_quickstart_installation_decision_tree_issues(project_root))
    issues.extend(_quickstart_json_mode_issues(project_root))
    issues.extend(_quickstart_rng_behavior_issues(project_root))
    issues.extend(_known_limitations_issues(project_root))
    issues.extend(_end_to_end_tutorial_public_command_issues(project_root))
    issues.extend(_slurm_example_lifecycle_issues(project_root))
    issues.extend(_snakemake_example_gate_issues(project_root))
    issues.extend(_nextflow_example_gate_issues(project_root))
    issues.extend(_workflow_examples_overview_gate_issues(project_root))
    issues.extend(_api_contract_json_mode_issues(project_root))
    issues.extend(_api_contract_compatibility_policy_issues(project_root))
    issues.extend(_api_contract_exit_code_issues(project_root))
    issues.extend(_input_validation_fixture_issue_shape_issues(project_root))
    issues.extend(_input_validation_fixture_category_issues(project_root))
    issues.extend(_input_validation_fixture_cpu_safe_issues(project_root))
    issues.extend(_docs_map_user_vs_research_scope_issues(project_root))
    issues.extend(_glossary_core_term_issues(project_root))
    issues.extend(_optimization_guide_goal_defaults_issues(project_root))

    license_files = [project_root / "LICENSE", project_root / "LICENSE.txt"]
    has_license_file = any(path.is_file() for path in license_files)
    license_classifier = any(str(item).startswith("License ::") for item in classifiers)
    if not has_license_file:
        issues.append("missing top-level LICENSE file")
    issues.extend(_license_metadata_issues(project, project_root))
    if not license_classifier:
        issues.append("pyproject.toml [project] must include a license classifier")

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing pyproject.toml",
    )
    args = parser.parse_args(argv)

    issues = release_metadata_issues(args.root)
    if not issues:
        print("release metadata check passed")
        return 0
    print("release metadata check failed:")
    for issue in issues:
        print(f"- {issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

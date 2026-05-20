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
            continue
        if table not in {"project", "project.urls"} or "=" not in line:
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


def _set_project_value(data: dict[str, Any], table: str | None, key: str, value: Any) -> None:
    project = data["project"]
    if table == "project.urls":
        project.setdefault("urls", {})[key] = value
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

    urls = project.get("urls") or {}
    missing_urls = sorted(REQUIRED_URLS - set(urls))
    if missing_urls:
        issues.append(
            "pyproject.toml [project.urls] is missing: " + ", ".join(missing_urls)
        )

    license_files = [project_root / "LICENSE", project_root / "LICENSE.txt"]
    has_license_file = any(path.exists() for path in license_files)
    has_license_metadata = bool(project.get("license"))
    license_classifier = any(str(item).startswith("License ::") for item in classifiers)
    if not has_license_file:
        issues.append("missing top-level LICENSE file")
    if not has_license_metadata:
        issues.append("pyproject.toml [project] must declare license metadata")
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

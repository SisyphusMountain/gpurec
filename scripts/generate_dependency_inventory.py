#!/usr/bin/env python3
"""Generate dependency inventories and optional supply-chain checks."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Any
import hashlib
import json
import subprocess
import sys

if sys.version_info < (3, 11):  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]
else:  # pragma: no cover
    import tomllib


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_dependency(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return ", ".join(f"{key}={value!r}" for key, value in sorted(value.items()))
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _collect_python_inventory(root: Path) -> dict[str, Any]:
    project = _read_toml(root / "pyproject.toml").get("project", {})
    optional_dependencies = project.get("optional-dependencies", {})
    return {
        "project": {
            "name": project.get("name"),
            "version": project.get("version"),
            "requires_python": project.get("requires-python"),
            "dependencies": project.get("dependencies", []),
            "optional_dependencies": {
                name: sorted(values)
                for name, values in sorted(optional_dependencies.items())
            },
        },
        "source": {
            "path": "pyproject.toml",
            "sha256": _hash_file(root / "pyproject.toml"),
        },
    }


def _iter_direct_rust_deps(manifest: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        section_value = manifest.get(section, {})
        for name, value in section_value.items():
            yield name, value


def _collect_rust_inventory_for_crate(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _read_toml(manifest_path)
    package_name = manifest["package"]["name"]
    lock_path = manifest_path.parent / "Cargo.lock"
    if not lock_path.exists():
        raise FileNotFoundError(f"missing Cargo.lock for {manifest_path}")

    lock_data = _read_toml(lock_path)
    lock_packages = [
        package
        for package in lock_data.get("package", [])
        if (
            package.get("name") == package_name
            or package.get("name") in manifest.get("dependencies", {})
            or package.get("name") in manifest.get("dev-dependencies", {})
            or package.get("name") in manifest.get("build-dependencies", {})
        )
    ]

    direct_dependencies: dict[str, str] = {}
    git_dependencies: list[dict[str, str | None]] = []
    for dep_name, value in _iter_direct_rust_deps(manifest):
        direct_dependencies[dep_name] = _normalize_dependency(value)
        if isinstance(value, dict) and value.get("git"):
            git_dependencies.append(
                {
                    "name": dep_name,
                    "source": str(value["git"]),
                    "rev": str(value.get("rev", "")) if "rev" in value else None,
                }
            )

    sorted_packages = sorted(
        lock_packages,
        key=lambda item: (item.get("name", ""), item.get("version", "")),
    )

    return {
        "crate": package_name,
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": _hash_file(manifest_path),
        "lockfile_path": str(lock_path.relative_to(root)),
        "lockfile_sha256": _hash_file(lock_path),
        "direct_dependencies": direct_dependencies,
        "locked_package_count": len(sorted_packages),
        "locked_packages": [
            {
                "name": package.get("name"),
                "version": package.get("version"),
                "source": package.get("source"),
                "checksum": package.get("checksum"),
                "dependencies": package.get("dependencies", []),
            }
            for package in sorted_packages
        ],
        "git_dependencies": git_dependencies,
    }


def _collect_rust_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        _collect_rust_inventory_for_crate(root, manifest)
        for manifest in sorted((root / "crates").glob("*/Cargo.toml"))
    ]


def _git_metadata(root: Path) -> dict[str, Any]:
    status: dict[str, str | bool | None] = {"commit": None, "dirty": True}
    try:
        status["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
            )
            .strip()
        )
        status["dirty"] = (
            subprocess.call(["git", "diff", "--quiet"], cwd=str(root)) != 0
        )
    except Exception:
        pass
    return status


def _build_inventory(root: Path) -> dict[str, Any]:
    return {
        "schema": "gpurec.dependency_inventory.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(root.resolve()),
            "git": _git_metadata(root),
        },
        "python": _collect_python_inventory(root),
        "rust": _collect_rust_inventory(root),
    }


def _check_git_dependencies(inventory: dict[str, Any]) -> list[str]:
    violations = []
    for crate in inventory["rust"]:
        for item in crate["git_dependencies"]:
            if not item.get("rev"):
                violations.append(f'{crate["crate"]}:{item["name"]}')
    return violations


def _parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root to inspect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON inventory path.",
    )
    parser.add_argument(
        "--check-git-dependency-pins",
        action="store_true",
        help=(
            "Exit non-zero if any direct Rust git dependency is missing "
            "a pinned revision."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent width for JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    inventory = _build_inventory(root)
    if args.check_git_dependency_pins:
        violations = _check_git_dependencies(inventory)
        if violations:
            print(
                "unresolved git dependencies: "
                + ", ".join(sorted(violations)),
                file=sys.stderr,
            )
            return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, sort_keys=True, indent=args.indent) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

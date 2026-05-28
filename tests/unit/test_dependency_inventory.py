"""Dependency inventory generation smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_inventory(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(ROOT / "scripts" / "generate_dependency_inventory.py")]
    return subprocess.run(
        command + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def test_dependency_inventory_includes_python_and_rust_sections(tmp_path: Path) -> None:
    output = tmp_path / "dependency-inventory.json"
    result = _run_inventory(
        [
            "--root",
            str(ROOT),
            "--output",
            str(output),
            "--check-git-dependency-pins",
        ],
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.exists()

    inventory = json.loads(output.read_text(encoding="utf-8"))
    assert inventory["schema"] == "gpurec.dependency_inventory.v1"
    assert "python" in inventory
    assert "project" in inventory["python"]
    assert inventory["python"]["project"]["name"] == "gpurec"
    assert "rust" in inventory
    assert {crate["crate"] for crate in inventory["rust"]} == {
        "gpurec-backtrack",
        "gpurec-preprocess",
    }


def test_inventory_checker_catches_unpinned_git_dependency(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "dependency-inventory-fixture"\nversion="0.0.0"\nrequires-python=">=3.10"\n', encoding="utf-8")
    crate_root = tmp_path / "crates" / "fixture"
    crate_root.mkdir(parents=True)
    (crate_root / "Cargo.toml").write_text(
        "\n".join(
            [
                '[package]',
                'name = "fixture"',
                'version = "0.0.1"',
                '',
                '[dependencies]',
                'rustree = { git = "https://example.com/rustree.git" }',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (crate_root / "Cargo.lock").write_text(
        "\n".join(
            [
                'version = 4',
                '',
                '[[package]]',
                'name = "fixture"',
                'version = "0.0.1"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "dependency-inventory.json"
    result = _run_inventory(
        [
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--check-git-dependency-pins",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert "unresolved git dependencies" in result.stderr

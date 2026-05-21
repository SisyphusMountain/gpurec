from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "crates" / "gpurec-backtrack" / "Cargo.toml"
FIXTURE = ROOT / "tests" / "fixtures" / "backtracking" / "speciation.json"
SUBPROCESS_TIMEOUT = 120


def test_rust_backtracking_cli_reads_json_fixture_and_writes_recphyloxml(
    tmp_path: Path,
):
    output = tmp_path / "speciation.xml"

    result = subprocess.run(
        [
            "cargo",
            "run",
            "--locked",
            "--quiet",
            "--manifest-path",
            str(MANIFEST),
            "--",
            str(FIXTURE),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    tree = ET.parse(output)
    assert tree.getroot().tag.endswith("recPhylo")

    xml = output.read_text(encoding="utf-8")
    assert '<speciation speciesLocation="Root"/>' in xml
    assert '<leaf speciesLocation="A"/>' in xml
    assert '<leaf speciesLocation="B"/>' in xml

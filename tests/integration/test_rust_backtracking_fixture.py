from __future__ import annotations

import json
import gzip
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "crates" / "gpurec-backtrack" / "Cargo.toml"
FIXTURE = ROOT / "tests" / "fixtures" / "backtracking" / "speciation.json"
SUBPROCESS_TIMEOUT = 120


def _assert_speciation_recphyloxml(path: Path) -> None:
    tree = ET.parse(path)
    assert tree.getroot().tag.endswith("recPhylo")

    xml = path.read_text(encoding="utf-8")
    assert '<speciation speciesLocation="Root"/>' in xml
    assert '<leaf speciesLocation="A"/>' in xml
    assert '<leaf speciesLocation="B"/>' in xml


def test_rust_backtracking_json_fixture_matches_documented_contract():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["species_newick"] == "(A:1,B:1)Root:0;"
    assert payload["species_names_postorder"] == ["A", "B", "Root"]
    assert payload["root_clade"] == 2
    assert payload["leaf_species"] == [0, 1, None]
    assert payload["clade_leaf_labels"] == ["a", "b", ""]
    assert payload["splits"] == [
        {"parent": 2, "left": 0, "right": 1, "log_prob": 0.0}
    ]
    assert payload["pi"]["rows"] == 3
    assert payload["pi"]["cols"] == 3
    assert len(payload["pi"]["data"]) == 9
    assert payload["pibar"]["rows"] == 3
    assert payload["pibar"]["cols"] == 3
    assert len(payload["pibar"]["data"]) == 9
    assert payload["ebar"] == [-1.0e300, -1.0e300, -1.0e300]
    assert payload["origination_probs"] == [0.0, 0.0, 1.0]
    assert payload["seed"] == 7
    assert payload["max_events"] == 32


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
    _assert_speciation_recphyloxml(output)


def test_rust_backtracking_cli_writes_multi_sample_output_dir(
    tmp_path: Path,
):
    output_dir = tmp_path / "samples"

    result = subprocess.run(
        [
            "cargo",
            "run",
            "--locked",
            "--quiet",
            "--manifest-path",
            str(MANIFEST),
            "--",
            "--samples",
            "2",
            "--output-dir",
            str(output_dir),
            str(FIXTURE),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "sample_0.xml",
        "sample_1.xml",
    ]
    _assert_speciation_recphyloxml(output_dir / "sample_0.xml")
    _assert_speciation_recphyloxml(output_dir / "sample_1.xml")


def test_rust_backtracking_cli_writes_gzip_output_dir(
    tmp_path: Path,
):
    output_dir = tmp_path / "samples-gz"

    result = subprocess.run(
        [
            "cargo",
            "run",
            "--locked",
            "--quiet",
            "--manifest-path",
            str(MANIFEST),
            "--",
            "--samples",
            "1",
            "--output-dir",
            str(output_dir),
            "--compression",
            "gzip",
            "--serial",
            str(FIXTURE),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output_dir.iterdir()) == ["sample_0.xml.gz"]
    xml = gzip.decompress((output_dir / "sample_0.xml.gz").read_bytes()).decode("utf-8")
    assert '<speciation speciesLocation="Root"/>' in xml
    assert '<leaf speciesLocation="A"/>' in xml

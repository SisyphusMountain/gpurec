from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch

from gpurec.core.model import parse_alerax_family_file
from gpurec.core.preprocess_rust import RustPreprocessExtension
from gpurec.core.preprocess_rust import RustPreprocessSubprocessExtension


ROOT = Path(__file__).resolve().parents[2]
RUST_MANIFEST = ROOT / "crates" / "gpurec-preprocess" / "Cargo.toml"
HOGENOM_STYLE_FIXTURE = ROOT / "tests" / "fixtures" / "alerax_hogenom_style"
SUBPROCESS_TIMEOUT = 60


def _plain(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _finite_or_neg_inf(value: float | None) -> float:
    return -math.inf if value is None else float(value)


def _assert_float_list_close(actual: list[float | None], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        got_value = _finite_or_neg_inf(got)
        if math.isinf(want):
            assert got_value == want
        else:
            assert got_value == pytest.approx(want)


def _flatten_float_list(values: list[Any]) -> list[float]:
    flat: list[float] = []
    for value in values:
        if isinstance(value, list):
            flat.extend(_flatten_float_list(value))
        else:
            flat.append(float(value))
    return flat


def _assert_preprocess_outputs_match(
    rust: dict[str, Any],
    expected: dict[str, Any],
    *,
    family_name: str,
    include_species_matrices: bool,
) -> None:
    expected_species = expected["species"]
    rust_species = rust["species"]
    assert rust_species["s"] == expected_species["S"]
    assert rust_species["names"] == expected_species["names"]
    assert rust_species["s_p_indexes"] == expected_species["s_P_indexes"]
    assert rust_species["s_c12_indexes"] == expected_species["s_C12_indexes"]
    _assert_float_list_close(rust_species["unnorm_row_max"], expected_species["unnorm_row_max"])
    if include_species_matrices:
        _assert_float_list_close(
            rust_species["ancestors_dense"],
            _flatten_float_list(expected_species["ancestors_dense"]),
        )
        _assert_float_list_close(
            rust_species["recipients_mat"],
            _flatten_float_list(expected_species["Recipients_mat"]),
        )

    expected_family = expected["families"][family_name]
    rust_family = rust["families"][family_name]
    assert rust_family["root_clade_id"] == expected_family["root_clade_id"]
    assert rust_family["leaf_row_index"] == expected_family["leaf_row_index"]
    assert rust_family["leaf_col_index"] == expected_family["leaf_col_index"]

    expected_ccp = expected_family["ccp"]
    rust_ccp = rust_family["ccp"]
    assert rust_ccp["c"] == expected_ccp["C"]
    assert rust_ccp["n_splits"] == expected_ccp["N_splits"]
    assert rust_ccp["root_clade_id"] == expected_ccp["root_clade_id"]
    assert rust_ccp["split_counts"] == expected_ccp["split_counts"]
    assert rust_ccp["split_parents_sorted"] == expected_ccp["split_parents_sorted"]
    assert rust_ccp["split_leftrights_sorted"] == expected_ccp["split_leftrights_sorted"]
    assert rust_ccp["num_segs_ge2"] == expected_ccp["num_segs_ge2"]
    assert rust_ccp["num_segs_eq1"] == expected_ccp["num_segs_eq1"]
    assert rust_ccp["end_rows_ge2"] == expected_ccp["end_rows_ge2"]
    assert rust_ccp["clade_leaf_labels"] == expected_ccp["clade_leaf_labels"]
    _assert_float_list_close(
        rust_ccp["log_split_probs_sorted"],
        expected_ccp["log_split_probs_sorted"],
    )


def _assert_raw_outputs_match(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    family_name: str,
    include_species_matrices: bool,
) -> None:
    actual_species = actual["species"]
    expected_species = expected["species"]
    assert actual_species["S"] == expected_species["S"]
    assert actual_species["names"] == expected_species["names"]
    assert actual_species["s_P_indexes"] == expected_species["s_P_indexes"]
    assert actual_species["s_C12_indexes"] == expected_species["s_C12_indexes"]
    _assert_float_list_close(
        actual_species["unnorm_row_max"],
        expected_species["unnorm_row_max"],
    )
    if include_species_matrices:
        _assert_float_list_close(
            _flatten_float_list(actual_species["ancestors_dense"]),
            _flatten_float_list(expected_species["ancestors_dense"]),
        )
        _assert_float_list_close(
            _flatten_float_list(actual_species["Recipients_mat"]),
            _flatten_float_list(expected_species["Recipients_mat"]),
        )

    actual_family = actual["families"][family_name]
    expected_family = expected["families"][family_name]
    assert actual_family["root_clade_id"] == expected_family["root_clade_id"]
    assert actual_family["leaf_row_index"] == expected_family["leaf_row_index"]
    assert actual_family["leaf_col_index"] == expected_family["leaf_col_index"]

    actual_ccp = actual_family["ccp"]
    expected_ccp = expected_family["ccp"]
    assert actual_ccp["C"] == expected_ccp["C"]
    assert actual_ccp["N_splits"] == expected_ccp["N_splits"]
    assert actual_ccp["root_clade_id"] == expected_ccp["root_clade_id"]
    assert actual_ccp["split_counts"] == expected_ccp["split_counts"]
    assert actual_ccp["split_parents_sorted"] == expected_ccp["split_parents_sorted"]
    assert actual_ccp["split_leftrights_sorted"] == expected_ccp["split_leftrights_sorted"]
    assert actual_ccp["num_segs_ge2"] == expected_ccp["num_segs_ge2"]
    assert actual_ccp["num_segs_eq1"] == expected_ccp["num_segs_eq1"]
    assert actual_ccp["end_rows_ge2"] == expected_ccp["end_rows_ge2"]
    assert actual_ccp["clade_leaf_labels"] == expected_ccp["clade_leaf_labels"]
    _assert_float_list_close(
        actual_ccp["log_split_probs_sorted"],
        expected_ccp["log_split_probs_sorted"],
    )


def _run_rust_preprocess(request_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--locked",
            "--quiet",
            "--manifest-path",
            str(RUST_MANIFEST),
            "--",
            str(request_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=SUBPROCESS_TIMEOUT,
    )
    return json.loads(result.stdout)


def _preprocess_with_adapter(
    adapter: RustPreprocessExtension | RustPreprocessSubprocessExtension,
    species: Path,
    families: dict[str, list[str]],
    *,
    leaf_species_maps: dict[str, dict[str, str]],
) -> dict[str, Any]:
    return adapter.preprocess_multiple_families(
        str(species),
        families,
        leaf_species_maps=leaf_species_maps,
        include_details=True,
        include_species_matrices=False,
        include_debug_details=False,
        include_scheduler_details=False,
        include_legacy_ccp_details=False,
        num_threads=1,
    )


def test_rust_preprocess_cli_matches_native_adapter_compact_arrays(tmp_path: Path):
    species = tmp_path / "species.nwk"
    gene = tmp_path / "gene.nwk"
    request_path = tmp_path / "request.json"
    species.write_text("((A,B)AB,C)Root;\n", encoding="utf-8")
    gene.write_text("((a,b),c);((a,c),b)\n", encoding="utf-8")
    leaf_map = {"a": "A", "b": "B", "c": "C"}
    request_path.write_text(
        json.dumps(
            {
                "species_path": str(species),
                "families": {"fam": [str(gene)]},
                "leaf_species_maps": {"fam": leaf_map},
                "include_species_matrices": False,
            }
        ),
        encoding="utf-8",
    )

    rust = _run_rust_preprocess(request_path)
    expected = _plain(
        RustPreprocessExtension().preprocess_multiple_families(
            str(species),
            {"fam": [str(gene)]},
            leaf_species_maps={"fam": leaf_map},
            include_details=True,
            include_species_matrices=False,
            include_debug_details=False,
            include_scheduler_details=False,
            include_legacy_ccp_details=False,
            num_threads=1,
        )
    )

    _assert_preprocess_outputs_match(
        rust,
        expected,
        family_name="fam",
        include_species_matrices=False,
    )


def test_rust_preprocess_cli_matches_native_adapter_multifurcating_gene_tree_and_species_matrices(
    tmp_path: Path,
):
    species = tmp_path / "species.nwk"
    gene = tmp_path / "gene.nwk"
    request_path = tmp_path / "request.json"
    species.write_text("(((A,B)AB,C)ABC,D)Root;\n", encoding="utf-8")
    gene.write_text("(a,b,c,d)g;\n", encoding="utf-8")
    leaf_map = {"a": "A", "b": "B", "c": "C", "d": "D"}
    request_path.write_text(
        json.dumps(
            {
                "species_path": str(species),
                "families": {"fam": [str(gene)]},
                "leaf_species_maps": {"fam": leaf_map},
                "include_species_matrices": True,
                "num_threads": 2,
            }
        ),
        encoding="utf-8",
    )

    rust = _run_rust_preprocess(request_path)
    expected = _plain(
        RustPreprocessExtension().preprocess_multiple_families(
            str(species),
            {"fam": [str(gene)]},
            leaf_species_maps={"fam": leaf_map},
            include_details=True,
            include_species_matrices=True,
            include_debug_details=False,
            include_scheduler_details=False,
            include_legacy_ccp_details=False,
            num_threads=1,
        )
    )

    _assert_preprocess_outputs_match(
        rust,
        expected,
        family_name="fam",
        include_species_matrices=True,
    )


def test_rust_preprocess_cli_matches_native_adapter_multiple_families_and_branch_lengths(
    tmp_path: Path,
):
    species = tmp_path / "species.nwk"
    gene_one = tmp_path / "gene_one.nwk"
    gene_two = tmp_path / "gene_two.nwk"
    request_path = tmp_path / "request.json"
    species.write_text(
        "((A:1e-3,B:+2.0E+1)AB:-0.3,(C:.4,D:5e-1)CD:6E-2)Root;\n",
        encoding="utf-8",
    )
    gene_one.write_text(
        "((a:1e-3,b:+2.0E+1)x:-0.3,(c:.4,d:5e-1)y)r;\n",
        encoding="utf-8",
    )
    gene_two.write_text(
        "(a:1.0e0,c:+2.0,d:-3.0E-1)r:4.0;((a,d:1e+0)ad,c:.25)r\n",
        encoding="utf-8",
    )
    leaf_maps = {
        "fam1": {"a": "A", "b": "B", "c": "C", "d": "D"},
        "fam2": {"a": "A", "c": "C", "d": "D"},
    }
    families = {"fam1": [str(gene_one)], "fam2": [str(gene_two)]}
    request_path.write_text(
        json.dumps(
            {
                "species_path": str(species),
                "families": families,
                "leaf_species_maps": leaf_maps,
                "include_species_matrices": False,
                "num_threads": 2,
            }
        ),
        encoding="utf-8",
    )

    rust = _run_rust_preprocess(request_path)
    expected = _plain(
        RustPreprocessExtension().preprocess_multiple_families(
            str(species),
            families,
            leaf_species_maps=leaf_maps,
            include_details=True,
            include_species_matrices=False,
            include_debug_details=False,
            include_scheduler_details=False,
            include_legacy_ccp_details=False,
            num_threads=2,
        )
    )

    _assert_preprocess_outputs_match(
        rust,
        expected,
        family_name="fam1",
        include_species_matrices=False,
    )
    _assert_preprocess_outputs_match(
        rust,
        expected,
        family_name="fam2",
        include_species_matrices=False,
    )


def test_rust_preprocess_cli_matches_native_adapter_hogenom_style_alerax_fixture(
    tmp_path: Path,
):
    species = HOGENOM_STYLE_FIXTURE / "species.nwk"
    families_file = HOGENOM_STYLE_FIXTURE / "families.txt"
    names, tree_paths, leaf_maps = parse_alerax_family_file(families_file)
    families = {name: paths for name, paths in zip(names, tree_paths)}
    leaf_species_maps = {
        name: mapping
        for name, mapping in zip(names, leaf_maps)
        if mapping
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "species_path": str(species),
                "families": families,
                "leaf_species_maps": leaf_species_maps,
                "include_species_matrices": True,
                "num_threads": 2,
            }
        ),
        encoding="utf-8",
    )

    rust = _run_rust_preprocess(request_path)
    expected = _plain(
        RustPreprocessExtension().preprocess_multiple_families(
            str(species),
            families,
            leaf_species_maps=leaf_species_maps,
            include_details=True,
            include_species_matrices=True,
            include_debug_details=False,
            include_scheduler_details=False,
            include_legacy_ccp_details=False,
            num_threads=2,
        )
    )

    assert names == ["ribosomal_like", "transfer_like"]
    for family_name in names:
        _assert_preprocess_outputs_match(
            rust,
            expected,
            family_name=family_name,
            include_species_matrices=True,
        )


@pytest.mark.parametrize(
    ("leaf_map", "expected"),
    [
        (
            {"a": "A", "b": "B"},
            "Gene leaf c is missing from mapping for family fam",
        ),
        (
            {"a": "A", "b": "B", "c": "Z"},
            "Species Z not found for gene leaf c",
        ),
    ],
)
def test_rust_preprocess_adapters_report_mapping_errors_consistently(
    tmp_path: Path,
    leaf_map: dict[str, str],
    expected: str,
):
    species = tmp_path / "species.nwk"
    gene = tmp_path / "gene.nwk"
    species.write_text("((A,B)AB,C)Root;\n", encoding="utf-8")
    gene.write_text("((a,b),c);\n", encoding="utf-8")
    families = {"fam": [str(gene)]}

    for adapter in (RustPreprocessExtension(), RustPreprocessSubprocessExtension()):
        with pytest.raises(RuntimeError) as exc_info:
            _preprocess_with_adapter(
                adapter,
                species,
                families,
                leaf_species_maps={"fam": leaf_map},
            )
        assert expected in str(exc_info.value)


def test_rust_preprocess_adapters_report_malformed_newick_consistently(tmp_path: Path):
    species = tmp_path / "species.nwk"
    gene = tmp_path / "gene.nwk"
    species.write_text("((A,B)AB,C)Root;\n", encoding="utf-8")
    gene.write_text("((a,b),c\n", encoding="utf-8")
    families = {"fam": [str(gene)]}
    leaf_map = {"a": "A", "b": "B", "c": "C"}

    for adapter in (RustPreprocessExtension(), RustPreprocessSubprocessExtension()):
        with pytest.raises(RuntimeError) as exc_info:
            _preprocess_with_adapter(
                adapter,
                species,
                families,
                leaf_species_maps={"fam": leaf_map},
            )
        assert "Unexpected end while parsing children" in str(exc_info.value)
        assert str(gene) in str(exc_info.value)


def test_rust_preprocess_native_adapter_matches_subprocess_adapter_raw_contract(tmp_path: Path):
    species = tmp_path / "species.nwk"
    gene = tmp_path / "gene.nwk"
    species.write_text("((A,B)AB,C)Root;\n", encoding="utf-8")
    gene.write_text("((a,b),c);((a,c),b)\n", encoding="utf-8")
    leaf_map = {"a": "A", "b": "B", "c": "C"}

    rust = _plain(
        RustPreprocessExtension().preprocess_multiple_families(
            str(species),
            {"fam": [str(gene)]},
            leaf_species_maps={"fam": leaf_map},
            include_details=True,
            include_species_matrices=True,
            include_debug_details=False,
            include_scheduler_details=False,
            include_legacy_ccp_details=False,
            num_threads=2,
        )
    )
    subprocess_adapter = _plain(
        RustPreprocessSubprocessExtension().preprocess_multiple_families(
            str(species),
            {"fam": [str(gene)]},
            leaf_species_maps={"fam": leaf_map},
            include_details=True,
            include_species_matrices=True,
            include_debug_details=False,
            include_scheduler_details=False,
            include_legacy_ccp_details=False,
            num_threads=2,
        )
    )

    _assert_raw_outputs_match(
        rust,
        subprocess_adapter,
        family_name="fam",
        include_species_matrices=True,
    )

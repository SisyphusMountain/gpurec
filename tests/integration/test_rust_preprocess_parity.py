from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.preprocess_rust import RustPreprocessExtension


ROOT = Path(__file__).resolve().parents[2]
RUST_MANIFEST = ROOT / "crates" / "gpurec-preprocess" / "Cargo.toml"
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
    cpp: dict[str, Any],
    *,
    family_name: str,
    include_species_matrices: bool,
) -> None:
    cpp_species = cpp["species"]
    rust_species = rust["species"]
    assert rust_species["s"] == cpp_species["S"]
    assert rust_species["names"] == cpp_species["names"]
    assert rust_species["s_p_indexes"] == cpp_species["s_P_indexes"]
    assert rust_species["s_c12_indexes"] == cpp_species["s_C12_indexes"]
    _assert_float_list_close(rust_species["unnorm_row_max"], cpp_species["unnorm_row_max"])
    if include_species_matrices:
        _assert_float_list_close(
            rust_species["ancestors_dense"],
            _flatten_float_list(cpp_species["ancestors_dense"]),
        )
        _assert_float_list_close(
            rust_species["recipients_mat"],
            _flatten_float_list(cpp_species["Recipients_mat"]),
        )

    cpp_family = cpp["families"][family_name]
    rust_family = rust["families"][family_name]
    assert rust_family["root_clade_id"] == cpp_family["root_clade_id"]
    assert rust_family["leaf_row_index"] == cpp_family["leaf_row_index"]
    assert rust_family["leaf_col_index"] == cpp_family["leaf_col_index"]

    cpp_ccp = cpp_family["ccp"]
    rust_ccp = rust_family["ccp"]
    assert rust_ccp["c"] == cpp_ccp["C"]
    assert rust_ccp["n_splits"] == cpp_ccp["N_splits"]
    assert rust_ccp["root_clade_id"] == cpp_ccp["root_clade_id"]
    assert rust_ccp["split_counts"] == cpp_ccp["split_counts"]
    assert rust_ccp["split_parents_sorted"] == cpp_ccp["split_parents_sorted"]
    assert rust_ccp["split_leftrights_sorted"] == cpp_ccp["split_leftrights_sorted"]
    assert rust_ccp["num_segs_ge2"] == cpp_ccp["num_segs_ge2"]
    assert rust_ccp["num_segs_eq1"] == cpp_ccp["num_segs_eq1"]
    assert rust_ccp["end_rows_ge2"] == cpp_ccp["end_rows_ge2"]
    assert rust_ccp["clade_leaf_labels"] == cpp_ccp["clade_leaf_labels"]
    _assert_float_list_close(
        rust_ccp["log_split_probs_sorted"],
        cpp_ccp["log_split_probs_sorted"],
    )


def _assert_cpp_shaped_outputs_match(
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


def test_rust_preprocess_matches_cpp_compact_arrays(tmp_path: Path):
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
    cpp = _plain(
        _load_extension().preprocess_multiple_families(
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
        cpp,
        family_name="fam",
        include_species_matrices=False,
    )


def test_rust_preprocess_matches_cpp_multifurcating_gene_tree_and_species_matrices(
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
    cpp = _plain(
        _load_extension().preprocess_multiple_families(
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
        cpp,
        family_name="fam",
        include_species_matrices=True,
    )


def test_rust_preprocess_matches_cpp_multiple_families_and_branch_lengths(
    tmp_path: Path,
):
    species = tmp_path / "species.nwk"
    gene_one = tmp_path / "gene_one.nwk"
    gene_two = tmp_path / "gene_two.nwk"
    request_path = tmp_path / "request.json"
    species.write_text(
        "((A:0.1,B:0.2)AB:0.3,(C:0.4,D:0.5)CD:0.6)Root;\n",
        encoding="utf-8",
    )
    gene_one.write_text("((a:0.1,b:0.2)x:0.3,(c:0.4,d:0.5)y)r;\n", encoding="utf-8")
    gene_two.write_text("(a:1.0,c:2.0,d:3.0)r:4.0;((a,d)ad,c)r\n", encoding="utf-8")
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
    cpp = _plain(
        _load_extension().preprocess_multiple_families(
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
        cpp,
        family_name="fam1",
        include_species_matrices=False,
    )
    _assert_preprocess_outputs_match(
        rust,
        cpp,
        family_name="fam2",
        include_species_matrices=False,
    )


def test_rust_preprocess_python_adapter_matches_cpp_raw_contract(tmp_path: Path):
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
    cpp = _plain(
        _load_extension().preprocess_multiple_families(
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

    _assert_cpp_shaped_outputs_match(
        rust,
        cpp,
        family_name="fam",
        include_species_matrices=True,
    )

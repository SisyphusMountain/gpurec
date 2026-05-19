from __future__ import annotations

import pytest
import torch

from gpurec.core.model import (
    GeneDataset,
    _load_preprocess_cache,
    _validate_family_preprocess_cache,
    _validate_species_preprocess_cache,
    parse_alerax_family_file,
    parse_alerax_mapping_file,
)


def _write(path, text: str):
    path.write_text(text)
    return path


def test_alerax_family_selection_validates_before_io(tmp_path):
    missing = tmp_path / "missing_families.txt"

    with pytest.raises(ValueError, match="start"):
        parse_alerax_family_file(missing, start=-1)
    with pytest.raises(ValueError, match="max_families"):
        parse_alerax_family_file(missing, max_families=0)


def test_preprocess_cache_load_uses_weights_only(tmp_path, monkeypatch):
    calls = []

    def fake_load(path, *, map_location, weights_only):
        calls.append(
            {
                "path": path,
                "map_location": map_location,
                "weights_only": weights_only,
            }
        )
        return {"S": 3}

    monkeypatch.setattr("gpurec.core.model.torch.load", fake_load)

    payload = _load_preprocess_cache(
        tmp_path / "species.pt",
        label="species",
        required_keys=("S",),
    )

    assert payload == {"S": 3}
    assert calls == [
        {
            "path": tmp_path / "species.pt",
            "map_location": "cpu",
            "weights_only": True,
        }
    ]


def _valid_species_cache_payload() -> dict[str, object]:
    return {
        "S": 3,
        "names": ["Root", "A", "B"],
        "s_P_indexes": torch.tensor([0, 3], dtype=torch.long),
        "s_C12_indexes": torch.tensor([1, 2], dtype=torch.long),
        "unnorm_row_max": torch.zeros(3, dtype=torch.float64),
        "species_name_to_index": {"Root": 0, "A": 1, "B": 2},
    }


def _valid_family_cache_payload() -> dict[str, object]:
    return {
        "ccp": {
            "C": 3,
            "N_splits": 1,
            "root_clade_id": 0,
            "num_segs_eq1": 1,
            "end_rows_ge2": 0,
            "split_counts": torch.tensor([1, 0, 0], dtype=torch.long),
            "split_parents_sorted": torch.tensor([0], dtype=torch.long),
            "split_leftrights_sorted": torch.tensor([1, 2], dtype=torch.long),
            "log_split_probs_sorted": torch.zeros(1, dtype=torch.float64),
            "clade_leaf_labels": ["", "a", "b"],
        },
        "leaf_row_index": torch.tensor([1, 2], dtype=torch.long),
        "leaf_col_index": torch.tensor([1, 2], dtype=torch.long),
    }


def test_species_preprocess_cache_rejects_missing_nested_helpers(tmp_path):
    path = tmp_path / "species.pt"
    torch.save({"S": 3}, path)

    with pytest.raises(RuntimeError, match="names"):
        _load_preprocess_cache(
            path,
            label="species",
            required_keys=("S",),
            validator=_validate_species_preprocess_cache,
        )


def test_species_preprocess_cache_rejects_inconsistent_topology_lengths(tmp_path):
    path = tmp_path / "species.pt"
    payload = _valid_species_cache_payload()
    payload["s_C12_indexes"] = torch.tensor([1], dtype=torch.long)
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="same length"):
        _load_preprocess_cache(
            path,
            label="species",
            required_keys=("S",),
            validator=_validate_species_preprocess_cache,
        )


def test_family_preprocess_cache_rejects_missing_nested_ccp_helpers(tmp_path):
    path = tmp_path / "family.pt"
    torch.save(
        {
            "ccp": {"C": 3, "N_splits": 1},
            "leaf_row_index": torch.tensor([1, 2], dtype=torch.long),
            "leaf_col_index": torch.tensor([1, 2], dtype=torch.long),
        },
        path,
    )

    with pytest.raises(RuntimeError, match="root_clade_id"):
        _load_preprocess_cache(
            path,
            label="family 'fam0'",
            required_keys=("ccp", "leaf_row_index", "leaf_col_index"),
            validator=_validate_family_preprocess_cache,
        )


def test_family_preprocess_cache_rejects_wrong_index_tensor_dtype(tmp_path):
    path = tmp_path / "family.pt"
    payload = _valid_family_cache_payload()
    payload["ccp"]["split_parents_sorted"] = torch.tensor([0.0], dtype=torch.float32)
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="split_parents_sorted.*torch.int64"):
        _load_preprocess_cache(
            path,
            label="family 'fam0'",
            required_keys=("ccp", "leaf_row_index", "leaf_col_index"),
            validator=_validate_family_preprocess_cache,
        )


def test_family_preprocess_cache_rejects_inconsistent_split_lengths(tmp_path):
    path = tmp_path / "family.pt"
    payload = _valid_family_cache_payload()
    payload["ccp"]["split_leftrights_sorted"] = torch.tensor([1], dtype=torch.long)
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="split_leftrights_sorted.*expected 2"):
        _load_preprocess_cache(
            path,
            label="family 'fam0'",
            required_keys=("ccp", "leaf_row_index", "leaf_col_index"),
            validator=_validate_family_preprocess_cache,
        )


@pytest.mark.parametrize(
    ("bad_line", "message"),
    [
        ("maping = fam.map", "unknown AleRax family key"),
        ("gene_trees = fam.nwk", "unknown AleRax family key"),
        ("mapping fam.map", "invalid AleRax family line"),
        ("mapping = ", "empty AleRax family value"),
    ],
)
def test_alerax_family_file_rejects_malformed_family_entries(
    tmp_path,
    bad_line: str,
    message: str,
):
    _write(tmp_path / "fam.nwk", "(a:1,b:1);\n")
    _write(tmp_path / "fam.map", "A:a\nB:b\n")
    families = _write(
        tmp_path / "families.txt",
        "\n".join(
            [
                "[FAMILIES]",
                "- fam0",
                "starting_gene_tree = fam.nwk",
                bad_line,
                "",
            ]
        ),
    )

    with pytest.raises(ValueError, match=message):
        parse_alerax_family_file(families)


def test_alerax_family_file_rejects_duplicate_family_names(tmp_path):
    _write(tmp_path / "a.nwk", "(a:1,b:1);\n")
    _write(tmp_path / "b.nwk", "(c:1,d:1);\n")
    families = _write(
        tmp_path / "families.txt",
        "\n".join(
            [
                "[FAMILIES]",
                "- duplicated",
                "starting_gene_tree = a.nwk",
                "- duplicated",
                "starting_gene_tree = b.nwk",
                "",
            ]
        ),
    )

    with pytest.raises(ValueError, match="duplicate AleRax family name 'duplicated'"):
        parse_alerax_family_file(families)


def test_alerax_mapping_file_rejects_duplicate_gene_assignments(tmp_path):
    mapping = _write(
        tmp_path / "fam.map",
        "SpeciesA: gene1\nSpeciesB: gene1\n",
    )

    with pytest.raises(ValueError, match="gene1.*SpeciesA.*SpeciesB"):
        parse_alerax_mapping_file(mapping)


def test_alerax_family_file_multi_tree_ccp_matches_split_files(tmp_path):
    species_tree = _write(tmp_path / "sp.nwk", "((A:1,B:1)AB:1,C:1)root;\n")
    tree_a = _write(tmp_path / "a.nwk", "((a1:1,b1:1):1,c1:1);\n")
    tree_b = _write(tmp_path / "b.nwk", "(a1:1,(b1:1,c1:1):1);\n")
    tree_dist = _write(
        tmp_path / "fam.trees",
        tree_a.read_text() + tree_b.read_text(),
    )
    mapping = _write(tmp_path / "fam.map", "A:a1\nB:b1\nC:c1\n")
    families = _write(
        tmp_path / "families.txt",
        "\n".join(
            [
                "[FAMILIES]",
                "- fam0",
                f"starting_gene_tree = {tree_dist.name}",
                f"mapping = {mapping.name}",
                "",
            ]
        ),
    )

    names, tree_paths, leaf_maps = parse_alerax_family_file(families)
    assert names == ["fam0"]
    assert tree_paths == [[str(tree_dist)]]
    assert leaf_maps == [{"a1": "A", "b1": "B", "c1": "C"}]

    from_distribution = GeneDataset(
        species_tree,
        tree_paths,
        genewise=False,
        specieswise=False,
        dtype=torch.float64,
        device="cpu",
        family_names=names,
        leaf_species_maps=leaf_maps,
    )
    from_split_files = GeneDataset(
        species_tree,
        [[str(tree_a), str(tree_b)]],
        genewise=False,
        specieswise=False,
        dtype=torch.float64,
        device="cpu",
        family_names=names,
        leaf_species_maps=leaf_maps,
    )

    ccp_dist = from_distribution.families[0]["ccp_helpers"]
    ccp_split = from_split_files.families[0]["ccp_helpers"]
    labels_dist = from_distribution.families[0]["clade_leaf_labels"]
    labels_split = from_split_files.families[0]["clade_leaf_labels"]
    assert int(ccp_dist["C"]) == int(ccp_split["C"])
    assert int(ccp_dist["N_splits"]) == int(ccp_split["N_splits"])
    assert int(ccp_dist["root_clade_id"]) == int(ccp_split["root_clade_id"])
    assert "clade_leaves" not in ccp_dist
    assert "clade_is_leaf" not in ccp_dist
    assert labels_dist == labels_split
    assert sorted(label for label in labels_dist if label) == ["a1", "b1", "c1"]
    for key in (
        "split_counts",
        "split_parents_sorted",
        "split_leftrights_sorted",
        "parents_sorted",
        "seg_counts",
        "ptr",
        "log_split_probs_sorted",
    ):
        torch.testing.assert_close(ccp_dist[key], ccp_split[key])
    torch.testing.assert_close(
        from_distribution.families[0]["leaf_col_index"],
        from_split_files.families[0]["leaf_col_index"],
    )

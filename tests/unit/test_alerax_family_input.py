from __future__ import annotations

import pytest
import torch

from gpurec.core.model import (
    GeneDataset,
    parse_alerax_family_file,
    parse_alerax_family_file_with_paths,
    parse_alerax_mapping_file,
)
from tests.unit.alerax_helpers import write_tiny_alerax_inputs


def _write(path, text: str):
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start": -1}, "start"),
        ({"start": 0.5}, "start"),
        ({"start": True}, "start"),
        ({"max_families": 0}, "max_families"),
        ({"max_families": 1.5}, "max_families"),
        ({"max_families": True}, "max_families"),
    ],
)
def test_alerax_family_selection_validates_before_io(tmp_path, kwargs, message):
    missing = tmp_path / "missing_families.txt"

    with pytest.raises(ValueError, match=message):
        parse_alerax_family_file(missing, **kwargs)


def _valid_species_payload() -> dict[str, object]:
    return {
        "S": 3,
        "names": ["Root", "A", "B"],
        "s_P_indexes": torch.tensor([0, 3], dtype=torch.long),
        "s_C12_indexes": torch.tensor([1, 2], dtype=torch.long),
        "unnorm_row_max": torch.zeros(3, dtype=torch.float64),
        "species_name_to_index": {"Root": 0, "A": 1, "B": 2},
    }


def _valid_family_payload() -> dict[str, object]:
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


class _BatchRecordingPreprocessExt:
    def __init__(self, species_payload):
        self.species_payload = species_payload
        self.calls: list[dict[str, object]] = []

    def preprocess_multiple_families(
        self,
        species_tree,
        gene_tree_paths,
        *,
        leaf_species_maps=None,
        include_details=False,
        include_species_matrices=False,
        include_debug_details=True,
        include_scheduler_details=True,
        include_legacy_ccp_details=True,
        num_threads=0,
    ):
        del (
            species_tree,
            include_details,
            include_species_matrices,
            include_debug_details,
            include_scheduler_details,
            include_legacy_ccp_details,
        )
        leaf_species_maps = leaf_species_maps or {}
        names = list(gene_tree_paths)
        self.calls.append(
            {
                "names": names,
                "leaf_species_maps": dict(leaf_species_maps),
                "num_threads": num_threads,
            }
        )
        return {
            "species": self.species_payload,
            "families": {name: _valid_family_payload() for name in names},
        }


@pytest.mark.parametrize("bad_value", [0, -1, True, 1.5, "2"])
def test_gene_dataset_rejects_invalid_preprocess_cpu_cores_before_extension_load(
    tmp_path,
    bad_value,
):
    with pytest.raises(ValueError, match="preprocess_cpu_cores"):
        GeneDataset(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            genewise=False,
            specieswise=False,
            dtype=torch.float64,
            device="cpu",
            preprocess_cpu_cores=bad_value,
        )


def test_uncached_family_preprocessing_is_batched_and_forwards_cpu_cores(tmp_path):
    species_tree = _write(tmp_path / "species.nwk", "(A:1,B:1)Root;\n")
    gene_trees = [
        _write(tmp_path / f"gene{i}.nwk", f"(a:1,b:{i + 1});\n")
        for i in range(5)
    ]
    family_names = [f"fam{i}" for i in range(5)]
    leaf_maps = [{}, {"a": "A"}, {}, {"b": "B"}, {}]
    events: list[tuple[str, dict[str, object]]] = []

    def progress(event: str, **fields: object) -> None:
        events.append((event, fields))

    ext = _BatchRecordingPreprocessExt(_valid_species_payload())
    species_helpers, raw_by_family = GeneDataset._preprocess_without_cache(
        ext,
        species_tree,
        [[str(path)] for path in gene_trees],
        family_names,
        leaf_maps,
        progress=progress,
        preprocess_cpu_cores=3,
        _batch_size=2,
    )

    assert species_helpers["S"] == 3
    assert list(raw_by_family) == family_names
    assert [call["names"] for call in ext.calls] == [
        ["fam0", "fam1"],
        ["fam2", "fam3"],
        ["fam4"],
    ]
    assert [call["num_threads"] for call in ext.calls] == [3, 3, 3]
    assert ext.calls[0]["leaf_species_maps"] == {"fam1": {"a": "A"}}
    assert ext.calls[1]["leaf_species_maps"] == {"fam3": {"b": "B"}}
    assert ext.calls[2]["leaf_species_maps"] == {}

    batch_starts = [
        fields
        for event, fields in events
        if event == "uncached_preprocess_batch_start"
    ]
    batch_dones = [
        fields
        for event, fields in events
        if event == "uncached_preprocess_batch_done"
    ]
    assert events[0] == (
        "uncached_preprocess_start",
        {
            "families": 5,
            "families_with_leaf_maps": 2,
            "batch_size": 2,
            "batches": 3,
            "preprocess_cpu_cores": 3,
        },
    )
    assert batch_starts == [
        {
            "batch_idx": 0,
            "batches": 3,
            "families": 2,
            "families_with_leaf_maps": 1,
            "first_family": "fam0",
            "last_family": "fam1",
        },
        {
            "batch_idx": 1,
            "batches": 3,
            "families": 2,
            "families_with_leaf_maps": 1,
            "first_family": "fam2",
            "last_family": "fam3",
        },
        {
            "batch_idx": 2,
            "batches": 3,
            "families": 1,
            "families_with_leaf_maps": 0,
            "first_family": "fam4",
            "last_family": "fam4",
        },
    ]
    assert [fields["families"] for fields in batch_dones] == [2, 2, 1]
    assert [fields["total_built"] for fields in batch_dones] == [2, 4, 5]
    assert events[-1] == ("uncached_preprocess_done", {"families": 5, "batches": 3})


def test_uncached_preprocessing_uses_rayon_default_when_cpu_cores_is_none(tmp_path):
    species_tree = _write(tmp_path / "species.nwk", "(A:1,B:1)Root;\n")
    gene_tree = _write(tmp_path / "gene.nwk", "(a:1,b:1);\n")

    ext = _BatchRecordingPreprocessExt(_valid_species_payload())
    GeneDataset._preprocess_without_cache(
        ext,
        species_tree,
        [[str(gene_tree)]],
        ["fam0"],
        [{}],
        preprocess_cpu_cores=None,
    )

    assert ext.calls == [
        {
            "names": ["fam0"],
            "leaf_species_maps": {},
            "num_threads": 0,
        }
    ]


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
    inputs = write_tiny_alerax_inputs(
        tmp_path,
        family_name="fam0",
        gene_tree_name="fam.nwk",
        mapping_name="fam.map",
        family_lines=("starting_gene_tree = fam.nwk", bad_line),
    )

    with pytest.raises(ValueError, match=message):
        parse_alerax_family_file(inputs.families_file)


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


def test_alerax_mapping_file_accepts_whitespace_comments_and_duplicate_species(tmp_path):
    mapping = _write(
        tmp_path / "fam.map",
        "\n".join(
            [
                "# comment line",
                " SpeciesA : gene1 ; gene2 ;  ",
                "",
                "SpeciesB:gene3",
                "SpeciesA:gene4",
                "   # trailing comment line",
                "",
            ]
        )
        + "\n",
    )

    parsed = parse_alerax_mapping_file(mapping)
    assert parsed == {
        "gene1": "SpeciesA",
        "gene2": "SpeciesA",
        "gene3": "SpeciesB",
        "gene4": "SpeciesA",
    }


def test_alerax_mapping_file_empty_or_comment_only_is_empty_mapping(tmp_path):
    mapping = _write(
        tmp_path / "empty.map",
        "\n".join(["", "   ", "# one", "   # two", ""]),
    )
    assert parse_alerax_mapping_file(mapping) == {}


def test_alerax_family_file_large_manifest_selection_is_stable(tmp_path):
    family_lines = ["[FAMILIES]"]
    expected_names: list[str] = []
    expected_tree_paths: list[list[str]] = []
    expected_mapping_paths: list[str] = []
    for idx in range(120):
        name = f"fam_{idx:04d}"
        gene_name = f"{name}.nwk"
        map_name = f"{name}.map"
        _write(tmp_path / gene_name, "(a:1,b:1);\n")
        _write(tmp_path / map_name, "A:a\nB:b\n")
        family_lines.extend(
            [
                f"- {name}",
                f"starting_gene_tree = {gene_name}",
                f"mapping = {map_name}",
            ]
        )
        if 33 <= idx < 40:
            expected_names.append(name)
            expected_tree_paths.append([str((tmp_path / gene_name).resolve())])
            expected_mapping_paths.append(str((tmp_path / map_name).resolve()))
    families = _write(tmp_path / "families.txt", "\n".join(family_lines) + "\n")

    names, tree_paths, mapping_paths = parse_alerax_family_file_with_paths(
        families,
        start=33,
        max_families=7,
    )
    assert names == expected_names
    assert tree_paths == expected_tree_paths
    assert mapping_paths == expected_mapping_paths

    names_full, tree_paths_full, maps_full = parse_alerax_family_file(
        families,
        start=33,
        max_families=7,
    )
    assert names_full == expected_names
    assert tree_paths_full == expected_tree_paths
    assert len(maps_full) == 7
    assert all(mapping == {"a": "A", "b": "B"} for mapping in maps_full)


def test_gene_dataset_accepts_documented_simple_newick_subset(tmp_path):
    species_tree = _write(
        tmp_path / "sp.nwk",
        " ( ( A :1e-3 , B :+2.0E-1 ) AB :-0.0 , C :.5 ) Root \n",
    )
    gene_tree = _write(
        tmp_path / "gene.nwk",
        " ( a :1e-2 , b :+3.0E+1 , c :-0.0 ) GeneRoot \n",
    )

    dataset = GeneDataset(
        species_tree,
        [[str(gene_tree)]],
        genewise=False,
        specieswise=False,
        dtype=torch.float64,
        device="cpu",
        family_names=["fam0"],
        leaf_species_maps=[{"a": "A", "b": "B", "c": "C"}],
        preprocess_cpu_cores=1,
    )

    family = dataset.families[0]
    labels = sorted(label for label in family["clade_leaf_labels"] if label)
    assert labels == ["a", "b", "c"]
    assert dataset.S == 5
    assert int(family["N_splits"]) > 0
    assert set(map(int, family["leaf_col_index"].tolist())) == {0, 1, 3}


@pytest.mark.parametrize(
    "branch_length",
    [
        "0",
        "1",
        ".5",
        "1.",
        "1e-3",
        "+2.0E-1",
        "-0.0",
    ],
)
def test_gene_dataset_accepts_supported_branch_length_variants(
    tmp_path: Path,
    branch_length: str,
):
    species_tree = _write(
        tmp_path / "sp.nwk",
        f"((A:{branch_length},B:{branch_length})AB:{branch_length},C:{branch_length})Root;",
    )
    gene_tree = _write(
        tmp_path / "gene.nwk",
        f"(a:{branch_length},b:{branch_length},c:{branch_length})GeneRoot;",
    )
    dataset = GeneDataset(
        species_tree,
        [[str(gene_tree)]],
        genewise=False,
        specieswise=False,
        dtype=torch.float64,
        device="cpu",
        family_names=["fam0"],
        leaf_species_maps=[{"a": "A", "b": "B", "c": "C"}],
    )
    assert dataset.S == 5
    assert int(dataset.families[0]["N_splits"]) > 0


@pytest.mark.parametrize(
    ("species_newick", "gene_newick", "message"),
    [
        (
            "(A:1,B:1)Root;(A:1,B:1)Other;",
            "(a:1,b:1)Gene;",
            "Unexpected trailing characters",
        ),
        (
            "(A:1,B:1,C:1)Root;",
            "(a:1,b:1)Gene;",
            "Species tree must be strictly binary",
        ),
        (
            "(A:1,B:1)Root;",
            "((a:1)Unary:1,b:1)Gene;",
            "Unary node in gene tree",
        ),
        (
            "(A:1[&rate=1],B:1)Root;",
            "(a:1,b:1)Gene;",
            "Expected ',' or '\\)'",
        ),
    ],
)
def test_gene_dataset_rejects_unsupported_newick_dialect_cases(
    tmp_path,
    species_newick: str,
    gene_newick: str,
    message: str,
):
    species_tree = _write(tmp_path / "sp.nwk", species_newick)
    gene_tree = _write(tmp_path / "gene.nwk", gene_newick)

    with pytest.raises(RuntimeError, match=message):
        GeneDataset(
            species_tree,
            [[str(gene_tree)]],
            genewise=False,
            specieswise=False,
            dtype=torch.float64,
            device="cpu",
            family_names=["fam0"],
            leaf_species_maps=[{"a": "A", "b": "B"}],
        )


def test_alerax_family_file_multi_tree_ccp_matches_split_files(tmp_path):
    species_tree = _write(tmp_path / "sp.nwk", "((A:1,B:1)AB:1,C:1)root;\n")
    tree_a = _write(tmp_path / "a.nwk", "((a1:1,b1:1):1,c1:1);\n")
    tree_b = _write(tmp_path / "b.nwk", "(a1:1,(b1:1,c1:1):1);\n")
    tree_dist_text = tree_a.read_text() + tree_b.read_text().strip()
    assert tree_dist_text.endswith(";")
    tree_dist = _write(
        tmp_path / "fam.trees",
        tree_dist_text[:-1] + "\n",
    )
    assert tree_dist.read_text().count(";") == 1
    assert not tree_dist.read_text().rstrip().endswith(";")
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
    assert "inclusion_children" not in ccp_dist
    assert "inclusion_parents" not in ccp_dist
    assert "ubiquitous_clade_id" not in ccp_dist
    assert "parents_sorted" not in ccp_dist
    assert "seg_parent_ids" not in ccp_dist
    assert "seg_counts" not in ccp_dist
    assert "ptr" not in ccp_dist
    assert "ptr_ge2" not in ccp_dist
    assert "split_order" not in ccp_dist
    assert labels_dist == labels_split
    assert sorted(label for label in labels_dist if label) == ["a1", "b1", "c1"]
    for key in (
        "split_counts",
        "split_parents_sorted",
        "split_leftrights_sorted",
        "log_split_probs_sorted",
    ):
        torch.testing.assert_close(ccp_dist[key], ccp_split[key])
    torch.testing.assert_close(
        from_distribution.families[0]["leaf_col_index"],
        from_split_files.families[0]["leaf_col_index"],
    )

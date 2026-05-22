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


def test_gene_dataset_rejects_nonbool_cache_refresh_before_extension_load(tmp_path):
    with pytest.raises(ValueError, match="refresh_preprocess_cache"):
        GeneDataset(
            tmp_path / "missing_species.nwk",
            [tmp_path / "missing_gene.nwk"],
            genewise=False,
            specieswise=False,
            dtype=torch.float64,
            device="cpu",
            preprocess_cache_dir=tmp_path / "cache",
            refresh_preprocess_cache="false",
        )


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


class _FakePreprocessExt:
    def __init__(self, species_payload, family_payload):
        self.species_payload = species_payload
        self.family_payload = family_payload
        self.calls = 0

    def preprocess_multiple_families(
        self,
        species_tree,
        gene_tree_paths,
        *,
        leaf_species_maps=None,
        include_details=False,
        include_species_matrices=False,
    ):
        self.calls += 1
        return {
            "species": self.species_payload,
            "families": {"fam0": self.family_payload},
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
    ):
        del species_tree, include_details, include_species_matrices
        leaf_species_maps = leaf_species_maps or {}
        names = list(gene_tree_paths)
        self.calls.append(
            {
                "names": names,
                "leaf_species_maps": dict(leaf_species_maps),
            }
        )
        return {
            "species": self.species_payload,
            "families": {
                name: _valid_family_cache_payload()
                for name in names
            },
        }


class _NoPreprocessExt:
    def preprocess_multiple_families(self, *args, **kwargs):
        raise AssertionError("cache hit should not invoke preprocessing")


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


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("s_P_indexes", [-1, 3]),
        ("s_P_indexes", [0, 6]),
        ("s_C12_indexes", [-1, 2]),
        ("s_C12_indexes", [1, 3]),
    ],
)
def test_species_preprocess_cache_rejects_topology_ids_outside_range(
    tmp_path,
    field: str,
    values: list[int],
):
    path = tmp_path / "species.pt"
    payload = _valid_species_cache_payload()
    payload[field] = torch.tensor(values, dtype=torch.long)
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match=f"{field}.*outside range"):
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


def test_cached_family_preprocess_rejects_leaf_species_indexes_outside_species_range(
    tmp_path,
):
    species_tree = _write(tmp_path / "species.nwk", "(A:1,B:1)Root;\n")
    gene_tree = _write(tmp_path / "gene.nwk", "(a:1,b:1);\n")
    cache_dir = tmp_path / "cache"

    ext = _FakePreprocessExt(
        _valid_species_cache_payload(),
        _valid_family_cache_payload(),
    )
    GeneDataset._preprocess_with_cache(
        ext,
        species_tree,
        [[str(gene_tree)]],
        ["fam0"],
        [{}],
        preprocess_cache_dir=cache_dir,
        refresh=False,
    )
    assert ext.calls == 1

    family_cache = next(cache_dir.glob("family-*.pt"))
    cached_family = torch.load(family_cache, map_location="cpu", weights_only=True)
    cached_family["leaf_col_index"] = torch.tensor([1, 3], dtype=torch.long)
    torch.save(cached_family, family_cache)

    with pytest.raises(RuntimeError, match="leaf_col_index.*outside range \\[0, 3\\)"):
        GeneDataset._preprocess_with_cache(
            _NoPreprocessExt(),
            species_tree,
            [[str(gene_tree)]],
            ["fam0"],
            [{}],
            preprocess_cache_dir=cache_dir,
            refresh=False,
        )


def test_preprocess_cache_progress_reports_miss_build_and_hit_load(tmp_path):
    species_tree = _write(tmp_path / "species.nwk", "(A:1,B:1)Root;\n")
    gene_tree = _write(tmp_path / "gene.nwk", "(a:1,b:1);\n")
    cache_dir = tmp_path / "cache"
    events: list[tuple[str, dict[str, object]]] = []

    def progress(event: str, **fields: object) -> None:
        events.append((event, fields))

    ext = _FakePreprocessExt(
        _valid_species_cache_payload(),
        _valid_family_cache_payload(),
    )
    GeneDataset._preprocess_with_cache(
        ext,
        species_tree,
        [[str(gene_tree)]],
        ["fam0"],
        [{}],
        preprocess_cache_dir=cache_dir,
        refresh=False,
        progress=progress,
    )

    first_pass = [event for event, _fields in events]
    assert "cache_start" in first_pass
    assert "species_cache_miss" in first_pass
    assert "family_cache_miss" in first_pass
    assert "missing_families_preprocess_start" in first_pass
    assert "missing_families_preprocess_done" in first_pass
    assert "species_cache_save_done" in first_pass
    assert "family_cache_save_done" in first_pass
    assert first_pass[-1] == "cache_done"
    assert ext.calls == 1

    events.clear()
    GeneDataset._preprocess_with_cache(
        _NoPreprocessExt(),
        species_tree,
        [[str(gene_tree)]],
        ["fam0"],
        [{}],
        preprocess_cache_dir=cache_dir,
        refresh=False,
        progress=progress,
    )

    second_pass = [event for event, _fields in events]
    assert "species_cache_load_start" in second_pass
    assert "species_cache_load_done" in second_pass
    assert "family_cache_load_start" in second_pass
    assert "family_cache_hit" in second_pass
    assert "missing_families_preprocess_start" not in second_pass
    assert second_pass[-1] == "cache_done"


def test_missing_family_preprocessing_is_batched_and_cached_incrementally(tmp_path):
    species_tree = _write(tmp_path / "species.nwk", "(A:1,B:1)Root;\n")
    gene_trees = [
        _write(tmp_path / f"gene{i}.nwk", f"(a:1,b:{i + 1});\n")
        for i in range(5)
    ]
    family_names = [f"fam{i}" for i in range(5)]
    leaf_maps = [{}, {"a": "A"}, {}, {"b": "B"}, {}]
    cache_dir = tmp_path / "cache"
    events: list[tuple[str, dict[str, object]]] = []

    def progress(event: str, **fields: object) -> None:
        events.append((event, fields))

    ext = _BatchRecordingPreprocessExt(_valid_species_cache_payload())
    species_helpers, raw_by_family = GeneDataset._preprocess_with_cache(
        ext,
        species_tree,
        [[str(path)] for path in gene_trees],
        family_names,
        leaf_maps,
        preprocess_cache_dir=cache_dir,
        refresh=False,
        progress=progress,
        _missing_family_batch_size=2,
    )

    assert species_helpers["S"] == 3
    assert list(raw_by_family) == family_names
    assert [call["names"] for call in ext.calls] == [
        ["fam0", "fam1"],
        ["fam2", "fam3"],
        ["fam4"],
    ]
    assert ext.calls[0]["leaf_species_maps"] == {"fam1": {"a": "A"}}
    assert ext.calls[1]["leaf_species_maps"] == {"fam3": {"b": "B"}}
    assert ext.calls[2]["leaf_species_maps"] == {}

    batch_starts = [
        fields
        for event, fields in events
        if event == "missing_families_preprocess_batch_start"
    ]
    batch_dones = [
        fields
        for event, fields in events
        if event == "missing_families_preprocess_batch_done"
    ]
    batch_cached = [
        fields
        for event, fields in events
        if event == "missing_families_preprocess_batch_cached"
    ]
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
    assert [fields["total_built"] for fields in batch_cached] == [2, 4, 5]
    assert len(list(cache_dir.glob("family-*.pt"))) == 5
    assert len(list(cache_dir.glob("species-*.pt"))) == 1
    assert events[-1] == ("cache_done", {"hits": 0, "misses": 5})


def test_uncached_family_preprocessing_is_batched_without_cache_writes(tmp_path):
    species_tree = _write(tmp_path / "species.nwk", "(A:1,B:1)Root;\n")
    gene_trees = [
        _write(tmp_path / f"gene{i}.nwk", f"(a:1,b:{i + 1});\n")
        for i in range(5)
    ]
    family_names = [f"fam{i}" for i in range(5)]
    leaf_maps = [{}, {"a": "A"}, {}, {"b": "B"}, {}]
    cache_dir = tmp_path / "cache"
    events: list[tuple[str, dict[str, object]]] = []

    def progress(event: str, **fields: object) -> None:
        events.append((event, fields))

    ext = _BatchRecordingPreprocessExt(_valid_species_cache_payload())
    species_helpers, raw_by_family = GeneDataset._preprocess_without_cache(
        ext,
        species_tree,
        [[str(path)] for path in gene_trees],
        family_names,
        leaf_maps,
        progress=progress,
        _batch_size=2,
    )

    assert species_helpers["S"] == 3
    assert list(raw_by_family) == family_names
    assert [call["names"] for call in ext.calls] == [
        ["fam0", "fam1"],
        ["fam2", "fam3"],
        ["fam4"],
    ]
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
    assert not cache_dir.exists()


def test_all_cached_family_preprocessing_uses_no_batches(tmp_path):
    species_tree = _write(tmp_path / "species.nwk", "(A:1,B:1)Root;\n")
    gene_trees = [
        _write(tmp_path / f"gene{i}.nwk", f"(a:1,b:{i + 1});\n")
        for i in range(3)
    ]
    family_names = [f"fam{i}" for i in range(3)]
    cache_dir = tmp_path / "cache"

    ext = _BatchRecordingPreprocessExt(_valid_species_cache_payload())
    GeneDataset._preprocess_with_cache(
        ext,
        species_tree,
        [[str(path)] for path in gene_trees],
        family_names,
        [{} for _ in family_names],
        preprocess_cache_dir=cache_dir,
        refresh=False,
        _missing_family_batch_size=2,
    )
    assert len(ext.calls) == 2

    events: list[str] = []
    GeneDataset._preprocess_with_cache(
        _NoPreprocessExt(),
        species_tree,
        [[str(path)] for path in gene_trees],
        family_names,
        [{} for _ in family_names],
        preprocess_cache_dir=cache_dir,
        refresh=False,
        progress=lambda event, **_fields: events.append(event),
        _missing_family_batch_size=2,
    )

    assert "family_cache_hit" in events
    assert "missing_families_preprocess_start" not in events
    assert "missing_families_preprocess_batch_start" not in events
    assert events[-1] == "cache_done"


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("split_parents_sorted", [-1]),
        ("split_parents_sorted", [3]),
        ("split_leftrights_sorted", [-1, 2]),
        ("split_leftrights_sorted", [1, 3]),
    ],
)
def test_family_preprocess_cache_rejects_split_clade_ids_outside_range(
    tmp_path,
    field: str,
    values: list[int],
):
    path = tmp_path / "family.pt"
    payload = _valid_family_cache_payload()
    payload["ccp"][field] = torch.tensor(values, dtype=torch.long)
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match=f"{field}.*outside range"):
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
    )

    family = dataset.families[0]
    labels = sorted(label for label in family["clade_leaf_labels"] if label)
    assert labels == ["a", "b", "c"]
    assert dataset.S == 5
    assert int(family["N_splits"]) > 0
    assert set(map(int, family["leaf_col_index"].tolist())) == {0, 1, 3}


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

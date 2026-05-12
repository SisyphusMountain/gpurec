from __future__ import annotations

import torch

from gpurec.core.model import GeneDataset, parse_alerax_family_file


def _write(path, text: str):
    path.write_text(text)
    return path


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
    assert int(ccp_dist["C"]) == int(ccp_split["C"])
    assert int(ccp_dist["N_splits"]) == int(ccp_split["N_splits"])
    assert int(ccp_dist["root_clade_id"]) == int(ccp_split["root_clade_id"])
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

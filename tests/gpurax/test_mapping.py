"""Tests for gpurax.io.prepare.prepare_family: relabeling gene-tree leaves
(and matching alignment headers) from the gene->species mapping file, so
real GeneRax-style datasets with non-species-prefixed gene names (e.g. `g1`)
work with gpurec (which derives species from the leaf-name prefix before the
first `_`).
"""
import pathlib

import dendropy
import pytest

from gpurax.driver import run
from gpurax.io.families import Family
from gpurax.io.prepare import prepare_family

FX = pathlib.Path(__file__).parent / "fixtures"
SIX = FX / "six"


def _leaf_labels(nwk_path):
    tree = dendropy.Tree.get(path=str(nwk_path), schema="newick", preserve_underscores=True)
    return {leaf.taxon.label for leaf in tree.leaf_node_iter()}


def _fasta_headers(path):
    headers = []
    for line in open(path):
        if line.startswith(">"):
            headers.append(line[1:].strip())
    return headers


def test_prepare_family_is_noop_on_already_prefixed_leaves(tmp_path):
    family = Family(
        name="fam1",
        starting_tree=str(SIX / "start_gene.nwk"),
        alignment=str(SIX / "aln.fasta"),
        mapping=str(SIX / "map.link"),
        model="GTR",
    )

    tree_path, aln_path = prepare_family(family, family.starting_tree, tmp_path)

    assert _leaf_labels(tree_path) == _leaf_labels(family.starting_tree)
    assert sorted(_fasta_headers(aln_path)) == sorted(_fasta_headers(family.alignment))


def test_prepare_family_no_mapping_returns_originals_unchanged(tmp_path):
    family = Family(
        name="fam1",
        starting_tree=str(SIX / "start_gene.nwk"),
        alignment=str(SIX / "aln.fasta"),
        mapping=None,
        model="GTR",
    )

    tree_path, aln_path = prepare_family(family, family.starting_tree, tmp_path)

    assert tree_path == family.starting_tree
    assert aln_path == family.alignment


def test_prepare_family_relabels_generic_gene_names(tmp_path):
    mapping_path = tmp_path / "map.link"
    mapping_path.write_text("g1 A\ng2 B\ng3 C\ng4 D\n")

    tree_path = tmp_path / "gene.nwk"
    tree_path.write_text("((g1:0.1,g2:0.1):0.1,(g3:0.1,g4:0.1):0.1);\n")

    aln_path = tmp_path / "aln.fasta"
    aln_path.write_text(">g1\nACGT\n>g2\nACGT\n>g3\nACGT\n>g4\nACGT\n")

    family = Family(
        name="fam1", starting_tree=str(tree_path), alignment=str(aln_path),
        mapping=str(mapping_path), model="GTR",
    )

    out_tree, out_aln = prepare_family(family, str(tree_path), tmp_path)

    assert _leaf_labels(out_tree) == {"A_g1", "B_g2", "C_g3", "D_g4"}
    assert sorted(_fasta_headers(out_aln)) == ["A_g1", "B_g2", "C_g3", "D_g4"]


def test_prepare_family_missing_gene_in_mapping_raises(tmp_path):
    mapping_path = tmp_path / "map.link"
    mapping_path.write_text("g1 A\n")  # g2 missing

    tree_path = tmp_path / "gene.nwk"
    tree_path.write_text("(g1:0.1,g2:0.1);\n")

    aln_path = tmp_path / "aln.fasta"
    aln_path.write_text(">g1\nACGT\n>g2\nACGT\n")

    family = Family(
        name="badfam", starting_tree=str(tree_path), alignment=str(aln_path),
        mapping=str(mapping_path), model="GTR",
    )

    with pytest.raises((KeyError, ValueError), match="badfam"):
        prepare_family(family, str(tree_path), tmp_path)


def test_prepare_family_species_with_underscore_raises(tmp_path):
    mapping_path = tmp_path / "map.link"
    mapping_path.write_text("g1 species_x\ng2 B\n")

    tree_path = tmp_path / "gene.nwk"
    tree_path.write_text("(g1:0.1,g2:0.1);\n")

    aln_path = tmp_path / "aln.fasta"
    aln_path.write_text(">g1\nACGT\n>g2\nACGT\n")

    family = Family(
        name="fam1", starting_tree=str(tree_path), alignment=str(aln_path),
        mapping=str(mapping_path), model="GTR",
    )

    with pytest.raises(ValueError, match="species_x"):
        prepare_family(family, str(tree_path), tmp_path)


def test_driver_handles_generic_gene_names_via_mapping(tmp_path):
    """Real-data-style leaf names (not species-prefixed) must work end-to-end
    once a mapping file is supplied, instead of crashing inside gpurec."""
    species_path = tmp_path / "species.nwk"
    species_path.write_text("((A,B),(C,D));\n")

    gene_path = tmp_path / "gene.nwk"
    gene_path.write_text("((g1:0.1,g2:0.1):0.1,(g3:0.1,g4:0.1):0.1);\n")

    aln_path = tmp_path / "aln.fasta"
    aln_path.write_text(
        ">g1\nACGTACGTACGTAAGT\n"
        ">g2\nACGTACGTACGTAAGT\n"
        ">g3\nACGTTCGTACGTTAGT\n"
        ">g4\nACGTTCGTACGTTAGT\n"
    )

    mapping_path = tmp_path / "map.link"
    mapping_path.write_text("g1 A\ng2 B\ng3 C\ng4 D\n")

    families_path = tmp_path / "families.txt"
    families_path.write_text(
        "[FAMILIES]\n"
        "- fam1\n"
        f"starting_gene_tree = {gene_path}\n"
        f"alignment = {aln_path}\n"
        f"mapping = {mapping_path}\n"
        "subst_model = GTR\n"
    )

    out_dir = tmp_path / "out"
    run(str(families_path), str(species_path), str(out_dir),
        max_spr_radius=2, rec_radius=1, device="cuda")

    assert (out_dir / "scores.tsv").exists()
    assert (out_dir / "fam1.reconciled.nwk").exists()

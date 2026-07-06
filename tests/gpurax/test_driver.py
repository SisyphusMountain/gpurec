import pathlib

import pytest

from gpurax.driver import run
from gpurax.io.families import Family

FX = pathlib.Path(__file__).parent / "fixtures"
SIX = FX / "six"


def _write_families_file(path):
    path.write_text(
        "[FAMILIES]\n"
        "- fam1\n"
        f"starting_gene_tree = {SIX / 'start_gene.nwk'}\n"
        f"alignment = {SIX / 'aln.fasta'}\n"
        f"mapping = {SIX / 'map.link'}\n"
        "subst_model = GTR\n"
    )


def test_end_to_end_small(tmp_path):
    families_file = tmp_path / "families.txt"
    _write_families_file(families_file)

    run(str(families_file), str(SIX / "species.nwk"), str(tmp_path),
        max_spr_radius=2, rec_radius=1, device="cuda")

    assert (tmp_path / "scores.tsv").exists()
    assert (tmp_path / "fam1.reconciled.nwk").exists()
    assert (tmp_path / "fam1.recphyloxml.xml").exists()
    assert (tmp_path / "fam1.notung.nhx").exists()

    tsv = (tmp_path / "scores.tsv").read_text().splitlines()
    assert tsv[0].split("\t") == ["name", "joint", "rec", "seq"]
    assert tsv[1].split("\t")[0] == "fam1"


def test_malformed_starting_tree_raises_clean_error(tmp_path):
    bad_tree = tmp_path / "bad.nwk"
    bad_tree.write_text("(((A,B),C);\n")  # unbalanced parens

    families_file = tmp_path / "families.txt"
    families_file.write_text(
        "[FAMILIES]\n"
        "- badfam\n"
        f"starting_gene_tree = {bad_tree}\n"
        f"alignment = {SIX / 'aln.fasta'}\n"
        f"mapping = {SIX / 'map.link'}\n"
        "subst_model = GTR\n"
    )

    with pytest.raises(ValueError, match="malformed gene tree"):
        run(str(families_file), str(SIX / "species.nwk"), str(tmp_path / "out"),
            max_spr_radius=1, rec_radius=0, device="cuda")

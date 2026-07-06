import pathlib

from gpurax.io.families import Family
from gpurax.io.initial_trees import ensure_starting_tree

FX = pathlib.Path(__file__).parent / "fixtures"


def test_builds_when_absent(tmp_path):
    fam = Family("f", None, str(FX / "aln.fasta"), str(FX / "map.link"), "GTR")
    tree = ensure_starting_tree(fam, str(tmp_path))
    txt = pathlib.Path(tree).read_text()
    assert txt.count("(") >= 1 and ";" in txt


def test_uses_provided_when_present(tmp_path):
    fam = Family("f", str(FX / "gene.nwk"), str(FX / "aln.fasta"), str(FX / "map.link"), "GTR")
    assert ensure_starting_tree(fam, str(tmp_path)) == str(FX / "gene.nwk")

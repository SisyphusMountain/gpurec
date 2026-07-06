import pathlib
from gpurax.io.families import parse_families, parse_mapping

FX = pathlib.Path(__file__).parent / "fixtures"


def test_parse_families():
    fams = parse_families(str(FX / "families.txt"))
    assert len(fams) == 1
    f = fams[0]
    assert f.name == "fam1" and f.model == "GTR"
    assert f.alignment.endswith("aln.fasta")


def test_parse_mapping():
    m = parse_mapping(str(FX / "map.link"))
    assert m == {"A_a": "A", "B_b": "B", "C_c": "C", "D_d": "D"}

import json
import pathlib
import re

from gpurax.driver import run

FX = pathlib.Path(__file__).parent / "fixtures" / "six"


def _family_name(families_txt: pathlib.Path) -> str:
    for line in families_txt.read_text().splitlines():
        line = line.strip()
        m = re.match(r"^-\s*(\S+)", line)
        if m:
            return m.group(1)
    raise ValueError(f"no family name found in {families_txt}")


def _rf(nwk_a, nwk_b):
    import dendropy

    tns = dendropy.TaxonNamespace()
    a = dendropy.Tree.get(data=nwk_a, schema="newick", taxon_namespace=tns)
    b = dendropy.Tree.get(data=nwk_b, schema="newick", taxon_namespace=tns)
    return dendropy.calculate.treecompare.symmetric_difference(a, b)


def test_rf_parity_vs_generax(tmp_path):
    generax_ref = json.loads((FX / "generax_ref.json").read_text())
    if not generax_ref.get("available", True):
        import pytest

        pytest.xfail("no GeneRax binary")

    fam_name = _family_name(FX / "families.txt")

    run(
        str(FX / "families.txt"),
        str(FX / "species.nwk"),
        str(tmp_path),
        max_spr_radius=3,
        rec_radius=1,
        device="cuda",
    )

    ours = (tmp_path / f"{fam_name}.reconciled.nwk").read_text()

    # RF/tree-quality parity (NOT joint-loglk equality): our reconciled tree
    # should agree closely with GeneRax's on this 6-taxon fixture.
    rf_ours_vs_generax = _rf(ours, generax_ref["reconciled_newick"])
    assert rf_ours_vs_generax <= 2

    # sanity: scores.tsv well-formed with the joint/rec/seq columns
    rows = (tmp_path / "scores.tsv").read_text().splitlines()
    assert rows[0].split("\t") == ["name", "joint", "rec", "seq"]

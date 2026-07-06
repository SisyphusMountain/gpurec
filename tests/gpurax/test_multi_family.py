"""End-to-end multi-family driver test: validates the tool's headline
capability -- processing MANY gene families at once through a single
`gpurax.driver.run` call (co-batched reconciliation search, jointly
optimized global DTL rates, per-family export).

The three families below all reference the same existing 6-taxon fixture
inputs (`tests/gpurax/fixtures/six/`). The driver optimizes DTL rates
GLOBALLY over all families jointly, so reusing identical inputs across
`fam1`/`fam2`/`fam3` is fine here: this test validates multi-family
ORCHESTRATION and per-family export, not that families differ. (Cross-family
batching correctness with differentiated inputs is covered elsewhere at the
search level, e.g. `tests/gpurax/test_spr_multi.py`.)
"""
import pathlib

import dendropy

from gpurax.driver import run

SIX = pathlib.Path(__file__).parent / "fixtures" / "six"
NAMES = ("fam1", "fam2", "fam3")


def _write_families_file(path):
    lines = ["[FAMILIES]"]
    for name in NAMES:
        lines.append(f"- {name}")
        lines.append(f"starting_gene_tree = {SIX / 'start_gene.nwk'}")
        lines.append(f"alignment = {SIX / 'aln.fasta'}")
        lines.append(f"mapping = {SIX / 'map.link'}")
        lines.append("subst_model = GTR")
    path.write_text("\n".join(lines) + "\n")


def test_multi_family_end_to_end(tmp_path):
    families_file = tmp_path / "families.txt"
    _write_families_file(families_file)

    out_dir = tmp_path / "out"
    run(str(families_file), str(SIX / "species.nwk"), str(out_dir),
        max_spr_radius=2, rec_radius=1, device="cuda")

    scores_path = out_dir / "scores.tsv"
    assert scores_path.exists()
    rows = scores_path.read_text().splitlines()
    assert rows[0].split("\t") == ["name", "joint", "rec", "seq"]
    assert len(rows) == 1 + len(NAMES), f"expected header + {len(NAMES)} rows, got {rows}"

    data_names = {r.split("\t")[0] for r in rows[1:]}
    assert data_names == set(NAMES)

    for name in NAMES:
        nwk_path = out_dir / f"{name}.reconciled.nwk"
        xml_path = out_dir / f"{name}.recphyloxml.xml"
        nhx_path = out_dir / f"{name}.notung.nhx"

        assert nwk_path.exists(), f"missing reconciled newick for {name}"
        assert xml_path.exists(), f"missing recphyloxml for {name}"
        assert nhx_path.exists(), f"missing notung nhx for {name}"

        newick_text = nwk_path.read_text().strip()
        assert newick_text
        tree = dendropy.Tree.get(data=newick_text, schema="newick")
        assert len(tree.leaf_nodes()) > 0

        assert xml_path.read_text().strip()
        assert nhx_path.read_text().strip()

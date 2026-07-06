from gpurax.reconcile.export import write_newick, write_scores

def test_writes_newick_and_scores(tmp_path):
    write_newick(str(tmp_path), "fam1", "((a,b),(c,d));")
    write_scores(str(tmp_path), [{"name": "fam1", "joint": -10.0, "rec": -3.0, "seq": -7.0}])
    assert (tmp_path / "fam1.reconciled.nwk").read_text().strip() == "((a,b),(c,d));"
    tsv = (tmp_path / "scores.tsv").read_text().splitlines()
    assert tsv[0].split("\t") == ["name", "joint", "rec", "seq"]
    assert tsv[1].split("\t")[0] == "fam1"

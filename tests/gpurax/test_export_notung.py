from gpurax.reconcile.export import write_notung


def test_notung_has_nhx_and_events(tmp_path):
    nodes = [("duplication", 4, [1, 2]), ("leaf", 0, [-1, -1]), ("leaf", 1, [-1, -1])]
    out = write_notung(str(tmp_path), "fam1", "((a,b),c);", nodes)
    txt = open(out).read()
    assert "[&&NHX" in txt and txt.strip().endswith(";")


def test_notung_marks_duplication_and_transfer(tmp_path):
    # node 0: transfer from species 2 -> species 5, donor lineage lost (child0=loss,
    # child1=leaf continuing in recipient species 5); node 1: loss at species 2;
    # node 2: leaf at species 5.
    nodes = [
        ("transfer", 2, [1, 2]),
        ("loss", 2, [-1, -1]),
        ("leaf", 5, [-1, -1]),
    ]
    out = write_notung(str(tmp_path), "fam2", "(a,b);", nodes)
    txt = open(out).read()
    assert "D=Y" not in txt  # no duplication in this scenario
    assert "H=Y" in txt  # transfer node present
    assert "@species_2@species_5" in txt  # donor@recipient species tag
    assert txt.strip().endswith(";")


def test_notung_uses_species_names_when_provided(tmp_path):
    nodes = [("duplication", 0, [1, 2]), ("leaf", 0, [-1, -1]), ("leaf", 0, [-1, -1])]
    out = write_notung(
        str(tmp_path), "fam3", "(a,b);", nodes, species_newick="A;",
    )
    txt = open(out).read()
    assert "S=A" in txt
    assert "D=Y" in txt

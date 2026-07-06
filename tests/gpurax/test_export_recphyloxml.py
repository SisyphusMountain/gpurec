import xml.dom.minidom as md

from gpurax.reconcile.export import write_recphyloxml


def test_recphyloxml_is_valid(tmp_path):
    nodes = [("speciation", 4, [1, 2]), ("leaf", 0, [-1, -1]), ("leaf", 1, [-1, -1])]
    out = write_recphyloxml(str(tmp_path), "fam1", "((a,b),c);", nodes, "((A,B),C);")
    doc = md.parse(out)  # raises if malformed
    tags = {n.tagName for n in doc.getElementsByTagName("*")}
    assert "recPhylo" in tags and "eventsRec" in tags


def test_recphyloxml_event_tags_and_species_names(tmp_path):
    # species tree "((A,B),C);" postorder indices: A=0, B=1, (A,B)=2, C=3, root=4
    nodes = [
        ("speciation", 4, [1, 2]),   # root
        ("duplication", 0, [3, 4]),  # under A
        ("leaf", 3, [-1, -1]),
        ("leaf", 0, [-1, -1]),
        ("transfer", 0, [5, 6]),
        ("loss", 0, [-1, -1]),
        ("leaf", 1, [-1, -1]),
    ]
    out = write_recphyloxml(str(tmp_path), "fam2", "(((a,b),c),d);", nodes, "((A,B),C);")
    doc = md.parse(out)
    tags = {n.tagName for n in doc.getElementsByTagName("*")}
    for expected in ("recPhylo", "spTree", "recGeneTree", "eventsRec",
                     "speciation", "duplication", "leaf", "transferBack", "loss"):
        assert expected in tags, expected

    leaf_locations = {
        el.getAttribute("speciesLocation") for el in doc.getElementsByTagName("leaf")
    }
    assert leaf_locations == {"A", "B", "C"}

    transfer_back = doc.getElementsByTagName("transferBack")[0]
    assert transfer_back.getAttribute("destinationSpecies") == "B"

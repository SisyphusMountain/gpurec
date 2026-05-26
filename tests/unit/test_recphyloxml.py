from __future__ import annotations

from gpurec.backtracking import recphyloxml_event_counts
from gpurec.recphyloxml import (
    EVENT_KEYS,
    direct_children,
    events_for_clade,
    iter_rec_gene_tree_clades,
    local_name,
    primary_event_name,
)
from gpurec.workflow.sampling import _xml_species_and_transfer_counts


def test_recphyloxml_helpers_handle_namespaced_documents():
    xml = """
    <rp:recPhylo xmlns:rp="urn:test">
      <rp:recGeneTree>
        <rp:phylogeny>
          <rp:clade>
            <rp:eventsRec><rp:speciation speciesLocation="Root"/></rp:eventsRec>
            <rp:clade><rp:eventsRec><rp:loss speciesLocation="Lost"/></rp:eventsRec></rp:clade>
          </rp:clade>
        </rp:phylogeny>
      </rp:recGeneTree>
      <rp:recGeneTree>
        <rp:phylogeny>
          <rp:clade>
            <rp:eventsRec><rp:duplication speciesLocation="Copy"/></rp:eventsRec>
            <rp:clade><rp:eventsRec><rp:leaf speciesLocation="Copy"/></rp:eventsRec></rp:clade>
          </rp:clade>
        </rp:phylogeny>
      </rp:recGeneTree>
    </rp:recPhylo>
    """

    roots = list(iter_rec_gene_tree_clades(xml))

    assert [primary_event_name(root) for root in roots] == ["speciation", "duplication"]
    assert local_name(roots[0].tag) == "clade"
    assert [primary_event_name(child) for child in direct_children(roots[0], "clade")] == [
        "loss"
    ]
    assert [local_name(event.tag) for event in events_for_clade(roots[1])] == [
        "duplication"
    ]


def test_backtracking_and_sampling_use_shared_recphyloxml_traversal():
    xml = """
    <recPhylo>
      <recGeneTree>
        <phylogeny>
          <clade>
            <eventsRec><speciation speciesLocation="Root"/></eventsRec>
            <clade>
              <eventsRec>
                <branchingOut speciesLocation="Donor"/>
                <transferBack destinationSpecies="Recipient"/>
              </eventsRec>
              <clade><eventsRec><leaf speciesLocation="Recipient"/></eventsRec></clade>
            </clade>
            <clade><eventsRec><loss speciesLocation="Lost"/></eventsRec></clade>
          </clade>
        </phylogeny>
      </recGeneTree>
    </recPhylo>
    """

    event_counts = recphyloxml_event_counts(xml)
    species_counts, transfers = _xml_species_and_transfer_counts(xml)

    assert tuple(event_counts) == EVENT_KEYS
    assert event_counts["SL"] == 1
    assert event_counts["T"] == 1
    assert event_counts["Leaf"] == 1
    assert species_counts["Root"]["origination"] == 1.0
    assert species_counts["Donor"]["transfers"] == 1.0
    assert species_counts["Recipient"]["transfers_to"] == 1.0
    assert transfers[("Donor", "Recipient")] == 1.0


def test_recphyloxml_traversal_uses_first_root_clade_per_gene_tree():
    xml = """
    <recPhylo>
      <recGeneTree>
        <phylogeny>
          <clade>
            <eventsRec><speciation speciesLocation="RootA"/></eventsRec>
            <clade><eventsRec><leaf speciesLocation="LeafA"/></eventsRec></clade>
          </clade>
          <clade>
            <eventsRec><duplication speciesLocation="IgnoredRoot"/></eventsRec>
            <clade><eventsRec><leaf speciesLocation="IgnoredLeaf"/></eventsRec></clade>
          </clade>
        </phylogeny>
      </recGeneTree>
      <recGeneTree>
        <phylogeny>
          <clade>
            <eventsRec><duplication speciesLocation="RootB"/></eventsRec>
            <clade><eventsRec><leaf speciesLocation="LeafB"/></eventsRec></clade>
          </clade>
        </phylogeny>
      </recGeneTree>
    </recPhylo>
    """

    roots = list(iter_rec_gene_tree_clades(xml))
    event_counts = recphyloxml_event_counts(xml, alerax_style=False)

    assert [primary_event_name(root) for root in roots] == [
        "speciation",
        "duplication",
    ]
    assert event_counts == {
        "S": 1,
        "SL": 0,
        "D": 1,
        "DL": 0,
        "T": 0,
        "TL": 0,
        "L": 0,
        "Leaf": 2,
    }


def test_recphyloxml_species_counts_use_one_file_level_origination():
    xml = """
    <recPhylo>
      <recGeneTree>
        <phylogeny>
          <clade>
            <eventsRec><speciation speciesLocation="FirstRoot"/></eventsRec>
            <clade><eventsRec><leaf speciesLocation="FirstLeaf"/></eventsRec></clade>
          </clade>
        </phylogeny>
      </recGeneTree>
      <recGeneTree>
        <phylogeny>
          <clade>
            <eventsRec><speciation speciesLocation="SecondRoot"/></eventsRec>
            <clade><eventsRec><leaf speciesLocation="SecondLeaf"/></eventsRec></clade>
          </clade>
        </phylogeny>
      </recGeneTree>
    </recPhylo>
    """

    species_counts, transfers = _xml_species_and_transfer_counts(xml)

    assert species_counts["FirstRoot"]["origination"] == 1.0
    assert species_counts["SecondRoot"]["origination"] == 0.0
    assert species_counts["FirstRoot"]["speciations"] == 1.0
    assert species_counts["SecondRoot"]["speciations"] == 1.0
    assert species_counts["FirstLeaf"]["copies"] == 1.0
    assert species_counts["SecondLeaf"]["copies"] == 1.0
    assert transfers == {}

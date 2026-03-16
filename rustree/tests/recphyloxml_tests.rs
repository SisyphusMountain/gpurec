use rustree::{RecTree, Event};

/// Test parsing a simple RecPhyloXML with minimal structure
#[test]
fn test_parse_simple_recphyloxml() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<recPhylo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.recg.org ./recGeneTreeXML.xsd"
    xmlns="http://www.recg.org">
<spTree>
<phylogeny>
<clade>
    <name>Root</name>
    <branchLength>0.0</branchLength>
    <clade>
        <name>A</name>
        <branchLength>1.0</branchLength>
    </clade>
    <clade>
        <name>B</name>
        <branchLength>1.0</branchLength>
    </clade>
</clade>
</phylogeny>
</spTree>
<recGeneTree>
<phylogeny rooted="true">
<clade>
    <name>NULL</name>
    <branchLength>0.0</branchLength>
    <eventsRec>
        <speciation speciesLocation="Root"/>
    </eventsRec>
    <clade>
        <name>gene_A</name>
        <branchLength>1.0</branchLength>
        <eventsRec>
            <leaf speciesLocation="A"/>
        </eventsRec>
    </clade>
    <clade>
        <name>gene_B</name>
        <branchLength>1.0</branchLength>
        <eventsRec>
            <leaf speciesLocation="B"/>
        </eventsRec>
    </clade>
</clade>
</phylogeny>
</recGeneTree>
</recPhylo>"#;

    let result = RecTree::from_xml(xml);
    assert!(result.is_ok(), "Failed to parse simple XML: {:?}", result.err());

    let rec_tree = result.unwrap();

    // Check species tree
    assert_eq!(rec_tree.species_tree.nodes.len(), 3, "Species tree should have 3 nodes");
    assert_eq!(rec_tree.species_tree.nodes[rec_tree.species_tree.root].name, "Root");

    // Check gene tree
    assert_eq!(rec_tree.gene_tree.nodes.len(), 3, "Gene tree should have 3 nodes");

    // Check mappings
    assert_eq!(rec_tree.node_mapping.len(), 3);
    assert_eq!(rec_tree.event_mapping.len(), 3);

    // Check event types
    assert_eq!(rec_tree.event_mapping[rec_tree.gene_tree.root], Event::Speciation);
}

/// Test parsing with missing branch lengths (should default to 0.0)
#[test]
fn test_parse_missing_branch_lengths() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<recPhylo xmlns="http://www.recg.org">
<spTree>
<phylogeny>
<clade>
    <name>Root</name>
    <clade>
        <name>A</name>
    </clade>
    <clade>
        <name>B</name>
    </clade>
</clade>
</phylogeny>
</spTree>
<recGeneTree>
<phylogeny rooted="true">
<clade>
    <name>NULL</name>
    <eventsRec>
        <duplication speciesLocation="Root"/>
    </eventsRec>
    <clade>
        <name>gene_A</name>
        <eventsRec>
            <leaf speciesLocation="A"/>
        </eventsRec>
    </clade>
    <clade>
        <name>gene_B</name>
        <eventsRec>
            <leaf speciesLocation="B"/>
        </eventsRec>
    </clade>
</clade>
</phylogeny>
</recGeneTree>
</recPhylo>"#;

    let result = RecTree::from_xml(xml);
    assert!(result.is_ok(), "Failed to parse XML without branch lengths");

    let rec_tree = result.unwrap();

    // All branch lengths should be 0.0 by default
    for node in &rec_tree.species_tree.nodes {
        assert_eq!(node.length, 0.0);
    }
}

/// Test parsing all event types
#[test]
fn test_parse_all_event_types() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<recPhylo xmlns="http://www.recg.org">
<spTree>
<phylogeny>
<clade>
    <name>Root</name>
    <clade>
        <name>A</name>
    </clade>
    <clade>
        <name>B</name>
    </clade>
</clade>
</phylogeny>
</spTree>
<recGeneTree>
<phylogeny rooted="true">
<clade>
    <name>NULL</name>
    <eventsRec>
        <duplication speciesLocation="Root"/>
    </eventsRec>
    <clade>
        <name>NULL</name>
        <eventsRec>
            <speciation speciesLocation="Root"/>
        </eventsRec>
        <clade>
            <name>loss</name>
            <eventsRec>
                <loss speciesLocation="A"/>
            </eventsRec>
        </clade>
        <clade>
            <name>gene_B1</name>
            <eventsRec>
                <leaf speciesLocation="B"/>
            </eventsRec>
        </clade>
    </clade>
    <clade>
        <name>NULL</name>
        <eventsRec>
            <branchingOut speciesLocation="Root"/>
        </eventsRec>
        <clade>
            <name>gene_A</name>
            <eventsRec>
                <leaf speciesLocation="A"/>
            </eventsRec>
        </clade>
        <clade>
            <name>gene_B2</name>
            <eventsRec>
                <transferBack destinationSpecies="B"/>
                <leaf speciesLocation="B"/>
            </eventsRec>
        </clade>
    </clade>
</clade>
</phylogeny>
</recGeneTree>
</recPhylo>"#;

    let result = RecTree::from_xml(xml);
    assert!(result.is_ok(), "Failed to parse XML with all event types");

    let rec_tree = result.unwrap();

    // Check that we have different event types
    let has_duplication = rec_tree.event_mapping.iter().any(|e| *e == Event::Duplication);
    let has_speciation = rec_tree.event_mapping.iter().any(|e| *e == Event::Speciation);
    let has_loss = rec_tree.event_mapping.iter().any(|e| *e == Event::Loss);
    let has_leaf = rec_tree.event_mapping.iter().any(|e| *e == Event::Leaf);
    let has_transfer = rec_tree.event_mapping.iter().any(|e| *e == Event::Transfer);

    assert!(has_duplication, "Should have duplication event");
    assert!(has_speciation, "Should have speciation event");
    assert!(has_loss, "Should have loss event");
    assert!(has_leaf, "Should have leaf event");
    assert!(has_transfer, "Should have transfer event");
}

/// Test that to_xml() produces valid XML that can be parsed back
#[test]
fn test_round_trip() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<recPhylo xmlns="http://www.recg.org">
<spTree>
<phylogeny>
<clade>
    <name>Root</name>
    <branchLength>0.0</branchLength>
    <clade>
        <name>A</name>
        <branchLength>1.0</branchLength>
    </clade>
    <clade>
        <name>B</name>
        <branchLength>1.0</branchLength>
    </clade>
</clade>
</phylogeny>
</spTree>
<recGeneTree>
<phylogeny rooted="true">
<clade>
    <name>NULL</name>
    <branchLength>0.0</branchLength>
    <eventsRec>
        <duplication speciesLocation="Root"/>
    </eventsRec>
    <clade>
        <name>gene_A</name>
        <branchLength>1.0</branchLength>
        <eventsRec>
            <leaf speciesLocation="A"/>
        </eventsRec>
    </clade>
    <clade>
        <name>gene_B</name>
        <branchLength>1.0</branchLength>
        <eventsRec>
            <leaf speciesLocation="B"/>
        </eventsRec>
    </clade>
</clade>
</phylogeny>
</recGeneTree>
</recPhylo>"#;

    let rec_tree1 = RecTree::from_xml(xml).expect("Failed to parse first time");
    let xml2 = rec_tree1.to_xml();
    let rec_tree2 = RecTree::from_xml(&xml2).expect("Failed to parse second time");

    // Check that structure is preserved
    assert_eq!(rec_tree1.species_tree.nodes.len(), rec_tree2.species_tree.nodes.len());
    assert_eq!(rec_tree1.gene_tree.nodes.len(), rec_tree2.gene_tree.nodes.len());
    assert_eq!(rec_tree1.node_mapping.len(), rec_tree2.node_mapping.len());
    assert_eq!(rec_tree1.event_mapping.len(), rec_tree2.event_mapping.len());
}

/// Test RecTree methods
#[test]
fn test_rectree_owned_methods() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<recPhylo xmlns="http://www.recg.org">
<spTree>
<phylogeny>
<clade>
    <name>Root</name>
    <clade>
        <name>A</name>
    </clade>
    <clade>
        <name>B</name>
    </clade>
</clade>
</phylogeny>
</spTree>
<recGeneTree>
<phylogeny rooted="true">
<clade>
    <name>NULL</name>
    <eventsRec>
        <speciation speciesLocation="Root"/>
    </eventsRec>
    <clade>
        <name>gene_A</name>
        <eventsRec>
            <leaf speciesLocation="A"/>
        </eventsRec>
    </clade>
    <clade>
        <name>gene_B</name>
        <eventsRec>
            <leaf speciesLocation="B"/>
        </eventsRec>
    </clade>
</clade>
</phylogeny>
</recGeneTree>
</recPhylo>"#;

    let rec_tree = RecTree::from_xml(xml).expect("Failed to parse XML");

    // Test species_node_for
    let root_species_idx = rec_tree.species_node_for(rec_tree.gene_tree.root);
    assert_eq!(rec_tree.species_tree.nodes[root_species_idx.unwrap()].name, "Root");

    // Test event_for
    let root_event = rec_tree.event_for(rec_tree.gene_tree.root);
    assert_eq!(*root_event, Event::Speciation);

    // Test get_full_info
    let (gene_node, species_idx, event) = rec_tree.get_full_info(rec_tree.gene_tree.root);
    assert_eq!(gene_node.name, "NULL");
    assert_eq!(rec_tree.species_tree.nodes[species_idx.unwrap()].name, "Root");
    assert_eq!(*event, Event::Speciation);

}

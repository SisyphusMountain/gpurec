use rustree::{RecTree, Event, parse_newick};

/// Test parsing gene-tree-only XML with <P> event tags
#[test]
fn test_parse_gene_tree_only_with_p_tags() {
    // Create a simple species tree
    let species_newick = "((A:1.0,B:1.0)AB:1.0,C:2.0)Root:0.0;";
    let mut species_nodes = parse_newick(species_newick).expect("Failed to parse species tree");
    let species_root = species_nodes.pop().unwrap();
    let species_tree = species_root.to_flat_tree();

    // Gene tree XML with only recGeneTree section and <P> tags
    let gene_xml = r#"<recGeneTree>
<phylogeny rooted="true">
<clade>
    <name>Root_1</name>
    <eventsRec>
        <speciation speciesLocation="Root" ts="0"></speciation>
    </eventsRec>
    <clade>
        <name>A_gene</name>
        <eventsRec>
            <P speciesLocation="A" ts="1"></P>
        </eventsRec>
    </clade>
    <clade>
        <name>B_gene</name>
        <eventsRec>
            <P speciesLocation="B" ts="1"></P>
        </eventsRec>
    </clade>
</clade>
</phylogeny>
</recGeneTree>"#;

    let result = RecTree::from_gene_tree_xml(gene_xml, species_tree);
    assert!(result.is_ok(), "Failed to parse gene-tree-only XML: {:?}", result.err());

    let rec_tree = result.unwrap();

    // Check gene tree structure
    assert_eq!(rec_tree.gene_tree.nodes.len(), 3, "Should have 3 gene nodes");

    // Check that <P> tags were parsed as Event::Leaf
    let leaf_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Leaf).count();
    assert_eq!(leaf_count, 2, "Should have 2 leaf events from <P> tags");

    let speciation_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Speciation).count();
    assert_eq!(speciation_count, 1, "Should have 1 speciation event");
}

/// Test from_separate_files with simple example
#[test]
fn test_from_separate_files_simple() {
    use std::fs;

    // Create temporary files
    let temp_dir = std::env::temp_dir();
    let species_path = temp_dir.join("test_species.nwk");
    let gene_path = temp_dir.join("test_gene.xml");

    let species_newick = "((A:1.0,B:1.0)AB:1.0,C:2.0)Root:0.0;";
    let gene_xml = r#"<recGeneTree>
<phylogeny rooted="true">
<clade>
    <name>Root_1</name>
    <eventsRec>
        <duplication speciesLocation="Root"></duplication>
    </eventsRec>
    <clade>
        <name>gene_A</name>
        <eventsRec>
            <P speciesLocation="A"></P>
        </eventsRec>
    </clade>
    <clade>
        <name>gene_C</name>
        <eventsRec>
            <P speciesLocation="C"></P>
        </eventsRec>
    </clade>
</clade>
</phylogeny>
</recGeneTree>"#;

    fs::write(&species_path, species_newick).expect("Failed to write species file");
    fs::write(&gene_path, gene_xml).expect("Failed to write gene file");

    // Test from_separate_files
    let result = RecTree::from_separate_files(
        species_path.to_str().unwrap(),
        gene_path.to_str().unwrap()
    );

    // Clean up
    let _ = fs::remove_file(&species_path);
    let _ = fs::remove_file(&gene_path);

    assert!(result.is_ok(), "Failed to parse from separate files: {:?}", result.err());

    let rec_tree = result.unwrap();
    assert_eq!(rec_tree.species_tree.nodes.len(), 5, "Species tree should have 5 nodes");
    assert_eq!(rec_tree.gene_tree.nodes.len(), 3, "Gene tree should have 3 nodes");

    // Check events
    let duplication_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Duplication).count();
    assert_eq!(duplication_count, 1, "Should have 1 duplication");

    let leaf_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Leaf).count();
    assert_eq!(leaf_count, 2, "Should have 2 leaves");
}

/// Test parsing with missing speciesLocation attribute
#[test]
fn test_gene_tree_only_with_ts_attribute() {
    // Create species tree
    let species_newick = "(A:1.0,B:1.0)Root:0.0;";
    let mut species_nodes = parse_newick(species_newick).expect("Failed to parse");
    let species_root = species_nodes.pop().unwrap();
    let species_tree = species_root.to_flat_tree();

    // Gene tree with ts attributes (should be ignored)
    let gene_xml = r#"<recGeneTree>
<phylogeny rooted="true">
<clade>
    <name>Root_1</name>
    <eventsRec>
        <speciation speciesLocation="Root" ts="0"></speciation>
    </eventsRec>
    <clade>
        <name>A_gene</name>
        <eventsRec>
            <P speciesLocation="A" ts="1234"></P>
        </eventsRec>
    </clade>
    <clade>
        <name>B_gene</name>
        <eventsRec>
            <P speciesLocation="B" ts="5678"></P>
        </eventsRec>
    </clade>
</clade>
</phylogeny>
</recGeneTree>"#;

    let result = RecTree::from_gene_tree_xml(gene_xml, species_tree);
    assert!(result.is_ok(), "Should ignore ts attribute");

    let rec_tree = result.unwrap();
    assert_eq!(rec_tree.gene_tree.nodes.len(), 3);
}

/// Test that gene-tree-only parsing validates species names
#[test]
fn test_gene_tree_only_missing_species() {
    // Create species tree with limited species
    let species_newick = "(A:1.0,B:1.0)Root:0.0;";
    let mut species_nodes = parse_newick(species_newick).expect("Failed to parse");
    let species_root = species_nodes.pop().unwrap();
    let species_tree = species_root.to_flat_tree();

    // Gene tree referencing non-existent species "C"
    let gene_xml = r#"<recGeneTree>
<phylogeny rooted="true">
<clade>
    <name>Root_1</name>
    <eventsRec>
        <speciation speciesLocation="Root"></speciation>
    </eventsRec>
    <clade>
        <name>A_gene</name>
        <eventsRec>
            <P speciesLocation="A"></P>
        </eventsRec>
    </clade>
    <clade>
        <name>C_gene</name>
        <eventsRec>
            <P speciesLocation="C"></P>
        </eventsRec>
    </clade>
</clade>
</phylogeny>
</recGeneTree>"#;

    let result = RecTree::from_gene_tree_xml(gene_xml, species_tree);
    assert!(result.is_err(), "Should fail when species 'C' not found");
    assert!(result.unwrap_err().to_string().contains("Species 'C' not found"));
}

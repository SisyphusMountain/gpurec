use rustree::{RecTree, Event};

#[test]
#[ignore] // Requires external WP2 data — set RUSTREE_WP2_DATA to the data dir
fn test_parse_alerax_file() {
    let data_dir = std::env::var("RUSTREE_WP2_DATA").expect("Set RUSTREE_WP2_DATA to WP2 data dir");
    let filepath = format!("{}/test_data/test_2/alerax_g.xml", data_dir);

    let result = RecTree::from_xml_file(&filepath);
    assert!(result.is_ok(), "Failed to parse ALERax file: {:?}", result.err());

    let rec_tree = result.unwrap();

    println!("\n=== ALERax File Parsing Results ===");
    println!("Species Tree:");
    println!("  - Number of nodes: {}", rec_tree.species_tree.nodes.len());
    println!("  - Root node: {}", rec_tree.species_tree.nodes[rec_tree.species_tree.root].name);

    println!("\nGene Tree:");
    println!("  - Number of nodes: {}", rec_tree.gene_tree.nodes.len());
    println!("  - Root node: {}", rec_tree.gene_tree.nodes[rec_tree.gene_tree.root].name);

    // Count events
    let speciation_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Speciation).count();
    let duplication_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Duplication).count();
    let transfer_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Transfer).count();
    let loss_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Loss).count();
    let leaf_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Leaf).count();

    println!("\nEvent Counts:");
    println!("  - Speciation: {}", speciation_count);
    println!("  - Duplication: {}", duplication_count);
    println!("  - Transfer: {}", transfer_count);
    println!("  - Loss: {}", loss_count);
    println!("  - Leaf: {}", leaf_count);

    // Verify structure
    assert!(rec_tree.species_tree.nodes.len() > 0, "Species tree should have nodes");
    assert!(rec_tree.gene_tree.nodes.len() > 0, "Gene tree should have nodes");
    assert_eq!(rec_tree.gene_tree.nodes.len(), rec_tree.node_mapping.len());
    assert_eq!(rec_tree.gene_tree.nodes.len(), rec_tree.event_mapping.len());

    // Verify we have various event types
    assert!(speciation_count > 0, "Should have speciation events");
    assert!(duplication_count > 0, "Should have duplication events");
    assert!(leaf_count > 0, "Should have leaf events");

    // List some leaf names
    println!("\nSample Gene Tree Leaves:");
    let leaf_nodes: Vec<_> = rec_tree.gene_tree.nodes.iter()
        .enumerate()
        .filter(|(idx, _)| rec_tree.event_mapping[*idx] == Event::Leaf)
        .take(10)
        .collect();

    for (idx, node) in leaf_nodes {
        let species_idx = rec_tree.node_mapping[idx].unwrap();
        let species_name = &rec_tree.species_tree.nodes[species_idx].name;
        println!("  - {} (in species: {})", node.name, species_name);
    }

    // Test XML export and round-trip
    println!("\nTesting XML export and round-trip...");
    let xml_output = rec_tree.to_xml();
    println!("  - Generated XML length: {} bytes", xml_output.len());

    let rec_tree2 = RecTree::from_xml(&xml_output).expect("Failed to parse exported XML");
    println!("  ✓ Successfully parsed exported XML!");
    println!("  - Species nodes: {}", rec_tree2.species_tree.nodes.len());
    println!("  - Gene nodes: {}", rec_tree2.gene_tree.nodes.len());

    // Verify structure preserved
    assert_eq!(rec_tree.species_tree.nodes.len(), rec_tree2.species_tree.nodes.len());
    assert_eq!(rec_tree.gene_tree.nodes.len(), rec_tree2.gene_tree.nodes.len());
}

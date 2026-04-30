use rustree::{RecTree, Event};

#[test]
#[ignore] // Requires external WP2 data — set RUSTREE_WP2_DATA to the data dir
fn test_parse_real_separate_files() {
    let data_dir = std::env::var("RUSTREE_WP2_DATA").expect("Set RUSTREE_WP2_DATA to WP2 data dir");
    let species_path = format!("{}/output_1/T/CompleteTree.nwk", data_dir);
    let gene_path = format!("{}/output_1/G/Gene_trees/1_rec.xml", data_dir);

    println!("\n=== Parsing Real Separate Files ===");
    println!("Species tree: {}", species_path);
    println!("Gene tree: {}", gene_path);

    let result = RecTree::from_separate_files(&species_path, &gene_path);

    assert!(
        result.is_ok(),
        "Failed to parse real separate files: {:?}",
        result.err()
    );

    let rec_tree = result.unwrap();

    println!("\n=== Results ===");
    println!("Species tree nodes: {}", rec_tree.species_tree.nodes.len());
    println!("Gene tree nodes: {}", rec_tree.gene_tree.nodes.len());

    // Count events
    let speciation_count = rec_tree
        .event_mapping
        .iter()
        .filter(|e| **e == Event::Speciation)
        .count();
    let duplication_count = rec_tree
        .event_mapping
        .iter()
        .filter(|e| **e == Event::Duplication)
        .count();
    let transfer_count = rec_tree
        .event_mapping
        .iter()
        .filter(|e| **e == Event::Transfer)
        .count();
    let loss_count = rec_tree
        .event_mapping
        .iter()
        .filter(|e| **e == Event::Loss)
        .count();
    let leaf_count = rec_tree
        .event_mapping
        .iter()
        .filter(|e| **e == Event::Leaf)
        .count();

    println!("\n=== Event Counts ===");
    println!("  Speciation: {}", speciation_count);
    println!("  Duplication: {}", duplication_count);
    println!("  Transfer: {}", transfer_count);
    println!("  Loss: {}", loss_count);
    println!("  Leaf: {}", leaf_count);
    println!("  Total: {}", rec_tree.event_mapping.len());

    // Verify structure
    assert!(
        rec_tree.species_tree.nodes.len() > 0,
        "Species tree should have nodes"
    );
    assert!(
        rec_tree.gene_tree.nodes.len() > 0,
        "Gene tree should have nodes"
    );
    assert_eq!(
        rec_tree.gene_tree.nodes.len(),
        rec_tree.node_mapping.len(),
        "Mapping length should match gene tree size"
    );
    assert_eq!(
        rec_tree.gene_tree.nodes.len(),
        rec_tree.event_mapping.len(),
        "Event mapping length should match gene tree size"
    );

    // Verify we have various event types
    assert!(leaf_count > 0, "Should have leaf events (from <P> tags)");

    // Display some sample leaves
    println!("\n=== Sample Gene Tree Leaves (from <P> tags) ===");
    let leaf_nodes: Vec<_> = rec_tree
        .gene_tree
        .nodes
        .iter()
        .enumerate()
        .filter(|(idx, _)| rec_tree.event_mapping[*idx] == Event::Leaf)
        .take(10)
        .collect();

    for (idx, node) in leaf_nodes {
        let species_idx = rec_tree.node_mapping[idx].unwrap();
        let species_name = &rec_tree.species_tree.nodes[species_idx].name;
        println!("  {} → {}", node.name, species_name);
    }

    println!("\n✓ Successfully parsed real separate files!");
}

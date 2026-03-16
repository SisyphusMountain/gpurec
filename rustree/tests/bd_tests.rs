// Tests for birth-death tree simulation

use rustree::bd::{simulate_bd_tree_bwd, save_events_to_csv, BDEvent};
use rustree::node::TraversalOrder;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs;

#[test]
fn test_bd_tree_basic() {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 10;
    let lambda = 1.0;
    let mu = 0.5;

    let (tree, events) = simulate_bd_tree_bwd(n, lambda, mu, &mut rng);

    // Count leaf nodes (nodes with no children)
    let leaf_count = tree.nodes.iter()
        .filter(|node| node.left_child.is_none() && node.right_child.is_none())
        .count();

    println!("Tree has {} nodes total, {} leaves", tree.nodes.len(), leaf_count);
    println!("Root index: {}", tree.root);
    println!("Number of events: {}", events.len());

    // Export tree to Newick format and save
    let newick = tree.to_newick();
    let newick_with_semicolon = format!("{};", newick.expect("Failed to convert to Newick"));
    fs::write("bd_tree_basic.nwk", &newick_with_semicolon)
        .expect("Failed to write tree to file");
    println!("Tree saved to bd_tree_basic.nwk");
    println!("Newick: {}", newick_with_semicolon);

    // Save events to CSV
    save_events_to_csv(&events, &tree, "bd_tree_basic_events.csv")
        .expect("Failed to write events to file");
    println!("Events saved to bd_tree_basic_events.csv");

    // The tree should have some nodes
    assert!(tree.nodes.len() > 0);
    assert!(events.len() > 0);

    // Cleanup test files
    let _ = std::fs::remove_file("bd_tree_basic.nwk");
    let _ = std::fs::remove_file("bd_tree_basic_events.csv");
}

#[test]
fn test_bd_tree_traversal() {
    let mut rng = StdRng::seed_from_u64(123);
    let n = 5;
    let lambda = 1.0;
    let mu = 0.3;

    let (tree, _events) = simulate_bd_tree_bwd(n, lambda, mu, &mut rng);

    // Test that we can traverse the tree
    let mut node_count = 0;
    for _node in tree.iter(TraversalOrder::PostOrder) {
        node_count += 1;
    }

    assert_eq!(node_count, tree.nodes.len());
}

#[test]
#[should_panic(expected = "Speciation rate must be strictly greater than extinction rate")]
fn test_bd_tree_invalid_rates() {
    let mut rng = StdRng::seed_from_u64(999);
    let n = 10;
    let lambda = 0.5;
    let mu = 1.0; // mu >= lambda should panic

    simulate_bd_tree_bwd(n, lambda, mu, &mut rng);
}

#[test]
fn test_bd_tree_pure_birth() {
    let mut rng = StdRng::seed_from_u64(777);
    let n = 8;
    let lambda = 1.0;
    let mu = 0.0; // Pure birth (Yule) process

    let (tree, events) = simulate_bd_tree_bwd(n, lambda, mu, &mut rng);

    // In a pure birth process, there should be no extinct lineages
    // So we should have exactly 2n-1 nodes (n leaves + n-1 internal nodes)
    let leaf_count = tree.nodes.iter()
        .filter(|node| node.left_child.is_none() && node.right_child.is_none())
        .count();

    let internal_count = tree.nodes.iter()
        .filter(|node| node.left_child.is_some() || node.right_child.is_some())
        .count();

    println!("Pure birth tree: {} leaves, {} internal nodes", leaf_count, internal_count);
    println!("Number of events: {}", events.len());

    // Export tree to Newick format and save
    let newick = tree.to_newick();
    let newick_with_semicolon = format!("{};", newick.expect("Failed to convert to Newick"));
    fs::write("bd_tree_pure_birth.nwk", &newick_with_semicolon)
        .expect("Failed to write tree to file");
    println!("Pure birth tree saved to bd_tree_pure_birth.nwk");
    println!("Newick: {}", newick_with_semicolon);

    // Save events to CSV
    save_events_to_csv(&events, &tree, "bd_tree_pure_birth_events.csv")
        .expect("Failed to write events to file");
    println!("Events saved to bd_tree_pure_birth_events.csv");

    // For pure birth, all leaves should be extant (not extinct)
    // We should have n leaves numbered 0 to n-1
    let extant_leaves = tree.nodes.iter()
        .filter(|node| node.left_child.is_none() && node.right_child.is_none())
        .filter(|node| {
            if let Ok(id) = node.name.parse::<usize>() {
                id < n
            } else {
                false
            }
        })
        .count();

    assert_eq!(extant_leaves, n);
    // In pure birth, we should have only leaf and speciation events
    assert!(events.iter().all(|e| e.event_type == BDEvent::Leaf || e.event_type == BDEvent::Speciation));

    // Cleanup test files
    let _ = std::fs::remove_file("bd_tree_pure_birth.nwk");
    let _ = std::fs::remove_file("bd_tree_pure_birth_events.csv");
}

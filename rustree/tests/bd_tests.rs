// Tests for birth-death tree simulation

use rustree::bd::{simulate_bd_tree_bwd, save_events_to_csv, BDEvent};
use rustree::node::TraversalOrder;
use rustree::parse_newick;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs;

#[test]
fn test_bd_tree_basic() {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 10;
    let lambda = 1.0;
    let mu = 0.5;

    let (tree, events) = simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap();

    // Structural assertions: tree should have nodes and valid root
    assert!(!tree.nodes.is_empty());
    assert!(tree.root < tree.nodes.len());

    // Count leaf and internal nodes
    let leaf_count = tree.nodes.iter()
        .filter(|node| node.left_child.is_none() && node.right_child.is_none())
        .count();
    let internal_count = tree.nodes.len() - leaf_count;

    // Binary tree: internal nodes have exactly 2 children, leaves have 0
    for node in &tree.nodes {
        let has_left = node.left_child.is_some();
        let has_right = node.right_child.is_some();
        assert!(
            (has_left && has_right) || (!has_left && !has_right),
            "Node '{}' has unary branching (left={}, right={})",
            node.name, has_left, has_right
        );
    }

    // Every internal node has exactly 2 children, so nodes = 2 * internal + 1
    // (for a full binary tree: leaves = internal + 1)
    assert_eq!(leaf_count, internal_count + 1, "Binary tree must have leaves = internal + 1");

    // Parent-child consistency: every child's parent points back
    for (i, node) in tree.nodes.iter().enumerate() {
        if let Some(left) = node.left_child {
            assert_eq!(tree.nodes[left].parent, Some(i), "Left child's parent mismatch at node {}", i);
        }
        if let Some(right) = node.right_child {
            assert_eq!(tree.nodes[right].parent, Some(i), "Right child's parent mismatch at node {}", i);
        }
    }

    // Root has no parent
    assert!(tree.nodes[tree.root].parent.is_none(), "Root should have no parent");

    // Events should be non-empty
    assert!(!events.is_empty());

    // Export and cleanup
    let newick = tree.to_newick().expect("Failed to convert to Newick");
    let newick_with_semicolon = format!("{};", newick);
    fs::write("bd_tree_basic.nwk", &newick_with_semicolon).expect("Failed to write tree");
    save_events_to_csv(&events, &tree, "bd_tree_basic_events.csv").expect("Failed to write events");
    let _ = std::fs::remove_file("bd_tree_basic.nwk");
    let _ = std::fs::remove_file("bd_tree_basic_events.csv");
}

#[test]
fn test_bd_tree_traversal() {
    let mut rng = StdRng::seed_from_u64(123);
    let n = 5;
    let lambda = 1.0;
    let mu = 0.3;

    let (tree, _events) = simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap();

    // Test that we can traverse the tree
    let mut node_count = 0;
    for _node in tree.iter(TraversalOrder::PostOrder) {
        node_count += 1;
    }

    assert_eq!(node_count, tree.nodes.len());
}

#[test]
fn test_bd_tree_invalid_rates() {
    let mut rng = StdRng::seed_from_u64(999);
    let result = simulate_bd_tree_bwd(10, 0.5, 1.0, &mut rng);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("strictly greater than extinction rate"));
}

#[test]
fn test_bd_tree_pure_birth() {
    let mut rng = StdRng::seed_from_u64(777);
    let n = 8;
    let lambda = 1.0;
    let mu = 0.0; // Pure birth (Yule) process

    let (tree, events) = simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap();

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

#[test]
fn test_bd_newick_roundtrip() {
    let mut rng = StdRng::seed_from_u64(42);
    let (tree, _) = simulate_bd_tree_bwd(10, 1.0, 0.5, &mut rng).unwrap();

    // Export to Newick
    let newick = tree.to_newick().expect("Failed to convert to Newick");
    let newick_str = format!("{};", newick);

    // Re-parse the Newick string
    let parsed_nodes = parse_newick(&newick_str).expect("Failed to re-parse Newick");
    assert_eq!(parsed_nodes.len(), 1, "Should parse exactly one tree");

    // Convert back to FlatTree and compare structure
    let reparsed_tree = parsed_nodes[0].to_flat_tree();
    assert_eq!(reparsed_tree.nodes.len(), tree.nodes.len(),
        "Roundtrip should preserve node count");

    // Count leaves in both
    let original_leaves = tree.nodes.iter()
        .filter(|n| n.left_child.is_none() && n.right_child.is_none())
        .count();
    let reparsed_leaves = reparsed_tree.nodes.iter()
        .filter(|n| n.left_child.is_none() && n.right_child.is_none())
        .count();
    assert_eq!(original_leaves, reparsed_leaves, "Roundtrip should preserve leaf count");
}

#[test]
fn test_bd_tree_error_zero_species() {
    let mut rng = StdRng::seed_from_u64(42);
    let result = simulate_bd_tree_bwd(0, 1.0, 0.5, &mut rng);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("positive"));
}

#[test]
fn test_bd_tree_error_negative_lambda() {
    let mut rng = StdRng::seed_from_u64(42);
    let result = simulate_bd_tree_bwd(10, -1.0, 0.5, &mut rng);
    assert!(result.is_err());
}

#[test]
fn test_bd_tree_error_nan_rate() {
    let mut rng = StdRng::seed_from_u64(42);
    let result = simulate_bd_tree_bwd(10, f64::NAN, 0.5, &mut rng);
    assert!(result.is_err());
}

#[test]
fn test_bd_tree_error_infinite_rate() {
    let mut rng = StdRng::seed_from_u64(42);
    let result = simulate_bd_tree_bwd(10, f64::INFINITY, 0.5, &mut rng);
    assert!(result.is_err());
}

#[test]
fn test_bd_tree_single_species() {
    let mut rng = StdRng::seed_from_u64(42);
    let (tree, _events) = simulate_bd_tree_bwd(1, 1.0, 0.0, &mut rng).unwrap();
    // Single species: just one leaf node
    assert_eq!(tree.nodes.len(), 1);
    assert!(tree.nodes[0].left_child.is_none());
    assert!(tree.nodes[0].right_child.is_none());
}

#[test]
fn test_bd_tree_two_species() {
    let mut rng = StdRng::seed_from_u64(42);
    let (tree, _events) = simulate_bd_tree_bwd(2, 1.0, 0.0, &mut rng).unwrap();
    // Two species: root + 2 leaves = 3 nodes
    assert_eq!(tree.nodes.len(), 3);
}

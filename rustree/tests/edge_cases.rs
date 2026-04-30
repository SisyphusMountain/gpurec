// Edge case tests for various tree operations (#78)

use rustree::node::{FlatTree, FlatNode, TraversalOrder};
use rustree::parse_newick;
use rustree::sampling::{extract_induced_subtree, extract_induced_subtree_by_names};
use std::collections::HashSet;

// ============================================================================
// Single-node trees
// ============================================================================

#[test]
fn test_single_node_tree() {
    let tree = FlatTree {
        nodes: vec![FlatNode {
            name: "root".to_string(),
            left_child: None,
            right_child: None,
            parent: None,
            depth: None,
            length: 0.0,
            bd_event: None,
        }],
        root: 0,
    };
    assert_eq!(tree.len(), 1);
    assert!(!tree.is_empty());
    assert_eq!(tree.get_leaves().len(), 1);
}

#[test]
fn test_single_node_to_newick() {
    let tree = FlatTree {
        nodes: vec![FlatNode {
            name: "A".to_string(),
            left_child: None,
            right_child: None,
            parent: None,
            depth: None,
            length: 0.0,
            bd_event: None,
        }],
        root: 0,
    };
    let newick = tree.to_newick().expect("Single node should produce valid Newick");
    assert_eq!(newick, "A:0.000000");
}

#[test]
fn test_single_node_traversal() {
    let tree = FlatTree {
        nodes: vec![FlatNode {
            name: "only".to_string(),
            left_child: None,
            right_child: None,
            parent: None,
            depth: None,
            length: 0.0,
            bd_event: None,
        }],
        root: 0,
    };
    let count = tree.iter(TraversalOrder::PreOrder).count();
    assert_eq!(count, 1);
}

#[test]
fn test_single_node_assign_depths() {
    let mut tree = FlatTree {
        nodes: vec![FlatNode {
            name: "root".to_string(),
            left_child: None,
            right_child: None,
            parent: None,
            depth: None,
            length: 1.5,
            bd_event: None,
        }],
        root: 0,
    };
    tree.assign_depths();
    assert_eq!(tree.nodes[0].depth, Some(1.5));
}

// ============================================================================
// Sampling edge cases
// ============================================================================

#[test]
fn test_extract_empty_keep_set() {
    let nodes = parse_newick("(A:1.0,B:1.0):0.0;").unwrap();
    let tree = nodes[0].to_flat_tree();
    let empty: HashSet<usize> = HashSet::new();
    let result = extract_induced_subtree(&tree, &empty);
    assert!(result.is_none(), "Empty keep set should return None");
}

#[test]
fn test_extract_single_leaf_from_pair() {
    let nodes = parse_newick("(A:1.0,B:1.0):0.0;").unwrap();
    let tree = nodes[0].to_flat_tree();

    // Find leaf "A"
    let a_idx = tree.nodes.iter().position(|n| n.name == "A").unwrap();
    let mut keep = HashSet::new();
    keep.insert(a_idx);

    let result = extract_induced_subtree(&tree, &keep);
    assert!(result.is_some(), "Should extract a single-leaf subtree");
    let (subtree, _) = result.unwrap();
    assert_eq!(subtree.nodes.len(), 1);
    assert_eq!(subtree.nodes[0].name, "A");
}

#[test]
fn test_extract_by_nonexistent_name() {
    let nodes = parse_newick("(A:1.0,B:1.0):0.0;").unwrap();
    let tree = nodes[0].to_flat_tree();
    let names = vec!["NonExistent".to_string()];
    let result = extract_induced_subtree_by_names(&tree, &names);
    assert!(result.is_none(), "Non-existent name should return None");
}

// ============================================================================
// Newick parsing edge cases
// ============================================================================

#[test]
fn test_parse_single_leaf_newick() {
    let result = parse_newick("A:1.0;");
    assert!(result.is_ok());
    let nodes = result.unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].name, "A");
    assert_eq!(nodes[0].length, 1.0);
}

#[test]
fn test_parse_deeply_nested_newick() {
    // 10-level deep caterpillar tree
    let newick = "((((((((((A:1,B:1):1,C:2):1,D:3):1,E:4):1,F:5):1,G:6):1,H:7):1,I:8):1,J:9):1,K:10):0;";
    let result = parse_newick(newick);
    assert!(result.is_ok());
    let nodes = result.unwrap();
    assert_eq!(nodes.len(), 1);
    // Should have 11 leaves
    let tree = nodes[0].to_flat_tree();
    let leaf_count = tree.get_leaves().len();
    assert_eq!(leaf_count, 11);
}

#[test]
fn test_parse_malformed_newick_missing_semicolon() {
    let result = parse_newick("(A:1.0,B:2.0):0.0");
    assert!(result.is_err(), "Missing semicolon should fail");
}

#[test]
fn test_parse_malformed_newick_unmatched_paren() {
    let result = parse_newick("(A:1.0,B:2.0:0.0;");
    assert!(result.is_err(), "Unmatched paren should fail");
}

#[test]
fn test_parse_ternary_tree_rejected() {
    let result = parse_newick("(A:1,B:1,C:1):0;");
    assert!(result.is_err(), "Ternary node should be rejected");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Non-binary") || err.contains("3 children"),
        "Error should mention non-binary: {}", err);
}

// ============================================================================
// Conversion edge cases
// ============================================================================

#[test]
fn test_node_flat_tree_roundtrip() {
    let nodes = parse_newick("((A:1.0,B:2.0):0.5,(C:1.5,D:2.5):0.3):0.0;").unwrap();
    let original = &nodes[0];

    // Node → FlatTree → Node
    let flat = original.to_flat_tree();
    let reconstructed = flat.to_node();

    // Verify structure preserved
    let flat2 = reconstructed.to_flat_tree();
    assert_eq!(flat.nodes.len(), flat2.nodes.len());
    assert_eq!(flat.get_leaves().len(), flat2.get_leaves().len());
}

// ============================================================================
// Depth assignment edge cases
// ============================================================================

#[test]
fn test_assign_depths_twice_idempotent() {
    let nodes = parse_newick("(A:1.0,B:2.0):0.0;").unwrap();
    let mut tree = nodes[0].to_flat_tree();

    tree.assign_depths();
    let depths_first: Vec<_> = tree.nodes.iter().map(|n| n.depth).collect();

    tree.assign_depths();
    let depths_second: Vec<_> = tree.nodes.iter().map(|n| n.depth).collect();

    assert_eq!(depths_first, depths_second, "assign_depths should be idempotent");
}

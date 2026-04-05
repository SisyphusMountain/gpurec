// API parity contract tests for Python and R bindings.
//
// Both Python and R bindings delegate core validation/utility logic to
// `bindings_common`. These tests verify that shared contract by testing
// the underlying functions directly. Any regression here means both
// bindings break simultaneously.
//
// The parity checklist (features exposed in both bindings):
//   [x] simulate_species_tree  — both via simulate_bd_tree_bwd
//   [x] parse_newick           — both via newick::parse_newick
//   [x] DTL simulation (per-gene, per-species, batch, streaming)
//   [x] to_newick / to_xml     — both via FlatTree/RecTree methods
//   [x] pairwise_distances     — both via metric_functions
//   [x] sample_extant          — both via extract_extant_gene_tree
//   [x] sample_by_names        — both via extract_induced_subtree_by_names
//   [x] parse_recphyloxml      — both via RecTree::from_xml_file
//   [x] induced_transfers      — both via induced_transfers module
//   [x] name_internal_nodes    — both via FlatTree::name_internal_nodes
//   [x] DTL rate validation    — both via validate_dtl_rates
//   [x] Distance type parsing  — both via parse_distance_type
//   [x] Replacement transfer   — both via validate_replacement_transfer

use rustree::bindings_common::{
    validate_dtl_rates, validate_replacement_transfer, parse_distance_type,
    extract_extant_gene_tree, is_leaf, digit_width, init_rng,
};
use rustree::metric_functions::DistanceType;

// ============================================================================
// DTL rate validation (shared by Python + R)
// ============================================================================

#[test]
fn test_validate_dtl_rates_valid() {
    assert!(validate_dtl_rates(0.5, 0.2, 0.3).is_ok());
    assert!(validate_dtl_rates(0.0, 0.0, 0.0).is_ok());
}

#[test]
fn test_validate_dtl_rates_negative() {
    assert!(validate_dtl_rates(-0.1, 0.2, 0.3).is_err());
    assert!(validate_dtl_rates(0.5, -0.1, 0.3).is_err());
    assert!(validate_dtl_rates(0.5, 0.2, -0.1).is_err());
}

// ============================================================================
// Replacement transfer validation (shared by Python + R)
// ============================================================================

#[test]
fn test_validate_replacement_transfer_valid() {
    assert!(validate_replacement_transfer(None).is_ok());
    assert!(validate_replacement_transfer(Some(0.0)).is_ok());
    assert!(validate_replacement_transfer(Some(0.5)).is_ok());
    assert!(validate_replacement_transfer(Some(1.0)).is_ok());
}

#[test]
fn test_validate_replacement_transfer_invalid() {
    assert!(validate_replacement_transfer(Some(-0.1)).is_err());
    assert!(validate_replacement_transfer(Some(1.1)).is_err());
}

// ============================================================================
// Distance type parsing (shared by Python + R)
// ============================================================================

#[test]
fn test_parse_distance_type_aliases() {
    // Both bindings accept the same string aliases
    assert_eq!(parse_distance_type("topological").unwrap(), DistanceType::Topological);
    assert_eq!(parse_distance_type("topo").unwrap(), DistanceType::Topological);
    assert_eq!(parse_distance_type("metric").unwrap(), DistanceType::Metric);
    assert_eq!(parse_distance_type("branch").unwrap(), DistanceType::Metric);
    assert_eq!(parse_distance_type("patristic").unwrap(), DistanceType::Metric);
    // Case insensitive
    assert_eq!(parse_distance_type("Topological").unwrap(), DistanceType::Topological);
    assert_eq!(parse_distance_type("METRIC").unwrap(), DistanceType::Metric);
}

#[test]
fn test_parse_distance_type_invalid() {
    assert!(parse_distance_type("unknown").is_err());
    assert!(parse_distance_type("").is_err());
}

// ============================================================================
// Extant gene tree extraction (shared by Python + R)
// ============================================================================

#[test]
fn test_extract_extant_gene_tree() {
    use rustree::parse_newick;
    use rustree::node::{FlatTree, FlatNode, RecTree, Event};
    use std::sync::Arc;

    // Build a simple species tree: (A:1,B:1):0
    let sp_nodes = parse_newick("(A:1.0,B:1.0):0.0;").unwrap();
    let sp_tree = Arc::new(sp_nodes[0].to_flat_tree());

    // Build a gene tree with 3 leaves: one leaf, one leaf, one loss
    // Gene tree: (g1:1, (g2:0.5, loss:0.5):0.5):0
    let mut gene_tree = FlatTree {
        nodes: vec![
            FlatNode { name: "root".into(), left_child: Some(1), right_child: Some(2),
                       parent: None, depth: None, length: 0.0, bd_event: None },
            FlatNode { name: "g1".into(), left_child: None, right_child: None,
                       parent: Some(0), depth: None, length: 1.0, bd_event: None },
            FlatNode { name: "int".into(), left_child: Some(3), right_child: Some(4),
                       parent: Some(0), depth: None, length: 0.5, bd_event: None },
            FlatNode { name: "g2".into(), left_child: None, right_child: None,
                       parent: Some(2), depth: None, length: 0.5, bd_event: None },
            FlatNode { name: "loss".into(), left_child: None, right_child: None,
                       parent: Some(2), depth: None, length: 0.5, bd_event: None },
        ],
        root: 0,
    };
    gene_tree.assign_depths();

    let node_mapping = vec![Some(0), Some(0), Some(0), Some(1), Some(1)];
    let event_mapping = vec![
        Event::Speciation, Event::Leaf, Event::Speciation, Event::Leaf, Event::Loss,
    ];

    let rec_tree = RecTree::new_owned(
        (*sp_tree).clone(), gene_tree, node_mapping, event_mapping,
    );

    let extant = extract_extant_gene_tree(&rec_tree).unwrap();
    // Should have 2 extant leaves (g1, g2), not the loss node
    let leaf_count = extant.nodes.iter().filter(|n| is_leaf(n)).count();
    assert_eq!(leaf_count, 2, "Should extract exactly 2 extant leaves");
}

// ============================================================================
// Utility functions (shared by Python + R)
// ============================================================================

#[test]
fn test_digit_width() {
    assert_eq!(digit_width(0), 1);
    assert_eq!(digit_width(9), 1);
    assert_eq!(digit_width(10), 2);
    assert_eq!(digit_width(99), 2);
    assert_eq!(digit_width(100), 3);
    assert_eq!(digit_width(1000), 4);
}

#[test]
fn test_init_rng_deterministic() {
    use rand::Rng;
    let mut rng1 = init_rng(Some(42));
    let mut rng2 = init_rng(Some(42));
    let v1: f64 = rng1.gen();
    let v2: f64 = rng2.gen();
    assert_eq!(v1, v2, "Same seed should produce same values");
}

#[test]
fn test_is_leaf() {
    use rustree::node::FlatNode;
    let leaf = FlatNode {
        name: "leaf".into(), left_child: None, right_child: None,
        parent: Some(0), depth: None, length: 1.0, bd_event: None,
    };
    let internal = FlatNode {
        name: "int".into(), left_child: Some(1), right_child: Some(2),
        parent: None, depth: None, length: 0.0, bd_event: None,
    };
    assert!(is_leaf(&leaf));
    assert!(!is_leaf(&internal));
}

// ============================================================================
// Core operations parity: both bindings use the same underlying functions
// ============================================================================

#[test]
fn test_newick_roundtrip_parity() {
    // Both Python and R use parse_newick + to_newick
    use rustree::parse_newick;

    let input = "((A:1.0,B:2.0):0.5,(C:1.5,D:2.5):0.3):0.0;";
    let nodes = parse_newick(input).unwrap();
    let tree = nodes[0].to_flat_tree();
    let newick = tree.to_newick().unwrap();

    // Re-parse the output
    let reparsed = parse_newick(&format!("{};", newick)).unwrap();
    let tree2 = reparsed[0].to_flat_tree();

    assert_eq!(tree.get_leaves().len(), tree2.get_leaves().len());
    assert_eq!(tree.nodes.len(), tree2.nodes.len());
}

#[test]
fn test_pairwise_distances_parity() {
    // Both Python and R delegate to tree.pairwise_distances()
    use rustree::parse_newick;
    use rustree::metric_functions::DistanceType;

    let nodes = parse_newick("((A:1.0,B:2.0):0.5,(C:1.5,D:2.5):0.3):0.0;").unwrap();
    let tree = nodes[0].to_flat_tree();

    let topo = tree.pairwise_distances(DistanceType::Topological, true).unwrap();
    let metric = tree.pairwise_distances(DistanceType::Metric, true).unwrap();

    // 4 leaves → 6 pairs
    assert_eq!(topo.len(), 6);
    assert_eq!(metric.len(), 6);

    // Topological distances are integers (number of edges)
    for d in &topo {
        assert!(d.distance == d.distance.floor());
    }
}

#[test]
fn test_name_internal_nodes_parity() {
    // Both Python and R delegate to tree.name_internal_nodes()
    use rustree::parse_newick;

    let nodes = parse_newick("((A:1.0,B:2.0):0.5,(C:1.5,D:2.5):0.3):0.0;").unwrap();
    let mut tree = nodes[0].to_flat_tree();
    tree.name_internal_nodes().unwrap();

    let internal_names: Vec<&str> = tree.nodes.iter()
        .filter(|n| n.left_child.is_some())
        .map(|n| n.name.as_str())
        .collect();

    // All internal nodes should now have names starting with "internal"
    for name in &internal_names {
        assert!(name.starts_with("internal"), "Expected 'internal*' got '{}'", name);
    }
}

#[test]
fn test_species_tree_simulation_parity() {
    // Both Python and R delegate to simulate_bd_tree_bwd
    use rustree::bd::simulate_bd_tree_bwd;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(42);
    let (tree, events) = simulate_bd_tree_bwd(10, 1.0, 0.5, &mut rng).unwrap();

    // get_leaves() returns ALL leaves (extant + extinct); extant count >= n
    assert!(tree.get_leaves().len() >= 10, "Tree should have at least 10 leaves");
    assert!(!events.is_empty());
}

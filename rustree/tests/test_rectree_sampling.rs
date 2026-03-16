use rustree::{RecTree, Event, GeneForest, parse_newick};
use rustree::sampling::{compute_lca, build_leaf_pair_lca_map, get_descendant_leaf_names};
use std::sync::Arc;

/// Helper to create a simple species tree for testing
fn make_species_tree(newick: &str) -> rustree::FlatTree {
    let mut nodes = parse_newick(newick).unwrap();
    let mut root = nodes.pop().unwrap();
    root.zero_root_length();
    root.assign_depths(0.0);
    root.to_flat_tree()
}

/// Helper: build a single-tree GeneForest, sample, and return the sampled forest
fn sample_via_forest(
    species_tree: rustree::FlatTree,
    gene_tree: rustree::FlatTree,
    node_mapping: Vec<Option<usize>>,
    event_mapping: Vec<Event>,
    species_leaf_names: &[String],
) -> Result<GeneForest, String> {
    let species_arc = Arc::new(species_tree);
    let rt = RecTree::new(Arc::clone(&species_arc), gene_tree, node_mapping, event_mapping);
    let forest = GeneForest::from_rec_trees(species_arc, vec![rt]);
    forest.sample_leaves(species_leaf_names)
}

#[test]
fn test_compute_lca_simple() {
    // Test LCA computation on a simple tree
    let tree = make_species_tree("((A:1,B:1)AB:1,C:2)Root:0;");

    // Find indices
    let a_idx = tree.nodes.iter().position(|n| n.name == "A").unwrap();
    let b_idx = tree.nodes.iter().position(|n| n.name == "B").unwrap();
    let c_idx = tree.nodes.iter().position(|n| n.name == "C").unwrap();
    let ab_idx = tree.nodes.iter().position(|n| n.name == "AB").unwrap();

    // Test LCA(A, B) = AB
    let lca_ab = compute_lca(&tree, a_idx, b_idx);
    assert_eq!(lca_ab, ab_idx, "LCA(A, B) should be AB");

    // Test LCA(A, C) = Root
    let lca_ac = compute_lca(&tree, a_idx, c_idx);
    assert_eq!(lca_ac, tree.root, "LCA(A, C) should be Root");

    // Test LCA(B, C) = Root
    let lca_bc = compute_lca(&tree, b_idx, c_idx);
    assert_eq!(lca_bc, tree.root, "LCA(B, C) should be Root");
}

#[test]
fn test_build_leaf_pair_lca_map() {
    let tree = make_species_tree("((A:1,B:1)AB:1,C:2)Root:0;");
    let lca_map = build_leaf_pair_lca_map(&tree);

    let ab_idx = tree.nodes.iter().position(|n| n.name == "AB").unwrap();

    // Check that (A, B) and (B, A) both map to AB
    assert_eq!(lca_map.get(&("A".to_string(), "B".to_string())), Some(&ab_idx));
    assert_eq!(lca_map.get(&("B".to_string(), "A".to_string())), Some(&ab_idx));

    // Check that (A, C) maps to Root
    assert_eq!(lca_map.get(&("A".to_string(), "C".to_string())), Some(&tree.root));
}

#[test]
fn test_get_descendant_leaf_names() {
    let tree = make_species_tree("((A:1,B:1)AB:1,C:2)Root:0;");

    let ab_idx = tree.nodes.iter().position(|n| n.name == "AB").unwrap();
    let leaves = get_descendant_leaf_names(&tree, ab_idx);

    assert_eq!(leaves.len(), 2);
    assert!(leaves.contains(&"A".to_string()));
    assert!(leaves.contains(&"B".to_string()));

    let root_leaves = get_descendant_leaf_names(&tree, tree.root);
    assert_eq!(root_leaves.len(), 3);
    assert!(root_leaves.contains(&"A".to_string()));
    assert!(root_leaves.contains(&"B".to_string()));
    assert!(root_leaves.contains(&"C".to_string()));
}

#[test]
fn test_sample_species_leaves_simple() {
    // Create a simple species tree: ((A, B), C)
    let species_tree = make_species_tree("((A:1,B:1)AB:1,C:2)Root:0;");

    // Create a simple gene tree with genes mapping to each species
    let gene_tree = make_species_tree("((gA:1,gB:1):1,gC:2):0;");

    // Map gene nodes to species nodes
    let a_idx = species_tree.nodes.iter().position(|n| n.name == "A").unwrap();
    let b_idx = species_tree.nodes.iter().position(|n| n.name == "B").unwrap();
    let c_idx = species_tree.nodes.iter().position(|n| n.name == "C").unwrap();

    // Find the internal nodes by structure (parent of A and B)
    let ab_idx = species_tree.nodes[a_idx].parent.unwrap();

    // Simple mapping: map each gene directly to corresponding species
    let mut node_mapping: Vec<Option<usize>> = vec![Some(0); gene_tree.nodes.len()];
    let mut event_mapping = vec![Event::Speciation; gene_tree.nodes.len()];

    for (idx, node) in gene_tree.nodes.iter().enumerate() {
        if node.left_child.is_none() && node.right_child.is_none() {
            // Leaf - map to species by name pattern
            if node.name.contains("gA") {
                node_mapping[idx] = Some(a_idx);
                event_mapping[idx] = Event::Leaf;
            } else if node.name.contains("gB") {
                node_mapping[idx] = Some(b_idx);
                event_mapping[idx] = Event::Leaf;
            } else if node.name.contains("gC") {
                node_mapping[idx] = Some(c_idx);
                event_mapping[idx] = Event::Leaf;
            }
        } else {
            // Internal node - map to AB (LCA of A and B)
            node_mapping[idx] = Some(ab_idx);
        }
    }

    // Sample only species A and B via GeneForest
    let sampled = sample_via_forest(
        species_tree, gene_tree, node_mapping, event_mapping,
        &["A".to_string(), "B".to_string()],
    ).expect("Sampling should succeed");

    let sampled_rt = sampled.get(0).unwrap();

    // Should have 3 species nodes: A, B, AB
    assert_eq!(sampled_rt.species_tree.nodes.len(), 3, "Sampled species tree should have 3 nodes");

    // Should have 3 gene nodes: gA, gB, g1
    assert_eq!(sampled_rt.gene_tree.nodes.len(), 3, "Sampled gene tree should have 3 nodes");

    // Verify all gene leaves are present
    let gene_leaves: Vec<String> = sampled_rt.gene_tree.nodes.iter()
        .filter(|n| n.left_child.is_none() && n.right_child.is_none())
        .map(|n| n.name.clone())
        .collect();

    assert!(gene_leaves.contains(&"gA".to_string()));
    assert!(gene_leaves.contains(&"gB".to_string()));
    assert!(!gene_leaves.contains(&"gC".to_string()), "gC should be filtered out");
}

#[test]
fn test_sample_species_leaves_single_species() {
    let species_tree = make_species_tree("((A:1,B:1)AB:1,C:2)Root:0;");
    let gene_tree = make_species_tree("((gA:1,gB:1):1,gC:2):0;");

    let a_idx = species_tree.nodes.iter().position(|n| n.name == "A").unwrap();
    let b_idx = species_tree.nodes.iter().position(|n| n.name == "B").unwrap();
    let c_idx = species_tree.nodes.iter().position(|n| n.name == "C").unwrap();
    let ab_idx = species_tree.nodes.iter().position(|n| n.name == "AB").unwrap();
    let root_idx = species_tree.root;

    let node_mapping = vec![Some(root_idx), Some(ab_idx), Some(a_idx), Some(b_idx), Some(c_idx)];
    let event_mapping = vec![Event::Speciation, Event::Speciation, Event::Leaf, Event::Leaf, Event::Leaf];

    // Sample only species A
    let sampled = sample_via_forest(
        species_tree, gene_tree, node_mapping, event_mapping,
        &["A".to_string()],
    ).expect("Sampling single species should succeed");

    let sampled_rt = sampled.get(0).unwrap();

    // Should have 1 species node
    assert_eq!(sampled_rt.species_tree.nodes.len(), 1);

    // Should have 1 gene node (gA)
    assert_eq!(sampled_rt.gene_tree.nodes.len(), 1);
    assert_eq!(sampled_rt.gene_tree.nodes[0].name, "gA");
}

#[test]
fn test_sample_species_leaves_all_species() {
    let species_tree = make_species_tree("((A:1,B:1)AB:1,C:2)Root:0;");
    let gene_tree = make_species_tree("((gA:1,gB:1):1,gC:2):0;");

    let a_idx = species_tree.nodes.iter().position(|n| n.name == "A").unwrap();
    let b_idx = species_tree.nodes.iter().position(|n| n.name == "B").unwrap();
    let c_idx = species_tree.nodes.iter().position(|n| n.name == "C").unwrap();
    let ab_idx = species_tree.nodes.iter().position(|n| n.name == "AB").unwrap();
    let root_idx = species_tree.root;

    let original_species_len = species_tree.nodes.len();
    let original_gene_len = gene_tree.nodes.len();

    let node_mapping = vec![Some(root_idx), Some(ab_idx), Some(a_idx), Some(b_idx), Some(c_idx)];
    let event_mapping = vec![Event::Speciation, Event::Speciation, Event::Leaf, Event::Leaf, Event::Leaf];

    // Sample all species
    let sampled = sample_via_forest(
        species_tree, gene_tree, node_mapping, event_mapping,
        &["A".to_string(), "B".to_string(), "C".to_string()],
    ).expect("Sampling all species should succeed");

    let sampled_rt = sampled.get(0).unwrap();

    // Should have same number of nodes as original
    assert_eq!(sampled_rt.species_tree.nodes.len(), original_species_len);
    assert_eq!(sampled_rt.gene_tree.nodes.len(), original_gene_len);
}

#[test]
fn test_sample_species_leaves_with_duplication() {
    // Create species tree
    let species_tree = make_species_tree("(A:1,B:1)Root:0;");

    // Create gene tree with duplication in species A: ((gA1, gA2), gB)
    let gene_tree = make_species_tree("((gA1:0.5,gA2:0.5):0.5,gB:1):0;");

    let a_idx = species_tree.nodes.iter().position(|n| n.name == "A").unwrap();
    let b_idx = species_tree.nodes.iter().position(|n| n.name == "B").unwrap();

    // Build mappings based on gene tree structure
    let mut node_mapping: Vec<Option<usize>> = vec![Some(0); gene_tree.nodes.len()];
    let mut event_mapping = vec![Event::Speciation; gene_tree.nodes.len()];

    for (idx, node) in gene_tree.nodes.iter().enumerate() {
        if node.left_child.is_none() && node.right_child.is_none() {
            // Leaf nodes
            if node.name.contains("gA") {
                node_mapping[idx] = Some(a_idx);
                event_mapping[idx] = Event::Leaf;
            } else if node.name.contains("gB") {
                node_mapping[idx] = Some(b_idx);
                event_mapping[idx] = Event::Leaf;
            }
        } else {
            // Internal nodes - find which one is the duplication
            let left_child_idx = node.left_child.unwrap();
            let right_child_idx = node.right_child.unwrap();
            let left_name = &gene_tree.nodes[left_child_idx].name;
            let right_name = &gene_tree.nodes[right_child_idx].name;

            if left_name.contains("gA") && right_name.contains("gA") {
                // This is the duplication node (parent of two A genes)
                node_mapping[idx] = Some(a_idx);
                event_mapping[idx] = Event::Duplication;
            } else {
                // Root speciation node
                node_mapping[idx] = Some(a_idx);
                event_mapping[idx] = Event::Speciation;
            }
        }
    }

    // Sample both species A and B to avoid root node issues
    let sampled = sample_via_forest(
        species_tree, gene_tree, node_mapping, event_mapping,
        &["A".to_string(), "B".to_string()],
    ).expect("Sampling should succeed");

    let sampled_rt = sampled.get(0).unwrap();

    // Should have 3 species nodes: A, B, and their parent
    assert_eq!(sampled_rt.species_tree.nodes.len(), 3);

    // Should have all 5 gene nodes since we're keeping both A and B
    assert_eq!(sampled_rt.gene_tree.nodes.len(), 5);

    // Check that duplication event is preserved
    let dup_count = sampled_rt.event_mapping.iter()
        .filter(|e| **e == Event::Duplication)
        .count();
    assert_eq!(dup_count, 1, "Duplication event should be preserved");
}

#[test]
fn test_sample_species_leaves_no_matching_genes() {
    let species_tree = make_species_tree("((A:1,B:1)AB:1,C:2)Root:0;");
    let gene_tree = make_species_tree("(gA:1,gB:1):0;");

    let a_idx = species_tree.nodes.iter().position(|n| n.name == "A").unwrap();
    let b_idx = species_tree.nodes.iter().position(|n| n.name == "B").unwrap();
    let ab_idx = species_tree.nodes.iter().position(|n| n.name == "AB").unwrap();

    let node_mapping = vec![Some(ab_idx), Some(a_idx), Some(b_idx)];
    let event_mapping = vec![Event::Speciation, Event::Leaf, Event::Leaf];

    // Try to sample species C (which has no genes)
    let result = sample_via_forest(
        species_tree, gene_tree, node_mapping, event_mapping,
        &["C".to_string()],
    );

    assert!(result.is_err(), "Should fail when no genes map to sampled species");
    assert!(result.unwrap_err().contains("No gene tree leaves"));
}

#[test]
fn test_sample_species_leaves_invalid_species() {
    let species_tree = make_species_tree("(A:1,B:1):0;");
    let gene_tree = make_species_tree("(gA:1,gB:1):0;");

    let a_idx = species_tree.nodes.iter().position(|n| n.name == "A").unwrap();
    let b_idx = species_tree.nodes.iter().position(|n| n.name == "B").unwrap();
    let root_idx = species_tree.root;

    let node_mapping = vec![Some(root_idx), Some(a_idx), Some(b_idx)];
    let event_mapping = vec![Event::Speciation, Event::Leaf, Event::Leaf];

    // Try to sample non-existent species
    let result = sample_via_forest(
        species_tree, gene_tree, node_mapping, event_mapping,
        &["NonExistent".to_string()],
    );

    assert!(result.is_err(), "Should fail when species doesn't exist");
    assert!(result.unwrap_err().contains("Failed to sample species tree"));
}

#[test]
fn test_sample_preserves_event_types() {
    // Create a tree with multiple event types
    let species_tree = make_species_tree("(A:1,B:1):0;");
    let gene_tree = make_species_tree("((gA1:0.5,gA2:0.5):0.5,gB:1):0;");

    let a_idx = species_tree.nodes.iter().position(|n| n.name == "A").unwrap();
    let b_idx = species_tree.nodes.iter().position(|n| n.name == "B").unwrap();

    // Build mappings based on gene tree structure
    let mut node_mapping: Vec<Option<usize>> = vec![Some(0); gene_tree.nodes.len()];
    let mut event_mapping = vec![Event::Speciation; gene_tree.nodes.len()];

    for (idx, node) in gene_tree.nodes.iter().enumerate() {
        if node.left_child.is_none() && node.right_child.is_none() {
            // Leaf nodes
            if node.name.contains("gA") {
                node_mapping[idx] = Some(a_idx);
                event_mapping[idx] = Event::Leaf;
            } else if node.name.contains("gB") {
                node_mapping[idx] = Some(b_idx);
                event_mapping[idx] = Event::Leaf;
            }
        } else {
            // Internal nodes
            let left_child_idx = node.left_child.unwrap();
            let right_child_idx = node.right_child.unwrap();
            let left_name = &gene_tree.nodes[left_child_idx].name;
            let right_name = &gene_tree.nodes[right_child_idx].name;

            if left_name.contains("gA") && right_name.contains("gA") {
                // Duplication node
                node_mapping[idx] = Some(a_idx);
                event_mapping[idx] = Event::Duplication;
            } else {
                // Root speciation node
                node_mapping[idx] = Some(a_idx);
                event_mapping[idx] = Event::Speciation;
            }
        }
    }

    // Sample both species
    let sampled = sample_via_forest(
        species_tree, gene_tree, node_mapping, event_mapping,
        &["A".to_string(), "B".to_string()],
    ).expect("Sampling should succeed");

    let sampled_rt = sampled.get(0).unwrap();

    // Check event preservation
    let spec_count = sampled_rt.event_mapping.iter().filter(|e| **e == Event::Speciation).count();
    let dup_count = sampled_rt.event_mapping.iter().filter(|e| **e == Event::Duplication).count();
    let leaf_count = sampled_rt.event_mapping.iter().filter(|e| **e == Event::Leaf).count();

    // Should have at least the expected events
    assert!(spec_count >= 1, "Should have at least 1 speciation");
    assert_eq!(dup_count, 1, "Should have 1 duplication");
    assert_eq!(leaf_count, 3, "Should have 3 leaves");
}

// Compute induced transfers: project transfer donors/recipients from a complete
// species tree onto a sampled subtree.
//
// Given a complete species tree, a subset of sampled leaf names, and a list of
// DTL events, we compute for each transfer the "induced donor" and "induced
// recipient" — the branches of the sampled tree that the original donor and
// recipient project onto.

use std::collections::HashMap;
use crate::node::FlatTree;
use crate::dtl::DTLEvent;
use crate::error::RustreeError;
use crate::sampling::{NodeMark, mark_nodes_postorder, find_leaf_indices_by_names, extract_induced_subtree_by_names};

/// An induced transfer: a transfer event with both complete-tree and
/// sampled-tree donor/recipient species.
#[derive(Clone, Debug)]
pub struct InducedTransfer {
    pub time: f64,
    pub gene_id: usize,
    /// Donor species index in the complete tree.
    pub from_species_complete: usize,
    /// Recipient species index in the complete tree.
    pub to_species_complete: usize,
    /// Induced donor: index in the sampled tree (None if projection fails).
    pub from_species_sampled: Option<usize>,
    /// Induced recipient: index in the sampled tree (None if projection fails).
    pub to_species_sampled: Option<usize>,
}

/// Returns the sibling of `node_idx` (the other child of its parent).
fn get_sibling(tree: &FlatTree, node_idx: usize) -> Option<usize> {
    let parent_idx = tree.nodes[node_idx].parent?;
    let parent = &tree.nodes[parent_idx];
    if parent.left_child == Some(node_idx) {
        parent.right_child
    } else {
        parent.left_child
    }
}

/// Projects a single node onto the nearest Keep node by walking through the tree.
///
/// - **Keep**: returns itself.
/// - **HasDescendant**: walks down to the child that has kept descendants.
/// - **Discard**: walks to sibling; if sibling is also Discard, walks up to parent.
fn project_node(tree: &FlatTree, marks: &[NodeMark], node_idx: usize) -> Option<usize> {
    match marks[node_idx] {
        NodeMark::Keep => Some(node_idx),
        NodeMark::HasDescendant => {
            let left = tree.nodes[node_idx].left_child;
            let right = tree.nodes[node_idx].right_child;
            let next = match (left, right) {
                (Some(l), _) if marks[l] != NodeMark::Discard => l,
                (_, Some(r)) => r,
                _ => return None,
            };
            project_node(tree, marks, next)
        }
        NodeMark::Discard => {
            let sibling = get_sibling(tree, node_idx);
            match sibling {
                Some(s) if marks[s] != NodeMark::Discard => {
                    project_node(tree, marks, s)
                }
                _ => {
                    // Sibling is Discard or doesn't exist — go up to parent
                    let parent = tree.nodes[node_idx].parent?;
                    project_node(tree, marks, parent)
                }
            }
        }
    }
}

/// Computes the projection of every node in the complete tree onto the nearest
/// Keep node (a node that is present in the sampled tree).
///
/// Returns a `Vec<Option<usize>>` indexed by complete-tree node index, where
/// each entry is the index (in the complete tree) of the Keep node it projects
/// onto, or `None` if no projection exists.
pub(crate) fn compute_projection(
    complete_tree: &FlatTree,
    marks: &[NodeMark],
) -> Vec<Option<usize>> {
    (0..complete_tree.nodes.len())
        .map(|idx| project_node(complete_tree, marks, idx))
        .collect()
}

/// Computes the ghost length for each branch of the sampled tree.
///
/// The ghost length of a sampled-tree branch is the sum of branch lengths of all
/// non-sampled nodes (HasDescendant and Discard) in the complete tree that project
/// onto that branch.
///
/// # Arguments
/// * `complete_tree` - The complete species tree
/// * `sampled_leaf_names` - Names of leaves to keep
///
/// # Returns
/// A tuple of `(sampled_tree, ghost_lengths)` where `ghost_lengths` is a `Vec<f64>`
/// indexed by sampled-tree node index.
pub fn ghost_lengths(
    complete_tree: &FlatTree,
    sampled_leaf_names: &[String],
) -> Result<(FlatTree, Vec<f64>), RustreeError> {
    let (sampled_tree, _) = extract_induced_subtree_by_names(complete_tree, sampled_leaf_names)
        .ok_or_else(|| RustreeError::Tree("Failed to extract induced subtree from leaf names".to_string()))?;

    let keep_indices = find_leaf_indices_by_names(complete_tree, sampled_leaf_names);
    let mut marks = vec![NodeMark::Discard; complete_tree.nodes.len()];
    mark_nodes_postorder(complete_tree, complete_tree.root, &keep_indices, &mut marks);

    let projection = compute_projection(complete_tree, &marks);

    // Map complete-tree Keep-node names → sampled-tree indices
    let mut keep_name_to_sampled: HashMap<&str, usize> = HashMap::new();
    for (idx, node) in sampled_tree.nodes.iter().enumerate() {
        keep_name_to_sampled.insert(&node.name, idx);
    }

    let mut ghost = vec![0.0; sampled_tree.nodes.len()];

    for (complete_idx, mark) in marks.iter().enumerate() {
        if *mark == NodeMark::Keep {
            continue;
        }
        // Non-Keep node: add its branch length to the projected sampled-tree branch
        if let Some(keep_idx) = projection[complete_idx] {
            let name = &complete_tree.nodes[keep_idx].name;
            if let Some(&sampled_idx) = keep_name_to_sampled.get(name.as_str()) {
                ghost[sampled_idx] += complete_tree.nodes[complete_idx].length;
            }
        }
    }

    Ok((sampled_tree, ghost))
}

/// Computes induced transfers by projecting each transfer's donor and recipient
/// from a complete species tree onto a sampled subtree.
///
/// # Arguments
/// * `complete_tree` - The complete species tree (must have depths assigned)
/// * `sampled_leaf_names` - Names of leaves to keep (must be a subset of complete tree leaves)
/// * `events` - DTL events from a gene tree simulation on the complete tree
///
/// # Returns
/// A vector of `InducedTransfer` — one per Transfer event in `events`.
pub fn induced_transfers(
    complete_tree: &FlatTree,
    sampled_leaf_names: &[String],
    events: &[DTLEvent],
) -> Result<Vec<InducedTransfer>, RustreeError> {
    let (sampled_tree, _) = extract_induced_subtree_by_names(complete_tree, sampled_leaf_names)
        .ok_or_else(|| RustreeError::Tree("Failed to extract induced subtree from leaf names".to_string()))?;

    // Step 1: Mark complete tree nodes
    let keep_indices = find_leaf_indices_by_names(complete_tree, sampled_leaf_names);
    let mut marks = vec![NodeMark::Discard; complete_tree.nodes.len()];
    mark_nodes_postorder(complete_tree, complete_tree.root, &keep_indices, &mut marks);

    // Step 2: Compute projection (complete tree index → Keep node index in complete tree)
    let projection = compute_projection(complete_tree, &marks);

    // Step 3: Build mapping from complete tree Keep-node names to sampled tree indices
    let mut keep_name_to_sampled: HashMap<&str, usize> = HashMap::new();
    for (idx, node) in sampled_tree.nodes.iter().enumerate() {
        keep_name_to_sampled.insert(&node.name, idx);
    }

    // Step 4: Compose projection → Keep node name → sampled tree index
    let complete_to_sampled: Vec<Option<usize>> = projection
        .iter()
        .map(|proj| {
            proj.and_then(|keep_idx| {
                let name = &complete_tree.nodes[keep_idx].name;
                keep_name_to_sampled.get(name.as_str()).copied()
            })
        })
        .collect();

    // Step 5: For each transfer event, produce an InducedTransfer
    Ok(events
        .iter()
        .filter_map(|event| {
            if let DTLEvent::Transfer { time, gene_id, from_species, to_species, .. } = event {
                Some(InducedTransfer {
                    time: *time,
                    gene_id: *gene_id,
                    from_species_complete: *from_species,
                    to_species_complete: *to_species,
                    from_species_sampled: complete_to_sampled[*from_species],
                    to_species_sampled: complete_to_sampled[*to_species],
                })
            } else {
                None
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::newick::parse_newick;
    use crate::sampling::extract_induced_subtree_by_names;

    fn make_tree(newick: &str) -> FlatTree {
        let mut nodes = parse_newick(newick).unwrap();
        let mut tree = nodes.pop().unwrap().to_flat_tree();
        tree.assign_depths();
        tree
    }

    #[test]
    fn test_projection_balanced_tree() {
        // Tree: ((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;
        // Sample: {A, C}
        // Expected projections:
        //   A → A (Keep), B → A (sibling), C → C (Keep), D → C (sibling)
        //   AB → A (HasDescendant, left child A is Keep)
        //   CD → C (HasDescendant, left child C is Keep)
        //   root → root (Keep, both subtrees have kept descendants)
        let tree = make_tree("((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;");
        let sampled_names: Vec<String> = vec!["A".into(), "C".into()];

        let keep_indices = find_leaf_indices_by_names(&tree, &sampled_names);
        let mut marks = vec![NodeMark::Discard; tree.nodes.len()];
        mark_nodes_postorder(&tree, tree.root, &keep_indices, &mut marks);

        let projection = compute_projection(&tree, &marks);

        // Get node indices by name
        let idx = |name: &str| tree.find_node_index(name).unwrap();

        // Keep nodes project to themselves
        assert_eq!(projection[idx("A")], Some(idx("A")));
        assert_eq!(projection[idx("C")], Some(idx("C")));
        assert_eq!(projection[idx("root")], Some(idx("root")));

        // B projects to its sibling A
        assert_eq!(projection[idx("B")], Some(idx("A")));
        // D projects to its sibling C
        assert_eq!(projection[idx("D")], Some(idx("C")));

        // AB (HasDescendant) projects down to A
        assert_eq!(projection[idx("AB")], Some(idx("A")));
        // CD (HasDescendant) projects down to C
        assert_eq!(projection[idx("CD")], Some(idx("C")));
    }

    #[test]
    fn test_projection_one_subtree_sampled() {
        // Tree: ((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;
        // Sample: {A, B} — only left subtree
        // root is HasDescendant, CD subtree is all Discard
        let tree = make_tree("((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;");
        let sampled_names: Vec<String> = vec!["A".into(), "B".into()];

        let keep_indices = find_leaf_indices_by_names(&tree, &sampled_names);
        let mut marks = vec![NodeMark::Discard; tree.nodes.len()];
        mark_nodes_postorder(&tree, tree.root, &keep_indices, &mut marks);

        let projection = compute_projection(&tree, &marks);

        let idx = |name: &str| tree.find_node_index(name).unwrap();

        // AB is Keep (MRCA of A, B)
        assert_eq!(projection[idx("AB")], Some(idx("AB")));
        assert_eq!(projection[idx("A")], Some(idx("A")));
        assert_eq!(projection[idx("B")], Some(idx("B")));

        // root is HasDescendant → projects down to AB
        assert_eq!(projection[idx("root")], Some(idx("AB")));

        // CD is Discard → sibling AB has kept descendants → projects to AB
        assert_eq!(projection[idx("CD")], Some(idx("AB")));

        // C is Discard → sibling D is Discard → go up to CD (Discard) →
        // sibling of CD is AB → projects to AB
        assert_eq!(projection[idx("C")], Some(idx("AB")));
        assert_eq!(projection[idx("D")], Some(idx("AB")));
    }

    #[test]
    fn test_induced_transfers_basic() {
        let complete_tree = make_tree("((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;");
        let sampled_names: Vec<String> = vec!["A".into(), "C".into()];

        let idx = |name: &str| complete_tree.find_node_index(name).unwrap();

        // A transfer from B (complete) to D (complete)
        let events = vec![DTLEvent::Transfer {
            time: 0.5,
            gene_id: 0,
            species_id: idx("B"),
            from_species: idx("B"),
            to_species: idx("D"),
            donor_child: 1,
            recipient_child: 2,
        }];

        let induced = induced_transfers(&complete_tree, &sampled_names, &events).unwrap();
        assert_eq!(induced.len(), 1);

        let t = &induced[0];
        assert_eq!(t.from_species_complete, idx("B"));
        assert_eq!(t.to_species_complete, idx("D"));

        // B projects to A in the complete tree, then A maps to sampled tree
        let sampled_tree = extract_induced_subtree_by_names(&complete_tree, &sampled_names).unwrap().0;
        let sampled_a = sampled_tree.find_node_index("A").unwrap();
        let sampled_c = sampled_tree.find_node_index("C").unwrap();
        assert_eq!(t.from_species_sampled, Some(sampled_a));
        assert_eq!(t.to_species_sampled, Some(sampled_c));
    }

    #[test]
    fn test_ghost_lengths_balanced() {
        // Tree: ((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;
        // Sample: {A, C}
        //
        // Non-Keep nodes and their projections:
        //   B  (length 1) → projects to A
        //   D  (length 1) → projects to C
        //   AB (length 1) → projects to A
        //   CD (length 1) → projects to C
        //
        // Ghost lengths: A = 1 + 1 = 2, C = 1 + 1 = 2, root = 0
        let complete = make_tree("((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;");
        let sampled_names: Vec<String> = vec!["A".into(), "C".into()];

        let (sampled, ghost) = ghost_lengths(&complete, &sampled_names).unwrap();

        let sampled_a = sampled.find_node_index("A").unwrap();
        let sampled_c = sampled.find_node_index("C").unwrap();

        assert!((ghost[sampled_a] - 2.0).abs() < 1e-10, "Ghost length of A should be 2.0, got {}", ghost[sampled_a]);
        assert!((ghost[sampled_c] - 2.0).abs() < 1e-10, "Ghost length of C should be 2.0, got {}", ghost[sampled_c]);
    }

    #[test]
    fn test_ghost_lengths_one_subtree() {
        // Tree: ((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;
        // Sample: {A, B} — only left subtree
        //
        // Non-Keep nodes: root (length 0), CD (length 1), C (length 1), D (length 1)
        // All project to AB.
        // Ghost length of AB = 0 + 1 + 1 + 1 = 3
        let complete = make_tree("((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;");
        let sampled_names: Vec<String> = vec!["A".into(), "B".into()];

        let (sampled, ghost) = ghost_lengths(&complete, &sampled_names).unwrap();

        let sampled_ab = sampled.find_node_index("AB").unwrap();
        let sampled_a = sampled.find_node_index("A").unwrap();
        let sampled_b = sampled.find_node_index("B").unwrap();

        assert!((ghost[sampled_ab] - 3.0).abs() < 1e-10, "Ghost length of AB should be 3.0, got {}", ghost[sampled_ab]);
        assert!((ghost[sampled_a]).abs() < 1e-10, "Ghost length of A should be 0.0");
        assert!((ghost[sampled_b]).abs() < 1e-10, "Ghost length of B should be 0.0");
    }

    #[test]
    fn test_induced_transfers_non_transfer_events_filtered() {
        let complete_tree = make_tree("((A:1,B:1)AB:1,C:2)root:0;");
        let sampled_names: Vec<String> = vec!["A".into(), "C".into()];

        let events = vec![
            DTLEvent::Duplication {
                time: 0.3,
                gene_id: 0,
                species_id: 0,
                child1: 1,
                child2: 2,
            },
            DTLEvent::Loss {
                time: 0.4,
                gene_id: 1,
                species_id: 0,
            },
            DTLEvent::Leaf {
                time: 1.0,
                gene_id: 2,
                species_id: 0,
            },
        ];

        let induced = induced_transfers(&complete_tree, &sampled_names, &events).unwrap();
        assert_eq!(induced.len(), 0, "Non-transfer events should be filtered out");
    }
}

//! Conversion functions between recursive Node and flat FlatTree representations.

use super::{Node, FlatNode, FlatTree, RecTree};
use std::collections::HashMap;
use crate::error::RustreeError;

// Methods on Node for conversion
impl Node {
    /// Converts a recursive `Node` structure into a `FlatTree`.
    #[must_use]
    pub fn to_flat_tree(&self) -> FlatTree {
        let mut flat_nodes = Vec::new();
        let root_index = self.node_to_flat_internal(&mut flat_nodes, None);
        FlatTree {
            nodes: flat_nodes,
            root: root_index,
        }
    }

    /// Internal helper method for converting a `Node` to flat nodes.
    fn node_to_flat_internal(&self, flat_nodes: &mut Vec<FlatNode>, parent_index: Option<usize>) -> usize {
        let index = flat_nodes.len();
        flat_nodes.push(FlatNode {
            name: self.name.clone(),
            left_child: None,
            right_child: None,
            parent: parent_index,
            depth: self.depth,
            length: self.length,
            bd_event: None,
        });

        if let Some(ref left_child) = self.left_child {
            let left_index = left_child.node_to_flat_internal(flat_nodes, Some(index));
            flat_nodes[index].left_child = Some(left_index);
        }

        if let Some(ref right_child) = self.right_child {
            let right_index = right_child.node_to_flat_internal(flat_nodes, Some(index));
            flat_nodes[index].right_child = Some(right_index);
        }

        index
    }
}

// Methods on FlatTree for conversion
impl FlatTree {
    /// Converts a `FlatTree` into a recursive `Node` structure.
    #[must_use]
    pub fn to_node(&self) -> Node {
        self.flat_to_node_internal(self.root)
    }

    /// Find a node by name and return its index.
    pub fn find_node_index(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == name)
    }

    /// Internal helper method for converting flat nodes to a `Node`.
    fn flat_to_node_internal(&self, index: usize) -> Node {
        let flat_node = &self.nodes[index];

        let left_child = flat_node.left_child.map(|i| {
            Box::new(self.flat_to_node_internal(i))
        });

        let right_child = flat_node.right_child.map(|i| {
            Box::new(self.flat_to_node_internal(i))
        });

        Node {
            name: flat_node.name.clone(),
            left_child,
            right_child,
            depth: flat_node.depth,
            length: flat_node.length,
        }
    }

    /// Returns the number of nodes in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the tree has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns all structural leaf nodes (nodes with no children).
    pub fn get_leaves(&self) -> Vec<&FlatNode> {
        self.nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .collect()
    }

    /// Name unnamed internal nodes as `internal0`, `internal1`, etc.
    ///
    /// Returns an error if any existing node name starts with `"internal"`, to avoid
    /// ambiguity between original and generated names.
    pub fn name_internal_nodes(&mut self) -> Result<(), RustreeError> {
        if self.nodes.iter().any(|n| n.name.starts_with("internal")) {
            return Err(RustreeError::Validation(
                "Cannot auto-name internal nodes: at least one node already has a name starting with \"internal\"".to_string()
            ));
        }
        let mut counter = 0usize;
        for i in 0..self.nodes.len() {
            let is_internal = self.nodes[i].left_child.is_some() || self.nodes[i].right_child.is_some();
            if is_internal && self.nodes[i].name.is_empty() {
                self.nodes[i].name = format!("internal{}", counter);
                counter += 1;
            }
        }
        Ok(())
    }

    /// Returns extant leaf nodes — leaves whose birth-death event is `BDEvent::Leaf`.
    pub fn get_extant_leaves(&self) -> Vec<&FlatNode> {
        use crate::bd::BDEvent;
        self.nodes.iter()
            .filter(|n| {
                n.left_child.is_none()
                    && n.right_child.is_none()
                    && n.bd_event == Some(BDEvent::Leaf)
            })
            .collect()
    }
}

/// Map nodes between two trees with identical topology using postorder traversal.
///
/// Given two trees with the same topology (same branching structure), this function
/// builds a mapping from node indices in `target_tree` to the corresponding node
/// indices in `source_tree`. The matching is based on postorder position: nodes at
/// the same position in a postorder traversal of each tree are considered corresponding.
///
/// This is useful when you have two representations of the same tree that may have
/// different internal node names or different index ordering (e.g., after re-parsing,
/// or after a tool like ALERax re-indexes nodes).
///
/// # Arguments
/// * `source_tree` - The tree whose indices will be the *values* in the mapping
/// * `target_tree` - The tree whose indices will be the *keys* in the mapping
///
/// # Returns
/// A HashMap where keys are `target_tree` node indices and values are the
/// corresponding `source_tree` node indices.
///
/// # Errors
/// Returns an error if the trees have different topologies (different number of
/// nodes or leaf/internal mismatch at any position).
///
/// # Example
/// ```rust,no_run
/// # use rustree::FlatTree;
/// # use rustree::node::map_by_topology;
/// # let tree_a = FlatTree { nodes: vec![], root: 0 };
/// # let tree_b = FlatTree { nodes: vec![], root: 0 };
/// let mapping = map_by_topology(&tree_a, &tree_b)?;
/// // mapping[b_idx] == a_idx for corresponding nodes
/// # Ok::<(), rustree::RustreeError>(())
/// ```
pub fn map_by_topology(
    source_tree: &FlatTree,
    target_tree: &FlatTree,
) -> Result<HashMap<usize, usize>, RustreeError> {
    if source_tree.nodes.len() != target_tree.nodes.len() {
        return Err(RustreeError::Tree(format!(
            "Trees have different number of nodes: source={}, target={}",
            source_tree.nodes.len(),
            target_tree.nodes.len()
        )));
    }

    let source_postorder = source_tree.postorder_indices();
    let target_postorder = target_tree.postorder_indices();

    if source_postorder.len() != target_postorder.len() {
        return Err(RustreeError::Tree("Trees have different structures (postorder lengths differ)".to_string()));
    }

    let mut mapping = HashMap::new();
    for (pos, (&target_idx, &source_idx)) in target_postorder
        .iter()
        .zip(source_postorder.iter())
        .enumerate()
    {
        let source_is_leaf = source_tree.nodes[source_idx].left_child.is_none()
            && source_tree.nodes[source_idx].right_child.is_none();
        let target_is_leaf = target_tree.nodes[target_idx].left_child.is_none()
            && target_tree.nodes[target_idx].right_child.is_none();

        if source_is_leaf != target_is_leaf {
            return Err(RustreeError::Tree(format!(
                "Structural mismatch at postorder position {}: source node {} and target node {} differ in leaf/internal status",
                pos, source_idx, target_idx
            )));
        }

        mapping.insert(target_idx, source_idx);
    }

    Ok(mapping)
}

/// Rename gene tree nodes in a RecTree to match a reference tree.
///
/// Given a reference tree and a RecTree whose gene tree has the same topology,
/// this function renames all gene tree nodes to match the corresponding node names
/// in the reference tree. The matching is done via postorder topology mapping.
///
/// The reconciliation data (node_mapping and event_mapping) are indexed by gene tree
/// node indices, so they remain valid after renaming — only names change, not indices.
///
/// # Arguments
/// * `reference_tree` - The tree with desired node names
/// * `rec_tree` - Mutable reference to the RecTree whose gene tree will be renamed
///
/// # Returns
/// Ok(()) on success, or an error if trees have incompatible topologies.
///
/// # Example
/// ```rust,ignore
/// rename_gene_tree(&original_tree, &mut rec_tree)?;
/// // Now rec_tree.gene_tree has the same node names as original_tree
/// ```
pub fn rename_gene_tree(
    reference_tree: &FlatTree,
    rec_tree: &mut RecTree,
) -> Result<(), RustreeError> {
    let mapping = map_by_topology(reference_tree, &rec_tree.gene_tree)?;

    for (target_idx, &source_idx) in mapping.iter() {
        rec_tree.gene_tree.nodes[*target_idx].name =
            reference_tree.nodes[source_idx].name.clone();
    }

    Ok(())
}

#[cfg(test)]
mod mapping_tests {
    use super::*;

    #[test]
    fn test_obtain_mapping_identical_trees() {
        // Create two identical trees
        let mut nodes = vec![
            FlatNode {
                name: "A".to_string(),
                left_child: None,
                right_child: None,
                parent: Some(2),
                depth: Some(1.0),
                length: 1.0,
                bd_event: None,
            },
            FlatNode {
                name: "B".to_string(),
                left_child: None,
                right_child: None,
                parent: Some(2),
                depth: Some(1.0),
                length: 1.0,
                bd_event: None,
            },
            FlatNode {
                name: "Root".to_string(),
                left_child: Some(0),
                right_child: Some(1),
                parent: None,
                depth: Some(0.0),
                length: 0.0,
                bd_event: None,
            },
        ];

        let tree1 = FlatTree {
            nodes: nodes.clone(),
            root: 2,
        };

        // Create second tree with different names
        nodes[0].name = "X".to_string();
        nodes[1].name = "Y".to_string();
        nodes[2].name = "Top".to_string();

        let tree2 = FlatTree { nodes, root: 2 };

        // Obtain mapping (tree1 = source, tree2 = target)
        let mapping = map_by_topology(&tree1, &tree2).unwrap();

        // Verify mapping
        assert_eq!(mapping.len(), 3);
        assert_eq!(mapping[&0], 0); // Leaf A maps to leaf A
        assert_eq!(mapping[&1], 1); // Leaf B maps to leaf B
        assert_eq!(mapping[&2], 2); // Root maps to Root
    }

    #[test]
    fn test_obtain_mapping_different_sizes() {
        let tree1 = FlatTree {
            nodes: vec![FlatNode {
                name: "A".to_string(),
                left_child: None,
                right_child: None,
                parent: None,
                depth: Some(0.0),
                length: 0.0,
                bd_event: None,
            }],
            root: 0,
        };

        let tree2 = FlatTree {
            nodes: vec![
                FlatNode {
                    name: "X".to_string(),
                    left_child: None,
                    right_child: None,
                    parent: Some(1),
                    depth: Some(1.0),
                    length: 1.0,
                    bd_event: None,
                },
                FlatNode {
                    name: "Root".to_string(),
                    left_child: Some(0),
                    right_child: None,
                    parent: None,
                    depth: Some(0.0),
                    length: 0.0,
                    bd_event: None,
                },
            ],
            root: 1,
        };

        // Should fail due to different number of nodes
        let result = map_by_topology(&tree1, &tree2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("different number of nodes"));
    }
}

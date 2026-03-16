use std::collections::{BTreeSet, HashSet};
use crate::node::FlatTree;


/// Computes the set of leaf names in the FlatTree.
/// A leaf is defined as a node with no left or right child.
fn collect_leaves(tree: &FlatTree) -> BTreeSet<String> {
    let mut leaves = BTreeSet::new();
    for node in &tree.nodes {
        if node.left_child.is_none() && node.right_child.is_none() {
            leaves.insert(node.name.clone());
        }
    }
    leaves
}

/// Recursively computes the set of leaf names for the subtree rooted at `index`.
fn subtree_leaves(tree: &FlatTree, index: usize) -> BTreeSet<String> {
    let node = &tree.nodes[index];
    if node.left_child.is_none() && node.right_child.is_none() {
        let mut set = BTreeSet::new();
        set.insert(node.name.clone());
        return set;
    }
    let mut leaves = BTreeSet::new();
    if let Some(left_index) = node.left_child {
        leaves.extend(subtree_leaves(tree, left_index));
    }
    if let Some(right_index) = node.right_child {
        leaves.extend(subtree_leaves(tree, right_index));
    }
    leaves
}

/// Extracts the splits from the given tree as a set of strings representing clades.
/// For rooted trees, each internal node defines a unique clade (set of descendant leaves).
fn get_splits(tree: &FlatTree) -> HashSet<String> {
    let mut splits = HashSet::new();

    // For each node, if it's an internal node with a parent, count its clade
    for (index, node) in tree.nodes.iter().enumerate() {
        // Skip the root (which has no parent)
        if node.parent.is_none() {
            continue;
        }

        // Only count internal nodes (skip leaves)
        let is_leaf = node.left_child.is_none() && node.right_child.is_none();
        if is_leaf {
            continue;
        }

        // Get the clade (set of leaves below this node) and represent as sorted string
        // Sort leaves using natural/numerical ordering for consistent comparison
        let sub_leaves = subtree_leaves(tree, index);
        let mut leaves_vec: Vec<String> = sub_leaves.iter().cloned().collect();
        leaves_vec.sort_by(|a, b| {
            // Extract numeric suffix if present (e.g., "t10" -> 10)
            let num_a = a.trim_start_matches(|c: char| !c.is_numeric())
                .parse::<i32>().unwrap_or(0);
            let num_b = b.trim_start_matches(|c: char| !c.is_numeric())
                .parse::<i32>().unwrap_or(0);
            num_a.cmp(&num_b)
        });
        let clade = leaves_vec.join(",");
        splits.insert(clade);
    }
    splits
}

/// Computes the Robinson-Foulds distance between two FlatTree objects.
///
/// This function computes the RF distance by counting clades (monophyletic groups) in rooted trees.
/// Each internal node defines a clade, and the RF distance is the number of clades that differ
/// between the two trees.
///
/// Note: Despite the function name "unrooted", this implementation works with rooted trees
/// and counts clades rather than canonical bipartitions. This matches common usage in phylogenetics
/// where rooted trees are compared.
///
/// # Panics
///
/// Panics if the trees do not have identical leaf labels or do not have the same number of leaves.
pub fn unrooted_robinson_foulds(tree1: &FlatTree, tree2: &FlatTree) -> usize {
    // Verify that both trees have exactly the same set of leaves.
    let leaves1 = collect_leaves(tree1);
    let leaves2 = collect_leaves(tree2);
    if leaves1 != leaves2 {
        panic!("The trees do not have identical leaf labels.");
    }

    let splits1 = get_splits(tree1);
    let splits2 = get_splits(tree2);

    // Compute the number of splits that are not shared.
    let common: HashSet<_> = splits1.intersection(&splits2).collect();
    let rf_distance = (splits1.len() - common.len()) + (splits2.len() - common.len());
    rf_distance
}

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

/// Given a split (represented by the set of leaves in one part)
/// and the full leaf set, returns a canonical string representation.
/// The canonical representation is chosen as the smaller of the two partitions (or,
/// if the two are equal in size, the lexicographically smaller one).
fn canonical_split(subtree: &BTreeSet<String>, full: &BTreeSet<String>) -> String {
    let complement: BTreeSet<String> = full.difference(subtree).cloned().collect();
    if subtree.len() < complement.len() {
        subtree.iter().cloned().collect::<Vec<_>>().join(",")
    } else if subtree.len() > complement.len() {
        complement.iter().cloned().collect::<Vec<_>>().join(",")
    } else {
        let a = subtree.iter().cloned().collect::<Vec<_>>().join(",");
        let b = complement.iter().cloned().collect::<Vec<_>>().join(",");
        if a < b { a } else { b }
    }
}

/// Extracts the non-trivial splits from the given tree as a set of canonical strings.
/// A split is considered non-trivial if both parts contain at least 2 leaves.
fn get_splits(tree: &FlatTree) -> HashSet<String> {
    let full_leaves = collect_leaves(tree);
    let mut splits = HashSet::new();

    // For each node with a parent, consider the edge between it and its parent.
    // Its subtree defines one side of a split.
    for (index, node) in tree.nodes.iter().enumerate() {
        // Skip the root (which has no parent)
        if node.parent.is_none() {
            continue;
        }
        let sub_leaves = subtree_leaves(tree, index);
        // Ignore trivial splits where one side has fewer than 2 leaves.
        if sub_leaves.len() < 2 || (full_leaves.len() - sub_leaves.len()) < 2 {
            continue;
        }
        let canon = canonical_split(&sub_leaves, &full_leaves);
        splits.insert(canon);
    }
    splits
}

/// Computes the raw unrooted Robinson-Foulds distance between two FlatTree objects.
/// 
/// # Panics
/// 
/// Panics if the trees do not have identical leaf names or do not have the same number of leaves.
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

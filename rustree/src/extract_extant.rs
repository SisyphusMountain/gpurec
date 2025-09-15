use crate::node::{FlatTree, TraversalOrder};

/// Finds the deepest nodes (leaves) in the flat tree.
/// This function sorts the leaves by their depth in descending order
/// and returns the top `nb_leaves` deepest nodes.
pub fn find_deepest_nodes(flat_tree: &FlatTree, nb_leaves: usize) -> Vec<usize> {
    let mut leaves_with_depths: Vec<(usize, f64)> = flat_tree
        .iter(TraversalOrder::PreOrder)
        .enumerate()
        .filter(|(_, node)| node.left_child.is_none() && node.right_child.is_none())
        .map(|(i, node)| (i, node.depth.unwrap()))
        .collect();

    leaves_with_depths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    leaves_with_depths.iter().take(nb_leaves).map(|(i, _)| *i).collect()
}
// Event generation functions for birth-death trees
// I did this so we could easily generate events 
// from trees opened from Newick files, which don't have
// event annotations. If the tree is dated, we can use depths
// to discriminate between extinctions and leaves.


use crate::node::FlatTree;

use super::types::{BDEvent, TreeEvent};

/// Generate birth-death events from an existing tree.
///
/// This function respects the `bd_event` field when present (from BD simulation),
/// allowing it to distinguish extinctions from extant leaves. For trees without
/// annotations (e.g., from Newick), it infers events from tree structure.
///
/// # Arguments
/// * `tree` - A FlatTree with depths assigned
///
/// # Returns
/// A vector of TreeEvent representing the tree's evolutionary history, sorted by time
///
/// # Errors
/// Returns an error if any node doesn't have a depth assigned
///
/// # Example
/// ```ignore
/// let tree = parse_newick("((A:1,B:1):1,C:2):0;").unwrap();
/// let mut flat_tree = tree.to_flat_tree();
/// flat_tree.assign_depths();
/// let events = generate_events_from_tree(&flat_tree).unwrap();
/// ```
pub fn generate_events_from_tree(tree: &FlatTree) -> Result<Vec<TreeEvent>, String> {
    let mut events = Vec::with_capacity(tree.nodes.len());

    for (idx, node) in tree.nodes.iter().enumerate() {
        let depth = node.depth.ok_or_else(||
            format!("Node {} ('{}') has no assigned depth. Call assign_depths() before generating events.", idx, node.name)
        )?;

        // Use bd_event annotation if available, otherwise infer from structure
        let event_type = if let Some(bd_event) = node.bd_event {
            // Use the annotation (can distinguish Extinction from Leaf)
            bd_event
        } else {
            // Fall back to structure-based inference
            if node.left_child.is_none() && node.right_child.is_none() {
                BDEvent::Leaf  // Assume extant if no annotation
            } else {
                BDEvent::Speciation
            }
        };

        events.push(TreeEvent {
            time: depth,
            node_id: idx,
            event_type,
            child1: node.left_child,
            child2: node.right_child,
        });
    }

    // Sort by time (depth) for consistency
    events.sort_by(|a, b| a.time.total_cmp(&b.time));

    Ok(events)
}

/// Generate birth-death events, distinguishing extant leaves from extinctions by depth.
///
/// Childless nodes with depth close to the maximum leaf depth (within relative
/// threshold `eps`) are classified as `Leaf`; all other childless nodes are `Extinction`.
/// This ignores any existing `bd_event` annotations on nodes.
///
/// A childless node at depth `d` is classified as `Leaf` if
/// `(max_depth - d) / max_depth <= eps`, and as `Extinction` otherwise.
///
/// # Arguments
/// * `tree` - A FlatTree with depths assigned
/// * `eps` - Relative threshold for leaf classification (e.g., 0.01 for 1% tolerance)
///
/// # Returns
/// A vector of TreeEvent sorted by time
///
/// # Errors
/// Returns an error if any node doesn't have a depth assigned, or if there are no leaves
///
/// # Example
/// ```ignore
/// let tree = parse_newick("((A:1,B:1):1,(C:0.5):1.5):0;").unwrap();
/// let mut flat_tree = tree.to_flat_tree();
/// flat_tree.assign_depths();
/// let events = generate_events_with_extinction(&flat_tree, 0.01).unwrap();
/// ```
pub fn generate_events_with_extinction(tree: &FlatTree, eps: f64) -> Result<Vec<TreeEvent>, String> {
    // First pass: find max depth among childless nodes
    let mut max_depth: f64 = 0.0;
    for node in &tree.nodes {
        if node.left_child.is_none() && node.right_child.is_none() {
            let depth = node.depth.ok_or_else(||
                format!("Node '{}' has no assigned depth. Call assign_depths() before generating events.", node.name)
            )?;
            if depth > max_depth {
                max_depth = depth;
            }
        }
    }

    if max_depth == 0.0 {
        return Err("No leaves with positive depth found in the tree.".to_string());
    }

    // Second pass: classify all nodes
    let mut events = Vec::with_capacity(tree.nodes.len());

    for (idx, node) in tree.nodes.iter().enumerate() {
        let depth = node.depth.ok_or_else(||
            format!("Node {} ('{}') has no assigned depth. Call assign_depths() before generating events.", idx, node.name)
        )?;

        let event_type = if node.left_child.is_some() || node.right_child.is_some() {
            BDEvent::Speciation
        } else if (max_depth - depth) / max_depth <= eps {
            BDEvent::Leaf
        } else {
            BDEvent::Extinction
        };

        events.push(TreeEvent {
            time: depth,
            node_id: idx,
            event_type,
            child1: node.left_child,
            child2: node.right_child,
        });
    }

    events.sort_by(|a, b| a.time.total_cmp(&b.time));

    Ok(events)
}

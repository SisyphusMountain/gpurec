// Comparing two trees, to see whether they are identical.


use crate::node::Node;


/// Recursively compares two Node objects (ignoring the order of children).
pub fn compare_nodes(n1: &Node, n2: &Node, use_lengths: bool, tol: f64) -> Result<bool, String> {
    if n1.name != n2.name { return Ok(false); }

    if use_lengths {
        if (n1.length - n2.length).abs() > tol { return Ok(false); }
    }



    // Collect non-None children for each node.
    let mut children1 = Vec::new();
    if let Some(child) = &n1.left_child { children1.push(child); }
    if let Some(child) = &n1.right_child { children1.push(child); }

    let mut children2 = Vec::new();
    if let Some(child) = &n2.left_child { children2.push(child); }
    if let Some(child) = &n2.right_child { children2.push(child); }

    // Check if the number of children is the same.
    // It may be the case that one node has 2 children (internal node) and the other has 1

    if children1.len() != children2.len() {
        return Ok(false);
    }

    match children1.len() {
        0 => Ok(true),
        2 => {
            Ok((compare_nodes(children1[0], children2[0], use_lengths, tol)? && compare_nodes(children1[1], children2[1], use_lengths, tol)?)
            || (compare_nodes(children1[0], children2[1], use_lengths, tol)? && compare_nodes(children1[1], children2[0], use_lengths, tol)?))
        },
        n => Err(format!("Invalid binary tree: node '{}' has {} children (expected 0 or 2)", n1.name, n))
    }
}

/// Convenience wrapper for topology-only comparison (ignores branch lengths).
pub fn compare_nodes_topology(n1: &Node, n2: &Node) -> Result<bool, String> {
    compare_nodes(n1, n2, false, 0.0)
}

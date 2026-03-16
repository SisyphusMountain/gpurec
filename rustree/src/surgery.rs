use crate::node::FlatTree;

/// Validates ancestry relationship. Returns true if ancestor_idx is an ancestor of descendant_idx.
pub fn is_ancestor(
    flat_tree: &FlatTree,
    potential_ancestor_idx: usize,
    node_idx: usize,
) -> bool {
    // Check if indices are valid first
    if potential_ancestor_idx >= flat_tree.nodes.len() || node_idx >= flat_tree.nodes.len() {
        return false; // Or consider returning an error/panic depending on desired strictness
    }
    // A node is not its own ancestor in the context of SPR movement checks
    if potential_ancestor_idx == node_idx {
        return false;
    }

    let mut current_idx_opt = flat_tree[node_idx].parent;
    while let Some(current_idx) = current_idx_opt {
        if current_idx == potential_ancestor_idx {
            return true;
        }
        // Basic cycle check / prevent infinite loop if root's parent is not None
        if current_idx == flat_tree.root && flat_tree[flat_tree.root].parent.is_some() {
            eprintln!(
                "Warning: Root node {} has an unexpected parent during is_ancestor check.",
                flat_tree.root
            );
            return false; // Avoid potential infinite loop
        }
        // More robust cycle check (if current node points back to original node_idx)
        if current_idx == node_idx {
            eprintln!(
                "Warning: Cycle detected involving node {} during is_ancestor check.",
                node_idx
            );
            return false;
        }

        current_idx_opt = flat_tree[current_idx].parent;
    }
    false
}

/// Detaches the subtree rooted at `node_to_detach_idx` (`N`).
/// Reuses N's parent `P`, promotes N's sibling `S`.
/// Returns the index of the detached subtree's root (`P`).
/// Assumes `node_to_detach_idx` is valid and not the root (checked by `spr_topology`).
/// Uses `Result` for errors like missing sibling (violates binary assumption).
fn detach(flat_tree: &mut FlatTree, node_to_detach_idx: usize) -> Result<usize, String> {
    // 1. Get parent P. Error if non-root node has no parent (consistency issue).
    let parent_idx = flat_tree[node_to_detach_idx]
        .parent
        .ok_or_else(|| {
            format!(
                "detach consistency error: Non-root node {} has no parent.",
                node_to_detach_idx
            )
        })?;

    // 2. Get sibling S and identify if N was left/right child. Error if no sibling.
    let (sibling_idx, n_was_left) = {
        let p_node = &flat_tree[parent_idx];
        match (p_node.left_child, p_node.right_child) {
            (Some(left_idx), Some(right_idx)) if left_idx == node_to_detach_idx => {
                (right_idx, true) // N is left, S is right
            }
            (Some(left_idx), Some(right_idx)) if right_idx == node_to_detach_idx => {
                (left_idx, false) // N is right, S is left
            }
            // Cases where N isn't actually a child of P (consistency error)
            (Some(l), _) if l == node_to_detach_idx => {
                return Err(format!(
                    "detach error: Parent P ({}) of node N ({}) is missing its right child (sibling). Tree must be binary.",
                    parent_idx, node_to_detach_idx
                ));
            }
            (_, Some(r)) if r == node_to_detach_idx => {
                return Err(format!(
                    "detach error: Parent P ({}) of node N ({}) is missing its left child (sibling). Tree must be binary.",
                    parent_idx, node_to_detach_idx
                ));
            }
            _ => {
                // N is not a child of P, or P is malformed
                return Err(format!(
                    "detach consistency error: Node {} is not a child of its claimed parent {} or parent is malformed. L:{:?}, R:{:?}",
                    node_to_detach_idx, parent_idx, p_node.left_child, p_node.right_child
                ));
            }
        }
    }; // sibling_idx is now usize

    // 3. Get grandparent G (as Option<usize>)
    let grandparent_idx_opt = flat_tree[parent_idx].parent; // This is inherently Option

    // 4. Promote sibling S.
    match grandparent_idx_opt {
        Some(gp_idx) => {
            // G exists. Update G's child pointer from P to S.
            let gp_node = &mut flat_tree.nodes[gp_idx];
            // Use if-else-if, matching on Option is safer
            if gp_node.left_child == Some(parent_idx) {
                gp_node.left_child = Some(sibling_idx);
            } else if gp_node.right_child == Some(parent_idx) {
                gp_node.right_child = Some(sibling_idx);
            } else {
                // G doesn't point to P - consistency error
                return Err(format!(
                    "detach consistency error: Grandparent {} does not list parent {} as a child.",
                    gp_idx, parent_idx
                ));
            }
            // Update S's parent to G.
            flat_tree.nodes[sibling_idx].parent = Some(gp_idx);
        }
        None => {
            // G doesn't exist (P was child of root). S becomes the new root.
            flat_tree.root = sibling_idx;
            flat_tree.nodes[sibling_idx].parent = None; // Root has no parent
        }
    }

    // 5. Isolate the P -> N subtree.
    {
        let p_node = &mut flat_tree.nodes[parent_idx];
        p_node.parent = None; // P is now detached root
        if n_was_left {
            // N was left child, remove P's pointer to S (which was right)
            p_node.right_child = None;
            // p_node.left_child remains Some(node_to_detach_idx)
        } else {
            // N was right child, remove P's pointer to S (which was left)
            p_node.left_child = None;
            // p_node.right_child remains Some(node_to_detach_idx)
        }
    }
    // N's parent remains Some(parent_idx).

    Ok(parent_idx) // Return the index of P, the root of the detached subtree
}

/// Attaches a detached subtree (rooted at `P` = `detached_root_idx`)
/// above a target node `R` (`target_node_idx`).
/// Handles `Option`s gracefully. Uses `Result` for consistency errors.
fn attach(
    flat_tree: &mut FlatTree,
    detached_root_idx: usize,
    target_node_idx: usize,
) -> Result<(), String> {
    // 1. Validate P's state (detached, one child N) and find N.
    let original_donor_n_idx = {
        let p_node = flat_tree
            .nodes
            .get(detached_root_idx)
            .ok_or(format!(
                "Detached root index {} out of bounds.",
                detached_root_idx
            ))?; // Use get for bounds check

        if p_node.parent.is_some() {
            return Err(format!(
                "Attach error: Node P ({}) to attach is not detached (parent: {:?}).",
                detached_root_idx, p_node.parent
            ));
        }

        match (p_node.left_child, p_node.right_child) {
            (Some(n_idx), None) => n_idx,     // N is left child
            (None, Some(n_idx)) => n_idx,     // N is right child
            (None, None) => {
                return Err(format!(
                    "Attach error: Detached node P ({}) has no children.",
                    detached_root_idx
                ))
            }
            (Some(_), Some(_)) => {
                return Err(format!(
                    "Attach error: Detached node P ({}) should only have one child.",
                    detached_root_idx
                ))
            }
        }
    }; // original_donor_n_idx is usize

    // 2. Get original parent of R (`RP`) as Option<usize>. Check R exists.
    let recipient_parent_idx_opt = flat_tree
        .nodes
        .get(target_node_idx)
        .ok_or(format!("Target node index {} out of bounds.", target_node_idx))?
        .parent; // Get R's parent Option

    // 3. Link RP to P (or set P as new root).
    match recipient_parent_idx_opt {
        Some(rp_idx) => {
            // RP exists. Update RP's child pointer from R to P.
            let rp_node = &mut flat_tree.nodes[rp_idx]; // rp_idx must be valid if R pointed to it
            if rp_node.left_child == Some(target_node_idx) {
                rp_node.left_child = Some(detached_root_idx);
            } else if rp_node.right_child == Some(target_node_idx) {
                rp_node.right_child = Some(detached_root_idx);
            } else {
                // R points to RP, but RP doesn't point back - consistency error
                return Err(format!(
                    "Attach consistency error: Recipient's parent {} does not list recipient {} as child.",
                    rp_idx, target_node_idx
                ));
            }
            // Update P's parent to RP.
            flat_tree.nodes[detached_root_idx].parent = Some(rp_idx);
        }
        None => {
            // RP doesn't exist. R must have been the root. Make P the new root.
            if target_node_idx != flat_tree.root {
                // R had no parent but wasn't root - consistency error
                return Err(format!(
                    "Attach consistency error: Target node {} has no parent but was not the tree root ({}).",
                    target_node_idx, flat_tree.root
                ));
            }
            flat_tree.root = detached_root_idx;
            flat_tree.nodes[detached_root_idx].parent = None; // P is new root, has no parent
        }
    }

    // 4. Link P to R. Find the None slot in P and place R there.
    {
        let p_node = &mut flat_tree.nodes[detached_root_idx];
        match (p_node.left_child, p_node.right_child) {
            // Case 1: N was left, right must be None (validated in step 1)
            (Some(n_idx), None) if n_idx == original_donor_n_idx => {
                p_node.right_child = Some(target_node_idx);
            }
            // Case 2: N was right, left must be None (validated in step 1)
            (None, Some(n_idx)) if n_idx == original_donor_n_idx => {
                p_node.left_child = Some(target_node_idx);
            }
            // Error cases (should not happen if step 1 passed)
            (Some(l), Some(r)) => {
                return Err(format!(
                    "Attach consistency error: P ({}) has two children ({:?}, {:?}) before linking R.",
                    detached_root_idx, l, r
                ));
            }
            (None, None) => {
                return Err(format!(
                    "Attach consistency error: P ({}) lost its child N ({}) before linking R.",
                    detached_root_idx, original_donor_n_idx
                ));
            }
            // Case where N identified in step 1 is not actually P's child anymore
            _ => {
                return Err(format!(
                    "Attach consistency error: Child structure of P ({}) changed unexpectedly. L:{:?}, R:{:?}",
                    detached_root_idx, p_node.left_child, p_node.right_child
                ));
            }
        }
    }

    // 5. Link R to P. Update R's parent pointer.
    flat_tree.nodes[target_node_idx].parent = Some(detached_root_idx);

    Ok(())
}

/// SPR moves for ROOTED trees. Otherwise, another kind of SPR move exists.
///
/// Performs SPR using the "reuse parent" method, handling Options explicitly.
/// See `detach` and `attach` for implementation details.
/// Rules & Limitations:
/// - Moved node `N` cannot be the root.
/// - Recipient `R` can be the root (P becomes new root).
/// - Cycle Prevention: Recipient `R` cannot be an ancestor of Donor `N`.
/// - Descendants OK: Recipient `R` can be a descendant of `N` or `P`.
/// - Trivial Moves (R is sibling of N) result in no change.
pub fn spr_topology(
    flat_tree: &mut FlatTree,
    moving_node_index: usize,
    recipient_idx: usize,
) -> Result<(), String> {
    // --- Pre-checks ---
    // 1. Basic index validity (using .get().is_none() is safer than direct index)
    if flat_tree.nodes.get(moving_node_index).is_none() {
        return Err(format!("Donor index {} out of bounds.", moving_node_index));
    }
    if flat_tree.nodes.get(recipient_idx).is_none() {
        return Err(format!("Recipient index {} out of bounds.", recipient_idx));
    }

    // 2. Moving node N cannot be an ancestor of the recipient node R.
    // --- Ancestry Check (Cycle Prevention) ---
    if is_ancestor(flat_tree, moving_node_index, recipient_idx) {
        return Err(format!(
            "Invalid SPR: moving node '{moving}' is an ancestor of recipient '{recipient}'. Move would create a cycle.",
            moving = flat_tree[moving_node_index].name,
            recipient = flat_tree[recipient_idx].name,
        ));
    }

    // 3. If the moved node and recipient are the same node, do nothing
    if moving_node_index == recipient_idx {
        return Ok(()); // Valid move, no change needed
    }

    // 4. Get moving node's parent P. Error if non-root has no parent.
    let moving_node_parent_index = flat_tree[moving_node_index].parent.ok_or_else(|| {
        format!(
            "SPR consistency error: moving node {} has no parent.",
            moving_node_index
        )
    })?;

    // 5. If the recipient is the parent of the moving node, do nothing.
    if moving_node_parent_index == recipient_idx {
        return Ok(()); // Valid move, no change needed
    }

    // --- Check for Trivial Move ---
    // 6. If R is the sibling of N. Access parent P safely.
    // Note: We already know moving_node_parent_index is valid from step 4.
    let moving_node_parent = &flat_tree[moving_node_parent_index];
    let sibling_is_recipient =
        // Check if R is the right child and N is the left
        (moving_node_parent.left_child == Some(moving_node_index) && moving_node_parent.right_child == Some(recipient_idx)) ||
        // Check if R is the left child and N is the right
        (moving_node_parent.right_child == Some(moving_node_index) && moving_node_parent.left_child == Some(recipient_idx));

    if sibling_is_recipient {
        return Ok(()); // Valid move, no change needed
    }

    // --- Perform the SPR move ---
    // Use `?` to propagate potential errors from detach/attach
    let detached_parent_idx = detach(flat_tree, moving_node_index)?;
    attach(flat_tree, detached_parent_idx, recipient_idx)?;

    Ok(())
}

use crate::node::FlatTree;

/// spr moves for ROOTED trees. Otherwise, another kind of SPR move exists.
pub fn spr_topology(
    flat_tree: &mut FlatTree,
    donor: usize,
    recipient: usize,
) -> () {
    // Early return if the receiver is one of the childs of the donor: we get the same tree.
    if flat_tree[donor].left_child == Some(recipient) || flat_tree[donor].right_child == Some(recipient) {
        return;
    }

    // Get initial state
    let donor_parent = flat_tree[donor]
        .parent
        .expect("The donor node should not be the root");
    let recipient_parent = flat_tree[recipient]
        .parent
        .expect("The recipient node should not be the root");

    let recipient_sibling = if flat_tree[recipient_parent].left_child.unwrap() == recipient {
        flat_tree[recipient_parent].right_child.unwrap()
    } else {
        flat_tree[recipient_parent].left_child.unwrap()
    };


    // First case: check if the donor and receiver are siblings
    if let (Some(left), Some(right)) = (flat_tree[donor_parent].left_child, flat_tree[donor_parent].right_child) {
        if (left == donor && right == recipient) || (left == recipient && right == donor) {
            // There are no changes in the topology of the tree
            // don't do anything
            return;
        }
    }
    // Check if recipient's parent is the root
    else if flat_tree[recipient_parent].parent.is_none() {
        // The recipient's sibling becomes the new root.
        flat_tree[recipient_sibling].parent = None;
        flat_tree.root = recipient_sibling;
        // Reassign the recipient's parent: attach it under the donor's parent.
        flat_tree[recipient_parent].parent = Some(donor_parent);

        // Update the child pointer in recipient_parent: replace the recipient's parent's sister with the donor.
        if flat_tree[recipient_parent].left_child.unwrap() == recipient {
            flat_tree[recipient_parent].right_child = Some(donor);
        } else {
            flat_tree[recipient_parent].left_child = Some(donor);
        }


        // Update donor_parent so that its child pointer now points to recipient_parent.
        if flat_tree[donor_parent].left_child.unwrap() == donor {
            flat_tree[donor_parent].left_child = Some(recipient_parent);
        } else {
            flat_tree[donor_parent].right_child = Some(recipient_parent);
        }
        // Finally, attach the donor under recipient_parent.
        flat_tree[donor].parent = Some(recipient_parent);
    } else {
        // Normal case: recipient_parent is not the root.
        let recipient_grandparent = flat_tree[recipient_parent].parent;

        flat_tree[recipient_parent].parent = Some(donor_parent);
        if flat_tree[recipient_parent].left_child.unwrap() == recipient {
            flat_tree[recipient_parent].right_child = Some(donor);
        } else {
            flat_tree[recipient_parent].left_child = Some(donor);
        }


        if let Some(gp) = recipient_grandparent {
            if flat_tree[gp].left_child.unwrap() == recipient_parent {
                flat_tree[gp].left_child = Some(recipient_sibling);
            } else {
                flat_tree[gp].right_child = Some(recipient_sibling);
            }
            flat_tree[recipient_sibling].parent = Some(gp);
                }
        if flat_tree[donor_parent].left_child.unwrap() == donor {
            flat_tree[donor_parent].left_child = Some(recipient_parent);
        } else {
            flat_tree[donor_parent].right_child = Some(recipient_parent);
        }
        flat_tree[donor].parent = Some(recipient_parent);
    }
}
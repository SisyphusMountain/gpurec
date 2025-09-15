// rustree/src/node.rs

use std::ops::{Index, IndexMut};
use std::fmt;
#[derive(Clone, Debug)]
pub struct FlatNode {
    pub name: String,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub parent: Option<usize>,
    pub depth: Option<f64>,
    pub length: f64,
}
#[derive(Clone, Debug)]
/// Flat tree implementation (vector of nodes)
pub struct FlatTree {
    pub nodes: Vec<FlatNode>,
    pub root: usize,
}
#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub left_child: Option<Box<Node>>,
    pub right_child: Option<Box<Node>>,
    pub depth: Option<f64>,
    pub length: f64,
}
// Implement HasName for &Node
impl<'a> HasName for &'a Node {
    fn name(&self) -> &str {
        &self.name
    }
}

// Implement HasName for &FlatNode
impl<'a> HasName for &'a FlatNode {
    fn name(&self) -> &str {
        &self.name
    }
}

// --- Keep existing error enum ---
#[derive(Debug)]
pub enum SprError {
    NodeNotFound(String),
    InvalidDonor(String),
    InvalidRecipient(String),
    InvalidMove(String),
    TreeError(String), // For internal consistency issues
}

impl fmt::Display for SprError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SprError::NodeNotFound(name) => write!(f, "Node not found: {}", name),
            SprError::InvalidDonor(msg) => write!(f, "Invalid donor node: {}", msg),
            SprError::InvalidRecipient(msg) => write!(f, "Invalid recipient node: {}", msg),
            SprError::InvalidMove(msg) => write!(f, "Invalid SPR move: {}", msg),
            SprError::TreeError(msg) => write!(f, "Tree integrity error: {}", msg),
        }
    }
}impl std::error::Error for SprError {}


pub enum TraversalOrder {
    PreOrder,
    InOrder,
    PostOrder,
}

pub struct NodeIter<'a> {
    stack: Vec<NodeState<'a>>,
    order: TraversalOrder,
}

enum NodeState<'a> {
    Start(&'a Node),
    Left(&'a Node),
    Right(&'a Node),
    End(&'a Node),
}

impl<'a> Iterator for NodeIter<'a> {
    type Item = &'a Node;
    // Do we need mutable reference?
    fn next(&mut self) -> Option<Self::Item>{
        while let Some(state) = self.stack.pop(){
            match state {
                NodeState::Start(node) => match self.order {
                    TraversalOrder::PreOrder => {
                        // NLR
                        self.stack.push(NodeState::Right(node));
                        self.stack.push(NodeState::Left(node));
                        return Some(node);
                    }
                    TraversalOrder::InOrder => {
                        // LNR
                        self.stack.push(NodeState::Right(node));
                        self.stack.push(NodeState::End(node));
                        self.stack.push(NodeState::Left(node));
                    }
                    TraversalOrder::PostOrder => {
                        // LRN
                        self.stack.push(NodeState::End(node));
                        self.stack.push(NodeState::Right(node));
                        self.stack.push(NodeState::Left(node));
                    }
                }
                NodeState::Left(node) => {
                    if let Some(ref left) = node.left_child {
                        self.stack.push(NodeState::Start(left));
                    }
                }
                NodeState::Right(node) => {
                    if let Some(ref right) = node.right_child {
                        self.stack.push(NodeState::Start(right));
                    }
                }

                NodeState::End(node) => match self.order {
                    TraversalOrder::InOrder => {
                        return Some(node);
                    }
                    TraversalOrder::PostOrder => {
                        return Some(node);
                    }
                    TraversalOrder::PreOrder => {} // No return there
                }
            }
        }
    return None;
}
}
impl Node {
    pub fn iter(&self, order: TraversalOrder) -> NodeIter {
        NodeIter {
            stack: vec![NodeState::Start(self)],
            order,
        }
    }
    /// Converts a recursive `Node` structure into a `FlatTree`.
    ///
    /// # Returns
    /// A `FlatTree` representation of the `Node`.
    pub fn to_flat_tree(&self) -> FlatTree {
        let mut flat_nodes = Vec::new();
        let root_index = self.node_to_flat_internal(&mut flat_nodes, None);
        FlatTree {
            nodes: flat_nodes,
            root: root_index,
        }
    }

    /// Internal helper method for converting a `Node` to flat nodes.
    ///
    /// # Arguments
    /// * `flat_nodes` - The vector to store `FlatNode` instances.
    /// * `parent_index` - The index of the parent node (if any).
    ///
    /// # Returns
    /// The index of the current node in `flat_nodes`.
    fn node_to_flat_internal(&self, flat_nodes: &mut Vec<FlatNode>, parent_index: Option<usize>) -> usize {
        let index = flat_nodes.len();
        flat_nodes.push(FlatNode {
            name: self.name.clone(),
            left_child: None,  // Will be updated after recursive calls
            right_child: None, // Will be updated after recursive calls
            parent: parent_index,
            depth: self.depth,
            length: self.length,
        });

        // Process left child
        if let Some(ref left_child) = self.left_child {
            let left_index = left_child.node_to_flat_internal(flat_nodes, Some(index));
            flat_nodes[index].left_child = Some(left_index);
        }

        // Process right child
        if let Some(ref right_child) = self.right_child {
            let right_index = right_child.node_to_flat_internal(flat_nodes, Some(index));
            flat_nodes[index].right_child = Some(right_index);
        }

        index
    }


}
pub struct FlatTreeIter<'a> {
    tree: &'a FlatTree,
    stack: Vec<FlatTreeState>,
    order: TraversalOrder,
}

enum FlatTreeState {
    Start(usize),
    Left(usize),
    Right(usize),
    End(usize),
}

impl<'a> Iterator for FlatTreeIter<'a> {
    type Item = &'a FlatNode;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(state) = self.stack.pop() {
            match state {
                FlatTreeState::Start(index) => {
                    let node = &self.tree.nodes[index];
                    match self.order {
                        TraversalOrder::PreOrder => {
                            // PreOrder: Visit node, left, right
                            self.stack.push(FlatTreeState::Right(index));
                            self.stack.push(FlatTreeState::Left(index));
                            return Some(node);
                        }
                        TraversalOrder::InOrder => {
                            // InOrder: Left, visit node, right
                            self.stack.push(FlatTreeState::Right(index));
                            self.stack.push(FlatTreeState::End(index));
                            self.stack.push(FlatTreeState::Left(index));
                        }
                        TraversalOrder::PostOrder => {
                            // PostOrder: Left, right, visit node
                            self.stack.push(FlatTreeState::End(index));
                            self.stack.push(FlatTreeState::Right(index));
                            self.stack.push(FlatTreeState::Left(index));
                        }
                    }
                }
                FlatTreeState::Left(index) => {
                    if let Some(left_index) = self.tree.nodes[index].left_child {
                        self.stack.push(FlatTreeState::Start(left_index));
                    }
                }
                FlatTreeState::Right(index) => {
                    if let Some(right_index) = self.tree.nodes[index].right_child {
                        self.stack.push(FlatTreeState::Start(right_index));
                    }
                }
                FlatTreeState::End(index) => {
                    let node = &self.tree.nodes[index];
                    match self.order {
                        TraversalOrder::InOrder | TraversalOrder::PostOrder => {
                            return Some(node);
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
}

impl FlatTree {
    pub fn iter(&self, order: TraversalOrder) -> FlatTreeIter {
        FlatTreeIter {
            tree: self,
            stack: vec![FlatTreeState::Start(self.root)],
            order,
        }
    }
    /// Converts a `FlatTree` into a recursive `Node` structure.
    ///
    /// # Returns
    /// A `Node` representation of the `FlatTree`.
    pub fn to_node(&self) -> Node {
        self.flat_to_node_internal(self.root)
    }
    pub fn find_node_index(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == name)
    }
    /// Internal helper method for converting flat nodes to a `Node`.
    ///
    /// # Arguments
    /// * `index` - The index of the current node in `self.nodes`.
    ///
    /// # Returns
    /// A `Node` representing the current node.
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
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_ancestor(&self, potential_ancestor_idx: usize, node_idx: usize) -> bool {
        // Check if indices are valid first
        if potential_ancestor_idx >= self.nodes.len() || node_idx >= self.nodes.len() {
            return false; // Or consider returning an error/panic depending on desired strictness
        }
        // A node is not its own ancestor in the context of SPR movement checks
        if potential_ancestor_idx == node_idx {
            return false;
        }

        let mut current_idx_opt = self.nodes[node_idx].parent;
        while let Some(current_idx) = current_idx_opt {
            if current_idx == potential_ancestor_idx {
                return true;
            }
            // Basic cycle check / prevent infinite loop if root's parent is not None
             if current_idx == self.root && self.nodes[self.root].parent.is_some() {
                 eprintln!("Warning: Root node {} has an unexpected parent during is_ancestor check.", self.root);
                 return false; // Avoid potential infinite loop
             }
             // More robust cycle check (if current node points back to original node_idx)
             if current_idx == node_idx {
                eprintln!("Warning: Cycle detected involving node {} during is_ancestor check.", node_idx);
                return false;
            }

            current_idx_opt = self.nodes[current_idx].parent;
        }
        false
    }
        /// Detaches the subtree rooted at `node_to_detach_idx` (`N`).
    /// Reuses N's parent `P`, promotes N's sibling `S`.
    /// Returns the index of the detached subtree's root (`P`).
    /// Assumes `node_to_detach_idx` is valid and not the root (checked by `spr`).
    /// Uses `Result` for errors like missing sibling (violates binary assumption).
    fn detach(&mut self, node_to_detach_idx: usize) -> Result<usize, SprError> {
        // 1. Get parent P. Error if non-root node has no parent (consistency issue).
        let parent_idx = self.nodes[node_to_detach_idx].parent
            .ok_or_else(|| SprError::TreeError(format!(
                "detach consistency error: Non-root node {} has no parent.", node_to_detach_idx
            )))?;

        // 2. Get sibling S and identify if N was left/right child. Error if no sibling.
        let (sibling_idx, n_was_left) = {
            let p_node = &self.nodes[parent_idx];
            match (p_node.left_child, p_node.right_child) {
                (Some(left_idx), Some(right_idx)) if left_idx == node_to_detach_idx => {
                    (right_idx, true) // N is left, S is right
                }
                (Some(left_idx), Some(right_idx)) if right_idx == node_to_detach_idx => {
                    (left_idx, false) // N is right, S is left
                }
                 // Cases where N isn't actually a child of P (consistency error)
                (Some(l), _) if l == node_to_detach_idx => {
                    return Err(SprError::TreeError(format!(
                        "detach error: Parent P ({}) of node N ({}) is missing its right child (sibling). Tree must be binary.", parent_idx, node_to_detach_idx
                    )));
                 }
                 (_, Some(r)) if r == node_to_detach_idx => {
                      return Err(SprError::TreeError(format!(
                        "detach error: Parent P ({}) of node N ({}) is missing its left child (sibling). Tree must be binary.", parent_idx, node_to_detach_idx
                     )));
                 }
                _ => {
                    // N is not a child of P, or P is malformed
                    return Err(SprError::TreeError(format!(
                        "detach consistency error: Node {} is not a child of its claimed parent {} or parent is malformed. L:{:?}, R:{:?}",
                        node_to_detach_idx, parent_idx, p_node.left_child, p_node.right_child
                    )));
                }
            }
        }; // sibling_idx is now usize

        // 3. Get grandparent G (as Option<usize>)
        let grandparent_idx_opt = self.nodes[parent_idx].parent; // This is inherently Option

        // 4. Promote sibling S.
        match grandparent_idx_opt {
            Some(gp_idx) => {
                // G exists. Update G's child pointer from P to S.
                let gp_node = &mut self.nodes[gp_idx];
                 // Use if-else-if, matching on Option is safer
                 if gp_node.left_child == Some(parent_idx) {
                     gp_node.left_child = Some(sibling_idx);
                 } else if gp_node.right_child == Some(parent_idx) {
                      gp_node.right_child = Some(sibling_idx);
                 } else {
                     // G doesn't point to P - consistency error
                     return Err(SprError::TreeError(format!(
                         "detach consistency error: Grandparent {} does not list parent {} as a child.",
                         gp_idx, parent_idx
                     )));
                 }
                // Update S's parent to G.
                self.nodes[sibling_idx].parent = Some(gp_idx);
            }
            None => {
                // G doesn't exist (P was child of root). S becomes the new root.
                self.root = sibling_idx;
                self.nodes[sibling_idx].parent = None; // Root has no parent
            }
        }

        // 5. Isolate the P -> N subtree.
        {
            let p_node = &mut self.nodes[parent_idx];
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
    fn attach(&mut self, detached_root_idx: usize, target_node_idx: usize) -> Result<(), SprError> {
        // 1. Validate P's state (detached, one child N) and find N.
        let original_donor_n_idx = {
            let p_node = self.nodes.get(detached_root_idx)
                .ok_or(SprError::InvalidDonor(format!("Detached root index {} out of bounds.", detached_root_idx)))?; // Use get for bounds check

            if p_node.parent.is_some() {
                return Err(SprError::TreeError(format!("Attach error: Node P ({}) to attach is not detached (parent: {:?}).", detached_root_idx, p_node.parent)));
            }

            match (p_node.left_child, p_node.right_child) {
                (Some(n_idx), None) => n_idx, // N is left child
                (None, Some(n_idx)) => n_idx, // N is right child
                (None, None) => return Err(SprError::TreeError(format!("Attach error: Detached node P ({}) has no children.", detached_root_idx))),
                (Some(_), Some(_)) => return Err(SprError::TreeError(format!("Attach error: Detached node P ({}) should only have one child.", detached_root_idx))),
            }
        }; // original_donor_n_idx is usize

        // 2. Get original parent of R (`RP`) as Option<usize>. Check R exists.
        let recipient_parent_idx_opt = self.nodes.get(target_node_idx)
            .ok_or(SprError::InvalidRecipient(format!("Target node index {} out of bounds.", target_node_idx)))?
            .parent; // Get R's parent Option

        // 3. Link RP to P (or set P as new root).
        match recipient_parent_idx_opt {
            Some(rp_idx) => {
                // RP exists. Update RP's child pointer from R to P.
                let rp_node = &mut self.nodes[rp_idx]; // rp_idx must be valid if R pointed to it
                if rp_node.left_child == Some(target_node_idx) {
                    rp_node.left_child = Some(detached_root_idx);
                } else if rp_node.right_child == Some(target_node_idx) {
                    rp_node.right_child = Some(detached_root_idx);
                } else {
                    // R points to RP, but RP doesn't point back - consistency error
                    return Err(SprError::TreeError(format!(
                        "Attach consistency error: Recipient's parent {} does not list recipient {} as child.", rp_idx, target_node_idx
                    )));
                }
                // Update P's parent to RP.
                self.nodes[detached_root_idx].parent = Some(rp_idx);
            }
            None => {
                // RP doesn't exist. R must have been the root. Make P the new root.
                if target_node_idx != self.root {
                    // R had no parent but wasn't root - consistency error
                    return Err(SprError::TreeError(format!(
                        "Attach consistency error: Target node {} has no parent but was not the tree root ({}).", target_node_idx, self.root
                    )));
                }
                self.root = detached_root_idx;
                self.nodes[detached_root_idx].parent = None; // P is new root, has no parent
            }
        }

        // 4. Link P to R. Find the None slot in P and place R there.
        {
            let p_node = &mut self.nodes[detached_root_idx];
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
                     return Err(SprError::TreeError(format!(
                         "Attach consistency error: P ({}) has two children ({:?}, {:?}) before linking R.", detached_root_idx, l, r
                     )));
                 }
                 (None, None) => {
                      return Err(SprError::TreeError(format!(
                          "Attach consistency error: P ({}) lost its child N ({}) before linking R.", detached_root_idx, original_donor_n_idx
                      )));
                 }
                 // Case where N identified in step 1 is not actually P's child anymore
                 _ => {
                      return Err(SprError::TreeError(format!(
                           "Attach consistency error: Child structure of P ({}) changed unexpectedly. L:{:?}, R:{:?}", detached_root_idx, p_node.left_child, p_node.right_child
                      )));
                 }
             }
        }

        // 5. Link R to P. Update R's parent pointer.
        self.nodes[target_node_idx].parent = Some(detached_root_idx);

        Ok(())
    }


    /// Performs SPR using the "reuse parent" method, handling Options explicitly.
    /// See `detach` and `attach` for implementation details.
    /// Rules & Limitations:
    /// - Moved node `N` cannot be the root.
    /// - Recipient `R` can be the root (P becomes new root).
    /// - Cycle Prevention: Recipient `R` cannot be an ancestor of Donor `N`.
    /// - Descendants OK: Recipient `R` can be a descendant of `N` or `P`.
    /// - Trivial Moves (R is sibling of N) result in no change.
    pub fn spr_topology(
        &mut self,
        moving_node_index: usize,
        recipient_idx: usize
    ) -> Result<(), SprError> {

        // --- Pre-checks ---
        // 1. Basic index validity (using .get().is_none() is safer than direct index)
        if self.nodes.get(moving_node_index).is_none() {
            return Err(SprError::InvalidDonor(format!("Donor index {} out of bounds.", moving_node_index)));
        }
        if self.nodes.get(recipient_idx).is_none() {
            return Err(SprError::InvalidRecipient(format!("Recipient index {} out of bounds.", recipient_idx)));
        }

        // 2. Moving node N cannot be an ancestor of the recipient node R.
        // --- Ancestry Check (Cycle Prevention) ---
        if self.is_ancestor(moving_node_index, recipient_idx) {
            return Err(SprError::InvalidMove(format!(
                "Moving node R ({}) cannot be an ancestor of the moving node N ({}). Move would create a cycle.",
                self.nodes[recipient_idx].name, self.nodes[moving_node_index].name
            )));
       }

        // 3. If the moved node and recipient are the same node, do nothing
        if moving_node_index == recipient_idx {
            return Ok(()); // Valid move, no change needed
        }

        // 4. Get moving node's parent P. Error if non-root has no parent.
        let moving_node_parent_index = self.nodes[moving_node_index].parent
            .ok_or_else(|| SprError::TreeError(format!(
                "SPR consistency error: moving node {} has no parent.", moving_node_index
            )))?;

        // 5. If the recipient is the parent of the moving node, do nothing.
        if moving_node_parent_index == recipient_idx {
            return Ok(()); // Valid move, no change needed
        }



        // --- Check for Trivial Move ---
        // 6. If R is the sibling of N. Access parent P safely.
        // Note: We already know moving_node_parent_index is valid from step 4.
        let moving_node_parent = &self.nodes[moving_node_parent_index];
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
        let detached_parent_idx = self.detach(moving_node_index)?;
        self.attach(detached_parent_idx, recipient_idx)?;

        Ok(())
    }



}
impl Index<usize> for FlatTree {
    type Output = FlatNode;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}
impl IndexMut<usize> for FlatTree {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

pub trait HasName {
    fn name(&self) -> &str;
}

impl HasName for Node {
    fn name(&self) -> &str {
        &self.name
    }
}

impl HasName for FlatNode {
    fn name(&self) -> &str {
        &self.name
    }
}


/// Converts an arborescent tree (a tree where each node owns its children)
/// into a flat structure (a vector of FlatNodes).
///
/// # Arguments
/// * `node` - The root node of the tree.
/// * `flat_tree` - The vector to be filled with flat nodes.
/// * `parent` - The parent index (if any).
///
/// # Returns
/// The index of the node in the flat tree.
pub fn node_to_flat(node: &Node, flat_tree: &mut Vec<FlatNode>, parent: Option<usize>) -> usize {
    let index = flat_tree.len();
    flat_tree.push(FlatNode {
        name: node.name.clone(),
        left_child: None,  // Will fill this in a moment
        right_child: None, // Will fill this in a moment
        parent,
        depth: node.depth,
        length: node.length,
    });

    if let Some(left) = &node.left_child {
        let left_index = node_to_flat(left, flat_tree, Some(index));
        flat_tree[index].left_child = Some(left_index);
    }

    if let Some(right) = &node.right_child {
        let right_index = node_to_flat(right, flat_tree, Some(index));
        flat_tree[index].right_child = Some(right_index);
    }

    index
}

/// Converts a flat tree into a nested `Node` tree structure.
///
/// # Arguments
/// * `flat_tree` - The vector representing the flat tree.
/// * `index` - The index of the node in the flat tree.
/// * `parent_index` - The index of the parent node (if any).
///
/// # Returns
/// The corresponding `Node` tree.
pub fn flat_to_node(
    flat_tree: &[FlatNode],
    index: usize,
) -> Option<Node> {
    let flat_node = &flat_tree[index];
    let left_child = flat_node
        .left_child
        .and_then(|i| flat_to_node(flat_tree, i).map(Box::new));
    let right_child = flat_node
        .right_child
        .and_then(|i| flat_to_node(flat_tree, i,).map(Box::new));

    Some(Node {
        name: flat_node.name.clone(),
        left_child,
        right_child,
        depth: flat_node.depth,
        length: flat_node.length,
    })
}


/// Recursively compares two Node objects (ignoring the order of children).
pub fn compare_nodes(n1: &Node, n2: &Node) -> bool {
    if n1.name != n2.name { return false; }
    if (n1.length - n2.length).abs() > 1e-3 { return false; }
    if n1.depth != n2.depth { return false; }

    // Collect non-None children for each node.
    let mut children1 = Vec::new();
    if let Some(child) = &n1.left_child { children1.push(child); }
    if let Some(child) = &n1.right_child { children1.push(child); }

    let mut children2 = Vec::new();
    if let Some(child) = &n2.left_child { children2.push(child); }
    if let Some(child) = &n2.right_child { children2.push(child); }

    if children1.len() != children2.len() {
        return false;
    }

    match children1.len() {
        0 => true,
        1 => compare_nodes(children1[0], children2[0]),
        2 => {
            (compare_nodes(children1[0], children2[0]) && compare_nodes(children1[1], children2[1]))
            || (compare_nodes(children1[0], children2[1]) && compare_nodes(children1[1], children2[0]))
        },
        _ => true // for trees with more than 2 children (unexpected)
    }
}

pub fn compare_nodes_topology(n1: &Node, n2: &Node) -> bool {
    if n1.name != n2.name { return false; }

    // Collect non-None children for each node.
    let mut children1 = Vec::new();
    if let Some(child) = &n1.left_child { children1.push(child); }
    if let Some(child) = &n1.right_child { children1.push(child); }

    let mut children2 = Vec::new();
    if let Some(child) = &n2.left_child { children2.push(child); }
    if let Some(child) = &n2.right_child { children2.push(child); }

    if children1.len() != children2.len() {
        return false;
    }

    match children1.len() {
        0 => true,
        1 => compare_nodes_topology(children1[0], children2[0]),
        2 => {
            (compare_nodes_topology(children1[0], children2[0]) && compare_nodes_topology(children1[1], children2[1]))
            || (compare_nodes_topology(children1[0], children2[1]) && compare_nodes_topology(children1[1], children2[0]))
        },
        _ => {
            // For more than 2 children, compare without relying on order.
            let mut matched = vec![false; children2.len()];
            for child1 in &children1 {
                let mut found = false;
                for (i, child2) in children2.iter().enumerate() {
                    if !matched[i] && compare_nodes_topology(child1, child2) {
                        matched[i] = true;
                        found = true;
                        break;
                    }
                }
                if !found { return false; }
            }
            true
        }
    }
}

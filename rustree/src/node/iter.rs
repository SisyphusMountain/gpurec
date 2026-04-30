//! Tree iteration utilities for recursive and flat trees.

use super::{Node, FlatNode, FlatTree, TraversalOrder};

// ============================================================================
// NodeIter - Iterator for recursive Node trees
// ============================================================================

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

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(state) = self.stack.pop() {
            match state {
                NodeState::Start(node) => match self.order {
                    TraversalOrder::PreOrder => {
                        self.stack.push(NodeState::Right(node));
                        self.stack.push(NodeState::Left(node));
                        return Some(node);
                    }
                    TraversalOrder::InOrder => {
                        self.stack.push(NodeState::Right(node));
                        self.stack.push(NodeState::End(node));
                        self.stack.push(NodeState::Left(node));
                    }
                    TraversalOrder::PostOrder => {
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
                    TraversalOrder::InOrder | TraversalOrder::PostOrder => {
                        return Some(node);
                    }
                    TraversalOrder::PreOrder => {}
                }
            }
        }
        None
    }
}

impl Node {
    pub fn iter(&self, order: TraversalOrder) -> NodeIter<'_> {
        NodeIter {
            stack: vec![NodeState::Start(self)],
            order,
        }
    }
}

// ============================================================================
// FlatTreeIter - Iterator for FlatTree
// ============================================================================

pub struct FlatTreeIter<'a> {
    tree: &'a FlatTree,
    stack: Vec<FlatTreeState>,
    order: TraversalOrder,
}

pub enum FlatTreeState {
    Start(usize),
    Left(usize),
    Right(usize),
    End(usize),
}

/// Advances the traversal state machine by one step, returning the next
/// node index if one should be emitted for the given traversal order.
///
/// Shared by both `FlatTreeIter` (yields `&FlatNode`) and
/// `FlatTreeIndexIter` (yields `usize`).
pub fn advance_flat_tree(
    tree: &FlatTree,
    stack: &mut Vec<FlatTreeState>,
    order: &TraversalOrder,
) -> Option<usize> {
    while let Some(state) = stack.pop() {
        match state {
            FlatTreeState::Start(index) => match order {
                TraversalOrder::PreOrder => {
                    stack.push(FlatTreeState::Right(index));
                    stack.push(FlatTreeState::Left(index));
                    return Some(index);
                }
                TraversalOrder::InOrder => {
                    stack.push(FlatTreeState::Right(index));
                    stack.push(FlatTreeState::End(index));
                    stack.push(FlatTreeState::Left(index));
                }
                TraversalOrder::PostOrder => {
                    stack.push(FlatTreeState::End(index));
                    stack.push(FlatTreeState::Right(index));
                    stack.push(FlatTreeState::Left(index));
                }
            }
            FlatTreeState::Left(index) => {
                if let Some(left_index) = tree.nodes[index].left_child {
                    stack.push(FlatTreeState::Start(left_index));
                }
            }
            FlatTreeState::Right(index) => {
                if let Some(right_index) = tree.nodes[index].right_child {
                    stack.push(FlatTreeState::Start(right_index));
                }
            }
            FlatTreeState::End(index) => match order {
                TraversalOrder::InOrder | TraversalOrder::PostOrder => {
                    return Some(index);
                }
                TraversalOrder::PreOrder => {}
            }
        }
    }
    None
}

impl<'a> Iterator for FlatTreeIter<'a> {
    type Item = &'a FlatNode;

    fn next(&mut self) -> Option<Self::Item> {
        advance_flat_tree(self.tree, &mut self.stack, &self.order)
            .map(|idx| &self.tree.nodes[idx])
    }
}

// ============================================================================
// FlatTreeIndexIter - Index iterator for FlatTree
// ============================================================================

pub struct FlatTreeIndexIter<'a> {
    tree: &'a FlatTree,
    stack: Vec<FlatTreeState>,
    order: TraversalOrder,
}

impl<'a> Iterator for FlatTreeIndexIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        advance_flat_tree(self.tree, &mut self.stack, &self.order)
    }
}

impl FlatTree {
    pub fn iter(&self, order: TraversalOrder) -> FlatTreeIter<'_> {
        FlatTreeIter {
            tree: self,
            stack: vec![FlatTreeState::Start(self.root)],
            order,
        }
    }

    /// Iterate over node indices in the given traversal order.
    pub fn iter_indices(&self, order: TraversalOrder) -> FlatTreeIndexIter<'_> {
        FlatTreeIndexIter {
            tree: self,
            stack: vec![FlatTreeState::Start(self.root)],
            order,
        }
    }

    /// Returns node indices in postorder (children before parents).
    pub fn postorder_indices(&self) -> Vec<usize> {
        self.iter_indices(TraversalOrder::PostOrder).collect()
    }
}

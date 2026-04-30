//! Traits and operator implementations for tree types.

use std::ops::{Index, IndexMut};
use super::{Node, FlatNode, FlatTree};

/// Trait for types that have a name field.
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

impl HasName for &Node {
    fn name(&self) -> &str {
        &self.name
    }
}

impl HasName for &FlatNode {
    fn name(&self) -> &str {
        &self.name
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

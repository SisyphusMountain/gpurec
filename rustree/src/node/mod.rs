//! Core tree data structures for phylogenetic trees.
//!
//! This module provides two tree representations:
//! - `Node`: Recursive tree with Box-based children (heap-allocated)
//! - `FlatTree`/`FlatNode`: Vector-based tree with index references
//!
//! Both representations support traversal (pre-order, in-order, post-order)
//! and can be converted between each other.

mod iter;
mod traits;
mod conversion;
pub mod rectree;
pub mod gene_forest;

// Re-export iterator types
pub use iter::{NodeIter, FlatTreeIter, FlatTreeIndexIter, advance_flat_tree, FlatTreeState};

// Re-export traits
pub use traits::HasName;

// Re-export conversion functions
pub use conversion::{map_by_topology, rename_gene_tree};

// Re-export reconciliation types
pub use rectree::{Event, RecTree};
pub use gene_forest::{GeneForest, remap_gene_tree_indices};
pub use crate::io::rectree_csv::RecTreeColumns;

// Re-export XML parsing functions (now in io module)
pub use crate::io::recphyloxml::{parse_recphyloxml, parse_recphyloxml_file, parse_gene_tree_only, parse_gene_tree_only_file};

// ============================================================================
// Core Data Structures
// ============================================================================

/// A node in a flat (vector-based) tree representation.
#[derive(Clone, Debug)]
pub struct FlatNode {
    pub name: String,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub parent: Option<usize>,
    pub depth: Option<f64>,
    pub length: f64,
    /// Birth-death event type for this node (only set for species trees from BD simulation)
    pub bd_event: Option<crate::bd::BDEvent>,
}

/// A flat tree representation using a vector of nodes.
#[derive(Clone, Debug)]
pub struct FlatTree {
    pub nodes: Vec<FlatNode>,
    pub root: usize,
}

/// A node in a recursive (Box-based) tree representation.
#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub left_child: Option<Box<Node>>,
    pub right_child: Option<Box<Node>>,
    pub depth: Option<f64>,
    pub length: f64,
}

/// Traversal order for tree iteration.
pub enum TraversalOrder {
    PreOrder,
    InOrder,
    PostOrder,
}

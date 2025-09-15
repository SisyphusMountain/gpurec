// rustree/src/lib.rs
#[macro_use]
extern crate pest_derive;


// Expose the modules
pub mod newick; // Contains newick parsing logic (newick.rs, newick.pest, etc.)
pub mod node;   // Contains the Node, FlatNode, and traversal logic
pub mod metric_functions;  // added metric functions
pub mod sampling; // added sampling functions
pub mod extract_extant; // re-export the new module
pub mod surgery; // added surgery functions
pub mod comparison; // added comparison functions
pub mod debug; // added debug functions
pub mod robinson_foulds; // added robinson foulds functions

// Re-export key functions for easier access:
pub use newick::newick::{newick_to_tree, handle_pair, node_to_newick};
pub use node::{Node, FlatNode, FlatTree, TraversalOrder, node_to_flat, flat_to_node, SprError};
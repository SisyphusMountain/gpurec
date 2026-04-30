//! Birth-death process simulation for phylogenetic trees.
//!
//! Provides forward- and backward-time birth-death tree generation,
//! event extraction (speciation, extinction), and CSV export of
//! the resulting event sequences.

mod types;
mod simulation;
mod events;

// Re-export public API
pub use types::{BDEvent, TreeEvent};
pub use simulation::simulate_bd_tree_bwd;
pub use events::{generate_events_from_tree, generate_events_with_extinction};

// Re-export from io module for backward compatibility
pub use crate::io::save_bd_events_to_csv as save_events_to_csv;

// rustree/src/lib.rs
#[macro_use]
extern crate pest_derive;

// Core modules
pub mod node;   // Tree data structures and traversal
pub mod newick; // Newick parsing
pub mod io;     // I/O utilities (CSV export)

// Simulation modules
pub mod simulation;
pub use simulation::bd;
pub use simulation::dtl;

// Tree operations
pub mod sampling;
pub mod surgery;
pub mod comparison;
pub mod metric_functions;
pub mod robinson_foulds;
pub mod debug;

// Analysis
pub mod induced_transfers;

// External tool integration
pub mod external;
pub use external::alerax;

// Language bindings (feature-gated)
#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "r")]
pub mod r;

// Re-export key types for easier access
pub use newick::newick::parse_newick;
pub use node::{Node, FlatNode, FlatTree, TraversalOrder, map_by_topology, rename_gene_tree};
pub use node::{RecTree, Event, GeneForest, parse_recphyloxml, parse_recphyloxml_file};
pub use metric_functions::{DistanceType, PairwiseDistance, LcaTable};
pub use sampling::{
    extract_induced_subtree, extract_induced_subtree_by_names,
    compute_lca, build_leaf_pair_lca_map, build_sampled_to_original_mapping,
    get_descendant_leaf_names, find_all_leaf_indices, find_leaf_indices_by_names,
};
pub use alerax::{
    run_alerax, reconcile_forest, AleRaxConfig, AleRaxFamilyResult, AleRaxForestResult,
    GeneFamily, ModelType, validate_inputs, SpeciesEventRow, TransferRow,
};
pub use robinson_foulds::unrooted_robinson_foulds;
//! Reconciled tree structure for DTL (Duplication-Transfer-Loss) model.

use super::{FlatNode, FlatTree};
use std::sync::Arc;
use crate::dtl::DTLEvent;

/// Events that can occur during DTL reconciliation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    /// Speciation event - gene tree lineage follows species tree split
    Speciation,
    /// Duplication event - gene duplicates within a species
    Duplication,
    /// Horizontal gene transfer event
    Transfer,
    /// Gene loss event
    Loss,
    /// Leaf node (extant gene)
    Leaf,
}

/// Reconciled tree structure for DTL model.
///
/// Represents a gene tree reconciled with a species tree, storing the mapping
/// between gene tree nodes and species tree nodes, as well as the evolutionary
/// events (Duplication, Transfer, Loss) at each node.
///
/// The species tree is shared via `Arc<FlatTree>`, allowing multiple gene trees
/// to reference the same species tree without cloning.
#[derive(Clone, Debug)]
pub struct RecTree {
    /// The species tree (shared via Arc)
    pub species_tree: Arc<FlatTree>,
    /// The gene tree
    pub gene_tree: FlatTree,
    /// Maps each gene tree node index to its corresponding species tree node index.
    /// `None` means the mapping is unknown (e.g., after pruning/sampling).
    pub node_mapping: Vec<Option<usize>>,
    /// Maps each gene tree node index to its evolutionary event
    pub event_mapping: Vec<Event>,
    /// Detailed DTL events from simulation (if available).
    /// `None` for trees parsed from files or after sampling operations.
    pub dtl_events: Option<Vec<DTLEvent>>,
}

impl RecTree {
    /// Creates a new RecTree with a shared species tree.
    ///
    /// # Panics
    /// Panics if the mappings don't match the gene tree size.
    pub fn new(
        species_tree: Arc<FlatTree>,
        gene_tree: FlatTree,
        node_mapping: Vec<Option<usize>>,
        event_mapping: Vec<Event>,
    ) -> Self {
        assert_eq!(
            gene_tree.nodes.len(),
            node_mapping.len(),
            "node_mapping must have same length as gene_tree nodes"
        );
        assert_eq!(
            gene_tree.nodes.len(),
            event_mapping.len(),
            "event_mapping must have same length as gene_tree nodes"
        );

        RecTree {
            species_tree,
            gene_tree,
            node_mapping,
            event_mapping,
            dtl_events: None,
        }
    }

    /// Creates a new RecTree with DTL events.
    ///
    /// # Panics
    /// Panics if the mappings don't match the gene tree size.
    pub fn with_dtl_events(
        species_tree: Arc<FlatTree>,
        gene_tree: FlatTree,
        node_mapping: Vec<Option<usize>>,
        event_mapping: Vec<Event>,
        dtl_events: Vec<DTLEvent>,
    ) -> Self {
        assert_eq!(
            gene_tree.nodes.len(),
            node_mapping.len(),
            "node_mapping must have same length as gene_tree nodes"
        );
        assert_eq!(
            gene_tree.nodes.len(),
            event_mapping.len(),
            "event_mapping must have same length as gene_tree nodes"
        );

        RecTree {
            species_tree,
            gene_tree,
            node_mapping,
            event_mapping,
            dtl_events: Some(dtl_events),
        }
    }

    /// Creates a new RecTree, wrapping the species tree in an Arc.
    ///
    /// Convenience constructor for when you have an owned FlatTree.
    pub fn new_owned(
        species_tree: FlatTree,
        gene_tree: FlatTree,
        node_mapping: Vec<Option<usize>>,
        event_mapping: Vec<Event>,
    ) -> Self {
        Self::new(Arc::new(species_tree), gene_tree, node_mapping, event_mapping)
    }

    /// Creates a new RecTree with DTL events, wrapping the species tree in an Arc.
    pub fn new_owned_with_events(
        species_tree: FlatTree,
        gene_tree: FlatTree,
        node_mapping: Vec<Option<usize>>,
        event_mapping: Vec<Event>,
        dtl_events: Vec<DTLEvent>,
    ) -> Self {
        Self::with_dtl_events(Arc::new(species_tree), gene_tree, node_mapping, event_mapping, dtl_events)
    }

    /// Gets the species tree node index for a given gene tree node.
    /// Returns `None` if the mapping is unknown (e.g., after pruning).
    pub fn species_node_for(&self, gene_node_idx: usize) -> Option<usize> {
        self.node_mapping[gene_node_idx]
    }

    /// Gets the event type for a given gene tree node.
    pub fn event_for(&self, gene_node_idx: usize) -> &Event {
        &self.event_mapping[gene_node_idx]
    }

    /// Gets the gene tree node and its corresponding species node and event.
    pub fn get_full_info(&self, gene_node_idx: usize) -> (&FlatNode, Option<usize>, &Event) {
        (
            &self.gene_tree.nodes[gene_node_idx],
            self.node_mapping[gene_node_idx],
            &self.event_mapping[gene_node_idx],
        )
    }

}

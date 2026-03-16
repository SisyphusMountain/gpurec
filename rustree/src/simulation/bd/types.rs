// birth-death process types

use crate::node::FlatTree;

/// Event types in a birth-death process
// This type only refers to the event of a given node.
// TreeEvent includes all the other information about a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BDEvent {
    /// Speciation event - lineage splits into two
    Speciation,
    /// Extinction event - lineage goes extinct
    Extinction,
    /// Leaf node - extant species at present time
    Leaf,
}

impl BDEvent {
    /// Convert to string representation for display/CSV
    pub fn as_str(&self) -> &'static str {
        match self {
            BDEvent::Speciation => "Speciation",
            BDEvent::Extinction => "Extinction",
            BDEvent::Leaf => "Leaf",
        }
    }

    /// Parse from string representation
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Speciation" => Some(BDEvent::Speciation),
            "Extinction" => Some(BDEvent::Extinction),
            "Leaf" => Some(BDEvent::Leaf),
            _ => None,
        }
    }
}

/// Represents an event in the birth-death process
#[derive(Clone, Debug)]
pub struct TreeEvent {
    /// Time when the event occurred (going backwards from present at 0)
    pub time: f64,
    /// Node ID where the event occurred
    pub node_id: usize,
    /// Type of event
    pub event_type: BDEvent,
    /// First child node ID (for speciation events)
    pub child1: Option<usize>,
    /// Second child node ID (for speciation events)
    pub child2: Option<usize>,
}

impl TreeEvent {
    /// Convert event to CSV row format, resolving node names from the tree
    pub fn to_csv_row(&self, tree: &FlatTree) -> Result<String, String> {
        let node_name = tree.nodes.get(self.node_id)
            .map(|n| n.name.as_str())
            .ok_or_else(|| format!("Invalid node_id {} (tree has {} nodes)", self.node_id, tree.nodes.len()))?;
        let child1_name = match self.child1 {
            Some(c) => tree.nodes.get(c)
                .map(|n| n.name.clone())
                .ok_or_else(|| format!("Invalid child1 index {} (tree has {} nodes)", c, tree.nodes.len()))?,
            None => String::new(),
        };
        let child2_name = match self.child2 {
            Some(c) => tree.nodes.get(c)
                .map(|n| n.name.clone())
                .ok_or_else(|| format!("Invalid child2 index {} (tree has {} nodes)", c, tree.nodes.len()))?,
            None => String::new(),
        };
        Ok(format!("{},{},{},{},{}", self.time, node_name, self.event_type.as_str(), child1_name, child2_name))
    }

    /// CSV header for event data
    pub fn csv_header() -> &'static str {
        "time,node_name,event_type,child1_name,child2_name"
    }
}

// birth-death process types

use crate::node::FlatTree;
use std::str::FromStr;

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

}

impl FromStr for BDEvent {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Speciation" => Ok(BDEvent::Speciation),
            "Extinction" => Ok(BDEvent::Extinction),
            "Leaf" => Ok(BDEvent::Leaf),
            _ => Err(format!("Unknown BDEvent '{}'. Valid values: Speciation, Extinction, Leaf", s)),
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
                .map(|n| n.name.as_str())
                .ok_or_else(|| format!("Invalid child1 index {} (tree has {} nodes)", c, tree.nodes.len()))?,
            None => "",
        };
        let child2_name = match self.child2 {
            Some(c) => tree.nodes.get(c)
                .map(|n| n.name.as_str())
                .ok_or_else(|| format!("Invalid child2 index {} (tree has {} nodes)", c, tree.nodes.len()))?,
            None => "",
        };
        Ok(format!("{},{},{},{},{}", self.time, node_name, self.event_type.as_str(), child1_name, child2_name))
    }

    /// CSV header for event data
    pub fn csv_header() -> &'static str {
        "time,node_name,event_type,child1_name,child2_name"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn bd_event_from_str_valid() {
        assert_eq!(BDEvent::from_str("Speciation"), Ok(BDEvent::Speciation));
        assert_eq!(BDEvent::from_str("Extinction"), Ok(BDEvent::Extinction));
        assert_eq!(BDEvent::from_str("Leaf"), Ok(BDEvent::Leaf));
    }

    #[test]
    fn bd_event_from_str_invalid() {
        assert!(BDEvent::from_str("").is_err());
        assert!(BDEvent::from_str("speciation").is_err()); // case-sensitive
        assert!(BDEvent::from_str("SPECIATION").is_err());
        assert!(BDEvent::from_str("Unknown").is_err());
        assert!(BDEvent::from_str("Transfer").is_err()); // valid DTL event but not BD
    }

    #[test]
    fn bd_event_roundtrip() {
        for event in [BDEvent::Speciation, BDEvent::Extinction, BDEvent::Leaf] {
            let s = event.as_str();
            let parsed = BDEvent::from_str(s);
            assert_eq!(parsed, Ok(event), "Roundtrip failed for {:?}", event);
        }
    }
}

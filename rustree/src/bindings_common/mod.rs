//! Shared validation and utility logic for Python and R bindings.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::error::RustreeError;
use crate::metric_functions::DistanceType;
use crate::node::{FlatNode, FlatTree, Event, RecTree};
use crate::sampling::extract_induced_subtree;

/// Validate that DTL rates are all non-negative.
pub fn validate_dtl_rates(lambda_d: f64, lambda_t: f64, lambda_l: f64) -> Result<(), RustreeError> {
    if lambda_d < 0.0 || lambda_t < 0.0 || lambda_l < 0.0 {
        return Err(RustreeError::Validation("Rates must be non-negative".to_string()));
    }
    Ok(())
}

/// Validate that replacement_transfer probability is in [0.0, 1.0].
pub fn validate_replacement_transfer(replacement_transfer: Option<f64>) -> Result<(), RustreeError> {
    if let Some(p) = replacement_transfer {
        if !(0.0..=1.0).contains(&p) {
            return Err(RustreeError::Validation("replacement_transfer must be between 0.0 and 1.0".to_string()));
        }
    }
    Ok(())
}

/// Parse a distance type string into a DistanceType enum.
pub fn parse_distance_type(distance_type: &str) -> Result<DistanceType, RustreeError> {
    match distance_type.to_lowercase().as_str() {
        "topological" | "topo" => Ok(DistanceType::Topological),
        "metric" | "branch" | "patristic" => Ok(DistanceType::Metric),
        _ => Err(RustreeError::Validation(format!(
            "Invalid distance_type '{}'. Use 'topological' or 'metric'.",
            distance_type
        ))),
    }
}

/// Extract the gene tree with only extant (Event::Leaf) leaves, stripping loss nodes.
pub fn extract_extant_gene_tree(rec_tree: &RecTree) -> Result<FlatTree, RustreeError> {
    let extant_indices: std::collections::HashSet<usize> = rec_tree.gene_tree.nodes.iter()
        .enumerate()
        .filter(|(i, n)| {
            n.left_child.is_none()
            && n.right_child.is_none()
            && rec_tree.event_mapping[*i] == Event::Leaf
        })
        .map(|(i, _)| i)
        .collect();

    if extant_indices.is_empty() {
        return Err(RustreeError::Tree("No extant leaves".to_string()));
    }

    let (tree, _) = extract_induced_subtree(&rec_tree.gene_tree, &extant_indices)
        .ok_or_else(|| RustreeError::Tree("Failed to extract extant subtree".to_string()))?;
    Ok(tree)
}

/// Check if a node is a leaf (no children).
pub fn is_leaf(node: &FlatNode) -> bool {
    node.left_child.is_none() && node.right_child.is_none()
}

/// Compute the number of digits needed to represent n (for zero-padded filenames).
pub fn digit_width(n: usize) -> usize {
    if n == 0 { 1 } else { ((n as f64).log10().floor() as usize) + 1 }
}

/// Initialize an RNG from an optional seed.
pub fn init_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

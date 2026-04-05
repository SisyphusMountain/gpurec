//! Python bindings for rustree using PyO3.
//!
//! Provides Python access to:
//! - Birth-death species tree simulation
//! - DTL gene tree simulation
//! - Gene tree sampling (induced subtree extraction)

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::bd::simulate_bd_tree_bwd;
use crate::node::{FlatTree, RecTree};

use std::sync::Arc;
use std::fs;

// Submodules
pub mod species_tree;
pub mod gene_tree;
pub mod sim_iter;
pub mod types;
pub mod reconciliation;
pub mod alerax;
pub mod forest;
pub mod training;

// Re-export submodule types for the pymodule registration and cross-module use
pub use species_tree::{PySpeciesTree, PySpeciesNode, PySpeciesTreeIter};
pub use gene_tree::{PyGeneTree, PyInducedTransfer};
pub use sim_iter::PyDtlSimIter;
pub use types::{PyEventCounts, PyReconciliationStatistics};
pub use reconciliation::{PyReconciliationComparison, PyMultiSampleComparison};
pub use alerax::PyAleRaxResult;
pub use forest::{PyGeneForest, PyAleRaxForestResult};

// ============================================================================
// Helper Functions (thin wrappers around bindings_common, converting to PyResult)
// ============================================================================

pub(crate) fn validate_dtl_rates(lambda_d: f64, lambda_t: f64, lambda_l: f64) -> PyResult<()> {
    crate::bindings_common::validate_dtl_rates(lambda_d, lambda_t, lambda_l)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn validate_replacement_transfer(replacement_transfer: Option<f64>) -> PyResult<()> {
    crate::bindings_common::validate_replacement_transfer(replacement_transfer)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn extract_extant_gene_tree(rec_tree: &RecTree) -> Result<FlatTree, crate::error::RustreeError> {
    crate::bindings_common::extract_extant_gene_tree(rec_tree)
}

pub(crate) fn init_rng(seed: Option<u64>) -> StdRng {
    crate::bindings_common::init_rng(seed)
}

pub(crate) fn is_leaf(node: &crate::node::FlatNode) -> bool {
    crate::bindings_common::is_leaf(node)
}

pub(crate) fn parse_distance_type(distance_type: &str) -> PyResult<crate::metric_functions::DistanceType> {
    crate::bindings_common::parse_distance_type(distance_type)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Import a Python module with a helpful error message if it's not installed.
pub(crate) fn import_pymodule<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    py.import(name).map_err(|_| {
        let install_hint = match name {
            "pandas" => "pip install pandas",
            "matplotlib.pyplot" | "matplotlib" => "pip install matplotlib",
            "IPython.display" | "IPython" => "pip install ipython",
            _ => "pip install <package>",
        };
        PyValueError::new_err(format!(
            "Required package '{}' is not installed. Install with: {}",
            name, install_hint
        ))
    })
}

// ============================================================================
// Module-level functions
// ============================================================================

/// Simulate a birth-death species tree.
#[pyfunction]
#[pyo3(signature = (n, lambda_, mu, seed=None))]
fn simulate_species_tree(n: usize, lambda_: f64, mu: f64, seed: Option<u64>) -> PyResult<PySpeciesTree> {
    if n == 0 {
        return Err(PyValueError::new_err("Number of species must be positive"));
    }
    if lambda_ <= 0.0 {
        return Err(PyValueError::new_err("Speciation rate must be positive"));
    }
    if mu < 0.0 {
        return Err(PyValueError::new_err("Extinction rate must be non-negative"));
    }
    if lambda_ <= mu {
        return Err(PyValueError::new_err("Speciation rate must be strictly greater than extinction rate"));
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let (mut tree, _events) = simulate_bd_tree_bwd(n, lambda_, mu, &mut rng)
        .map_err(PyValueError::new_err)?;
    tree.assign_depths();

    Ok(PySpeciesTree { tree: Arc::new(tree) })
}

/// Parse a Newick string or file into a species tree.
#[pyfunction]
pub(crate) fn parse_species_tree(newick_str: &str) -> PyResult<PySpeciesTree> {
    use crate::newick::parse_newick;
    use std::path::Path;

    // Check if input is a file path
    let newick_content = if Path::new(newick_str).exists() {
        fs::read_to_string(newick_str)
            .map_err(|e| PyValueError::new_err(format!("Failed to read file '{}': {}", newick_str, e)))?
    } else {
        newick_str.to_string()
    };

    let mut nodes = parse_newick(&newick_content)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse Newick: {}", e)))?;

    let mut root = nodes.pop()
        .ok_or_else(|| PyValueError::new_err("No tree found in Newick string"))?;

    root.assign_depths(0.0);
    let tree = root.to_flat_tree();

    Ok(PySpeciesTree { tree: Arc::new(tree) })
}

/// Parse a RecPhyloXML file into a gene tree with reconciliation information.
#[pyfunction]
fn parse_recphyloxml(filepath: &str) -> PyResult<PyGeneTree> {
    let rec_tree = RecTree::from_xml_file(filepath)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse RecPhyloXML: {}", e)))?;

    Ok(PyGeneTree { rec_tree })
}

// ============================================================================
// Python module registration
// ============================================================================

/// Python module for rustree.
#[pymodule]
fn rustree(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core functions
    m.add_function(wrap_pyfunction!(simulate_species_tree, m)?)?;
    m.add_function(wrap_pyfunction!(parse_species_tree, m)?)?;
    m.add_function(wrap_pyfunction!(parse_recphyloxml, m)?)?;

    // ALERax
    m.add_function(wrap_pyfunction!(alerax::reconcile_with_alerax, m)?)?;

    // Training / ML
    m.add_function(wrap_pyfunction!(training::create_training_sample, m)?)?;
    m.add_function(wrap_pyfunction!(training::create_training_sample_from_sim, m)?)?;
    m.add_function(wrap_pyfunction!(training::build_training_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(training::build_otf_batch, m)?)?;
    m.add_function(wrap_pyfunction!(training::compute_gcn_norm, m)?)?;
    m.add_function(wrap_pyfunction!(training::build_inference_batch, m)?)?;
    m.add_function(wrap_pyfunction!(training::from_reconciliation, m)?)?;

    // Reconciliation comparison
    m.add_function(wrap_pyfunction!(reconciliation::compare_reconciliations, m)?)?;
    m.add_function(wrap_pyfunction!(reconciliation::compare_reconciliations_multi, m)?)?;

    // Classes
    m.add_class::<PySpeciesTree>()?;
    m.add_class::<PySpeciesNode>()?;
    m.add_class::<PySpeciesTreeIter>()?;
    m.add_class::<PyGeneTree>()?;
    m.add_class::<PyDtlSimIter>()?;
    m.add_class::<PyEventCounts>()?;
    m.add_class::<PyReconciliationStatistics>()?;
    m.add_class::<PyAleRaxResult>()?;
    m.add_class::<PyGeneForest>()?;
    m.add_class::<PyAleRaxForestResult>()?;
    m.add_class::<PyReconciliationComparison>()?;
    m.add_class::<PyMultiSampleComparison>()?;
    Ok(())
}

//! GeneForest and ALERax forest result Python bindings.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::sync::Arc;

use crate::node::RecTree;

use super::{PyGeneTree, PySpeciesTree, PyEventCounts, PyReconciliationStatistics};
use super::alerax::PyAleRaxResult;

// ============================================================================
// GeneForest
// ============================================================================

/// A collection of gene trees associated with a single species tree.
///
/// PyGeneForest wraps a GeneForest and provides methods for:
/// - Pruning by species tree or species leaf names
/// - Reconciliation with ALERax
/// - Batch access to gene trees
///
/// # Example
/// ```python
/// import rustree
///
/// st = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)
/// gts = st.simulate_dtl_batch(n=5, lambda_d=0.5, lambda_t=0.2, lambda_l=0.3, seed=123)
/// forest = rustree.GeneForest(st, gts)
/// print(forest)  # GeneForest(species_leaves=20, gene_trees=5)
/// ```
#[pyclass(name = "GeneForest")]
#[derive(Clone)]
pub struct PyGeneForest {
    pub(crate) forest: crate::node::gene_forest::GeneForest,
}

#[pymethods]
impl PyGeneForest {
    /// Create a new GeneForest from a species tree and list of gene trees.
    #[new]
    fn new(species_tree: &PySpeciesTree, gene_trees: Vec<PyGeneTree>) -> PyResult<Self> {
        let species_arc = Arc::clone(&species_tree.tree);
        let rec_trees: Vec<RecTree> = gene_trees.into_iter()
            .map(|gt| {
                let mut rt = gt.rec_tree;
                rt.species_tree = Arc::clone(&species_arc);
                rt
            })
            .collect();

        Ok(PyGeneForest {
            forest: crate::node::gene_forest::GeneForest::from_rec_trees(species_arc, rec_trees),
        })
    }

    /// Number of gene trees in the forest.
    fn __len__(&self) -> usize {
        self.forest.len()
    }

    /// Get a gene tree by index.
    fn __getitem__(&self, idx: usize) -> PyResult<PyGeneTree> {
        self.forest.get(idx)
            .map(|rt| PyGeneTree { rec_tree: rt.clone() })
            .ok_or_else(|| PyValueError::new_err(format!("Index {} out of range (len={})", idx, self.forest.len())))
    }

    /// Get the species tree.
    #[getter]
    fn species_tree(&self) -> PySpeciesTree {
        PySpeciesTree {
            tree: Arc::clone(&self.forest.species_tree),
        }
    }

    /// Get all gene trees.
    #[getter]
    fn gene_trees(&self) -> Vec<PyGeneTree> {
        self.forest.gene_trees.iter()
            .map(|rt| PyGeneTree { rec_tree: rt.clone() })
            .collect()
    }

    /// Return a new forest with each gene tree pruned to extant leaves only.
    ///
    /// Keeps only gene tree leaves whose event is Leaf (i.e., extant genes).
    /// Gene trees with zero extant leaves are dropped.
    fn sample_extant(&self) -> PyResult<PyGeneForest> {
        let sampled = self.forest.sample_extant()
            .map_err(|e| PyValueError::new_err(format!("Failed to sample extant: {}", e)))?;
        Ok(PyGeneForest { forest: sampled })
    }

    /// Prune the forest to match a target species tree.
    ///
    /// The target species tree's leaves must be a subset of this forest's
    /// species tree leaves. Returns a new GeneForest with pruned trees.
    fn prune_to_species_tree(&self, target: &PySpeciesTree) -> PyResult<PyGeneForest> {
        let pruned = self.forest.prune_to_species_tree(&target.tree)
            .map_err(|e| PyValueError::new_err(format!("Failed to prune: {}", e)))?;
        Ok(PyGeneForest { forest: pruned })
    }

    /// Sample leaves and filter all trees accordingly.
    ///
    /// # Arguments
    /// * `names` - List of species leaf names to keep
    fn sample_leaves(&self, names: Vec<String>) -> PyResult<PyGeneForest> {
        let sampled = self.forest.sample_leaves(&names)
            .map_err(|e| PyValueError::new_err(format!("Failed to sample: {}", e)))?;
        Ok(PyGeneForest { forest: sampled })
    }

    /// Reconcile the gene forest with ALERax.
    ///
    /// Runs ALERax on all gene trees, parses all results, and auto-renames
    /// species tree nodes back to their original names.
    ///
    /// # Arguments
    /// * `output_dir` - Optional output directory (default: temp directory)
    /// * `num_samples` - Number of reconciliation samples per family (default: 100)
    /// * `model` - Model parametrization: "PER-FAMILY" or "GLOBAL" (default: "PER-FAMILY")
    /// * `seed` - Random seed for reproducibility (optional)
    /// * `keep_output` - Whether to preserve ALERax output files (default: False)
    /// * `alerax_path` - Path to alerax executable (default: "alerax")
    ///
    /// # Returns
    /// An AleRaxForestResult containing all reconciliation data.
    #[pyo3(signature = (output_dir=None, num_samples=100, model="PER-FAMILY".to_string(), gene_tree_rooting=None, seed=None, keep_output=false, alerax_path="alerax".to_string()))]
    #[allow(clippy::too_many_arguments)]
    fn reconcile_with_alerax(
        &self,
        output_dir: Option<String>,
        num_samples: usize,
        model: String,
        gene_tree_rooting: Option<String>,
        seed: Option<u64>,
        keep_output: bool,
        alerax_path: String,
    ) -> PyResult<PyAleRaxForestResult> {
        use crate::alerax::{reconcile_forest, ModelType};
        use std::path::PathBuf;

        let model_type = match model.to_uppercase().as_str() {
            "PER-FAMILY" | "PER_FAMILY" => ModelType::PerFamily,
            "GLOBAL" => ModelType::Global,
            _ => return Err(PyValueError::new_err(format!(
                "Invalid model type '{}'. Must be 'PER-FAMILY' or 'GLOBAL'", model
            ))),
        };

        let output_path = output_dir.map(PathBuf::from);

        let result = reconcile_forest(
            &self.forest,
            output_path,
            num_samples,
            model_type,
            gene_tree_rooting,
            seed,
            &alerax_path,
            keep_output,
        ).map_err(|e| PyValueError::new_err(format!("ALERax reconciliation failed: {}", e)))?;

        Ok(PyAleRaxForestResult::from_rust(result))
    }

    fn __iter__(&self) -> PyGeneForestIter {
        PyGeneForestIter {
            forest: self.forest.clone(),
            pos: 0,
        }
    }

    fn __repr__(&self) -> String {
        let species_leaves = self.forest.species_tree().nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .count();
        format!(
            "GeneForest(species_leaves={}, gene_trees={})",
            species_leaves,
            self.forest.len()
        )
    }
}

/// Iterator over gene trees in a GeneForest.
#[pyclass]
struct PyGeneForestIter {
    forest: crate::node::gene_forest::GeneForest,
    pos: usize,
}

#[pymethods]
impl PyGeneForestIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&mut self) -> Option<PyGeneTree> {
        let gt = self.forest.get(self.pos)?;
        self.pos += 1;
        Some(PyGeneTree { rec_tree: gt.clone() })
    }
}

// ============================================================================
// ALERax Forest Result
// ============================================================================

/// Comprehensive result of reconciling a GeneForest with ALERax.
///
/// Contains per-family results (reconciled trees, rates, likelihoods)
/// plus aggregate statistics across all families, accessible as DataFrames.
#[pyclass(name = "AleRaxForestResult")]
pub struct PyAleRaxForestResult {
    #[pyo3(get)]
    family_results: HashMap<String, PyAleRaxResult>,
    mean_species_event_counts: HashMap<String, Vec<crate::alerax::SpeciesEventRow>>,
    total_species_event_counts_data: Vec<crate::alerax::SpeciesEventRow>,
    total_transfers_data: Vec<crate::alerax::TransferRow>,
    #[pyo3(get)]
    output_dir: Option<String>,
}

impl PyAleRaxForestResult {
    fn from_rust(result: crate::alerax::AleRaxForestResult) -> Self {
        let mut py_family_results = HashMap::new();
        for (name, family_result) in result.family_results {
            let py_gene_trees: Vec<PyGeneTree> = family_result.reconciled_trees
                .into_iter()
                .map(|rt| PyGeneTree { rec_tree: rt })
                .collect();

            let mean_event_counts = PyEventCounts {
                speciations: family_result.statistics.mean_event_counts.speciations,
                speciation_losses: family_result.statistics.mean_event_counts.speciation_losses,
                duplications: family_result.statistics.mean_event_counts.duplications,
                duplication_losses: family_result.statistics.mean_event_counts.duplication_losses,
                transfers: family_result.statistics.mean_event_counts.transfers,
                transfer_losses: family_result.statistics.mean_event_counts.transfer_losses,
                losses: family_result.statistics.mean_event_counts.losses,
                leaves: family_result.statistics.mean_event_counts.leaves,
            };

            let events_per_species: HashMap<String, PyEventCounts> = family_result.statistics.events_per_species
                .into_iter()
                .map(|(species, counts)| {
                    (species, PyEventCounts {
                        speciations: counts.speciations,
                        speciation_losses: counts.speciation_losses,
                        duplications: counts.duplications,
                        duplication_losses: counts.duplication_losses,
                        transfers: counts.transfers,
                        transfer_losses: counts.transfer_losses,
                        losses: counts.losses,
                        leaves: counts.leaves,
                    })
                })
                .collect();

            let statistics = PyReconciliationStatistics {
                mean_event_counts,
                mean_transfers: family_result.statistics.mean_transfers,
                events_per_species,
            };

            py_family_results.insert(name, PyAleRaxResult {
                gene_trees: py_gene_trees,
                duplication_rate: family_result.duplication_rate,
                loss_rate: family_result.loss_rate,
                transfer_rate: family_result.transfer_rate,
                likelihood: family_result.likelihood,
                statistics,
            });
        }

        PyAleRaxForestResult {
            family_results: py_family_results,
            mean_species_event_counts: result.mean_species_event_counts,
            total_species_event_counts_data: result.total_species_event_counts,
            total_transfers_data: result.total_transfers,
            output_dir: result.output_dir.map(|p| p.to_string_lossy().to_string()),
        }
    }
}

/// Helper: convert SpeciesEventRow slice to a pandas DataFrame.
fn species_event_rows_to_df(
    py: Python,
    rows: &[crate::alerax::SpeciesEventRow],
) -> PyResult<PyObject> {
    let pandas = super::import_pymodule(py, "pandas")?;
    let dict = pyo3::types::PyDict::new(py);

    let labels: Vec<&str> = rows.iter().map(|r| r.species_label.as_str()).collect();
    let speciations: Vec<f64> = rows.iter().map(|r| r.speciations).collect();
    let duplications: Vec<f64> = rows.iter().map(|r| r.duplications).collect();
    let losses: Vec<f64> = rows.iter().map(|r| r.losses).collect();
    let transfers: Vec<f64> = rows.iter().map(|r| r.transfers).collect();
    let presence: Vec<f64> = rows.iter().map(|r| r.presence).collect();
    let origination: Vec<f64> = rows.iter().map(|r| r.origination).collect();
    let copies: Vec<f64> = rows.iter().map(|r| r.copies).collect();
    let singletons: Vec<f64> = rows.iter().map(|r| r.singletons).collect();
    let transfers_to: Vec<f64> = rows.iter().map(|r| r.transfers_to).collect();

    dict.set_item("species_label", labels)?;
    dict.set_item("speciations", speciations)?;
    dict.set_item("duplications", duplications)?;
    dict.set_item("losses", losses)?;
    dict.set_item("transfers", transfers)?;
    dict.set_item("presence", presence)?;
    dict.set_item("origination", origination)?;
    dict.set_item("copies", copies)?;
    dict.set_item("singletons", singletons)?;
    dict.set_item("transfers_to", transfers_to)?;

    let df = pandas.call_method1("DataFrame", (dict,))?;
    Ok(df.into())
}

#[pymethods]
impl PyAleRaxForestResult {
    /// Get mean species event counts for a specific family as a pandas DataFrame.
    ///
    /// Columns: species_label, speciations, duplications, losses, transfers,
    ///          presence, origination, copies, singletons, transfers_to
    fn mean_species_event_counts(&self, py: Python, family_name: &str) -> PyResult<PyObject> {
        let rows = self.mean_species_event_counts.get(family_name)
            .ok_or_else(|| PyValueError::new_err(format!(
                "No mean species event counts for family '{}'", family_name
            )))?;
        species_event_rows_to_df(py, rows)
    }

    /// Get total (aggregate) species event counts as a pandas DataFrame.
    ///
    /// Columns: species_label, speciations, duplications, losses, transfers,
    ///          presence, origination, copies, singletons, transfers_to
    #[getter]
    fn total_species_event_counts(&self, py: Python) -> PyResult<PyObject> {
        species_event_rows_to_df(py, &self.total_species_event_counts_data)
    }

    /// Get total (aggregate) transfers as a pandas DataFrame.
    ///
    /// Columns: source, destination, count
    #[getter]
    fn total_transfers(&self, py: Python) -> PyResult<PyObject> {
        let pandas = super::import_pymodule(py, "pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        let sources: Vec<&str> = self.total_transfers_data.iter().map(|r| r.source.as_str()).collect();
        let dests: Vec<&str> = self.total_transfers_data.iter().map(|r| r.destination.as_str()).collect();
        let counts: Vec<f64> = self.total_transfers_data.iter().map(|r| r.count).collect();

        dict.set_item("source", sources)?;
        dict.set_item("destination", dests)?;
        dict.set_item("count", counts)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// List all family names.
    fn family_names(&self) -> Vec<String> {
        self.family_results.keys().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "AleRaxForestResult(families={}, total_species={})",
            self.family_results.len(),
            self.total_species_event_counts_data.len()
        )
    }
}


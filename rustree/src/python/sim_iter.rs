//! Streaming DTL simulation iterator for Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rand::rngs::StdRng;
use std::sync::Arc;
use std::fs;

use crate::bd::TreeEvent;
use crate::dtl::{count_extant_genes, DTLConfig};
use crate::node::{FlatTree, RecTree};
use crate::simulation::dtl::gillespie::{DTLMode, simulate_dtl_gillespie};

use super::gene_tree::PyGeneTree;
use super::forest::PyGeneForest;

/// Compute the number of digits needed to represent n (for zero-padded filenames).
pub(crate) fn digit_width(n: usize) -> usize {
    if n == 0 { 1 } else { ((n as f64).log10().floor() as usize) + 1 }
}

/// Lazy iterator for DTL gene tree simulation.
///
/// Generates one gene tree per `next()` call using the Gillespie algorithm.
/// Owns all its state (species tree, pre-computed data, RNG) so it can be
/// used as a Python iterator without lifetime issues.
///
/// Created by `PySpeciesTree.simulate_dtl_iter()` or
/// `PySpeciesTree.simulate_dtl_per_species_iter()`.
///
/// # Example
/// ```python
/// import rustree
///
/// species_tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)
///
/// # Create a lazy iterator for 1000 gene trees
/// it = species_tree.simulate_dtl_iter(
///     lambda_d=0.5, lambda_t=0.2, lambda_l=0.3, n=1000, seed=123
/// )
///
/// # Iterate one at a time
/// for gene_tree in it:
///     print(gene_tree.num_extant())
///
/// # Or use convenience methods:
/// it = species_tree.simulate_dtl_iter(lambda_d=0.5, lambda_t=0.2, lambda_l=0.3, n=100, seed=42)
/// it.save_xml("output/")
///
/// it = species_tree.simulate_dtl_iter(lambda_d=0.5, lambda_t=0.2, lambda_l=0.3, n=1, seed=42)
/// gene_tree = it.single()
/// ```
#[pyclass]
pub struct PyDtlSimIter {
    // Pre-computed state (owned)
    pub(crate) species_arc: Arc<FlatTree>,
    pub(crate) species_events: Vec<TreeEvent>,
    pub(crate) depths: Vec<f64>,
    pub(crate) contemporaneity: Vec<Vec<usize>>,
    pub(crate) lca_depths: Option<Vec<Vec<f64>>>,
    // Simulation parameters
    pub(crate) origin_species: usize,
    pub(crate) config: DTLConfig,
    pub(crate) n_simulations: usize,
    pub(crate) require_extant: bool,
    pub(crate) mode: DTLMode,
    // Mutable state
    pub(crate) rng: StdRng,
    pub(crate) completed: usize,
}

impl PyDtlSimIter {
    /// Run one Gillespie simulation and return the result.
    ///
    /// Retries if `require_extant` is true and the tree has no extant genes.
    /// Returns `None` when all requested simulations have been completed.
    fn next_simulation(&mut self) -> Option<Result<RecTree, String>> {
        if self.completed >= self.n_simulations {
            return None;
        }

        loop {
            let lca_ref = self.lca_depths.as_deref();
            let result = simulate_dtl_gillespie(
                self.mode,
                &self.species_arc,
                &self.species_events,
                &self.depths,
                &self.contemporaneity,
                lca_ref,
                self.origin_species,
                &self.config,
                &mut self.rng,
            );

            match result {
                Ok((mut rec_tree, events)) => {
                    if !self.require_extant || count_extant_genes(&rec_tree) > 0 {
                        self.completed += 1;
                        rec_tree.dtl_events = Some(events);
                        return Some(Ok(rec_tree));
                    }
                    // Retry: tree had no extant genes
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

#[pymethods]
impl PyDtlSimIter {
    /// Python iterator protocol: return self.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Python iterator protocol: generate the next gene tree.
    ///
    /// Each call runs one Gillespie simulation. Returns a PyGeneTree or
    /// raises StopIteration when all requested simulations are done.
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyGeneTree>> {
        let species_arc = Arc::clone(&slf.species_arc);
        match slf.next_simulation() {
            None => Ok(None), // StopIteration
            Some(Ok(mut rec_tree)) => {
                rec_tree.species_tree = species_arc;
                Ok(Some(PyGeneTree { rec_tree }))
            }
            Some(Err(e)) => Err(PyValueError::new_err(e)),
        }
    }

    /// Number of simulations remaining.
    #[getter]
    fn remaining(&self) -> usize {
        self.n_simulations.saturating_sub(self.completed)
    }

    /// Total number of simulations requested.
    #[getter]
    fn total(&self) -> usize {
        self.n_simulations
    }

    /// Number of simulations completed so far.
    #[getter]
    fn completed(&self) -> usize {
        self.completed
    }

    /// Run a single simulation and return the result directly.
    ///
    /// Convenience for the common case of generating one tree.
    /// Equivalent to calling `next()` once.
    ///
    /// # Raises
    /// ValueError if no simulations remain or the simulation fails.
    fn single(&mut self) -> PyResult<PyGeneTree> {
        let species_arc = Arc::clone(&self.species_arc);
        match self.next_simulation() {
            None => Err(PyValueError::new_err("No simulations remaining")),
            Some(Ok(mut rec_tree)) => {
                rec_tree.species_tree = species_arc;
                Ok(PyGeneTree { rec_tree })
            }
            Some(Err(e)) => Err(PyValueError::new_err(e)),
        }
    }

    /// Collect all remaining simulations into a GeneForest.
    ///
    /// Warning: this loads all trees into memory at once.
    ///
    /// # Returns
    /// A GeneForest containing all remaining gene trees.
    fn collect_all(&mut self) -> PyResult<PyGeneForest> {
        let mut trees = Vec::with_capacity(self.n_simulations.saturating_sub(self.completed));
        loop {
            let species_arc = Arc::clone(&self.species_arc);
            match self.next_simulation() {
                None => break,
                Some(Ok(mut rec_tree)) => {
                    rec_tree.species_tree = species_arc;
                    trees.push(rec_tree);
                }
                Some(Err(e)) => return Err(PyValueError::new_err(e)),
            }
        }
        Ok(PyGeneForest {
            forest: crate::node::gene_forest::GeneForest::from_rec_trees(
                Arc::clone(&self.species_arc), trees,
            ),
        })
    }

    /// Save each remaining gene tree as RecPhyloXML to the given directory.
    ///
    /// Files are named `gene_0000.xml`, `gene_0001.xml`, etc.
    /// Creates the directory if it does not exist.
    ///
    /// # Arguments
    /// * `dir` - Directory path to save the XML files
    fn save_xml(&mut self, dir: &str) -> PyResult<()> {
        fs::create_dir_all(dir)
            .map_err(|e| PyValueError::new_err(format!("Failed to create directory: {}", e)))?;
        let width = digit_width(self.n_simulations);
        let mut idx = self.completed;
        loop {
            let species_arc = Arc::clone(&self.species_arc);
            match self.next_simulation() {
                None => break,
                Some(Ok(mut rec_tree)) => {
                    rec_tree.species_tree = species_arc;
                    let xml = rec_tree.to_xml();
                    let path = format!("{}/gene_{:0>width$}.xml", dir, idx, width = width);
                    fs::write(&path, &xml)
                        .map_err(|e| PyValueError::new_err(format!("Failed to write {}: {}", path, e)))?;
                    idx += 1;
                }
                Some(Err(e)) => return Err(PyValueError::new_err(e)),
            }
        }
        Ok(())
    }

    /// Save each remaining gene tree as Newick to the given directory.
    ///
    /// Files are named `gene_0000.nwk`, `gene_0001.nwk`, etc.
    /// Creates the directory if it does not exist.
    ///
    /// Note: Newick format does not preserve reconciliation information
    /// (species mapping, events). Use `save_xml` for full reconciled trees.
    ///
    /// # Arguments
    /// * `dir` - Directory path to save the Newick files
    fn save_newick(&mut self, dir: &str) -> PyResult<()> {
        fs::create_dir_all(dir)
            .map_err(|e| PyValueError::new_err(format!("Failed to create directory: {}", e)))?;
        let width = digit_width(self.n_simulations);
        let mut idx = self.completed;
        loop {
            let species_arc = Arc::clone(&self.species_arc);
            match self.next_simulation() {
                None => break,
                Some(Ok(mut rec_tree)) => {
                    rec_tree.species_tree = species_arc;
                    let newick = rec_tree.gene_tree.to_newick()
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let path = format!("{}/gene_{:0>width$}.nwk", dir, idx, width = width);
                    fs::write(&path, format!("{};", newick))
                        .map_err(|e| PyValueError::new_err(format!("Failed to write {}: {}", path, e)))?;
                    idx += 1;
                }
                Some(Err(e)) => return Err(PyValueError::new_err(e)),
            }
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        let mode_str = match self.mode {
            DTLMode::PerGene => "per_gene",
            DTLMode::PerSpecies => "per_species",
        };
        format!(
            "PyDtlSimIter(mode={}, completed={}/{}, D={}, T={}, L={})",
            mode_str, self.completed, self.n_simulations,
            self.config.lambda_d, self.config.lambda_t, self.config.lambda_l
        )
    }

    fn __len__(&self) -> usize {
        self.n_simulations.saturating_sub(self.completed)
    }
}

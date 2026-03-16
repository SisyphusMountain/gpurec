// Streaming DTL simulation iterator
//
// Provides DtlSimIter, a lazy iterator that generates one gene tree at a time
// without accumulating all trees in memory. Supports chainable convenience
// methods like .save_xml(), .save_newick(), and .collect_all().

use crate::bd::TreeEvent;
use crate::node::{FlatTree, RecTree};
use rand::Rng;
use std::sync::Arc;

use super::event::DTLEvent;
use super::gillespie::{DTLMode, simulate_dtl_gillespie};
use super::utils::count_extant_genes;

/// Lazy iterator that generates DTL-simulated gene trees one at a time.
///
/// Created by [`simulate_dtl_iter`](super::simulate_dtl_iter) or
/// [`simulate_dtl_per_species_iter`](super::simulate_dtl_per_species_iter).
///
/// Each call to `next()` runs one Gillespie simulation and yields the result.
/// Only one gene tree lives in memory at a time.
///
/// # Examples
///
/// ```no_run
/// use rustree::dtl::simulate_dtl_iter;
/// # use rustree::node::FlatTree;
/// # use rand::SeedableRng;
/// # let species_tree = FlatTree { nodes: vec![], root: 0 };
/// # let mut rng = rand::rngs::StdRng::seed_from_u64(42);
///
/// // Stream 10,000 trees directly to XML files:
/// simulate_dtl_iter(&species_tree, 0, 0.2, 0.2, 0.1, None, None, 10_000, true, &mut rng)
///     .unwrap()
///     .save_xml("output/")
///     .unwrap();
/// ```
pub struct DtlSimIter<'a, R: Rng> {
    // Pre-computed state (owned)
    species_arc: Arc<FlatTree>,
    species_events: Vec<TreeEvent>,
    depths: Vec<f64>,
    contemporaneity: Vec<Vec<usize>>,
    lca_depths: Option<Vec<Vec<f64>>>,
    // Simulation parameters
    origin_species: usize,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Option<f64>,
    replacement_transfer: Option<f64>,
    n_simulations: usize,
    require_extant: bool,
    mode: DTLMode,
    // Mutable state
    rng: &'a mut R,
    completed: usize,
}

impl<'a, R: Rng> DtlSimIter<'a, R> {
    /// Creates a new DtlSimIter with pre-computed state.
    pub(crate) fn new(
        mode: DTLMode,
        species_arc: Arc<FlatTree>,
        species_events: Vec<TreeEvent>,
        depths: Vec<f64>,
        contemporaneity: Vec<Vec<usize>>,
        lca_depths: Option<Vec<Vec<f64>>>,
        origin_species: usize,
        lambda_d: f64,
        lambda_t: f64,
        lambda_l: f64,
        transfer_alpha: Option<f64>,
        replacement_transfer: Option<f64>,
        n_simulations: usize,
        require_extant: bool,
        rng: &'a mut R,
    ) -> Self {
        DtlSimIter {
            species_arc,
            species_events,
            depths,
            contemporaneity,
            lca_depths,
            origin_species,
            lambda_d,
            lambda_t,
            lambda_l,
            transfer_alpha,
            replacement_transfer,
            n_simulations,
            require_extant,
            mode,
            rng,
            completed: 0,
        }
    }

    /// Save each tree as RecPhyloXML to the given directory.
    ///
    /// Files are named `gene_0000.xml`, `gene_0001.xml`, etc.
    /// Creates the directory if it doesn't exist.
    pub fn save_xml(self, dir: &str) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let width = digit_width(self.n_simulations);
        for (i, result) in self.enumerate() {
            let (rec_tree, _events) = result?;
            let xml = rec_tree.to_xml();
            let path = format!("{}/gene_{:0>width$}.xml", dir, i, width = width);
            std::fs::write(&path, &xml).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Save each gene tree as Newick to the given directory.
    ///
    /// Files are named `gene_0000.nwk`, `gene_0001.nwk`, etc.
    /// Creates the directory if it doesn't exist.
    ///
    /// Note: Newick format does not preserve reconciliation information
    /// (species mapping, events). Use `save_xml` for full reconciled trees.
    pub fn save_newick(self, dir: &str) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let width = digit_width(self.n_simulations);
        for (i, result) in self.enumerate() {
            let (rec_tree, _events) = result?;
            let newick = rec_tree.gene_tree.to_newick()?;
            let path = format!("{}/gene_{:0>width$}.nwk", dir, i, width = width);
            std::fs::write(&path, &newick).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Run a single simulation and return the result directly.
    ///
    /// Convenience for the common case of generating one tree.
    /// Equivalent to creating the iterator with `n_simulations=1`
    /// and calling `.next()`.
    pub fn single(mut self) -> Result<(RecTree, Vec<DTLEvent>), String> {
        self.next()
            .ok_or_else(|| "No simulations requested".to_string())?
    }

    /// Collect all trees and events into vectors.
    ///
    /// Equivalent to the old `simulate_dtl_batch` behavior.
    /// Warning: this loads everything into memory.
    pub fn collect_all(self) -> Result<(Vec<RecTree>, Vec<Vec<DTLEvent>>), String> {
        let mut trees = Vec::with_capacity(self.n_simulations);
        let mut all_events = Vec::with_capacity(self.n_simulations);
        for result in self {
            let (rec_tree, events) = result?;
            trees.push(rec_tree);
            all_events.push(events);
        }
        Ok((trees, all_events))
    }
}

impl<'a, R: Rng> Iterator for DtlSimIter<'a, R> {
    type Item = Result<(RecTree, Vec<DTLEvent>), String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.completed >= self.n_simulations {
            return None;
        }

        loop {
            let lca_ref = self.lca_depths.as_ref().map(|v| v.as_slice());
            let result = simulate_dtl_gillespie(
                self.mode,
                &self.species_arc,
                &self.species_events,
                &self.depths,
                &self.contemporaneity,
                lca_ref,
                self.origin_species,
                self.lambda_d,
                self.lambda_t,
                self.lambda_l,
                self.transfer_alpha,
                self.replacement_transfer,
                &mut *self.rng,
            );

            match result {
                Ok((rec_tree, events)) => {
                    if !self.require_extant || count_extant_genes(&rec_tree) > 0 {
                        self.completed += 1;
                        return Some(Ok((rec_tree, events)));
                    }
                    // Retry: tree had no extant genes
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n_simulations - self.completed;
        (remaining, Some(remaining))
    }
}

/// Compute the number of digits needed to represent n (for zero-padded filenames).
fn digit_width(n: usize) -> usize {
    if n == 0 { 1 } else { ((n as f64).log10().floor() as usize) + 1 }
}

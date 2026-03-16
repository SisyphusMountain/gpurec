// Per-species DTL (Duplication-Transfer-Loss) simulation (Zombi-style)
//
// This module contains the per-species model where DTL event rates are
// proportional to the number of alive species, not the number of gene copies.
// Uses the shared Gillespie loop with PerSpecies mode.

use crate::bd::generate_events_from_tree;
use crate::node::{FlatTree, RecTree};
use rand::Rng;
use std::sync::Arc;

use super::event::DTLEvent;
use super::gillespie::DTLMode;
use super::stream::DtlSimIter;
use super::utils::{validate_rates, precompute_lca};

/// Returns a lazy iterator that generates gene trees one at a time (per-species model).
///
/// This is the core implementation for the Zombi-style per-species DTL model.
/// In this model, the DTL event rate is proportional to the number of ALIVE species,
/// NOT the number of gene copies.
///
/// All other functions in this module are thin wrappers around this iterator.
///
/// # Examples
///
/// ```no_run
/// # use rustree::dtl::simulate_dtl_per_species_iter;
/// # use rand::SeedableRng;
/// # let species_tree = rustree::FlatTree { nodes: vec![], root: 0 };
/// # let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// // Save 10,000 trees as XML (streaming):
/// simulate_dtl_per_species_iter(&species_tree, 0, 0.2, 0.2, 0.1, None, None, 10_000, true, &mut rng)
///     .unwrap()
///     .save_xml("output/").unwrap();
///
/// // Single tree:
/// let (tree, events) = simulate_dtl_per_species_iter(&species_tree, 0, 0.2, 0.2, 0.1, None, None, 1, true, &mut rng)
///     .unwrap()
///     .single().unwrap();
///
/// // Collect into vectors:
/// let (trees, events) = simulate_dtl_per_species_iter(&species_tree, 0, 0.2, 0.2, 0.1, None, None, 100, true, &mut rng)
///     .unwrap()
///     .collect_all().unwrap();
/// ```
pub fn simulate_dtl_per_species_iter<'a, R: Rng>(
    species_tree: &FlatTree,
    origin_species: usize,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Option<f64>,
    replacement_transfer: Option<f64>,
    n_simulations: usize,
    require_extant: bool,
    rng: &'a mut R,
) -> Result<DtlSimIter<'a, R>, String> {
    validate_rates(lambda_d, lambda_t, lambda_l)?;

    let species_arc = Arc::new(species_tree.clone());
    let species_events = generate_events_from_tree(species_tree)?;

    // Include origin start time in depths so the stem period is covered
    // by contemporaneity (make_subdivision only has node depths, not branch starts).
    let mut depths = species_tree.make_subdivision();
    let origin_start = species_tree.nodes[origin_species].depth
        .ok_or_else(|| format!("Origin species {} has no depth", origin_species))?
        - species_tree.nodes[origin_species].length;
    depths.push(origin_start);
    depths.sort_by(|a, b| a.total_cmp(b));
    depths.dedup();
    let contemporaneity = species_tree.find_contemporaneity(&depths);
    let lca_depths = precompute_lca(species_tree, transfer_alpha);

    Ok(DtlSimIter::new(
        DTLMode::PerSpecies,
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
        rng,
    ))
}

/// Simulates a gene tree using Zombi-style per-species rates.
///
/// Backward-compatible wrapper around [`simulate_dtl_per_species_iter`].
pub fn simulate_dtl_per_species<R: Rng>(
    species_tree: &FlatTree,
    origin_species: usize,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Option<f64>,
    replacement_transfer: Option<f64>,
    require_extant: bool,
    rng: &mut R,
) -> Result<(RecTree, Vec<DTLEvent>), String> {
    simulate_dtl_per_species_iter(
        species_tree, origin_species,
        lambda_d, lambda_t, lambda_l,
        transfer_alpha, replacement_transfer,
        1, require_extant, rng,
    )?.single()
}

/// Simulates multiple gene trees using the Zombi-style per-species DTL model.
///
/// Backward-compatible wrapper around [`simulate_dtl_per_species_iter`].
pub fn simulate_dtl_per_species_batch<R: Rng>(
    species_tree: &FlatTree,
    origin_species: usize,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Option<f64>,
    replacement_transfer: Option<f64>,
    n_simulations: usize,
    require_extant: bool,
    rng: &mut R,
) -> Result<(Vec<RecTree>, Vec<Vec<DTLEvent>>), String> {
    simulate_dtl_per_species_iter(
        species_tree, origin_species,
        lambda_d, lambda_t, lambda_l,
        transfer_alpha, replacement_transfer,
        n_simulations, require_extant, rng,
    )?.collect_all()
}

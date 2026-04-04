// Shared Gillespie-style DTL simulation loop
// Both per-gene and per-species DTL models use the same chronological
// event loop, differing only in how the total event rate is computed
// and how the affected gene copy is selected.

use crate::bd::{BDEvent, TreeEvent};
use crate::node::{FlatTree, Event, RecTree};
use rand::Rng;
use std::sync::Arc;

use super::event::DTLEvent;
use super::state::SimulationState;
use crate::simulation::utils::draw_waiting_time;
use super::utils::{
    find_time_index,
    select_transfer_recipient, select_transfer_recipient_assortative,
    finalize_simulation,
};

/// Determines how DTL event rates scale with the simulation state.
#[derive(Clone, Copy)]
pub(crate) enum DTLMode {
    /// Rate proportional to number of gene copies (per-gene model)
    PerGene,
    /// Rate proportional to number of alive species (per-species/Zombi model)
    PerSpecies,
}

/// Shared Gillespie-style DTL simulation.
///
/// Both per-gene and per-species models use this function. The `mode` parameter
/// determines:
/// - **PerGene**: total rate = n_copies × (λ_D + λ_T + λ_L), random copy is selected
/// - **PerSpecies**: total rate = n_alive_species × (λ_D + λ_T + λ_L), random species
///   is selected then random gene in that species (event fails if species has no genes)
/// For now, rates are the same for all species in the tree. We can change it later if we want.
/// Doing so will involve changing the rates at which events are proposed,
/// as well as the probability of a given event affecting each gene or species at time t.
/// We can have auxillary functions to flexibly handle the different generalizations without loss of performance.
pub(crate) fn simulate_dtl_gillespie<R: Rng>(
    mode: DTLMode,
    species_tree: &Arc<FlatTree>,
    species_events: &[TreeEvent],
    depths: &[f64],
    contemporaneity: &[Vec<usize>],
    lca_depths: Option<&[Vec<f64>]>,
    origin_species: usize,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Option<f64>,
    replacement_transfer: Option<f64>,
    rng: &mut R,
) -> Result<(RecTree, Vec<DTLEvent>), String> {

    let total_dtl_rate = lambda_d + lambda_t + lambda_l;
    let dup_threshold = if total_dtl_rate > 0.0 { lambda_d / total_dtl_rate } else { 0.0 };
    let transfer_threshold = if total_dtl_rate > 0.0 { (lambda_d + lambda_t) / total_dtl_rate } else { 0.0 };
    // Preallocate as many gene nodes as species nodes (for default case where there are no transfers, no duplications, no losses)
    let estimated_capacity = species_tree.nodes.len();
    let mut state = SimulationState::new(estimated_capacity, species_tree);

    // Find when origin species starts (beginning of its branch)
    let origin_start_time = species_tree.nodes[origin_species].depth
        .ok_or_else(|| format!("Species node {} has no assigned depth. Call assign_depths() first.", origin_species))?
        - species_tree.nodes[origin_species].length;

    let mut current_time = origin_start_time;

    // Skip species events before origin time
    // The following assumes species_events are sorted by time, which should always be the case.
    let mut species_event_idx = 0;
    while species_event_idx < species_events.len()
        && species_events[species_event_idx].time < origin_start_time
    {
        species_event_idx += 1;
    }



    // For more flexibility, maybe just create a function handle_origination in 
    // SimulationState that we can call here.
    // This would avoid hardcoding the logic, and would make the code more uniform.
    // Create initial gene at origin
    // We don't have an "Origination" like in ALE_dated or ALE_undated, so we just use a speciation event with no parent, 
    // on the origination species.
    let initial_gene_idx = state.create_gene_node(
        None, origin_species, Event::Speciation, origin_start_time,
    );
    state.add_gene_to_species(origin_species, initial_gene_idx);

    // Nudge past origin so the rate computation sees the stem interval
    // (same half-open boundary issue as speciations).
    current_time = current_time.next_up();

    // Main Gillespie loop
    loop {
        // We can enter here and have species_event_idx be out of bounds. In this case, next event time will be infinity.
        // what does it mean to be out of bounds?


        let next_species_event_time = species_events.get(species_event_idx)
            .map_or(f64::INFINITY, |e| e.time);

        // Check if all genes are lost
        if state.total_gene_copies() == 0 {
            break;
        }

        // Compute rate based on mode
        let dtl_total_rate = match mode {
            DTLMode::PerGene => {
                // In PerGene mode, copies evolve independently, so rates
                // are proportional to the number of copies. (n_copies * (lambda+mu+tau))
                state.total_gene_copies() as f64 * total_dtl_rate
            }
            DTLMode::PerSpecies => {
                // In PerSpecies mode, events are species-level.
                // They affect species, and events can fail if there
                // are no genes which can undergo the event we selected.
                // Rates are proportional to the number of alive species. (n_alive_species * (lambda+mu+tau))
                let time_idx = find_time_index(depths, current_time);
                contemporaneity[time_idx].len() as f64 * total_dtl_rate
            }
        };

        let next_dtl_time = current_time + draw_waiting_time(dtl_total_rate, rng);

        if next_species_event_time == f64::INFINITY && next_dtl_time == f64::INFINITY {
            break;
        }
        // next DTL event was supposed to occur AFTER
        // the next species event, so we actually don't do the
        // DTL event and instead process a species-level event which will affect genes.
        if next_species_event_time <= next_dtl_time && species_event_idx < species_events.len() {
            // === Process species event ===
            let sp_event = species_events[species_event_idx].clone();
            current_time = sp_event.time;

            match sp_event.event_type {
                BDEvent::Speciation => {
                    let child1 = sp_event.child1.ok_or_else(|| format!("Speciation event for node {} has no child1", sp_event.node_id))?;
                    let child2 = sp_event.child2.ok_or_else(|| format!("Speciation event for node {} has no child2", sp_event.node_id))?;
                    let gps = state.genes_per_species.as_mut().ok_or("Internal error: genes_per_species not initialized")?;
                    if let Some(genes) = gps.remove(&sp_event.node_id) {
                        for gene_idx in genes {
                            state.handle_speciation(
                                gene_idx, sp_event.node_id, child1, child2, current_time,
                            );
                        }
                    }
                }
                BDEvent::Extinction => {
                    let gps = state.genes_per_species.as_mut().ok_or("Internal error: genes_per_species not initialized")?;
                    if let Some(genes) = gps.remove(&sp_event.node_id) {
                        for gene_idx in genes {
                            state.handle_loss(gene_idx, sp_event.node_id, current_time);
                        }
                    }
                }
                BDEvent::Leaf => {
                    let gps = state.genes_per_species.as_mut().ok_or("Internal error: genes_per_species not initialized")?;
                    if let Some(genes) = gps.remove(&sp_event.node_id) {
                        for gene_idx in genes {
                            state.handle_leaf(gene_idx, sp_event.node_id, current_time);
                        }
                    }
                }
            }

            species_event_idx += 1;

            // Nudge time past the event boundary so the next rate computation
            // uses post-event contemporaneity (children instead of parent after
            // speciations, excludes extinct/leaf species after those events).
            current_time = current_time.next_up();
        } else {
            // === Process DTL event ===
            current_time = next_dtl_time;

            // Select affected gene based on mode
            let selection = match mode {
                DTLMode::PerGene => {
                    state.random_gene_copy(rng)
                }
                DTLMode::PerSpecies => {
                    let time_idx = find_time_index(depths, current_time);
                    let alive_species = &contemporaneity[time_idx];
                    if alive_species.is_empty() { None }
                    else {
                        let species = alive_species[rng.gen_range(0..alive_species.len())];
                        let gps = state.genes_per_species.as_ref().ok_or("Internal error: genes_per_species not initialized")?;
                        match gps.get(&species) {
                            Some(genes) if !genes.is_empty() => {
                                let gene = genes[rng.gen_range(0..genes.len())];
                                Some((species, gene))
                            }
                            _ => None, // Event fails: no genes in chosen species
                        }
                    }
                }
            };

            let (affected_species, affected_gene) = match selection {
                Some(s) => s,
                None => continue, // Event failed, time still advances. This can happen in PerSpecies mode if we select a species with no genes, or if there are no alive species at this time.
            };

            // Determine event type
            let event_prob: f64 = rng.gen();
            if event_prob < dup_threshold {
                // Duplication
                state.handle_duplication(affected_gene, affected_species, current_time);
            } else if event_prob < transfer_threshold {
                // Transfer
                let recipient = match (transfer_alpha, lca_depths) {
                    (Some(alpha), Some(lca)) => select_transfer_recipient_assortative(
                        depths, contemporaneity, lca, current_time, affected_species, alpha, rng,
                    ),
                    _ => select_transfer_recipient(
                        depths, contemporaneity, current_time, affected_species, rng,
                    ),
                };

                if let Some(recipient_species) = recipient {
                    // Replacement transfer: find and remove victim BEFORE adding transfer
                    let is_replacement = replacement_transfer
                        .map_or(false, |p| p > 0.0 && rng.gen::<f64>() < p);
                    if is_replacement {
                        if let Some(victim) = state.random_gene_in_species(recipient_species, rng) {
                            state.handle_loss(victim, recipient_species, current_time);
                        }
                    }
                    state.handle_transfer(
                        affected_gene, affected_species, recipient_species, current_time,
                    );
                }
            } else {
                // Loss
                state.handle_loss(affected_gene, affected_species, current_time);
            }
        }

        // Check if simulation is complete
        if species_event_idx >= species_events.len() && state.total_gene_copies() == 0 {
            break;
        }
    }

    // Finalize tree
    let rec_tree = finalize_simulation(
        species_tree.clone(),
        state.gene_nodes,
        state.node_mapping,
        state.event_mapping,
        origin_species,
        origin_start_time,
    );
    Ok((rec_tree, state.events))
}

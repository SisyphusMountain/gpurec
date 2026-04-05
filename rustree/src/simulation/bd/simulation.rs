// birth-death simulation functions

use crate::node::{FlatTree, FlatNode};
use crate::simulation::utils::draw_waiting_time;
use rand::Rng;

use super::types::{BDEvent, TreeEvent};

/// Handle extinction event (backward in time: a new lineage appears that will go extinct
/// forward in time).
fn handle_d_bwd(
    nodes: &mut Vec<FlatNode>,
    active_lineages: &mut Vec<(usize, f64)>,
    events: &mut Vec<TreeEvent>,
    time: f64,
) {
    let extinct_idx = nodes.len();
    nodes.push(FlatNode {
        name: extinct_idx.to_string(),
        left_child: None,
        right_child: None,
        parent: None,
        depth: Some(time),
        length: 0.0,
        bd_event: Some(BDEvent::Extinction),
    });
    active_lineages.push((extinct_idx, time));

    events.push(TreeEvent {
        time,
        node_id: extinct_idx,
        event_type: BDEvent::Extinction,
        child1: None,
        child2: None,
    });
}

/// Handle speciation/coalescence event (backward in time: two lineages merge into one parent).
/// Returns the parent node index.
fn handle_b_bwd<R: Rng>(
    nodes: &mut Vec<FlatNode>,
    active_lineages: &mut Vec<(usize, f64)>,
    events: &mut Vec<TreeEvent>,
    time: f64,
    rng: &mut R,
) -> usize {
    // Pick two random lineages uniformly and coalesce them
    // Use swap_remove instead of remove for O(1) performance
    let idx1 = rng.gen_range(0..active_lineages.len());
    let (child1_idx, child1_start) = active_lineages.swap_remove(idx1);

    let idx2 = rng.gen_range(0..active_lineages.len());
    let (child2_idx, child2_start) = active_lineages.swap_remove(idx2);

    // Create parent node (the speciation event)
    let parent_idx = nodes.len();
    nodes.push(FlatNode {
        name: parent_idx.to_string(),
        left_child: Some(child1_idx),
        right_child: Some(child2_idx),
        parent: None,
        depth: Some(time),
        length: 0.0, // Will be set when we add its parent
        bd_event: Some(BDEvent::Speciation),
    });

    // Update children to point to parent and set branch lengths
    nodes[child1_idx].parent = Some(parent_idx);
    nodes[child1_idx].length = time - child1_start;

    nodes[child2_idx].parent = Some(parent_idx);
    nodes[child2_idx].length = time - child2_start;

    // Parent becomes an active lineage
    active_lineages.push((parent_idx, time));

    // Record speciation event
    events.push(TreeEvent {
        time,
        node_id: parent_idx,
        event_type: BDEvent::Speciation,
        child1: Some(child1_idx),
        child2: Some(child2_idx),
    });

    parent_idx
}


/// Create `n` extant species as leaf nodes at present time (depth = 0).
/// Populates `nodes`, `active_lineages`, and `events`.
fn handle_initial_nodes(
    n: usize,
    nodes: &mut Vec<FlatNode>,
    active_lineages: &mut Vec<(usize, f64)>,
    events: &mut Vec<TreeEvent>,
) {
    for i in 0..n {
        nodes.push(FlatNode {
            name: i.to_string(),
            left_child: None,
            right_child: None,
            parent: None,
            depth: Some(0.0),
            length: 0.0, // Will be set when we add the parent
            bd_event: Some(BDEvent::Leaf),
        });
        active_lineages.push((i, 0.0));

        events.push(TreeEvent {
            time: 0.0,
            node_id: i,
            event_type: BDEvent::Leaf,
            child1: None,
            child2: None,
        });
    }
}

/// Simulates a tree with a fixed number of extant species using the
/// constant rate birth-death backward process (Stadler 2011)
///
/// This implements the EBDP backward algorithm for the simple case of
/// constant rates (no mass extinctions or rate shifts).
///
/// # Arguments
/// * `n` - Number of extant species (must be > 0)
/// * `lambda` - Speciation/birth rate (must be > 0)
/// * `mu` - Extinction/death rate (must be >= 0 and < lambda)
/// * `rng` - Random number generator
///
/// # Returns
/// A tuple containing:
/// - `FlatTree` with `n` extant species simulated under the birth-death process
/// - `Vec<TreeEvent>` list of events that occurred during the simulation
///
/// Node names are integers: 0 to n-1 are extant species, n onwards are other nodes
///
/// # Errors
/// Returns an error if:
/// - `n` is 0
/// - `lambda` is not finite or not positive
/// - `mu` is not finite or negative
/// - `lambda <= mu`
///
/// # References
/// Stadler, T. (2011). Simulating trees with a fixed number of extant species.
/// Systematic Biology, 60(5), 676-684.
pub fn simulate_bd_tree_bwd<R: Rng>(n: usize, lambda: f64, mu: f64, rng: &mut R) -> Result<(FlatTree, Vec<TreeEvent>), String> {
    if n == 0 {
        return Err("Number of species must be positive".to_string());
    }
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(format!("Speciation rate must be finite and positive, got {}", lambda));
    }
    if !mu.is_finite() || mu < 0.0 {
        return Err(format!("Extinction rate must be finite and non-negative, got {}", mu));
    }
    if lambda <= mu {
        return Err(format!("Speciation rate ({}) must be strictly greater than extinction rate ({})", lambda, mu));
    }

    // Start at present (time = 0) with n isolated vertices
    let mut time = 0.0;
    let mut num_lineages = n;

    // Pre-allocate vectors based on expected tree size
    // With extinction rate mu, expected total nodes ≈ n * (1 + r) / (1 - r) where r = mu/lambda
    // For safety, we use a slightly larger estimate,
    // but we still know that the variance is very large, so we can't be too aggressive with pre-allocation.
    // We will optimize later if we need extreme speed, but this should be fine for moderate n and reasonable rates. a
    let extinction_ratio = mu / lambda;
    let expected_nodes = if extinction_ratio < 0.99 {
        (n as f64 * (1.0 + extinction_ratio) / (1.0 - extinction_ratio) * 1.2) as usize
    } else {
        n * 20 // High extinction - be conservative
    };

    // Track active lineages: (node_index, time_when_lineage_started)
    let mut active_lineages: Vec<(usize, f64)> = Vec::with_capacity(n);

    // Build the tree nodes
    let mut nodes: Vec<FlatNode> = Vec::with_capacity(expected_nodes);

    // Track events
    let mut events: Vec<TreeEvent> = Vec::with_capacity(expected_nodes);

    // Create n extant species (leaf nodes) at present
    handle_initial_nodes(n, &mut nodes, &mut active_lineages, &mut events);

    // Pre-compute constant ratios
    let total_rate = lambda + mu;
    let extinction_prob = mu / total_rate;

    // Simulate backwards in time until we reach the origin
    let mut root_idx = 0; // Will track the final root

    while num_lineages > 0 {
        let rate = total_rate * (num_lineages as f64);
        let waiting_time = draw_waiting_time(rate, rng);
        time += waiting_time;

        if rng.gen_bool(extinction_prob) {
            handle_d_bwd(&mut nodes, &mut active_lineages, &mut events, time);
            num_lineages += 1;
        } else if num_lineages == 1 {
            // Reached the origin (root of the tree)
            let (final_root, root_start) = active_lineages[0];
            nodes[final_root].length = time - root_start;
            root_idx = final_root;
            num_lineages = 0;
        } else {
            handle_b_bwd(&mut nodes, &mut active_lineages, &mut events, time, rng);
            num_lineages -= 1;
        }
    }

    let tree = FlatTree {
        nodes,
        root: root_idx,
    };

    Ok((tree, events))
}

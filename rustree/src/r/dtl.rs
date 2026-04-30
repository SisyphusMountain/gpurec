// DTL simulation R binding functions

/// Simulate a single DTL gene tree.
///
/// @param species_tree_list Species tree from simulate_species_tree_r
/// @param lambda_d Duplication rate
/// @param lambda_t Transfer rate
/// @param lambda_l Loss rate
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, retry until a tree with extant genes is produced (default FALSE)
/// @param seed Optional random seed
/// @return A list containing the gene tree data with reconciliation info
/// @export
#[extendr]
fn simulate_dtl_r(
    species_tree_list: List,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
) -> Result<List> {
    crate::bindings_common::validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;

    let species_tree = rlist_to_flattree(&species_tree_list)?;

    let mut rng = if seed.is_null() || seed.is_na() {
        StdRng::from_entropy()
    } else {
        match seed.as_integer() {
            Some(s) => StdRng::seed_from_u64(s as u64),
            None => return Err("seed must be an integer".into()),
        }
    };

    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    let (rec_tree, events) = simulate_dtl(&species_tree, origin_species, lambda_d, lambda_t, lambda_l, alpha, repl, require_extant, &mut rng)
        .map_err(Error::Other)?;

    let result = rectree_to_rlist(&species_tree, &rec_tree.gene_tree, &rec_tree.node_mapping, &rec_tree.event_mapping)?;
    result.set_attrib("dtl_events", dtl_events_to_rlist(&events, &species_tree))?;
    Ok(result)
}

/// Simulate a batch of DTL gene trees efficiently.
///
/// @param species_tree_list Species tree from simulate_species_tree_r
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate
/// @param lambda_t Transfer rate
/// @param lambda_l Loss rate
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, only include trees with extant genes (default FALSE)
/// @param seed Optional random seed
/// @return A list of gene tree lists
/// @export
#[extendr]
fn simulate_dtl_batch_r(
    species_tree_list: List,
    n: i32,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
) -> Result<List> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    crate::bindings_common::validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;

    let species_tree = rlist_to_flattree(&species_tree_list)?;

    let mut rng = if seed.is_null() || seed.is_na() {
        StdRng::from_entropy()
    } else {
        match seed.as_integer() {
            Some(s) => StdRng::seed_from_u64(s as u64),
            None => return Err("seed must be an integer".into()),
        }
    };

    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    let (rec_trees, all_events) = simulate_dtl_batch(
        &species_tree, origin_species,
        lambda_d, lambda_t, lambda_l, alpha, repl,
        n as usize, require_extant, &mut rng,
    ).map_err(Error::Other)?;

    let gene_tree_lists: Vec<List> = rec_trees
        .iter()
        .zip(all_events.iter())
        .map(|(rec_tree, events)| {
            let gt_list = rectree_to_rlist(&species_tree, &rec_tree.gene_tree, &rec_tree.node_mapping, &rec_tree.event_mapping)?;
            gt_list.set_attrib("dtl_events", dtl_events_to_rlist(events, &species_tree))?;
            Ok(gt_list)
        })
        .collect::<Result<Vec<List>>>()?;

    Ok(List::from_values(gene_tree_lists))
}

/// Simulate a single DTL gene tree using the Zombi-style per-species model.
///
/// In this model, the event rate is proportional to the number of SPECIES
/// with gene copies, NOT the number of gene copies themselves.
///
/// @param species_tree_list Species tree from simulate_species_tree_r
/// @param lambda_d Duplication rate per species per unit time
/// @param lambda_t Transfer rate per species per unit time
/// @param lambda_l Loss rate per species per unit time
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, retry until a tree with extant genes is produced (default FALSE)
/// @param seed Optional random seed
/// @return A list containing the gene tree data with reconciliation info
/// @export
#[extendr]
fn simulate_dtl_per_species_r(
    species_tree_list: List,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
) -> Result<List> {
    crate::bindings_common::validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;

    let species_tree = rlist_to_flattree(&species_tree_list)?;

    let mut rng = if seed.is_null() || seed.is_na() {
        StdRng::from_entropy()
    } else {
        match seed.as_integer() {
            Some(s) => StdRng::seed_from_u64(s as u64),
            None => return Err("seed must be an integer".into()),
        }
    };

    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    let (rec_tree, events) = simulate_dtl_per_species(&species_tree, origin_species, lambda_d, lambda_t, lambda_l, alpha, repl, require_extant, &mut rng)
        .map_err(Error::Other)?;

    let result = rectree_to_rlist(&species_tree, &rec_tree.gene_tree, &rec_tree.node_mapping, &rec_tree.event_mapping)?;
    result.set_attrib("dtl_events", dtl_events_to_rlist(&events, &species_tree))?;
    Ok(result)
}

/// Simulate a batch of DTL gene trees using the Zombi-style per-species model.
///
/// @param species_tree_list Species tree from simulate_species_tree_r
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate per species per unit time
/// @param lambda_t Transfer rate per species per unit time
/// @param lambda_l Loss rate per species per unit time
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, only include trees with extant genes (default FALSE)
/// @param seed Optional random seed
/// @return A list of gene tree lists
/// @export
#[extendr]
fn simulate_dtl_per_species_batch_r(
    species_tree_list: List,
    n: i32,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
) -> Result<List> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    crate::bindings_common::validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;

    let species_tree = rlist_to_flattree(&species_tree_list)?;

    let mut rng = if seed.is_null() || seed.is_na() {
        StdRng::from_entropy()
    } else {
        match seed.as_integer() {
            Some(s) => StdRng::seed_from_u64(s as u64),
            None => return Err("seed must be an integer".into()),
        }
    };

    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    let (rec_trees, all_events) = simulate_dtl_per_species_batch(
        &species_tree, origin_species,
        lambda_d, lambda_t, lambda_l, alpha, repl,
        n as usize, require_extant, &mut rng,
    ).map_err(Error::Other)?;

    let gene_tree_lists: Vec<List> = rec_trees
        .iter()
        .zip(all_events.iter())
        .map(|(rec_tree, events)| {
            let gt_list = rectree_to_rlist(&species_tree, &rec_tree.gene_tree, &rec_tree.node_mapping, &rec_tree.event_mapping)?;
            gt_list.set_attrib("dtl_events", dtl_events_to_rlist(events, &species_tree))?;
            Ok(gt_list)
        })
        .collect::<Result<Vec<List>>>()?;

    Ok(List::from_values(gene_tree_lists))
}

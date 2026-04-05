// Analysis & metrics R binding functions

/// Compute all pairwise distances between nodes in a tree.
///
/// @param tree_list A tree list from simulate_species_tree_r or parse_newick_r
/// @param distance_type Type of distance: "topological" (number of edges) or "metric" (sum of branch lengths)
/// @param leaves_only If TRUE, only compute distances between leaf nodes (default TRUE)
/// @return A list with three vectors: node1, node2, distance (suitable for conversion to data.frame)
/// @export
#[extendr]
fn pairwise_distances_r(tree_list: List, distance_type: &str, leaves_only: bool) -> Result<List> {
    let tree = rlist_to_flattree(&tree_list)?;

    let dist_type = crate::bindings_common::parse_distance_type(distance_type)?;

    let distances = tree.pairwise_distances(dist_type, leaves_only)
        .map_err(|e| format!("Failed to compute pairwise distances: {}", e))?;

    let node1: Vec<String> = distances.iter().map(|d| d.node1.to_string()).collect();
    let node2: Vec<String> = distances.iter().map(|d| d.node2.to_string()).collect();
    let dist: Vec<f64> = distances.iter().map(|d| d.distance).collect();

    Ok(list!(
        node1 = node1,
        node2 = node2,
        distance = dist
    ))
}

/// Save pairwise distances to a CSV file.
///
/// @param tree_list A tree list from simulate_species_tree_r or parse_newick_r
/// @param filepath Path to save the CSV file
/// @param distance_type Type of distance: "topological" or "metric"
/// @param leaves_only If TRUE, only compute distances between leaf nodes (default TRUE)
/// @export
#[extendr]
fn save_pairwise_distances_csv_r(tree_list: List, filepath: &str, distance_type: &str, leaves_only: bool) -> Result<()> {
    use crate::metric_functions::PairwiseDistance;

    let tree = rlist_to_flattree(&tree_list)?;

    let dist_type = crate::bindings_common::parse_distance_type(distance_type)?;

    let distances = tree.pairwise_distances(dist_type, leaves_only)
        .map_err(|e| format!("Failed to compute pairwise distances: {}", e))?;

    // Write to CSV
    let mut csv = String::from(PairwiseDistance::csv_header());
    csv.push('\n');
    for d in &distances {
        csv.push_str(&d.to_csv_row());
        csv.push('\n');
    }

    fs::write(filepath, csv)
        .map_err(|e| format!("Failed to write CSV file: {}", e))?;

    Ok(())
}

/// Parse a RecPhyloXML file into a gene tree with reconciliation information.
///
/// @param filepath Path to the RecPhyloXML file (e.g., from ALERax)
/// @return A gene tree list with reconciliation information
/// @export
#[extendr]
fn parse_recphyloxml_r(filepath: &str) -> Result<List> {
    let rec_tree = RecTree::from_xml_file(filepath)?;

    // Convert to R list using the rectree_to_rlist function
    rectree_to_rlist(
        &rec_tree.species_tree,
        &rec_tree.gene_tree,
        &rec_tree.node_mapping,
        &rec_tree.event_mapping,
    )
}

/// Compute induced transfers by projecting transfer events onto a sampled subtree.
///
/// Given a species tree, a set of sampled leaf names, and DTL events from
/// simulation, this function projects each transfer's donor and recipient
/// species onto the sampled subtree.
///
/// @param species_tree_list The complete species tree
/// @param sampled_leaf_names Character vector of species leaf names to keep
/// @param dtl_events_list DTL events list (from attr(gene_tree, "dtl_events"))
/// @return A data.frame with columns: time, gene_id, from_complete, to_complete,
///         from_sampled, to_sampled
/// @export
#[extendr]
fn induced_transfers_r(
    species_tree_list: List,
    sampled_leaf_names: Robj,
    dtl_events_list: List,
) -> Result<List> {
    let species_tree = rlist_to_flattree(&species_tree_list)?;

    let names: Vec<String> = sampled_leaf_names.as_str_vector()
        .ok_or("sampled_leaf_names must be a character vector")?
        .iter().map(|s| s.to_string()).collect();

    if names.is_empty() {
        return Err("sampled_leaf_names cannot be empty".into());
    }

    let events = rlist_to_dtl_events(&dtl_events_list, &species_tree)?;

    let result = induced_transfers(&species_tree, &names, &events)?;

    // Build the sampled tree to resolve sampled indices to names
    let (sampled_tree, _) = extract_induced_subtree_by_names(&species_tree, &names)
        .ok_or("Failed to extract sampled subtree")?;

    let n = result.len();
    let mut times: Vec<f64> = Vec::with_capacity(n);
    let mut gene_ids: Vec<i32> = Vec::with_capacity(n);
    let mut from_complete: Vec<String> = Vec::with_capacity(n);
    let mut to_complete: Vec<String> = Vec::with_capacity(n);
    let mut from_sampled: Vec<Rstr> = Vec::with_capacity(n);
    let mut to_sampled: Vec<Rstr> = Vec::with_capacity(n);

    for it in &result {
        times.push(it.time);
        gene_ids.push(it.gene_id as i32);
        from_complete.push(species_tree.nodes[it.from_species_complete].name.clone());
        to_complete.push(species_tree.nodes[it.to_species_complete].name.clone());
        from_sampled.push(match it.from_species_sampled {
            Some(idx) => Rstr::from(sampled_tree.nodes[idx].name.clone()),
            None => Rstr::na(),
        });
        to_sampled.push(match it.to_species_sampled {
            Some(idx) => Rstr::from(sampled_tree.nodes[idx].name.clone()),
            None => Rstr::na(),
        });
    }

    Ok(list!(
        time = times,
        gene_id = gene_ids,
        from_complete = from_complete,
        to_complete = to_complete,
        from_sampled = from_sampled,
        to_sampled = to_sampled
    ))
}

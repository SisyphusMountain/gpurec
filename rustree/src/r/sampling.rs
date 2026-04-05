// Sampling R binding functions

/// Sample extant genes from a gene tree.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r
/// @return A new gene tree containing only extant genes
/// @export
#[extendr]
fn sample_extant_r(gene_tree_list: List) -> Result<List> {
    let (gene_tree, species_tree, node_mapping, event_mapping) = rlist_to_genetree(&gene_tree_list)?;

    let extant_indices: std::collections::HashSet<usize> = gene_tree.nodes.iter()
        .enumerate()
        .filter(|(i, n)| {
            n.left_child.is_none()
            && n.right_child.is_none()
            && event_mapping[*i] == Event::Leaf
        })
        .map(|(i, _)| i)
        .collect();

    if extant_indices.is_empty() {
        return Err("No extant genes to sample".into());
    }

    let (sampled_tree, gene_old_to_new) = extract_induced_subtree(&gene_tree, &extant_indices)
        .ok_or("Failed to extract induced subtree")?;

    let species_identity: Vec<Option<usize>> = (0..species_tree.nodes.len())
        .map(Some)
        .collect();

    let (new_node_mapping, new_event_mapping) = remap_gene_tree_indices(
        &sampled_tree, &gene_old_to_new,
        &node_mapping, &event_mapping,
        &species_identity,
    ).map_err(|e| format!("Remap failed: {}", e))?;

    rectree_to_rlist(&species_tree, &sampled_tree, &new_node_mapping, &new_event_mapping)
}

/// Sample leaves and filter all trees accordingly.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r or parse_recphyloxml_r
/// @param species_leaf_names Character vector of species leaf names to keep
/// @return A new gene tree with sampled species and gene trees
/// @export
#[extendr]
fn sample_leaves_r(gene_tree_list: List, species_leaf_names: Robj) -> Result<List> {
    let species_names: Vec<String> = if species_leaf_names.is_string() {
        species_leaf_names.as_str_vector()
            .ok_or("Failed to convert species_leaf_names to string vector")?
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        return Err("species_leaf_names must be a character vector".into());
    };

    let (gene_tree, species_tree, node_mapping, event_mapping) = rlist_to_genetree(&gene_tree_list)?;

    let forest = GeneForest::with_gene_trees(species_tree, vec![(gene_tree, node_mapping, event_mapping)]);
    let sampled_forest = forest.sample_leaves(&species_names)
        .map_err(|e| format!("Failed to sample leaves: {}", e))?;
    let sampled = sampled_forest.get(0)
        .ok_or("No gene trees in sampled result")?;

    rectree_to_rlist(
        &sampled.species_tree,
        &sampled.gene_tree,
        &sampled.node_mapping,
        &sampled.event_mapping,
    )
}

/// Extract an induced subtree keeping only specified leaves.
///
/// @param tree_list A tree list (species tree or gene tree)
/// @param leaf_names Character vector of leaf names to keep
/// @return A new tree containing only the specified leaves and their MRCAs
/// @export
#[extendr]
fn extract_induced_subtree_by_names_r(tree_list: List, leaf_names: Robj) -> Result<List> {
    let names: Vec<String> = leaf_names.as_str_vector()
        .ok_or("leaf_names must be a character vector")?
        .iter()
        .map(|s| s.to_string())
        .collect();

    if names.is_empty() {
        return Err("leaf_names cannot be empty".into());
    }

    let tree = rlist_to_flattree(&tree_list)?;

    let (induced_tree, _) = extract_induced_subtree_by_names(&tree, &names)
        .ok_or("Failed to extract induced subtree (no matching leaves found)")?;

    Ok(flattree_to_rlist(&induced_tree))
}

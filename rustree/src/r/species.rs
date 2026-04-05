// Species tree & gene tree R binding functions

/// Simulate a birth-death species tree.
///
/// @param n Number of extant species
/// @param lambda Speciation rate (birth rate)
/// @param mu Extinction rate (death rate), must be less than lambda
/// @param seed Optional random seed for reproducibility
/// @return A list containing the tree data
/// @export
#[extendr]
fn simulate_species_tree_r(n: i32, lambda: f64, mu: f64, seed: Robj) -> Result<List> {
    if n <= 0 {
        return Err("Number of species must be positive".into());
    }
    if lambda <= 0.0 {
        return Err("Speciation rate must be positive".into());
    }
    if mu < 0.0 {
        return Err("Extinction rate must be non-negative".into());
    }
    if lambda <= mu {
        return Err("Speciation rate must be greater than extinction rate".into());
    }

    let mut rng = if seed.is_null() || seed.is_na() {
        StdRng::from_entropy()
    } else {
        match seed.as_integer() {
            Some(s) => StdRng::seed_from_u64(s as u64),
            None => return Err("seed must be an integer".into()),
        }
    };

    let (mut tree, events) = simulate_bd_tree_bwd(n as usize, lambda, mu, &mut rng)
        .map_err(Error::Other)?;
    tree.assign_depths();

    let names: Vec<String> = tree.nodes.iter().map(|n| n.name.clone()).collect();
    let parents: Vec<Rint> = tree.nodes.iter()
        .map(|n| n.parent.map(|p| Rint::from(p as i32)).unwrap_or(Rint::na()))
        .collect();
    let left_children: Vec<Rint> = tree.nodes.iter()
        .map(|n| n.left_child.map(|c| Rint::from(c as i32)).unwrap_or(Rint::na()))
        .collect();
    let right_children: Vec<Rint> = tree.nodes.iter()
        .map(|n| n.right_child.map(|c| Rint::from(c as i32)).unwrap_or(Rint::na()))
        .collect();
    let lengths: Vec<f64> = tree.nodes.iter().map(|n| n.length).collect();
    let depths: Vec<Rfloat> = tree.nodes.iter()
        .map(|n| n.depth.map(Rfloat::from).unwrap_or(Rfloat::na()))
        .collect();
    let bd_events: Vec<Rstr> = tree.nodes.iter()
        .map(|n| match n.bd_event {
            Some(e) => Rstr::from(e.as_str()),
            None => Rstr::na(),
        })
        .collect();

    let events_list = bd_events_to_rlist(&events);

    Ok(list!(
        name = names,
        parent = parents,
        left_child = left_children,
        right_child = right_children,
        length = lengths,
        depth = depths,
        root = tree.root as i32,
        bd_event = bd_events,
        events = events_list
    ))
}

/// Parse a Newick string into a species tree.
///
/// @param newick_str A Newick formatted string
/// @return A list containing the tree data
/// @export
#[extendr]
fn parse_newick_r(newick_str: &str) -> Result<List> {
    use crate::newick::parse_newick;

    let mut nodes = parse_newick(newick_str)
        .map_err(|e| format!("Failed to parse Newick: {}", e))?;

    let mut root = nodes.pop()
        .ok_or("No tree found in Newick string")?;

    root.zero_root_length();
    root.assign_depths(0.0);
    let tree = root.to_flat_tree();

    Ok(flattree_to_rlist(&tree))
}

/// Convert a species tree to Newick format.
///
/// @param tree_list A tree list from simulate_species_tree_r or parse_newick_r
/// @return Newick string representation
/// @export
#[extendr]
fn tree_to_newick_r(tree_list: List) -> Result<String> {
    let tree = rlist_to_flattree(&tree_list)?;
    let nwk = tree.to_newick()?;
    Ok(nwk + ";")
}

/// Get the number of leaves in a tree.
///
/// @param tree_list A tree list
/// @return Number of leaves
/// @export
#[extendr]
fn tree_num_leaves_r(tree_list: List) -> Result<i32> {
    let tree = rlist_to_flattree(&tree_list)?;
    let count = tree.nodes.iter()
        .filter(|n| n.left_child.is_none() && n.right_child.is_none())
        .count();
    Ok(count as i32)
}

/// Get leaf names from a tree.
///
/// @param tree_list A tree list
/// @return Character vector of leaf names
/// @export
#[extendr]
fn tree_leaf_names_r(tree_list: List) -> Result<Vec<String>> {
    let tree = rlist_to_flattree(&tree_list)?;
    let names: Vec<String> = tree.nodes.iter()
        .filter(|n| n.left_child.is_none() && n.right_child.is_none())
        .map(|n| n.name.clone())
        .collect();
    Ok(names)
}

/// Name unnamed internal nodes as `internal0`, `internal1`, etc.
///
/// Assigns names to internal nodes (nodes with children) that have empty names.
/// Errors if any existing node name starts with "internal".
///
/// @param tree_list A tree list from parse_newick or simulate_species_tree
/// @return The tree with internal nodes named
/// @export
#[extendr]
fn name_internal_nodes_r(tree_list: List) -> Result<List> {
    let mut tree = rlist_to_flattree(&tree_list)?;
    tree.name_internal_nodes()?;
    Ok(flattree_to_rlist(&tree))
}

/// Get the number of extant genes in a gene tree.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r
/// @return Number of extant genes (non-loss leaves)
/// @export
#[extendr]
fn gene_tree_num_extant_r(gene_tree_list: List) -> Result<i32> {
    let events: Vec<String> = gene_tree_list.dollar("event")?.as_str_vector()
        .ok_or("Failed to get event column")?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let left_children: Vec<i32> = gene_tree_list.dollar("left_child")?.as_integer_vector()
        .ok_or("Failed to get left_child column")?;

    let count = events.iter()
        .zip(left_children.iter())
        .filter(|(e, lc)| *e == "Leaf" && lc.is_na())
        .count();

    Ok(count as i32)
}

/// Convert a gene tree to Newick format.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r
/// @return Newick string representation
/// @export
#[extendr]
fn gene_tree_to_newick_r(gene_tree_list: List) -> Result<String> {
    let (gene_tree, _, _, _) = rlist_to_genetree(&gene_tree_list)?;
    let nwk = gene_tree.to_newick()?;
    Ok(nwk + ";")
}

/// Export a gene tree to RecPhyloXML format.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r
/// @return RecPhyloXML string
/// @export
#[extendr]
fn gene_tree_to_xml_r(gene_tree_list: List) -> Result<String> {
    let (gene_tree, species_tree, node_mapping, event_mapping) = rlist_to_genetree(&gene_tree_list)?;
    let rec_tree = RecTree::new_owned(species_tree, gene_tree, node_mapping, event_mapping);
    Ok(rec_tree.to_xml())
}

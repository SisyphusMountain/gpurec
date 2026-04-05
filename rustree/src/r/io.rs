// File I/O R binding functions

/// Save a tree to a Newick file.
///
/// @param tree_list A tree list (species or gene tree)
/// @param filepath Path to save the file
/// @export
#[extendr]
fn save_newick_r(tree_list: List, filepath: &str) -> Result<()> {
    let is_gene_tree = tree_list.dollar("species_node")
        .map(|robj| !robj.is_null())
        .unwrap_or(false);

    let newick = if is_gene_tree {
        gene_tree_to_newick_r(tree_list)?
    } else {
        tree_to_newick_r(tree_list)?
    };

    fs::write(filepath, newick)
        .map_err(|e| format!("Failed to write file: {}", e))?;
    Ok(())
}

/// Save a gene tree to RecPhyloXML format.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r
/// @param filepath Path to save the XML file
/// @export
#[extendr]
fn save_xml_r(gene_tree_list: List, filepath: &str) -> Result<()> {
    let xml = gene_tree_to_xml_r(gene_tree_list)?;
    fs::write(filepath, xml)
        .map_err(|e| format!("Failed to write file: {}", e))?;
    Ok(())
}

/// Save a gene tree to CSV format.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r
/// @param filepath Path to save the CSV file
/// @export
#[extendr]
fn save_csv_r(gene_tree_list: List, filepath: &str) -> Result<()> {
    let (gene_tree, species_tree, node_mapping, event_mapping) = rlist_to_genetree(&gene_tree_list)?;
    let rec_tree = RecTree::new_owned(species_tree, gene_tree, node_mapping, event_mapping);
    rec_tree.save_csv(filepath)
        .map_err(|e| extendr_api::Error::Other(format!("Failed to write CSV: {}", e)))?;
    Ok(())
}

/// Save birth-death events from a species tree to CSV format.
///
/// @param species_tree_list A species tree list from simulate_species_tree_r or parse_newick_r
/// @param filepath Path to save the CSV file
/// @export
#[extendr]
fn save_bd_events_csv_r(species_tree_list: List, filepath: &str) -> Result<()> {
    let tree = rlist_to_flattree(&species_tree_list)?;

    let events = match species_tree_list.dollar("events") {
        Ok(events_robj) if !events_robj.is_null() => {
            let events_list: List = events_robj
                .try_into()
                .map_err(|_| "Failed to parse events list")?;
            rlist_to_bd_events(&events_list)?
        }
        _ => {
            generate_events_from_tree(&tree).map_err(Error::Other)?
        }
    };

    use crate::io::save_bd_events_to_csv;
    save_bd_events_to_csv(&events, &tree, filepath)
        .map_err(|e| format!("Failed to write CSV file: {}", e))?;

    Ok(())
}

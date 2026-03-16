//! R bindings for rustree using extendr.
//!
//! Provides R access to:
//! - Newick tree parsing
//! - Birth-death species tree simulation
//! - DTL gene tree simulation
//! - Export to Newick, XML, CSV
//! - SVG visualization via thirdkind

use extendr_api::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs;
use std::process::Command;

use crate::bd::{simulate_bd_tree_bwd, generate_events_from_tree, TreeEvent, BDEvent};
use crate::dtl::{simulate_dtl, simulate_dtl_batch, simulate_dtl_per_species, simulate_dtl_per_species_batch};
use crate::dtl::{simulate_dtl_iter, simulate_dtl_per_species_iter};
use crate::node::{FlatTree, FlatNode, Event, RecTree, GeneForest};
use crate::sampling::{extract_induced_subtree, extract_induced_subtree_by_names};

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

    let (mut tree, events) = simulate_bd_tree_bwd(n as usize, lambda, mu, &mut rng);
    tree.assign_depths();

    // Serialize tree and events to R list
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
        .map(|n| n.depth.map(|d| Rfloat::from(d)).unwrap_or(Rfloat::na()))
        .collect();
    // bd_event: convert Option<BDEvent> to character vector for R
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
    use crate::newick::newick::parse_newick;

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
    let nwk = tree.to_newick().map_err(|e| Error::Other(e))?;
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

/// Simulate a single DTL gene tree.
///
/// @param species_tree_list Species tree from simulate_species_tree_r
/// @param lambda_d Duplication rate
/// @param lambda_t Transfer rate
/// @param lambda_l Loss rate
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
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
    require_extant: bool,
    seed: Robj,
) -> Result<List> {
    if lambda_d < 0.0 || lambda_t < 0.0 || lambda_l < 0.0 {
        return Err("Rates must be non-negative".into());
    }

    let species_tree = rlist_to_flattree(&species_tree_list)?;

    let mut rng = if seed.is_null() || seed.is_na() {
        StdRng::from_entropy()
    } else {
        match seed.as_integer() {
            Some(s) => StdRng::seed_from_u64(s as u64),
            None => return Err("seed must be an integer".into()),
        }
    };

    let alpha = if transfer_alpha.is_null() || transfer_alpha.is_na() {
        None
    } else {
        Some(transfer_alpha.as_real().ok_or("transfer_alpha must be a number")?)
    };

    let origin_species = species_tree.root;
    let (rec_tree, _events) = simulate_dtl(&species_tree, origin_species, lambda_d, lambda_t, lambda_l, alpha, None, require_extant, &mut rng)
        .map_err(|e| Error::Other(e))?;

    rectree_to_rlist(&species_tree, &rec_tree.gene_tree, &rec_tree.node_mapping, &rec_tree.event_mapping)
}

/// Simulate a batch of DTL gene trees efficiently.
///
/// @param species_tree_list Species tree from simulate_species_tree_r
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate
/// @param lambda_t Transfer rate
/// @param lambda_l Loss rate
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
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
    require_extant: bool,
    seed: Robj,
) -> Result<List> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    if lambda_d < 0.0 || lambda_t < 0.0 || lambda_l < 0.0 {
        return Err("Rates must be non-negative".into());
    }

    let species_tree = rlist_to_flattree(&species_tree_list)?;

    let mut rng = if seed.is_null() || seed.is_na() {
        StdRng::from_entropy()
    } else {
        match seed.as_integer() {
            Some(s) => StdRng::seed_from_u64(s as u64),
            None => return Err("seed must be an integer".into()),
        }
    };

    let alpha = if transfer_alpha.is_null() || transfer_alpha.is_na() {
        None
    } else {
        Some(transfer_alpha.as_real().ok_or("transfer_alpha must be a number")?)
    };

    let origin_species = species_tree.root;
    let (rec_trees, _all_events) = simulate_dtl_batch(
        &species_tree,
        origin_species,
        lambda_d,
        lambda_t,
        lambda_l,
        alpha,
        None,
        n as usize,
        require_extant,
        &mut rng,
    ).map_err(|e| Error::Other(e))?;

    // Convert to list of lists
    let gene_tree_lists: Vec<List> = rec_trees
        .iter()
        .map(|rec_tree| {
            rectree_to_rlist(&species_tree, &rec_tree.gene_tree, &rec_tree.node_mapping, &rec_tree.event_mapping)
        })
        .collect::<Result<Vec<List>>>()?;

    Ok(List::from_values(gene_tree_lists))
}

/// Simulate a single DTL gene tree using the Zombi-style per-species model.
///
/// In this model, the event rate is proportional to the number of SPECIES
/// with gene copies, NOT the number of gene copies themselves. This means
/// duplications don't increase the event rate.
///
/// @param species_tree_list Species tree from simulate_species_tree_r
/// @param lambda_d Duplication rate per species per unit time
/// @param lambda_t Transfer rate per species per unit time
/// @param lambda_l Loss rate per species per unit time
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
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
    require_extant: bool,
    seed: Robj,
) -> Result<List> {
    if lambda_d < 0.0 || lambda_t < 0.0 || lambda_l < 0.0 {
        return Err("Rates must be non-negative".into());
    }

    let species_tree = rlist_to_flattree(&species_tree_list)?;

    let mut rng = if seed.is_null() || seed.is_na() {
        StdRng::from_entropy()
    } else {
        match seed.as_integer() {
            Some(s) => StdRng::seed_from_u64(s as u64),
            None => return Err("seed must be an integer".into()),
        }
    };

    let alpha = if transfer_alpha.is_null() || transfer_alpha.is_na() {
        None
    } else {
        Some(transfer_alpha.as_real().ok_or("transfer_alpha must be a number")?)
    };

    let origin_species = species_tree.root;
    let (rec_tree, _events) = simulate_dtl_per_species(&species_tree, origin_species, lambda_d, lambda_t, lambda_l, alpha, None, require_extant, &mut rng)
        .map_err(|e| Error::Other(e))?;

    rectree_to_rlist(&species_tree, &rec_tree.gene_tree, &rec_tree.node_mapping, &rec_tree.event_mapping)
}

/// Simulate a batch of DTL gene trees using the Zombi-style per-species model.
///
/// This is faster than calling simulate_dtl_per_species_r multiple times because
/// species tree events and LCA depths are computed only once.
///
/// @param species_tree_list Species tree from simulate_species_tree_r
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate per species per unit time
/// @param lambda_t Transfer rate per species per unit time
/// @param lambda_l Loss rate per species per unit time
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
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
    require_extant: bool,
    seed: Robj,
) -> Result<List> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    if lambda_d < 0.0 || lambda_t < 0.0 || lambda_l < 0.0 {
        return Err("Rates must be non-negative".into());
    }

    let species_tree = rlist_to_flattree(&species_tree_list)?;

    let mut rng = if seed.is_null() || seed.is_na() {
        StdRng::from_entropy()
    } else {
        match seed.as_integer() {
            Some(s) => StdRng::seed_from_u64(s as u64),
            None => return Err("seed must be an integer".into()),
        }
    };

    let alpha = if transfer_alpha.is_null() || transfer_alpha.is_na() {
        None
    } else {
        Some(transfer_alpha.as_real().ok_or("transfer_alpha must be a number")?)
    };

    let origin_species = species_tree.root;
    let (rec_trees, _all_events) = simulate_dtl_per_species_batch(
        &species_tree,
        origin_species,
        lambda_d,
        lambda_t,
        lambda_l,
        alpha,
        None,
        n as usize,
        require_extant,
        &mut rng,
    ).map_err(|e| Error::Other(e))?;

    // Convert to list of lists
    let gene_tree_lists: Vec<List> = rec_trees
        .iter()
        .map(|rec_tree| {
            rectree_to_rlist(&species_tree, &rec_tree.gene_tree, &rec_tree.node_mapping, &rec_tree.event_mapping)
        })
        .collect::<Result<Vec<List>>>()?;

    Ok(List::from_values(gene_tree_lists))
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
    let nwk = gene_tree.to_newick().map_err(|e| Error::Other(e))?;
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

/// Save a tree to a Newick file.
///
/// @param tree_list A tree list (species or gene tree)
/// @param filepath Path to save the file
/// @export
#[extendr]
fn save_newick_r(tree_list: List, filepath: &str) -> Result<()> {
    // Check for species_node field to determine if this is a gene tree
    // Gene trees have species_node, species trees do not
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
    let (species_tree, gene_tree, node_mapping, event_mapping) = rlist_to_genetree(&gene_tree_list)?;
    let rec_tree = RecTree::new_owned(species_tree, gene_tree, node_mapping, event_mapping);
    rec_tree.save_csv(filepath)
        .map_err(|e| extendr_api::Error::Other(format!("Failed to write CSV: {}", e)))?;
    Ok(())
}

/// Save birth-death events from a species tree to CSV format.
///
/// For trees from simulate_species_tree, saves the actual simulation events.
/// For trees from parse_newick, generates events by treating internal nodes
/// as speciations and leaves as extant species (no extinction events).
///
/// @param species_tree_list A species tree list from simulate_species_tree_r or parse_newick_r
/// @param filepath Path to save the CSV file
/// @export
#[extendr]
fn save_bd_events_csv_r(species_tree_list: List, filepath: &str) -> Result<()> {
    // Get the tree structure first (needed in both cases)
    let tree = rlist_to_flattree(&species_tree_list)?;

    // Try to get events from the tree (exists for simulated trees)
    let events = match species_tree_list.dollar("events") {
        Ok(events_robj) if !events_robj.is_null() => {
            // Tree has events (from simulate_species_tree)
            let events_list: List = events_robj
                .try_into()
                .map_err(|_| "Failed to parse events list")?;
            rlist_to_bd_events(&events_list)?
        }
        _ => {
            // No events (from parse_newick) - generate them from tree structure
            generate_events_from_tree(&tree).map_err(|e| Error::Other(e))?
        }
    };

    // Call the existing Rust function
    use crate::io::save_bd_events_to_csv;
    save_bd_events_to_csv(&events, &tree, filepath)
        .map_err(|e| format!("Failed to write CSV file: {}", e))?;

    Ok(())
}

/// Generate SVG visualization using thirdkind.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r
/// @param filepath Optional path to save the SVG (NULL to return string only)
/// @param open_browser Whether to open in browser (default FALSE)
/// @return SVG content as string
/// @export
#[extendr]
fn gene_tree_to_svg_r(gene_tree_list: List, filepath: Robj, open_browser: bool) -> Result<String> {
    let xml = gene_tree_to_xml_r(gene_tree_list)?;

    let temp_dir = std::env::temp_dir();
    let xml_path = temp_dir.join("rustree_temp.recphyloxml");
    let svg_path = temp_dir.join("rustree_temp.svg");

    fs::write(&xml_path, &xml)
        .map_err(|e| format!("Failed to write temp XML: {}", e))?;

    let mut cmd = Command::new("thirdkind");
    cmd.arg("-f").arg(&xml_path)
       .arg("-o").arg(&svg_path);

    if open_browser {
        cmd.arg("-b");
    }

    let output = cmd.output()
        .map_err(|e| format!("Failed to run thirdkind. Is it installed? (`cargo install thirdkind`)\nError: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("thirdkind failed: {}", stderr).into());
    }

    let svg = fs::read_to_string(&svg_path)
        .map_err(|e| format!("Failed to read SVG output: {}", e))?;

    // Save to user-specified path if provided
    if !filepath.is_null() {
        let path = filepath.as_str().ok_or("filepath must be a string")?;
        fs::write(path, &svg)
            .map_err(|e| format!("Failed to write SVG file: {}", e))?;
    }

    // Cleanup
    let _ = fs::remove_file(&xml_path);
    let _ = fs::remove_file(&svg_path);

    Ok(svg)
}

/// Sample extant genes from a gene tree.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r
/// @return A new gene tree containing only extant genes
/// @export
#[extendr]
fn sample_extant_r(gene_tree_list: List) -> Result<List> {
    let (gene_tree, species_tree, _node_mapping, event_mapping) = rlist_to_genetree(&gene_tree_list)?;

    // Find extant gene indices
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

    let (sampled_tree, old_to_new) = extract_induced_subtree(&gene_tree, &extant_indices)
        .ok_or("Failed to extract induced subtree")?;

    // Build new→old mapping by inverting old_to_new
    let mut new_to_old: Vec<Option<usize>> = vec![None; sampled_tree.nodes.len()];
    for (old_idx, new_idx_opt) in old_to_new.iter().enumerate() {
        if let Some(new_idx) = new_idx_opt {
            new_to_old[*new_idx] = Some(old_idx);
        }
    }

    // Rebuild event mapping using index-based lookup
    let new_event_mapping: Vec<Event> = new_to_old.iter()
        .enumerate()
        .map(|(new_idx, old_idx_opt)| {
            if let Some(old_idx) = old_idx_opt {
                event_mapping[*old_idx].clone()
            } else {
                if sampled_tree.nodes[new_idx].left_child.is_none() {
                    Event::Leaf
                } else {
                    Event::Speciation
                }
            }
        })
        .collect();

    let new_node_mapping: Vec<Option<usize>> = sampled_tree.nodes.iter()
        .map(|n| {
            if n.left_child.is_none() && n.right_child.is_none() {
                if let Some(pos) = n.name.rfind('_') {
                    let species_name = &n.name[..pos];
                    species_tree.nodes.iter()
                        .position(|sn| sn.name == species_name)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    rectree_to_rlist(&species_tree, &sampled_tree, &new_node_mapping, &new_event_mapping)
}

/// Sample leaves and filter all trees accordingly.
///
/// This function samples a subset of species from the species tree and automatically
/// filters the gene tree to keep only genes that map to the sampled species. The
/// reconciliation mappings are preserved using an LCA-based approach.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r or parse_recphyloxml_r
/// @param species_leaf_names Character vector of species leaf names to keep
/// @return A new gene tree with sampled species and gene trees
/// @export
#[extendr]
fn sample_leaves_r(gene_tree_list: List, species_leaf_names: Robj) -> Result<List> {
    // Convert species_leaf_names to Vec<String>
    let species_names: Vec<String> = if species_leaf_names.is_string() {
        species_leaf_names.as_str_vector()
            .ok_or("Failed to convert species_leaf_names to string vector")?
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        return Err("species_leaf_names must be a character vector".into());
    };

    // Parse gene tree list
    let (gene_tree, species_tree, node_mapping, event_mapping) = rlist_to_genetree(&gene_tree_list)?;

    // Build a single-tree GeneForest and sample
    let forest = GeneForest::with_gene_trees(species_tree, vec![(gene_tree, node_mapping, event_mapping)]);
    let sampled_forest = forest.sample_leaves(&species_names)
        .map_err(|e| format!("Failed to sample leaves: {}", e))?;
    let sampled = sampled_forest.get(0)
        .ok_or("No gene trees in sampled result")?;

    // Convert back to R list
    rectree_to_rlist(
        &sampled.species_tree,
        &sampled.gene_tree,
        &sampled.node_mapping,
        &sampled.event_mapping,
    )
}

/// Extract an induced subtree keeping only specified leaves.
///
/// This function works on both species trees and gene trees.
/// For species trees, returns a pruned tree with only the specified leaves.
/// For gene trees, returns the pruned gene tree (reconciliation info is not preserved).
///
/// @param tree_list A tree list (species tree or gene tree)
/// @param leaf_names Character vector of leaf names to keep
/// @return A new tree containing only the specified leaves and their MRCAs
/// @export
#[extendr]
fn extract_induced_subtree_by_names_r(tree_list: List, leaf_names: Robj) -> Result<List> {
    // Convert leaf_names to Vec<String>
    let names: Vec<String> = leaf_names.as_str_vector()
        .ok_or("leaf_names must be a character vector")?
        .iter()
        .map(|s| s.to_string())
        .collect();

    if names.is_empty() {
        return Err("leaf_names cannot be empty".into());
    }

    // Extract the tree structure (works for both species and gene trees)
    let tree = rlist_to_flattree(&tree_list)?;

    // Extract the induced subtree
    let (induced_tree, _) = extract_induced_subtree_by_names(&tree, &names)
        .ok_or("Failed to extract induced subtree (no matching leaves found)")?;

    // Return as a simple tree (no reconciliation info)
    Ok(flattree_to_rlist(&induced_tree))
}

// Helper functions

fn flattree_to_rlist(tree: &FlatTree) -> List {
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
        .map(|n| n.depth.map(|d| Rfloat::from(d)).unwrap_or(Rfloat::na()))
        .collect();
    // bd_event: convert Option<BDEvent> to Rstr for proper NA handling in R
    let bd_events: Vec<Rstr> = tree.nodes.iter()
        .map(|n| match n.bd_event {
            Some(e) => Rstr::from(e.as_str()),
            None => Rstr::na(),
        })
        .collect();

    list!(
        name = names,
        parent = parents,
        left_child = left_children,
        right_child = right_children,
        length = lengths,
        depth = depths,
        root = tree.root as i32,
        bd_event = bd_events
    )
}

fn rlist_to_flattree(list: &List) -> Result<FlatTree> {
    let names: Vec<String> = list.dollar("name")?.as_str_vector()
        .ok_or("Failed to get name column")?
        .iter().map(|s| s.to_string()).collect();
    let parents: Vec<i32> = list.dollar("parent")?.as_integer_vector()
        .ok_or("Failed to get parent column")?;
    let left_children: Vec<i32> = list.dollar("left_child")?.as_integer_vector()
        .ok_or("Failed to get left_child column")?;
    let right_children: Vec<i32> = list.dollar("right_child")?.as_integer_vector()
        .ok_or("Failed to get right_child column")?;
    let lengths: Vec<f64> = list.dollar("length")?.as_real_vector()
        .ok_or("Failed to get length column")?;
    let depths: Vec<f64> = list.dollar("depth")?.as_real_vector()
        .ok_or("Failed to get depth column")?;
    let root: i32 = list.dollar("root")?.as_integer()
        .ok_or("Failed to get root")?;

    // Try to get bd_event (optional, for backward compatibility)
    // Use Rstr to properly handle NA values
    let bd_events_robj = list.dollar("bd_event").ok();

    let nodes: Vec<FlatNode> = (0..names.len())
        .map(|i| {
            let bd_event = bd_events_robj.as_ref().and_then(|robj| {
                // Get the i-th element as Rstr to check for NA
                if let Some(str_iter) = robj.as_str_iter() {
                    let strs: Vec<&str> = str_iter.collect();
                    if i < strs.len() {
                        let s = strs[i];
                        // Check if it's NA (empty or "NA" string indicates NA from R)
                        if s.is_empty() || s == "NA" {
                            return None;
                        }
                        return BDEvent::from_str(s);
                    }
                }
                None
            });
            FlatNode {
                name: names[i].clone(),
                parent: if parents[i].is_na() { None } else { Some(parents[i] as usize) },
                left_child: if left_children[i].is_na() { None } else { Some(left_children[i] as usize) },
                right_child: if right_children[i].is_na() { None } else { Some(right_children[i] as usize) },
                length: lengths[i],
                depth: if depths[i].is_na() { None } else { Some(depths[i]) },
                bd_event,
            }
        })
        .collect();

    Ok(FlatTree {
        nodes,
        root: root as usize,
    })
}

fn bd_events_to_rlist(events: &[TreeEvent]) -> List {
    let times: Vec<f64> = events.iter().map(|e| e.time).collect();
    let node_ids: Vec<i32> = events.iter().map(|e| e.node_id as i32).collect();
    let event_types: Vec<String> = events.iter().map(|e| e.event_type.as_str().to_string()).collect();
    let child1s: Vec<Rint> = events.iter()
        .map(|e| e.child1.map(|c| Rint::from(c as i32)).unwrap_or(Rint::na()))
        .collect();
    let child2s: Vec<Rint> = events.iter()
        .map(|e| e.child2.map(|c| Rint::from(c as i32)).unwrap_or(Rint::na()))
        .collect();

    list!(
        time = times,
        node_id = node_ids,
        event_type = event_types,
        child1 = child1s,
        child2 = child2s
    )
}

fn rlist_to_bd_events(list: &List) -> Result<Vec<TreeEvent>> {
    let times: Vec<f64> = list.dollar("time")?.as_real_vector()
        .ok_or("Failed to get time column")?;
    let node_ids: Vec<i32> = list.dollar("node_id")?.as_integer_vector()
        .ok_or("Failed to get node_id column")?;
    let event_types: Vec<String> = list.dollar("event_type")?.as_str_vector()
        .ok_or("Failed to get event_type column")?
        .iter().map(|s| s.to_string()).collect();
    let child1s: Vec<i32> = list.dollar("child1")?.as_integer_vector()
        .ok_or("Failed to get child1 column")?;
    let child2s: Vec<i32> = list.dollar("child2")?.as_integer_vector()
        .ok_or("Failed to get child2 column")?;

    let events: Vec<TreeEvent> = (0..times.len())
        .map(|i| {
            let event_type = BDEvent::from_str(&event_types[i])
                .ok_or_else(|| format!(
                    "Unknown BD event type '{}' at index {}. Expected one of: Speciation, Extinction, Leaf",
                    event_types[i], i
                ))?;
            Ok(TreeEvent {
                time: times[i],
                node_id: node_ids[i] as usize,
                event_type,
                child1: if child1s[i].is_na() { None } else { Some(child1s[i] as usize) },
                child2: if child2s[i].is_na() { None } else { Some(child2s[i] as usize) },
            })
        })
        .collect::<std::result::Result<Vec<TreeEvent>, String>>()
        .map_err(|e| extendr_api::Error::Other(e))?;

    Ok(events)
}

fn rectree_to_rlist(species_tree: &FlatTree, gene_tree: &FlatTree, node_mapping: &[Option<usize>], event_mapping: &[Event]) -> Result<List> {
    let names: Vec<String> = gene_tree.nodes.iter().map(|n| n.name.clone()).collect();
    let parents: Vec<Rint> = gene_tree.nodes.iter()
        .map(|n| n.parent.map(|p| Rint::from(p as i32)).unwrap_or(Rint::na()))
        .collect();
    let left_children: Vec<Rint> = gene_tree.nodes.iter()
        .map(|n| n.left_child.map(|c| Rint::from(c as i32)).unwrap_or(Rint::na()))
        .collect();
    let right_children: Vec<Rint> = gene_tree.nodes.iter()
        .map(|n| n.right_child.map(|c| Rint::from(c as i32)).unwrap_or(Rint::na()))
        .collect();
    let lengths: Vec<f64> = gene_tree.nodes.iter().map(|n| n.length).collect();
    let depths: Vec<Rfloat> = gene_tree.nodes.iter()
        .map(|n| n.depth.map(|d| Rfloat::from(d)).unwrap_or(Rfloat::na()))
        .collect();

    // Validate all node_mapping indices before using them
    for (gene_idx, opt_idx) in node_mapping.iter().enumerate() {
        if let Some(idx) = opt_idx {
            if *idx >= species_tree.nodes.len() {
                return Err(Error::Other(format!(
                    "Invalid species node index {} in node_mapping at gene node {}. Species tree has {} nodes.",
                    idx, gene_idx, species_tree.nodes.len()
                )));
            }
        }
    }

    let species_nodes: Vec<String> = node_mapping.iter()
        .map(|opt_idx| match opt_idx {
            Some(idx) => species_tree.nodes[*idx].name.clone(),
            None => "NA".to_string(),
        })
        .collect();

    let events: Vec<String> = event_mapping.iter()
        .map(|e| match e {
            Event::Speciation => "Speciation".to_string(),
            Event::Duplication => "Duplication".to_string(),
            Event::Transfer => "Transfer".to_string(),
            Event::Loss => "Loss".to_string(),
            Event::Leaf => "Leaf".to_string(),
        })
        .collect();

    // Also store the species tree for later use
    let species_tree_list = flattree_to_rlist(species_tree);

    Ok(list!(
        name = names,
        parent = parents,
        left_child = left_children,
        right_child = right_children,
        length = lengths,
        depth = depths,
        species_node = species_nodes,
        event = events,
        root = gene_tree.root as i32,
        species_tree = species_tree_list
    ))
}

fn rlist_to_genetree(list: &List) -> Result<(FlatTree, FlatTree, Vec<Option<usize>>, Vec<Event>)> {
    // Get gene tree data
    let names: Vec<String> = list.dollar("name")?.as_str_vector()
        .ok_or("Failed to get name column")?
        .iter().map(|s| s.to_string()).collect();
    let parents: Vec<i32> = list.dollar("parent")?.as_integer_vector()
        .ok_or("Failed to get parent column")?;
    let left_children: Vec<i32> = list.dollar("left_child")?.as_integer_vector()
        .ok_or("Failed to get left_child column")?;
    let right_children: Vec<i32> = list.dollar("right_child")?.as_integer_vector()
        .ok_or("Failed to get right_child column")?;
    let lengths: Vec<f64> = list.dollar("length")?.as_real_vector()
        .ok_or("Failed to get length column")?;
    let depths: Vec<f64> = list.dollar("depth")?.as_real_vector()
        .ok_or("Failed to get depth column")?;
    let species_node_names: Vec<String> = list.dollar("species_node")?.as_str_vector()
        .ok_or("Failed to get species_node column")?
        .iter().map(|s| s.to_string()).collect();
    let event_strs: Vec<String> = list.dollar("event")?.as_str_vector()
        .ok_or("Failed to get event column")?
        .iter().map(|s| s.to_string()).collect();
    let root: i32 = list.dollar("root")?.as_integer()
        .ok_or("Failed to get root")?;

    // Get species tree
    let species_tree_list: List = list.dollar("species_tree")?.try_into()
        .map_err(|_| "Failed to get species_tree")?;
    let species_tree = rlist_to_flattree(&species_tree_list)?;

    // Build gene tree
    let gene_nodes: Vec<FlatNode> = (0..names.len())
        .map(|i| FlatNode {
            name: names[i].clone(),
            parent: if parents[i].is_na() { None } else { Some(parents[i] as usize) },
            left_child: if left_children[i].is_na() { None } else { Some(left_children[i] as usize) },
            right_child: if right_children[i].is_na() { None } else { Some(right_children[i] as usize) },
            length: lengths[i],
            depth: if depths[i].is_na() { None } else { Some(depths[i]) },
            bd_event: None,
        })
        .collect();

    let gene_tree = FlatTree {
        nodes: gene_nodes,
        root: root as usize,
    };

    // Build node mapping (find species node index by name)
    let node_mapping: Vec<Option<usize>> = species_node_names.iter()
        .enumerate()
        .map(|(i, name)| {
            if name == "NA" || name.is_empty() {
                Ok(None)
            } else {
                species_tree.nodes.iter().position(|n| &n.name == name)
                    .map(Some)
                    .ok_or_else(|| format!(
                        "Species node name '{}' (at gene node index {}) not found in species tree",
                        name, i
                    ))
            }
        })
        .collect::<std::result::Result<Vec<Option<usize>>, String>>()
        .map_err(|e| extendr_api::Error::Other(e))?;

    // Build event mapping
    let event_mapping: Vec<Event> = event_strs.iter()
        .enumerate()
        .map(|(i, s)| match s.as_str() {
            "Speciation" => Ok(Event::Speciation),
            "Duplication" => Ok(Event::Duplication),
            "Transfer" => Ok(Event::Transfer),
            "Loss" => Ok(Event::Loss),
            "Leaf" => Ok(Event::Leaf),
            unknown => Err(format!(
                "Unknown reconciliation event type '{}' at index {}. Expected one of: Speciation, Duplication, Transfer, Loss, Leaf",
                unknown, i
            )),
        })
        .collect::<std::result::Result<Vec<Event>, String>>()
        .map_err(|e| extendr_api::Error::Other(e))?;

    Ok((gene_tree, species_tree, node_mapping, event_mapping))
}

/// Compute all pairwise distances between nodes in a tree.
///
/// @param tree_list A tree list from simulate_species_tree_r or parse_newick_r
/// @param distance_type Type of distance: "topological" (number of edges) or "metric" (sum of branch lengths)
/// @param leaves_only If TRUE, only compute distances between leaf nodes (default TRUE)
/// @return A list with three vectors: node1, node2, distance (suitable for conversion to data.frame)
/// @export
#[extendr]
fn pairwise_distances_r(tree_list: List, distance_type: &str, leaves_only: bool) -> Result<List> {
    use crate::metric_functions::DistanceType;

    let tree = rlist_to_flattree(&tree_list)?;

    let dist_type = match distance_type.to_lowercase().as_str() {
        "topological" | "topo" => DistanceType::Topological,
        "metric" | "branch" | "patristic" => DistanceType::Metric,
        _ => return Err(format!(
            "Invalid distance_type '{}'. Use 'topological' or 'metric'.",
            distance_type
        ).into()),
    };

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
    use crate::metric_functions::{DistanceType, PairwiseDistance};

    let tree = rlist_to_flattree(&tree_list)?;

    let dist_type = match distance_type.to_lowercase().as_str() {
        "topological" | "topo" => DistanceType::Topological,
        "metric" | "branch" | "patristic" => DistanceType::Metric,
        _ => return Err(format!(
            "Invalid distance_type '{}'. Use 'topological' or 'metric'.",
            distance_type
        ).into()),
    };

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
    let rec_tree = RecTree::from_xml_file(filepath)
        .map_err(|e| format!("Failed to parse RecPhyloXML: {}", e))?;

    // Convert to R list using the rectree_to_rlist function
    rectree_to_rlist(
        &rec_tree.species_tree,
        &rec_tree.gene_tree,
        &rec_tree.node_mapping,
        &rec_tree.event_mapping,
    )
}

// ---------------------------------------------------------------------------
// Streaming / lazy-iterator DTL functions
// ---------------------------------------------------------------------------

/// Helper: parse a Newick string into a FlatTree with depths assigned.
fn parse_newick_to_flattree(newick: &str) -> Result<FlatTree> {
    use crate::newick::newick::parse_newick;

    let mut nodes = parse_newick(newick)
        .map_err(|e| format!("Failed to parse Newick: {}", e))?;
    let mut root = nodes.pop()
        .ok_or("No tree found in Newick string")?;
    root.zero_root_length();
    root.assign_depths(0.0);
    let mut tree = root.to_flat_tree();
    tree.assign_depths();
    Ok(tree)
}

/// Helper: create an StdRng from an optional R seed value.
fn make_rng(seed: &Robj) -> Result<StdRng> {
    if seed.is_null() || seed.is_na() {
        Ok(StdRng::from_entropy())
    } else {
        match seed.as_integer() {
            Some(s) => Ok(StdRng::seed_from_u64(s as u64)),
            None => Err("seed must be an integer".into()),
        }
    }
}

/// Helper: extract optional transfer_alpha from an Robj.
fn extract_alpha(transfer_alpha: &Robj) -> Result<Option<f64>> {
    if transfer_alpha.is_null() || transfer_alpha.is_na() {
        Ok(None)
    } else {
        Ok(Some(transfer_alpha.as_real().ok_or("transfer_alpha must be a number")?))
    }
}

/// Helper: extract optional replacement_transfer from an Robj.
fn extract_replacement(replacement_transfer: &Robj) -> Result<Option<f64>> {
    if replacement_transfer.is_null() || replacement_transfer.is_na() {
        Ok(None)
    } else {
        Ok(Some(replacement_transfer.as_real().ok_or("replacement_transfer must be a number")?))
    }
}

/// Stream DTL gene trees (per-gene model) to RecPhyloXML files.
///
/// Simulates `n` gene trees one at a time using the per-gene-copy DTL model
/// and writes each tree as a RecPhyloXML file in `output_dir`. Files are
/// named `gene_0000.xml`, `gene_0001.xml`, etc.
///
/// This is memory-efficient for large batches because only one tree is held
/// in memory at a time.
///
/// @param newick Species tree in Newick format
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate
/// @param lambda_t Transfer rate
/// @param lambda_l Loss rate
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, retry until a tree with extant genes is produced
/// @param seed Optional random seed for reproducibility
/// @param output_dir Directory to write XML files to (created if it doesn't exist)
/// @return The output directory path
/// @export
#[extendr]
fn simulate_dtl_stream_xml_r(
    newick: &str,
    n: i32,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
    output_dir: &str,
) -> Result<String> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    if lambda_d < 0.0 || lambda_t < 0.0 || lambda_l < 0.0 {
        return Err("Rates must be non-negative".into());
    }

    let species_tree = parse_newick_to_flattree(newick)?;
    let mut rng = make_rng(&seed)?;
    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    simulate_dtl_iter(
        &species_tree, origin_species,
        lambda_d, lambda_t, lambda_l,
        alpha, repl,
        n as usize, require_extant, &mut rng,
    )
    .map_err(|e| Error::Other(e))?
    .save_xml(output_dir)
    .map_err(|e| Error::Other(e))?;

    Ok(output_dir.to_string())
}

/// Stream DTL gene trees (per-gene model) to Newick files.
///
/// Simulates `n` gene trees one at a time using the per-gene-copy DTL model
/// and writes each gene tree in Newick format to `output_dir`. Files are
/// named `gene_0000.nwk`, `gene_0001.nwk`, etc.
///
/// Note: Newick format does not preserve reconciliation information (species
/// mapping, events). Use `simulate_dtl_stream_xml_r` for full reconciled trees.
///
/// @param newick Species tree in Newick format
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate
/// @param lambda_t Transfer rate
/// @param lambda_l Loss rate
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, retry until a tree with extant genes is produced
/// @param seed Optional random seed for reproducibility
/// @param output_dir Directory to write Newick files to (created if it doesn't exist)
/// @return The output directory path
/// @export
#[extendr]
fn simulate_dtl_stream_newick_r(
    newick: &str,
    n: i32,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
    output_dir: &str,
) -> Result<String> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    if lambda_d < 0.0 || lambda_t < 0.0 || lambda_l < 0.0 {
        return Err("Rates must be non-negative".into());
    }

    let species_tree = parse_newick_to_flattree(newick)?;
    let mut rng = make_rng(&seed)?;
    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    simulate_dtl_iter(
        &species_tree, origin_species,
        lambda_d, lambda_t, lambda_l,
        alpha, repl,
        n as usize, require_extant, &mut rng,
    )
    .map_err(|e| Error::Other(e))?
    .save_newick(output_dir)
    .map_err(|e| Error::Other(e))?;

    Ok(output_dir.to_string())
}

/// Stream DTL gene trees (per-species/Zombi model) to RecPhyloXML files.
///
/// Simulates `n` gene trees one at a time using the Zombi-style per-species
/// DTL model and writes each tree as a RecPhyloXML file in `output_dir`.
/// Files are named `gene_0000.xml`, `gene_0001.xml`, etc.
///
/// In the per-species model, the event rate is proportional to the number of
/// SPECIES with gene copies, NOT the number of gene copies themselves.
///
/// @param newick Species tree in Newick format
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate per species per unit time
/// @param lambda_t Transfer rate per species per unit time
/// @param lambda_l Loss rate per species per unit time
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, retry until a tree with extant genes is produced
/// @param seed Optional random seed for reproducibility
/// @param output_dir Directory to write XML files to (created if it doesn't exist)
/// @return The output directory path
/// @export
#[extendr]
fn simulate_dtl_per_species_stream_xml_r(
    newick: &str,
    n: i32,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
    output_dir: &str,
) -> Result<String> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    if lambda_d < 0.0 || lambda_t < 0.0 || lambda_l < 0.0 {
        return Err("Rates must be non-negative".into());
    }

    let species_tree = parse_newick_to_flattree(newick)?;
    let mut rng = make_rng(&seed)?;
    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    simulate_dtl_per_species_iter(
        &species_tree, origin_species,
        lambda_d, lambda_t, lambda_l,
        alpha, repl,
        n as usize, require_extant, &mut rng,
    )
    .map_err(|e| Error::Other(e))?
    .save_xml(output_dir)
    .map_err(|e| Error::Other(e))?;

    Ok(output_dir.to_string())
}

/// Stream DTL gene trees (per-species/Zombi model) to Newick files.
///
/// Simulates `n` gene trees one at a time using the Zombi-style per-species
/// DTL model and writes each gene tree in Newick format to `output_dir`.
/// Files are named `gene_0000.nwk`, `gene_0001.nwk`, etc.
///
/// Note: Newick format does not preserve reconciliation information (species
/// mapping, events). Use `simulate_dtl_per_species_stream_xml_r` for full
/// reconciled trees.
///
/// @param newick Species tree in Newick format
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate per species per unit time
/// @param lambda_t Transfer rate per species per unit time
/// @param lambda_l Loss rate per species per unit time
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, retry until a tree with extant genes is produced
/// @param seed Optional random seed for reproducibility
/// @param output_dir Directory to write Newick files to (created if it doesn't exist)
/// @return The output directory path
/// @export
#[extendr]
fn simulate_dtl_per_species_stream_newick_r(
    newick: &str,
    n: i32,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
    output_dir: &str,
) -> Result<String> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    if lambda_d < 0.0 || lambda_t < 0.0 || lambda_l < 0.0 {
        return Err("Rates must be non-negative".into());
    }

    let species_tree = parse_newick_to_flattree(newick)?;
    let mut rng = make_rng(&seed)?;
    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    simulate_dtl_per_species_iter(
        &species_tree, origin_species,
        lambda_d, lambda_t, lambda_l,
        alpha, repl,
        n as usize, require_extant, &mut rng,
    )
    .map_err(|e| Error::Other(e))?
    .save_newick(output_dir)
    .map_err(|e| Error::Other(e))?;

    Ok(output_dir.to_string())
}

// Macro to generate exports
extendr_module! {
    mod rustree;
    fn simulate_species_tree_r;
    fn parse_newick_r;
    fn tree_to_newick_r;
    fn tree_num_leaves_r;
    fn tree_leaf_names_r;
    fn simulate_dtl_r;
    fn simulate_dtl_batch_r;
    fn simulate_dtl_per_species_r;
    fn simulate_dtl_per_species_batch_r;
    fn gene_tree_num_extant_r;
    fn gene_tree_to_newick_r;
    fn gene_tree_to_xml_r;
    fn save_newick_r;
    fn save_xml_r;
    fn save_csv_r;
    fn save_bd_events_csv_r;
    fn gene_tree_to_svg_r;
    fn sample_extant_r;
    fn sample_leaves_r;
    fn extract_induced_subtree_by_names_r;
    fn pairwise_distances_r;
    fn save_pairwise_distances_csv_r;
    fn parse_recphyloxml_r;
    fn simulate_dtl_stream_xml_r;
    fn simulate_dtl_stream_newick_r;
    fn simulate_dtl_per_species_stream_xml_r;
    fn simulate_dtl_per_species_stream_newick_r;
}
//! Data extraction and parsing for training samples.
//!
//! Contains base sample extraction from RecTree, base sample dict parsing,
//! and the create_training_sample / create_training_sample_from_sim / from_reconciliation pyfunctions.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::fs;

use crate::node::{Event, RecTree};
use crate::python::PyGeneTree;

/// Internal representation of a training sample (mirrors Python base_sample dict).
pub(super) struct RustBaseSample {
    pub(super) g_root_name: String,
    pub(super) nb_g_leaves: usize,
    pub(super) species_names: Vec<String>,
    pub(super) g_leaves_names: Vec<String>,
    pub(super) true_root: Vec<String>,
    pub(super) g_neighbors: HashMap<String, Vec<String>>,
    pub(super) sp_children: HashMap<String, Vec<String>>,
    pub(super) true_states: HashMap<String, String>,
    pub(super) true_events: HashMap<String, i32>,
}

/// Extract base sample data from a RecTree (no Python boundary).
pub(super) fn extract_base_sample_internal(rt: &RecTree) -> Result<RustBaseSample, String> {
    use crate::node::TraversalOrder;

    let sp_tree = &*rt.species_tree;
    let g_tree = &rt.gene_tree;

    let sp_preorder: Vec<usize> = sp_tree.iter_indices(TraversalOrder::PreOrder).collect();
    let species_names: Vec<String> = sp_preorder.iter()
        .map(|&i| sp_tree.nodes[i].name.clone())
        .collect();
    let sp_name_set: std::collections::HashSet<&str> = species_names.iter().map(|s| s.as_str()).collect();

    let mut sp_children: HashMap<String, Vec<String>> = HashMap::new();
    for &idx in &sp_preorder {
        let node = &sp_tree.nodes[idx];
        let mut children = Vec::new();
        if let Some(left) = node.left_child {
            children.push(sp_tree.nodes[left].name.clone());
        }
        if let Some(right) = node.right_child {
            children.push(sp_tree.nodes[right].name.clone());
        }
        if !children.is_empty() {
            sp_children.insert(node.name.clone(), children);
        }
    }

    let g_preorder: Vec<usize> = g_tree.iter_indices(TraversalOrder::PreOrder).collect();
    let g_root_name = g_tree.nodes[g_tree.root].name.clone();

    let g_leaves_names: Vec<String> = g_preorder.iter()
        .filter(|&&i| g_tree.nodes[i].left_child.is_none() && g_tree.nodes[i].right_child.is_none())
        .map(|&i| g_tree.nodes[i].name.clone())
        .collect();
    let nb_g_leaves = g_leaves_names.len();

    let mut g_neighbors: HashMap<String, Vec<String>> = HashMap::new();
    for &idx in &g_preorder {
        let node = &g_tree.nodes[idx];
        let mut neighbors = Vec::new();
        if let Some(parent) = node.parent {
            neighbors.push(g_tree.nodes[parent].name.clone());
        }
        if let Some(left) = node.left_child {
            neighbors.push(g_tree.nodes[left].name.clone());
        }
        if let Some(right) = node.right_child {
            neighbors.push(g_tree.nodes[right].name.clone());
        }
        g_neighbors.insert(node.name.clone(), neighbors);
    }

    let g_root_node = &g_tree.nodes[g_tree.root];
    let mut true_root = Vec::new();
    if let Some(left) = g_root_node.left_child {
        true_root.push(g_tree.nodes[left].name.clone());
    }
    if let Some(right) = g_root_node.right_child {
        true_root.push(g_tree.nodes[right].name.clone());
    }

    let mut true_states: HashMap<String, String> = HashMap::new();
    let mut true_events: HashMap<String, i32> = HashMap::new();
    for &idx in &g_preorder {
        let g_name = g_tree.nodes[idx].name.clone();
        let sp_idx = rt.node_mapping[idx]
            .ok_or_else(|| format!("Gene node '{}' has no species mapping", g_name))?;
        let sp_name = sp_tree.nodes[sp_idx].name.clone();
        if !sp_name_set.contains(sp_name.as_str()) {
            return Err(format!("Gene node '{}' maps to unknown species '{}'", g_name, sp_name));
        }
        true_states.insert(g_name.clone(), sp_name);
        let event_idx: i32 = match &rt.event_mapping[idx] {
            Event::Speciation => 0,
            Event::Duplication => 1,
            Event::Transfer => 2,
            Event::Leaf => 3,
            Event::Loss => 3,
        };
        true_events.insert(g_name, event_idx);
    }

    Ok(RustBaseSample {
        g_root_name, nb_g_leaves, species_names, g_leaves_names,
        true_root, g_neighbors, sp_children, true_states, true_events,
    })
}

/// Parse a Python base_sample dict into a RustBaseSample.
pub(super) fn parse_base_sample_dict(d: &Bound<'_, pyo3::types::PyDict>) -> PyResult<RustBaseSample> {
    let species_names: Vec<String> = d.get_item("species_names")?
        .ok_or_else(|| PyValueError::new_err("missing species_names"))?.extract()?;
    let g_root_name: String = d.get_item("g_root_name")?
        .ok_or_else(|| PyValueError::new_err("missing g_root_name"))?.extract()?;
    let g_leaves_names: Vec<String> = d.get_item("g_leaves_names")?
        .ok_or_else(|| PyValueError::new_err("missing g_leaves_names"))?.extract()?;
    let nb_g_leaves = g_leaves_names.len();
    let true_root: Vec<String> = d.get_item("true_root")?
        .ok_or_else(|| PyValueError::new_err("missing true_root"))?.extract()?;
    let g_neighbors: HashMap<String, Vec<String>> = d.get_item("g_neighbors")?
        .ok_or_else(|| PyValueError::new_err("missing g_neighbors"))?.extract()?;
    let sp_children: HashMap<String, Vec<String>> = d.get_item("sp_children")?
        .ok_or_else(|| PyValueError::new_err("missing sp_children"))?.extract()?;
    let true_states: HashMap<String, String> = d.get_item("true_states")?
        .ok_or_else(|| PyValueError::new_err("missing true_states"))?.extract()?;
    let true_events: HashMap<String, i32> = d.get_item("true_events")?
        .ok_or_else(|| PyValueError::new_err("missing true_events"))?.extract()?;

    Ok(RustBaseSample {
        g_root_name, nb_g_leaves, species_names, g_leaves_names,
        true_root, g_neighbors, sp_children, true_states, true_events,
    })
}


/// Create a training sample from species tree, pruned gene tree, and reconciliation XML.
///
/// This replaces the Python `create_sample()` function with a fast Rust implementation.
/// The Zombi-format XML (containing `<recGeneTree>` without `<spTree>`) is parsed for
/// event annotations. The pruned gene tree from Newick provides the observable topology.
///
/// # Arguments
/// * `sp_newick_path` - Path to species tree Newick file
/// * `g_newick_path` - Path to pruned gene tree Newick file
/// * `xml_path` - Path to reconciliation XML file (Zombi format)
///
/// # Returns
/// A Python dict with the same keys as the Python `create_sample()`:
/// species_names, g_root_name, gene_names, g_neighbors, g_leaves_names,
/// sp_children, sp_parents, true_states, true_events, true_root,
/// nb_sp_leaves, nb_g_leaves, sp_tree_path, g_tree_path, xml_g_tree_path
#[pyfunction]
pub fn create_training_sample(
    py: Python,
    sp_newick_path: &str,
    g_newick_path: &str,
    xml_path: &str,
) -> PyResult<PyObject> {
    use crate::newick::parse_newick;
    use crate::node::TraversalOrder;
    use crate::io::recphyloxml::parse_xml_gene_annotations_file;
    use pyo3::types::{PyDict, PyList};

    // 1. Parse species tree
    let sp_newick = fs::read_to_string(sp_newick_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read species tree: {}", e)))?;
    let mut sp_nodes = parse_newick(&sp_newick)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse species Newick: {}", e)))?;
    let sp_root = sp_nodes.pop()
        .ok_or_else(|| PyValueError::new_err("No tree found in species Newick"))?;
    let sp_tree = sp_root.to_flat_tree();

    // 2. Parse pruned gene tree
    let g_newick = fs::read_to_string(g_newick_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read gene tree: {}", e)))?;
    let mut g_nodes = parse_newick(&g_newick)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse gene Newick: {}", e)))?;
    let g_root = g_nodes.pop()
        .ok_or_else(|| PyValueError::new_err("No tree found in gene Newick"))?;
    let g_tree = g_root.to_flat_tree();

    // 3. Parse XML annotations
    let annotations = parse_xml_gene_annotations_file(xml_path)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse XML: {}", e)))?;

    // 4. Build species names (preorder)
    let sp_preorder: Vec<usize> = sp_tree.iter_indices(TraversalOrder::PreOrder).collect();
    let species_names: Vec<&str> = sp_preorder.iter()
        .map(|&i| sp_tree.nodes[i].name.as_str())
        .collect();
    let sp_name_set: std::collections::HashSet<&str> = species_names.iter().copied().collect();

    // 5. Build species children/parents dicts
    let sp_children = PyDict::new(py);
    let sp_parents = PyDict::new(py);
    for &idx in &sp_preorder {
        let node = &sp_tree.nodes[idx];
        let name = &node.name;
        let mut children_list = Vec::new();
        if let Some(left) = node.left_child {
            children_list.push(sp_tree.nodes[left].name.as_str());
        }
        if let Some(right) = node.right_child {
            children_list.push(sp_tree.nodes[right].name.as_str());
        }
        if !children_list.is_empty() {
            sp_children.set_item(name, children_list)?;
        }
        if let Some(parent) = node.parent {
            sp_parents.set_item(name, vec![sp_tree.nodes[parent].name.as_str()])?;
        }
    }

    // 6. Build gene names (preorder), separate root
    let g_preorder: Vec<usize> = g_tree.iter_indices(TraversalOrder::PreOrder).collect();
    let g_root_name = &g_tree.nodes[g_tree.root].name;
    let gene_names: Vec<&str> = g_preorder.iter()
        .filter(|&&i| i != g_tree.root)
        .map(|&i| g_tree.nodes[i].name.as_str())
        .collect();

    // 7. Build gene leaves
    let g_leaves_names: Vec<&str> = g_preorder.iter()
        .filter(|&&i| {
            let n = &g_tree.nodes[i];
            n.left_child.is_none() && n.right_child.is_none()
        })
        .map(|&i| g_tree.nodes[i].name.as_str())
        .collect();
    let nb_g_leaves = g_leaves_names.len();

    // 8. Build gene undirected adjacency (including root)
    let g_neighbors = PyDict::new(py);
    for &idx in &g_preorder {
        let node = &g_tree.nodes[idx];
        let mut neighbors = Vec::new();
        if let Some(parent) = node.parent {
            neighbors.push(g_tree.nodes[parent].name.as_str());
        }
        if let Some(left) = node.left_child {
            neighbors.push(g_tree.nodes[left].name.as_str());
        }
        if let Some(right) = node.right_child {
            neighbors.push(g_tree.nodes[right].name.as_str());
        }
        g_neighbors.set_item(&node.name, neighbors)?;
    }

    // 9. Build true_root (children of gene tree root)
    let g_root_node = &g_tree.nodes[g_tree.root];
    let mut root_children = Vec::new();
    if let Some(left) = g_root_node.left_child {
        root_children.push(g_tree.nodes[left].name.as_str());
    }
    if let Some(right) = g_root_node.right_child {
        root_children.push(g_tree.nodes[right].name.as_str());
    }

    // 10. Build true_states and true_events from XML annotations
    // Event mapping: speciation=0, duplication=1, T=2, P=3
    let true_states = PyDict::new(py);
    let true_events = PyDict::new(py);
    let all_g_names: Vec<&str> = g_preorder.iter()
        .map(|&i| g_tree.nodes[i].name.as_str())
        .collect();
    for &g_name in &all_g_names {
        let (sp_loc, event) = annotations.get(g_name)
            .ok_or_else(|| PyValueError::new_err(
                format!("Gene node '{}' not found in XML annotations", g_name)
            ))?;
        if !sp_name_set.contains(sp_loc.as_str()) {
            return Err(PyValueError::new_err(
                format!("Gene node '{}' maps to unknown species '{}'", g_name, sp_loc)
            ));
        }
        true_states.set_item(g_name, sp_loc)?;
        let event_idx: i32 = match event {
            Event::Speciation => 0,
            Event::Duplication => 1,
            Event::Transfer => 2,
            Event::Leaf => 3,
            Event::Loss => 3,  // shouldn't appear in pruned tree, but map safely
        };
        true_events.set_item(g_name, event_idx)?;
    }

    // 11. Count species leaves
    let nb_sp_leaves = sp_preorder.iter()
        .filter(|&&i| {
            let n = &sp_tree.nodes[i];
            n.left_child.is_none() && n.right_child.is_none()
        })
        .count();

    // 12. Build result dict
    let result = PyDict::new(py);
    result.set_item("species_names", PyList::new(py, &species_names)?)?;
    result.set_item("g_root_name", g_root_name)?;
    result.set_item("_gene_root_name", g_root_name)?;
    result.set_item("gene_names", PyList::new(py, &gene_names)?)?;
    result.set_item("g_neighbors", g_neighbors)?;
    result.set_item("g_leaves_names", PyList::new(py, &g_leaves_names)?)?;
    result.set_item("sp_children", sp_children)?;
    result.set_item("sp_parents", sp_parents)?;
    result.set_item("true_states", true_states)?;
    result.set_item("true_events", true_events)?;
    result.set_item("true_root", PyList::new(py, &root_children)?)?;
    result.set_item("nb_sp_leaves", nb_sp_leaves)?;
    result.set_item("nb_g_leaves", nb_g_leaves)?;
    result.set_item("sp_tree_path", sp_newick_path)?;
    result.set_item("g_tree_path", g_newick_path)?;
    result.set_item("xml_g_tree_path", xml_path)?;

    Ok(result.into())
}

/// Create a training sample dict directly from simulated tree objects.
///
/// Equivalent to `create_training_sample` but takes a `PyGeneTree` (after
/// `sample_extant()`) instead of file paths.  All reconciliation information
/// (node mapping, event types) is read from the internal `RecTree`, so no
/// intermediate files or pandas DataFrames are needed.
///
/// # Arguments
/// * `gene_tree` - a `PyGeneTree` returned by `sample_extant()`.
///
/// # Returns
/// A Python dict with the same keys as `create_training_sample` (minus the
/// three `*_path` keys which are set to empty strings).
#[cfg(feature = "python")]
#[pyfunction]
pub fn create_training_sample_from_sim(
    py: Python,
    gene_tree: &PyGeneTree,
) -> PyResult<PyObject> {
    use crate::node::TraversalOrder;
    use pyo3::types::{PyDict, PyList};

    let rt = &gene_tree.rec_tree;
    let sp_tree = &*rt.species_tree;
    let g_tree = &rt.gene_tree;

    // 1. Species names (preorder)
    let sp_preorder: Vec<usize> = sp_tree.iter_indices(TraversalOrder::PreOrder).collect();
    let species_names: Vec<&str> = sp_preorder.iter()
        .map(|&i| sp_tree.nodes[i].name.as_str())
        .collect();
    let sp_name_set: std::collections::HashSet<&str> = species_names.iter().copied().collect();

    // 2. Species children / parents
    let sp_children = PyDict::new(py);
    let sp_parents = PyDict::new(py);
    for &idx in &sp_preorder {
        let node = &sp_tree.nodes[idx];
        let mut children_list = Vec::new();
        if let Some(left) = node.left_child {
            children_list.push(sp_tree.nodes[left].name.as_str());
        }
        if let Some(right) = node.right_child {
            children_list.push(sp_tree.nodes[right].name.as_str());
        }
        if !children_list.is_empty() {
            sp_children.set_item(&node.name, children_list)?;
        }
        if let Some(parent) = node.parent {
            sp_parents.set_item(&node.name, vec![sp_tree.nodes[parent].name.as_str()])?;
        }
    }

    // 3. Gene names (preorder), root separate
    let g_preorder: Vec<usize> = g_tree.iter_indices(TraversalOrder::PreOrder).collect();
    let g_root_name = &g_tree.nodes[g_tree.root].name;
    let gene_names: Vec<&str> = g_preorder.iter()
        .filter(|&&i| i != g_tree.root)
        .map(|&i| g_tree.nodes[i].name.as_str())
        .collect();

    // 4. Gene leaves
    let g_leaves_names: Vec<&str> = g_preorder.iter()
        .filter(|&&i| {
            let n = &g_tree.nodes[i];
            n.left_child.is_none() && n.right_child.is_none()
        })
        .map(|&i| g_tree.nodes[i].name.as_str())
        .collect();
    let nb_g_leaves = g_leaves_names.len();

    // 5. Gene undirected adjacency
    let g_neighbors = PyDict::new(py);
    for &idx in &g_preorder {
        let node = &g_tree.nodes[idx];
        let mut neighbors = Vec::new();
        if let Some(parent) = node.parent {
            neighbors.push(g_tree.nodes[parent].name.as_str());
        }
        if let Some(left) = node.left_child {
            neighbors.push(g_tree.nodes[left].name.as_str());
        }
        if let Some(right) = node.right_child {
            neighbors.push(g_tree.nodes[right].name.as_str());
        }
        g_neighbors.set_item(&node.name, neighbors)?;
    }

    // 6. true_root (children of gene root)
    let g_root_node = &g_tree.nodes[g_tree.root];
    let mut root_children = Vec::new();
    if let Some(left) = g_root_node.left_child {
        root_children.push(g_tree.nodes[left].name.as_str());
    }
    if let Some(right) = g_root_node.right_child {
        root_children.push(g_tree.nodes[right].name.as_str());
    }

    // 7. true_states and true_events from RecTree mappings
    let true_states = PyDict::new(py);
    let true_events = PyDict::new(py);
    for &idx in &g_preorder {
        let g_name = g_tree.nodes[idx].name.as_str();
        // Species mapping
        let sp_idx = rt.node_mapping[idx]
            .ok_or_else(|| PyValueError::new_err(
                format!("Gene node '{}' has no species mapping", g_name)
            ))?;
        let sp_name = sp_tree.nodes[sp_idx].name.as_str();
        if !sp_name_set.contains(sp_name) {
            return Err(PyValueError::new_err(
                format!("Gene node '{}' maps to unknown species '{}'", g_name, sp_name)
            ));
        }
        true_states.set_item(g_name, sp_name)?;
        // Event mapping
        let event_idx: i32 = match &rt.event_mapping[idx] {
            Event::Speciation => 0,
            Event::Duplication => 1,
            Event::Transfer => 2,
            Event::Leaf => 3,
            Event::Loss => 3,
        };
        true_events.set_item(g_name, event_idx)?;
    }

    // 8. Count species leaves
    let nb_sp_leaves = sp_preorder.iter()
        .filter(|&&i| {
            let n = &sp_tree.nodes[i];
            n.left_child.is_none() && n.right_child.is_none()
        })
        .count();

    // 9. Build result dict
    let result = PyDict::new(py);
    result.set_item("species_names", PyList::new(py, &species_names)?)?;
    result.set_item("g_root_name", g_root_name)?;
    result.set_item("_gene_root_name", g_root_name)?;
    result.set_item("gene_names", PyList::new(py, &gene_names)?)?;
    result.set_item("g_neighbors", g_neighbors)?;
    result.set_item("g_leaves_names", PyList::new(py, &g_leaves_names)?)?;
    result.set_item("sp_children", sp_children)?;
    result.set_item("sp_parents", sp_parents)?;
    result.set_item("true_states", true_states)?;
    result.set_item("true_events", true_events)?;
    result.set_item("true_root", PyList::new(py, &root_children)?)?;
    result.set_item("nb_sp_leaves", nb_sp_leaves)?;
    result.set_item("nb_g_leaves", nb_g_leaves)?;
    result.set_item("sp_tree_path", "")?;
    result.set_item("g_tree_path", "")?;
    result.set_item("xml_g_tree_path", "")?;

    Ok(result.into())
}

/// Build a PyGeneTree from a species tree, gene tree, and per-node reconciliation annotations.
///
/// This allows constructing a fully annotated reconciled gene tree without going through
/// RecPhyloXML, e.g. from model predictions.
///
/// # Arguments
/// * `sp_newick` - Species tree in Newick format (string, not file path)
/// * `g_newick` - Gene tree in Newick format (string, not file path)
/// * `node_species` - Dict mapping gene node name -> species node name
/// * `node_events` - Dict mapping gene node name -> event code (0=Speciation, 1=Duplication, 2=Transfer, 3=Leaf)
///
/// # Returns
/// A PyGeneTree with the given reconciliation annotations.
#[pyfunction]
#[pyo3(signature = (sp_newick, g_newick, node_species, node_events))]
pub fn from_reconciliation(
    sp_newick: &str,
    g_newick: &str,
    node_species: &Bound<'_, pyo3::types::PyDict>,
    node_events: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<PyGeneTree> {
    use crate::newick::parse_newick;
    use crate::node::rectree::{Event, RecTree};

    // Parse species tree
    let mut sp_nodes = parse_newick(sp_newick)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse species Newick: {}", e)))?;
    let sp_root = sp_nodes.pop()
        .ok_or_else(|| PyValueError::new_err("No tree found in species Newick"))?;
    let mut sp_tree = sp_root.to_flat_tree();
    sp_tree.assign_depths();

    // Build species name -> index map
    let sp_name_to_idx: std::collections::HashMap<String, usize> = sp_tree
        .nodes
        .iter()
        .enumerate()
        .map(|(idx, node)| (node.name.clone(), idx))
        .collect();

    // Parse gene tree
    let mut g_nodes = parse_newick(g_newick)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse gene Newick: {}", e)))?;
    let g_root = g_nodes.pop()
        .ok_or_else(|| PyValueError::new_err("No tree found in gene Newick"))?;
    let mut g_tree = g_root.to_flat_tree();
    g_tree.assign_depths();

    let n_gene_nodes = g_tree.nodes.len();

    // Build node_mapping and event_mapping
    let mut node_mapping: Vec<Option<usize>> = vec![None; n_gene_nodes];
    let mut event_mapping: Vec<Event> = vec![Event::Speciation; n_gene_nodes];

    for (idx, node) in g_tree.nodes.iter().enumerate() {
        let name = &node.name;

        // Species mapping
        if node_species.contains(name.as_str())? {
            let sp_name_obj = node_species.get_item(name.as_str())?
                .ok_or_else(|| PyValueError::new_err(format!("Key '{}' disappeared from node_species", name)))?;
            let sp_name: String = sp_name_obj.extract()?;
            if let Some(&sp_idx) = sp_name_to_idx.get(&sp_name) {
                node_mapping[idx] = Some(sp_idx);
            } else {
                return Err(PyValueError::new_err(format!(
                    "Species '{}' (mapped from gene node '{}') not found in species tree",
                    sp_name, name
                )));
            }
        }
        // If not in dict, mapping stays None (unknown)

        // Event mapping
        if node_events.contains(name.as_str())? {
            let evt_obj = node_events.get_item(name.as_str())?
                .ok_or_else(|| PyValueError::new_err(format!("Key '{}' disappeared from node_events", name)))?;
            let evt_code: i32 = evt_obj.extract()?;
            event_mapping[idx] = match evt_code {
                0 => Event::Speciation,
                1 => Event::Duplication,
                2 => Event::Transfer,
                3 => Event::Leaf,
                _ => Event::Speciation, // fallback
            };
        } else {
            // Default: leaf if no children, speciation otherwise
            event_mapping[idx] = if node.left_child.is_none() && node.right_child.is_none() {
                Event::Leaf
            } else {
                Event::Speciation
            };
        }
    }

    let rec_tree = RecTree::new_owned(sp_tree, g_tree, node_mapping, event_mapping);
    Ok(PyGeneTree { rec_tree })
}

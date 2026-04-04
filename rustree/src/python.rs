//! Python bindings for rustree using PyO3.
//!
//! Provides Python access to:
//! - Birth-death species tree simulation
//! - DTL gene tree simulation
//! - Gene tree sampling (induced subtree extraction)

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::bd::{simulate_bd_tree_bwd, generate_events_from_tree, TreeEvent};
use crate::dtl::{simulate_dtl, simulate_dtl_batch, simulate_dtl_per_species, simulate_dtl_per_species_batch, count_extant_genes};
use crate::node::{FlatTree, Event, RecTree, remap_gene_tree_indices};
use crate::io::rectree_csv::RecTreeColumns;
use crate::sampling::{extract_induced_subtree, extract_induced_subtree_by_names, find_leaf_indices_by_names, extract_extant_subtree, NodeMark, mark_nodes_postorder};
use crate::simulation::dtl::gillespie::{DTLMode, simulate_dtl_gillespie};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::sync::Arc;

// ============================================================================
// Helper Functions (Refactored to reduce duplication)
// ============================================================================

/// Validate DTL rates (all must be non-negative)
fn validate_dtl_rates(lambda_d: f64, lambda_t: f64, lambda_l: f64) -> PyResult<()> {
    if lambda_d < 0.0 || lambda_t < 0.0 || lambda_l < 0.0 {
        return Err(PyValueError::new_err("Rates must be non-negative"));
    }
    Ok(())
}

/// Validate replacement_transfer probability parameter
fn validate_replacement_transfer(replacement_transfer: Option<f64>) -> PyResult<()> {
    if let Some(p) = replacement_transfer {
        if !(0.0..=1.0).contains(&p) {
            return Err(PyValueError::new_err(
                "replacement_transfer must be between 0.0 and 1.0"
            ));
        }
    }
    Ok(())
}

/// Extract the gene tree with only extant (Event::Leaf) leaves, stripping loss nodes.
fn extract_extant_gene_tree(rec_tree: &RecTree) -> Result<FlatTree, String> {
    let extant_indices: std::collections::HashSet<usize> = rec_tree.gene_tree.nodes.iter()
        .enumerate()
        .filter(|(i, n)| {
            n.left_child.is_none()
            && n.right_child.is_none()
            && rec_tree.event_mapping[*i] == Event::Leaf
        })
        .map(|(i, _)| i)
        .collect();

    if extant_indices.is_empty() {
        return Err("No extant leaves".to_string());
    }

    let (tree, _) = extract_induced_subtree(&rec_tree.gene_tree, &extant_indices)
        .ok_or_else(|| "Failed to extract extant subtree".to_string())?;
    Ok(tree)
}

/// Initialize RNG from optional seed
fn init_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

/// Check if a node is a leaf (no children)
fn is_leaf(node: &crate::node::FlatNode) -> bool {
    node.left_child.is_none() && node.right_child.is_none()
}

// ============================================================================

/// A species tree simulated under the birth-death process.
#[pyclass]
#[derive(Clone)]
pub struct PySpeciesTree {
    tree: Arc<FlatTree>,
}

#[pymethods]
impl PySpeciesTree {
    /// Convert the species tree to Newick format.
    fn to_newick(&self) -> PyResult<String> {
        let nwk = self.tree.to_newick()
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(nwk + ";")
    }

    /// Get the number of nodes in the tree.
    fn num_nodes(&self) -> usize {
        self.tree.nodes.len()
    }

    /// Get the number of extant species (leaves).
    fn num_leaves(&self) -> usize {
        self.tree.nodes.iter().filter(|n| is_leaf(n)).count()
    }

    /// Get the total tree height (depth of deepest leaf).
    fn tree_height(&self) -> f64 {
        self.tree.nodes.iter()
            .filter_map(|n| n.depth)
            .fold(0.0, f64::max)
    }

    /// Get the root node index.
    fn root_index(&self) -> usize {
        self.tree.root
    }

    /// Get leaf names.
    fn leaf_names(&self) -> Vec<String> {
        self.tree.nodes.iter()
            .filter(|n| is_leaf(n))
            .map(|n| n.name.clone())
            .collect()
    }

    /// Get a node by its index.
    fn get_node(&self, index: usize) -> PyResult<PySpeciesNode> {
        if index >= self.tree.nodes.len() {
            return Err(PyValueError::new_err(format!(
                "Node index {} out of range (tree has {} nodes)", index, self.tree.nodes.len()
            )));
        }
        let node = &self.tree.nodes[index];
        Ok(PySpeciesNode {
            name: node.name.clone(),
            index,
            depth: node.depth,
            length: node.length,
            left_child: node.left_child,
            right_child: node.right_child,
            parent: node.parent,
            bd_event: node.bd_event,
        })
    }

    /// Iterate over nodes in a given traversal order.
    ///
    /// # Arguments
    /// * `order` - Traversal order: "preorder", "inorder", or "postorder" (default: "preorder")
    ///
    /// # Example
    /// ```python
    /// for node in species_tree.iter("postorder"):
    ///     print(node.name, node.length)
    /// ```
    #[pyo3(signature = (order="preorder"))]
    fn iter(&self, order: &str) -> PyResult<PySpeciesTreeIter> {
        use crate::node::TraversalOrder;
        let traversal = match order.to_lowercase().as_str() {
            "preorder" | "pre" => TraversalOrder::PreOrder,
            "inorder" | "in" => TraversalOrder::InOrder,
            "postorder" | "post" => TraversalOrder::PostOrder,
            _ => return Err(PyValueError::new_err(
                "order must be 'preorder', 'inorder', or 'postorder'"
            )),
        };
        let indices: Vec<usize> = self.tree.iter_indices(traversal).collect();
        Ok(PySpeciesTreeIter {
            tree: Arc::clone(&self.tree),
            indices,
            pos: 0,
        })
    }

    /// Default iteration (preorder traversal).
    ///
    /// # Example
    /// ```python
    /// for node in species_tree:
    ///     print(node.name, node.length)
    /// total_length = sum(node.length for node in species_tree)
    /// ```
    fn __iter__(&self) -> PySpeciesTreeIter {
        use crate::node::TraversalOrder;
        let indices: Vec<usize> = self.tree.iter_indices(TraversalOrder::PreOrder).collect();
        PySpeciesTreeIter {
            tree: Arc::clone(&self.tree),
            indices,
            pos: 0,
        }
    }

    /// Uniformly sample leaf names from extant species in the tree.
    ///
    /// Exactly one of `n` or `fraction` must be provided.
    ///
    /// # Arguments
    /// * `n` - Number of leaf names to sample
    /// * `fraction` - Fraction of leaves to sample (0.0–1.0)
    /// * `seed` - Optional random seed for reproducibility
    ///
    /// # Returns
    /// A list of sampled leaf names.
    ///
    /// # Example
    /// ```python
    /// species_tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)
    /// names = species_tree.sample_leaf_names(n=20, seed=42)
    /// names = species_tree.sample_leaf_names(fraction=0.5, seed=42)
    /// ```
    #[pyo3(signature = (n=None, fraction=None, seed=None))]
    fn sample_leaf_names(&self, n: Option<usize>, fraction: Option<f64>, seed: Option<u64>) -> PyResult<Vec<String>> {
        use rand::seq::SliceRandom;

        let extant_leaves = self.tree.get_extant_leaves();
        if extant_leaves.is_empty() {
            // Fall back to structural leaves for trees without bd_event annotations
            let all_leaves = self.tree.get_leaves();
            if all_leaves.is_empty() {
                return Err(PyValueError::new_err("Tree has no leaves"));
            }
            let count = match (n, fraction) {
                (Some(n), None) => n,
                (None, Some(f)) => {
                    if !(0.0..=1.0).contains(&f) {
                        return Err(PyValueError::new_err("fraction must be between 0.0 and 1.0"));
                    }
                    (all_leaves.len() as f64 * f).round() as usize
                }
                _ => return Err(PyValueError::new_err("Exactly one of 'n' or 'fraction' must be provided")),
            };
            if count == 0 || count > all_leaves.len() {
                return Err(PyValueError::new_err(
                    format!("Cannot sample {} leaves from {} available", count, all_leaves.len())
                ));
            }
            let mut rng = init_rng(seed);
            let names: Vec<String> = all_leaves
                .choose_multiple(&mut rng, count)
                .map(|n| n.name.clone())
                .collect();
            return Ok(names);
        }

        let count = match (n, fraction) {
            (Some(n), None) => n,
            (None, Some(f)) => {
                if !(0.0..=1.0).contains(&f) {
                    return Err(PyValueError::new_err("fraction must be between 0.0 and 1.0"));
                }
                (extant_leaves.len() as f64 * f).round() as usize
            }
            _ => return Err(PyValueError::new_err("Exactly one of 'n' or 'fraction' must be provided")),
        };
        if count == 0 || count > extant_leaves.len() {
            return Err(PyValueError::new_err(
                format!("Cannot sample {} leaves from {} extant leaves", count, extant_leaves.len())
            ));
        }

        let mut rng = init_rng(seed);
        let names: Vec<String> = extant_leaves
            .choose_multiple(&mut rng, count)
            .map(|n| n.name.clone())
            .collect();
        Ok(names)
    }

    /// Extract an induced subtree keeping only the specified leaf names.
    ///
    /// This method creates a new species tree containing only the specified leaves
    /// and their most recent common ancestors (MRCAs). The resulting tree preserves
    /// the topology and branch lengths among the selected species.
    ///
    /// # Arguments
    /// * `names` - List of leaf names to keep in the subtree
    ///
    /// # Returns
    /// A new PySpeciesTree containing only the specified leaves and their MRCAs.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The names list is empty
    /// - No matching leaves are found in the tree
    /// - Failed to construct the induced subtree
    ///
    /// # Example
    /// ```python
    /// import rustree
    ///
    /// # Simulate or load a species tree
    /// tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.5, seed=42)
    ///
    /// # Get all leaf names
    /// all_leaves = tree.leaf_names()
    /// print(f"Original tree has {len(all_leaves)} leaves")
    ///
    /// # Extract a subtree with only 3 species
    /// selected_species = all_leaves[:3]
    /// subtree = tree.extract_induced_subtree_by_names(selected_species)
    /// print(f"Subtree has {subtree.num_leaves()} leaves")
    /// ```
    fn extract_induced_subtree_by_names(&self, names: Vec<String>) -> PyResult<PySpeciesTree> {
        if names.is_empty() {
            return Err(PyValueError::new_err("Leaf names list cannot be empty"));
        }

        let (induced_tree, _) = extract_induced_subtree_by_names(&self.tree, &names)
            .ok_or_else(|| PyValueError::new_err(
                "Failed to extract induced subtree (no matching leaves found or subtree extraction failed)"
            ))?;

        Ok(PySpeciesTree {
            tree: Arc::new(induced_tree),
        })
    }

    /// Extract only extant species from the tree (remove extinct lineages).
    ///
    /// Returns a new species tree containing only leaves marked as extant
    /// (`bd_event == Some(BDEvent::Leaf)`). Extinct lineages
    /// (`bd_event == Some(BDEvent::Extinction)`) are removed.
    ///
    /// This method only works on trees from `simulate_species_tree()`.
    /// Trees parsed from Newick files have `bd_event == None` and cannot
    /// be filtered this way.
    ///
    /// Returns
    /// -------
    /// PySpeciesTree
    ///     A new species tree with only extant species.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If no extant species are found (all lineages went extinct or
    ///     the tree lacks bd_event annotations).
    ///
    /// Examples
    /// --------
    /// >>> import rustree
    /// >>> # Simulate tree with 20 extant species (may include extinct lineages)
    /// >>> tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)
    /// >>> print(f"Original: {tree.num_leaves()} leaves")
    /// Original: 61 leaves
    /// >>> # Extract only extant species
    /// >>> extant_only = tree.sample_extant()
    /// >>> print(f"Extant: {extant_only.num_leaves()} leaves")
    /// Extant: 20 leaves
    fn sample_extant(&self) -> PyResult<PySpeciesTree> {
        // Call Rust function
        let (extant_tree, _) = extract_extant_subtree(&self.tree)
            .ok_or_else(|| PyValueError::new_err(
                "No extant species found in tree. Either all lineages went extinct, \
                 or the tree lacks bd_event annotations (trees from parse_species_tree \
                 cannot be filtered by extinction status)."
            ))?;

        // Return new PySpeciesTree
        Ok(PySpeciesTree {
            tree: Arc::new(extant_tree),
        })
    }

    /// Save the species tree to a Newick file.
    ///
    /// # Arguments
    /// * `filepath` - Path to save the Newick file
    fn save_newick(&self, filepath: &str) -> PyResult<()> {
        let newick = self.to_newick()?;
        fs::write(filepath, newick)
            .map_err(|e| PyValueError::new_err(format!("Failed to write Newick file: {}", e)))?;
        Ok(())
    }

    /// Generate an SVG visualization of the species tree using thirdkind.
    ///
    /// Requires thirdkind to be installed (`cargo install thirdkind`).
    ///
    /// # Arguments
    /// * `filepath` - Optional path to save the SVG file. If None, returns SVG as string.
    /// * `open_browser` - If True, opens the SVG in a web browser (default: False)
    /// * `internal_names` - If True, display internal node names (default: True)
    /// * `color` - Color for the tree (e.g. "blue", "#4A38C4")
    /// * `fontsize` - Font size for node labels
    /// * `thickness` - Thickness of tree lines
    /// * `symbol_size` - Size of event symbols (circles, etc.)
    /// * `background` - Background color (e.g. "white", "black")
    /// * `landscape` - If True, display in landscape orientation (default: False)
    /// * `sampled_species_names` - Optional list of sampled species leaf names. When provided,
    ///   node labels are colored by their NodeMark: Keep, HasDescendant, or Discard.
    /// * `keep_color` - Color for Keep nodes (default: "green")
    /// * `has_descendant_color` - Color for HasDescendant nodes (default: "orange")
    /// * `discard_color` - Color for Discard nodes (default: "grey")
    ///
    /// # Returns
    /// The SVG content as a string.
    #[pyo3(signature = (
        filepath=None, open_browser=false, internal_names=true,
        color=None, fontsize=None, thickness=None,
        symbol_size=None, background=None, landscape=false,
        sampled_species_names=None,
        keep_color="green", has_descendant_color="orange", discard_color="grey"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn to_svg(
        &self,
        filepath: Option<&str>,
        open_browser: bool,
        internal_names: bool,
        color: Option<&str>,
        fontsize: Option<f64>,
        thickness: Option<f64>,
        symbol_size: Option<f64>,
        background: Option<&str>,
        landscape: bool,
        sampled_species_names: Option<Vec<String>>,
        keep_color: &str,
        has_descendant_color: &str,
        discard_color: &str,
    ) -> PyResult<String> {
        let marking_nodes = sampled_species_names.is_some();

        let temp_dir = std::env::temp_dir();
        let nwk_path = temp_dir.join("rustree_species_temp.nwk");
        let svg_path = temp_dir.join("rustree_species_temp.svg");
        let conf_path = temp_dir.join("rustree_species_thirdkind.conf");

        // Write Newick to temp file
        let newick = self.to_newick()?;
        fs::write(&nwk_path, &newick)
            .map_err(|e| PyValueError::new_err(format!("Failed to write temp Newick: {}", e)))?;

        // Build thirdkind config file
        let mut conf_lines = Vec::new();
        if let Some(c) = color { conf_lines.push(format!("single_gene_color:{}", c)); }
        if let Some(s) = fontsize { conf_lines.push(format!("gene_police_size:{}", s)); }
        fs::write(&conf_path, conf_lines.join("\n"))
            .map_err(|e| PyValueError::new_err(format!("Failed to write config file: {}", e)))?;

        // Call thirdkind
        let mut cmd = Command::new("thirdkind");
        cmd.arg("--input-file").arg(&nwk_path)
           .arg("-o").arg(&svg_path)
           .arg("-c").arg(&conf_path);

        if internal_names || marking_nodes { cmd.arg("-i"); }
        if open_browser { cmd.arg("-b"); }
        if landscape { cmd.arg("-L"); }
        if let Some(c) = color { cmd.arg("-C").arg(c); }
        if let Some(t) = thickness { cmd.arg("-z").arg(t.to_string()); }
        if let Some(s) = symbol_size { cmd.arg("-k").arg(s.to_string()); }
        if let Some(c) = background { cmd.arg("-Q").arg(c); }

        let output = cmd.output()
            .map_err(|e| PyValueError::new_err(format!(
                "Failed to run thirdkind. Is it installed? (`cargo install thirdkind`)\nError: {}", e
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PyValueError::new_err(format!("thirdkind failed: {}", stderr)));
        }

        let mut svg = fs::read_to_string(&svg_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to read SVG output: {}", e)))?;

        // Post-process SVG to color node labels by NodeMark
        // Thirdkind renders Newick trees with class="gene" for text elements:
        //   <text class="gene" ...>\nNodeName\n</text>
        if let Some(ref names) = sampled_species_names {
            let keep_indices = find_leaf_indices_by_names(&self.tree, names);
            let mut marks = vec![NodeMark::Discard; self.tree.nodes.len()];
            mark_nodes_postorder(&self.tree, self.tree.root, &keep_indices, &mut marks);

            for (idx, node) in self.tree.nodes.iter().enumerate() {
                let node_color = match marks[idx] {
                    NodeMark::Keep => keep_color,
                    NodeMark::HasDescendant => has_descendant_color,
                    NodeMark::Discard => discard_color,
                };
                let new_attr = format!("class=\"gene\" style=\"fill: {}\"", node_color);
                let search = "class=\"gene\"";
                let name_pattern = format!("\n{}\n</text>", node.name);
                if let Some(pos) = svg.find(&name_pattern) {
                    if let Some(class_start) = svg[..pos].rfind(search) {
                        let class_end = class_start + search.len();
                        svg = format!(
                            "{}{}{}",
                            &svg[..class_start],
                            &new_attr,
                            &svg[class_end..]
                        );
                    }
                }
            }
        }

        if let Some(path) = filepath {
            fs::write(path, &svg)
                .map_err(|e| PyValueError::new_err(format!("Failed to write SVG file: {}", e)))?;
        }

        let _ = fs::remove_file(&nwk_path);
        let _ = fs::remove_file(&svg_path);
        let _ = fs::remove_file(&conf_path);

        Ok(svg)
    }

    /// Display the species tree visualization in a Jupyter notebook.
    ///
    /// Requires thirdkind to be installed and IPython/Jupyter environment.
    /// Accepts the same styling options as `to_svg()`.
    #[pyo3(signature = (
        internal_names=true,
        color=None, fontsize=None, thickness=None,
        symbol_size=None, background=None, landscape=false,
        sampled_species_names=None,
        keep_color="green", has_descendant_color="orange", discard_color="grey"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn display(
        &self,
        py: Python,
        internal_names: bool,
        color: Option<&str>,
        fontsize: Option<f64>,
        thickness: Option<f64>,
        symbol_size: Option<f64>,
        background: Option<&str>,
        landscape: bool,
        sampled_species_names: Option<Vec<String>>,
        keep_color: &str,
        has_descendant_color: &str,
        discard_color: &str,
    ) -> PyResult<PyObject> {
        let svg = self.to_svg(
            None, false, internal_names,
            color, fontsize, thickness,
            symbol_size, background, landscape,
            sampled_species_names,
            keep_color, has_descendant_color, discard_color,
        )?;

        let ipython_display = py.import("IPython.display")?;
        let svg_class = ipython_display.getattr("SVG")?;
        let display_obj = svg_class.call1((svg,))?;

        Ok(display_obj.into())
    }

    /// Save birth-death events to a CSV file.
    ///
    /// For trees from simulate_species_tree, saves the actual simulation events.
    /// For trees from parse_species_tree, generates events by treating internal nodes
    /// as speciations and leaves as extant species (no extinction events).
    ///
    /// # Arguments
    /// * `filepath` - Path to save the CSV file
    ///
    /// # Example
    /// ```python
    /// import rustree
    ///
    /// # Simulate a tree
    /// tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.3, seed=42)
    ///
    /// # Save birth-death events to CSV
    /// tree.save_bd_events_csv("events.csv")
    /// ```
    #[pyo3(signature = (filepath, eps=None))]
    fn save_bd_events_csv(&self, filepath: &str, eps: Option<f64>) -> PyResult<()> {
        use crate::bd::{generate_events_from_tree, generate_events_with_extinction};
        use crate::io::save_bd_events_to_csv;

        let events = match eps {
            Some(eps) => generate_events_with_extinction(&self.tree, eps),
            None => generate_events_from_tree(&self.tree),
        }.map_err(|e| PyValueError::new_err(e))?;

        save_bd_events_to_csv(&events, &self.tree, filepath)
            .map_err(|e| PyValueError::new_err(format!("Failed to write CSV file: {}", e)))?;

        Ok(())
    }

    /// Get birth-death events as a dictionary.
    ///
    /// Returns a dictionary with keys:
    /// - 'time': List of event times (floats)
    /// - 'node_name': List of node names (strings)
    /// - 'event_type': List of event types ('Speciation', 'Extinction', or 'Leaf')
    /// - 'child1_name': List of first child names (strings, empty for non-speciation events)
    /// - 'child2_name': List of second child names (strings, empty for non-speciation events)
    ///
    /// For trees from simulate_species_tree, returns the actual simulation events.
    /// For trees from parse_species_tree, generates events from tree structure.
    ///
    /// # Returns
    /// A dictionary containing lists of event data, suitable for pandas DataFrame creation.
    ///
    /// # Example
    /// ```python
    /// import rustree
    /// import pandas as pd
    ///
    /// # Simulate a tree
    /// tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.3, seed=42)
    ///
    /// # Get events as dict
    /// events = tree.get_bd_events()
    ///
    /// # Convert to pandas DataFrame
    /// df = pd.DataFrame(events)
    /// print(df.head())
    /// ```
    #[pyo3(signature = (eps=None))]
    fn get_bd_events(&self, py: Python, eps: Option<f64>) -> PyResult<PyObject> {
        use crate::bd::{generate_events_from_tree, generate_events_with_extinction};
        use pyo3::types::PyDict;

        let events = match eps {
            Some(eps) => generate_events_with_extinction(&self.tree, eps),
            None => generate_events_from_tree(&self.tree),
        }.map_err(|e| PyValueError::new_err(e))?;

        // Build dictionary with lists
        let dict = PyDict::new(py);

        let times: Vec<f64> = events.iter().map(|e| e.time).collect();
        let node_names: Vec<String> = events.iter()
            .map(|e| self.tree.nodes[e.node_id].name.clone())
            .collect();
        let event_types: Vec<&str> = events.iter()
            .map(|e| e.event_type.as_str())
            .collect();
        let child1_names: Vec<String> = events.iter()
            .map(|e| e.child1.map_or(String::new(), |c| self.tree.nodes[c].name.clone()))
            .collect();
        let child2_names: Vec<String> = events.iter()
            .map(|e| e.child2.map_or(String::new(), |c| self.tree.nodes[c].name.clone()))
            .collect();

        dict.set_item("time", times)?;
        dict.set_item("node_name", node_names)?;
        dict.set_item("event_type", event_types)?;
        dict.set_item("child1_name", child1_names)?;
        dict.set_item("child2_name", child2_names)?;

        Ok(dict.into())
    }

    /// Get Lineages Through Time (LTT) data for plotting.
    ///
    /// Returns a dictionary with:
    /// - `times`: Time points (going backward from present at 0)
    /// - `lineages`: Number of lineages alive at each time point
    ///
    /// This data can be directly plotted to visualize lineage accumulation/extinction
    /// over the tree's history.
    ///
    /// # Example
    /// ```python
    /// import rustree
    /// import matplotlib.pyplot as plt
    ///
    /// # Simulate or parse a species tree
    /// tree = rustree.simulate_species_tree(100, lambda_val=1.0, mu=0.5, seed=42)
    ///
    /// # Get LTT data
    /// ltt = tree.get_ltt_data()
    ///
    /// # Plot
    /// plt.plot(ltt['times'], ltt['lineages'])
    /// plt.xlabel('Time (past to present)')
    /// plt.ylabel('Number of lineages')
    /// plt.title('Lineages Through Time')
    /// plt.show()
    /// ```
    #[pyo3(signature = (eps=None))]
    fn get_ltt_data(&self, py: Python, eps: Option<f64>) -> PyResult<PyObject> {
        use crate::bd::{generate_events_from_tree, generate_events_with_extinction};
        use pyo3::types::PyDict;

        // Get all events
        let mut events = match eps {
            Some(eps) => generate_events_with_extinction(&self.tree, eps),
            None => generate_events_from_tree(&self.tree),
        }.map_err(|e| PyValueError::new_err(e))?;

        // Sort events by time (ascending, from present to past)
        events.sort_by(|a, b| a.time.total_cmp(&b.time));

        // Build LTT data
        let mut times = Vec::new();
        let mut lineages = Vec::new();

        // Start at present (time = 0) with n extant species
        let n_extant = self.tree.nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .count();

        let mut current_lineages = n_extant;
        times.push(0.0);
        lineages.push(current_lineages);

        // Process events backward in time
        for event in events.iter() {
            match event.event_type.as_str() {
                "Leaf" => {
                    // Leaf events at present don't change count (already counted)
                    continue;
                }
                "Extinction" => {
                    // Going backward: an extinction means a lineage appears
                    current_lineages += 1;
                }
                "Speciation" => {
                    // Going backward: a speciation means two lineages merge into one
                    current_lineages -= 1;
                }
                _ => continue,
            }
            times.push(event.time);
            lineages.push(current_lineages);
        }

        // Create dictionary
        let dict = PyDict::new(py);
        dict.set_item("times", times)?;
        dict.set_item("lineages", lineages)?;

        Ok(dict.into())
    }

    /// Plot Lineages Through Time (LTT) using matplotlib.
    ///
    /// Creates and displays an LTT plot showing how the number of lineages
    /// changes over time.
    ///
    /// # Arguments
    /// * `filepath` - Optional path to save the plot. If None, displays interactively.
    /// * `title` - Optional plot title (default: "Lineages Through Time")
    /// * `xlabel` - Optional x-axis label (default: "Time (past to present)")
    /// * `ylabel` - Optional y-axis label (default: "Number of lineages")
    ///
    /// # Example
    /// ```python
    /// import rustree
    ///
    /// tree = rustree.simulate_species_tree(100, lambda_val=1.0, mu=0.5, seed=42)
    ///
    /// # Display plot
    /// tree.plot_ltt()
    ///
    /// # Save to file
    /// tree.plot_ltt(filepath="ltt_plot.png", title="My LTT Plot")
    /// ```
    #[pyo3(signature = (filepath=None, title=None, xlabel=None, ylabel=None))]
    fn plot_ltt(
        &self,
        py: Python,
        filepath: Option<&str>,
        title: Option<&str>,
        xlabel: Option<&str>,
        ylabel: Option<&str>,
    ) -> PyResult<()> {
        // Get LTT data
        let ltt_data = self.get_ltt_data(py, None)?;

        // Import matplotlib
        let plt = py.import("matplotlib.pyplot")?;

        // Extract data from dictionary
        let dict = ltt_data.downcast_bound::<pyo3::types::PyDict>(py)?;
        let times = dict.get_item("times")?
            .ok_or_else(|| PyValueError::new_err("Missing 'times' in LTT data"))?;
        let lineages = dict.get_item("lineages")?
            .ok_or_else(|| PyValueError::new_err("Missing 'lineages' in LTT data"))?;

        // Create plot
        plt.call_method1("plot", (times, lineages))?;
        plt.call_method1("xlabel", (xlabel.unwrap_or("Time (past to present)"),))?;
        plt.call_method1("ylabel", (ylabel.unwrap_or("Number of lineages)"),))?;
        plt.call_method1("title", (title.unwrap_or("Lineages Through Time"),))?;
        plt.call_method0("grid")?;

        // Save or show
        if let Some(path) = filepath {
            plt.call_method1("savefig", (path,))?;
        } else {
            plt.call_method0("show")?;
        }

        Ok(())
    }

    /// Simulate a gene tree along this species tree using the DTL model.
    ///
    /// # Arguments
    /// * `lambda_d` - Duplication rate per unit time
    /// * `lambda_t` - Transfer rate per unit time
    /// * `lambda_l` - Loss rate per unit time
    /// * `transfer_alpha` - Distance decay for assortative transfers (optional, None = uniform)
    /// * `require_extant` - If true, retry until a tree with extant genes is produced (default false)
    /// * `seed` - Random seed for reproducibility (optional)
    ///
    /// # Returns
    /// A PyGeneTree containing the simulated gene tree with its mapping to the species tree.
    #[pyo3(signature = (lambda_d, lambda_t, lambda_l, transfer_alpha=None, replacement_transfer=None, require_extant=false, seed=None))]
    fn simulate_dtl(&self, lambda_d: f64, lambda_t: f64, lambda_l: f64, transfer_alpha: Option<f64>, replacement_transfer: Option<f64>, require_extant: bool, seed: Option<u64>) -> PyResult<PyGeneTree> {
        validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;
        validate_replacement_transfer(replacement_transfer)?;
        let mut rng = init_rng(seed);
        let origin_species = self.tree.root;
        let (mut rec_tree, events) = simulate_dtl(&self.tree, origin_species, lambda_d, lambda_t, lambda_l, transfer_alpha, replacement_transfer, require_extant, &mut rng)
            .map_err(|e| PyValueError::new_err(e))?;

        // Share the PySpeciesTree's Arc instead of the simulation's internal clone
        rec_tree.species_tree = Arc::clone(&self.tree);
        rec_tree.dtl_events = Some(events);
        Ok(PyGeneTree { rec_tree })
    }

    /// Simulate multiple gene trees efficiently with shared pre-computed data.
    ///
    /// This is faster than calling simulate_dtl multiple times because depths
    /// and contemporaneity are computed only once.
    ///
    /// # Arguments
    /// * `n` - Number of gene trees to simulate
    /// * `lambda_d` - Duplication rate per unit time
    /// * `lambda_t` - Transfer rate per unit time
    /// * `lambda_l` - Loss rate per unit time
    /// * `transfer_alpha` - Distance decay for assortative transfers (optional, None = uniform)
    /// * `require_extant` - If true, only include trees with extant genes (default false)
    /// * `seed` - Random seed for reproducibility (optional)
    ///
    /// # Returns
    /// A list of PyGeneTree objects.
    #[pyo3(signature = (n, lambda_d, lambda_t, lambda_l, transfer_alpha=None, replacement_transfer=None, require_extant=false, seed=None))]
    fn simulate_dtl_batch(&self, n: usize, lambda_d: f64, lambda_t: f64, lambda_l: f64, transfer_alpha: Option<f64>, replacement_transfer: Option<f64>, require_extant: bool, seed: Option<u64>) -> PyResult<PyGeneForest> {
        validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;
        validate_replacement_transfer(replacement_transfer)?;
        let mut rng = init_rng(seed);

        let origin_species = self.tree.root;
        let (rec_trees, all_events) = simulate_dtl_batch(
            &self.tree,
            origin_species,
            lambda_d,
            lambda_t,
            lambda_l,
            transfer_alpha,
            replacement_transfer,
            n,
            require_extant,
            &mut rng,
        ).map_err(|e| PyValueError::new_err(e))?;

        let gene_trees: Vec<RecTree> = rec_trees
            .into_iter()
            .zip(all_events.into_iter())
            .map(|(mut rec_tree, events)| {
                rec_tree.species_tree = Arc::clone(&self.tree);
                rec_tree.dtl_events = Some(events);
                rec_tree
            })
            .collect();

        Ok(PyGeneForest {
            forest: crate::node::gene_forest::GeneForest::from_rec_trees(
                Arc::clone(&self.tree), gene_trees,
            ),
        })
    }

    /// Simulate a gene tree using the Zombi-style per-species DTL model.
    ///
    /// In this model, the event rate is proportional to the number of SPECIES
    /// with gene copies, NOT the number of gene copies themselves. This means
    /// duplications don't increase the event rate.
    ///
    /// When an event occurs:
    /// 1. A random species (among those with genes) is chosen
    /// 2. A random gene copy within that species is affected
    ///
    /// This is the model used by Zombi and is appropriate when DTL events are
    /// driven by species-level processes rather than gene-copy-level processes.
    ///
    /// # Arguments
    /// * `lambda_d` - Duplication rate per species per unit time
    /// * `lambda_t` - Transfer rate per species per unit time
    /// * `lambda_l` - Loss rate per species per unit time
    /// * `transfer_alpha` - Distance decay for assortative transfers (optional, None = uniform)
    /// * `require_extant` - If true, retry until at least one extant gene exists (default false)
    /// * `seed` - Random seed for reproducibility (optional)
    ///
    /// # Returns
    /// A PyGeneTree containing the simulated gene tree with its mapping to the species tree.
    ///
    /// # Example
    /// ```python
    /// import rustree
    ///
    /// # Simulate species tree
    /// species_tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)
    ///
    /// # Compare per-gene-copy model vs per-species model
    /// gene_tree_per_copy = species_tree.simulate_dtl(0.5, 0.2, 0.3, seed=123)
    /// gene_tree_per_species = species_tree.simulate_dtl_per_species(0.5, 0.2, 0.3, seed=123)
    ///
    /// print(f"Per-copy model: {gene_tree_per_copy.num_leaves()} leaves")
    /// print(f"Per-species model: {gene_tree_per_species.num_leaves()} leaves")
    /// # Per-species typically has fewer leaves since duplications don't increase rate
    /// ```
    #[pyo3(signature = (lambda_d, lambda_t, lambda_l, transfer_alpha=None, replacement_transfer=None, require_extant=false, seed=None))]
    fn simulate_dtl_per_species(&self, lambda_d: f64, lambda_t: f64, lambda_l: f64, transfer_alpha: Option<f64>, replacement_transfer: Option<f64>, require_extant: bool, seed: Option<u64>) -> PyResult<PyGeneTree> {
        validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;
        validate_replacement_transfer(replacement_transfer)?;
        let mut rng = init_rng(seed);
        let origin_species = self.tree.root;
        let (mut rec_tree, events) = simulate_dtl_per_species(&self.tree, origin_species, lambda_d, lambda_t, lambda_l, transfer_alpha, replacement_transfer, require_extant, &mut rng)
            .map_err(|e| PyValueError::new_err(e))?;

        rec_tree.species_tree = Arc::clone(&self.tree);
        rec_tree.dtl_events = Some(events);
        Ok(PyGeneTree { rec_tree })
    }

    /// Simulate multiple gene trees using the Zombi-style per-species DTL model.
    ///
    /// This is faster than calling simulate_dtl_per_species multiple times because
    /// species tree events and LCA depths are computed only once.
    ///
    /// # Arguments
    /// * `n` - Number of gene trees to simulate
    /// * `lambda_d` - Duplication rate per species per unit time
    /// * `lambda_t` - Transfer rate per species per unit time
    /// * `lambda_l` - Loss rate per species per unit time
    /// * `transfer_alpha` - Distance decay for assortative transfers (optional, None = uniform)
    /// * `require_extant` - If true, only include trees with extant genes (default false)
    /// * `seed` - Random seed for reproducibility (optional)
    ///
    /// # Returns
    /// A list of PyGeneTree objects.
    ///
    /// # Example
    /// ```python
    /// import rustree
    ///
    /// # Simulate species tree
    /// species_tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)
    ///
    /// # Simulate 100 gene trees efficiently
    /// gene_trees = species_tree.simulate_dtl_per_species_batch(
    ///     n=100,
    ///     lambda_d=0.5,
    ///     lambda_t=0.2,
    ///     lambda_l=0.3,
    ///     seed=123
    /// )
    /// print(f"Generated {len(gene_trees)} gene trees")
    /// ```
    #[pyo3(signature = (n, lambda_d, lambda_t, lambda_l, transfer_alpha=None, replacement_transfer=None, require_extant=false, seed=None))]
    fn simulate_dtl_per_species_batch(&self, n: usize, lambda_d: f64, lambda_t: f64, lambda_l: f64, transfer_alpha: Option<f64>, replacement_transfer: Option<f64>, require_extant: bool, seed: Option<u64>) -> PyResult<PyGeneForest> {
        validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;
        validate_replacement_transfer(replacement_transfer)?;
        let mut rng = init_rng(seed);
        let origin_species = self.tree.root;
        let (rec_trees, all_events) = simulate_dtl_per_species_batch(
            &self.tree,
            origin_species,
            lambda_d,
            lambda_t,
            lambda_l,
            transfer_alpha,
            replacement_transfer,
            n,
            require_extant,
            &mut rng,
        ).map_err(|e| PyValueError::new_err(e))?;

        let gene_trees: Vec<RecTree> = rec_trees
            .into_iter()
            .zip(all_events.into_iter())
            .map(|(mut rec_tree, events)| {
                rec_tree.species_tree = Arc::clone(&self.tree);
                rec_tree.dtl_events = Some(events);
                rec_tree
            })
            .collect();

        Ok(PyGeneForest {
            forest: crate::node::gene_forest::GeneForest::from_rec_trees(
                Arc::clone(&self.tree), gene_trees,
            ),
        })
    }

    /// Create a lazy iterator that generates gene trees one at a time (per-gene-copy model).
    ///
    /// Each call to `next()` on the returned iterator runs one Gillespie simulation
    /// and yields a PyGeneTree. Only one gene tree lives in memory at a time.
    ///
    /// The per-gene-copy model means the DTL event rate scales with the number
    /// of gene copies (more copies = higher total rate).
    ///
    /// # Arguments
    /// * `lambda_d` - Duplication rate per unit time
    /// * `lambda_t` - Transfer rate per unit time
    /// * `lambda_l` - Loss rate per unit time
    /// * `transfer_alpha` - Distance decay for assortative transfers (optional, None = uniform)
    /// * `replacement_transfer` - Probability of replacement transfer (optional, 0.0-1.0)
    /// * `n` - Number of gene trees to generate (default 1)
    /// * `require_extant` - If true, retry until a tree with extant genes is produced (default false)
    /// * `seed` - Random seed for reproducibility (optional)
    ///
    /// # Returns
    /// A PyDtlSimIter that implements the Python iterator protocol.
    ///
    /// # Example
    /// ```python
    /// import rustree
    ///
    /// species_tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)
    ///
    /// # Stream 1000 trees directly to XML files:
    /// species_tree.simulate_dtl_iter(0.5, 0.2, 0.3, n=1000, seed=42).save_xml("output/")
    ///
    /// # Iterate one at a time:
    /// for gene_tree in species_tree.simulate_dtl_iter(0.5, 0.2, 0.3, n=100, seed=42):
    ///     print(gene_tree.num_extant())
    ///
    /// # Get a single tree:
    /// gene_tree = species_tree.simulate_dtl_iter(0.5, 0.2, 0.3, seed=42).single()
    /// ```
    #[pyo3(signature = (lambda_d, lambda_t, lambda_l, transfer_alpha=None, replacement_transfer=None, n=1, require_extant=false, seed=None))]
    fn simulate_dtl_iter(
        &self,
        lambda_d: f64,
        lambda_t: f64,
        lambda_l: f64,
        transfer_alpha: Option<f64>,
        replacement_transfer: Option<f64>,
        n: usize,
        require_extant: bool,
        seed: Option<u64>,
    ) -> PyResult<PyDtlSimIter> {
        validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;
        validate_replacement_transfer(replacement_transfer)?;

        let species_arc = Arc::clone(&self.tree);
        let species_events = generate_events_from_tree(&self.tree)
            .map_err(|e| PyValueError::new_err(e))?;
        let depths = self.tree.make_subdivision();
        let contemporaneity = self.tree.find_contemporaneity(&depths);
        let lca_depths = transfer_alpha.map(|_| {
            self.tree.precompute_lca_depths()
                .expect("Failed to precompute LCA depths")
        });
        let rng = init_rng(seed);
        let origin_species = self.tree.root;

        Ok(PyDtlSimIter {
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
            n_simulations: n,
            require_extant,
            mode: DTLMode::PerGene,
            rng,
            completed: 0,
        })
    }

    /// Create a lazy iterator that generates gene trees one at a time (per-species model).
    ///
    /// Each call to `next()` on the returned iterator runs one Gillespie simulation
    /// and yields a PyGeneTree. Only one gene tree lives in memory at a time.
    ///
    /// In the per-species (Zombi-style) model, the DTL event rate scales with
    /// the number of alive species, NOT the number of gene copies.
    ///
    /// # Arguments
    /// * `lambda_d` - Duplication rate per species per unit time
    /// * `lambda_t` - Transfer rate per species per unit time
    /// * `lambda_l` - Loss rate per species per unit time
    /// * `transfer_alpha` - Distance decay for assortative transfers (optional, None = uniform)
    /// * `replacement_transfer` - Probability of replacement transfer (optional, 0.0-1.0)
    /// * `n` - Number of gene trees to generate (default 1)
    /// * `require_extant` - If true, retry until a tree with extant genes is produced (default false)
    /// * `seed` - Random seed for reproducibility (optional)
    ///
    /// # Returns
    /// A PyDtlSimIter that implements the Python iterator protocol.
    ///
    /// # Example
    /// ```python
    /// import rustree
    ///
    /// species_tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)
    ///
    /// # Stream 1000 trees directly to Newick files:
    /// species_tree.simulate_dtl_per_species_iter(0.5, 0.2, 0.3, n=1000, seed=42).save_newick("output/")
    ///
    /// # Collect all into a list:
    /// gene_trees = species_tree.simulate_dtl_per_species_iter(0.5, 0.2, 0.3, n=50, seed=42).collect_all()
    /// ```
    #[pyo3(signature = (lambda_d, lambda_t, lambda_l, transfer_alpha=None, replacement_transfer=None, n=1, require_extant=false, seed=None))]
    fn simulate_dtl_per_species_iter(
        &self,
        lambda_d: f64,
        lambda_t: f64,
        lambda_l: f64,
        transfer_alpha: Option<f64>,
        replacement_transfer: Option<f64>,
        n: usize,
        require_extant: bool,
        seed: Option<u64>,
    ) -> PyResult<PyDtlSimIter> {
        validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;
        validate_replacement_transfer(replacement_transfer)?;

        let species_arc = Arc::clone(&self.tree);
        let species_events = generate_events_from_tree(&self.tree)
            .map_err(|e| PyValueError::new_err(e))?;
        let depths = self.tree.make_subdivision();
        let contemporaneity = self.tree.find_contemporaneity(&depths);
        let lca_depths = transfer_alpha.map(|_| {
            self.tree.precompute_lca_depths()
                .expect("Failed to precompute LCA depths")
        });
        let rng = init_rng(seed);
        let origin_species = self.tree.root;

        Ok(PyDtlSimIter {
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
            n_simulations: n,
            require_extant,
            mode: DTLMode::PerSpecies,
            rng,
            completed: 0,
        })
    }

    /// Compute all pairwise distances between nodes in the species tree.
    ///
    /// # Arguments
    /// * `distance_type` - Type of distance: "topological" (number of edges) or "metric" (sum of branch lengths)
    /// * `leaves_only` - If true, only compute distances between leaf nodes (default true)
    ///
    /// # Returns
    /// A pandas DataFrame with columns: node1, node2, distance
    ///
    /// # Example
    /// ```python
    /// import rustree
    /// tree = rustree.simulate_species_tree(5, 1.0, 0.5, seed=42)
    /// df = tree.pairwise_distances("metric", leaves_only=True)
    /// print(df)
    /// ```
    #[pyo3(signature = (distance_type, leaves_only=true))]
    fn pairwise_distances(&self, py: Python, distance_type: &str, leaves_only: bool) -> PyResult<PyObject> {
        use crate::metric_functions::DistanceType;

        let dist_type = match distance_type.to_lowercase().as_str() {
            "topological" | "topo" => DistanceType::Topological,
            "metric" | "branch" | "patristic" => DistanceType::Metric,
            _ => return Err(PyValueError::new_err(format!(
                "Invalid distance_type '{}'. Use 'topological' or 'metric'.",
                distance_type
            ))),
        };

        let distances = self.tree.pairwise_distances(dist_type, leaves_only)
            .map_err(|e| PyValueError::new_err(format!("Failed to compute pairwise distances: {}", e)))?;

        let node1: Vec<&str> = distances.iter().map(|d| d.node1).collect();
        let node2: Vec<&str> = distances.iter().map(|d| d.node2).collect();
        let dist: Vec<f64> = distances.iter().map(|d| d.distance).collect();

        // Create pandas DataFrame
        let pandas = py.import("pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("node1", node1)?;
        dict.set_item("node2", node2)?;
        dict.set_item("distance", dist)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Save pairwise distances between nodes to a CSV file.
    ///
    /// Computes all pairwise distances and writes them to a CSV file with the format:
    /// node1,node2,distance
    ///
    /// # Arguments
    /// * `filepath` - Path to save the CSV file
    /// * `distance_type` - Type of distance: "topological" (number of edges) or "metric" (sum of branch lengths)
    /// * `leaves_only` - If true, only compute distances between leaf nodes (default: true)
    ///
    /// # Example
    /// ```python
    /// import rustree
    ///
    /// # Simulate a species tree
    /// tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.5, seed=42)
    ///
    /// # Save topological distances between all leaves
    /// tree.save_pairwise_distances_csv("distances.csv", "topological", leaves_only=True)
    ///
    /// # Save metric distances between all nodes
    /// tree.save_pairwise_distances_csv("all_distances.csv", "metric", leaves_only=False)
    /// ```
    #[pyo3(signature = (filepath, distance_type, leaves_only=true))]
    fn save_pairwise_distances_csv(&self, filepath: &str, distance_type: &str, leaves_only: bool) -> PyResult<()> {
        use crate::metric_functions::{DistanceType, PairwiseDistance};

        // Parse distance type
        let dist_type = match distance_type.to_lowercase().as_str() {
            "topological" | "topo" => DistanceType::Topological,
            "metric" | "branch" | "patristic" => DistanceType::Metric,
            _ => return Err(PyValueError::new_err(format!(
                "Invalid distance_type '{}'. Use 'topological' or 'metric'.",
                distance_type
            ))),
        };

        // Compute pairwise distances
        let distances = self.tree.pairwise_distances(dist_type, leaves_only)
            .map_err(|e| PyValueError::new_err(format!("Failed to compute pairwise distances: {}", e)))?;

        // Write to CSV
        let mut csv = String::from(PairwiseDistance::csv_header());
        csv.push('\n');
        for d in &distances {
            csv.push_str(&d.to_csv_row());
            csv.push('\n');
        }

        fs::write(filepath, csv)
            .map_err(|e| PyValueError::new_err(format!("Failed to write CSV file: {}", e)))?;

        Ok(())
    }

    /// Compute ghost branch lengths for a sampled species tree.
    ///
    /// The ghost length of a sampled-tree branch is the sum of branch lengths of all
    /// non-sampled nodes in the complete tree that project onto that branch.
    ///
    /// # Arguments
    /// * `sampled_leaf_names` - Names of species leaves to keep
    ///
    /// # Returns
    /// A pandas DataFrame with columns: node_index, node_name, ghost_length.
    ///
    /// # Example
    /// ```python
    /// species_tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)
    ///
    /// sampled_leaf_names = species_tree.sample_leaf_names(n=20, seed=42)
    /// df = species_tree.compute_ghost_lengths(sampled_leaf_names)
    /// print(df)
    /// ```
    fn compute_ghost_lengths(&self, py: Python, sampled_leaf_names: Vec<String>) -> PyResult<PyObject> {
        use crate::induced_transfers::ghost_lengths;
        let (sampled_tree, ghosts) = ghost_lengths(
            &self.tree,
            &sampled_leaf_names,
        );

        let node_index: Vec<usize> = (0..sampled_tree.nodes.len()).collect();
        let node_name: Vec<&str> = sampled_tree.nodes.iter().map(|n| n.name.as_str()).collect();

        let pandas = py.import("pandas")?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("node_index", node_index)?;
        dict.set_item("node_name", node_name)?;
        dict.set_item("ghost_length", ghosts)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }
}

// ============================================================================
// Species Tree Node and Iterator
// ============================================================================

/// A single node from a species tree, exposed to Python.
#[pyclass]
#[derive(Clone)]
pub struct PySpeciesNode {
    name: String,
    index: usize,
    depth: Option<f64>,
    length: f64,
    left_child: Option<usize>,
    right_child: Option<usize>,
    parent: Option<usize>,
    bd_event: Option<crate::bd::BDEvent>,
}

#[pymethods]
impl PySpeciesNode {
    #[getter]
    fn name(&self) -> &str { &self.name }
    #[getter]
    fn index(&self) -> usize { self.index }
    #[getter]
    fn depth(&self) -> Option<f64> { self.depth }
    #[getter]
    fn length(&self) -> f64 { self.length }
    #[getter]
    fn left_child(&self) -> Option<usize> { self.left_child }
    #[getter]
    fn right_child(&self) -> Option<usize> { self.right_child }
    #[getter]
    fn parent(&self) -> Option<usize> { self.parent }
    #[getter]
    fn bd_event(&self) -> Option<String> {
        self.bd_event.map(|e| format!("{:?}", e))
    }

    fn __repr__(&self) -> String {
        format!("SpeciesNode(name='{}', index={}, depth={:?}, length={:.6})",
            self.name, self.index, self.depth, self.length)
    }
}

/// Iterator over species tree nodes in a given traversal order.
#[pyclass]
struct PySpeciesTreeIter {
    tree: Arc<FlatTree>,
    indices: Vec<usize>,
    pos: usize,
}

#[pymethods]
impl PySpeciesTreeIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&mut self) -> Option<PySpeciesNode> {
        if self.pos >= self.indices.len() {
            return None;
        }
        let idx = self.indices[self.pos];
        self.pos += 1;
        let node = &self.tree.nodes[idx];
        Some(PySpeciesNode {
            name: node.name.clone(),
            index: idx,
            depth: node.depth,
            length: node.length,
            left_child: node.left_child,
            right_child: node.right_child,
            parent: node.parent,
            bd_event: node.bd_event,
        })
    }
}

// ============================================================================
// Streaming DTL Simulation Iterator
// ============================================================================

/// Lazy iterator for DTL gene tree simulation.
///
/// Generates one gene tree per `next()` call using the Gillespie algorithm.
/// Owns all its state (species tree, pre-computed data, RNG) so it can be
/// used as a Python iterator without lifetime issues.
///
/// Created by `PySpeciesTree.simulate_dtl_iter()` or
/// `PySpeciesTree.simulate_dtl_per_species_iter()`.
///
/// # Example
/// ```python
/// import rustree
///
/// species_tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)
///
/// # Create a lazy iterator for 1000 gene trees
/// it = species_tree.simulate_dtl_iter(
///     lambda_d=0.5, lambda_t=0.2, lambda_l=0.3, n=1000, seed=123
/// )
///
/// # Iterate one at a time
/// for gene_tree in it:
///     print(gene_tree.num_extant())
///
/// # Or use convenience methods:
/// it = species_tree.simulate_dtl_iter(lambda_d=0.5, lambda_t=0.2, lambda_l=0.3, n=100, seed=42)
/// it.save_xml("output/")
///
/// it = species_tree.simulate_dtl_iter(lambda_d=0.5, lambda_t=0.2, lambda_l=0.3, n=1, seed=42)
/// gene_tree = it.single()
/// ```
#[pyclass]
pub struct PyDtlSimIter {
    // Pre-computed state (owned)
    species_arc: Arc<FlatTree>,
    species_events: Vec<TreeEvent>,
    depths: Vec<f64>,
    contemporaneity: Vec<Vec<usize>>,
    lca_depths: Option<Vec<Vec<f64>>>,
    // Simulation parameters
    origin_species: usize,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Option<f64>,
    replacement_transfer: Option<f64>,
    n_simulations: usize,
    require_extant: bool,
    mode: DTLMode,
    // Mutable state
    rng: StdRng,
    completed: usize,
}

impl PyDtlSimIter {
    /// Run one Gillespie simulation and return the result.
    ///
    /// Retries if `require_extant` is true and the tree has no extant genes.
    /// Returns `None` when all requested simulations have been completed.
    fn next_simulation(&mut self) -> Option<Result<RecTree, String>> {
        if self.completed >= self.n_simulations {
            return None;
        }

        loop {
            let lca_ref = self.lca_depths.as_ref().map(|v| v.as_slice());
            let result = simulate_dtl_gillespie(
                self.mode,
                &self.species_arc,
                &self.species_events,
                &self.depths,
                &self.contemporaneity,
                lca_ref,
                self.origin_species,
                self.lambda_d,
                self.lambda_t,
                self.lambda_l,
                self.transfer_alpha,
                self.replacement_transfer,
                &mut self.rng,
            );

            match result {
                Ok((mut rec_tree, events)) => {
                    if !self.require_extant || count_extant_genes(&rec_tree) > 0 {
                        self.completed += 1;
                        rec_tree.dtl_events = Some(events);
                        return Some(Ok(rec_tree));
                    }
                    // Retry: tree had no extant genes
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

#[pymethods]
impl PyDtlSimIter {
    /// Python iterator protocol: return self.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Python iterator protocol: generate the next gene tree.
    ///
    /// Each call runs one Gillespie simulation. Returns a PyGeneTree or
    /// raises StopIteration when all requested simulations are done.
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyGeneTree>> {
        let species_arc = Arc::clone(&slf.species_arc);
        match slf.next_simulation() {
            None => Ok(None), // StopIteration
            Some(Ok(mut rec_tree)) => {
                rec_tree.species_tree = species_arc;
                Ok(Some(PyGeneTree { rec_tree }))
            }
            Some(Err(e)) => Err(PyValueError::new_err(e)),
        }
    }

    /// Number of simulations remaining.
    #[getter]
    fn remaining(&self) -> usize {
        self.n_simulations.saturating_sub(self.completed)
    }

    /// Total number of simulations requested.
    #[getter]
    fn total(&self) -> usize {
        self.n_simulations
    }

    /// Number of simulations completed so far.
    #[getter]
    fn completed(&self) -> usize {
        self.completed
    }

    /// Run a single simulation and return the result directly.
    ///
    /// Convenience for the common case of generating one tree.
    /// Equivalent to calling `next()` once.
    ///
    /// # Raises
    /// ValueError if no simulations remain or the simulation fails.
    fn single(&mut self) -> PyResult<PyGeneTree> {
        let species_arc = Arc::clone(&self.species_arc);
        match self.next_simulation() {
            None => Err(PyValueError::new_err("No simulations remaining")),
            Some(Ok(mut rec_tree)) => {
                rec_tree.species_tree = species_arc;
                Ok(PyGeneTree { rec_tree })
            }
            Some(Err(e)) => Err(PyValueError::new_err(e)),
        }
    }

    /// Collect all remaining simulations into a GeneForest.
    ///
    /// Warning: this loads all trees into memory at once.
    ///
    /// # Returns
    /// A GeneForest containing all remaining gene trees.
    fn collect_all(&mut self) -> PyResult<PyGeneForest> {
        let mut trees = Vec::with_capacity(self.n_simulations.saturating_sub(self.completed));
        loop {
            let species_arc = Arc::clone(&self.species_arc);
            match self.next_simulation() {
                None => break,
                Some(Ok(mut rec_tree)) => {
                    rec_tree.species_tree = species_arc;
                    trees.push(rec_tree);
                }
                Some(Err(e)) => return Err(PyValueError::new_err(e)),
            }
        }
        Ok(PyGeneForest {
            forest: crate::node::gene_forest::GeneForest::from_rec_trees(
                Arc::clone(&self.species_arc), trees,
            ),
        })
    }

    /// Save each remaining gene tree as RecPhyloXML to the given directory.
    ///
    /// Files are named `gene_0000.xml`, `gene_0001.xml`, etc.
    /// Creates the directory if it does not exist.
    ///
    /// # Arguments
    /// * `dir` - Directory path to save the XML files
    fn save_xml(&mut self, dir: &str) -> PyResult<()> {
        fs::create_dir_all(dir)
            .map_err(|e| PyValueError::new_err(format!("Failed to create directory: {}", e)))?;
        let width = digit_width(self.n_simulations);
        let mut idx = self.completed;
        loop {
            let species_arc = Arc::clone(&self.species_arc);
            match self.next_simulation() {
                None => break,
                Some(Ok(mut rec_tree)) => {
                    rec_tree.species_tree = species_arc;
                    let xml = rec_tree.to_xml();
                    let path = format!("{}/gene_{:0>width$}.xml", dir, idx, width = width);
                    fs::write(&path, &xml)
                        .map_err(|e| PyValueError::new_err(format!("Failed to write {}: {}", path, e)))?;
                    idx += 1;
                }
                Some(Err(e)) => return Err(PyValueError::new_err(e)),
            }
        }
        Ok(())
    }

    /// Save each remaining gene tree as Newick to the given directory.
    ///
    /// Files are named `gene_0000.nwk`, `gene_0001.nwk`, etc.
    /// Creates the directory if it does not exist.
    ///
    /// Note: Newick format does not preserve reconciliation information
    /// (species mapping, events). Use `save_xml` for full reconciled trees.
    ///
    /// # Arguments
    /// * `dir` - Directory path to save the Newick files
    fn save_newick(&mut self, dir: &str) -> PyResult<()> {
        fs::create_dir_all(dir)
            .map_err(|e| PyValueError::new_err(format!("Failed to create directory: {}", e)))?;
        let width = digit_width(self.n_simulations);
        let mut idx = self.completed;
        loop {
            let species_arc = Arc::clone(&self.species_arc);
            match self.next_simulation() {
                None => break,
                Some(Ok(mut rec_tree)) => {
                    rec_tree.species_tree = species_arc;
                    let newick = rec_tree.gene_tree.to_newick()
                        .map_err(|e| PyValueError::new_err(e))?;
                    let path = format!("{}/gene_{:0>width$}.nwk", dir, idx, width = width);
                    fs::write(&path, format!("{};", newick))
                        .map_err(|e| PyValueError::new_err(format!("Failed to write {}: {}", path, e)))?;
                    idx += 1;
                }
                Some(Err(e)) => return Err(PyValueError::new_err(e)),
            }
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        let mode_str = match self.mode {
            DTLMode::PerGene => "per_gene",
            DTLMode::PerSpecies => "per_species",
        };
        format!(
            "PyDtlSimIter(mode={}, completed={}/{}, D={}, T={}, L={})",
            mode_str, self.completed, self.n_simulations,
            self.lambda_d, self.lambda_t, self.lambda_l
        )
    }

    fn __len__(&self) -> usize {
        self.n_simulations.saturating_sub(self.completed)
    }
}

/// Compute the number of digits needed to represent n (for zero-padded filenames).
fn digit_width(n: usize) -> usize {
    if n == 0 { 1 } else { ((n as f64).log10().floor() as usize) + 1 }
}

// ============================================================================

/// Convert RecTreeColumns to a pandas DataFrame.
fn columns_to_dataframe(py: Python, cols: &RecTreeColumns) -> PyResult<PyObject> {
    let pandas = py.import("pandas")?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("node_id", &cols.node_id)?;
    dict.set_item("name", &cols.name)?;
    dict.set_item("parent", &cols.parent)?;
    dict.set_item("left_child", &cols.left_child)?;
    dict.set_item("left_child_name", &cols.left_child_name)?;
    dict.set_item("right_child", &cols.right_child)?;
    dict.set_item("right_child_name", &cols.right_child_name)?;
    dict.set_item("length", &cols.length)?;
    dict.set_item("depth", &cols.depth)?;
    dict.set_item("species_node", &cols.species_node)?;
    dict.set_item("species_node_left", &cols.species_node_left)?;
    dict.set_item("species_node_right", &cols.species_node_right)?;
    dict.set_item("event", &cols.event)?;
    let df = pandas.call_method1("DataFrame", (dict,))?;
    Ok(df.into())
}

/// A gene tree simulated under the DTL model.
#[pyclass]
#[derive(Clone)]
pub struct PyGeneTree {
    rec_tree: RecTree,
}

#[pymethods]
impl PyGeneTree {
    /// Convert the gene tree to Newick format.
    fn to_newick(&self) -> PyResult<String> {
        let nwk = self.rec_tree.gene_tree.to_newick()
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(nwk + ";")
    }

    /// Save the gene tree to a Newick file.
    ///
    /// # Arguments
    /// * `filepath` - Path to save the Newick file
    fn save_newick(&self, filepath: &str) -> PyResult<()> {
        let newick = self.to_newick()?;
        fs::write(filepath, newick)
            .map_err(|e| PyValueError::new_err(format!("Failed to write Newick file: {}", e)))?;
        Ok(())
    }

    /// Get the number of nodes in the gene tree.
    fn num_nodes(&self) -> usize {
        self.rec_tree.gene_tree.nodes.len()
    }

    /// Get the number of extant genes (leaves that survived, not losses).
    fn num_extant(&self) -> usize {
        self.rec_tree.gene_tree.nodes.iter()
            .enumerate()
            .filter(|(i, n)| {
                n.left_child.is_none()
                && n.right_child.is_none()
                && self.rec_tree.event_mapping[*i] == Event::Leaf
            })
            .count()
    }

    /// Get the number of events by type as a dictionary.
    fn count_events(&self) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        counts.insert("speciations".to_string(), 0);
        counts.insert("duplications".to_string(), 0);
        counts.insert("transfers".to_string(), 0);
        counts.insert("losses".to_string(), 0);
        counts.insert("leaves".to_string(), 0);

        for event in &self.rec_tree.event_mapping {
            match event {
                Event::Speciation => *counts.get_mut("speciations").unwrap() += 1,
                Event::Duplication => *counts.get_mut("duplications").unwrap() += 1,
                Event::Transfer => *counts.get_mut("transfers").unwrap() += 1,
                Event::Loss => *counts.get_mut("losses").unwrap() += 1,
                Event::Leaf => *counts.get_mut("leaves").unwrap() += 1,
            }
        }

        counts
    }

    /// Get names of extant genes (genes that survived to present).
    fn extant_gene_names(&self) -> Vec<String> {
        self.rec_tree.gene_tree.nodes.iter()
            .enumerate()
            .filter(|(i, n)| {
                n.left_child.is_none()
                && n.right_child.is_none()
                && self.rec_tree.event_mapping[*i] == Event::Leaf
            })
            .map(|(_, n)| n.name.clone())
            .collect()
    }

    /// Sample the gene tree by keeping only genes from extant species.
    ///
    /// Returns a new gene tree containing only the induced subtree of extant genes.
    /// If all genes are lost, returns None (represented as an error in Python).
    fn sample_extant(&self) -> PyResult<PyGeneTree> {
        // Find indices of extant genes (leaves with Event::Leaf)
        let extant_indices: std::collections::HashSet<usize> = self.rec_tree.gene_tree.nodes.iter()
            .enumerate()
            .filter(|(i, n)| {
                n.left_child.is_none()
                && n.right_child.is_none()
                && self.rec_tree.event_mapping[*i] == Event::Leaf
            })
            .map(|(i, _)| i)
            .collect();

        if extant_indices.is_empty() {
            return Err(PyValueError::new_err("No extant genes to sample"));
        }

        // Extract induced gene subtree
        let (sampled_gene_tree, gene_old_to_new) = extract_induced_subtree(&self.rec_tree.gene_tree, &extant_indices)
            .ok_or_else(|| PyValueError::new_err("Failed to extract induced subtree"))?;

        // Also prune the species tree to extant-only and get species_old_to_new mapping.
        // This ensures the returned gene tree's node_mapping references the pruned species tree,
        // which is consistent with what ALERax sees during reconciliation.
        let (sampled_species_tree, species_old_to_new) = extract_extant_subtree(&self.rec_tree.species_tree)
            .ok_or_else(|| PyValueError::new_err("Failed to extract extant species subtree"))?;

        // Use the existing remap function to correctly translate both gene and species indices
        let (new_node_mapping, new_event_mapping) = remap_gene_tree_indices(
            &sampled_gene_tree, &gene_old_to_new,
            &self.rec_tree.node_mapping, &self.rec_tree.event_mapping,
            &species_old_to_new,
        ).map_err(|e| PyValueError::new_err(e))?;

        Ok(PyGeneTree {
            rec_tree: RecTree::new(
                Arc::new(sampled_species_tree),
                sampled_gene_tree,
                new_node_mapping,
                new_event_mapping,
            ),
        })
    }

    /// Sample the gene tree by keeping only genes with the specified names.
    ///
    /// # Arguments
    /// * `names` - List of gene names to keep
    ///
    /// # Returns
    /// A new gene tree containing only the induced subtree of the specified genes.
    fn sample_by_names(&self, names: Vec<String>) -> PyResult<PyGeneTree> {
        let keep_indices = find_leaf_indices_by_names(&self.rec_tree.gene_tree, &names);

        if keep_indices.is_empty() {
            return Err(PyValueError::new_err("No matching genes found"));
        }

        let (sampled_tree, old_to_new) = extract_induced_subtree(&self.rec_tree.gene_tree, &keep_indices)
            .ok_or_else(|| PyValueError::new_err("Failed to extract induced subtree"))?;

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
                    self.rec_tree.event_mapping[*old_idx].clone()
                } else {
                    if sampled_tree.nodes[new_idx].left_child.is_none() {
                        Event::Leaf
                    } else {
                        Event::Speciation
                    }
                }
            })
            .collect();

        // Rebuild node mapping: leaves get species from gene name, internal nodes get None
        let new_node_mapping: Vec<Option<usize>> = sampled_tree.nodes.iter()
            .map(|n| {
                if n.left_child.is_none() && n.right_child.is_none() {
                    if let Some(pos) = n.name.rfind('_') {
                        let species_name = &n.name[..pos];
                        self.rec_tree.species_tree.nodes.iter()
                            .position(|sn| sn.name == species_name)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        Ok(PyGeneTree {
            rec_tree: RecTree::new(
                Arc::clone(&self.rec_tree.species_tree),
                sampled_tree,
                new_node_mapping,
                new_event_mapping,
            ),
        })
    }

    /// Sample the gene tree by keeping only genes from the specified species.
    ///
    /// Gene leaves are matched to species by the naming convention `speciesName_geneId`.
    ///
    /// # Arguments
    /// * `species_names` - List of species names to keep
    ///
    /// # Returns
    /// A new gene tree containing only genes from the specified species.
    ///
    /// # Example
    /// ```python
    /// species_tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)
    /// gene_tree = species_tree.simulate_dtl(0.5, 0.2, 0.3, seed=123)
    ///
    /// sampled_names = species_tree.sample_leaf_names(n=20, seed=42)
    /// pruned_gene_tree = gene_tree.sample_by_species_names(sampled_names)
    /// ```
    fn sample_by_species_names(&self, species_names: Vec<String>) -> PyResult<PyGeneTree> {
        let species_set: std::collections::HashSet<&str> = species_names.iter().map(|s| s.as_str()).collect();

        // Find gene leaves whose species is in the set
        let keep_indices: std::collections::HashSet<usize> = self.rec_tree.gene_tree.nodes.iter()
            .enumerate()
            .filter(|(_, n)| {
                if n.left_child.is_some() || n.right_child.is_some() {
                    return false;
                }
                if let Some(pos) = n.name.rfind('_') {
                    let species_name = &n.name[..pos];
                    species_set.contains(species_name)
                } else {
                    false
                }
            })
            .map(|(i, _)| i)
            .collect();

        if keep_indices.is_empty() {
            return Err(PyValueError::new_err("No genes found for the specified species"));
        }

        let (sampled_tree, old_to_new) = extract_induced_subtree(&self.rec_tree.gene_tree, &keep_indices)
            .ok_or_else(|| PyValueError::new_err("Failed to extract induced subtree"))?;

        // Build new→old mapping
        let mut new_to_old: Vec<Option<usize>> = vec![None; sampled_tree.nodes.len()];
        for (old_idx, new_idx_opt) in old_to_new.iter().enumerate() {
            if let Some(new_idx) = new_idx_opt {
                new_to_old[*new_idx] = Some(old_idx);
            }
        }

        let new_event_mapping: Vec<Event> = new_to_old.iter()
            .enumerate()
            .map(|(new_idx, old_idx_opt)| {
                if let Some(old_idx) = old_idx_opt {
                    self.rec_tree.event_mapping[*old_idx].clone()
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
                        self.rec_tree.species_tree.nodes.iter()
                            .position(|sn| sn.name == species_name)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        Ok(PyGeneTree {
            rec_tree: RecTree::new(
                Arc::clone(&self.rec_tree.species_tree),
                sampled_tree,
                new_node_mapping,
                new_event_mapping,
            ),
        })
    }

    /// Export the reconciled tree to RecPhyloXML format as a string.
    fn to_xml(&self) -> String {
        self.rec_tree.to_xml()
    }

    /// Save the reconciled tree to a RecPhyloXML file.
    ///
    /// # Arguments
    /// * `filepath` - Path to save the XML file
    fn save_xml(&self, filepath: &str) -> PyResult<()> {
        let xml = self.to_xml();
        fs::write(filepath, xml)
            .map_err(|e| PyValueError::new_err(format!("Failed to write XML file: {}", e)))?;
        Ok(())
    }

    /// Generate an SVG visualization of the reconciled tree using thirdkind.
    ///
    /// Requires thirdkind to be installed (`cargo install thirdkind`).
    ///
    /// # Arguments
    /// * `filepath` - Optional path to save the SVG file. If None, returns SVG as string.
    /// * `open_browser` - If True, opens the SVG in a web browser (default: False)
    /// * `gene_colors` - Comma-separated colors for gene trees (e.g. "red,blue,#4A38C4")
    /// * `species_color` - Color for the species tree (e.g. "grey", "#CCCCCC")
    /// * `internal_gene_names` - If True, display internal gene node names (default: False)
    /// * `internal_species_names` - If True, display internal species node names (default: False)
    /// * `gene_fontsize` - Font size for gene tree labels
    /// * `species_fontsize` - Font size for species tree labels
    /// * `gene_thickness` - Thickness of gene tree lines
    /// * `species_thickness` - Thickness of species tree lines
    /// * `symbol_size` - Size of event symbols (circles, squares, etc.)
    /// * `background` - Background color (e.g. "white", "black")
    /// * `landscape` - If True, display in landscape orientation (default: False)
    /// * `fill_species` - If True, fill the species tree (default: False)
    /// * `sampled_species_names` - Optional list of sampled species leaf names. When provided,
    ///   species node labels are colored by their NodeMark: Keep, HasDescendant, or Discard.
    ///   Internal species names are automatically shown.
    /// * `keep_color` - Color for Keep nodes (default: "green")
    /// * `has_descendant_color` - Color for HasDescendant nodes (default: "orange")
    /// * `discard_color` - Color for Discard nodes (default: "grey")
    ///
    /// # Returns
    /// The SVG content as a string.
    #[pyo3(signature = (
        filepath=None, open_browser=false,
        gene_colors=None, species_color=None,
        internal_gene_names=false, internal_species_names=false,
        gene_fontsize=None, species_fontsize=None,
        gene_thickness=None, species_thickness=None,
        symbol_size=None, background=None,
        landscape=false, fill_species=false,
        gene_only=false,
        sampled_species_names=None,
        keep_color="green", has_descendant_color="orange", discard_color="grey",
        color_transfers_by=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn to_svg(
        &self,
        filepath: Option<&str>,
        open_browser: bool,
        gene_colors: Option<&str>,
        species_color: Option<&str>,
        internal_gene_names: bool,
        internal_species_names: bool,
        gene_fontsize: Option<f64>,
        species_fontsize: Option<f64>,
        gene_thickness: Option<f64>,
        species_thickness: Option<f64>,
        symbol_size: Option<f64>,
        background: Option<&str>,
        landscape: bool,
        fill_species: bool,
        gene_only: bool,
        sampled_species_names: Option<Vec<String>>,
        keep_color: &str,
        has_descendant_color: &str,
        discard_color: &str,
        color_transfers_by: Option<&str>,
    ) -> PyResult<String> {
        let marking_nodes = sampled_species_names.is_some();

        // Create temp files
        let temp_dir = std::env::temp_dir();
        let input_path = if gene_only {
            temp_dir.join("rustree_gene_temp.nwk")
        } else {
            temp_dir.join("rustree_temp.recphyloxml")
        };
        let svg_path = temp_dir.join("rustree_temp.svg");
        let conf_path = temp_dir.join("rustree_thirdkind.conf");

        // Write input file: Newick for gene_only, RecPhyloXML otherwise
        if gene_only {
            let nwk = self.to_newick()?;
            fs::write(&input_path, &nwk)
                .map_err(|e| PyValueError::new_err(format!("Failed to write temp Newick: {}", e)))?;
        } else {
            let xml = self.to_xml();
            fs::write(&input_path, &xml)
                .map_err(|e| PyValueError::new_err(format!("Failed to write temp XML: {}", e)))?;
        }

        // Build thirdkind config file for styling options
        let mut conf_lines = Vec::new();
        if !gene_only {
            if let Some(c) = species_color { conf_lines.push(format!("species_color:{}", c)); }
        }
        if let Some(c) = gene_colors {
            // single_gene_color applies when one color is given
            conf_lines.push(format!("single_gene_color:{}", c.split(',').next().unwrap_or(c)));
        }
        if let Some(s) = species_fontsize { conf_lines.push(format!("species_police_size:{}", s)); }
        if let Some(s) = gene_fontsize { conf_lines.push(format!("gene_police_size:{}", s)); }

        // Write transfer color config entries based on NodeMark
        if let Some(ref names) = sampled_species_names {
            if let Some(color_by) = color_transfers_by {
                let species_tree = &self.rec_tree.species_tree;
                let keep_indices = find_leaf_indices_by_names(species_tree, names);
                let mut marks = vec![NodeMark::Discard; species_tree.nodes.len()];
                mark_nodes_postorder(species_tree, species_tree.root, &keep_indices, &mut marks);

                let config_key = match color_by {
                    "donor" => "transfer_donor_color",
                    _ => "transfer_color",  // "recipient" or any other value defaults to recipient
                };
                for (idx, node) in species_tree.nodes.iter().enumerate() {
                    let color = match marks[idx] {
                        NodeMark::Keep => keep_color,
                        NodeMark::HasDescendant => has_descendant_color,
                        NodeMark::Discard => discard_color,
                    };
                    conf_lines.push(format!("{}:{}:{}", config_key, node.name, color));
                }
            }
        }

        fs::write(&conf_path, conf_lines.join("\n"))
            .map_err(|e| PyValueError::new_err(format!("Failed to write config file: {}", e)))?;

        // Call thirdkind with config file + CLI-only flags
        let mut cmd = Command::new("thirdkind");
        cmd.arg("-f").arg(&input_path)
           .arg("-o").arg(&svg_path)
           .arg("-c").arg(&conf_path);

        if open_browser { cmd.arg("-b"); }
        if internal_gene_names { cmd.arg("-i"); }
        if !gene_only {
            if internal_species_names || marking_nodes { cmd.arg("-I"); }
            if fill_species { cmd.arg("-P"); }
        }
        if landscape { cmd.arg("-L"); }
        // -C supports comma-separated multi-color (config only handles single)
        if let Some(c) = gene_colors { cmd.arg("-C").arg(c); }
        if let Some(t) = gene_thickness { cmd.arg("-z").arg(t.to_string()); }
        if !gene_only {
            if let Some(t) = species_thickness { cmd.arg("-Z").arg(t.to_string()); }
        }
        if let Some(s) = symbol_size { cmd.arg("-k").arg(s.to_string()); }
        if let Some(c) = background { cmd.arg("-Q").arg(c); }

        let output = cmd.output()
            .map_err(|e| PyValueError::new_err(format!(
                "Failed to run thirdkind. Is it installed? (`cargo install thirdkind`)\nError: {}", e
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PyValueError::new_err(format!("thirdkind failed: {}", stderr)));
        }

        // Read SVG output
        let mut svg = fs::read_to_string(&svg_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to read SVG output: {}", e)))?;

        // Post-process SVG to color species node labels by NodeMark
        // Thirdkind generates species text as:
        //   <text class="species" ...>\nNodeName\n</text>
        // We match each species <text> element by the node name on the next line
        // and add an inline style to override the CSS class fill color.
        if let Some(ref names) = sampled_species_names {
            let species_tree = &self.rec_tree.species_tree;
            let keep_indices = find_leaf_indices_by_names(species_tree, names);
            let mut marks = vec![NodeMark::Discard; species_tree.nodes.len()];
            mark_nodes_postorder(species_tree, species_tree.root, &keep_indices, &mut marks);

            for (idx, node) in species_tree.nodes.iter().enumerate() {
                let color = match marks[idx] {
                    NodeMark::Keep => keep_color,
                    NodeMark::HasDescendant => has_descendant_color,
                    NodeMark::Discard => discard_color,
                };
                // Match the multi-line pattern: class="species" ...>\nNodeName\n</text>
                let new_attr = format!("class=\"species\" style=\"fill: {}\"", color);
                let search = "class=\"species\"";
                let name_pattern = format!("\n{}\n</text>", node.name);
                // Find position of this specific text element
                if let Some(pos) = svg.find(&name_pattern) {
                    // Search backwards from the name to find the class="species" in this element
                    if let Some(class_start) = svg[..pos].rfind(&search) {
                        let class_end = class_start + search.len();
                        svg = format!(
                            "{}{}{}",
                            &svg[..class_start],
                            &new_attr,
                            &svg[class_end..]
                        );
                    }
                }
            }

            // Also color gene node labels by the NodeMark of their mapped species.
            // Gene text elements have class="gene" (or "gene_0", "node_X").
            // We match by gene name text content and override fill.
            let species_tree = &self.rec_tree.species_tree;
            let mut species_color_map: HashMap<String, &str> = HashMap::new();
            for (idx, node) in species_tree.nodes.iter().enumerate() {
                let c = match marks[idx] {
                    NodeMark::Keep => keep_color,
                    NodeMark::HasDescendant => has_descendant_color,
                    NodeMark::Discard => discard_color,
                };
                species_color_map.insert(node.name.clone(), c);
            }

            for (gene_idx, gene_node) in self.rec_tree.gene_tree.nodes.iter().enumerate() {
                if gene_node.name.is_empty() {
                    continue;
                }
                // Get the species this gene node maps to
                let species_name = match self.rec_tree.node_mapping[gene_idx] {
                    Some(sp_idx) => &species_tree.nodes[sp_idx].name,
                    None => continue,
                };
                let color = match species_color_map.get(species_name) {
                    Some(c) => *c,
                    None => continue,
                };
                // Find the gene text element by name, looking for class="gene..." pattern
                let name_pattern = format!("\n{}\n</text>", gene_node.name);
                if let Some(pos) = svg.find(&name_pattern) {
                    // Search backwards for the class attribute in this <text> element
                    if let Some(tag_start) = svg[..pos].rfind("<text") {
                        // Find the closing > of the opening tag
                        let tag_content = &svg[tag_start..pos];
                        // Only modify gene text elements (class contains "gene" or "node_")
                        if tag_content.contains("class=\"gene") || tag_content.contains("class=\"node_") {
                            if let Some(class_rel) = tag_content.find("class=\"") {
                                let class_abs = tag_start + class_rel;
                                let class_val_start = class_abs + 7; // after class="
                                if let Some(class_val_end) = svg[class_val_start..].find('"') {
                                    let old_class = &svg[class_val_start..class_val_start + class_val_end];
                                    let new_attr = format!("class=\"{}\" style=\"fill: {}\"", old_class, color);
                                    let replace_start = class_abs;
                                    let replace_end = class_val_start + class_val_end + 1; // include closing "
                                    svg = format!(
                                        "{}{}{}",
                                        &svg[..replace_start],
                                        &new_attr,
                                        &svg[replace_end..]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Save to user-specified path if provided
        if let Some(path) = filepath {
            fs::write(path, &svg)
                .map_err(|e| PyValueError::new_err(format!("Failed to write SVG file: {}", e)))?;
        }

        // Cleanup temp files
        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&svg_path);
        let _ = fs::remove_file(&conf_path);

        Ok(svg)
    }

    /// Display the reconciled tree visualization in a Jupyter notebook.
    ///
    /// Requires thirdkind to be installed and IPython/Jupyter environment.
    /// Accepts the same styling options as `to_svg()`.
    #[pyo3(signature = (
        gene_colors=None, species_color=None,
        internal_gene_names=false, internal_species_names=false,
        gene_fontsize=None, species_fontsize=None,
        gene_thickness=None, species_thickness=None,
        symbol_size=None, background=None,
        landscape=false, fill_species=false,
        gene_only=false,
        sampled_species_names=None,
        keep_color="green", has_descendant_color="orange", discard_color="grey",
        color_transfers_by=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn display(
        &self,
        py: Python,
        gene_colors: Option<&str>,
        species_color: Option<&str>,
        internal_gene_names: bool,
        internal_species_names: bool,
        gene_fontsize: Option<f64>,
        species_fontsize: Option<f64>,
        gene_thickness: Option<f64>,
        species_thickness: Option<f64>,
        symbol_size: Option<f64>,
        background: Option<&str>,
        landscape: bool,
        fill_species: bool,
        gene_only: bool,
        sampled_species_names: Option<Vec<String>>,
        keep_color: &str,
        has_descendant_color: &str,
        discard_color: &str,
        color_transfers_by: Option<&str>,
    ) -> PyResult<PyObject> {
        let svg = self.to_svg(
            None, false,
            gene_colors, species_color,
            internal_gene_names, internal_species_names,
            gene_fontsize, species_fontsize,
            gene_thickness, species_thickness,
            symbol_size, background,
            landscape, fill_species,
            gene_only,
            sampled_species_names,
            keep_color, has_descendant_color, discard_color,
            color_transfers_by,
        )?;

        let ipython_display = py.import("IPython.display")?;
        let svg_class = ipython_display.getattr("SVG")?;
        let display_obj = svg_class.call1((svg,))?;

        Ok(display_obj.into())
    }

    /// Export gene tree data as a pandas DataFrame.
    ///
    /// Returns a DataFrame with columns: node_id, name, parent, left_child,
    /// left_child_name, right_child, right_child_name, length, depth,
    /// species_node, species_node_left, species_node_right, event
    ///
    /// # Arguments
    /// * `filepath` - Optional path to save the CSV file
    ///
    /// # Returns
    /// A pandas DataFrame with the gene tree data.
    #[pyo3(signature = (filepath=None))]
    fn to_csv(&self, py: Python, filepath: Option<&str>) -> PyResult<PyObject> {
        let cols = self.rec_tree.to_columns();

        if let Some(path) = filepath {
            cols.save_csv(path)
                .map_err(|e| PyValueError::new_err(format!("Failed to write CSV: {}", e)))?;
        }

        columns_to_dataframe(py, &cols)
    }

    /// Return a DataFrame of transfer events (subset of `to_csv()` where event == "Transfer").
    fn transfers(&self, py: Python) -> PyResult<PyObject> {
        let cols = self.rec_tree.to_columns().filter_by_event("Transfer");
        columns_to_dataframe(py, &cols)
    }

    /// Return a DataFrame of duplication events (subset of `to_csv()` where event == "Duplication").
    fn duplications(&self, py: Python) -> PyResult<PyObject> {
        let cols = self.rec_tree.to_columns().filter_by_event("Duplication");
        columns_to_dataframe(py, &cols)
    }

    /// Return a DataFrame of loss events (subset of `to_csv()` where event == "Loss").
    fn losses(&self, py: Python) -> PyResult<PyObject> {
        let cols = self.rec_tree.to_columns().filter_by_event("Loss");
        columns_to_dataframe(py, &cols)
    }

    /// Return a DataFrame of speciation events (subset of `to_csv()` where event == "Speciation").
    fn speciations(&self, py: Python) -> PyResult<PyObject> {
        let cols = self.rec_tree.to_columns().filter_by_event("Speciation");
        columns_to_dataframe(py, &cols)
    }

    /// Return a DataFrame of leaf events (subset of `to_csv()` where event == "Leaf").
    fn leaves(&self, py: Python) -> PyResult<PyObject> {
        let cols = self.rec_tree.to_columns().filter_by_event("Leaf");
        columns_to_dataframe(py, &cols)
    }

    /// Compute all pairwise distances between nodes in the gene tree.
    ///
    /// # Arguments
    /// * `distance_type` - Type of distance: "topological" (number of edges) or "metric" (sum of branch lengths)
    /// * `leaves_only` - If true, only compute distances between leaf nodes (default true)
    ///
    /// # Returns
    /// A pandas DataFrame with columns: node1, node2, distance
    ///
    /// # Example
    /// ```python
    /// import rustree
    /// species_tree = rustree.simulate_species_tree(5, 1.0, 0.5, seed=42)
    /// gene_tree = species_tree.simulate_dtl(0.5, 0.2, 0.3, seed=123)
    /// df = gene_tree.pairwise_distances("metric", leaves_only=True)
    /// print(df)
    /// ```
    #[pyo3(signature = (distance_type, leaves_only=true))]
    fn pairwise_distances(&self, py: Python, distance_type: &str, leaves_only: bool) -> PyResult<PyObject> {
        use crate::metric_functions::DistanceType;

        let dist_type = match distance_type.to_lowercase().as_str() {
            "topological" | "topo" => DistanceType::Topological,
            "metric" | "branch" | "patristic" => DistanceType::Metric,
            _ => return Err(PyValueError::new_err(format!(
                "Invalid distance_type '{}'. Use 'topological' or 'metric'.",
                distance_type
            ))),
        };

        let distances = self.rec_tree.gene_tree.pairwise_distances(dist_type, leaves_only)
            .map_err(|e| PyValueError::new_err(format!("Failed to compute pairwise distances: {}", e)))?;

        let node1: Vec<&str> = distances.iter().map(|d| d.node1).collect();
        let node2: Vec<&str> = distances.iter().map(|d| d.node2).collect();
        let dist: Vec<f64> = distances.iter().map(|d| d.distance).collect();

        // Create pandas DataFrame
        let pandas = py.import("pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("node1", node1)?;
        dict.set_item("node2", node2)?;
        dict.set_item("distance", dist)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Save pairwise distances between nodes in the gene tree to a CSV file.
    ///
    /// Computes all pairwise distances and writes them to a CSV file with the format:
    /// node1,node2,distance
    ///
    /// # Arguments
    /// * `filepath` - Path to save the CSV file
    /// * `distance_type` - Type of distance: "topological" (number of edges) or "metric" (sum of branch lengths)
    /// * `leaves_only` - If true, only compute distances between leaf nodes (default: true)
    ///
    /// # Example
    /// ```python
    /// import rustree
    ///
    /// # Simulate a species tree and gene tree
    /// species_tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.5, seed=42)
    /// gene_tree = species_tree.simulate_dtl(0.5, 0.2, 0.3, seed=123)
    ///
    /// # Save topological distances between all leaves
    /// gene_tree.save_pairwise_distances_csv("gene_distances.csv", "topological", leaves_only=True)
    ///
    /// # Save metric distances between all nodes
    /// gene_tree.save_pairwise_distances_csv("all_gene_distances.csv", "metric", leaves_only=False)
    /// ```
    #[pyo3(signature = (filepath, distance_type, leaves_only=true))]
    fn save_pairwise_distances_csv(&self, filepath: &str, distance_type: &str, leaves_only: bool) -> PyResult<()> {
        use crate::metric_functions::{DistanceType, PairwiseDistance};

        // Parse distance type
        let dist_type = match distance_type.to_lowercase().as_str() {
            "topological" | "topo" => DistanceType::Topological,
            "metric" | "branch" | "patristic" => DistanceType::Metric,
            _ => return Err(PyValueError::new_err(format!(
                "Invalid distance_type '{}'. Use 'topological' or 'metric'.",
                distance_type
            ))),
        };

        // Compute pairwise distances
        let distances = self.rec_tree.gene_tree.pairwise_distances(dist_type, leaves_only)
            .map_err(|e| PyValueError::new_err(format!("Failed to compute pairwise distances: {}", e)))?;

        // Write to CSV
        let mut csv = String::from(PairwiseDistance::csv_header());
        csv.push('\n');
        for d in &distances {
            csv.push_str(&d.to_csv_row());
            csv.push('\n');
        }

        fs::write(filepath, csv)
            .map_err(|e| PyValueError::new_err(format!("Failed to write CSV file: {}", e)))?;

        Ok(())
    }

    /// Compute induced transfers by projecting transfers onto a sampled species tree.
    ///
    /// Given leaf names to keep, this internally builds the sampled tree and projects
    /// each transfer's donor and recipient species onto the nearest branches.
    ///
    /// # Arguments
    /// * `sampled_leaf_names` - Names of species leaves to keep
    ///
    /// # Returns
    /// A pandas DataFrame with columns: time, gene_id, from_species_complete,
    /// to_species_complete, from_species_sampled, to_species_sampled.
    ///
    /// # Example
    /// ```python
    /// species_tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)
    /// gene_tree = species_tree.simulate_dtl(0.5, 0.2, 0.3, seed=123)
    ///
    /// sampled_leaf_names = species_tree.sample_leaf_names(n=20, seed=42)
    /// df = gene_tree.compute_induced_transfers(sampled_leaf_names)
    /// print(df)
    /// ```
    fn compute_induced_transfers(&self, py: Python, sampled_leaf_names: Vec<String>) -> PyResult<PyObject> {
        let events = self.rec_tree.dtl_events.as_ref()
            .ok_or_else(|| PyValueError::new_err(
                "DTL events not available. Gene tree must be simulated (not parsed from file)."
            ))?;

        use crate::induced_transfers::induced_transfers;
        let induced = induced_transfers(
            &self.rec_tree.species_tree,
            &sampled_leaf_names,
            events,
        );

        let time: Vec<f64> = induced.iter().map(|t| t.time).collect();
        let gene_id: Vec<usize> = induced.iter().map(|t| t.gene_id).collect();
        let from_complete: Vec<usize> = induced.iter().map(|t| t.from_species_complete).collect();
        let to_complete: Vec<usize> = induced.iter().map(|t| t.to_species_complete).collect();
        let from_sampled: Vec<Option<usize>> = induced.iter().map(|t| t.from_species_sampled).collect();
        let to_sampled: Vec<Option<usize>> = induced.iter().map(|t| t.to_species_sampled).collect();

        let pandas = py.import("pandas")?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("time", time)?;
        dict.set_item("gene_id", gene_id)?;
        dict.set_item("from_species_complete", from_complete)?;
        dict.set_item("to_species_complete", to_complete)?;
        dict.set_item("from_species_sampled", from_sampled)?;
        dict.set_item("to_species_sampled", to_sampled)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Compare this reconciliation (truth) against another (inferred).
    ///
    /// Matches nodes by their clade (set of descendant extant leaf names).
    /// Both trees must have the same extant leaf set.
    ///
    /// # Arguments
    /// * `other` - The inferred reconciliation to compare against
    ///
    /// # Returns
    /// A PyReconciliationComparison with accuracy metrics and per-node details.
    fn compare_reconciliation(&self, other: &PyGeneTree) -> PyResult<PyReconciliationComparison> {
        let result = crate::comparison::compare_reconciliations(&self.rec_tree, &other.rec_tree)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(PyReconciliationComparison {
            inner: result,
            trees: Some((self.rec_tree.clone(), other.rec_tree.clone())),
        })
    }

    /// Compare this reconciliation (truth) against multiple inferred samples.
    ///
    /// Computes per-sample metrics and a consensus comparison (majority vote per clade).
    ///
    /// # Arguments
    /// * `samples` - List of inferred reconciliations (e.g., from PyAleRaxResult.gene_trees)
    ///
    /// # Returns
    /// A PyMultiSampleComparison with per-sample and consensus accuracy.
    fn compare_reconciliation_multi(&self, samples: Vec<PyGeneTree>) -> PyResult<PyMultiSampleComparison> {
        let sample_recs: Vec<_> = samples.iter().map(|s| s.rec_tree.clone()).collect();
        let result = crate::comparison::compare_reconciliations_multi(&self.rec_tree, &sample_recs)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(PyMultiSampleComparison { inner: result })
    }

    /// Compute Robinson-Foulds distance to another gene tree.
    ///
    /// Loss nodes are automatically stripped before comparison, so this works
    /// directly between a truth tree and an ALERax reconciliation.
    ///
    /// # Arguments
    /// * `other` - The other gene tree
    /// * `rooted` - If True, compare rooted clades. If False (default), compare
    ///              unrooted bipartitions (ignores root placement).
    ///
    /// # Returns
    /// The RF distance (number of differing splits).
    #[pyo3(signature = (other, rooted=false))]
    fn rf_distance(&self, other: &PyGeneTree, rooted: bool) -> PyResult<usize> {
        // Extract extant-only subtrees (strips loss nodes from ALERax trees)
        let tree1 = extract_extant_gene_tree(&self.rec_tree)
            .map_err(|e| PyValueError::new_err(e))?;
        let tree2 = extract_extant_gene_tree(&other.rec_tree)
            .map_err(|e| PyValueError::new_err(e))?;

        if rooted {
            Ok(crate::robinson_foulds::unrooted_robinson_foulds(&tree1, &tree2))
        } else {
            Ok(crate::robinson_foulds::true_unrooted_robinson_foulds(&tree1, &tree2))
        }
    }

}

/// An induced transfer: a transfer event projected onto a sampled species tree.
///
/// When a gene tree is simulated on a complete species tree and then analyzed
/// with respect to a sampled (pruned) species tree, each transfer's donor and
/// recipient species may not be present in the sampled tree. This struct records
/// both the original transfer (in the complete tree) and its projection onto the
/// nearest branches in the sampled tree.
#[pyclass]
#[derive(Clone)]
pub struct PyInducedTransfer {
    /// Time of the transfer event
    #[pyo3(get)]
    pub time: f64,

    /// Gene tree node index
    #[pyo3(get)]
    pub gene_id: usize,

    /// Donor species index in the complete tree
    #[pyo3(get)]
    pub from_species_complete: usize,

    /// Recipient species index in the complete tree
    #[pyo3(get)]
    pub to_species_complete: usize,

    /// Induced donor: index in the sampled tree (None if projection fails)
    #[pyo3(get)]
    pub from_species_sampled: Option<usize>,

    /// Induced recipient: index in the sampled tree (None if projection fails)
    #[pyo3(get)]
    pub to_species_sampled: Option<usize>,
}

#[pymethods]
impl PyInducedTransfer {
    fn __repr__(&self) -> String {
        let from_sampled = self.from_species_sampled
            .map(|i| i.to_string())
            .unwrap_or_else(|| "None".to_string());
        let to_sampled = self.to_species_sampled
            .map(|i| i.to_string())
            .unwrap_or_else(|| "None".to_string());
        format!(
            "InducedTransfer(time={:.3}, gene_id={}, from_complete={}, to_complete={}, from_sampled={}, to_sampled={})",
            self.time, self.gene_id, self.from_species_complete, self.to_species_complete,
            from_sampled, to_sampled
        )
    }
}

/// Simulate a birth-death species tree.
///
/// # Arguments
/// * `n` - Number of extant species (must be > 0)
/// * `lambda_` - Speciation/birth rate (must be > 0)
/// * `mu` - Extinction/death rate (must be >= 0 and < lambda)
/// * `seed` - Random seed for reproducibility (optional)
///
/// # Returns
/// A PySpeciesTree with n extant species simulated under the birth-death process.
#[pyfunction]
#[pyo3(signature = (n, lambda_, mu, seed=None))]
fn simulate_species_tree(n: usize, lambda_: f64, mu: f64, seed: Option<u64>) -> PyResult<PySpeciesTree> {
    if n == 0 {
        return Err(PyValueError::new_err("Number of species must be positive"));
    }
    if lambda_ <= 0.0 {
        return Err(PyValueError::new_err("Speciation rate must be positive"));
    }
    if mu < 0.0 {
        return Err(PyValueError::new_err("Extinction rate must be non-negative"));
    }
    if lambda_ <= mu {
        return Err(PyValueError::new_err("Speciation rate must be strictly greater than extinction rate"));
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let (mut tree, _events) = simulate_bd_tree_bwd(n, lambda_, mu, &mut rng);
    tree.assign_depths();

    Ok(PySpeciesTree { tree: Arc::new(tree) })
}

/// Parse a Newick string or file into a species tree.
///
/// # Arguments
/// * `newick_str` - A Newick formatted string or path to a Newick file
///
/// # Returns
/// A PySpeciesTree parsed from the Newick string.
///
/// # Example
/// ```python
/// # From Newick string
/// tree = rustree.parse_species_tree("(A:1.0,B:1.0):0.5;")
///
/// # From file
/// tree = rustree.parse_species_tree("species.nwk")
/// ```
#[pyfunction]
fn parse_species_tree(newick_str: &str) -> PyResult<PySpeciesTree> {
    use crate::newick::newick::parse_newick;
    use std::path::Path;

    // Check if input is a file path
    let newick_content = if Path::new(newick_str).exists() {
        fs::read_to_string(newick_str)
            .map_err(|e| PyValueError::new_err(format!("Failed to read file '{}': {}", newick_str, e)))?
    } else {
        newick_str.to_string()
    };

    let mut nodes = parse_newick(&newick_content)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse Newick: {}", e)))?;

    let mut root = nodes.pop()
        .ok_or_else(|| PyValueError::new_err("No tree found in Newick string"))?;

    root.assign_depths(0.0);
    let tree = root.to_flat_tree();

    Ok(PySpeciesTree { tree: Arc::new(tree) })
}

/// Parse a RecPhyloXML file into a gene tree with reconciliation information.
///
/// # Arguments
/// * `filepath` - Path to the RecPhyloXML file (e.g., from ALERax)
///
/// # Returns
/// A PyGeneTree parsed from the RecPhyloXML file, containing both the species tree
/// and reconciled gene tree with event mappings.
#[pyfunction]
fn parse_recphyloxml(filepath: &str) -> PyResult<PyGeneTree> {
    let rec_tree = RecTree::from_xml_file(filepath)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse RecPhyloXML: {}", e)))?;

    Ok(PyGeneTree { rec_tree })
}

/// Event counts from reconciliation analysis
#[pyclass]
#[derive(Clone)]
pub struct PyEventCounts {
    /// Number of speciation events
    #[pyo3(get)]
    pub speciations: f64,

    /// Number of speciation loss events
    #[pyo3(get)]
    pub speciation_losses: f64,

    /// Number of duplication events
    #[pyo3(get)]
    pub duplications: f64,

    /// Number of duplication loss events
    #[pyo3(get)]
    pub duplication_losses: f64,

    /// Number of transfer events
    #[pyo3(get)]
    pub transfers: f64,

    /// Number of transfer loss events
    #[pyo3(get)]
    pub transfer_losses: f64,

    /// Number of loss events
    #[pyo3(get)]
    pub losses: f64,

    /// Number of leaf nodes
    #[pyo3(get)]
    pub leaves: f64,
}

#[pymethods]
impl PyEventCounts {
    fn __repr__(&self) -> String {
        format!(
            "EventCounts(S={:.1}, D={:.1}, T={:.1}, L={:.1})",
            self.speciations,
            self.duplications,
            self.transfers,
            self.losses
        )
    }
}

/// Summary statistics for reconciliation analysis
#[pyclass]
#[derive(Clone)]
pub struct PyReconciliationStatistics {
    /// Mean event counts across all samples
    #[pyo3(get)]
    pub mean_event_counts: PyEventCounts,

    /// Mean transfers between species pairs
    /// Returns a dict of dicts: {source_species: {dest_species: mean_count}}
    #[pyo3(get)]
    pub mean_transfers: HashMap<String, HashMap<String, f64>>,

    /// Mean events per species node
    /// Returns a dict: {species_name: EventCounts}
    #[pyo3(get)]
    pub events_per_species: HashMap<String, PyEventCounts>,
}

#[pymethods]
impl PyReconciliationStatistics {
    fn __repr__(&self) -> String {
        let total_transfers: f64 = self.mean_transfers.values()
            .flat_map(|dests| dests.values())
            .sum();
        format!(
            "ReconciliationStatistics(mean_events={}, mean_transfers={:.1}, species_count={})",
            self.mean_event_counts.__repr__(),
            total_transfers,
            self.events_per_species.len()
        )
    }

    /// Returns a pandas DataFrame of mean transfers between species pairs.
    ///
    /// Columns: source, destination, mean_count
    fn transfers_df(&self, py: Python) -> PyResult<PyObject> {
        let pandas = py.import("pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        let mut sources: Vec<String> = Vec::new();
        let mut destinations: Vec<String> = Vec::new();
        let mut counts: Vec<f64> = Vec::new();

        for (source, dests) in &self.mean_transfers {
            for (dest, count) in dests {
                sources.push(source.clone());
                destinations.push(dest.clone());
                counts.push(*count);
            }
        }

        dict.set_item("source", sources)?;
        dict.set_item("destination", destinations)?;
        dict.set_item("mean_count", counts)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Returns a pandas DataFrame of mean events per species.
    ///
    /// Columns: species, speciations, speciation_losses, duplications,
    ///          duplication_losses, transfers, transfer_losses, losses, leaves
    fn events_df(&self, py: Python) -> PyResult<PyObject> {
        let pandas = py.import("pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        let mut species_names: Vec<String> = Vec::new();
        let mut speciations: Vec<f64> = Vec::new();
        let mut speciation_losses: Vec<f64> = Vec::new();
        let mut duplications: Vec<f64> = Vec::new();
        let mut duplication_losses: Vec<f64> = Vec::new();
        let mut transfers: Vec<f64> = Vec::new();
        let mut transfer_losses: Vec<f64> = Vec::new();
        let mut losses: Vec<f64> = Vec::new();
        let mut leaves: Vec<f64> = Vec::new();

        // Sort by species name for consistent output
        let mut sorted_species: Vec<(&String, &PyEventCounts)> =
            self.events_per_species.iter().collect();
        sorted_species.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (species, counts) in sorted_species {
            species_names.push(species.clone());
            speciations.push(counts.speciations);
            speciation_losses.push(counts.speciation_losses);
            duplications.push(counts.duplications);
            duplication_losses.push(counts.duplication_losses);
            transfers.push(counts.transfers);
            transfer_losses.push(counts.transfer_losses);
            losses.push(counts.losses);
            leaves.push(counts.leaves);
        }

        dict.set_item("species", species_names)?;
        dict.set_item("speciations", speciations)?;
        dict.set_item("speciation_losses", speciation_losses)?;
        dict.set_item("duplications", duplications)?;
        dict.set_item("duplication_losses", duplication_losses)?;
        dict.set_item("transfers", transfers)?;
        dict.set_item("transfer_losses", transfer_losses)?;
        dict.set_item("losses", losses)?;
        dict.set_item("leaves", leaves)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }
}

// ============================================================================
// Reconciliation Comparison Python Bindings
// ============================================================================

fn event_to_string(event: &crate::node::rectree::Event) -> String {
    match event {
        Event::Speciation => "S".to_string(),
        Event::Duplication => "D".to_string(),
        Event::Transfer => "T".to_string(),
        Event::Loss => "L".to_string(),
        Event::Leaf => "Leaf".to_string(),
    }
}

/// Result of comparing two reconciliations (truth vs inferred).
///
/// Provides accuracy metrics, per-node details, and confusion matrix.
#[pyclass]
#[derive(Clone)]
pub struct PyReconciliationComparison {
    inner: crate::comparison::ReconciliationComparison,
    /// Truth and inferred RecTrees, stored for display(). None for sub-comparisons
    /// extracted from MultiSampleComparison (consensus, per-sample).
    trees: Option<(RecTree, RecTree)>,
}

impl PyReconciliationComparison {
    /// Generate SVG string from both reconciliations using thirdkind.
    fn generate_svg(
        &self,
        truth_color: &str,
        inferred_color: &str,
        internal_gene_names: bool,
        internal_species_names: bool,
        landscape: bool,
        fill_species: bool,
        species_color: &str,
        species_fontsize: f64,
        background: &str,
    ) -> PyResult<String> {
        let (truth, inferred) = self.trees.as_ref()
            .ok_or_else(|| PyValueError::new_err(
                "display()/to_svg() requires trees stored from compare_reconciliation(). \
                 Not available for consensus/per-sample sub-comparisons."
            ))?;

        // Generate combined RecPhyloXML with both gene trees
        let xml = crate::io::rectree_xml::multi_to_xml(&[truth, inferred]);

        // Write to temp file
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("rustree_comparison.recphyloxml");
        let svg_path = temp_dir.join("rustree_comparison.svg");
        let conf_path = temp_dir.join("rustree_comparison.conf");

        fs::write(&input_path, &xml)
            .map_err(|e| PyValueError::new_err(format!("Failed to write temp XML: {}", e)))?;

        // Write config file for styling
        let conf_lines = vec![
            format!("species_color:{}", species_color),
            format!("species_police_size:{}", species_fontsize),
        ];
        fs::write(&conf_path, conf_lines.join("\n"))
            .map_err(|e| PyValueError::new_err(format!("Failed to write config: {}", e)))?;

        // Call thirdkind with two gene tree colors
        let colors = format!("{},{}", truth_color, inferred_color);
        let mut cmd = Command::new("thirdkind");
        cmd.arg("-f").arg(&input_path)
           .arg("-o").arg(&svg_path)
           .arg("-c").arg(&conf_path)
           .arg("-C").arg(&colors)
           .arg("-Q").arg(background);

        if internal_gene_names { cmd.arg("-i"); }
        if internal_species_names { cmd.arg("-I"); }
        if landscape { cmd.arg("-L"); }
        if fill_species { cmd.arg("-P"); }

        let output = cmd.output()
            .map_err(|e| PyValueError::new_err(format!(
                "Failed to run thirdkind. Is it installed? (`cargo install thirdkind`)\nError: {}", e
            )))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PyValueError::new_err(format!("thirdkind failed: {}", stderr)));
        }

        let mut svg = fs::read_to_string(&svg_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to read SVG output: {}", e)))?;

        // Post-process SVG: override species text style (thirdkind CSS defaults to orange/small)
        // Replace the CSS rule for .species text elements
        if let Some(pos) = svg.find(".species") {
            if let Some(brace_offset) = svg[pos..].find('{') {
                if let Some(end_offset) = svg[pos + brace_offset..].find('}') {
                    let rule_end = pos + brace_offset + end_offset + 1;
                    svg = format!(
                        "{}.species {{ font-size: {}px; fill: black; }}{}",
                        &svg[..pos],
                        species_fontsize,
                        &svg[rule_end..],
                    );
                }
            }
        }

        // Inject a legend before the closing </svg> tag
        if let Some(close_pos) = svg.rfind("</svg>") {
            // Parse the SVG viewBox/dimensions to position the legend at the top-right
            let (legend_x, legend_y) = if let Some(vb_start) = svg.find("viewBox=\"") {
                let vb = &svg[vb_start + 9..];
                if let Some(vb_end) = vb.find('"') {
                    let parts: Vec<f64> = vb[..vb_end]
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if parts.len() == 4 {
                        (parts[0] + parts[2] - 220.0, parts[1] + 20.0)
                    } else {
                        (20.0, 20.0)
                    }
                } else {
                    (20.0, 20.0)
                }
            } else {
                (20.0, 20.0)
            };

            let legend = format!(
                r##"<g id="legend" transform="translate({},{})">
  <rect x="0" y="0" width="200" height="64" rx="6" fill="white" stroke="#ccc" stroke-width="1" opacity="0.92"/>
  <line x1="12" y1="22" x2="36" y2="22" stroke="{}" stroke-width="3"/>
  <text x="44" y="26" font-family="sans-serif" font-size="14" fill="#333">Truth</text>
  <line x1="12" y1="46" x2="36" y2="46" stroke="{}" stroke-width="3"/>
  <text x="44" y="50" font-family="sans-serif" font-size="14" fill="#333">Predicted</text>
</g>
"##,
                legend_x, legend_y, truth_color, inferred_color
            );

            svg.insert_str(close_pos, &legend);
        }

        Ok(svg)
    }
}

#[pymethods]
impl PyReconciliationComparison {
    /// Fraction of correctly inferred species mappings.
    #[getter]
    fn mapping_accuracy(&self) -> f64 {
        self.inner.mapping_accuracy()
    }

    /// Fraction of correctly inferred events.
    #[getter]
    fn event_accuracy(&self) -> f64 {
        self.inner.event_accuracy()
    }

    /// Fraction of nodes correct on both mapping and event.
    #[getter]
    fn both_accuracy(&self) -> f64 {
        self.inner.both_accuracy()
    }

    /// Number of nodes matched by clade (topology agreement).
    #[getter]
    fn nodes_compared(&self) -> usize {
        self.inner.nodes_compared
    }

    /// Number of correct species mappings.
    #[getter]
    fn correct_mappings(&self) -> usize {
        self.inner.correct_mappings
    }

    /// Number of correct events.
    #[getter]
    fn correct_events(&self) -> usize {
        self.inner.correct_events
    }

    /// Number of nodes where mapping was evaluable (both sides had species info).
    #[getter]
    fn mappings_evaluated(&self) -> usize {
        self.inner.mappings_evaluated
    }

    /// Number of truth clades not found in inferred tree.
    #[getter]
    fn unmatched_truth_clades(&self) -> usize {
        self.inner.unmatched_truth_clades
    }

    /// Number of inferred clades not found in truth tree.
    #[getter]
    fn unmatched_inferred_clades(&self) -> usize {
        self.inner.unmatched_inferred_clades
    }

    /// Leaf mapping sanity check: (correct, total).
    #[getter]
    fn leaf_check(&self) -> (usize, usize) {
        (self.inner.leaf_correct, self.inner.leaf_total)
    }

    /// Per-node comparison as a pandas DataFrame.
    ///
    /// Columns: truth_node_idx, truth_node_name, inferred_node_idx, inferred_node_name,
    /// clade, truth_species, inferred_species, truth_event,
    /// inferred_event, mapping_correct, event_correct
    fn to_dataframe(&self, py: Python) -> PyResult<PyObject> {
        let pandas = py.import("pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        let n = self.inner.node_details.len();
        let mut truth_node_idxs: Vec<usize> = Vec::with_capacity(n);
        let mut truth_node_names: Vec<String> = Vec::with_capacity(n);
        let mut inf_node_idxs: Vec<Option<usize>> = Vec::with_capacity(n);
        let mut inf_node_names: Vec<String> = Vec::with_capacity(n);
        let mut clades: Vec<String> = Vec::with_capacity(n);
        let mut truth_species: Vec<Option<String>> = Vec::with_capacity(n);
        let mut inf_species: Vec<Option<String>> = Vec::with_capacity(n);
        let mut truth_events: Vec<String> = Vec::with_capacity(n);
        let mut inf_events: Vec<String> = Vec::with_capacity(n);
        let mut mapping_correct: Vec<Option<bool>> = Vec::with_capacity(n);
        let mut event_correct: Vec<bool> = Vec::with_capacity(n);

        for detail in &self.inner.node_details {
            truth_node_idxs.push(detail.truth_node_idx);
            truth_node_names.push(detail.truth_node_name.clone());
            // usize::MAX sentinel means consensus (no single inferred node)
            if detail.inferred_node_idx == usize::MAX {
                inf_node_idxs.push(None);
            } else {
                inf_node_idxs.push(Some(detail.inferred_node_idx));
            }
            inf_node_names.push(detail.inferred_node_name.clone());
            let clade_str: Vec<&str> = detail.clade.iter().map(|s| s.as_str()).collect();
            clades.push(clade_str.join(","));
            truth_species.push(detail.truth_species.clone());
            inf_species.push(detail.inferred_species.clone());
            truth_events.push(event_to_string(&detail.truth_event));
            inf_events.push(event_to_string(&detail.inferred_event));
            mapping_correct.push(detail.mapping_correct);
            event_correct.push(detail.event_correct);
        }

        dict.set_item("truth_node_idx", truth_node_idxs)?;
        dict.set_item("truth_node_name", truth_node_names)?;
        dict.set_item("inferred_node_idx", inf_node_idxs)?;
        dict.set_item("inferred_node_name", inf_node_names)?;
        dict.set_item("clade", clades)?;
        dict.set_item("truth_species", truth_species)?;
        dict.set_item("inferred_species", inf_species)?;
        dict.set_item("truth_event", truth_events)?;
        dict.set_item("inferred_event", inf_events)?;
        dict.set_item("mapping_correct", mapping_correct)?;
        dict.set_item("event_correct", event_correct)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Event confusion matrix as a pandas DataFrame.
    ///
    /// Columns: truth_event, inferred_event, count
    fn confusion_matrix(&self, py: Python) -> PyResult<PyObject> {
        let pandas = py.import("pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        let n = self.inner.event_confusion.len();
        let mut truth_events: Vec<String> = Vec::with_capacity(n);
        let mut inf_events: Vec<String> = Vec::with_capacity(n);
        let mut counts: Vec<usize> = Vec::with_capacity(n);

        for ((truth, inferred), &count) in &self.inner.event_confusion {
            truth_events.push(event_to_string(truth));
            inf_events.push(event_to_string(inferred));
            counts.push(count);
        }

        dict.set_item("truth_event", truth_events)?;
        dict.set_item("inferred_event", inf_events)?;
        dict.set_item("count", counts)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Generate SVG showing both reconciliations overlaid on the same species tree.
    ///
    /// Returns the post-processed SVG string. Use `display()` for Jupyter rendering.
    #[pyo3(signature = (
        truth_color="green",
        inferred_color="red",
        internal_gene_names=false,
        internal_species_names=true,
        landscape=false,
        fill_species=true,
        species_color="#cccccc",
        species_fontsize=40.0,
        background="white",
    ))]
    fn to_svg(
        &self,
        truth_color: &str,
        inferred_color: &str,
        internal_gene_names: bool,
        internal_species_names: bool,
        landscape: bool,
        fill_species: bool,
        species_color: &str,
        species_fontsize: f64,
        background: &str,
    ) -> PyResult<String> {
        self.generate_svg(
            truth_color, inferred_color, internal_gene_names, internal_species_names,
            landscape, fill_species, species_color, species_fontsize, background,
        )
    }

    /// Display both reconciliations overlaid on the same species tree using thirdkind.
    ///
    /// The truth gene tree is shown in green, the inferred (ALERax) tree in red.
    #[pyo3(signature = (
        truth_color="green",
        inferred_color="red",
        internal_gene_names=false,
        internal_species_names=true,
        landscape=false,
        fill_species=true,
        species_color="#cccccc",
        species_fontsize=40.0,
        background="white",
    ))]
    fn display(
        &self,
        py: Python,
        truth_color: &str,
        inferred_color: &str,
        internal_gene_names: bool,
        internal_species_names: bool,
        landscape: bool,
        fill_species: bool,
        species_color: &str,
        species_fontsize: f64,
        background: &str,
    ) -> PyResult<PyObject> {
        let svg = self.generate_svg(
            truth_color, inferred_color, internal_gene_names, internal_species_names,
            landscape, fill_species, species_color, species_fontsize, background,
        )?;

        let ipython_display = py.import("IPython.display")?;
        let svg_class = ipython_display.getattr("SVG")?;
        let display_obj = svg_class.call1((svg,))?;
        Ok(display_obj.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "ReconciliationComparison(nodes={}, mapping={:.1}%, event={:.1}%, both={:.1}%)",
            self.inner.nodes_compared,
            self.inner.mapping_accuracy() * 100.0,
            self.inner.event_accuracy() * 100.0,
            self.inner.both_accuracy() * 100.0,
        )
    }
}

/// Result of comparing truth against multiple reconciliation samples.
///
/// Provides per-sample metrics, consensus comparison, and mean accuracies.
#[pyclass]
#[derive(Clone)]
pub struct PyMultiSampleComparison {
    inner: crate::comparison::MultiSampleComparison,
}

#[pymethods]
impl PyMultiSampleComparison {
    /// Mean mapping accuracy across all samples.
    #[getter]
    fn mean_mapping_accuracy(&self) -> f64 {
        self.inner.mean_mapping_accuracy
    }

    /// Mean event accuracy across all samples.
    #[getter]
    fn mean_event_accuracy(&self) -> f64 {
        self.inner.mean_event_accuracy
    }

    /// Number of samples compared.
    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.per_sample.len()
    }

    /// Consensus comparison (majority vote across samples).
    #[getter]
    fn consensus(&self) -> PyReconciliationComparison {
        PyReconciliationComparison { inner: self.inner.consensus.clone(), trees: None }
    }

    /// Per-sample mapping accuracies as a list.
    #[getter]
    fn sample_mapping_accuracies(&self) -> Vec<f64> {
        self.inner.per_sample.iter().map(|c| c.mapping_accuracy()).collect()
    }

    /// Per-sample event accuracies as a list.
    #[getter]
    fn sample_event_accuracies(&self) -> Vec<f64> {
        self.inner.per_sample.iter().map(|c| c.event_accuracy()).collect()
    }

    /// Get the comparison for a specific sample index.
    fn get_sample(&self, index: usize) -> PyResult<PyReconciliationComparison> {
        if index >= self.inner.per_sample.len() {
            return Err(PyValueError::new_err(format!(
                "Sample index {} out of range (0..{})", index, self.inner.per_sample.len()
            )));
        }
        Ok(PyReconciliationComparison { inner: self.inner.per_sample[index].clone(), trees: None })
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiSampleComparison(samples={}, mean_mapping={:.1}%, mean_event={:.1}%, \
             consensus_mapping={:.1}%, consensus_event={:.1}%)",
            self.inner.per_sample.len(),
            self.inner.mean_mapping_accuracy * 100.0,
            self.inner.mean_event_accuracy * 100.0,
            self.inner.consensus.mapping_accuracy() * 100.0,
            self.inner.consensus.event_accuracy() * 100.0,
        )
    }
}

/// Result of ALERax reconciliation for a gene family.
///
/// Contains all reconciliation samples, estimated evolutionary rates,
/// log-likelihood, and summary statistics.
#[pyclass]
#[derive(Clone)]
pub struct PyAleRaxResult {
    /// All reconciliation samples (typically 100)
    #[pyo3(get)]
    gene_trees: Vec<PyGeneTree>,

    /// Estimated duplication rate
    #[pyo3(get)]
    duplication_rate: f64,

    /// Estimated loss rate
    #[pyo3(get)]
    loss_rate: f64,

    /// Estimated transfer rate
    #[pyo3(get)]
    transfer_rate: f64,

    /// Log-likelihood of the reconciliation
    #[pyo3(get)]
    likelihood: f64,

    /// Summary statistics across all reconciliation samples
    #[pyo3(get)]
    statistics: PyReconciliationStatistics,
}

#[pymethods]
impl PyAleRaxResult {
    /// Get a summary string of the reconciliation result.
    fn __repr__(&self) -> String {
        format!(
            "PyAleRaxResult(samples={}, D={:.4}, L={:.4}, T={:.4}, logL={:.2})",
            self.gene_trees.len(),
            self.duplication_rate,
            self.loss_rate,
            self.transfer_rate,
            self.likelihood
        )
    }
}

/// Reconcile gene trees with species tree using ALERax.
///
/// Calls the external ALERax tool to perform phylogenetic reconciliation,
/// inferring duplication, transfer, and loss events. Returns reconciliation
/// samples that can be analyzed, visualized, and exported.
///
/// # Arguments
/// * `species_tree` - Species tree as PySpeciesTree, Newick string, or file path
/// * `gene_trees` - Gene trees as:
///   - Single Newick string or file path (creates "family_0")
///   - List of Newick strings/paths (creates "family_0", "family_1", ...)
///   - Dict mapping family names to Newick strings/paths
/// * `output_dir` - Optional output directory (default: temporary directory)
/// * `num_samples` - Number of reconciliation samples per family (default: 100)
/// * `model` - Model parametrization: "PER-FAMILY" or "GLOBAL" (default: "PER-FAMILY")
/// * `seed` - Random seed for reproducibility (default: None)
/// * `keep_output` - Whether to preserve ALERax output files (default: False)
/// * `alerax_path` - Path to alerax executable (default: "alerax")
///
/// # Returns
/// Dictionary mapping family names to PyAleRaxResult objects containing
/// reconciled gene trees and estimated evolutionary rates.
///
/// # Example
/// ```python
/// import rustree
///
/// species_tree = rustree.parse_species_tree("species.nwk")
/// results = rustree.reconcile_with_alerax(species_tree, "gene.nwk", seed=42)
///
/// result = results["family_0"]
/// print(f"D={result.duplication_rate}, T={result.transfer_rate}")
///
/// # Access reconciliation samples
/// best_tree = result.gene_trees[0]
/// best_tree.to_svg("reconciled.svg")
/// ```
#[pyfunction]
#[pyo3(signature = (species_tree, gene_trees, output_dir=None, num_samples=100, model="PER-FAMILY".to_string(), gene_tree_rooting=None, seed=None, keep_output=false, alerax_path="alerax".to_string()))]
fn reconcile_with_alerax(
    py: Python,
    species_tree: PyObject,
    gene_trees: PyObject,
    output_dir: Option<String>,
    num_samples: usize,
    model: String,
    gene_tree_rooting: Option<String>,
    seed: Option<u64>,
    keep_output: bool,
    alerax_path: String,
) -> PyResult<HashMap<String, PyAleRaxResult>> {
    use crate::alerax::{run_alerax, AleRaxConfig, GeneFamily, ModelType, validate_inputs};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // Parse model type
    let model_type = match model.to_uppercase().as_str() {
        "PER-FAMILY" | "PER_FAMILY" => ModelType::PerFamily,
        "GLOBAL" => ModelType::Global,
        _ => return Err(PyValueError::new_err(format!(
            "Invalid model type '{}'. Must be 'PER-FAMILY' or 'GLOBAL'", model
        ))),
    };

    // Parse species tree
    let species_tree_obj = species_tree.extract::<PySpeciesTree>(py)
        .or_else(|_| {
            // Try as string (Newick or file path)
            let tree_str = species_tree.extract::<String>(py)?;

            // Check if it's a file path
            if PathBuf::from(&tree_str).exists() {
                // Read file and parse
                let content = fs::read_to_string(&tree_str)
                    .map_err(|e| PyValueError::new_err(format!("Failed to read species tree file: {}", e)))?;
                parse_species_tree(&content)
            } else {
                // Parse as Newick string
                parse_species_tree(&tree_str)
            }
        })?;

    let species_tree_newick = species_tree_obj.tree.to_newick()
        .map_err(|e| PyValueError::new_err(e))?;

    // Parse gene trees — accepts PyGeneTree, PyGeneForest, list of PyGeneTree,
    // Newick string, file path, list of strings/paths, or dict of names→strings/paths.
    let mut gene_tree_list: Vec<(String, String)> = Vec::new();

    // Try PyGeneTree objects first (single, list, or forest)
    if let Ok(single_gt) = gene_trees.extract::<PyGeneTree>(py) {
        let newick = single_gt.rec_tree.gene_tree.to_newick()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert gene tree to Newick: {}", e)))?;
        gene_tree_list.push(("family_0".to_string(), newick + ";"));
    } else if let Ok(gt_list) = gene_trees.extract::<Vec<PyGeneTree>>(py) {
        for (idx, gt) in gt_list.iter().enumerate() {
            let newick = gt.rec_tree.gene_tree.to_newick()
                .map_err(|e| PyValueError::new_err(format!("Failed to convert gene tree to Newick: {}", e)))?;
            gene_tree_list.push((format!("family_{}", idx), newick + ";"));
        }
    } else if let Ok(forest) = gene_trees.extract::<PyGeneForest>(py) {
        for (idx, rec_tree) in forest.forest.gene_trees.iter().enumerate() {
            let newick = rec_tree.gene_tree.to_newick()
                .map_err(|e| PyValueError::new_err(format!("Failed to convert gene tree to Newick: {}", e)))?;
            gene_tree_list.push((format!("family_{}", idx), newick + ";"));
        }
    } else if let Ok(dict) = gene_trees.extract::<HashMap<String, String>>(py) {
        for (name, newick_or_path) in dict {
            let newick = if PathBuf::from(&newick_or_path).exists() {
                fs::read_to_string(&newick_or_path)
                    .map_err(|e| PyValueError::new_err(format!(
                        "Failed to read gene tree file '{}': {}", newick_or_path, e
                    )))?
            } else {
                newick_or_path
            };
            gene_tree_list.push((name, newick));
        }
    } else if let Ok(list) = gene_trees.extract::<Vec<String>>(py) {
        // List of Newick strings or file paths
        for (idx, newick_or_path) in list.iter().enumerate() {
            let (name, newick) = if PathBuf::from(newick_or_path).exists() {
                let path = PathBuf::from(newick_or_path);
                let name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(&format!("family_{}", idx))
                    .to_string();
                let newick = fs::read_to_string(&path)
                    .map_err(|e| PyValueError::new_err(format!(
                        "Failed to read gene tree file '{}': {}", newick_or_path, e
                    )))?;
                (name, newick)
            } else {
                (format!("family_{}", idx), newick_or_path.clone())
            };
            gene_tree_list.push((name, newick));
        }
    } else if let Ok(single) = gene_trees.extract::<String>(py) {
        // Single Newick string or file path
        let (name, newick) = if PathBuf::from(&single).exists() {
            let path = PathBuf::from(&single);
            let name = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("family_0")
                .to_string();
            let newick = fs::read_to_string(&path)
                .map_err(|e| PyValueError::new_err(format!(
                    "Failed to read gene tree file '{}': {}", single, e
                )))?;
            (name, newick)
        } else {
            ("family_0".to_string(), single)
        };
        gene_tree_list.push((name, newick));
    } else {
        return Err(PyValueError::new_err(
            "gene_trees must be a PyGeneTree, PyGeneForest, list of PyGeneTree, \
             Newick string, file path, list of strings/paths, or dict mapping names to strings/paths"
        ));
    }

    if gene_tree_list.is_empty() {
        return Err(PyValueError::new_err("No gene trees provided"));
    }

    // Validate inputs before running ALERax
    validate_inputs(&species_tree_obj.tree, &gene_tree_list)
        .map_err(|e| PyValueError::new_err(format!("Input validation failed: {}", e)))?;

    // Create output directory
    let (output_path, _temp_dir) = if let Some(dir) = output_dir {
        (PathBuf::from(dir), None)
    } else {
        let temp_dir = TempDir::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create temp directory: {}", e)))?;
        let path = temp_dir.path().to_path_buf();
        (path, Some(temp_dir))
    };

    // Create input directory for ALERax files
    let input_dir = output_path.join("input");
    fs::create_dir_all(&input_dir)
        .map_err(|e| PyValueError::new_err(format!("Failed to create input directory: {}", e)))?;

    let species_tree_path = input_dir.join("species_tree.newick");
    let families_file_path = input_dir.join("families.txt");

    // Prepare gene families
    let families: Vec<GeneFamily> = gene_tree_list.iter().map(|(name, newick)| {
        GeneFamily {
            name: name.clone(),
            gene_tree_path: input_dir.join(format!("{}.newick", name)),
            gene_tree_newick: newick.clone(),
        }
    }).collect();

    let alerax_output_dir = output_path.join("alerax_output");

    // Create ALERax config
    let config = AleRaxConfig {
        species_tree_path,
        families,
        families_file_path,
        output_dir: alerax_output_dir,
        num_samples,
        model_parametrization: model_type,
        gene_tree_rooting,
        seed,
        alerax_path,
    };

    // Run ALERax
    let mut results = run_alerax(config, &species_tree_newick)
        .map_err(|e| PyValueError::new_err(format!("ALERax execution failed: {}", e)))?;

    // Auto-rename: map ALERax species names back to original names.
    // ALERax internally renames species tree nodes; reconcile_forest() does this
    // rename automatically, but run_alerax() does not, so we replicate it here.
    {
        use crate::node::map_by_topology;

        // Get the ALERax species tree from the first parsed result
        let first_result = results.values().next()
            .ok_or_else(|| PyValueError::new_err("No ALERax results"))?;
        if let Some(first_rec) = first_result.reconciled_trees.first() {
            let alerax_species_tree = &first_rec.species_tree;

            // Build topology mapping: alerax_idx -> original_idx
            let topo_mapping = map_by_topology(&species_tree_obj.tree, alerax_species_tree)
                .map_err(|e| PyValueError::new_err(format!("Failed to build topology mapping: {}", e)))?;

            // Build name mapping: alerax_name -> original_name
            let mut alerax_to_original: HashMap<String, String> = HashMap::new();
            for (alerax_idx, original_idx) in &topo_mapping {
                let alerax_name = &alerax_species_tree.nodes[*alerax_idx].name;
                let original_name = &species_tree_obj.tree.nodes[*original_idx].name;
                alerax_to_original.insert(alerax_name.clone(), original_name.clone());
            }

            // Build a single renamed species tree
            let renamed_species = {
                let mut tree = (**alerax_species_tree).clone();
                for node in &mut tree.nodes {
                    if let Some(orig) = alerax_to_original.get(&node.name) {
                        node.name = orig.clone();
                    }
                }
                Arc::new(tree)
            };

            // Apply rename to all family results
            for result in results.values_mut() {
                for rec_tree in &mut result.reconciled_trees {
                    rec_tree.species_tree = Arc::clone(&renamed_species);
                    // Rename gene tree leaf names
                    for node in &mut rec_tree.gene_tree.nodes {
                        if node.left_child.is_none() && node.right_child.is_none() {
                            if let Some(pos) = node.name.rfind('_') {
                                let alerax_species = &node.name[..pos];
                                let suffix = &node.name[pos..];
                                if let Some(original_species) = alerax_to_original.get(alerax_species) {
                                    node.name = format!("{}{}", original_species, suffix);
                                }
                            }
                        }
                    }
                }

                // Rename species keys in statistics
                let old_eps = std::mem::take(&mut result.statistics.events_per_species);
                for (species_name, counts) in old_eps {
                    let renamed = alerax_to_original.get(&species_name)
                        .cloned()
                        .unwrap_or(species_name);
                    result.statistics.events_per_species.insert(renamed, counts);
                }

                let old_mt = std::mem::take(&mut result.statistics.mean_transfers);
                for (source, dests) in old_mt {
                    let renamed_source = alerax_to_original.get(&source)
                        .cloned()
                        .unwrap_or(source);
                    let mut renamed_dests = HashMap::new();
                    for (dest, count) in dests {
                        let renamed_dest = alerax_to_original.get(&dest)
                            .cloned()
                            .unwrap_or(dest);
                        renamed_dests.insert(renamed_dest, count);
                    }
                    result.statistics.mean_transfers.insert(renamed_source, renamed_dests);
                }
            }
        }
    }

    // Convert to Python types
    let mut py_results = HashMap::new();
    for (family_name, result) in results {
        // Share a single species tree Arc across all samples in this family.
        // Take the Arc from the first reconciled tree.
        let mut rec_trees_iter = result.reconciled_trees.into_iter();
        let first_rec_tree = rec_trees_iter.next();
        let (shared_species_tree, first_py_gene_tree) = match first_rec_tree {
            Some(rt) => {
                let shared = Arc::clone(&rt.species_tree);
                (shared, Some(PyGeneTree { rec_tree: rt }))
            }
            None => {
                (Arc::new(species_tree_obj.tree.as_ref().clone()), None)
            }
        };
        let mut py_gene_trees: Vec<PyGeneTree> = first_py_gene_tree.into_iter().collect();
        py_gene_trees.extend(rec_trees_iter.map(|mut rec_tree| {
            rec_tree.species_tree = Arc::clone(&shared_species_tree);
            PyGeneTree { rec_tree }
        }));

        // Convert statistics
        let mean_event_counts = PyEventCounts {
            speciations: result.statistics.mean_event_counts.speciations,
            speciation_losses: result.statistics.mean_event_counts.speciation_losses,
            duplications: result.statistics.mean_event_counts.duplications,
            duplication_losses: result.statistics.mean_event_counts.duplication_losses,
            transfers: result.statistics.mean_event_counts.transfers,
            transfer_losses: result.statistics.mean_event_counts.transfer_losses,
            losses: result.statistics.mean_event_counts.losses,
            leaves: result.statistics.mean_event_counts.leaves,
        };

        let events_per_species: HashMap<String, PyEventCounts> = result.statistics.events_per_species
            .into_iter()
            .map(|(species, counts)| {
                (species, PyEventCounts {
                    speciations: counts.speciations,
                    speciation_losses: counts.speciation_losses,
                    duplications: counts.duplications,
                    duplication_losses: counts.duplication_losses,
                    transfers: counts.transfers,
                    transfer_losses: counts.transfer_losses,
                    losses: counts.losses,
                    leaves: counts.leaves,
                })
            })
            .collect();

        let statistics = PyReconciliationStatistics {
            mean_event_counts,
            mean_transfers: result.statistics.mean_transfers,
            events_per_species,
        };

        py_results.insert(family_name, PyAleRaxResult {
            gene_trees: py_gene_trees,
            duplication_rate: result.duplication_rate,
            loss_rate: result.loss_rate,
            transfer_rate: result.transfer_rate,
            likelihood: result.likelihood,
            statistics,
        });
    }

    // Clean up temp directory if not keeping output
    if !keep_output {
        drop(_temp_dir);
    }

    Ok(py_results)
}

// ============================================================================
// GeneForest
// ============================================================================

/// A collection of gene trees associated with a single species tree.
///
/// PyGeneForest wraps a GeneForest and provides methods for:
/// - Pruning by species tree or species leaf names
/// - Reconciliation with ALERax
/// - Batch access to gene trees
///
/// # Example
/// ```python
/// import rustree
///
/// st = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)
/// gts = st.simulate_dtl_batch(n=5, lambda_d=0.5, lambda_t=0.2, lambda_l=0.3, seed=123)
/// forest = rustree.GeneForest(st, gts)
/// print(forest)  # GeneForest(species_leaves=20, gene_trees=5)
/// ```
#[pyclass(name = "GeneForest")]
#[derive(Clone)]
pub struct PyGeneForest {
    forest: crate::node::gene_forest::GeneForest,
}

#[pymethods]
impl PyGeneForest {
    /// Create a new GeneForest from a species tree and list of gene trees.
    #[new]
    fn new(species_tree: &PySpeciesTree, gene_trees: Vec<PyGeneTree>) -> PyResult<Self> {
        let species_arc = Arc::clone(&species_tree.tree);
        let rec_trees: Vec<RecTree> = gene_trees.into_iter()
            .map(|gt| {
                let mut rt = gt.rec_tree;
                rt.species_tree = Arc::clone(&species_arc);
                rt
            })
            .collect();

        Ok(PyGeneForest {
            forest: crate::node::gene_forest::GeneForest::from_rec_trees(species_arc, rec_trees),
        })
    }

    /// Number of gene trees in the forest.
    fn __len__(&self) -> usize {
        self.forest.len()
    }

    /// Get a gene tree by index.
    fn __getitem__(&self, idx: usize) -> PyResult<PyGeneTree> {
        self.forest.get(idx)
            .map(|rt| PyGeneTree { rec_tree: rt.clone() })
            .ok_or_else(|| PyValueError::new_err(format!("Index {} out of range (len={})", idx, self.forest.len())))
    }

    /// Get the species tree.
    #[getter]
    fn species_tree(&self) -> PySpeciesTree {
        PySpeciesTree {
            tree: Arc::clone(&self.forest.species_tree),
        }
    }

    /// Get all gene trees.
    #[getter]
    fn gene_trees(&self) -> Vec<PyGeneTree> {
        self.forest.gene_trees.iter()
            .map(|rt| PyGeneTree { rec_tree: rt.clone() })
            .collect()
    }

    /// Return a new forest with each gene tree pruned to extant leaves only.
    ///
    /// Keeps only gene tree leaves whose event is Leaf (i.e., extant genes).
    /// Gene trees with zero extant leaves are dropped.
    fn sample_extant(&self) -> PyResult<PyGeneForest> {
        let sampled = self.forest.sample_extant()
            .map_err(|e| PyValueError::new_err(format!("Failed to sample extant: {}", e)))?;
        Ok(PyGeneForest { forest: sampled })
    }

    /// Prune the forest to match a target species tree.
    ///
    /// The target species tree's leaves must be a subset of this forest's
    /// species tree leaves. Returns a new GeneForest with pruned trees.
    fn prune_to_species_tree(&self, target: &PySpeciesTree) -> PyResult<PyGeneForest> {
        let pruned = self.forest.prune_to_species_tree(&target.tree)
            .map_err(|e| PyValueError::new_err(format!("Failed to prune: {}", e)))?;
        Ok(PyGeneForest { forest: pruned })
    }

    /// Sample leaves and filter all trees accordingly.
    ///
    /// # Arguments
    /// * `names` - List of species leaf names to keep
    fn sample_leaves(&self, names: Vec<String>) -> PyResult<PyGeneForest> {
        let sampled = self.forest.sample_leaves(&names)
            .map_err(|e| PyValueError::new_err(format!("Failed to sample: {}", e)))?;
        Ok(PyGeneForest { forest: sampled })
    }

    /// Reconcile the gene forest with ALERax.
    ///
    /// Runs ALERax on all gene trees, parses all results, and auto-renames
    /// species tree nodes back to their original names.
    ///
    /// # Arguments
    /// * `output_dir` - Optional output directory (default: temp directory)
    /// * `num_samples` - Number of reconciliation samples per family (default: 100)
    /// * `model` - Model parametrization: "PER-FAMILY" or "GLOBAL" (default: "PER-FAMILY")
    /// * `seed` - Random seed for reproducibility (optional)
    /// * `keep_output` - Whether to preserve ALERax output files (default: False)
    /// * `alerax_path` - Path to alerax executable (default: "alerax")
    ///
    /// # Returns
    /// An AleRaxForestResult containing all reconciliation data.
    #[pyo3(signature = (output_dir=None, num_samples=100, model="PER-FAMILY".to_string(), gene_tree_rooting=None, seed=None, keep_output=false, alerax_path="alerax".to_string()))]
    fn reconcile_with_alerax(
        &self,
        output_dir: Option<String>,
        num_samples: usize,
        model: String,
        gene_tree_rooting: Option<String>,
        seed: Option<u64>,
        keep_output: bool,
        alerax_path: String,
    ) -> PyResult<PyAleRaxForestResult> {
        use crate::alerax::{reconcile_forest, ModelType};
        use std::path::PathBuf;

        let model_type = match model.to_uppercase().as_str() {
            "PER-FAMILY" | "PER_FAMILY" => ModelType::PerFamily,
            "GLOBAL" => ModelType::Global,
            _ => return Err(PyValueError::new_err(format!(
                "Invalid model type '{}'. Must be 'PER-FAMILY' or 'GLOBAL'", model
            ))),
        };

        let output_path = output_dir.map(PathBuf::from);

        let result = reconcile_forest(
            &self.forest,
            output_path,
            num_samples,
            model_type,
            gene_tree_rooting,
            seed,
            &alerax_path,
            keep_output,
        ).map_err(|e| PyValueError::new_err(format!("ALERax reconciliation failed: {}", e)))?;

        Ok(PyAleRaxForestResult::from_rust(result))
    }

    fn __iter__(&self) -> PyGeneForestIter {
        PyGeneForestIter {
            forest: self.forest.clone(),
            pos: 0,
        }
    }

    fn __repr__(&self) -> String {
        let species_leaves = self.forest.species_tree().nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .count();
        format!(
            "GeneForest(species_leaves={}, gene_trees={})",
            species_leaves,
            self.forest.len()
        )
    }
}

/// Iterator over gene trees in a GeneForest.
#[pyclass]
struct PyGeneForestIter {
    forest: crate::node::gene_forest::GeneForest,
    pos: usize,
}

#[pymethods]
impl PyGeneForestIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&mut self) -> Option<PyGeneTree> {
        let gt = self.forest.get(self.pos)?;
        self.pos += 1;
        Some(PyGeneTree { rec_tree: gt.clone() })
    }
}

// ============================================================================
// ALERax Forest Result
// ============================================================================

/// Comprehensive result of reconciling a GeneForest with ALERax.
///
/// Contains per-family results (reconciled trees, rates, likelihoods)
/// plus aggregate statistics across all families, accessible as DataFrames.
#[pyclass(name = "AleRaxForestResult")]
pub struct PyAleRaxForestResult {
    #[pyo3(get)]
    family_results: HashMap<String, PyAleRaxResult>,
    mean_species_event_counts: HashMap<String, Vec<crate::alerax::SpeciesEventRow>>,
    total_species_event_counts_data: Vec<crate::alerax::SpeciesEventRow>,
    total_transfers_data: Vec<crate::alerax::TransferRow>,
    #[pyo3(get)]
    output_dir: Option<String>,
}

impl PyAleRaxForestResult {
    fn from_rust(result: crate::alerax::AleRaxForestResult) -> Self {
        let mut py_family_results = HashMap::new();
        for (name, family_result) in result.family_results {
            let py_gene_trees: Vec<PyGeneTree> = family_result.reconciled_trees
                .into_iter()
                .map(|rt| PyGeneTree { rec_tree: rt })
                .collect();

            let mean_event_counts = PyEventCounts {
                speciations: family_result.statistics.mean_event_counts.speciations,
                speciation_losses: family_result.statistics.mean_event_counts.speciation_losses,
                duplications: family_result.statistics.mean_event_counts.duplications,
                duplication_losses: family_result.statistics.mean_event_counts.duplication_losses,
                transfers: family_result.statistics.mean_event_counts.transfers,
                transfer_losses: family_result.statistics.mean_event_counts.transfer_losses,
                losses: family_result.statistics.mean_event_counts.losses,
                leaves: family_result.statistics.mean_event_counts.leaves,
            };

            let events_per_species: HashMap<String, PyEventCounts> = family_result.statistics.events_per_species
                .into_iter()
                .map(|(species, counts)| {
                    (species, PyEventCounts {
                        speciations: counts.speciations,
                        speciation_losses: counts.speciation_losses,
                        duplications: counts.duplications,
                        duplication_losses: counts.duplication_losses,
                        transfers: counts.transfers,
                        transfer_losses: counts.transfer_losses,
                        losses: counts.losses,
                        leaves: counts.leaves,
                    })
                })
                .collect();

            let statistics = PyReconciliationStatistics {
                mean_event_counts,
                mean_transfers: family_result.statistics.mean_transfers,
                events_per_species,
            };

            py_family_results.insert(name, PyAleRaxResult {
                gene_trees: py_gene_trees,
                duplication_rate: family_result.duplication_rate,
                loss_rate: family_result.loss_rate,
                transfer_rate: family_result.transfer_rate,
                likelihood: family_result.likelihood,
                statistics,
            });
        }

        PyAleRaxForestResult {
            family_results: py_family_results,
            mean_species_event_counts: result.mean_species_event_counts,
            total_species_event_counts_data: result.total_species_event_counts,
            total_transfers_data: result.total_transfers,
            output_dir: result.output_dir.map(|p| p.to_string_lossy().to_string()),
        }
    }
}

/// Helper: convert SpeciesEventRow slice to a pandas DataFrame.
fn species_event_rows_to_df(
    py: Python,
    rows: &[crate::alerax::SpeciesEventRow],
) -> PyResult<PyObject> {
    let pandas = py.import("pandas")?;
    let dict = pyo3::types::PyDict::new(py);

    let labels: Vec<&str> = rows.iter().map(|r| r.species_label.as_str()).collect();
    let speciations: Vec<f64> = rows.iter().map(|r| r.speciations).collect();
    let duplications: Vec<f64> = rows.iter().map(|r| r.duplications).collect();
    let losses: Vec<f64> = rows.iter().map(|r| r.losses).collect();
    let transfers: Vec<f64> = rows.iter().map(|r| r.transfers).collect();
    let presence: Vec<f64> = rows.iter().map(|r| r.presence).collect();
    let origination: Vec<f64> = rows.iter().map(|r| r.origination).collect();
    let copies: Vec<f64> = rows.iter().map(|r| r.copies).collect();
    let singletons: Vec<f64> = rows.iter().map(|r| r.singletons).collect();
    let transfers_to: Vec<f64> = rows.iter().map(|r| r.transfers_to).collect();

    dict.set_item("species_label", labels)?;
    dict.set_item("speciations", speciations)?;
    dict.set_item("duplications", duplications)?;
    dict.set_item("losses", losses)?;
    dict.set_item("transfers", transfers)?;
    dict.set_item("presence", presence)?;
    dict.set_item("origination", origination)?;
    dict.set_item("copies", copies)?;
    dict.set_item("singletons", singletons)?;
    dict.set_item("transfers_to", transfers_to)?;

    let df = pandas.call_method1("DataFrame", (dict,))?;
    Ok(df.into())
}

#[pymethods]
impl PyAleRaxForestResult {
    /// Get mean species event counts for a specific family as a pandas DataFrame.
    ///
    /// Columns: species_label, speciations, duplications, losses, transfers,
    ///          presence, origination, copies, singletons, transfers_to
    fn mean_species_event_counts(&self, py: Python, family_name: &str) -> PyResult<PyObject> {
        let rows = self.mean_species_event_counts.get(family_name)
            .ok_or_else(|| PyValueError::new_err(format!(
                "No mean species event counts for family '{}'", family_name
            )))?;
        species_event_rows_to_df(py, rows)
    }

    /// Get total (aggregate) species event counts as a pandas DataFrame.
    ///
    /// Columns: species_label, speciations, duplications, losses, transfers,
    ///          presence, origination, copies, singletons, transfers_to
    #[getter]
    fn total_species_event_counts(&self, py: Python) -> PyResult<PyObject> {
        species_event_rows_to_df(py, &self.total_species_event_counts_data)
    }

    /// Get total (aggregate) transfers as a pandas DataFrame.
    ///
    /// Columns: source, destination, count
    #[getter]
    fn total_transfers(&self, py: Python) -> PyResult<PyObject> {
        let pandas = py.import("pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        let sources: Vec<&str> = self.total_transfers_data.iter().map(|r| r.source.as_str()).collect();
        let dests: Vec<&str> = self.total_transfers_data.iter().map(|r| r.destination.as_str()).collect();
        let counts: Vec<f64> = self.total_transfers_data.iter().map(|r| r.count).collect();

        dict.set_item("source", sources)?;
        dict.set_item("destination", dests)?;
        dict.set_item("count", counts)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// List all family names.
    fn family_names(&self) -> Vec<String> {
        self.family_results.keys().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "AleRaxForestResult(families={}, total_species={})",
            self.family_results.len(),
            self.total_species_event_counts_data.len()
        )
    }
}

/// Compare two reconciliations (standalone function).
///
/// Matches nodes by their clade (set of descendant extant leaf names).
/// Both gene trees must have the same extant leaf set.
///
/// # Arguments
/// * `truth` - Ground truth reconciliation (e.g., from simulation + sample_extant())
/// * `inferred` - Inferred reconciliation (e.g., from ALERax)
///
/// # Returns
/// A PyReconciliationComparison with accuracy metrics and per-node details.
#[pyfunction]
fn compare_reconciliations(truth: &PyGeneTree, inferred: &PyGeneTree) -> PyResult<PyReconciliationComparison> {
    let result = crate::comparison::compare_reconciliations(&truth.rec_tree, &inferred.rec_tree)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyReconciliationComparison {
        inner: result,
        trees: Some((truth.rec_tree.clone(), inferred.rec_tree.clone())),
    })
}

/// Compare truth reconciliation against multiple inferred samples (standalone function).
///
/// Computes per-sample metrics and a consensus comparison (majority vote per clade).
///
/// # Arguments
/// * `truth` - Ground truth reconciliation
/// * `samples` - List of inferred reconciliations
///
/// # Returns
/// A PyMultiSampleComparison with per-sample and consensus accuracy.
#[pyfunction]
fn compare_reconciliations_multi(truth: &PyGeneTree, samples: Vec<PyGeneTree>) -> PyResult<PyMultiSampleComparison> {
    let sample_recs: Vec<_> = samples.iter().map(|s| s.rec_tree.clone()).collect();
    let result = crate::comparison::compare_reconciliations_multi(&truth.rec_tree, &sample_recs)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyMultiSampleComparison { inner: result })
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
fn create_training_sample(
    py: Python,
    sp_newick_path: &str,
    g_newick_path: &str,
    xml_path: &str,
) -> PyResult<PyObject> {
    use crate::newick::newick::parse_newick;
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
/// * `gene_tree` – a `PyGeneTree` returned by `sample_extant()`.
///
/// # Returns
/// A Python dict with the same keys as `create_training_sample` (minus the
/// three `*_path` keys which are set to empty strings).
#[cfg(feature = "python")]
#[pyfunction]
fn create_training_sample_from_sim(
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

/// Build all tensors needed for a training sample in one Rust call.
///
/// Replaces the Python functions: unroot_tree, sample_gene_coloring,
/// _build_species_tensors, _build_gene_edges, _build_gene_event_types,
/// and the masking/assembly logic in get_sample.
///
/// Returns a dict of numpy arrays ready to be wrapped into HeteroData.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (base_sample, seed, force_mask_last_added, predict_root_position, sample_order = "random"))]
fn build_training_tensors(
    py: Python,
    base_sample: &Bound<'_, pyo3::types::PyDict>,
    seed: u64,
    force_mask_last_added: bool,
    predict_root_position: bool,
    sample_order: &str,
) -> PyResult<PyObject> {
    use numpy::PyArray1;
    use pyo3::types::PyDict;
    use rand::seq::SliceRandom;
    use rand::Rng;
    use std::collections::{HashMap, HashSet, VecDeque};

    // ---- Extract fields from base_sample dict ----
    let g_root_name: String = base_sample
        .get_item("g_root_name")?.ok_or_else(|| PyValueError::new_err("missing g_root_name"))?
        .extract::<String>()?;
    let nb_g_leaves: usize = base_sample
        .get_item("nb_g_leaves")?.ok_or_else(|| PyValueError::new_err("missing nb_g_leaves"))?
        .extract::<usize>()?;
    let species_names: Vec<String> = base_sample
        .get_item("species_names")?.ok_or_else(|| PyValueError::new_err("missing species_names"))?
        .extract::<Vec<String>>()?;
    let g_leaves_names: Vec<String> = base_sample
        .get_item("g_leaves_names")?.ok_or_else(|| PyValueError::new_err("missing g_leaves_names"))?
        .extract::<Vec<String>>()?;
    let true_root: Vec<String> = base_sample
        .get_item("true_root")?.ok_or_else(|| PyValueError::new_err("missing true_root"))?
        .extract::<Vec<String>>()?;

    // g_neighbors: Dict[str, List[str]]
    let g_neighbors: HashMap<String, Vec<String>> = base_sample
        .get_item("g_neighbors")?.ok_or_else(|| PyValueError::new_err("missing g_neighbors"))?
        .extract()?;

    // sp_children: Dict[str, List[str]]
    let sp_children: HashMap<String, Vec<String>> = base_sample
        .get_item("sp_children")?.ok_or_else(|| PyValueError::new_err("missing sp_children"))?
        .extract()?;

    // true_states: Dict[str, str]
    let true_states: HashMap<String, String> = base_sample
        .get_item("true_states")?.ok_or_else(|| PyValueError::new_err("missing true_states"))?
        .extract()?;

    // true_events: Dict[str, int]
    let true_events: HashMap<String, i32> = base_sample
        .get_item("true_events")?.ok_or_else(|| PyValueError::new_err("missing true_events"))?
        .extract()?;

    // ---- RNG setup ----
    let mut rng = StdRng::seed_from_u64(seed);

    // ---- Determine sampling parameters ----
    let max_to_sample = if !force_mask_last_added || predict_root_position {
        nb_g_leaves.saturating_sub(2)
    } else {
        nb_g_leaves.saturating_sub(1)
    };

    let nb_sampled = if predict_root_position {
        max_to_sample
    } else {
        rng.gen_range(0..=max_to_sample)
    };

    // ---- Unroot tree ----
    let mut g_neighbors_unrooted: HashMap<String, Vec<String>> = HashMap::new();
    for (k, v) in &g_neighbors {
        if k != &g_root_name {
            let filtered: Vec<String> = v.iter()
                .filter(|n| *n != &g_root_name)
                .cloned()
                .collect();
            g_neighbors_unrooted.insert(k.clone(), filtered);
        }
    }
    // Connect root's two children
    if let Some(root_nbrs) = g_neighbors.get(&g_root_name) {
        if root_nbrs.len() == 2 {
            let c1 = &root_nbrs[0];
            let c2 = &root_nbrs[1];
            g_neighbors_unrooted.entry(c1.clone()).or_default().push(c2.clone());
            g_neighbors_unrooted.entry(c2.clone()).or_default().push(c1.clone());
        }
    }

    // ---- Frontier sampling / node selection ----
    let include_root_node: bool;
    let selected: HashSet<String>;
    let frontier_names: Vec<String>;
    let last_added_name: Option<String>;

    if predict_root_position {
        include_root_node = false;
        selected = g_neighbors_unrooted.keys().cloned().collect();
        frontier_names = Vec::new();
        last_added_name = None;
    } else if nb_sampled == max_to_sample && !force_mask_last_added {
        include_root_node = true;
        selected = g_neighbors.keys()
            .filter(|k| *k != &g_root_name)
            .cloned().collect();
        frontier_names = vec![g_root_name.clone()];
        last_added_name = None;
    } else if nb_sampled == max_to_sample && force_mask_last_added {
        include_root_node = true;
        selected = g_neighbors.keys().cloned().collect();
        frontier_names = Vec::new();
        last_added_name = Some(g_root_name.clone());
    } else {
        include_root_node = false;
        // ---- Inline frontier sampling (sample_gene_coloring) ----
        let node_list: Vec<String> = g_neighbors_unrooted.keys().cloned().collect();
        let n = node_list.len();
        let name_to_idx: HashMap<&str, usize> = node_list.iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (u, vs) in &g_neighbors_unrooted {
            let ui = name_to_idx[u.as_str()];
            for v in vs {
                let vi = name_to_idx[v.as_str()];
                if ui != vi {
                    adj[ui].push(vi);
                }
            }
        }
        // Deduplicate adjacency
        for a in &mut adj {
            a.sort_unstable();
            a.dedup();
        }

        let degrees: Vec<usize> = adj.iter().map(|a| a.len()).collect();
        let leaves_idx: Vec<usize> = degrees.iter().enumerate()
            .filter(|(_, &d)| d == 1)
            .map(|(i, _)| i)
            .collect();
        let internal_idx: Vec<usize> = degrees.iter().enumerate()
            .filter(|(_, &d)| d == 3)
            .map(|(i, _)| i)
            .collect();

        // Leaf distance BFS for bottom_up ordering
        let leaf_dist: Option<Vec<i32>> = if sample_order == "bottom_up" {
            let mut dist = vec![-1i32; n];
            let mut queue = VecDeque::new();
            for &li in &leaves_idx {
                dist[li] = 0;
                queue.push_back(li);
            }
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if dist[v] == -1 {
                        dist[v] = dist[u] + 1;
                        queue.push_back(v);
                    }
                }
            }
            Some(dist)
        } else {
            None
        };

        let mut mapped = vec![false; n];
        for &li in &leaves_idx {
            mapped[li] = true;
        }
        let internal_set: HashSet<usize> = internal_idx.iter().copied().collect();

        let mut mapped_cnt = vec![0u32; n];
        for &ii in &internal_idx {
            let cnt = adj[ii].iter().filter(|&&nb| mapped[nb]).count() as u32;
            mapped_cnt[ii] = cnt;
        }

        let mut frontier_idx: Vec<usize> = internal_idx.iter()
            .filter(|&&ii| mapped_cnt[ii] >= 2)
            .copied()
            .collect();
        let mut added_order: Vec<usize> = Vec::new();
        let mut last_added_i: Option<usize> = None;

        while added_order.len() < nb_sampled && !frontier_idx.is_empty() {
            let pick = if let Some(ref dist) = leaf_dist {
                let min_d = frontier_idx.iter().map(|&v| dist[v]).min().unwrap();
                let closest: Vec<usize> = frontier_idx.iter().enumerate()
                    .filter(|(_, &v)| dist[v] == min_d)
                    .map(|(i, _)| i)
                    .collect();
                *closest.choose(&mut rng).unwrap()
            } else {
                rng.gen_range(0..frontier_idx.len())
            };
            let v = frontier_idx.swap_remove(pick);
            mapped[v] = true;
            added_order.push(v);
            last_added_i = Some(v);

            for &u in &adj[v] {
                if !mapped[u] {
                    mapped_cnt[u] += 1;
                    if internal_set.contains(&u) && mapped_cnt[u] == 2 {
                        frontier_idx.push(u);
                    }
                }
            }
        }

        let mut sel = HashSet::new();
        for &li in &leaves_idx {
            sel.insert(node_list[li].clone());
        }
        for &ai in &added_order {
            sel.insert(node_list[ai].clone());
        }
        selected = sel;
        frontier_names = frontier_idx.iter().map(|&i| node_list[i].clone()).collect();
        last_added_name = last_added_i.map(|i| node_list[i].clone());
    };

    // ---- Species tensors ----
    let n_sp = species_names.len();
    let mut sp_name_to_id: HashMap<&str, i32> = HashMap::new();
    let mut sp_name_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, name) in species_names.iter().enumerate() {
        sp_name_to_id.insert(name.as_str(), (i + 1) as i32);
        sp_name_to_idx.insert(name.as_str(), i);
    }
    let x_sp: Vec<i32> = species_names.iter()
        .map(|name: &String| sp_name_to_id[name.as_str()])
        .collect();

    let mut sp_child_src: Vec<i32> = Vec::new();
    let mut sp_child_dst: Vec<i32> = Vec::new();
    let mut sp_parent_src: Vec<i32> = Vec::new();
    let mut sp_parent_dst: Vec<i32> = Vec::new();
    for (p, children) in &sp_children {
        let pi = sp_name_to_idx[p.as_str()] as i32;
        for c in children {
            let ci = sp_name_to_idx[c.as_str()] as i32;
            sp_child_src.push(pi);
            sp_child_dst.push(ci);
            sp_parent_src.push(ci);
            sp_parent_dst.push(pi);
        }
    }

    // ---- Gene node ordering ----
    let active_neighbors = if include_root_node { &g_neighbors } else { &g_neighbors_unrooted };
    let mut g_names: Vec<String> = active_neighbors.keys().cloned().collect();
    g_names.sort();
    let n_gene = g_names.len();
    let g_name_to_idx: HashMap<&str, usize> = g_names.iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    // ---- Gene feature tensors ----
    let leaf_set: HashSet<&str> = g_leaves_names.iter().map(|s: &String| s.as_str()).collect();
    let frontier_set: HashSet<&str> = frontier_names.iter().map(|s| s.as_str()).collect();

    let mut x_gene = vec![0i32; n_gene];
    let mut g_true_sp = vec![0i32; n_gene];
    let mut event_true = vec![0i32; n_gene];
    let mut event_input = vec![0i32; n_gene];
    let mut frontier_mask = vec![0u8; n_gene]; // bool as u8
    let mut is_leaf = vec![0u8; n_gene];
    let mut mask_label_node = vec![0u8; n_gene];

    for (i, name) in g_names.iter().enumerate() {
        // Species mapping
        let sp_name = true_states.get(name)
            .ok_or_else(|| PyValueError::new_err(format!("Gene node '{}' not in true_states", name)))?;
        let sp_id = sp_name_to_id.get(sp_name.as_str())
            .ok_or_else(|| PyValueError::new_err(format!("Species '{}' not found", sp_name)))?;
        g_true_sp[i] = *sp_id;

        // Input features: masked for unselected nodes
        if selected.contains(name) {
            x_gene[i] = *sp_id;
        }
        // else x_gene[i] stays 0

        // Event types
        if leaf_set.contains(name.as_str()) {
            event_true[i] = 3; // P/leaf
        } else {
            event_true[i] = *true_events.get(name)
                .ok_or_else(|| PyValueError::new_err(format!("Gene node '{}' not in true_events", name)))?;
        }

        // Event input: masked for unselected nodes
        if selected.contains(name) {
            event_input[i] = event_true[i];
        } else {
            event_input[i] = 4; // mask
        }

        // Frontier mask
        if frontier_set.contains(name.as_str()) {
            frontier_mask[i] = 1;
        }

        // Is leaf
        if leaf_set.contains(name.as_str()) {
            is_leaf[i] = 1;
        }
    }

    // Handle last_added masking
    let last_idx: i32;
    let mask_last_added_event: bool;
    if let Some(ref la) = last_added_name {
        if let Some(&idx) = g_name_to_idx.get(la.as_str()) {
            last_idx = idx as i32;
            if force_mask_last_added {
                event_input[idx] = 4; // mask
                mask_last_added_event = true;
                mask_label_node[idx] = 1;
            } else {
                mask_last_added_event = false;
            }
        } else {
            last_idx = -1;
            mask_last_added_event = false;
        }
    } else {
        last_idx = -1;
        mask_last_added_event = false;
    }

    // ---- Gene edges ----
    let mut undirected_edges: HashSet<(i32, i32)> = HashSet::new();
    for (u, vs) in active_neighbors {
        let ui = g_name_to_idx[u.as_str()] as i32;
        for v in vs {
            if let Some(&vi_usize) = g_name_to_idx.get(v.as_str()) {
                let vi = vi_usize as i32;
                if ui != vi {
                    let e = if ui < vi { (ui, vi) } else { (vi, ui) };
                    undirected_edges.insert(e);
                }
            }
        }
    }

    let mut directed: Vec<(i32, i32)> = undirected_edges.into_iter().collect();
    directed.sort();

    let n_edges = directed.len();
    let mut g_edge_src = Vec::with_capacity(n_edges * 2);
    let mut g_edge_dst = Vec::with_capacity(n_edges * 2);
    let mut g_dir_src = Vec::with_capacity(n_edges);
    let mut g_dir_dst = Vec::with_capacity(n_edges);
    for &(a, b) in &directed {
        // Directed (canonical)
        g_dir_src.push(a);
        g_dir_dst.push(b);
        // Undirected (both directions)
        g_edge_src.push(a);
        g_edge_dst.push(b);
        g_edge_src.push(b);
        g_edge_dst.push(a);
    }

    // Root edge target
    let root_pair_0 = g_name_to_idx.get(true_root[0].as_str()).map(|&v| v as i32).unwrap_or(-1);
    let root_pair_1 = g_name_to_idx.get(true_root[1].as_str()).map(|&v| v as i32).unwrap_or(-1);
    let root_edge_target: i32 = if !include_root_node {
        let pair_fwd = (root_pair_0.min(root_pair_1), root_pair_0.max(root_pair_1));
        directed.iter().position(|&e| e == pair_fwd)
            .map(|i| i as i32)
            .unwrap_or(-1)
    } else {
        -1
    };

    // ---- Build result dict of numpy arrays ----
    // Use from_slice / into_pyarray so numpy fully owns the memory (no Rust
    // allocator callback on dealloc). This prevents cross-allocator corruption
    // when DataLoader workers tear down.
    let result = PyDict::new(py);
    result.set_item("x_sp", PyArray1::from_slice(py, &x_sp))?;
    result.set_item("x_gene", PyArray1::from_slice(py, &x_gene))?;
    result.set_item("g_true_sp", PyArray1::from_slice(py, &g_true_sp))?;
    result.set_item("event_true", PyArray1::from_slice(py, &event_true))?;
    result.set_item("event_input", PyArray1::from_slice(py, &event_input))?;
    result.set_item("frontier_mask", PyArray1::from_slice(py, &frontier_mask))?;
    result.set_item("is_leaf", PyArray1::from_slice(py, &is_leaf))?;
    result.set_item("mask_label_node", PyArray1::from_slice(py, &mask_label_node))?;
    result.set_item("last_added_index", last_idx)?;
    result.set_item("mask_last_added_event", mask_last_added_event)?;

    // Edge indices as 2×E arrays — build via ndarray then convert to numpy.
    let edge_2d = |src: &[i32], dst: &[i32], name: &str| -> PyResult<PyObject> {
        use numpy::ndarray::Array2;
        use numpy::ToPyArray;
        let n = src.len();
        let mut flat = Vec::with_capacity(2 * n);
        flat.extend_from_slice(src);
        flat.extend_from_slice(dst);
        let arr = Array2::from_shape_vec([2, n], flat)
            .map_err(|e| PyValueError::new_err(format!("{}: {}", name, e)))?;
        Ok(arr.to_pyarray(py).into_any().unbind())
    };
    result.set_item("sp_child_edge", edge_2d(&sp_child_src, &sp_child_dst, "sp_child_edge")?)?;
    result.set_item("sp_parent_edge", edge_2d(&sp_parent_src, &sp_parent_dst, "sp_parent_edge")?)?;
    result.set_item("g_edge", edge_2d(&g_edge_src, &g_edge_dst, "g_edge")?)?;
    result.set_item("g_dir_edge", edge_2d(&g_dir_src, &g_dir_dst, "g_dir_edge")?)?;
    result.set_item("root_edge_target", root_edge_target)?;

    Ok(result.into())
}

// ============================================================================
// Batch OTF generation: internal helpers and build_otf_batch
// ============================================================================

/// Internal representation of a training sample (mirrors Python base_sample dict).
struct RustBaseSample {
    g_root_name: String,
    nb_g_leaves: usize,
    species_names: Vec<String>,
    g_leaves_names: Vec<String>,
    true_root: Vec<String>,
    g_neighbors: HashMap<String, Vec<String>>,
    sp_children: HashMap<String, Vec<String>>,
    true_states: HashMap<String, String>,
    true_events: HashMap<String, i32>,
}

/// Per-sample task tensors (before collation).
struct TaskTensors {
    x_gene: Vec<i32>,
    g_true_sp: Vec<i32>,   // 1-based species IDs
    event_true: Vec<i32>,
    event_input: Vec<i32>,
    frontier_mask: Vec<u8>,
    is_leaf: Vec<u8>,
    mask_label_node: Vec<u8>,
    g_edge_src: Vec<i32>,
    g_edge_dst: Vec<i32>,
    g_dir_src: Vec<i32>,
    g_dir_dst: Vec<i32>,
    root_edge_target: i32,
}

/// GCN normalization: add self-loops, compute D^{-1/2} A D^{-1/2} edge weights.
///
/// Matches PyG's `gcn_norm(edge_index, None, num_nodes, improved=False,
/// add_self_loops=True, flow="source_to_target")`.
///
/// Returns (new_src, new_dst, edge_weights) with self-loops included.
fn gcn_norm_internal(src: &[i32], dst: &[i32], num_nodes: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let n_orig = src.len();
    let n_total = n_orig + num_nodes; // original edges + self-loops

    let mut new_src = Vec::with_capacity(n_total);
    let mut new_dst = Vec::with_capacity(n_total);
    new_src.extend_from_slice(src);
    new_dst.extend_from_slice(dst);
    for i in 0..num_nodes {
        new_src.push(i as i32);
        new_dst.push(i as i32);
    }

    // Degree = number of edges targeting each node (flow="source_to_target")
    let mut deg = vec![0u32; num_nodes];
    for &d in &new_dst {
        deg[d as usize] += 1;
    }

    // D^{-1/2}
    let inv_sqrt: Vec<f32> = deg.iter().map(|&d| {
        if d == 0 { 0.0f32 } else { (d as f32).powf(-0.5) }
    }).collect();

    // Edge weights: inv_sqrt[src] * inv_sqrt[dst]
    let weights: Vec<f32> = new_src.iter().zip(new_dst.iter())
        .map(|(&s, &d)| inv_sqrt[s as usize] * inv_sqrt[d as usize])
        .collect();

    (new_src, new_dst, weights)
}

/// Compute varlen attention metadata from sizes array.
/// Returns (prefix_sums_with_leading_zero, max_size).
fn compute_varlen_metadata(sizes: &[i32]) -> (Vec<i32>, i32) {
    let mut cu = Vec::with_capacity(sizes.len() + 1);
    cu.push(0);
    let mut cumsum: i32 = 0;
    let mut max_val: i32 = 0;
    for &s in sizes {
        cumsum += s;
        cu.push(cumsum);
        if s > max_val { max_val = s; }
    }
    (cu, max_val)
}

/// Collated tensors for one task across a full batch.
struct CollatedTask {
    gene_x: Vec<i32>,
    gene_y: Vec<i32>,
    event: Vec<i32>,
    event_true: Vec<i32>,
    frontier_mask: Vec<u8>,
    is_leaf: Vec<u8>,
    mask_label_node: Vec<u8>,
    // GCN-normalized gene edges (includes self-loops)
    gene_ei_src: Vec<i32>,
    gene_ei_dst: Vec<i32>,
    gene_ew: Vec<f32>,
    g_dir_src: Vec<i32>,
    g_dir_dst: Vec<i32>,
    true_root_index: Vec<i32>,
    g_sizes: Vec<i32>,
    g_ptr: Vec<i32>,
    g_batch: Vec<i32>,
    root_edges_ptr: Vec<i32>,
    // Varlen attention metadata
    cu_g: Vec<i32>,
    max_g: i32,
}

/// Extract base sample data from a RecTree (no Python boundary).
fn extract_base_sample_internal(rt: &RecTree) -> Result<RustBaseSample, String> {
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

/// Build task tensors for one sample (internal; mirrors build_training_tensors logic).
/// sp_name_to_id maps species name (owned) → 1-based ID.
fn build_task_tensors_internal(
    sample: &RustBaseSample,
    sp_name_to_id: &HashMap<String, i32>,
    rng: &mut StdRng,
    force_mask_last_added: bool,
    predict_root_position: bool,
    sample_order: &str,
) -> Result<TaskTensors, String> {
    use rand::Rng;
    use rand::seq::SliceRandom;
    use std::collections::{HashSet, VecDeque};

    let g_root_name = &sample.g_root_name;
    let nb_g_leaves = sample.nb_g_leaves;
    let g_neighbors = &sample.g_neighbors;
    let g_leaves_names = &sample.g_leaves_names;
    let true_root = &sample.true_root;
    let true_states = &sample.true_states;
    let true_events = &sample.true_events;

    let max_to_sample = if !force_mask_last_added || predict_root_position {
        nb_g_leaves.saturating_sub(2)
    } else {
        nb_g_leaves.saturating_sub(1)
    };
    let nb_sampled = if predict_root_position {
        max_to_sample
    } else {
        rng.gen_range(0..=max_to_sample)
    };

    // Unroot tree: remove root node and reconnect its children
    let mut g_neighbors_unrooted: HashMap<String, Vec<String>> = HashMap::new();
    for (k, v) in g_neighbors {
        if k != g_root_name {
            let filtered: Vec<String> = v.iter()
                .filter(|n| n.as_str() != g_root_name.as_str())
                .cloned()
                .collect();
            g_neighbors_unrooted.insert(k.clone(), filtered);
        }
    }
    if let Some(root_nbrs) = g_neighbors.get(g_root_name) {
        // Connect all former root children to each other (handles bifurcating
        // and multifurcating roots alike).
        for i in 0..root_nbrs.len() {
            for j in (i + 1)..root_nbrs.len() {
                let ci = root_nbrs[i].clone();
                let cj = root_nbrs[j].clone();
                g_neighbors_unrooted.entry(ci.clone()).or_default().push(cj.clone());
                g_neighbors_unrooted.entry(cj.clone()).or_default().push(ci.clone());
            }
        }
    }
    // Safety: ensure every node referenced as a neighbor is also a key.
    // Collect missing keys first to avoid borrowing issues.
    let all_referenced: Vec<String> = g_neighbors_unrooted.values()
        .flat_map(|vs| vs.iter().cloned())
        .collect();
    for name in all_referenced {
        g_neighbors_unrooted.entry(name).or_default();
    }

    // Node selection
    let include_root_node: bool;
    let selected: HashSet<String>;
    let frontier_names: Vec<String>;
    let last_added_name: Option<String>;

    if predict_root_position {
        include_root_node = false;
        selected = g_neighbors_unrooted.keys().cloned().collect();
        frontier_names = Vec::new();
        last_added_name = None;
    } else if nb_sampled == max_to_sample && !force_mask_last_added {
        include_root_node = true;
        selected = g_neighbors.keys().filter(|k| k.as_str() != g_root_name.as_str()).cloned().collect();
        frontier_names = vec![g_root_name.clone()];
        last_added_name = None;
    } else if nb_sampled == max_to_sample && force_mask_last_added {
        include_root_node = true;
        selected = g_neighbors.keys().cloned().collect();
        frontier_names = Vec::new();
        last_added_name = Some(g_root_name.clone());
    } else {
        include_root_node = false;
        let node_list: Vec<String> = g_neighbors_unrooted.keys().cloned().collect();
        let n = node_list.len();
        let name_to_idx: HashMap<&str, usize> = node_list.iter().enumerate()
            .map(|(i, name)| (name.as_str(), i)).collect();

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (u, vs) in &g_neighbors_unrooted {
            let ui = name_to_idx[u.as_str()];
            for v in vs {
                let vi = match name_to_idx.get(v.as_str()) {
                    Some(&idx) => idx,
                    None => {
                        eprintln!(
                            "[rustree] WARNING: neighbor '{}' of node '{}' not found \
                             in g_neighbors_unrooted keys — skipping edge",
                            v, u
                        );
                        continue;
                    }
                };
                if ui != vi { adj[ui].push(vi); }
            }
        }
        for a in &mut adj { a.sort_unstable(); a.dedup(); }

        let degrees: Vec<usize> = adj.iter().map(|a| a.len()).collect();
        let leaves_idx: Vec<usize> = degrees.iter().enumerate()
            .filter(|(_, &d)| d == 1).map(|(i, _)| i).collect();
        let internal_idx: Vec<usize> = degrees.iter().enumerate()
            .filter(|(_, &d)| d == 3).map(|(i, _)| i).collect();

        let leaf_dist: Option<Vec<i32>> = if sample_order == "bottom_up" {
            let mut dist = vec![-1i32; n];
            let mut queue = VecDeque::new();
            for &li in &leaves_idx { dist[li] = 0; queue.push_back(li); }
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if dist[v] == -1 { dist[v] = dist[u] + 1; queue.push_back(v); }
                }
            }
            Some(dist)
        } else { None };

        let mut mapped = vec![false; n];
        for &li in &leaves_idx { mapped[li] = true; }
        let internal_set: HashSet<usize> = internal_idx.iter().copied().collect();

        let mut mapped_cnt = vec![0u32; n];
        for &ii in &internal_idx {
            mapped_cnt[ii] = adj[ii].iter().filter(|&&nb| mapped[nb]).count() as u32;
        }

        let mut frontier_idx: Vec<usize> = internal_idx.iter()
            .filter(|&&ii| mapped_cnt[ii] >= 2).copied().collect();
        let mut added_order: Vec<usize> = Vec::new();
        let mut last_added_i: Option<usize> = None;

        while added_order.len() < nb_sampled && !frontier_idx.is_empty() {
            let pick = if let Some(ref dist) = leaf_dist {
                let min_d = frontier_idx.iter().map(|&v| dist[v]).min().unwrap();
                let closest: Vec<usize> = frontier_idx.iter().enumerate()
                    .filter(|(_, &v)| dist[v] == min_d).map(|(i, _)| i).collect();
                *closest.choose(rng).unwrap()
            } else {
                rng.gen_range(0..frontier_idx.len())
            };
            let v = frontier_idx.swap_remove(pick);
            mapped[v] = true;
            added_order.push(v);
            last_added_i = Some(v);
            for &u in &adj[v] {
                if !mapped[u] {
                    mapped_cnt[u] += 1;
                    if internal_set.contains(&u) && mapped_cnt[u] == 2 {
                        frontier_idx.push(u);
                    }
                }
            }
        }

        let mut sel = HashSet::new();
        for &li in &leaves_idx { sel.insert(node_list[li].clone()); }
        for &ai in &added_order { sel.insert(node_list[ai].clone()); }
        selected = sel;
        frontier_names = frontier_idx.iter().map(|&i| node_list[i].clone()).collect();
        last_added_name = last_added_i.map(|i| node_list[i].clone());
    }

    let active_neighbors = if include_root_node { g_neighbors } else { &g_neighbors_unrooted };
    let mut g_names: Vec<String> = active_neighbors.keys().cloned().collect();
    g_names.sort();
    let n_gene = g_names.len();
    let g_name_to_idx: HashMap<&str, usize> = g_names.iter().enumerate()
        .map(|(i, name)| (name.as_str(), i)).collect();

    let leaf_set: HashSet<&str> = g_leaves_names.iter().map(|s| s.as_str()).collect();
    let frontier_set: HashSet<&str> = frontier_names.iter().map(|s| s.as_str()).collect();

    let mut x_gene = vec![0i32; n_gene];
    let mut g_true_sp = vec![0i32; n_gene];
    let mut event_true = vec![0i32; n_gene];
    let mut event_input = vec![0i32; n_gene];
    let mut frontier_mask = vec![0u8; n_gene];
    let mut is_leaf = vec![0u8; n_gene];
    let mut mask_label_node = vec![0u8; n_gene];

    for (i, name) in g_names.iter().enumerate() {
        let sp_name = true_states.get(name)
            .ok_or_else(|| format!("Gene node '{}' not in true_states", name))?;
        let sp_id = sp_name_to_id.get(sp_name.as_str())
            .ok_or_else(|| format!("Species '{}' not found in sp_name_to_id", sp_name))?;
        g_true_sp[i] = *sp_id;

        if selected.contains(name) { x_gene[i] = *sp_id; }

        if leaf_set.contains(name.as_str()) {
            event_true[i] = 3;
        } else {
            event_true[i] = *true_events.get(name)
                .ok_or_else(|| format!("Gene node '{}' not in true_events", name))?;
        }

        event_input[i] = if selected.contains(name) { event_true[i] } else { 4 };

        if frontier_set.contains(name.as_str()) { frontier_mask[i] = 1; }
        if leaf_set.contains(name.as_str()) { is_leaf[i] = 1; }
    }

    if let Some(ref la) = last_added_name {
        if let Some(&idx) = g_name_to_idx.get(la.as_str()) {
            if force_mask_last_added {
                event_input[idx] = 4;
                mask_label_node[idx] = 1;
            }
        }
    }

    let mut undirected_edges: HashSet<(i32, i32)> = HashSet::new();
    for (u, vs) in active_neighbors {
        let ui = g_name_to_idx[u.as_str()] as i32;
        for v in vs {
            if let Some(&vi_usize) = g_name_to_idx.get(v.as_str()) {
                let vi = vi_usize as i32;
                if ui != vi {
                    let e = if ui < vi { (ui, vi) } else { (vi, ui) };
                    undirected_edges.insert(e);
                }
            }
        }
    }

    let mut directed: Vec<(i32, i32)> = undirected_edges.into_iter().collect();
    directed.sort();
    let n_edges = directed.len();
    let mut g_edge_src = Vec::with_capacity(n_edges * 2);
    let mut g_edge_dst = Vec::with_capacity(n_edges * 2);
    let mut g_dir_src = Vec::with_capacity(n_edges);
    let mut g_dir_dst = Vec::with_capacity(n_edges);
    for &(a, b) in &directed {
        g_dir_src.push(a); g_dir_dst.push(b);
        g_edge_src.push(a); g_edge_dst.push(b);
        g_edge_src.push(b); g_edge_dst.push(a);
    }

    let root_pair_0 = true_root.first()
        .and_then(|s| g_name_to_idx.get(s.as_str())).map(|&v| v as i32).unwrap_or(-1);
    let root_pair_1 = true_root.get(1)
        .and_then(|s| g_name_to_idx.get(s.as_str())).map(|&v| v as i32).unwrap_or(-1);
    let root_edge_target: i32 = if !include_root_node && root_pair_0 >= 0 && root_pair_1 >= 0 {
        let pair_fwd = (root_pair_0.min(root_pair_1), root_pair_0.max(root_pair_1));
        directed.iter().position(|&e| e == pair_fwd).map(|i| i as i32).unwrap_or(-1)
    } else {
        -1
    };

    Ok(TaskTensors {
        x_gene, g_true_sp, event_true, event_input,
        frontier_mask, is_leaf, mask_label_node,
        g_edge_src, g_edge_dst, g_dir_src, g_dir_dst, root_edge_target,
    })
}

/// Collate per-sample task tensors into a single batch.
///
/// Also applies GCN normalization to gene edges and computes varlen attention metadata.
fn collate_task_tensors(tensors: &[TaskTensors]) -> CollatedTask {
    let mut gene_x = Vec::new();
    let mut gene_y = Vec::new();
    let mut event = Vec::new();
    let mut event_true = Vec::new();
    let mut frontier_mask = Vec::new();
    let mut is_leaf = Vec::new();
    let mut mask_label_node = Vec::new();
    let mut g_edge_src_raw = Vec::new();
    let mut g_edge_dst_raw = Vec::new();
    let mut g_dir_src = Vec::new();
    let mut g_dir_dst = Vec::new();
    let mut true_root_index = Vec::new();
    let mut g_sizes: Vec<i32> = Vec::new();
    let mut g_ptr: Vec<i32> = vec![0];
    let mut g_batch: Vec<i32> = Vec::new();
    let mut root_edges_ptr: Vec<i32> = vec![0];

    let mut cum_g: i32 = 0;
    let mut cum_re: i32 = 0;

    for (s_idx, t) in tensors.iter().enumerate() {
        let ng = t.x_gene.len() as i32;
        let ndir = t.g_dir_src.len() as i32;

        gene_x.extend_from_slice(&t.x_gene);
        gene_y.extend(t.g_true_sp.iter().map(|&v| v - 1));
        event.extend_from_slice(&t.event_input);
        event_true.extend_from_slice(&t.event_true);
        frontier_mask.extend_from_slice(&t.frontier_mask);
        is_leaf.extend_from_slice(&t.is_leaf);
        mask_label_node.extend_from_slice(&t.mask_label_node);

        for &s in &t.g_edge_src { g_edge_src_raw.push(s + cum_g); }
        for &d in &t.g_edge_dst { g_edge_dst_raw.push(d + cum_g); }
        for &s in &t.g_dir_src { g_dir_src.push(s + cum_g); }
        for &d in &t.g_dir_dst { g_dir_dst.push(d + cum_g); }

        true_root_index.push(t.root_edge_target);
        g_sizes.push(ng);
        for _ in 0..ng { g_batch.push(s_idx as i32); }
        cum_g += ng;
        g_ptr.push(cum_g);
        cum_re += ndir;
        root_edges_ptr.push(cum_re);
    }

    // Apply GCN normalization to gene edges
    let total_gene_nodes = cum_g as usize;
    let (gene_ei_src, gene_ei_dst, gene_ew) =
        gcn_norm_internal(&g_edge_src_raw, &g_edge_dst_raw, total_gene_nodes);

    // Compute varlen attention metadata
    let (cu_g, max_g) = compute_varlen_metadata(&g_sizes);

    CollatedTask {
        gene_x, gene_y, event, event_true,
        frontier_mask, is_leaf, mask_label_node,
        gene_ei_src, gene_ei_dst, gene_ew,
        g_dir_src, g_dir_dst,
        true_root_index, g_sizes, g_ptr, g_batch, root_edges_ptr,
        cu_g, max_g,
    }
}

/// Build a complete batched dict of numpy arrays for on-the-fly training.
///
/// Simulates one species tree and n_gene_trees gene trees, builds all three
/// task tensors (map / evt / root), collates them, and returns flat numpy arrays
/// ready to be passed to batch_to_heterodata() in Python.
///
/// Parameters
/// ----------
/// n_sp : int          Target number of extant species.
/// lambda_birth : float  Speciation rate (must be > mu_death).
/// mu_death : float    Extinction rate (must be >= 0).
/// sp_seed : int       Seed for species tree simulation.
/// n_gene_trees : int  Number of gene trees per batch.
/// gt_seeds : list[int]  Per-gene-tree simulation seeds (length n_gene_trees).
/// lambda_d/t/l : float  DTL rates for gene tree simulation.
/// enable_event : bool   Include evt_* keys in returned dict.
/// enable_root : bool    Include root_* keys in returned dict.
/// sample_order : str    "random" or "bottom_up".
/// map/evt/root_coloring_seeds : list[int]  Per-sample RNG seeds for tensor construction.
/// min_gene_leaves : int   Minimum extant gene leaves required.
/// max_gene_nodes : int    Maximum total gene nodes (0 = no limit).
/// max_retries : int       Retry attempts per gene tree slot on simulation failure.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (n_sp, lambda_birth, mu_death, sp_seed, n_gene_trees, gt_seeds, lambda_d, lambda_t, lambda_l, enable_event, enable_root, sample_order, map_coloring_seeds, evt_coloring_seeds, root_coloring_seeds, min_gene_leaves, max_gene_nodes, max_retries))]
fn build_otf_batch(
    py: Python,
    n_sp: usize,
    lambda_birth: f64,
    mu_death: f64,
    sp_seed: u64,
    n_gene_trees: usize,
    gt_seeds: Vec<u64>,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    enable_event: bool,
    enable_root: bool,
    sample_order: &str,
    map_coloring_seeds: Vec<u64>,
    evt_coloring_seeds: Vec<u64>,
    root_coloring_seeds: Vec<u64>,
    min_gene_leaves: usize,
    max_gene_nodes: usize,
    max_retries: usize,
) -> PyResult<PyObject> {
    use numpy::PyArray1;
    use pyo3::types::PyDict;
    use crate::node::TraversalOrder;

    // Validate
    if gt_seeds.len() != n_gene_trees {
        return Err(PyValueError::new_err(format!(
            "gt_seeds length {} != n_gene_trees {}", gt_seeds.len(), n_gene_trees)));
    }
    if map_coloring_seeds.len() != n_gene_trees {
        return Err(PyValueError::new_err(format!(
            "map_coloring_seeds length {} != n_gene_trees {}", map_coloring_seeds.len(), n_gene_trees)));
    }
    if enable_event && evt_coloring_seeds.len() != n_gene_trees {
        return Err(PyValueError::new_err(format!(
            "evt_coloring_seeds length {} != n_gene_trees {}", evt_coloring_seeds.len(), n_gene_trees)));
    }
    if enable_root && root_coloring_seeds.len() != n_gene_trees {
        return Err(PyValueError::new_err(format!(
            "root_coloring_seeds length {} != n_gene_trees {}", root_coloring_seeds.len(), n_gene_trees)));
    }
    validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;

    // ---- GIL-free section ------------------------------------------------
    // All simulation, tensor building, and collation is pure Rust with no
    // Python object interaction.  Releasing the GIL here lets other Python
    // threads (prefetcher producers, GPU transfer thread) run concurrently.
    let sample_order_owned = sample_order.to_owned();
    let result_or_err = py.allow_threads(move || -> Result<_, String> {
        let sample_order: &str = &sample_order_owned;

        // 1. Simulate species tree
        let mut sp_rng = StdRng::seed_from_u64(sp_seed);
        let (mut sp_tree_raw, _) = simulate_bd_tree_bwd(n_sp, lambda_birth, mu_death, &mut sp_rng);
        sp_tree_raw.assign_depths();
        let sp_tree_raw_arc = Arc::new(sp_tree_raw);
        let (extant_sp, _) = extract_extant_subtree(&sp_tree_raw_arc)
            .ok_or_else(|| "build_otf_batch: failed to extract extant species subtree".to_string())?;
        let extant_sp_arc = Arc::new(extant_sp);
        let n_sp_nodes = extant_sp_arc.nodes.len();

        // 2. Simulate and filter gene trees (one per slot, individual seeds)
        //    Parallelised with rayon: each slot is independent (own seed, own RNG).
        let species_identity: Vec<Option<usize>> = (0..n_sp_nodes).map(|i| Some(i)).collect();

        let valid_gene_trees: Vec<RecTree> = (0..n_gene_trees).into_par_iter()
            .map(|i| {
                for attempt in 0..=(max_retries as u64) {
                    let gt_seed = gt_seeds[i].wrapping_add(attempt);
                    let mut gt_rng = StdRng::seed_from_u64(gt_seed);

                    let (mut rec_tree, events) = match simulate_dtl_per_species(
                        &extant_sp_arc, extant_sp_arc.root,
                        lambda_d, lambda_t, lambda_l,
                        None, None, false,
                        &mut gt_rng,
                    ) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    rec_tree.species_tree = Arc::clone(&extant_sp_arc);
                    rec_tree.dtl_events = Some(events);

                    // Extract extant gene leaves
                    let extant_gene_indices: std::collections::HashSet<usize> = rec_tree.gene_tree.nodes
                        .iter().enumerate()
                        .filter(|(idx, n)| {
                            n.left_child.is_none() && n.right_child.is_none()
                                && rec_tree.event_mapping[*idx] == Event::Leaf
                        })
                        .map(|(idx, _)| idx)
                        .collect();

                    if extant_gene_indices.is_empty() { continue; }
                    if extant_gene_indices.len() < min_gene_leaves { continue; }

                    let (extant_gene_tree, gene_old_to_new) = match extract_induced_subtree(
                        &rec_tree.gene_tree, &extant_gene_indices,
                    ) {
                        Some(r) => r,
                        None => continue,
                    };

                    if max_gene_nodes > 0 && extant_gene_tree.nodes.len() > max_gene_nodes { continue; }

                    let (new_node_mapping, new_event_mapping) = match remap_gene_tree_indices(
                        &extant_gene_tree, &gene_old_to_new,
                        &rec_tree.node_mapping, &rec_tree.event_mapping,
                        &species_identity,
                    ) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };

                    return Ok(RecTree::new(
                        Arc::clone(&extant_sp_arc),
                        extant_gene_tree,
                        new_node_mapping,
                        new_event_mapping,
                    ));
                }
                Err(format!(
                    "build_otf_batch: gene tree slot {} failed after {} retries", i, max_retries))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // 3. Species tree tensors (computed once, replicated B times during collation)
        let sp_preorder: Vec<usize> = extant_sp_arc.iter_indices(TraversalOrder::PreOrder).collect();
        let species_names: Vec<String> = sp_preorder.iter()
            .map(|&i| extant_sp_arc.nodes[i].name.clone()).collect();

        let mut sp_name_to_id: HashMap<String, i32> = HashMap::new();
        let mut sp_name_to_idx_map: HashMap<String, usize> = HashMap::new();
        for (i, name) in species_names.iter().enumerate() {
            sp_name_to_id.insert(name.clone(), (i + 1) as i32);
            sp_name_to_idx_map.insert(name.clone(), i);
        }

        let x_sp_single: Vec<i32> = species_names.iter()
            .map(|name| sp_name_to_id[name]).collect();

        let mut sp_child_src_s: Vec<i32> = Vec::new();
        let mut sp_child_dst_s: Vec<i32> = Vec::new();
        let mut sp_parent_src_s: Vec<i32> = Vec::new();
        let mut sp_parent_dst_s: Vec<i32> = Vec::new();
        for &idx in &sp_preorder {
            let node = &extant_sp_arc.nodes[idx];
            let pi = sp_name_to_idx_map[&node.name] as i32;
            for child_opt in [node.left_child, node.right_child] {
                if let Some(child) = child_opt {
                    let ci = sp_name_to_idx_map[&extant_sp_arc.nodes[child].name] as i32;
                    sp_child_src_s.push(pi); sp_child_dst_s.push(ci);
                    sp_parent_src_s.push(ci); sp_parent_dst_s.push(pi);
                }
            }
        }

        // 4. Extract base samples from gene trees (parallel)
        let base_samples: Vec<RustBaseSample> = valid_gene_trees.par_iter()
            .map(|rt| extract_base_sample_internal(rt))
            .collect::<Result<Vec<_>, _>>()?;

        // 5. Build task tensors (parallel — each sample has its own seed/RNG)
        let map_tensors: Vec<TaskTensors> = base_samples.par_iter().enumerate()
            .map(|(i, s)| {
                let mut rng = StdRng::seed_from_u64(map_coloring_seeds[i]);
                build_task_tensors_internal(s, &sp_name_to_id, &mut rng, false, false, sample_order)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let evt_tensors: Vec<TaskTensors> = if enable_event {
            base_samples.par_iter().enumerate()
                .map(|(i, s)| {
                    let mut rng = StdRng::seed_from_u64(evt_coloring_seeds[i]);
                    build_task_tensors_internal(s, &sp_name_to_id, &mut rng, true, false, sample_order)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else { Vec::new() };

        let root_tensors: Vec<TaskTensors> = if enable_root {
            base_samples.par_iter().enumerate()
                .map(|(i, s)| {
                    let mut rng = StdRng::seed_from_u64(root_coloring_seeds[i]);
                    build_task_tensors_internal(s, &sp_name_to_id, &mut rng, false, true, sample_order)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else { Vec::new() };

        // 6. Compute species GCN norm once on single-tree edges, then replicate B times
        let (sp_child_norm_src_s, sp_child_norm_dst_s, sp_child_ew_s) =
            gcn_norm_internal(&sp_child_src_s, &sp_child_dst_s, n_sp_nodes);
        let (sp_parent_norm_src_s, sp_parent_norm_dst_s, sp_parent_ew_s) =
            gcn_norm_internal(&sp_parent_src_s, &sp_parent_dst_s, n_sp_nodes);

        let b = n_gene_trees;
        let n_sp_child_e = sp_child_norm_src_s.len(); // includes self-loops
        let n_sp_parent_e = sp_parent_norm_src_s.len();
        let mut sp_x: Vec<i32> = Vec::with_capacity(b * n_sp_nodes);
        let mut sp_child_src: Vec<i32> = Vec::with_capacity(b * n_sp_child_e);
        let mut sp_child_dst: Vec<i32> = Vec::with_capacity(b * n_sp_child_e);
        let mut sp_child_ew: Vec<f32> = Vec::with_capacity(b * n_sp_child_e);
        let mut sp_parent_src: Vec<i32> = Vec::with_capacity(b * n_sp_parent_e);
        let mut sp_parent_dst: Vec<i32> = Vec::with_capacity(b * n_sp_parent_e);
        let mut sp_parent_ew: Vec<f32> = Vec::with_capacity(b * n_sp_parent_e);
        let mut sp_batch: Vec<i32> = Vec::with_capacity(b * n_sp_nodes);
        let mut sp_sizes: Vec<i32> = Vec::with_capacity(b);
        let mut sp_ptr: Vec<i32> = Vec::with_capacity(b + 1);
        sp_ptr.push(0);

        for s_idx in 0..b {
            let offset = (s_idx * n_sp_nodes) as i32;
            sp_x.extend_from_slice(&x_sp_single);
            for (&s, &d) in sp_child_norm_src_s.iter().zip(sp_child_norm_dst_s.iter()) {
                sp_child_src.push(s + offset); sp_child_dst.push(d + offset);
            }
            sp_child_ew.extend_from_slice(&sp_child_ew_s);
            for (&s, &d) in sp_parent_norm_src_s.iter().zip(sp_parent_norm_dst_s.iter()) {
                sp_parent_src.push(s + offset); sp_parent_dst.push(d + offset);
            }
            sp_parent_ew.extend_from_slice(&sp_parent_ew_s);
            for _ in 0..n_sp_nodes { sp_batch.push(s_idx as i32); }
            sp_sizes.push(n_sp_nodes as i32);
            sp_ptr.push((s_idx + 1) as i32 * n_sp_nodes as i32);
        }

        // Compute species varlen attention metadata
        let (cu_sp, max_sp) = compute_varlen_metadata(&sp_sizes);

        // 7. Collate task tensors (includes gene GCN norm + varlen metadata)
        let map_c = collate_task_tensors(&map_tensors);
        let evt_c = if enable_event { Some(collate_task_tensors(&evt_tensors)) } else { None };
        let root_c = if enable_root { Some(collate_task_tensors(&root_tensors)) } else { None };

        Ok((sp_x, sp_child_src, sp_child_dst, sp_child_ew,
            sp_parent_src, sp_parent_dst, sp_parent_ew,
            sp_ptr, sp_sizes, sp_batch, cu_sp, max_sp,
            map_c, evt_c, root_c))
    });
    // ---- End GIL-free section --------------------------------------------

    let (sp_x, sp_child_src, sp_child_dst, sp_child_ew,
         sp_parent_src, sp_parent_dst, sp_parent_ew,
         sp_ptr, sp_sizes, sp_batch, cu_sp, max_sp,
         map_c, evt_c, root_c) =
        result_or_err.map_err(|e| PyValueError::new_err(e))?;

    // 8. Build result dict of numpy arrays
    let result = PyDict::new(py);

    let edge_2d = |src: &[i32], dst: &[i32], name: &str| -> PyResult<PyObject> {
        use numpy::ndarray::Array2;
        use numpy::ToPyArray;
        let n = src.len();
        let mut flat = Vec::with_capacity(2 * n);
        flat.extend_from_slice(src);
        flat.extend_from_slice(dst);
        let arr = Array2::from_shape_vec([2, n], flat)
            .map_err(|e| PyValueError::new_err(format!("{}: {}", name, e)))?;
        Ok(arr.to_pyarray(py).into_any().unbind())
    };

    // Species (GCN-normalized edges with self-loops)
    result.set_item("sp_x", PyArray1::from_slice(py, &sp_x))?;
    result.set_item("sp_child_ei", edge_2d(&sp_child_src, &sp_child_dst, "sp_child_ei")?)?;
    result.set_item("sp_child_ew", PyArray1::from_slice(py, &sp_child_ew))?;
    result.set_item("sp_parent_ei", edge_2d(&sp_parent_src, &sp_parent_dst, "sp_parent_ei")?)?;
    result.set_item("sp_parent_ew", PyArray1::from_slice(py, &sp_parent_ew))?;
    result.set_item("sp_ptr", PyArray1::from_slice(py, &sp_ptr))?;
    result.set_item("sp_sizes", PyArray1::from_slice(py, &sp_sizes))?;
    result.set_item("sp_batch", PyArray1::from_slice(py, &sp_batch))?;
    result.set_item("cu_sp", PyArray1::from_slice(py, &cu_sp))?;
    result.set_item("max_sp", max_sp)?;

    // Helper: write one collated task into result dict under a given prefix
    let write_task = |c: &CollatedTask, prefix: &str| -> PyResult<()> {
        result.set_item(format!("{}_gene_x", prefix), PyArray1::from_slice(py, &c.gene_x))?;
        result.set_item(format!("{}_gene_y", prefix), PyArray1::from_slice(py, &c.gene_y))?;
        result.set_item(format!("{}_event", prefix), PyArray1::from_slice(py, &c.event))?;
        result.set_item(format!("{}_event_true", prefix), PyArray1::from_slice(py, &c.event_true))?;
        result.set_item(format!("{}_is_leaf", prefix), PyArray1::from_slice(py, &c.is_leaf))?;
        result.set_item(format!("{}_frontier_mask", prefix), PyArray1::from_slice(py, &c.frontier_mask))?;
        result.set_item(format!("{}_mask_label_node", prefix), PyArray1::from_slice(py, &c.mask_label_node))?;
        result.set_item(format!("{}_gene_ei", prefix), edge_2d(&c.gene_ei_src, &c.gene_ei_dst, &format!("{}_gene_ei", prefix))?)?;
        result.set_item(format!("{}_gene_ew", prefix), PyArray1::from_slice(py, &c.gene_ew))?;
        result.set_item(format!("{}_g_dir_edge", prefix), edge_2d(&c.g_dir_src, &c.g_dir_dst, &format!("{}_g_dir_edge", prefix))?)?;
        result.set_item(format!("{}_true_root_index", prefix), PyArray1::from_slice(py, &c.true_root_index))?;
        result.set_item(format!("{}_g_ptr", prefix), PyArray1::from_slice(py, &c.g_ptr))?;
        result.set_item(format!("{}_g_sizes", prefix), PyArray1::from_slice(py, &c.g_sizes))?;
        result.set_item(format!("{}_g_batch", prefix), PyArray1::from_slice(py, &c.g_batch))?;
        result.set_item(format!("{}_root_edges_ptr", prefix), PyArray1::from_slice(py, &c.root_edges_ptr))?;
        result.set_item(format!("{}_cu_g", prefix), PyArray1::from_slice(py, &c.cu_g))?;
        result.set_item(format!("{}_max_g", prefix), c.max_g)?;
        Ok(())
    };

    write_task(&map_c, "map")?;
    if let Some(ref c) = evt_c { write_task(c, "evt")?; }
    if let Some(ref c) = root_c { write_task(c, "root")?; }

    Ok(result.into())
}

/// Compute GCN normalization on edge indices.
///
/// Adds self-loops and computes D^{-1/2} A D^{-1/2} edge weights.
/// Matches PyG's `gcn_norm(edge_index, None, num_nodes, improved=False,
/// add_self_loops=True, flow="source_to_target")`.
///
/// Parameters
/// ----------
/// edge_index : numpy array of shape (2, E), int32
/// num_nodes : int
///
/// Returns (edge_index_with_self_loops, edge_weights) as numpy arrays.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (edge_index, num_nodes))]
fn compute_gcn_norm(py: Python, edge_index: &Bound<'_, numpy::PyArray2<i32>>, num_nodes: usize) -> PyResult<(PyObject, PyObject)> {
    use numpy::{PyArray1, PyArrayMethods};
    use numpy::ndarray::Array2;
    use numpy::ToPyArray;

    let ei = unsafe { edge_index.as_array() };
    let src: Vec<i32> = ei.row(0).to_vec();
    let dst: Vec<i32> = ei.row(1).to_vec();

    let (new_src, new_dst, weights) = gcn_norm_internal(&src, &dst, num_nodes);

    let n = new_src.len();
    let mut flat = Vec::with_capacity(2 * n);
    flat.extend_from_slice(&new_src);
    flat.extend_from_slice(&new_dst);
    let arr = Array2::from_shape_vec([2, n], flat)
        .map_err(|e| PyValueError::new_err(format!("compute_gcn_norm: {}", e)))?;

    let ei_out = arr.to_pyarray(py).into_any().unbind();
    let ew_out = PyArray1::from_slice(py, &weights).into_any().unbind();
    Ok((ei_out, ew_out))
}

/// Build a batched dict of numpy arrays for inference from a base_sample dict.
///
/// Replaces the Python get_sample() + collate() path. Builds task tensors,
/// collates n_copies, applies GCN norm and computes varlen metadata.
/// Returns a flat numpy dict in the same format as build_otf_batch().
///
/// Only mapping task tensors are returned (inference doesn't need separate evt/root batches).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (base_sample, n_copies, seed, sample_order))]
fn build_inference_batch(
    py: Python,
    base_sample: &Bound<'_, pyo3::types::PyDict>,
    n_copies: usize,
    seed: u64,
    sample_order: &str,
) -> PyResult<PyObject> {
    use numpy::PyArray1;
    use pyo3::types::PyDict;

    // Parse the base_sample dict into RustBaseSample
    let base = parse_base_sample_dict(base_sample)?;

    // Build species name → 1-based ID mapping
    let mut sp_name_to_id: HashMap<String, i32> = HashMap::new();
    for (i, name) in base.species_names.iter().enumerate() {
        sp_name_to_id.insert(name.clone(), (i + 1) as i32);
    }

    let n_sp_nodes = base.species_names.len();
    let sample_order_owned = sample_order.to_owned();

    let result_or_err = py.allow_threads(move || -> Result<_, String> {
        let sample_order: &str = &sample_order_owned;

        // Build species tensors (single copy)
        let x_sp_single: Vec<i32> = base.species_names.iter()
            .map(|name| sp_name_to_id[name]).collect();

        let mut sp_child_src_s: Vec<i32> = Vec::new();
        let mut sp_child_dst_s: Vec<i32> = Vec::new();
        let mut sp_parent_src_s: Vec<i32> = Vec::new();
        let mut sp_parent_dst_s: Vec<i32> = Vec::new();

        // Build species name to index mapping
        let mut sp_name_to_idx: HashMap<String, usize> = HashMap::new();
        for (i, name) in base.species_names.iter().enumerate() {
            sp_name_to_idx.insert(name.clone(), i);
        }

        for (parent_name, children) in &base.sp_children {
            if let Some(&pi) = sp_name_to_idx.get(parent_name) {
                for child_name in children {
                    if let Some(&ci) = sp_name_to_idx.get(child_name) {
                        sp_child_src_s.push(pi as i32);
                        sp_child_dst_s.push(ci as i32);
                        sp_parent_src_s.push(ci as i32);
                        sp_parent_dst_s.push(pi as i32);
                    }
                }
            }
        }

        // GCN norm on species edges (single copy)
        let (sp_child_norm_src_s, sp_child_norm_dst_s, sp_child_ew_s) =
            gcn_norm_internal(&sp_child_src_s, &sp_child_dst_s, n_sp_nodes);
        let (sp_parent_norm_src_s, sp_parent_norm_dst_s, sp_parent_ew_s) =
            gcn_norm_internal(&sp_parent_src_s, &sp_parent_dst_s, n_sp_nodes);

        // Build mapping task tensors for each copy
        let mut map_tensors: Vec<TaskTensors> = Vec::with_capacity(n_copies);
        for copy_idx in 0..n_copies {
            let copy_seed = seed.wrapping_add(copy_idx as u64 * 1000003);
            let mut rng = StdRng::seed_from_u64(copy_seed);
            let t = build_task_tensors_internal(&base, &sp_name_to_id, &mut rng, false, false, sample_order)?;
            map_tensors.push(t);
        }

        // Collate
        let map_c = collate_task_tensors(&map_tensors);

        // Replicate species B times
        let b = n_copies;
        let n_sp_child_e = sp_child_norm_src_s.len();
        let n_sp_parent_e = sp_parent_norm_src_s.len();
        let mut sp_x: Vec<i32> = Vec::with_capacity(b * n_sp_nodes);
        let mut sp_child_src: Vec<i32> = Vec::with_capacity(b * n_sp_child_e);
        let mut sp_child_dst: Vec<i32> = Vec::with_capacity(b * n_sp_child_e);
        let mut sp_child_ew: Vec<f32> = Vec::with_capacity(b * n_sp_child_e);
        let mut sp_parent_src: Vec<i32> = Vec::with_capacity(b * n_sp_parent_e);
        let mut sp_parent_dst: Vec<i32> = Vec::with_capacity(b * n_sp_parent_e);
        let mut sp_parent_ew: Vec<f32> = Vec::with_capacity(b * n_sp_parent_e);
        let mut sp_batch: Vec<i32> = Vec::with_capacity(b * n_sp_nodes);
        let mut sp_sizes: Vec<i32> = Vec::with_capacity(b);
        let mut sp_ptr: Vec<i32> = Vec::with_capacity(b + 1);
        sp_ptr.push(0);

        for s_idx in 0..b {
            let offset = (s_idx * n_sp_nodes) as i32;
            sp_x.extend_from_slice(&x_sp_single);
            for (&s, &d) in sp_child_norm_src_s.iter().zip(sp_child_norm_dst_s.iter()) {
                sp_child_src.push(s + offset); sp_child_dst.push(d + offset);
            }
            sp_child_ew.extend_from_slice(&sp_child_ew_s);
            for (&s, &d) in sp_parent_norm_src_s.iter().zip(sp_parent_norm_dst_s.iter()) {
                sp_parent_src.push(s + offset); sp_parent_dst.push(d + offset);
            }
            sp_parent_ew.extend_from_slice(&sp_parent_ew_s);
            for _ in 0..n_sp_nodes { sp_batch.push(s_idx as i32); }
            sp_sizes.push(n_sp_nodes as i32);
            sp_ptr.push((s_idx + 1) as i32 * n_sp_nodes as i32);
        }

        let (cu_sp, max_sp) = compute_varlen_metadata(&sp_sizes);

        Ok((sp_x, sp_child_src, sp_child_dst, sp_child_ew,
            sp_parent_src, sp_parent_dst, sp_parent_ew,
            sp_ptr, sp_sizes, sp_batch, cu_sp, max_sp, map_c))
    });

    let (sp_x, sp_child_src, sp_child_dst, sp_child_ew,
         sp_parent_src, sp_parent_dst, sp_parent_ew,
         sp_ptr, sp_sizes, sp_batch, cu_sp, max_sp, map_c) =
        result_or_err.map_err(|e| PyValueError::new_err(e))?;

    let result = PyDict::new(py);

    let edge_2d = |src: &[i32], dst: &[i32], name: &str| -> PyResult<PyObject> {
        use numpy::ndarray::Array2;
        use numpy::ToPyArray;
        let n = src.len();
        let mut flat = Vec::with_capacity(2 * n);
        flat.extend_from_slice(src);
        flat.extend_from_slice(dst);
        let arr = Array2::from_shape_vec([2, n], flat)
            .map_err(|e| PyValueError::new_err(format!("{}: {}", name, e)))?;
        Ok(arr.to_pyarray(py).into_any().unbind())
    };

    // Species
    result.set_item("sp_x", PyArray1::from_slice(py, &sp_x))?;
    result.set_item("sp_child_ei", edge_2d(&sp_child_src, &sp_child_dst, "sp_child_ei")?)?;
    result.set_item("sp_child_ew", PyArray1::from_slice(py, &sp_child_ew))?;
    result.set_item("sp_parent_ei", edge_2d(&sp_parent_src, &sp_parent_dst, "sp_parent_ei")?)?;
    result.set_item("sp_parent_ew", PyArray1::from_slice(py, &sp_parent_ew))?;
    result.set_item("sp_ptr", PyArray1::from_slice(py, &sp_ptr))?;
    result.set_item("sp_sizes", PyArray1::from_slice(py, &sp_sizes))?;
    result.set_item("sp_batch", PyArray1::from_slice(py, &sp_batch))?;
    result.set_item("cu_sp", PyArray1::from_slice(py, &cu_sp))?;
    result.set_item("max_sp", max_sp)?;

    // Mapping task (only task needed for inference)
    let c = &map_c;
    result.set_item("map_gene_x", PyArray1::from_slice(py, &c.gene_x))?;
    result.set_item("map_gene_y", PyArray1::from_slice(py, &c.gene_y))?;
    result.set_item("map_event", PyArray1::from_slice(py, &c.event))?;
    result.set_item("map_event_true", PyArray1::from_slice(py, &c.event_true))?;
    result.set_item("map_is_leaf", PyArray1::from_slice(py, &c.is_leaf))?;
    result.set_item("map_frontier_mask", PyArray1::from_slice(py, &c.frontier_mask))?;
    result.set_item("map_mask_label_node", PyArray1::from_slice(py, &c.mask_label_node))?;
    result.set_item("map_gene_ei", edge_2d(&c.gene_ei_src, &c.gene_ei_dst, "map_gene_ei")?)?;
    result.set_item("map_gene_ew", PyArray1::from_slice(py, &c.gene_ew))?;
    result.set_item("map_g_dir_edge", edge_2d(&c.g_dir_src, &c.g_dir_dst, "map_g_dir_edge")?)?;
    result.set_item("map_true_root_index", PyArray1::from_slice(py, &c.true_root_index))?;
    result.set_item("map_g_ptr", PyArray1::from_slice(py, &c.g_ptr))?;
    result.set_item("map_g_sizes", PyArray1::from_slice(py, &c.g_sizes))?;
    result.set_item("map_g_batch", PyArray1::from_slice(py, &c.g_batch))?;
    result.set_item("map_root_edges_ptr", PyArray1::from_slice(py, &c.root_edges_ptr))?;
    result.set_item("map_cu_g", PyArray1::from_slice(py, &c.cu_g))?;
    result.set_item("map_max_g", c.max_g)?;

    Ok(result.into())
}

/// Parse a Python base_sample dict into a RustBaseSample.
fn parse_base_sample_dict(d: &Bound<'_, pyo3::types::PyDict>) -> PyResult<RustBaseSample> {
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

/// Build a PyGeneTree from a species tree, gene tree, and per-node reconciliation annotations.
///
/// This allows constructing a fully annotated reconciled gene tree without going through
/// RecPhyloXML, e.g. from model predictions.
///
/// # Arguments
/// * `sp_newick` - Species tree in Newick format (string, not file path)
/// * `g_newick` - Gene tree in Newick format (string, not file path)
/// * `node_species` - Dict mapping gene node name → species node name
/// * `node_events` - Dict mapping gene node name → event code (0=Speciation, 1=Duplication, 2=Transfer, 3=Leaf)
///
/// # Returns
/// A PyGeneTree with the given reconciliation annotations.
#[pyfunction]
#[pyo3(signature = (sp_newick, g_newick, node_species, node_events))]
fn from_reconciliation(
    sp_newick: &str,
    g_newick: &str,
    node_species: &Bound<'_, pyo3::types::PyDict>,
    node_events: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<PyGeneTree> {
    use crate::newick::newick::parse_newick;
    use crate::node::rectree::{Event, RecTree};

    // Parse species tree
    let mut sp_nodes = parse_newick(sp_newick)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse species Newick: {}", e)))?;
    let sp_root = sp_nodes.pop()
        .ok_or_else(|| PyValueError::new_err("No tree found in species Newick"))?;
    let mut sp_tree = sp_root.to_flat_tree();
    sp_tree.assign_depths();

    // Build species name → index map
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

/// Python module for rustree.
#[pymodule]
fn rustree(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_species_tree, m)?)?;
    m.add_function(wrap_pyfunction!(parse_species_tree, m)?)?;
    m.add_function(wrap_pyfunction!(parse_recphyloxml, m)?)?;
    m.add_function(wrap_pyfunction!(reconcile_with_alerax, m)?)?;
    m.add_function(wrap_pyfunction!(create_training_sample, m)?)?;
    m.add_function(wrap_pyfunction!(create_training_sample_from_sim, m)?)?;
    m.add_function(wrap_pyfunction!(build_training_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(build_otf_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gcn_norm, m)?)?;
    m.add_function(wrap_pyfunction!(build_inference_batch, m)?)?;
    m.add_function(wrap_pyfunction!(from_reconciliation, m)?)?;
    m.add_class::<PySpeciesTree>()?;
    m.add_class::<PySpeciesNode>()?;
    m.add_class::<PySpeciesTreeIter>()?;
    m.add_class::<PyGeneTree>()?;
    m.add_class::<PyDtlSimIter>()?;
    m.add_class::<PyEventCounts>()?;
    m.add_class::<PyReconciliationStatistics>()?;
    m.add_class::<PyAleRaxResult>()?;
    m.add_class::<PyGeneForest>()?;
    m.add_class::<PyAleRaxForestResult>()?;
    m.add_class::<PyReconciliationComparison>()?;
    m.add_class::<PyMultiSampleComparison>()?;
    m.add_function(wrap_pyfunction!(compare_reconciliations, m)?)?;
    m.add_function(wrap_pyfunction!(compare_reconciliations_multi, m)?)?;
    Ok(())
}

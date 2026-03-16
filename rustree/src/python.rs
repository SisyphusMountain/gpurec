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
use crate::node::{FlatTree, Event, RecTree};
use crate::io::rectree_csv::RecTreeColumns;
use crate::sampling::{extract_induced_subtree, extract_induced_subtree_by_names, find_leaf_indices_by_names, extract_extant_subtree, NodeMark, mark_nodes_postorder};
use crate::simulation::dtl::gillespie::{DTLMode, simulate_dtl_gillespie};
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
        let dict = ltt_data.downcast::<pyo3::types::PyDict>(py)?;
        let times = dict.get_item("times")
            .ok_or_else(|| PyValueError::new_err("Missing 'times' in LTT data"))?;
        let lineages = dict.get_item("lineages")
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

        // Extract induced subtree with old→new index mapping
        let (sampled_tree, old_to_new) = extract_induced_subtree(&self.rec_tree.gene_tree, &extant_indices)
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
                    // Collapsed node — shouldn't happen since extract_induced_subtree
                    // only keeps nodes marked Keep
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
                    None // Internal node: no valid mapping after sampling
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
#[pyo3(signature = (species_tree, gene_trees, output_dir=None, num_samples=100, model="PER-FAMILY".to_string(), seed=None, keep_output=false, alerax_path="alerax".to_string()))]
fn reconcile_with_alerax(
    py: Python,
    species_tree: PyObject,
    gene_trees: PyObject,
    output_dir: Option<String>,
    num_samples: usize,
    model: String,
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

    // Parse gene trees
    let mut gene_tree_list: Vec<(String, String)> = Vec::new();

    // Try as dict first
    if let Ok(dict) = gene_trees.extract::<HashMap<String, String>>(py) {
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
            "gene_trees must be a Newick string, file path, list of strings/paths, or dict mapping names to strings/paths"
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
        seed,
        alerax_path,
    };

    // Run ALERax
    let results = run_alerax(config, &species_tree_newick)
        .map_err(|e| PyValueError::new_err(format!("ALERax execution failed: {}", e)))?;

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
    #[pyo3(signature = (output_dir=None, num_samples=100, model="PER-FAMILY".to_string(), seed=None, keep_output=false, alerax_path="alerax".to_string()))]
    fn reconcile_with_alerax(
        &self,
        output_dir: Option<String>,
        num_samples: usize,
        model: String,
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

/// Python module for rustree.
#[pymodule]
fn rustree(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_species_tree, m)?)?;
    m.add_function(wrap_pyfunction!(parse_species_tree, m)?)?;
    m.add_function(wrap_pyfunction!(parse_recphyloxml, m)?)?;
    m.add_function(wrap_pyfunction!(reconcile_with_alerax, m)?)?;
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
    Ok(())
}

#![allow(clippy::too_many_arguments)]
//! PySpeciesTree, PySpeciesNode, and PySpeciesTreeIter for Python bindings.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;
use std::fs;
use std::process::Command;

use crate::bd::generate_events_from_tree;
use crate::dtl::{simulate_dtl, simulate_dtl_batch, simulate_dtl_per_species, simulate_dtl_per_species_batch};
use crate::node::{FlatTree, RecTree};
use crate::sampling::{extract_induced_subtree_by_names, extract_extant_subtree, find_leaf_indices_by_names, NodeMark, mark_nodes_postorder};
use crate::simulation::dtl::gillespie::DTLMode;

use super::{validate_dtl_rates, validate_replacement_transfer, init_rng, is_leaf, parse_distance_type};
use super::gene_tree::PyGeneTree;
use super::forest::PyGeneForest;
use super::sim_iter::PyDtlSimIter;

/// A species tree simulated under the birth-death process.
#[pyclass]
#[derive(Clone)]
pub struct PySpeciesTree {
    pub(crate) tree: Arc<FlatTree>,
}

#[pymethods]
impl PySpeciesTree {
    fn __repr__(&self) -> String {
        let n_leaves = self.tree.nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .count();
        let height = self.tree.nodes.iter()
            .filter_map(|n| n.depth)
            .fold(0.0_f64, f64::max);
        format!("SpeciesTree(leaves={}, nodes={}, height={:.4})", n_leaves, self.tree.nodes.len(), height)
    }

    /// Convert the species tree to Newick format.
    fn to_newick(&self) -> PyResult<String> {
        let nwk = self.tree.to_newick()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
    fn sample_extant(&self) -> PyResult<PySpeciesTree> {
        let (extant_tree, _) = extract_extant_subtree(&self.tree)
            .ok_or_else(|| PyValueError::new_err(
                "No extant species found in tree. Either all lineages went extinct, \
                 or the tree lacks bd_event annotations (trees from parse_species_tree \
                 cannot be filtered by extinction status)."
            ))?;

        Ok(PySpeciesTree {
            tree: Arc::new(extant_tree),
        })
    }

    /// Save the species tree to a Newick file.
    fn save_newick(&self, filepath: &str) -> PyResult<()> {
        let newick = self.to_newick()?;
        fs::write(filepath, newick)
            .map_err(|e| PyValueError::new_err(format!("Failed to write Newick file: {}", e)))?;
        Ok(())
    }

    /// Generate an SVG visualization of the species tree using thirdkind.
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

        let ipython_display = super::import_pymodule(py, "IPython.display")?;
        let svg_class = ipython_display.getattr("SVG")?;
        let display_obj = svg_class.call1((svg,))?;

        Ok(display_obj.into())
    }

    /// Save birth-death events to a CSV file.
    #[pyo3(signature = (filepath, eps=None))]
    fn save_bd_events_csv(&self, filepath: &str, eps: Option<f64>) -> PyResult<()> {
        use crate::bd::{generate_events_from_tree, generate_events_with_extinction};
        use crate::io::save_bd_events_to_csv;

        let events = match eps {
            Some(eps) => generate_events_with_extinction(&self.tree, eps),
            None => generate_events_from_tree(&self.tree),
        }.map_err(PyValueError::new_err)?;

        save_bd_events_to_csv(&events, &self.tree, filepath)
            .map_err(|e| PyValueError::new_err(format!("Failed to write CSV file: {}", e)))?;

        Ok(())
    }

    /// Get birth-death events as a dictionary.
    #[pyo3(signature = (eps=None))]
    fn get_bd_events(&self, py: Python, eps: Option<f64>) -> PyResult<PyObject> {
        use crate::bd::{generate_events_from_tree, generate_events_with_extinction};
        use pyo3::types::PyDict;

        let events = match eps {
            Some(eps) => generate_events_with_extinction(&self.tree, eps),
            None => generate_events_from_tree(&self.tree),
        }.map_err(PyValueError::new_err)?;

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
    #[pyo3(signature = (eps=None))]
    fn get_ltt_data(&self, py: Python, eps: Option<f64>) -> PyResult<PyObject> {
        use crate::bd::{generate_events_from_tree, generate_events_with_extinction};
        use pyo3::types::PyDict;

        let mut events = match eps {
            Some(eps) => generate_events_with_extinction(&self.tree, eps),
            None => generate_events_from_tree(&self.tree),
        }.map_err(PyValueError::new_err)?;

        events.sort_by(|a, b| a.time.total_cmp(&b.time));

        let mut times = Vec::new();
        let mut lineages = Vec::new();

        let n_extant = self.tree.nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .count();

        let mut current_lineages = n_extant;
        times.push(0.0);
        lineages.push(current_lineages);

        for event in events.iter() {
            match event.event_type.as_str() {
                "Leaf" => {
                    continue;
                }
                "Extinction" => {
                    current_lineages += 1;
                }
                "Speciation" => {
                    current_lineages -= 1;
                }
                _ => continue,
            }
            times.push(event.time);
            lineages.push(current_lineages);
        }

        let dict = PyDict::new(py);
        dict.set_item("times", times)?;
        dict.set_item("lineages", lineages)?;

        Ok(dict.into())
    }

    /// Plot Lineages Through Time (LTT) using matplotlib.
    #[pyo3(signature = (filepath=None, title=None, xlabel=None, ylabel=None))]
    fn plot_ltt(
        &self,
        py: Python,
        filepath: Option<&str>,
        title: Option<&str>,
        xlabel: Option<&str>,
        ylabel: Option<&str>,
    ) -> PyResult<()> {
        let ltt_data = self.get_ltt_data(py, None)?;

        let plt = super::import_pymodule(py, "matplotlib.pyplot")?;

        let dict = ltt_data.downcast_bound::<pyo3::types::PyDict>(py)?;
        let times = dict.get_item("times")?
            .ok_or_else(|| PyValueError::new_err("Missing 'times' in LTT data"))?;
        let lineages = dict.get_item("lineages")?
            .ok_or_else(|| PyValueError::new_err("Missing 'lineages' in LTT data"))?;

        plt.call_method1("plot", (times, lineages))?;
        plt.call_method1("xlabel", (xlabel.unwrap_or("Time (past to present)"),))?;
        plt.call_method1("ylabel", (ylabel.unwrap_or("Number of lineages)"),))?;
        plt.call_method1("title", (title.unwrap_or("Lineages Through Time"),))?;
        plt.call_method0("grid")?;

        if let Some(path) = filepath {
            plt.call_method1("savefig", (path,))?;
        } else {
            plt.call_method0("show")?;
        }

        Ok(())
    }

    /// Simulate a gene tree along this species tree using the DTL model.
    #[pyo3(signature = (lambda_d, lambda_t, lambda_l, transfer_alpha=None, replacement_transfer=None, require_extant=false, seed=None))]
    fn simulate_dtl(&self, lambda_d: f64, lambda_t: f64, lambda_l: f64, transfer_alpha: Option<f64>, replacement_transfer: Option<f64>, require_extant: bool, seed: Option<u64>) -> PyResult<PyGeneTree> {
        validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;
        validate_replacement_transfer(replacement_transfer)?;
        let mut rng = init_rng(seed);
        let origin_species = self.tree.root;
        let (mut rec_tree, events) = simulate_dtl(&self.tree, origin_species, lambda_d, lambda_t, lambda_l, transfer_alpha, replacement_transfer, require_extant, &mut rng)
            .map_err(PyValueError::new_err)?;

        rec_tree.species_tree = Arc::clone(&self.tree);
        rec_tree.dtl_events = Some(events);
        Ok(PyGeneTree { rec_tree })
    }

    /// Simulate multiple gene trees efficiently with shared pre-computed data.
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
        ).map_err(PyValueError::new_err)?;

        let gene_trees: Vec<RecTree> = rec_trees
            .into_iter()
            .zip(all_events)
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
    #[pyo3(signature = (lambda_d, lambda_t, lambda_l, transfer_alpha=None, replacement_transfer=None, require_extant=false, seed=None))]
    fn simulate_dtl_per_species(&self, lambda_d: f64, lambda_t: f64, lambda_l: f64, transfer_alpha: Option<f64>, replacement_transfer: Option<f64>, require_extant: bool, seed: Option<u64>) -> PyResult<PyGeneTree> {
        validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;
        validate_replacement_transfer(replacement_transfer)?;
        let mut rng = init_rng(seed);
        let origin_species = self.tree.root;
        let (mut rec_tree, events) = simulate_dtl_per_species(&self.tree, origin_species, lambda_d, lambda_t, lambda_l, transfer_alpha, replacement_transfer, require_extant, &mut rng)
            .map_err(PyValueError::new_err)?;

        rec_tree.species_tree = Arc::clone(&self.tree);
        rec_tree.dtl_events = Some(events);
        Ok(PyGeneTree { rec_tree })
    }

    /// Simulate multiple gene trees using the Zombi-style per-species DTL model.
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
        ).map_err(PyValueError::new_err)?;

        let gene_trees: Vec<RecTree> = rec_trees
            .into_iter()
            .zip(all_events)
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
            .map_err(PyValueError::new_err)?;
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
            config: crate::dtl::DTLConfig { lambda_d, lambda_t, lambda_l, transfer_alpha, replacement_transfer },
            n_simulations: n,
            require_extant,
            mode: DTLMode::PerGene,
            rng,
            completed: 0,
        })
    }

    /// Create a lazy iterator that generates gene trees one at a time (per-species model).
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
            .map_err(PyValueError::new_err)?;
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
            config: crate::dtl::DTLConfig { lambda_d, lambda_t, lambda_l, transfer_alpha, replacement_transfer },
            n_simulations: n,
            require_extant,
            mode: DTLMode::PerSpecies,
            rng,
            completed: 0,
        })
    }

    /// Compute all pairwise distances between nodes in the species tree.
    #[pyo3(signature = (distance_type, leaves_only=true))]
    fn pairwise_distances(&self, py: Python, distance_type: &str, leaves_only: bool) -> PyResult<PyObject> {
        let dist_type = parse_distance_type(distance_type)?;

        let distances = self.tree.pairwise_distances(dist_type, leaves_only)
            .map_err(|e| PyValueError::new_err(format!("Failed to compute pairwise distances: {}", e)))?;

        let node1: Vec<&str> = distances.iter().map(|d| d.node1).collect();
        let node2: Vec<&str> = distances.iter().map(|d| d.node2).collect();
        let dist: Vec<f64> = distances.iter().map(|d| d.distance).collect();

        let pandas = super::import_pymodule(py, "pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("node1", node1)?;
        dict.set_item("node2", node2)?;
        dict.set_item("distance", dist)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Save pairwise distances between nodes to a CSV file.
    #[pyo3(signature = (filepath, distance_type, leaves_only=true))]
    fn save_pairwise_distances_csv(&self, filepath: &str, distance_type: &str, leaves_only: bool) -> PyResult<()> {
        use crate::metric_functions::{DistanceType, PairwiseDistance};

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
    fn compute_ghost_lengths(&self, py: Python, sampled_leaf_names: Vec<String>) -> PyResult<PyObject> {
        use crate::induced_transfers::ghost_lengths;
        let (sampled_tree, ghosts) = ghost_lengths(
            &self.tree,
            &sampled_leaf_names,
        ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let node_index: Vec<usize> = (0..sampled_tree.nodes.len()).collect();
        let node_name: Vec<&str> = sampled_tree.nodes.iter().map(|n| n.name.as_str()).collect();

        let pandas = super::import_pymodule(py, "pandas")?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("node_index", node_index)?;
        dict.set_item("node_name", node_name)?;
        dict.set_item("ghost_length", ghosts)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Compute the Robinson-Foulds distance to another species tree.
    fn robinson_foulds(&self, other: &PySpeciesTree) -> PyResult<usize> {
        let rf = std::panic::catch_unwind(|| {
            crate::robinson_foulds::unrooted_robinson_foulds(&self.tree, &other.tree)
        }).map_err(|e| {
            let msg = e.downcast_ref::<String>().map(|s| s.as_str())
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("Robinson-Foulds computation failed");
            PyValueError::new_err(msg.to_string())
        })?;
        Ok(rf)
    }

    /// Compute the true unrooted Robinson-Foulds distance to another species tree.
    fn unrooted_robinson_foulds(&self, other: &PySpeciesTree) -> PyResult<usize> {
        let rf = std::panic::catch_unwind(|| {
            crate::robinson_foulds::true_unrooted_robinson_foulds(&self.tree, &other.tree)
        }).map_err(|e| {
            let msg = e.downcast_ref::<String>().map(|s| s.as_str())
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("Robinson-Foulds computation failed");
            PyValueError::new_err(msg.to_string())
        })?;
        Ok(rf)
    }

    /// Find the index of a node by its name.
    fn find_node_index(&self, name: &str) -> PyResult<usize> {
        self.tree.nodes.iter()
            .position(|n| n.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("Node '{}' not found", name)))
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
pub struct PySpeciesTreeIter {
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

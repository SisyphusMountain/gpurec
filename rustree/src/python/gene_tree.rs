//! PyGeneTree and related types for Python bindings.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::sync::Arc;
use std::fs;
use std::process::Command;

use crate::node::{Event, RecTree, remap_gene_tree_indices};
use crate::io::rectree_csv::RecTreeColumns;
use crate::sampling::{extract_induced_subtree, extract_extant_subtree, find_leaf_indices_by_names, NodeMark, mark_nodes_postorder};

use super::{extract_extant_gene_tree, parse_distance_type};
use super::reconciliation::{PyReconciliationComparison, PyMultiSampleComparison};

/// Convert RecTreeColumns to a pandas DataFrame.
pub(crate) fn columns_to_dataframe(py: Python, cols: &RecTreeColumns) -> PyResult<PyObject> {
    let pandas = super::import_pymodule(py, "pandas")?;
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
    pub(crate) rec_tree: RecTree,
}

#[pymethods]
impl PyGeneTree {
    fn __repr__(&self) -> String {
        let n_leaves = self.rec_tree.gene_tree.nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .count();
        let s = self.rec_tree.event_mapping.iter().filter(|e| **e == Event::Speciation).count();
        let d = self.rec_tree.event_mapping.iter().filter(|e| **e == Event::Duplication).count();
        let t = self.rec_tree.event_mapping.iter().filter(|e| **e == Event::Transfer).count();
        let l = self.rec_tree.event_mapping.iter().filter(|e| **e == Event::Loss).count();
        format!("GeneTree(leaves={}, events={{S:{}, D:{}, T:{}, L:{}}})", n_leaves, s, d, t, l)
    }

    /// Convert the gene tree to Newick format.
    fn to_newick(&self) -> PyResult<String> {
        let nwk = self.rec_tree.gene_tree.to_newick()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
        ).map_err(PyValueError::new_err)?;

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

        let (sampled_tree, gene_old_to_new) = extract_induced_subtree(&self.rec_tree.gene_tree, &keep_indices)
            .ok_or_else(|| PyValueError::new_err("Failed to extract induced subtree"))?;

        // Species tree is unchanged, so build an identity species mapping
        let species_identity: Vec<Option<usize>> = (0..self.rec_tree.species_tree.nodes.len())
            .map(Some)
            .collect();

        let (new_node_mapping, new_event_mapping) = remap_gene_tree_indices(
            &sampled_tree, &gene_old_to_new,
            &self.rec_tree.node_mapping, &self.rec_tree.event_mapping,
            &species_identity,
        ).map_err(PyValueError::new_err)?;

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

        let (sampled_tree, gene_old_to_new) = extract_induced_subtree(&self.rec_tree.gene_tree, &keep_indices)
            .ok_or_else(|| PyValueError::new_err("Failed to extract induced subtree"))?;

        // Species tree is unchanged, so build an identity species mapping
        let species_identity: Vec<Option<usize>> = (0..self.rec_tree.species_tree.nodes.len())
            .map(Some)
            .collect();

        let (new_node_mapping, new_event_mapping) = remap_gene_tree_indices(
            &sampled_tree, &gene_old_to_new,
            &self.rec_tree.node_mapping, &self.rec_tree.event_mapping,
            &species_identity,
        ).map_err(PyValueError::new_err)?;

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
                let new_attr = format!("class=\"species\" style=\"fill: {}\"", color);
                let search = "class=\"species\"";
                let name_pattern = format!("\n{}\n</text>", node.name);
                if let Some(pos) = svg.find(&name_pattern) {
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
                let species_name = match self.rec_tree.node_mapping[gene_idx] {
                    Some(sp_idx) => &species_tree.nodes[sp_idx].name,
                    None => continue,
                };
                let color = match species_color_map.get(species_name) {
                    Some(c) => *c,
                    None => continue,
                };
                let name_pattern = format!("\n{}\n</text>", gene_node.name);
                if let Some(pos) = svg.find(&name_pattern) {
                    if let Some(tag_start) = svg[..pos].rfind("<text") {
                        let tag_content = &svg[tag_start..pos];
                        if tag_content.contains("class=\"gene") || tag_content.contains("class=\"node_") {
                            if let Some(class_rel) = tag_content.find("class=\"") {
                                let class_abs = tag_start + class_rel;
                                let class_val_start = class_abs + 7;
                                if let Some(class_val_end) = svg[class_val_start..].find('"') {
                                    let old_class = &svg[class_val_start..class_val_start + class_val_end];
                                    let new_attr = format!("class=\"{}\" style=\"fill: {}\"", old_class, color);
                                    let replace_start = class_abs;
                                    let replace_end = class_val_start + class_val_end + 1;
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

        let ipython_display = super::import_pymodule(py, "IPython.display")?;
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
    #[pyo3(signature = (distance_type, leaves_only=true))]
    fn pairwise_distances(&self, py: Python, distance_type: &str, leaves_only: bool) -> PyResult<PyObject> {
        let dist_type = parse_distance_type(distance_type)?;

        let distances = self.rec_tree.gene_tree.pairwise_distances(dist_type, leaves_only)
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

    /// Save pairwise distances between nodes in the gene tree to a CSV file.
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

        let distances = self.rec_tree.gene_tree.pairwise_distances(dist_type, leaves_only)
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

    /// Compute induced transfers by projecting transfers onto a sampled species tree.
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
        ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let time: Vec<f64> = induced.iter().map(|t| t.time).collect();
        let gene_id: Vec<usize> = induced.iter().map(|t| t.gene_id).collect();
        let from_complete: Vec<usize> = induced.iter().map(|t| t.from_species_complete).collect();
        let to_complete: Vec<usize> = induced.iter().map(|t| t.to_species_complete).collect();
        let from_sampled: Vec<Option<usize>> = induced.iter().map(|t| t.from_species_sampled).collect();
        let to_sampled: Vec<Option<usize>> = induced.iter().map(|t| t.to_species_sampled).collect();

        let pandas = super::import_pymodule(py, "pandas")?;
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
    fn compare_reconciliation(&self, other: &PyGeneTree) -> PyResult<PyReconciliationComparison> {
        let result = crate::comparison::compare_reconciliations(&self.rec_tree, &other.rec_tree)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyReconciliationComparison {
            inner: result,
            trees: Some((self.rec_tree.clone(), other.rec_tree.clone())),
        })
    }

    /// Compare this reconciliation (truth) against multiple inferred samples.
    fn compare_reconciliation_multi(&self, samples: Vec<PyGeneTree>) -> PyResult<PyMultiSampleComparison> {
        let sample_recs: Vec<_> = samples.iter().map(|s| s.rec_tree.clone()).collect();
        let result = crate::comparison::compare_reconciliations_multi(&self.rec_tree, &sample_recs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyMultiSampleComparison { inner: result })
    }

    /// Compute Robinson-Foulds distance to another gene tree.
    #[pyo3(signature = (other, rooted=false))]
    fn rf_distance(&self, other: &PyGeneTree, rooted: bool) -> PyResult<usize> {
        let tree1 = extract_extant_gene_tree(&self.rec_tree)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let tree2 = extract_extant_gene_tree(&other.rec_tree)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if rooted {
            Ok(crate::robinson_foulds::unrooted_robinson_foulds(&tree1, &tree2))
        } else {
            Ok(crate::robinson_foulds::true_unrooted_robinson_foulds(&tree1, &tree2))
        }
    }

    /// Find the index of a gene node by its name.
    fn find_node_index(&self, name: &str) -> PyResult<usize> {
        self.rec_tree.gene_tree.nodes.iter()
            .position(|n| n.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("Gene node '{}' not found", name)))
    }
}

/// An induced transfer: a transfer event projected onto a sampled species tree.
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

//! ALERax reconciliation Python bindings.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use std::sync::Arc;

use super::{PyGeneTree, PySpeciesTree, PyEventCounts, PyReconciliationStatistics, parse_species_tree};
use super::forest::PyGeneForest;

/// Result of ALERax reconciliation for a gene family.
///
/// Contains all reconciliation samples, estimated evolutionary rates,
/// log-likelihood, and summary statistics.
#[pyclass]
#[derive(Clone)]
pub struct PyAleRaxResult {
    /// All reconciliation samples (typically 100)
    #[pyo3(get)]
    pub(crate) gene_trees: Vec<PyGeneTree>,

    /// Estimated duplication rate
    #[pyo3(get)]
    pub(crate) duplication_rate: f64,

    /// Estimated loss rate
    #[pyo3(get)]
    pub(crate) loss_rate: f64,

    /// Estimated transfer rate
    #[pyo3(get)]
    pub(crate) transfer_rate: f64,

    /// Log-likelihood of the reconciliation
    #[pyo3(get)]
    pub(crate) likelihood: f64,

    /// Summary statistics across all reconciliation samples
    #[pyo3(get)]
    pub(crate) statistics: PyReconciliationStatistics,
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
#[allow(clippy::too_many_arguments)]
pub fn reconcile_with_alerax(
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
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

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

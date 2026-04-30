#![allow(clippy::too_many_arguments)]
//! Reconciliation comparison Python bindings.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::fs;
use std::process::Command;

use crate::node::{Event, RecTree};

use super::PyGeneTree;

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
    pub(crate) inner: crate::comparison::ReconciliationComparison,
    /// Truth and inferred RecTrees, stored for display(). None for sub-comparisons
    /// extracted from MultiSampleComparison (consensus, per-sample).
    pub(crate) trees: Option<(RecTree, RecTree)>,
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
        let conf_lines = [format!("species_color:{}", species_color),
            format!("species_police_size:{}", species_fontsize)];
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
        let pandas = super::import_pymodule(py, "pandas")?;
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
        let pandas = super::import_pymodule(py, "pandas")?;
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

        let ipython_display = super::import_pymodule(py, "IPython.display")?;
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
    pub(crate) inner: crate::comparison::MultiSampleComparison,
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
pub fn compare_reconciliations(truth: &PyGeneTree, inferred: &PyGeneTree) -> PyResult<PyReconciliationComparison> {
    let result = crate::comparison::compare_reconciliations(&truth.rec_tree, &inferred.rec_tree)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
pub fn compare_reconciliations_multi(truth: &PyGeneTree, samples: Vec<PyGeneTree>) -> PyResult<PyMultiSampleComparison> {
    let sample_recs: Vec<_> = samples.iter().map(|s| s.rec_tree.clone()).collect();
    let result = crate::comparison::compare_reconciliations_multi(&truth.rec_tree, &sample_recs)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyMultiSampleComparison { inner: result })
}

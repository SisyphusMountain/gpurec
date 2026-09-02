//! Binary (numpy) bridge between the Rust preprocessor and Python.
//!
//! The original `preprocess_dataset` pyfunction serialises everything -- including the
//! multi-gigabyte per-family split arrays -- to one JSON string that Python then re-parses
//! and walks element by element. This module keeps the parsed families resident in Rust
//! (`ParsedFamilies`) so a fit parses each `.ale` file exactly once, and hands numeric arrays
//! to Python as `numpy` arrays instead of JSON text.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

use crate::{
    build_species_output, parse_one_newick_file, plan_batches_and_layouts,
    preprocess_one_family, PreprocessError, JSON_SCHEMA_VERSION,
};

/// Payload keys whose arrays MUST reach Python as float64 even when every element happens to
/// be integral. Both are consumed as floating point downstream: `log_split_probs_sorted`
/// feeds `_materialize_split_probabilities` (log2 split priors) and `unnorm_row_max` feeds the
/// transfer softmax row maximum. Anything else in the payload is an index or a count.
const FORCE_F64_KEYS: [&str; 2] = ["log_split_probs_sorted", "unnorm_row_max"];

/// Recursively convert a `serde_json::Value` into Python objects, turning homogeneous numeric
/// arrays into 1-D numpy arrays (int64 or float64) instead of Python lists.
///
/// Mixed arrays, arrays of objects/arrays/strings, and arrays holding a JSON `null` (which is
/// how a non-finite f64 is encoded) fall back to a Python list of converted items, so the
/// result is element-for-element what `json.loads` produced before.
pub fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    value_to_py_inner(py, value, false)
}

fn value_to_py_inner(py: Python<'_>, value: &Value, force_f64: bool) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(flag) => Ok(flag.into_py(py)),
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                Ok(int.into_py(py))
            } else if let Some(uint) = number.as_u64() {
                Ok(uint.into_py(py))
            } else {
                Ok(number.as_f64().unwrap_or(f64::NAN).into_py(py))
            }
        }
        Value::String(text) => Ok(text.into_py(py)),
        Value::Array(items) => array_to_py(py, items, force_f64),
        Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (key, item) in map {
                let force = FORCE_F64_KEYS.contains(&key.as_str());
                dict.set_item(key.as_str(), value_to_py_inner(py, item, force)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

fn array_to_py(py: Python<'_>, items: &[Value], force_f64: bool) -> PyResult<PyObject> {
    if items.is_empty() {
        return Ok(if force_f64 {
            PyArray1::<f64>::from_vec_bound(py, Vec::new())
                .into_any()
                .unbind()
        } else {
            PyArray1::<i64>::from_vec_bound(py, Vec::new())
                .into_any()
                .unbind()
        });
    }
    if items.iter().all(Value::is_number) {
        if !force_f64 && items.iter().all(|item| item.as_i64().is_some()) {
            let data = items
                .iter()
                .map(|item| item.as_i64().unwrap_or_default())
                .collect::<Vec<i64>>();
            return Ok(PyArray1::from_vec_bound(py, data).into_any().unbind());
        }
        if items.iter().all(|item| item.as_f64().is_some()) {
            let data = items
                .iter()
                .map(|item| item.as_f64().unwrap_or(f64::NAN))
                .collect::<Vec<f64>>();
            return Ok(PyArray1::from_vec_bound(py, data).into_any().unbind());
        }
    }
    let list = PyList::empty_bound(py);
    for item in items {
        list.append(value_to_py_inner(py, item, force_f64)?)?;
    }
    Ok(list.into_any().unbind())
}

/// A parsed dataset held resident in Rust: the species tree output plus one `Value` per gene
/// family, in the order the family paths were given. Re-planning a subset never re-reads a file.
#[pyclass]
pub struct ParsedFamilies {
    species: Value,
    species_name_to_index: BTreeMap<String, usize>,
    families: Vec<Value>,
}

#[pymethods]
impl ParsedFamilies {
    /// Parse the species tree and every gene family once, in parallel (same code path as
    /// `preprocess_dataset`).
    #[staticmethod]
    fn parse(py: Python<'_>, species_path: String, family_paths: Vec<String>) -> PyResult<Self> {
        let parsed = py.allow_threads(|| -> Result<Self, PreprocessError> {
            let species_tree = parse_one_newick_file(Path::new(&species_path))?;
            let (species, species_name_to_index) = build_species_output(&species_tree);
            let families = family_paths
                .par_iter()
                .map(|gene_path| {
                    preprocess_one_family(Path::new(gene_path), &species_name_to_index)
                        .map_err(PreprocessError::InvalidInput)
                })
                .collect::<Result<Vec<Value>, PreprocessError>>()?;
            Ok(Self {
                species,
                species_name_to_index,
                families,
            })
        })?;
        Ok(parsed)
    }

    fn n_families(&self) -> usize {
        self.families.len()
    }

    /// The species-tree payload (`S`, `sp_child1`, ... ) with numeric arrays as numpy arrays.
    fn species(&self, py: Python<'_>) -> PyResult<PyObject> {
        value_to_py(py, &self.species)
    }

    /// `{species_name: post-order index}`, the map used to resolve gene leaves onto species.
    fn species_name_to_index(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (name, index) in &self.species_name_to_index {
            dict.set_item(name.as_str(), *index)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// The selected family payloads, in the order of `indices`, numeric arrays as numpy arrays.
    fn families(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<PyObject> {
        let list = PyList::empty_bound(py);
        for index in &indices {
            list.append(value_to_py(py, self.family_at(*index)?)?)?;
        }
        Ok(list.into_any().unbind())
    }

    /// Plan batches + per-batch wave layouts over the selected subset, without touching any file.
    ///
    /// The returned `batches` hold positions into `indices` (not the original dataset indices),
    /// matching `preprocess_dataset`, whose batches index into the families it returned.
    #[pyo3(signature = (indices, family_chunk_size, clade_budget, batch_packing, max_wave_size, family_group_assignments))]
    fn plan(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
        family_chunk_size: Option<usize>,
        clade_budget: Option<usize>,
        batch_packing: Option<String>,
        max_wave_size: usize,
        family_group_assignments: Option<Vec<i64>>,
    ) -> PyResult<PyObject> {
        let mut selected = Vec::with_capacity(indices.len());
        for index in &indices {
            selected.push(self.family_at(*index)?);
        }
        let (batches, batch_wave_layouts) = py.allow_threads(|| {
            plan_batches_and_layouts(
                &selected,
                family_chunk_size,
                clade_budget,
                batch_packing.as_deref().unwrap_or("depth_first_fit"),
                max_wave_size,
                family_group_assignments.as_deref(),
            )
            .map_err(PreprocessError::InvalidInput)
        })?;

        let batch_list = PyList::empty_bound(py);
        for batch in &batches {
            let entries = batch.iter().map(|index| *index as i64).collect::<Vec<i64>>();
            batch_list.append(PyArray1::from_vec_bound(py, entries))?;
        }
        let layout_list = PyList::empty_bound(py);
        for layout in &batch_wave_layouts {
            layout_list.append(value_to_py(py, layout)?)?;
        }
        let out = PyDict::new_bound(py);
        out.set_item("schema_version", JSON_SCHEMA_VERSION)?;
        out.set_item("batches", batch_list)?;
        out.set_item("batch_wave_layouts", layout_list)?;
        Ok(out.into_any().unbind())
    }
}

impl ParsedFamilies {
    fn family_at(&self, index: usize) -> PyResult<&Value> {
        self.families.get(index).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "family index {index} out of range for {} parsed families",
                self.families.len()
            ))
        })
    }
}

//! Rust preprocessing prototype for GPUREC.
//!
//! This crate mirrors the C++ `preprocess_multiple_families` data path closely:
//! species nodes are indexed in postorder, gene-family clades use sorted global
//! leaf labels, split weights are accumulated across tree samples, and CCP split
//! arrays are emitted in the same parent-ranked order as the current pybind.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub mod batch_planning;
pub mod scheduler;
pub use batch_planning::{plan_family_batches_request, BatchPlanRequest, FamilyBatchPlanOutput};
pub use scheduler::{
    family_schedule_summary, schedule_global_phased_waves_request, FamilyScheduleSummary,
    ScheduleCcp, ScheduleOutput, ScheduleRequest,
};

#[cfg(feature = "python-extension")]
use numpy::{ndarray::Array2, IntoPyArray};
#[cfg(feature = "python-extension")]
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBytes, PyDict, PyList, PyString},
};

const BITS_PER_WORD: usize = 64;
const BINARY_MAGIC: &[u8; 8] = b"GPREP001";
static THREAD_POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();

#[derive(Debug, thiserror::Error)]
pub enum PreprocessError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to build preprocessing thread pool: {0}")]
    ThreadPoolBuild(#[from] rayon::ThreadPoolBuildError),
    #[error(transparent)]
    Rustree(#[from] rustree::RustreeError),
}

#[derive(Clone, Debug, Deserialize)]
pub struct PreprocessRequest {
    pub species_path: PathBuf,
    pub families: BTreeMap<String, Vec<PathBuf>>,
    #[serde(default)]
    pub leaf_species_maps: BTreeMap<String, BTreeMap<String, String>>,
    #[serde(default = "default_include_species_matrices")]
    pub include_species_matrices: bool,
    #[serde(default)]
    pub num_threads: usize,
}

fn default_include_species_matrices() -> bool {
    true
}

#[derive(Clone, Debug, Serialize)]
pub struct PreprocessOutput {
    pub species: SpeciesOutput,
    pub families: BTreeMap<String, FamilyOutput>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpeciesOutput {
    pub s: usize,
    pub names: Vec<String>,
    pub s_p_indexes: Vec<i64>,
    pub s_c12_indexes: Vec<i64>,
    pub unnorm_row_max: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ancestors_dense: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipients_mat: Option<Vec<f64>>,
    pub species_name_to_index: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FamilyOutput {
    pub ccp: CcpOutput,
    pub root_clade_id: i64,
    pub leaf_row_index: Vec<i64>,
    pub leaf_col_index: Vec<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CcpOutput {
    pub split_counts: Vec<i64>,
    pub split_parents_sorted: Vec<i64>,
    pub split_leftrights_sorted: Vec<i64>,
    pub log_split_probs_sorted: Vec<f64>,
    pub num_segs_ge2: i64,
    pub num_segs_eq1: i64,
    pub end_rows_ge2: i64,
    pub c: i64,
    pub n_splits: i64,
    pub root_clade_id: i64,
    pub clade_leaf_labels: Vec<String>,
}

pub fn write_binary_output<W: Write>(output: &PreprocessOutput, mut writer: W) -> io::Result<()> {
    writer.write_all(BINARY_MAGIC)?;
    write_species_binary(&output.species, &mut writer)?;
    write_u64(&mut writer, output.families.len() as u64)?;
    for (name, family) in &output.families {
        write_string(&mut writer, name)?;
        write_family_binary(family, &mut writer)?;
    }
    Ok(())
}

fn write_species_binary<W: Write>(species: &SpeciesOutput, writer: &mut W) -> io::Result<()> {
    write_u64(writer, species.s as u64)?;
    write_strings(writer, &species.names)?;
    write_i64s(writer, &species.s_p_indexes)?;
    write_i64s(writer, &species.s_c12_indexes)?;
    write_f64s(writer, &species.unnorm_row_max)?;
    write_optional_f64s(writer, species.ancestors_dense.as_deref())?;
    write_optional_f64s(writer, species.recipients_mat.as_deref())?;
    write_u64(writer, species.species_name_to_index.len() as u64)?;
    for (name, index) in &species.species_name_to_index {
        write_string(writer, name)?;
        write_u64(writer, *index as u64)?;
    }
    Ok(())
}

fn write_family_binary<W: Write>(family: &FamilyOutput, writer: &mut W) -> io::Result<()> {
    let ccp = &family.ccp;
    write_i64s(writer, &ccp.split_counts)?;
    write_i64s(writer, &ccp.split_parents_sorted)?;
    write_i64s(writer, &ccp.split_leftrights_sorted)?;
    write_f64s(writer, &ccp.log_split_probs_sorted)?;
    write_i64(writer, ccp.num_segs_ge2)?;
    write_i64(writer, ccp.num_segs_eq1)?;
    write_i64(writer, ccp.end_rows_ge2)?;
    write_i64(writer, ccp.c)?;
    write_i64(writer, ccp.n_splits)?;
    write_i64(writer, ccp.root_clade_id)?;
    write_sparse_strings(writer, &ccp.clade_leaf_labels)?;
    write_i64(writer, family.root_clade_id)?;
    write_i64s(writer, &family.leaf_row_index)?;
    write_i64s(writer, &family.leaf_col_index)?;
    Ok(())
}

fn write_optional_f64s<W: Write>(writer: &mut W, values: Option<&[f64]>) -> io::Result<()> {
    match values {
        Some(values) => {
            writer.write_all(&[1])?;
            write_f64s(writer, values)
        }
        None => writer.write_all(&[0]),
    }
}

fn write_strings<W: Write>(writer: &mut W, values: &[String]) -> io::Result<()> {
    write_u64(writer, values.len() as u64)?;
    for value in values {
        write_string(writer, value)?;
    }
    Ok(())
}

fn write_sparse_strings<W: Write>(writer: &mut W, values: &[String]) -> io::Result<()> {
    write_u64(writer, values.len() as u64)?;
    write_u64(
        writer,
        values.iter().filter(|value| !value.is_empty()).count() as u64,
    )?;
    for (idx, value) in values.iter().enumerate() {
        if value.is_empty() {
            continue;
        }
        write_u64(writer, idx as u64)?;
        write_string(writer, value)?;
    }
    Ok(())
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> io::Result<()> {
    let bytes = value.as_bytes();
    write_u64(writer, bytes.len() as u64)?;
    writer.write_all(bytes)
}

fn write_i64s<W: Write>(writer: &mut W, values: &[i64]) -> io::Result<()> {
    write_u64(writer, values.len() as u64)?;
    for value in values {
        write_i64(writer, *value)?;
    }
    Ok(())
}

fn write_f64s<W: Write>(writer: &mut W, values: &[f64]) -> io::Result<()> {
    write_u64(writer, values.len() as u64)?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_i64<W: Write>(writer: &mut W, value: i64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[derive(Clone, Debug)]
struct SpeciesTopology {
    names: Vec<String>,
    children: Vec<Vec<usize>>,
    parent: Vec<Option<usize>>,
}

#[derive(Clone, Debug)]
struct GeneTree {
    nodes: Vec<GeneNode>,
    root: usize,
}

#[derive(Clone, Debug)]
struct GeneNode {
    name: String,
    children: Vec<usize>,
    parent: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Clade {
    bits: Vec<u64>,
    size: usize,
}

#[derive(Clone, Debug)]
struct CladeRegistry {
    clades: Vec<Clade>,
    ids_by_bits: HashMap<Vec<u64>, usize>,
}

#[derive(Clone, Debug)]
struct CladeSplit {
    parent: usize,
    left: usize,
    right: usize,
    weight: f64,
}

#[derive(Clone, Debug)]
struct CladeData {
    clades: CladeRegistry,
    splits: Vec<CladeSplit>,
    root_clade_id: usize,
}

#[derive(Clone, Debug)]
struct CcpArrays {
    split_counts: Vec<i64>,
    split_parents_sorted: Vec<i64>,
    split_leftrights_sorted: Vec<i64>,
    log_split_probs_sorted: Vec<f64>,
    num_segs_ge2: i64,
    num_segs_eq1: i64,
    end_rows_ge2: i64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SplitKey {
    parent: usize,
    left: usize,
    right: usize,
}

pub fn preprocess_request(
    request: &PreprocessRequest,
) -> Result<PreprocessOutput, PreprocessError> {
    preprocess_multiple_families_with_threads(
        &request.species_path,
        &request.families,
        &request.leaf_species_maps,
        request.include_species_matrices,
        request.num_threads,
    )
}

#[cfg(feature = "python-extension")]
#[pyfunction]
fn preprocess_request_binary<'py>(
    py: Python<'py>,
    request_json: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let request: PreprocessRequest = serde_json::from_str(request_json)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let output = py
        .allow_threads(|| preprocess_request(&request))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let mut bytes = Vec::new();
    write_binary_output(&output, &mut bytes)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyBytes::new_bound(py, &bytes))
}

#[cfg(feature = "python-extension")]
#[pyfunction]
fn preprocess_request_numpy<'py>(
    py: Python<'py>,
    request_json: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let request: PreprocessRequest = serde_json::from_str(request_json)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let output = py
        .allow_threads(|| preprocess_request(&request))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    output_to_python(py, output)
}

#[cfg(feature = "python-extension")]
#[pyfunction]
fn preprocess_request_torch<'py>(
    py: Python<'py>,
    request_json: &str,
    from_numpy: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let request: PreprocessRequest = serde_json::from_str(request_json)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let output = py
        .allow_threads(|| preprocess_request(&request))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    output_to_torch_python(py, from_numpy, output)
}

#[cfg(feature = "python-extension")]
#[pyfunction]
fn schedule_global_phased_waves_json(request_json: &str) -> PyResult<String> {
    let request: ScheduleRequest = serde_json::from_str(request_json)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let output = schedule_global_phased_waves_request(&request)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    serde_json::to_string(&output).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[cfg(feature = "python-extension")]
#[pyfunction]
fn family_schedule_summary_json(ccp_json: &str) -> PyResult<String> {
    let ccp: ScheduleCcp =
        serde_json::from_str(ccp_json).map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let output =
        family_schedule_summary(&ccp).map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    serde_json::to_string(&output).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[cfg(feature = "python-extension")]
#[pyfunction]
fn plan_family_batches_json(request_json: &str) -> PyResult<String> {
    let request: BatchPlanRequest = serde_json::from_str(request_json)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let output = plan_family_batches_request(&request)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    serde_json::to_string(&output).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[cfg(feature = "python-extension")]
#[pymodule]
fn gpurec_preprocess(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(preprocess_request_binary, module)?)?;
    module.add_function(wrap_pyfunction!(preprocess_request_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(preprocess_request_torch, module)?)?;
    module.add_function(wrap_pyfunction!(schedule_global_phased_waves_json, module)?)?;
    module.add_function(wrap_pyfunction!(family_schedule_summary_json, module)?)?;
    module.add_function(wrap_pyfunction!(plan_family_batches_json, module)?)?;
    Ok(())
}

#[cfg(feature = "python-extension")]
fn output_to_python<'py>(
    py: Python<'py>,
    output: PreprocessOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new_bound(py);
    result.set_item("species", species_to_python(py, output.species)?)?;

    let families = PyDict::new_bound(py);
    for (name, family) in output.families {
        families.set_item(name, family_to_python(py, family)?)?;
    }
    result.set_item("families", families)?;
    Ok(result)
}

#[cfg(feature = "python-extension")]
fn species_to_python<'py>(py: Python<'py>, species: SpeciesOutput) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new_bound(py);
    let s = species.s;
    result.set_item("S", s)?;
    result.set_item("names", species.names)?;
    result.set_item("s_P_indexes", species.s_p_indexes.into_pyarray_bound(py))?;
    result.set_item(
        "s_C12_indexes",
        species.s_c12_indexes.into_pyarray_bound(py),
    )?;
    result.set_item(
        "unnorm_row_max",
        species.unnorm_row_max.into_pyarray_bound(py),
    )?;

    if let Some(ancestors) = species.ancestors_dense {
        result.set_item("ancestors_dense", vec_to_pyarray2(py, ancestors, s, s)?)?;
    }
    if let Some(recipients) = species.recipients_mat {
        result.set_item("Recipients_mat", vec_to_pyarray2(py, recipients, s, s)?)?;
    }

    let name_to_index = PyDict::new_bound(py);
    for (name, index) in species.species_name_to_index {
        name_to_index.set_item(name, index)?;
    }
    result.set_item("species_name_to_index", name_to_index)?;
    Ok(result)
}

#[cfg(feature = "python-extension")]
fn family_to_python<'py>(py: Python<'py>, family: FamilyOutput) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new_bound(py);
    let ccp = family.ccp;
    let ccp_dict = PyDict::new_bound(py);
    ccp_dict.set_item("split_counts", ccp.split_counts.into_pyarray_bound(py))?;
    ccp_dict.set_item(
        "split_parents_sorted",
        ccp.split_parents_sorted.into_pyarray_bound(py),
    )?;
    ccp_dict.set_item(
        "split_leftrights_sorted",
        ccp.split_leftrights_sorted.into_pyarray_bound(py),
    )?;
    ccp_dict.set_item(
        "log_split_probs_sorted",
        ccp.log_split_probs_sorted.into_pyarray_bound(py),
    )?;
    ccp_dict.set_item("num_segs_ge2", ccp.num_segs_ge2)?;
    ccp_dict.set_item("num_segs_eq1", ccp.num_segs_eq1)?;
    ccp_dict.set_item("end_rows_ge2", ccp.end_rows_ge2)?;
    ccp_dict.set_item("C", ccp.c)?;
    ccp_dict.set_item("N_splits", ccp.n_splits)?;
    ccp_dict.set_item("root_clade_id", ccp.root_clade_id)?;
    ccp_dict.set_item(
        "clade_leaf_labels",
        clade_labels_to_pylist(py, ccp.clade_leaf_labels)?,
    )?;

    result.set_item("ccp", ccp_dict)?;
    result.set_item("root_clade_id", family.root_clade_id)?;
    result.set_item(
        "leaf_row_index",
        family.leaf_row_index.into_pyarray_bound(py),
    )?;
    result.set_item(
        "leaf_col_index",
        family.leaf_col_index.into_pyarray_bound(py),
    )?;
    Ok(result)
}

#[cfg(feature = "python-extension")]
fn vec_to_pyarray2<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    Array2::from_shape_vec((rows, cols), values)
        .map_err(|err| PyValueError::new_err(err.to_string()))
        .map(|array| array.into_pyarray_bound(py))
}

#[cfg(feature = "python-extension")]
fn output_to_torch_python<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    output: PreprocessOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new_bound(py);
    result.set_item(
        "species",
        species_to_torch_python(py, from_numpy, output.species)?,
    )?;

    let families = PyDict::new_bound(py);
    for (name, family) in output.families {
        families.set_item(name, family_to_torch_python(py, from_numpy, family)?)?;
    }
    result.set_item("families", families)?;
    Ok(result)
}

#[cfg(feature = "python-extension")]
fn species_to_torch_python<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    species: SpeciesOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new_bound(py);
    let s = species.s;
    result.set_item("S", s)?;
    result.set_item("names", species.names)?;
    result.set_item(
        "s_P_indexes",
        vec_i64_to_torch(py, from_numpy, species.s_p_indexes)?,
    )?;
    result.set_item(
        "s_C12_indexes",
        vec_i64_to_torch(py, from_numpy, species.s_c12_indexes)?,
    )?;
    result.set_item(
        "unnorm_row_max",
        vec_f64_to_torch(py, from_numpy, species.unnorm_row_max)?,
    )?;

    if let Some(ancestors) = species.ancestors_dense {
        result.set_item(
            "ancestors_dense",
            vec_f64_matrix_to_torch(py, from_numpy, ancestors, s, s)?,
        )?;
    }
    if let Some(recipients) = species.recipients_mat {
        result.set_item(
            "Recipients_mat",
            vec_f64_matrix_to_torch(py, from_numpy, recipients, s, s)?,
        )?;
    }

    let name_to_index = PyDict::new_bound(py);
    for (name, index) in species.species_name_to_index {
        name_to_index.set_item(name, index)?;
    }
    result.set_item("species_name_to_index", name_to_index)?;
    Ok(result)
}

#[cfg(feature = "python-extension")]
fn family_to_torch_python<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    family: FamilyOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new_bound(py);
    let ccp = family.ccp;
    let ccp_dict = PyDict::new_bound(py);
    ccp_dict.set_item(
        "split_counts",
        vec_i64_to_torch(py, from_numpy, ccp.split_counts)?,
    )?;
    ccp_dict.set_item(
        "split_parents_sorted",
        vec_i64_to_torch(py, from_numpy, ccp.split_parents_sorted)?,
    )?;
    ccp_dict.set_item(
        "split_leftrights_sorted",
        vec_i64_to_torch(py, from_numpy, ccp.split_leftrights_sorted)?,
    )?;
    ccp_dict.set_item(
        "log_split_probs_sorted",
        vec_f64_to_torch(py, from_numpy, ccp.log_split_probs_sorted)?,
    )?;
    ccp_dict.set_item("num_segs_ge2", ccp.num_segs_ge2)?;
    ccp_dict.set_item("num_segs_eq1", ccp.num_segs_eq1)?;
    ccp_dict.set_item("end_rows_ge2", ccp.end_rows_ge2)?;
    ccp_dict.set_item("C", ccp.c)?;
    ccp_dict.set_item("N_splits", ccp.n_splits)?;
    ccp_dict.set_item("root_clade_id", ccp.root_clade_id)?;
    ccp_dict.set_item(
        "clade_leaf_labels",
        clade_labels_to_pylist(py, ccp.clade_leaf_labels)?,
    )?;

    result.set_item("ccp", ccp_dict)?;
    result.set_item("root_clade_id", family.root_clade_id)?;
    result.set_item(
        "leaf_row_index",
        vec_i64_to_torch(py, from_numpy, family.leaf_row_index)?,
    )?;
    result.set_item(
        "leaf_col_index",
        vec_i64_to_torch(py, from_numpy, family.leaf_col_index)?,
    )?;
    Ok(result)
}

#[cfg(feature = "python-extension")]
fn vec_i64_to_torch<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    values: Vec<i64>,
) -> PyResult<Py<PyAny>> {
    let array = values.into_pyarray_bound(py);
    from_numpy.call1((array,)).map(Bound::unbind)
}

#[cfg(feature = "python-extension")]
fn vec_f64_to_torch<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    values: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    let array = values.into_pyarray_bound(py);
    from_numpy.call1((array,)).map(Bound::unbind)
}

#[cfg(feature = "python-extension")]
fn vec_f64_matrix_to_torch<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    values: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Py<PyAny>> {
    let array = vec_to_pyarray2(py, values, rows, cols)?;
    from_numpy.call1((array,)).map(Bound::unbind)
}

#[cfg(feature = "python-extension")]
fn clade_labels_to_pylist<'py>(
    py: Python<'py>,
    labels: Vec<String>,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty_bound(py);
    let empty = PyString::new_bound(py, "");
    for label in labels {
        if label.is_empty() {
            list.append(&empty)?;
        } else {
            list.append(label)?;
        }
    }
    Ok(list)
}

pub fn preprocess_multiple_families(
    species_path: &Path,
    families: &BTreeMap<String, Vec<PathBuf>>,
    leaf_species_maps: &BTreeMap<String, BTreeMap<String, String>>,
    include_species_matrices: bool,
) -> Result<PreprocessOutput, PreprocessError> {
    preprocess_multiple_families_with_threads(
        species_path,
        families,
        leaf_species_maps,
        include_species_matrices,
        0,
    )
}

pub fn preprocess_multiple_families_with_threads(
    species_path: &Path,
    families: &BTreeMap<String, Vec<PathBuf>>,
    leaf_species_maps: &BTreeMap<String, BTreeMap<String, String>>,
    include_species_matrices: bool,
    num_threads: usize,
) -> Result<PreprocessOutput, PreprocessError> {
    let species_tree = parse_one_newick_file(species_path)?;
    let species = enumerate_species(&species_tree)?;
    let species_output = build_species_output(&species, include_species_matrices);
    let family_entries: Vec<_> = families.iter().collect();
    let run_families = || -> Result<Vec<(String, FamilyOutput)>, PreprocessError> {
        family_entries
            .par_iter()
            .map(|(family_name, gene_paths)| {
                let leaf_map = leaf_species_maps.get(*family_name);
                let family = preprocess_one_family(
                    family_name,
                    gene_paths,
                    leaf_map,
                    &species_output.species_name_to_index,
                )?;
                Ok(((*family_name).clone(), family))
            })
            .collect()
    };

    let family_outputs = if num_threads > 0 {
        get_thread_pool(num_threads)?.install(run_families)?
    } else {
        run_families()?
    }
    .into_iter()
    .collect();

    Ok(PreprocessOutput {
        species: species_output,
        families: family_outputs,
    })
}

fn get_thread_pool(num_threads: usize) -> Result<Arc<rayon::ThreadPool>, PreprocessError> {
    let pools = THREAD_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pool) = pools
        .lock()
        .map_err(|err| PreprocessError::InvalidInput(err.to_string()))?
        .get(&num_threads)
        .cloned()
    {
        return Ok(pool);
    }

    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()?,
    );
    let mut guard = pools
        .lock()
        .map_err(|err| PreprocessError::InvalidInput(err.to_string()))?;
    Ok(guard
        .entry(num_threads)
        .or_insert_with(|| Arc::clone(&pool))
        .clone())
}

fn parse_one_newick_file(path: &Path) -> Result<GeneTree, PreprocessError> {
    let text = read_text(path)?;
    GeneNewickParser::new(text.trim()).parse(path)
}

fn parse_gene_newick_file_records(path: &Path) -> Result<Vec<GeneTree>, PreprocessError> {
    let text = read_text(path)?;
    parse_gene_newick_records(&text, path)
}

fn read_text(path: &Path) -> Result<String, PreprocessError> {
    fs::read_to_string(path).map_err(|source| PreprocessError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn parse_gene_newick_records(text: &str, path: &Path) -> Result<Vec<GeneTree>, PreprocessError> {
    let mut trees = Vec::new();
    for raw in text.split(';') {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut tree = GeneNewickParser::new(trimmed).parse(path)?;
        binarize_gene_tree(&mut tree, path)?;
        trees.push(tree);
    }
    if trees.is_empty() {
        return Err(PreprocessError::InvalidInput(format!(
            "No Newick trees found in file: {}",
            path.display()
        )));
    }
    Ok(trees)
}

struct GeneNewickParser<'a> {
    text: &'a str,
    pos: usize,
    nodes: Vec<GeneNode>,
}

impl<'a> GeneNewickParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            pos: 0,
            nodes: Vec::new(),
        }
    }

    fn parse(mut self, path: &Path) -> Result<GeneTree, PreprocessError> {
        let root = self.parse_subtree(path)?;
        self.skip_whitespace();
        if self.peek_byte() == Some(b';') {
            self.pos += 1;
        }
        self.skip_whitespace();
        if self.pos != self.text.len() {
            return Err(newick_error(
                path,
                "Unexpected trailing characters in Newick string",
            ));
        }
        Ok(GeneTree {
            nodes: self.nodes,
            root,
        })
    }

    fn parse_subtree(&mut self, path: &Path) -> Result<usize, PreprocessError> {
        self.skip_whitespace();
        let node = self.make_node();
        if self.pos >= self.text.len() {
            return Err(newick_error(path, "Unexpected end of Newick string"));
        }

        if self.peek_byte() == Some(b'(') {
            self.pos += 1;
            loop {
                let child = self.parse_subtree(path)?;
                self.nodes[child].parent = Some(node);
                self.nodes[node].children.push(child);
                self.skip_whitespace();
                let Some(c) = self.peek_byte() else {
                    return Err(newick_error(path, "Unexpected end while parsing children"));
                };
                match c {
                    b',' => {
                        self.pos += 1;
                    }
                    b')' => {
                        self.pos += 1;
                        break;
                    }
                    _ => {
                        return Err(newick_error(path, "Expected ',' or ')' in Newick string"));
                    }
                }
            }
            self.nodes[node].name = self.parse_label();
        } else {
            self.nodes[node].name = self.parse_label();
        }

        self.skip_branch_length();
        Ok(node)
    }

    fn make_node(&mut self) -> usize {
        let node = self.nodes.len();
        self.nodes.push(GeneNode {
            name: String::new(),
            children: Vec::new(),
            parent: None,
        });
        node
    }

    fn parse_label(&mut self) -> String {
        self.skip_whitespace();
        let start = self.pos;
        while let Some(c) = self.peek_byte() {
            if matches!(c, b':' | b',' | b')' | b'(' | b';') {
                break;
            }
            self.pos += 1;
        }
        self.text[start..self.pos].trim().to_string()
    }

    fn skip_branch_length(&mut self) {
        self.skip_whitespace();
        if self.peek_byte() != Some(b':') {
            return;
        }
        self.pos += 1;
        while let Some(c) = self.peek_byte() {
            if c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn skip_whitespace(&mut self) {
        while self.peek_byte().is_some_and(|c| c.is_ascii_whitespace()) {
            self.pos += 1;
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }
}

fn newick_error(path: &Path, message: &str) -> PreprocessError {
    PreprocessError::InvalidInput(format!("{}: {message}", path.display()))
}

fn binarize_gene_tree(tree: &mut GeneTree, path: &Path) -> Result<(), PreprocessError> {
    binarize_gene_node(tree, tree.root, path)
}

fn binarize_gene_node(
    tree: &mut GeneTree,
    node: usize,
    path: &Path,
) -> Result<(), PreprocessError> {
    let children = tree.nodes[node].children.clone();
    for child in children {
        binarize_gene_node(tree, child, path)?;
    }

    if tree.nodes[node].children.len() == 1 {
        return Err(PreprocessError::InvalidInput(format!(
            "Unary node in gene tree {}. GPUREC expects binary gene trees or multifurcations that can be resolved deterministically.",
            path.display()
        )));
    }

    while tree.nodes[node].children.len() > 2 {
        let right = tree.nodes[node].children.pop().expect("checked length > 2");
        let left = tree.nodes[node].children.pop().expect("checked length > 2");
        let internal = tree.nodes.len();
        tree.nodes.push(GeneNode {
            name: String::new(),
            children: vec![left, right],
            parent: Some(node),
        });
        tree.nodes[left].parent = Some(internal);
        tree.nodes[right].parent = Some(internal);
        tree.nodes[node].children.push(internal);
    }
    Ok(())
}

fn gene_postorder(tree: &GeneTree) -> Vec<usize> {
    fn visit(tree: &GeneTree, node: usize, out: &mut Vec<usize>) {
        for child in &tree.nodes[node].children {
            visit(tree, *child, out);
        }
        out.push(node);
    }

    let mut out = Vec::with_capacity(tree.nodes.len());
    visit(tree, tree.root, &mut out);
    out
}

fn enumerate_species(tree: &GeneTree) -> Result<SpeciesTopology, PreprocessError> {
    let postorder = gene_postorder(tree);
    let s = postorder.len();
    let mut node_to_gp = vec![usize::MAX; tree.nodes.len()];
    for (gp_idx, &node_idx) in postorder.iter().enumerate() {
        node_to_gp[node_idx] = gp_idx;
    }

    let mut names = vec![String::new(); s];
    let mut children = vec![Vec::new(); s];
    let mut parent = vec![None; s];
    for (gp_idx, &node_idx) in postorder.iter().enumerate() {
        let node = &tree.nodes[node_idx];
        names[gp_idx] = node.name.clone();
        if let Some(parent_idx) = node.parent {
            parent[gp_idx] = Some(node_to_gp[parent_idx]);
        }
        match node.children.as_slice() {
            [left, right] => {
                children[gp_idx].push(node_to_gp[*left]);
                children[gp_idx].push(node_to_gp[*right]);
            }
            [] => {}
            _ => {
                return Err(PreprocessError::InvalidInput(format!(
                    "Species tree must be strictly binary: node at postorder index {gp_idx} has {} children",
                    node.children.len()
                )));
            }
        }
    }

    Ok(SpeciesTopology {
        names,
        children,
        parent,
    })
}

fn build_species_output(
    species: &SpeciesTopology,
    include_species_matrices: bool,
) -> SpeciesOutput {
    let s = species.names.len();
    let mut s_p_indexes = Vec::new();
    let mut s_c1_indexes = Vec::new();
    let mut s_c2_indexes = Vec::new();
    for (idx, children) in species.children.iter().enumerate() {
        if children.len() == 2 {
            s_p_indexes.push(idx as i64);
            s_c1_indexes.push(children[0] as i64);
            s_c2_indexes.push(children[1] as i64);
        }
    }
    let mut s_p_indexes_ext = s_p_indexes.clone();
    s_p_indexes_ext.extend(s_p_indexes.iter().map(|idx| idx + s as i64));

    let mut s_c12_indexes = s_c1_indexes;
    s_c12_indexes.extend(s_c2_indexes);

    let mut species_name_to_index = BTreeMap::new();
    for (idx, name) in species.names.iter().enumerate() {
        if !name.is_empty() {
            species_name_to_index.insert(name.clone(), idx);
        }
    }

    let ancestors_dense = include_species_matrices.then(|| compute_ancestors(species));
    let recipients_mat = ancestors_dense
        .as_ref()
        .map(|ancestors| compute_recipients(ancestors, s));

    SpeciesOutput {
        s,
        names: species.names.clone(),
        s_p_indexes: s_p_indexes_ext,
        s_c12_indexes,
        unnorm_row_max: compute_uniform_unnorm_row_max(species),
        ancestors_dense,
        recipients_mat,
        species_name_to_index,
    }
}

fn compute_ancestors(species: &SpeciesTopology) -> Vec<f64> {
    let s = species.names.len();
    let mut ancestors = vec![0.0; s * s];
    for start in 0..s {
        let mut cur = Some(start);
        while let Some(idx) = cur {
            ancestors[start * s + idx] = 1.0;
            cur = species.parent[idx];
        }
    }
    ancestors
}

fn compute_recipients(ancestors: &[f64], s: usize) -> Vec<f64> {
    let mut recipients = vec![0.0; s * s];
    for i in 0..s {
        let mut total = 0.0;
        for j in 0..s {
            if ancestors[i * s + j] == 0.0 {
                recipients[i * s + j] = 1.0;
                total += 1.0;
            }
        }
        if total > 0.0 {
            for j in 0..s {
                recipients[i * s + j] /= total;
            }
        }
    }
    recipients
}

fn compute_uniform_unnorm_row_max(species: &SpeciesTopology) -> Vec<f64> {
    let s = species.names.len();
    let mut row_max = vec![f64::NEG_INFINITY; s];
    for (idx, out) in row_max.iter_mut().enumerate() {
        let mut depth = 0usize;
        let mut cur = Some(idx);
        while let Some(node) = cur {
            depth += 1;
            cur = species.parent[node];
        }
        let recipients = s.saturating_sub(depth);
        if recipients > 0 {
            *out = -(recipients as f64).log2();
        }
    }
    row_max
}

fn preprocess_one_family(
    family_name: &str,
    gene_paths: &[PathBuf],
    leaf_species_map: Option<&BTreeMap<String, String>>,
    species_name_to_index: &BTreeMap<String, usize>,
) -> Result<FamilyOutput, PreprocessError> {
    let (clade_data, leaf_names) = amalgamate_clades_and_splits(gene_paths)?;
    let ccp = build_ccp_arrays(&clade_data);
    let c = clade_data.clades.clades.len();
    let mut clade_leaf_labels = vec![String::new(); c];
    let mut leaf_row_index = Vec::new();
    let mut leaf_col_index = Vec::new();

    for cid in 0..c {
        let clade = &clade_data.clades.clades[cid];
        if clade.size != 1 {
            continue;
        }
        let Some(leaf_idx) = first_set_bit(&clade.bits) else {
            continue;
        };
        let leaf_name = &leaf_names[leaf_idx];
        let species = species_for_gene_leaf(leaf_name, leaf_species_map, family_name)?;
        let species_idx = species_name_to_index.get(&species).ok_or_else(|| {
            PreprocessError::InvalidInput(format!(
                "Species {species} not found for gene leaf {leaf_name}"
            ))
        })?;
        leaf_row_index.push(cid as i64);
        leaf_col_index.push(*species_idx as i64);
        clade_leaf_labels[cid] = leaf_name.clone();
    }

    Ok(FamilyOutput {
        ccp: CcpOutput {
            split_counts: ccp.split_counts,
            split_parents_sorted: ccp.split_parents_sorted,
            split_leftrights_sorted: ccp.split_leftrights_sorted,
            log_split_probs_sorted: ccp.log_split_probs_sorted,
            num_segs_ge2: ccp.num_segs_ge2,
            num_segs_eq1: ccp.num_segs_eq1,
            end_rows_ge2: ccp.end_rows_ge2,
            c: c as i64,
            n_splits: clade_data.splits.len() as i64,
            root_clade_id: clade_data.root_clade_id as i64,
            clade_leaf_labels,
        },
        root_clade_id: clade_data.root_clade_id as i64,
        leaf_row_index,
        leaf_col_index,
    })
}

fn species_for_gene_leaf(
    leaf_name: &str,
    leaf_species_map: Option<&BTreeMap<String, String>>,
    family_name: &str,
) -> Result<String, PreprocessError> {
    if let Some(mapping) = leaf_species_map {
        if !mapping.is_empty() {
            return mapping.get(leaf_name).cloned().ok_or_else(|| {
                PreprocessError::InvalidInput(format!(
                    "Gene leaf {leaf_name} is missing from mapping for family {family_name}"
                ))
            });
        }
    }
    Ok(leaf_name
        .split_once('_')
        .map(|(species, _)| species)
        .unwrap_or(leaf_name)
        .to_string())
}

fn amalgamate_clades_and_splits(
    gene_paths: &[PathBuf],
) -> Result<(CladeData, Vec<String>), PreprocessError> {
    if gene_paths.is_empty() {
        return Err(PreprocessError::InvalidInput(
            "No gene tree paths provided".to_string(),
        ));
    }

    let mut gene_trees = Vec::new();
    let mut all_leaves = BTreeSet::new();
    for path in gene_paths {
        for tree in parse_gene_newick_file_records(path)? {
            collect_leaf_names(&tree, &mut all_leaves);
            gene_trees.push(tree);
        }
    }

    let leaf_names: Vec<String> = all_leaves.into_iter().collect();
    if leaf_names.is_empty() {
        return Err(PreprocessError::InvalidInput(
            "No leaves found in gene trees".to_string(),
        ));
    }
    let leaf_to_index: HashMap<String, usize> = leaf_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect();
    let num_words = bitvec_num_words(leaf_names.len());

    let mut result = CladeData {
        clades: CladeRegistry::new(),
        splits: Vec::new(),
        root_clade_id: 0,
    };
    let mut root_bits = vec![0; num_words];
    for idx in 0..leaf_names.len() {
        set_bit(&mut root_bits, idx);
    }
    result.root_clade_id = result.clades.get_or_create(root_bits);
    let mut split_index_map = HashMap::new();

    for tree in &gene_trees {
        process_gene_tree(
            tree,
            &leaf_to_index,
            num_words,
            &mut result,
            &mut split_index_map,
        )?;
    }

    Ok((result, leaf_names))
}

fn collect_leaf_names(tree: &GeneTree, out: &mut BTreeSet<String>) {
    for node in &tree.nodes {
        if node.children.is_empty() {
            out.insert(node.name.clone());
        }
    }
}

fn process_gene_tree(
    tree: &GeneTree,
    leaf_to_index: &HashMap<String, usize>,
    num_words: usize,
    result: &mut CladeData,
    split_index_map: &mut HashMap<SplitKey, usize>,
) -> Result<(), PreprocessError> {
    let postorder = gene_postorder(tree);
    let mut node_clades = vec![vec![0; num_words]; tree.nodes.len()];
    let mut node_clade_ids = vec![usize::MAX; tree.nodes.len()];
    let mut node_above_ids = vec![None; tree.nodes.len()];

    for &node_idx in &postorder {
        let node = &tree.nodes[node_idx];
        let bits = match node.children.as_slice() {
            [] => {
                let leaf_idx = leaf_to_index.get(&node.name).ok_or_else(|| {
                    PreprocessError::InvalidInput(format!(
                        "Gene leaf {:?} was not present in the family leaf set",
                        node.name
                    ))
                })?;
                let mut bits = vec![0; num_words];
                set_bit(&mut bits, *leaf_idx);
                bits
            }
            [left, right] => bit_or(&node_clades[*left], &node_clades[*right]),
            [_] => {
                return Err(PreprocessError::InvalidInput(
                    "Unary node in gene tree".to_string(),
                ));
            }
            _ => {
                return Err(PreprocessError::InvalidInput(
                    "Multifurcating node remained after gene tree binarization".to_string(),
                ));
            }
        };
        node_clades[node_idx] = bits;
        node_clade_ids[node_idx] = result.clades.get_or_create(node_clades[node_idx].clone());
    }

    let tree_root_bits = node_clades[tree.root].clone();
    let tree_root_id = node_clade_ids[tree.root];
    let mut tree_root_split_keys = HashSet::new();

    for &node_idx in &postorder {
        if node_idx == tree.root {
            continue;
        }
        let above_bits = bit_difference(&tree_root_bits, &node_clades[node_idx]);
        if is_empty(&above_bits) {
            continue;
        }
        let below_id = node_clade_ids[node_idx];
        let above_id = result.clades.get_or_create(above_bits);
        node_above_ids[node_idx] = Some(above_id);
        let root_key = canonical_root_key(&result.clades, below_id, above_id);
        if tree_root_split_keys.insert(root_key) {
            add_or_accumulate_split(
                result,
                split_index_map,
                CladeSplit {
                    parent: tree_root_id,
                    left: below_id,
                    right: above_id,
                    weight: 1.0,
                },
            );
        }
    }

    for &node_idx in &postorder {
        if node_idx == tree.root {
            continue;
        }
        let node = &tree.nodes[node_idx];
        let [left_idx, right_idx] = node.children.as_slice() else {
            continue;
        };

        let parent_id = node_clade_ids[node_idx];
        let left_id = node_clade_ids[*left_idx];
        let right_id = node_clade_ids[*right_idx];
        add_or_accumulate_split(
            result,
            split_index_map,
            CladeSplit {
                parent: parent_id,
                left: left_id,
                right: right_id,
                weight: 1.0,
            },
        );

        if let Some(above_id) = node_above_ids[node_idx] {
            let Some(left_plus_id) = node_above_ids[*right_idx] else {
                return Err(PreprocessError::InvalidInput(
                    "Missing left-plus clade while building rerooted split".to_string(),
                ));
            };
            let Some(right_plus_id) = node_above_ids[*left_idx] else {
                return Err(PreprocessError::InvalidInput(
                    "Missing right-plus clade while building rerooted split".to_string(),
                ));
            };
            add_or_accumulate_split(
                result,
                split_index_map,
                CladeSplit {
                    parent: left_plus_id,
                    left: left_id,
                    right: above_id,
                    weight: 1.0,
                },
            );
            add_or_accumulate_split(
                result,
                split_index_map,
                CladeSplit {
                    parent: right_plus_id,
                    left: right_id,
                    right: above_id,
                    weight: 1.0,
                },
            );
        }
    }

    Ok(())
}

fn add_or_accumulate_split(
    data: &mut CladeData,
    split_index_map: &mut HashMap<SplitKey, usize>,
    split: CladeSplit,
) {
    let key = split.canonical_key();
    if let Some(idx) = split_index_map.get(&key) {
        data.splits[*idx].weight += split.weight;
    } else {
        let idx = data.splits.len();
        data.splits.push(split);
        split_index_map.insert(key, idx);
    }
}

fn build_ccp_arrays(data: &CladeData) -> CcpArrays {
    let c = data.clades.clades.len();
    let n = data.splits.len();
    let mut split_counts = vec![0i64; c];
    let mut split_weights = vec![0.0; n];
    for (idx, split) in data.splits.iter().enumerate() {
        split_counts[split.parent] += 1;
        split_weights[idx] = split.weight;
    }

    let mut sum_weights = vec![0.0; c];
    for (idx, split) in data.splits.iter().enumerate() {
        sum_weights[split.parent] += split_weights[idx];
    }

    let mut log_split_probs = vec![f64::NEG_INFINITY; n];
    for (idx, split) in data.splits.iter().enumerate() {
        let denom = sum_weights[split.parent];
        if denom > 0.0 && split_weights[idx] > 0.0 {
            log_split_probs[idx] = (split_weights[idx] / denom).ln();
        }
    }

    let mut parents_sorted: Vec<usize> = (0..c).collect();
    parents_sorted.sort_by(|&a, &b| {
        split_counts[b]
            .cmp(&split_counts[a])
            .then_with(|| a.cmp(&b))
    });
    let mut parent_rank = vec![0usize; c];
    for (rank, parent) in parents_sorted.iter().enumerate() {
        parent_rank[*parent] = rank;
    }

    let mut split_order: Vec<usize> = (0..n).collect();
    split_order.sort_by(|&lhs, &rhs| {
        parent_rank[data.splits[lhs].parent]
            .cmp(&parent_rank[data.splits[rhs].parent])
            .then_with(|| lhs.cmp(&rhs))
    });

    let mut split_parents_sorted = Vec::with_capacity(n);
    let mut split_lefts_sorted = Vec::with_capacity(n);
    let mut split_rights_sorted = Vec::with_capacity(n);
    let mut log_split_probs_sorted = Vec::with_capacity(n);
    for idx in split_order {
        let split = &data.splits[idx];
        split_parents_sorted.push(split.parent as i64);
        split_lefts_sorted.push(split.left as i64);
        split_rights_sorted.push(split.right as i64);
        log_split_probs_sorted.push(log_split_probs[idx]);
    }

    let mut split_leftrights_sorted = split_lefts_sorted.clone();
    split_leftrights_sorted.extend(split_rights_sorted.iter().copied());

    let seg_counts: Vec<i64> = parents_sorted
        .iter()
        .map(|&parent| split_counts[parent])
        .collect();
    let mut ptr = Vec::with_capacity(c + 1);
    ptr.push(0i64);
    for count in &seg_counts {
        ptr.push(ptr.last().copied().unwrap_or(0) + count);
    }

    let mut num_segs_ge2 = 0;
    let mut num_segs_eq1 = 0;
    for count in &seg_counts {
        if *count >= 2 {
            num_segs_ge2 += 1;
        } else if *count == 1 {
            num_segs_eq1 += 1;
        }
    }
    let end_rows_ge2 = ptr[num_segs_ge2 as usize];

    CcpArrays {
        split_counts,
        split_parents_sorted,
        split_leftrights_sorted,
        log_split_probs_sorted,
        num_segs_ge2,
        num_segs_eq1,
        end_rows_ge2,
    }
}

impl CladeRegistry {
    fn new() -> Self {
        Self {
            clades: Vec::new(),
            ids_by_bits: HashMap::new(),
        }
    }

    fn get_or_create(&mut self, bits: Vec<u64>) -> usize {
        if let Some(id) = self.ids_by_bits.get(&bits) {
            return *id;
        }
        let id = self.clades.len();
        let clade = Clade {
            size: bit_count(&bits),
            bits: bits.clone(),
        };
        self.clades.push(clade);
        self.ids_by_bits.insert(bits, id);
        id
    }
}

impl CladeSplit {
    fn canonical_key(&self) -> SplitKey {
        let (left, right) = if self.left <= self.right {
            (self.left, self.right)
        } else {
            (self.right, self.left)
        };
        SplitKey {
            parent: self.parent,
            left,
            right,
        }
    }
}

fn canonical_root_key(clades: &CladeRegistry, a: usize, b: usize) -> (usize, usize) {
    let left = &clades.clades[a];
    let right = &clades.clades[b];
    match left
        .size
        .cmp(&right.size)
        .then_with(|| bitvec_cmp(&left.bits, &right.bits))
    {
        Ordering::Less => (a, b),
        Ordering::Greater => (b, a),
        Ordering::Equal => (a.min(b), a.max(b)),
    }
}

fn bitvec_num_words(num_leaves: usize) -> usize {
    (num_leaves + BITS_PER_WORD - 1) / BITS_PER_WORD
}

fn set_bit(bits: &mut [u64], index: usize) {
    let word = index >> 6;
    let offset = index & 63;
    bits[word] |= 1u64 << offset;
}

fn first_set_bit(bits: &[u64]) -> Option<usize> {
    bits.iter()
        .enumerate()
        .find(|(_, word)| **word != 0)
        .map(|(word_idx, word)| word_idx * BITS_PER_WORD + word.trailing_zeros() as usize)
}

fn bit_or(left: &[u64], right: &[u64]) -> Vec<u64> {
    left.iter().zip(right).map(|(a, b)| a | b).collect()
}

fn bit_difference(parent: &[u64], child: &[u64]) -> Vec<u64> {
    parent.iter().zip(child).map(|(p, c)| p & !c).collect()
}

fn is_empty(bits: &[u64]) -> bool {
    bits.iter().all(|word| *word == 0)
}

fn bit_count(bits: &[u64]) -> usize {
    bits.iter().map(|word| word.count_ones() as usize).sum()
}

fn bitvec_cmp(left: &[u64], right: &[u64]) -> Ordering {
    left.cmp(right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_leaf_tree_matches_expected_ccp_shape() {
        let tmp = tempfile_dir("two_leaf_tree_matches_expected_ccp_shape");
        let species_path = tmp.join("species.nwk");
        let gene_path = tmp.join("gene.nwk");
        fs::write(&species_path, "(A,B)Root;\n").unwrap();
        fs::write(&gene_path, "(a,b)g;\n").unwrap();

        let mut families = BTreeMap::new();
        families.insert("fam".to_string(), vec![gene_path]);
        let mut leaf_map = BTreeMap::new();
        leaf_map.insert(
            "fam".to_string(),
            BTreeMap::from([
                ("a".to_string(), "A".to_string()),
                ("b".to_string(), "B".to_string()),
            ]),
        );

        let output =
            preprocess_multiple_families(&species_path, &families, &leaf_map, false).unwrap();
        assert_eq!(output.species.names, vec!["A", "B", "Root"]);
        assert_eq!(output.species.s_p_indexes, vec![2, 5]);
        assert_eq!(output.species.s_c12_indexes, vec![0, 1]);

        let family = output.families.get("fam").unwrap();
        assert_eq!(family.root_clade_id, 0);
        assert_eq!(family.leaf_row_index, vec![1, 2]);
        assert_eq!(family.leaf_col_index, vec![0, 1]);
        assert_eq!(family.ccp.c, 3);
        assert_eq!(family.ccp.n_splits, 1);
        assert_eq!(family.ccp.split_counts, vec![1, 0, 0]);
        assert_eq!(family.ccp.split_parents_sorted, vec![0]);
        assert_eq!(family.ccp.split_leftrights_sorted, vec![1, 2]);
        assert_eq!(family.ccp.clade_leaf_labels, vec!["", "a", "b"]);
    }

    #[test]
    fn multiple_records_accumulate_split_weights() {
        let tmp = tempfile_dir("multiple_records_accumulate_split_weights");
        let species_path = tmp.join("species.nwk");
        let gene_path = tmp.join("gene.nwk");
        fs::write(&species_path, "((a,b)ab,c)root;\n").unwrap();
        fs::write(&gene_path, "((a,b),c);((a,b),c)\n").unwrap();

        let mut families = BTreeMap::new();
        families.insert("fam".to_string(), vec![gene_path]);
        let output =
            preprocess_multiple_families(&species_path, &families, &BTreeMap::new(), false)
                .unwrap();
        let family = output.families.get("fam").unwrap();
        assert!(family.ccp.n_splits > 1);
        assert_eq!(
            family
                .ccp
                .clade_leaf_labels
                .iter()
                .filter(|s| !s.is_empty())
                .count(),
            3
        );
    }

    #[test]
    fn multifurcating_gene_tree_is_right_binarized() {
        let path = Path::new("gene.nwk");
        let trees = parse_gene_newick_records("(a,b,c,d)g;", path).unwrap();
        assert_eq!(trees.len(), 1);

        let tree = &trees[0];
        let root_children = &tree.nodes[tree.root].children;
        assert_eq!(root_children.len(), 2);
        assert_eq!(tree.nodes[root_children[0]].name, "a");

        let bcd = root_children[1];
        let bcd_children = &tree.nodes[bcd].children;
        assert_eq!(bcd_children.len(), 2);
        assert_eq!(tree.nodes[bcd_children[0]].name, "b");

        let cd = bcd_children[1];
        let cd_children = &tree.nodes[cd].children;
        assert_eq!(cd_children.len(), 2);
        assert_eq!(tree.nodes[cd_children[0]].name, "c");
        assert_eq!(tree.nodes[cd_children[1]].name, "d");
    }

    fn tempfile_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("gpurec-preprocess-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }
}

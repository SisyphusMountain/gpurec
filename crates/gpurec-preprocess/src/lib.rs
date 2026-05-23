//! Rust preprocessing implementation for GPUREC.
//!
//! Species nodes are indexed in postorder, gene-family clades use sorted global
//! leaf labels, split weights are accumulated across tree samples, and CCP split
//! arrays are emitted in parent-ranked order for the Python likelihood runtime.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub mod batch_planning;
pub mod layout;
pub mod scheduler;
pub use batch_planning::{plan_family_batches_request, BatchPlanRequest, FamilyBatchPlanOutput};
pub use layout::{build_wave_layout_plan_request, WaveLayoutPlan, WaveLayoutRequest};
pub use scheduler::{
    family_schedule_summary, schedule_global_phased_waves_request, FamilyScheduleSummary,
    ScheduleCcp, ScheduleOutput, ScheduleRequest,
};

#[cfg(feature = "python-extension")]
use numpy::{ndarray::Array2, IntoPyArray, PyReadonlyArray1};
#[cfg(feature = "python-extension")]
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBytes, PyDict, PyList, PyString},
};

const BITS_PER_WORD: usize = 64;
const BINARY_MAGIC: &[u8; 8] = b"GPREP001";
#[cfg(feature = "python-extension")]
const INV_LN2: f64 = std::f64::consts::LOG2_E;
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

#[cfg(feature = "python-extension")]
#[derive(Clone, Debug, Deserialize)]
pub struct PreprocessDatasetRequest {
    #[serde(flatten)]
    pub preprocess: PreprocessRequest,
    #[serde(default)]
    pub family_order: Vec<String>,
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

#[cfg(feature = "python-extension")]
#[derive(Clone, Debug, Deserialize)]
pub struct ChunkedLayoutRequest {
    pub family_chunk_size: i64,
    #[serde(default)]
    pub clade_budget: Option<i64>,
    #[serde(default = "default_layout_batch_packing")]
    pub batch_packing: String,
    #[serde(default)]
    pub max_wave_size: Option<i64>,
    #[serde(default)]
    pub max_root_wave_size: Option<usize>,
    #[serde(default)]
    pub max_dts_partial_rows: Option<usize>,
    #[serde(default = "scheduler::default_dts_partial_tile_splits")]
    pub dts_partial_tile_splits: usize,
    #[serde(default = "default_layout_dtype")]
    pub dtype: String,
    #[serde(default)]
    pub num_threads: usize,
}

#[cfg(feature = "python-extension")]
fn default_layout_batch_packing() -> String {
    "sequential".to_string()
}

#[cfg(feature = "python-extension")]
fn default_layout_dtype() -> String {
    "float32".to_string()
}

#[cfg(feature = "python-extension")]
#[derive(Clone, Debug)]
struct CollatedFamilyBatch {
    c: usize,
    n_splits: usize,
    split_leftrights_sorted: Vec<i64>,
    split_parents_sorted: Vec<i64>,
    log_split_probs_sorted: Vec<f64>,
    leaf_row_index: Vec<i64>,
    leaf_col_index: Vec<i64>,
    root_clade_ids: Vec<i64>,
    family_clade_counts: Vec<i64>,
    family_clade_offsets: Vec<i64>,
}

#[cfg(feature = "python-extension")]
#[derive(Clone, Debug)]
struct FusedChunkOutput {
    indices: Vec<i64>,
    clades: i64,
    splits: i64,
    wave_layout: WaveLayoutPlan,
    log_split_probs_sorted: Vec<f64>,
    waves: i64,
    max_wave: i64,
    split_rows: i64,
    max_wave_split_rows: i64,
}

#[cfg(feature = "python-extension")]
#[derive(Clone, Debug)]
struct SpeciesHelperTopology {
    sp_child1_cpu: Vec<i64>,
    sp_child2_cpu: Vec<i64>,
    sp_parent_cpu: Vec<i64>,
    max_ancestor_depth: i64,
    compact_level_ptr: Vec<i64>,
    compact_level_parents: Vec<i32>,
    compact_level_child1: Vec<i32>,
    compact_level_child2: Vec<i32>,
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
fn ordered_family_names(
    output: &PreprocessOutput,
    requested_order: &[String],
) -> Result<Vec<String>, PreprocessError> {
    let names = if requested_order.is_empty() {
        output.families.keys().cloned().collect::<Vec<_>>()
    } else {
        requested_order.to_vec()
    };
    let mut seen = HashSet::with_capacity(names.len());
    for name in &names {
        if !seen.insert(name.clone()) {
            return Err(PreprocessError::InvalidInput(format!(
                "duplicate family name {name:?} in family_order"
            )));
        }
        if !output.families.contains_key(name) {
            return Err(PreprocessError::InvalidInput(format!(
                "family_order references unknown family {name:?}"
            )));
        }
    }
    Ok(names)
}

#[cfg(feature = "python-extension")]
fn schedule_ccp_from_family(family: &FamilyOutput) -> ScheduleCcp {
    let ccp = &family.ccp;
    ScheduleCcp {
        c: ccp.c as usize,
        n_splits: ccp.n_splits as usize,
        split_counts: Some(ccp.split_counts.clone()),
        split_parents_sorted: ccp.split_parents_sorted.clone(),
        split_leftrights_sorted: ccp.split_leftrights_sorted.clone(),
        root_clade_id: ccp.root_clade_id,
    }
}

#[cfg(feature = "python-extension")]
fn family_basic_counts(
    output: &PreprocessOutput,
    family_order: &[String],
) -> Result<(Vec<i64>, Vec<i64>), PreprocessError> {
    let mut clade_counts = Vec::with_capacity(family_order.len());
    let mut split_counts = Vec::with_capacity(family_order.len());
    for name in family_order {
        let family = output
            .families
            .get(name)
            .ok_or_else(|| PreprocessError::InvalidInput(format!("unknown family {name:?}")))?;
        clade_counts.push(family.ccp.c);
        split_counts.push(family.ccp.n_splits);
    }
    Ok((clade_counts, split_counts))
}

#[cfg(feature = "python-extension")]
fn family_counts_and_summaries(
    output: &PreprocessOutput,
    family_order: &[String],
) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>), PreprocessError> {
    let (clade_counts, split_counts) = family_basic_counts(output, family_order)?;
    let mut leaf_counts = Vec::with_capacity(family_order.len());
    let mut nonleaf_counts = Vec::with_capacity(family_order.len());
    let mut schedule_depths = Vec::with_capacity(family_order.len());
    for name in family_order {
        let family = output
            .families
            .get(name)
            .ok_or_else(|| PreprocessError::InvalidInput(format!("unknown family {name:?}")))?;
        let summary = family_schedule_summary(&schedule_ccp_from_family(family))?;
        leaf_counts.push(summary.leaf_count);
        nonleaf_counts.push(summary.nonleaf_count);
        schedule_depths.push(summary.max_level);
    }
    Ok((
        clade_counts,
        split_counts,
        leaf_counts,
        nonleaf_counts,
        schedule_depths,
    ))
}

#[cfg(feature = "python-extension")]
fn collate_family_batch(
    output: &PreprocessOutput,
    family_order: &[String],
    indices: &[i64],
) -> Result<CollatedFamilyBatch, PreprocessError> {
    if indices.is_empty() {
        return Err(PreprocessError::InvalidInput(
            "chunk must contain at least one family".to_string(),
        ));
    }

    let mut clade_offset = 0i64;
    let mut root_clade_ids = Vec::with_capacity(indices.len());
    let mut leaf_row_index = Vec::new();
    let mut leaf_col_index = Vec::new();
    let mut ge2_left = Vec::new();
    let mut ge2_right = Vec::new();
    let mut ge2_logp = Vec::new();
    let mut ge2_parents = Vec::new();
    let mut eq1_left = Vec::new();
    let mut eq1_right = Vec::new();
    let mut eq1_logp = Vec::new();
    let mut eq1_parents = Vec::new();
    let mut family_clade_counts = Vec::with_capacity(indices.len());
    let mut family_clade_offsets = Vec::with_capacity(indices.len());
    let mut total_splits = 0usize;

    for index in indices {
        if *index < 0 || (*index as usize) >= family_order.len() {
            return Err(PreprocessError::InvalidInput(format!(
                "family index {index} is outside valid range [0, {})",
                family_order.len()
            )));
        }
        let name = &family_order[*index as usize];
        let family = output
            .families
            .get(name)
            .ok_or_else(|| PreprocessError::InvalidInput(format!("unknown family {name:?}")))?;
        let ccp = &family.ccp;
        let c = ccp.c;
        let n_splits = ccp.n_splits as usize;
        let end_rows_ge2 = ccp.end_rows_ge2 as usize;
        let num_eq1 = ccp.num_segs_eq1 as usize;
        if end_rows_ge2 + num_eq1 != n_splits {
            return Err(PreprocessError::InvalidInput(format!(
                "family {name:?} split block lengths cover {} rows but N_splits={n_splits}",
                end_rows_ge2 + num_eq1
            )));
        }
        if ccp.split_leftrights_sorted.len() != 2 * n_splits {
            return Err(PreprocessError::InvalidInput(format!(
                "family {name:?} split_leftrights_sorted has length {} but expected {}",
                ccp.split_leftrights_sorted.len(),
                2 * n_splits
            )));
        }
        if ccp.split_parents_sorted.len() != n_splits {
            return Err(PreprocessError::InvalidInput(format!(
                "family {name:?} split_parents_sorted has length {} but expected {n_splits}",
                ccp.split_parents_sorted.len()
            )));
        }
        if ccp.log_split_probs_sorted.len() != n_splits {
            return Err(PreprocessError::InvalidInput(format!(
                "family {name:?} log_split_probs_sorted has length {} but expected {n_splits}",
                ccp.log_split_probs_sorted.len()
            )));
        }

        let lefts = &ccp.split_leftrights_sorted[..n_splits];
        let rights = &ccp.split_leftrights_sorted[n_splits..];
        for row in 0..end_rows_ge2 {
            ge2_left.push(lefts[row] + clade_offset);
            ge2_right.push(rights[row] + clade_offset);
            ge2_logp.push(ccp.log_split_probs_sorted[row] * INV_LN2);
            ge2_parents.push(ccp.split_parents_sorted[row] + clade_offset);
        }
        for row in end_rows_ge2..(end_rows_ge2 + num_eq1) {
            eq1_left.push(lefts[row] + clade_offset);
            eq1_right.push(rights[row] + clade_offset);
            eq1_logp.push(ccp.log_split_probs_sorted[row] * INV_LN2);
            eq1_parents.push(ccp.split_parents_sorted[row] + clade_offset);
        }

        for row in &family.leaf_row_index {
            leaf_row_index.push(*row + clade_offset);
        }
        leaf_col_index.extend(family.leaf_col_index.iter().copied());
        root_clade_ids.push(family.root_clade_id + clade_offset);
        family_clade_offsets.push(clade_offset);
        family_clade_counts.push(c);
        clade_offset += c;
        total_splits += n_splits;
    }

    let mut split_leftrights_sorted = Vec::with_capacity(2 * total_splits);
    split_leftrights_sorted.extend(ge2_left);
    split_leftrights_sorted.extend(eq1_left);
    split_leftrights_sorted.extend(ge2_right);
    split_leftrights_sorted.extend(eq1_right);

    let mut log_split_probs_sorted = Vec::with_capacity(total_splits);
    log_split_probs_sorted.extend(ge2_logp);
    log_split_probs_sorted.extend(eq1_logp);

    let mut split_parents_sorted = Vec::with_capacity(total_splits);
    split_parents_sorted.extend(ge2_parents);
    split_parents_sorted.extend(eq1_parents);

    Ok(CollatedFamilyBatch {
        c: clade_offset as usize,
        n_splits: total_splits,
        split_leftrights_sorted,
        split_parents_sorted,
        log_split_probs_sorted,
        leaf_row_index,
        leaf_col_index,
        root_clade_ids,
        family_clade_counts,
        family_clade_offsets,
    })
}

#[cfg(feature = "python-extension")]
fn build_one_fused_chunk(
    output: &PreprocessOutput,
    family_order: &[String],
    request: &ChunkedLayoutRequest,
    plan: FamilyBatchPlanOutput,
) -> Result<FusedChunkOutput, PreprocessError> {
    let collated = collate_family_batch(output, family_order, &plan.indices)?;
    let mut items = Vec::with_capacity(plan.indices.len());
    for index in &plan.indices {
        let name = &family_order[*index as usize];
        let family = output
            .families
            .get(name)
            .ok_or_else(|| PreprocessError::InvalidInput(format!("unknown family {name:?}")))?;
        items.push(scheduler::ScheduleItem {
            ccp: schedule_ccp_from_family(family),
        });
    }
    let schedule = scheduler::schedule_global_phased_waves(
        &items,
        &collated.family_clade_offsets,
        request.max_wave_size.map(|value| value as usize),
        request.max_root_wave_size,
        request.max_dts_partial_rows,
        request.dts_partial_tile_splits,
    )?;
    let layout = layout::build_wave_layout_plan(
        &schedule.waves,
        &schedule.phases,
        collated.c,
        collated.n_splits,
        &collated.split_leftrights_sorted,
        &collated.split_parents_sorted,
        &collated.leaf_row_index,
        &collated.leaf_col_index,
        &collated.root_clade_ids,
        Some(&collated.family_clade_counts),
        Some(&collated.family_clade_offsets),
    )?;

    let mut max_wave = 0i64;
    let mut split_rows = 0i64;
    let mut max_wave_split_rows = 0i64;
    for meta in &layout.wave_metas {
        max_wave = max_wave.max(meta.w);
        let rows = meta
            .sl
            .as_ref()
            .map(|values| values.len() as i64)
            .unwrap_or(0);
        split_rows += rows;
        max_wave_split_rows = max_wave_split_rows.max(rows);
    }

    Ok(FusedChunkOutput {
        indices: plan.indices,
        clades: plan.clades,
        splits: plan.splits,
        wave_layout: layout,
        log_split_probs_sorted: collated.log_split_probs_sorted,
        waves: schedule.waves.len() as i64,
        max_wave,
        split_rows,
        max_wave_split_rows,
    })
}

#[cfg(feature = "python-extension")]
fn build_fused_chunked_layouts(
    output: &PreprocessOutput,
    family_order: &[String],
    request: &ChunkedLayoutRequest,
) -> Result<Vec<FusedChunkOutput>, PreprocessError> {
    let (clade_counts, split_counts) = family_basic_counts(output, family_order)?;
    let needs_depth_stats = batch_packing_needs_depth_stats(&request.batch_packing);
    let (leaf_counts, nonleaf_counts, schedule_depths) = if needs_depth_stats {
        let (_, _, leaf_counts, nonleaf_counts, schedule_depths) =
            family_counts_and_summaries(output, family_order)?;
        (
            Some(leaf_counts),
            Some(nonleaf_counts),
            Some(schedule_depths),
        )
    } else {
        (None, None, None)
    };
    let plans = batch_planning::plan_family_batches(
        &clade_counts,
        request.family_chunk_size,
        request.clade_budget,
        &request.batch_packing,
        None,
        Some(family_order.len() as i64),
        Some(&split_counts),
        leaf_counts.as_deref(),
        nonleaf_counts.as_deref(),
        schedule_depths.as_deref(),
        request.max_wave_size,
    )?;

    let build = || {
        plans
            .into_par_iter()
            .map(|plan| build_one_fused_chunk(output, family_order, request, plan))
            .collect::<Result<Vec<_>, _>>()
    };
    if request.num_threads > 0 {
        get_thread_pool(request.num_threads)?.install(build)
    } else {
        build()
    }
}

#[cfg(feature = "python-extension")]
fn species_indexes_from_numpy(
    s_p_indexes: PyReadonlyArray1<'_, i64>,
    s_c12_indexes: PyReadonlyArray1<'_, i64>,
) -> (Vec<i64>, Vec<i64>) {
    (
        s_p_indexes.as_array().iter().copied().collect(),
        s_c12_indexes.as_array().iter().copied().collect(),
    )
}

#[cfg(feature = "python-extension")]
fn validate_species_index_lengths(s_p_indexes: &[i64], s_c12_indexes: &[i64]) -> PyResult<()> {
    if s_p_indexes.len() != s_c12_indexes.len() {
        return Err(PyValueError::new_err(
            "s_P_indexes and s_C12_indexes must have the same length",
        ));
    }
    Ok(())
}

#[cfg(feature = "python-extension")]
fn species_parent_from_indexes(
    s: usize,
    s_p_indexes: &[i64],
    s_c12_indexes: &[i64],
) -> PyResult<Vec<i64>> {
    validate_species_index_lengths(s_p_indexes, s_c12_indexes)?;
    let mut parent = vec![-1i64; s];
    for (&parent_code, &child) in s_p_indexes.iter().zip(s_c12_indexes.iter()) {
        if child < 0 || child as usize >= s {
            return Err(PyValueError::new_err(format!(
                "s_C12_indexes value {child} is outside valid range [0, {s})"
            )));
        }
        if parent_code < 0 || parent_code as usize >= 2 * s {
            return Err(PyValueError::new_err(format!(
                "s_P_indexes value {parent_code} is outside valid range [0, {})",
                2 * s
            )));
        }
        let parent_idx = if parent_code < s as i64 {
            parent_code
        } else {
            parent_code - s as i64
        };
        parent[child as usize] = parent_idx;
    }
    Ok(parent)
}

#[cfg(feature = "python-extension")]
fn compute_max_ancestor_depth(parent: &[i64]) -> PyResult<i64> {
    let s = parent.len();
    let mut max_ancestor_depth = 0usize;
    for s_idx in 0..s {
        let mut depth = 0usize;
        let mut cur = s_idx as i64;
        while cur >= 0 {
            depth += 1;
            if depth > s {
                return Err(PyRuntimeError::new_err(
                    "Cycle detected in species parent pointers",
                ));
            }
            cur = parent[cur as usize];
        }
        max_ancestor_depth = max_ancestor_depth.max(depth);
    }
    Ok(max_ancestor_depth as i64)
}

#[cfg(feature = "python-extension")]
fn species_helper_topology_from_indexes(
    s: usize,
    s_p_indexes: &[i64],
    s_c12_indexes: &[i64],
) -> PyResult<SpeciesHelperTopology> {
    validate_species_index_lengths(s_p_indexes, s_c12_indexes)?;
    let mut sp_child1_cpu = vec![s as i64; s];
    let mut sp_child2_cpu = vec![s as i64; s];
    for (&parent_code, &child) in s_p_indexes.iter().zip(s_c12_indexes.iter()) {
        if child < 0 || child as usize >= s {
            return Err(PyValueError::new_err(format!(
                "s_C12_indexes value {child} is outside valid range [0, {s})"
            )));
        }
        if parent_code < 0 || parent_code as usize >= 2 * s {
            return Err(PyValueError::new_err(format!(
                "s_P_indexes value {parent_code} is outside valid range [0, {})",
                2 * s
            )));
        }
        if parent_code < s as i64 {
            sp_child1_cpu[parent_code as usize] = child;
        } else {
            sp_child2_cpu[(parent_code - s as i64) as usize] = child;
        }
    }

    let sp_parent_cpu = species_parent_from_indexes(s, s_p_indexes, s_c12_indexes)?;
    let max_ancestor_depth = compute_max_ancestor_depth(&sp_parent_cpu)?;

    let mut levels = vec![-1i64; s];
    for s_idx in 0..s {
        if levels[s_idx] >= 0 {
            continue;
        }
        let mut stack = vec![(s_idx, false)];
        while let Some((node, expanded)) = stack.pop() {
            if levels[node] >= 0 {
                continue;
            }
            let c1 = sp_child1_cpu[node];
            let c2 = sp_child2_cpu[node];
            if !expanded {
                stack.push((node, true));
                if c2 >= 0 && (c2 as usize) < s && levels[c2 as usize] < 0 {
                    stack.push((c2 as usize, false));
                }
                if c1 >= 0 && (c1 as usize) < s && levels[c1 as usize] < 0 {
                    stack.push((c1 as usize, false));
                }
                continue;
            }
            let mut max_child_level = -1i64;
            if c1 >= 0 && (c1 as usize) < s {
                max_child_level = max_child_level.max(levels[c1 as usize]);
            }
            if c2 >= 0 && (c2 as usize) < s {
                max_child_level = max_child_level.max(levels[c2 as usize]);
            }
            levels[node] = if max_child_level >= 0 {
                max_child_level + 1
            } else {
                0
            };
        }
    }

    let max_level = levels.iter().copied().max().unwrap_or(0);
    let mut compact_level_ptr = vec![0i64];
    let mut compact_level_parents = Vec::new();
    let mut compact_level_child1 = Vec::new();
    let mut compact_level_child2 = Vec::new();
    for level in 1..=max_level {
        for (idx, &node_level) in levels.iter().enumerate() {
            if node_level == level
                && ((sp_child1_cpu[idx] as usize) < s || (sp_child2_cpu[idx] as usize) < s)
            {
                compact_level_parents.push(i32::try_from(idx).map_err(|_| {
                    PyValueError::new_err(format!("species parent index {idx} does not fit int32"))
                })?);
                compact_level_child1.push(i32::try_from(sp_child1_cpu[idx]).map_err(|_| {
                    PyValueError::new_err(format!(
                        "species child1 index {} does not fit int32",
                        sp_child1_cpu[idx]
                    ))
                })?);
                compact_level_child2.push(i32::try_from(sp_child2_cpu[idx]).map_err(|_| {
                    PyValueError::new_err(format!(
                        "species child2 index {} does not fit int32",
                        sp_child2_cpu[idx]
                    ))
                })?);
            }
        }
        compact_level_ptr.push(compact_level_parents.len() as i64);
    }
    if compact_level_ptr.len() == 1 {
        compact_level_ptr.push(0);
    }

    Ok(SpeciesHelperTopology {
        sp_child1_cpu,
        sp_child2_cpu,
        sp_parent_cpu,
        max_ancestor_depth,
        compact_level_ptr,
        compact_level_parents,
        compact_level_child1,
        compact_level_child2,
    })
}

#[cfg(feature = "python-extension")]
fn uniform_ancestor_index_pairs_from_indexes(
    s: usize,
    s_p_indexes: &[i64],
    s_c12_indexes: &[i64],
) -> PyResult<Vec<i64>> {
    let parent = species_parent_from_indexes(s, s_p_indexes, s_c12_indexes)?;
    let mut pairs = Vec::new();
    for desc in 0..s {
        let mut depth = 0usize;
        let mut cur = desc as i64;
        while cur >= 0 {
            pairs.push((cur, desc as i64));
            depth += 1;
            if depth > s {
                return Err(PyRuntimeError::new_err(
                    "Cycle detected in species parent pointers",
                ));
            }
            cur = parent[cur as usize];
        }
    }
    pairs.sort_unstable();

    let mut indices = Vec::with_capacity(2 * pairs.len());
    indices.extend(pairs.iter().map(|(row, _)| *row));
    indices.extend(pairs.into_iter().map(|(_, col)| col));
    Ok(indices)
}

#[cfg(feature = "python-extension")]
fn batch_packing_needs_depth_stats(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().replace('-', "_").as_str(),
        "depth_first_fit"
            | "depth_ffd"
            | "critical_path_first_fit"
            | "critical_first_fit"
            | "wave_first_fit"
    )
}

#[cfg(feature = "python-extension")]
#[pyclass]
struct PyPreprocessedDataset {
    output: PreprocessOutput,
    family_order: Vec<String>,
}

#[cfg(feature = "python-extension")]
#[pymethods]
impl PyPreprocessedDataset {
    fn family_basic_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (clade_counts, split_counts) = family_basic_counts(&self.output, &self.family_order)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let result = PyDict::new_bound(py);
        result.set_item("family_names", self.family_order.clone())?;
        result.set_item("clade_counts", clade_counts)?;
        result.set_item("split_counts", split_counts)?;
        Ok(result)
    }

    fn family_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (clade_counts, split_counts, leaf_counts, nonleaf_counts, schedule_depths) =
            family_counts_and_summaries(&self.output, &self.family_order)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let result = PyDict::new_bound(py);
        result.set_item("family_names", self.family_order.clone())?;
        result.set_item("clade_counts", clade_counts)?;
        result.set_item("split_counts", split_counts)?;
        result.set_item("leaf_counts", leaf_counts)?;
        result.set_item("nonleaf_counts", nonleaf_counts)?;
        result.set_item("schedule_depths", schedule_depths)?;
        Ok(result)
    }

    fn to_torch<'py>(
        &self,
        py: Python<'py>,
        from_numpy: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        output_to_torch_python(py, from_numpy, self.output.clone())
    }

    fn build_chunked_layouts_torch<'py>(
        &self,
        py: Python<'py>,
        request_json: &str,
        from_numpy: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let request: ChunkedLayoutRequest = serde_json::from_str(request_json)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let chunks = py
            .allow_threads(|| {
                build_fused_chunked_layouts(&self.output, &self.family_order, &request)
            })
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        fused_chunks_to_torch_python(py, from_numpy, chunks, &request.dtype)
    }
}

#[cfg(feature = "python-extension")]
#[pyfunction]
fn preprocess_dataset(py: Python<'_>, request_json: &str) -> PyResult<PyPreprocessedDataset> {
    let request: PreprocessDatasetRequest = serde_json::from_str(request_json)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let output = py
        .allow_threads(|| preprocess_request(&request.preprocess))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let family_order = ordered_family_names(&output, &request.family_order)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyPreprocessedDataset {
        output,
        family_order,
    })
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
fn species_parent_from_indexes_torch<'py>(
    py: Python<'py>,
    s: usize,
    s_p_indexes: PyReadonlyArray1<'py, i64>,
    s_c12_indexes: PyReadonlyArray1<'py, i64>,
    from_numpy: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let (s_p_indexes, s_c12_indexes) = species_indexes_from_numpy(s_p_indexes, s_c12_indexes);
    let parent = species_parent_from_indexes(s, &s_p_indexes, &s_c12_indexes)?;
    vec_i64_to_torch(py, from_numpy, parent)
}

#[cfg(feature = "python-extension")]
#[pyfunction]
fn species_wave_topology_torch<'py>(
    py: Python<'py>,
    s: usize,
    s_p_indexes: PyReadonlyArray1<'py, i64>,
    s_c12_indexes: PyReadonlyArray1<'py, i64>,
    from_numpy: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let (s_p_indexes, s_c12_indexes) = species_indexes_from_numpy(s_p_indexes, s_c12_indexes);
    let topology = species_helper_topology_from_indexes(s, &s_p_indexes, &s_c12_indexes)?;
    let result = PyDict::new_bound(py);
    result.set_item("S", s)?;
    result.set_item(
        "sp_child1_cpu",
        vec_i64_to_torch(py, from_numpy, topology.sp_child1_cpu)?,
    )?;
    result.set_item(
        "sp_child2_cpu",
        vec_i64_to_torch(py, from_numpy, topology.sp_child2_cpu)?,
    )?;
    result.set_item(
        "sp_parent_cpu",
        vec_i64_to_torch(py, from_numpy, topology.sp_parent_cpu)?,
    )?;
    result.set_item("max_ancestor_depth", topology.max_ancestor_depth)?;
    result.set_item(
        "compact_level_ptr",
        vec_i64_to_torch(py, from_numpy, topology.compact_level_ptr)?,
    )?;
    result.set_item(
        "compact_level_parents",
        vec_i32_to_torch(py, from_numpy, topology.compact_level_parents)?,
    )?;
    result.set_item(
        "compact_level_child1",
        vec_i32_to_torch(py, from_numpy, topology.compact_level_child1)?,
    )?;
    result.set_item(
        "compact_level_child2",
        vec_i32_to_torch(py, from_numpy, topology.compact_level_child2)?,
    )?;
    Ok(result)
}

#[cfg(feature = "python-extension")]
#[pyfunction]
fn uniform_ancestors_t_indices_torch<'py>(
    py: Python<'py>,
    s: usize,
    s_p_indexes: PyReadonlyArray1<'py, i64>,
    s_c12_indexes: PyReadonlyArray1<'py, i64>,
    from_numpy: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let (s_p_indexes, s_c12_indexes) = species_indexes_from_numpy(s_p_indexes, s_c12_indexes);
    let indices = uniform_ancestor_index_pairs_from_indexes(s, &s_p_indexes, &s_c12_indexes)?;
    let cols = indices.len() / 2;
    vec_i64_matrix_to_torch(py, from_numpy, indices, 2, cols)
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
#[pyfunction]
fn build_wave_layout_plan_json(request_json: &str) -> PyResult<String> {
    let request: WaveLayoutRequest = serde_json::from_str(request_json)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let output = build_wave_layout_plan_request(&request)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    serde_json::to_string(&output).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

#[cfg(feature = "python-extension")]
#[pymodule]
fn gpurec_preprocess(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPreprocessedDataset>()?;
    module.add_function(wrap_pyfunction!(preprocess_dataset, module)?)?;
    module.add_function(wrap_pyfunction!(preprocess_request_binary, module)?)?;
    module.add_function(wrap_pyfunction!(preprocess_request_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(preprocess_request_torch, module)?)?;
    module.add_function(wrap_pyfunction!(species_parent_from_indexes_torch, module)?)?;
    module.add_function(wrap_pyfunction!(species_wave_topology_torch, module)?)?;
    module.add_function(wrap_pyfunction!(uniform_ancestors_t_indices_torch, module)?)?;
    module.add_function(wrap_pyfunction!(schedule_global_phased_waves_json, module)?)?;
    module.add_function(wrap_pyfunction!(family_schedule_summary_json, module)?)?;
    module.add_function(wrap_pyfunction!(plan_family_batches_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_wave_layout_plan_json, module)?)?;
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
fn fused_chunks_to_torch_python<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    chunks: Vec<FusedChunkOutput>,
    dtype: &str,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty_bound(py);
    for chunk in chunks {
        let item = PyDict::new_bound(py);
        item.set_item("indices", chunk.indices)?;
        item.set_item("clades", chunk.clades)?;
        item.set_item("splits", chunk.splits)?;
        item.set_item("waves", chunk.waves)?;
        item.set_item("max_wave", chunk.max_wave)?;
        item.set_item("split_rows", chunk.split_rows)?;
        item.set_item("max_wave_split_rows", chunk.max_wave_split_rows)?;
        item.set_item(
            "wave_layout",
            wave_layout_to_torch_python(
                py,
                from_numpy,
                chunk.wave_layout,
                chunk.log_split_probs_sorted,
                dtype,
            )?,
        )?;
        list.append(item)?;
    }
    Ok(list)
}

#[cfg(feature = "python-extension")]
fn wave_layout_to_torch_python<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    plan: WaveLayoutPlan,
    log_split_probs_sorted: Vec<f64>,
    dtype: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new_bound(py);
    result.set_item("perm", vec_i64_to_torch(py, from_numpy, plan.perm)?)?;
    result.set_item("C", plan.c)?;
    result.set_item(
        "leaf_row_index",
        vec_i64_to_torch(py, from_numpy, plan.leaf_row_index)?,
    )?;
    result.set_item(
        "leaf_species_index",
        vec_i64_to_torch(py, from_numpy, plan.leaf_species_index)?,
    )?;
    result.set_item(
        "root_clade_ids",
        vec_i64_to_torch(py, from_numpy, plan.root_clade_ids)?,
    )?;
    result.set_item("root_clade_ids_cpu", plan.root_clade_ids_cpu)?;
    if let Some(family_idx) = plan.family_idx {
        result.set_item("family_idx", vec_i64_to_torch(py, from_numpy, family_idx)?)?;
    }

    let metas = PyList::empty_bound(py);
    for meta_plan in plan.wave_metas {
        let meta = PyDict::new_bound(py);
        meta.set_item("start", meta_plan.start)?;
        meta.set_item("end", meta_plan.end)?;
        meta.set_item("W", meta_plan.w)?;
        meta.set_item("has_splits", meta_plan.has_splits)?;
        meta.set_item("phase", meta_plan.phase)?;
        if meta_plan.has_splits {
            let split_indices = meta_plan.split_indices.unwrap_or_default();
            let log_probs = gather_log_split_probs(&log_split_probs_sorted, &split_indices)?;
            meta.set_item(
                "sl",
                vec_i32_to_torch(
                    py,
                    from_numpy,
                    i64s_to_i32s("sl", meta_plan.sl.unwrap_or_default())?,
                )?,
            )?;
            meta.set_item(
                "sr",
                vec_i32_to_torch(
                    py,
                    from_numpy,
                    i64s_to_i32s("sr", meta_plan.sr.unwrap_or_default())?,
                )?,
            )?;
            meta.set_item(
                "log_split_probs",
                log_probs_to_torch(py, from_numpy, log_probs, dtype)?,
            )?;
            meta.set_item(
                "reduce_idx",
                vec_i32_to_torch(
                    py,
                    from_numpy,
                    i64s_to_i32s("reduce_idx", meta_plan.reduce_idx.unwrap_or_default())?,
                )?,
            )?;
            meta.set_item("n_eq1", meta_plan.n_eq1.unwrap_or(0))?;
            if let Some(values) = meta_plan.eq1_reduce_idx {
                meta.set_item(
                    "eq1_reduce_idx",
                    vec_i32_to_torch(py, from_numpy, i64s_to_i32s("eq1_reduce_idx", values)?)?,
                )?;
            }
            if let Some(values) = meta_plan.ge2_ptr {
                meta.set_item("ge2_ptr", vec_i64_to_torch(py, from_numpy, values)?)?;
            }
            if let Some(values) = meta_plan.ge2_parent_ids {
                meta.set_item(
                    "ge2_parent_ids",
                    vec_i32_to_torch(py, from_numpy, i64s_to_i32s("ge2_parent_ids", values)?)?,
                )?;
            }
            if let Some(value) = meta_plan.ge2_max_fanout {
                meta.set_item("ge2_max_fanout", value)?;
            }
        }
        metas.append(meta)?;
    }
    result.set_item("wave_metas", metas)?;
    Ok(result)
}

#[cfg(feature = "python-extension")]
fn gather_log_split_probs(values: &[f64], split_indices: &[i64]) -> PyResult<Vec<f64>> {
    let mut gathered = Vec::with_capacity(split_indices.len());
    for split_idx in split_indices {
        if *split_idx < 0 || (*split_idx as usize) >= values.len() {
            return Err(PyRuntimeError::new_err(format!(
                "split index {split_idx} outside log_split_probs length {}",
                values.len()
            )));
        }
        gathered.push(values[*split_idx as usize]);
    }
    Ok(gathered)
}

#[cfg(feature = "python-extension")]
fn i64s_to_i32s(name: &str, values: Vec<i64>) -> PyResult<Vec<i32>> {
    values
        .into_iter()
        .map(|value| {
            i32::try_from(value).map_err(|_| {
                PyValueError::new_err(format!("{name} value {value} does not fit int32"))
            })
        })
        .collect()
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
fn vec_i32_to_torch<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    values: Vec<i32>,
) -> PyResult<Py<PyAny>> {
    let array = values.into_pyarray_bound(py);
    from_numpy.call1((array,)).map(Bound::unbind)
}

#[cfg(feature = "python-extension")]
fn vec_i64_matrix_to_torch<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    values: Vec<i64>,
    rows: usize,
    cols: usize,
) -> PyResult<Py<PyAny>> {
    let array = Array2::from_shape_vec((rows, cols), values)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray_bound(py);
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
fn vec_f32_matrix_to_torch<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    values: Vec<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<Py<PyAny>> {
    let array = Array2::from_shape_vec((rows, cols), values)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .into_pyarray_bound(py);
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
fn log_probs_to_torch<'py>(
    py: Python<'py>,
    from_numpy: &Bound<'py, PyAny>,
    values: Vec<f64>,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let rows = values.len();
    match dtype.trim().to_ascii_lowercase().as_str() {
        "float32" | "fp32" | "single" | "torch.float32" => {
            let values = values.into_iter().map(|value| value as f32).collect();
            vec_f32_matrix_to_torch(py, from_numpy, values, rows, 1)
        }
        "float64" | "fp64" | "double" | "torch.float64" => {
            vec_f64_matrix_to_torch(py, from_numpy, values, rows, 1)
        }
        other => Err(PyValueError::new_err(format!(
            "dtype must be float32 or float64, got {other:?}"
        ))),
    }
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

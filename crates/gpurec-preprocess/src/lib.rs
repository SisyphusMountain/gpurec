use rayon::prelude::*;
use rustree::{parse_newick, FlatTree};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::Path;

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub mod batch_planning;
pub mod layout;
pub mod scheduler;

const BITS_PER_WORD: usize = 64;
const JSON_SCHEMA_VERSION: u64 = 1;

#[derive(Debug)]
pub enum PreprocessError {
    Io(String),
    InvalidInput(String),
    Parse(String),
}

impl fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PreprocessError::Io(message) => write!(f, "{message}"),
            PreprocessError::InvalidInput(message) => write!(f, "invalid input: {message}"),
            PreprocessError::Parse(message) => write!(f, "failed to parse Newick: {message}"),
        }
    }
}

impl std::error::Error for PreprocessError {}

impl From<PreprocessError> for PyErr {
    fn from(err: PreprocessError) -> Self {
        match err {
            PreprocessError::Io(message) => PyOSError::new_err(message),
            PreprocessError::InvalidInput(message) => PyValueError::new_err(message),
            PreprocessError::Parse(message) => PyValueError::new_err(message),
        }
    }
}

fn species_subtree_intervals(root: usize, child1: &[i32], child2: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let s = child1.len();
    let mut start = vec![0i32; s];
    let mut end = vec![0i32; s];
    let mut cursor = 0i32;
    let mut stack = vec![(root, false)];
    while let Some((node, exiting)) = stack.pop() {
        if exiting {
            end[node] = cursor;
            continue;
        }
        start[node] = cursor;
        cursor += 1;
        stack.push((node, true));
        for child in [child2[node], child1[node]] {
            if child < s as i32 {
                stack.push((child as usize, false));
            }
        }
    }
    (start, end)
}

#[pyfunction]
#[pyo3(signature = (species_path, families, family_chunk_size=None, clade_budget=None, batch_packing=None, max_wave_size=8192))]
fn preprocess_dataset(
    py: Python<'_>,
    species_path: String,
    families: Vec<String>,
    family_chunk_size: Option<usize>,
    clade_budget: Option<usize>,
    batch_packing: Option<String>,
    max_wave_size: usize,
) -> PyResult<String> {
    let output = py.allow_threads(|| {
        let species_tree = parse_one_newick_file(Path::new(&species_path))?;
        let (species, species_name_to_index) = build_species_output(&species_tree);
        let family_outputs: Result<Vec<Value>, PreprocessError> = families
            .par_iter()
            .map(|gene_path| preprocess_one_family(Path::new(gene_path), &species_name_to_index))
            .map(|result| result.map_err(PreprocessError::InvalidInput))
            .collect();
        family_outputs.and_then(|families| {
            let (batches, batch_wave_layouts) = plan_batches_and_layouts(
                &families,
                family_chunk_size,
                clade_budget,
                batch_packing.as_deref().unwrap_or("depth_first_fit"),
                max_wave_size,
            )
            .map_err(PreprocessError::InvalidInput)?;
            Ok(json!({
                "schema_version": JSON_SCHEMA_VERSION,
                "species": species,
                "families": families,
                "batches": batches,
                "batch_wave_layouts": batch_wave_layouts,
            }))
        })
    });
    output.map(|value| value.to_string()).map_err(PyErr::from)
}

#[pymodule]
fn gpurec_preprocess(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(preprocess_dataset, module)?)?;
    module.add_function(wrap_pyfunction!(plan_batch_layouts, module)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (families_json, family_chunk_size=None, clade_budget=None, batch_packing=None, max_wave_size=8192))]
fn plan_batch_layouts(
    families_json: String,
    family_chunk_size: Option<usize>,
    clade_budget: Option<usize>,
    batch_packing: Option<String>,
    max_wave_size: usize,
) -> PyResult<String> {
    let families: Vec<Value> = serde_json::from_str(&families_json)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let (batches, batch_wave_layouts) = plan_batches_and_layouts(
        &families,
        family_chunk_size,
        clade_budget,
        batch_packing.as_deref().unwrap_or("depth_first_fit"),
        max_wave_size,
    )
    .map_err(PyRuntimeError::new_err)?;
    Ok(json!({
        "schema_version": JSON_SCHEMA_VERSION,
        "batches": batches,
        "batch_wave_layouts": batch_wave_layouts,
    })
    .to_string())
}

fn parse_one_newick_file(path: &Path) -> Result<FlatTree, PreprocessError> {
    let text = fs::read_to_string(path)
        .map_err(|err| PreprocessError::Io(format!("{}: {err}", path.display())))?;
    let mut roots = parse_newick(text.trim())
        .map_err(|err| PreprocessError::Parse(format!("{}: {err}", path.display())))?;
    let root = roots
        .drain(..)
        .next()
        .ok_or_else(|| PreprocessError::Parse(format!("{}: no tree found", path.display())))?;
    Ok(root.to_flat_tree())
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
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SplitKey {
    parent: usize,
    left: usize,
    right: usize,
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

    fn parse(mut self, path: &Path) -> Result<GeneTree, String> {
        let root = self.parse_subtree(path)?;
        self.skip_whitespace();
        if self.pos != self.text.len() {
            return Err(format!(
                "{}: unexpected trailing Newick text",
                path.display()
            ));
        }
        Ok(GeneTree {
            nodes: self.nodes,
            root,
        })
    }

    fn parse_subtree(&mut self, path: &Path) -> Result<usize, String> {
        self.skip_whitespace();
        let node = self.make_node();
        if self.peek_byte() == Some(b'(') {
            self.pos += 1;
            loop {
                let child = self.parse_subtree(path)?;
                self.nodes[child].parent = Some(node);
                self.nodes[node].children.push(child);
                self.skip_whitespace();
                match self.peek_byte() {
                    Some(b',') => self.pos += 1,
                    Some(b')') => {
                        self.pos += 1;
                        break;
                    }
                    _ => return Err(format!("{}: expected ',' or ')'", path.display())),
                }
            }
        }
        self.nodes[node].name = self.parse_label();
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

fn parse_gene_newick_file_records(path: &Path) -> Result<Vec<GeneTree>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("{}: {err}", path.display()))?;
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
        return Err(format!("{}: no Newick tree records found", path.display()));
    }
    Ok(trees)
}

fn binarize_gene_tree(tree: &mut GeneTree, path: &Path) -> Result<(), String> {
    binarize_gene_node(tree, tree.root, path)
}

fn binarize_gene_node(tree: &mut GeneTree, node: usize, path: &Path) -> Result<(), String> {
    let children = tree.nodes[node].children.clone();
    for child in children {
        binarize_gene_node(tree, child, path)?;
    }
    if tree.nodes[node].children.len() == 1 {
        return Err(format!(
            "{}: unary gene nodes are not supported",
            path.display()
        ));
    }
    while tree.nodes[node].children.len() > 2 {
        let right = tree.nodes[node].children.pop().ok_or_else(|| {
            format!(
                "{}: missing right child during binarization",
                path.display()
            )
        })?;
        let left = tree.nodes[node]
            .children
            .pop()
            .ok_or_else(|| format!("{}: missing left child during binarization", path.display()))?;
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

fn build_species_output(tree: &FlatTree) -> (Value, BTreeMap<String, usize>) {
    let postorder = tree.postorder_indices();
    let s = postorder.len();
    let sentinel = s as i32;
    let mut node_to_gp = vec![usize::MAX; tree.nodes.len()];
    for (gp_idx, &node_idx) in postorder.iter().enumerate() {
        node_to_gp[node_idx] = gp_idx;
    }

    let mut sp_child1 = vec![sentinel; s];
    let mut sp_child2 = vec![sentinel; s];
    let mut sp_parent = vec![-1i32; s];
    let mut species_name_to_index = BTreeMap::new();
    for (gp_idx, &node_idx) in postorder.iter().enumerate() {
        let node = &tree.nodes[node_idx];
        species_name_to_index.insert(node.name.clone(), gp_idx);
        if let Some(parent_idx) = node.parent {
            sp_parent[gp_idx] = node_to_gp[parent_idx] as i32;
        }
        if let Some((left, right)) = node.left_child.zip(node.right_child) {
            sp_child1[gp_idx] = node_to_gp[left] as i32;
            sp_child2[gp_idx] = node_to_gp[right] as i32;
        }
    }

    let mut max_ancestor_depth = 0usize;
    let mut unnorm_row_max = Vec::with_capacity(s);
    for idx in 0..s {
        let mut depth = 0usize;
        let mut cur = idx as i32;
        while cur >= 0 {
            depth += 1;
            cur = sp_parent[cur as usize];
        }
        max_ancestor_depth = max_ancestor_depth.max(depth);
        unnorm_row_max.push(match s.saturating_sub(depth) {
            0 => f64::NEG_INFINITY,
            recipients => -(recipients as f64).log2(),
        });
    }
    let (sp_subtree_start, sp_subtree_end) =
        species_subtree_intervals(node_to_gp[tree.root], &sp_child1, &sp_child2);

    let mut levels = vec![0i64; s];
    for idx in 0..s {
        levels[idx] = [sp_child1[idx], sp_child2[idx]]
            .into_iter()
            .filter(|&child| child < sentinel)
            .map(|child| levels[child as usize] + 1)
            .max()
            .unwrap_or(0);
    }

    let max_level = levels.iter().copied().max().unwrap_or(0);
    let mut compact_level_ptr = vec![0i64];
    let mut compact_level_parents = Vec::new();
    let mut compact_level_child1 = Vec::new();
    let mut compact_level_child2 = Vec::new();
    for level in 1..=max_level {
        for (idx, &node_level) in levels.iter().enumerate() {
            if node_level == level {
                compact_level_parents.push(idx as i32);
                compact_level_child1.push(sp_child1[idx]);
                compact_level_child2.push(sp_child2[idx]);
            }
        }
        compact_level_ptr.push(compact_level_parents.len() as i64);
    }
    (
        json!({
            "S": s,
            "unnorm_row_max": unnorm_row_max,
            "sp_child1": sp_child1,
            "sp_child2": sp_child2,
            "sp_parent": sp_parent,
            "sp_subtree_start": sp_subtree_start,
            "sp_subtree_end": sp_subtree_end,
            "max_ancestor_depth": max_ancestor_depth as i64,
            "compact_level_ptr": compact_level_ptr,
            "compact_level_parents": compact_level_parents,
            "compact_level_child1": compact_level_child1,
            "compact_level_child2": compact_level_child2,
        }),
        species_name_to_index,
    )
}

fn preprocess_one_family(
    gene_path: &Path,
    species_name_to_index: &BTreeMap<String, usize>,
) -> Result<Value, String> {
    let (clade_data, leaf_names) = amalgamate_clades_and_splits(gene_path)?;
    let ccp = build_ccp_arrays(&clade_data);
    let c = clade_data.clades.clades.len();
    let schedule_depth = clade_schedule_depth(&clade_data);
    let mut leaf_row_index = Vec::new();
    let mut leaf_col_index = Vec::new();

    for cid in 0..c {
        let clade = &clade_data.clades.clades[cid];
        if clade.size != 1 {
            continue;
        }
        if let Some(leaf_idx) = first_set_bit(&clade.bits) {
            let leaf_name = &leaf_names[leaf_idx];
            let species = leaf_name
                .split_once('_')
                .map_or(leaf_name.as_str(), |(species, _)| species);
            let species_idx = species_name_to_index.get(species).ok_or_else(|| {
                format!(
                    "species {species:?} not found for gene leaf {:?}",
                    leaf_name
                )
            })?;
            leaf_row_index.push(cid as i64);
            leaf_col_index.push(*species_idx as i64);
        }
    }

    Ok(json!({
        "split_leftrights_sorted": ccp.split_leftrights_sorted,
        "split_parents_sorted": ccp.split_parents_sorted,
        "log_split_probs_sorted": ccp.log_split_probs_sorted,
        "split_counts": ccp.split_counts,
        "N_splits": clade_data.splits.len() as i64,
        "root_clade_id": clade_data.root_clade_id as i64,
        "C": c as i64,
        "schedule_depth": schedule_depth as i64,
        "leaf_row_index": leaf_row_index,
        "leaf_col_index": leaf_col_index,
    }))
}

fn clade_schedule_depth(clade_data: &CladeData) -> usize {
    let c = clade_data.clades.clades.len();
    let mut levels = vec![-1isize; c];
    for (idx, clade) in clade_data.clades.clades.iter().enumerate() {
        if clade.size == 1 {
            levels[idx] = 0;
        }
    }
    for _ in 0..c {
        let mut changed = false;
        for split in &clade_data.splits {
            let child_level = levels[split.left].max(levels[split.right]);
            if child_level >= 0 && levels[split.parent] < child_level + 1 {
                levels[split.parent] = child_level + 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    levels.into_iter().max().unwrap_or(0).max(0) as usize
}

fn plan_batches_and_layouts(
    families: &[Value],
    family_chunk_size: Option<usize>,
    clade_budget: Option<usize>,
    batch_packing: &str,
    max_wave_size: usize,
) -> Result<(Vec<Vec<usize>>, Vec<Value>), String> {
    let effective_batch_packing = if clade_budget.is_none()
        && matches!(batch_packing, "depth_first_fit" | "clade_first_fit")
    {
        "sequential"
    } else {
        batch_packing
    };
    let clade_counts = families
        .iter()
        .map(|family| value_i64(family, "C"))
        .collect::<Result<Vec<_>, _>>()?;
    let split_counts_total = families
        .iter()
        .map(|family| value_i64(family, "N_splits"))
        .collect::<Result<Vec<_>, _>>()?;
    let leaf_counts = families
        .iter()
        .map(|family| value_vec_i64(family, "leaf_row_index").map(|rows| rows.len() as i64))
        .collect::<Result<Vec<_>, _>>()?;
    let nonleaf_counts = families
        .iter()
        .map(|family| {
            value_vec_i64(family, "split_counts")
                .map(|counts| counts.into_iter().filter(|count| *count > 0).count() as i64)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let schedule_depths = families
        .iter()
        .map(|family| value_i64(family, "schedule_depth"))
        .collect::<Result<Vec<_>, _>>()?;
    let plans = batch_planning::plan_family_batches(
        &clade_counts,
        family_chunk_size.unwrap_or(0) as i64,
        clade_budget.map(|value| value as i64),
        effective_batch_packing,
        None,
        Some(families.len() as i64),
        Some(&split_counts_total),
        Some(&leaf_counts),
        Some(&nonleaf_counts),
        Some(&schedule_depths),
        Some(max_wave_size as i64),
    )
    .map_err(|err| err.to_string())?;

    let batches = plans
        .into_iter()
        .map(|plan| {
            plan.indices
                .into_iter()
                .map(|index| index as usize)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let layouts = batches
        .iter()
        .map(|batch| batch_wave_layout(families, batch, max_wave_size))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((batches, layouts))
}

fn value_i64(family: &Value, key: &str) -> Result<i64, String> {
    family[key]
        .as_i64()
        .ok_or_else(|| format!("family field {key:?} is missing or not an integer"))
}

fn value_vec_i64(family: &Value, key: &str) -> Result<Vec<i64>, String> {
    family[key]
        .as_array()
        .ok_or_else(|| format!("family field {key:?} is missing or not an array"))?
        .iter()
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| format!("family field {key:?} contains a non-integer"))
        })
        .collect()
}

fn value_vec_f64(family: &Value, key: &str) -> Result<Vec<f64>, String> {
    family[key]
        .as_array()
        .ok_or_else(|| format!("family field {key:?} is missing or not an array"))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| format!("family field {key:?} contains a non-number"))
        })
        .collect()
}

fn batch_wave_layout(
    families: &[Value],
    batch: &[usize],
    max_wave_size: usize,
) -> Result<Value, String> {
    let mut offsets = Vec::with_capacity(batch.len());
    let mut counts = Vec::with_capacity(batch.len());
    let mut items = Vec::with_capacity(batch.len());
    let mut parents = Vec::new();
    let mut lefts = Vec::new();
    let mut rights = Vec::new();
    let mut log_split_probs = Vec::new();
    let mut leaf_rows = Vec::new();
    let mut leaf_cols = Vec::new();
    let mut root_ids = Vec::new();
    let mut offset = 0i64;

    for index in batch {
        let family = &families[*index];
        let c = value_i64(family, "C")?;
        let n = value_i64(family, "N_splits")? as usize;
        let split_counts = value_vec_i64(family, "split_counts")?;
        let split_parents = value_vec_i64(family, "split_parents_sorted")?;
        let leftrights = value_vec_i64(family, "split_leftrights_sorted")?;
        let logp = value_vec_f64(family, "log_split_probs_sorted")?;
        if split_parents.len() != n || leftrights.len() != 2 * n || logp.len() != n {
            return Err(format!("family {index} has inconsistent split arrays"));
        }

        offsets.push(offset);
        counts.push(c);
        items.push(scheduler::ScheduleItem {
            ccp: scheduler::ScheduleCcp {
                c: c as usize,
                n_splits: n,
                split_counts: Some(split_counts),
                split_parents_sorted: split_parents.clone(),
                split_leftrights_sorted: leftrights.clone(),
                root_clade_id: value_i64(family, "root_clade_id")?,
            },
        });
        parents.extend(split_parents.into_iter().map(|value| value + offset));
        lefts.extend(leftrights[..n].iter().map(|value| *value + offset));
        rights.extend(leftrights[n..].iter().map(|value| *value + offset));
        log_split_probs.extend(logp);
        leaf_rows.extend(
            value_vec_i64(family, "leaf_row_index")?
                .into_iter()
                .map(|row| row + offset),
        );
        leaf_cols.extend(value_vec_i64(family, "leaf_col_index")?);
        root_ids.push(value_i64(family, "root_clade_id")? + offset);
        offset += c;
    }

    let schedule = scheduler::schedule_global_phased_waves(
        &items,
        &offsets,
        Some(max_wave_size),
        None,
        None,
        scheduler::default_dts_partial_tile_splits(),
    )
    .map_err(|err| err.to_string())?;
    let mut leftrights = lefts;
    leftrights.extend(rights);
    let plan = layout::build_wave_layout_plan(
        &schedule.waves,
        &schedule.phases,
        offset as usize,
        parents.len(),
        &leftrights,
        &parents,
        &leaf_rows,
        &leaf_cols,
        &root_ids,
        Some(&counts),
        Some(&offsets),
    )
    .map_err(|err| err.to_string())?;

    Ok(json!({
        "family_indices": batch,
        "plan": plan,
        "log_split_probs_sorted": log_split_probs,
    }))
}

fn amalgamate_clades_and_splits(gene_path: &Path) -> Result<(CladeData, Vec<String>), String> {
    let trees = parse_gene_newick_file_records(gene_path)?;
    let mut all_leaves = BTreeSet::new();
    for tree in &trees {
        collect_leaf_names(tree, &mut all_leaves);
    }
    let leaf_names: Vec<String> = all_leaves.into_iter().collect();
    if leaf_names.is_empty() {
        return Err(format!("{}: no leaves found", gene_path.display()));
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
    for tree in &trees {
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
) -> Result<(), String> {
    let postorder = gene_postorder(tree);
    let mut node_clades = vec![vec![0; num_words]; tree.nodes.len()];
    let mut node_clade_ids = vec![usize::MAX; tree.nodes.len()];
    let mut node_above_ids = vec![None; tree.nodes.len()];

    for &node_idx in &postorder {
        let node = &tree.nodes[node_idx];
        let bits = match node.children.as_slice() {
            [] => {
                let leaf_idx = leaf_to_index
                    .get(&node.name)
                    .ok_or_else(|| format!("unknown gene leaf {:?}", node.name))?;
                let mut bits = vec![0; num_words];
                set_bit(&mut bits, *leaf_idx);
                bits
            }
            [left, right] => bit_or(&node_clades[*left], &node_clades[*right]),
            _ => return Err("gene tree was not binarized".to_string()),
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
            let left_plus_id =
                node_above_ids[*right_idx].ok_or_else(|| "missing left-plus clade".to_string())?;
            let right_plus_id =
                node_above_ids[*left_idx].ok_or_else(|| "missing right-plus clade".to_string())?;
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
    let mut sum_weights = vec![0.0; c];
    for split in &data.splits {
        split_counts[split.parent] += 1;
        sum_weights[split.parent] += split.weight;
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
        log_split_probs_sorted.push((split.weight / sum_weights[split.parent]).log2());
    }

    let mut split_leftrights_sorted = split_lefts_sorted;
    split_leftrights_sorted.extend(split_rights_sorted);
    CcpArrays {
        split_counts,
        split_parents_sorted,
        split_leftrights_sorted,
        log_split_probs_sorted,
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
        self.clades.push(Clade {
            size: bit_count(&bits),
            bits: bits.clone(),
        });
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
        .then_with(|| left.bits.cmp(&right.bits))
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
    bits[index >> 6] |= 1u64 << (index & 63);
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

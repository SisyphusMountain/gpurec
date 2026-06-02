use rustree::{parse_newick, FlatTree};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use pyo3::prelude::*;

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
fn preprocess_dataset(py: Python<'_>, species_path: String, families: Vec<String>) -> String {
    let output = py.allow_threads(|| {
        let species_tree = parse_one_newick_file(Path::new(&species_path));
        let (species, species_name_to_index) = build_species_output(&species_tree);
        let family_outputs: Vec<Value> = families
            .iter()
            .map(|gene_path| preprocess_one_family(Path::new(gene_path), &species_name_to_index))
            .collect();
        json!({ "species": species, "families": family_outputs })
    });
    output.to_string()
}

#[pymodule]
fn gpurec_preprocess(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(preprocess_dataset, module)?)?;
    Ok(())
}

fn parse_one_newick_file(path: &Path) -> FlatTree {
    let text = fs::read_to_string(path).unwrap();
    let mut roots = parse_newick(text.trim()).unwrap();
    roots.remove(0).to_flat_tree()
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
) -> Value {
    let tree = parse_one_newick_file(gene_path);
    let postorder = tree.postorder_indices();
    let c = postorder.len();
    let mut node_to_clade = vec![usize::MAX; tree.nodes.len()];
    for (clade_id, &node_idx) in postorder.iter().enumerate() {
        node_to_clade[node_idx] = clade_id;
    }

    let mut leaf_row_index = Vec::new();
    let mut leaf_col_index = Vec::new();
    let mut split_lefts_sorted = Vec::new();
    let mut split_rights_sorted = Vec::new();

    for (clade_id, &node_idx) in postorder.iter().enumerate() {
        let node = &tree.nodes[node_idx];
        if let Some((left, right)) = node.left_child.zip(node.right_child) {
            split_lefts_sorted.push(node_to_clade[left] as i64);
            split_rights_sorted.push(node_to_clade[right] as i64);
        } else {
            let species = node
                .name
                .split_once('_')
                .map_or(node.name.as_str(), |(species, _)| species);
            let species_idx = species_name_to_index[species];
            leaf_row_index.push(clade_id as i64);
            leaf_col_index.push(species_idx as i64);
        };
    }
    let mut split_leftrights_sorted = split_lefts_sorted;
    split_leftrights_sorted.extend(split_rights_sorted);

    json!({
        "split_leftrights_sorted": split_leftrights_sorted,
        "C": c as i64,
        "leaf_row_index": leaf_row_index,
        "leaf_col_index": leaf_col_index,
    })
}

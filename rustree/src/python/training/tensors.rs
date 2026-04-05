//! Per-sample tensor construction for training.
//!
//! Contains TaskTensors struct, build_task_tensors_internal helper,
//! and the build_training_tensors pyfunction.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

use super::extraction::RustBaseSample;

/// Per-sample task tensors (before collation).
pub(super) struct TaskTensors {
    pub(super) x_gene: Vec<i32>,
    pub(super) g_true_sp: Vec<i32>,   // 1-based species IDs
    pub(super) event_true: Vec<i32>,
    pub(super) event_input: Vec<i32>,
    pub(super) frontier_mask: Vec<u8>,
    pub(super) is_leaf: Vec<u8>,
    pub(super) mask_label_node: Vec<u8>,
    pub(super) g_edge_src: Vec<i32>,
    pub(super) g_edge_dst: Vec<i32>,
    pub(super) g_dir_src: Vec<i32>,
    pub(super) g_dir_dst: Vec<i32>,
    pub(super) root_edge_target: i32,
}

/// Build task tensors for one sample (internal; mirrors build_training_tensors logic).
/// sp_name_to_id maps species name (owned) -> 1-based ID.
pub(super) fn build_task_tensors_internal(
    sample: &RustBaseSample,
    sp_name_to_id: &HashMap<String, i32>,
    rng: &mut StdRng,
    force_mask_last_added: bool,
    predict_root_position: bool,
    sample_order: &str,
) -> Result<TaskTensors, String> {
    use rand::Rng;
    use rand::seq::SliceRandom;
    use std::collections::{HashSet, VecDeque};

    let g_root_name = &sample.g_root_name;
    let nb_g_leaves = sample.nb_g_leaves;
    let g_neighbors = &sample.g_neighbors;
    let g_leaves_names = &sample.g_leaves_names;
    let true_root = &sample.true_root;
    let true_states = &sample.true_states;
    let true_events = &sample.true_events;

    let max_to_sample = if !force_mask_last_added || predict_root_position {
        nb_g_leaves.saturating_sub(2)
    } else {
        nb_g_leaves.saturating_sub(1)
    };
    let nb_sampled = if predict_root_position {
        max_to_sample
    } else {
        rng.gen_range(0..=max_to_sample)
    };

    // Unroot tree: remove root node and reconnect its children
    let mut g_neighbors_unrooted: HashMap<String, Vec<String>> = HashMap::new();
    for (k, v) in g_neighbors {
        if k != g_root_name {
            let filtered: Vec<String> = v.iter()
                .filter(|n| n.as_str() != g_root_name.as_str())
                .cloned()
                .collect();
            g_neighbors_unrooted.insert(k.clone(), filtered);
        }
    }
    if let Some(root_nbrs) = g_neighbors.get(g_root_name) {
        // Connect all former root children to each other (handles bifurcating
        // and multifurcating roots alike).
        for i in 0..root_nbrs.len() {
            for j in (i + 1)..root_nbrs.len() {
                let ci = root_nbrs[i].clone();
                let cj = root_nbrs[j].clone();
                g_neighbors_unrooted.entry(ci.clone()).or_default().push(cj.clone());
                g_neighbors_unrooted.entry(cj.clone()).or_default().push(ci.clone());
            }
        }
    }
    // Safety: ensure every node referenced as a neighbor is also a key.
    // Collect missing keys first to avoid borrowing issues.
    let all_referenced: Vec<String> = g_neighbors_unrooted.values()
        .flat_map(|vs| vs.iter().cloned())
        .collect();
    for name in all_referenced {
        g_neighbors_unrooted.entry(name).or_default();
    }

    // Node selection
    let include_root_node: bool;
    let selected: HashSet<String>;
    let frontier_names: Vec<String>;
    let last_added_name: Option<String>;

    if predict_root_position {
        include_root_node = false;
        selected = g_neighbors_unrooted.keys().cloned().collect();
        frontier_names = Vec::new();
        last_added_name = None;
    } else if nb_sampled == max_to_sample && !force_mask_last_added {
        include_root_node = true;
        selected = g_neighbors.keys().filter(|k| k.as_str() != g_root_name.as_str()).cloned().collect();
        frontier_names = vec![g_root_name.clone()];
        last_added_name = None;
    } else if nb_sampled == max_to_sample && force_mask_last_added {
        include_root_node = true;
        selected = g_neighbors.keys().cloned().collect();
        frontier_names = Vec::new();
        last_added_name = Some(g_root_name.clone());
    } else {
        include_root_node = false;
        let node_list: Vec<String> = g_neighbors_unrooted.keys().cloned().collect();
        let n = node_list.len();
        let name_to_idx: HashMap<&str, usize> = node_list.iter().enumerate()
            .map(|(i, name)| (name.as_str(), i)).collect();

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (u, vs) in &g_neighbors_unrooted {
            let ui = name_to_idx[u.as_str()];
            for v in vs {
                let vi = match name_to_idx.get(v.as_str()) {
                    Some(&idx) => idx,
                    None => {
                        log::warn!(
                            "neighbor '{}' of node '{}' not found \
                             in g_neighbors_unrooted keys — skipping edge",
                            v, u
                        );
                        continue;
                    }
                };
                if ui != vi { adj[ui].push(vi); }
            }
        }
        for a in &mut adj { a.sort_unstable(); a.dedup(); }

        let degrees: Vec<usize> = adj.iter().map(|a| a.len()).collect();
        let leaves_idx: Vec<usize> = degrees.iter().enumerate()
            .filter(|(_, &d)| d == 1).map(|(i, _)| i).collect();
        let internal_idx: Vec<usize> = degrees.iter().enumerate()
            .filter(|(_, &d)| d == 3).map(|(i, _)| i).collect();

        let leaf_dist: Option<Vec<i32>> = if sample_order == "bottom_up" {
            let mut dist = vec![-1i32; n];
            let mut queue = VecDeque::new();
            for &li in &leaves_idx { dist[li] = 0; queue.push_back(li); }
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if dist[v] == -1 { dist[v] = dist[u] + 1; queue.push_back(v); }
                }
            }
            Some(dist)
        } else { None };

        let mut mapped = vec![false; n];
        for &li in &leaves_idx { mapped[li] = true; }
        let internal_set: HashSet<usize> = internal_idx.iter().copied().collect();

        let mut mapped_cnt = vec![0u32; n];
        for &ii in &internal_idx {
            mapped_cnt[ii] = adj[ii].iter().filter(|&&nb| mapped[nb]).count() as u32;
        }

        let mut frontier_idx: Vec<usize> = internal_idx.iter()
            .filter(|&&ii| mapped_cnt[ii] >= 2).copied().collect();
        let mut added_order: Vec<usize> = Vec::new();
        let mut last_added_i: Option<usize> = None;

        while added_order.len() < nb_sampled && !frontier_idx.is_empty() {
            let pick = if let Some(ref dist) = leaf_dist {
                let min_d = frontier_idx.iter().map(|&v| dist[v]).min().unwrap();
                let closest: Vec<usize> = frontier_idx.iter().enumerate()
                    .filter(|(_, &v)| dist[v] == min_d).map(|(i, _)| i).collect();
                *closest.choose(rng).unwrap()
            } else {
                rng.gen_range(0..frontier_idx.len())
            };
            let v = frontier_idx.swap_remove(pick);
            mapped[v] = true;
            added_order.push(v);
            last_added_i = Some(v);
            for &u in &adj[v] {
                if !mapped[u] {
                    mapped_cnt[u] += 1;
                    if internal_set.contains(&u) && mapped_cnt[u] == 2 {
                        frontier_idx.push(u);
                    }
                }
            }
        }

        let mut sel = HashSet::new();
        for &li in &leaves_idx { sel.insert(node_list[li].clone()); }
        for &ai in &added_order { sel.insert(node_list[ai].clone()); }
        selected = sel;
        frontier_names = frontier_idx.iter().map(|&i| node_list[i].clone()).collect();
        last_added_name = last_added_i.map(|i| node_list[i].clone());
    }

    let active_neighbors = if include_root_node { g_neighbors } else { &g_neighbors_unrooted };
    let mut g_names: Vec<String> = active_neighbors.keys().cloned().collect();
    g_names.sort();
    let n_gene = g_names.len();
    let g_name_to_idx: HashMap<&str, usize> = g_names.iter().enumerate()
        .map(|(i, name)| (name.as_str(), i)).collect();

    let leaf_set: HashSet<&str> = g_leaves_names.iter().map(|s| s.as_str()).collect();
    let frontier_set: HashSet<&str> = frontier_names.iter().map(|s| s.as_str()).collect();

    let mut x_gene = vec![0i32; n_gene];
    let mut g_true_sp = vec![0i32; n_gene];
    let mut event_true = vec![0i32; n_gene];
    let mut event_input = vec![0i32; n_gene];
    let mut frontier_mask = vec![0u8; n_gene];
    let mut is_leaf = vec![0u8; n_gene];
    let mut mask_label_node = vec![0u8; n_gene];

    for (i, name) in g_names.iter().enumerate() {
        let sp_name = true_states.get(name)
            .ok_or_else(|| format!("Gene node '{}' not in true_states", name))?;
        let sp_id = sp_name_to_id.get(sp_name.as_str())
            .ok_or_else(|| format!("Species '{}' not found in sp_name_to_id", sp_name))?;
        g_true_sp[i] = *sp_id;

        if selected.contains(name) { x_gene[i] = *sp_id; }

        if leaf_set.contains(name.as_str()) {
            event_true[i] = 3;
        } else {
            event_true[i] = *true_events.get(name)
                .ok_or_else(|| format!("Gene node '{}' not in true_events", name))?;
        }

        event_input[i] = if selected.contains(name) { event_true[i] } else { 4 };

        if frontier_set.contains(name.as_str()) { frontier_mask[i] = 1; }
        if leaf_set.contains(name.as_str()) { is_leaf[i] = 1; }
    }

    if let Some(ref la) = last_added_name {
        if let Some(&idx) = g_name_to_idx.get(la.as_str()) {
            if force_mask_last_added {
                event_input[idx] = 4;
                mask_label_node[idx] = 1;
            }
        }
    }

    let mut undirected_edges: HashSet<(i32, i32)> = HashSet::new();
    for (u, vs) in active_neighbors {
        let ui = g_name_to_idx[u.as_str()] as i32;
        for v in vs {
            if let Some(&vi_usize) = g_name_to_idx.get(v.as_str()) {
                let vi = vi_usize as i32;
                if ui != vi {
                    let e = if ui < vi { (ui, vi) } else { (vi, ui) };
                    undirected_edges.insert(e);
                }
            }
        }
    }

    let mut directed: Vec<(i32, i32)> = undirected_edges.into_iter().collect();
    directed.sort();
    let n_edges = directed.len();
    let mut g_edge_src = Vec::with_capacity(n_edges * 2);
    let mut g_edge_dst = Vec::with_capacity(n_edges * 2);
    let mut g_dir_src = Vec::with_capacity(n_edges);
    let mut g_dir_dst = Vec::with_capacity(n_edges);
    for &(a, b) in &directed {
        g_dir_src.push(a); g_dir_dst.push(b);
        g_edge_src.push(a); g_edge_dst.push(b);
        g_edge_src.push(b); g_edge_dst.push(a);
    }

    let root_pair_0 = true_root.first()
        .and_then(|s| g_name_to_idx.get(s.as_str())).map(|&v| v as i32).unwrap_or(-1);
    let root_pair_1 = true_root.get(1)
        .and_then(|s| g_name_to_idx.get(s.as_str())).map(|&v| v as i32).unwrap_or(-1);
    let root_edge_target: i32 = if !include_root_node && root_pair_0 >= 0 && root_pair_1 >= 0 {
        let pair_fwd = (root_pair_0.min(root_pair_1), root_pair_0.max(root_pair_1));
        directed.iter().position(|&e| e == pair_fwd).map(|i| i as i32).unwrap_or(-1)
    } else {
        -1
    };

    Ok(TaskTensors {
        x_gene, g_true_sp, event_true, event_input,
        frontier_mask, is_leaf, mask_label_node,
        g_edge_src, g_edge_dst, g_dir_src, g_dir_dst, root_edge_target,
    })
}


/// Build all tensors needed for a training sample in one Rust call.
///
/// Replaces the Python functions: unroot_tree, sample_gene_coloring,
/// _build_species_tensors, _build_gene_edges, _build_gene_event_types,
/// and the masking/assembly logic in get_sample.
///
/// Returns a dict of numpy arrays ready to be wrapped into HeteroData.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (base_sample, seed, force_mask_last_added, predict_root_position, sample_order = "random"))]
pub fn build_training_tensors(
    py: Python,
    base_sample: &Bound<'_, pyo3::types::PyDict>,
    seed: u64,
    force_mask_last_added: bool,
    predict_root_position: bool,
    sample_order: &str,
) -> PyResult<PyObject> {
    use numpy::PyArray1;
    use pyo3::types::PyDict;
    use rand::seq::SliceRandom;
    use rand::Rng;
    use std::collections::{HashMap, HashSet, VecDeque};

    // ---- Extract fields from base_sample dict ----
    let g_root_name: String = base_sample
        .get_item("g_root_name")?.ok_or_else(|| PyValueError::new_err("missing g_root_name"))?
        .extract::<String>()?;
    let nb_g_leaves: usize = base_sample
        .get_item("nb_g_leaves")?.ok_or_else(|| PyValueError::new_err("missing nb_g_leaves"))?
        .extract::<usize>()?;
    let species_names: Vec<String> = base_sample
        .get_item("species_names")?.ok_or_else(|| PyValueError::new_err("missing species_names"))?
        .extract::<Vec<String>>()?;
    let g_leaves_names: Vec<String> = base_sample
        .get_item("g_leaves_names")?.ok_or_else(|| PyValueError::new_err("missing g_leaves_names"))?
        .extract::<Vec<String>>()?;
    let true_root: Vec<String> = base_sample
        .get_item("true_root")?.ok_or_else(|| PyValueError::new_err("missing true_root"))?
        .extract::<Vec<String>>()?;

    // g_neighbors: Dict[str, List[str]]
    let g_neighbors: HashMap<String, Vec<String>> = base_sample
        .get_item("g_neighbors")?.ok_or_else(|| PyValueError::new_err("missing g_neighbors"))?
        .extract()?;

    // sp_children: Dict[str, List[str]]
    let sp_children: HashMap<String, Vec<String>> = base_sample
        .get_item("sp_children")?.ok_or_else(|| PyValueError::new_err("missing sp_children"))?
        .extract()?;

    // true_states: Dict[str, str]
    let true_states: HashMap<String, String> = base_sample
        .get_item("true_states")?.ok_or_else(|| PyValueError::new_err("missing true_states"))?
        .extract()?;

    // true_events: Dict[str, int]
    let true_events: HashMap<String, i32> = base_sample
        .get_item("true_events")?.ok_or_else(|| PyValueError::new_err("missing true_events"))?
        .extract()?;

    // ---- RNG setup ----
    let mut rng = StdRng::seed_from_u64(seed);

    // ---- Determine sampling parameters ----
    let max_to_sample = if !force_mask_last_added || predict_root_position {
        nb_g_leaves.saturating_sub(2)
    } else {
        nb_g_leaves.saturating_sub(1)
    };

    let nb_sampled = if predict_root_position {
        max_to_sample
    } else {
        rng.gen_range(0..=max_to_sample)
    };

    // ---- Unroot tree ----
    let mut g_neighbors_unrooted: HashMap<String, Vec<String>> = HashMap::new();
    for (k, v) in &g_neighbors {
        if k != &g_root_name {
            let filtered: Vec<String> = v.iter()
                .filter(|n| *n != &g_root_name)
                .cloned()
                .collect();
            g_neighbors_unrooted.insert(k.clone(), filtered);
        }
    }
    // Connect root's two children
    if let Some(root_nbrs) = g_neighbors.get(&g_root_name) {
        if root_nbrs.len() == 2 {
            let c1 = &root_nbrs[0];
            let c2 = &root_nbrs[1];
            g_neighbors_unrooted.entry(c1.clone()).or_default().push(c2.clone());
            g_neighbors_unrooted.entry(c2.clone()).or_default().push(c1.clone());
        }
    }

    // ---- Frontier sampling / node selection ----
    let include_root_node: bool;
    let selected: HashSet<String>;
    let frontier_names: Vec<String>;
    let last_added_name: Option<String>;

    if predict_root_position {
        include_root_node = false;
        selected = g_neighbors_unrooted.keys().cloned().collect();
        frontier_names = Vec::new();
        last_added_name = None;
    } else if nb_sampled == max_to_sample && !force_mask_last_added {
        include_root_node = true;
        selected = g_neighbors.keys()
            .filter(|k| *k != &g_root_name)
            .cloned().collect();
        frontier_names = vec![g_root_name.clone()];
        last_added_name = None;
    } else if nb_sampled == max_to_sample && force_mask_last_added {
        include_root_node = true;
        selected = g_neighbors.keys().cloned().collect();
        frontier_names = Vec::new();
        last_added_name = Some(g_root_name.clone());
    } else {
        include_root_node = false;
        // ---- Inline frontier sampling (sample_gene_coloring) ----
        let node_list: Vec<String> = g_neighbors_unrooted.keys().cloned().collect();
        let n = node_list.len();
        let name_to_idx: HashMap<&str, usize> = node_list.iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (u, vs) in &g_neighbors_unrooted {
            let ui = name_to_idx[u.as_str()];
            for v in vs {
                let vi = name_to_idx[v.as_str()];
                if ui != vi {
                    adj[ui].push(vi);
                }
            }
        }
        // Deduplicate adjacency
        for a in &mut adj {
            a.sort_unstable();
            a.dedup();
        }

        let degrees: Vec<usize> = adj.iter().map(|a| a.len()).collect();
        let leaves_idx: Vec<usize> = degrees.iter().enumerate()
            .filter(|(_, &d)| d == 1)
            .map(|(i, _)| i)
            .collect();
        let internal_idx: Vec<usize> = degrees.iter().enumerate()
            .filter(|(_, &d)| d == 3)
            .map(|(i, _)| i)
            .collect();

        // Leaf distance BFS for bottom_up ordering
        let leaf_dist: Option<Vec<i32>> = if sample_order == "bottom_up" {
            let mut dist = vec![-1i32; n];
            let mut queue = VecDeque::new();
            for &li in &leaves_idx {
                dist[li] = 0;
                queue.push_back(li);
            }
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if dist[v] == -1 {
                        dist[v] = dist[u] + 1;
                        queue.push_back(v);
                    }
                }
            }
            Some(dist)
        } else {
            None
        };

        let mut mapped = vec![false; n];
        for &li in &leaves_idx {
            mapped[li] = true;
        }
        let internal_set: HashSet<usize> = internal_idx.iter().copied().collect();

        let mut mapped_cnt = vec![0u32; n];
        for &ii in &internal_idx {
            let cnt = adj[ii].iter().filter(|&&nb| mapped[nb]).count() as u32;
            mapped_cnt[ii] = cnt;
        }

        let mut frontier_idx: Vec<usize> = internal_idx.iter()
            .filter(|&&ii| mapped_cnt[ii] >= 2)
            .copied()
            .collect();
        let mut added_order: Vec<usize> = Vec::new();
        let mut last_added_i: Option<usize> = None;

        while added_order.len() < nb_sampled && !frontier_idx.is_empty() {
            let pick = if let Some(ref dist) = leaf_dist {
                let min_d = frontier_idx.iter().map(|&v| dist[v]).min().unwrap();
                let closest: Vec<usize> = frontier_idx.iter().enumerate()
                    .filter(|(_, &v)| dist[v] == min_d)
                    .map(|(i, _)| i)
                    .collect();
                *closest.choose(&mut rng).unwrap()
            } else {
                rng.gen_range(0..frontier_idx.len())
            };
            let v = frontier_idx.swap_remove(pick);
            mapped[v] = true;
            added_order.push(v);
            last_added_i = Some(v);

            for &u in &adj[v] {
                if !mapped[u] {
                    mapped_cnt[u] += 1;
                    if internal_set.contains(&u) && mapped_cnt[u] == 2 {
                        frontier_idx.push(u);
                    }
                }
            }
        }

        let mut sel = HashSet::new();
        for &li in &leaves_idx {
            sel.insert(node_list[li].clone());
        }
        for &ai in &added_order {
            sel.insert(node_list[ai].clone());
        }
        selected = sel;
        frontier_names = frontier_idx.iter().map(|&i| node_list[i].clone()).collect();
        last_added_name = last_added_i.map(|i| node_list[i].clone());
    };

    // ---- Species tensors ----
    let _n_sp = species_names.len();
    let mut sp_name_to_id: HashMap<&str, i32> = HashMap::new();
    let mut sp_name_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, name) in species_names.iter().enumerate() {
        sp_name_to_id.insert(name.as_str(), (i + 1) as i32);
        sp_name_to_idx.insert(name.as_str(), i);
    }
    let x_sp: Vec<i32> = species_names.iter()
        .map(|name: &String| sp_name_to_id[name.as_str()])
        .collect();

    let mut sp_child_src: Vec<i32> = Vec::new();
    let mut sp_child_dst: Vec<i32> = Vec::new();
    let mut sp_parent_src: Vec<i32> = Vec::new();
    let mut sp_parent_dst: Vec<i32> = Vec::new();
    for (p, children) in &sp_children {
        let pi = sp_name_to_idx[p.as_str()] as i32;
        for c in children {
            let ci = sp_name_to_idx[c.as_str()] as i32;
            sp_child_src.push(pi);
            sp_child_dst.push(ci);
            sp_parent_src.push(ci);
            sp_parent_dst.push(pi);
        }
    }

    // ---- Gene node ordering ----
    let active_neighbors = if include_root_node { &g_neighbors } else { &g_neighbors_unrooted };
    let mut g_names: Vec<String> = active_neighbors.keys().cloned().collect();
    g_names.sort();
    let n_gene = g_names.len();
    let g_name_to_idx: HashMap<&str, usize> = g_names.iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    // ---- Gene feature tensors ----
    let leaf_set: HashSet<&str> = g_leaves_names.iter().map(|s: &String| s.as_str()).collect();
    let frontier_set: HashSet<&str> = frontier_names.iter().map(|s| s.as_str()).collect();

    let mut x_gene = vec![0i32; n_gene];
    let mut g_true_sp = vec![0i32; n_gene];
    let mut event_true = vec![0i32; n_gene];
    let mut event_input = vec![0i32; n_gene];
    let mut frontier_mask = vec![0u8; n_gene]; // bool as u8
    let mut is_leaf = vec![0u8; n_gene];
    let mut mask_label_node = vec![0u8; n_gene];

    for (i, name) in g_names.iter().enumerate() {
        // Species mapping
        let sp_name = true_states.get(name)
            .ok_or_else(|| PyValueError::new_err(format!("Gene node '{}' not in true_states", name)))?;
        let sp_id = sp_name_to_id.get(sp_name.as_str())
            .ok_or_else(|| PyValueError::new_err(format!("Species '{}' not found", sp_name)))?;
        g_true_sp[i] = *sp_id;

        // Input features: masked for unselected nodes
        if selected.contains(name) {
            x_gene[i] = *sp_id;
        }
        // else x_gene[i] stays 0

        // Event types
        if leaf_set.contains(name.as_str()) {
            event_true[i] = 3; // P/leaf
        } else {
            event_true[i] = *true_events.get(name)
                .ok_or_else(|| PyValueError::new_err(format!("Gene node '{}' not in true_events", name)))?;
        }

        // Event input: masked for unselected nodes
        if selected.contains(name) {
            event_input[i] = event_true[i];
        } else {
            event_input[i] = 4; // mask
        }

        // Frontier mask
        if frontier_set.contains(name.as_str()) {
            frontier_mask[i] = 1;
        }

        // Is leaf
        if leaf_set.contains(name.as_str()) {
            is_leaf[i] = 1;
        }
    }

    // Handle last_added masking
    let last_idx: i32;
    let mask_last_added_event: bool;
    if let Some(ref la) = last_added_name {
        if let Some(&idx) = g_name_to_idx.get(la.as_str()) {
            last_idx = idx as i32;
            if force_mask_last_added {
                event_input[idx] = 4; // mask
                mask_last_added_event = true;
                mask_label_node[idx] = 1;
            } else {
                mask_last_added_event = false;
            }
        } else {
            last_idx = -1;
            mask_last_added_event = false;
        }
    } else {
        last_idx = -1;
        mask_last_added_event = false;
    }

    // ---- Gene edges ----
    let mut undirected_edges: HashSet<(i32, i32)> = HashSet::new();
    for (u, vs) in active_neighbors {
        let ui = g_name_to_idx[u.as_str()] as i32;
        for v in vs {
            if let Some(&vi_usize) = g_name_to_idx.get(v.as_str()) {
                let vi = vi_usize as i32;
                if ui != vi {
                    let e = if ui < vi { (ui, vi) } else { (vi, ui) };
                    undirected_edges.insert(e);
                }
            }
        }
    }

    let mut directed: Vec<(i32, i32)> = undirected_edges.into_iter().collect();
    directed.sort();

    let n_edges = directed.len();
    let mut g_edge_src = Vec::with_capacity(n_edges * 2);
    let mut g_edge_dst = Vec::with_capacity(n_edges * 2);
    let mut g_dir_src = Vec::with_capacity(n_edges);
    let mut g_dir_dst = Vec::with_capacity(n_edges);
    for &(a, b) in &directed {
        // Directed (canonical)
        g_dir_src.push(a);
        g_dir_dst.push(b);
        // Undirected (both directions)
        g_edge_src.push(a);
        g_edge_dst.push(b);
        g_edge_src.push(b);
        g_edge_dst.push(a);
    }

    // Root edge target
    let root_pair_0 = g_name_to_idx.get(true_root[0].as_str()).map(|&v| v as i32).unwrap_or(-1);
    let root_pair_1 = g_name_to_idx.get(true_root[1].as_str()).map(|&v| v as i32).unwrap_or(-1);
    let root_edge_target: i32 = if !include_root_node {
        let pair_fwd = (root_pair_0.min(root_pair_1), root_pair_0.max(root_pair_1));
        directed.iter().position(|&e| e == pair_fwd)
            .map(|i| i as i32)
            .unwrap_or(-1)
    } else {
        -1
    };

    // ---- Build result dict of numpy arrays ----
    // Use from_slice / into_pyarray so numpy fully owns the memory (no Rust
    // allocator callback on dealloc). This prevents cross-allocator corruption
    // when DataLoader workers tear down.
    let result = PyDict::new(py);
    result.set_item("x_sp", PyArray1::from_slice(py, &x_sp))?;
    result.set_item("x_gene", PyArray1::from_slice(py, &x_gene))?;
    result.set_item("g_true_sp", PyArray1::from_slice(py, &g_true_sp))?;
    result.set_item("event_true", PyArray1::from_slice(py, &event_true))?;
    result.set_item("event_input", PyArray1::from_slice(py, &event_input))?;
    result.set_item("frontier_mask", PyArray1::from_slice(py, &frontier_mask))?;
    result.set_item("is_leaf", PyArray1::from_slice(py, &is_leaf))?;
    result.set_item("mask_label_node", PyArray1::from_slice(py, &mask_label_node))?;
    result.set_item("last_added_index", last_idx)?;
    result.set_item("mask_last_added_event", mask_last_added_event)?;

    // Edge indices as 2xE arrays -- build via ndarray then convert to numpy.
    let edge_2d = |src: &[i32], dst: &[i32], name: &str| -> PyResult<PyObject> {
        use numpy::ndarray::Array2;
        use numpy::ToPyArray;
        let n = src.len();
        let mut flat = Vec::with_capacity(2 * n);
        flat.extend_from_slice(src);
        flat.extend_from_slice(dst);
        let arr = Array2::from_shape_vec([2, n], flat)
            .map_err(|e| PyValueError::new_err(format!("{}: {}", name, e)))?;
        Ok(arr.to_pyarray(py).into_any().unbind())
    };
    result.set_item("sp_child_edge", edge_2d(&sp_child_src, &sp_child_dst, "sp_child_edge")?)?;
    result.set_item("sp_parent_edge", edge_2d(&sp_parent_src, &sp_parent_dst, "sp_parent_edge")?)?;
    result.set_item("g_edge", edge_2d(&g_edge_src, &g_edge_dst, "g_edge")?)?;
    result.set_item("g_dir_edge", edge_2d(&g_dir_src, &g_dir_dst, "g_dir_edge")?)?;
    result.set_item("root_edge_target", root_edge_target)?;

    Ok(result.into())
}

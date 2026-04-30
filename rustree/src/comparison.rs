// Comparing two trees, to see whether they are identical.
// + Reconciliation accuracy comparison (clade-based).

use std::collections::{BTreeSet, HashSet, HashMap};
use crate::node::{Node, FlatTree, RecTree, rectree::Event};
use crate::error::RustreeError;
use crate::sampling::{mark_nodes_postorder, NodeMark, get_descendant_leaf_names};


// ============================================================================
// Topology comparison (existing)
// ============================================================================

/// Returns the lexicographically smallest leaf name in the subtree.
/// Used for canonical child ordering to avoid exponential comparison.
fn min_leaf_name(node: &Node) -> &str {
    match (&node.left_child, &node.right_child) {
        (None, None) => &node.name,
        (Some(left), Some(right)) => {
            let l = min_leaf_name(left);
            let r = min_leaf_name(right);
            if l <= r { l } else { r }
        }
        (Some(child), None) | (None, Some(child)) => min_leaf_name(child),
    }
}

/// Recursively compares two Node objects (ignoring the order of children).
///
/// Uses canonical child ordering (by minimum leaf name) to achieve O(n)
/// comparison instead of the O(2^h) worst case of trying both orderings.
pub fn compare_nodes(n1: &Node, n2: &Node, use_lengths: bool, tol: f64) -> Result<bool, RustreeError> {
    if n1.name != n2.name { return Ok(false); }

    if use_lengths
        && (n1.length - n2.length).abs() > tol { return Ok(false); }

    // Collect non-None children for each node.
    let mut children1 = Vec::new();
    if let Some(child) = &n1.left_child { children1.push(child.as_ref()); }
    if let Some(child) = &n1.right_child { children1.push(child.as_ref()); }

    let mut children2 = Vec::new();
    if let Some(child) = &n2.left_child { children2.push(child.as_ref()); }
    if let Some(child) = &n2.right_child { children2.push(child.as_ref()); }

    if children1.len() != children2.len() {
        return Ok(false);
    }

    match children1.len() {
        0 => Ok(true),
        2 => {
            // Sort children by min leaf name for canonical ordering
            children1.sort_by_key(|c| min_leaf_name(c));
            children2.sort_by_key(|c| min_leaf_name(c));
            Ok(compare_nodes(children1[0], children2[0], use_lengths, tol)?
                && compare_nodes(children1[1], children2[1], use_lengths, tol)?)
        },
        n => Err(RustreeError::Tree(format!("Invalid binary tree: node '{}' has {} children (expected 0 or 2)", n1.name, n)))
    }
}

/// Convenience wrapper for topology-only comparison (ignores branch lengths).
pub fn compare_nodes_topology(n1: &Node, n2: &Node) -> Result<bool, RustreeError> {
    compare_nodes(n1, n2, false, 0.0)
}


// ============================================================================
// Reconciliation accuracy comparison
// ============================================================================

/// Per-node comparison detail.
#[derive(Clone, Debug)]
pub struct NodeComparison {
    /// The clade (set of descendant extant leaf names) identifying this node
    pub clade: BTreeSet<String>,
    /// Node index in the truth gene tree
    pub truth_node_idx: usize,
    /// Node name in the truth gene tree
    pub truth_node_name: String,
    /// Node index in the inferred gene tree
    pub inferred_node_idx: usize,
    /// Node name in the inferred gene tree
    pub inferred_node_name: String,
    /// Species name from truth tree (None if mapping unknown)
    pub truth_species: Option<String>,
    /// Species name from inferred tree (None if mapping unknown)
    pub inferred_species: Option<String>,
    /// Event from truth tree
    pub truth_event: Event,
    /// Event from inferred tree
    pub inferred_event: Event,
    /// Whether species mapping was correct (None if either side unknown)
    pub mapping_correct: Option<bool>,
    /// Whether event was correct
    pub event_correct: bool,
}

/// Result of comparing two reconciliations (truth vs inferred).
#[derive(Clone, Debug)]
pub struct ReconciliationComparison {
    /// Number of nodes matched by clade
    pub nodes_compared: usize,
    /// Number of correct species mappings (among evaluated)
    pub correct_mappings: usize,
    /// Number of correct events
    pub correct_events: usize,
    /// Number of nodes correct on both mapping and event
    pub correct_both: usize,
    /// Number of nodes where both sides had species info
    pub mappings_evaluated: usize,
    /// Per-node comparison details
    pub node_details: Vec<NodeComparison>,
    /// Confusion matrix: (truth_event, predicted_event) -> count
    pub event_confusion: HashMap<(Event, Event), usize>,
    /// Leaf mapping sanity check: correct count
    pub leaf_correct: usize,
    /// Leaf mapping sanity check: total leaves
    pub leaf_total: usize,
    /// Number of truth clades not found in inferred tree (topology mismatches)
    pub unmatched_truth_clades: usize,
    /// Number of inferred clades not found in truth tree
    pub unmatched_inferred_clades: usize,
}

impl ReconciliationComparison {
    /// Fraction of correctly inferred species mappings.
    pub fn mapping_accuracy(&self) -> f64 {
        if self.mappings_evaluated == 0 { return 0.0; }
        self.correct_mappings as f64 / self.mappings_evaluated as f64
    }

    /// Fraction of correctly inferred events.
    pub fn event_accuracy(&self) -> f64 {
        if self.nodes_compared == 0 { return 0.0; }
        self.correct_events as f64 / self.nodes_compared as f64
    }

    /// Fraction of nodes correct on both mapping and event.
    pub fn both_accuracy(&self) -> f64 {
        if self.mappings_evaluated == 0 { return 0.0; }
        self.correct_both as f64 / self.mappings_evaluated as f64
    }
}

/// Result of comparing truth against multiple reconciliation samples.
#[derive(Clone, Debug)]
pub struct MultiSampleComparison {
    /// Per-sample comparisons
    pub per_sample: Vec<ReconciliationComparison>,
    /// Consensus comparison (majority vote across samples per clade)
    pub consensus: ReconciliationComparison,
    /// Mean mapping accuracy across samples
    pub mean_mapping_accuracy: f64,
    /// Mean event accuracy across samples
    pub mean_event_accuracy: f64,
}


// ============================================================================
// Internal helpers
// ============================================================================

/// Find indices of extant leaves (Event::Leaf) in the gene tree.
fn extant_leaf_indices(tree: &FlatTree, event_mapping: &[Event]) -> HashSet<usize> {
    tree.nodes.iter().enumerate()
        .filter(|(i, n)| {
            n.left_child.is_none()
            && n.right_child.is_none()
            && event_mapping[*i] == Event::Leaf
        })
        .map(|(i, _)| i)
        .collect()
}

/// Collect names of leaves with Event::Leaf (extant genes).
fn extant_leaf_names(tree: &FlatTree, event_mapping: &[Event]) -> BTreeSet<String> {
    extant_leaf_indices(tree, event_mapping).iter()
        .map(|&i| tree.nodes[i].name.clone())
        .collect()
}

/// Build a mapping from clade (set of extant leaf names) to node index.
///
/// Uses `mark_nodes_postorder` from sampling to identify nodes with extant
/// descendants, then `get_descendant_leaf_names` to compute each node's clade
/// (filtered to extant leaves only).
///
/// When multiple nodes share the same clade (e.g., SL/DL chains in ALERax),
/// keeps the most rootward one (last inserted overwrites in postorder).
fn build_extant_clade_map(
    tree: &FlatTree,
    event_mapping: &[Event],
) -> HashMap<BTreeSet<String>, usize> {
    let extant_indices = extant_leaf_indices(tree, event_mapping);
    let extant_names: HashSet<String> = extant_indices.iter()
        .map(|&i| tree.nodes[i].name.clone())
        .collect();

    // Mark which nodes have extant descendants (reuse sampling infrastructure)
    let mut marks = vec![NodeMark::Discard; tree.nodes.len()];
    mark_nodes_postorder(tree, tree.root, &extant_indices, &mut marks);

    // For each node with extant descendants, compute its extant clade
    let postorder = tree.postorder_indices();
    let mut map = HashMap::new();
    for &idx in &postorder {
        if marks[idx] == NodeMark::Discard {
            continue;
        }
        // get_descendant_leaf_names returns ALL leaves; filter to extant only
        // Safety: idx comes from tree.postorder_indices(), so it is a valid tree node index
        let all_descendants = get_descendant_leaf_names(tree, idx)
            .expect("internal error: postorder index should be valid");
        let clade: BTreeSet<String> = all_descendants.into_iter()
            .filter(|name| extant_names.contains(name))
            .collect();

        if !clade.is_empty() {
            // Postorder: children before parents. Overwriting keeps most rootward.
            map.insert(clade, idx);
        }
    }
    map
}

/// Get species name for a gene node, or None if mapping is absent.
fn species_name_for(rec_tree: &RecTree, gene_node_idx: usize) -> Option<String> {
    rec_tree.node_mapping[gene_node_idx]
        .map(|sp_idx| rec_tree.species_tree.nodes[sp_idx].name.clone())
}

/// Precompute species clade (set of descendant leaf names) for each species node.
/// Used for comparing species mappings across different species tree instances.
fn build_species_clades(species_tree: &FlatTree) -> Vec<BTreeSet<String>> {
    let mut clades = vec![BTreeSet::new(); species_tree.nodes.len()];
    let postorder = species_tree.postorder_indices();
    for &idx in &postorder {
        let node = &species_tree.nodes[idx];
        if node.left_child.is_none() && node.right_child.is_none() {
            clades[idx].insert(node.name.clone());
        } else {
            if let Some(left) = node.left_child {
                let left_clade = clades[left].clone();
                clades[idx].extend(left_clade);
            }
            if let Some(right) = node.right_child {
                let right_clade = clades[right].clone();
                clades[idx].extend(right_clade);
            }
        }
    }
    clades
}

/// Get species clade for a gene node, or None if mapping is absent.
fn species_clade_for(
    rec_tree: &RecTree,
    gene_node_idx: usize,
    species_clades: &[BTreeSet<String>],
) -> Option<BTreeSet<String>> {
    rec_tree.node_mapping[gene_node_idx]
        .map(|sp_idx| species_clades[sp_idx].clone())
}


// ============================================================================
// Public API
// ============================================================================

/// Compare two reconciliations by matching nodes via their extant leaf clades.
///
/// Both trees must have the same set of extant leaves (leaves with Event::Leaf).
/// The truth tree should typically come from simulation + `sample_extant()`.
/// The inferred tree typically comes from ALERax reconciliation.
///
/// Returns per-node comparison details and aggregate accuracy metrics.
pub fn compare_reconciliations(
    truth: &RecTree,
    inferred: &RecTree,
) -> Result<ReconciliationComparison, RustreeError> {
    // Verify same extant leaf sets
    let truth_extant = extant_leaf_names(&truth.gene_tree, &truth.event_mapping);
    let inf_extant = extant_leaf_names(&inferred.gene_tree, &inferred.event_mapping);

    if truth_extant != inf_extant {
        let only_truth: Vec<_> = truth_extant.difference(&inf_extant).take(5).collect();
        let only_inf: Vec<_> = inf_extant.difference(&truth_extant).take(5).collect();
        return Err(RustreeError::Validation(format!(
            "Extant leaf sets differ: truth has {}, inferred has {}. \
             Only in truth (first 5): {:?}. Only in inferred (first 5): {:?}",
            truth_extant.len(), inf_extant.len(), only_truth, only_inf
        )));
    }

    // Build clade maps for gene trees
    let truth_clades = build_extant_clade_map(&truth.gene_tree, &truth.event_mapping);
    let inf_clades = build_extant_clade_map(&inferred.gene_tree, &inferred.event_mapping);

    // Precompute species clades for clade-based comparison.
    // This handles the case where truth and inferred reference different species tree
    // instances (e.g., truth from simulation vs ALERax's parsed species tree).
    let truth_sp_clades = build_species_clades(&truth.species_tree);
    let inf_sp_clades = build_species_clades(&inferred.species_tree);

    let mut comparison = ReconciliationComparison {
        nodes_compared: 0,
        correct_mappings: 0,
        correct_events: 0,
        correct_both: 0,
        mappings_evaluated: 0,
        node_details: Vec::new(),
        event_confusion: HashMap::new(),
        leaf_correct: 0,
        leaf_total: truth_extant.len(),
        unmatched_truth_clades: 0,
        unmatched_inferred_clades: 0,
    };

    // Leaf mapping sanity check (by species clade)
    for leaf_name in &truth_extant {
        let truth_leaf_idx = truth.gene_tree.nodes.iter()
            .position(|n| n.name == *leaf_name);
        let inf_leaf_idx = inferred.gene_tree.nodes.iter()
            .position(|n| n.name == *leaf_name);

        if let (Some(ti), Some(ii)) = (truth_leaf_idx, inf_leaf_idx) {
            let truth_sp_clade = species_clade_for(truth, ti, &truth_sp_clades);
            let inf_sp_clade = species_clade_for(inferred, ii, &inf_sp_clades);
            if truth_sp_clade.is_some() && truth_sp_clade == inf_sp_clade {
                comparison.leaf_correct += 1;
            }
        }
    }

    // Compare internal + leaf nodes by clade
    for (clade, &truth_idx) in &truth_clades {
        if let Some(&inf_idx) = inf_clades.get(clade) {
            comparison.nodes_compared += 1;

            let truth_event = &truth.event_mapping[truth_idx];
            let inf_event = &inferred.event_mapping[inf_idx];

            // Compare species mappings by clade (robust across different species tree instances)
            let truth_sp_clade = species_clade_for(truth, truth_idx, &truth_sp_clades);
            let inf_sp_clade = species_clade_for(inferred, inf_idx, &inf_sp_clades);

            // Keep names for display purposes
            let truth_sp = species_name_for(truth, truth_idx);
            let inf_sp = species_name_for(inferred, inf_idx);

            let event_correct = truth_event == inf_event;
            let mapping_correct = match (&truth_sp_clade, &inf_sp_clade) {
                (Some(tc), Some(ic)) => {
                    comparison.mappings_evaluated += 1;
                    let correct = tc == ic;
                    if correct { comparison.correct_mappings += 1; }
                    Some(correct)
                },
                _ => None,
            };

            if event_correct { comparison.correct_events += 1; }
            if event_correct && mapping_correct == Some(true) {
                comparison.correct_both += 1;
            }

            *comparison.event_confusion
                .entry((truth_event.clone(), inf_event.clone()))
                .or_insert(0) += 1;

            comparison.node_details.push(NodeComparison {
                clade: clade.clone(),
                truth_node_idx: truth_idx,
                truth_node_name: truth.gene_tree.nodes[truth_idx].name.clone(),
                inferred_node_idx: inf_idx,
                inferred_node_name: inferred.gene_tree.nodes[inf_idx].name.clone(),
                truth_species: truth_sp,
                inferred_species: inf_sp,
                truth_event: truth_event.clone(),
                inferred_event: inf_event.clone(),
                mapping_correct,
                event_correct,
            });
        } else {
            comparison.unmatched_truth_clades += 1;
        }
    }

    // Count inferred clades not in truth
    for clade in inf_clades.keys() {
        if !truth_clades.contains_key(clade) {
            comparison.unmatched_inferred_clades += 1;
        }
    }

    Ok(comparison)
}


/// Compare truth reconciliation against multiple inferred samples.
///
/// Computes per-sample metrics and a consensus comparison where
/// the species mapping and event for each clade are determined by
/// majority vote across all samples.
pub fn compare_reconciliations_multi(
    truth: &RecTree,
    samples: &[RecTree],
) -> Result<MultiSampleComparison, RustreeError> {
    if samples.is_empty() {
        return Err(RustreeError::Validation("No samples to compare".to_string()));
    }

    // Per-sample comparisons
    let per_sample: Vec<ReconciliationComparison> = samples.iter()
        .map(|s| compare_reconciliations(truth, s))
        .collect::<Result<Vec<_>, _>>()?;

    // Mean accuracies
    let mean_mapping_accuracy = if per_sample.is_empty() { 0.0 } else {
        per_sample.iter().map(|c| c.mapping_accuracy()).sum::<f64>() / per_sample.len() as f64
    };
    let mean_event_accuracy = if per_sample.is_empty() { 0.0 } else {
        per_sample.iter().map(|c| c.event_accuracy()).sum::<f64>() / per_sample.len() as f64
    };

    // Build consensus: majority vote per clade
    let truth_clades = build_extant_clade_map(&truth.gene_tree, &truth.event_mapping);
    let truth_extant = extant_leaf_names(&truth.gene_tree, &truth.event_mapping);
    let truth_sp_clades = build_species_clades(&truth.species_tree);

    // For each clade, collect votes across samples using species clades (not names)
    let mut species_clade_votes: HashMap<BTreeSet<String>, HashMap<BTreeSet<String>, usize>> = HashMap::new();
    let mut species_name_for_clade: HashMap<BTreeSet<String>, HashMap<BTreeSet<String>, String>> = HashMap::new();
    let mut event_votes: HashMap<BTreeSet<String>, HashMap<Event, usize>> = HashMap::new();

    for sample in samples {
        let inf_clade_map = build_extant_clade_map(&sample.gene_tree, &sample.event_mapping);
        let sample_sp_clades = build_species_clades(&sample.species_tree);

        for (clade, &inf_idx) in &inf_clade_map {
            if !truth_clades.contains_key(clade) {
                continue; // Only vote on clades present in truth
            }

            if let Some(sp_clade) = species_clade_for(sample, inf_idx, &sample_sp_clades) {
                *species_clade_votes
                    .entry(clade.clone())
                    .or_default()
                    .entry(sp_clade.clone())
                    .or_insert(0) += 1;
                // Store the species name for display
                if let Some(sp_name) = species_name_for(sample, inf_idx) {
                    species_name_for_clade
                        .entry(clade.clone())
                        .or_default()
                        .entry(sp_clade)
                        .or_insert(sp_name);
                }
            }

            let event = &sample.event_mapping[inf_idx];
            *event_votes
                .entry(clade.clone())
                .or_default()
                .entry(event.clone())
                .or_insert(0) += 1;
        }
    }

    // Build consensus comparison
    let mut consensus = ReconciliationComparison {
        nodes_compared: 0,
        correct_mappings: 0,
        correct_events: 0,
        correct_both: 0,
        mappings_evaluated: 0,
        node_details: Vec::new(),
        event_confusion: HashMap::new(),
        leaf_correct: 0,
        leaf_total: truth_extant.len(),
        unmatched_truth_clades: 0,
        unmatched_inferred_clades: 0,
    };

    // Leaf sanity check (same as single sample, using first sample)
    consensus.leaf_correct = per_sample[0].leaf_correct;

    for (clade, &truth_idx) in &truth_clades {
        let truth_event = &truth.event_mapping[truth_idx];
        let truth_sp = species_name_for(truth, truth_idx);
        let truth_sp_clade_opt = species_clade_for(truth, truth_idx, &truth_sp_clades);

        // Get majority-vote species clade
        let consensus_sp_clade = species_clade_votes.get(clade).and_then(|votes| {
            votes.iter().max_by_key(|(_, &count)| count).map(|(sc, _)| sc.clone())
        });
        // Get the display name for the winning species clade
        let consensus_sp = consensus_sp_clade.as_ref().and_then(|sc| {
            species_name_for_clade.get(clade).and_then(|m| m.get(sc).cloned())
        });

        // Get majority-vote event
        let consensus_event = event_votes.get(clade).and_then(|votes| {
            votes.iter().max_by_key(|(_, &count)| count).map(|(ev, _)| ev.clone())
        });

        if let Some(ref cons_event) = consensus_event {
            consensus.nodes_compared += 1;

            let event_correct = truth_event == cons_event;
            let mapping_correct = match (&truth_sp_clade_opt, &consensus_sp_clade) {
                (Some(tc), Some(cc)) => {
                    consensus.mappings_evaluated += 1;
                    let correct = tc == cc;
                    if correct { consensus.correct_mappings += 1; }
                    Some(correct)
                },
                _ => None,
            };

            if event_correct { consensus.correct_events += 1; }
            if event_correct && mapping_correct == Some(true) {
                consensus.correct_both += 1;
            }

            *consensus.event_confusion
                .entry((truth_event.clone(), cons_event.clone()))
                .or_insert(0) += 1;

            consensus.node_details.push(NodeComparison {
                clade: clade.clone(),
                truth_node_idx: truth_idx,
                truth_node_name: truth.gene_tree.nodes[truth_idx].name.clone(),
                inferred_node_idx: usize::MAX, // no single inferred node for consensus
                inferred_node_name: String::new(),
                truth_species: truth_sp,
                inferred_species: consensus_sp,
                truth_event: truth_event.clone(),
                inferred_event: cons_event.clone(),
                mapping_correct,
                event_correct,
            });
        } else {
            consensus.unmatched_truth_clades += 1;
        }
    }

    Ok(MultiSampleComparison {
        per_sample,
        consensus,
        mean_mapping_accuracy,
        mean_event_accuracy,
    })
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{FlatNode, FlatTree, RecTree, rectree::Event};
    use std::sync::Arc;

    /// Build a simple species tree: (A, B)root
    fn make_species_tree() -> FlatTree {
        FlatTree {
            nodes: vec![
                FlatNode { name: "root".into(), left_child: Some(1), right_child: Some(2), parent: None, depth: None, length: 0.0, bd_event: None },
                FlatNode { name: "A".into(), left_child: None, right_child: None, parent: Some(0), depth: None, length: 1.0, bd_event: None },
                FlatNode { name: "B".into(), left_child: None, right_child: None, parent: Some(0), depth: None, length: 1.0, bd_event: None },
            ],
            root: 0,
        }
    }

    /// Build a gene tree: ((a1, a2), b1)
    /// Mapped as: a1->A, a2->A, b1->B, internal1->A (dup), root->root (spec)
    fn make_truth_gene_tree(species: &Arc<FlatTree>) -> RecTree {
        let gene_tree = FlatTree {
            nodes: vec![
                // 0: root (speciation at root)
                FlatNode { name: "root".into(), left_child: Some(1), right_child: Some(4), parent: None, depth: None, length: 0.0, bd_event: None },
                // 1: internal (duplication in A)
                FlatNode { name: "dup_A".into(), left_child: Some(2), right_child: Some(3), parent: Some(0), depth: None, length: 0.5, bd_event: None },
                // 2: leaf a1
                FlatNode { name: "A_1".into(), left_child: None, right_child: None, parent: Some(1), depth: None, length: 0.5, bd_event: None },
                // 3: leaf a2
                FlatNode { name: "A_2".into(), left_child: None, right_child: None, parent: Some(1), depth: None, length: 0.5, bd_event: None },
                // 4: leaf b1
                FlatNode { name: "B_1".into(), left_child: None, right_child: None, parent: Some(0), depth: None, length: 1.0, bd_event: None },
            ],
            root: 0,
        };
        // species indices: root=0, A=1, B=2
        let node_mapping = vec![Some(0), Some(1), Some(1), Some(1), Some(2)];
        let event_mapping = vec![
            Event::Speciation, Event::Duplication, Event::Leaf, Event::Leaf, Event::Leaf
        ];
        RecTree::new(Arc::clone(species), gene_tree, node_mapping, event_mapping)
    }

    #[test]
    fn test_compare_identical() {
        let sp = Arc::new(make_species_tree());
        let truth = make_truth_gene_tree(&sp);
        let inferred = make_truth_gene_tree(&sp);

        let result = compare_reconciliations(&truth, &inferred).unwrap();
        assert_eq!(result.mapping_accuracy(), 1.0);
        assert_eq!(result.event_accuracy(), 1.0);
        assert_eq!(result.both_accuracy(), 1.0);
        assert_eq!(result.leaf_correct, 3);
        assert_eq!(result.leaf_total, 3);
        assert_eq!(result.unmatched_truth_clades, 0);
    }

    #[test]
    fn test_compare_wrong_event() {
        let sp = Arc::new(make_species_tree());
        let truth = make_truth_gene_tree(&sp);

        // Inferred: same tree but duplication is marked as speciation
        let mut inferred = make_truth_gene_tree(&sp);
        inferred.event_mapping[1] = Event::Speciation; // was Duplication

        let result = compare_reconciliations(&truth, &inferred).unwrap();
        assert_eq!(result.mapping_accuracy(), 1.0); // mappings still correct
        // 5 nodes total, all matched by clade. 4 correct events, 1 wrong.
        assert_eq!(result.correct_events, 4);
        assert_eq!(result.nodes_compared, 5);
    }

    #[test]
    fn test_compare_wrong_mapping() {
        let sp = Arc::new(make_species_tree());
        let truth = make_truth_gene_tree(&sp);

        // Inferred: duplication node mapped to B instead of A
        let mut inferred = make_truth_gene_tree(&sp);
        inferred.node_mapping[1] = Some(2); // mapped to B instead of A

        let result = compare_reconciliations(&truth, &inferred).unwrap();
        // 5 nodes compared, 4 correct mappings (one wrong), 5 correct events
        assert_eq!(result.correct_mappings, 4);
        assert_eq!(result.mappings_evaluated, 5);
        assert_eq!(result.correct_events, 5);
    }

    #[test]
    fn test_compare_with_loss_nodes() {
        // Inferred tree has extra loss nodes (like ALERax output)
        let sp = Arc::new(make_species_tree());
        let truth = make_truth_gene_tree(&sp);

        // Inferred: same topology + a loss node under root (SL pattern)
        // Tree: ((a1, a2)dup, (b1, LOSS)sl)root
        let gene_tree = FlatTree {
            nodes: vec![
                // 0: root (speciation)
                FlatNode { name: "root".into(), left_child: Some(1), right_child: Some(4), parent: None, depth: None, length: 0.0, bd_event: None },
                // 1: duplication in A
                FlatNode { name: "dup_A".into(), left_child: Some(2), right_child: Some(3), parent: Some(0), depth: None, length: 0.5, bd_event: None },
                // 2: A_1
                FlatNode { name: "A_1".into(), left_child: None, right_child: None, parent: Some(1), depth: None, length: 0.5, bd_event: None },
                // 3: A_2
                FlatNode { name: "A_2".into(), left_child: None, right_child: None, parent: Some(1), depth: None, length: 0.5, bd_event: None },
                // 4: SL node (speciation, one child survives, one is loss)
                FlatNode { name: "sl_B".into(), left_child: Some(5), right_child: Some(6), parent: Some(0), depth: None, length: 0.8, bd_event: None },
                // 5: B_1
                FlatNode { name: "B_1".into(), left_child: None, right_child: None, parent: Some(4), depth: None, length: 0.2, bd_event: None },
                // 6: loss
                FlatNode { name: "loss_1".into(), left_child: None, right_child: None, parent: Some(4), depth: None, length: 0.2, bd_event: None },
            ],
            root: 0,
        };
        let node_mapping = vec![Some(0), Some(1), Some(1), Some(1), Some(2), Some(2), Some(2)];
        let event_mapping = vec![
            Event::Speciation, Event::Duplication, Event::Leaf, Event::Leaf,
            Event::Speciation, Event::Leaf, Event::Loss,
        ];
        let inferred = RecTree::new(Arc::clone(&sp), gene_tree, node_mapping, event_mapping);

        let result = compare_reconciliations(&truth, &inferred).unwrap();
        // Clades in truth: {A_1}=leaf, {A_2}=leaf, {B_1}=leaf, {A_1,A_2}=dup, {A_1,A_2,B_1}=root
        // Clades in inferred: same leaves, {A_1,A_2}=dup, {B_1}=SL node (overwritten to idx 4),
        //   {A_1,A_2,B_1}=root
        // The SL node (idx 4) has clade {B_1} and overwrites the leaf B_1 (idx 5).
        // So comparing {B_1}: truth event=Leaf (idx 4 in truth), inferred event=Speciation (idx 4 in inferred)
        // Wait, {B_1} clade: in truth it maps to the leaf B_1. In inferred, it maps to SL node (idx 4).
        // The SL node event is Speciation. Truth leaf event is Leaf. So event differs.
        //
        // Actually: the SL pattern means B_1's clade appears for both the SL node and the B_1 leaf.
        // In postorder, leaf B_1 (idx 5) comes before SL (idx 4). So the map overwrites to idx 4.
        // So {B_1} → idx 4 (SL, Speciation event, species B).
        // In truth: {B_1} → idx 4 (B_1 leaf, Leaf event, species B).
        // Mapping: both B → correct. Event: Leaf vs Speciation → wrong.
        assert_eq!(result.nodes_compared, 5);
        assert_eq!(result.correct_mappings, 5); // all mappings correct
        assert_eq!(result.correct_events, 4); // SL vs Leaf mismatch
    }

    #[test]
    fn test_compare_different_leaves() {
        let sp = Arc::new(make_species_tree());
        let truth = make_truth_gene_tree(&sp);

        // Inferred with different leaf set
        let gene_tree = FlatTree {
            nodes: vec![
                FlatNode { name: "root".into(), left_child: Some(1), right_child: Some(2), parent: None, depth: None, length: 0.0, bd_event: None },
                FlatNode { name: "A_1".into(), left_child: None, right_child: None, parent: Some(0), depth: None, length: 1.0, bd_event: None },
                FlatNode { name: "C_1".into(), left_child: None, right_child: None, parent: Some(0), depth: None, length: 1.0, bd_event: None },
            ],
            root: 0,
        };
        let inferred = RecTree::new(
            Arc::clone(&sp), gene_tree,
            vec![Some(0), Some(1), Some(2)],
            vec![Event::Speciation, Event::Leaf, Event::Leaf],
        );

        let result = compare_reconciliations(&truth, &inferred);
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_multi_identical() {
        let sp = Arc::new(make_species_tree());
        let truth = make_truth_gene_tree(&sp);
        let samples = vec![
            make_truth_gene_tree(&sp),
            make_truth_gene_tree(&sp),
            make_truth_gene_tree(&sp),
        ];

        let result = compare_reconciliations_multi(&truth, &samples).unwrap();
        assert_eq!(result.per_sample.len(), 3);
        assert_eq!(result.mean_mapping_accuracy, 1.0);
        assert_eq!(result.mean_event_accuracy, 1.0);
        assert_eq!(result.consensus.mapping_accuracy(), 1.0);
        assert_eq!(result.consensus.event_accuracy(), 1.0);
    }

    #[test]
    fn test_compare_multi_consensus() {
        let sp = Arc::new(make_species_tree());
        let truth = make_truth_gene_tree(&sp);

        // 3 samples: 2 agree with truth, 1 has wrong event for dup node
        let s1 = make_truth_gene_tree(&sp);
        let s2 = make_truth_gene_tree(&sp);
        let mut s3 = make_truth_gene_tree(&sp);
        s3.event_mapping[1] = Event::Transfer; // wrong

        let result = compare_reconciliations_multi(&truth, &[s1, s2, s3]).unwrap();
        // Consensus should pick Duplication (2 votes) over Transfer (1 vote)
        assert_eq!(result.consensus.event_accuracy(), 1.0);
    }

    #[test]
    fn test_extant_leaf_names() {
        let sp = Arc::new(make_species_tree());
        let truth = make_truth_gene_tree(&sp);
        let names = extant_leaf_names(&truth.gene_tree, &truth.event_mapping);
        assert_eq!(names.len(), 3);
        assert!(names.contains("A_1"));
        assert!(names.contains("A_2"));
        assert!(names.contains("B_1"));
    }

    #[test]
    fn test_clade_map() {
        let sp = Arc::new(make_species_tree());
        let truth = make_truth_gene_tree(&sp);
        let clade_map = build_extant_clade_map(&truth.gene_tree, &truth.event_mapping);

        // Expected clades: {A_1}, {A_2}, {B_1}, {A_1,A_2}, {A_1,A_2,B_1}
        assert_eq!(clade_map.len(), 5);

        let a1: BTreeSet<String> = ["A_1".to_string()].into();
        let a2: BTreeSet<String> = ["A_2".to_string()].into();
        let b1: BTreeSet<String> = ["B_1".to_string()].into();
        let a1a2: BTreeSet<String> = ["A_1".to_string(), "A_2".to_string()].into();
        let all: BTreeSet<String> = ["A_1".to_string(), "A_2".to_string(), "B_1".to_string()].into();

        assert_eq!(clade_map[&a1], 2);   // leaf A_1
        assert_eq!(clade_map[&a2], 3);   // leaf A_2
        assert_eq!(clade_map[&b1], 4);   // leaf B_1
        assert_eq!(clade_map[&a1a2], 1); // dup node
        assert_eq!(clade_map[&all], 0);  // root
    }
}

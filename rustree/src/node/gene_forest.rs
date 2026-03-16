//! A container for a species tree and its associated gene trees.
//!
//! `GeneForest` groups a single species tree (shared via `Arc`) with
//! multiple reconciled gene trees. This is useful for:
//! - Pruning a species tree and all associated gene trees at once
//! - Batch reconciliation with ALERax
//! - Batch export to Newick/XML/CSV

use std::collections::HashSet;
use std::sync::Arc;
use crate::node::{FlatTree, RecTree};
use crate::node::rectree::Event;
use crate::sampling::{extract_induced_subtree, extract_induced_subtree_by_names};

/// A collection of gene trees associated with a single species tree.
///
/// All `RecTree` instances in the forest share the same `Arc<FlatTree>`
/// species tree, so cloning the forest is cheap (O(n) in gene trees,
/// O(1) for the species tree).
#[derive(Clone, Debug)]
pub struct GeneForest {
    pub species_tree: Arc<FlatTree>,
    pub gene_trees: Vec<RecTree>,
}

impl GeneForest {
    /// Create an empty forest with the given species tree.
    pub fn new(species_tree: FlatTree) -> Self {
        GeneForest {
            species_tree: Arc::new(species_tree),
            gene_trees: Vec::new(),
        }
    }

    /// Create a forest from a species tree and pre-built gene tree components.
    ///
    /// Each tuple is `(gene_tree, node_mapping, event_mapping)`.
    /// All resulting `RecTree`s share the same species tree Arc.
    pub fn with_gene_trees(
        species_tree: FlatTree,
        gene_trees: Vec<(FlatTree, Vec<Option<usize>>, Vec<Event>)>,
    ) -> Self {
        let species_arc = Arc::new(species_tree);
        let rec_trees = gene_trees
            .into_iter()
            .map(|(gt, nm, em)| RecTree::new(Arc::clone(&species_arc), gt, nm, em))
            .collect();
        GeneForest {
            species_tree: species_arc,
            gene_trees: rec_trees,
        }
    }

    /// Create a forest from a shared species tree Arc and existing RecTrees.
    ///
    /// The RecTrees' species_tree fields are replaced with the shared Arc.
    pub fn from_rec_trees(species_tree: Arc<FlatTree>, mut gene_trees: Vec<RecTree>) -> Self {
        for rt in &mut gene_trees {
            rt.species_tree = Arc::clone(&species_tree);
        }
        GeneForest {
            species_tree,
            gene_trees,
        }
    }

    /// Add a gene tree to the forest.
    ///
    /// The gene tree is wrapped in a `RecTree` sharing this forest's species tree.
    pub fn push(
        &mut self,
        gene_tree: FlatTree,
        node_mapping: Vec<Option<usize>>,
        event_mapping: Vec<Event>,
    ) {
        self.gene_trees.push(RecTree::new(
            Arc::clone(&self.species_tree),
            gene_tree,
            node_mapping,
            event_mapping,
        ));
    }

    /// Number of gene trees in the forest.
    pub fn len(&self) -> usize {
        self.gene_trees.len()
    }

    /// Returns true if the forest has no gene trees.
    pub fn is_empty(&self) -> bool {
        self.gene_trees.is_empty()
    }

    /// Iterate over the gene trees.
    pub fn iter(&self) -> impl Iterator<Item = &RecTree> {
        self.gene_trees.iter()
    }

    /// Get a gene tree by index.
    pub fn get(&self, idx: usize) -> Option<&RecTree> {
        self.gene_trees.get(idx)
    }

    /// Get a reference to the shared species tree.
    pub fn species_tree(&self) -> &FlatTree {
        &self.species_tree
    }

    /// Sample leaves and filter all trees accordingly.
    ///
    /// Prunes the species tree to keep only the specified leaves, then
    /// filters each gene tree to keep only genes mapping to the remaining
    /// species. Returns a new `GeneForest` with the pruned trees.
    pub fn sample_leaves(&self, names: &[String]) -> Result<Self, String> {
        let (sampled_species_tree, species_old_to_new) =
            extract_induced_subtree_by_names(&self.species_tree, names)
                .ok_or_else(|| "Failed to sample species tree".to_string())?;

        if self.gene_trees.is_empty() {
            return Ok(GeneForest::new(sampled_species_tree));
        }

        let shared_species = Arc::new(sampled_species_tree);
        let species_leaf_name_set: HashSet<&str> = names.iter().map(|s| s.as_str()).collect();

        let mut sampled_trees = Vec::with_capacity(self.gene_trees.len());
        for rt in &self.gene_trees {
            let sampled = sample_single_gene_tree(
                rt, &species_leaf_name_set, &species_old_to_new, &shared_species,
            )?;
            sampled_trees.push(sampled);
        }

        Ok(GeneForest {
            species_tree: shared_species,
            gene_trees: sampled_trees,
        })
    }

    /// Prune the forest to match a target species tree.
    ///
    /// The target species tree's leaves must be a subset of this forest's
    /// species tree leaves. The forest's species tree is pruned to keep only
    /// those leaves, and all gene trees are filtered accordingly.
    ///
    /// This is a convenience wrapper around `sample_leaves` that
    /// accepts a `FlatTree` instead of a list of leaf names.
    pub fn prune_to_species_tree(&self, target_species_tree: &FlatTree) -> Result<Self, String> {
        let leaf_names: Vec<String> = target_species_tree.nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .map(|n| n.name.clone())
            .collect();
        if leaf_names.is_empty() {
            return Err("Target species tree has no leaves".to_string());
        }
        self.sample_leaves(&leaf_names)
    }

    /// Return a new forest with each gene tree pruned to extant leaves only.
    ///
    /// For each gene tree, keeps only leaf nodes whose event is `Event::Leaf`
    /// (i.e., extant genes). Gene trees with zero extant leaves are dropped.
    pub fn sample_extant(&self) -> Result<Self, String> {
        let shared_species = Arc::clone(&self.species_tree);
        let mut sampled_trees = Vec::with_capacity(self.gene_trees.len());

        for rt in &self.gene_trees {
            // Find extant leaf indices
            let extant_indices: HashSet<usize> = rt.gene_tree.nodes.iter()
                .enumerate()
                .filter(|(i, n)| {
                    n.left_child.is_none()
                        && n.right_child.is_none()
                        && rt.event_mapping[*i] == Event::Leaf
                })
                .map(|(i, _)| i)
                .collect();

            if extant_indices.is_empty() {
                continue; // skip gene trees with no extant genes
            }

            let (sampled_gene_tree, gene_old_to_new) =
                extract_induced_subtree(&rt.gene_tree, &extant_indices)
                    .ok_or_else(|| "Failed to extract extant gene subtree".to_string())?;

            // Identity species mapping (species tree is unchanged)
            let identity: Vec<Option<usize>> = (0..self.species_tree.nodes.len())
                .map(Some)
                .collect();

            let (new_node_mapping, new_event_mapping) = remap_gene_tree_indices(
                &sampled_gene_tree, &gene_old_to_new,
                &rt.node_mapping, &rt.event_mapping,
                &identity,
            )?;

            sampled_trees.push(RecTree::new(
                Arc::clone(&shared_species),
                sampled_gene_tree,
                new_node_mapping,
                new_event_mapping,
            ));
        }

        Ok(GeneForest {
            species_tree: shared_species,
            gene_trees: sampled_trees,
        })
    }

    /// Consume the forest and return the gene trees.
    pub fn into_gene_trees(self) -> Vec<RecTree> {
        self.gene_trees
    }
}

/// Sample a single gene tree to match a pruned species tree.
///
/// Given a `RecTree` and the set of species leaf names being kept,
/// this function:
/// 1. Identifies which gene leaves map to the kept species
/// 2. Extracts the induced gene subtree for those leaves
/// 3. Remaps node_mapping and event_mapping to the new indices
fn sample_single_gene_tree(
    rt: &RecTree,
    species_leaf_names: &HashSet<&str>,
    species_old_to_new: &[Option<usize>],
    shared_species: &Arc<FlatTree>,
) -> Result<RecTree, String> {
    let gene_leaves_to_keep = find_gene_leaves_for_species(rt, species_leaf_names);

    if gene_leaves_to_keep.is_empty() {
        if rt.node_mapping.iter().all(|m| m.is_none()) {
            return Err("No gene tree leaves map to the sampled species".to_string());
        } else {
            return Err(
                "No gene tree leaves map to the sampled species, \
                 but some leaves were mapped to species".to_string()
            );
        }
    }

    let (sampled_gene_tree, gene_old_to_new) =
        extract_induced_subtree(&rt.gene_tree, &gene_leaves_to_keep)
            .ok_or_else(|| "Failed to extract gene subtree".to_string())?;

    let (new_node_mapping, new_event_mapping) = remap_gene_tree_indices(
        &sampled_gene_tree, &gene_old_to_new,
        &rt.node_mapping, &rt.event_mapping,
        species_old_to_new,
    )?;

    Ok(RecTree::new(
        Arc::clone(shared_species),
        sampled_gene_tree,
        new_node_mapping,
        new_event_mapping,
    ))
}

/// Find gene tree leaf indices whose species mapping is in the kept set.
fn find_gene_leaves_for_species(
    rt: &RecTree,
    species_leaf_names: &HashSet<&str>,
) -> HashSet<usize> {
    rt.gene_tree.nodes.iter().enumerate()
        .filter(|(idx, node)| {
            node.left_child.is_none()
                && node.right_child.is_none()
                && rt.node_mapping[*idx]
                    .map(|sp_idx| species_leaf_names.contains(rt.species_tree.nodes[sp_idx].name.as_str()))
                    .unwrap_or(false)
        })
        .map(|(idx, _)| idx)
        .collect()
}

/// Rebuild node_mapping and event_mapping for a sampled gene tree.
///
/// `gene_old_to_new` maps original gene indices to new indices (None if pruned).
/// `species_old_to_new` maps original species indices to new indices.
/// Returns `(new_node_mapping, new_event_mapping)` indexed by new gene tree nodes.
fn remap_gene_tree_indices(
    sampled_gene_tree: &FlatTree,
    gene_old_to_new: &[Option<usize>],
    old_node_mapping: &[Option<usize>],
    old_event_mapping: &[Event],
    species_old_to_new: &[Option<usize>],
) -> Result<(Vec<Option<usize>>, Vec<Event>), String> {
    let n = sampled_gene_tree.nodes.len();

    // Invert gene_old_to_new: for each new index, find the original index
    let mut gene_new_to_old: Vec<Option<usize>> = vec![None; n];
    for (old_idx, new_idx_opt) in gene_old_to_new.iter().enumerate() {
        if let Some(new_idx) = new_idx_opt {
            gene_new_to_old[*new_idx] = Some(old_idx);
        }
    }

    let mut new_node_mapping: Vec<Option<usize>> = vec![None; n];
    let mut new_event_mapping = vec![Event::Leaf; n];

    for new_idx in 0..n {
        let old_idx = gene_new_to_old[new_idx]
            .ok_or_else(|| format!("New gene node {} has no original mapping", new_idx))?;

        new_event_mapping[new_idx] = old_event_mapping[old_idx].clone();

        if let Some(old_species_idx) = old_node_mapping[old_idx] {
            if let Some(new_species_idx) = species_old_to_new[old_species_idx] {
                new_node_mapping[new_idx] = Some(new_species_idx);
            }
        }
    }

    Ok((new_node_mapping, new_event_mapping))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::newick::newick::parse_newick;

    fn make_species_tree(newick: &str) -> FlatTree {
        let mut nodes = parse_newick(newick).unwrap();
        let root = nodes.pop().unwrap();
        let mut tree = root.to_flat_tree();
        tree.assign_depths();
        tree
    }

    #[test]
    fn test_new_empty_forest() {
        let species = make_species_tree("((A:1,B:1)AB:1,C:2)root:0;");
        let forest = GeneForest::new(species);
        assert!(forest.is_empty());
        assert_eq!(forest.len(), 0);
    }

    #[test]
    fn test_push_and_iterate() {
        let species = make_species_tree("((A:1,B:1)AB:1,C:2)root:0;");
        let mut forest = GeneForest::new(species);

        // Push a minimal gene tree (just a root)
        let gene = FlatTree {
            nodes: vec![crate::node::FlatNode {
                name: "g0".to_string(),
                left_child: None,
                right_child: None,
                parent: None,
                depth: Some(0.0),
                length: 0.0,
                bd_event: None,
            }],
            root: 0,
        };
        forest.push(gene, vec![Some(0)], vec![Event::Leaf]);

        assert_eq!(forest.len(), 1);
        assert_eq!(forest.get(0).unwrap().gene_tree.nodes[0].name, "g0");

        // All gene trees share the same species tree Arc
        assert!(Arc::ptr_eq(&forest.species_tree, &forest.gene_trees[0].species_tree));
    }

    #[test]
    fn test_prune_to_species_tree() {
        let species = make_species_tree("((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;");
        let species_arc = Arc::new(species);

        // Create a gene tree with leaves mapping to A and C
        let gene = FlatTree {
            nodes: vec![
                crate::node::FlatNode {
                    name: "A_0".to_string(),
                    left_child: None, right_child: None,
                    parent: Some(2), depth: Some(0.0), length: 1.0, bd_event: None,
                },
                crate::node::FlatNode {
                    name: "C_0".to_string(),
                    left_child: None, right_child: None,
                    parent: Some(2), depth: Some(0.0), length: 1.0, bd_event: None,
                },
                crate::node::FlatNode {
                    name: "root".to_string(),
                    left_child: Some(0), right_child: Some(1),
                    parent: None, depth: Some(1.0), length: 0.0, bd_event: None,
                },
            ],
            root: 2,
        };

        // Map leaves to species: A_0 -> A (idx 1), C_0 -> C (idx 3)
        // Species tree nodes: root(0), AB(1), A(2), B(3), CD(4), C(5), D(6)
        // Wait, let's find the actual indices
        let a_idx = species_arc.nodes.iter().position(|n| n.name == "A").unwrap();
        let c_idx = species_arc.nodes.iter().position(|n| n.name == "C").unwrap();
        let root_sp_idx = species_arc.nodes.iter().position(|n| n.name == "root").unwrap();

        let node_mapping = vec![Some(a_idx), Some(c_idx), Some(root_sp_idx)];
        let event_mapping = vec![Event::Leaf, Event::Leaf, Event::Speciation];

        let rt = RecTree::new(Arc::clone(&species_arc), gene, node_mapping, event_mapping);
        let forest = GeneForest::from_rec_trees(species_arc, vec![rt]);

        // Target species tree with only A and C
        let target = make_species_tree("(A:1,C:1)root:0;");
        let pruned = forest.prune_to_species_tree(&target).unwrap();

        // Check pruned species tree has only A and C as leaves
        let leaf_names: Vec<String> = pruned.species_tree().nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .map(|n| n.name.clone())
            .collect();
        assert!(leaf_names.contains(&"A".to_string()));
        assert!(leaf_names.contains(&"C".to_string()));
        assert!(!leaf_names.contains(&"B".to_string()));
        assert!(!leaf_names.contains(&"D".to_string()));

        // Gene tree should be preserved (both leaves map to sampled species)
        assert_eq!(pruned.len(), 1);
    }

    #[test]
    fn test_prune_to_species_tree_empty_target() {
        let species = make_species_tree("((A:1,B:1)AB:1,C:2)root:0;");
        let forest = GeneForest::new(species);

        let empty_tree = FlatTree { nodes: vec![], root: 0 };
        let result = forest.prune_to_species_tree(&empty_tree);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_rec_trees_shares_arc() {
        let species = make_species_tree("((A:1,B:1)AB:1,C:2)root:0;");
        let species_arc = Arc::new(species);

        let rt1 = RecTree::new(
            Arc::clone(&species_arc),
            FlatTree { nodes: vec![], root: 0 },
            vec![],
            vec![],
        );
        let rt2 = RecTree::new(
            Arc::new(make_species_tree("(X:1,Y:1)root:0;")), // different Arc
            FlatTree { nodes: vec![], root: 0 },
            vec![],
            vec![],
        );

        let forest = GeneForest::from_rec_trees(Arc::clone(&species_arc), vec![rt1, rt2]);

        // Both should now share the same species tree Arc
        assert!(Arc::ptr_eq(&forest.species_tree, &forest.gene_trees[0].species_tree));
        assert!(Arc::ptr_eq(&forest.species_tree, &forest.gene_trees[1].species_tree));
    }
}

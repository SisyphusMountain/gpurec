//! Sampling utilities for GPUREC backtracking payloads.
//!
//! The JSON schema mirrors the Python exporter. Probability-like tensors are
//! base-2 log values, with `-1e300` used as the practical negative-infinity
//! sentinel. Species indices use the postorder order supplied by
//! `species_names_postorder`.

use rand::distributions::Uniform;
use rand::prelude::*;
use rustree::{parse_newick, Event, FlatNode, FlatTree, RecTree};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const NEG_INF: f64 = -1.0e300;

#[derive(Debug, thiserror::Error)]
pub enum BacktrackError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("sampling failed: {0}")]
    Sampling(String),
    #[error(transparent)]
    Rustree(#[from] rustree::RustreeError),
}

/// Row-major matrix used by the JSON backtracking schema.
///
/// `data[row * cols + col]` must exist for every `row < rows` and `col < cols`.
/// Matrix entries are base-2 log values unless a specific field documents a
/// different unit.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Matrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major matrix values with length exactly `rows * cols`.
    pub data: Vec<f64>,
}

impl Matrix {
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    fn validate(&self, name: &str) -> Result<(), BacktrackError> {
        let expected = self.rows.checked_mul(self.cols).ok_or_else(|| {
            BacktrackError::InvalidInput(format!(
                "{name} shape is {}x{} but rows*cols overflows usize",
                self.rows, self.cols
            ))
        })?;
        if expected != self.data.len() {
            return Err(BacktrackError::InvalidInput(format!(
                "{name} shape is {}x{} but has {} values",
                self.rows,
                self.cols,
                self.data.len()
            )));
        }
        Ok(())
    }
}

/// One possible binary split for a reconciliation clade.
///
/// `parent`, `left`, and `right` are clade indices into the payload arrays.
/// `log_prob` is the base-2 log conditional split probability.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SplitInput {
    pub parent: usize,
    pub left: usize,
    pub right: usize,
    pub log_prob: f64,
}

/// Complete JSON payload consumed by the Rust backtracking sampler.
///
/// Clade-indexed arrays use the Python exporter clade order. Species-indexed
/// arrays use `species_names_postorder`, matching the postorder indexing of the
/// exported species tree. `pi`, `e`, `log_p_s`, `log_p_d`, and `max_transfer`
/// are base-2 log values. `origination_probs`, when present, are ordinary
/// nonnegative weights over species; nonpositive weights are treated as
/// impossible origination species during sampling.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BacktrackInput {
    /// Species tree in Newick format.
    pub species_newick: String,
    /// Species names in GPUREC postorder index order.
    pub species_names_postorder: Vec<String>,
    /// Root clade index for the gene-family reconciliation.
    pub root_clade: usize,
    /// Optional species index for each leaf clade.
    pub leaf_species: Vec<Option<usize>>,
    /// Leaf label for each clade, empty for internal clades.
    pub clade_leaf_labels: Vec<String>,
    /// Candidate binary splits for internal clades.
    pub splits: Vec<SplitInput>,
    /// Clade-by-species Pi matrix in base-2 log space.
    pub pi: Matrix,
    /// Species extinction/survival fixed-point values in base-2 log space.
    pub e: Vec<f64>,
    /// Speciation probabilities by species in base-2 log space.
    pub log_p_s: Vec<f64>,
    /// Duplication probabilities by species in base-2 log space.
    pub log_p_d: Vec<f64>,
    /// Maximum transfer probabilities by donor species in base-2 log space.
    pub max_transfer: Vec<f64>,
    /// Optional nonnegative origination weights by species.
    pub origination_probs: Option<Vec<f64>>,
    /// Optional deterministic random seed.
    pub seed: Option<u64>,
    /// Optional cap on sampled event expansions.
    pub max_events: Option<usize>,
}

#[derive(Clone, Debug)]
struct SpeciesTopology {
    rust_tree: FlatTree,
    gp_to_rust: Vec<usize>,
    child1: Vec<Option<usize>>,
    child2: Vec<Option<usize>>,
    ancestors: Vec<HashSet<usize>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Term {
    HiddenDupLoss,
    HiddenTransferLossRecipient,
    HiddenTransferLossDonor,
    HiddenSpeciationLeft,
    HiddenSpeciationRight,
    Leaf,
    SplitDup(usize),
    SplitTransferRight(usize),
    SplitTransferLeft(usize),
    SplitSpeciation(usize, bool),
}

#[derive(Clone, Debug)]
struct Candidate {
    term: Term,
    log_weight: f64,
}

#[derive(Clone, Debug)]
struct WorkItem {
    node_idx: usize,
    clade: Option<usize>,
    species: usize,
}

#[derive(Clone, Debug)]
struct PreparedBacktracker<'a> {
    input: &'a BacktrackInput,
    species: SpeciesTopology,
    splits_by_parent: Vec<Vec<usize>>,
    pibar: Matrix,
    ebar: Vec<f64>,
    max_events: usize,
}

#[derive(Clone, Debug)]
struct Sampler<'a> {
    prepared: &'a PreparedBacktracker<'a>,
    rng: StdRng,
    nodes: Vec<FlatNode>,
    node_mapping: Vec<Option<usize>>,
    event_mapping: Vec<Event>,
    sampled_terms: usize,
}

/// Sample one reconciliation as RecPhyloXML.
pub fn sample_recphyloxml(input: &BacktrackInput) -> Result<String, BacktrackError> {
    let prepared = PreparedBacktracker::new(input)?;
    prepared.sample_to_xml(input.seed.unwrap_or(0))
}

/// Sample multiple reconciliations with consecutive seeds.
///
/// `num_samples` must be positive. The first sample uses `base_seed`, and each
/// subsequent sample increments that seed by one.
pub fn sample_recphyloxmls(
    input: &BacktrackInput,
    num_samples: usize,
    base_seed: u64,
) -> Result<Vec<String>, BacktrackError> {
    if num_samples == 0 {
        return Err(BacktrackError::InvalidInput(
            "num_samples must be positive".to_string(),
        ));
    }
    let prepared = PreparedBacktracker::new(input)?;
    let mut out = Vec::with_capacity(num_samples);
    for sample_idx in 0..num_samples {
        let seed_offset = u64::try_from(sample_idx).map_err(|_| {
            BacktrackError::InvalidInput(format!("sample index {sample_idx} does not fit in u64"))
        })?;
        let seed = base_seed.checked_add(seed_offset).ok_or_else(|| {
            BacktrackError::InvalidInput(format!(
                "seed range exceeds u64 maximum: base_seed={base_seed} sample_idx={sample_idx}"
            ))
        })?;
        out.push(prepared.sample_to_xml(seed)?);
    }
    Ok(out)
}

impl<'a> PreparedBacktracker<'a> {
    fn new(input: &'a BacktrackInput) -> Result<Self, BacktrackError> {
        input.pi.validate("pi")?;
        let c = input.pi.rows;
        let s = input.pi.cols;
        validate_len("leaf_species", input.leaf_species.len(), c)?;
        validate_len("clade_leaf_labels", input.clade_leaf_labels.len(), c)?;
        validate_len("e", input.e.len(), s)?;
        validate_len("log_p_s", input.log_p_s.len(), s)?;
        validate_len("log_p_d", input.log_p_d.len(), s)?;
        validate_len("max_transfer", input.max_transfer.len(), s)?;
        if let Some(probs) = &input.origination_probs {
            validate_len("origination_probs", probs.len(), s)?;
        }
        if input.max_events == Some(0) {
            return Err(BacktrackError::InvalidInput(
                "max_events must be positive".to_string(),
            ));
        }
        if input.root_clade >= c {
            return Err(BacktrackError::InvalidInput(format!(
                "root_clade {} is out of bounds for {c} clades",
                input.root_clade
            )));
        }

        let species =
            parse_species_topology(&input.species_newick, &input.species_names_postorder)?;
        validate_len("species_names_postorder", species.gp_to_rust.len(), s)?;

        let mut splits_by_parent = vec![Vec::new(); c];
        for (idx, split) in input.splits.iter().enumerate() {
            if split.parent >= c || split.left >= c || split.right >= c {
                return Err(BacktrackError::InvalidInput(format!(
                    "split {idx} has clade outside 0..{c}: parent={} left={} right={}",
                    split.parent, split.left, split.right
                )));
            }
            splits_by_parent[split.parent].push(idx);
        }

        let pibar = compute_pibar(&input.pi, &input.max_transfer, &species);
        let ebar = compute_ebar(&input.e, &input.max_transfer, &species);
        Ok(Self {
            input,
            species,
            splits_by_parent,
            pibar,
            ebar,
            max_events: input.max_events.unwrap_or(100_000),
        })
    }

    fn sample_to_xml(&'a self, seed: u64) -> Result<String, BacktrackError> {
        let mut sampler = Sampler::new(self, seed);
        let rec_tree = sampler.sample()?;
        Ok(rec_tree.to_xml())
    }
}

impl<'a> Sampler<'a> {
    fn new(prepared: &'a PreparedBacktracker<'a>, seed: u64) -> Self {
        Self {
            prepared,
            rng: StdRng::seed_from_u64(seed),
            nodes: Vec::new(),
            node_mapping: Vec::new(),
            event_mapping: Vec::new(),
            sampled_terms: 0,
        }
    }

    fn sample(&mut self) -> Result<RecTree, BacktrackError> {
        let root_species = self.sample_root_species()?;
        let root = self.add_node("", Event::Speciation, root_species, None);
        let mut stack = vec![WorkItem {
            node_idx: root,
            clade: Some(self.prepared.input.root_clade),
            species: root_species,
        }];

        while let Some(item) = stack.pop() {
            if self.nodes.len() > self.prepared.max_events {
                return Err(BacktrackError::Sampling(format!(
                    "sample exceeded max_events={}",
                    self.prepared.max_events
                )));
            }
            self.sampled_terms += 1;
            if self.sampled_terms > self.prepared.max_events {
                return Err(BacktrackError::Sampling(format!(
                    "sample exceeded max_events={} sampled terms",
                    self.prepared.max_events
                )));
            }
            if let Some(clade) = item.clade {
                let children = self.expand_state(item.node_idx, clade, item.species)?;
                stack.extend(children.into_iter().rev());
            }
        }

        Ok(RecTree::new_owned(
            self.prepared.species.rust_tree.clone(),
            FlatTree {
                nodes: std::mem::take(&mut self.nodes),
                root,
            },
            std::mem::take(&mut self.node_mapping),
            std::mem::take(&mut self.event_mapping),
        ))
    }

    fn sample_root_species(&mut self) -> Result<usize, BacktrackError> {
        let input = self.prepared.input;
        let s = input.pi.cols;
        let mut candidates = Vec::with_capacity(s);
        for species in 0..s {
            let prior = match &input.origination_probs {
                Some(probs) => {
                    if probs[species] <= 0.0 {
                        NEG_INF
                    } else {
                        probs[species].log2()
                    }
                }
                None => -(s as f64).log2(),
            };
            candidates.push((species, prior + input.pi.get(input.root_clade, species)));
        }
        sample_index(&candidates, &mut self.rng)
    }

    fn expand_state(
        &mut self,
        node_idx: usize,
        clade: usize,
        species: usize,
    ) -> Result<Vec<WorkItem>, BacktrackError> {
        let term = self.sample_term(clade, species)?;
        self.apply_term(node_idx, clade, species, term)
    }

    fn sample_term(&mut self, clade: usize, species: usize) -> Result<Term, BacktrackError> {
        let input = self.prepared.input;
        let species_topology = &self.prepared.species;
        let mut candidates =
            Vec::with_capacity(6 + self.prepared.splits_by_parent[clade].len() * 5);
        let pi_cs = input.pi.get(clade, species);
        let e_s = input.e[species];
        let child1 = species_topology.child1[species];
        let child2 = species_topology.child2[species];

        candidates.push(Candidate {
            term: Term::HiddenDupLoss,
            log_weight: 1.0 + input.log_p_d[species] + e_s + pi_cs,
        });
        candidates.push(Candidate {
            term: Term::HiddenTransferLossRecipient,
            log_weight: pi_cs + self.prepared.ebar[species],
        });
        candidates.push(Candidate {
            term: Term::HiddenTransferLossDonor,
            log_weight: self.prepared.pibar.get(clade, species) + e_s,
        });

        if let (Some(c1), Some(c2)) = (child1, child2) {
            candidates.push(Candidate {
                term: Term::HiddenSpeciationLeft,
                log_weight: input.log_p_s[species] + input.e[c2] + input.pi.get(clade, c1),
            });
            candidates.push(Candidate {
                term: Term::HiddenSpeciationRight,
                log_weight: input.log_p_s[species] + input.e[c1] + input.pi.get(clade, c2),
            });
        }

        if input.leaf_species[clade] == Some(species) {
            candidates.push(Candidate {
                term: Term::Leaf,
                log_weight: input.log_p_s[species],
            });
        }

        for split_idx in &self.prepared.splits_by_parent[clade] {
            let split = &input.splits[*split_idx];
            let left = split.left;
            let right = split.right;
            let base = split.log_prob;
            candidates.push(Candidate {
                term: Term::SplitDup(*split_idx),
                log_weight: base
                    + input.log_p_d[species]
                    + input.pi.get(left, species)
                    + input.pi.get(right, species),
            });
            candidates.push(Candidate {
                term: Term::SplitTransferRight(*split_idx),
                log_weight: base
                    + input.pi.get(left, species)
                    + self.prepared.pibar.get(right, species),
            });
            candidates.push(Candidate {
                term: Term::SplitTransferLeft(*split_idx),
                log_weight: base
                    + input.pi.get(right, species)
                    + self.prepared.pibar.get(left, species),
            });
            if let (Some(c1), Some(c2)) = (child1, child2) {
                candidates.push(Candidate {
                    term: Term::SplitSpeciation(*split_idx, false),
                    log_weight: base
                        + input.log_p_s[species]
                        + input.pi.get(left, c1)
                        + input.pi.get(right, c2),
                });
                candidates.push(Candidate {
                    term: Term::SplitSpeciation(*split_idx, true),
                    log_weight: base
                        + input.log_p_s[species]
                        + input.pi.get(right, c1)
                        + input.pi.get(left, c2),
                });
            }
        }

        let weighted: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, c.log_weight))
            .collect();
        let idx = sample_index(&weighted, &mut self.rng)?;
        Ok(candidates[idx].term)
    }

    fn apply_term(
        &mut self,
        node_idx: usize,
        clade: usize,
        species: usize,
        term: Term,
    ) -> Result<Vec<WorkItem>, BacktrackError> {
        match term {
            Term::Leaf => {
                self.nodes[node_idx].name = leaf_name(self.prepared.input, clade);
                self.event_mapping[node_idx] = Event::Leaf;
                Ok(Vec::new())
            }
            Term::HiddenDupLoss => {
                // AleRax marginalizes this unobservable same-species self-loop
                // out of RecPhyloXML, unlike visible speciation/transfer losses.
                Ok(vec![WorkItem {
                    node_idx,
                    clade: Some(clade),
                    species,
                }])
            }
            Term::HiddenTransferLossRecipient => {
                // A transfer to an immediately extinct recipient leaves the
                // retained lineage on the donor branch, so AleRax does not emit
                // a RecPhyloXML transfer-loss node for it.
                Ok(vec![WorkItem {
                    node_idx,
                    clade: Some(clade),
                    species,
                }])
            }
            Term::HiddenTransferLossDonor => {
                let recipient = self.sample_pibar_recipient(clade, species)?;
                self.event_mapping[node_idx] = Event::Transfer;
                let loss = self.add_node("loss", Event::Loss, species, Some(node_idx));
                let cont = self.add_node("", Event::Leaf, recipient, Some(node_idx));
                self.set_children_random(node_idx, loss, cont);
                Ok(vec![WorkItem {
                    node_idx: cont,
                    clade: Some(clade),
                    species: recipient,
                }])
            }
            Term::HiddenSpeciationLeft | Term::HiddenSpeciationRight => {
                let c1 = self.prepared.species.child1[species].ok_or_else(|| {
                    BacktrackError::Sampling(
                        "sampled hidden speciation at a leaf species".to_string(),
                    )
                })?;
                let c2 = self.prepared.species.child2[species].ok_or_else(|| {
                    BacktrackError::Sampling(
                        "sampled hidden speciation at a unary species".to_string(),
                    )
                })?;
                self.event_mapping[node_idx] = Event::Speciation;
                let (cont_species, loss_species) = if term == Term::HiddenSpeciationLeft {
                    (c1, c2)
                } else {
                    (c2, c1)
                };
                let cont = self.add_node("", Event::Leaf, cont_species, Some(node_idx));
                let loss = self.add_node("loss", Event::Loss, loss_species, Some(node_idx));
                self.set_children(node_idx, cont, loss);
                Ok(vec![WorkItem {
                    node_idx: cont,
                    clade: Some(clade),
                    species: cont_species,
                }])
            }
            Term::SplitDup(split_idx) => {
                let split = self.prepared.input.splits[split_idx].clone();
                self.event_mapping[node_idx] = Event::Duplication;
                let left = self.add_node("", Event::Leaf, species, Some(node_idx));
                let right = self.add_node("", Event::Leaf, species, Some(node_idx));
                self.set_children(node_idx, left, right);
                Ok(vec![
                    WorkItem {
                        node_idx: left,
                        clade: Some(split.left),
                        species,
                    },
                    WorkItem {
                        node_idx: right,
                        clade: Some(split.right),
                        species,
                    },
                ])
            }
            Term::SplitTransferRight(split_idx) => {
                let split = self.prepared.input.splits[split_idx].clone();
                let recipient = self.sample_pibar_recipient(split.right, species)?;
                self.event_mapping[node_idx] = Event::Transfer;
                let donor_child = self.add_node("", Event::Leaf, species, Some(node_idx));
                let recipient_child = self.add_node("", Event::Leaf, recipient, Some(node_idx));
                self.set_children(node_idx, donor_child, recipient_child);
                Ok(vec![
                    WorkItem {
                        node_idx: donor_child,
                        clade: Some(split.left),
                        species,
                    },
                    WorkItem {
                        node_idx: recipient_child,
                        clade: Some(split.right),
                        species: recipient,
                    },
                ])
            }
            Term::SplitTransferLeft(split_idx) => {
                let split = self.prepared.input.splits[split_idx].clone();
                let recipient = self.sample_pibar_recipient(split.left, species)?;
                self.event_mapping[node_idx] = Event::Transfer;
                let donor_child = self.add_node("", Event::Leaf, species, Some(node_idx));
                let recipient_child = self.add_node("", Event::Leaf, recipient, Some(node_idx));
                self.set_children(node_idx, recipient_child, donor_child);
                Ok(vec![
                    WorkItem {
                        node_idx: recipient_child,
                        clade: Some(split.left),
                        species: recipient,
                    },
                    WorkItem {
                        node_idx: donor_child,
                        clade: Some(split.right),
                        species,
                    },
                ])
            }
            Term::SplitSpeciation(split_idx, swapped) => {
                let split = self.prepared.input.splits[split_idx].clone();
                let c1 = self.prepared.species.child1[species].ok_or_else(|| {
                    BacktrackError::Sampling(
                        "sampled split speciation at a leaf species".to_string(),
                    )
                })?;
                let c2 = self.prepared.species.child2[species].ok_or_else(|| {
                    BacktrackError::Sampling(
                        "sampled split speciation at a unary species".to_string(),
                    )
                })?;
                self.event_mapping[node_idx] = Event::Speciation;
                let left_node = self.add_node("", Event::Leaf, c1, Some(node_idx));
                let right_node = self.add_node("", Event::Leaf, c2, Some(node_idx));
                self.set_children(node_idx, left_node, right_node);
                if swapped {
                    Ok(vec![
                        WorkItem {
                            node_idx: left_node,
                            clade: Some(split.right),
                            species: c1,
                        },
                        WorkItem {
                            node_idx: right_node,
                            clade: Some(split.left),
                            species: c2,
                        },
                    ])
                } else {
                    Ok(vec![
                        WorkItem {
                            node_idx: left_node,
                            clade: Some(split.left),
                            species: c1,
                        },
                        WorkItem {
                            node_idx: right_node,
                            clade: Some(split.right),
                            species: c2,
                        },
                    ])
                }
            }
        }
    }

    fn sample_pibar_recipient(
        &mut self,
        clade: usize,
        donor: usize,
    ) -> Result<usize, BacktrackError> {
        let input = self.prepared.input;
        let candidates = (0..input.pi.cols)
            .filter(|recipient| !self.prepared.species.ancestors[donor].contains(recipient))
            .map(|recipient| {
                (
                    recipient,
                    input.pi.get(clade, recipient) + input.max_transfer[donor],
                )
            })
            .collect::<Vec<_>>();
        sample_index(&candidates, &mut self.rng)
    }

    fn add_node(
        &mut self,
        name: impl Into<String>,
        event: Event,
        gp_species: usize,
        parent: Option<usize>,
    ) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(FlatNode {
            name: name.into(),
            left_child: None,
            right_child: None,
            parent,
            depth: None,
            length: 0.0,
            bd_event: None,
        });
        self.node_mapping
            .push(Some(self.prepared.species.gp_to_rust[gp_species]));
        self.event_mapping.push(event);
        idx
    }

    fn set_children(&mut self, parent: usize, left: usize, right: usize) {
        self.nodes[parent].left_child = Some(left);
        self.nodes[parent].right_child = Some(right);
    }

    fn set_children_random(&mut self, parent: usize, a: usize, b: usize) {
        if self.rng.gen_bool(0.5) {
            self.set_children(parent, a, b);
        } else {
            self.set_children(parent, b, a);
        }
    }
}

fn validate_len(name: &str, got: usize, expected: usize) -> Result<(), BacktrackError> {
    if got != expected {
        Err(BacktrackError::InvalidInput(format!(
            "{name} has length {got}, expected {expected}"
        )))
    } else {
        Ok(())
    }
}

fn leaf_name(input: &BacktrackInput, clade: usize) -> String {
    let label = &input.clade_leaf_labels[clade];
    if label.is_empty() {
        format!("leaf_{clade}")
    } else {
        label.clone()
    }
}

fn parse_species_topology(
    newick: &str,
    expected_postorder_names: &[String],
) -> Result<SpeciesTopology, BacktrackError> {
    let mut parsed = parse_newick(newick)?;
    if parsed.len() != 1 {
        return Err(BacktrackError::InvalidInput(format!(
            "species_newick must contain exactly one tree, found {}",
            parsed.len()
        )));
    }
    let mut rust_tree = parsed.remove(0).to_flat_tree();
    rust_tree.assign_depths();
    let postorder = rust_tree.postorder_indices();
    if postorder.len() != expected_postorder_names.len() {
        return Err(BacktrackError::InvalidInput(format!(
            "species tree has {} nodes but species_names_postorder has {}",
            postorder.len(),
            expected_postorder_names.len()
        )));
    }
    for (gp_idx, &rust_idx) in postorder.iter().enumerate() {
        let expected = &expected_postorder_names[gp_idx];
        let found = &rust_tree.nodes[rust_idx].name;
        if !expected.is_empty() && !found.is_empty() && expected != found {
            return Err(BacktrackError::InvalidInput(format!(
                "species postorder mismatch at {gp_idx}: gpurec={expected:?}, rustree={found:?}"
            )));
        }
        if found.is_empty() && !expected.is_empty() {
            rust_tree.nodes[rust_idx].name = expected.clone();
        }
    }

    let s = postorder.len();
    let mut rust_to_gp = vec![usize::MAX; rust_tree.nodes.len()];
    for (gp_idx, &rust_idx) in postorder.iter().enumerate() {
        rust_to_gp[rust_idx] = gp_idx;
    }
    let mut parent = vec![None; s];
    let mut child1 = vec![None; s];
    let mut child2 = vec![None; s];
    for (gp_idx, &rust_idx) in postorder.iter().enumerate() {
        if let Some(p) = rust_tree.nodes[rust_idx].parent {
            parent[gp_idx] = Some(rust_to_gp[p]);
        }
        if let Some(left) = rust_tree.nodes[rust_idx].left_child {
            child1[gp_idx] = Some(rust_to_gp[left]);
        }
        if let Some(right) = rust_tree.nodes[rust_idx].right_child {
            child2[gp_idx] = Some(rust_to_gp[right]);
        }
        if child1[gp_idx].is_some() != child2[gp_idx].is_some() {
            return Err(BacktrackError::InvalidInput(format!(
                "species node {gp_idx} is unary; gpurec requires binary species trees"
            )));
        }
    }

    let mut ancestors = vec![HashSet::new(); s];
    for start in 0..s {
        let mut cur = Some(start);
        while let Some(idx) = cur {
            ancestors[start].insert(idx);
            cur = parent[idx];
        }
    }

    Ok(SpeciesTopology {
        rust_tree,
        gp_to_rust: postorder,
        child1,
        child2,
        ancestors,
    })
}

fn compute_pibar(pi: &Matrix, max_transfer: &[f64], species: &SpeciesTopology) -> Matrix {
    let mut data = vec![NEG_INF; pi.rows * pi.cols];
    for clade in 0..pi.rows {
        for donor in 0..pi.cols {
            let mut terms = Vec::new();
            for recipient in 0..pi.cols {
                if !species.ancestors[donor].contains(&recipient) {
                    terms.push(pi.get(clade, recipient));
                }
            }
            data[clade * pi.cols + donor] = logsumexp2(&terms) + max_transfer[donor];
        }
    }
    Matrix {
        rows: pi.rows,
        cols: pi.cols,
        data,
    }
}

fn compute_ebar(e: &[f64], max_transfer: &[f64], species: &SpeciesTopology) -> Vec<f64> {
    (0..e.len())
        .map(|donor| {
            let terms = (0..e.len())
                .filter(|recipient| !species.ancestors[donor].contains(recipient))
                .map(|recipient| e[recipient])
                .collect::<Vec<_>>();
            logsumexp2(&terms) + max_transfer[donor]
        })
        .collect()
}

fn logsumexp2(values: &[f64]) -> f64 {
    let max = values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > NEG_INF / 2.0)
        .fold(NEG_INF, f64::max);
    if max <= NEG_INF / 2.0 {
        return NEG_INF;
    }
    let sum = values.iter().map(|v| 2.0_f64.powf(*v - max)).sum::<f64>();
    max + sum.log2()
}

fn sample_index<T: Copy>(weighted: &[(T, f64)], rng: &mut StdRng) -> Result<T, BacktrackError> {
    let logs = weighted.iter().map(|(_, w)| *w).collect::<Vec<_>>();
    let norm = logsumexp2(&logs);
    if norm <= NEG_INF / 2.0 || !norm.is_finite() {
        return Err(BacktrackError::Sampling(
            "all candidate backtracking weights are zero".to_string(),
        ));
    }
    let dist = Uniform::new(0.0, 1.0);
    let mut draw = dist.sample(rng);
    let mut last = weighted[0].0;
    for (item, log_w) in weighted {
        last = *item;
        if *log_w <= NEG_INF / 2.0 {
            continue;
        }
        let p = 2.0_f64.powf(*log_w - norm);
        if draw <= p {
            return Ok(*item);
        }
        draw -= p;
    }
    Ok(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speciation_input() -> BacktrackInput {
        let neg = NEG_INF;
        BacktrackInput {
            species_newick: "(A:1,B:1)Root:0;".to_string(),
            species_names_postorder: vec!["A".into(), "B".into(), "Root".into()],
            root_clade: 2,
            leaf_species: vec![Some(0), Some(1), None],
            clade_leaf_labels: vec!["a".into(), "b".into(), String::new()],
            splits: vec![SplitInput {
                parent: 2,
                left: 0,
                right: 1,
                log_prob: 0.0,
            }],
            pi: Matrix {
                rows: 3,
                cols: 3,
                data: vec![0.0, neg, neg, neg, 0.0, neg, neg, neg, 0.0],
            },
            e: vec![neg, neg, neg],
            log_p_s: vec![0.0, 0.0, 0.0],
            log_p_d: vec![neg, neg, neg],
            max_transfer: vec![neg, neg, neg],
            origination_probs: Some(vec![0.0, 0.0, 1.0]),
            seed: Some(7),
            max_events: Some(32),
        }
    }

    fn transfer_input(seed: u64) -> BacktrackInput {
        let neg = NEG_INF;
        BacktrackInput {
            species_newick: "(A:1,B:1)Root:0;".to_string(),
            species_names_postorder: vec!["A".into(), "B".into(), "Root".into()],
            root_clade: 0,
            leaf_species: vec![Some(1)],
            clade_leaf_labels: vec!["b".into()],
            splits: Vec::new(),
            pi: Matrix {
                rows: 1,
                cols: 3,
                data: vec![0.0, 0.0, neg],
            },
            e: vec![0.0, neg, neg],
            log_p_s: vec![0.0, 0.0, neg],
            log_p_d: vec![neg, neg, neg],
            max_transfer: vec![0.0, neg, neg],
            origination_probs: Some(vec![1.0, 0.0, 0.0]),
            seed: Some(seed),
            max_events: Some(32),
        }
    }

    #[test]
    fn samples_forced_speciation_xml() {
        let input = speciation_input();
        let xml = sample_recphyloxml(&input).unwrap();
        assert!(xml.contains("<speciation speciesLocation=\"Root\"/>"));
        assert!(xml.contains("<leaf speciesLocation=\"A\"/>"));
        assert!(xml.contains("<leaf speciesLocation=\"B\"/>"));
        let parsed = RecTree::from_xml(&xml).unwrap();
        assert_eq!(
            parsed
                .event_mapping
                .iter()
                .filter(|e| **e == Event::Leaf)
                .count(),
            2
        );
        assert_eq!(
            parsed
                .event_mapping
                .iter()
                .filter(|e| **e == Event::Speciation)
                .count(),
            1
        );
    }

    #[test]
    fn seeded_sampling_replays_transfer_xml() {
        let input = transfer_input(19);
        let first = sample_recphyloxml(&input).unwrap();
        let second = sample_recphyloxml(&input).unwrap();

        assert_eq!(first, second);
        assert!(first.contains("<branchingOut speciesLocation=\"A\"/>"));
        assert!(first.contains("<transferBack destinationSpecies=\"B\"/>"));
        assert!(first.contains("<leaf speciesLocation=\"B\"/>"));
    }

    #[test]
    fn multi_sample_uses_consecutive_seeds() {
        let base_seed = 23;
        let input = transfer_input(base_seed);
        let batch = sample_recphyloxmls(&input, 3, base_seed).unwrap();

        let expected = (0..3)
            .map(|idx| {
                let mut seeded = input.clone();
                seeded.seed = Some(base_seed + idx as u64);
                sample_recphyloxml(&seeded).unwrap()
            })
            .collect::<Vec<_>>();

        assert_eq!(batch, expected);
    }

    #[test]
    fn multi_sample_rejects_seed_range_overflow() {
        let input = speciation_input();

        let err = sample_recphyloxmls(&input, 2, u64::MAX)
            .unwrap_err()
            .to_string();

        assert!(err.contains("seed range exceeds u64 maximum"));
    }

    #[test]
    fn rejects_nonpositive_max_events() {
        let mut input = speciation_input();
        input.max_events = Some(0);

        let err = sample_recphyloxml(&input).unwrap_err().to_string();

        assert!(err.contains("max_events must be positive"));
    }

    #[test]
    fn max_events_caps_expansion() {
        let mut input = speciation_input();
        input.max_events = Some(1);

        let err = sample_recphyloxml(&input).unwrap_err().to_string();

        assert!(err.contains("sample exceeded max_events=1"));
    }

    #[test]
    fn hidden_dup_loss_requeues_without_xml_nodes() {
        let input = speciation_input();
        let prepared = PreparedBacktracker::new(&input).unwrap();
        let mut sampler = Sampler::new(&prepared, 3);
        let root = sampler.add_node("", Event::Speciation, 2, None);

        let children = sampler.apply_term(root, 2, 2, Term::HiddenDupLoss).unwrap();

        assert_eq!(sampler.nodes.len(), 1);
        assert_eq!(sampler.event_mapping[root], Event::Speciation);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].node_idx, root);
        assert_eq!(children[0].clade, Some(2));
        assert_eq!(children[0].species, 2);
    }

    #[test]
    fn hidden_transfer_recipient_loss_requeues_without_xml_nodes() {
        let input = speciation_input();
        let prepared = PreparedBacktracker::new(&input).unwrap();
        let mut sampler = Sampler::new(&prepared, 3);
        let root = sampler.add_node("", Event::Speciation, 2, None);

        let children = sampler
            .apply_term(root, 2, 2, Term::HiddenTransferLossRecipient)
            .unwrap();

        assert_eq!(sampler.nodes.len(), 1);
        assert_eq!(sampler.event_mapping[root], Event::Speciation);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].node_idx, root);
        assert_eq!(children[0].clade, Some(2));
        assert_eq!(children[0].species, 2);
    }

    #[test]
    fn rejects_bad_matrix_shape() {
        let mut input = speciation_input();
        input.pi.data.pop();
        let err = sample_recphyloxml(&input).unwrap_err().to_string();
        assert!(err.contains("pi shape"));
    }

    #[test]
    fn rejects_matrix_shape_overflow_without_panicking() {
        let matrix = Matrix {
            rows: usize::MAX,
            cols: 2,
            data: Vec::new(),
        };

        let err = matrix.validate("pi").unwrap_err().to_string();

        assert!(err.contains("pi shape"));
        assert!(err.contains("overflows usize"));
    }
}

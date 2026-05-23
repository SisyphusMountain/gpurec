//! Sampling utilities for GPUREC backtracking payloads.
//!
//! The JSON schema mirrors the Python exporter. Probability-like tensors are
//! base-2 log values, with `-1e300` used as the practical negative-infinity
//! sentinel. Species indices use the postorder order supplied by
//! `species_names_postorder`. `origination_probs`, when present, are ordinary
//! nonnegative weights over species; zero weights are treated as impossible.

use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use rustree::{parse_newick, Event, FlatNode, FlatTree, RecTree};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use flate2::{write::GzEncoder, Compression as GzipCompression};
#[cfg(feature = "python-extension")]
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
#[cfg(feature = "python-extension")]
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList},
};

const NEG_INF: f64 = -1.0e300;

#[derive(Debug, thiserror::Error)]
pub enum BacktrackError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("sampling failed: {0}")]
    Sampling(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Rustree(#[from] rustree::RustreeError),
}

/// Per-scenario event counts using GPUREC/AleRax event labels.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct EventCounts {
    #[serde(rename = "S")]
    pub s: usize,
    #[serde(rename = "SL")]
    pub sl: usize,
    #[serde(rename = "D")]
    pub d: usize,
    #[serde(rename = "DL")]
    pub dl: usize,
    #[serde(rename = "T")]
    pub t: usize,
    #[serde(rename = "TL")]
    pub tl: usize,
    #[serde(rename = "L")]
    pub l: usize,
    #[serde(rename = "Leaf")]
    pub leaf: usize,
}

/// Metadata for one sampled scenario.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SampleSummary {
    pub seed: u64,
    pub event_counts: EventCounts,
    /// Base-2 log probability of the sampled backtracking path.
    pub log_probability: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputCompression {
    None,
    Gzip,
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

#[derive(Clone, Copy, Debug)]
struct MatrixView<'a> {
    rows: usize,
    cols: usize,
    data: &'a [f64],
}

impl MatrixView<'_> {
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
        validate_finite_values(name, self.data)?;
        Ok(())
    }
}

impl<'a> From<&'a Matrix> for MatrixView<'a> {
    fn from(matrix: &'a Matrix) -> Self {
        Self {
            rows: matrix.rows,
            cols: matrix.cols,
            data: &matrix.data,
        }
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
/// exported species tree. `pi`, `pibar`, `e`, `ebar`, `log_p_s`, `log_p_d`,
/// and `max_transfer` are base-2 log values. `origination_probs`, when
/// present, are ordinary nonnegative weights over species; zero weights are
/// treated as impossible origination species during sampling.
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
    /// Precomputed clade-by-donor transfer aggregate matrix in base-2 log space.
    pub pibar: Matrix,
    /// Species extinction/survival fixed-point values in base-2 log space.
    pub e: Vec<f64>,
    /// Precomputed donor transfer-loss aggregate vector in base-2 log space.
    pub ebar: Vec<f64>,
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

#[derive(Clone, Copy, Debug)]
struct BacktrackInputView<'a> {
    species_newick: &'a str,
    species_names_postorder: &'a [String],
    root_clade: usize,
    leaf_species: &'a [Option<usize>],
    clade_leaf_labels: &'a [String],
    splits: &'a [SplitInput],
    pi: MatrixView<'a>,
    pibar: MatrixView<'a>,
    e: &'a [f64],
    ebar: &'a [f64],
    log_p_s: &'a [f64],
    log_p_d: &'a [f64],
    max_transfer: &'a [f64],
    origination_probs: Option<&'a [f64]>,
    seed: Option<u64>,
    max_events: Option<usize>,
}

impl<'a> From<&'a BacktrackInput> for BacktrackInputView<'a> {
    fn from(input: &'a BacktrackInput) -> Self {
        Self {
            species_newick: &input.species_newick,
            species_names_postorder: &input.species_names_postorder,
            root_clade: input.root_clade,
            leaf_species: &input.leaf_species,
            clade_leaf_labels: &input.clade_leaf_labels,
            splits: &input.splits,
            pi: MatrixView::from(&input.pi),
            pibar: MatrixView::from(&input.pibar),
            e: &input.e,
            ebar: &input.ebar,
            log_p_s: &input.log_p_s,
            log_p_d: &input.log_p_d,
            max_transfer: &input.max_transfer,
            origination_probs: input.origination_probs.as_deref(),
            seed: input.seed,
            max_events: input.max_events,
        }
    }
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
    clade: usize,
    species: usize,
}

#[derive(Clone, Debug)]
struct SampledTree {
    rec_tree: RecTree,
    summary: SampleSummary,
}

#[derive(Clone, Copy, Debug)]
struct SampledIndex<T> {
    item: T,
    log_probability: f64,
}

#[derive(Clone, Debug)]
struct PreparedBacktracker<'a> {
    input: BacktrackInputView<'a>,
    species: SpeciesTopology,
    splits_by_parent: Vec<Vec<usize>>,
    max_events: usize,
}

#[derive(Clone, Debug)]
struct Sampler<'a> {
    prepared: &'a PreparedBacktracker<'a>,
    rng: StdRng,
    nodes: Vec<FlatNode>,
    node_mapping: Vec<Option<usize>>,
    event_mapping: Vec<Event>,
    event_counts: EventCounts,
    log_probability: f64,
    sampled_terms: usize,
}

/// Sample one reconciliation as RecPhyloXML.
pub fn sample_recphyloxml(input: &BacktrackInput) -> Result<String, BacktrackError> {
    sample_recphyloxml_view(BacktrackInputView::from(input))
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
    sample_recphyloxmls_view(BacktrackInputView::from(input), num_samples, base_seed)
}

fn sample_recphyloxml_view(input: BacktrackInputView<'_>) -> Result<String, BacktrackError> {
    let prepared = PreparedBacktracker::new(input)?;
    prepared.sample_to_xml(input.seed.unwrap_or(0))
}

fn sample_recphyloxmls_view(
    input: BacktrackInputView<'_>,
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
        let seed = sample_seed(base_seed, sample_idx)?;
        out.push(prepared.sample_to_xml(seed)?);
    }
    Ok(out)
}

/// Sample scenario summaries with consecutive deterministic seeds.
///
/// When `parallel` is true, sampling is distributed with Rayon while preserving
/// output order and the `seed = base_seed + sample_idx` contract.
pub fn sample_summaries(
    input: &BacktrackInput,
    num_samples: usize,
    base_seed: u64,
    parallel: bool,
) -> Result<Vec<SampleSummary>, BacktrackError> {
    sample_summaries_view(
        BacktrackInputView::from(input),
        num_samples,
        base_seed,
        parallel,
    )
}

fn sample_summaries_view(
    input: BacktrackInputView<'_>,
    num_samples: usize,
    base_seed: u64,
    parallel: bool,
) -> Result<Vec<SampleSummary>, BacktrackError> {
    if num_samples == 0 {
        return Err(BacktrackError::InvalidInput(
            "num_samples must be positive".to_string(),
        ));
    }
    let prepared = PreparedBacktracker::new(input)?;
    if parallel {
        (0..num_samples)
            .into_par_iter()
            .map(|sample_idx| {
                let seed = sample_seed(base_seed, sample_idx)?;
                prepared.sample_summary(seed)
            })
            .collect()
    } else {
        let mut out = Vec::with_capacity(num_samples);
        for sample_idx in 0..num_samples {
            let seed = sample_seed(base_seed, sample_idx)?;
            out.push(prepared.sample_summary(seed)?);
        }
        Ok(out)
    }
}

/// Sample RecPhyloXML files directly to `output_dir`.
///
/// Files are named `sample_{i}.xml` or `sample_{i}.xml.gz` using zero-based
/// sample indices. This streams each sampled XML to disk instead of collecting
/// all XML strings in memory.
pub fn sample_recphyloxmls_to_dir(
    input: &BacktrackInput,
    num_samples: usize,
    base_seed: u64,
    output_dir: impl AsRef<Path>,
    parallel: bool,
    compression: OutputCompression,
) -> Result<Vec<SampleSummary>, BacktrackError> {
    sample_recphyloxmls_to_dir_view(
        BacktrackInputView::from(input),
        num_samples,
        base_seed,
        output_dir.as_ref(),
        parallel,
        compression,
    )
}

fn sample_recphyloxmls_to_dir_view(
    input: BacktrackInputView<'_>,
    num_samples: usize,
    base_seed: u64,
    output_dir: &Path,
    parallel: bool,
    compression: OutputCompression,
) -> Result<Vec<SampleSummary>, BacktrackError> {
    if num_samples == 0 {
        return Err(BacktrackError::InvalidInput(
            "num_samples must be positive".to_string(),
        ));
    }
    fs::create_dir_all(output_dir)?;
    let prepared = PreparedBacktracker::new(input)?;

    if parallel {
        (0..num_samples)
            .into_par_iter()
            .map(|sample_idx| {
                let seed = sample_seed(base_seed, sample_idx)?;
                let sampled = prepared.sample_with_summary(seed)?;
                let xml = sampled.rec_tree.to_xml();
                write_sample_xml(output_dir, sample_idx, &xml, compression)?;
                Ok(sampled.summary)
            })
            .collect()
    } else {
        let mut out = Vec::with_capacity(num_samples);
        for sample_idx in 0..num_samples {
            let seed = sample_seed(base_seed, sample_idx)?;
            let sampled = prepared.sample_with_summary(seed)?;
            let xml = sampled.rec_tree.to_xml();
            write_sample_xml(output_dir, sample_idx, &xml, compression)?;
            out.push(sampled.summary);
        }
        Ok(out)
    }
}

impl<'a> PreparedBacktracker<'a> {
    fn new(input: impl Into<BacktrackInputView<'a>>) -> Result<Self, BacktrackError> {
        let input = input.into();
        input.pi.validate("pi")?;
        input.pibar.validate("pibar")?;
        let c = input.pi.rows;
        let s = input.pi.cols;
        if input.pibar.rows != c || input.pibar.cols != s {
            return Err(BacktrackError::InvalidInput(format!(
                "pibar shape is {}x{}, expected {c}x{s}",
                input.pibar.rows, input.pibar.cols
            )));
        }
        validate_len("leaf_species", input.leaf_species.len(), c)?;
        validate_len("clade_leaf_labels", input.clade_leaf_labels.len(), c)?;
        validate_len("e", input.e.len(), s)?;
        validate_finite_values("e", input.e)?;
        validate_len("ebar", input.ebar.len(), s)?;
        validate_finite_values("ebar", input.ebar)?;
        validate_len("log_p_s", input.log_p_s.len(), s)?;
        validate_finite_values("log_p_s", input.log_p_s)?;
        validate_len("log_p_d", input.log_p_d.len(), s)?;
        validate_finite_values("log_p_d", input.log_p_d)?;
        validate_len("max_transfer", input.max_transfer.len(), s)?;
        validate_finite_values("max_transfer", input.max_transfer)?;
        if let Some(probs) = input.origination_probs {
            validate_len("origination_probs", probs.len(), s)?;
            validate_finite_values("origination_probs", probs)?;
            if let Some((idx, value)) = probs.iter().enumerate().find(|(_, value)| **value < 0.0) {
                return Err(BacktrackError::InvalidInput(format!(
                    "origination_probs contains negative value at index {idx}: {value}"
                )));
            }
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
        for (idx, leaf_species) in input.leaf_species.iter().enumerate() {
            if let Some(species) = leaf_species {
                if *species >= s {
                    return Err(BacktrackError::InvalidInput(format!(
                        "leaf_species[{idx}] is out of bounds for {s} species: {species}"
                    )));
                }
            }
        }

        let species = parse_species_topology(input.species_newick, input.species_names_postorder)?;
        validate_len("species_names_postorder", species.gp_to_rust.len(), s)?;

        let mut splits_by_parent = vec![Vec::new(); c];
        for (idx, split) in input.splits.iter().enumerate() {
            if !split.log_prob.is_finite() {
                return Err(BacktrackError::InvalidInput(format!(
                    "split {idx} log_prob is non-finite: {}",
                    split.log_prob
                )));
            }
            if split.parent >= c || split.left >= c || split.right >= c {
                return Err(BacktrackError::InvalidInput(format!(
                    "split {idx} has clade outside 0..{c}: parent={} left={} right={}",
                    split.parent, split.left, split.right
                )));
            }
            splits_by_parent[split.parent].push(idx);
        }

        Ok(Self {
            input,
            species,
            splits_by_parent,
            max_events: input.max_events.unwrap_or(100_000),
        })
    }

    fn sample_to_xml(&'a self, seed: u64) -> Result<String, BacktrackError> {
        Ok(self.sample_with_summary(seed)?.rec_tree.to_xml())
    }

    fn sample_summary(&'a self, seed: u64) -> Result<SampleSummary, BacktrackError> {
        Ok(self.sample_with_summary(seed)?.summary)
    }

    fn sample_with_summary(&'a self, seed: u64) -> Result<SampledTree, BacktrackError> {
        let mut sampler = Sampler::new(self, seed);
        sampler.sample(seed)
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
            event_counts: EventCounts::default(),
            log_probability: 0.0,
            sampled_terms: 0,
        }
    }

    fn sample(&mut self, seed: u64) -> Result<SampledTree, BacktrackError> {
        let root_species = self.sample_root_species()?;
        let root = self.add_node("", Event::Speciation, root_species, None);
        let mut stack = vec![WorkItem {
            node_idx: root,
            clade: self.prepared.input.root_clade,
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
            let children = self.expand_state(item.node_idx, item.clade, item.species)?;
            stack.extend(children.into_iter().rev());
        }

        Ok(SampledTree {
            rec_tree: RecTree::new_owned(
                self.prepared.species.rust_tree.clone(),
                FlatTree {
                    nodes: std::mem::take(&mut self.nodes),
                    root,
                },
                std::mem::take(&mut self.node_mapping),
                std::mem::take(&mut self.event_mapping),
            ),
            summary: SampleSummary {
                seed,
                event_counts: std::mem::take(&mut self.event_counts),
                log_probability: self.log_probability,
            },
        })
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
        let sampled = sample_index(&candidates, &mut self.rng)?;
        self.log_probability += sampled.log_probability;
        Ok(sampled.item)
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
            log_weight: pi_cs + input.ebar[species],
        });
        candidates.push(Candidate {
            term: Term::HiddenTransferLossDonor,
            log_weight: input.pibar.get(clade, species) + e_s,
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
                log_weight: base + input.pi.get(left, species) + input.pibar.get(right, species),
            });
            candidates.push(Candidate {
                term: Term::SplitTransferLeft(*split_idx),
                log_weight: base + input.pi.get(right, species) + input.pibar.get(left, species),
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
        let sampled = sample_index(&weighted, &mut self.rng)?;
        self.log_probability += sampled.log_probability;
        let term = candidates[sampled.item].term;
        self.count_term(term);
        Ok(term)
    }

    fn count_term(&mut self, term: Term) {
        match term {
            Term::HiddenDupLoss | Term::HiddenTransferLossRecipient => {}
            Term::HiddenTransferLossDonor => {
                self.event_counts.tl += 1;
            }
            Term::HiddenSpeciationLeft | Term::HiddenSpeciationRight => {
                self.event_counts.sl += 1;
            }
            Term::Leaf => {
                self.event_counts.leaf += 1;
            }
            Term::SplitDup(_) => {
                self.event_counts.d += 1;
            }
            Term::SplitTransferRight(_) | Term::SplitTransferLeft(_) => {
                self.event_counts.t += 1;
            }
            Term::SplitSpeciation(_, _) => {
                self.event_counts.s += 1;
            }
        }
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
                    clade,
                    species,
                }])
            }
            Term::HiddenTransferLossRecipient => {
                // A transfer to an immediately extinct recipient leaves the
                // retained lineage on the donor branch, so AleRax does not emit
                // a RecPhyloXML transfer-loss node for it.
                Ok(vec![WorkItem {
                    node_idx,
                    clade,
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
                    clade,
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
                    clade,
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
                        clade: split.left,
                        species,
                    },
                    WorkItem {
                        node_idx: right,
                        clade: split.right,
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
                        clade: split.left,
                        species,
                    },
                    WorkItem {
                        node_idx: recipient_child,
                        clade: split.right,
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
                        clade: split.left,
                        species: recipient,
                    },
                    WorkItem {
                        node_idx: donor_child,
                        clade: split.right,
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
                            clade: split.right,
                            species: c1,
                        },
                        WorkItem {
                            node_idx: right_node,
                            clade: split.left,
                            species: c2,
                        },
                    ])
                } else {
                    Ok(vec![
                        WorkItem {
                            node_idx: left_node,
                            clade: split.left,
                            species: c1,
                        },
                        WorkItem {
                            node_idx: right_node,
                            clade: split.right,
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
        let sampled = sample_index(&candidates, &mut self.rng)?;
        self.log_probability += sampled.log_probability;
        Ok(sampled.item)
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

fn sample_seed(base_seed: u64, sample_idx: usize) -> Result<u64, BacktrackError> {
    let seed_offset = u64::try_from(sample_idx).map_err(|_| {
        BacktrackError::InvalidInput(format!("sample index {sample_idx} does not fit in u64"))
    })?;
    base_seed.checked_add(seed_offset).ok_or_else(|| {
        BacktrackError::InvalidInput(format!(
            "seed range exceeds u64 maximum: base_seed={base_seed} sample_idx={sample_idx}"
        ))
    })
}

fn write_sample_xml(
    output_dir: &Path,
    sample_idx: usize,
    xml: &str,
    compression: OutputCompression,
) -> Result<(), BacktrackError> {
    let filename = match compression {
        OutputCompression::None => format!("sample_{sample_idx}.xml"),
        OutputCompression::Gzip => format!("sample_{sample_idx}.xml.gz"),
    };
    let path = output_dir.join(filename);
    match compression {
        OutputCompression::None => {
            fs::write(path, xml)?;
        }
        OutputCompression::Gzip => {
            let file = File::create(path)?;
            let mut encoder = GzEncoder::new(file, GzipCompression::default());
            encoder.write_all(xml.as_bytes())?;
            encoder.finish()?;
        }
    }
    Ok(())
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

fn validate_finite_values(name: &str, values: &[f64]) -> Result<(), BacktrackError> {
    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(BacktrackError::InvalidInput(format!(
            "{name} contains non-finite value at index {idx}: {value}"
        )));
    }
    Ok(())
}

fn leaf_name(input: BacktrackInputView<'_>, clade: usize) -> String {
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

fn sample_index<T: Copy>(
    weighted: &[(T, f64)],
    rng: &mut StdRng,
) -> Result<SampledIndex<T>, BacktrackError> {
    let logs = weighted.iter().map(|(_, w)| *w).collect::<Vec<_>>();
    let norm = logsumexp2(&logs);
    if norm <= NEG_INF / 2.0 || !norm.is_finite() {
        return Err(BacktrackError::Sampling(
            "all candidate backtracking weights are zero".to_string(),
        ));
    }
    let dist = Uniform::new(0.0, 1.0);
    let mut draw = dist.sample(rng);
    let mut last_selectable = None;
    for (item, log_w) in weighted {
        if *log_w <= NEG_INF / 2.0 {
            continue;
        }
        last_selectable = Some((*item, *log_w));
        let p = 2.0_f64.powf(*log_w - norm);
        if draw <= p {
            return Ok(SampledIndex {
                item: *item,
                log_probability: *log_w - norm,
            });
        }
        draw -= p;
    }
    let (last, last_log_w) = last_selectable.ok_or_else(|| {
        BacktrackError::Sampling("all candidate backtracking weights are zero".to_string())
    })?;
    Ok(SampledIndex {
        item: last,
        log_probability: last_log_w - norm,
    })
}

#[cfg(feature = "python-extension")]
fn py_backtrack_error(error: BacktrackError) -> PyErr {
    match error {
        BacktrackError::InvalidInput(message) => PyValueError::new_err(message),
        BacktrackError::Sampling(message) => PyRuntimeError::new_err(message),
        BacktrackError::Io(source) => PyRuntimeError::new_err(source.to_string()),
        BacktrackError::Rustree(source) => PyValueError::new_err(source.to_string()),
    }
}

#[cfg(feature = "python-extension")]
fn nonnegative_i64_to_usize(name: &str, idx: usize, value: i64) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("{name}[{idx}] must be non-negative, got {value}"))
    })
}

#[cfg(feature = "python-extension")]
fn i64_slice_from_numpy<'a>(
    name: &str,
    values: &'a PyReadonlyArray1<'_, i64>,
) -> PyResult<&'a [i64]> {
    if !values.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

#[cfg(feature = "python-extension")]
fn f64_slice_from_numpy<'a>(
    name: &str,
    values: &'a PyReadonlyArray1<'_, f64>,
) -> PyResult<&'a [f64]> {
    if !values.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

#[cfg(feature = "python-extension")]
fn optional_species_from_i64(values: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<Option<usize>>> {
    i64_slice_from_numpy("leaf_species", values)?
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if *value < 0 {
                Ok(None)
            } else {
                Ok(Some(nonnegative_i64_to_usize("leaf_species", idx, *value)?))
            }
        })
        .collect()
}

#[cfg(feature = "python-extension")]
fn matrix_view_from_numpy<'a>(
    name: &str,
    values: &'a PyReadonlyArray2<'_, f64>,
) -> PyResult<MatrixView<'a>> {
    let shape = values.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must be two-dimensional"
        )));
    }
    if !values.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    let data = values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))?;
    Ok(MatrixView {
        rows: shape[0],
        cols: shape[1],
        data,
    })
}

#[cfg(feature = "python-extension")]
fn parse_output_compression(value: &str) -> PyResult<OutputCompression> {
    match value {
        "none" | "" => Ok(OutputCompression::None),
        "gzip" | "gz" => Ok(OutputCompression::Gzip),
        other => Err(PyValueError::new_err(format!(
            "compression must be 'none' or 'gzip', got {other:?}"
        ))),
    }
}

#[cfg(feature = "python-extension")]
fn event_counts_to_dict<'py>(
    py: Python<'py>,
    event_counts: &EventCounts,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("S", event_counts.s)?;
    dict.set_item("SL", event_counts.sl)?;
    dict.set_item("D", event_counts.d)?;
    dict.set_item("DL", event_counts.dl)?;
    dict.set_item("T", event_counts.t)?;
    dict.set_item("TL", event_counts.tl)?;
    dict.set_item("L", event_counts.l)?;
    dict.set_item("Leaf", event_counts.leaf)?;
    Ok(dict)
}

#[cfg(feature = "python-extension")]
fn summaries_to_py(py: Python<'_>, summaries: Vec<SampleSummary>) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for summary in summaries {
        let dict = PyDict::new_bound(py);
        dict.set_item("seed", summary.seed)?;
        dict.set_item(
            "event_counts",
            event_counts_to_dict(py, &summary.event_counts)?,
        )?;
        dict.set_item("log_probability", summary.log_probability)?;
        list.append(dict)?;
    }
    Ok(list.into_py(py))
}

#[cfg(feature = "python-extension")]
#[allow(clippy::too_many_arguments)]
struct NumpyBacktrackInput<'a> {
    species_newick: String,
    species_names_postorder: Vec<String>,
    root_clade: usize,
    leaf_species: Vec<Option<usize>>,
    clade_leaf_labels: Vec<String>,
    splits: Vec<SplitInput>,
    pi: MatrixView<'a>,
    pibar: MatrixView<'a>,
    e: &'a [f64],
    ebar: &'a [f64],
    log_p_s: &'a [f64],
    log_p_d: &'a [f64],
    max_transfer: &'a [f64],
    origination_probs: Option<&'a [f64]>,
    seed: Option<u64>,
    max_events: Option<usize>,
}

#[cfg(feature = "python-extension")]
impl NumpyBacktrackInput<'_> {
    fn as_view(&self) -> BacktrackInputView<'_> {
        BacktrackInputView {
            species_newick: &self.species_newick,
            species_names_postorder: &self.species_names_postorder,
            root_clade: self.root_clade,
            leaf_species: &self.leaf_species,
            clade_leaf_labels: &self.clade_leaf_labels,
            splits: &self.splits,
            pi: self.pi,
            pibar: self.pibar,
            e: self.e,
            ebar: self.ebar,
            log_p_s: self.log_p_s,
            log_p_d: self.log_p_d,
            max_transfer: self.max_transfer,
            origination_probs: self.origination_probs,
            seed: self.seed,
            max_events: self.max_events,
        }
    }
}

#[cfg(feature = "python-extension")]
#[allow(clippy::too_many_arguments)]
fn build_borrowed_input_from_numpy<'a>(
    species_newick: String,
    species_names_postorder: Vec<String>,
    root_clade: usize,
    leaf_species: &'a PyReadonlyArray1<'_, i64>,
    clade_leaf_labels: Vec<String>,
    split_parents: &'a PyReadonlyArray1<'_, i64>,
    split_lefts: &'a PyReadonlyArray1<'_, i64>,
    split_rights: &'a PyReadonlyArray1<'_, i64>,
    split_log_probs: &'a PyReadonlyArray1<'_, f64>,
    pi: &'a PyReadonlyArray2<'_, f64>,
    pibar: &'a PyReadonlyArray2<'_, f64>,
    e: &'a PyReadonlyArray1<'_, f64>,
    ebar: &'a PyReadonlyArray1<'_, f64>,
    log_p_s: &'a PyReadonlyArray1<'_, f64>,
    log_p_d: &'a PyReadonlyArray1<'_, f64>,
    max_transfer: &'a PyReadonlyArray1<'_, f64>,
    origination_probs: Option<&'a PyReadonlyArray1<'_, f64>>,
    seed: Option<u64>,
    max_events: Option<usize>,
) -> PyResult<NumpyBacktrackInput<'a>> {
    let split_len = split_parents.len();
    if split_lefts.len() != split_len
        || split_rights.len() != split_len
        || split_log_probs.len() != split_len
    {
        return Err(PyValueError::new_err(
            "split arrays must have the same length",
        ));
    }

    let split_parents = i64_slice_from_numpy("split_parents", split_parents)?;
    let split_lefts = i64_slice_from_numpy("split_lefts", split_lefts)?;
    let split_rights = i64_slice_from_numpy("split_rights", split_rights)?;
    let split_log_probs = f64_slice_from_numpy("split_log_probs", split_log_probs)?;
    let mut splits = Vec::with_capacity(split_len);
    for idx in 0..split_len {
        splits.push(SplitInput {
            parent: nonnegative_i64_to_usize("split_parents", idx, split_parents[idx])?,
            left: nonnegative_i64_to_usize("split_lefts", idx, split_lefts[idx])?,
            right: nonnegative_i64_to_usize("split_rights", idx, split_rights[idx])?,
            log_prob: split_log_probs[idx],
        });
    }

    Ok(NumpyBacktrackInput {
        species_newick,
        species_names_postorder,
        root_clade,
        leaf_species: optional_species_from_i64(leaf_species)?,
        clade_leaf_labels,
        splits,
        pi: matrix_view_from_numpy("pi", pi)?,
        pibar: matrix_view_from_numpy("pibar", pibar)?,
        e: f64_slice_from_numpy("e", e)?,
        ebar: f64_slice_from_numpy("ebar", ebar)?,
        log_p_s: f64_slice_from_numpy("log_p_s", log_p_s)?,
        log_p_d: f64_slice_from_numpy("log_p_d", log_p_d)?,
        max_transfer: f64_slice_from_numpy("max_transfer", max_transfer)?,
        origination_probs: origination_probs
            .map(|values| f64_slice_from_numpy("origination_probs", values))
            .transpose()?,
        seed,
        max_events,
    })
}

#[cfg(feature = "python-extension")]
#[pyfunction]
#[pyo3(signature = (
    species_newick,
    species_names_postorder,
    root_clade,
    leaf_species,
    clade_leaf_labels,
    split_parents,
    split_lefts,
    split_rights,
    split_log_probs,
    pi,
    pibar,
    e,
    ebar,
    log_p_s,
    log_p_d,
    max_transfer,
    origination_probs=None,
    seed=None,
    max_events=None
))]
#[allow(clippy::too_many_arguments)]
fn sample_recphyloxml_torch(
    py: Python<'_>,
    species_newick: String,
    species_names_postorder: Vec<String>,
    root_clade: usize,
    leaf_species: PyReadonlyArray1<'_, i64>,
    clade_leaf_labels: Vec<String>,
    split_parents: PyReadonlyArray1<'_, i64>,
    split_lefts: PyReadonlyArray1<'_, i64>,
    split_rights: PyReadonlyArray1<'_, i64>,
    split_log_probs: PyReadonlyArray1<'_, f64>,
    pi: PyReadonlyArray2<'_, f64>,
    pibar: PyReadonlyArray2<'_, f64>,
    e: PyReadonlyArray1<'_, f64>,
    ebar: PyReadonlyArray1<'_, f64>,
    log_p_s: PyReadonlyArray1<'_, f64>,
    log_p_d: PyReadonlyArray1<'_, f64>,
    max_transfer: PyReadonlyArray1<'_, f64>,
    origination_probs: Option<PyReadonlyArray1<'_, f64>>,
    seed: Option<u64>,
    max_events: Option<usize>,
) -> PyResult<String> {
    let input = build_borrowed_input_from_numpy(
        species_newick,
        species_names_postorder,
        root_clade,
        &leaf_species,
        clade_leaf_labels,
        &split_parents,
        &split_lefts,
        &split_rights,
        &split_log_probs,
        &pi,
        &pibar,
        &e,
        &ebar,
        &log_p_s,
        &log_p_d,
        &max_transfer,
        origination_probs.as_ref(),
        seed,
        max_events,
    )?;
    let input_view = input.as_view();
    py.allow_threads(move || sample_recphyloxml_view(input_view))
        .map_err(py_backtrack_error)
}

#[cfg(feature = "python-extension")]
#[pyfunction]
#[pyo3(signature = (
    species_newick,
    species_names_postorder,
    root_clade,
    leaf_species,
    clade_leaf_labels,
    split_parents,
    split_lefts,
    split_rights,
    split_log_probs,
    pi,
    pibar,
    e,
    ebar,
    log_p_s,
    log_p_d,
    max_transfer,
    seed,
    num_samples,
    max_events=None,
    origination_probs=None
))]
#[allow(clippy::too_many_arguments)]
fn sample_recphyloxmls_torch(
    py: Python<'_>,
    species_newick: String,
    species_names_postorder: Vec<String>,
    root_clade: usize,
    leaf_species: PyReadonlyArray1<'_, i64>,
    clade_leaf_labels: Vec<String>,
    split_parents: PyReadonlyArray1<'_, i64>,
    split_lefts: PyReadonlyArray1<'_, i64>,
    split_rights: PyReadonlyArray1<'_, i64>,
    split_log_probs: PyReadonlyArray1<'_, f64>,
    pi: PyReadonlyArray2<'_, f64>,
    pibar: PyReadonlyArray2<'_, f64>,
    e: PyReadonlyArray1<'_, f64>,
    ebar: PyReadonlyArray1<'_, f64>,
    log_p_s: PyReadonlyArray1<'_, f64>,
    log_p_d: PyReadonlyArray1<'_, f64>,
    max_transfer: PyReadonlyArray1<'_, f64>,
    seed: u64,
    num_samples: usize,
    max_events: Option<usize>,
    origination_probs: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Vec<String>> {
    let input = build_borrowed_input_from_numpy(
        species_newick,
        species_names_postorder,
        root_clade,
        &leaf_species,
        clade_leaf_labels,
        &split_parents,
        &split_lefts,
        &split_rights,
        &split_log_probs,
        &pi,
        &pibar,
        &e,
        &ebar,
        &log_p_s,
        &log_p_d,
        &max_transfer,
        origination_probs.as_ref(),
        Some(seed),
        max_events,
    )?;
    let input_view = input.as_view();
    py.allow_threads(move || sample_recphyloxmls_view(input_view, num_samples, seed))
        .map_err(py_backtrack_error)
}

#[cfg(feature = "python-extension")]
#[pyfunction]
#[pyo3(signature = (
    species_newick,
    species_names_postorder,
    root_clade,
    leaf_species,
    clade_leaf_labels,
    split_parents,
    split_lefts,
    split_rights,
    split_log_probs,
    pi,
    pibar,
    e,
    ebar,
    log_p_s,
    log_p_d,
    max_transfer,
    seed,
    num_samples,
    max_events=None,
    origination_probs=None,
    parallel=true
))]
#[allow(clippy::too_many_arguments)]
fn sample_summaries_torch(
    py: Python<'_>,
    species_newick: String,
    species_names_postorder: Vec<String>,
    root_clade: usize,
    leaf_species: PyReadonlyArray1<'_, i64>,
    clade_leaf_labels: Vec<String>,
    split_parents: PyReadonlyArray1<'_, i64>,
    split_lefts: PyReadonlyArray1<'_, i64>,
    split_rights: PyReadonlyArray1<'_, i64>,
    split_log_probs: PyReadonlyArray1<'_, f64>,
    pi: PyReadonlyArray2<'_, f64>,
    pibar: PyReadonlyArray2<'_, f64>,
    e: PyReadonlyArray1<'_, f64>,
    ebar: PyReadonlyArray1<'_, f64>,
    log_p_s: PyReadonlyArray1<'_, f64>,
    log_p_d: PyReadonlyArray1<'_, f64>,
    max_transfer: PyReadonlyArray1<'_, f64>,
    seed: u64,
    num_samples: usize,
    max_events: Option<usize>,
    origination_probs: Option<PyReadonlyArray1<'_, f64>>,
    parallel: bool,
) -> PyResult<PyObject> {
    let input = build_borrowed_input_from_numpy(
        species_newick,
        species_names_postorder,
        root_clade,
        &leaf_species,
        clade_leaf_labels,
        &split_parents,
        &split_lefts,
        &split_rights,
        &split_log_probs,
        &pi,
        &pibar,
        &e,
        &ebar,
        &log_p_s,
        &log_p_d,
        &max_transfer,
        origination_probs.as_ref(),
        Some(seed),
        max_events,
    )?;
    let input_view = input.as_view();
    let summaries = py
        .allow_threads(move || sample_summaries_view(input_view, num_samples, seed, parallel))
        .map_err(py_backtrack_error)?;
    summaries_to_py(py, summaries)
}

#[cfg(feature = "python-extension")]
#[pyfunction]
#[pyo3(signature = (
    species_newick,
    species_names_postorder,
    root_clade,
    leaf_species,
    clade_leaf_labels,
    split_parents,
    split_lefts,
    split_rights,
    split_log_probs,
    pi,
    pibar,
    e,
    ebar,
    log_p_s,
    log_p_d,
    max_transfer,
    output_dir,
    seed,
    num_samples,
    max_events=None,
    origination_probs=None,
    parallel=true,
    compression="none"
))]
#[allow(clippy::too_many_arguments)]
fn sample_recphyloxmls_to_dir_torch(
    py: Python<'_>,
    species_newick: String,
    species_names_postorder: Vec<String>,
    root_clade: usize,
    leaf_species: PyReadonlyArray1<'_, i64>,
    clade_leaf_labels: Vec<String>,
    split_parents: PyReadonlyArray1<'_, i64>,
    split_lefts: PyReadonlyArray1<'_, i64>,
    split_rights: PyReadonlyArray1<'_, i64>,
    split_log_probs: PyReadonlyArray1<'_, f64>,
    pi: PyReadonlyArray2<'_, f64>,
    pibar: PyReadonlyArray2<'_, f64>,
    e: PyReadonlyArray1<'_, f64>,
    ebar: PyReadonlyArray1<'_, f64>,
    log_p_s: PyReadonlyArray1<'_, f64>,
    log_p_d: PyReadonlyArray1<'_, f64>,
    max_transfer: PyReadonlyArray1<'_, f64>,
    output_dir: String,
    seed: u64,
    num_samples: usize,
    max_events: Option<usize>,
    origination_probs: Option<PyReadonlyArray1<'_, f64>>,
    parallel: bool,
    compression: &str,
) -> PyResult<PyObject> {
    let compression = parse_output_compression(compression)?;
    let input = build_borrowed_input_from_numpy(
        species_newick,
        species_names_postorder,
        root_clade,
        &leaf_species,
        clade_leaf_labels,
        &split_parents,
        &split_lefts,
        &split_rights,
        &split_log_probs,
        &pi,
        &pibar,
        &e,
        &ebar,
        &log_p_s,
        &log_p_d,
        &max_transfer,
        origination_probs.as_ref(),
        Some(seed),
        max_events,
    )?;
    let input_view = input.as_view();
    let summaries = py
        .allow_threads(move || {
            sample_recphyloxmls_to_dir_view(
                input_view,
                num_samples,
                seed,
                Path::new(&output_dir),
                parallel,
                compression,
            )
        })
        .map_err(py_backtrack_error)?;
    summaries_to_py(py, summaries)
}

#[cfg(feature = "python-extension")]
#[pymodule]
fn gpurec_backtrack(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sample_recphyloxml_torch, module)?)?;
    module.add_function(wrap_pyfunction!(sample_recphyloxmls_torch, module)?)?;
    module.add_function(wrap_pyfunction!(sample_summaries_torch, module)?)?;
    module.add_function(wrap_pyfunction!(sample_recphyloxmls_to_dir_torch, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::fs;
    use std::io::Read;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

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
            pibar: Matrix {
                rows: 3,
                cols: 3,
                data: vec![neg; 9],
            },
            e: vec![neg, neg, neg],
            ebar: vec![neg, neg, neg],
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
            pibar: Matrix {
                rows: 1,
                cols: 3,
                data: vec![0.0, neg, neg],
            },
            e: vec![0.0, neg, neg],
            ebar: vec![neg, neg, neg],
            log_p_s: vec![0.0, 0.0, neg],
            log_p_d: vec![neg, neg, neg],
            max_transfer: vec![0.0, neg, neg],
            origination_probs: Some(vec![1.0, 0.0, 0.0]),
            seed: Some(seed),
            max_events: Some(32),
        }
    }

    fn assert_work_item(item: &WorkItem, node_idx: usize, clade: usize, species: usize) {
        assert_eq!(item.node_idx, node_idx);
        assert_eq!(item.clade, clade);
        assert_eq!(item.species, species);
    }

    fn temp_output_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("gpurec-backtrack-{name}-{nanos}"))
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
    fn sample_summaries_match_forced_speciation_counts() {
        let input = speciation_input();

        let summaries = sample_summaries(&input, 1, 7, false).unwrap();

        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].seed, 7);
        assert_eq!(
            summaries[0].event_counts,
            EventCounts {
                s: 1,
                leaf: 2,
                ..EventCounts::default()
            }
        );
        assert!(summaries[0].log_probability.is_finite());
    }

    #[test]
    fn sample_summaries_match_transfer_loss_counts() {
        let input = transfer_input(19);

        let summaries = sample_summaries(&input, 1, 19, false).unwrap();

        assert_eq!(
            summaries[0].event_counts,
            EventCounts {
                tl: 1,
                leaf: 1,
                ..EventCounts::default()
            }
        );
    }

    #[test]
    fn borrowed_input_view_samples_from_local_slices() {
        let neg = NEG_INF;
        let species_newick = "(A:1,B:1)Root:0;".to_string();
        let species_names_postorder = vec!["A".into(), "B".into(), "Root".into()];
        let leaf_species = vec![Some(0), Some(1), None];
        let clade_leaf_labels = vec!["a".into(), "b".into(), String::new()];
        let splits = vec![SplitInput {
            parent: 2,
            left: 0,
            right: 1,
            log_prob: 0.0,
        }];
        let pi = vec![0.0, neg, neg, neg, 0.0, neg, neg, neg, 0.0];
        let pibar = vec![neg; 9];
        let e = vec![neg, neg, neg];
        let ebar = vec![neg, neg, neg];
        let log_p_s = vec![0.0, 0.0, 0.0];
        let log_p_d = vec![neg, neg, neg];
        let max_transfer = vec![neg, neg, neg];
        let origination_probs = vec![0.0, 0.0, 1.0];
        let input = BacktrackInputView {
            species_newick: &species_newick,
            species_names_postorder: &species_names_postorder,
            root_clade: 2,
            leaf_species: &leaf_species,
            clade_leaf_labels: &clade_leaf_labels,
            splits: &splits,
            pi: MatrixView {
                rows: 3,
                cols: 3,
                data: &pi,
            },
            pibar: MatrixView {
                rows: 3,
                cols: 3,
                data: &pibar,
            },
            e: &e,
            ebar: &ebar,
            log_p_s: &log_p_s,
            log_p_d: &log_p_d,
            max_transfer: &max_transfer,
            origination_probs: Some(&origination_probs),
            seed: Some(7),
            max_events: Some(32),
        };

        let xml = sample_recphyloxml_view(input).unwrap();
        let summaries = sample_summaries_view(input, 2, 7, true).unwrap();

        assert!(xml.contains("<speciation speciesLocation=\"Root\"/>"));
        assert_eq!(summaries.len(), 2);
        assert_eq!(
            summaries[0].event_counts,
            EventCounts {
                s: 1,
                leaf: 2,
                ..EventCounts::default()
            }
        );
    }

    #[test]
    fn writes_samples_to_dir_without_collecting_returned_xml() {
        let input = speciation_input();
        let dir = temp_output_dir("plain");

        let summaries =
            sample_recphyloxmls_to_dir(&input, 2, 7, &dir, false, OutputCompression::None).unwrap();

        assert_eq!(summaries.len(), 2);
        assert!(dir.join("sample_0.xml").is_file());
        assert!(dir.join("sample_1.xml").is_file());
        let xml = fs::read_to_string(dir.join("sample_0.xml")).unwrap();
        assert!(xml.contains("<speciation speciesLocation=\"Root\"/>"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn parallel_dir_sampling_matches_serial_output() {
        let input = transfer_input(19);
        let serial_dir = temp_output_dir("serial");
        let parallel_dir = temp_output_dir("parallel");

        let serial =
            sample_recphyloxmls_to_dir(&input, 4, 19, &serial_dir, false, OutputCompression::None)
                .unwrap();
        let parallel =
            sample_recphyloxmls_to_dir(&input, 4, 19, &parallel_dir, true, OutputCompression::None)
                .unwrap();

        assert_eq!(parallel, serial);
        for idx in 0..4 {
            let serial_xml =
                fs::read_to_string(serial_dir.join(format!("sample_{idx}.xml"))).unwrap();
            let parallel_xml =
                fs::read_to_string(parallel_dir.join(format!("sample_{idx}.xml"))).unwrap();
            assert_eq!(parallel_xml, serial_xml);
        }
        fs::remove_dir_all(serial_dir).unwrap();
        fs::remove_dir_all(parallel_dir).unwrap();
    }

    #[test]
    fn writes_gzip_samples_to_dir() {
        let input = speciation_input();
        let dir = temp_output_dir("gzip");

        sample_recphyloxmls_to_dir(&input, 1, 7, &dir, false, OutputCompression::Gzip).unwrap();

        let path = dir.join("sample_0.xml.gz");
        assert!(path.is_file());
        let mut decoder = GzDecoder::new(fs::File::open(path).unwrap());
        let mut xml = String::new();
        decoder.read_to_string(&mut xml).unwrap();
        assert!(xml.contains("recPhylo"));
        assert!(xml.contains("<leaf speciesLocation=\"A\"/>"));
        fs::remove_dir_all(dir).unwrap();
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
        assert_eq!(children[0].clade, 2);
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
        assert_eq!(children[0].clade, 2);
        assert_eq!(children[0].species, 2);
    }

    #[test]
    fn hidden_transfer_donor_emits_transfer_loss_and_requeues_recipient() {
        let input = transfer_input(3);
        let prepared = PreparedBacktracker::new(&input).unwrap();
        let mut sampler = Sampler::new(&prepared, 3);
        let root = sampler.add_node("", Event::Speciation, 0, None);

        let children = sampler
            .apply_term(root, 0, 0, Term::HiddenTransferLossDonor)
            .unwrap();

        assert_eq!(sampler.nodes.len(), 3);
        assert_eq!(sampler.event_mapping[root], Event::Transfer);
        let left = sampler.nodes[root].left_child.unwrap();
        let right = sampler.nodes[root].right_child.unwrap();
        assert_ne!(left, right);
        let loss = (1..sampler.nodes.len())
            .find(|idx| sampler.event_mapping[*idx] == Event::Loss)
            .unwrap();
        let cont = (1..sampler.nodes.len())
            .find(|idx| sampler.event_mapping[*idx] == Event::Leaf)
            .unwrap();
        assert!(left == loss || right == loss);
        assert!(left == cont || right == cont);
        assert_eq!(sampler.nodes[loss].name, "loss");
        assert_eq!(
            sampler.node_mapping[loss],
            Some(prepared.species.gp_to_rust[0])
        );
        assert_eq!(
            sampler.node_mapping[cont],
            Some(prepared.species.gp_to_rust[1])
        );
        assert_eq!(children.len(), 1);
        assert_work_item(&children[0], cont, 0, 1);
    }

    #[test]
    fn hidden_speciation_left_emits_loss_on_right_species() {
        let input = speciation_input();
        let prepared = PreparedBacktracker::new(&input).unwrap();
        let mut sampler = Sampler::new(&prepared, 3);
        let root = sampler.add_node("", Event::Leaf, 2, None);

        let children = sampler
            .apply_term(root, 2, 2, Term::HiddenSpeciationLeft)
            .unwrap();

        assert_eq!(sampler.nodes.len(), 3);
        assert_eq!(sampler.event_mapping[root], Event::Speciation);
        let cont = sampler.nodes[root].left_child.unwrap();
        let loss = sampler.nodes[root].right_child.unwrap();
        assert_eq!(sampler.event_mapping[cont], Event::Leaf);
        assert_eq!(sampler.event_mapping[loss], Event::Loss);
        assert_eq!(
            sampler.node_mapping[cont],
            Some(prepared.species.gp_to_rust[0])
        );
        assert_eq!(
            sampler.node_mapping[loss],
            Some(prepared.species.gp_to_rust[1])
        );
        assert_eq!(children.len(), 1);
        assert_work_item(&children[0], cont, 2, 0);
    }

    #[test]
    fn hidden_speciation_right_emits_loss_on_left_species() {
        let input = speciation_input();
        let prepared = PreparedBacktracker::new(&input).unwrap();
        let mut sampler = Sampler::new(&prepared, 3);
        let root = sampler.add_node("", Event::Leaf, 2, None);

        let children = sampler
            .apply_term(root, 2, 2, Term::HiddenSpeciationRight)
            .unwrap();

        assert_eq!(sampler.nodes.len(), 3);
        assert_eq!(sampler.event_mapping[root], Event::Speciation);
        let cont = sampler.nodes[root].left_child.unwrap();
        let loss = sampler.nodes[root].right_child.unwrap();
        assert_eq!(sampler.event_mapping[cont], Event::Leaf);
        assert_eq!(sampler.event_mapping[loss], Event::Loss);
        assert_eq!(
            sampler.node_mapping[cont],
            Some(prepared.species.gp_to_rust[1])
        );
        assert_eq!(
            sampler.node_mapping[loss],
            Some(prepared.species.gp_to_rust[0])
        );
        assert_eq!(children.len(), 1);
        assert_work_item(&children[0], cont, 2, 1);
    }

    #[test]
    fn split_transfer_right_keeps_left_child_on_donor_branch() {
        let mut input = speciation_input();
        input.max_transfer[0] = 0.0;
        let prepared = PreparedBacktracker::new(&input).unwrap();
        let mut sampler = Sampler::new(&prepared, 3);
        let root = sampler.add_node("", Event::Leaf, 0, None);

        let children = sampler
            .apply_term(root, 2, 0, Term::SplitTransferRight(0))
            .unwrap();

        assert_eq!(sampler.nodes.len(), 3);
        assert_eq!(sampler.event_mapping[root], Event::Transfer);
        let donor_child = sampler.nodes[root].left_child.unwrap();
        let recipient_child = sampler.nodes[root].right_child.unwrap();
        assert_eq!(
            sampler.node_mapping[donor_child],
            Some(prepared.species.gp_to_rust[0])
        );
        assert_eq!(
            sampler.node_mapping[recipient_child],
            Some(prepared.species.gp_to_rust[1])
        );
        assert_eq!(children.len(), 2);
        assert_work_item(&children[0], donor_child, 0, 0);
        assert_work_item(&children[1], recipient_child, 1, 1);
    }

    #[test]
    fn split_transfer_left_keeps_right_child_on_donor_branch() {
        let mut input = speciation_input();
        input.pi.data[1] = 0.0;
        input.max_transfer[0] = 0.0;
        let prepared = PreparedBacktracker::new(&input).unwrap();
        let mut sampler = Sampler::new(&prepared, 3);
        let root = sampler.add_node("", Event::Leaf, 0, None);

        let children = sampler
            .apply_term(root, 2, 0, Term::SplitTransferLeft(0))
            .unwrap();

        assert_eq!(sampler.nodes.len(), 3);
        assert_eq!(sampler.event_mapping[root], Event::Transfer);
        let recipient_child = sampler.nodes[root].left_child.unwrap();
        let donor_child = sampler.nodes[root].right_child.unwrap();
        assert_eq!(
            sampler.node_mapping[recipient_child],
            Some(prepared.species.gp_to_rust[1])
        );
        assert_eq!(
            sampler.node_mapping[donor_child],
            Some(prepared.species.gp_to_rust[0])
        );
        assert_eq!(children.len(), 2);
        assert_work_item(&children[0], recipient_child, 0, 1);
        assert_work_item(&children[1], donor_child, 1, 0);
    }

    #[test]
    fn split_speciation_assigns_left_and_right_clades_to_species_children() {
        let input = speciation_input();
        let prepared = PreparedBacktracker::new(&input).unwrap();
        let mut sampler = Sampler::new(&prepared, 3);
        let root = sampler.add_node("", Event::Leaf, 2, None);

        let children = sampler
            .apply_term(root, 2, 2, Term::SplitSpeciation(0, false))
            .unwrap();

        assert_eq!(sampler.nodes.len(), 3);
        assert_eq!(sampler.event_mapping[root], Event::Speciation);
        let left = sampler.nodes[root].left_child.unwrap();
        let right = sampler.nodes[root].right_child.unwrap();
        assert_eq!(
            sampler.node_mapping[left],
            Some(prepared.species.gp_to_rust[0])
        );
        assert_eq!(
            sampler.node_mapping[right],
            Some(prepared.species.gp_to_rust[1])
        );
        assert_eq!(children.len(), 2);
        assert_work_item(&children[0], left, 0, 0);
        assert_work_item(&children[1], right, 1, 1);
    }

    #[test]
    fn swapped_split_speciation_swaps_clades_not_species_children() {
        let input = speciation_input();
        let prepared = PreparedBacktracker::new(&input).unwrap();
        let mut sampler = Sampler::new(&prepared, 3);
        let root = sampler.add_node("", Event::Leaf, 2, None);

        let children = sampler
            .apply_term(root, 2, 2, Term::SplitSpeciation(0, true))
            .unwrap();

        assert_eq!(sampler.nodes.len(), 3);
        assert_eq!(sampler.event_mapping[root], Event::Speciation);
        let left = sampler.nodes[root].left_child.unwrap();
        let right = sampler.nodes[root].right_child.unwrap();
        assert_eq!(
            sampler.node_mapping[left],
            Some(prepared.species.gp_to_rust[0])
        );
        assert_eq!(
            sampler.node_mapping[right],
            Some(prepared.species.gp_to_rust[1])
        );
        assert_eq!(children.len(), 2);
        assert_work_item(&children[0], left, 1, 0);
        assert_work_item(&children[1], right, 0, 1);
    }

    #[test]
    fn rejects_bad_matrix_shape() {
        let mut input = speciation_input();
        input.pi.data.pop();
        let err = sample_recphyloxml(&input).unwrap_err().to_string();
        assert!(err.contains("pi shape"));
    }

    #[test]
    fn rejects_nonfinite_log_payload_values() {
        let mut input = speciation_input();
        input.pi.data[0] = f64::NAN;

        let err = sample_recphyloxml(&input).unwrap_err().to_string();

        assert!(err.contains("pi contains non-finite value at index 0"));
    }

    #[test]
    fn rejects_leaf_species_outside_species_range() {
        let mut input = speciation_input();
        input.leaf_species[0] = Some(3);

        let err = sample_recphyloxml(&input).unwrap_err().to_string();

        assert!(err.contains("leaf_species[0] is out of bounds for 3 species"));
    }

    #[test]
    fn rejects_matrix_shape_overflow_without_panicking() {
        let matrix = Matrix {
            rows: usize::MAX,
            cols: 2,
            data: Vec::new(),
        };

        let err = MatrixView::from(&matrix)
            .validate("pi")
            .unwrap_err()
            .to_string();

        assert!(err.contains("pi shape"));
        assert!(err.contains("overflows usize"));
    }
}

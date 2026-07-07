use rand::distributions::Uniform;
use rand::prelude::*;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt;

const NEG_INF: f64 = f64::NEG_INFINITY;

#[derive(Debug)]
enum BacktrackError {
    InvalidInput(String),
    Sampling(String),
}

impl fmt::Display for BacktrackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BacktrackError::InvalidInput(message) => {
                write!(f, "invalid backtracking input: {message}")
            }
            BacktrackError::Sampling(message) => {
                write!(f, "backtracking sampling failed: {message}")
            }
        }
    }
}

impl std::error::Error for BacktrackError {}

impl From<BacktrackError> for PyErr {
    fn from(err: BacktrackError) -> Self {
        match err {
            BacktrackError::InvalidInput(message) => PyValueError::new_err(message),
            BacktrackError::Sampling(message) => PyRuntimeError::new_err(message),
        }
    }
}

#[derive(Clone, Copy)]
struct BacktrackInputView<'a> {
    cols: usize,
    root_clade: usize,
    leaf_species: &'a [i64],
    split_parents: &'a [i64],
    split_leftrights: &'a [i64],
    log_split_probs: &'a [f64],
    pi: &'a [f64],
    pibar: &'a [f64],
    e: &'a [f64],
    ebar: &'a [f64],
    log_p_s: &'a [f64],
    log_p_d: &'a [f64],
    receiver_log_probs: &'a [f64],
}

impl BacktrackInputView<'_> {
    fn pi(&self, row: usize, col: usize) -> f64 {
        self.pi[row * self.cols + col]
    }

    fn pibar(&self, row: usize, col: usize) -> f64 {
        self.pibar[row * self.cols + col]
    }

    fn split_left(&self, idx: usize) -> usize {
        self.split_leftrights[idx] as usize
    }

    fn split_right(&self, idx: usize) -> usize {
        self.split_leftrights[self.split_leftrights.len() / 2 + idx] as usize
    }
}

/// Per-family CSR index mapping each clade (parent) to the split rows whose parent
/// is that clade, in ascending split-row order -- exactly the order the former
/// `BacktrackInputView::splits_for_parent` linear scan yielded, but with O(1) lookup
/// + O(k) iteration instead of O(N_splits) per call. Built once per family and shared
/// across all K tracebacks, so the yielded split indices and their order are unchanged.
struct SplitIndex {
    offsets: Vec<usize>,
    indices: Vec<usize>,
}

impl SplitIndex {
    fn build(split_parents: &[i64], clade_count: usize) -> Self {
        let mut offsets = vec![0usize; clade_count + 1];
        for &parent in split_parents {
            offsets[parent as usize + 1] += 1;
        }
        for clade in 0..clade_count {
            offsets[clade + 1] += offsets[clade];
        }
        let mut cursor = offsets[..clade_count].to_vec();
        let mut indices = vec![0usize; split_parents.len()];
        for (idx, &parent) in split_parents.iter().enumerate() {
            let parent = parent as usize;
            indices[cursor[parent]] = idx;
            cursor[parent] += 1;
        }
        SplitIndex { offsets, indices }
    }

    fn splits_for_parent(&self, parent: usize) -> &[usize] {
        &self.indices[self.offsets[parent]..self.offsets[parent + 1]]
    }
}

/// Per-clade transfer-recipient sampler, built once per family and reused across
/// all K tracebacks. Stores the prefix-sum CDF of `2^(w[r] - max)` over EVERY
/// species r, with `w[r] = receiver_log_probs[r] + pi(clade, r)` and `max` the
/// maximum finite weight -- exactly the weights and `is_finite` selectability the
/// former per-transfer `sample_index` applied. A non-finite weight contributes
/// zero mass (an empty CDF interval) so it is never drawn. Sampling is O(log S)
/// via binary search; the caller conditions on non-ancestor recipients by
/// rejection. Building this once and reusing it over all K samples replaces the
/// former O(S) exclude-then-normalize rebuild on every transfer event.
struct RecipientSampler {
    cdf: Vec<f64>,
}

impl RecipientSampler {
    fn build(input: &BacktrackInputView<'_>, clade: usize) -> Self {
        let s = input.cols;
        // Same max-fold as sample_index: maximum over finite weights only.
        let mut max = NEG_INF;
        for r in 0..s {
            let w = input.receiver_log_probs[r] + input.pi(clade, r);
            if w.is_finite() && w > max {
                max = w;
            }
        }
        // Prefix sums of 2^(w-max) under the identical is_finite filter. If every
        // weight is non-finite, `max` stays NEG_INF, no term is finite, and the
        // CDF is all zeros (total mass 0) -> sample() returns None, matching the
        // former sample_index "all candidate weights are invalid" failure.
        let mut cdf = Vec::with_capacity(s);
        let mut acc = 0.0_f64;
        for r in 0..s {
            let w = input.receiver_log_probs[r] + input.pi(clade, r);
            if w.is_finite() {
                acc += 2.0_f64.powf(w - max);
            }
            cdf.push(acc);
        }
        RecipientSampler { cdf }
    }

    /// Draw one species index with probability proportional to its stored mass,
    /// or `None` when the total mass is zero (all weights non-finite).
    fn sample(&self, rng: &mut StdRng) -> Option<usize> {
        let total = *self.cdf.last()?;
        if !(total > 0.0) {
            return None;
        }
        // `Uniform` is half-open [0, total), so draw < total = cdf.last(); the
        // first prefix sum reaching `draw` therefore has an index < cdf.len().
        // This is the same species the former ordered sample_index walk
        // (subtracting 2^(w-max) until <= 0) would return for these cumulative
        // masses; zero-mass species have empty intervals and are skipped.
        let draw = Uniform::new(0.0, total).sample(rng);
        Some(self.cdf.partition_point(|&c| c < draw))
    }
}

#[derive(Clone, Copy)]
struct SpeciesTopology<'a> {
    child1: &'a [i64],
    child2: &'a [i64],
    subtree_start: &'a [i64],
    subtree_end: &'a [i64],
}

#[derive(Clone, Copy)]
enum Term {
    Continue,
    TransferLossDonor,
    HiddenSpeciation(bool),
    Leaf,
    SplitDup(usize),
    SplitTransfer(usize, bool),
    SplitSpeciation(usize, bool),
}

use Term::*;
type SampleNode = (&'static str, usize, [i64; 2]);

struct Sampler<'a, 'c> {
    input: BacktrackInputView<'a>,
    species: SpeciesTopology<'a>,
    split_index: &'a SplitIndex,
    /// Per-family cache of lazily-built full-species recipient samplers, one slot
    /// per clade (`None` until the first transfer under that clade). Owned by the
    /// caller (count_family / the single-draw entry point) and shared across all K
    /// tracebacks, so each clade's O(S) CDF is built at most once per family.
    recipient_cache: &'c mut Vec<Option<RecipientSampler>>,
    rng: StdRng,
    nodes: Vec<SampleNode>,
    scratch: Vec<(usize, f64)>,
}

impl<'a, 'c> Sampler<'a, 'c> {
    fn sample(&mut self) -> Result<Vec<SampleNode>, BacktrackError> {
        let root_species = self.sample_root_species()?;
        let root = self.add_node("speciation", root_species);
        let root_clade = self.input.root_clade;
        let mut stack = vec![(root, root_clade, root_species)];

        while let Some((node_idx, clade, species)) = stack.pop() {
            let term = self.sample_term(clade, species)?;
            self.apply_term(node_idx, clade, species, term, &mut stack)?;
        }

        Ok(std::mem::take(&mut self.nodes))
    }

    fn sample_root_species(&mut self) -> Result<usize, BacktrackError> {
        let input = self.input;
        let root_clade = input.root_clade;
        self.scratch.clear();
        self.scratch
            .extend((0..input.cols).map(|species| (species, input.pi(root_clade, species))));
        sample_index(&self.scratch, &mut self.rng, || "root species".to_string())
    }

    fn sample_term(&mut self, clade: usize, species: usize) -> Result<Term, BacktrackError> {
        let input = self.input;
        let split_index = self.split_index;
        let mut candidates = Vec::with_capacity(11);
        let pi_cs = input.pi(clade, species);
        let e_s = input.e[species];
        let p_s = input.log_p_s[species];
        let p_d = input.log_p_d[species];
        let children = self.species.children(species);

        let mut add = |term, weight| candidates.push((term, weight));
        add(Continue, 1.0 + p_d + e_s + pi_cs);
        add(Continue, pi_cs + input.ebar[species]);
        add(TransferLossDonor, input.pibar(clade, species) + e_s);

        if let Some((c1, c2)) = children {
            add(
                HiddenSpeciation(false),
                p_s + input.e[c2] + input.pi(clade, c1),
            );
            add(
                HiddenSpeciation(true),
                p_s + input.e[c1] + input.pi(clade, c2),
            );
        }

        if input.leaf_species[clade] == species as i64 {
            add(Leaf, p_s);
        }

        for split_idx in split_index.splits_for_parent(clade).iter().copied() {
            let left = input.split_left(split_idx);
            let right = input.split_right(split_idx);
            let lsp = input.log_split_probs[split_idx];
            add(
                SplitDup(split_idx),
                lsp + p_d + input.pi(left, species) + input.pi(right, species),
            );
            add(
                SplitTransfer(split_idx, true),
                lsp + input.pi(left, species) + input.pibar(right, species),
            );
            add(
                SplitTransfer(split_idx, false),
                lsp + input.pi(right, species) + input.pibar(left, species),
            );
            if let Some((c1, c2)) = children {
                add(
                    SplitSpeciation(split_idx, false),
                    lsp + p_s + input.pi(left, c1) + input.pi(right, c2),
                );
                add(
                    SplitSpeciation(split_idx, true),
                    lsp + p_s + input.pi(right, c1) + input.pi(left, c2),
                );
            }
        }

        sample_index(&candidates, &mut self.rng, || {
            format!("event term for clade {clade} and species {species}")
        })
    }

    fn apply_term(
        &mut self,
        node_idx: usize,
        clade: usize,
        species: usize,
        term: Term,
        stack: &mut Vec<(usize, usize, usize)>,
    ) -> Result<(), BacktrackError> {
        match term {
            Leaf => {
                self.nodes[node_idx].0 = "leaf";
            }
            Continue => {
                stack.push((node_idx, clade, species));
            }
            TransferLossDonor => {
                let recipient = self.sample_pibar_recipient(clade, species)?;
                self.nodes[node_idx].0 = "transfer";
                let loss = self.add_node("loss", species);
                let cont = self.add_node("leaf", recipient);
                self.set_children(node_idx, loss, cont);
                stack.push((cont, clade, recipient));
            }
            HiddenSpeciation(swapped) => {
                let (c1, c2) = self.species.children(species).ok_or_else(|| {
                    BacktrackError::Sampling(format!(
                        "hidden speciation selected for leaf species {species}"
                    ))
                })?;
                let (cont_species, loss_species) = if swapped { (c2, c1) } else { (c1, c2) };
                self.nodes[node_idx].0 = "speciation";
                let cont = self.add_node("leaf", cont_species);
                let loss = self.add_node("loss", loss_species);
                self.set_children(node_idx, cont, loss);
                stack.push((cont, clade, cont_species));
            }
            SplitDup(split_idx) => {
                let left_clade = self.input.split_left(split_idx);
                let right_clade = self.input.split_right(split_idx);
                self.nodes[node_idx].0 = "duplication";
                let left = self.add_node("leaf", species);
                let right = self.add_node("leaf", species);
                self.set_children(node_idx, left, right);
                stack.push((right, right_clade, species));
                stack.push((left, left_clade, species));
            }
            SplitTransfer(split_idx, recipient_on_right) => {
                let left_clade = self.input.split_left(split_idx);
                let right_clade = self.input.split_right(split_idx);
                let recipient_clade = if recipient_on_right {
                    right_clade
                } else {
                    left_clade
                };
                let recipient = self.sample_pibar_recipient(recipient_clade, species)?;
                self.nodes[node_idx].0 = "transfer";
                let donor_child = self.add_node("leaf", species);
                let recipient_child = self.add_node("leaf", recipient);
                let (left_child, right_child, left_species, right_species) = if recipient_on_right {
                    (donor_child, recipient_child, species, recipient)
                } else {
                    (recipient_child, donor_child, recipient, species)
                };
                self.set_children(node_idx, left_child, right_child);
                stack.push((right_child, right_clade, right_species));
                stack.push((left_child, left_clade, left_species));
            }
            SplitSpeciation(split_idx, swapped) => {
                let left_clade = self.input.split_left(split_idx);
                let right_clade = self.input.split_right(split_idx);
                let (c1, c2) = self.species.children(species).ok_or_else(|| {
                    BacktrackError::Sampling(format!(
                        "split speciation selected for leaf species {species}"
                    ))
                })?;
                self.nodes[node_idx].0 = "speciation";
                let left_node = self.add_node("leaf", c1);
                let right_node = self.add_node("leaf", c2);
                self.set_children(node_idx, left_node, right_node);
                let (left_clade, right_clade) = if swapped {
                    (right_clade, left_clade)
                } else {
                    (left_clade, right_clade)
                };
                stack.push((right_node, right_clade, c2));
                stack.push((left_node, left_clade, c1));
            }
        }
        Ok(())
    }

    /// Sample a transfer recipient for `donor` under `clade`'s recipient
    /// distribution `w[r] = receiver_log_probs[r] + pi(clade, r)`, excluding the
    /// donor's ancestors. The full-species weights depend only on `clade` (fixed
    /// across all K tracebacks of a family), so instead of rebuilding the O(S)
    /// exclude-then-normalize vector on every transfer we lazily build and cache a
    /// per-clade CDF over ALL species once, then rejection-sample: draw r from the
    /// full distribution and reject it if it is an ancestor of `donor`.
    ///
    /// Rejection-conditioning a categorical on the admissible (non-ancestor) subset
    /// yields exactly `w[r] / sum_{r' not ancestor(donor)} w[r']` for admissible r,
    /// which is the same distribution the former exclude-then-normalize produced
    /// (same weights, same `is_finite` selectability, same normalizer). Ancestors
    /// are ~1.5% of the mass, so acceptance is O(1) expected per transfer.
    fn sample_pibar_recipient(
        &mut self,
        clade: usize,
        donor: usize,
    ) -> Result<usize, BacktrackError> {
        // Consecutive ancestor draws tolerated before falling back to the exact
        // exhaustive draw. The fallback samples the *same* admissible-conditional
        // law, so bounding the loop is distribution-identical (see module note on
        // rejection sampling); it only guards against an unbounded loop when the
        // admissible mass is ~0, in which case the exhaustive path reproduces the
        // former error exactly.
        const MAX_REJECT: usize = 64;

        if self.recipient_cache[clade].is_none() {
            self.recipient_cache[clade] = Some(RecipientSampler::build(&self.input, clade));
        }
        // Disjoint field borrows: the cached sampler (shared), the RNG (unique),
        // and the Copy species topology.
        let sampler = self.recipient_cache[clade].as_ref().unwrap();
        let species = self.species;
        let rng = &mut self.rng;
        for _ in 0..MAX_REJECT {
            match sampler.sample(rng) {
                Some(recipient) => {
                    if !species.is_ancestor(recipient, donor) {
                        return Ok(recipient);
                    }
                    // recipient is an ancestor of the donor -> reject and redraw
                }
                None => break, // no admissible mass at all -> exact path reports it
            }
        }
        // Pathological fallback (admissible mass ~0, or an unlucky reject streak):
        // exact exclude-then-normalize, identical to the former implementation.
        self.sample_pibar_recipient_exhaustive(clade, donor)
    }

    /// Exact exclude-then-normalize recipient draw -- the former implementation,
    /// kept as the guaranteed-terminating fallback for `sample_pibar_recipient`.
    fn sample_pibar_recipient_exhaustive(
        &mut self,
        clade: usize,
        donor: usize,
    ) -> Result<usize, BacktrackError> {
        let input = self.input;
        let topology = self.species;
        self.scratch.clear();
        self.scratch.extend(
            (0..input.cols)
                .filter(|recipient| !topology.is_ancestor(*recipient, donor))
                .map(|recipient| {
                    (
                        recipient,
                        input.receiver_log_probs[recipient] + input.pi(clade, recipient),
                    )
                }),
        );
        sample_index(&self.scratch, &mut self.rng, || {
            format!("transfer recipient for clade {clade} and donor species {donor}")
        })
    }

    fn add_node(&mut self, event: &'static str, gp_species: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push((event, gp_species, [-1; 2]));
        idx
    }

    fn set_children(&mut self, parent: usize, left: usize, right: usize) {
        self.nodes[parent].2 = [left as i64, right as i64];
    }
}

impl SpeciesTopology<'_> {
    fn is_ancestor(&self, ancestor: usize, node: usize) -> bool {
        self.subtree_start[ancestor] <= self.subtree_start[node]
            && self.subtree_start[node] < self.subtree_end[ancestor]
    }

    fn children(&self, species: usize) -> Option<(usize, usize)> {
        let c1 = self.child1[species];
        let c2 = self.child2[species];
        let s = self.child1.len() as i64;
        (0 <= c1 && c1 < s && 0 <= c2 && c2 < s).then_some((c1 as usize, c2 as usize))
    }
}

/// Draw one item with probability proportional to `2^weight`. The candidate buffer
/// is read three times (max, normalizing total, ordered walk) but every weight is
/// computed exactly ONCE by the caller when filling `weighted` -- previously the
/// callers passed lazy `Clone` iterators that recomputed each weight on all three
/// passes. The math is byte-for-byte identical to the previous version: same
/// `is_finite` filter, same `max` fold, same `total` sum over `2^(w-max)` in the same
/// order, the same single `Uniform::new(0.0, total)` draw, and the same ordered walk
/// subtracting `2^(w-max)`. Because the weights, max, total, the one uniform draw, and
/// the iteration order are unchanged, the sampled result is bit-identical.
fn sample_index<T, F>(weighted: &[(T, f64)], rng: &mut StdRng, context: F) -> Result<T, BacktrackError>
where
    T: Copy,
    F: Fn() -> String,
{
    // `context` is a closure so the error-message string is only allocated on the
    // (rare) failure paths, not built on every call.
    let selectable = |weight: f64| weight.is_finite();
    let max = weighted
        .iter()
        .copied()
        .map(|(_, weight)| weight)
        .filter(|weight| selectable(*weight))
        .fold(NEG_INF, f64::max);
    if max == NEG_INF {
        return Err(BacktrackError::Sampling(format!(
            "all candidate weights are invalid for {}",
            context()
        )));
    }
    let total = weighted
        .iter()
        .copied()
        .filter(|(_, weight)| selectable(*weight))
        .map(|(_, weight)| 2.0_f64.powf(weight - max))
        .sum::<f64>();
    let mut draw = Uniform::new(0.0, total).sample(rng);
    for (item, log_w) in weighted.iter().copied() {
        if !selectable(log_w) {
            continue;
        }
        draw -= 2.0_f64.powf(log_w - max);
        if draw <= 0.0 {
            return Ok(item);
        }
    }
    Err(BacktrackError::Sampling(format!(
        "failed to draw from candidate weights for {}",
        context()
    )))
}

fn slice_from_numpy<'a, T: numpy::Element>(
    name: &str,
    values: &'a PyReadonlyArray1<'_, T>,
) -> Result<&'a [T], BacktrackError> {
    values.as_slice().map_err(|_| {
        BacktrackError::InvalidInput(format!(
            "{name} must be a contiguous one-dimensional NumPy array"
        ))
    })
}

fn matrix_slice_from_numpy<'a>(
    name: &str,
    values: &'a PyReadonlyArray2<'_, f64>,
) -> Result<&'a [f64], BacktrackError> {
    values.as_slice().map_err(|_| {
        BacktrackError::InvalidInput(format!(
            "{name} must be a contiguous two-dimensional NumPy array"
        ))
    })
}

fn require_len<T>(name: &str, values: &[T], expected: usize) -> Result<(), BacktrackError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(BacktrackError::InvalidInput(format!(
            "{name} has length {} but expected {expected}",
            values.len()
        )))
    }
}

fn require_nonempty(name: &str, len: usize) -> Result<(), BacktrackError> {
    if len > 0 {
        Ok(())
    } else {
        Err(BacktrackError::InvalidInput(format!(
            "{name} must not be empty"
        )))
    }
}

fn validate_clade_indices(
    name: &str,
    values: &[i64],
    clade_count: usize,
) -> Result<(), BacktrackError> {
    for (idx, value) in values.iter().copied().enumerate() {
        if value < 0 || value as usize >= clade_count {
            return Err(BacktrackError::InvalidInput(format!(
                "{name}[{idx}]={value} is outside clade range 0..{clade_count}"
            )));
        }
    }
    Ok(())
}

fn validate_leaf_species(values: &[i64], species_count: usize) -> Result<(), BacktrackError> {
    for (idx, value) in values.iter().copied().enumerate() {
        if value != -1 && (value < 0 || value as usize >= species_count) {
            return Err(BacktrackError::InvalidInput(format!(
                "leaf_species[{idx}]={value} is neither -1 nor in species range 0..{species_count}"
            )));
        }
    }
    Ok(())
}

fn validate_species_topology(
    child1: &[i64],
    child2: &[i64],
    subtree_start: &[i64],
    subtree_end: &[i64],
) -> Result<(), BacktrackError> {
    let species_count = child1.len();
    require_len("sp_child2", child2, species_count)?;
    require_len("sp_subtree_start", subtree_start, species_count)?;
    require_len("sp_subtree_end", subtree_end, species_count)?;
    for idx in 0..species_count {
        let c1_valid = 0 <= child1[idx] && (child1[idx] as usize) < species_count;
        let c2_valid = 0 <= child2[idx] && (child2[idx] as usize) < species_count;
        if c1_valid != c2_valid {
            return Err(BacktrackError::InvalidInput(format!(
                "species node {idx} has only one valid child: sp_child1={}, sp_child2={}",
                child1[idx], child2[idx]
            )));
        }
        let start = subtree_start[idx];
        let end = subtree_end[idx];
        if start < 0 || end <= start || end as usize > species_count {
            return Err(BacktrackError::InvalidInput(format!(
                "invalid subtree interval for species node {idx}: [{start}, {end}) with S={species_count}"
            )));
        }
    }
    Ok(())
}

/// Core stochastic-backtracking draw shared by every entry point so the sampling
/// math is defined exactly once. Given already-validated input views and a seed,
/// this constructs a fresh RNG and runs the identical `Sampler` used everywhere.
fn draw_sample<'a, 'c>(
    input: BacktrackInputView<'a>,
    species: SpeciesTopology<'a>,
    split_index: &'a SplitIndex,
    recipient_cache: &'c mut Vec<Option<RecipientSampler>>,
    seed: u64,
) -> Result<Vec<SampleNode>, BacktrackError> {
    Sampler {
        input,
        species,
        split_index,
        recipient_cache,
        rng: StdRng::seed_from_u64(seed),
        nodes: Vec::new(),
        scratch: Vec::new(),
    }
    .sample()
}

/// Validate the shared species-topology arrays and build a `SpeciesTopology` view.
fn build_species_topology<'a>(
    child1: &'a [i64],
    child2: &'a [i64],
    subtree_start: &'a [i64],
    subtree_end: &'a [i64],
) -> Result<SpeciesTopology<'a>, BacktrackError> {
    validate_species_topology(child1, child2, subtree_start, subtree_end)?;
    Ok(SpeciesTopology {
        child1,
        child2,
        subtree_start,
        subtree_end,
    })
}

/// Validate one family's arrays and build its `BacktrackInputView`. Performs the
/// exact same per-family checks and field mapping as the inline validation in
/// `sample_reconciliations_torch`, so a batched family is checked and constructed
/// identically.
#[allow(clippy::too_many_arguments)]
fn build_family_view<'a>(
    root_clade: u64,
    leaf_species: &'a [i64],
    split_parents: &'a [i64],
    split_leftrights: &'a [i64],
    log_split_probs: &'a [f64],
    pi_values: &'a [f64],
    pi_shape: [usize; 2],
    pibar_values: &'a [f64],
    pibar_shape: [usize; 2],
    e: &'a [f64],
    ebar: &'a [f64],
    log_p_s: &'a [f64],
    log_p_d: &'a [f64],
    receiver_log_probs: &'a [f64],
) -> Result<BacktrackInputView<'a>, BacktrackError> {
    let clade_count = pi_shape[0];
    let species_count = pi_shape[1];
    require_nonempty("pi rows", clade_count)?;
    require_nonempty("pi columns", species_count)?;
    if pibar_shape != pi_shape {
        return Err(BacktrackError::InvalidInput(format!(
            "pibar shape {:?} does not match pi shape {:?}",
            pibar_shape, pi_shape
        )));
    }
    if root_clade > usize::MAX as u64 || root_clade as usize >= clade_count {
        return Err(BacktrackError::InvalidInput(format!(
            "root_clade {root_clade} is outside clade range 0..{clade_count}"
        )));
    }
    require_len("leaf_species", leaf_species, clade_count)?;
    require_len("split_parents", split_parents, log_split_probs.len())?;
    require_len("split_leftrights", split_leftrights, 2 * log_split_probs.len())?;
    require_len("e", e, species_count)?;
    require_len("ebar", ebar, species_count)?;
    require_len("log_p_s", log_p_s, species_count)?;
    require_len("log_p_d", log_p_d, species_count)?;
    require_len("receiver_log_probs", receiver_log_probs, species_count)?;
    validate_clade_indices("split_parents", split_parents, clade_count)?;
    validate_clade_indices("split_leftrights", split_leftrights, clade_count)?;
    validate_leaf_species(leaf_species, species_count)?;
    Ok(BacktrackInputView {
        cols: species_count,
        root_clade: root_clade as usize,
        leaf_species,
        split_parents,
        split_leftrights,
        log_split_probs,
        pi: pi_values,
        pibar: pibar_values,
        e,
        ebar,
        log_p_s,
        log_p_d,
        receiver_log_probs,
    })
}

/// Draw `n_samples` reconciliations for one family (seeds `seed..seed+n_samples`)
/// and aggregate the event statistics with the GIL released. Replicates the Python
/// per-event counting exactly: every "transfer" node increments the transfer total
/// (and, when a child sits at a different species than the donor, its
/// (donor, recipient) edge histogram entry); every "duplication" node increments
/// the duplication total.
fn count_family(
    input: BacktrackInputView<'_>,
    species: SpeciesTopology<'_>,
    n_samples: usize,
    seed: u64,
) -> Result<(usize, usize, Vec<(i64, i64, u32)>), BacktrackError> {
    // Built once per family and reused across all K tracebacks; yields identical
    // split rows in identical order to the previous per-step linear scan.
    let clade_count = input.leaf_species.len();
    let split_index = SplitIndex::build(input.split_parents, clade_count);
    // Per-family recipient-sampler cache: one lazily-filled slot per clade, shared
    // across all K tracebacks so each clade's O(S) CDF is built at most once.
    let mut recipient_cache: Vec<Option<RecipientSampler>> =
        (0..clade_count).map(|_| None).collect();
    let mut n_transfers_total: usize = 0;
    let mut n_dup_total: usize = 0;
    let mut edges: HashMap<(i64, i64), u32> = HashMap::new();
    for k in 0..n_samples {
        let nodes = draw_sample(
            input,
            species,
            &split_index,
            &mut recipient_cache,
            seed + k as u64,
        )?;
        let n = nodes.len();
        for i in 0..n {
            let (ev_type, sp, kids) = nodes[i];
            match ev_type {
                "transfer" => {
                    n_transfers_total += 1;
                    let donor = sp as i64;
                    for &child in kids.iter() {
                        if child >= 0 && (child as usize) < n {
                            let recipient = nodes[child as usize].1 as i64;
                            if recipient != donor {
                                *edges.entry((donor, recipient)).or_insert(0) += 1;
                                break;
                            }
                        }
                    }
                }
                "duplication" => n_dup_total += 1,
                _ => {}
            }
        }
    }
    let edges_vec: Vec<(i64, i64, u32)> =
        edges.into_iter().map(|((d, r), c)| (d, r, c)).collect();
    Ok((n_transfers_total, n_dup_total, edges_vec))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_reconciliations_torch(
    py: Python<'_>,
    root_clade: u64,
    leaf_species: PyReadonlyArray1<'_, i64>,
    split_parents: PyReadonlyArray1<'_, i64>,
    split_leftrights: PyReadonlyArray1<'_, i64>,
    log_split_probs: PyReadonlyArray1<'_, f64>,
    pi: PyReadonlyArray2<'_, f64>,
    pibar: PyReadonlyArray2<'_, f64>,
    e: PyReadonlyArray1<'_, f64>,
    ebar: PyReadonlyArray1<'_, f64>,
    log_p_s: PyReadonlyArray1<'_, f64>,
    log_p_d: PyReadonlyArray1<'_, f64>,
    receiver_log_probs: PyReadonlyArray1<'_, f64>,
    sp_child1: PyReadonlyArray1<'_, i64>,
    sp_child2: PyReadonlyArray1<'_, i64>,
    sp_subtree_start: PyReadonlyArray1<'_, i64>,
    sp_subtree_end: PyReadonlyArray1<'_, i64>,
    seed: u64,
) -> PyResult<PyObject> {
    let pi_shape = pi.shape();
    let pibar_shape = pibar.shape();
    let clade_count = pi_shape[0];
    let species_count = pi_shape[1];
    require_nonempty("pi rows", clade_count).map_err(PyErr::from)?;
    require_nonempty("pi columns", species_count).map_err(PyErr::from)?;
    if pibar_shape != pi_shape {
        return Err(PyValueError::new_err(format!(
            "pibar shape {:?} does not match pi shape {:?}",
            pibar_shape, pi_shape
        )));
    }
    if root_clade > usize::MAX as u64 || root_clade as usize >= clade_count {
        return Err(PyValueError::new_err(format!(
            "root_clade {root_clade} is outside clade range 0..{clade_count}"
        )));
    }

    let leaf_species = slice_from_numpy("leaf_species", &leaf_species).map_err(PyErr::from)?;
    let split_parents = slice_from_numpy("split_parents", &split_parents).map_err(PyErr::from)?;
    let split_leftrights =
        slice_from_numpy("split_leftrights", &split_leftrights).map_err(PyErr::from)?;
    let log_split_probs =
        slice_from_numpy("log_split_probs", &log_split_probs).map_err(PyErr::from)?;
    let pi_values = matrix_slice_from_numpy("pi", &pi).map_err(PyErr::from)?;
    let pibar_values = matrix_slice_from_numpy("pibar", &pibar).map_err(PyErr::from)?;
    let e = slice_from_numpy("e", &e).map_err(PyErr::from)?;
    let ebar = slice_from_numpy("ebar", &ebar).map_err(PyErr::from)?;
    let log_p_s = slice_from_numpy("log_p_s", &log_p_s).map_err(PyErr::from)?;
    let log_p_d = slice_from_numpy("log_p_d", &log_p_d).map_err(PyErr::from)?;
    let receiver_log_probs =
        slice_from_numpy("receiver_log_probs", &receiver_log_probs).map_err(PyErr::from)?;
    let child1 = slice_from_numpy("sp_child1", &sp_child1).map_err(PyErr::from)?;
    let child2 = slice_from_numpy("sp_child2", &sp_child2).map_err(PyErr::from)?;
    let subtree_start =
        slice_from_numpy("sp_subtree_start", &sp_subtree_start).map_err(PyErr::from)?;
    let subtree_end = slice_from_numpy("sp_subtree_end", &sp_subtree_end).map_err(PyErr::from)?;

    require_len("leaf_species", leaf_species, clade_count).map_err(PyErr::from)?;
    require_len("split_parents", split_parents, log_split_probs.len()).map_err(PyErr::from)?;
    require_len(
        "split_leftrights",
        split_leftrights,
        2 * log_split_probs.len(),
    )
    .map_err(PyErr::from)?;
    require_len("e", e, species_count).map_err(PyErr::from)?;
    require_len("ebar", ebar, species_count).map_err(PyErr::from)?;
    require_len("log_p_s", log_p_s, species_count).map_err(PyErr::from)?;
    require_len("log_p_d", log_p_d, species_count).map_err(PyErr::from)?;
    require_len("receiver_log_probs", receiver_log_probs, species_count).map_err(PyErr::from)?;
    validate_clade_indices("split_parents", split_parents, clade_count).map_err(PyErr::from)?;
    validate_clade_indices("split_leftrights", split_leftrights, clade_count)
        .map_err(PyErr::from)?;
    validate_leaf_species(leaf_species, species_count).map_err(PyErr::from)?;
    validate_species_topology(child1, child2, subtree_start, subtree_end).map_err(PyErr::from)?;

    let input_view = BacktrackInputView {
        cols: species_count,
        root_clade: root_clade as usize,
        leaf_species,
        split_parents,
        split_leftrights,
        log_split_probs,
        pi: pi_values,
        pibar: pibar_values,
        e,
        ebar,
        log_p_s,
        log_p_d,
        receiver_log_probs,
    };
    let species = SpeciesTopology {
        child1,
        child2,
        subtree_start,
        subtree_end,
    };
    let split_index = SplitIndex::build(split_parents, clade_count);
    let mut recipient_cache: Vec<Option<RecipientSampler>> =
        (0..clade_count).map(|_| None).collect();
    let sample_nodes = py.allow_threads(move || {
        draw_sample(input_view, species, &split_index, &mut recipient_cache, seed)
    });
    sample_nodes
        .map(|nodes| nodes.into_py(py))
        .map_err(PyErr::from)
}

/// Draw `n_samples` reconciliations for a single gene family and return the
/// aggregated event statistics, with the whole K-sample loop running GIL-free.
///
/// Positional arguments are identical to `sample_reconciliations_torch` (root
/// clade, family CCP arrays, forward-solve outputs, and species topology) plus
/// `n_samples` and a base `seed`; the k-th draw uses `seed + k` so results match a
/// Python loop calling `sample_reconciliations_torch(..., seed + k)`. Returns
/// `(n_transfers_total, n_dup_total, [(donor_species, recipient_species, count), ...])`.
/// The counting logic mirrors the Python per-event counting exactly.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_family_counts(
    py: Python<'_>,
    root_clade: u64,
    leaf_species: PyReadonlyArray1<'_, i64>,
    split_parents: PyReadonlyArray1<'_, i64>,
    split_leftrights: PyReadonlyArray1<'_, i64>,
    log_split_probs: PyReadonlyArray1<'_, f64>,
    pi: PyReadonlyArray2<'_, f64>,
    pibar: PyReadonlyArray2<'_, f64>,
    e: PyReadonlyArray1<'_, f64>,
    ebar: PyReadonlyArray1<'_, f64>,
    log_p_s: PyReadonlyArray1<'_, f64>,
    log_p_d: PyReadonlyArray1<'_, f64>,
    receiver_log_probs: PyReadonlyArray1<'_, f64>,
    sp_child1: PyReadonlyArray1<'_, i64>,
    sp_child2: PyReadonlyArray1<'_, i64>,
    sp_subtree_start: PyReadonlyArray1<'_, i64>,
    sp_subtree_end: PyReadonlyArray1<'_, i64>,
    n_samples: usize,
    seed: u64,
) -> PyResult<(usize, usize, Vec<(i64, i64, u32)>)> {
    let leaf_species = slice_from_numpy("leaf_species", &leaf_species)?;
    let split_parents = slice_from_numpy("split_parents", &split_parents)?;
    let split_leftrights = slice_from_numpy("split_leftrights", &split_leftrights)?;
    let log_split_probs = slice_from_numpy("log_split_probs", &log_split_probs)?;
    let pi_shape = pi.shape();
    let pi_shape = [pi_shape[0], pi_shape[1]];
    let pibar_shape = pibar.shape();
    let pibar_shape = [pibar_shape[0], pibar_shape[1]];
    let pi_values = matrix_slice_from_numpy("pi", &pi)?;
    let pibar_values = matrix_slice_from_numpy("pibar", &pibar)?;
    let e = slice_from_numpy("e", &e)?;
    let ebar = slice_from_numpy("ebar", &ebar)?;
    let log_p_s = slice_from_numpy("log_p_s", &log_p_s)?;
    let log_p_d = slice_from_numpy("log_p_d", &log_p_d)?;
    let receiver = slice_from_numpy("receiver_log_probs", &receiver_log_probs)?;
    let child1 = slice_from_numpy("sp_child1", &sp_child1)?;
    let child2 = slice_from_numpy("sp_child2", &sp_child2)?;
    let subtree_start = slice_from_numpy("sp_subtree_start", &sp_subtree_start)?;
    let subtree_end = slice_from_numpy("sp_subtree_end", &sp_subtree_end)?;

    let species = build_species_topology(child1, child2, subtree_start, subtree_end)?;
    let input_view = build_family_view(
        root_clade,
        leaf_species,
        split_parents,
        split_leftrights,
        log_split_probs,
        pi_values,
        pi_shape,
        pibar_values,
        pibar_shape,
        e,
        ebar,
        log_p_s,
        log_p_d,
        receiver,
    )?;

    // Entire K-sample loop + per-event counting runs with the GIL released, so the
    // Python ThreadPoolExecutor over families scales past the previous GIL plateau.
    let result = py.allow_threads(move || count_family(input_view, species, n_samples, seed));
    result.map_err(PyErr::from)
}

#[pymodule]
fn gpurec_backtrack(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sample_reconciliations_torch, module)?)?;
    module.add_function(wrap_pyfunction!(sample_family_counts, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{RecipientSampler, NEG_INF};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // Build a RecipientSampler from an explicit weight vector using the SAME CDF
    // construction as RecipientSampler::build (2^(w-max) prefix sums, is_finite
    // filter, max over finite weights only).
    fn cdf_from_weights(w: &[f64]) -> RecipientSampler {
        let mut max = NEG_INF;
        for &x in w {
            if x.is_finite() && x > max {
                max = x;
            }
        }
        let mut cdf = Vec::with_capacity(w.len());
        let mut acc = 0.0_f64;
        for &x in w {
            if x.is_finite() {
                acc += 2.0_f64.powf(x - max);
            }
            cdf.push(acc);
        }
        RecipientSampler { cdf }
    }

    // Per-species mass from the CDF (prefix-sum differences) -- what sample() draws
    // proportionally to.
    fn masses(s: &RecipientSampler) -> Vec<f64> {
        let mut m = Vec::with_capacity(s.cdf.len());
        let mut prev = 0.0_f64;
        for &c in &s.cdf {
            m.push(c - prev);
            prev = c;
        }
        m
    }

    // OLD behaviour: exclude ancestors, then normalize over the finite remainder,
    // with its OWN max-fold over the admissible set (exactly what the former
    // sample_index over the filtered scratch did).
    fn exclude_then_normalize(w: &[f64], is_anc: &[bool]) -> Vec<f64> {
        let mut max = NEG_INF;
        for (r, &x) in w.iter().enumerate() {
            if !is_anc[r] && x.is_finite() && x > max {
                max = x;
            }
        }
        let mut p = vec![0.0_f64; w.len()];
        if max == NEG_INF {
            return p; // no admissible finite weight
        }
        let mut total = 0.0_f64;
        for (r, &x) in w.iter().enumerate() {
            if !is_anc[r] && x.is_finite() {
                let e = 2.0_f64.powf(x - max);
                p[r] = e;
                total += e;
            }
        }
        for v in &mut p {
            *v /= total;
        }
        p
    }

    // NEW behaviour: condition the full-species CDF distribution on non-ancestors
    // (exactly what the rejection loop yields in the limit).
    fn full_then_condition(s: &RecipientSampler, is_anc: &[bool]) -> Vec<f64> {
        let m = masses(s);
        let mut p = vec![0.0_f64; m.len()];
        let mut total = 0.0_f64;
        for (r, &mr) in m.iter().enumerate() {
            if !is_anc[r] {
                p[r] = mr;
                total += mr;
            }
        }
        for v in &mut p {
            *v /= total;
        }
        p
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    /// HARD GATE (1), Rust level: the NEW full-CDF-then-condition distribution is
    /// bit-identical (to fp) to the OLD exclude-then-normalize distribution, over
    /// hand-picked cases (incl. -inf weights, ancestors carrying finite mass) and a
    /// deterministic pseudo-random sweep.
    #[test]
    fn rejection_conditioning_matches_exclude_then_normalize() {
        let cases: Vec<(Vec<f64>, Vec<bool>)> = vec![
            (
                vec![0.0, -1.0, -3.0, -2.0, -0.5],
                vec![false, true, false, false, true],
            ),
            (
                vec![-5.0, -5.1, -4.9, NEG_INF, -5.05, -6.0],
                vec![true, false, false, false, true, false],
            ),
            (
                vec![NEG_INF, -2.0, -2.0, -2.0],
                vec![false, false, true, false],
            ),
        ];
        for (w, anc) in &cases {
            let s = cdf_from_weights(w);
            let a = exclude_then_normalize(w, anc);
            let b = full_then_condition(&s, anc);
            let md = max_abs_diff(&a, &b);
            assert!(md < 1e-12, "max abs diff {md} for w={w:?} anc={anc:?}");
        }

        // Deterministic LCG so the sweep is reproducible everywhere.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for _ in 0..5000 {
            let n = 2 + (next() % 40) as usize;
            let mut w = Vec::with_capacity(n);
            let mut anc = Vec::with_capacity(n);
            let mut any_admissible = false;
            for _ in 0..n {
                let x = if next() % 10 == 0 {
                    NEG_INF
                } else {
                    -8.0 * (next() as f64 / u64::MAX as f64)
                };
                let is_a = next() % 5 == 0;
                if !is_a && x.is_finite() {
                    any_admissible = true;
                }
                w.push(x);
                anc.push(is_a);
            }
            if !any_admissible {
                continue; // both distributions are all-zero; nothing to compare
            }
            let s = cdf_from_weights(&w);
            let a = exclude_then_normalize(&w, &anc);
            let b = full_then_condition(&s, &anc);
            let md = max_abs_diff(&a, &b);
            assert!(md < 1e-12, "random case max abs diff {md}");
        }
    }

    /// The compiled CDF sampler + ancestor rejection empirically reproduces the OLD
    /// exclude-then-normalize probabilities within Monte-Carlo error.
    #[test]
    fn sampler_empirical_frequencies_match_weights() {
        let w = vec![0.0, -0.5, -1.0, -2.0, -3.0, -0.2, NEG_INF, -1.5];
        let anc = vec![false, false, true, false, false, false, false, true];
        let target = exclude_then_normalize(&w, &anc);
        let s = cdf_from_weights(&w);
        let mut rng = StdRng::seed_from_u64(42);
        let n: usize = 300_000;
        let mut counts = vec![0u64; w.len()];
        let mut drawn = 0u64;
        while drawn < n as u64 {
            match s.sample(&mut rng) {
                Some(r) if !anc[r] => {
                    counts[r] += 1;
                    drawn += 1;
                }
                _ => {} // ancestor or None -> reject
            }
        }
        for r in 0..w.len() {
            let freq = counts[r] as f64 / n as f64;
            let sd = (target[r] * (1.0 - target[r]) / n as f64).sqrt();
            assert!(
                (freq - target[r]).abs() < 8.0 * sd + 1e-3,
                "species {r}: empirical {freq} vs target {} (sd {sd})",
                target[r]
            );
        }
    }
}

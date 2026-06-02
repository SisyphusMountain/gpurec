use rand::distributions::Uniform;
use rand::prelude::*;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
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

    fn splits_for_parent(&self, parent: usize) -> impl Iterator<Item = usize> + '_ {
        self.split_parents
            .iter()
            .enumerate()
            .filter(move |(_, &split_parent)| split_parent == parent as i64)
            .map(|(idx, _)| idx)
    }
}

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

struct Sampler<'a> {
    input: BacktrackInputView<'a>,
    species: SpeciesTopology<'a>,
    rng: StdRng,
    nodes: Vec<SampleNode>,
}

impl<'a> Sampler<'a> {
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
        sample_index(
            (0..input.cols).map(|species| (species, input.pi(root_clade, species))),
            &mut self.rng,
            "root species",
        )
    }

    fn sample_term(&mut self, clade: usize, species: usize) -> Result<Term, BacktrackError> {
        let input = self.input;
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

        for split_idx in input.splits_for_parent(clade) {
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

        let context = format!("event term for clade {clade} and species {species}");
        sample_index(candidates.iter().copied(), &mut self.rng, &context)
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

    fn sample_pibar_recipient(
        &mut self,
        clade: usize,
        donor: usize,
    ) -> Result<usize, BacktrackError> {
        let input = self.input;
        let topology = &self.species;
        let context = format!("transfer recipient for clade {clade} and donor species {donor}");
        sample_index(
            (0..input.cols)
                .filter(|recipient| !topology.is_ancestor(*recipient, donor))
                .map(|recipient| (recipient, input.pi(clade, recipient))),
            &mut self.rng,
            &context,
        )
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

fn sample_index<T, I>(weighted: I, rng: &mut StdRng, context: &str) -> Result<T, BacktrackError>
where
    T: Copy,
    I: Iterator<Item = (T, f64)> + Clone,
{
    let selectable = |weight: f64| weight.is_finite();
    let max = weighted
        .clone()
        .map(|(_, weight)| weight)
        .filter(|weight| selectable(*weight))
        .fold(NEG_INF, f64::max);
    if max == NEG_INF {
        return Err(BacktrackError::Sampling(format!(
            "all candidate weights are invalid for {context}"
        )));
    }
    let total = weighted
        .clone()
        .filter(|(_, weight)| selectable(*weight))
        .map(|(_, weight)| 2.0_f64.powf(weight - max))
        .sum::<f64>();
    let mut draw = Uniform::new(0.0, total).sample(rng);
    for (item, log_w) in weighted {
        if !selectable(log_w) {
            continue;
        }
        draw -= 2.0_f64.powf(log_w - max);
        if draw <= 0.0 {
            return Ok(item);
        }
    }
    Err(BacktrackError::Sampling(format!(
        "failed to draw from candidate weights for {context}"
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
    };
    let species = SpeciesTopology {
        child1,
        child2,
        subtree_start,
        subtree_end,
    };
    let sample_nodes = py.allow_threads(move || {
        Sampler {
            input: input_view,
            species,
            rng: StdRng::seed_from_u64(seed),
            nodes: Vec::new(),
        }
        .sample()
    });
    sample_nodes
        .map(|nodes| nodes.into_py(py))
        .map_err(PyErr::from)
}

#[pymodule]
fn gpurec_backtrack(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sample_reconciliations_torch, module)?)?;
    Ok(())
}

use rand::distributions::Uniform;
use rand::prelude::*;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

const NEG_INF: f64 = -1.0e300;

#[derive(Clone, Copy)]
struct BacktrackInputView<'a> {
    cols: usize,
    leaf_species: &'a [i64],
    split_leftrights: &'a [i64],
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

    fn split_index(&self, parent: usize) -> usize {
        self.leaf_species[..parent]
            .iter()
            .filter(|&&leaf| leaf < 0)
            .count()
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
    fn sample(&mut self) -> Vec<SampleNode> {
        let root_species = self.sample_root_species();
        let root = self.add_node("speciation", root_species);
        let root_clade = self.input.leaf_species.len() - 1;
        let mut stack = vec![(root, root_clade, root_species)];

        while let Some((node_idx, clade, species)) = stack.pop() {
            let term = self.sample_term(clade, species);
            self.apply_term(node_idx, clade, species, term, &mut stack);
        }

        std::mem::take(&mut self.nodes)
    }

    fn sample_root_species(&mut self) -> usize {
        let input = self.input;
        let root_clade = input.leaf_species.len() - 1;
        sample_index(
            (0..input.cols).map(|species| (species, input.pi(root_clade, species))),
            &mut self.rng,
        )
    }

    fn sample_term(&mut self, clade: usize, species: usize) -> Term {
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

        if input.leaf_species[clade] < 0 {
            let split_idx = input.split_index(clade);
            let left = input.split_left(split_idx);
            let right = input.split_right(split_idx);
            add(
                SplitDup(split_idx),
                p_d + input.pi(left, species) + input.pi(right, species),
            );
            add(
                SplitTransfer(split_idx, true),
                input.pi(left, species) + input.pibar(right, species),
            );
            add(
                SplitTransfer(split_idx, false),
                input.pi(right, species) + input.pibar(left, species),
            );
            if let Some((c1, c2)) = children {
                add(
                    SplitSpeciation(split_idx, false),
                    p_s + input.pi(left, c1) + input.pi(right, c2),
                );
                add(
                    SplitSpeciation(split_idx, true),
                    p_s + input.pi(right, c1) + input.pi(left, c2),
                );
            }
        }

        sample_index(candidates.iter().copied(), &mut self.rng)
    }

    fn apply_term(
        &mut self,
        node_idx: usize,
        clade: usize,
        species: usize,
        term: Term,
        stack: &mut Vec<(usize, usize, usize)>,
    ) {
        match term {
            Leaf => {
                self.nodes[node_idx].0 = "leaf";
            }
            Continue => {
                stack.push((node_idx, clade, species));
            }
            TransferLossDonor => {
                let recipient = self.sample_pibar_recipient(clade, species);
                self.nodes[node_idx].0 = "transfer";
                let loss = self.add_node("loss", species);
                let cont = self.add_node("leaf", recipient);
                self.set_children(node_idx, loss, cont);
                stack.push((cont, clade, recipient));
            }
            HiddenSpeciation(swapped) => {
                let (c1, c2) = self.species.children(species).unwrap();
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
                let recipient = self.sample_pibar_recipient(recipient_clade, species);
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
                let (c1, c2) = self.species.children(species).unwrap();
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
    }

    fn sample_pibar_recipient(&mut self, clade: usize, donor: usize) -> usize {
        let input = self.input;
        let topology = &self.species;
        sample_index(
            (0..input.cols)
                .filter(|recipient| !topology.is_ancestor(*recipient, donor))
                .map(|recipient| (recipient, input.pi(clade, recipient))),
            &mut self.rng,
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

fn sample_index<T, I>(weighted: I, rng: &mut StdRng) -> T
where
    T: Copy,
    I: Iterator<Item = (T, f64)> + Clone,
{
    let selectable = |weight: f64| weight.is_finite() && weight > NEG_INF / 2.0;
    let max = weighted
        .clone()
        .map(|(_, weight)| weight)
        .filter(|weight| selectable(*weight))
        .fold(NEG_INF, f64::max);
    if max <= NEG_INF / 2.0 {
        panic!("all candidate backtracking weights are zero");
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
            return item;
        }
    }
    panic!("all candidate backtracking weights are zero")
}

fn slice_from_numpy<'a, T: numpy::Element>(values: &'a PyReadonlyArray1<'_, T>) -> &'a [T] {
    values.as_slice().unwrap()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_reconciliations_torch(
    py: Python<'_>,
    leaf_species: PyReadonlyArray1<'_, i64>,
    split_leftrights: PyReadonlyArray1<'_, i64>,
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
) -> PyObject {
    let input_view = BacktrackInputView {
        cols: pi.shape()[1],
        leaf_species: slice_from_numpy(&leaf_species),
        split_leftrights: slice_from_numpy(&split_leftrights),
        pi: pi.as_slice().unwrap(),
        pibar: pibar.as_slice().unwrap(),
        e: slice_from_numpy(&e),
        ebar: slice_from_numpy(&ebar),
        log_p_s: slice_from_numpy(&log_p_s),
        log_p_d: slice_from_numpy(&log_p_d),
    };
    let species = SpeciesTopology {
        child1: slice_from_numpy(&sp_child1),
        child2: slice_from_numpy(&sp_child2),
        subtree_start: slice_from_numpy(&sp_subtree_start),
        subtree_end: slice_from_numpy(&sp_subtree_end),
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
    sample_nodes.into_py(py)
}

#[pymodule]
fn gpurec_backtrack(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sample_reconciliations_torch, module)?)?;
    Ok(())
}

# gpurec

`gpurec` is an experimental GPU-accelerated likelihood engine for
gene-tree/species-tree reconciliation under a DTL-style model.  It takes a
fixed species tree and a collection of gene-tree clade distributions, evaluates
the negative log-likelihood on CUDA/Triton kernels, and exposes the result as a
PyTorch module so reconciliation rates can be optimized with standard
optimizers.

The package is intended for large batches of gene families where the expensive
parts of reconciliation can be scheduled into wavefront kernels and streamed
through GPU memory.

## What It Computes

For each gene family, `gpurec` computes reconciliation likelihoods over clades
against a species tree.  The main public model returns a scalar negative
log-likelihood in bits.  Its parameters are log2 event-rate logits for the
underlying reconciliation events.

Parameter sharing is selected with `mode`:

- `global`: one set of rates is shared across all species and families.
- `specieswise`: one rate vector is learned per species.
- `genewise`: one rate vector is learned per gene family.

## How It Works

The high-level execution path is:

1. Parse and preprocess the species tree and gene-family inputs.
2. Build clade layouts and topological wave schedules with the Rust
   `gpurec-preprocess` crate.
3. Convert `theta` log2-rate logits into normalized log probabilities.
4. Solve the resident `E` fixed point with Triton kernels.
5. Propagate `Pi` and `Pibar` reconciliation quantities wave by wave.
6. Read root rows to produce the negative log-likelihood.
7. Compute gradients with an implicit adjoint solve using Neumann terms and
   BiCGSTAB refinement.

The Python layer owns the PyTorch module and optimization interface.  The Rust
crates own CPU preprocessing and backtracking helpers.  The CUDA-critical
recurrences live in Triton kernels under `gpurec/core/kernels`.

## Main Interface

```python
from gpurec import GeneReconModel, SolverOptions

model = GeneReconModel(
    species_tree="examples/tiny/species.nwk",
    gene_trees=["examples/tiny/gene.nwk"],
    mode="global",
    device="cuda",
    solver_options=SolverOptions(
        e_max_iter=2000,
        e_tol=1e-8,
        pi_iters=6,
        neumann_terms=3,
        bicgstab_max_iter=500,
    ),
)

loss = model()
loss.backward()
```

Important objects and helpers:

- `GeneReconModel`: PyTorch module that owns `theta`, batch schedules, warm
  starts, and the forward/backward likelihood path.
- `SolverOptions`: mutable runtime controls for fixed-point and implicit
  gradient solves.
- `model.configure_solver(...)`: updates solver knobs after model
  instantiation.
- `model.replan_batches(...)`: rebuilds batch schedules when chunking or memory
  limits should change.
- `model.clear_warm_starts()`: clears cached resident `E` warm starts.
- `project_rate_gradient_`: projects gradients at active rate bounds before an
  optimizer step.
- `clamp_log_rate_`: projects log-rate parameters back into feasible bounds
  after an optimizer step.
- `sample_reconciliations`: Rust-backed helper for sampling reconciliations
  from prepared inputs.

Example bounded optimization step:

```python
from gpurec import clamp_log_rate_, project_rate_gradient_

loss = model()
loss.backward()
project_rate_gradient_(model.theta, min_rate=1e-10, max_rate=1e-2)
optimizer.step()
clamp_log_rate_(model.theta, min_rate=1e-10, max_rate=1e-2)
optimizer.zero_grad(set_to_none=True)
```

## Solver Knobs

`SolverOptions` can be passed at construction time, assigned as a dictionary, or
changed in place while optimizing:

```python
model.configure_solver(pi_iters=10, neumann_terms=6, bicgstab_max_iter=1000)
model.solver_options.e_tol = 1e-10
```

The most important controls are:

- `e_max_iter`, `e_tol`, `e_init`: resident `E` fixed-point solve.
- `pi_iters`: number of Pi/Pibar wave iterations.  It must be an even integer
  at least 2.
- `neumann_terms`: number of Neumann-series terms used before BiCGSTAB.
- `bicgstab_max_iter`, `bicgstab_tol`, `bicgstab_breakdown_tol`: implicit
  adjoint linear-solve controls.
- `use_adjoint_pruning`, `adjoint_pruning_threshold`: skip low-signal adjoint
  rows in approximate gradient computations.
- `pibar_side_threshold`: threshold for side-term work in Pibar adjoints.

## Repository Map

- `gpurec/`: Python package, PyTorch model, optimizer helpers, scheduling
  wrappers, and Triton kernels.
- `gpurec/api/`: public model implementation and implicit-gradient bridge.
- `gpurec/core/`: lower-level inference, scheduling, parameter, kernel, and
  backtracking modules.
- `crates/gpurec-preprocess/`: Rust preprocessing crate for tree layouts,
  batches, and wave schedules.
- `crates/gpurec-backtrack/`: Rust/PyO3 backtracking and reconciliation sampling
  helper.
- `examples/tiny/`: minimal Newick/map/family inputs for smoke tests and small
  experiments.
- `notebooks/`: end-to-end optimization notebook.

## Requirements

The Python package requires Python 3.10-3.12, PyTorch, Triton, and NumPy.  The
main likelihood path expects a CUDA-capable PyTorch/Triton installation.  The
Rust crates are built separately when their native helpers are needed.

## Development Notes

This repository also contains external/reference projects used during
development.  The installable package is the `gpurec` Python package plus the
native crates under `crates/`.

Root-level `pytest` can collect tests from vendored/reference directories.  For
focused validation, run targeted checks against `gpurec` and the individual Rust
crates instead of assuming the repository root is a single isolated test suite.

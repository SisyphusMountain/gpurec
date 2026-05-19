# Lean Test Suite

This branch keeps focused tests for the retained performance path while avoiding
accidental traversal of large local fixtures under `tests/data`.

Fast CPU gate:

```bash
CUDA_VISIBLE_DEVICES='' pytest -q -m "unit and not gpu"
```

The GitHub Actions workflow in `.github/workflows/cpu-unit.yml` runs this same
gate on Python 3.10, 3.11, and 3.12 for pushes and pull requests.  It also
builds/checks distribution artifacts in a CPU packaging job.

The explicit equivalent is useful when bisecting a specific audit surface:

```bash
CUDA_VISIBLE_DEVICES='' pytest -q \
  tests/unit/test_batched_lbfgs.py \
  tests/unit/test_memory_policy.py \
  tests/unit/test_global_wave_scheduler.py \
  tests/unit/test_implicit_grad_solver.py \
  tests/unit/test_origination_probs.py \
  tests/unit/test_alerax_family_input.py \
  tests/unit/test_examples.py \
  tests/unit/test_release_metadata.py \
  tests/unit/test_workflow.py
```

Small CUDA smoke, when GPU memory allows:

```bash
pytest -q tests/kernels/test_wave_step_uniform_forward_kernel.py
pytest -q tests/integration/test_gene_recon_model.py::test_gene_recon_model_forward_backward_modes
```

Backtracking smoke should prefer a prebuilt Rust binary to avoid `cargo run`
startup during Python tests:

```bash
GPUREC_BACKTRACK_BIN=crates/gpurec-backtrack/target/release/gpurec-backtrack \
  pytest -q tests/integration/test_stochastic_backtracking.py::test_rust_stochastic_backtracking_exports_recphyloxml
```

`pytest.ini` declares the coarse test markers; `tests/conftest.py` auto-applies
`unit` and `integration` by directory, marks known CUDA modules as `gpu`, marks
selected expensive checks as `slow`, and excludes large data/output directories
from recursive collection.  Use explicit test paths for targeted audit gates
when a local checkout contains large generated datasets.

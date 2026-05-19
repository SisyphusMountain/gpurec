# Lean Test Suite

This branch keeps focused tests for the retained performance path while avoiding
accidental traversal of large local fixtures under `tests/data`.

Fast CPU gate:

```bash
CUDA_VISIBLE_DEVICES='' pytest -q \
  tests/unit/test_batched_lbfgs.py \
  tests/unit/test_memory_policy.py \
  tests/unit/test_global_wave_scheduler.py \
  tests/unit/test_implicit_grad_solver.py \
  tests/unit/test_origination_probs.py \
  tests/unit/test_alerax_family_input.py \
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

`pytest.ini` marks known CUDA modules as `gpu`, marks selected expensive checks
as `slow`, and excludes large data/output directories from recursive
collection.  Use explicit test paths for audit gates rather than bare `pytest`
when a local checkout contains large generated datasets.

# Lean Test Suite

This branch keeps focused tests for the retained performance path while avoiding
accidental traversal of large local fixtures under `tests/data`.

Fast CPU gate:

```bash
CUDA_VISIBLE_DEVICES='' pytest -q -m "unit and not gpu"
```

The GitHub Actions workflow in `.github/workflows/cpu-unit.yml` runs this same
gate on Python 3.10, 3.11, and 3.12 for pushes and pull requests.  It also
builds/checks distribution artifacts on Python 3.10 and 3.12 in a CPU
packaging job and runs the Rust backtracking crate plus JSON fixture gate.

The explicit equivalent is useful when bisecting a specific audit surface:

```bash
CUDA_VISIBLE_DEVICES='' pytest -q \
  tests/unit/test_batched_lbfgs.py \
  tests/unit/test_memory_policy.py \
  tests/unit/test_global_wave_scheduler.py \
  tests/unit/test_implicit_grad_solver.py \
  tests/unit/test_legacy_scripts.py \
  tests/unit/test_log2_utils.py \
  tests/unit/test_origination_probs.py \
  tests/unit/test_optimization_workflow.py \
  tests/unit/test_alerax_family_input.py \
  tests/unit/test_cli_workflow.py \
  tests/unit/test_core_backward.py \
  tests/unit/test_core_helpers.py \
  tests/unit/test_examples.py \
  tests/unit/test_family_layout.py \
  tests/unit/test_recphyloxml.py \
  tests/unit/test_release_metadata.py \
  tests/unit/test_repository_hygiene.py \
  tests/unit/test_species_helpers.py \
  tests/unit/test_terms.py \
  tests/unit/test_validation.py \
  tests/unit/test_workflow.py \
  tests/unit/test_workflow_artifacts.py
```

Small CUDA smoke, when GPU memory allows:

```bash
pytest -q tests/kernels/test_wave_step_uniform_forward_kernel.py
pytest -q tests/integration/test_gene_recon_model.py::test_gene_recon_model_forward_backward_modes
```

Rust backtracking checks are CPU-safe:

```bash
cargo test --locked --manifest-path crates/gpurec-backtrack/Cargo.toml
cargo run --locked --quiet --manifest-path crates/gpurec-backtrack/Cargo.toml -- --help
pytest -q tests/integration/test_rust_backtracking_fixture.py
```

Backtracking fixture smokes should use the CPU-only JSON fixture when the goal
is to validate the Rust binary without constructing a CUDA model:

```bash
pytest -q tests/integration/test_rust_backtracking_fixture.py::test_rust_backtracking_cli_reads_json_fixture_and_writes_recphyloxml
```

The checked fixture contracts live beside the fixtures:

- `tests/fixtures/backtracking/README.md` documents the CPU-only Rust JSON
  fixture and the RecPhyloXML output shape expected from the CLI smoke.
- `tests/data/test_trees_3/README.md` documents the smallest CUDA
  stochastic-backtracking fixture, including the expected 35 x 15 exported
  `pi` shape and one-tree sampling smoke.

Ignored local test data roots are intentionally outside the distributable fixture
contract:

| Surface | Current owner | Use | Decision before relying on it |
| --- | --- | --- | --- |
| `tests/data/test_trees_20/`, `tests/data/test_trees_100/`, `tests/data/test_trees_1000/`, `tests/data/test_trees_10000/` | Generated tree-scale fixtures. | Optional CUDA, profiling, and large-family regression runs. | Keep out of the CPU gate; add a small tracked fixture or documented generator before making a required test depend on them. |
| `tests/data/test_trees_dtl01/` | Local DTL experiment fixture plus generated output. | Legacy DTL/reference checks. | Migrate the reusable contract into a tracked fixture before using it in CI, otherwise treat it as local scratch. |
| `tests/data/HOGENOM/`, `tests/data/hogenom_bench/`, `tests/data/davin/` | External or checkout-local biological datasets. | HOGENOM notebooks, scripts, profiling, and validation runs. | Archive/delete local copies or migrate unique behavior into tracked fixtures before promoting any dependent workflow. |
| `tests/data.tar.gz` | Local transfer/archive artifact for generated data. | Convenience restore bundle. | Do not treat as source of truth; replace with a documented source or generator before any required workflow depends on it. |
| `.preprocess_cache/` and `tests/data/**/output/` | Runtime-generated cache and output trees. | Speedups and previous local run outputs. | Delete/regenerate as needed; never use them as expected fixtures. |

After installing the normal Python test dependencies, the broader non-GPU
integration marker can be collected with:

```bash
pytest -q -m "integration and not gpu"
```

`pytest.ini` declares the coarse test markers; `tests/conftest.py` auto-applies
`unit` to `tests/unit`, `integration` to `tests/integration`, and both
`integration` and `kernel` to `tests/kernels`.  It also excludes large
data/output directories from recursive collection.  GPU-only modules declare
`pytestmark = pytest.mark.gpu` at module scope, and expensive checks use local
`@pytest.mark.slow` decorators, so test intent stays beside the test instead of
a filename or nodeid list in conftest.
Some GPU-marked tests still live under `tests/unit` because they exercise a
single module or contract but require CUDA fixtures; use `-m "unit and not gpu"`
for CPU-only unit gates.
Use explicit test paths for targeted audit gates when a local checkout contains
large generated datasets.

## Adding Tests

Prefer public behavior assertions when a public API can exercise the contract.
Private helpers and partially initialized objects are acceptable for narrow
guardrail tests when full construction would require CUDA, external HOGENOM
data, or expensive preprocessing; keep those tests focused on one documented
contract and add the contract to the audit notes first.

Smoke tests should prove more than importability.  For optimization paths, use
loss decrease, reference-close values, or explicit failure/status assertions
when practical.  For docs and release hygiene, prefer parsing structured files
over long wording snapshots.

Rust checks belong in the CPU-safe gate when they validate the packaged
backtracking contract.  Use `cargo test --locked` for crate behavior and the
JSON fixture integration test when the goal is to validate the Python/Rust
boundary without constructing a CUDA model.

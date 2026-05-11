# Lean Test Suite

This branch keeps focused tests for the retained performance path:

- `tests/integration/test_gene_recon_model.py`
- `tests/integration/test_uniform_chunked_model.py`
- `tests/gradients/test_autograd_bridge.py`
- `tests/gradients/test_genewise_fused_backward.py`
- `tests/kernels/*uniform*`, DTS, and backward kernel tests
- `tests/unit/test_batched_lbfgs.py`
- `tests/unit/test_memory_policy.py`
- `tests/unit/test_genewise_wave.py`
- `tests/unit/test_specieswise_uniform.py`

Many scale tests skip unless the corresponding local datasets
(`test_trees_20`, `test_trees_100`, `test_trees_1000`) are present.  The
small tracked dataset `test_trees_3` is retained for smoke coverage.

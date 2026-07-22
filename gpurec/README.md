# gpurec Package

This package contains the Python-facing reconciliation model and the GPU
execution path.  It combines a PyTorch API, Rust preprocessing helpers, and
Triton kernels.

## Public Exports

`gpurec/__init__.py` re-exports the main user-facing interface:

- `GeneReconModel`: PyTorch module for evaluating reconciliation likelihoods.
- `SolverOptions`: mutable solver controls for fixed-point, Pi/Pibar, Neumann,
  and pruning behavior.
- `sample_reconciliations`: backtracking helper exposed from the Rust
  `gpurec-backtrack` extension.
- `log2_rate_bounds`: converts natural-rate bounds to log2 parameter bounds.
- `project_rate_gradient_`: in-place projected-gradient helper for bounded
  optimization.
- `clamp_log_rate_`: in-place parameter projection helper for bounded
  optimization.

## File And Folder Roles

- `__init__.py`: defines the package-level import surface.
- `optimization.py`: contains optimizer-agnostic projection helpers for
  log2-rate parameters and gradients.
- `api/`: high-level PyTorch model, solver options, and implicit-gradient entry
  points.
- `core/`: implementation modules for preprocessing wrappers, parameter
  extraction, forward inference, backtracking input, memory policy, and Triton
  kernels.

## Execution Boundary

The package keeps the stable interface in `gpurec.api` and leaves most
performance-sensitive details in `gpurec.core`.  User code should normally
import from `gpurec` rather than reaching directly into kernel or scheduling
modules, unless it is debugging or extending the implementation.

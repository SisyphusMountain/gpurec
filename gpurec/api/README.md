# gpurec.api

Public Python API for building, evaluating, and differentiating GPURec models.

## Files

- `model.py`: Defines `GeneReconModel`, the main `torch.nn.Module` wrapper. It preprocesses species and gene trees, builds batched wave layouts, owns the learnable `theta` parameters, solves resident `E` and `Pi` states, streams batched losses, and exposes solver reconfiguration and warm-start controls.
- `_implicit_grad.py`: Implements the custom implicit-gradient path used by `GeneReconModel.backward`. It propagates adjoints through wave dynamic-programming kernels, solves the fixed-point adjoint system with BiCGSTAB, and maps parameter-space sensitivities back to `theta`.
- `solver_options.py`: Defines the `SolverOptions` dataclass and validation rules for fixed-point, Pi iteration, implicit-gradient, and pruning controls.

Generated `__pycache__` files are interpreter artifacts and are not part of the source API.

# gpurec.api

Public Python API for building, evaluating, and differentiating GPURec models.

## Files

- `model.py`: Defines `GeneReconModel`, the main `torch.nn.Module` wrapper. It owns the public model API, learnable `theta` parameters, solver reconfiguration, warm-start controls, and delegates batch construction/execution to smaller helpers.
- `_batch_state.py`: Builds and stores per-batch static state, including wave layouts, rate-family indices, family index tensors, and warm-start storage.
- `_execution.py`: Streams batched losses and implicit gradients while mapping full-model `theta` tensors to per-batch tensors.
- `_autograd.py`: Defines the custom `torch.autograd.Function` bridge for single-batch and streamed full-model losses.
- `_implicit_grad.py`: Implements the custom implicit-gradient path used by `GeneReconModel.backward`. It propagates adjoints through wave dynamic-programming kernels, solves the fixed-point adjoint system with BiCGSTAB, and maps parameter-space sensitivities back to `theta`.
- `solver_options.py`: Defines the `SolverOptions` dataclass and validation rules for fixed-point, Pi iteration, implicit-gradient, and pruning controls. See [`../../docs/pi_storage.md`](../../docs/pi_storage.md) for the Pi storage execution contract.

Generated `__pycache__` files are interpreter artifacts and are not part of the source API.

CUDA Pi/Pibar state always uses centered residuals in the model dtype plus fp64
row offsets. The final likelihood head and streamed family/batch loss
reductions use fp64, including for fp32 models; parameter gradients are
returned in their configured dtype.

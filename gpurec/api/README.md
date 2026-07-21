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

CUDA Pi/Pibar state always uses centered residuals in the model dtype plus row
offsets in the configured accumulator dtype. `[precision].model_dtype` controls
parameters, dense E/Pi residual state, and dense-kernel wave metadata;
`[precision].accumulator_dtype` controls row offsets, the final likelihood
head, streamed family/batch reductions, small parameter softmaxes, and
accumulator-domain preprocessing statics.
Parameter gradients are returned in the model dtype.

The supported model/accumulator pairs are `float32/float32`,
`float32/float64`, and `float64/float64`; `float64/float32` is rejected. The
default is `float32/float64`. `GeneReconModel(dtype=...)` overrides only the
configured model dtype. `GeneReconModel(config=...)` always supplies the
accumulator dtype and supplies the model dtype when no explicit `dtype=` is
given.

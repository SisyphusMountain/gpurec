# gpurec.core.parameters

Parameter-shaping utilities that convert learnable optimization variables into log-probability tensors consumed by kernels and inference code.

## Files

- `extract_parameters.py`: Provides helpers for broadcasting global, specieswise, and genewise parameters to family/species shapes. Inference always normalizes receiver logits into base-2 log weights; equal logits give the uniform receiver measure.

Generated `__pycache__` files are interpreter artifacts and are not part of the source implementation.

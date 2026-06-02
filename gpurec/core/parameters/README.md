# gpurec.core.parameters

Parameter-shaping utilities that convert learnable optimization variables into log-probability tensors consumed by kernels and inference code.

## Files

- `extract_parameters.py`: Provides helpers for broadcasting global, specieswise, and genewise parameters to family/species shapes. `extract_parameters_uniform` turns unconstrained `theta` values into base-2 log probabilities for speciation, duplication, loss, and transfer terms, including transfer row-max normalization.

Generated `__pycache__` files are interpreter artifacts and are not part of the source implementation.

# gpurec.core.inference

Forward dynamic-programming routines for evaluating reconciliation likelihood state over scheduled gene-tree clade waves.

## Files

- `forward.py`: Implements `Pi_wave_forward`, which iterates over wave-layout metadata, computes duplication-transfer-speciation split reductions when needed, alternates `Pi` and `Pibar` buffers for a fixed number of Pi iterations, initializes leaf waves, applies wave-step kernels, stores final `Pibar` row maxima, and returns root rows plus full wave buffers.

Generated `__pycache__` files are interpreter artifacts and are not part of the source implementation.

# gpurec.core.backtracking

Backtracking support for sampling concrete reconciliations from a fitted model after forward probabilities have been computed.

## Files

- `input.py`: Loads the Rust native backtracking extension, extracts the selected family from the model's batch layout, recomputes resident `E` and wave probabilities, converts tensors to CPU NumPy inputs, selects family/species-specific parameter views, and calls `sample_reconciliations_torch`.

Generated `__pycache__` files are interpreter artifacts and are not part of the source implementation.

# gpurec.core

Core numerical and scheduling code for GPU reconciliation. This package turns
preprocessed family/species data into wave layouts, runs fixed-point inference,
and exposes helpers used by the native backtracking path.

## Subfolders

- `backtracking/`: Python adapter for the native reconciliation sampler.
- `inference/`: Forward inference orchestration over wave layouts and kernels.
- `kernels/`: Triton kernels for E-step updates, Pi/Pibar wave propagation, and
  retained backward passes.
- `parameters/`: Tensor shape normalization and parameter extraction helpers.
- `scheduling/`: Batch and wave-layout planning, including Rust-backed
  preprocessing entry points and Python fallback layout builders.

## Files

- `memory_policy.py`: Estimates CUDA memory budgets and scratch requirements for
  retained wave-backward kernels. Environment variables
  `GPUREC_MEMORY_POLICY_FRACTION` and `GPUREC_MEMORY_POLICY_RESERVE_GIB` tune the
  usable-memory gate.
- `backtracking/input.py`: Loads the native `gpurec_backtrack` extension,
  prepares model tensors as NumPy arrays, solves resident E/Pi values, and calls
  the native sampler for one family.
- `inference/forward.py`: Implements `pi_wave_forward`, the wave-by-wave forward
  pass that combines direct transfer/speciation splits with iterative Pi/Pibar
  self-loop updates.
- `inference/solver.py`: Runs the resident E fixed-point solve, Pi/Pibar wave
  propagation, and root-row negative log-likelihood shared by the API model and
  backtracking adapter.
- `inference/logspace.py`: Shared base-2 log-space tensor helpers.
- `parameters/extract_parameters.py`: Converts scalar, family-level, and
  family/species tensors into consistent matrix layouts and extracts normalized
  log probabilities from model parameters.
- `scheduling/batching.py`: Builds family batches and wave layouts, either by
  calling the native preprocessing extension or by constructing a Python layout
  from in-memory family dictionaries.

See `kernels/README.md` and `scheduling/README.md` for local file details in
those subfolders.

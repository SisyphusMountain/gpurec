# gpurec-backtrack Source

This folder contains the Rust implementation of the reconciliation sampler.
The crate is exposed to Python through PyO3 and consumes NumPy views prepared by
`gpurec.core.backtracking.input`.

## Files

- `lib.rs`: defines the PyO3 module, input views, species-topology helpers,
  seeded sampler, log-space weighted sampling, and event expansion for leaf,
  speciation, duplication, transfer, and loss nodes.

The sampler reads dynamic-programming tensors computed by the Python/Triton
forward path.  It does not recompute the likelihood; it stochastically
backtracks through already-computed `Pi`, `Pibar`, `E`, and event-probability
arrays.

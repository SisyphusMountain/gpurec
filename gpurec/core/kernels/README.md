# gpurec.core.kernels

Triton GPU kernels and Python launch wrappers used by core inference and
backward propagation. These modules assume tensors are already laid out by the
scheduling layer and generally operate on wave rows by species columns.

## Files

- `_dts_layout_contract.py`: CPU-testable parameter layout contract for the
  retained DTS backward kernels.
- `pi_forward.py`: Canonical row-gauged forward kernels. It handles leaf
  initialization, Pi/Pibar wave propagation, and single- or multi-split DTS
  reduction using model-dtype residuals plus fp64 row offsets.
- `e_step.py`: E fixed-point update kernels with autograd support. It computes
  E, child-indexed E terms, Ebar, convergence diffs, and backward gradients for
  species-tree parameters.
- `wave_backward.py`: Retained wave-backward fast path. It provides active-row
  pruning, Neumann self-loop solves, parameter-gradient accumulation, DTS cross
  backward accumulation, and compact-tree Pibar VJP correction helpers.

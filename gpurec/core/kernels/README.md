# gpurec.core.kernels

Triton GPU kernels and Python launch wrappers used by core inference and
backward propagation. These modules assume tensors are already laid out by the
scheduling layer and generally operate on wave rows by species columns.

## Files

- `_dts_layout_contract.py`: CPU-testable parameter layout contract for the
  retained DTS backward kernels.
- `pi_forward.py`: Canonical row-gauged forward kernels. It handles leaf
  initialization, Pi/Pibar wave propagation, and single- or multi-split DTS
  reduction using model-dtype residuals plus configured-accumulator row
  offsets.
- `e_step.py`: E fixed-point update kernels with autograd support. It computes
  E, child-indexed E terms, Ebar, convergence diffs, and backward gradients for
  species-tree parameters.
- `wave_backward.py`: Retained wave-backward fast path. It provides active-row
  pruning, Neumann self-loop solves, parameter-gradient accumulation, DTS cross
  backward accumulation, and compact-tree Pibar VJP correction helpers.

Centered kernel wrappers derive accumulator arithmetic from the offset tensor
dtype; kernels do not choose fp64 as a local precision policy. Supported
model/accumulator pairs are fp32/fp32, fp32/fp64, and fp64/fp64. Fp64/fp32 is
rejected before launch because offsets and small reductions may not be narrower
than model state.

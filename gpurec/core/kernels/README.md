# gpurec.core.kernels

Triton GPU kernels and Python launch wrappers used by core inference and
backward propagation. These modules assume tensors are already laid out by the
scheduling layer and generally operate on wave rows by species columns.

## Files

- `_dts_layout_contract.py`: CPU-testable layout contract for direct transfer
  and speciation (DTS) parameters. It documents and classifies scalar,
  species-shared, family-scalar, and family/species tensor layouts for forward
  and retained backward kernels.
- `dts_fused.py`: Forward DTS reduction kernels. It combines split child Pi and
  Pibar rows into parent contributions, with separate paths for single-split
  parents and multi-split parents.
- `e_step.py`: E fixed-point update kernels with autograd support. It computes
  E, child-indexed E terms, Ebar, convergence diffs, and backward gradients for
  species-tree parameters.
- `wave_step.py`: Forward wave kernels for Pi/Pibar propagation. It handles leaf
  initialization, row log-sum-exp/Pibar computation, optional DTS contributions,
  and final Pibar materialization.
- `wave_backward.py`: Retained wave-backward fast path. It provides active-row
  pruning, Neumann self-loop solves, parameter-gradient accumulation, DTS cross
  backward accumulation, and compact-tree Pibar VJP correction helpers.

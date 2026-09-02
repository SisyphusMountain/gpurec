# gpurec.core.kernels

Triton GPU kernels and Python launch wrappers used by core inference and
backward propagation. These modules assume tensors are already laid out by the
scheduling layer and generally operate on wave rows by species columns.

The canonical equations and a complete device-kernel index are in
[`docs/latex/kernel_mathematics.tex`](../../../docs/latex/kernel_mathematics.tex).
Python docstrings are intentionally limited to interface and launch behavior.

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
- `dts_tangent.py`: Directional derivative of the five-event duplication,
  transfer, and speciation log-sum-exp recurrence.
- `dts_so.py`: Directional derivative of the DTS adjoint, including the
  complementary-subtree transfer term used by exact Hessian-vector products.
- `e_step_tangent.py` and `e_step_so.py`: First- and second-order derivatives
  of the extinction-probability fixed point.
- `wave_tangent.py` and `wave_so.py`: First- and second-order derivatives of
  the within-wave reconciliation recurrence.

Centered kernel wrappers derive accumulator arithmetic from the offset tensor
dtype; kernels do not choose fp64 as a local precision policy. Supported
model/accumulator pairs are fp32/fp32, fp32/fp64, and fp64/fp64. Fp64/fp32 is
rejected before launch because offsets and small reductions may not be narrower
than model state.

## Why some kernels pass `do_not_specialize` to `@triton.jit`

Triton keys its compiled-kernel cache on more than the kernel source and its
`tl.constexpr` values. For every ordinary integer argument it also records
whether the value is exactly 1 (which turns it into a compile-time constant) and
whether it is divisible by 16; for every pointer argument it records whether the
address is 16-byte aligned. A new combination is a new just-in-time compile of
otherwise identical code, taking seconds each and leaving another directory in
`TRITON_CACHE_DIR`.

Wave loops hit both rules on every launch: `ws`, `W`, `n_ws`, `family_offset`,
`split_offset` and the tile counts change with each wave, and a wave's slice of a
`[clades, species]` buffer (`accumulated_rhs[ws : ws + W]`, the right half of the
stacked donor-adjoint staging buffer) starts at a byte offset that is only
sometimes a multiple of 16 because the species count is odd. Measured on one
100-family forward + gradient + analytic Hessian with an empty cache, that cost
93 compiles where 35 distinct kernel configurations exist.

The kernels affected list those arguments in `do_not_specialize` (integers) or
`do_not_specialize_on_alignment` (sliced pointers), which pins one compile per
genuine configuration. Only arguments that were measured to vary are listed:
alignment specialization is what lets Triton emit vectorized loads, so it is
dropped only where it was already being lost half the time. `tl.constexpr`
arguments stay compile-time -- they select which code is generated.

Species-tree node ids and split-row ids are discrete layout metadata, not
numerical model precision. Scheduling stores them as int32 to reduce memory
traffic. Parent-chain kernels explicitly keep their loop-carried node id int32
because Triton requires a stable loop type even when a direct caller supplies
an int64 tensor. Values used in flattened row addresses are instead widened to
int64 locally so products such as ``row * species_count`` cannot overflow.

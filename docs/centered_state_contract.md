# Centered log-state contract

This document is the storage and API contract for centered Pi-family state. It
is normative for kernels, inference orchestration, diagnostics, and derivative
code. A buffer is not centered merely because its values happen to be small;
the residual buffer and its offset buffer are one logical value.

## Representation

For every global clade row `c` and species lane `s`,

```text
Pi_abs[c, s]    = Pi_residual[c, s]    + Pi_offset[c]
Pibar_abs[c, s] = Pibar_residual[c, s] + Pibar_offset[c]
```

`Pi_residual` and `Pibar_residual` are contiguous fp32 CUDA matrices with shape
`[C, S]`. Their offsets are contiguous fp64 CUDA vectors with shape `[C]` on
the same device. A temporary split reduction has the same contract with shape
`[W, S]` and an fp64 offset vector of shape `[W]`; its owner is the wave that
created it and its row index is wave-local.

Offsets are gauges, not model variables. Choosing a different finite offset and
applying the opposite shift to every finite residual lane represents exactly the
same state and must not change loss or derivatives. Production wave iterations
therefore use a cheap heuristic gauge; consumers must not assume that the row
maximum is zero. Leaf seeding uses an exact maximum. The first split-wave
consumer may fold its already-computed raw residual maximum into a temporary
fp64 virtual DTS offset. Storage remains in its original gauge; the consumer
combines the virtual shift with its existing fp64 correction. This avoids
another launch, row-maximum pass, or mutable input.

## Ownership and lifetime

- `pi_wave_forward_centered` allocates and owns the global Pi/Pibar residual and
  offset buffers for one solve.
- `CenteredPiForwardState` carries the two global offset vectors alongside the
  residual matrices returned through the existing solver tuple. Autograd must
  save the offsets belonging to its own forward; reading a later value from a
  mutable batch static is unsafe.
- `compute_dts_forward_centered` allocates a wave-local residual and offset pair.
  The pair must remain together until the consuming wave step or derivative is
  complete.
- The first DTS-consuming iteration publishes one temporary fp64 virtual offset
  per wave-local row. Later iterations use it for base selection while reading
  the original residual/offset pair. The virtual offset is not published
  forward state, and derivative consumers recompute their own DTS state.
- Root rows are an intentional reconstruction point. They are reconstructed in
  fp64 before the likelihood reduction so the scalar objective retains the
  offset precision.
- The fp64 absolute backward adapter is an intentional correctness reference,
  not a production storage path. Native consumers should operate in a local
  frame by applying offset differences before fp32 arithmetic.

## Metadata frames

`pibar_row_max[c]` is historical naming. It contains the (receiver-weighted when
enabled) maximum of the final **Pi residual** row used to construct Pibar. It is
therefore in the Pi residual frame:

```text
Pi_row_max_abs[c] = pibar_row_max[c] + Pi_offset[c]
```

It must never be reconstructed with `Pibar_offset`. Any new row metadata must
state its frame in its name or docstring.

## Consumer rule

A centered consumer must do exactly one of the following:

1. stay in a local target frame and add fp64 offset differences before casting
   the small correction to the compute dtype; or
2. reconstruct the complete input in fp64 at a named reference boundary.

Reconstructing a large absolute row and casting it to fp32 is invalid because it
reintroduces the quantization that centered storage is designed to avoid.

For a wave row whose target frame is `Pi_offset[parent]`, the important source
corrections are:

| source term | correction before local fp32 log-sum-exp |
|---|---:|
| Pi from the same row | `0` |
| Pibar from the same row | `Pibar_offset - Pi_offset` |
| leaf observation | `-Pi_offset` |
| split DTS | `DTS_offset - Pi_offset` |
| DTS `Pi_l + Pi_r` term vs parent | `Pi_offset[l] + Pi_offset[r] - Pi_offset[parent]` |
| DTS `Pi_l + Pibar_r` term vs parent | `Pi_offset[l] + Pibar_offset[r] - Pi_offset[parent]` |

The analogous correction applies to the right/Pibar-left term. Row-local
softmaxes, receiver-weighted Pi sums, and ancestor exclusions need no correction
because a common Pi offset cancels. Pibar inverse-denominator expressions that
mix Pi row-max metadata and Pibar require `Pi_offset - Pibar_offset`.

## Non-finite and inactive rows

- An impossible species lane is stored as residual `-inf`; adding a finite
  offset must leave it `-inf`.
- The canonical empty or wholly inactive row is an all-`-inf` residual row with
  offset `0.0`. Producers must initialize both parts even when pruning skips a
  row.
- This canonical rule applies to published `CenteredPiState` and DTS state.
  Ping-pong rows from non-final fixed-point iterations are internal scratch and
  may temporarily carry any finite gauge; the final iteration canonicalizes
  them before the sidecar becomes visible to consumers.
- A finite lane requires a finite offset. NaN offsets and `+inf` offsets are
  invalid state and must not be silently clamped by reconstruction helpers.
- Reductions over an all-`-inf` row return `-inf`, never NaN. Safe temporary
  maxima may use zero only behind a mask that restores this result.

## Dtypes, gradients, and public support

The production centered representation currently targets fp32 residuals plus
fp64 offsets on CUDA. Parameter/E tensors retain their configured dtype. Offset
selection is bookkeeping and is not differentiated; derivatives are derivatives
of the represented absolute recurrence. Returned gradients must match the model
parameter dtype.

`SolverOptions.pi_representation` is the only supported selection mechanism.
The private environment experiment is deliberately not part of this contract.
An operation may advertise centered support only after its uniform/weighted,
split fanout, nonzero-leaf, and finite/non-finite parity gates pass.

## Consumer inventory

| layer | files | centered obligation |
|---|---|---|
| forward Pi/Pibar | `core/kernels/centered_pi_forward.py`, `core/inference/forward.py` | native residual+offset producer; fp64 root reconstruction |
| split reduction | `core/kernels/centered_pi_forward.py` | native DTS residual+offset producer, including one and multiple splits |
| likelihood/streaming | `core/inference/solver.py`, `api/_execution.py`, `api/_autograd.py` | preserve fp64 reconstructed scalar across batches and save the matching offsets |
| first-order adjoint | `core/kernels/wave_backward*.py`, `api/_implicit_grad.py` | offset-aware wave and cross-DTS ratios; fp64 reconstruction adapter remains the oracle |
| convergence diagnostics | `core/inference/solver.py`, `api/_execution.py` | compare represented absolute iterates or frame-aligned residuals |
| backtracking | `core/backtracking/*` | never serialize or replay a residual without its offsets |
| JVP/tangent | `core/kernels/wave_tangent.py`, `dts_tangent.py`, `solver/forward_tangent.py` | use offset-aware primal weights; tangent values themselves are gauge-invariant |
| second order/HVP | `core/kernels/wave_so.py`, `dts_so.py`, `solver/hvp_exact.py` | use the same offset-aware primal weights as first order |
| curvature consumers | `solver/*curvature.py`, `solver/ggn.py` | require a centered-capable tangent/HVP provider or fail before reading state |

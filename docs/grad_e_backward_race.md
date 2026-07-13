# `grad_E` Backward Race and Centered Precision Costs

This note documents the `grad_E` backward race fixed in
`gpurec/core/kernels/e_step.py`, and why the centered fp64 offsets and fp64
outer accumulations are not expected to be the dominant runtime cost.

## Problem

`_e_step_backward_prepare_2d_kernel` computes the local E-step adjoint for one
family row. Within that Triton program, each species lane first writes the
species-local contribution:

```python
tl.store(grad_E_ptr + base + offs, 2.0 * q1 + q2, mask=mask)
```

Then each lane adds parent-to-child contributions into the child species slots:

```python
tl.atomic_add(grad_E_ptr + base + c1, q0 + g_s1_out, sem="relaxed", mask=c1_valid)
tl.atomic_add(grad_E_ptr + base + c2, q0 + g_s2_out, sem="relaxed", mask=c2_valid)
```

The intended value is:

```text
grad_E[species] =
    local contribution for species
  + sum(parent contributions whose child is species)
```

The old ordering allowed this failure mode:

1. Lane `p` atomically adds a parent contribution into child slot `c`.
2. Lane `c` later performs its own initializing `tl.store` to `grad_E[c]`.
3. That store overwrites the atomic contribution from lane `p`.

This is a lost-update race. It is not a mathematical precision issue and Kahan
compensation cannot fix it, because the contribution is sometimes discarded
before it can participate in any sum.

## Fix

The fix is a CTA-level barrier after every lane stores its local initializer and
before any lane performs child atomics:

```python
tl.store(grad_E_ptr + base + offs, 2.0 * q1 + q2, mask=mask)
tl.debug_barrier()

tl.atomic_add(grad_E_ptr + base + c1, q0 + g_s1_out, sem="relaxed", mask=c1_valid)
tl.atomic_add(grad_E_ptr + base + c2, q0 + g_s2_out, sem="relaxed", mask=c2_valid)
```

After the barrier, all child atomics add into initialized slots instead of racing
against the initialization stores.

This is the narrowest fix because it preserves the existing one-kernel layout.
The alternatives would be heavier: split initialization and child scattering into
separate kernels, or turn more of the local initialization path into atomics.

## Effect on Gradient Stability

Before the barrier, repeated evaluations could differ because some child
contributions were occasionally lost. That made finite-difference gradients and
line-search decisions harder to interpret: the observed noise was a mixture of
actual numerical precision limits and a memory-ordering bug.

After the barrier, the remaining repeat-to-repeat variation in the centered
Archaea diagnostic runs is much smaller than the current residual gradient. That
means the remaining `~1e-2` to `~1e-1` gradient scale is not explained by this
`grad_E` race.

## Cost of fp64 Offsets

The centered Pi path does not convert the full Pi/Pibar matrices to fp64. The
bulk arrays remain fp32 with shape roughly:

```text
num_clade_rows * num_species
```

The fp64 state is a small side channel:

```text
pi_offset:    num_clade_rows
pibar_offset: num_clade_rows
dts_offset:   per active split wave when needed
```

So the extra persistent storage for Pi/Pibar offsets is two fp64 vectors, not two
fp64 matrices. For a run with `S = 119` species, that is about:

```text
2 * 8 bytes per clade row = 16 bytes per clade row
```

compared with:

```text
2 * 119 * 4 bytes per clade row = 952 bytes per clade row
```

for the fp32 Pi/Pibar matrix storage. The offset vectors are therefore small in
memory bandwidth and footprint relative to the fp32 matrix work.

There is some fp64 arithmetic in the centered kernels: row offsets are loaded,
combined, and then offset differences are converted back to the kernel dtype.
That is materially cheaper than running the full kernel algebra in fp64 because
the species-wise dense work stays fp32.

## Cost of fp64 Accumulations

The fp64 outer accumulations are also small relative to the kernels:

- Batch losses are scalar fp64 additions.
- Full-gradient accumulation is over the model parameter tensor, not over every
  dynamic-programming intermediate.
- The diagnostic block-Newton scripts build Jacobians and solve small dense
  systems in fp64, but that is optimization-driver cost, not production kernel
  cost.

The expensive part of the Archaea runs remains the repeated forward/backward
dynamic programming over all families and waves. The fp64 scalar/vector
accounting is there to avoid loss quantization and gradient cancellation error;
it should not be the main runtime driver unless the kernel work has already been
made very small.

The centered path can still cost extra runtime for other reasons:

- it may take additional centered kernel variants;
- the centered retained backward currently uses the Neumann self-loop path;
- the diagnostic optimizer intentionally performs many repeated evaluations.

Those costs are distinct from the fp64 offsets themselves.

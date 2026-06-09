# Centered Pi Heuristic Offsets

The centered Pi representation does not need each stored row to be centered at
the exact row maximum. It only needs to avoid storing values such as
`1500 + small_residual`, where fp32 loses the small residual. An offset that
keeps row-local values in a moderate range is enough.

## Problem With Exact Max Recentering

The current exact-centered prototype computes an output row max before storing
`Pi_centered = Pi_abs - Pi_offset`. In the fused forward kernels this forces an
extra pass over the species row:

1. evaluate the wave-step terms to discover the row max;
2. evaluate the same terms again to store centered output values.

For a four-step fused wave kernel, this turns four wave evaluations into roughly
five evaluations. That explains why the fused centered forward still has about
`1.25x` structural overhead even after reusing pair/quad fusion.

## Better Invariant

The useful invariant is not:

```text
max_s Pi_centered[row, s] = 0
```

The useful invariant is:

```text
Pi_centered[row, s] stays far from the large absolute offset scale.
```

Values in a range such as `[-100, 100]` or even wider are still much better than
values around `1500` when the gradient depends on small row-local differences.

## Heuristic Offset Rule

A cheap first implementation should pick the output offset from available scalar
offsets, without a preliminary max pass:

```text
Pi_out_offset ~= max(
    Pi_in_offset,
    DTS_offset if the wave has splits,
    0 if the wave has a leaf term
)
```

The kernel then computes each output value directly in that frame:

```text
Pi_out_centered[s] = logsumexp_terms_shifted_by(Pi_out_offset)
```

This preserves the cancellation fix because all large terms are still shifted
before conversion to fp32 row-local values. The offset is just no longer
guaranteed to place the largest output species at zero.

## Drift Control

If heuristic offsets drift, we can add cheap guards before returning to exact
per-row maxima:

1. Track min/max or absolute range diagnostics on selected hard families.
2. Add a periodic exact recenter every `k` Pi iterations if the range grows too
   large.
3. Use a conservative scalar correction from known family constants if the simple
   `max(input offsets)` rule is too low.
4. Keep exact recentering for final `Pibar` construction only if needed by the
   backward probability ratios.

## Expected Performance Impact

Removing the exact output-max pass should bring centered fused wave kernels close
to the current absolute fused kernels. The remaining overhead should mostly be:

```text
fp64 offset loads/stores
offset correction arithmetic
centered DTS offset bookkeeping
```

Those are small compared with an extra full species-row evaluation pass, so this
is the next implementation direction for a performance-preserving centered Pi
path.

# Family 2595 fp32 Error Source Report

Date: 2026-06-07

Family `2595` is `CLU_000688_21_14_C`.

## Summary

The remaining `~2.1e-4` relative gradient error after correcting `Pi/Pibar`
does not come from GMRES, Neumann convergence, or uncentered `Pi/Pibar`
storage.

The dominant isolated source is the fp32 theta-derived scalar `log_pS`, with
the fp32 `E` state also contributing. These small forward-state errors enter
backward source-weight computations and are amplified by the highly
cancellation-sensitive theta projection for this family.

## Key Diagnostic

The relevant lower-bound row is:

```text
prod_state_with_ref_pi_pibar_two_fp32_residual
rel L2 = 2.102e-4
abs L2 = 2.399e-5
```

This row uses the current fp32 production state, replaces only `Pi/Pibar` with
near-exact two-fp32 reference values, and then runs fp64 backward. The residual
therefore measures error that remains after `Pi/Pibar` has effectively been
removed as the cause.

Artifact:

```text
benchmarks/large_dataset_capacity/output/hogenom_centered_production_forward_20260607/error_source_diagnostic_2595_all_state_floor_bounds.json
```

## Field-Level Isolation

The following read-only diagnostic substituted one fp32 production field at a
time into the fp64 reference state and ran fp64 backward. This isolates forward
state precision error from backward arithmetic.

| Substituted fp32 field | rel L2 gradient error | abs L2 gradient error |
|---|---:|---:|
| `log_pS` | `1.793e-4` | `2.046e-5` |
| `E` | `1.161e-4` | `1.325e-5` |
| `log_pL` | `4.837e-5` | `5.519e-6` |
| `theta` | `1.576e-5` | `1.799e-6` |
| `max_transfer` | `1.491e-5` | `1.701e-6` |
| `E_s1` | `1.546e-5` | `1.764e-6` |
| `E_s2` | `1.085e-5` | `1.238e-6` |
| `Ebar` | `5.948e-6` | `6.787e-7` |
| `log_pD` | `2.255e-6` | `2.573e-7` |
| `receiver_log_probs` | `1.069e-12` | `1.220e-13` |

Bundle checks from the saved diagnostic agree with this:

| Substituted fp32 bundle | rel L2 gradient error |
|---|---:|
| `prod_param_bundle_only_abs_fp64_backward` | `2.180e-4` |
| `prod_E_bundle_only_abs_fp64_backward` | `9.893e-5` |
| `prod_E_plus_Pi_bundle_abs_fp64_backward` | `2.551e-4` |

The param bundle here means:

```text
log_pS
log_pD
log_pL
max_transfer
receiver_log_probs
theta
receiver_weights
```

## Exact `log_pS` Difference

The isolated `log_pS` error is caused by a tiny fp32 rounding difference:

```text
fp32 log_pS = -1.0489670038223267
fp64 log_pS = -1.0489671346297689
delta       =  1.308074422e-7
```

This is normal fp32-scale rounding. It becomes large in the final gradient only
because family `2595` has a very small final theta-gradient component and severe
projection cancellation.

## Where The Error Enters

`log_pS` enters the E/backward source weights through the S-event terms:

```text
SL1 = log_pS + E_s2
SL2 = log_pS + E_s1
```

Code:

```text
gpurec/api/_implicit_grad.py
```

It also enters DTS branch weights:

```text
d3 = log_pS + Pi_l_s1 + Pi_r_s2 + ...
d4 = log_pS + Pi_r_s1 + Pi_l_s2 + ...
```

Code:

```text
gpurec/core/kernels/wave_backward.py
```

These weights produce `grad_log_pS` and related source terms. The final theta
projection then applies:

```text
theta_grad_i = adj_i - p_i * sum(adj)
```

For `2595`, that projection has extreme cancellation factors. In the corrected
diagnostic, the cancellation factors were approximately:

| Component | cancellation factor |
|---|---:|
| `D` | `7.0e3` |
| `L` | `9.0e3` |
| `T` | `5.8e5` |

Thus a `~1e-7` error in a scalar log-parameter can become a `~1e-4` relative
error in the final theta gradient.

## What It Is Not

The current evidence rules out these as primary causes of the `2.1e-4`
residual:

| Candidate | Evidence |
|---|---|
| GMRES | The diagnostic uses Neumann and fp64 backward substitutions. |
| Neumann convergence | `E/Pi=256` and Neumann substitution checks isolate representation error. |
| raw `Pi/Pibar` storage | The `2.1e-4` residual remains after replacing `Pi/Pibar` with near-exact values. |
| `pibar_row_max` | Isolated substitution error is `~2.5e-12`. |
| `receiver_log_probs` | Isolated substitution error is `~1e-12`. |
| fast math exp/log | Earlier libdevice exp/log tests did not move the floor. |

## Implication

For family `2595`, improving `Pi/Pibar` alone cannot reach a `<1e-6` final
relative theta-gradient target.

The next precision fix should target higher-effective precision for:

1. theta-derived log parameters, especially `log_pS`;
2. `E` and related E-state values consumed by backward;
3. the backward source-weight path that combines these with centered
   `Pi/Pibar`.

The two-fp32 floor diagnostic shows that higher-effective fp32 representation is
sufficient in principle:

| Representation | rel L2 gradient error |
|---|---:|
| all exact ref state rounded to single fp32 | `1.147e-3` |
| all exact ref state reconstructed from two fp32 values | `1.451e-11` |
| exact centered `Pi/Pibar` as single fp32 local plus offset | `1.086e-5` |
| exact centered `Pi/Pibar` reconstructed from two fp32 values | `3.638e-12` |

So the clean direction is not more GMRES work. It is a two-fp32 or equivalent
higher-effective-precision representation for the sensitive parameter/E/source
state that backward consumes.

# Forward fp32/fp64 precision audit

## Scope and method

This historical component audit ran the same resident HOGENOM family through
absolute fp64, absolute fp32, and centered fp32 diagnostic implementations.
Production CUDA execution now always uses centered residuals plus fp64 row
offsets for both fp32 and fp64 models. The diagnostic wraps Python launch
boundaries and copies outputs before ping-pong buffers are reused. The capture
includes parameter extraction, every extinction iteration, every Pi wave
iteration, split-DTS outputs, final Pi/Pibar state, root rows, and the
likelihood-head decomposition. Centered residuals are reconstructed with their
fp64 offsets before comparison.

The historical weighted audit command is shown below. Its comparison driver
was removed with the alternate absolute implementation and remains available
from Git history at commit `f6d2ac37`.

```bash
GPUREC_HOGENOM_ROOT=/path/to/hogenom \
  .venv/bin/python benchmark/diagnose_forward_precision.py \
  --families 1 --weighted --e-max-iter 128 --e-tol 1e-8 \
  --pi-iters 64 \
  --output output/forward_precision_diagnostic_family1_weighted_final.json
```

This records 2,007 matched tensors for each comparison, spanning 18 extinction
iterations and 64 iterations for every Pi wave. The centered variant records 52
additional native residual/virtual-offset tensors around DTS centering.
There were no missing shared capture keys, finite-mask mismatches, or
negative-infinity-mask mismatches. Passing
`--tensor-output output/forward_precision_tensors.pt` additionally saves the
raw CPU captures; the JSON summary is normally sufficient and is much smaller.

## What caused the error

The component comparison and mixed-input controls separate three effects:

| component | fp32 maximum error | observed effect |
|---|---:|---|
| extracted transfer/rate constants | approximately `4.3e-7` | small upstream contribution |
| final extinction `E` | `3.255e-7` | substituting fp32 E into the fp64 head changes loss by `1.53e-8` |
| final `Ebar` | `7.749e-7` | still too small to explain the result |
| absolute split DTS | grows from `4.916e-6` to `1.083e-4` across waves | dominant accumulated error |
| absolute final root row | `1.208e-4` | dominant input to the likelihood head |
| fp32 likelihood reduction | about `1.1e-5` on the unweighted probe | cheap independent source |

A causal hybrid run recomputed Pi in fp64 while keeping all upstream
parameters and E rounded to fp32. Its loss error was approximately `3.4e-6`,
compared with `6.2e-5` for the absolute fp32 solve. Replacing only fp32 E in an
otherwise fp64 solve changed loss by less than `1e-6` in the unweighted probe
and by `1.53e-8` in the weighted trace. The main problem is therefore fp32
Pi/DTS propagation at large negative absolute log values, not extinction
convergence.

The centered implementation initially failed to realize that benefit for two
specific framing reasons:

1. A later leaf-wave iteration treated the leaf source as if it lived in a
   zero offset frame. Negative leaf observations therefore pulled a correctly
   centered row back toward the absolute frame.
2. Split DTS used a mathematically valid fp64 offset but retained a large
   common negative value in its fp32 residual. The consuming recurrence then
   combined poorly framed DTS and Pi terms and lost the low bits again.

## Applied low-cost fixes

1. The likelihood head now promotes only root rows, E, and optional
   origination tensors to fp64. Streamed scalar and genewise reductions also
   retain fp64. Its small root and survival adjoint seeds are evaluated in the
   same fp64 head and rounded once to the configured state dtype. Dense
   `[clades, species]` Pi/Pibar storage remains fp32.
2. Small rate, receiver, and origination log-softmax operations are evaluated
   in fp64 for fp32 inputs and rounded once back to fp32. Nonuniform
   origination weights remain fp64 through the likelihood head so the direct
   gradient differentiates the same function returned by the forward pass.
3. Leaf terms now use the actual observation log-probability as their frame.
4. A split wave computes a raw-residual shift in its first existing consuming
   traversal and publishes a one-fp64-scalar-per-row virtual centered offset.
   DTS storage remains immutable. The shift is folded algebraically into the
   existing fp64 correction, so there is no additional launch, maximum pass,
   or per-species operation.

The remaining error is ordinary fp32 arithmetic inside the hot Pi recurrence
and its approximate `exp2`, rather than an avoidable global frame loss.

## Accuracy and cost gates

The one-family fp64 comparison uses 413 clades and 1,331 species nodes. The
full HOGENOM timing uses 1,055 families, 1,036,963 clades, five streamed
batches, three warmups, and 12 alternating samples.

| weighted forward quantity | absolute fp32 | centered fp32 | fp64 reference |
|---|---:|---:|---:|
| production loss | `323.435630504` | `323.435583954` | `323.435577046` |
| loss absolute error | `5.346e-5` | `6.908e-6` | 0 |
| root maximum error | `1.209e-4` | `8.156e-6` | 0 |
| root L2 error | `1.812e-3` | `2.300e-4` | 0 |
| full Pi maximum error | `1.364e-4` | `1.044e-5` | 0 |

On the unweighted forward/gradient oracle, loss error falls from `6.199e-5`
to `9.096e-6`, gradient relative L2 error from `8.478e-6` to `7.941e-7`,
and gradient infinity error from `8.696e-5` to `5.288e-6`.

On the full workload, 12 alternating warmed samples give `+2.56%` median
centered forward overhead (absolute `2,822.64 ms`, centered `2,880.39 ms`) and
`+0.71%` median loss+gradient overhead (absolute `6,959.41 ms`, centered
`7,030.91 ms`). Both variants return an identical scalar loss in every repeat.
Centered peak allocation remains approximately `+0.53%` for forward and
`+0.30%` for loss+gradient.

The full fp64 HOGENOM solve was not attempted on the 24 GiB benchmark GPU
because it lacked an adequate memory margin. CUDA fp64 remains supported by the
same centered-state path; the one-family fp64 run is the accuracy oracle and
the full fp32 run is the memory and performance gate.

## Deliberately rejected changes

- Moving the extinction solve to fp64 is not justified by the substitution
  measurements and would enlarge a dense hot state.
- Moving an fp32 model's whole Pi wave reductions to fp64 would address the
  remaining error, but it is not a low-cost fix on this workload. A model
  configured as fp64 instead uses fp64 centered residuals throughout.
- Computing CCP split log-probabilities or every row maximum in fp64 offers a
  much smaller possible gain and touches derivative contracts. It is deferred
  unless a harder weighted/specieswise trace identifies it as dominant.

For fp32 models, the result is a targeted mixed-precision design: fp64 for
small reductions and row gauges, fp32 for dense storage and hot kernels, with
explicit trace data showing where that boundary is useful. Fp64 models retain
the same centered contract with fp64 dense residuals.

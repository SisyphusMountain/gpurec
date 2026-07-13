# Centered-kernel HOGENOM report

## Result

Centered Pi/Pibar storage is the CUDA execution path for fp32 and fp64 models,
including forward, first-order, tangent, second-order, exact-HVP, and GGN
computations. A component-level fp32/fp64 audit found
that absolute fp32 Pi/DTS propagation, rather than the extinction fixed point,
was the dominant forward-error source. Fixing the centered leaf and split-DTS
frames makes centered fp32 materially more accurate on the one-family fp64
oracle:

This is a historical benchmark report. Its centered fp32 runs used
`model_dtype="float32"` with `accumulator_dtype="float64"`, and the fp64 oracle
used `float64/float64`. Current production precision is configurable; the
reported numbers have not been reinterpreted as measurements of another dtype
pair.

| one-family quantity | absolute fp32 | centered fp32 | improvement |
|---|---:|---:|---:|
| loss absolute error | `6.199e-5` | `9.096e-6` | **6.8x** |
| root-row maximum error | `1.208e-4` | `1.277e-5` | **9.5x** |
| root-row L2 error | `2.552e-3` | `3.538e-4` | **7.2x** |
| gradient relative L2 error | `8.478e-6` | `7.941e-7` | **10.7x** |
| gradient infinity error | `8.696e-5` | `5.288e-6` | **16.4x** |

The weighted forward trace is also clear: centered root maximum error is
`8.156e-6` versus `1.209e-4`, and loss error is `6.908e-6` versus `5.346e-5`.
See [`forward_precision_report.md`](forward_precision_report.md) for the 2,007
matched component tensors and the causal substitutions.

On the 1,055-family HOGENOM workload it remains within the 3% paired performance
target:

| paired, warmed evaluation | absolute median | centered median | median paired overhead |
|---|---:|---:|---:|
| forward | `2,822.64 ms` | `2,880.39 ms` | **+2.56%** |
| loss + full gradient | `6,959.41 ms` | `7,030.91 ms` | **+0.71%** |

All 12 full-workload losses are identical within each historical variant. The
forward paired-overhead samples range from `+0.24%` to `+5.27%`; loss+gradient
ranges from `-3.28%` to `+2.27%`. Paired medians are used because they control
short-term GPU clock drift better than ratios of independent medians.

## Implementation and correctness gates

CUDA Pi/Pibar matrices are always stored as centered residuals in the model
dtype, with one configured-accumulator offset per row. Centered storage is a
solver invariant for both fp32 and fp64 models; Python/TOML/CLI configuration
selects precision, not an absolute-storage fallback. Supported
model/accumulator pairs are fp32/fp32, fp32/fp64, and fp64/fp64. Fp64/fp32 is
rejected because the accumulator may not be narrower than model state.

The state contract is specified in
[`centered_state_contract.md`](centered_state_contract.md). Inactive and wholly
impossible rows have residual `-inf` and offset zero. Native derivative paths
consume frame offsets directly; full matrix reconstruction is retained only as
a correctness oracle. Backtracking reconstructs only the selected family at
its sampler boundary.

The final precision changes are deliberately narrow:

- root rows, E, origination values, cross-batch loss reductions, and the small
  root/survival adjoint seeds use the configured accumulator dtype;
- parameter log-softmaxes are computed in the accumulator dtype and rounded
  once to the model dtype;
- leaf observations select their actual frame rather than an assumed zero
  frame; and
- the first split-DTS consumer tracks its raw residual maximum and publishes a
  temporary accumulator-dtype virtual centered offset. DTS storage stays
  immutable, and the shift folds into the existing correction without another
  per-species operation.

For every benchmark in this report, that accumulator dtype was fp64.

Coverage includes uniform and weighted receiver/origination models, nonzero
leaf observations, fanout-one and grouped DTS, pruned/all-`-inf` rows, streamed
batches, autograd lifetime, native-vs-fp64 first order, JVP, second-order
contractions, exact HVP, GGN, finite differences, parameter rounding, and fp64
head promotion in the historical fp32/fp64 configuration.

Final validation command:

```bash
.venv/bin/python -m pytest -q
```

Result after canonical-path consolidation: `307 passed, 11 skipped, 23 warnings
in 123.95s`. The warnings are the
existing TorchScript deprecation and fp32 GMRES tolerance-floor warnings.

## Benchmark method

Hardware and software:

- NVIDIA GeForce RTX 4090, 24 GiB, compute capability 8.9
- NVIDIA driver 580.173.02, CUDA 13.0
- Python 3.12.12, PyTorch 2.13.0+cu130, Triton 3.7.1
- Linux 6.8.0-134-generic
- branch `feat/centered-kernels`, working-tree base `b4ce4f69e89f0dabec1fa29eb6ee15f4800f7143`

The benchmark initializes log2-rate logits to zero, runs 128 E iterations at
`1e-8` tolerance, 64 Pi iterations, and 64 Neumann terms. CUDA is synchronized
around every host `perf_counter` sample. Three warmups exclude compilation and
12 raw samples are retained. The historical paired phase alternates which
experimental implementation runs first and reverses that order for gradient
evaluation. Peak values are PyTorch CUDA allocation peaks.

The centered fp32 benchmark precision corresponds to
`[precision] model_dtype="float32", accumulator_dtype="float64"`; the reference
uses both fields set to `"float64"`.

The full workload has 1,055 families, 1,036,963 clades, 1,331 species nodes,
and five streamed batches in specieswise mode. Warm-adjoint caching is disabled
for both variants because the estimated 5.1 GiB cache plus 15.6 GiB scratch
exceeds the 18.4--18.8 GiB safety budget.

Historical commands used to produce this report are shown below. The
comparison drivers were removed when the alternate absolute implementation was
deleted; they remain available from Git history at commit `f6d2ac37`.

```bash
export GPUREC_HOGENOM_ROOT=/path/to/hogenom

.venv/bin/python benchmark/diagnose_forward_precision.py \
  --families 1 --weighted --e-max-iter 128 --e-tol 1e-8 \
  --pi-iters 64 \
  --output output/forward_precision_diagnostic_family1_weighted_final.json

.venv/bin/python benchmark/bench_centered_kernels.py \
  --families 1 --reference-fp64 --head-control --paired-alternating \
  --warmups 3 --repeats 12 --optimizer-steps 5 \
  --output output/centered_kernels_family1_final.json

.venv/bin/python benchmark/bench_centered_kernels.py \
  --families 1055 --mode specieswise --paired-alternating \
  --warmups 3 --repeats 12 --optimizer-steps 3 \
  --output output/centered_kernels_hogenom1055_final.json
```

## Numerical evidence

### Component capture

The weighted one-family capture uses nonzero DTL logits and nonuniform receiver
and origination weights. All three variants converge E in 18 iterations and
produce 2,007 matched captures without missing shared tensors or mask
mismatches. The centered variant adds 52 native residual/virtual-offset captures for
the DTS framing side channel.

| weighted quantity | absolute fp32 | centered fp32 | absolute fp64 |
|---|---:|---:|---:|
| production loss | `323.435630504` | `323.435583954` | `323.435577046` |
| loss absolute error | `5.346e-5` | `6.908e-6` | 0 |
| root maximum error | `1.209e-4` | `8.156e-6` | 0 |
| root L2 error | `1.812e-3` | `2.300e-4` | 0 |
| full Pi maximum error | `1.364e-4` | `1.044e-5` | 0 |

Absolute split-DTS maximum error grows from `4.916e-6` at wave 1 to
`1.083e-4` at wave 26. Centered wave-26 DTS error is `8.145e-6`. In contrast,
final E maximum error is `3.255e-7`, final Ebar maximum error is `7.749e-7`,
and replacing only E in the fp64 head changes loss by `1.53e-8`.

### One-family forward and gradient

The unweighted fp64 oracle contains 413 clades and the same 1,331-node species
tree.

| quantity | absolute fp32 | centered fp32 | absolute fp64 |
|---|---:|---:|---:|
| loss | `304.852344664` | `304.852291772` | `304.852282676` |
| loss absolute error | `6.199e-5` | `9.096e-6` | 0 |
| gradient relative L2 error | `8.478e-6` | `7.941e-7` | 0 |
| gradient infinity error | `8.696e-5` | `5.288e-6` | 0 |
| repeated-gradient maximum delta | `8.583e-6` | `2.861e-6` | `4.44e-14` |

The repeatability delta reflects nondeterministic fp32 atomic order and is not
an error-to-reference measurement. Both historical fp32 variants have stable repeated losses,
and five fixed SGD steps remain finite and monotonically decreasing.

The fp64 likelihood head used by this audit is independent of Pi storage
framing. For the same historical absolute fp32 roots, its loss is `304.852344664`, versus
`304.852355957` for a manual fp32 head. Centering supplies the larger additional
gain by preserving Pi/DTS row-local low bits.

### Full 1,055-family workload

No full fp64 run is attempted because two dense fp64 Pi-family matrices plus
streamed solver state leave an inadequate safety margin on this 24 GiB GPU.
The one-family reference is the accuracy gate; the full workload is the timing
and memory gate. Previous allocation measurements for the same centered state
shape were 4,780,808,704 B absolute versus approximately 4,805,954,048 B
centered for forward (+0.53%), and 6,670,900,736 B versus 6,691,231,232 B for
loss+gradient (+0.30%). In this fp32/fp64 configuration, the virtual DTS frame
adds only one temporary fp64 scalar per split-wave row.

## Why these fixes

The trace rules out several more expensive ideas. Recomputing Pi in fp64 while
keeping fp32-rounded upstream inputs reduces the unweighted loss error to about
`3.4e-6`; changing only E has a sub-micro effect. Thus an fp64 E state or all-
fp64 hot wave reduction has a poor cost/benefit ratio. Correctly rounded small
softmaxes and fp64 head reductions were cheap in the audited fp32/fp64 policy,
while maintaining local Pi/DTS frames attacks the measured dominant error
without changing dense storage.

## Conclusion

Centered state is the canonical CUDA representation with complete derivative
coverage for supported model/accumulator precision pairs. The fp32/fp64 tensor
capture demonstrates a real centered-Pi benefit: roughly 7--16x on the
unweighted loss/gradient gates and 8--15x on the weighted forward/root gates.
In the measured configuration, the fixes retain fp32 dense kernels and confine
fp64 to small reductions and row gauges. Production also supports fp32/fp32
and fp64/fp64 with the same centered-state contract; fp64/fp32 is rejected.

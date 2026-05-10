# Bug Report: AleRax `ScaledValue` Drops Probability Mass Across Scale Buckets

Date: 2026-05-08

## Summary

AleRax's undated DTL likelihood can be underestimated because the
`ScaledValue` arithmetic used by GeneRaxCore drops terms whenever two operands
live in different scale buckets. This affects additions and subtractions in the
dynamic program that accumulates reconciliation probabilities.

The affected source file is:

```text
AleRax/ext/GeneRaxCore/src/maths/ScaledValue.hpp
```

The problem is not in the biological DTL recurrence itself. When the same
recurrence is evaluated in exact log-space, it matches GPUREC. The discrepancy
appears when AleRax's `ScaledValue` operators approximate cross-bucket
addition/subtraction by keeping only one operand.

For the diagnostic HOGENOM case:

```text
family: family_0444
rates:  D = 1, L = 1, T = 1
```

the observed likelihoods were:

```text
unpatched AleRax NLL: 1043.251554916500027 nats
GPUREC exact NLL:     1042.786263344967665 nats
patched AleRax NLL:   1042.786263344967665 nats
```

After fixing `ScaledValue` to rescale operands before adding/subtracting, AleRax
matches GPUREC exactly at the printed precision.

## Affected Code

The old `ScaledValue::operator+=` logic was equivalent to:

```cpp
if (v.scaler == scaler) {
  value += v.value;
} else if (v.scaler < scaler) {
  value = v.value;
  scaler = v.scaler;
}
return *this;
```

The old `operator+` and `operator-` used the same scale-only decision rule.

This is not exact arithmetic. It treats different scale buckets as if the
smaller-scale bucket always dominates and the other contribution can be ignored.
That approximation is false near bucket boundaries.

## Why This Is Wrong

`ScaledValue` represents a real value as:

```text
real_value = value * JS_SCALE_THRESHOLD^scaler
```

where:

```text
JS_SCALE_THRESHOLD = 2^-256
```

Call this threshold `T`. Two adjacent scale buckets can represent comparable
numbers:

```text
A = T * T^0 = T
B = 1 * T^1 = T
```

Both values are equal. But the old `operator+=` sees different scalers and keeps
only the operand with smaller `scaler`. Therefore:

```text
A + B should be 2T
old ScaledValue addition returns T
```

That is a factor-two error for a single addition. In log likelihood units, that
is a `log(2)` error if this lost term reaches the final likelihood.

The approximation is therefore not only a tiny underflow shortcut. It can drop
non-negligible positive probability mass whenever adjacent scale buckets meet
near their boundary. In a large dynamic program, this loss can accumulate across
many clades, species branches, and fixed-point iterations.

## Where The Loss Enters The DTL Dynamic Program

The faulty arithmetic is used in `UndatedDTLMultiModel::computeProbability`:

```text
AleRax/src/ale/UndatedDTLMultiModel.hpp
```

Examples include:

```cpp
temp = _dtlclvs[cidLeft]._uq[fc] * _dtlclvs[cidRight]._uq[gc] *
       (_PS[ec] * freq);
scale(temp);
proba += temp;
```

and:

```cpp
temp = _dtlclvs[cid]._uq[ec] * (_uE[ec] * _PD[ec] * 2.0);
scale(temp);
proba += temp;
```

Similar `proba += temp` operations accumulate speciation, duplication,
transfer, loss, duplication-loss, and transfer-loss terms. The transfer
normalization path also relies on scaled addition/subtraction:

```cpp
transferSum += p;
tp = transferSum - tp;
```

So the bug affects both the direct reconciliation event sum and the
transfer-normalized vectors.

## Evidence From The Source-Level Debugging

I instrumented AleRax to dump:

```text
species topology and event probabilities
E and Ebar fixed-pass vectors
clade split definitions
per-clade Pi and Pibar vectors
root likelihood vectors
```

The diagnostic dumps ruled out the following explanations:

```text
wrong species tree
wrong species indexing after label mapping
wrong E recurrence
wrong Ebar transfer normalization
wrong leaf-clade initialization
wrong CCP splits or split frequencies
wrong GPUREC log-space recurrence
```

The first divergent clade in `family_0444` was:

```text
cid: 265
leaf count: 25
children: 227, 264
split count: 1
```

Both child clades matched the exact recurrence. The parent did not.

I recomputed `cid=265` independently in Python using AleRax's own dumped child
CLVs, species traversal order, event probabilities, transfer candidates, and E
vectors. That independent log-space recurrence matched GPUREC, not unpatched
AleRax.

For clade `265`, old AleRax differed from the exact recurrence by:

```text
Pi    max absolute log difference: 0.937229916255404
Pi    mean absolute log difference: 0.256030047304301
Pibar max absolute log difference: 1.75344708287756
Pibar mean absolute log difference: 0.743354198148379
```

After patching `ScaledValue`, the same clade matched the exact recurrence:

```text
Pi    max absolute log difference: 8.5265128291212e-14
Pi    mean absolute log difference: 1.91337092920154e-14
Pibar max absolute log difference: 8.5265128291212e-14
Pibar mean absolute log difference: 1.55944020799273e-14
```

This isolates the bug to AleRax's arithmetic layer. The recurrence formula is
not the source of the discrepancy.

## Patch

The diagnostic patch changes `operator+`, `operator+=`, and `operator-` to
rescale operands to a common scale bucket before performing arithmetic.

Current patched logic:

```cpp
const int commonScaler = scaler < v.scaler ? scaler : v.scaler;
const double lhs =
    value * std::pow(JS_SCALE_THRESHOLD, scaler - commonScaler);
const double rhs =
    v.value * std::pow(JS_SCALE_THRESHOLD, v.scaler - commonScaler);
```

Then addition/subtraction is performed on `lhs` and `rhs`, and the result is
scaled.

This makes adjacent bucket contributions exact up to ordinary floating-point
roundoff instead of silently dropping one side.

## Reproduction

Build patched AleRax:

```bash
cd /home/enzo/Documents/git/gpurec/gpurec/AleRax
cmake --build build -j 8 --target alerax
```

Run the fixed-rate diagnostic:

```bash
ALERAX_DEBUG_DUMP_DIR=/home/enzo/Documents/git/gpurec/gpurec/tests/data/hogenom_bench/diagnostics/alerax_debug_family_0444_x1_scaledvalue_exact/dump \
ALERAX_DEBUG_DUMP_PREFIX=family_0444_x1_scaledvalue_exact \
ALERAX_DEBUG_DUMP_CLV_ITERS_CIDS=265,root \
/home/enzo/Documents/git/gpurec/gpurec/AleRax/build/bin/alerax \
  -s /home/enzo/Documents/git/gpurec/gpurec/tests/data/hogenom_bench/sp.nwk \
  -f /home/enzo/Documents/git/gpurec/gpurec/tests/data/hogenom_bench/diagnostics/alerax_debug_family_0444_x1_scaledvalue_exact/families.txt \
  -p /home/enzo/Documents/git/gpurec/gpurec/tests/data/hogenom_bench/diagnostics/alerax_debug_family_0444_x1_scaledvalue_exact/alerax_out \
  --model-parametrization GLOBAL \
  --d 1 --l 1 --t 1 \
  --fix-rates \
  --species-tree-search SKIP \
  -g 0
```

The patched run reports:

```text
Initial ll=-1042.79
```

The exact value recomputed from the root dump is:

```text
patched AleRax NLL = 1042.786263344967665 nats
```

which matches GPUREC:

```text
GPUREC NLL = 1042.786263344967665 nats
```

## Impact

The bug makes unpatched AleRax undercount positive reconciliation probability
mass in some cases. The visible effect is an inflated NLL. It is most likely to
matter for:

```text
large gene trees
deep dynamic programs with many accumulated terms
high D/T/L rates
families where probabilities frequently cross ScaledValue bucket boundaries
```

This explains why GPUREC sometimes appeared to find a much better likelihood
than AleRax at the same fixed parameters: GPUREC's log-space evaluator was
keeping probability mass that AleRax's scaled arithmetic dropped.

## Recommendation

The patch used here is sufficient to prove the cause of the discrepancy, but an
upstream-quality fix should be reviewed carefully for performance and numerical
policy. Reasonable options are:

```text
1. Keep the common-scale addition/subtraction patch.
2. Normalize ScaledValue more aggressively after multiplication and addition.
3. Replace the lossy scaled arithmetic with log-space logsumexp where feasible.
4. Add unit tests for adjacent-bucket addition/subtraction.
```

A minimal unit test should include the adjacent-bucket equality case:

```text
ScaledValue(T, 0) + ScaledValue(1, 1) == ScaledValue(2T, 0)
```

where `T = JS_SCALE_THRESHOLD`.

That test fails under the old scale-only addition rule and captures the core
mistake directly.

# AleRax ScaledValue Likelihood Divergence

Date: 2026-05-08

This note records the final source-level diagnosis for the residual
AleRax/GPUREC likelihood mismatch on the HOGENOM benchmark after aligning the
model semantics to AleRax's fixed four-pass evaluator.

## Executive Summary

For the tested fixed-rate case, GPUREC is evaluating the source-level DTL
recurrence correctly. The remaining discrepancy was in AleRax's numeric
representation, specifically `ScaledValue` addition and subtraction in:

```text
AleRax/ext/GeneRaxCore/src/maths/ScaledValue.hpp
```

The old `ScaledValue` implementation chose an operand based only on the integer
`scaler` when two operands had different scale buckets. That is only safe if the
stored mantissas are normalized. They are not always normalized, because
`ScaledValue` multiplication does not normalize immediately and `scale()` only
shifts by one bucket.

As a result, AleRax sometimes dropped the larger real-valued contribution during
`proba += temp`, `transferSum += p`, or `transferSum - forbidden_sum`. This made
AleRax report a lower likelihood than the exact recurrence and than GPUREC.

After patching `ScaledValue::{operator+, operator+=, operator-}` to rescale both
operands to a common bucket before adding or subtracting, the fixed-rate
AleRax likelihood for the diagnostic family exactly matches GPUREC.

## Diagnostic Target

The decisive test case was:

```text
family: family_0444
gene tree: tests/data/hogenom_bench/g_444.nwk
species tree: tests/data/hogenom_bench/sp.nwk
rates: D = 1, L = 1, T = 1
model parametrization: GLOBAL
species tree search: SKIP
gene tree rootings: all rootings considered
E passes: 4
Pi passes: 4
transfer mode: uniform, AleRax parent-transfer constraint
```

Before the `ScaledValue` patch:

```text
AleRax NLL: 1043.251554916500027 nats
AleRax NLL: 1505.093844677734751 bits
```

GPUREC, with the AleRax-compatible fixed-pass evaluator:

```text
GPUREC NLL: 1042.786263344967665 nats
GPUREC NLL: 1504.422570834917451 bits
```

After the `ScaledValue` patch, AleRax gives:

```text
patched AleRax NLL: 1042.786263344967665 nats
patched AleRax NLL: 1504.422570834917451 bits
```

So the patched AleRax value and GPUREC value are identical at the printed
precision for this case.

## Instrumentation Added To AleRax

I added debug dumps to `AleRax/src/ale/UndatedDTLMultiModel.hpp`. They are
enabled with environment variables:

```bash
ALERAX_DEBUG_DUMP_DIR=/path/to/dump
ALERAX_DEBUG_DUMP_PREFIX=family_0444_x1
ALERAX_DEBUG_DUMP_CLV_ITERS_CIDS=265,root
ALERAX_DEBUG_DUMP_ALL_CLVS=1
```

The dumps include:

```text
*_species.tsv
*_clades.tsv
*_splits.tsv
*_e_iters.tsv
*_clv_iters.tsv
*_clv_final.tsv
*_root_final.tsv
```

I also bypassed the likelihood cache while debug dumping in:

```text
AleRax/src/ale/MultiModel.hpp
```

Without that, repeated likelihood evaluations could reuse cached values and
skip the vector dumps.

To align clades by contents rather than by unstable IDs alone, I added:

```text
ConditionalClades::getCladeLeafLabels(...)
```

in:

```text
AleRax/ext/GeneRaxCore/src/ccp/ConditionalClades.hpp
AleRax/ext/GeneRaxCore/src/ccp/ConditionalClades.cpp
```

## What Was Ruled Out

The source-level dumps ruled out the following as the cause of this residual
case:

```text
species tree topology
species vector ordering, after mapping by species labels
extinction vector E
transfer-normalized extinction vector Ebar
leaf-clade Pi initialization
small internal-clade Pi recursion
CCP split frequencies
GPUREC log-space recurrence formula
```

Important alignment detail: AleRax and GPUREC do not use the same species node
integer IDs. Comparing vectors by raw index is misleading. The vectors must be
mapped by species label, with the root handled as the unlabeled GPUREC root.
After label mapping, `E` and `Ebar` matched to floating-point precision.

## First Divergence

The first nontrivial divergence in `family_0444` occurred at AleRax clade:

```text
cid = 265
leaf count = 25
split count = 1
children = 227, 264
```

Both child clades matched the independent recurrence. The parent did not.

I recomputed clade `265` with an independent log-space implementation of the
AleRax recurrence, using:

```text
AleRax child CLVs for cids 227 and 264
AleRax species topology and traversal order
AleRax event probabilities PD, PL, PT, PS
AleRax transfer candidate sets
AleRax E and Ebar vectors
the same four in-place Pi passes
```

Comparison against the old AleRax binary:

```text
old AleRax u  - exact u: max_abs_ln  = 0.937229916255404
old AleRax u  - exact u: mean_abs_ln = 0.256030047304301
old AleRax t  - exact t: max_abs_ln  = 1.75344708287756
old AleRax t  - exact t: mean_abs_ln = 0.743354198148379
```

Comparison against patched AleRax:

```text
patched AleRax u - exact u: max_abs_ln  = 8.5265128291212e-14
patched AleRax u - exact u: mean_abs_ln = 1.91337092920154e-14
patched AleRax t - exact t: max_abs_ln  = 8.5265128291212e-14
patched AleRax t - exact t: mean_abs_ln = 1.55944020799273e-14
```

That isolates the divergence to arithmetic inside AleRax's recurrence
evaluation, not to the recurrence definition.

## The Broken Operation

The old `ScaledValue::operator+=` effectively did this:

```cpp
if (v.scaler == scaler) {
  value += v.value;
} else if (v.scaler < scaler) {
  value = v.value;
  scaler = v.scaler;
}
```

The old `operator+` and `operator-` used the same scale-only logic.

This assumes the object invariant:

```text
value is normalized for its scaler
```

But multiplication can break that invariant:

```cpp
inline ScaledValue operator*(const ScaledValue &v) const {
  auto res = ScaledValue(v.value * value, v.scaler + scaler);
  return res;
}
```

The multiplication result is not normalized. Then callers often normalize only
once:

```cpp
scale(temp);
proba += temp;
```

and `scale()` itself only shifts by one scale bucket:

```cpp
if (value < JS_SCALE_THRESHOLD) {
  scaler += 1;
  value *= JS_SCALE_FACTOR;
}
```

Therefore a value with a numerically smaller `scaler` can still represent a
smaller real number than another value with a larger `scaler`. The old
addition/subtraction code ignored that possibility and could discard the larger
real contribution.

The problematic additions are reached in the DTL probability recurrence in:

```text
AleRax/src/ale/UndatedDTLMultiModel.hpp
```

notably:

```cpp
proba += temp;
transferSum += p;
tp = transferSum - tp;
```

Those operations are used for speciation, duplication, transfer, loss, DL, and
TL contributions.

## Patch Applied

I changed `ScaledValue::{operator+, operator+=, operator-}` so operands with
different scale buckets are first represented in the smaller common scaler:

```cpp
const int commonScaler = scaler < v.scaler ? scaler : v.scaler;
const double lhs =
    value * std::pow(JS_SCALE_THRESHOLD, scaler - commonScaler);
const double rhs =
    v.value * std::pow(JS_SCALE_THRESHOLD, v.scaler - commonScaler);
```

Then the operation is performed on `lhs` and `rhs`, and the result is scaled.

This is a diagnostic/proof patch. A more production-grade upstream fix should
probably also enforce normalization after multiplication or implement a more
systematic log-space or arbitrary-scale arithmetic policy. The current patch is
enough to prove that the residual GPUREC/AleRax mismatch came from
cross-scaler arithmetic in AleRax.

## Verification Commands

Build patched AleRax:

```bash
cd /home/enzo/Documents/git/gpurec/gpurec/AleRax
cmake --build build -j 8 --target alerax
```

Run the fixed-rate AleRax check:

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

Run the GPUREC check:

```bash
GPUREC_ALERAX_COMPAT=1 \
GPUREC_ALERAX_COMPAT_E_ITERS=4 \
GPUREC_ALERAX_COMPAT_PI_ITERS=4 \
GPUREC_FORWARD_PARENT_REDUCED_DTS=0 \
python - <<'PY'
from pathlib import Path
import math
import torch
from gpurec import GeneReconModel

root = Path("/home/enzo/Documents/git/gpurec/gpurec/tests/data/hogenom_bench")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeneReconModel.from_trees(
    species_tree=str(root / "sp.nwk"),
    gene_trees=[str(root / "g_444.nwk")],
    mode="global",
    pibar_mode="uniform",
    device=device,
    dtype=torch.float64,
    theta_init_rates=(1.0, 1.0, 1.0),
    preprocess_cache_dir="/tmp/gpurec_notebook_cache",
    fixed_iters_E=4,
    max_iters_E=4,
    fixed_iters_Pi=4,
    max_iters_Pi=4,
    neumann_terms=4,
    use_pruning=False,
    pruning_threshold=1e-6,
    max_wave_size=32768,
)
with torch.no_grad():
    nll_bits = float(model.nll().item())
print(f"gpurec_nll_bits={nll_bits:.15f}")
print(f"gpurec_nll_nats={nll_bits * math.log(2):.15f}")
PY
```

Observed GPUREC output:

```text
gpurec_nll_bits=1504.422570834917451
gpurec_nll_nats=1042.786263344967665
```

Observed patched AleRax output from the root dump:

```text
alerax_patched_nll_bits=1504.422570834917451
alerax_patched_nll_nats=1042.786263344967665
```

## Conclusion

For this fixed-rate HOGENOM discrepancy, GPUREC was not finding an incorrect
likelihood. Once GPUREC was put into AleRax-compatible fixed-pass mode, its
log-space recurrence matched the exact source-level recurrence. The residual
mismatch was caused by AleRax's `ScaledValue` arithmetic dropping or selecting
terms incorrectly when scale buckets differed and mantissas were not normalized.

The immediate consequence is that GPUREC can report a better NLL than unpatched
AleRax at the same rates because unpatched AleRax is undercounting some
positive probability mass.

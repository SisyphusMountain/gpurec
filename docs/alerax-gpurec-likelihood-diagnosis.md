# AleRax / GPUREC Likelihood Diagnosis

Date: 2026-05-08

Update: this document records the earlier compatibility-mode investigation.
The residual high-rate discrepancy was later isolated to AleRax's
`ScaledValue` arithmetic, not to GPUREC's log-space recurrence. See
`docs/alerax-scaledvalue-diagnosis.md` for the final source-level diagnosis and
the exact family_0444 verification.

This document explains the likelihood discrepancy investigation around the
HOGENOM benchmark, the code changes made in GPUREC, what those changes do, and
what remains unresolved.

## Executive Summary

The original GPUREC likelihood evaluator was not evaluating the same objective
as AleRax. The largest issue was not a simple log-base conversion problem. It
was that GPUREC was solving the `E` and `Pi` recurrences as converged fixed
points, while AleRax v1.4 evaluates a fixed number of in-place postorder
Gauss-Seidel passes.

That difference is small at low rates, but it becomes very visible when the DTL
rates are pushed toward 1-2. This explains the weird upward-curving
GPUREC-vs-AleRax plots: the curves were comparing two different numerical
objectives.

I added an explicit AleRax-compatible likelihood path behind:

```bash
GPUREC_ALERAX_COMPAT=1
```

This path changes only the likelihood evaluation semantics. It is currently
limited to global/shared uniform-transfer evaluation. I also disabled the
current custom implicit-gradient backward under this flag, because that backward
still differentiates GPUREC's converged fixed-point objective, not AleRax's
fixed four-pass evaluator.

## What Was Wrong

### 1. Fixed Point vs Fixed Four-Pass Evaluation

GPUREC's default objective solves recurrences to convergence:

- `E_fixed_point(...)` iterates until tolerance or `max_iters`.
- `Pi_wave_forward(...)` iterates each wave until tolerance or a caller-provided
  fixed iteration count.

AleRax v1.4 does something different for the undated DTL model. It applies a
small fixed number of in-place postorder passes. In the cases we inspected, this
was four passes.

Those are mathematically different objectives. If the recurrence is close to a
contraction, the difference may be tiny. When D/T/L are high, the difference can
be large enough to distort line scans and optimization comparisons.

The new code path implements this by:

- running `E` in probability space for a fixed number of postorder passes;
- running `Pi` as fixed in-place species-postorder passes inside each clade wave;
- recomputing `Pibar` after each full species pass.

The relevant code is:

- `gpurec/core/likelihood.py`: `_E_fixed_point_alerax_gs(...)`
- `gpurec/core/forward.py`: `_run_wave_self_loop_alerax(...)`

### 2. Leaf Observation Semantics

The old GPUREC initialization seeded observed gene/species leaf cells with
`Pi = 0` before the recurrence. That is natural for the converged fixed-point
formulation, but it is not equivalent to the AleRax recurrence.

In AleRax-compatible mode, leaf observations are added as a term during every
pass:

```text
base = logsumexp(DL, TL, LT, leaf_observation, DTS)
```

They are not just terminal initial values. This matters because a leaf clade can
still participate in continuation terms during subsequent passes.

Change:

- In `gpurec/core/forward.py`, when `GPUREC_ALERAX_COMPAT=1`, `Pi` is no longer
  pre-seeded with leaf observations. The leaf term is included inside
  `_run_wave_self_loop_alerax(...)`.

### 3. DL Loss Multiplier

The DL self-loop term needs the `2 * pD * E * Pi` contribution. In log2 space
that is:

```text
log2(2) + log_pD + E + Pi
```

The code now makes that multiplier explicit and configurable:

```bash
GPUREC_DL_LOSS_MULTIPLIER=2.0
```

Relevant files:

- `gpurec/core/forward.py`
- `gpurec/core/terms.py`

The default remains `2.0`.

### 4. Transfer Sum Update Ordering

For uniform transfer, AleRax's recurrence effectively uses transfer sums that
are recomputed after each full species pass. Within a species pass, the transfer
sum is lagged from the previous pass.

The compatible `Pi` path therefore:

1. takes the current `Pi` and `Pibar`;
2. computes the base continuation terms;
3. updates species states in postorder in-place for the current pass;
4. recomputes `Pibar` from the updated full species vector;
5. repeats four times by default.

That ordering is intentionally different from GPUREC's optimized converged
fixed-point kernels.

## Code Changes

### `gpurec/core/likelihood.py`

Added `_E_fixed_point_alerax_gs(...)`.

This is an AleRax-compatible `E` evaluator for uniform-transfer global/shared
runs. It works in probability space, updates species in postorder, and computes
`Ebar` after each full pass.

The path is selected when either environment variable is set:

```bash
GPUREC_ALERAX_COMPAT=1
GPUREC_E_ALERAX_GS=1
```

In full compatibility mode, the iteration count defaults to four:

```bash
GPUREC_ALERAX_COMPAT_E_ITERS=4
```

I also added a fail-fast check: compatibility mode currently requires
`pibar_mode="uniform"` and an `ancestors_T` topology matrix. It will raise
`NotImplementedError` for unsupported modes instead of silently evaluating a
hybrid objective.

### `gpurec/core/forward.py`

Added AleRax-compatible `Pi` evaluation behind `GPUREC_ALERAX_COMPAT=1`.

Important behavior changes under this flag:

- only `pibar_mode="uniform"` is supported;
- only shared/global rates are supported;
- leaf observations are not used as initial `Pi = 0` seeds;
- the evaluator forces a fixed number of passes;
- the default number of passes is four:

```bash
GPUREC_ALERAX_COMPAT_PI_ITERS=4
```

The core implementation is `_run_wave_self_loop_alerax(...)`, which performs:

```text
for pass in 1..4:
    base = logsumexp(DL, TL, LT, leaf, DTS)
    for species in postorder:
        add SL terms using child values already updated in this pass
    recompute Pibar from updated Pi
```

I also disabled forward DTS-overlap scheduling under compatibility mode. That
keeps the execution order simple and deterministic while matching AleRax's
pass ordering.

### `gpurec/core/terms.py`

Made the DL multiplier explicit:

```python
dl_loss_multiplier = float(os.environ.get("GPUREC_DL_LOSS_MULTIPLIER", "2.0"))
```

This keeps the old mathematical default but makes the factor visible and
testable.

### `gpurec/core/legacy.py`

Added diagnostic switches used during the investigation:

```bash
GPUREC_PI_INIT_LEAF_OBSERVATIONS=0
GPUREC_TERMINAL_LEAF_SPECIES=1
```

These were useful for testing hypotheses independently. They are not the main
fix.

### `gpurec/api/autograd.py`

Blocked custom autograd when `GPUREC_ALERAX_COMPAT=1`.

This is intentional. The current backward path uses an implicit-gradient solve
for the converged GPUREC fixed-point objective. After enabling AleRax-compatible
forward evaluation, using that same backward would produce gradients for the
wrong objective.

So this now raises `NotImplementedError` if someone tries:

```python
GPUREC_ALERAX_COMPAT=1
loss = model()
loss.backward()
```

No-grad likelihood evaluation still works:

```python
with torch.no_grad():
    nll = model.nll()
```

## Diagnostic Process

### Step 1: Fix Units

The comparison initially mixed GPUREC log2 values with AleRax natural-log
values. That was fixed by converting AleRax likelihoods from nats to bits:

```text
alerax_nll_bits = -alerax_log_likelihood_nats / log(2)
```

After that conversion, discrepancies remained. So the issue was not only units.

### Step 2: Scan D/L/T Together from 0 to 2

The user-requested diagnostic was to evaluate fixed parameters with:

```text
D = L = T = x, x in [0, 2]
```

This showed that the old GPUREC evaluator and AleRax diverged strongly as rates
increased. That was the key evidence that this was a likelihood-evaluation
problem, not only an optimizer problem.

### Step 3: Test Candidate Causes Independently

Several possible explanations were tested separately:

- log2 vs nats conversion;
- parameterization of rates into probabilities;
- transfer denominator / recipient set;
- species tree topology parsing;
- missing species coverage;
- CCP parsing and split frequencies;
- duplicate species / repeated gene leaves;
- numerical precision;
- fixed iteration count vs convergence.

The decisive result was that changing iteration semantics from converged
fixed-point evaluation to AleRax-style fixed in-place passes collapsed most of
the discrepancy.

### Step 4: Verify CCP Equality

For `family_0631`, I parsed the AleRax-generated CCP file and compared it
against GPUREC's C++ preprocessing output.

The clade sets, split sets, root split count, and split frequencies matched.
This ruled out the gene-tree/CCP preprocessing as the primary explanation for
the residual discrepancy.

Important observed values:

```text
AleRax CCP leaves: 169
AleRax clades:     671
AleRax splits:     836
GPUREC clades:     671
GPUREC splits:     836
Root split count:  335
```

The clade and split key sets matched exactly. The max split-frequency
difference was around machine precision.

### Step 5: Independent Reference Implementation

I wrote an independent vectorized Python reference for the source-level AleRax
v1.4 recurrence using:

- the AleRax-generated CCP file;
- the GPUREC species topology;
- the AleRax fixed four-pass recurrence semantics;
- the same D/L/T rates.

For the difficult transfer-heavy case:

```text
family_0631
D = 1e-10
L = 1e-10
T = 2.0
```

the independent reference matched GPUREC, not the installed AleRax binary:

```text
independent reference NLL: 2621.891513526077 bits
GPUREC compat NLL:        2621.891513117995 bits
installed AleRax NLL:     2623.843897815167 bits
```

That is why I do not classify the remaining `family_0631` transfer-heavy gap as
an identified GPUREC formula bug. For that specific case, GPUREC matches the
source-level recurrence we independently implemented.

## Validation Results

After enabling:

```bash
GPUREC_ALERAX_COMPAT=1
```

I reran compact HOGENOM scans.

### `family_0276`, D=L=T

```text
x=0.00  AleRax=138.083949  GPUREC=138.084015  diff=+0.000066 bits
x=1.00  AleRax=46.334748   GPUREC=46.334808   diff=+0.000060 bits
x=2.00  AleRax=47.483133   GPUREC=47.483196   diff=+0.000063 bits
```

### `family_0000`, D=L=T

Observed max absolute difference over `x = 0, 0.5, 1, 1.5, 2`:

```text
~0.0012 bits
```

### `family_0631`, D=L=T

This family is much more sensitive, especially in transfer-heavy regimes:

```text
x=0.00  diff=+0.005114 bits
x=0.50  diff=-0.000246 bits
x=1.00  diff=-0.013593 bits
x=1.50  diff=-0.114615 bits
x=2.00  diff=-0.040825 bits
```

The remaining discrepancy here was isolated further with event-axis scans. The
largest residual came from near-boundary transfer-only cases, not from the full
DTL scan uniformly.

For the specific `T=2, D=L=1e-10` case, the independent source-level reference
matched GPUREC to roughly `4e-7` bits and disagreed with the installed AleRax
binary by about `1.95` bits.

### Broader Random-20 Family Scan

On 2026-05-08, I ran a seeded random sample of 20 additional HOGENOM families
not used in the first examples:

```text
103 293 33 342 542 223 508 952 299 880
777 211 951 416 718 491 444 870 47 315
```

Command:

```bash
GPUREC_ALERAX_COMPAT=1 \
python profiling/scan_hogenom_dtl_triplet.py \
  --families 103 293 33 342 542 223 508 952 299 880 \
             777 211 951 416 718 491 444 870 47 315 \
  --points 5 \
  --out-dir tests/data/hogenom_bench/diagnostics/dtl_triplet_scan_compat_random20_20260508
```

This produced 100 fixed-parameter comparisons (`20 families * 5 rates`).

Aggregate result:

```text
max_abs_diff_bits:  0.772823
mean_abs_diff_bits: 0.038885
abs diff > 1e-3:    17 / 100
abs diff > 1e-2:     8 / 100
abs diff > 1e-1:     8 / 100
abs diff > 0.5:      4 / 100
```

Most families matched tightly, but the scan found clear counterexamples:

```text
family_0952 x=1.50 diff=-0.772823 bits
family_0952 x=1.00 diff=-0.699327 bits
family_0444 x=1.00 diff=-0.668750 bits
family_0444 x=0.50 diff=-0.514418 bits
family_0223 x=0.00 diff=-0.453268 bits
family_0444 x=1.50 diff=-0.363025 bits
family_0952 x=0.50 diff=-0.226263 bits
family_0444 x=2.00 diff=-0.133955 bits
```

I reran the worst points in float64. The residuals remained essentially the
same, so these are not fp32 roundoff artifacts:

```text
family_0223 x=0.00 diff=-0.453162 bits
family_0444 x=0.50 diff=-0.514063 bits
family_0444 x=1.00 diff=-0.669031 bits
family_0444 x=1.50 diff=-0.363162 bits
family_0444 x=2.00 diff=-0.134142 bits
family_0952 x=0.50 diff=-0.226109 bits
family_0952 x=1.00 diff=-0.699110 bits
family_0952 x=1.50 diff=-0.772738 bits
```

I also swept integer fixed-pass counts around the worst `family_0444, x=1.0`
case. With `E=4`, varying `Pi` gives:

```text
Pi=1 diff=+196.128 bits
Pi=2 diff= +51.556 bits
Pi=3 diff= +11.826 bits
Pi=4 diff=  -0.669 bits
Pi=5 diff=  -4.761 bits
Pi=6 diff=  -6.119 bits
Pi=7 diff=  -6.573 bits
Pi=8 diff=  -6.724 bits
```

With `Pi=4`, varying `E` gives:

```text
E=1 diff=+46.312 bits
E=2 diff=+12.783 bits
E=3 diff= +2.637 bits
E=4 diff= -0.669 bits
E=5 diff= -1.771 bits
E=6 diff= -2.141 bits
E=7 diff= -2.266 bits
E=8 diff= -2.308 bits
```

So the remaining mismatch is not explained by simply choosing a different
integer number of `E` or `Pi` passes.

## What This Means for Optimization

There are now two different objectives:

### GPUREC Default Objective

This is the converged fixed-point objective. It is the objective currently
supported by GPUREC's custom implicit-gradient backward and high-performance
optimization path.

Use this for normal GPUREC optimization unless the goal is exact fixed-parameter
comparison to AleRax's reported likelihood.

### AleRax-Compatible Objective

This is the fixed four-pass evaluator behind:

```bash
GPUREC_ALERAX_COMPAT=1
```

It should be used for fixed-parameter likelihood comparisons against AleRax.

It should not currently be used with `loss.backward()`, because the existing
custom backward differentiates a different objective. I explicitly blocked that
case to avoid silent false optimization results.

To optimize the AleRax-compatible objective, we need one of:

1. an unrolled differentiable implementation of the four fixed passes;
2. a derived backward for the fixed-pass recurrence;
3. finite-difference or external black-box optimization, which is slower but
   can be useful for validation.

## How To Reproduce the Key Check

Run:

```bash
GPUREC_ALERAX_COMPAT=1 \
python profiling/scan_hogenom_dtl_triplet.py \
  --families 276 0 631 \
  --points 5 \
  --out-dir tests/data/hogenom_bench/diagnostics/dtl_triplet_scan_compat_final
```

The script writes:

```text
tests/data/hogenom_bench/diagnostics/dtl_triplet_scan_compat_final/summary.csv
```

The quick smoke version is:

```bash
GPUREC_ALERAX_COMPAT=1 \
python profiling/scan_hogenom_dtl_triplet.py \
  --families 276 \
  --points 3 \
  --out-dir tests/data/hogenom_bench/diagnostics/dtl_triplet_scan_compat_smoke
```

Expected output shape:

```text
family_0276 x=0.00: diff around 6e-5 bits
family_0276 x=1.00: diff around 6e-5 bits
family_0276 x=2.00: diff around 6e-5 bits
```

## Verification Commands Run

Compilation:

```bash
python -m py_compile \
  gpurec/core/forward.py \
  gpurec/core/likelihood.py \
  gpurec/core/legacy.py \
  gpurec/core/terms.py \
  gpurec/api/autograd.py
```

API smoke tests:

```bash
python -m pytest \
  tests/integration/test_gene_recon_model.py::test_preprocess_cache_matches_single_path \
  tests/integration/test_gene_recon_model.py::test_multi_family_preprocess_defaults_to_light \
  -q
```

Result:

```text
2 passed
```

Autograd guard smoke check:

```bash
GPUREC_ALERAX_COMPAT=1 python - <<'PY'
import torch
from pathlib import Path
from gpurec import GeneReconModel

root = Path("tests/data/test_trees_20")
model = GeneReconModel.from_trees(
    species_tree=str(root / "sp.nwk"),
    gene_trees=[str(root / "g_0000.nwk")],
    mode="global",
    pibar_mode="uniform",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float32,
    theta_init_rates=(0.05, 0.05, 0.05),
)

with torch.no_grad():
    print(float(model.nll().item()))

try:
    model.nll()
except NotImplementedError as exc:
    print(str(exc).split(".")[0])
PY
```

This confirms:

- no-grad compatibility likelihood evaluation works;
- gradient-enabled evaluation is blocked.

## Current Limitations

Compatibility mode currently supports only:

- global/shared DTL rates;
- uniform transfer mode;
- no-grad likelihood evaluation.

It does not yet support:

- genewise rates;
- specieswise rates;
- dense or top-k transfer modes;
- differentiating the AleRax-compatible objective with the existing custom
  backward.

Those limitations are deliberate. Extending the mode is straightforward in
principle, but silently mixing the AleRax-compatible forward with the existing
implicit backward would be worse than failing loudly.

## Bottom Line

The main problem was a likelihood-semantics mismatch:

```text
GPUREC default: converged fixed-point likelihood
AleRax v1.4:    fixed four-pass in-place likelihood
```

The new `GPUREC_ALERAX_COMPAT=1` path implements the AleRax-style evaluator for
fixed-parameter likelihood checks. It collapses the large high-rate divergence
for the tested HOGENOM families, and for the remaining pathological transfer
case, an independent implementation of the source-level recurrence matches
GPUREC rather than the installed AleRax binary.

The next engineering step, if exact AleRax-style optimization is required, is
to add a correct differentiable backward for the fixed four-pass objective or
use an explicit unrolled differentiable implementation for that compatibility
mode.

# Shared-EM2 hierarchical continuation result

## Gate and implementation

The experiment starts all arms at the point-consistent float32-evaluated EM2 endpoint in
`em/hybrid_shared_200_v2.pt`. Native uses its independently derived native complete-information
seed; the two hierarchical arms use the direct `(u,v,w)` diagonal complete-information seed and
the same transformed EM gradient secant. The two hierarchical choices differ only in the trust
metric. Likelihood, original rate box, native projected-gradient stopping (`1e-3`), BFGS update,
refresh 15, trust acceptance/rejection, replanning, and certification remain production logic.

`source_adapter.py` compiles a private in-memory continuation from the current production function
and fails unless each intended substitution occurs exactly once. Native is a direct call to the
unmodified production function. The mirrored production source SHA-256 was
`7d398d3e596f150c7eb5fc36f5eccce3b94b7dd14561c337bbbf5d7363409cd5`.

The CPU suite has 12 passing tests, including the full Hessian chain rule, all eight authoritative
active masks, Lagrangian face curvature, model-dtype bound roundoff, both trust metrics, and a
deterministic coupled-bound case where the initial Newton direction points out of a lower native
face despite an inward gradient. A step-only monotone working set repairs that case. It is distinct
from projected-gradient/freeze/certification logic. Five-family CUDA gates covered ordinary steps
and the scheduled exact-Hessian refresh; all three arms remained finite and bounded.

## First 200-family paired run

All arms ran from the same saved endpoint and paid the same two EM gradient passes. “Gradient-clade
equivalents” includes those two shared passes and every resident family in each continuation
gradient, including settled-but-not-replanned rows. It does not turn the analytic-Hessian kernel
into a guessed gradient-pass multiplier; the table reports its measured time separately, while
wall time includes it.

| arm | wall s | gradient-clade eq | Hessian s / rounds | builds | NLL reported by fit | fit certificate |
|---|---:|---:|---:|---:|---:|---:|
| native (ran first/cold) | 26.025 | 13.5207 | 0.789 / 1 | 10 | 613261.801129 | 200/200 |
| hierarchy, native metric | 25.433 | 13.9118 | 0.604 / 1 | 6 | 613261.798351 | 200/200 |
| hierarchy, coordinate metric | 26.402 | 14.0257 | 0.644 / 1 | 8 | 613261.813913 | 200/200 |

The hierarchy’s apparent first-run wall advantage is confounded by native running cold and by its
four additional rebuilds. It does not represent less likelihood work: hierarchical-native costs
0.3911 extra full gradient-clade equivalents and coordinate metric costs 0.5050 extra.

The independent fresh common-model audit gives:

| endpoint | fresh NLL | fresh `Pg<1e-3` count | fresh max Pg |
|---|---:|---:|---:|
| native | 613261.799799 | 194 | 0.002157 |
| hierarchy, native metric | 613261.796900 | 195 | 0.001237 |
| hierarchy, coordinate metric | 613261.814100 | 190 | 0.001844 |
| identical native repeat | 613261.799799 | 195 | 0.001989 |

The native repeat has bit-identical NLL but a slightly different projected gradient, demonstrating
that fresh FP32 pruning/gradient noise crosses the `1e-3` threshold for a few endpoint rows. The
native-metric hierarchy is 0.002899 bits lower than native, with no family differing by 0.01 bits
and maximum per-family absolute difference 0.001356 bits. Coordinate metric is 0.014301 bits worse,
including one 0.018651-bit family regression.

## Reverse-order warmed repeat

| arm | order | wall s | gradient-clade eq | Hessian s / rounds | builds | NLL reported by fit | fit certificate |
|---|---:|---:|---:|---:|---:|---:|---:|
| hierarchy, native metric | first | 28.122 | 14.0899 | 1.045 / 1 | 8 | 613261.797115 | 200/200 |
| native | second | 24.894 | 13.5590 | 0.452 / 1 | 10 | 613261.802217 | 200/200 |

The reversed ordering removes the putative wall advantage. Across the two paired runs,
hierarchical-native costs 0.35–0.53 additional gradient-clade equivalents (about 2.6–3.9% of the
native total) despite fewer rebuilds. Both hierarchical runs have zero nonconverged zero-direction,
zero-ray, near-zero-ray, boundary-zero, and model-rounded native-zero diagnostics; minimum ray
fractions were about 0.0448. Thus the result is not explained by the known false-alpha-zero bug.

The two-run means are 25.459 s and 13.5399 gradient-clade equivalents for native versus 26.777 s
and 14.0009 for hierarchical-native: 5.2% more wall time and about 3.4% more charged gradient work.
The separate audit of the reverse-run tensors again found equivalent endpoint quality:
613261.798281 bits for hierarchical-native versus 613261.800815 for native (−0.002534 bits), maximum
per-family absolute difference 0.001358 bits, and no family difference above 0.01 bits. The native
repeat was NLL-identical. Its fresh certificate count differed by one (198 versus 197) and the
hierarchical count was 192, consistent with threshold-level FP32 projected-gradient variability;
these audit calls are quality checks outside fit timing, not replacements for the fit certificates.

## Decision and limitations

The native-metric hierarchy is robust and reaches essentially the same basin, but does not reduce
charged likelihood work on 200 families. Coordinate metric is both more expensive and has a small
NLL regression. Neither is a clear candidate for a 500-family or H100 campaign on this evidence.
This is a bounded post-EM2 result; it corrects the earlier report, whose negative coordinate test
started post-Adam and could not answer the hybrid question.

The monotone working set guarantees a feasible tested direction but is not an exact inequality QP:
it does not release a promoted face during one solve and can overconstrain a corner. Any future run
with a nonconverged zero/near-zero diagnostic would invalidate a negative geometry conclusion and
would require a releasing active-set method. Endpoint ray bisection tests feasibility, not a
globally first face crossing, because the nonlinear native image of a phi ray need not be monotone.

Reproduce the CPU tests and the paired comparison with:

```bash
cd experiments/coleman_sol_20260906/geometry/hybrid
PYTHONPATH=../../../.. ../../../../.venv/bin/python -m pytest -q test_hierarchical_adapter.py

cd ../../../..
PYTHONPATH=. .venv/bin/python experiments/coleman_sol_20260906/geometry/hybrid/run_shared_endpoint.py \
  --limit 200 --max-iter 200 --check-every 2 --pi-tiers 16,64 --certify \
  --out experiments/coleman_sol_20260906/geometry/hybrid/comparison200.json
```

The JSON records timing/work/certification and ray diagnostics. The sibling PT stores every arm’s
theta, curvature, history, work ledger, and diagnostic tensors. `comparison200_audit.json` is the
separate common-model audit; logs and the reverse-order JSON/PT are preserved beside it.

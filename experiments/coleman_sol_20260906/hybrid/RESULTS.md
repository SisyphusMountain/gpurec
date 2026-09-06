# Matched post-EM hierarchical-coordinate experiment

The user's correction was right: the previous post-Adam experiment did not
test this hybrid. We have now tested it from one shared EM2 endpoint, using
the same two GPT-5.6 Sol workers and an independently reviewed continuation
adapter. This implementation did not demonstrate a performance benefit in
the 200-family gate. It was not promoted to 500 families or a full H100 run.
That is a bounded empirical result, not a dismissal of every hierarchical
optimizer or evidence about untested full-dataset performance.

Follow-up diagnosis: [matched traces and first-step probes](diagnosis/REPORT.md)
identify a lagged opening curvature seed and a rebuild-threshold interaction.
Fresh-pair recalibration greatly improves the first proposal, but has not yet
been tested as a full continuation or installed in production. Bounds are not
the main loss in the traced cohort. The benchmark results below remain unchanged.

## Comparison contract

The first 200 Coleman families contain 1,491,100 clades. Every arm receives
exactly the same FP32-evaluated EM2 endpoint. Native uses its own complete
information plus native secant seed. Hierarchical coordinates use directly
constructed diagonal complete-history information and their own consistently
transformed gradient pair, never the native BFGS matrix. The shared artifact
and seed validations are linked from [PLAN.md](PLAN.md).

The likelihood, FP32 model/FP64 accumulation, pruning threshold 1e-6, native
rate box [1e-6,2], projected-gradient tolerance 1e-3, freeze/replan policy,
rejection policy, and exact-Hessian refresh every 15 steps are unchanged.
Native calls the unmodified production optimizer. Hierarchical exact refreshes
include the nonlinear gradient-times-map-curvature term. All five fits below
took one scheduled exact Hessian refresh and certified 200/200 at freeze time.

The primary hierarchical arm measures trust steps in native physical units.
One sensitivity arm instead measures the tangent step in hierarchical units;
both enforce the same native box. The adapter corrects curvature for held
native faces and maintains a separate step working set to avoid an outward
coupled direction collapsing the whole feasible ray. Working faces never
replace the native stopping/certificate mask.

## Results on the local RTX4090

Wall times are **continuation-only**, including each continuation's parse,
builds, curvature, and certification. Work includes the common two EM passes
and all resident clades in subsequent gradient calls. Hessian work is separate
from gradient equivalents; its time is included in wall time.

| Run order | Arm | Continuation seconds | Gradient/clade equivalents incl. EM | Builds |
| --- | --- | ---: | ---: | ---: |
| A, first | Native log rates | 26.025 | 13.5207 | 10 |
| A, second | Hierarchical, native metric | 25.433 | 13.9118 | 6 |
| A, third | Hierarchical, coordinate metric | 26.402 | 14.0257 | 8 |
| B, first | Hierarchical, native metric | 28.122 | 14.0899 | 8 |
| B, second | Native log rates | 24.894 | 13.5590 | 10 |

The order-balanced two-run means are 25.459 s / 13.5399 equivalents for native
and 26.777 s / 14.0009 for hierarchical/native-metric: the latter measured
5.2% more continuation time and 3.4% more total gradient/clade work. Its fewer
builds did not offset the extra gradient work consistently. Two repetitions
do not establish a precise timing distribution, but they do not support
promoting the initial single-run 2.3% apparent advantage as a speedup.

The shared EM artifact cost 5.929 s including parse/build and two reverse
passes (4.601 s). Adding it equally gives prototype means 31.388 s versus
32.706 s. These are not integrated end-to-end production timings: artifact
generation and continuation each construct a model. No EM cost is omitted
from a claimed production speedup, because no such speedup is claimed here.

There were no nonconverged zero directions, zero/near-zero rays, boundary
zero rays, or actual-FP32-zero steps in either hierarchical arm, including
the repeated primary arm. The minimum ray fractions in run A were 0.0448
and 0.000225 for the native and coordinate metrics respectively. Thus the
known complete-step stall was repaired before interpreting these results.

## Fresh quality audit

A fresh common model, outside fit time, evaluated all run-A endpoints and
repeated the identical native endpoint. Hierarchical/native-metric changed
total NLL by -0.002899 bits; no family changed by more than 0.001357 bits.
Its quality is effectively equivalent at this scale. The coordinate-metric
arm was +0.014301 bits worse overall, with one family +0.018651 bits worse
(COG0006_2); it has no favorable work or likelihood tradeoff in this screen.

Fresh strict certificates were 194/200 native, 195/200 hierarchy/native-metric,
and 190/200 hierarchy/coordinate-metric. Repeating identical native parameters
gave exactly the same NLL but 195/200 fresh certificates, with projected-gradient
changes up to 0.000673. As in the full H100 audits, freeze-time certification
is not a claim of strict fresh or unpruned stationarity. These fresh counts
do not replace the common optimizer stopping rule.

The separate reverse-order audit confirms equivalent quality: hierarchical
NLL was -0.002534 bits relative to native, with maximum per-family difference
0.001359 bits and no change above 0.01 bits. Identical native parameters again
repeated their NLL exactly. Fresh passing counts were 192 hierarchy, 198 native,
and 197 native-repeat; these counts remain threshold-sensitive, not evidence
that either fit satisfies a strict fresh 200/200 certificate.

## Decision and limitations

Retain the validated EM2/EM3 warm-up followed by native log-rate BFGS/Newton:
its full 5,124-family H100 results remain about 400 seconds (best single run
396.305 s). Keep the hierarchical implementation and artifacts experiment-only.
No production coordinate change or default change was made in this follow-up.

The monotone per-step working set is a conservative feasible heuristic, not
an exact tangent-cone QP with multiplier-based face releases. It can constrain
a step more than necessary even when the recorded stall diagnostics are zero.
A stronger bound solver, different coordinate-aware globalization, or a full
dataset can behave differently. The evidence supports "this tested hybrid
did not improve the screening result," not "EM cannot help reparameterization."

## Evidence and reproduction

- [Independent adapter review](../em/HYBRID_ADAPTER_REVIEW.md): endpoint,
  rollback, bounds, exact refresh, certificate, and accounting review.
- [Run A](../geometry/hybrid/comparison200.json),
  [reverse-order run B](../geometry/hybrid/comparison200_reverse.json),
  [run-A fresh audit](../geometry/hybrid/comparison200_audit.json), and
  [run-B fresh audit](../geometry/hybrid/comparison200_reverse_audit.json).
  Corresponding `.pt` and `.log` files retain endpoints, curvature, histories,
  per-family audit vectors, and execution traces.
- Twelve CPU geometry tests pass; an independent 256-case retraction/Hessian
  autodiff check has maximum curvature discrepancy 4.0e-15. A tiny 20-pass
  GPU smoke exercised the scheduled exact refresh in all three arms.

Run from the repository root with the saved V2 artifact:

```bash
PYTHONPATH=. .venv/bin/python experiments/coleman_sol_20260906/geometry/hybrid/run_shared_endpoint.py \
  --limit 200 --max-iter 200 --check-every 2 --pi-tiers 16,64 --certify \
  --out /tmp/hybrid_comparison.json
PYTHONPATH=. .venv/bin/python experiments/coleman_sol_20260906/geometry/hybrid/run_shared_endpoint.py \
  --limit 200 --max-iter 200 --check-every 2 --pi-tiers 16,64 --certify \
  --arms hierarchical_native_metric,native --out /tmp/hybrid_reverse.json
```

Use fresh output names to preserve earlier results. Current exact elimination
internally collapses the legacy accuracy-tier tuple to its first tier for all
arms; the source adapter preserves that production behavior.

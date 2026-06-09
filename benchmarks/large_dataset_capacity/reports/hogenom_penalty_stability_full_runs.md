# HOGENOM Full-Dataset Penalty Stability Experiments

Date: 2026-06-09

## Scope

These runs extend the bounded CV pilots to the full 1055-family HOGENOM specieswise objective.  All runs use the normal forward fixed-point solver with `pi_iters=16`, plus backward `self_loop_solver=gmres` and `neumann_terms=16`.

The scalar 80-family CV pilots selected `lambda=0.01` by held-out raw NLL, but the full-dataset continuation showed that this penalty is too weak for long optimization: rates reached the configured `2.0` upper clamp after about 120 total Adam steps.

## Main Full-Dataset Runs

Rows below resumed from the same clean 20-step checkpoint unless marked as continuations:

`benchmarks/large_dataset_capacity/output/hogenom_specieswise_map/20260609_012044/theta_final.pt`

| Run | Penalty | Steps | LR | Final raw NLL | Final objective | Rate max | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `20260609_012644` | scalar `0.01` | 60 | 0.04 | 547822.9375 | 547974.3125 | 1.223 | Best raw NLL before cap pressure; still drifting upward. |
| `20260609_013800` | scalar `0.01` continuation | 40 | 0.02 | 538758.5000 | 538944.9375 | 2.000 | Hit upper rate clamp; rejected as unstable. |
| `20260609_014433` | scalar `0.1` | 60 | 0.04 | 547737.8750 | 549110.0625 | 1.228 | Similar drift to `0.01`; too weak. |
| `20260609_015404` | scalar `1.0` | 60 | 0.04 | 548411.4375 | 558691.2500 | 1.289 | Still drifts, with worse objective. |
| `20260609_020331` | scalar `10.0` | 60 | 0.04 | 579161.4375 | 609076.3125 | 1.317 | More stable p95 rates but large raw-NLL cost. |
| `20260609_021922` | scalar `10.0` continuation | 40 | 0.01 | 578258.3125 | 608245.3125 | 1.762 | Still drifting toward cap. |
| `20260609_021252` | scalar `100.0` | 40 | 0.02 | 628302.0000 | 721036.7500 | 0.262 | Stable but over-regularized; raw NLL worsened. |
| `20260609_023256` | event `D,T,L=(0.1,10,10)` | 60 | 0.04 | 576069.0625 | 604115.3125 | 1.300 | Best stability/likelihood compromise tested. |
| `20260609_025255` | event `D,T,L=(0.1,10,10)` continuation | 40 | 0.01 | 574787.6875 | 603194.8750 | 1.736 | Objective still improves, but tail rate continues upward. |
| `20260609_024226` | event `D,T,L=(0.1,10,30)` | 60 | 0.04 | 578943.8750 | 610305.3750 | 1.310 | Stronger loss prior did not reduce max; worse raw NLL. |
| `20260609_022551` | scalar `30.0` | 40 | 0.02 | 615217.3125 | 651586.9375 | 0.289 | Stable but raw NLL degraded. |
| `20260609_031202` | tree/root `D,T,L=(1,10,10)` + weak L2 | 60 | 0.04 | 553611.0625 | 559636.8125 | 1.259 | Best objective among stable tree-prior candidates. |
| `20260609_032300` | tree/root `D,T,L=(1,10,10)` continuation | 40 | 0.01 | 549579.1875 | 555980.0000 | 1.645 | Objective improves, but transfer tail keeps climbing. |
| `20260609_032930` | tree/root `D,T,L=(1,30,10)` + weak L2 | 60 | 0.04 | 559326.6875 | 567359.2500 | 1.172 | Stronger transfer smoothing improves tail control. |
| `20260609_033920` | tree/root `D,T,L=(1,30,10)` continuation | 40 | 0.01 | 555893.3125 | 563950.1250 | 1.509 | Initial selected tail-control/likelihood compromise. |
| `20260609_035727` | tree/root `D,T,L=(1,30,10)` guarded continuation | 30 | 0.003 | 555248.0000 | 563314.3125 | 1.603 | Improved objective under `--stop-rate-max 1.8`. |
| `20260609_040244` | tree/root `D,T,L=(1,30,10)` guarded continuation | 30 | 0.003 | 554667.3750 | 562724.4375 | 1.701 | Best current selected run; still below the guard. |

## Recommendation

Use the unit-branch tree/root GBM-style prior in log2-rate space, with a weak eventwise L2 anchor:

```text
weak L2 lambda_D,T,L = 0.01,0.01,0.01
tree/root lambda_D,T,L = 1,30,10
target rate = 0.05
```

This is a pragmatic MAP prior rather than the short-CV scalar-L2 optimum.  The scalar `0.01` prior won early held-out NLL but failed the full-run stability check by driving rates into the upper clamp.  A topology-based GBM-style penalty is a better fit for specieswise parameters because it directly penalizes rough species-to-parent log-rate jumps without forcing every species rate back to a single global value.

The `D,T,L=(1,30,10)` tree/root prior gives a better tail-control/likelihood tradeoff than scalar/eventwise L2: after 160 continuation steps from the clean 20-step checkpoint it reached raw NLL `554667.3750`, objective `562724.4375`, and rate max `1.701`, still below the `1.8` continuation guard and the hard `2.0` cap.  The looser `D,T,L=(1,10,10)` tree prior gives a better objective earlier (`555980.0000`) but a higher tail rate (`1.645`) after only 100 continuation steps.

Forward Pi GMRES diagnostics for the selected full-dataset theta were refreshed after the Triton forward-GMRES apply fix in `benchmarks/large_dataset_capacity/output/hogenom_specieswise_map_diagnostics/20260609_101618/summary.json`: across all batches, 41 of 46 attempted forward GMRES waves were accepted and 5 batches fell back to the fixed-point forward path after an invalid GMRES wave.  The selected run still evaluated to raw NLL `554667.3750`, objective `562724.4375`, and max rate `1.7011`.  This verifies substantial forward GMRES use with objective-preserving fallback, but not pure GMRES on every wave.

Final selected tree-prior rate summaries for `20260609_040244`:

| Event | min | median | mean | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| D | 0.00600 | 0.05322 | 0.10025 | 0.35713 | 1.27102 |
| T | 0.06141 | 0.54100 | 0.60607 | 1.28108 | 1.70110 |
| L | 0.00621 | 0.10676 | 0.21111 | 0.86812 | 1.49317 |

The remaining high rates are still tail values, not a fully converged MAP optimum.  A production run should monitor the tail and stop or retune before the hard upper bound is reached.

A lower-learning-rate continuation of the earlier eventwise L2 prior `(0.1,10,10)` reduced the MAP objective (`604115.3125 -> 603194.8750`) but raised the maximum rate (`1.300 -> 1.736`).  The selected tree/root prior is materially better on objective and comparable or better on tail control.

## Commands

Representative selected continuation:

```bash
python benchmarks/large_dataset_capacity/hogenom_specieswise_map_optimize.py \
  --steps 30 --lr 0.003 --stop-rate-max 1.8 \
  --penalty-lambda 0.01 --penalty-lambdas 0.01,0.01,0.01 \
  --tree-penalty-lambdas 1,30,10 --root-penalty-lambdas 1,30,10 \
  --pi-solver gmres --self-loop-solver gmres \
  --pi-iters 16 --neumann-terms 16 \
  --init-theta benchmarks/large_dataset_capacity/output/hogenom_specieswise_map/20260609_035727/theta_final.pt \
  --output-root benchmarks/large_dataset_capacity/output/hogenom_specieswise_map
```

Earlier scalar CV reports:

- `benchmarks/large_dataset_capacity/reports/hogenom_penalty_cv_subagent.md`
- `benchmarks/large_dataset_capacity/reports/hogenom_penalty_cv_long_stability.md`

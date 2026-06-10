# Archaea Genewise Optimization Experiments

Date: 2026-06-09

## Best result so far

The best practical setup so far is **hierarchical EB + unbounded rates + Rprop**, with the final optimization phase run in `float64` and the backward self-loop solver set to `gmres`.

It converged cleanly on the 256-family complete-CCP archaeal screen. The Adam reset/ramp/beta sweep below did not produce a converged Adam run, so Rprop remains the best-supported optimizer for this screen.

| Setup | Converged | Total objective | Data loss | Prior loss | Final theta grad | Rate range | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hierarchical EB, Rprop, fp64 GMRES finish | yes | 5803.7300488 | 5286.1745989 | 517.5554499 | 4.57e-05 | 0.1180..0.8595 | 63.0 s |
| Hierarchical EB, old default Adam, fp64 GMRES finish | yes | 5803.7300648 | 5286.1979681 | 517.5320967 | 7.22e-03 | 0.1180..0.8595 | 49.4 s |
| Hierarchical EB, tuned RMSProp, fp64 GMRES finish | yes | 5803.7300766 | 5286.1737399 | 517.5563367 | 8.90e-03 | 0.1180..0.8595 | 69.6 s |
| Hierarchical EB, Adafactor, fp64 GMRES finish | no | 5803.9519411 | 5286.2554630 | 517.6964781 | 6.81e-01 | 0.1170..0.8514 | 56.6 s |

Rprop is the best current choice because it reaches the same EB optimum as old default Adam/RMSProp but with a much smaller final theta gradient and a flatter loss tail. The new tuned Adam reset/ramp runs did not converge. Tuned RMSProp can work, but it is more sensitive to learning-rate scheduling.

Current recommendation:

1. Use Rprop for the robust baseline.
2. Use final `float64` + backward self-loop GMRES.
3. Do not treat the old default Adam result as evidence for tuned Adam.
4. If continuing adaptive-optimizer work, use phase-boundary optimizer resets and LR ramps, but do not promote Adam from the current 256-family screen.

## Scope

The top genewise optimizer comparison uses:

- Dataset root: `tests/data/alerax_archaea_davin2017`
- Input: complete ALE CCP files, parsed directly from `.ale`
- Mode: `genewise`
- Families: 256 selected families, ordered by smallest eligible families
- Leaf filter: exclude families with fewer than 4 leaves
- Rate bounds: disabled for EB/unbounded experiments
- Backtracking: enabled only in the final Rprop run; optimizer screen timings usually used `--backtrack-families 0`

The 256-family screen is not a whole-dataset result. It is enough to compare optimizer behavior, not enough to claim final production-scale performance over every eligible archaeal family.

The specieswise experiment later in this document is a whole-dataset run over all 5,379 eligible archaeal families.

## Solver And Transition Schedule

Corrected hierarchical-EB runs used this solver schedule:

| Steps | Pi iterations | Self-loop iterations | Dtype | Backward self-loop solver |
|---:|---:|---:|---|---|
| 40 | 4 | 4 | float32 | neumann |
| 40 | 8 | 8 | float32 | neumann |
| 40 | 12 | 12 | float32 | neumann |
| 120 | 16 | 16 | float64 | gmres |

In this code path, `self_loop_solver=gmres` means the backward/adjoint self-loop solve uses GMRES. The forward likelihood solve is still the normal forward path.

I updated `scripts/optimize_alerax_archaea_genewise_adam.py` so the schedule can specify the solver per phase:

```text
STEPS:PI_ITERS:SELF_LOOP_ITERS:DTYPE:SELF_LOOP_SOLVER
```

and also per-phase optimizer LR:

```text
STEPS:PI_ITERS:SELF_LOOP_ITERS:DTYPE:SELF_LOOP_SOLVER:OPTIMIZER_LR
```

## Optimizer Transition Policy

Optimizer state is now reset at schedule phase boundaries by default in `scripts/optimize_alerax_archaea_genewise_adam.py`. This matters for Adam/RMSProp/Adafactor-like methods because changing Pi iterations, self-loop iterations, dtype, or solver changes the gradient scale and makes old accumulator statistics stale.

The previous behavior can be recovered with:

```text
--preserve-optimizer-state-across-phases
```

The script also supports a phase-entry LR ramp:

```text
--lr-ramp-steps N --lr-ramp-start-factor F
```

For example, with `--lr-ramp-steps 3 --lr-ramp-start-factor 0.2` and a phase target LR of `0.01`, the phase starts with `0.002`, then `0.006`, then reaches `0.01`.

It also supports an explicit phase-entry LR decay:

```text
--lr-decay-steps N --lr-decay-end-factor F
```

With decay enabled, the schedule LR is the start LR for the phase, and the LR decays linearly over the first `N` phase steps to `F * start_lr`.

A smoke test confirmed the JSON history records the transition correctly:

```text
step phase phase_step lr    target_lr reset
0    0     0          0.05  0.05      false
1    0     1          0.05  0.05      false
2    1     0          0.002 0.01      true
3    1     1          0.006 0.01      true
4    1     2          0.01  0.01      true
```

The intended adaptive-optimizer pattern is:

```text
fp32 warmup phases: larger LR, fresh statistics at each Pi/Neumann transition
fp64 GMRES finish: lower target LR, fresh statistics, short LR ramp
```

For RMSProp, the best pre-reset/pre-ramp phase-LR result suggests a starting point:

```text
alpha = 0.99
fp32 target LR = 0.05
fp64 GMRES target LR = 0.01
try --lr-ramp-steps 5 to 10
```

For Adam/Adamax/Adafactor, the important variable is the second-moment memory. Defaults such as Adam `beta2=0.999` are designed for much longer stochastic training horizons and are probably inappropriate for a 100-200 step deterministic optimization. The Adam sweep below tested shorter beta memories with phase resets and LR ramps, but none converged on the 256-family screen.

## Adam Reset/Ramp/Beta Sweep

This is the reset/ramp Adam sweep on the same 256-family genewise complete-CCP screen as the top optimizer comparison. It is **not** a whole-dataset result. Common controls:

```text
mode: genewise
families: 256 smallest eligible families
min leaves: 4
hierarchical EB: enabled
unbounded/unprojected: enabled
backtracking: disabled
schedule: 40:4:4:float32:neumann:LR32,40:8:8:float32:neumann:LR32,40:12:12:float32:neumann:LR32B,120:16:16:float64:gmres:LR64
lr ramp: --lr-ramp-steps 10 --lr-ramp-start-factor 0.2
optimizer state: reset at every non-first phase boundary
```

Output JSONs are under `output/alerax_archaea_genewise_adam/` with `adam_reset_ramp` and the beta values in the filename.

| Run | Betas | LR32 / LR32B / LR64 | Converged | Final objective | Best objective | Tail slope | Final joint grad | Data loss | Prior loss | Rate min..max (mean) | Time | Resets | LR ramp visible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `adam_b09_b099` | 0.9 / 0.99 | 0.03 / 0.03 / 0.003 | no | 5813.6374962 | 5813.6374962 | -0.107312 | 3.07963 | 5295.7180343 | 517.9194619 | 0.108090..0.701354 (0.309205) | 51.4 s | yes | yes |
| `adam_b05_b09` | 0.5 / 0.90 | 0.01 / 0.005 / 0.001 | no | 6209.5015289 | 6209.5015289 | -0.404998 | 56.7453 | 5739.6915990 | 469.8099299 | 0.050782..0.105806 (0.085812) | 45.1 s | yes | yes |
| `adam_b05_b095` | 0.5 / 0.95 | 0.01 / 0.005 / 0.001 | no | 6210.6627336 | 6210.6627336 | -0.407547 | 60.8264 | 5740.4755089 | 470.1872247 | 0.050299..0.106672 (0.085710) | 43.6 s | yes | yes |
| `adam_b07_b095` | 0.7 / 0.95 | 0.02 / 0.01 / 0.002 | no | 5958.7045391 | 5958.7045391 | -0.347067 | 13.3267 | 5479.5015629 | 479.2029763 | 0.083912..0.229507 (0.156667) | 47.8 s | yes | yes |

Verification details:

- Every JSON records `optimizer_reset_at_phase_start: true` for phases 1, 2, and 3.
- The first 10 steps of each post-transition phase ramp from `0.2 * target_lr` to `target_lr`.
- Example ramp endpoints: `adam_b09_b099` uses phase ramps `0.006 -> 0.03`, `0.006 -> 0.03`, and `0.0006 -> 0.003`; `adam_b07_b095` uses `0.004 -> 0.02`, `0.002 -> 0.01`, and `0.0004 -> 0.002`.

The old default Adam result remains a separate pre-reset/pre-ramp result:

```text
old default Adam total:      5803.7300648
old default Adam theta grad: 7.22e-03
```

None of the reset/ramp beta-tuned Adam runs beat or match Rprop by the gradient criterion. The best tuned Adam run by final objective, `adam_b09_b099`, stopped at final joint gradient `3.08`, while Rprop reached final theta gradient `4.57e-05` on the same 256-family screen. Since no tuned Adam configuration converged cleanly, I did not run an Adam specieswise all-family continuation from the strong-prior specieswise checkpoint.

## Adam High-LR Decay Diagnostic

The low-LR reset/ramp sweep above looked underpowered: several runs were still descending fast with high gradients. I therefore ran a separate diagnostic using Adam `beta1=0.9`, `beta2=0.99`, starting each phase with a high LR and decaying over the first 10 phase steps. This is still the same 256-family genewise screen, not a whole-dataset result.

Common controls:

```text
mode: genewise
families: 256 smallest eligible families
hierarchical EB: enabled
unbounded/unprojected: enabled
schedule: 40:4:4:float32:neumann:LR,40:8:8:float32:neumann:LR,40:12:12:float32:neumann:LR,120:16:16:float64:gmres:LR
lr decay: --lr-decay-steps 10
backtracking: disabled
```

Results:

| Run | Start -> steady LR | Converged | Final objective | Best objective | Tail slope | Final joint grad | Final theta grad | Data loss | Prior loss | Rate min..max (mean) | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `adam_highdecay_0p5_end0p05` | 0.5 -> 0.05 | no | 5803.7427512 | 5803.7313992 | 9.9531e-05 | 0.384513 | 0.351331 | 5286.2928534 | 517.4498978 | 0.118061..0.858990 (0.391327) | 80.0 s |
| `adam_highdecay_1p0_end0p1` | 1.0 -> 0.1 | no | 5803.7310479 | 5803.7310479 | -0.000418112 | 0.288895 | 0.051416 | 5286.1074250 | 517.6236230 | 0.117936..0.859888 (0.391515) | 150.8 s |
| `adam_highdecay_1p0_end0p05` | 1.0 -> 0.05 | no | 5803.7321424 | 5803.7318390 | -0.000708011 | 0.155694 | 0.096871 | 5286.3463232 | 517.3858192 | 0.117786..0.860522 (0.391454) | 147.7 s |
| `adam_highdecay_2p0_end0p2` | 2.0 -> 0.2 | no | 5803.7401757 | 5803.7393987 | -0.00252255 | 0.490862 | 0.225537 | 5286.2042839 | 517.5358918 | 0.117680..0.858392 (0.391603) | 344.1 s |

Interpretation:

- The high-start schedules support the hypothesis that the earlier Adam LRs were too low. The `1.0 -> 0.1` run reached the EB optimum basin and a near-flat tail, far better than the low-LR reset/ramp runs.
- None of these high-LR decay runs converged by the joint-gradient criterion. The best final joint gradient was `0.155694` for `1.0 -> 0.05`, still above the `0.05` threshold and much larger than Rprop.
- `2.0 -> 0.2` was too aggressive under this decay factor. It produced large phase-transition spikes and took much longer per run.
- The promising region is probably a high start near `1.0`, followed by stronger or longer decay in the fp64 phase, rather than simply holding `0.1` or `0.05` steady after 10 steps.

## Adam fp32 Neumann Max-12 Schedule

I then removed the fp64/GMRES finish and capped the solver schedule at `float32`, 12 Pi iterations, 12 Neumann terms:

```text
20:4:4:float32:neumann,
40:8:8:float32:neumann,
140:12:12:float32:neumann
```

All runs below are the same 256-family genewise screen with hierarchical EB, unbounded rates, no backtracking, and phase-entry LR decay.

| Run | Betas | Schedule | Start -> steady LR | Converged | Final objective | Best objective | Tail slope | Final joint grad | Final theta grad | Final prior grad | Data loss | Prior loss | Rate min..max | Time |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `adam_b09_b099_end0p05` | 0.9 / 0.99 | 20x4 + 40x8 + 140x12 | 1.0 -> 0.05 | no | 5803.7314453 | 5803.7304688 | -2.441e-05 | 0.054208 | 0.015767 | 0.051864 | 5286.1621094 | 517.5691528 | 0.118044..0.859262 | 74.6 s |
| `adam_b09_b099_end0p025` | 0.9 / 0.99 | 20x4 + 40x8 + 140x12 | 1.0 -> 0.025 | no | 5803.7514648 | 5803.7514648 | -0.002651 | 0.401592 | 0.070580 | 0.395342 | 5286.8750000 | 516.8765259 | 0.119498..0.865122 | 82.6 s |
| `adam_b09_b095_end0p05` | 0.9 / 0.95 | 20x4 + 40x8 + 140x12 | 1.0 -> 0.05 | no | 5803.7309570 | 5803.7304688 | -7.888e-05 | 0.108428 | 0.026527 | 0.105132 | 5286.1245117 | 517.6065063 | 0.118040..0.859349 | 75.6 s |
| `adam_b07_b095_end0p05` | 0.7 / 0.95 | 20x4 + 40x8 + 140x12 | 1.0 -> 0.05 | no | 5803.7353516 | 5803.7299805 | 0.002172 | 1.33878 | 0.103941 | 1.33474 | 5286.1171875 | 517.6179810 | 0.118056..0.859367 | 82.2 s |
| `adam_b09_b099_split_lowfinish` | 0.9 / 0.99 | 20x4 + 40x8 + 120x12 + 20x12 | 1.0 -> 0.05, then 0.05 -> 0.0025 | no | 5803.8251953 | 5803.7309570 | -0.014537 | 2.61386 | 0.494802 | 2.56660 | 5286.2280273 | 517.5972290 | 0.117140..0.863088 | 76.0 s |

Interpretation:

- The requested fp32/Neumann max-12 schedule almost converged with Adam `0.9/0.99`, `1.0 -> 0.05`: final theta gradient was already low (`0.0158`), tail slope was flat, and final joint gradient missed the `0.05` threshold narrowly at `0.0542`.
- The remaining blocker was the EB hyperparameter gradient, not genewise theta.
- Lowering the steady LR to `0.025` was too slow within the fixed 140-step max-12 tail.
- Shorter Adam memory (`beta2=0.95`) made the final EB hyperparameter gradient worse.
- Splitting the 140-step max-12 block into `120 + 20` with a low-LR finish introduced a same-solver optimizer reset and moved away from the best point.

This is the best Adam result so far under the requested no-fp64/no-GMRES cap, but it is still technically unconverged by the joint-gradient criterion.

## Hierarchical EB behavior

The EB model places a learned hierarchical normal prior on genewise log2 D/T/L rates:

- learned population means: `mu_D, mu_T, mu_L`
- learned population sigmas: `sigma_D, sigma_T, sigma_L`
- lower sigma floor: `prior_min_sigma = 0.1`
- stable log-sigma hyperprior used here: `prior_log_sigma_sigma = 0.05`

Final Rprop EB hyperparameters:

```text
prior_mu    = [-1.4566127, -0.8101932, -2.4828398]
prior_sigma = [ 1.2486479,  1.0954681,  1.1783685]
```

The weak sigma hyperprior was bad. With `prior_log_sigma_sigma = 1.0`, Rprop collapsed all sigmas to the floor:

```text
prior_sigma = [0.1, 0.1, 0.1]
```

That made the prior term strongly negative and dominated the objective:

```text
total = 3281.1014
data  = 5810.8264
prior = -2529.7250
```

This is a joint MAP/EB pathology, not a useful fit. The tighter log-sigma hyperprior avoids that collapse.

## Bounded vs unbounded vs EB

The best bounded unregularized optimizer result was:

| Setup | Converged | Objective | Final theta grad | Rate range |
|---|---:|---:|---:|---:|
| Bounded Rprop, no EB | yes | 4959.2418168 | 2.28e-03 | 1e-10..2.0 |

That has the best likelihood value, but it hits both artificial rate bounds. It is not the answer to the divergence problem; it is the constrained optimum under the imposed box.

Unbounded without EB is not usable. A long unbounded Adam run kept improving the likelihood by sending rates upward:

| Setup | Converged | Objective | Tail slope | Final theta grad | Rate range |
|---|---:|---:|---:|---:|---:|
| Unbounded Adam, no EB, long run | no | 3682.3954018 | -2.7185 bits/step | 16.77 | 1.36e-4..7547.88 |

This is runaway behavior, not convergence. Hierarchical EB fixes that by making large family-specific rate excursions pay a prior penalty.

The cost is that the EB data loss is worse than the bounded unregularized likelihood:

```text
bounded no-EB Rprop objective: 4959.2418 bits
EB Rprop data loss only:       5286.1746 bits
EB Rprop total objective:      5803.7300 bits
```

That is expected: EB trades raw data fit for stable finite genewise rates.

## Specieswise Whole-Dataset EB Run

I ran specieswise mode over the full archaeal ALE dataset with the same complete-CCP input and the same `<4` leaf exclusion:

```text
candidate families:        5446
excluded below 4 leaves:     67
eligible / selected:       5379
specieswise theta rows:     119
```

The specieswise EB prior is the same empirical hierarchical Bayes penalty, but its rows are **species**, not gene families. In other words, the prior regularizes species-specific log2 D/T/L rates around learned population means and sigmas.

The run used:

```text
mode: specieswise
optimizer: Rprop
rate bounds: disabled
hierarchical EB: enabled
family_chunk_size: 0
clade_budget: 1000000
backtracking: disabled
```

The solver schedule was the corrected 240-step schedule:

| Steps | Pi iterations | Self-loop iterations | Dtype | Backward self-loop solver |
|---:|---:|---:|---|---|
| 40 | 4 | 4 | float32 | neumann |
| 40 | 8 | 8 | float32 | neumann |
| 40 | 12 | 12 | float32 | neumann |
| 120 | 16 | 16 | float64 | gmres |

Result:

| Run | Families | Steps | Converged | Total objective | Data loss | Prior loss | Final theta grad | Tail slope | Rate range | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Specieswise EB Rprop, 80-step screen | 5379 | 80 | no | 357496.3978 | 356708.7754 | 787.6224 | 181.1 | -50.30 | 0.00132..5.46 | 432 s |
| Specieswise EB Rprop, 240-step run | 5379 | 240 | no | 354562.2692 | 353645.6521 | 916.6171 | 1.28e4 | -27.39 | 0.000462..663.38 | 2308 s |

The 240-step run did not converge. The final loss was still descending and was not near the best point in the trajectory:

```text
final total: 354562.269215
best total:  354551.921496 at step 235
tail slope:  -27.3862 bits/step
theta grad:  12783.76
```

The EB prior stayed finite and did not collapse:

```text
prior_mu    = [-4.1863591, -1.7485839, -3.8102903]
prior_sigma = [ 2.5927780,  2.4999587,  2.1965134]
```

However, the specieswise unbounded rates still spread aggressively:

```text
rate_min  = 0.000462
rate_max  = 663.378
rate_mean = 4.447
```

Interpretation of that first whole-dataset run: hierarchical EB prevented immediate numerical blow-up, but the schedule was wrong for specieswise mode. It moved into the expensive `float64` GMRES phase while the full objective was still far from optimized. The fp64 phase was being used as an optimizer instead of as a verification/refinement pass.

Operational notes:

- A fully unbatched all-family fp32 step was fast after preprocessing, but the unbatched fp64 GMRES phase hit the backward self-loop scratch guard.
- `clade_budget=1000000` was needed for the fp64 GMRES phase to complete.
- The 240-step all-dataset run took about 38.5 minutes.
- No stochastic backtracking was run for this specieswise experiment.

### Strong-Prior FP32 Continuations

I then switched to the more stable specieswise EB prior:

```text
prior_initial_sigma = 0.5
prior_log_sigma_sigma = 0.02
```

and kept the optimizer in `float32` instead of going straight to `float64`.

These continuation runs start from the strong-prior checkpoint and use all 5,379 eligible families:

| Run | Extra fp32 steps | Backward self-loop solver | Rprop LR schedule | Converged | Total objective | Data loss | Prior loss | Final joint grad | Tail slope | Rate range | Time |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strong-prior checkpoint | - | mixed | mixed warmup + short fp64 check | no | 359099.7018 | 356774.4581 | 2325.2436 | 13.36 | -0.0768 | 0.00946..11.14 | 443 s |
| fp32 GMRES high-LR probe | 40 | gmres | `0.005 -> 0.001` decay | no | 359093.4688 | 356760.6250 | 2332.8501 | 14.24 | -0.0721 | 0.00946..12.08 | 113 s |
| matched fp32 Neumann high-LR probe | 40 | neumann | `0.005 -> 0.001` decay | no | 359088.4375 | 356752.6250 | 2335.7976 | 14.26 | -0.2149 | 0.00945..12.92 | 39 s |
| fp32 GMRES continuation | 160 | gmres | `0.001`, then `0.0002` | no | 359062.0938 | 356710.5625 | 2351.5378 | 7.53 | -0.1992 | 0.00942..16.38 | 435 s |
| fp32 GMRES small-step continuation | 100 | gmres | `0.00005` | no | 359030.5312 | 356667.9375 | 2362.6006 | 12.58 | -0.3820 | 0.00940..19.65 | 212 s |

This is the clearest result from the specieswise runs so far. The main issue was entering `float64` too early, not failing to use GMRES during optimization. A matched 40-step fp32 control from the same checkpoint with the same LR schedule did **better with truncated Neumann** than with GMRES:

```text
GMRES:   359099.4062 -> 359093.4688, median post-first step 2.74 s
Neumann: 359099.4062 -> 359088.4375, median post-first step 0.94 s
```

So GMRES is not currently justified as the default specieswise fp32 optimizer phase. Its role is better interpreted as a final gradient-accuracy/refinement option once the cheap fp32 truncated-Neumann objective is already close to flat.

I also ran a matched 20-step `float64` refinement probe from the same checkpoint with the same Rprop LR `0.0001`. In fp64, GMRES was marginally more accurate by the gradient/objective numbers, but the effect was tiny and not a convergence-speed win:

```text
GMRES:   359099.3980 -> 359098.5495, final joint grad 12.37, median post-first step 18.12 s
Neumann: 359099.3980 -> 359098.5506, final joint grad 13.26, median post-first step 15.56 s
```

That supports using fp64+GMRES as a final accuracy check, not as a materially better optimizer phase.

The longer fp32 GMRES continuations still show that cheap fp32 work was available. After 300 additional fp32 GMRES steps, the objective improved by about `69.17` bits from the strong-prior checkpoint:

```text
359099.7018 -> 359030.5312
```

The run is still not converged: the final fp32 tail slope is `-0.3820 bits/step`, far above the `0.02` tolerance. That means no final fp64 run should be interpreted as convergence yet. The next specieswise schedule should keep optimizing cheaply in fp32, probably starting with truncated Neumann, until the fp32 tail is nearly flat, then switch to a short `float64`+GMRES verification/refinement phase.

The strong prior is controlling most rates, but the upper tail is still moving. The final rate quantiles after the fp32 continuations are:

```text
min, q25, median, q75, q90, q95, q99, max
0.00940, 0.05230, 0.13677, 0.48771, 1.40977, 2.34608, 5.71133, 19.65298
```

So this is not a broad rate explosion; it is mostly a small high-rate tail. The script now saves `final_theta`, so those high-rate species coordinates can be inspected directly.

## Optimizer experiments

### Rprop

Rprop is currently the most reliable optimizer for this deterministic, low-dimensional genewise problem. It does not depend on an exponential moving average of gradient norms, so it is less sensitive to the 100-200 step optimization horizon.

Corrected final run:

```text
optimizer: rprop
lr: 0.01
steps: 240
final phase: 120 fp64 GMRES steps
converged: yes
total: 5803.7300488
data: 5286.1745989
prior: 517.5554499
theta grad: 4.57e-05
tail slope: -1.32e-09 bits/step
```

### Adam

The old default Adam run converged and was faster than Rprop, but the final gradient was two orders of magnitude larger:

```text
optimizer: adam
lr: 0.03
steps: 240
final phase: 120 fp64 GMRES steps
converged: yes
total: 5803.7300648
theta grad: 7.22e-03
tail slope: -6.38e-05 bits/step
```

That row is not a tuned Adam result. It came before the explicit reset/ramp beta sweep and should stay separate from the tuned-Adam comparison. PyTorch-style default Adam beta memory is neural-network-oriented:

```text
betas = (0.9, 0.999)
```

For a 100-200 step deterministic optimization, `beta2=0.999` is almost certainly too slow to adapt. The reset/ramp sweep above tried shorter beta memories, but none of those prescribed Adam configurations converged. The later high-LR decay diagnostic suggests the learning-rate scale was at least as important as beta memory: Adam needs a much larger early LR to reach the basin quickly, then a more careful decay to finish cleanly.

### RMSProp

The old RMSProp result was not good and was not the corrected EB/fp64-GMRES experiment. Under the corrected setup, default-style fixed LR was still bad:

| RMSProp alpha | LR schedule | Converged | Total | Final theta grad | Tail slope |
|---:|---|---:|---:|---:|---:|
| 0.99 | fixed 0.01 | no | 5806.0678067 | 1.661 | -0.1443 |
| 0.98 | fixed 0.01 | no | 5826.9699511 | 5.720 | -0.6345 |
| 0.95 | fixed 0.01 | no | 5873.7606019 | 9.565 | -1.1712 |
| 0.90 | fixed 0.01 | no | 5895.7478393 | 10.788 | -1.3641 |
| 0.80 | fixed 0.01 | no | 5906.3875839 | 11.295 | -1.4586 |

Larger fixed LRs reached the optimum faster but oscillated:

| RMSProp alpha | Fixed LR | Converged | Total | Final theta grad |
|---:|---:|---:|---:|---:|
| 0.99 | 0.03 | no | 5804.0737141 | 1.200 |
| 0.99 | 0.05 | no | 5804.1991505 | 1.265 |
| 0.99 | 0.10 | no | 5809.1485267 | 3.252 |
| 0.98 | 0.03 | no | 5804.3160448 | 1.571 |
| 0.98 | 0.05 | no | 5806.2865139 | 3.008 |

The best RMSProp setup used phase-specific LR:

| RMSProp alpha | fp32 LR | fp64 GMRES LR | Converged | Total | Data | Prior | Final theta grad |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.99 | 0.05 | 0.01 | yes | 5803.7300766 | 5286.1737399 | 517.5563367 | 8.90e-03 |
| 0.99 | 0.05 | 0.005 | no | 5803.7405237 | 5286.0290830 | 517.7114400 | 1.71e-01 |
| 0.99 | 0.03 | 0.01 | no | 5803.7761382 | 5286.3918440 | 517.3842940 | 4.40e-01 |
| 0.98 | 0.05 | 0.01 | no | 5803.8427554 | 5286.0880640 | 517.7546920 | 6.81e-01 |

So RMSProp can be made to work, but only after choosing a faster fp32 LR and a damped fp64 finishing LR. It is competitive with Adam but still worse than Rprop by final gradient and robustness.

A next RMSProp pass should rerun the RMSProp sweep with phase-boundary optimizer resets plus a short LR ramp. The best fixed phase-LR setting above was found before adding explicit same-dtype phase resets and ramping; those controls are likely the right way to reduce the late-phase oscillations seen in the failed RMSProp runs.

### Adafactor

Adafactor did not converge in the corrected 240-step screen:

```text
total: 5803.9519411
theta grad: 0.6808
tail slope: -0.02528 bits/step
```

It was close in objective but not stationary enough. Like Adam/RMSProp, it probably needs optimizer-specific tuning for this deterministic short-horizon problem.

## Current interpretation

Rprop works best because the optimization is not neural-network-like. We have a deterministic objective, a small number of parameters per family, and a target of roughly 100-200 gradient steps. Optimizers whose behavior is dominated by exponential moving averages of past squared gradients need their memory length and final learning rate matched to that horizon.

For now:

1. Use hierarchical EB to avoid unbounded divergence.
2. Use Rprop as the default robust optimizer.
3. Keep the final phase in `float64` with backward self-loop GMRES.
4. If using RMSProp, start from `alpha=0.99`, fp32 LR `0.05`, fp64 GMRES LR `0.01`, phase-boundary resets, and a short fp64 LR ramp.
5. Treat the current Adam reset/ramp/beta sweep as negative for the prescribed configurations; future Adam work should start from the high-LR decay result and tune the fp64 decay/finish, not just beta memory.

## Remaining caveats

- Older optimizer JSONs recorded the genewise `theta` or projected gradient only. The reset/ramp Adam JSONs also record `final_joint_grad_norm`, including EB hyperparameters.
- These are 256-family screens, not whole-dataset archaeal runs.
- The main optimizer table's old Adam/RMSProp/Adafactor rows were measured before adding same-dtype phase resets and LR ramping. The Adam reset/ramp/beta sweep in this document is separate and did not converge.
- The final `gmres` setting is for the backward/adjoint self-loop solve, not a replacement of the forward likelihood solver.

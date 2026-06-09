# Archaea Genewise Optimization Experiments

Date: 2026-06-09

## Best result so far

The best practical setup so far is **hierarchical EB + unbounded rates + Rprop**, with the final optimization phase run in `float64` and the backward self-loop solver set to `gmres`.

It converged cleanly on the 256-family complete-CCP archaeal screen. These measured rows were produced before adding same-dtype phase-boundary optimizer resets and LR ramping; they are still the best evidence so far, but adaptive optimizers should be rerun under the new transition policy before treating the comparison as final.

| Setup | Converged | Total objective | Data loss | Prior loss | Final theta grad | Rate range | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hierarchical EB, Rprop, fp64 GMRES finish | yes | 5803.7300488 | 5286.1745989 | 517.5554499 | 4.57e-05 | 0.1180..0.8595 | 63.0 s |
| Hierarchical EB, Adam, fp64 GMRES finish | yes | 5803.7300648 | 5286.1979681 | 517.5320967 | 7.22e-03 | 0.1180..0.8595 | 49.4 s |
| Hierarchical EB, tuned RMSProp, fp64 GMRES finish | yes | 5803.7300766 | 5286.1737399 | 517.5563367 | 8.90e-03 | 0.1180..0.8595 | 69.6 s |
| Hierarchical EB, Adafactor, fp64 GMRES finish | no | 5803.9519411 | 5286.2554630 | 517.6964781 | 6.81e-01 | 0.1170..0.8514 | 56.6 s |

Rprop is the best current choice because it reaches the same EB optimum as Adam/RMSProp but with a much smaller final theta gradient and a flatter loss tail. Adam is faster wall-clock but less clean by the gradient criterion. Tuned RMSProp can work, but it is more sensitive to learning-rate scheduling.

Current recommendation:

1. Use Rprop for the robust baseline.
2. Use final `float64` + backward self-loop GMRES.
3. For Adam/RMSProp/Adafactor-style optimizers, reset optimizer state at every solver-schedule transition and ramp LR back up at phase entry.
4. Rerun the adaptive-optimizer sweeps with those transition controls before selecting a non-Rprop optimizer.

## Scope

These numbers are for:

- Dataset root: `tests/data/alerax_archaea_davin2017`
- Input: complete ALE CCP files, parsed directly from `.ale`
- Mode: `genewise`
- Families: 256 selected families, ordered by smallest eligible families
- Leaf filter: exclude families with fewer than 4 leaves
- Rate bounds: disabled for EB/unbounded experiments
- Backtracking: enabled only in the final Rprop run; optimizer screen timings usually used `--backtrack-families 0`

The 256-family screen is not a whole-dataset result. It is enough to compare optimizer behavior, not enough to claim final production-scale performance over every eligible archaeal family.

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

For Adam/Adamax/Adafactor, the important missing sweep is the second-moment memory. Defaults such as Adam `beta2=0.999` are designed for much longer stochastic training horizons and are probably inappropriate for a 100-200 step deterministic optimization.

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

Adam converged and was faster than Rprop, but the final gradient was two orders of magnitude larger:

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

Adam should probably be retuned with shorter-memory beta values. The current script still uses PyTorch defaults, which are neural-network defaults:

```text
betas = (0.9, 0.999)
```

For a 100-200 step deterministic optimization, `beta2=0.999` is almost certainly too slow to adapt.

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

A next RMSProp pass should rerun this sweep with phase-boundary optimizer resets plus a short LR ramp. The best fixed phase-LR setting above was found before adding explicit same-dtype phase resets and ramping; those controls are likely the right way to reduce the late-phase oscillations seen in the failed RMSProp runs.

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
5. Retune Adam/Adamax/Adafactor with shorter second-moment memory before drawing conclusions about them.

## Remaining caveats

- The convergence gradient currently recorded in the JSON is the genewise `theta` gradient. The script does not yet separately record gradient norms for the EB hyperparameters `mu` and `raw_sigma`.
- These are 256-family screens, not whole-dataset archaeal runs.
- The main optimizer tables were measured before adding same-dtype phase resets and LR ramping. The script now resets optimizer state at every phase boundary by default, but the adaptive optimizer sweeps need to be rerun under that policy.
- The final `gmres` setting is for the backward/adjoint self-loop solve, not a replacement of the forward likelihood solver.

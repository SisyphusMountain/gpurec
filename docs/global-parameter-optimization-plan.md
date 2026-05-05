# Global Parameter Optimization Plan

Date: 2026-05-05.

Scope: optimizing the three global DTL rates `(D, L, T)` in uniform mode using
the gradients already exposed through `GeneReconModel` / `_GeneReconFunction`.
The reference target for correctness is:

```text
tests/data/test_trees_100/output_global
```

AleRax reports natural-log likelihoods.  gpurec optimizes NLL in log2 units, so
the comparison target is:

```text
AleRax sum log-likelihood       = -9737.072900 nats
gpurec target NLL               = 14047.626786 bits
AleRax global rates:
  D = 0.0191209
  L = 0.0199312
  T = 0.0208267
```

At those AleRax rates, gpurec fp32 evaluates:

```text
NLL = 14047.632812 bits
gap = 0.006027 bits = 0.004177 nats
```

This is within the rounding error expected from the three-decimal per-family
likelihood file.

## Recommendation

Use projected PyTorch `LBFGS` with `line_search_fn="strong_wolfe"` in fp32 as
the default global optimizer.

For the 100-family benchmark, starting from `(0.05, 0.05, 0.05)`:

```text
hit AleRax rates: 5 gradient evaluations, 1.11 s optimizer time
full LBFGS run:   11 gradient evaluations, 1.89 s optimizer time
process wall:     4.09 s including Python startup + model construction
```

Use the lower bound as a constraint, not as an initialization:

```python
min_rate = 1e-10
theta_min = log2(min_rate)

def closure():
    with torch.no_grad():
        model.theta.clamp_(min=theta_min)
    opt.zero_grad(set_to_none=True)
    loss = model()
    loss.backward()
    return loss
```

Do not start from exactly `1e-10`.  On this dataset, starting all three rates at
the floor produced `-inf`/`nan` likelihood values and did not reach the target
within 61 evaluations.  A robust default initialization is an interior rate,
for example `(0.02, 0.02, 0.02)` if no prior is available, or `(0.05, 0.05,
0.05)` to match the existing benchmark.

## Tested Strategies

Benchmark script:

```text
profiling/bench_global_parameter_optimization.py
```

Common settings:

```text
dataset = tests/data/test_trees_100
mode = global
pibar_mode = uniform
fixed_iters_Pi = 6
max_wave_size = 32768
min_rate = 1e-10
target condition = NLL gap <= 0.05 bits and max relative rate error <= 1%
```

Times below are measured after model construction unless otherwise stated.

| Strategy | Init rates | Evals | Hit eval | Hit time | Final time | Best NLL gap | Best rate rel err | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PyTorch LBFGS fp32 + strong Wolfe | `0.05` | `11` | `5` | `1.11 s` | `1.89 s` | `0.00212 bits` | `6.9e-5` | best default |
| PyTorch LBFGS fp32 + strong Wolfe | `0.01` | `10` | `5` | `1.08 s` | `1.74 s` | `0.00017 bits` | `1.2e-4` | also good |
| PyTorch LBFGS fp32 + strong Wolfe | `0.10` | `13` | `6` | `1.26 s` | `2.19 s` | `0.00114 bits` | `2.7e-4` | robust |
| SciPy L-BFGS-B fp32 | `0.05` | `25` | `5` | `1.46 s` | `4.13 s` | `0.00212 bits` | `1.3e-4` | correct but extra line-search calls |
| Adam 3 steps then SciPy L-BFGS-B fp32 | `0.05` | `47` | `7` | `1.26 s` | `6.42 s` | `0.00212 bits` | `1.4e-4` | not worth it here |
| SciPy L-BFGS-B fp64 | `0.05` | `7` | `5` | `4.58 s` | `5.02 s` | `0.00440 bits` | `2.0e-5` | good polish, slower first use |
| bf16 target smoke | target | n/a | n/a | n/a | n/a | n/a | n/a | unsupported: sparse addmm lacks bf16 |
| PyTorch LBFGS fp32 | `1e-10` | `61` | none | none | `31.24 s` | invalid | invalid | bad initialization |

The fastest path to the AleRax parameter neighborhood is therefore plain fp32
LBFGS, not Adam warmup and not fp64 from scratch.

## Why LBFGS Works Well Here

The global problem has only three parameters.  Each gradient evaluation is
expensive because it runs the full forward+backward dynamic program, but the
curvature information from L-BFGS is very valuable in three dimensions.

The observed fp32 LBFGS trajectory from `(0.05, 0.05, 0.05)` was:

```text
eval 1: NLL 14951.2031, rate rel err 1.61
eval 3: NLL 14077.4902, rate rel err 0.231
eval 5: NLL 14047.6396, rate rel err 0.00396  <-- target reached
eval 6: NLL 14047.6279, rate rel err 0.000068
```

Adam is not competitive because it spends full gradient evaluations just to
discover scale.  A few Adam steps move in the right direction, but L-BFGS from
the original point already gets there in about the same number of evaluations.

SciPy L-BFGS-B gives cleaner bound handling, but with this objective it does
many redundant line-search evaluations near the fp32 precision floor.  PyTorch
LBFGS with explicit projection is faster in wall time.

## Precision Policy

Recommended staged policy:

1. Run fp32 LBFGS from an interior initialization.
2. Stop when either:
   - relative rate step is below `1e-4`, or
   - NLL improvement is below about `1e-3` bits for two accepted steps, or
   - a fixed budget of roughly `8-12` gradient evaluations is reached.
3. If a stricter final report is needed, rebuild/cast to fp64 at the fp32 rates
   and run a short fp64 L-BFGS-B polish.

Do not use bf16 for the current backward path.  The smoke test failed with:

```text
"addmm_sparse_cuda" not implemented for 'BFloat16'
```

Even if bf16 were made to run, it is unlikely to be useful for the final line
search because the optimum is identified by sub-percent changes in three small
rates.

## Production Shape

The production optimizer should keep the model and static wave layout resident
and only update `theta`:

```python
model = GeneReconModel.from_trees(
    species_tree=sp,
    gene_trees=genes,
    mode="global",
    pibar_mode="uniform",
    dtype=torch.float32,
    theta_init_rates=(0.05, 0.05, 0.05),
    fixed_iters_Pi=6,
    max_wave_size=32768,
)

opt = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=12,
    history_size=10,
    tolerance_grad=1e-3,
    tolerance_change=1e-7,
    line_search_fn="strong_wolfe",
)

theta_min = math.log2(1e-10)

def closure():
    with torch.no_grad():
        model.theta.clamp_(min=theta_min)
    opt.zero_grad(set_to_none=True)
    loss = model()
    loss.backward()
    return loss

opt.step(closure)
model.clamp_theta_(min_rate=1e-10)
```

For larger datasets that do not fit in a single resident `GeneReconModel`
state, the same optimizer should call a chunked objective closure:

```python
def objective_and_grad(theta):
    E, params = E_fixed_point_once(theta)
    nll = 0
    adj = zero_adjoint()

    for chunk in resident_chunks:
        Pi, Pibar = forward_chunk(chunk, E, params)
        nll += root_nll(Pi, E, chunk.roots)
        adj += backward_chunk(chunk, Pi, Pibar, E, params)

    grad = theta_vjp_once(adj, E, params)
    return nll, grad
```

This is the same computational structure as the uniform full-pipeline harness.
It avoids rebuilding layouts or keeping all families' full `Pi/Pibar` state
resident.

## Open Implementation Tasks

1. Add a production `optimize_global_rates_lbfgs` helper around the public
   `GeneReconModel` API.
2. Add a chunked closure variant for 1000-family training, using the existing
   full-pipeline forward+backward harness as the template.
3. Expose an explicit stopping policy in terms of NLL improvement and rate-step
   size, not just raw gradient norm.  The fp32 gradient norm jitters near the
   optimum even when the rates and NLL are already correct.
4. Add a small correctness test using `tests/data/test_trees_100/output_global`:
   the optimizer should reach the AleRax rounded rates within `1%` and the
   AleRax NLL within `0.05` bits from `(0.05, 0.05, 0.05)`.
5. Keep the lower clipping bound at `1e-10`, but reject or override
   initialization exactly at the bound.


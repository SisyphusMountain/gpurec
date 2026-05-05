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

## Round Status

This round confirmed the global-rate path through the public
`GeneReconModel` API and refreshed the 100-family benchmark on the local
RTX 4090 / torch `2.11.0+cu128` environment.

Confirmed implemented API:

- `GeneReconModel.from_trees(...)` constructs a resident model from Newick
  files with `mode="global"` and `pibar_mode="uniform"`.
- `theta_init_rates=(D, L, T)` initializes rates in natural space.
- `preprocess_cache_dir=...` reuses CPU preprocessing artifacts.
- `model.rates`, `model.clamp_theta_(min_rate=...)`, `model.nll()`,
  `model.log_likelihood()`, and `model.nll_per_family()` are available.

Implemented in this round:

- `gpurec.optimization.optimize_global_rates_lbfgs` wraps the resident
  `GeneReconModel` path and keeps only `theta` changing between objective
  evaluations.
- `tests/integration/test_global_parameter_optimization.py` checks the AleRax
  rounded optimum and the floor-initialization guard.
- `profiling/bench_global_parameter_optimization.py` now benchmarks the
  production helper through the `recommended-fp32` strategy, while preserving
  the direct and SciPy comparison paths.
- No 1000-family chunked-closure measurement was supplied for this round.

## Recommendation

Use projected PyTorch `LBFGS` with `line_search_fn="strong_wolfe"` in fp32 as
the default global optimizer.

For the 100-family benchmark, starting from `(0.05, 0.05, 0.05)`, the landed
production helper measured:

```text
hit AleRax rates: 5 gradient evaluations, 1.47 s optimizer time
full LBFGS run:   11 gradient evaluations, 2.25 s optimizer time
process wall:     3.70 s including Python startup + model construction
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

Do not start optimization from exactly `1e-10`.  On this dataset, starting all
three rates at the floor produced `-inf`/`nan` likelihood values and did not
reach the target within 61 evaluations.  The production helper therefore
defaults to moving floor-initialized model state back to `(0.05, 0.05, 0.05)`.
The benchmark CLI still rejects explicit `--init-rate 1e-10` unless the user
opts into the floor path, because this is almost always a user mistake.

## Correctness Tests

Focused API and autograd tests run for this report:

```bash
pytest -q tests/integration/test_gene_recon_model.py::test_lbfgs_closure_runs \
  tests/integration/test_gene_recon_model.py::test_log_likelihood_helper \
  tests/integration/test_gene_recon_model.py::test_preprocess_cache_matches_single_path
```

Result:

```text
3 passed in 6.00 s
```

```bash
pytest -q 'tests/gradients/test_autograd_bridge.py::test_model_nll_matches_compute_likelihood_batch[global-uniform]' \
  'tests/gradients/test_autograd_bridge.py::test_autograd_matches_fd[global-uniform]' \
  tests/gradients/test_autograd_bridge.py::test_per_family_rejects_non_genewise
```

Result:

```text
3 passed in 1.87 s
```

Production optimizer regression:

```bash
pytest -q tests/integration/test_global_parameter_optimization.py
```

Result:

```text
2 passed in 3.17 s
```

The test optimizes `tests/data/test_trees_100` from `(0.05, 0.05, 0.05)` and
asserts:

```text
max relative rate error <= 1%
NLL gap to AleRax rounded likelihood <= 0.05 bits
```

It also constructs the model at `(1e-10, 1e-10, 1e-10)` and verifies that the
helper moves that initial state to the configured interior rates before the
first closure evaluation.

## Before / After Benchmark Comparison

AleRax fixture baseline, from
`tests/data/test_trees_100/output_global/alerax.log`:

```text
command: alerax -s sp.nwk -f families.txt -p output_global --model-parametrization GLOBAL
MPI ranks: 24
non-thorough rate optimization reached ll=-9737.08 at 4 s
thorough rate optimization completed at 7 s
full AleRax run including reconciliation export ended at 9 s
```

gpurec comparison:

| Measurement | Hit target | Full optimizer | Process wall | Notes |
|---|---:|---:|---:|---|
| Previous report baseline | `5 evals / 1.11 s` | `11 evals / 1.89 s` | `4.09 s` | same dataset and strategy |
| This round, timed single-strategy run | `5 evals / 1.12 s` | `11 evals / 1.92 s` | `3.80 s` | no observed regression |
| Production helper landed | `5 evals / 1.47 s` | `11 evals / 2.25 s` | `3.70 s` | includes helper history/final-eval bookkeeping |

Exact timed command:

```bash
/usr/bin/time -f 'process_wall_s %e' \
  python profiling/bench_global_parameter_optimization.py \
    --dataset tests/data/test_trees_100 \
    --cache-dir /tmp/gpurec_paramopt_final_cache \
    --strategies recommended-fp32 \
    --init-rate 0.05 \
    --no-print-evals
```

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

Exact multi-strategy command:

```bash
python profiling/bench_global_parameter_optimization.py \
  --dataset tests/data/test_trees_100 \
  --cache-dir /tmp/gpurec_paramopt_final_cache \
  --strategies recommended-fp32,scipy-lbfgsb-fp32,scipy-lbfgsb-fp64-polish,bad-floor-init-guard \
  --init-rate 0.05 \
  --no-print-evals
```

Times below are measured after model construction unless otherwise stated.

| Strategy | Init rates | Evals | Hit eval | Hit time | Final time | Best NLL gap | Best rate rel err | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| AleRax target eval fp32 | target | `1` | `1` | `0.980 s` | `0.980 s` | `0.00603 bits` | `9.7e-8` | reference evaluation |
| Production helper fp32 LBFGS + strong Wolfe, timed alone | `0.05` | `11` | `5` | `1.467 s` | `2.250 s` | `0.00212 bits` | `6.8e-5` | landed default |
| Production helper fp32 LBFGS + strong Wolfe, multi-strategy run | `0.05` | `7` | `5` | `1.464 s` | `1.741 s` | `0.00212 bits` | `1.2e-4` | same target; LBFGS stopped earlier |
| SciPy L-BFGS-B fp32 | `0.05` | `7` | `5` | `0.683 s` | `0.963 s` | `0.00407 bits` | `1.2e-4` | correct; clean bounds, less production-shaped |
| SciPy L-BFGS-B fp64 polish after fp32 seed | fp32 seed | `3` polish | `1` polish | `0.271 s` polish | `0.717 s` polish | `0.00441 bits` | `1.8e-5` | seed cost was `11 evals / 1.455 s` |
| Adam 3 steps then SciPy L-BFGS-B fp32 | `0.05` | `48` | `7` | `0.910 s` | `6.616 s` | `0.00212 bits` | `1.4e-4` | line search ended abnormal; not worth it |
| SciPy L-BFGS-B fp64 | `0.05` | `7` | `5` | `1.202 s` | `1.652 s` | `0.00440 bits` | `2.0e-5` | good polish option |
| bf16 target smoke | target | n/a | n/a | n/a | n/a | n/a | n/a | unsupported: sparse addmm lacks bf16 |
| PyTorch LBFGS fp32 | `0.01` | `10` previous | `5` | `1.08 s` | `1.74 s` | `0.00017 bits` | `1.2e-4` | previous report; also good |
| PyTorch LBFGS fp32 | `0.10` | `13` previous | `6` | `1.26 s` | `2.19 s` | `0.00114 bits` | `2.7e-4` | previous report; robust |
| PyTorch LBFGS fp32 | `1e-10` | `61` previous | none | none | `31.24 s` previous | invalid | invalid | bad initialization |
| Benchmark explicit floor init | `1e-10` | `0` | n/a | n/a | `0 s` | n/a | n/a | rejected before model run |

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
eval 5: NLL 14047.6406, rate rel err 0.00396  <-- target reached
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

The production helper keeps the model and static wave layout resident and only
updates `theta`:

```python
from gpurec.optimization import optimize_global_rates_lbfgs

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

result = optimize_global_rates_lbfgs(
    model,
    min_rate=1e-10,
    interior_init_rates=(0.05, 0.05, 0.05),
    override_floor_init=True,
    steps=12,
    max_eval=60,
)

rates = result["rates"]
nll = result["negative_log_likelihood"]
history = result["history"]
```

Internally this is still the projected PyTorch LBFGS closure:

```python
with torch.no_grad():
    model.theta.clamp_(min=log2(min_rate))
opt.zero_grad(set_to_none=True)
loss = model()
loss.backward()
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

1. Add a chunked closure variant for 1000-family training, using the existing
   full-pipeline forward+backward harness as the template.
   The 1000-family time/memory measurement was not part of this round.
2. Expose an explicit stopping policy in terms of NLL improvement and rate-step
   size, not just raw gradient norm.  The fp32 gradient norm jitters near the
   optimum even when the rates and NLL are already correct.
3. Keep the lower clipping bound at `1e-10`, but reject or override
   initialization exactly at the bound.

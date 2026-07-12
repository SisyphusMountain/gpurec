# Centered-kernel worktree brief

## Mission

Develop a production-quality centered representation for gpurec's log-domain
dynamic program. The immediate hypothesis is that row-wise fp64 offsets plus
fp32 residuals can improve numerical precision and repeatability on HOGENOM
without materially reducing throughput. The long-term aim is to port every
kernel that consumes, produces, differentiates, or reduces centered state, not
merely the forward Pi kernels.

This is an empirical optimization project. Preserve the ordinary likelihood and
prove improvements against baselines; do not assume that centering fixes every
precision floor. In particular, `docs/full_dataset_convergence.md` records an
earlier Archaea probe where centering did not address the dominant family-sum
precision floor. HOGENOM must be measured independently.

## Worktree

- Worktree: `/home/enzo/Documents/git/gpurec/gpurec-centered-kernels`
- Branch: `feat/centered-kernels`
- Base: `dev` at `bccad635`
- Canonical external data root:
  `/home/enzo/Documents/git/gpurec/gpurec-data/benchmarks/large_dataset_capacity`
- HOGENOM core:
  `/home/enzo/Documents/git/gpurec/gpurec-data/benchmarks/large_dataset_capacity/datasets/alerax_hogenom_core/hogenom`
- HOGENOM single-family probe:
  `/home/enzo/Documents/git/gpurec/gpurec-data/benchmarks/large_dataset_capacity/datasets/alerax_hogenom_single_family_probe/hogenom`

Use `GPUREC_HOGENOM_ROOT` for code that supports it. Do not copy the 7.1 GB
dataset into this worktree and do not commit generated results or captures.

## What already exists

The current prototype is opt-in through `GPUREC_CENTERED_PI_FORWARD=1` in
`gpurec/core/inference/forward.py`. It stores each absolute log2 row as an fp32
residual matrix plus one fp64 offset per row:

```text
Pi_absolute[row, species] = Pi_residual[row, species] + Pi_offset[row]
```

The same representation exists for Pibar and temporary DTS rows. Relevant code:

- `gpurec/core/kernels/centered_pi_forward.py`: centered leaf initialization,
  wave steps, Pibar construction, and DTS reduction.
- `gpurec/core/inference/forward.py`: environment gate, offset allocation, and
  root-row reconstruction.
- `docs/gmres/centered_pi_heuristic_offsets.md`: exact versus heuristic offset
  selection and the motivation for avoiding an extra species-row pass.
- `gpurec/core/kernels/wave_step.py` and `dts_fused.py`: supported absolute
  forward implementations and correctness reference.
- `gpurec/core/kernels/wave_backward.py`, `wave_backward_kernels.py`,
  `wave_tangent.py`, `wave_so.py`, `dts_tangent.py`, and `dts_so.py`: derivative
  paths that eventually need offset-aware equivalents.
- `docs/grad_e_backward_race.md`, if available in the source checkout, explains
  why offset storage itself is small relative to the fp32 matrices and separates
  precision problems from memory-ordering bugs.

The cleanup checkout at `/home/enzo/Documents/git/gpurec/gpurec` currently has
useful untracked evidence that is intentionally not part of this branch's base:

- `tests/test_centered_pi_forward.py`
- `docs/numerical_precision.md`
- `docs/grad_e_backward_race.md`
- `docs/backward_atomics_profiling.md`

Read these in place and port only the durable tests or conclusions needed here.
Do not modify or delete the cleanup checkout's copies. The historical commit
`9abbece3` also contains HOGENOM-centered probes that were removed during source
tree cleanup and can be recovered selectively with `git show`:

- `benchmarks/large_dataset_capacity/hogenom_centered_compensation_accuracy.py`
- `benchmarks/large_dataset_capacity/hogenom_centered_error_source_diagnostic.py`
- `benchmarks/large_dataset_capacity/hogenom_solver_accuracy_distribution.py`
- `benchmarks/large_dataset_capacity/plot_centered_fp32_vs_fp64_tail.py`
- `docs/gmres/2595_fp32_error_source_report.md`

That earlier family-2595 investigation reportedly found that centering reduced a
Pi/Pibar representation error while residual whole-gradient error was dominated
by other fp32 state (`log_pS`/E). Reproduce the finding before relying on it; use
it as a warning that the eventual design may need to extend beyond Pi-family
kernels to deliver an end-to-end HOGENOM improvement.

The prototype is incomplete and should be treated as evidence, not as an oracle:

- It supports CUDA fp32 residual storage and forward loss evaluation only.
- The single-batch autograd bridge deliberately rejects backward; other call
  paths do not consistently enforce that guard.
- Pi residual diagnostics are unsupported.
- Leaf initialization has drifted from the normal kernel: the normal path
  includes `leaf_obs_logp` in its seeded Pi/Pibar/child terms, while the centered
  prototype uses zero there.
- Centered DTS does not mirror the normal path's cast of `log_split_probs` to the
  Pi dtype, which can create Triton loop-carried fp32/fp64 type mismatches.
- Final Pibar metadata and every backward consumer must be audited for whether a
  value is an absolute log value or a centered residual.

Start by pinning these differences with tests and repairing forward parity. Do
not build backward on a forward recurrence that only happens to agree when leaf
observation log-probabilities are zero.

## Required outcomes

1. Establish reproducible normal-path baselines before changing kernels:
   forward loss, full gradient, repeated-run variation, optimizer convergence,
   peak memory, and steady-state wall time.
2. Make centered forward mathematically equivalent for uniform and weighted
   receiver/origination cases, single- and multi-split DTS, streamed batches,
   and nonzero leaf observation weights.
3. Define one explicit centered-state contract shared by all layers. Offset
   ownership, dtype, reconstruction points, `-inf` rows, and empty/inactive rows
   must be unambiguous.
4. Port the first-order backward/implicit-adjoint path so loss and gradients are
   both offset-aware. Then port diagnostics, backtracking, JVP/tangent kernels,
   second-order kernels, and HVP/curvature consumers in measured phases.
5. Replace the private environment experiment with an explicit supported solver
   option only after forward and backward gates pass. Keep a temporary absolute
   reference path until parity and performance are demonstrated.
6. Add focused unit tests plus end-to-end regression tests. Compare against
   float64 where feasible, not only against fp32 absolute output.
7. Record a before/after HOGENOM report containing exact commands, hardware,
   software versions, warmup policy, repeat counts, medians/distributions, loss,
   gradient norms, repeatability, convergence behavior, memory, and runtime.

## Numerical success criteria

On representative HOGENOM subsets and the 1,055-family workload, centered mode
must preserve finite outputs and the model's likelihood convention while showing
a measurable stability benefit in at least one relevant quantity, such as:

- closer agreement of loss or gradients with a converged float64 reference;
- lower repeated-evaluation variation;
- reduced finite-difference discrepancy;
- a lower trustworthy optimization gradient floor;
- fewer false line-search/convergence decisions caused by fp32 quantization.

Report absolute and relative errors and identify which stage supplies the gain.
Distinguish Pi row quantization from scalar family reduction, atomic-order noise,
fixed-point truncation, and optimizer conditioning. A neutral result is useful;
do not tune tolerances until the experiment appears successful.

## Performance budget

The target is no more than a few percent end-to-end slowdown relative to the
same commit's normal path on the same GPU. Use 3% as the working target. A result
above 5% is not acceptable without a profile-backed redesign and explicit
discussion. Measure at least:

- forward-only loss evaluation;
- loss plus full first-order gradient;
- a representative optimization iteration or fixed evaluation sequence;
- the full HOGENOM batch shape, because small fixtures can be launch-bound in a
  different way.

Use warmed Triton caches, synchronize CUDA around timings, alternate baseline
and candidate runs where practical, and report medians over enough repetitions
to resolve a 3% difference. Profile before optimizing. Avoid exact per-row-max
passes in steady wave iterations unless measurements show their benefit pays for
the extra row traversal; the existing heuristic-offset note was written to avoid
roughly 1.25x structural forward overhead.

## Initial gates and references

- `tests/test_optim_golden.py` and `gpurec/docs/optim/golden_test.md`: frozen
  666x80 value/gradient regression and runtime context.
- AleRax toy fidelity tests: independent likelihood-convention checks.
- `docs/full_dataset_convergence.md`: known fp32 floors and confounders.
- `gpurec/docs/optim/newton_polish.md`: HOGENOM precision observations.
- `gpurec/docs/optim/golden_baseline.md` and profiling notes: existing runtime
  methodology.

Before broad edits, create a kernel/consumer inventory and a small offset-aware
reference implementation in PyTorch or float64. Work in reviewable phases, with
each phase retaining correctness and timing evidence. Do not regenerate pinned
goldens merely to make a changed implementation pass; first explain and validate
every intentional numerical difference.

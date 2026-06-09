# Branch Intent Notes

This document records what each remaining branch appears to have been trying to do, based on the branch tip commit message, nearby history, and diff shape.

Some branches are not distinct ideas on their own:
- Several `codex/aggressive-*` and `codex/core-audit-*` refs all point to the same FD Newton Hessian refactor commit.
- Several `origin/*` refs are remote-tracking mirrors of local branches.
- Some `backup/*`, `list`, `main`, `production`, and `worktree-*` refs are snapshots or aliases rather than independent lines of work.

## Pi and backward experiments

- `agent/pi-anderson-prototype`: Prototype for warm-started, relaxed Pi-adjoint backward updates, likely exploring an Anderson-style acceleration idea.
- `agent/pi-budget-policy`: Turns warm-Pi benchmark results into policy recommendations, comparing cold vs warm runs and suggesting budgets.
- `agent/pi-warm-cache`: Correctness fix for Pi-adjoint warmstart caching so cached tensors and cache flags stay in sync.
- `agent/triton-direct-grad-accum`: Explores a Triton-side self-loop layout that writes child-edge weights directly to reduce extra accumulation steps.
- `candidate3-fused-self-loop`: Stronger prototype that fuses self-loop iteration and parameter-gradient accumulation into one Triton path.
- `centered-pi-param-vjp-diagnostics`: Tunes the centered-Pi forward/backward path by deferring fused-block recentering by default to reduce overhead.

## GMRES and exact/backward-path work

- `clean/gmres-fixed`: Adds a real `gmres` self-loop solver option through autograd, execution, solver options, and tests.
- `lean-basic-functionality`: Minimal GMRES diagnostic branch focused on backward graph replay measurement.
- `work/gmres-experiments`: Experiment harness for measuring backward overhead and CUDA-graph replay behavior.
- `sparse-corrected-backward`: Makes the Pi/Pibar path exact again via `sparse_corrected` mode and full-chain gradient validation.
- `tmp/weighted-pi-precompute`: Broad weighted-receiver/Pi refactor to support non-uniform receiver weights end to end.
- `worktree-agent-acdd66df`: Worktree snapshot of `sparse-corrected-backward`, with the same intent as that branch.

## Contracts, audits, and refactors

- `codex/backward-layout-audit`: Pins the current backward auto-wrap layout as a baseline.
- `codex/bwd-pruning-policy`: Extracts backward pruning into a dedicated policy module.
- `codex/bwd-self-loop-policy`: Characterizes how backward self-loops are handled.
- `codex/scheduler-policy-audit`: Characterizes scheduler candidate selection and locks down policy choice.
- `codex/dts-layout-contract`: Adds a DTS layout contract helper and wires it into DTS-backed code.
- `codex/layout-contracts`: Makes parameter layout assumptions explicit with contracts and tests.
- `codex/origination-contract`: Introduces a typed origination-prior contract with validation and normalization.
- `codex/pi-output-intent`: Makes forward-pass output behavior explicit with an internal intent object.
- `codex/validation-contract`: Centralizes uniform-chunk validation helpers.
- `codex/likelihood-root-rows`: Proves root-row likelihood behavior matches full-Pi likelihood behavior.
- `codex/chunk-gradient-accumulator`: Makes chunked Pi backward accumulation explicit.
- `codex/chunk-gradient-evaluator`: Gates the chunked bf16 gradient path so it only runs when eligible.
- `codex/chunk-readonly`: Consolidates the chunked read-only evaluation path.
- `codex/grad-accumulator`: Adds a reusable gradient accumulator helper.
- `codex/eval-autograd-consolidation`: Consolidates the resident autograd solve path.
- `codex/eval-export`: Consolidates the export-state solve path.
- `codex/eval-nograd`: Consolidates the resident no-grad evaluator path.
- `codex/eval-static-gradient`: Consolidates the resident gradient-evaluation boundary.
- `codex/eval-consolidation-tests`: Adds guard tests for the evaluator consolidation refactor.
- `codex/preprocess-cache-batching`: Batches cached family preprocessing misses.
- `codex/preprocess-no-cache-batching`: Batches uncached family preprocessing.
- `codex/bench-1000-preflight`: Adds a preflight/setup-only benchmark mode with JSONL progress reporting.
- `codex/bench-1000-diagnosis`: Adds more benchmark telemetry and knobs to diagnose the 1000-family case.
- `codex/bench-dataset-progress`: Threads progress reporting into dataset preprocessing and model setup.
- `codex/cpp-surface-audit`: Guards the public/legacy/diagnostic classification of the C++ pybind surface.
- `codex/script-test-surface`: Tightens ownership and migration/deletion rules for tracked scripts and tests.
- `codex/env-manifest`: Documents which `GPUREC_*` env vars are public/runtime surface vs internal.
- `codex/env-options-audit`: Stricter follow-up audit of env-flag ownership and categorization.
- `codex/source-archive-preflight`: Validates checkpoint theta before restore.
- `codex/candidate1-triton-child-edge`: Removes inclusion-DAG preprocessing payloads to reduce footprint on the lean path.
- `codex/create-production-from-lean-fast-path`: Documents why the lean fast path is or is not production-ready.
- `codex/e-step-bench`: Pushes likelihood accumulation to fp64 and threads a reduction helper through the E-step path.
- `codex/hogenom-end2end-optimization`: Adds retained step checkpoint archives and wires them through the HOGENOM optimization workflow.
- `codex/small-family-batching`: Adds entropy utilities and bounded LBFGS diagnostics for small-family runs.

## Shared FD Newton refactor line

These refs all point to the same tip commit and represent one shared refactor, not separate feature branches:

- `codex/aggressive-api`
- `codex/aggressive-backtracking`
- `codex/aggressive-data`
- `codex/aggressive-kernels`
- `codex/aggressive-rust`
- `codex/core-audit-backtracking`
- `codex/core-audit-common`
- `codex/core-audit-data`
- `codex/core-audit-inference`
- `codex/core-audit-kernels`
- `codex/core-audit-parameters`
- `codex/core-audit-scheduling`

Intent: extract FD Newton Hessian state, state-match checks, BFGS update, and refresh logic out of `_fd_newton.py` into `_fd_newton_hessian.py`, while keeping `_fd_newton.py` as the runtime entrypoint and documenting the compatibility boundary.

## Lean, mainline, and staging branches

- `clean-slate`: Planning snapshot that lays out a roadmap for parsing, batching, descent, backtracking, and adaptive exact-vs-approximate gradients.
- `lean-fast-path`: Prunes the runtime surface down to the measured high-performance uniform path for the 1000-tree benchmark.
- `lean-scheduled-optimizers`: Rebuilds the lean fast path around the scheduled-optimizer story while removing historical baggage.
- `list`: Another name for the current mainline tip, with no separate intent beyond mirroring `main`.
- `main`: Active baseline centered on native chunk mini-batch optimization.
- `production`: Workflow refactor branch that extracts FD Newton Hessian helpers into a private sibling module.

## Backup and mirror refs

- `backup/lean-scheduled-optimizers-before-fastpath-20260511-130658`: Snapshot documenting AleRax ScaledValue performance options.
- `backup/production-pre-final-2026-05-27-171734`: Snapshot of production likelihood-optimization work before later protection changes.
- `backup/production-pre-rebase-protection-2026-05-27-170428`: Another pre-protection snapshot of the same production line.
- `backup/production-protect-2026-05-27-171405`: Records the fallback-budget `test_trees` route.
- `backup/source-archive-preflight-protection-2026-05-27`: Validates checkpoint theta before restore.
- `origin/clean/gmres-fixed`: Remote-tracking mirror of `clean/gmres-fixed`.
- `origin/codex/e-step-bench`: Remote-tracking mirror of `codex/e-step-bench`.
- `origin/codex/hogenom-end2end-optimization`: Remote-tracking mirror of the HOGENOM optimization branch.
- `origin/lean-basic-functionality`: Remote-tracking mirror of `lean-basic-functionality`.
- `origin/lean-fast-path`: Remote-tracking mirror of `lean-fast-path`.
- `origin/main`: Remote-tracking mirror of `main`.
- `origin/production`: Remote-tracking mirror of `production`.

## Takeaways

- The repo has several stacked branches that are really one idea split into many checkpoints.
- The main recurring themes are Pi/backward solver experimentation, evaluator consolidation, contract/hygiene work, and benchmark instrumentation.
- A handful of branches are just snapshots, mirrors, or worktree aliases and should be treated as such when reconstructing history.

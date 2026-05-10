# gpurec Codex Optimization Conversation Summary

This document summarizes the long Codex conversation about optimizing `gpurec`, mainly the global/uniform likelihood and gradient path on `test_trees_1000`, with later work extending to genewise/specieswise modes and global parameter optimization.

## Overall Arc

The conversation was a sustained profiling and optimization campaign. The recurring workflow was:

- profile with Nsight Systems / Nsight Compute, not the Torch profiler;
- identify the actual bottleneck;
- prototype alternatives behind flags;
- verify parity, finite differences, and gradient behavior;
- document results in `docs/`;
- commit regularly.

The code moved from a sparse/PyTorch-heavy uniform implementation toward a more fused Triton/CUDA pipeline. By the later full-pipeline measurements, the 1000-family global/uniform likelihood was around `2.3-2.7 s`, and the full fp32 forward+backward baseline was around `8.68 s` before the newest Proposal 0 ancestor-batching experiments.

## Uniform Forward

Initial profiling showed the uniform forward path was dominated by orchestration, sparse/PyTorch pieces, and synchronization. The original path spent hundreds of milliseconds in sparse/Pibar handling, even though useful GPU kernel time was much smaller.

The major accepted forward change was a fused ancestor-aware Triton uniform wave kernel. Instead of materializing dense or sparse transfer matrices, the kernel computes uniform `Pibar` by subtracting ancestor contributions through species parent traversal. This became the best default after multiple alternatives lost in profiling.

Rejected or non-default alternatives included:

- a signed linear-operator / SpMM-like formulation, which was mathematically plausible but slower or numerically fragile without guards;
- a two-kernel "compute Pibar then DTS" path, which was correct but slower than the fused one-kernel path;
- precomputed ancestor tables and CSR/SpMM variants, which lost to parent-pointer traversal;
- a fused Triton CSR path, which removed PyTorch sparse overhead but still had worse memory access than the tiny cache-hot parent table.

Tensor cores were judged not very applicable to the current uniform fused kernel, because the work is dominated by reductions, `exp2` / `log2`, tree traversal, scalar/index traffic, and irregular memory access rather than dense GEMM.

A practical win was removing adaptive fixed-point convergence polling. A `fixed_iters_Pi=6` mode was added and retained as the high-level default. It removed many `.item()` synchronization points and reduced forward time while matching NLL at fp32 print precision in the tested workload.

Main forward document: `docs/uniform-forward-profile.md`.

## Scaling To 1000 Trees

When scaling from 3 families to 1000, a fully resident model was not viable. Preprocessing memory and full resident `Pi` / `Pibar` state would have been too large, so the solution became chunked execution.

Forward throughput for all 1000 trees reached about `2.688 s` total in an earlier chunked configuration, not per tree. Later optimized forward profiles were around `2.3 s` for 1000 families depending on chunking.

Preprocessing was initially the real wall-clock problem, around `590 s`. The follow-up work:

- skipped unused heavy CCP detail structures;
- added content-hash CPU caching;
- found and removed unnecessary O(C^2) clade inclusion DAG construction in light mode;
- made light multi-family preprocessing the default.

After that, public model build was roughly `0.157 s` for 10 trees, `0.726 s` for 50, `2.162 s` for 150, and about `15.6 s` for all 1000 without cache. Cache-hit construction was much faster.

Important commits from that phase included `cb04bdd`, `b8f0366`, and `3442acd`.

## Uniform Backward

Backward started much more PyTorch-heavy than forward. Profiling showed major time in sparse matmuls, `cat`, scatter/index kernels, direct child-gradient materialization, and the fused wave backward kernel.

The first safe cleanup avoided materializing several `[C, S]` constants, mostly reducing memory. Then the backward was progressively kernelized:

- fused uniform cross-`Pibar` VJP replaced `cat -> sparse mm -> sparse mm transpose -> index_add`;
- a descendant/tree gather version removed expensive ancestor-scatter atomics and became default;
- backward-only DTS recomputation and child-gradient materialization moved into Triton/fused paths;
- fp64 support was unified through the same fused kernels where possible.

The 10-family backward moved from roughly `210 ms` through `160 ms`, then to around `72 ms` after deeper kernelization. Later proposal rounds pushed 50-family backward from about `230 ms` to `181 ms`, and then through many follow-up optimizations toward about `121 ms` before a correctness fix.

A serious correctness issue was found: fused uniform backward had used incomplete ancestor correction in the self-loop `Pibar` VJP. This was fixed in commit `221f16a`, with stricter high-iteration correctness tests in `3b94b66`. After the fix, tight reference agreement was restored.

Accepted ideas included compact Pibar scratch, RHS views, active-mask kernels, merged DTS accumulation, device scalar params to remove syncs, staged Pibar VJP, gather-based self-loop/speciation rewrite, hybrid pruning, parent-reduced DTS recompute, and thresholded side pruning.

Rejected or opt-in ideas included broad no-split rewrites, CUDA graph replay, scratch pooling, parent-tiled/ragged DTS variants, and some zero-store or no-split shortcuts.

Main backward documents include:

- `docs/uniform-backward-profile.md`
- `docs/uniform-backward-fp32-fused-profile.md`
- `docs/uniform-backward-third-pass-proposals.md`
- `docs/uniform-backward-fourth-pass-proposals.md`

## Genewise And Specieswise

The genewise path was initially slower because it materialized `[W, S]` constants and DTS parameter tensors around kernels originally optimized for global uniform mode.

The main genewise forward fixes made Triton load `[G, S]` or `[G]` parameters through `family_idx` instead of materializing wave-shaped arrays. This brought genewise 1000-tree forward from about `4.468 s` to `2.328 s`, essentially matching global uniform.

Genewise/specieswise backward was then generalized from the optimized global fused DTS machinery. Proposal 3+4 work cut old generic DTS/Pibar behavior dramatically. For example, 10-family old generic was `344 ms` and `6.6 GB`, while Proposal 3+4 was about `95.8 ms` and `2.0 GB`.

A specieswise slowdown turned out mostly to be benchmark configuration: pruning and compact Pibar flags were not aligned. With aligned settings, specieswise was only about `1-5%` slower than genewise. A stronger AleRax specieswise regression test was added.

Important commits from this area included `1333b23`, `8c585a5`, `f2b3151`, `8346ad3`, `b953def`, `c43d762`, and `cc6d93e`.

## Full Forward+Backward Pipeline

A full efficient pipeline plan was written in `docs/forward-backward-full-pipeline-plan.md`. It covered static preprocessing, chunk scheduling, saved-state contracts, fused backward, E adjoints, memory pooling, correctness gates, and profiling gates.

Then a full pipeline harness was implemented in `profiling/bench_uniform_forward_backward_pipeline.py` and committed as `fbd7b6f`. It streams chunks through forward/backward, solves shared `E` once, accumulates adjoints, and runs theta/E VJP once.

Measured fp32 full 1000-family baseline:

- best chunk size around `75`;
- forward about `2.34 s`;
- backward about `6.34 s`;
- total about `8.68 s`;
- peak about `15.6 GiB`.

Chunk `50` was safer but slower, chunk `100` used about `20.4 GiB` and was slower, and chunk `125` OOMed. The full pipeline was clearly backward-dominated, with `_wave_backward_uniform_kernel` the major bottleneck.

## Global Parameter Optimization

A parameter optimization plan was written, then `optimize_global_rates_lbfgs` was added in commit `b216939`. The selected production strategy was fp32 PyTorch `LBFGS` with strong Wolfe line search, rate clipping, and an interior start such as `(0.05, 0.05, 0.05)`.

On `test_trees_100`, it reached the AleRax neighborhood quickly, around evaluation 5, with total helper time around `2.25 s`. On the first 100 trees from `test_trees_1000`, a longer run took about `31.6 s` optimizer time and plateaued around evaluation 22 / `19.4 s`.

A convergence check showed only fp32 objective plateau convergence, not a strict fp64 optimum. More Pi iterations and more Neumann terms did not move the fp32 result. fp64 forward shifted NLL by only about `0.0195` bits, but full fp64 backward did not fit in 24 GB.

## bf16 Work

The first bf16-start optimizer attempt cast saved bf16 state back to fp32 for backward. It was slower and used more memory, so it was rejected.

A resident-bf16 phase was then implemented: theta, saved forward tensors, static state, and backward inputs stayed bf16 during the initial phase, with fp32 internal accumulation where required. The best handoff strategy was one bf16 eval/update followed by fp32 polish; threshold-based handoff was unreliable or slower.

A major bf16 backward issue was found and fixed in commit `de2c5fd`: bf16 global atomics into `pibar_corr` were extremely slow. Keeping `pibar_corr` fp32 made bf16 backward drop from about `2.97 s` to `0.672 s` on the first 100 families, close to fp32 `0.633 s`. bf16 forward improved modestly, about `1.23x`.

The reason bf16 was not 2x faster is that the hot kernels are not simple coalesced fp32 DRAM streams. They are L2/cache/ancestor-walk limited, with fp32 correction scratch, index traffic, scalar math, atomics, and metadata.

## Ancestor Batching / Proposal 0

The user pushed on whether the ancestor walk had to remain uncoalesced. The conclusion was no: it is an artifact of one Triton program owning one clade row and walking parent pointers independently. A wave-batched tree-DP formulation was proposed.

A proposal document was created, then multiple prototypes were implemented and profiled:

- Proposal 0: 2D Triton self-loop prototype. Fastest, about `1.9x` backward speedup: 50 families `327 ms -> 171 ms`, 100 families `627 ms -> 319 ms`. Memory-heavy: about `14.2 GB` at 50 and `21.3 GB` at 100.
- Proposal 1: staged tree-DP. Correct but slower, around `456 ms` at 50 and OOM at 100. Rejected in current form.
- Proposal 5: CUDA tree/no-split path. Conservative candidate: about `1.46-1.49x` speedup with lower memory risk, but it only handles no-split waves.

The final report for this phase was `docs/uniform-backward-ancestor-batching-experiments.md`, finalized around HEAD `43853d7`.

The memory discussion concluded that `21 GB` is not inherently disqualifying if it buys runtime, but it is too close to the edge on a 24 GB GPU without full-pipeline proof. Chunk `50` with Proposal 0 looked promising because it would save roughly `7 GB` while costing only an estimated `233 ms` backward versus chunk `100`.

At the cutoff, agents were still running a Proposal 0 full-pipeline sweep over chunk sizes and `max_wave_size`. The memory audit found likely savings from moving Proposal 0 dispatch before generic scratch allocation, adding accumulated-parameter mode, reusing buffers, and using saved `Pibar` / row max for `inv_denom`.

## Current Practical Takeaways

- The forward path is much closer to optimized than the backward path.
- Conventional sparse matrix formulations for uniform ancestor handling are not competitive with the fused parent-walk kernel under the current layout.
- The full 1000-family training step is backward-dominated.
- Proposal 0 is the most promising speed path for backward, but needs memory work and full-pipeline validation.
- Proposal 5 is the safer conservative backward acceleration path, but only covers no-split waves.
- bf16 is useful but not transformative; one bf16 optimizer step before fp32 polish is the best measured policy so far.
- Parameter optimization with fp32 LBFGS is now productized and fast enough for the tested 100-family cases.

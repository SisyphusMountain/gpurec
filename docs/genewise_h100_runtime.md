# Genewise fit runtime on the Coleman 1007-leaf dataset (H100, 2026-09-02)

Goal: make `gpurec fit --mode genewise` (the `fit_dtl` genewise recipe, certificate on) faster on
the Coleman et al. bacterial dataset without changing the fitted likelihood.

Dataset: species tree `ReferenceTree.nwk` (1007 leaves, 2013 species nodes) and 5124 ALE
gene-family files (23.3M clades in total). The single 400,918-clade family `COG3676_X` is
excluded, leaving 5123 families and 22.9M clades. With the default `clade_budget=315_000` the
model is 79 batches of roughly 150-200 waves each.

Hardware: CC-IN2P3 cluster, one NVIDIA H100 NVL (94 GiB), 12 CPU cores per job. Cluster
scripts live in `benchmark/cc/` (see `env.sh`, `sbatch_h100.sh`, `run_genewise.py`,
`run_genewise_sharded.py`).

## Where the time went (before)

Measured with the code at commit `817007e6` (the state of `main` plus the job scripts):

| stage | measured | note |
|---|---|---|
| model build, 5123 families | 775 s | repeated at every rebatch, tier change and for the certificate (4-15 times per fit) |
| first gradients at full scale | > 300 s each | Triton JIT compilation, not GPU work |
| one gradient, steady state, 500 families / 13 batches | 10.8 s | GPU 96 % busy; 45 % of kernel time in `_update_reconciliation_likelihood_kernel`, 21 % in `_apply_reconciliation_self_loop_transpose_kernel` |
| one Hessian (3 HVP probes), 500 families | 173 s | forward solve + adjoint cache rebuilt once per probe |
| 40-family end-to-end fit | 354 s | 4 model builds |

Three root causes were found:

1. **The shipped Rust extension was a debug build.** `gpurec/gpurec_preprocess.abi3.so`
   contained the overflow-check strings only a debug build emits, and a release build of the
   same source produced byte-identical output 11x faster. On top of that, ALE files grow as
   clades x leaves (the `#set-id` section lists every leaf of every clade), so parsing looked
   quadratic in the clade count; the parser now reads those lists in place and computes the
   schedule depth in one topological pass (output byte-identical, another 1.5x on large
   families). Per-family parse time is now proportional to file size (~390 MB/s).
2. **Rust to Python interchange was one JSON string** (5.3 GB for 5123 families): `json.loads`
   took 98 s, the list-to-tensor conversion 54 s, and the process held 57 GB of Python objects.
   The families are now parsed once per fit (`ParsedFamilies` in
   `crates/gpurec-preprocess/src/pybridge.rs`); every rebuild re-plans the active subset and
   receives numpy arrays (zero-copy from Rust) that become tensors with `torch.from_numpy`.
   All batch tensors are `torch.equal` to the legacy path (`tests/test_parsed_families_equivalence.py`).
3. **Three Triton kernel arguments that change with every wave were `tl.constexpr`**
   (`n_ws` in the transfer-subtree VJP kernel and its second-order twin, `MAX_TILES` in the
   multi-split DTS reduction, `n_tiles` in the max-transfer VJP reduce). Triton therefore
   compiled one kernel variant per distinct wave shape: the shared cache held 19,615 variants,
   and a first full-dataset gradient spent more than 10 minutes compiling while a steady-state
   gradient takes 58 s. They are now runtime integers (only used for index arithmetic, so the
   floating-point work is unchanged).

Two smaller items:

- `_analytic_hessian` in `gpurec/fit/genewise_fit.py` now streams the batches in the outer
  loop and applies the 3 probes to one forward solve + adjoint cache per batch (the library's
  multi-batch HVP rebuilt both per probe). 2.6x on the Hessian at 500 families (173 s to 66 s);
  the difference to the old result (0.10) is below the run-to-run noise of the atomics (0.15-0.18).
- The warm-adjoint memory gate only counted the gradient cache; the HVP warm start keeps one
  more `[total_clades, S]` cache per probe (3x). On the H100 a 500-family run passed the gate at
  32 GiB and ran out of memory at 90 GiB inside the Hessian. `warm_adjoint_fits` now takes
  `resident_caches` and the model passes 1 + 3.

Things checked and ruled out: Python's cyclic garbage collector (disabling it changes nothing),
a larger `clade_budget` (13 to 3 batches gives 9 % on the gradient; the work is GPU-bound, not
launch-bound), warm vs cold adjoint (same speed), and Triton `num_warps` for the five hottest
kernels (`benchmark/cc/sweep_num_warps.py`: every alternative is within 2 % of the current
default or slower; 2 warps on the tangent self-loop kernel is 6.7x slower).

## After

| stage | before | after |
|---|---|---|
| model build, 5123 families (first) | 775 s | 18 s |
| rebuild over a subset | 775 s | a few seconds (re-plan only) |
| first full-dataset gradient (includes JIT) | > 300 s | 64 s |
| steady-state full-dataset gradient | 58 s (hidden behind JIT) | 58 s |
| 40-family end-to-end fit | 354 s | 68 s, same NLL (115604.90 bits), same 25 steps / 4 builds |
| 500-family end-to-end fit | 2290 s (old code, warm cache forced off so it fits the H100), NLL 1618463.746 bits, 230 steps, 12 builds, 485/500 certified | 1017 s, NLL 1618463.769 bits, 231 steps, 11 builds, 490/500 certified |
| 5123-family end-to-end fit, 1 GPU | crashed after 2 h 34 min (CUDA out of memory inside the old streaming Hessian after the first rebatch; it had reached iteration 20 with 2981 active families at 143 min) | **5353 s = 89 min**, NLL 9048956.572 bits, 226 steps, 16 builds, 5073/5123 certified converged, peak 70 GiB (reached iteration 20 with 2983 active at 52 min; first tier done at 62 min; pi=64 tier done at 67 min; certificate 22 min) |
| 5123-family fit, 2 GPUs sharded | - | job `full_v3_2gpu` (Slurm 57473861) was running when the session lost cluster access (IN2P3 Kerberos servers unreachable); both shards reached their first Newton iteration in 202 s vs 394 s single-GPU. Result: `/sps/biometr/emarsot/gpurec-runs/results/full_v3_2gpu.json` (per-shard logs alongside) |
| 5123-family fit, 4 GPUs sharded | - | not run: no node with 4 free H100s was available during the session (`GPUS=4 ... run_genewise_sharded.py --n-shards 4` is ready) |

Fitted per-family log2 rates (old vs new code, 500 families): median difference 1.5e-6, 99.6 % of
families within 1e-3, worst 0.063 on one family whose duplication rate is tiny (a flat direction of
its likelihood); on 40 families the largest difference is 4.6e-4. The certified total NLL agrees between old and new code to 0.02 bits on 500 families (1.6M bits
total, 1.4e-8 relative) and to 0.001 bits on 40 families; the gradient is not bitwise
reproducible (atomic accumulation), so the per-family convergence flags in the tail differ by a
few families between any two runs, old or new. The full-dataset fixed run followed the same
rebatch trajectory as the crashed baseline (19 %, 43 % converged at iterations 8 and 12; 2983 vs
2981 families active after the first drop).

## What is left

The gradient itself is GPU-bound (96 % busy) with two kernels taking two thirds of the time,
so further gains need kernel work (`ncu` on `_update_reconciliation_likelihood_kernel` and the
backward self-loop kernel). The Hessian costs about 7 gradients (and 4x more at the
`pi_iters=64` certificate tier, where the tangent self-loop kernel runs 64 fixed iterations);
the recipe evaluates it every 5 iterations and once for the certificate. Independent families
shard perfectly across GPUs (`run_genewise_sharded.py`).

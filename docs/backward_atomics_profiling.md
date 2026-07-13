# Backward `atomic_add` vs buffer-then-reduce: a profiling answer

**Question.** The backward gradient kernels accumulate with `tl.atomic_add` (scatter of the
clade adjoint into `accumulated_rhs`, and the D/S/T rate-gradient reduction). Would it be
more efficient to instead write partial contributions to a buffer and reduce them at the end
of wave processing?

**Short answer: no — it's not worth it.** The atomic-bearing kernels are ~4.3% of total GPU
time, and within the main one the atomics are only ~16% of its memory traffic while the
kernel is **latency-bound on its input gathers**, not on the atomic writes. A buffer-then-reduce
would *add* memory traffic and could only ever target a sub-1% slice. The real backward cost is
the **self-loop Neumann kernel at 57%**, which uses no atomics at all.

Hardware: RTX 4090 (sm_89). Path: full archaea dataset (5379 families, S=119, 25 batches),
**float32** (the production path), solver pi=64/nt=64, at the converged checkpoint. Harness:
`scripts/profile_backward_atomics.py`.

## 1. Where the backward time actually goes (nsys)

`nsys profile … ; nsys stats --report cuda_gpu_kern_sum` over 3 forward+backward iterations:

| kernel | % total GPU time | ms/iter | role | atomics? |
|---|---:|---:|---|---|
| `_wave_backward_uniform_2d_jt_kernel` | **57%** | 1577 | self-loop Neumann backward (J^T apply) | **no** — in-block tree reduction |
| `_wave_step_kernel` | 28% | 769 | forward Pi wave | no |
| `_dts_cross_backward_accum_kernel` | **4.0%** | 111 | clade-adjoint scatter + D/S/T rate grad | **yes** |
| `_uniform_cross_pibar_vjp_tree…` | 3.0% | 107 | pibar VJP backward | no |
| `_dts_ge2_stage1_kernel` | 1.6% | 45 | DTS gather | no |
| `_receiver_grad_from_pibar_self_loop_kernel` | 1.4% | 39 | receiver-weight grad | no |
| `_wave_backward_uniform_param_store_kernel` | **0.3%** | 8 | rate-grad store | **yes** |
| (everything else) | <3% | — | torch elementwise, reductions, e-step | — |

Total ≈ 2.77 s/iter. **The two atomic kernels together are ~4.3% of total** (4.0% + 0.3%).
The backward is dominated by the **self-loop kernel (57%)** — the same kernel that carries the
float64 race (see `docs/full_dataset_convergence.md`), and it contains no atomics.

## 2. Inside the atomic kernel (ncu)

`ncu` on `_dts_cross_backward_accum_kernel` (10 instances, grids of 9k–11.7k blocks × 256):

| metric | value | reading |
|---|---|---|
| compute SOL | 28–34% | not compute-bound |
| memory SOL | 28–34% | not bandwidth-bound |
| long-scoreboard stall | **26–30%** | **latency-bound on global loads** |
| duration | 16–33 µs | — |

Memory-traffic composition (sectors):

| op | share of global traffic |
|---|---:|
| **load** (`op_ld`) — gathering `Pi_l, Pi_r, Pibar, E, log_pD/pS, …` | **~79%** |
| **atomic** (`op_red`) — the `atomic_add` accumulations | **~16%** |
| store (`op_st`) | ~6% |

Note: a relaxed `tl.atomic_add` whose return value is unused compiles to a **`red`
(reduction)** instruction, not `atom` — so the atomic traffic shows up under `op_red`
(the `op_atom` counters are 0 here).

**Interpretation.** The kernel spends its time *waiting on the loads that feed the DTS event
terms* (~79% of traffic, ~28% long-scoreboard stalls), not on the atomic writes (~16%). It is
at ~30% speed-of-light — latency-bound, not throughput- or atomic-bound.

## 3. What this means for the buffer-then-reduce idea

- **Optimization ceiling is < ~1% of total.** Atomics are 16% of a kernel that is 4% of total
  → ~0.6% of total GPU time is the *traffic-share* ceiling, and even that is unreachable because
  the kernel's latency is gated by the **loads**, not the atomic stores. The param-store kernel
  (the other atomics) is 0.3% of total.
- **Buffer-then-reduce would likely be net-negative here.** Materializing per-edge partials
  *adds* store traffic, and the reduce pass *adds* load traffic — on a kernel already bound by
  load latency. You'd also reintroduce a scatter (arbitrary child-clade targets) for the reduce,
  i.e. atomics or a sort.
- **The contention is low.** With atomics only 16% of traffic and the kernel latency-bound on
  loads, there is no atomic-serialization hotspot to relieve. (Determinism, not speed, is the
  only thing a reduction would buy — and that wouldn't fix the self-loop race, which is a
  different kernel.)
- **Where the time is: the self-loop (57%).** If backward performance is the goal, that kernel
  is the target — and it is *also* the one to restructure for the float64 race fix (shared-memory
  reduction), so correctness and performance work coincide there.

## 4. Caveats
- Measured in **float32** (production). float64 is separately broken (the self-loop race) and was
  not benchmarked.
- Contention is **data-dependent** (CCP DAG shape); these numbers are the archaea operating point.
  The conclusion (atomics ≪ load latency; atomic kernels ≈ 4% of total) is unlikely to flip, but
  a different dataset/`S` should be re-measured rather than assumed.
- `nsys` time shares are wall-time per kernel name aggregated over the run; `ncu` SOL/stall/traffic
  are per-kernel-instance (a representative sample after `--launch-skip 250`).

## Reproduce
```
python scripts/profile_backward_atomics.py            # sanity-run the harness
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx \
    -o /tmp/bwd_prof -f true python scripts/profile_backward_atomics.py
nsys stats --report cuda_gpu_kern_sum /tmp/bwd_prof.nsys-rep
ncu --kernel-name regex:"_dts_cross_backward_accum_kernel" --launch-skip 250 -c 6 \
    --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    --csv python scripts/profile_backward_atomics.py
```

# Optimization Opportunities: Sparse k=16 Pibar at S=20,000

**Hardware**: RTX 4090 — 1,008 GB/s memory BW, 82.6 TFLOPS FP32, 72 MB L2 cache
**Problem size**: S=19,999 species, C=40,859 clades, 58 waves
**Source**: profiling of `tests/profiling/ncu_sparse_pibar.py`

---

## 0. Total Pi convergence time

**608 ms** for sparse k=16, vs **3,522 ms** for dense cuBLAS = **5.8× speedup**.

Convergence: 326 total iterations across 58 waves (39 waves @ 4 iters, 18 @ 9, 1 @ 8).

### Convergence check overhead

The per-iteration convergence check (`torch.abs(Pi_new - Pi_W)[sig].max().item()`)
forces a GPU→CPU sync via `.item()`, breaking kernel pipelining.

| Strategy | Total iters | Wall time | Notes |
|----------|----------:|----------:|-------|
| Check every iter (from 3) | 326 | **613 ms** | Current behavior |
| Check every 4 (from 3) | 380 | **548 ms** | 16% fewer checks, 17% more iters |
| Check every 8 (from 7) | 608 | **695 ms** | Too many wasted iters |
| Fixed 4 iters | 232 | **272 ms** | Underconverged for 19 waves |
| Fixed 9 iters | 522 | **562 ms** | 60% extra iters, 8% faster |

**Convergence checking costs ~50 ms (8%) due to sync-induced pipeline stalls.**

Check-every-4 is the sweet spot: saves 65 ms (11%) vs current by reducing sync
points while keeping iteration waste low (+17%).

### k=16 vs k=32 vs k=64

| Method | logL | logL diff | Pi max err | Time | Speedup vs Dense |
|--------|-----:|----------:|-----------:|-----:|-----------------:|
| Dense | 6226.82 | — | — | 3,522 ms | 1.00× |
| Sparse k=16 | 6227.05 | 0.234 | 11.20 | 628 ms | **5.61×** |
| Sparse k=32 | 6227.05 | 0.234 | 9.61 | 683 ms | **5.16×** |
| Sparse k=64 | 6227.05 | 0.233 | 8.41 | 828 ms | **4.26×** |

logL difference is identical across all k values — the transfer matrix is nearly
uniform, so increasing k only reduces error at extremely low-probability cells.
**k=16 is optimal**: fastest with no accuracy loss vs higher k.

---

## 1. DTS reduction optimization — IMPLEMENTED

### Problem

The old `_compute_dts_cross` reduced `dts_fused` output from [N_splits, S] → [W, S]
using **10 separate PyTorch kernel launches** (fill, scatter_amax, clone, fill, index,
sub, exp2, scatter_add, log2, add). Each op makes a full pass over [N,S] or [W,S]
data, totaling **~6.6 GB of memory traffic** for what should be a trivial operation.

This was **35% of total wave time** — more than either Triton kernel.

### Key insight

Splits within each wave were analyzed:
- **56/57 waves**: `reduce_idx` is a **permutation** (each clade has exactly 1 split).
  No reduction needed — just a permuted row copy.
- **Root wave** (wave 57): W=1, N=20,429. All splits reduce to a single clade.
  Requires actual logsumexp reduction.

The preprocessing already had this split between eq1/ge2 clades in the FP code path
(`seg_logsumexp` for ≥2 splits, `index_copy_` for ==1 splits in `likelihood.py:228-240`).
It just wasn't applied to the wave-based `_compute_dts_cross`.

### Implementation

**`build_wave_layout`** (`src/core/batching.py`): within each wave, splits are now
sorted with a composite key: eq1 clades first, then ge2 clades sorted by ascending
parent index (for CSR contiguity). Per-wave metadata includes:
- `n_eq1`, `n_ge2_clades` — split counts
- `eq1_reduce_idx` — permutation for direct copy
- `ge2_ptr`, `ge2_parent_ids` — CSR pointers for `seg_logsumexp`

**`_compute_dts_cross`** (`src/core/likelihood.py`):
```python
dts_term = dts_fused(...)  # [n_ws, S]
dts_r = torch.full((W, S), NEG_INF, ...)

if n_eq1 > 0:
    dts_r[meta['eq1_reduce_idx']] = dts_term[:n_eq1]    # permuted copy

if n_ge2_clades > 0:
    y_ge2 = seg_logsumexp(dts_term[n_eq1:], meta['ge2_ptr'])
    dts_r[meta['ge2_parent_ids']] = y_ge2                # fused reduction
```

### Results (nsys profiled, waves 1–3, S=20K)

**DTS reduction time (per wave):**

| Wave | Old (10 ops) | New (fill + index_copy) | Speedup |
|------|----------:|---------:|--------:|
| W=3411 | 6,554 µs | 829 µs | **7.9×** |
| W=2030 | 3,883 µs | 481 µs | **8.1×** |
| W=1367 | 2,590 µs | 315 µs | **8.2×** |

All 27 existing tests pass (16 wave_v2 + 11 wave_vs_fp).

---

## 2. Current per-wave cost breakdown (AFTER DTS optimization)

Per-kernel timing (average of 20 reps, isolated measurement):

| Wave | W | Pibar | wave_step | memcpy | DTS cross | Per-iter |
|------|------:|------:|----------:|-------:|----------:|---------:|
| 0 | 10,216 | 3.12 ms | 5.00 ms | 3.47 ms | — | 11.58 ms |
| 1 | 3,411 | 2.05 ms | 2.12 ms | 1.16 ms | 2.38 ms | 5.33 ms |
| 2 | 2,030 | 1.23 ms | 1.27 ms | 0.69 ms | 1.43 ms | 3.19 ms |
| 3 | 1,367 | 0.77 ms | 0.87 ms | 0.47 ms | 0.98 ms | 2.10 ms |

Per-iteration cost scales linearly with W — all kernels are memory-bandwidth bound.

### Amortized breakdown (wave 1, W=3411, 9 iterations):

| Component | Time | % of wave |
|-----------|-----:|:---------:|
| DTS cross (once) | 2.38 ms | 5% |
| Pibar kernel (9×) | 18.5 ms | 38% |
| wave_step kernel (9×) | 19.1 ms | 39% |
| memcpy DtoD (9×) | 10.4 ms | 21% |
| Conv check overhead | ~8 ms | — |

Note: conv check overhead is excluded from kernel percentages because it's a
sync-induced stall, not GPU compute. At the GPU level, the three kernels + memcpy
account for all time. At the wall-clock level, conv checks add ~15%.

---

## 3. Kernel-by-kernel analysis

### 3.1. `_wave_step_kernel` — 39% of amortized GPU time

**What it computes:**

For each clade `w` and species `s`, computes:
```
Pi_new[w,s] = logsumexp2(t0, t1, t2, t3, t4, t5, [t6])
```
where the 6–7 terms are:
- `t0 = DL_const[s] + Pi[w,s]` — duplication-loss
- `t1 = Pi[w,s] + Ebar[s]` — transfer (clade side)
- `t2 = Pibar[w,s] + E[s]` — transfer (species side, via Pibar)
- `t3 = SL1_const[s] + Pi[w, child1[s]]` — speciation (child 1)
- `t4 = SL2_const[s] + Pi[w, child2[s]]` — speciation (child 2)
- `t5 = leaf_term[w,s]` — leaf survival
- `t6 = DTS_reduced[w,s]` — cross-clade DTS (only for waves with splits)

Followed by numerically stable logsumexp2: `log2(Σ exp2(ti - max)) + max`.

**Grid**: `(W, ⌈S/32⌉)` — one thread block per (clade, species-block-of-32).

**For W=3411, S=19,999 (with DTS):**
- Unique traffic: W × (4 reads + 2 scattered + 1 write) × S × 4 = 1.91 GB

**Observed**: 2,120 µs → **effective BW ≈ 900 GB/s (89% of peak)**

**Optimization potential**: Low. Already at ~90% of roofline.

---

### 3.2. `_sparse_input_pibar_kernel` — 38% of amortized GPU time

**What it computes:**

For each clade `w` and output species `e`:
```
out[w,e] = log2(Σ_{j=0..k-1} M_T[topk_idx[w,j], e] × weights[w,j]) + c[w] + mt[e]
```

Sparse-input dense-output matmul: only k=16 input species contribute, but all S
output species are computed.

**Grid**: `(W,)` — one program per clade, tiles over S in chunks of BLOCK_S=256.

**Total across all clades (W=3411):**
- M_T reads: W × k × S × 4 = **4.37 GB** (with L2 reuse: M_T is 1.6 GB, ~2.7× reuse)
- Output writes: W × S × 4 = 273 MB

**Observed**: 2,050 µs → **effective BW ≈ 2,250 GB/s** (L2-amplified)
True DRAM BW ≈ 907 GB/s (90% of peak).

**Optimization potential**: Low. Saturates DRAM bandwidth.

---

### 3.3. `_dts_fused_kernel` — 5% of amortized GPU time

**What it computes:**

Per split, 5 DTS terms (D, T×2, S×2) + logsumexp + log_split_probs.

**Grid**: `(N, ⌈S/128⌉)`.

**Observed**: ~1,500 µs → **effective BW ≈ 900–1000 GB/s (90%+ of peak)**

**Optimization potential**: Low.

---

### 3.4. memcpy DtoD — 21% of amortized GPU time

The slice assignments `Pi[ws:we] = Pi_new` and `Pibar[ws:we] = Pibar_W` each copy
W×S×4 bytes. Two copies per self-loop iteration.

**For W=3411, S=19,999:**
- Two copies per iter: 1,160 µs at ~470 GB/s

**Optimization: write Pibar in-place**

The pibar kernel reads from Pi_W and writes to `out`. If `out` points to
`Pibar[ws:we]` directly, the Pibar memcpy is eliminated. No read-after-write
hazard (different tensors). Saves ~580 µs/iter = **11% of self-loop time**.

In-place Pi_new is NOT safe due to scattered child reads creating cross-block
dependencies within the same clade row. Double-buffering could help.

**Estimated savings**: 580 µs/iter × 9 iters = 5.2 ms per wave (~11%).

---

## 4. Remaining optimization opportunities

| Optimization | Savings (wave 1, 9 iters) | Complexity | Priority |
|-------------|---------------------------:|:----------:|:--------:|
| Check every 4 iters | ~65 ms total (11%) | Trivial | **HIGH** |
| Write Pibar in-place | ~5.2 ms/wave (11%) | Low | **HIGH** |
| Eliminate Pi copy (double-buffer) | ~5.2 ms/wave (11%) | Low | Medium |
| Permuted write in dts_fused | ~0.8 ms/wave (2%) | Very Low | Low |

### Estimated total speedup from remaining optimizations:

Current total: **613 ms** (check every iter from 3)
After check-every-4: ~548 ms (11% faster)
After also Pibar in-place + Pi double-buffer: eliminates ~21% of per-iter GPU cost
→ estimated ~440 ms (28% faster vs current)

---

## 5. Roofline summary

| Kernel | Traffic | Time | Eff. BW | % of peak | Bound |
|--------|--------:|-----:|--------:|----------:|-------|
| `_wave_step_kernel` | 1.91 GB | 2.12 ms | 900 GB/s | 89% | Memory |
| `_sparse_input_pibar_kernel` | 4.6 GB* | 2.05 ms | 2,250 GB/s* | 90%† | Memory |
| `_dts_fused_kernel` | ~1.4 GB | ~1.5 ms | ~930 GB/s | 92% | Memory |
| memcpy DtoD (2×) | 0.55 GB | 1.16 ms | 470 GB/s | 93%‡ | Memory (bidir) |

\* Includes L2 cache amplification (M_T reuse across clades)
† True DRAM BW ≈ 907 GB/s (90% of peak)
‡ Theoretical max for DtoD = peak/2 = 504 GB/s

**All three Triton kernels are within 90% of their memory-bandwidth roofline.**
The dominant remaining waste is the memcpy DtoD copies (21%) and convergence check
sync stalls (~8%).

---

## 6. Per-wave time distribution

Total: 608 ms across 58 waves. Top 10 by time:

| Rank | Wave | W | Iters | Time | % of total |
|------|------|------:|------:|-----:|-----------:|
| 1 | 0 | 10,216 | 8 | 188.2 ms | 31.0% |
| 2 | 1 | 3,411 | 9 | 89.0 ms | 14.6% |
| 3 | 2 | 2,030 | 9 | 53.2 ms | 8.7% |
| 4 | 3 | 1,367 | 9 | 35.4 ms | 5.8% |
| 5 | 4 | 948 | 9 | 24.7 ms | 4.1% |
| 6 | 5 | 668 | 9 | 17.5 ms | 2.9% |
| 7 | 43 | 2,130 | 4 | 14.1 ms | 2.3% |
| 8 | 42 | 2,080 | 4 | 13.5 ms | 2.2% |
| 9 | 44 | 2,040 | 4 | 13.4 ms | 2.2% |
| 10 | 57 | 1 | 4 | 12.4 ms | 2.0% |

Wave 0 (leaf wave, W=10,216, no DTS) dominates at 31%. Waves 1–5 converge in 9
iterations (vs 4 for most later waves), contributing 36% combined. Root wave 57
(W=1, 20K splits) is only 2% — dominated by `dts_fused`, not the self-loop.

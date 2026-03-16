# Fused Uniform-Pibar Kernel Optimization

## Summary

Three optimizations that together reduce compute_likelihood_batch time by 5x
at S=20K (1 family: 2.5s → 0.56s) and 3.4x at S=2K (10 families: 0.75s → 0.22s).

## 1. Fused Triton Kernel (`wave_step_uniform_fused`)

**File:** `src/core/kernels/wave_step.py` — `_wave_step_uniform_kernel`

**Problem:** The per-iteration self-loop had 5-6 kernel launches:
- `_compute_Pibar_inline`: 4 PyTorch ops (max, exp2, sum, log2+add) creating a [W,S] intermediate
- `wave_step_fused`: Triton kernel reading Pi + Pibar
- Convergence check: allocate [W,S] bool + diff tensors, GPU sync via `.item()`

**Solution:** Single Triton kernel per iteration that:
- **Pass 1 (online max+sum):** Single scan of Pi row computes both `row_max` and
  `row_sum = sum(exp2(Pi - max))` using the online rescaling trick. This avoids
  a separate max-then-sum two-pass approach.
- **Pass 2:** For each tile of S species:
  - Computes Pibar inline: `log2(row_sum - exp2(Pi - max)) + max + mt`
  - Writes Pibar to global tensor (needed by future waves' DTS cross)
  - Computes 6 DTS_L terms + logsumexp → Pi_new
  - Computes per-row convergence diff (max |Pi_new - Pi_old|)

**Grid:** 1D, one program per clade row (W blocks). Each block processes the
full S-element row in tiles of BLOCK_S=256. This gives good data reuse for
row_max/row_sum and avoids the need for inter-block communication.

**Memory savings:** Eliminates the [W,S] Pibar intermediate. Reads Pi only 2x
(down from 4-7x in the unfused path). The convergence check produces a [W]
max_diff vector instead of allocating [W,S] temporaries.

**Result:** Fused kernel achieves ~81% of peak memory bandwidth (625 GB/s on
a 768 GB/s device). At S=20K: 0.45ms/call average, 105ms for 232 calls.

## 2. Vectorized Species Child Setup

**File:** `src/core/likelihood.py` — `Pi_wave_forward` init

**Problem:** Building `sp_child1[S]` and `sp_child2[S]` from `s_P_indexes` and
`s_C12_indexes` used a Python loop with `.item()` calls:
```python
for i in range(len(sp_P_idx)):      # O(S) iterations
    p = int(sp_P_idx[i].item())     # GPU sync!
    c = int(sp_c12_idx[i].item())   # GPU sync!
    ...
```
At S=20K: 40K `.item()` calls × ~10μs each = **0.4s** (was 30% of total time).

**Solution:** Vectorized with a single `.cpu()` call + CPU tensor indexing:
```python
p_cpu = sp_P_idx.cpu().long()    # one GPU→CPU transfer
c_cpu = sp_c12_idx.cpu().long()
mask_c1 = p_cpu < S
sp_child1_cpu[p_cpu[mask_c1]] = c_cpu[mask_c1]
sp_child2_cpu[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1]
sp_child1 = sp_child1_cpu.to(device)
sp_child2 = sp_child2_cpu.to(device)
```

**Result:** 0.4s → <10ms (32x speedup for this step).

## 3. Skip Unused Inclusion Tensors

**File:** `src/core/model.py` — `compute_likelihood_batch`

**Problem:** Per-family `ccp_helpers` contains `inclusion_children` and
`inclusion_parents` tensors encoding the clade inclusion DAG. At S=20K with
C=40K clades, these are 417M entries × 8 bytes = **3.3GB each (6.6GB total)**.
The wave scheduling code moved ALL ccp_helpers tensors to GPU:
```python
ch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
          for k, v in fam['ccp_helpers'].items()}
```
Transferring 6.6GB over PCIe: **0.44s**.

But `compute_clade_waves` only reads `phased_waves` and `phased_phases`
(Python lists, not tensors). The inclusion tensors are computed by the C++
preprocessor but **never used** in scheduling, batching, or likelihood.

**Solution:** Pass `fam['ccp_helpers']` directly without GPU transfer:
```python
waves_i, phases_i = compute_clade_waves(fam['ccp_helpers'])
```

**Result:** 0.44s → 0ms.

## Profiling Results

### S=20K, 1 family (C=40,859)

```
                                     Before    After
End-to-end:                           2.5s      0.56s  (4.5x)
  extract_params (PCIe + compute):    0.29s     0.28s
  wave scheduling:                    0.42s     0.001s  ← fix #3
  Pi_wave_forward:                 0.63s     0.25s
    species child setup:              0.40s     0.01s   ← fix #2
    fused kernel (232 calls):          -        0.10s   ← fix #1
    DTS cross (57 calls):             0.10s     0.10s
    other (copies, leaf_wt, etc):     0.13s     0.04s
```

The fused kernel at S=20K: 0.45ms/call average, ~625 GB/s effective bandwidth
(81% of 768 GB/s peak).

### S=2K, 10 families (C=66,530)

```
                                     Before    After
End-to-end:                           0.75s     0.22s  (3.4x)
  Pi_wave_forward:                 0.17s     0.07s
    fused kernel (1724 calls):         -        0.06s
    DTS cross (421 calls):            0.05s     0.05s
```

### Remaining Bottleneck at S=20K

`extract_parameters` takes 0.28s (50% of total), dominated by:
- PCIe transfer of `transfer_mat_unnormalized` [S,S]: ~100ms
- Computation (log_softmax + exp2 over [S,S]): ~50ms
- `collate_gene_families`: ~50ms

For `pibar_mode='uniform'`, this [S,S] matrix is only needed by E-step.
A potential future optimization: apply the uniform approximation to E-step
too (`Ebar ≈ log2(sum(exp2(E)) - exp2(E_s)) + ...`), eliminating the need
for the [S,S] transfer matrix entirely.

## How to Run the Profiler

```bash
# S=20K, 1 family
python -m tests.profiling.profile_fused_kernel --dataset test_trees_10000 --n_families 1

# S=2K, 10 families
python -m tests.profiling.profile_fused_kernel --dataset test_trees_1000 --n_families 10

# Dense Pibar (for comparison)
python -m tests.profiling.profile_fused_kernel --dataset test_trees_1000 --pibar_mode dense
```

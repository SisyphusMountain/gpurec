# Segmented variant of your hybrid kernel.
# Differences vs. your version:
#  - V_scaled is now 3D: [NS, K, N]  (per-segment pre-scaled: exp(log_V_s - c_max_s[None,:]))
#  - c_max is now 2D:    [NS, N]
#  - ptr: int32 [NS+1] with 0 = ptr[0] < ... < ptr[NS] = G  (row boundaries)
#  - Each row g uses segment s determined by ptr, i.e., ptr[s] <= g < ptr[s+1].
#
# We keep the single-pass "online" row-max update and only accumulate
# contributions for rows that belong to each segment present in the tile.

import triton
import triton.language as tl
import torch
from statistics import median
import os

# Ensure Triton prints best config after autotune (once per key)
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")

# Track which autotune selections we've printed to avoid spam
_autotune_printed_tiled: set[tuple[int, int]] = set()  # (N, K)
_autotune_printed_h2: set[tuple[int, int, int]] = set()  # (G, N, K)


def _print_autotune_choice_tiled(N: int, K: int):
    key_short = (int(N), int(K))
    if key_short in _autotune_printed_tiled:
        return
    try:
        chosen = None
        for k, cfg in getattr(_seg_log_matmul_rowcolmax_hybrid_kernel_tiled, 'cache', {}).items():
            try:
                if isinstance(k, tuple) and len(k) >= 2 and int(k[0]) == int(N) and int(k[1]) == int(K):
                    chosen = cfg
                    break
            except Exception:
                continue
        if chosen is not None:
            print(f"[autotune] tiled  N={N} K={K} -> {chosen.kwargs}, num_warps={chosen.num_warps}, num_stages={chosen.num_stages}")
            _autotune_printed_tiled.add(key_short)
    except Exception:
        pass


def _print_autotune_choice_h2(G: int, N: int, K: int):
    key_short = (int(G), int(N), int(K))
    if key_short in _autotune_printed_h2:
        return
    try:
        chosen = None
        for k, cfg in getattr(_seg_log_matmul_rowcolmax_hybrid_kernel_2, 'cache', {}).items():
            try:
                if isinstance(k, tuple) and len(k) >= 3 and int(k[0]) == int(G) and int(k[1]) == int(N) and int(k[2]) == int(K):
                    chosen = cfg
                    break
            except Exception:
                continue
        if chosen is not None:
            print(f"[autotune] hybrid2 G={G} N={N} K={K} -> {chosen.kwargs}, num_warps={chosen.num_warps}, num_stages={chosen.num_stages}")
            _autotune_printed_h2.add(key_short)
    except Exception:
        pass


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    ],
    key=["G", "N", "K"],  # shapes
)
@triton.jit
def _seg_log_matmul_rowcolmax_hybrid_kernel_2(
    U_ptr, V_scaled_ptr, W_ptr, c_max_ptr, PTR_ptr,   # PTR_ptr: int32 [NS+1]
    G, N, K, NS,                                       # sizes: U=[G,K], V=[NS,K,N], W=[G,N]
    # U strides [G,K]
    stride_um, stride_uk,
    # V_scaled strides [NS,K,N]
    stride_vs, stride_vk, stride_vn,
    # W strides [G,N]
    stride_wm, stride_wn,
    # c_max strides [NS,N]
    stride_cs, stride_cn,
    # meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    NEG_INF = -float("inf")

    # ---- grouped PID mapping (same pattern as tutorial) ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(G, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BN]
    mask_m = offs_m < G
    mask_n = offs_n < N

    # ---- compute per-row segment id via ptr boundaries ----
    # seg_id[i] = count{s in 1..NS | offs_m[i] >= ptr[s]}
    offs_m_clamped = tl.minimum(offs_m, G - 1)
    seg_id = tl.zeros((BLOCK_M,), tl.int32)
    # linear scan over NS boundaries (works well for modest NS)
    for s in range(1, NS + 1):  # direct range instead of using tl.any
        end_s = tl.load(PTR_ptr + s, mask=s < NS + 1, other=0)   # ptr[s]
        seg_id += tl.where(offs_m_clamped >= end_s, 1, 0)

    # ---- initialize K traversals ----
    offs_k = tl.arange(0, BLOCK_K)

    # running row max r (over U) and accumulator S in probability space
    r = tl.full((BLOCK_M,), NEG_INF, tl.float32)        # [BM]
    S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)        # [BM,BN]

    # ---- single pass over K tiles ----
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k0 = kb * BLOCK_K
        k_offs = k0 + offs_k
        k_mask = k_offs < K

        # U chunk (streaming)
        u_ptrs = U_ptr + (offs_m[:, None] * stride_um + k_offs[None, :] * stride_uk)  # [BM,BK]
        Ublk = tl.load(u_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF)

        # online row-max update and rescale of S
        r_blk = tl.max(Ublk, axis=1)                 # [BM]
        new_r = tl.maximum(r, r_blk)
        S *= tl.exp(r - new_r)[:, None]
        r = new_r

        # exp(U - r) reused across segments
        Uexp = tl.exp(Ublk - r[:, None])             # [BM,BK]

        # for each segment present in this tile, accumulate only its rows
        # (rows in different segments see different V and c_max)
        for s in range(0, NS):
            rowmask = mask_m & (seg_id == s)
            # Check if any rows in this segment (simple approach without tl.any)
            rowmask_sum = tl.sum(tl.where(rowmask, 1, 0))
            if rowmask_sum > 0:
                # V_scaled chunk for segment s: shape [BK, BN] in *linear* space
                v_ptrs = (V_scaled_ptr
                          + s * stride_vs
                          + k_offs[:, None] * stride_vk
                          + offs_n[None, :] * stride_vn)
                V_scaled_blk = tl.load(
                    v_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0,
                )

                # mask out rows not in segment s
                Uexp_masked = Uexp * rowmask[:, None]    # [BM,BK]

                # accumulate contribution
                # Error is too large with default TF32 matmul, use IEEE. Exponentiation is the bottleneck.
                S += tl.dot(Uexp_masked, V_scaled_blk, input_precision="ieee")   # [BM,BN]

    # ---- finalize: W = log(S) + r + c_max[seg(row), :] ----
    Wtile = tl.where(S > 0, tl.log(S), tl.full((BLOCK_M, BLOCK_N), NEG_INF, dtype=tl.float32))
    Wtile = Wtile + r[:, None]

    # add the per-segment column offsets c_max
    for s in range(0, NS):
        rowmask = mask_m & (seg_id == s)
        rowmask_sum = tl.sum(tl.where(rowmask, 1, 0))
        if rowmask_sum > 0:
            c_ptrs = c_max_ptr + s * stride_cs + offs_n * stride_cn   # [BN]
            c_seg = tl.load(c_ptrs, mask=mask_n, other=NEG_INF)       # [BN]
            # add c_seg only to rows in this segment
            add_mask = tl.where(rowmask, 1.0, 0.0)[:, None]            # [BM,1] in {0,1}
            Wtile += add_mask * c_seg[None, :]

    # store
    w_ptrs = W_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
    tl.store(w_ptrs, Wtile, mask=mask_m[:, None] & mask_n[None, :])


# ----------------- Python launcher -----------------

def _launch_seg_log_matmul_rowcolmax_hybrid_2(U, V_scaled, W, c_max, ptr, G, N, K, NS):
    """Launch segmented hybrid v2 kernel"""
    grid = lambda meta: (triton.cdiv(G, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _seg_log_matmul_rowcolmax_hybrid_kernel_2[grid](
        U, V_scaled, W, c_max, ptr,
        G, N, K, NS,
        U.stride(0), U.stride(1),
        V_scaled.stride(0), V_scaled.stride(1), V_scaled.stride(2),
        W.stride(0), W.stride(1),
        c_max.stride(0), c_max.stride(1),
    )


def seg_log_matmul_rowcolmax_hybrid_2(U_log, V_log_list, ptr):
    """
    Segmented log-space matrix multiplication using hybrid v2 approach.
    
    Args:
        U_log: torch.Tensor [G, K] - input log matrix
        V_log_list: List[torch.Tensor] - list of [K, N] log matrices, one per segment
        ptr: torch.Tensor [NS+1] - segment boundaries, ptr[s] <= row < ptr[s+1] uses V_log_list[s]
        
    Returns:
        torch.Tensor [G, N] - log(segmented_matmul(exp(U_log), exp(V_log_list)))
    """
    import torch
    
    G, K = U_log.shape
    NS = len(V_log_list)

    
    # All V matrices should have same [K, N] shape
    K2, N = V_log_list[0].shape

    
    # Stack V matrices and precompute column maxima and scaling
    V_log_stacked = torch.stack(V_log_list, dim=0)  # [NS, K, N]
    c_max = torch.max(V_log_stacked, dim=1)[0]      # [NS, N] - column max per segment
    V_scaled = torch.exp(V_log_stacked - c_max[:, None, :])  # [NS, K, N] - pre-scaled
    
    # Prepare output and launch kernel
    W_log = torch.empty((G, N), device=U_log.device, dtype=torch.float32)
    _launch_seg_log_matmul_rowcolmax_hybrid_2(
        U_log, V_scaled, W_log, c_max, ptr.int(), G, N, K, NS
    )
    
    return W_log


def seg_log_matmul_triton_kernel_only(U_log, V_scaled, c_max, ptr):
    """
    Triton kernel only version for benchmarking - assumes preprocessing is already done.
    
    Parameters:
    - U_log: [G, K] log matrix  
    - V_scaled: [NS, K, N] pre-scaled V matrices: exp(V_log - c_max[:, None, :])
    - c_max: [NS, N] column maxima per segment
    - ptr: [NS+1] int32 tensor with segment boundaries
    
    Returns:
    - W_log: [G, N] result matrix in log space
    """
    G, K = U_log.shape
    NS, K2, N = V_scaled.shape
    
    # Prepare output and launch kernel (just the kernel, no preprocessing)
    W_log = torch.empty((G, N), device=U_log.device, dtype=torch.float32)
    _launch_seg_log_matmul_rowcolmax_hybrid_2(
        U_log, V_scaled, W_log, c_max, ptr.int(), G, N, K, NS
    )
    
    return W_log


# ----------------- PyTorch Reference Implementation -----------------

import torch

def segmented_log_matmul_pytorch_batched(U_log, V_log_list, ptr):
    """
    PyTorch reference using batched approach with reshape to [B, max_len_seg, S].
    This is the efficient approach you requested.
    """
    G, K = U_log.shape
    NS = len(V_log_list)
    _, N = V_log_list[0].shape
    
    # Compute segment lengths
    segment_lengths = (ptr[1:] - ptr[:-1]).tolist()
    max_len_seg = max(segment_lengths)
    
    # Create batched tensors [NS, max_len_seg, K] and [NS, K, N]
    U_batched = torch.full((NS, max_len_seg, K), float('-inf'), 
                          device=U_log.device, dtype=U_log.dtype)
    V_batched = torch.stack(V_log_list, dim=0)  # [NS, K, N]
    
    # Fill batched U tensor with appropriate segments
    for s in range(NS):
        start_idx = ptr[s]
        end_idx = ptr[s + 1]
        seg_len = end_idx - start_idx
        if seg_len > 0:
            U_batched[s, :seg_len, :] = U_log[start_idx:end_idx, :]
    
    # Apply log-space stabilization per segment
    # Row-wise max for each segment
    row_max_batched = torch.max(U_batched, dim=2, keepdim=True)[0]  # [NS, max_len_seg, 1]
    row_max_batched = torch.where(U_batched[:, :, :1] > float('-inf'), 
                                 row_max_batched, float('-inf'))
    
    # Column-wise max for each V matrix
    col_max_batched = torch.max(V_batched, dim=1, keepdim=True)[0]  # [NS, 1, N]
    
    # Stabilized computation
    U_stable = torch.exp(U_batched - row_max_batched)  # [NS, max_len_seg, K]
    V_stable = torch.exp(V_batched - col_max_batched)  # [NS, K, N]
    
    # Batched matrix multiplication
    result_linear_batched = torch.bmm(U_stable, V_stable)  # [NS, max_len_seg, N]
    
    # Back to log space
    result_log_batched = torch.log(torch.clamp(result_linear_batched, min=1e-45))
    result_log_batched = result_log_batched + row_max_batched + col_max_batched
    
    # Reassemble into original [G, N] shape
    W_log = torch.empty((G, N), device=U_log.device, dtype=U_log.dtype)
    for s in range(NS):
        start_idx = ptr[s]
        end_idx = ptr[s + 1]
        seg_len = end_idx - start_idx
        if seg_len > 0:
            W_log[start_idx:end_idx, :] = result_log_batched[s, :seg_len, :]
    
    return W_log


def segmented_log_matmul_pytorch_loop(U_log, V_log_list, ptr):
    """
    PyTorch reference using simple for loop approach.
    This is the naive approach for comparison.
    """
    G, K = U_log.shape
    NS = len(V_log_list)
    _, N = V_log_list[0].shape
    
    W_log = torch.empty((G, N), device=U_log.device, dtype=U_log.dtype)
    
    for s in range(NS):
        start_idx = ptr[s]
        end_idx = ptr[s + 1]
        if start_idx >= end_idx:
            continue
            
        # Extract segment
        U_seg = U_log[start_idx:end_idx, :]  # [seg_len, K]
        V_seg = V_log_list[s]                # [K, N]
        
        # Standard log-space matrix multiplication with stabilization
        # Row-wise max stabilization
        row_max = torch.max(U_seg, dim=1, keepdim=True)[0]  # [seg_len, 1]
        U_stable = torch.exp(U_seg - row_max)               # [seg_len, K]
        
        # Convert V to linear space
        V_linear = torch.exp(V_seg)                         # [K, N]
        
        # Matrix multiplication
        result_linear = torch.matmul(U_stable, V_linear)    # [seg_len, N]
        
        # Back to log space
        result_log = torch.log(torch.clamp(result_linear, min=1e-45)) + row_max
        
        # Store in output
        W_log[start_idx:end_idx, :] = result_log
    
    return W_log


def segmented_log_matmul_pytorch_loop_kernel_only(U_log, V_linear_list, ptr):
    """
    PyTorch loop implementation with preprocessing factored out.
    Assumes V matrices are already in linear space.
    """
    G, K = U_log.shape
    NS = len(V_linear_list)
    _, N = V_linear_list[0].shape
    
    W_log = torch.empty((G, N), device=U_log.device, dtype=U_log.dtype)
    
    for s in range(NS):
        start_idx = ptr[s]
        end_idx = ptr[s + 1]
        if start_idx >= end_idx:
            continue
            
        # Extract segment
        U_seg = U_log[start_idx:end_idx, :]  # [seg_len, K]
        V_linear = V_linear_list[s]          # [K, N] - already in linear space
        
        # Row-wise max stabilization
        row_max = torch.max(U_seg, dim=1, keepdim=True)[0]  # [seg_len, 1]
        U_stable = torch.exp(U_seg - row_max)               # [seg_len, K]
        
        # Matrix multiplication (no V conversion needed)
        result_linear = torch.matmul(U_stable, V_linear)    # [seg_len, N]
        
        # Back to log space
        result_log = torch.log(torch.clamp(result_linear, min=1e-45)) + row_max
        
        # Store in output
        W_log[start_idx:end_idx, :] = result_log
    
    return W_log


def segmented_log_matmul_pytorch_batched_kernel_only(U_batched, V_linear_batched, row_max_batched, ptr):
    # Removed in unified benchmark; keep interface if ever needed
    raise NotImplementedError("Batched kernel-only path is no longer benchmarked here.")



def _build_tile_lut(ptr: torch.Tensor, block_m: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build LUTs: tile2seg, tile2row0, tile2row1 given segment boundaries `ptr` and tile height `block_m`.
    Returns int32 tensors on `device`.
    """
    assert ptr.ndim == 1 and ptr.dtype in (torch.int32, torch.int64)
    ptr = ptr.to(torch.int64)
    ns = ptr.numel() - 1
    rows = []
    segs = []
    rows_end = []
    for s in range(ns):
        start = int(ptr[s].item())
        end   = int(ptr[s+1].item())
        if end <= start:
            continue
        seg_len = end - start
        tiles = (seg_len + block_m - 1) // block_m
        for t in range(tiles):
            r0 = start + t * block_m
            r1 = min(end, r0 + block_m)
            segs.append(s)
            rows.append(r0)
            rows_end.append(r1)
    tile2seg  = torch.tensor(segs, dtype=torch.int32, device=device)
    tile2row0 = torch.tensor(rows, dtype=torch.int32, device=device)
    tile2row1 = torch.tensor(rows_end, dtype=torch.int32, device=device)
    return tile2seg, tile2row0, tile2row1


@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 64,  "TILE_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"TILE_N": 128, "TILE_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"TILE_N": 128, "TILE_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"TILE_N": 256, "TILE_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"TILE_N": 256, "TILE_K": 64}, num_warps=16, num_stages=2),
        triton.Config({"TILE_N": 128, "TILE_K": 128}, num_warps=8, num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _seg_log_matmul_rowcolmax_hybrid_kernel_tiled(
    U_ptr,                 # float32*  [G, K]       (log-space)
    V_scaled_ptr,          # float32*  [NS, K, N]   (linear space)
    c_max_ptr,             # float32*  [NS, N]
    tile2seg_ptr,          # int32*    [T]
    tile2row0_ptr,         # int32*    [T]
    tile2row1_ptr,         # int32*    [T]
    W_ptr,                 # float32*  [G, N]
    G, N, K, NS,           # sizes
    # strides
    stride_um, stride_uk,                      # U [G,K]
    stride_vs, stride_vk, stride_vn,           # V_scaled [NS,K,N]
    stride_cm_s, stride_cm_n,                  # c_max [NS,N]
    stride_wm, stride_wn,                      # W [G,N]
    # meta
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    tl.assume(stride_um > 0)
    tl.assume(stride_uk > 0)
    tl.assume(stride_vs > 0)
    tl.assume(stride_vk > 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_wm > 0)
    tl.assume(stride_wn > 0)

    NEG_INF = -float("inf")

    # tile ids
    pid_m = tl.program_id(axis=0)  # which row tile (across all segments)
    pid_n = tl.program_id(axis=1)  # which col tile

    # look up segment and row range for this tile
    s  = tl.load(tile2seg_ptr   + pid_m)          # scalar int32
    r0 = tl.load(tile2row0_ptr  + pid_m)          # scalar int32
    r1 = tl.load(tile2row1_ptr  + pid_m)          # scalar int32

    # row/col indices
    offs_m = r0 + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask_m = offs_m < r1
    mask_n = offs_n < N

    # per-segment column offsets for these columns
    c_ptrs = c_max_ptr + s * stride_cm_s + offs_n * stride_cm_n
    c_seg  = tl.load(c_ptrs, mask=mask_n, other=NEG_INF)  # [BN]

    # single-pass over K with online row max
    offs_k = tl.arange(0, TILE_K)
    r = tl.full((TILE_M,), NEG_INF, tl.float32)          # row baselines
    S = tl.zeros((TILE_M, TILE_N), tl.float32)          # prob-space accumulator

    for kb in range(0, tl.cdiv(K, TILE_K)):
        k0 = kb * TILE_K
        k_offs = k0 + offs_k
        k_mask = k_offs < K

        # U chunk (streaming, log-space)
        u_ptrs = U_ptr + (offs_m[:, None] * stride_um + k_offs[None, :] * stride_uk)
        Ublk = tl.load(u_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF)

        # online row max + rescale
        r_blk = tl.max(Ublk, axis=1)
        new_r = tl.maximum(r, r_blk)
        S *= tl.exp(r - new_r)[:, None]
        r = new_r

        Uexp = tl.exp(Ublk - r[:, None])  # [BM, BK], safe

        # V_scaled chunk for this segment s and our columns (already linear space)
        v_ptrs = (V_scaled_ptr
                  + s * stride_vs
                  + k_offs[:, None] * stride_vk
                  + offs_n[None, :] * stride_vn)
        V_scaled_blk = tl.load(
            v_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0,
        )

        # accumulate
        S += tl.dot(Uexp, V_scaled_blk, input_precision="ieee")  # [BM,BN]

    # finalize: W = log(S) + r + c_seg
    Wtile = tl.where(S > 0, tl.log(S), tl.full((TILE_M, TILE_N), NEG_INF, dtype=tl.float32))
    Wtile = Wtile + r[:, None] + c_seg[None, :]

    # store
    w_ptrs = W_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
    tl.store(w_ptrs, Wtile, mask=mask_m[:, None] & mask_n[None, :])


def seg_log_matmul_rowcolmax_hybrid_tiled(
    U_log: torch.Tensor,        # [G, K] float32 (CUDA)
    V_scaled: torch.Tensor,     # [NS, K, N] float32 (CUDA), pre = exp(logV - c_max[None,:])
    c_max: torch.Tensor,        # [NS, N] float32 (CUDA)
    ptr: torch.Tensor,          # [NS+1] int32/int64 (CPU or CUDA)
    *,
    TILE_M=64, TILE_N=128, TILE_K=32,
    num_warps=8, num_stages=2,
):
    """
    Segmented log-matmul with one row-tile per segment tile (no inner loop over segments).
    Precompute LUTs so each pid_m knows (segment s, row0, row1).
    """

    G, K1 = U_log.shape
    NS, K2, N = V_scaled.shape

    device = U_log.device
    
    # build LUTs
    tile2seg, tile2row0, tile2row1 = _build_tile_lut(ptr, TILE_M, device)
    T = tile2seg.numel()
    if T == 0:
        return torch.empty((G, N), device=device, dtype=torch.float32)

    # output
    W_log = torch.empty((G, N), device=device, dtype=torch.float32)

    # 2D grid: (#row-tiles across all segments, #col-tiles using meta TILE_N)
    grid = lambda meta: (T, triton.cdiv(N, meta['TILE_N']))

    _seg_log_matmul_rowcolmax_hybrid_kernel_tiled[grid](
        U_log, V_scaled, c_max,
        tile2seg, tile2row0, tile2row1,
        W_log,
        G, N, K1, NS,
        U_log.stride(0), U_log.stride(1),
        V_scaled.stride(0), V_scaled.stride(1), V_scaled.stride(2),
        c_max.stride(0), c_max.stride(1),
        W_log.stride(0), W_log.stride(1),
        # Fix TILE_M to match LUT; let autotune pick TILE_N/K, warps, stages
        TILE_M=TILE_M,
    )
    return W_log


def seg_log_matmul_rowcolmax_hybrid_tiled_kernel_only(
    U_log: torch.Tensor,
    V_scaled: torch.Tensor,
    c_max: torch.Tensor,
    tile2seg: torch.Tensor,
    tile2row0: torch.Tensor,
    tile2row1: torch.Tensor,
    *,
    TILE_M=64, TILE_N=128, TILE_K=32,
    num_warps=8, num_stages=2,
):
    """
    Launch the tiled kernel without building LUTs (kernel-only timing helper).
    Expects precomputed tile2seg/tile2row* tensors on the same device.
    """
    G, K1 = U_log.shape
    NS, K2, N = V_scaled.shape

    device = U_log.device
    assert tile2seg.device == device and tile2row0.device == device and tile2row1.device == device

    T = tile2seg.numel()
    if T == 0:
        return torch.empty((G, N), device=device, dtype=torch.float32)

    W_log = torch.empty((G, N), device=device, dtype=torch.float32)
    grid = lambda meta: (T, triton.cdiv(N, meta['TILE_N']))

    _seg_log_matmul_rowcolmax_hybrid_kernel_tiled[grid](
        U_log, V_scaled, c_max,
        tile2seg, tile2row0, tile2row1,
        W_log,
        G, N, K1, NS,
        U_log.stride(0), U_log.stride(1),
        V_scaled.stride(0), V_scaled.stride(1), V_scaled.stride(2),
        c_max.stride(0), c_max.stride(1),
        W_log.stride(0), W_log.stride(1),
        TILE_M=TILE_M,
    )
    return W_log

# ----------------- Benchmarking and Testing Framework -----------------

def create_test_data(G, K, N, NS, seed=42):
    """Create test data for segmented matrix multiplication"""
    torch.manual_seed(seed)
    
    # Create input log matrix
    U_log = torch.randn(G, K, device='cuda', dtype=torch.float32) - 3.0
    
    # Create list of V log matrices
    V_log_list = []
    for s in range(NS):
        V_log = torch.randn(K, N, device='cuda', dtype=torch.float32) - 3.0
        V_log_list.append(V_log)
    
    # Create segment boundaries - distribute rows roughly equally
    segment_sizes = [G // NS] * NS
    # Distribute remainder
    for i in range(G % NS):
        segment_sizes[i] += 1
    
    ptr = torch.zeros(NS + 1, device='cuda', dtype=torch.int32)
    ptr[0] = 0
    for i in range(NS):
        ptr[i + 1] = ptr[i] + segment_sizes[i]
    
    return U_log, V_log_list, ptr


def _random_segments_ptr(G: int, NS: int, device) -> torch.Tensor:
    """Create random segment boundaries (ptr) with NS segments summing to G.
    Ensures each segment has at least 1 row. Average length ~ G/NS.
    """
    NS = int(min(NS, G))
    if NS <= 0:
        return torch.tensor([0, G], device=device, dtype=torch.int32)

    w = torch.rand(NS, device=device)
    w = w / w.sum()
    lengths = (w * G).round().to(torch.int64)
    lengths = torch.clamp(lengths, min=1)
    diff = int(lengths.sum().item() - G)
    if diff > 0:
        # subtract 1 from segments with length > 1
        idx = (lengths > 1).nonzero(as_tuple=False).flatten()
        if idx.numel() > 0:
            idx = idx[:diff]
            lengths[idx] -= 1
    elif diff < 0:
        # add 1 to some segments
        add = -diff
        idx = torch.arange(min(add, NS), device=device)
        lengths[idx] += 1
    # build ptr
    ptr = torch.zeros(NS + 1, device=device, dtype=torch.int32)
    ptr[1:] = lengths.to(torch.int32).cumsum(0)
    return ptr


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['G'],
        x_vals=[50_000, 100_000, 200_000, 300_000, 400_000],
        x_log=False,
        line_arg='provider',
        line_vals=['torch_loop', 'triton', 'triton_tiled'],
        line_names=['Torch (loop)', 'Triton', 'Triton Tiled'],
        styles=[('green', '-'), ('blue', '-'), ('purple', '-')],
        ylabel='GF/s',
        plot_name='segmented-log-matmul-large-G',
        args={'K': 500, 'N': 500, 'NS': 1000},
    )
)
def benchmark_large_G(G: int, K: int, N: int, NS: int, provider: str):
    device = 'cuda'
    torch.manual_seed(123)

    # Allocate inputs
    U_log = (torch.randn(G, K, device=device, dtype=torch.float32) - 3.0).contiguous()
    ptr = _random_segments_ptr(G, NS, device)

    # V per-segment in log-space
    V_log_stacked = (torch.randn(NS, K, N, device=device, dtype=torch.float32) - 3.0).contiguous()

    # Precompute for each provider
    if provider == 'torch_loop':
        V_linear_stacked = torch.exp(V_log_stacked)  # [NS, K, N]
        V_linear_list = [V_linear_stacked[s] for s in range(NS)]
        fn = lambda: segmented_log_matmul_pytorch_loop_kernel_only(U_log, V_linear_list, ptr)
    elif provider == 'triton':
        c_max = torch.max(V_log_stacked, dim=1)[0]  # [NS, N]
        V_scaled = torch.exp(V_log_stacked - c_max[:, None, :])  # [NS, K, N]
        fn = lambda: seg_log_matmul_triton_kernel_only(U_log, V_scaled, c_max, ptr)
    elif provider == 'triton_tiled':
        c_max = torch.max(V_log_stacked, dim=1)[0]
        V_scaled = torch.exp(V_log_stacked - c_max[:, None, :])
        TILE_M, TILE_N, TILE_K = 64, 128, 32
        tile2seg, tile2row0, tile2row1 = _build_tile_lut(ptr, TILE_M, U_log.device)
        fn = lambda: seg_log_matmul_rowcolmax_hybrid_tiled_kernel_only(
            U_log, V_scaled, c_max, tile2seg, tile2row0, tile2row1,
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
        )
    else:
        raise ValueError(provider)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # Print chosen configs once per (N,K) or (G,N,K)
    if provider == 'triton':
        _print_autotune_choice_h2(G, N, K)
    elif provider == 'triton_tiled':
        _print_autotune_choice_tiled(N, K)
    gfs = lambda t_ms: (G * K * N) / (t_ms * 1e6)
    return gfs(ms), gfs(max_ms), gfs(min_ms)


if __name__ == "__main__":
    # Unified benchmark: large-G, ~1000 segments; K=N=500
    benchmark_large_G.run(print_data=True, show_plots=True)
    
    # Also plot raw execution time (ms) for the same sweep
    try:
        import triton
    except Exception:
        pass

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['G'],
            x_vals=[50_000, 100_000, 200_000, 300_000, 400_000],
            x_log=False,
            line_arg='provider',
            line_vals=['torch_loop', 'triton', 'triton_tiled'],
            line_names=['Torch (loop)', 'Triton', 'Triton Tiled'],
            styles=[('green', '-'), ('blue', '-'), ('purple', '-')],
            ylabel='Time (ms)',
            plot_name='segmented-log-matmul-large-G-time',
            args={'K': 500, 'N': 500, 'NS': 1000},
        )
    )
    def benchmark_large_G_time(G: int, K: int, N: int, NS: int, provider: str):
        device = 'cuda'
        torch.manual_seed(123)

        U_log = (torch.randn(G, K, device=device, dtype=torch.float32) - 3.0).contiguous()
        ptr = _random_segments_ptr(G, NS, device)
        V_log_stacked = (torch.randn(NS, K, N, device=device, dtype=torch.float32) - 3.0).contiguous()

        if provider == 'torch_loop':
            V_linear_stacked = torch.exp(V_log_stacked)
            V_linear_list = [V_linear_stacked[s] for s in range(NS)]
            fn = lambda: segmented_log_matmul_pytorch_loop_kernel_only(U_log, V_linear_list, ptr)
        elif provider == 'triton':
            c_max = torch.max(V_log_stacked, dim=1)[0]
            V_scaled = torch.exp(V_log_stacked - c_max[:, None, :])
            fn = lambda: seg_log_matmul_triton_kernel_only(U_log, V_scaled, c_max, ptr)
        elif provider == 'triton_tiled':
            c_max = torch.max(V_log_stacked, dim=1)[0]
            V_scaled = torch.exp(V_log_stacked - c_max[:, None, :])
            TILE_M, TILE_N, TILE_K = 64, 128, 32
            tile2seg, tile2row0, tile2row1 = _build_tile_lut(ptr, TILE_M, U_log.device)
            fn = lambda: seg_log_matmul_rowcolmax_hybrid_tiled_kernel_only(
                U_log, V_scaled, c_max, tile2seg, tile2row0, tile2row1,
                TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            )
        else:
            raise ValueError(provider)

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        # Print chosen configs once per (N,K) or (G,N,K)
        if provider == 'triton':
            _print_autotune_choice_h2(G, N, K)
        elif provider == 'triton_tiled':
            _print_autotune_choice_tiled(N, K)
        # Return ms metrics directly to plot time
        return ms, max_ms, min_ms

    benchmark_large_G_time.run(print_data=True, show_plots=True)

    # Additional benchmark around A=[~1000,200], B=[200,200]
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['G'],
            x_vals=[512, 800, 1000, 1280, 1600, 2000],
            x_log=False,
            line_arg='provider',
            line_vals=['torch_loop', 'triton', 'triton_tiled'],
            line_names=['Torch (loop)', 'Triton', 'Triton Tiled'],
            styles=[('green', '-'), ('blue', '-'), ('purple', '-')],
            ylabel='GF/s',
            plot_name='segmented-log-matmul-small-200',
            args={'K': 200, 'N': 200, 'NS': 32},
        )
    )
    def benchmark_small_200(G: int, K: int, N: int, NS: int, provider: str):
        device = 'cuda'
        torch.manual_seed(321)

        U_log = (torch.randn(G, K, device=device, dtype=torch.float32) - 3.0).contiguous()
        ptr = _random_segments_ptr(G, NS, device)
        V_log_stacked = (torch.randn(NS, K, N, device=device, dtype=torch.float32) - 3.0).contiguous()

        if provider == 'torch_loop':
            V_linear_stacked = torch.exp(V_log_stacked)
            V_linear_list = [V_linear_stacked[s] for s in range(NS)]
            fn = lambda: segmented_log_matmul_pytorch_loop_kernel_only(U_log, V_linear_list, ptr)
        elif provider == 'triton':
            c_max = torch.max(V_log_stacked, dim=1)[0]
            V_scaled = torch.exp(V_log_stacked - c_max[:, None, :])
            fn = lambda: seg_log_matmul_triton_kernel_only(U_log, V_scaled, c_max, ptr)
        elif provider == 'triton_tiled':
            c_max = torch.max(V_log_stacked, dim=1)[0]
            V_scaled = torch.exp(V_log_stacked - c_max[:, None, :])
            TILE_M, TILE_N, TILE_K = 64, 128, 32
            tile2seg, tile2row0, tile2row1 = _build_tile_lut(ptr, TILE_M, U_log.device)
            fn = lambda: seg_log_matmul_rowcolmax_hybrid_tiled_kernel_only(
                U_log, V_scaled, c_max, tile2seg, tile2row0, tile2row1,
                TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            )
        else:
            raise ValueError(provider)

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        if provider == 'triton':
            _print_autotune_choice_h2(G, N, K)
        elif provider == 'triton_tiled':
            _print_autotune_choice_tiled(N, K)
        gfs = lambda t_ms: (G * K * N) / (t_ms * 1e6)
        return gfs(ms), gfs(max_ms), gfs(min_ms)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['G'],
            x_vals=[512, 800, 1000, 1280, 1600, 2000],
            x_log=False,
            line_arg='provider',
            line_vals=['torch_loop', 'triton', 'triton_tiled'],
            line_names=['Torch (loop)', 'Triton', 'Triton Tiled'],
            styles=[('green', '-'), ('blue', '-'), ('purple', '-')],
            ylabel='Time (ms)',
            plot_name='segmented-log-matmul-small-200-time',
            args={'K': 200, 'N': 200, 'NS': 32},
        )
    )
    def benchmark_small_200_time(G: int, K: int, N: int, NS: int, provider: str):
        device = 'cuda'
        torch.manual_seed(321)

        U_log = (torch.randn(G, K, device=device, dtype=torch.float32) - 3.0).contiguous()
        ptr = _random_segments_ptr(G, NS, device)
        V_log_stacked = (torch.randn(NS, K, N, device=device, dtype=torch.float32) - 3.0).contiguous()

        if provider == 'torch_loop':
            V_linear_stacked = torch.exp(V_log_stacked)
            V_linear_list = [V_linear_stacked[s] for s in range(NS)]
            fn = lambda: segmented_log_matmul_pytorch_loop_kernel_only(U_log, V_linear_list, ptr)
        elif provider == 'triton':
            c_max = torch.max(V_log_stacked, dim=1)[0]
            V_scaled = torch.exp(V_log_stacked - c_max[:, None, :])
            fn = lambda: seg_log_matmul_triton_kernel_only(U_log, V_scaled, c_max, ptr)
        elif provider == 'triton_tiled':
            c_max = torch.max(V_log_stacked, dim=1)[0]
            V_scaled = torch.exp(V_log_stacked - c_max[:, None, :])
            TILE_M, TILE_N, TILE_K = 64, 128, 32
            tile2seg, tile2row0, tile2row1 = _build_tile_lut(ptr, TILE_M, U_log.device)
            fn = lambda: seg_log_matmul_rowcolmax_hybrid_tiled_kernel_only(
                U_log, V_scaled, c_max, tile2seg, tile2row0, tile2row1,
                TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            )
        else:
            raise ValueError(provider)

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        if provider == 'triton':
            _print_autotune_choice_h2(G, N, K)
        elif provider == 'triton_tiled':
            _print_autotune_choice_tiled(N, K)
        return ms, max_ms, min_ms

    benchmark_small_200.run(print_data=True, show_plots=True)
    benchmark_small_200_time.run(print_data=True, show_plots=True)

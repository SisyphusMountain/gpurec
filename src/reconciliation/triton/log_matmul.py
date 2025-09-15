
import torch
import triton
import triton.language as tl


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
def _seg_log_matmul(
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




def segmented_log_matmul(
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
    # TODO: cache these if ptr is constant across calls
    tile2seg, tile2row0, tile2row1 = _build_tile_lut(ptr, TILE_M, device)
    T = tile2seg.numel()
    if T == 0:
        return torch.empty((G, N), device=device, dtype=torch.float32)

    # output
    W_log = torch.empty((G, N), device=device, dtype=torch.float32)

    # 2D grid: (#row-tiles across all segments, #col-tiles using meta TILE_N)
    grid = lambda meta: (T, triton.cdiv(N, meta['TILE_N']))

    _seg_log_matmul[grid](
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


# ---------------- Backward (tiled) ----------------

@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 64,  "TILE_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"TILE_N": 128, "TILE_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"TILE_N": 128, "TILE_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"TILE_N": 256, "TILE_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"TILE_N": 256, "TILE_K": 64}, num_warps=16, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit
def _seg_log_matmul_rowcolmax_hybrid_kernel_tiled_bwd(
    U_ptr,                 # float32*  [G, K]       (log-space)
    V_scaled_ptr,          # float32*  [NS, K, N]   (linear space)
    c_max_ptr,             # float32*  [NS, N]
    tile2seg_ptr,          # int32*    [T]
    tile2row0_ptr,         # int32*    [T]
    tile2row1_ptr,         # int32*    [T]
    dW_ptr,                # float32*  [G, N] upstream grad
    dU_ptr,                # float32*  [G, K] output grad U
    dV_ptr,                # float32*  [NS, K, N] output grad V_scaled
    G, N, K, NS,           # sizes
    # strides
    stride_um, stride_uk,                      # U [G,K]
    stride_vs, stride_vk, stride_vn,           # V_scaled [NS,K,N]
    stride_cm_s, stride_cm_n,                  # c_max [NS,N]
    stride_dm, stride_dn,                      # dW [G,N]
    stride_dum, stride_duk,                    # dU [G,K]
    stride_dvs, stride_dvk, stride_dvn,        # dV [NS,K,N]
    # meta
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
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

    # per-segment column offsets for these columns (not needed numerically for grads, but keep parity)
    # c_ptrs = c_max_ptr + s * stride_cm_s + offs_n * stride_cm_n
    # c_seg  = tl.load(c_ptrs, mask=mask_n, other=NEG_INF)

    offs_k = tl.arange(0, TILE_K)

    # Pass 1: compute row baseline r and S (prob-space sum)
    r = tl.full((TILE_M,), NEG_INF, tl.float32)
    S = tl.zeros((TILE_M, TILE_N), tl.float32)

    for kb in range(0, tl.cdiv(K, TILE_K)):
        k0 = kb * TILE_K
        k_offs = k0 + offs_k
        k_mask = k_offs < K

        # U chunk
        u_ptrs = U_ptr + (offs_m[:, None] * stride_um + k_offs[None, :] * stride_uk)
        Ublk = tl.load(u_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF)

        # online row max + rescale
        r_blk = tl.max(Ublk, axis=1)
        new_r = tl.maximum(r, r_blk)
        S *= tl.exp(r - new_r)[:, None]
        r = new_r

        # Uexp for current slice
        Uexp = tl.exp(Ublk - r[:, None])

        # V slice for this segment and columns
        v_ptrs = (V_scaled_ptr + s * stride_vs + k_offs[:, None] * stride_vk + offs_n[None, :] * stride_vn)
        V_scaled_blk = tl.load(v_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

        S += tl.dot(Uexp, V_scaled_blk, input_precision="ieee")

    # Load upstream grad and form Ghat = dW / S
    dW_ptrs = dW_ptr + (offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn)
    dW = tl.load(dW_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    eps = 1e-20
    Ghat = dW / tl.maximum(S, eps)

    # Pass 2: accumulate dU and dV using Uexp with final r
    for kb in range(0, tl.cdiv(K, TILE_K)):
        k0 = kb * TILE_K
        k_offs = k0 + offs_k
        k_mask = k_offs < K

        # U and V blocks
        u_ptrs = U_ptr + (offs_m[:, None] * stride_um + k_offs[None, :] * stride_uk)
        Ublk = tl.load(u_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF)
        Uexp = tl.exp(Ublk - r[:, None])  # [BM, BK]

        v_ptrs = (V_scaled_ptr + s * stride_vs + k_offs[:, None] * stride_vk + offs_n[None, :] * stride_vn)
        V_scaled_blk = tl.load(v_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

        # dU contribution: Uexp * (Ghat @ V_scaled^T)
        # tmp = [BM, BK]
        tmp = tl.dot(Ghat, tl.trans(V_scaled_blk), input_precision="ieee")
        dU_blk = Uexp * tmp

        # dV contribution: (Uexp^T @ Ghat)  -> [BK, BN]
        dV_blk = tl.dot(tl.trans(Uexp), Ghat, input_precision="ieee")

        # Scatter-add to dU
        dU_ptrs = dU_ptr + (offs_m[:, None] * stride_dum + k_offs[None, :] * stride_duk)
        tl.atomic_add(dU_ptrs, dU_blk, mask=mask_m[:, None] & k_mask[None, :])

        # Scatter-add to dV (segment s)
        dV_ptrs = dV_ptr + (s * stride_dvs + k_offs[:, None] * stride_dvk + offs_n[None, :] * stride_dvn)
        tl.atomic_add(dV_ptrs, dV_blk, mask=k_mask[:, None] & mask_n[None, :])


def segmented_log_matmul_backward(
    U_log: torch.Tensor,        # [G, K]
    V_scaled: torch.Tensor,     # [NS, K, N]
    c_max: torch.Tensor,        # [NS, N]
    ptr: torch.Tensor,          # [NS+1]
    dW: torch.Tensor,           # [G, N]
    *,
    TILE_M=64, TILE_N=128, TILE_K=32,
):
    """Backward pass for segmented_log_matmul (tiled forward).

    Returns (dU, dV_scaled). Gradients for c_max are not propagated (treated as constant).
    """
    assert U_log.is_cuda and V_scaled.is_cuda and dW.is_cuda
    G, K1 = U_log.shape
    NS, K2, N = V_scaled.shape
    assert K1 == K2 and dW.shape == (G, N)

    device = U_log.device
    tile2seg, tile2row0, tile2row1 = _build_tile_lut(ptr, TILE_M, device)
    T = tile2seg.numel()
    if T == 0:
        return torch.zeros_like(U_log), torch.zeros_like(V_scaled)

    dU = torch.zeros_like(U_log)
    dV = torch.zeros_like(V_scaled)

    grid = lambda meta: (T, triton.cdiv(N, meta['TILE_N']))
    _seg_log_matmul_rowcolmax_hybrid_kernel_tiled_bwd[grid](
        U_log, V_scaled, c_max,
        tile2seg, tile2row0, tile2row1,
        dW, dU, dV,
        G, N, K1, NS,
        U_log.stride(0), U_log.stride(1),
        V_scaled.stride(0), V_scaled.stride(1), V_scaled.stride(2),
        c_max.stride(0), c_max.stride(1),
        dW.stride(0), dW.stride(1),
        dU.stride(0), dU.stride(1),
        dV.stride(0), dV.stride(1), dV.stride(2),
        TILE_M=TILE_M,
    )

    return dU, dV


class SegmentedLogMatmulTiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                U_log: torch.Tensor,
                V_scaled: torch.Tensor,
                c_max: torch.Tensor,
                ptr: torch.Tensor,
                TILE_M: int = 64,
                TILE_N: int = 128,
                TILE_K: int = 32):
        ctx.TILE_M = int(TILE_M)
        ctx.TILE_N = int(TILE_N)
        ctx.TILE_K = int(TILE_K)
        ctx.save_for_backward(U_log, V_scaled, c_max, ptr)
        with torch.cuda.device_of(U_log):
            out = segmented_log_matmul(U_log, V_scaled, c_max, ptr,
                                       TILE_M=ctx.TILE_M, TILE_N=ctx.TILE_N, TILE_K=ctx.TILE_K)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        U_log, V_scaled, c_max, ptr = ctx.saved_tensors
        dU, dV = segmented_log_matmul_backward(U_log, V_scaled, c_max, ptr, grad_out,
                                               TILE_M=ctx.TILE_M, TILE_N=ctx.TILE_N, TILE_K=ctx.TILE_K)
        # No grads for c_max (stability term) and ptr (indices)
        return dU, dV, None, None, None, None, None


def segmented_log_matmul_autograd(U_log: torch.Tensor,
                                  V_scaled: torch.Tensor,
                                  c_max: torch.Tensor,
                                  ptr: torch.Tensor,
                                  *,
                                  TILE_M=64, TILE_N=128, TILE_K=32):
    """Autograd-ready wrapper using the tiled forward and backward kernels.

    Inputs:
      - U_log: [G,K] log-space
      - V_scaled: [NS,K,N] = exp(V_log - c_max)
      - c_max: [NS,N] (treated as constant)
      - ptr: [NS+1] segment boundaries
    Returns:
      - W_log: [G,N]
    """
    return SegmentedLogMatmulTiledFunction.apply(U_log, V_scaled, c_max, ptr, TILE_M, TILE_N, TILE_K)

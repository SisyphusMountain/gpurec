# Triton kernels for segmented log-matmul and its backward pass.
# We do that to compute the transfer probabilities in log-space 
# for the most general case: the transfer matrix can have any
# coefficients, and different transfer matrices can be applied 
# to the different gene families which are separated into 
# segments.

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 64,  "TILE_K": 32}, num_warps=4,  num_stages=2),
        triton.Config({"TILE_N": 128, "TILE_K": 32}, num_warps=4,  num_stages=2),
        triton.Config({"TILE_N": 128, "TILE_K": 64}, num_warps=8,  num_stages=2),
        triton.Config({"TILE_N": 256, "TILE_K": 32}, num_warps=8,  num_stages=2),
        triton.Config({"TILE_N": 256, "TILE_K": 64}, num_warps=16, num_stages=2),
        triton.Config({"TILE_N": 128, "TILE_K": 128},num_warps=8,  num_stages=3),
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
    R_ptr,                 # float32*  [G] row baselines (output)
    W_ptr,                 # float32*  [G, N]
    G, N, K, NS,           # sizes
    # strides
    stride_um, stride_uk,                      # U [G,K]
    stride_vs, stride_vk, stride_vn,           # V_scaled [NS,K,N]
    stride_cm_s, stride_cm_n,                  # c_max [NS,N]
    stride_rm,                                 # R [G]
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
    tl.assume(stride_rm > 0)

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
    S = tl.zeros((TILE_M, TILE_N), tl.float32)           # prob-space accumulator

    for kb in range(0, tl.cdiv(K, TILE_K)):
        k0 = kb * TILE_K
        k_offs = k0 + offs_k
        k_mask = k_offs < K

        # U chunk (log-space)
        u_ptrs = U_ptr + (offs_m[:, None] * stride_um + k_offs[None, :] * stride_uk)
        Ublk = tl.load(u_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF)

        # online row max + rescale
        r_blk = tl.max(Ublk, axis=1)
        new_r = tl.maximum(r, r_blk)
        S *= tl.exp2(r - new_r)[:, None]
        r = new_r

        Uexp = tl.exp2(Ublk - r[:, None])  # [BM, BK], safe

        # V_scaled chunk (linear space)
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
    Wtile = tl.where(S > 0, tl.log2(S), tl.full((TILE_M, TILE_N), NEG_INF, dtype=tl.float32))
    Wtile = Wtile + r[:, None] + c_seg[None, :]

    # store W
    w_ptrs = W_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
    tl.store(w_ptrs, Wtile, mask=mask_m[:, None] & mask_n[None, :])

    # store r once (pid_n==0)
    is_first_col_tile = pid_n == 0
    if is_first_col_tile:
        r_ptrs = R_ptr + offs_m * stride_rm
        tl.store(r_ptrs, r, mask=mask_m)


def segmented_log_matmul(
    U_log: torch.Tensor,        # [G, K] float32 (CUDA)
    V_scaled: torch.Tensor,     # [NS, K, N] float32 (CUDA)
    c_max: torch.Tensor,        # [NS, N] float32 (CUDA)
    ptr: torch.Tensor,          # [NS+1] int32/int64 (CPU or CUDA)
    *,
    TILE_M=64, TILE_N=128, TILE_K=32,
):
    G, K1 = U_log.shape
    NS, K2, N = V_scaled.shape
    assert K1 == K2

    device = U_log.device
    tile2seg, tile2row0, tile2row1 = _build_tile_lut(ptr, TILE_M, device)
    T = tile2seg.numel()
    W_log = torch.empty((G, N), device=device, dtype=torch.float32)
    R = torch.empty((G,),    device=device, dtype=torch.float32)  # row baselines

    if T == 0:
        W_log.zero_()
        R.fill_(float("-inf"))
        return W_log, R

    grid = lambda meta: (T, triton.cdiv(N, meta['TILE_N']))

    _seg_log_matmul[grid](
        U_log, V_scaled, c_max,
        tile2seg, tile2row0, tile2row1,
        R, W_log,
        G, N, K1, NS,
        U_log.stride(0), U_log.stride(1),
        V_scaled.stride(0), V_scaled.stride(1), V_scaled.stride(2),
        c_max.stride(0),    c_max.stride(1),
        R.stride(0),
        W_log.stride(0), W_log.stride(1),
        TILE_M=TILE_M,
    )
    return W_log, R

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


######## BACKWARD ########
@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 64,  "TILE_K": 32}, num_warps=4,  num_stages=2),
        triton.Config({"TILE_N": 128, "TILE_K": 32}, num_warps=4,  num_stages=2),
        triton.Config({"TILE_N": 128, "TILE_K": 64}, num_warps=8,  num_stages=2),
        triton.Config({"TILE_N": 256, "TILE_K": 32}, num_warps=8,  num_stages=2),
        triton.Config({"TILE_N": 256, "TILE_K": 64}, num_warps=16, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit
def _seg_dU_log(
    U_ptr,                 # float32* [G,K] (log-space)
    V_scaled_ptr,          # float32* [NS,K,N]
    c_max_ptr,             # float32* [NS,N]
    tile2seg_ptr,          # int32*   [T]
    tile2row0_ptr,         # int32*   [T]
    tile2row1_ptr,         # int32*   [T]
    R_ptr,                 # float32* [G]       (row baselines from fwd)
    Wlog_ptr,              # float32* [G,N]     (fwd output)
    Glog_ptr,              # float32* [G,N]     (incoming grad)
    dUlog_ptr,             # float32* [G,K]     (output)
    G, N, K, NS,           # sizes
    # strides
    stride_um, stride_uk,                  # U [G,K]
    stride_vs, stride_vk, stride_vn,       # V_scaled [NS,K,N]
    stride_cm_s, stride_cm_n,              # c_max [NS,N]
    stride_rm,                             # R [G]
    stride_wm, stride_wn,                  # W_log [G,N]
    stride_gm, stride_gn,                  # G_log [G,N]
    stride_dm, stride_dk,                  # dU_log [G,K]
    # meta
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    NEG_INF = float("-inf")

    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    # segment for this row tile
    s  = tl.load(tile2seg_ptr   + pid_m)
    r0 = tl.load(tile2row0_ptr  + pid_m)
    r1 = tl.load(tile2row1_ptr  + pid_m)

    offs_m = r0 + tl.arange(0, TILE_M)

    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    mask_m = offs_m < r1
    mask_k = offs_k < K

    # c_max for these cols will be fetched per-N tile
    # row baselines r[m]
    r = tl.load(R_ptr + offs_m * stride_rm, mask=mask_m, other=NEG_INF)  # [BM]

    # accumulator over N: [BM, BK]
    Acc = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    for nb in range(0, tl.cdiv(N, TILE_N)):
        offs_n = nb * TILE_N + tl.arange(0, TILE_N)
        mask_n = offs_n < N

        # V_scaled[N,K] view to do T @ V^T
        v_ptrs = (V_scaled_ptr
                  + s * stride_vs
                  + offs_k[None, :] * stride_vk
                  + offs_n[:, None] * stride_vn)
        Vblk_T = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)  # [BN, BK]

        # per-seg c_max for these columns
        c_ptrs = c_max_ptr + s * stride_cm_s + offs_n * stride_cm_n
        c_seg  = tl.load(c_ptrs, mask=mask_n, other=0.0)  # [BN]

        # tiles of W_log and G_log
        w_ptrs = Wlog_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
        g_ptrs = Glog_ptr + (offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn)
        Wtile  = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=NEG_INF)
        Gtile  = tl.load(g_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        # inv S* = exp(-(Wlog - c_max - r))
        invSstar = tl.exp2(-(Wtile - c_seg[None, :] - r[:, None]))
        # zero-out columns where W=-inf (S*=0)
        invSstar = tl.where(Wtile > NEG_INF, invSstar, 0.0)
        Ttile = Gtile * invSstar  # [BM, BN]

        # Accumulate: [BM,BN] @ [BN,BK] -> [BM,BK]
        Acc += tl.dot(Ttile, Vblk_T, input_precision="ieee")

    # multiply by exp(U_log - r) safely
    u_ptrs = U_ptr + (offs_m[:, None] * stride_um + offs_k[None, :] * stride_uk)
    Ublk_log = tl.load(u_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=NEG_INF)
    Uexp = tl.exp2(Ublk_log - r[:, None])  # <= 1
    dUtile = Uexp * Acc

    d_ptrs = dUlog_ptr + (offs_m[:, None] * stride_dm + offs_k[None, :] * stride_dk)
    tl.store(d_ptrs, dUtile, mask=mask_m[:, None] & mask_k[None, :])

@triton.autotune(
    configs=[
        triton.Config({"TILE_M": 64,  "TILE_N": 128, "TILE_K": 64}, num_warps=8,  num_stages=2),
        triton.Config({"TILE_M": 64,  "TILE_N": 256, "TILE_K": 32}, num_warps=8,  num_stages=2),
        triton.Config({"TILE_M": 128, "TILE_N": 128, "TILE_K": 32}, num_warps=8,  num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _seg_dV_scaled(
    U_ptr,                 # float32* [G,K] (log-space)
    c_max_ptr,             # float32* [NS,N]
    ptr_seg_bounds,        # int32*   [NS+1] (segment row pointers)
    R_ptr,                 # float32* [G]     (row baselines)
    Wlog_ptr,              # float32* [G,N]
    Glog_ptr,              # float32* [G,N]
    dV_ptr,                # float32* [NS,K,N] (output)
    G, N, K, NS,           # sizes
    # strides
    stride_um, stride_uk,                # U [G,K]
    stride_cm_s, stride_cm_n,            # c_max [NS,N]
    stride_rm,                           # R [G]
    stride_wm, stride_wn,                # W_log [G,N]
    stride_gm, stride_gn,                # G_log [G,N]
    stride_dvs, stride_dvk, stride_dvn,  # dV [NS,K,N]
    # meta
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    NEG_INF = float("-inf")

    s   = tl.program_id(axis=0)  # segment
    pk  = tl.program_id(axis=1)  # K-tile id
    pn  = tl.program_id(axis=2)  # N-tile id

    # segment row bounds [start, end)
    start = tl.load(ptr_seg_bounds + s)
    end   = tl.load(ptr_seg_bounds + s + 1)

    offs_k = pk * TILE_K + tl.arange(0, TILE_K)
    offs_n = pn * TILE_N + tl.arange(0, TILE_N)

    mask_k = offs_k < K
    mask_n = offs_n < N

    # c_max for this segment/columns
    c_ptrs = c_max_ptr + s * stride_cm_s + offs_n * stride_cm_n
    c_seg  = tl.load(c_ptrs, mask=mask_n, other=0.0)  # [BN]

    # accumulator for this tile: [BK, BN]
    Acc = tl.zeros((TILE_K, TILE_N), dtype=tl.float32)

    # loop over rows m in this segment (reduction over M)
    for mb in range(0, tl.cdiv(end - start, TILE_M)):
        m0 = start + mb * TILE_M
        offs_m = m0 + tl.arange(0, TILE_M)
        mask_m = offs_m < end

        # row baselines r[m]
        r = tl.load(R_ptr + offs_m * stride_rm, mask=mask_m, other=NEG_INF)  # [BM]

        # W_log and G_log tiles
        w_ptrs = Wlog_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
        g_ptrs = Glog_ptr + (offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn)
        Wtile  = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=NEG_INF)
        Gtile  = tl.load(g_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        # T = G_log * exp(-(Wlog - c_max - r))
        invSstar = tl.exp2(-(Wtile - c_seg[None, :] - r[:, None]))
        invSstar = tl.where(Wtile > NEG_INF, invSstar, 0.0)
        Ttile = Gtile * invSstar  # [BM, BN]

        # Uexp^T : [BK, BM]
        u_ptrs_T = U_ptr + (offs_m[None, :] * stride_um + offs_k[:, None] * stride_uk)
        Ublk_log_T = tl.load(u_ptrs_T, mask=mask_k[:, None] & mask_m[None, :], other=NEG_INF)
        Uexp_T = tl.exp2(Ublk_log_T - r[None, :])

        # Accumulate: [BK,BM] @ [BM,BN] -> [BK,BN]
        Acc += tl.dot(Uexp_T, Ttile, input_precision="ieee")

    # store dV for this segment tile
    dv_ptrs = dV_ptr + s * stride_dvs + (offs_k[:, None] * stride_dvk + offs_n[None, :] * stride_dvn)
    tl.store(dv_ptrs, Acc, mask=mask_k[:, None] & mask_n[None, :])



######## ADAPTED KERNELS FOR MANY VJPS ########


@triton.autotune(configs=[triton.Config({"TILE_M": 128, "TILE_K": 128}, num_warps=4, num_stages=2),
                          triton.Config({"TILE_M": 128, "TILE_K": 256}, num_warps=8, num_stages=2)],
                 key=["K"])
@triton.jit
def _row_max_kernel(U_ptr, R_ptr, G, K, stride_um, stride_uk,
                    TILE_M: tl.constexpr, TILE_K: tl.constexpr):
    NEG_INF = -float("inf")
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    mask_m = offs_m < G

    r = tl.full((TILE_M,), NEG_INF, tl.float32)
    offs_k = tl.arange(0, TILE_K)

    for kb in range(0, tl.cdiv(K, TILE_K)):
        k_offs = kb * TILE_K + offs_k
        k_mask = k_offs < K
        u_ptrs = U_ptr + (offs_m[:, None] * stride_um + k_offs[None, :] * stride_uk)
        Ublk = tl.load(u_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF)
        r = tl.maximum(r, tl.max(Ublk, axis=1))

    tl.store(R_ptr + offs_m, r, mask=mask_m)


def row_max(U_log: torch.Tensor):
    G, K = U_log.shape
    R = torch.empty((G,), device=U_log.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(G, meta["TILE_M"]),)
    _row_max_kernel[grid](U_log, R, G, K, U_log.stride(0), U_log.stride(1))
    return R


@triton.autotune(
    configs=[
        triton.Config({"TILE_M": 64,  "TILE_N": 128, "TILE_K": 32}, num_warps=8,  num_stages=2),
        triton.Config({"TILE_M": 64,  "TILE_N": 256, "TILE_K": 32}, num_warps=8,  num_stages=2),
        triton.Config({"TILE_M": 128, "TILE_N": 128, "TILE_K": 64}, num_warps=8,  num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _seg_record_forward(
    U_ptr,                 # float32* [G,K] (log-space)
    V_scaled_ptr,          # float32* [NS,K,N] (linear)
    c_max_ptr,             # float32* [NS,N]
    tile2seg_ptr,          # int32*   [T]
    tile2row0_ptr,         # int32*   [T]
    tile2row1_ptr,         # int32*   [T]
    R_ptr,                 # float32* [G]
    Uexp_ptr,              # *        [G,K]  (fp16/fp32) — written once (pid_n==0)
    W_ptr,                 # float32* [G,N]
    InvS_ptr,              # float32* [G,N]  (optional, STORE_INV_SSTAR)
    G, N, K, NS,
    # strides
    stride_um, stride_uk,                      # U [G,K]
    stride_vs, stride_vk, stride_vn,           # V_scaled [NS,K,N]
    stride_cm_s, stride_cm_n,                  # c_max [NS,N]
    stride_rm,                                 # R [G]
    stride_xm, stride_xk,                      # Uexp [G,K] (dtype may be f16 or f32)
    stride_wm, stride_wn,                      # W [G,N]
    stride_im, stride_in,                      # InvS [G,N] (if used)
    # meta
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    UEXP_DTYPE: tl.constexpr,                  # tl.float16 or tl.float32
    STORE_INV_SSTAR: tl.constexpr              # bool
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    NEG_INF = -float("inf")

    # segment + row window
    s  = tl.load(tile2seg_ptr   + pid_m)
    r0 = tl.load(tile2row0_ptr  + pid_m)
    r1 = tl.load(tile2row1_ptr  + pid_m)

    offs_m = r0 + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask_m = offs_m < r1
    mask_n = offs_n < N

    # r[m]
    r = tl.load(R_ptr + offs_m * stride_rm, mask=mask_m, other=NEG_INF)

    # c_max segment columns
    c_ptrs = c_max_ptr + s * stride_cm_s + offs_n * stride_cm_n
    c_seg  = tl.load(c_ptrs, mask=mask_n, other=NEG_INF)

    # accumulator S* in prob space
    S = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    offs_k = tl.arange(0, TILE_K)
    for kb in range(0, tl.cdiv(K, TILE_K)):
        k0 = kb * TILE_K
        k_offs = k0 + offs_k
        k_mask = k_offs < K

        # Uexp chunk
        u_ptrs = U_ptr + (offs_m[:, None] * stride_um + k_offs[None, :] * stride_uk)
        Ublk_log = tl.load(u_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF)
        Ublk_exp = tl.exp2(Ublk_log - r[:, None])  # <= 1, fp32 here

        # Store Uexp once (first column tile only), possibly in f16/bf16
        if pid_n == 0:
            x_ptrs = Uexp_ptr + (offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk)
            tl.store(x_ptrs, Ublk_exp.to(UEXP_DTYPE), mask=mask_m[:, None] & k_mask[None, :])

        # V chunk
        v_ptrs = (V_scaled_ptr
                  + s * stride_vs
                  + k_offs[:, None] * stride_vk
                  + offs_n[None, :] * stride_vn)
        Vblk = tl.load(v_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

        # accumulate S* += Uexp @ V
        S += tl.dot(Ublk_exp, Vblk, input_precision="ieee")

    # finalize W = log(S) + r + c_max
    Wtile = tl.where(S > 0, tl.log2(S), tl.full(S.shape, NEG_INF, tl.float32))
    Wtile = Wtile + r[:, None] + c_seg[None, :]

    # store W
    w_ptrs = W_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
    tl.store(w_ptrs, Wtile, mask=mask_m[:, None] & mask_n[None, :])

    # optional: store invS* = 1 / S*
    if STORE_INV_SSTAR:
        inv = tl.where(S > 0, 1.0 / S, 0.0)
        i_ptrs = InvS_ptr + (offs_m[:, None] * stride_im + offs_n[None, :] * stride_in)
        tl.store(i_ptrs, inv, mask=mask_m[:, None] & mask_n[None, :])

def segmented_log_matmul_record(
    U_log: torch.Tensor,        # [G,K], fp32
    V_scaled: torch.Tensor,     # [NS,K,N], fp32
    c_max: torch.Tensor,        # [NS,N], fp32
    ptr: torch.Tensor,          # [NS+1], int32/int64
    *,
    TILE_M=64, TILE_N=128, TILE_K=64,
    uexp_dtype=torch.float16,   # cache Uexp in f16/bf16/f32
    store_inv_sstar=False
):
    G, K = U_log.shape
    NS, K2, N = V_scaled.shape
    assert K2 == K and c_max.shape == (NS, N)
    device = U_log.device

    # 1) row-baseline
    R = row_max(U_log)

    # 2) build row-tiles across segments
    tile2seg, tile2row0, tile2row1 = _build_tile_lut(ptr, TILE_M, device)
    T = tile2seg.numel()

    # alloc caches / outputs
    Uexp = torch.empty((G, K), device=device, dtype=uexp_dtype)
    W_log = torch.empty((G, N), device=device, dtype=torch.float32)
    InvS = None
    if store_inv_sstar:
        InvS = torch.empty((G, N), device=device, dtype=torch.float32)

    # launch
    grid = lambda meta: (T, triton.cdiv(N, meta["TILE_N"]))
    _seg_record_forward[grid](
        U_log, V_scaled, c_max,
        tile2seg, tile2row0, tile2row1,
        R, Uexp, W_log, (InvS if store_inv_sstar else W_log),  # dummy ptr if not storing
        G, N, K, NS,
        U_log.stride(0), U_log.stride(1),
        V_scaled.stride(0), V_scaled.stride(1), V_scaled.stride(2),
        c_max.stride(0), c_max.stride(1),
        R.stride(0),
        Uexp.stride(0), Uexp.stride(1),
        W_log.stride(0), W_log.stride(1),
        (InvS.stride(0) if store_inv_sstar else W_log.stride(0)),
        (InvS.stride(1) if store_inv_sstar else W_log.stride(1)),
        TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
        UEXP_DTYPE=tl.float16 if uexp_dtype == torch.float16 else (tl.bfloat16 if uexp_dtype == torch.bfloat16 else tl.float32),
        STORE_INV_SSTAR=store_inv_sstar,
    )
    return W_log, R, Uexp, InvS

@triton.autotune(
    configs=[
        triton.Config({"TILE_M": 64, "TILE_N": 128, "TILE_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"TILE_M": 64, "TILE_N": 256, "TILE_K": 32}, num_warps=8, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit
def _seg_dU_log_cached(
    Uexp_ptr,               # *       [G,K] (f16/f32)
    V_scaled_ptr,           # float32 [NS,K,N]
    c_max_ptr,              # float32 [NS,N]
    tile2seg_ptr,           # int32   [T]
    tile2row0_ptr,          # int32   [T]
    tile2row1_ptr,          # int32   [T]
    R_ptr,                  # float32 [G]
    Wlog_ptr,               # float32 [G,N]
    Glog_ptr,               # float32 [G,N]
    InvS_ptr,               # float32 [G,N] (if cached; otherwise dummy)
    dUlog_ptr,              # float32 [G,K]
    G, N, K, NS,
    # strides
    stride_xm, stride_xk,                  # Uexp
    stride_vs, stride_vk, stride_vn,       # V_scaled
    stride_cm_s, stride_cm_n,              # c_max
    stride_rm,                             # R
    stride_wm, stride_wn,                  # W_log
    stride_gm, stride_gn,                  # G_log
    stride_im, stride_in,                  # InvS
    stride_dm, stride_dk,                  # dU_log
    # meta
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    UEXP_IS_F16: tl.constexpr,
    HAVE_INV_SSTAR: tl.constexpr
):
    NEG_INF = float("-inf")

    pid_m = tl.program_id(axis=0)  # row tiles across segments
    pid_k = tl.program_id(axis=1)  # K-tiles

    s  = tl.load(tile2seg_ptr  + pid_m)
    r0 = tl.load(tile2row0_ptr + pid_m)
    r1 = tl.load(tile2row1_ptr + pid_m)

    offs_m = r0 + tl.arange(0, TILE_M)
    mask_m = offs_m < r1
    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    mask_k = offs_k < K

    # accumulate over N
    Acc = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # row baselines r
    r = tl.load(R_ptr + offs_m * stride_rm, mask=mask_m, other=NEG_INF)

    for nb in range(0, tl.cdiv(N, TILE_N)):
        offs_n = nb * TILE_N + tl.arange(0, TILE_N)
        mask_n = offs_n < N

        # V^T tile
        v_ptrs = (V_scaled_ptr
                  + s * stride_vs
                  + offs_k[None, :] * stride_vk
                  + offs_n[:, None] * stride_vn)
        Vblk_T = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # build T = Glog * invS*
        g_ptrs = Glog_ptr + (offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn)
        Gtile = tl.load(g_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        if HAVE_INV_SSTAR:
            i_ptrs = InvS_ptr + (offs_m[:, None] * stride_im + offs_n[None, :] * stride_in)
            Inv = tl.load(i_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        else:
            # invS* = exp(-(Wlog - cmax - r))
            w_ptrs = Wlog_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
            Wtile  = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=NEG_INF)
            c_ptrs = c_max_ptr + s * stride_cm_s + offs_n * stride_cm_n
            c_seg  = tl.load(c_ptrs, mask=mask_n, other=0.0)
            Inv = tl.exp2(-(Wtile - c_seg[None, :] - r[:, None]))
            Inv = tl.where(Wtile > NEG_INF, Inv, 0.0)

        Ttile = Gtile * Inv  # [BM, BN]

        Acc += tl.dot(Ttile, Vblk_T, input_precision="ieee")

    # multiply by Uexp and store
    x_ptrs = Uexp_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    Xblk = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    if UEXP_IS_F16:
        Xblk = Xblk.to(tl.float32)
    dU = Xblk * Acc

    d_ptrs = dUlog_ptr + (offs_m[:, None] * stride_dm + offs_k[None, :] * stride_dk)
    tl.store(d_ptrs, dU, mask=mask_m[:, None] & mask_k[None, :])

@triton.autotune(
    configs=[
        triton.Config({"TILE_M": 64,  "TILE_N": 128, "TILE_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"TILE_M": 128, "TILE_N": 128, "TILE_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _seg_dV_scaled_cached(
    Uexp_ptr,               # *       [G,K]
    c_max_ptr,              # float32 [NS,N]
    ptr_seg_bounds,         # int32   [NS+1]
    R_ptr,                  # float32 [G]
    Wlog_ptr,               # float32 [G,N]
    Glog_ptr,               # float32 [G,N]
    InvS_ptr,               # float32 [G,N] (if cached; otherwise dummy)
    dV_ptr,                 # float32 [NS,K,N]
    G, N, K, NS,
    # strides
    stride_xm, stride_xk,                # Uexp
    stride_cm_s, stride_cm_n,            # c_max
    stride_rm,                           # R
    stride_wm, stride_wn,                # W_log
    stride_gm, stride_gn,                # G_log
    stride_im, stride_in,                # InvS
    stride_dvs, stride_dvk, stride_dvn,  # dV
    # meta
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    UEXP_IS_F16: tl.constexpr,
    HAVE_INV_SSTAR: tl.constexpr
):
    NEG_INF = float("-inf")

    s  = tl.program_id(axis=0)  # segment
    pk = tl.program_id(axis=1)  # K-tiles
    pn = tl.program_id(axis=2)  # N-tiles

    # segment row bounds
    start = tl.load(ptr_seg_bounds + s)
    end   = tl.load(ptr_seg_bounds + s + 1)

    offs_k = pk * TILE_K + tl.arange(0, TILE_K)
    offs_n = pn * TILE_N + tl.arange(0, TILE_N)
    mask_k = offs_k < K
    mask_n = offs_n < N

    # c_max for these columns
    c_ptrs = c_max_ptr + s * stride_cm_s + offs_n * stride_cm_n
    c_seg  = tl.load(c_ptrs, mask=mask_n, other=0.0)

    Acc = tl.zeros((TILE_K, TILE_N), dtype=tl.float32)

    for mb in range(0, tl.cdiv(end - start, TILE_M)):
        m0 = start + mb * TILE_M
        offs_m = m0 + tl.arange(0, TILE_M)
        mask_m = offs_m < end

        # r[m]
        r = tl.load(R_ptr + offs_m * stride_rm, mask=mask_m, other=NEG_INF)

        # form T
        g_ptrs = Glog_ptr + (offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn)
        Gtile = tl.load(g_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        if HAVE_INV_SSTAR:
            i_ptrs = InvS_ptr + (offs_m[:, None] * stride_im + offs_n[None, :] * stride_in)
            Inv = tl.load(i_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        else:
            w_ptrs = Wlog_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
            Wtile  = tl.load(w_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=NEG_INF)
            Inv = tl.exp2(-(Wtile - c_seg[None, :] - r[:, None]))
            Inv = tl.where(Wtile > NEG_INF, Inv, 0.0)

        Ttile = Gtile * Inv  # [BM, BN]

        # Uexp^T [BK, BM]
        x_ptrs_T = Uexp_ptr + (offs_m[None, :] * stride_xm + offs_k[:, None] * stride_xk)
        Xblk_T = tl.load(x_ptrs_T, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
        if UEXP_IS_F16:
            Xblk_T = Xblk_T.to(tl.float32)

        Acc += tl.dot(Xblk_T, Ttile, input_precision="ieee")

    # store dV for this segment tile
    dv_ptrs = dV_ptr + s * stride_dvs + (offs_k[:, None] * stride_dvk + offs_n[None, :] * stride_dvn)
    tl.store(dv_ptrs, Acc, mask=mask_k[:, None] & mask_n[None, :])


def vjp_cached_dU_dV(
    G_log: torch.Tensor,   # [G,N]
    Uexp: torch.Tensor,    # [G,K] (f16/bf16/f32)
    V_scaled: torch.Tensor,# [NS,K,N]
    c_max: torch.Tensor,   # [NS,N]
    ptr: torch.Tensor,     # [NS+1]
    R: torch.Tensor,       # [G]
    W_log: torch.Tensor,   # [G,N]
    invSstar: torch.Tensor | None,  # [G,N] or None
    *, TILE_M=64, TILE_N=128, TILE_K=64
):
    G, N = G_log.shape
    _, K = Uexp.shape
    NS = c_max.shape[0]
    device = G_log.device

    tile2seg, tile2row0, tile2row1 = _build_tile_lut(ptr, TILE_M, device)
    T = tile2seg.numel()

    dU_log = torch.empty((G, K), device=device, dtype=torch.float32)
    dV_scaled = torch.zeros_like(V_scaled)

    # dU
    if T > 0:
        grid_du = (T, triton.cdiv(K, TILE_K))
        _seg_dU_log_cached[grid_du](
            Uexp, V_scaled, c_max,
            tile2seg, tile2row0, tile2row1,
            R, W_log, G_log, (invSstar if invSstar is not None else W_log), dU_log,
            G, N, K, NS,
            Uexp.stride(0), Uexp.stride(1),
            V_scaled.stride(0), V_scaled.stride(1), V_scaled.stride(2),
            c_max.stride(0), c_max.stride(1),
            R.stride(0),
            W_log.stride(0), W_log.stride(1),
            G_log.stride(0), G_log.stride(1),
            (invSstar.stride(0) if invSstar is not None else W_log.stride(0)),
            (invSstar.stride(1) if invSstar is not None else W_log.stride(1)),
            dU_log.stride(0), dU_log.stride(1),
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            UEXP_IS_F16=(Uexp.dtype in (torch.float16, torch.bfloat16)),
            HAVE_INV_SSTAR=(invSstar is not None),
        )
    else:
        dU_log.zero_()

    # dV
    grid_dv = (NS, triton.cdiv(K, TILE_K), triton.cdiv(N, TILE_N))
    _seg_dV_scaled_cached[grid_dv](
        Uexp, c_max, ptr.to(torch.int32, device=device), R, W_log, G_log,
        (invSstar if invSstar is not None else W_log),
        dV_scaled,
        G, N, K, NS,
        Uexp.stride(0), Uexp.stride(1),
        c_max.stride(0), c_max.stride(1),
        R.stride(0),
        W_log.stride(0), W_log.stride(1),
        G_log.stride(0), G_log.stride(1),
        (invSstar.stride(0) if invSstar is not None else W_log.stride(0)),
        (invSstar.stride(1) if invSstar is not None else W_log.stride(1)),
        dV_scaled.stride(0), dV_scaled.stride(1), dV_scaled.stride(2),
        TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
        UEXP_IS_F16=(Uexp.dtype in (torch.float16, torch.bfloat16)),
        HAVE_INV_SSTAR=(invSstar is not None),
    )

    return dU_log, dV_scaled



def _make_scaled(V_log: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Given V_log [K,N], return V_scaled [K,N] = exp(V_log - c_max) and c_max [N]."""
    c_max = torch.max(V_log, dim=0, keepdim=False).values  # [N]
    V_scaled = torch.exp2(V_log - c_max.unsqueeze(0))       # [K,N]
    return V_scaled, c_max


def _ref_log_matmul(U_log: torch.Tensor, V_log: torch.Tensor) -> torch.Tensor:
    """Reference stabilized log-matmul: log(exp(U) @ exp(V)) with per-row/col stabilization.

    U_log: [G,K], V_log: [K,N]
    Returns: [G,N]
    """
    # Row stabilization for U
    U_max = torch.max(U_log, dim=1, keepdim=True).values  # [G,1]
    U_lin = torch.exp2(U_log - U_max)                      # [G,K]
    # Col stabilization for V
    V_scaled, c_max = _make_scaled(V_log)                 # [K,N], [N]
    Y_lin = U_lin @ V_scaled                              # [G,N]
    Y_log = torch.log2(Y_lin) + U_max + c_max.unsqueeze(0) # [G,N]
    return Y_log


def _test_segmented_log_matmul_basic():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping segmented_log_matmul test")
        return
    device = torch.device('cuda')
    torch.manual_seed(0)

    # Single segment case (NS=1)
    G, K, N = 128, 64, 96
    U_log = torch.randn(G, K, device=device, dtype=torch.float32)
    V_log = torch.randn(K, N, device=device, dtype=torch.float32)
    V_scaled, c_max = _make_scaled(V_log)
    V_scaled_b = V_scaled.unsqueeze(0)  # [1,K,N]
    c_max_b = c_max.unsqueeze(0)        # [1,N]
    ptr = torch.tensor([0, G], dtype=torch.int64, device=device)

    W_ref = _ref_log_matmul(U_log, V_log)
    W_kernel, R = segmented_log_matmul(U_log, V_scaled_b, c_max_b, ptr)
    assert torch.allclose(W_kernel, W_ref, rtol=1e-4, atol=1e-5), (
        f"Mismatch single segment: max_abs={torch.max(torch.abs(W_kernel - W_ref)).item():.3e}")

    # Two segments (NS=2), different V per segment
    G1 = G // 2
    G2 = G - G1
    U1 = U_log[:G1]
    U2 = U_log[G1:]
    V1_log = torch.randn(K, N, device=device, dtype=torch.float32)
    V2_log = torch.randn(K, N, device=device, dtype=torch.float32)
    V1_scaled, c1 = _make_scaled(V1_log)
    V2_scaled, c2 = _make_scaled(V2_log)
    Vb = torch.stack([V1_scaled, V2_scaled], dim=0)  # [2,K,N]
    cb = torch.stack([c1, c2], dim=0)                # [2,N]
    ptr2 = torch.tensor([0, G1, G], dtype=torch.int64, device=device)

    W1_ref = _ref_log_matmul(U1, V1_log)
    W2_ref = _ref_log_matmul(U2, V2_log)
    W_ref2 = torch.cat([W1_ref, W2_ref], dim=0)
    W_kernel2, _ = segmented_log_matmul(U_log, Vb, cb, ptr2)
    assert torch.allclose(W_kernel2, W_ref2, rtol=1e-4, atol=1e-5), (
        f"Mismatch two segments: max_abs={torch.max(torch.abs(W_kernel2 - W_ref2)).item():.3e}")
    print("segmented_log_matmul basic tests passed")


if __name__ == '__main__':
    _test_segmented_log_matmul_basic()
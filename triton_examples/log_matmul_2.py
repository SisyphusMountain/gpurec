# Triton kernel: stable log-matmul using row/col max offsets
# Computes W = log( exp(U) @ exp(V) ), with:
#   r_i = max_k U[i,k], c_j = max_k V[k,j]
#   W_ij = log( sum_k exp( U_ik - r_i ) * exp( V_kj - c_j ) ) + r_i + c_j
#
# Efficient streaming over K; keeps B=V hot in L2 via grouping + cache hints.
# Works best with float32 inputs.

import triton
import triton.language as tl
import torch




@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _log_matmul_rowcolmax_kernel(
    U_ptr, V_ptr, W_ptr,              # pointers
    M, N, K,                          # sizes
    stride_um, stride_uk,             # U strides: [M,K]
    stride_vk, stride_vn,             # V strides: [K,N]
    stride_wm, stride_wn,             # W strides: [M,N]
    # meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ---- grouped PID mapping (like Triton matmul tutorial) ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # tile row/col offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # base pointers for first K-block
    offs_k = tl.arange(0, BLOCK_K)
    u_ptrs = U_ptr + (offs_m[:, None] * stride_um + offs_k[None, :] * stride_uk)  # [BM, BK]
    v_ptrs = V_ptr + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)  # [BK, BN]

    NEG_INF = -float("inf")

    # -------- Pass 1: compute r (row max over K for this BM tile) and c (col max over K for this BN tile)
    r = tl.full((BLOCK_M,), NEG_INF, tl.float32)  # [BM]
    c = tl.full((BLOCK_N,), NEG_INF, tl.float32)  # [BN]

    ptrs_u = u_ptrs
    ptrs_v = v_ptrs
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - kb * BLOCK_K
        k_mask = offs_k < k_remaining

        # U load: streaming read (A is large, accessed once per row block)
        Ublk = tl.load(
            ptrs_u, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF
        )  # [BM, BK]
        # V load: B is small and reused across many A tiles - good cache locality
        Vblk = tl.load(
            ptrs_v, mask=k_mask[:, None] & mask_n[None, :], other=NEG_INF
        )  # [BK, BN]

        # Update r with row-wise max over BK: max_k U[i,k]
        r = tl.maximum(r, tl.max(Ublk, axis=1))  # [BM]
        # Update c with col-wise max over BK: max_k V[k,j]
        c = tl.maximum(c, tl.max(Vblk, axis=0))  # [BN]

        # advance to next K tile
        ptrs_u += BLOCK_K * stride_uk
        ptrs_v += BLOCK_K * stride_vk

    # -------- Pass 2: accumulate S = sum_k exp(U - r[:,None]) @ exp(V - c[None,:])
    S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # reset pointers
    ptrs_u = u_ptrs
    ptrs_v = v_ptrs
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - kb * BLOCK_K
        k_mask = offs_k < k_remaining

        Ublk = tl.load(
            ptrs_u, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF
        )  # [BM, BK]
        Vblk = tl.load(
            ptrs_v, mask=k_mask[:, None] & mask_n[None, :], other=NEG_INF
        )  # [BK, BN]

        # stable exponentials: subtract r and c (broadcast)
        Uexp = tl.exp(Ublk - r[:, None])         # [BM, BK]
        Vexp = tl.exp(Vblk - c[None, :])         # [BK, BN]

        # multiply-accumulate in probability space
        S += tl.dot(Uexp, Vexp)

        ptrs_u += BLOCK_K * stride_uk
        ptrs_v += BLOCK_K * stride_vk

    # finalize: log(S) + r + c, with zero-guard -> -inf
    # S == 0 means all contributions were effectively zero
    # Wtile = tl.where(S > 0, tl.log(S), tl.full((BLOCK_M, BLOCK_N), NEG_INF, tl.float32))
    
    Wtile = tl.log(S) + r[:, None] + c[None, :]

    # store
    w_ptrs = W_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
    tl.store(w_ptrs, Wtile, mask=mask_m[:, None] & mask_n[None, :])

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _log_matmul_rowcolmax_online_kernel(
    U_ptr, V_ptr, W_ptr,              # pointers
    M, N, K,                          # sizes
    stride_um, stride_uk,             # U strides: [M,K]
    stride_vk, stride_vn,             # V strides: [K,N]
    stride_wm, stride_wn,             # W strides: [M,N]
    # meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ---- grouped PID mapping (as in Triton matmul tutorial) ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # tile row/col offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # base pointers for first K-block
    offs_k = tl.arange(0, BLOCK_K)
    u_ptrs = U_ptr + (offs_m[:, None] * stride_um + offs_k[None, :] * stride_uk)  # [BM, BK]
    v_ptrs = V_ptr + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)  # [BK, BN]

    NEG_INF = -float("inf")

    # running row/col maxima and running probability-space accumulator
    r = tl.full((BLOCK_M,), NEG_INF, tl.float32)          # [BM]
    c = tl.full((BLOCK_N,), NEG_INF, tl.float32)          # [BN]
    S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)          # [BM, BN]

    # ---- single pass over K tiles ----
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - kb * BLOCK_K
        k_mask = offs_k < k_remaining

        # U load: streaming read (A is large, accessed once per row block)
        Ublk = tl.load(
            u_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF
        )  # [BM, BK]
        # V load: B is small and reused across many A tiles - good cache locality
        Vblk = tl.load(
            v_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=NEG_INF
        )  # [BK, BN]

        # block-local maxima over this BK slice
        r_blk = tl.max(Ublk, axis=1)   # [BM]
        c_blk = tl.max(Vblk, axis=0)   # [BN]

        # online update of row/col maxima
        new_r = tl.maximum(r, r_blk)   # [BM]
        new_c = tl.maximum(c, c_blk)   # [BN]

        # rescale the running sum S to the new baseline (outer product of row/col scales)
        # S *= exp( (r - new_r)[:,None] ) * exp( (c - new_c)[None,:] )
        row_scale = tl.exp(r - new_r)            # [BM]
        col_scale = tl.exp(c - new_c)            # [BN]
        S *= row_scale[:, None]
        S *= col_scale[None, :]

        # accumulate current BK slice in the new baseline
        Uexp = tl.exp(Ublk - new_r[:, None])     # [BM, BK]
        Vexp = tl.exp(Vblk - new_c[None, :])     # [BK, BN]
        S += tl.dot(Uexp, Vexp)

        # commit new baselines
        r = new_r
        c = new_c

        # advance K pointers
        u_ptrs += BLOCK_K * stride_uk
        v_ptrs += BLOCK_K * stride_vk

    # finalize: log(S) + r + c, with zero-guard -> -inf
    Wtile = tl.log(S)
    Wtile = Wtile + r[:, None] + c[None, :]

    # store
    w_ptrs = W_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
    tl.store(w_ptrs, Wtile, mask=mask_m[:, None] & mask_n[None, :])

def _launch_log_matmul_rowcolmax_online(U, V, W, M, N, K):
    # The autotuned kernel will automatically handle grid computation
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _log_matmul_rowcolmax_online_kernel[grid](
        U, V, W,
        M, N, K,
        U.stride(0), U.stride(1),
        V.stride(0), V.stride(1),
        W.stride(0), W.stride(1),
    )

def log_matmul_rowcolmax_online(U_log: torch.Tensor, V_log: torch.Tensor) -> torch.Tensor:
    """
    Compute W_log = log( exp(U_log) @ exp(V_log) ) using single-pass online row/col max stabilization.
    Shapes: U_log [M,K], V_log [K,N] -> W_log [M,N]. Uses float32.
    """
    assert U_log.is_cuda and V_log.is_cuda
    assert U_log.dtype == torch.float32 and V_log.dtype == torch.float32
    M, K = U_log.shape
    K2, N = V_log.shape
    assert K == K2
    W_log = torch.empty((M, N), device=U_log.device, dtype=torch.float32)
    _launch_log_matmul_rowcolmax_online(U_log, V_log, W_log, M, N, K)
    return W_log


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _log_matmul_rowcolmax_hybrid_kernel(
    U_ptr, V_ptr, W_ptr,              # pointers
    c_max_ptr,                        # precomputed column maxima of V [N]
    M, N, K,                          # sizes
    stride_um, stride_uk,             # U strides: [M,K]
    stride_vk, stride_vn,             # V strides: [K,N]
    stride_wm, stride_wn,             # W strides: [M,N]
    # meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ---- grouped PID mapping ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # tile row/col offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # base pointers for first K-block
    offs_k = tl.arange(0, BLOCK_K)
    u_ptrs = U_ptr + (offs_m[:, None] * stride_um + offs_k[None, :] * stride_uk)  # [BM, BK]
    v_ptrs = V_ptr + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)  # [BK, BN]
    
    # Load precomputed column maxima for this N block
    c_max_ptrs = c_max_ptr + offs_n  # [BN]
    c = tl.load(c_max_ptrs, mask=mask_n, other=-float("inf"))  # [BN]

    NEG_INF = -float("inf")

    # running row maxima and running probability-space accumulator
    r = tl.full((BLOCK_M,), NEG_INF, tl.float32)          # [BM]
    S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)          # [BM, BN]

    # ---- single pass over K tiles with online row max updates ----
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - kb * BLOCK_K
        k_mask = offs_k < k_remaining

        # U load: streaming read (A is large, accessed once per row block)
        Ublk = tl.load(
            u_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF
        )  # [BM, BK]
        # V load: B is small and reused across many A tiles - good cache locality
        Vblk = tl.load(
            v_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=NEG_INF
        )  # [BK, BN]

        # block-local row maxima over this BK slice
        r_blk = tl.max(Ublk, axis=1)   # [BM]

        # online update of row maxima only (column maxima are precomputed)
        new_r = tl.maximum(r, r_blk)   # [BM]

        # rescale the running sum S to the new row baseline
        row_scale = tl.exp(r - new_r)            # [BM]
        S *= row_scale[:, None]

        # accumulate current BK slice in the new baseline
        # Use precomputed column maxima c for V stabilization
        Uexp = tl.exp(Ublk - new_r[:, None])     # [BM, BK]
        Vexp = tl.exp(Vblk - c[None, :])         # [BK, BN]
        S += tl.dot(Uexp, Vexp, input_precision="ieee")

        # commit new row baseline
        r = new_r

        # advance K pointers
        u_ptrs += BLOCK_K * stride_uk
        v_ptrs += BLOCK_K * stride_vk

    # finalize: log(S) + r + c
    Wtile = tl.where(S > 0, tl.log(S), tl.full((BLOCK_M, BLOCK_N), NEG_INF, tl.float32))
    Wtile = Wtile + r[:, None] + c[None, :]

    # store
    w_ptrs = W_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
    tl.store(w_ptrs, Wtile, mask=mask_m[:, None] & mask_n[None, :])


# ----------------- Hybrid v2 kernel: takes pre-scaled V -----------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _log_matmul_rowcolmax_hybrid_kernel_2(
    U_ptr, V_scaled_ptr, W_ptr, c_max_ptr,      # V_scaled is already exp(log_V - c_max[None, :])
    M, N, K,                                     # sizes
    stride_um, stride_uk,                        # U strides: [M,K]
    stride_vk, stride_vn,                        # V_scaled strides: [K,N]
    stride_wm, stride_wn,                        # W strides: [M,N]
    stride_c,                                    # c_max stride: [N,]
    # meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Hybrid v2 kernel: Takes pre-scaled V matrix to avoid exp computation inside kernel.
    V_scaled = exp(log_V - c_max[None, :]) is computed on CPU/GPU before kernel call.
    This eliminates the expensive tl.exp(Vblk - c[None, :]) computation.
    """
    NEG_INF = -float("inf")
    
    # ---- grouped PID mapping (like Triton matmul tutorial) ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # tile row/col offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Load precomputed column maxima c_max [N]
    c = tl.load(c_max_ptr + offs_n * stride_c, mask=mask_n, other=NEG_INF)  # [BN]

    # K tiles offsets & pointers initialization
    offs_k = tl.arange(0, BLOCK_K)
    u_ptrs = U_ptr + (offs_m[:, None] * stride_um + offs_k[None, :] * stride_uk)
    v_ptrs = V_scaled_ptr + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)

    # Initialize running row maxima and sum
    r = tl.full((BLOCK_M,), NEG_INF, tl.float32)          # [BM]
    S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)          # [BM, BN]

    # ---- single pass over K tiles with online row max updates ----
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - kb * BLOCK_K
        k_mask = offs_k < k_remaining

        # U load: streaming read (A is large, accessed once per row block)
        Ublk = tl.load(
            u_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=NEG_INF
        )  # [BM, BK]
        
        # V_scaled load: pre-scaled matrix (already exp(log_V - c_max[None, :]))
        # Use evict_last policy to keep B warm across A row tiles
        V_scaled_blk = tl.load(
            v_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0, eviction_policy="evict_last"
        )  # [BK, BN] - note: other=0.0 since this is already in linear space

        # block-local row maxima over this BK slice
        r_blk = tl.max(Ublk, axis=1)   # [BM]

        # online update of row maxima only (column maxima are precomputed)
        new_r = tl.maximum(r, r_blk)   # [BM]

        # rescale the running sum S to the new row baseline
        row_scale = tl.exp(r - new_r)            # [BM]
        S *= row_scale[:, None]

        # accumulate current BK slice in the new baseline
        # V is already scaled, so we just need to scale U
        Uexp = tl.exp(Ublk - new_r[:, None])     # [BM, BK]
        # No need for: Vexp = tl.exp(Vblk - c[None, :]) - V is already scaled!
        S += tl.dot(Uexp, V_scaled_blk)

        # commit new row baseline
        r = new_r

        # advance K pointers
        u_ptrs += BLOCK_K * stride_uk
        v_ptrs += BLOCK_K * stride_vk

    # finalize: log(S) + r + c
    Wtile = tl.where(S > 0, tl.log(S), tl.full((BLOCK_M, BLOCK_N), NEG_INF, tl.float32))
    Wtile = Wtile + r[:, None] + c[None, :]

    # store
    w_ptrs = W_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
    tl.store(w_ptrs, Wtile, mask=mask_m[:, None] & mask_n[None, :])


def _launch_log_matmul_rowcolmax_hybrid(U, V, W, c_max, M, N, K):
    # The autotuned kernel will automatically handle grid computation
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _log_matmul_rowcolmax_hybrid_kernel[grid](
        U, V, W, c_max,
        M, N, K,
        U.stride(0), U.stride(1),
        V.stride(0), V.stride(1),
        W.stride(0), W.stride(1),
    )


def _launch_log_matmul_rowcolmax_hybrid_2(U, V_scaled, W, c_max, M, N, K):
    # Launcher for hybrid v2 kernel that takes pre-scaled V
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _log_matmul_rowcolmax_hybrid_kernel_2[grid](
        U, V_scaled, W, c_max,
        M, N, K,
        U.stride(0), U.stride(1),
        V_scaled.stride(0), V_scaled.stride(1),
        W.stride(0), W.stride(1),
        c_max.stride(0),  # stride for c_max array
    )


def log_matmul_rowcolmax_hybrid(U_log: torch.Tensor, V_log: torch.Tensor) -> torch.Tensor:
    """
    Compute W_log = log( exp(U_log) @ exp(V_log) ) using hybrid stabilization:
    - Precompute column maxima of V (since V is typically smaller)
    - Online row maxima updates for U (memory efficient for large U)
    Shapes: U_log [M,K], V_log [K,N] -> W_log [M,N]. Uses float32.
    """
    assert U_log.is_cuda and V_log.is_cuda
    assert U_log.dtype == torch.float32 and V_log.dtype == torch.float32
    M, K = U_log.shape
    K2, N = V_log.shape
    assert K == K2
    
    # Precompute column maxima of V_log
    c_max = torch.max(V_log, dim=0)[0]  # [N] - max over K dimension
    
    W_log = torch.empty((M, N), device=U_log.device, dtype=torch.float32)
    _launch_log_matmul_rowcolmax_hybrid(U_log, V_log, W_log, c_max, M, N, K)
    return W_log


def log_matmul_rowcolmax_hybrid_2(U_log: torch.Tensor, V_log: torch.Tensor) -> torch.Tensor:
    """
    Compute W_log = log( exp(U_log) @ exp(V_log) ) using hybrid v2 stabilization:
    - Precompute column maxima of V AND pre-scale V matrix on GPU
    - Online row maxima updates for U (memory efficient for large U)  
    - Eliminates expensive tl.exp(V - c_max) computation inside kernel
    Shapes: U_log [M,K], V_log [K,N] -> W_log [M,N]. Uses float32.
    """
    assert U_log.is_cuda and V_log.is_cuda
    assert U_log.dtype == torch.float32 and V_log.dtype == torch.float32
    M, K = U_log.shape
    K2, N = V_log.shape
    assert K == K2
    
    # Precompute column maxima of V_log
    c_max = torch.max(V_log, dim=0)[0]  # [N] - max over K dimension
    
    # Pre-scale V matrix: V_scaled = exp(V_log - c_max[None, :])
    V_scaled = torch.exp(V_log - c_max[None, :])  # [K, N]
    
    W_log = torch.empty((M, N), device=U_log.device, dtype=torch.float32)
    _launch_log_matmul_rowcolmax_hybrid_2(U_log, V_scaled, W_log, c_max, M, N, K)
    return W_log



# ----------------- Python launcher -----------------

def _launch_log_matmul_rowcolmax(U, V, W, M, N, K):
    # The autotuned kernel will automatically handle grid computation
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _log_matmul_rowcolmax_kernel[grid](
        U, V, W,
        M, N, K,
        U.stride(0), U.stride(1),
        V.stride(0), V.stride(1),
        W.stride(0), W.stride(1),
    )


def log_matmul_rowcolmax(U_log: torch.Tensor, V_log: torch.Tensor) -> torch.Tensor:
    """
    Compute W_log = log( exp(U_log) @ exp(V_log) ) using row/col max stabilization.
    This is equivalent to computing log(A @ B) where A = exp(U_log) and B = exp(V_log).
    Shapes: U_log [M,K], V_log [K,N] -> W_log [M,N]. Uses float32.
    """
    assert U_log.is_cuda and V_log.is_cuda
    assert U_log.dtype == torch.float32 and V_log.dtype == torch.float32
    M, K = U_log.shape
    K2, N = V_log.shape
    assert K == K2
    W_log = torch.empty((M, N), device=U_log.device, dtype=torch.float32)
    # Use autotuned kernel
    _launch_log_matmul_rowcolmax(U_log, V_log, W_log, M, N, K)
    return W_log


# ----------------- quick sanity test -----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N = 512, 500, 500
    
    # Create test matrices for log(exp(log_A) @ B) computation
    log_A = (torch.randn(M, K, device="cuda", dtype=torch.float32) - 5.0).contiguous()
    B_linear = torch.exp(torch.randn(K, N, device="cuda", dtype=torch.float32) - 5.0).contiguous()
    
    # For Triton: both inputs need to be in log space
    U = log_A
    V = torch.log(B_linear)
    
    # Triton results (all three versions)
    W_triton_2pass = log_matmul_rowcolmax(U, V)
    W_triton_online = log_matmul_rowcolmax_online(U, V)
    W_triton_hybrid = log_matmul_rowcolmax_hybrid(U, V)
    
    # PyTorch reference using row-wise max stabilization
    def pytorch_reference(log_A, B_linear):
        row_max = torch.max(log_A, dim=1, keepdim=True)[0]
        A_stable = torch.exp(log_A - row_max)
        result_linear = torch.matmul(A_stable, B_linear)
        return torch.log(result_linear) + row_max
    
    # Test on a small sample
    idx = torch.randint(0, M, (8,), device="cuda")
    ref = pytorch_reference(log_A[idx], B_linear)
    
    err_2pass = (W_triton_2pass[idx] - ref).abs().max().item()
    err_online = (W_triton_online[idx] - ref).abs().max().item()
    err_hybrid = (W_triton_hybrid[idx] - ref).abs().max().item()
    
    print("Max |Δ| on 8 sampled rows:")
    print(f"  Triton 2-pass:   {err_2pass:.6f}")
    print(f"  Triton online:   {err_online:.6f}")
    print(f"  Triton hybrid:   {err_hybrid:.6f}")
    
    # Also compare the Triton implementations with each other
    diff_2pass_online = (W_triton_2pass - W_triton_online).abs().max().item()
    diff_2pass_hybrid = (W_triton_2pass - W_triton_hybrid).abs().max().item()
    diff_online_hybrid = (W_triton_online - W_triton_hybrid).abs().max().item()
    print(f"  2-pass vs online: {diff_2pass_online:.6f}")
    print(f"  2-pass vs hybrid: {diff_2pass_hybrid:.6f}")
    print(f"  online vs hybrid: {diff_online_hybrid:.6f}")
    print()
    
    # ----------------- Benchmarking -----------------
    import numpy as np
    import triton.testing
    
    print("Benchmarking log-matmul with varying matrix sizes...")
    print("Matrix A: [G, S], Matrix B: [S, S] -> Output: [G, S]")
    print()
    
    # Generate G values: 10 steps logarithmically spaced from 400 to 10^6
    G_values = np.logspace(np.log10(400), np.log10(10**6), 10).astype(int)
    # Generate S values: 4 steps linearly spaced from 100 to 500
    S_values = np.linspace(100, 500, 4).astype(int)
    
    # Create separate benchmark function for each S value
    def create_benchmark_for_s(s_val):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=['G'],  # Only G varies, S is fixed
                x_vals=G_values,  # G values only
                line_arg='provider',  # Argument name for different lines
                line_vals=['triton', 'triton_online', 'triton_hybrid', 'triton_hybrid_v2', 'torch'],  # Different implementations
                line_names=['Triton (2-pass)', 'Triton (online)', 'Triton (hybrid)', 'Triton (hybrid v2)', 'PyTorch'],  # Labels for the lines
                styles=[('blue', '-'), ('red', '--'), ('purple', '-.'), ('orange', ':'), ('green', '-')],  # Line styles
                ylabel='Time (ms)',  # Label for y-axis
                plot_name=f'log-matmul-performance-S{s_val}',  # Unique plot name per S
                args={},  # No additional args needed
            ))
        def benchmark_s_fixed(G, provider):
            return benchmark_core(G, s_val, provider)
        
        return benchmark_s_fixed
    
    # Core benchmark function
    def benchmark_core(G, S, provider):
        # Create input matrices for log(exp(log_A) @ B) computation
        # A contains log values (log of original matrix)  
        log_A = torch.randn(G, S, device='cuda', dtype=torch.float32) - 5.0
        # B contains linear values (not log)
        B_linear = torch.exp(torch.randn(S, S, device='cuda', dtype=torch.float32) - 5.0)
        
        # For Triton kernel, we need both matrices in log space
        # So we'll pass log_A and log(B_linear)
        A = log_A.contiguous()
        B = torch.log(B_linear).contiguous()
        
        quantiles = [0.5, 0.2, 0.8]
        
        if provider == 'torch':
            # PyTorch implementation of log(exp(log_A) @ B) with row-wise max stabilization
            # This computes log(AB) where log_A is given and B is given
            # Similar to Triton version: subtract row max, do matmul, add row max back
            
            def torch_log_matmul():
                # A contains log(original_A), we have B_linear available
                log_A = A  # [G, S] - this is log of the original matrix
                
                # Step 1: Find row-wise max of log_A for numerical stability
                row_max = torch.max(log_A, dim=1, keepdim=True)[0]  # [G, 1]
                
                # Step 2: Subtract row max and exponentiate log_A
                A_stable = torch.exp(log_A - row_max)  # [G, S]
                
                # Step 3: Matrix multiplication in linear space using B_linear
                result_linear = torch.matmul(A_stable, B_linear)  # [G, S]
                
                # Step 4: Take log and add back the row max
                # Handle zeros by clamping to avoid log(0)
                result_log = torch.log(result_linear) + row_max  # [G, S]
                
                return result_log
            
            ms, min_ms, max_ms = triton.testing.do_bench(torch_log_matmul, quantiles=quantiles)
        
        elif provider == 'triton':
            # Triton implementation (2-pass)
            def triton_log_matmul():
                return log_matmul_rowcolmax(A, B)
            
            ms, min_ms, max_ms = triton.testing.do_bench(triton_log_matmul, quantiles=quantiles)
            
        elif provider == 'triton_online':
            # Triton online implementation (single-pass)
            def triton_online_log_matmul():
                return log_matmul_rowcolmax_online(A, B)
            
            ms, min_ms, max_ms = triton.testing.do_bench(triton_online_log_matmul, quantiles=quantiles)
            
        elif provider == 'triton_hybrid':
            # Triton hybrid implementation (precomputed col max + online row max)
            def triton_hybrid_log_matmul():
                return log_matmul_rowcolmax_hybrid(A, B)
            
            ms, min_ms, max_ms = triton.testing.do_bench(triton_hybrid_log_matmul, quantiles=quantiles)
            
        elif provider == 'triton_hybrid_v2':
            # Triton hybrid v2 implementation (pre-scaled V matrix)
            def triton_hybrid_v2_log_matmul():
                return log_matmul_rowcolmax_hybrid_2(A, B)
            
            ms, min_ms, max_ms = triton.testing.do_bench(triton_hybrid_v2_log_matmul, quantiles=quantiles)
        
        # Return times in milliseconds
        return ms, max_ms, min_ms
    
    # Option to run with or without PyTorch comparison
    RUN_PYTORCH_COMPARISON = True  # Set to False to only benchmark Triton
    
    if RUN_PYTORCH_COMPARISON:
        # Run separate benchmarks for each S value
        print("Running benchmarks with separate plots for each S value...")
        print("This will create 4 separate plots (one per S value)")
        print()
        
        for s_val in S_values:
            print(f"Running benchmark for S = {s_val}...")
            benchmark_func = create_benchmark_for_s(s_val)
            benchmark_func.run(print_data=True, show_plots=False)
            print()
    else:
        # Run only Triton benchmark
        print("Running Triton-only benchmark...")
        print()
        
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=['G', 'S'],  # Argument names to use as axes
                x_vals=[(g, s) for g in G_values for s in S_values],  # All combinations of G and S
                line_arg='provider',  # Argument name for different lines
                line_vals=['triton'],  # Only Triton
                line_names=['Triton'],  # Labels for the lines
                styles=[('blue', '-')],  # Line styles
                ylabel='Time (ms)',  # Label for y-axis
                plot_name='log-matmul-triton-performance',  # Plot name
                args={},  # Additional arguments
            ))
        def benchmark_triton_only(G, S, provider='triton'):
            # Create input matrices for log(exp(log_A) @ B) computation
            log_A = torch.randn(G, S, device='cuda', dtype=torch.float32) - 5.0
            B_linear = torch.exp(torch.randn(S, S, device='cuda', dtype=torch.float32) - 5.0)
            
            # For Triton kernel, we need both matrices in log space
            A = log_A.contiguous()
            B = torch.log(B_linear).contiguous()
            
            quantiles = [0.5, 0.2, 0.8]
            
            # Triton implementation
            def triton_log_matmul():
                return log_matmul_rowcolmax(A, B)
            
            ms, min_ms, max_ms = triton.testing.do_bench(triton_log_matmul, quantiles=quantiles)
            
            # Return times in milliseconds
            return ms, max_ms, min_ms
        
        benchmark_triton_only.run(print_data=True, show_plots=False)

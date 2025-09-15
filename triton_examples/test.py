# Triton kernel for C_log = log(A @ B) given logA = log(A) and B >= 0
# - Works fully in log-space: logsumexp over k of (logA[:, k] + log(B[k, :]))
# - Keeps exact -inf semantics
# - Simple, correct implementation (does per-k loads to avoid tricky tensor indexing)

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def _auto_configs():
    if is_cuda():
        return [
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ]
    else:
        sizes = [
            {'BLOCK_SIZE_M':  32, 'BLOCK_SIZE_N':  32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
            {'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N':  32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
            {'BLOCK_SIZE_M':  32, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
            {'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        ]
        return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]

@triton.jit
def _logaddexp(x, y):
    m = tl.maximum(x, y)
    res = m + tl.log(tl.exp(x - m) + tl.exp(y - m))
    # if both are -inf, keep -inf (avoid -inf + log(0) -> NaN)
    res = tl.where(m == float("-inf"), m, res)
    return res

@triton.autotune(configs=_auto_configs(), key=['M', 'N', 'K'])
@triton.jit
def log_matmul_kernel(
    loga_ptr, b_ptr, clog_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # program id mapping with grouped ordering for better L2 reuse of B columns
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0); tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0); tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0); tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0); tl.assume(stride_cn > 0)

    # row/col offsets for this block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_am_clamp = offs_am < M
    offs_bn_clamp = offs_bn < N

    # initialize accumulator to -inf (log(0))
    NEG_INF = tl.full((), float("-inf"), tl.float32)
    acc = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), float("-inf"), tl.float32)

    # sweep K in tiles, but load per-k slices to avoid 2D tensor indexing
    for kblk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # inner unrolled reduction over this K-tile
        for kk in range(0, BLOCK_SIZE_K):
            k_idx = kblk * BLOCK_SIZE_K + kk
            k_valid = k_idx < K  # scalar boolean

            # Pointers to logA[:, k_idx] and B[k_idx, :]
            a_ptrs_k = loga_ptr + offs_am * stride_am + k_idx * stride_ak         # [BM]
            b_ptrs_k = b_ptr   + k_idx * stride_bk   + offs_bn * stride_bn         # [BN]

            # Masks for loads
            a_mask = offs_am_clamp & k_valid
            b_mask = offs_bn_clamp & k_valid

            # Load logA col and log(B) row
            logA_k = tl.load(a_ptrs_k, mask=a_mask, other=NEG_INF).to(tl.float32)  # [BM]
            B_k    = tl.load(b_ptrs_k, mask=b_mask, other=0).to(tl.float32)        # [BN]
            logB_k = tl.where(B_k > 0, tl.log(B_k), NEG_INF)                       # [BN]

            # Broadcast and accumulate in log-space
            term = logA_k[:, None] + logB_k[None, :]                                # [BM, BN]
            acc = _logaddexp(acc, term)

    # write back
    c_ptrs = clog_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = offs_am_clamp[:, None] & offs_bn_clamp[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)

def matmul_logsumexp(logA: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C_log = log(A @ B) from logA=log(A) and B>=0.
    Shapes: logA [M,K], B [K,N] -> C_log [M,N] (float32)
    """
    assert logA.ndim == 2 and B.ndim == 2, "2D tensors required"
    assert logA.shape[1] == B.shape[0], "incompatible shapes"
    assert logA.is_contiguous() and B.is_contiguous(), "inputs must be contiguous"
    if torch.any(B < 0):
        raise ValueError("B contains negative values; log(B) undefined. Ensure B >= 0.")

    M, K = logA.shape
    _, N = B.shape
    C_log = torch.empty((M, N), device=logA.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    log_matmul_kernel[grid](
        logA, B, C_log,
        M, N, K,
        logA.stride(0), logA.stride(1),
        B.stride(0), B.stride(1),
        C_log.stride(0), C_log.stride(1),
    )
    return C_log

# ------------------------------
# Quick correctness test
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N = 128, 160, 96
    A = torch.exp(torch.rand((M, K), device=DEVICE, dtype=torch.float32))
    B = torch.exp(torch.rand((K, N), device=DEVICE, dtype=torch.float32))


    logA = torch.log(A)
    C_log = matmul_logsumexp(logA.contiguous(), B.contiguous())

    C_ref = A @ B
    C_log_ref = torch.where(C_ref > 0, torch.log(C_ref), torch.full_like(C_ref, float("-inf")))
    print("max |Δ| =", torch.max(torch.abs(C_log - C_log_ref)).item())
    print("allclose:", torch.allclose(C_log, C_log_ref, atol=1e-5, rtol=0))

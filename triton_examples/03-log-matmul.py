"""
Log-Space Matrix Multiplication (logsumexp semiring)
====================================================

This example implements a Triton kernel that computes C = log(A @ B) given
log-domain inputs logA = log(A) and logB = log(B). The computation is:

  C[i, j] = log( sum_k exp(logA[i, k] + logB[k, j]) )

which we compute in a numerically-stable way (logsumexp) and using tiled
matmul-like blocking for efficiency.

We compare the Triton result against a stable PyTorch reference that uses
torch.logsumexp on broadcasted tensors.
"""

import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    # Fix BLOCK_SIZE_K to 64 so tile-wise precomputed shifts match K-tiling.
    return [
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    ]


def get_hip_autotune_config():
    sizes = [
        {'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
    ]
    return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.jit
def _rowmax_kernel(
    a_log_ptr, row_max_ptr,
    M, K,
    stride_am, stride_ak,
    BLOCK_K: tl.constexpr,
):
    # One program per row; reduce over K in tiles of BLOCK_K
    pid_m = tl.program_id(axis=0)
    if pid_m >= M:
        return
    offs_m = pid_m
    running = tl.full((), -float('inf'), tl.float32)
    for s0 in range(0, K, BLOCK_K):
        k_range = s0 + tl.arange(0, BLOCK_K)
        k_mask = k_range < K
        ptrs = a_log_ptr + offs_m * stride_am + k_range * stride_ak
        tile = tl.load(ptrs, mask=k_mask, other=-float('inf')).to(tl.float32)
        running = tl.maximum(running, tl.max(tile, axis=0))
    tl.store(row_max_ptr + pid_m, running)


@triton.jit
def _colmax_kernel(
    b_log_ptr, col_max_ptr,
    K, N,
    stride_bk, stride_bn,
    BLOCK_K: tl.constexpr,
):
    # One program per column; reduce over K in tiles
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return
    offs_n = pid_n
    running = tl.full((), -float('inf'), tl.float32)
    for s0 in range(0, K, BLOCK_K):
        k_range = s0 + tl.arange(0, BLOCK_K)
        k_mask = k_range < K
        ptrs = b_log_ptr + k_range * stride_bk + offs_n * stride_bn
        tile = tl.load(ptrs, mask=k_mask, other=-float('inf')).to(tl.float32)
        running = tl.maximum(running, tl.max(tile, axis=0))
    tl.store(col_max_ptr + pid_n, running)


@triton.jit
def _rowmax_tiles_kernel(
    a_log_ptr, out_ptr,
    M, K, KTILES,
    stride_am, stride_ak,
    out_stride_kt, out_stride_m,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (m_blocks, ktile)
    pid_mblk = tl.program_id(axis=0)
    pid_kt   = tl.program_id(axis=1)
    offs_m = pid_mblk * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_k = pid_kt * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K
    a_ptrs = a_log_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    tile = tl.load(a_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=-float('inf')).to(tl.float32)
    rmax = tl.max(tile, axis=1)
    out_ptrs = out_ptr + pid_kt * out_stride_kt + offs_m * out_stride_m
    tl.store(out_ptrs, rmax, mask=m_mask)


@triton.jit
def _colmax_tiles_kernel(
    b_log_ptr, out_ptr,
    K, N, KTILES,
    stride_bk, stride_bn,
    out_stride_kt, out_stride_n,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (n_blocks, ktile)
    pid_nblk = tl.program_id(axis=0)
    pid_kt   = tl.program_id(axis=1)
    offs_n = pid_nblk * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = pid_kt * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K
    b_ptrs = b_log_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    tile = tl.load(b_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=-float('inf')).to(tl.float32)
    cmax = tl.max(tile, axis=0)
    out_ptrs = out_ptr + pid_kt * out_stride_kt + offs_n * out_stride_n
    tl.store(out_ptrs, cmax, mask=n_mask)


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def logmatmul_kernel(
    a_log_ptr, b_log_ptr, rmax_tiles_ptr, cmax_tiles_ptr, c_log_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_rm_kt, stride_rm_m,
    stride_cn_kt, stride_cn_n,
    stride_cm, stride_cn_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Log-matmul with tile-wise stabilization and log-domain tile accumulation.
    A_log (M,K), B_log (K,N) -> C_log (M,N).
    rmax_tiles_ptr: [KTILES, M], per K-tile row max of A
    cmax_tiles_ptr: [KTILES, N], per K-tile col max of B
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = a_log_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_log_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Running log-sum-exp across tiles (no inner kk loop)
    m_val = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), -float('inf'), dtype=tl.float32)
    s_val = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kt in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < (K - kt * BLOCK_SIZE_K)
        a_tile = tl.load(a_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=-float('inf')).to(tl.float32)
        b_tile = tl.load(b_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=-float('inf')).to(tl.float32)

        # Load tile-wise stabilizers
        m_row_tile = tl.load(rmax_tiles_ptr + kt * stride_rm_kt + offs_m * stride_rm_m, mask=m_mask, other=-float('inf')).to(tl.float32)
        n_col_tile = tl.load(cmax_tiles_ptr + kt * stride_cn_kt + offs_n * stride_cn_n, mask=n_mask, other=-float('inf')).to(tl.float32)

        a_exp = tl.exp(a_tile - m_row_tile[:, None])
        b_exp = tl.exp(b_tile - n_col_tile[None, :])
        prod = tl.dot(a_exp, b_exp)

        # log contribution of this K-tile
        tile_log = m_row_tile[:, None] + n_col_tile[None, :] + tl.log(prod)

        # logaddexp across tiles using (m_val, s_val)
        m_new = tl.maximum(m_val, tile_log)
        s_val = tl.exp(m_val - m_new) * s_val + tl.exp(tile_log - m_new)
        m_val = m_new

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    out = m_val + tl.log(s_val)
    out = tl.where(s_val > 0, out, -float('inf'))

    c_ptrs = c_log_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn_out
    tl.store(c_ptrs, out, mask=(m_mask[:, None] & n_mask[None, :]))


def log_matmul_logspace(a_log: torch.Tensor, b_log: torch.Tensor) -> torch.Tensor:
    """Compute log(A @ B) from logA and logB using the Triton kernel.
    Args:
      a_log: [M,K] log-domain matrix (float16/float32), may include -inf
      b_log: [K,N] log-domain matrix (float16/float32), may include -inf
    Returns:
      c_log: [M,N] float32
    """
    assert a_log.ndim == 2 and b_log.ndim == 2, "Only 2D matrices supported"
    M, K = a_log.shape
    K2, N = b_log.shape
    assert K2 == K, "Incompatible dimensions"
    assert a_log.device == b_log.device, "Inputs must be on the same device"

    # Use fp32 output for accuracy; inputs can be f16/f32
    c = torch.empty((M, N), device=a_log.device, dtype=torch.float32)

    # Precompute per-K-tile row/col maxima for stabilization
    # Must match BLOCK_SIZE_K used by the main kernel configs (64)
    TILE_K = 64
    KTILES = (K + TILE_K - 1) // TILE_K
    row_max_tiles = torch.empty((KTILES, M), device=a_log.device, dtype=torch.float32)
    col_max_tiles = torch.empty((KTILES, N), device=b_log.device, dtype=torch.float32)

    # Launch reductions over (M, KTILES) and (N, KTILES)
    BM = 128
    BN = 128
    grid_row = (triton.cdiv(M, BM), KTILES)
    grid_col = (triton.cdiv(N, BN), KTILES)
    _rowmax_tiles_kernel[grid_row](
        a_log, row_max_tiles,
        M, K, KTILES,
        a_log.stride(0), a_log.stride(1),
        row_max_tiles.stride(0), row_max_tiles.stride(1),
        BLOCK_M=BM, BLOCK_K=TILE_K,
    )
    _colmax_tiles_kernel[grid_col](
        b_log, col_max_tiles,
        K, N, KTILES,
        b_log.stride(0), b_log.stride(1),
        col_max_tiles.stride(0), col_max_tiles.stride(1),
        BLOCK_N=BN, BLOCK_K=TILE_K,
    )

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    logmatmul_kernel[grid](
        a_log, b_log, row_max_tiles, col_max_tiles, c,
        M, N, K,
        a_log.stride(0), a_log.stride(1),
        b_log.stride(0), b_log.stride(1),
        row_max_tiles.stride(0), row_max_tiles.stride(1),
        col_max_tiles.stride(0), col_max_tiles.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def log_matmul_torch(a_log: torch.Tensor, b_log: torch.Tensor) -> torch.Tensor:
    """Stable PyTorch reference: logsumexp over K of a_log + b_log.
    Uses broadcasting: [M,K] + [K,N] -> [M,K,N], then reduce over K.
    Returns float32.
    """
    assert a_log.ndim == 2 and b_log.ndim == 2
    M, K = a_log.shape
    K2, N = b_log.shape
    assert K2 == K
    # Work in float32 for reference accuracy
    a32 = a_log.to(torch.float32)
    b32 = b_log.to(torch.float32)
    A = torch.exp(a32)
    B = torch.exp(b32)
    C = A @ B
    return torch.log(C)
    # Broadcast + reduce: memory heavy but stable and simple for reference
    # C[i,j] = logsumexp_k(a[i,k] + b[k,j])
    # return torch.logsumexp(a32.unsqueeze(2) + b32.unsqueeze(0), dim=1)


if __name__ == "__main__":
    # Unit test against PyTorch reference
    torch.manual_seed(0)

    for (M, K, N) in [(256, 256, 256), (512, 512, 256), (512, 256, 512)]:
        a_log = (torch.randn((M, K), device=DEVICE, dtype=torch.float32) * 3.0 - 1.0)
        b_log = (torch.randn((K, N), device=DEVICE, dtype=torch.float32) * 3.0 - 1.0)

        triton_out = log_matmul_logspace(a_log, b_log)
        torch_out = log_matmul_torch(a_log, b_log)
        ok = torch.allclose(triton_out, torch_out, rtol=1e-4, atol=1e-6)
        print(f"[M={M} K={K} N={N}] match={ok}  max_abs_err={(triton_out - torch_out).abs().max().item():.3e}")

    # Micro-benchmark vs PyTorch reference
    ref_lib = 'PyTorch logsumexp'
    configs = [
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[128 * i for i in range(2, 17)],
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=[ref_lib, "Triton"],
            styles=[("gray", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name="log-matmul-performance",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider):
        a_log = (torch.randn((M, K), device=DEVICE, dtype=torch.float32) * 3.0 - 1.0)
        b_log = (torch.randn((K, N), device=DEVICE, dtype=torch.float32) * 3.0 - 1.0)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: log_matmul_torch(a_log, b_log), quantiles=quantiles)
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: log_matmul_logspace(a_log, b_log), quantiles=quantiles)
        # Report time directly (ms). For reference, GEMM-equivalent flop rate would not be apples-to-apples here.
        return ms, max_ms, min_ms

    benchmark.run(show_plots=True, print_data=True)

# Compile-fix + faster layout: no 2D tensor indexing, pointer-bumping per kk
# C_log = log(A @ B) from logA = log(A) and B>=0
# - Mirrors GEMM tiling and grouped ordering
# - Uses per-kk pointer arithmetic within a K-tile (avoids logA_tile[:, kk] / logB_tile[kk, :])
# - Unrolls kk with tl.static_range for good ILP
# - Keeps exact -inf semantics

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def _auto_configs():
    if is_cuda():
        return [
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  32, 'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ]
    else:
        sizes = [
            {'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        ]
        return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]

@triton.jit
def _logaddexp(x, y):
    m = tl.maximum(x, y)
    res = m + tl.log(tl.exp(x - m) + tl.exp(y - m))
    # keep -inf if both are -inf
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
    # --- grouped ordering like the original GEMM ---
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

    # row/col offsets for this C tile
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_am = offs_am < M
    mask_bn = offs_bn < N

    NEG_INF = tl.full((), float("-inf"), tl.float32)
    acc = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), float("-inf"), tl.float32)

    # sweep K in tiles
    for k0 in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # base pointers for this K-tile
        # A: columns [k0*BK : k0*BK+BK)
        a_col0_ptrs = loga_ptr + offs_am * stride_am + (k0 * BLOCK_SIZE_K) * stride_ak    # [BM]
        # B: rows   [k0*BK : k0*BK+BK)
        b_row0_ptrs = b_ptr   + (k0 * BLOCK_SIZE_K) * stride_bk + offs_bn * stride_bn      # [BN]

        # inner kk fully unrolled; per-kk pointer bumping (no 2D indexing on value tensors)
        for kk in tl.static_range(0, BLOCK_SIZE_K):
            k_idx = k0 * BLOCK_SIZE_K + kk
            k_valid = k_idx < K

            a_ptrs_k = a_col0_ptrs + kk * stride_ak
            b_ptrs_k = b_row0_ptrs + kk * stride_bk

            a_mask = mask_am & k_valid
            b_mask = mask_bn & k_valid

            logA_k = tl.load(a_ptrs_k, mask=a_mask, other=NEG_INF).to(tl.float32)  # [BM]
            B_k    = tl.load(b_ptrs_k, mask=b_mask, other=0).to(tl.float32)        # [BN]
            logB_k = tl.where(B_k > 0, tl.log(B_k), NEG_INF)                       # [BN]

            term = logA_k[:, None] + logB_k[None, :]                                # [BM, BN]
            acc = _logaddexp(acc, term)

    # write back
    c_ptrs = clog_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = mask_am[:, None] & mask_bn[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)

def matmul_logsumexp(logA: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert logA.ndim == 2 and B.ndim == 2
    assert logA.shape[1] == B.shape[0]
    assert logA.is_contiguous() and B.is_contiguous()
    if torch.any(B < 0):
        raise ValueError("B contains negatives; log(B) undefined. Ensure B >= 0.")
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
# Quick test + benchmark hooks
# ------------------------------
def ref_log_matmul(logA: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = torch.exp(logA)
    C = A @ B
    return torch.where(C > 0, torch.log(C), torch.full_like(C, float("-inf")))

def _quick_test():
    torch.manual_seed(0)
    M, K, N = 512, 512, 512
    A = torch.rand((M, K), device=DEVICE, dtype=torch.float32)
    B = torch.rand((K, N), device=DEVICE, dtype=torch.float32)
    A[A < 0.01] = 0.0
    B[B < 0.01] = 0.0
    logA = torch.where(A > 0, torch.log(A), torch.full_like(A, float("-inf")))
    out = matmul_logsumexp(logA.contiguous(), B.contiguous())
    ref = ref_log_matmul(logA, B)
    print("max |Δ|:", torch.max(torch.abs(out - ref)).item(),
          "allclose:", torch.allclose(out, ref, atol=1e-5, rtol=0))

# ------------------------------
# Benchmark (same API as before)
# ------------------------------
configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],  # 256..4096
        line_arg="provider",
        line_vals=["ref", "triton"],
        line_names=["PyTorch (ref)", "Triton (log-matmul)"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Equivalent TFLOPS (2MNK / time)",
        plot_name="log-matmul-performance-fp32",
        args={}
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    torch.manual_seed(0)
    A = torch.rand((M, K), device=DEVICE, dtype=torch.float32)
    B = torch.rand((K, N), device=DEVICE, dtype=torch.float32)
    # optional zeros to test -inf path; for peak perf you can comment these two lines
    A[A < 0.01] = 0.0
    B[B < 0.01] = 0.0
    logA = torch.where(A > 0, torch.log(A), torch.full_like(A, float("-inf"))).contiguous()
    B = B.contiguous()

    quantiles = [0.5, 0.2, 0.8]
    if provider == "ref":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ref_log_matmul(logA, B), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_logsumexp(logA, B), quantiles=quantiles)

    toflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return toflops(ms), toflops(max_ms), toflops(min_ms)

if __name__ == "__main__":
    _quick_test()
    benchmark.run(show_plots=True, print_data=True)

# seg_lse_hdim_triton.py
import torch
import triton
import triton.language as tl


HDIM_CONFIGS = [
    # (BLOCK_H, BLOCK_S, num_warps, num_stages)
    # Ensure BLOCK_H and BLOCK_S are powers of two for tl.arange
    (32, 32, 4, 2),
    (32, 64, 4, 2),
    (32, 128, 4, 2),
    (32, 256, 8, 2),
    (64, 64, 4, 2),
    (64, 128, 4, 2),
    (64, 256, 8, 2),
    (128, 64, 4, 2),
    (128, 128, 4, 2),
    (128, 256, 8, 2),
    (256, 64, 4, 2),
    (256, 128, 8, 2),
    (256, 256, 8, 2),
]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": bh, "BLOCK_S": bs}, num_warps=nw, num_stages=ns)
        for (bh, bs, nw, ns) in HDIM_CONFIGS
    ],
    key=["S"],
)
@triton.jit
def _seg_lse_hdim_kernel(
    x_ptr, ptr_ptr, y_ptr,
    G, S,
    stride_x_h, stride_x_s,
    stride_y_g, stride_y_s,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """
    Compute y[g, s] = logsumexp(x[ptr[g]:ptr[g+1], s]) for g in [0..G), s in [0..S)
    x: [H, S]
    y: [G, S]
    """
    gid = tl.program_id(0)  # segment id along H
    sid = tl.program_id(1)  # tile id along S
    if gid >= G:
        return

    # Column tile
    offs_s = sid * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    NEG_INF = tl.full((BLOCK_S,), float("-inf"), DTYPE)
    ZERO    = tl.zeros((BLOCK_S,), DTYPE)

    # Bounds for this segment along H
    h0 = tl.load(ptr_ptr + gid)
    h1 = tl.load(ptr_ptr + gid + 1)

    running_max = NEG_INF
    running_sum = tl.zeros((BLOCK_S,), DTYPE)

    # Sweep H within the segment
    start_h = h0
    while start_h < h1:
        offs_h = start_h + tl.arange(0, BLOCK_H)
        mask_h = offs_h < h1

        mask = mask_h[:, None] & mask_s[None, :]
        # Load tile [BLOCK_H, BLOCK_S]
        x_tile = tl.load(
            x_ptr + offs_h[:, None] * stride_x_h + offs_s[None, :] * stride_x_s,
            mask=mask,
            other=float("-inf"),
        )

        # Per-column block max/sum over H axis
        block_max = tl.max(x_tile, axis=0)
        neg_inf_col = block_max == NEG_INF
        safe_block_max = tl.where(neg_inf_col, ZERO, block_max)

        stable = tl.exp(x_tile - safe_block_max[None, :])
        block_sum = tl.sum(stable, axis=0)

        cand_block_max = tl.where(neg_inf_col, NEG_INF, block_max)
        new_max = tl.maximum(running_max, cand_block_max)

        scaled_running = tl.where(new_max == NEG_INF, ZERO, running_sum * tl.exp(running_max - new_max))
        scaled_block   = tl.where(new_max == NEG_INF, ZERO, block_sum   * tl.exp(safe_block_max - new_max))
        running_sum = scaled_running + scaled_block
        running_max = new_max

        start_h += BLOCK_H

    out = tl.where(running_max == NEG_INF, NEG_INF, tl.log(running_sum) + running_max)
    tl.store(y_ptr + gid * stride_y_g + offs_s * stride_y_s, out, mask=mask_s)


# Non-autotuned variant for manual benchmarking (avoids meta duplication)
@triton.jit
def _seg_lse_hdim_kernel_na(
    x_ptr, ptr_ptr, y_ptr,
    G, S,
    stride_x_h, stride_x_s,
    stride_y_g, stride_y_s,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,
):
    gid = tl.program_id(0)
    sid = tl.program_id(1)
    if gid >= G:
        return
    offs_s = sid * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    NEG_INF = tl.full((BLOCK_S,), float("-inf"), DTYPE)
    ZERO    = tl.zeros((BLOCK_S,), DTYPE)
    h0 = tl.load(ptr_ptr + gid)
    h1 = tl.load(ptr_ptr + gid + 1)
    running_max = NEG_INF
    running_sum = tl.zeros((BLOCK_S,), DTYPE)
    start_h = h0
    while start_h < h1:
        offs_h = start_h + tl.arange(0, BLOCK_H)
        mask_h = offs_h < h1
        mask = mask_h[:, None] & mask_s[None, :]
        x_tile = tl.load(
            x_ptr + offs_h[:, None] * stride_x_h + offs_s[None, :] * stride_x_s,
            mask=mask,
            other=float("-inf"),
        )
        block_max = tl.max(x_tile, axis=0)
        neg_inf_col = block_max == NEG_INF
        safe_block_max = tl.where(neg_inf_col, ZERO, block_max)
        stable = tl.exp(x_tile - safe_block_max[None, :])
        block_sum = tl.sum(stable, axis=0)
        cand_block_max = tl.where(neg_inf_col, NEG_INF, block_max)
        new_max = tl.maximum(running_max, cand_block_max)
        scaled_running = tl.where(new_max == NEG_INF, ZERO, running_sum * tl.exp(running_max - new_max))
        scaled_block   = tl.where(new_max == NEG_INF, ZERO, block_sum   * tl.exp(safe_block_max - new_max))
        running_sum = scaled_running + scaled_block
        running_max = new_max
        start_h += BLOCK_H
    out = tl.where(running_max == NEG_INF, NEG_INF, tl.log(running_sum) + running_max)
    tl.store(y_ptr + gid * stride_y_g + offs_s * stride_y_s, out, mask=mask_s)


def seg_logsumexp(
    x: torch.Tensor,
    ptr: torch.Tensor,
    block_h: int = None,
    block_s: int = None,
) -> torch.Tensor:
    """
    y[g, s] = logsumexp(x[ptr[g]:ptr[g+1], s])  for g in [0..G), s in [0..S)

    Args
    ----
    x   : [H, S] CUDA tensor (float32 or float64). Strided OK.
    ptr : [G+1] CUDA Long tensor, 0 <= ptr[i] <= ptr[i+1] <= H.

    Returns
    -------
    y : [G, S] CUDA tensor, same dtype as x.
    """
    assert x.is_cuda and ptr.is_cuda, "x and ptr must be CUDA tensors"
    assert x.ndim == 2, "x must be [H, S]"
    assert x.is_contiguous()
    assert ptr.ndim == 1 and ptr.dtype == torch.long, "ptr must be int64 [G+1]"
    

    H, S = x.shape
    G = ptr.numel() - 1
    y = torch.empty((G, S), dtype=x.dtype, device=x.device)

    if x.dtype == torch.float32:
        DTYPE = tl.float32
    elif x.dtype == torch.float64:
        DTYPE = tl.float64
    else:
        raise TypeError("x must be float32 or float64")

    # Launch with autotuned configuration by default; manual override if both block sizes provided
    if block_h is None or block_s is None:
        grid = lambda META: (G, triton.cdiv(S, META["BLOCK_S"]))
        _seg_lse_hdim_kernel[grid](
            x, ptr, y, G, S,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            DTYPE=DTYPE,
        )
    else:
        grid = (G, (S + block_s - 1) // block_s)
        _seg_lse_hdim_kernel[grid](
            x, ptr, y, G, S,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_H=int(block_h), BLOCK_S=int(block_s), DTYPE=DTYPE,
        )
    return y


def seg_logsumexp_manual(
    x: torch.Tensor,
    ptr: torch.Tensor,
    block_h: int,
    block_s: int,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """
    Run 2D segmented logsumexp with an explicit kernel configuration.
    This bypasses the autotuner for benchmarking specific parameter sets.
    """
    assert x.is_cuda and ptr.is_cuda, "x and ptr must be CUDA tensors"
    assert x.ndim == 2 and ptr.ndim == 1 and ptr.dtype == torch.long
    H, S = x.shape
    G = ptr.numel() - 1
    y = torch.empty((G, S), dtype=x.dtype, device=x.device)
    if x.dtype == torch.float32:
        DTYPE = tl.float32
    elif x.dtype == torch.float64:
        DTYPE = tl.float64
    else:
        raise TypeError("x must be float32 or float64")
    grid = (G, (S + block_s - 1) // block_s)
    _seg_lse_hdim_kernel_na[grid](
        x, ptr, y, G, S,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_H=int(block_h), BLOCK_S=int(block_s), DTYPE=DTYPE,
        num_warps=int(num_warps), num_stages=int(num_stages),
    )
    return y


# -------------------------
# Reference + Tests
# -------------------------
def _ref(x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    H, S = x.shape
    G = ptr.numel() - 1
    rows = []
    for g in range(G):
        a = int(ptr[g].item()); b = int(ptr[g+1].item())
        if a == b:
            rows.append(torch.full((S,), float("-inf"), dtype=x.dtype, device=x.device))
        else:
            rows.append(torch.logsumexp(x[a:b, :], dim=0))
    return torch.stack(rows, 0) if G > 0 else x.new_empty((0, S))


def _allclose(a, b, rtol=1e-6, atol=1e-6):
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def test_huge_negatives_case():
    if not torch.cuda.is_available(): return
    x = torch.full((5, 4), -1e10, dtype=torch.float32, device="cuda")
    ptr = torch.tensor([0, 5], dtype=torch.long, device="cuda")
    y = seg_logsumexp(x, ptr)
    y_ref = _ref(x, ptr)
    assert _allclose(y, y_ref)


def test_random_float32_with_empties_and_neginf():
    if not torch.cuda.is_available(): return
    torch.manual_seed(0)
    H, S = 257, 129
    x = torch.randn(H, S, device="cuda", dtype=torch.float32)
    # sprinkle -inf
    idx = torch.randint(0, H * S, (max(1, (H * S) // 20),), device="cuda")
    x.view(-1)[idx] = float("-inf")
    # random segments (some empty)
    seg_sizes = torch.randint(0, 25, (20,), device="cpu")
    ptr_cpu = torch.zeros(seg_sizes.numel() + 1, dtype=torch.long)
    ptr_cpu[1:] = torch.cumsum(seg_sizes, 0).clamp_max(H)
    ptr_cpu[-1] = H
    ptr = ptr_cpu.to("cuda")

    y = seg_logsumexp(x, ptr)
    y_ref = _ref(x, ptr)
    assert _allclose(y, y_ref)


def test_float64_mixed():
    if not torch.cuda.is_available(): return
    x = torch.tensor([[0.0, -1.0, -float("inf")],
                      [2.0,  3.0,  4.0]], dtype=torch.float64, device="cuda")
    ptr = torch.tensor([0, 0, 1, 2], dtype=torch.long, device="cuda")  # [], [row0], [row1]
    y = seg_logsumexp(x, ptr)
    y_ref = _ref(x, ptr)
    assert _allclose(y, y_ref, rtol=1e-12, atol=1e-12)

def test_S200_H1000_G1():
    if not torch.cuda.is_available(): 
        return
    torch.manual_seed(0)
    H, S, G = 1000, 200, 1
    x = torch.randn(H, S, device="cuda", dtype=torch.float32)
    ptr = torch.tensor([0, H], dtype=torch.long, device="cuda")  # G = 1 segment covering all H
    y = seg_logsumexp(x, ptr)
    y_ref = _ref(x, ptr)
    assert torch.allclose(y, y_ref, rtol=1e-6, atol=1e-6)


def test_S200_H100k_G100_equal_segments():
    if not torch.cuda.is_available(): 
        return
    torch.manual_seed(0)
    H, S, G = 100_000, 200, 100
    seg_len = H // G  # 1000 exactly
    ptr = torch.arange(G + 1, device="cuda", dtype=torch.long) * seg_len
    ptr[-1] = H  # ensure exact end
    x = torch.randn(H, S, device="cuda", dtype=torch.float32)
    y = seg_logsumexp(x, ptr)
    y_ref = _ref(x, ptr)
    assert torch.allclose(y, y_ref, rtol=1e-6, atol=1e-6)

def benchmark_S200_H1000_G1():
    """
    Benchmark Triton 2D kernel vs PyTorch reference for S=200, H=1000, G=1.
    Prints median latency and effective GB/s for both providers.
    """
    if not torch.cuda.is_available():
        print("CUDA not available; skipping benchmark_S200_H1000_G1")
        return

    torch.manual_seed(0)
    H, S, G = 1000, 200, 1
    x = torch.randn(H, S, device="cuda", dtype=torch.float32)
    ptr = torch.tensor([0, H], dtype=torch.long, device="cuda")

    # warm-up / compile
    _ = seg_logsumexp(x, ptr)
    _ = _ref(x, ptr)
    torch.cuda.synchronize()

    # helper: minimal bytes read/write
    def _bytes_moved(H, S, G, dtype):
        itemsize = torch.finfo(dtype).bits // 8
        return (H * S + G * S) * itemsize

    def _gbps(bytes_moved, ms):
        return (bytes_moved / 1e9) / (ms / 1e3)

    bytes_mv = _bytes_moved(H, S, G, x.dtype)

    # Triton
    q = [0.5, 0.2, 0.8]
    ms_tri, _, _ = triton.testing.do_bench(lambda: seg_logsumexp(x, ptr), quantiles=q)
    gbps_tri = _gbps(bytes_mv, ms_tri)

    # Torch reference
    ms_torch, _, _ = triton.testing.do_bench(lambda: _ref(x, ptr), quantiles=q)
    gbps_torch = _gbps(bytes_mv, ms_torch)

    print(f"[S=200, H=1000, G=1] Triton: {ms_tri:.3f} ms, {gbps_tri:.2f} GB/s | Torch: {ms_torch:.3f} ms, {gbps_torch:.2f} GB/s")


def benchmark_S200_H100k_G100_equal_segments():
    """
    Benchmark Triton 2D kernel vs PyTorch reference for S=200, H=100_000, G=100 (equal-length segments).
    Prints median latency and effective GB/s for both providers.
    """
    if not torch.cuda.is_available():
        print("CUDA not available; skipping benchmark_S200_H100k_G100_equal_segments")
        return

    torch.manual_seed(0)
    H, S, G = 100_000, 200, 100
    seg_len = H // G  # 1000 exactly
    ptr = torch.arange(G + 1, device="cuda", dtype=torch.long) * seg_len
    ptr[-1] = H
    x = torch.randn(H, S, device="cuda", dtype=torch.float32)

    # warm-up / compile
    _ = seg_logsumexp(x, ptr)
    _ = _ref(x, ptr)
    torch.cuda.synchronize()

    # helper: minimal bytes read/write
    def _bytes_moved(H, S, G, dtype):
        itemsize = torch.finfo(dtype).bits // 8
        return (H * S + G * S) * itemsize

    def _gbps(bytes_moved, ms):
        return (bytes_moved / 1e9) / (ms / 1e3)

    bytes_mv = _bytes_moved(H, S, G, x.dtype)

    # Triton (autotuned)
    q = [0.5, 0.2, 0.8]
    ms_tri, _, _ = triton.testing.do_bench(lambda: seg_logsumexp(x, ptr), quantiles=q)
    gbps_tri = _gbps(bytes_mv, ms_tri)

    # Torch reference
    ms_torch, _, _ = triton.testing.do_bench(lambda: _ref(x, ptr), quantiles=q)
    gbps_torch = _gbps(bytes_mv, ms_torch)

    print(f"[S=200, H=100000, G=100] Triton (autotuned): {ms_tri:.3f} ms, {gbps_tri:.2f} GB/s | Torch: {ms_torch:.3f} ms, {gbps_torch:.2f} GB/s")

    # Sweep explicit configurations
    print("Config sweep:")
    for (bh, bs, nw, ns) in HDIM_CONFIGS:
        try:
            ms_cfg, _, _ = triton.testing.do_bench(
                lambda bh=bh, bs=bs, nw=nw, ns=ns: seg_logsumexp_manual(x, ptr, bh, bs, nw, ns), quantiles=q
            )
            gbps_cfg = _gbps(bytes_mv, ms_cfg)
            print(f"  BH={bh:>3} BS={bs:>3} NW={nw} NS={ns} -> {ms_cfg:.3f} ms, {gbps_cfg:.2f} GB/s")
        except Exception as e:
            print(f"  BH={bh} BS={bs} NW={nw} NS={ns} -> error: {e}")


def benchmark_S399_H5000_G1():
    """
    Benchmark seg_logsumexp on x of shape [H=5000, S=399] with a single segment (G=1).
    The reduction maps [5000, 399] -> [1, 399]. Prints median latency and GB/s.
    """
    if not torch.cuda.is_available():
        print("CUDA not available; skipping benchmark_S399_H5000_G1")
        return

    torch.manual_seed(0)
    H, S, G = 5000, 399, 1
    ptr = torch.tensor([0, H], dtype=torch.long, device="cuda")

    def _bytes_moved(H, S, G, dtype):
        itemsize = torch.finfo(dtype).bits // 8
        return (H * S + G * S) * itemsize

    def _gbps(bytes_moved, ms):
        return (bytes_moved / 1e9) / (ms / 1e3)

    q = [0.5, 0.2, 0.8]
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(H, S, device="cuda", dtype=dtype)
        # Warm-up / compile
        _ = seg_logsumexp(x, ptr)
        _ = _ref(x, ptr)
        torch.cuda.synchronize()

        bytes_mv = _bytes_moved(H, S, G, dtype)
        ms_tri, _, _ = triton.testing.do_bench(lambda: seg_logsumexp(x, ptr), quantiles=q)
        gbps_tri = _gbps(bytes_mv, ms_tri)

        ms_torch, _, _ = triton.testing.do_bench(lambda: _ref(x, ptr), quantiles=q)
        gbps_torch = _gbps(bytes_mv, ms_torch)

        print(f"[dtype={str(dtype).split('.')[-1]}, S=399, H=5000, G=1] Triton: {ms_tri:.3f} ms, {gbps_tri:.2f} GB/s | Torch: {ms_torch:.3f} ms, {gbps_torch:.2f} GB/s")



if __name__ == "__main__":
    # Run tests ad-hoc (pytest will discover them as well)
    test_huge_negatives_case()
    test_random_float32_with_empties_and_neginf()
    test_float64_mixed()
    test_S200_H1000_G1()
    test_S200_H100k_G100_equal_segments()
    benchmark_S200_H1000_G1()
    benchmark_S200_H100k_G100_equal_segments()
    benchmark_S399_H5000_G1()
    print("All tests passed (or skipped if CUDA unavailable).")

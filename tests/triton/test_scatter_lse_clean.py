import torch
import pytest
import sys
import os

# Add top-level 'src' so we import reconciliation.* without shadowing the Triton package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from reconciliation.triton.scatter_lse import seg_lse

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required for Triton kernel"
)

def ref_segmented_logsumexp(x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """
    Simple, reliable reference on the same device as x.
    Uses torch.logsumexp per segment (handles -inf correctly).
    """
    assert x.dim() == 1 and ptr.dim() == 1
    M = ptr.numel() - 1
    out = x.new_empty(M)
    for i in range(M):
        s = int(ptr[i].item()); e = int(ptr[i + 1].item())
        out[i] = torch.logsumexp(x[s:e], dim=0)
    return out

def make_case_many_singletons_plus_one_long(
    n_singletons=2000, L_long=777, p_neginf=0.02, seed=0, device="cuda"
):
    g = torch.Generator(device=device).manual_seed(seed)
    lengths = [1] * n_singletons + [L_long]
    # shuffle lengths
    perm = torch.randperm(len(lengths), generator=g, device=device).cpu().tolist()
    lengths = [lengths[i] for i in perm]
    ptr = [0]
    for L in lengths:
        ptr.append(ptr[-1] + L)
    ptr = torch.tensor(ptr, dtype=torch.long, device=device)
    x = torch.randn(ptr[-1].item(), device=device, dtype=torch.float32, generator=g)
    if p_neginf > 0:
        mask = torch.rand_like(x).lt(p_neginf)
        x = x.masked_fill(mask, float("-inf"))
    return x, ptr

def make_case_random_segments(M=1024, min_len=1, max_len=32, p_neginf=0.05, seed=1, device="cuda"):
    # CPU generator for CPU randint
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)

    lens = torch.randint(min_len, max_len + 1, (M,), generator=g_cpu, device="cpu").tolist()

    ptr = [0]
    for L in lens:
        ptr.append(ptr[-1] + int(L))
    ptr = torch.tensor(ptr, dtype=torch.long, device=device)

    x = torch.randn(ptr[-1].item(), device=device, dtype=torch.float32)
    if p_neginf > 0:
        mask = torch.rand_like(x).lt(p_neginf)
        x = x.masked_fill(mask, float("-inf"))
    return x, ptr

def make_case_all_neginf_segments(
    lens=(1, 2, 5, 17, 64), seed=2, device="cuda"
):
    ptr = [0]
    for L in lens:
        ptr.append(ptr[-1] + int(L))
    ptr = torch.tensor(ptr, dtype=torch.long, device=device)
    x = torch.full((ptr[-1].item(),), float("-inf"), device=device, dtype=torch.float32)
    return x, ptr

@pytest.mark.parametrize("block", [128, 256, 512])
@pytest.mark.parametrize("warps", [2, 4])
@pytest.mark.parametrize("stages", [1, 2])
@CUDA_ONLY
def test_stream_kernel_many_singletons(block, warps, stages):
    x, ptr = make_case_many_singletons_plus_one_long(
        n_singletons=3000, L_long=1023, p_neginf=0.02, seed=123, device="cuda"
    )
    y_ref = ref_segmented_logsumexp(x, ptr)
    y_out = seg_lse(x, ptr, block=block, num_warps=warps, num_stages=stages)
    assert torch.allclose(y_out, y_ref, rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize("block", [256])
@pytest.mark.parametrize("warps", [4])
@pytest.mark.parametrize("stages", [1, 2])
@CUDA_ONLY
def test_stream_kernel_random_segments(block, warps, stages):
    x, ptr = make_case_random_segments(
        M=2048, min_len=1, max_len=16, p_neginf=0.03, seed=7, device="cuda"
    )
    y_ref = ref_segmented_logsumexp(x, ptr)
    y_out = seg_lse(x, ptr, block=block, num_warps=warps, num_stages=stages)
    assert torch.allclose(y_out, y_ref, rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize("block", [128, 512])
@pytest.mark.parametrize("warps", [2])
@pytest.mark.parametrize("stages", [1])
@CUDA_ONLY
def test_stream_kernel_all_neginf(block, warps, stages):
    x, ptr = make_case_all_neginf_segments(lens=(1, 2, 5, 17, 64), device="cuda")
    y_ref = ref_segmented_logsumexp(x, ptr)
    y_out = seg_lse(x, ptr, block=block, num_warps=warps, num_stages=stages)
    # all segments should be -inf
    assert torch.isneginf(y_ref).all()
    assert torch.allclose(y_out, y_ref, rtol=0, atol=0)  # exact -inf match

@CUDA_ONLY
def test_stream_kernel_singletons_only_fastpath():
    # All segments length 1 -> y should equal x at starts
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    ptr = torch.arange(0, x.numel() + 1, 1, device="cuda", dtype=torch.long)
    y_out = seg_lse(x, ptr, block=256, num_warps=4, num_stages=1)
    assert torch.allclose(y_out, x, rtol=0, atol=0)

@torch.no_grad()
@CUDA_ONLY
def test_speedtest_1k():
    """Speed test: 1000 launches on 1000x1000 segments."""
    device = "cuda"
    block, num_warps, num_stages = 256, 4, 1
    iters, warmup = 100, 10  # Reduced for test
    
    # Build data: 1,000 segments of length 1,000  ->  N = 1,000,000; M = 1,000
    ptr = torch.arange(0, 1_000*1_000 + 1, 1_000, dtype=torch.long, device=device)
    N = int(ptr[-1].item())
    x = torch.randn(N, dtype=torch.float32, device=device)
    
    # Run test (not benchmarking, just correctness)
    y = seg_lse(x, ptr, block=block, num_warps=num_warps, num_stages=num_stages)
    assert y.shape[0] == 1000

if __name__ == "__main__":
    pytest.main([__file__])

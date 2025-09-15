import os
import sys
import pytest
import torch

# Add top-level 'src' to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required for Triton kernel"
)


def make_random_segments(M=128, min_len=1, max_len=16, seed=0, device='cuda'):
    g_cpu = torch.Generator(device='cpu').manual_seed(seed)
    lens = torch.randint(min_len, max_len + 1, (M,), generator=g_cpu, device='cpu')
    ptr = torch.cat([torch.tensor([0], dtype=torch.long), lens.cumsum(0)]).to(device)
    return ptr


def ref_segmented_logsumexp_2d(x2d: torch.Tensor, ptr: torch.Tensor):
    # x2d: [N, S]
    M = ptr.numel() - 1
    S = x2d.shape[1]
    y = torch.empty((M, S), dtype=x2d.dtype, device=x2d.device)
    for i in range(M):
        s = int(ptr[i].item()); e = int(ptr[i + 1].item())
        y[i, :] = torch.logsumexp(x2d[s:e, :], dim=0)
    return y


@CUDA_ONLY
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_seg_lse_batched_random(dtype):
    from reconciliation.triton.scatter_lse import seg_lse_batched
    device = 'cuda'
    torch.manual_seed(0)
    ptr = make_random_segments(M=257, min_len=1, max_len=8, seed=3, device=device)
    N = int(ptr[-1].item())
    S = 97
    x2d = torch.randn(N, S, dtype=dtype, device=device)
    # sprinkle some -inf
    x2d = x2d.masked_fill(torch.rand_like(x2d) < 0.01, float('-inf'))

    y_ref = ref_segmented_logsumexp_2d(x2d, ptr)
    y = seg_lse_batched(x2d, ptr, block=256, num_warps=4, num_stages=1)

    rtol = 1e-6 if dtype == torch.float32 else 1e-12
    atol = 1e-6 if dtype == torch.float32 else 1e-12
    assert y.dtype == dtype
    assert torch.allclose(y, y_ref, rtol=rtol, atol=atol)


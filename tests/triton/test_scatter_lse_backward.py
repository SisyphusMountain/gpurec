import os
import sys
import pytest
import torch

# Add top-level 'src' to sys.path to import reconciliation.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required for Triton kernel"
)


def ref_lse_and_grad(x: torch.Tensor, ptr: torch.Tensor):
    assert x.dim() == 1 and ptr.dim() == 1
    device = x.device
    dtype = x.dtype
    M = ptr.numel() - 1
    y = torch.empty((M,), device=device, dtype=dtype)
    gx = torch.empty_like(x)
    for i in range(M):
        s = int(ptr[i].item()); e = int(ptr[i + 1].item())
        xi = x[s:e]
        # Compute stable softmax weights and y
        m = torch.max(xi)
        if torch.isneginf(m):
            y[i] = float('-inf')
            gx[s:e] = 0
            continue
        ex = torch.exp(xi - m)
        S = ex.sum()
        y[i] = torch.log(S) + m
        gx[s:e] = ex / S  # gradient when grad_y = 1 per segment
    return y, gx


@CUDA_ONLY
def test_scatter_lse_backward_random_segments_fp32():
    from reconciliation.triton.scatter_lse import seg_lse

    torch.manual_seed(0)
    device = 'cuda'
    M = 1024
    lens = torch.randint(1, 17, (M,), device='cpu')
    ptr = torch.cat([torch.tensor([0], dtype=torch.long), lens.cumsum(0)]).to(device)
    x = torch.randn(int(ptr[-1].item()), device=device, dtype=torch.float32, requires_grad=True)

    y = seg_lse(x, ptr, block=256, num_warps=4, num_stages=1)
    y.sum().backward()

    y_ref, gx_ref = ref_lse_and_grad(x.detach(), ptr)
    assert torch.allclose(y, y_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(x.grad, gx_ref, rtol=1e-6, atol=1e-6)


@CUDA_ONLY
def test_scatter_lse_backward_singletons_fp64():
    from reconciliation.triton.scatter_lse import seg_lse
    torch.manual_seed(1)
    device = 'cuda'
    n = 2048
    ptr = torch.arange(0, n + 1, 1, device=device, dtype=torch.long)
    x = torch.randn(n, device=device, dtype=torch.float64, requires_grad=True)
    # Inject some -inf values
    mask = torch.rand(n, device=device) < 0.05
    x = x.masked_fill(mask, float('-inf')).detach().requires_grad_(True)

    y = seg_lse(x, ptr, block=256, num_warps=4, num_stages=1)
    y.sum().backward()

    # For singletons: grad is 1 for finite x, 0 for -inf
    gx_ref = torch.where(torch.isfinite(x.detach()), torch.ones_like(x), torch.zeros_like(x))
    assert torch.allclose(y, x.detach(), rtol=0, atol=0)
    assert torch.allclose(x.grad, gx_ref, rtol=0, atol=0)


@CUDA_ONLY
def test_scatter_lse_backward_all_neginf():
    from reconciliation.triton.scatter_lse import seg_lse
    device = 'cuda'
    lengths = [1, 2, 5, 17, 64]
    ptr = [0]
    for L in lengths:
        ptr.append(ptr[-1] + L)
    ptr = torch.tensor(ptr, dtype=torch.long, device=device)
    x = torch.full((ptr[-1].item(),), float('-inf'), device=device, dtype=torch.float32, requires_grad=True)

    y = seg_lse(x, ptr, block=256, num_warps=4, num_stages=1)
    y.sum().backward()

    assert torch.isneginf(y).all()
    assert torch.allclose(x.grad, torch.zeros_like(x), rtol=0, atol=0)


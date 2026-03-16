import pytest
import torch

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for seg_logsumexp tests",
)


def _reference_seg_logsumexp(x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """Reference segmented logsumexp computed with PyTorch ops."""
    G = ptr.numel() - 1
    S = x.shape[1]
    out = x.new_full((G, S), float("-inf"))
    for g in range(G):
        start = int(ptr[g].item())
        end = int(ptr[g + 1].item())
        if start == end:
            continue
        out[g] = torch.logsumexp(x[start:end], dim=0)
    return out


def _reference_grad(x: torch.Tensor, ptr: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    """Closed-form gradient for seg_logsumexp."""
    grad = torch.zeros_like(x)
    G = ptr.numel() - 1
    for g in range(G):
        start = int(ptr[g].item())
        end = int(ptr[g + 1].item())
        if start == end:
            continue
        seg = x[start:end]
        col_max = torch.max(seg, dim=0).values
        finite_mask = torch.isfinite(col_max)
        if not finite_mask.any():
            continue
        safe_max = torch.where(finite_mask, col_max, torch.zeros_like(col_max))
        exp_terms = torch.exp(seg - safe_max)
        sums = exp_terms.sum(dim=0)
        probs = torch.zeros_like(seg)
        probs[:, finite_mask] = exp_terms[:, finite_mask] / sums[finite_mask]
        grad[start:end] += grad_out[g] * probs
    return grad


@CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_seg_logsumexp_matches_reference(dtype):
    from src.core.kernels.scatter_lse import seg_logsumexp

    device = "cuda"
    torch.manual_seed(1234)
    G = 41
    S = 13
    lengths = torch.randint(0, 25, (G,), dtype=torch.long)
    if G > 3:
        lengths[::7] = 0  # ensure empty segments appear
    if lengths.sum().item() == 0:
        lengths[0] = 5
    ptr_cpu = torch.cat([torch.tensor([0], dtype=torch.long), lengths.cumsum(0)])
    ptr = ptr_cpu.to(device)
    H = int(ptr_cpu[-1].item())
    x = torch.randn((H, S), device=device, dtype=dtype)
    if H > 0:
        mask = torch.rand((H, S), device=device) < 0.15
        x = x.masked_fill(mask, float("-inf"))
    y_ref = _reference_seg_logsumexp(x, ptr)
    y = seg_logsumexp(x.contiguous(), ptr)
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    assert torch.allclose(y, y_ref, rtol=tol, atol=tol)


@CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_seg_logsumexp_gradients(dtype):
    from src.core.kernels.scatter_lse import seg_logsumexp

    device = "cuda"
    torch.manual_seed(4321)
    G = 29
    S = 11
    lengths = torch.randint(0, 19, (G,), dtype=torch.long)
    if G > 4:
        lengths[4::5] = 0
    if lengths.sum().item() == 0:
        lengths[0] = 3
    ptr_cpu = torch.cat([torch.tensor([0], dtype=torch.long), lengths.cumsum(0)])
    ptr = ptr_cpu.to(device)
    H = int(ptr_cpu[-1].item())
    x = torch.randn((H, S), device=device, dtype=dtype)
    if H > 0:
        mask = torch.rand((H, S), device=device) < 0.2
        x = x.masked_fill(mask, float("-inf"))
    x = x.detach().requires_grad_(True)
    y = seg_logsumexp(x.contiguous(), ptr)
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    grad_expected = _reference_grad(x.detach(), ptr, grad_out)
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    assert torch.allclose(x.grad, grad_expected, rtol=tol, atol=tol)


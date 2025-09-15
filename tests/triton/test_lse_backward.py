import os
import sys
import pytest
import torch

# Add top-level 'src' to sys.path to import reconciliation.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required for Triton kernels"
)


def _rand_like(x, eps=0.1):
    return x + eps * torch.randn_like(x)


@CUDA_ONLY
class TestLSEBackward:
    def test_match_torch_grad_4(self):
        from reconciliation.triton.lse import lse4_triton_pair
        device = 'cuda'
        n = 8192
        xs = [torch.randn(n, device=device, dtype=torch.float32, requires_grad=True) for _ in range(4)]
        ys = [t.clone().detach().requires_grad_() for t in xs]

        y_triton = lse4_triton_pair(*xs)
        y_torch = torch.logsumexp(torch.stack(ys, dim=0), dim=0)

        loss_t = y_triton.sum()
        loss_r = y_torch.sum()
        loss_t.backward()
        loss_r.backward()

        for xt, yt in zip(xs, ys):
            assert torch.allclose(xt.grad, yt.grad, rtol=1e-5, atol=1e-6)

    def test_match_torch_grad_5(self):
        from reconciliation.triton.lse import lse5_triton_pair
        device = 'cuda'
        n = 4096
        xs = [torch.randn(n, device=device, dtype=torch.float32, requires_grad=True) for _ in range(5)]
        ys = [t.clone().detach().requires_grad_() for t in xs]

        y_triton = lse5_triton_pair(*xs)
        y_torch = torch.logsumexp(torch.stack(ys, dim=0), dim=0)

        y_triton.sum().backward()
        y_torch.sum().backward()

        for xt, yt in zip(xs, ys):
            assert torch.allclose(xt.grad, yt.grad, rtol=1e-5, atol=1e-6)

    def test_match_torch_grad_7(self):
        from reconciliation.triton.lse import lse7_triton_pair
        device = 'cuda'
        n = 2048
        xs = [torch.randn(n, device=device, dtype=torch.float32, requires_grad=True) for _ in range(7)]
        ys = [t.clone().detach().requires_grad_() for t in xs]

        y_triton = lse7_triton_pair(*xs)
        y_torch = torch.logsumexp(torch.stack(ys, dim=0), dim=0)

        y_triton.sum().backward()
        y_torch.sum().backward()

        for xt, yt in zip(xs, ys):
            assert torch.allclose(xt.grad, yt.grad, rtol=1e-5, atol=1e-6)

import os
import sys
import pytest
import torch

# Add top-level 'src' to sys.path to import without shadowing the Triton package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


def _sprinkle_neginf(x: torch.Tensor, p=0.001):
    m = torch.rand_like(x) < p
    x[m] = -float('inf')
    return x


CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required for Triton kernels"
)


@CUDA_ONLY
class TestLSE:
    def test_lse4_correctness(self):
        """Test 4-input logsumexp correctness vs torch."""
        from reconciliation.triton.lse import lse4_torch, lse4_triton_pair

        torch.manual_seed(0)
        device = torch.device('cuda')
        sizes = [1, 7, 513, 65536]

        for n in sizes:
            x0 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x1 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x2 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x3 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))

            yt = lse4_torch(x0, x1, x2, x3)
            yb = lse4_triton_pair(x0, x1, x2, x3)

            assert torch.allclose(yt, yb, atol=1e-6, rtol=1e-6)

    def test_lse5_correctness(self):
        """Test 5-input logsumexp correctness vs torch."""
        from reconciliation.triton.lse import lse5_torch, lse5_triton_pair

        torch.manual_seed(0)
        device = torch.device('cuda')
        sizes = [1, 7, 513, 65536]

        for n in sizes:
            x0 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x1 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x2 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x3 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x4 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))

            yt = lse5_torch(x0, x1, x2, x3, x4)
            yb = lse5_triton_pair(x0, x1, x2, x3, x4)

            assert torch.allclose(yt, yb, atol=1e-6, rtol=1e-6)

    def test_lse7_correctness(self):
        """Test 7-input logsumexp correctness vs torch."""
        from reconciliation.triton.lse import lse7_torch, lse7_triton_pair

        torch.manual_seed(0)
        device = torch.device('cuda')
        sizes = [1, 7, 513, 65536]

        for n in sizes:
            x0 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x1 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x2 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x3 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x4 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x5 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))
            x6 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float32))

            yt = lse7_torch(x0, x1, x2, x3, x4, x5, x6)
            yb = lse7_triton_pair(x0, x1, x2, x3, x4, x5, x6)

            assert torch.allclose(yt, yb, atol=1e-6, rtol=1e-6)

    def test_all_neginf_inputs(self):
        """All -inf inputs should yield -inf output."""
        from reconciliation.triton.lse import lse4_torch, lse4_triton_pair

        device = torch.device('cuda')
        n = 1024
        x = torch.full((n,), float('-inf'), device=device, dtype=torch.float32)
        yt = lse4_torch(x, x, x, x)
        yb = lse4_triton_pair(x, x, x, x)
        assert torch.isneginf(yt).all()
        assert torch.allclose(yt, yb, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__])

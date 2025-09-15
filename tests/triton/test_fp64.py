import os
import sys
import pytest
import torch

# Add top-level 'src' to sys.path to import from reconciliation.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required for Triton kernels"
)


def _sprinkle_neginf(x: torch.Tensor, p=0.01):
    mask = torch.rand_like(x) < p
    return x.masked_fill(mask, float('-inf'))


@CUDA_ONLY
class TestFP64LSE:
    def test_lse4_fp64(self):
        from reconciliation.triton.lse import lse4_torch, lse4_triton_pair
        device = torch.device('cuda')
        torch.manual_seed(0)
        for n in [1, 17, 4096]:
            x0 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float64))
            x1 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float64))
            x2 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float64))
            x3 = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float64))

            yt = lse4_torch(x0, x1, x2, x3)
            yk = lse4_triton_pair(x0, x1, x2, x3)

            assert yt.dtype == torch.float64 and yk.dtype == torch.float64
            assert torch.allclose(yt, yk, rtol=1e-12, atol=1e-12)

    def test_lse5_fp64(self):
        from reconciliation.triton.lse import lse5_torch, lse5_triton_pair
        device = torch.device('cuda')
        torch.manual_seed(1)
        for n in [1, 17, 4096]:
            xs = [_sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float64)) for _ in range(5)]
            yt = lse5_torch(*xs)
            yk = lse5_triton_pair(*xs)
            assert yt.dtype == torch.float64 and yk.dtype == torch.float64
            assert torch.allclose(yt, yk, rtol=1e-12, atol=1e-12)

    def test_lse7_fp64(self):
        from reconciliation.triton.lse import lse7_torch, lse7_triton_pair
        device = torch.device('cuda')
        torch.manual_seed(2)
        for n in [1, 17, 4096]:
            xs = [_sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float64)) for _ in range(7)]
            yt = lse7_torch(*xs)
            yk = lse7_triton_pair(*xs)
            assert yt.dtype == torch.float64 and yk.dtype == torch.float64
            assert torch.allclose(yt, yk, rtol=1e-12, atol=1e-12)


@CUDA_ONLY
class TestFP64ScatterLSE:
    def ref_segmented_logsumexp(self, x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 1 and ptr.dim() == 1
        M = ptr.numel() - 1
        out = x.new_empty(M)
        for i in range(M):
            s = int(ptr[i].item()); e = int(ptr[i + 1].item())
            out[i] = torch.logsumexp(x[s:e], dim=0)
        return out

    def make_random_segments(self, M=512, min_len=1, max_len=32, seed=0, device='cuda'):
        g_cpu = torch.Generator(device='cpu').manual_seed(seed)
        lens = torch.randint(min_len, max_len + 1, (M,), generator=g_cpu, device='cpu')
        ptr = torch.cat([torch.tensor([0], dtype=torch.long), lens.cumsum(0)]).to(device)
        return ptr

    def test_seg_lse_fp64_random(self):
        from reconciliation.triton.scatter_lse import seg_lse
        device = torch.device('cuda')
        torch.manual_seed(3)
        ptr = self.make_random_segments(M=512, min_len=1, max_len=32, seed=3, device=device)
        N = int(ptr[-1].item())
        x = _sprinkle_neginf(torch.randn(N, device=device, dtype=torch.float64), p=0.03)
        y_ref = self.ref_segmented_logsumexp(x, ptr)
        y_out = seg_lse(x, ptr, block=256, num_warps=4, num_stages=1)
        # Expect FP64 output and tight FP64 accuracy
        assert y_out.dtype == torch.float64
        assert torch.allclose(y_out, y_ref, rtol=1e-12, atol=1e-12)

    def test_seg_lse_fp64_singletons(self):
        from reconciliation.triton.scatter_lse import seg_lse
        device = torch.device('cuda')
        n = 2048
        ptr = torch.arange(0, n + 1, 1, device=device, dtype=torch.long)
        x = _sprinkle_neginf(torch.randn(n, device=device, dtype=torch.float64), p=0.1)
        y_ref = self.ref_segmented_logsumexp(x, ptr)
        y_out = seg_lse(x, ptr, block=256, num_warps=4, num_stages=1)
        assert y_out.dtype == torch.float64
        # For singletons, result should be exactly x
        assert torch.allclose(y_out, x, rtol=0, atol=0)


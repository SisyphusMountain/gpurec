import os
import sys
import pytest
import torch

# Add top-level 'src' to sys.path to avoid shadowing the Triton package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


def ref_segmented_logsumexp(x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 1 and ptr.dim() == 1
    M = ptr.numel() - 1
    out = x.new_empty(M)
    for i in range(M):
        s = int(ptr[i].item()); e = int(ptr[i + 1].item())
        out[i] = torch.logsumexp(x[s:e], dim=0)
    return out


def make_segments_with_one_long(n_singletons: int, L: int, n_smalls: int = 5, seed: int = 0, device: str = 'cuda'):
    g = torch.Generator(device=device).manual_seed(seed)
    lengths = [1] * n_singletons + [L] + [int(torch.randint(2, 9, (1,), generator=g, device=device).item()) for _ in range(n_smalls)]
    perm = torch.randperm(len(lengths), generator=g, device=device).cpu().tolist()
    lengths = [lengths[i] for i in perm]
    ptr = [0]
    for ll in lengths:
        ptr.append(ptr[-1] + ll)
    ptr = torch.tensor(ptr, dtype=torch.long, device=device)
    x = torch.randn(ptr[-1].item(), device=device, dtype=torch.float32, generator=g)
    # sprinkle some -inf
    mask = torch.rand_like(x) < 0.03
    x = x.masked_fill(mask, float('-inf'))
    return x, ptr


def make_segments_with_k_longs(n_singletons=500, long_count=10, long_len=100, n_smalls=20, seed=0, device='cuda'):
    g = torch.Generator(device=device).manual_seed(seed)
    lengths = [1] * n_singletons + [long_len] * long_count + [int(torch.randint(2, 17, (1,), generator=g, device=device).item()) for _ in range(n_smalls)]
    perm = torch.randperm(len(lengths), generator=g, device=device).cpu().tolist()
    lengths = [lengths[i] for i in perm]
    ptr = [0]
    for ll in lengths:
        ptr.append(ptr[-1] + ll)
    ptr = torch.tensor(ptr, dtype=torch.long, device=device)
    x = torch.randn(ptr[-1].item(), device=device, dtype=torch.float32, generator=g)
    # sprinkle some -inf
    mask = torch.rand_like(x) < 0.05
    x = x.masked_fill(mask, float('-inf'))
    return x, ptr


CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required for Triton kernel"
)


@CUDA_ONLY
class TestScatterLSE:
    def test_kernel_correctness_various_segments(self):
        """Test seg_lse correctness with various segment configurations."""
        from reconciliation.triton.scatter_lse import seg_lse

        torch.manual_seed(0)
        sizes = [(2000, 257), (2000, 777), (10000, 1023)]

        for n_s, L in sizes:
            x, ptr = make_segments_with_one_long(n_s, L, n_smalls=5, seed=L, device="cuda")

            y_ref = ref_segmented_logsumexp(x, ptr)
            y_out = seg_lse(x, ptr, block=256, num_warps=4, num_stages=1)

            assert torch.allclose(y_out, y_ref, rtol=1e-6, atol=1e-6)

    def test_singleton_segments_fastpath(self):
        """All segments length 1 should return x at starts."""
        from reconciliation.triton.scatter_lse import seg_lse

        torch.manual_seed(42)
        device = "cuda"

        n_singletons = 2000
        ptr = torch.arange(0, n_singletons + 1, 1, dtype=torch.long, device=device)
        x = torch.randn(n_singletons, dtype=torch.float32, device=device)

        # Add some -inf values
        mask = torch.rand_like(x) < 0.1
        x = x.masked_fill(mask, float('-inf'))

        y_ref = ref_segmented_logsumexp(x, ptr)
        y_out = seg_lse(x, ptr, block=256, num_warps=4, num_stages=1)

        assert torch.allclose(y_ref, x, rtol=0, atol=0)
        assert torch.allclose(y_out, x, rtol=0, atol=0)

    def test_all_neginf_segments(self):
        """Segments where all values are -inf produce -inf."""
        from reconciliation.triton.scatter_lse import seg_lse

        device = "cuda"
        lengths = [1, 2, 5, 17, 64]
        ptr = [0]
        for L in lengths:
            ptr.append(ptr[-1] + L)
        ptr = torch.tensor(ptr, dtype=torch.long, device=device)

        x = torch.full((ptr[-1].item(),), float("-inf"), device=device, dtype=torch.float32)

        y_ref = ref_segmented_logsumexp(x, ptr)
        y_out = seg_lse(x, ptr, block=256, num_warps=4, num_stages=1)

        assert torch.isneginf(y_ref).all()
        assert torch.allclose(y_out, y_ref, rtol=0, atol=0)

    def test_random_segments(self):
        """Random small segments with sprinkled -inf."""
        from reconciliation.triton.scatter_lse import seg_lse

        torch.manual_seed(7)
        device = 'cuda'
        M = 2048
        lens = torch.randint(1, 17, (M,), device='cpu')
        ptr = torch.cat([torch.tensor([0], dtype=torch.long), lens.cumsum(0)]).to(device)

        x = torch.randn(int(ptr[-1].item()), device=device, dtype=torch.float32)
        x = x.masked_fill(torch.rand_like(x) < 0.03, float('-inf'))

        y_ref = ref_segmented_logsumexp(x, ptr)
        y_out = seg_lse(x, ptr, block=256, num_warps=4, num_stages=1)

        assert torch.allclose(y_out, y_ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])

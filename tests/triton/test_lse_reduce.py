import os
import sys
import pytest
import torch

# Add top-level 'src' to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required for Triton kernels"
)


def sprinkle_neginf(x: torch.Tensor, p=0.02):
    m = torch.rand_like(x) < p
    return x.masked_fill(m, float('-inf'))


@CUDA_ONLY
@pytest.mark.parametrize("shape,dim,K,fn_name", [
    ((4, 1024), 0, 4, 'lse4_reduce'),
    ((1024, 4),  1, 4, 'lse4_reduce'),
    ((2, 5, 257), 1, 5, 'lse5_reduce'),
    ((7, 3, 129), 0, 7, 'lse7_reduce'),
    ((33, 7), -1, 7, 'lse7_reduce'),
])
def test_lseK_reduce_correctness(shape, dim, K, fn_name):
    from reconciliation.triton.lse import lse4_reduce, lse5_reduce, lse7_reduce
    fn = {'lse4_reduce': lse4_reduce, 'lse5_reduce': lse5_reduce, 'lse7_reduce': lse7_reduce}[fn_name]
    torch.manual_seed(0)
    x = torch.randn(*shape, device='cuda', dtype=torch.float32)
    x = sprinkle_neginf(x, p=0.01)
    y_ref = torch.logsumexp(x, dim=dim)
    y_out = fn(x, dim)
    assert torch.allclose(y_out, y_ref, rtol=1e-6, atol=1e-6)


@CUDA_ONLY
@pytest.mark.parametrize("shape,dim,K,fn_name", [
    ((4, 257), 0, 4, 'lse4_reduce'),
    ((257, 5), 1, 5, 'lse5_reduce'),
    ((7, 17, 9), 0, 7, 'lse7_reduce'),
])
def test_lseK_reduce_fp64(shape, dim, K, fn_name):
    from reconciliation.triton.lse import lse4_reduce, lse5_reduce, lse7_reduce
    fn = {'lse4_reduce': lse4_reduce, 'lse5_reduce': lse5_reduce, 'lse7_reduce': lse7_reduce}[fn_name]
    torch.manual_seed(1)
    x = torch.randn(*shape, device='cuda', dtype=torch.float64)
    x = sprinkle_neginf(x, p=0.01)
    y_ref = torch.logsumexp(x, dim=dim)
    y_out = fn(x, dim)
    assert y_out.dtype == torch.float64
    assert torch.allclose(y_out, y_ref, rtol=1e-12, atol=1e-12)


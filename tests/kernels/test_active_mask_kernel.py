"""Parity tests for fused backward active-mask construction."""

import pytest
import torch

from gpurec.core.kernels.wave_backward import active_mask_from_rhs_absmax_fused


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_active_mask_from_rhs_absmax_matches_pytorch(dtype):
    torch.manual_seed(123)
    device = torch.device("cuda")
    rhs = (torch.randn(17, 513, device=device, dtype=dtype) * 1e-5).contiguous()
    rhs[2].zero_()
    rhs[7, 11] = 1e-4
    rhs[9, 31] = -1e-4

    threshold = 5e-5
    actual = active_mask_from_rhs_absmax_fused(
        rhs, threshold, use_pruning=True
    )
    expected = rhs.abs().max(dim=1).values >= threshold

    torch.testing.assert_close(actual, expected)


def test_active_mask_from_rhs_absmax_zero_threshold_uses_strict_gt():
    device = torch.device("cuda")
    rhs = torch.zeros(4, 257, device=device, dtype=torch.float32)
    rhs[1, 3] = -0.0
    rhs[2, 13] = 1e-12
    rhs[3, 21] = -1e-12

    actual = active_mask_from_rhs_absmax_fused(
        rhs, 0.0, use_pruning=False
    )
    expected = rhs.abs().max(dim=1).values > 0

    torch.testing.assert_close(actual, expected)

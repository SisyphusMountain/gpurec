"""Parity tests for the fused DTS kernel parameter layouts."""

import pytest
import torch

from gpurec.core.kernels.dts_fused import dts_fused
from gpurec.core.log2_utils import logsumexp2


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(Pi, Pibar, lefts, rights, sp_child1, sp_child2, log_pD, log_pS, log_split_probs):
    N, S = lefts.numel(), Pi.shape[1]

    def expand_param(p):
        if p.ndim == 0:
            return p.expand(N, S)
        if p.ndim == 1:
            if p.numel() == S:
                return p.unsqueeze(0).expand(N, S)
            if p.numel() == N:
                return p.unsqueeze(1).expand(N, S)
        if p.ndim == 2 and p.shape == (N, 1):
            return p.expand(N, S)
        if p.ndim == 2 and p.shape == (N, S):
            return p
        raise AssertionError(f"unsupported test parameter shape {tuple(p.shape)}")

    pD = expand_param(log_pD)
    pS = expand_param(log_pS)
    Pi_l = Pi[lefts]
    Pi_r = Pi[rights]
    Pibar_l = Pibar[lefts]
    Pibar_r = Pibar[rights]
    Pi_pad = torch.cat([Pi, torch.full((Pi.shape[0], 1), -float("inf"), device=Pi.device, dtype=Pi.dtype)], dim=1)
    dts = torch.stack(
        [
            pD + Pi_l + Pi_r,
            Pi_l + Pibar_r,
            Pi_r + Pibar_l,
            pS + Pi_pad[lefts][:, sp_child1] + Pi_pad[rights][:, sp_child2],
            pS + Pi_pad[rights][:, sp_child1] + Pi_pad[lefts][:, sp_child2],
        ],
        dim=0,
    )
    return log_split_probs.reshape(N, 1) + logsumexp2(dts, dim=0)


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 2e-5, 2e-5), (torch.float64, 1e-10, 1e-10)])
@pytest.mark.parametrize("layout", ["shared_species", "split_scalar_1d", "split_scalar_2d", "split_species"])
def test_dts_fused_matches_reference_for_param_layouts(dtype, atol, rtol, layout):
    torch.manual_seed(0)
    device = torch.device("cuda")
    C, S, N = 7, 11, 5
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    Pibar = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    lefts = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.long)
    rights = torch.tensor([1, 2, 3, 4, 5], device=device, dtype=torch.long)
    sp_child1 = torch.tensor([1, 3, S, 5, S, 7, S, 9, S, S, S], device=device, dtype=torch.long)
    sp_child2 = torch.tensor([2, 4, S, 6, S, 8, S, 10, S, S, S], device=device, dtype=torch.long)
    log_split_probs = (torch.randn(N, 1, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()

    if layout == "shared_species":
        log_pD = (torch.randn(S, device=device, dtype=dtype) * 0.1 - 4.0).contiguous()
        log_pS = (torch.randn(S, device=device, dtype=dtype) * 0.1 - 4.0).contiguous()
    elif layout == "split_scalar_1d":
        log_pD = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 4.0).contiguous()
        log_pS = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 4.0).contiguous()
    elif layout == "split_scalar_2d":
        log_pD = (torch.randn(N, 1, device=device, dtype=dtype) * 0.1 - 4.0).contiguous()
        log_pS = (torch.randn(N, 1, device=device, dtype=dtype) * 0.1 - 4.0).contiguous()
    else:
        log_pD = (torch.randn(N, S, device=device, dtype=dtype) * 0.1 - 4.0).contiguous()
        log_pS = (torch.randn(N, S, device=device, dtype=dtype) * 0.1 - 4.0).contiguous()

    actual = dts_fused(Pi, Pibar, lefts, rights, sp_child1, sp_child2, log_pD, log_pS, log_split_probs)
    expected = _reference(Pi, Pibar, lefts, rights, sp_child1, sp_child2, log_pD, log_pS, log_split_probs)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 2e-5, 2e-5), (torch.float64, 1e-10, 1e-10)])
def test_dts_fused_active_mask_skips_inactive_parent_rows(dtype, atol, rtol):
    torch.manual_seed(1)
    device = torch.device("cuda")
    C, S, N = 7, 11, 5
    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    Pibar = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    lefts = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.long)
    rights = torch.tensor([1, 2, 3, 4, 5], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 3], device=device, dtype=torch.long)
    active_mask = torch.tensor([True, False, True, False], device=device)
    sp_child1 = torch.tensor([1, 3, S, 5, S, 7, S, 9, S, S, S], device=device, dtype=torch.long)
    sp_child2 = torch.tensor([2, 4, S, 6, S, 8, S, 10, S, S, S], device=device, dtype=torch.long)
    log_split_probs = (torch.randn(N, 1, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()
    log_pD = (torch.randn(S, device=device, dtype=dtype) * 0.1 - 4.0).contiguous()
    log_pS = (torch.randn(S, device=device, dtype=dtype) * 0.1 - 4.0).contiguous()

    actual = dts_fused(
        Pi, Pibar, lefts, rights, sp_child1, sp_child2, log_pD, log_pS, log_split_probs,
        active_mask=active_mask, reduce_idx=reduce_idx,
    )
    expected = _reference(Pi, Pibar, lefts, rights, sp_child1, sp_child2, log_pD, log_pS, log_split_probs)
    inactive = ~active_mask[reduce_idx]
    expected = expected.clone()
    expected[inactive] = -1e30

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

"""Parity tests for direct DTS backward accumulation variants."""

import pytest
import torch

from gpurec.core.kernels.wave_backward import dts_cross_backward_accum_fused


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _run_accum_variant(*, merge_s_term, active_mask=None, dtype=torch.float32):
    torch.manual_seed(17)
    device = torch.device("cuda")
    C, S, W, N = 9, 11, 3, 6
    ws = 6

    Pi = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    Pibar = (torch.randn(C, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    v_k = (torch.randn(W, S, device=device, dtype=dtype) * 0.1).contiguous()

    # Duplicated child rows exercise the atomic accumulation path.
    sl = torch.tensor([0, 1, 2, 3, 1, 4], device=device, dtype=torch.long)
    sr = torch.tensor([1, 2, 3, 4, 0, 2], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0, 1, 1, 2, 2, 0], device=device, dtype=torch.long)
    wlsp = (torch.randn(N, device=device, dtype=dtype) * 0.1 - 1.0).contiguous()

    sp_child1 = torch.tensor(
        [1, 3, S, 5, S, 7, S, 9, S, S, S],
        device=device,
        dtype=torch.long,
    )
    sp_child2 = torch.tensor(
        [2, 4, S, 6, S, 8, S, 10, S, S, S],
        device=device,
        dtype=torch.long,
    )
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    grad_pibar_l, grad_pibar_r, param_pD, param_pS = dts_cross_backward_accum_fused(
        Pi,
        Pibar,
        v_k,
        ws,
        sl,
        sr,
        reduce_idx,
        wlsp,
        torch.tensor(-4.0, device=device, dtype=dtype),
        torch.tensor(-5.0, device=device, dtype=dtype),
        sp_child1,
        sp_child2,
        accumulated_rhs,
        S,
        active_mask=active_mask,
        merge_s_term=merge_s_term,
    )
    torch.cuda.synchronize()
    return accumulated_rhs, grad_pibar_l, grad_pibar_r, param_pD, param_pS


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 2e-5, 2e-5), (torch.float64, 1e-10, 1e-10)])
def test_dts_backward_accum_merged_matches_two_loop(dtype, atol, rtol):
    old = _run_accum_variant(merge_s_term=False, dtype=dtype)
    merged = _run_accum_variant(merge_s_term=True, dtype=dtype)

    for actual, expected in zip(merged, old):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_dts_backward_accum_merged_matches_two_loop_with_active_mask():
    device = torch.device("cuda")
    active_mask = torch.tensor([True, False, True], device=device)

    old = _run_accum_variant(merge_s_term=False, active_mask=active_mask)
    merged = _run_accum_variant(merge_s_term=True, active_mask=active_mask)

    for actual, expected in zip(merged, old):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

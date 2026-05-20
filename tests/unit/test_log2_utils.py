from __future__ import annotations

import math

import torch

from gpurec.core.log2_utils import _safe_log2_internal, log2_softmax, logsumexp2


def test_safe_log2_returns_negative_infinity_with_zero_gradient_for_nonpositive():
    values = torch.tensor([-2.0, 0.0, 0.25, 4.0], dtype=torch.float64, requires_grad=True)

    result = _safe_log2_internal(values)
    torch.autograd.backward(result, torch.ones_like(result))

    assert torch.isneginf(result[:2]).all()
    torch.testing.assert_close(result[2:], torch.tensor([-2.0, 2.0], dtype=torch.float64))
    expected_grad = torch.tensor(
        [0.0, 0.0, 1.0 / (0.25 * math.log(2.0)), 1.0 / (4.0 * math.log(2.0))],
        dtype=torch.float64,
    )
    torch.testing.assert_close(values.grad, expected_grad)


def test_logsumexp2_matches_natural_log_reference_and_keeps_all_inf_slices():
    values = torch.tensor(
        [
            [0.0, -1.0, -float("inf")],
            [-float("inf"), -float("inf"), -float("inf")],
        ],
        dtype=torch.float64,
    )

    result = logsumexp2(values, dim=1)
    keepdim_result = logsumexp2(values, dim=1, keepdim=True)
    reference = torch.logsumexp(values * math.log(2.0), dim=1) / math.log(2.0)

    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(keepdim_result, reference[:, None])
    assert keepdim_result.shape == (2, 1)
    assert torch.isneginf(result[1])
    assert not torch.isnan(result).any()


def test_log2_softmax_matches_reference_and_normalizes_probabilities():
    values = torch.tensor(
        [[0.0, -1.5, -4.0], [2.0, 0.0, -3.0]],
        dtype=torch.float64,
    )

    result = log2_softmax(values, dim=1)
    reference = torch.log_softmax(values * math.log(2.0), dim=1) / math.log(2.0)

    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(
        torch.exp2(result).sum(dim=1),
        torch.ones(2, dtype=torch.float64),
    )


def test_log2_softmax_backward_is_finite_for_extreme_float32_inputs():
    values = torch.tensor([[0.0, -33.0, -100.0]], dtype=torch.float32, requires_grad=True)

    result = log2_softmax(values, dim=1)
    result.sum().backward()

    assert torch.isfinite(result).all()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()

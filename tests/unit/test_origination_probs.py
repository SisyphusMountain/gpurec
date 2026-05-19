import torch

from gpurec.core.likelihood import (
    compute_log_likelihood,
    compute_log_likelihood_root_rows,
)


def _manual_weighted_nll(root_rows, E, weights):
    weights = weights / weights.sum(dim=-1, keepdim=True)
    numerator = torch.log2((weights * torch.exp2(root_rows)).sum(dim=-1))
    denominator = torch.log2((weights * (1.0 - torch.exp2(E))).sum(dim=-1))
    return -(numerator - denominator)


def test_weighted_origination_likelihood_matches_manual_formula():
    dtype = torch.float64
    Pi = torch.tensor(
        [
            [-4.0, -2.0, -5.0],
            [-1.5, -3.0, -2.5],
            [-3.0, -1.0, -4.5],
            [-2.0, -2.5, -3.5],
        ],
        dtype=dtype,
    )
    roots = torch.tensor([1, 3], dtype=torch.long)
    E = torch.tensor([-3.0, -2.0, -4.0], dtype=dtype)
    weights = torch.tensor([0.15, 0.60, 0.25], dtype=dtype)

    expected = _manual_weighted_nll(Pi[roots], E, weights)
    actual = compute_log_likelihood(Pi, E, roots, origination_probs=weights)
    root_actual = compute_log_likelihood_root_rows(Pi[roots], E, origination_probs=weights)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(root_actual, expected, rtol=1e-12, atol=1e-12)


def test_family_specific_origination_likelihood_matches_manual_formula():
    dtype = torch.float64
    root_rows = torch.tensor(
        [
            [-4.0, -2.0, -5.0],
            [-2.0, -2.5, -3.5],
        ],
        dtype=dtype,
    )
    E = torch.tensor(
        [
            [-3.0, -2.0, -4.0],
            [-2.5, -2.25, -3.25],
        ],
        dtype=dtype,
    )
    weights = torch.tensor(
        [
            [0.15, 0.60, 0.25],
            [3.0, 1.0, 2.0],
        ],
        dtype=dtype,
    )

    expected = _manual_weighted_nll(root_rows, E, weights)
    actual = compute_log_likelihood_root_rows(root_rows, E, origination_probs=weights)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_uniform_origination_probs_preserve_default_likelihood():
    dtype = torch.float64
    Pi = torch.tensor(
        [
            [-4.0, -2.0, -5.0],
            [-1.5, -3.0, -2.5],
            [-3.0, -1.0, -4.5],
        ],
        dtype=dtype,
    )
    roots = torch.tensor([0, 2], dtype=torch.long)
    E = torch.tensor([-3.0, -2.0, -4.0], dtype=dtype)
    uniform = torch.ones(3, dtype=dtype)

    default = compute_log_likelihood(Pi, E, roots)
    weighted = compute_log_likelihood(Pi, E, roots, origination_probs=uniform)
    root_weighted = compute_log_likelihood_root_rows(
        Pi[roots],
        E,
        origination_probs=uniform,
    )

    torch.testing.assert_close(weighted, default, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(root_weighted, default, rtol=1e-12, atol=1e-12)

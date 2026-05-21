import torch
import pytest

from gpurec.optimization.implicit_grad import _bicgstab


def test_bicgstab_solves_nonsymmetric_system():
    A = torch.tensor(
        [
            [3.0, 2.0, 0.0],
            [0.0, 4.0, 1.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

    x, stats = _bicgstab(lambda v: A @ v, b, tol=1e-12, maxiter=20)

    torch.testing.assert_close(A @ x, b, rtol=1e-10, atol=1e-12)
    assert stats.method == "BiCGSTAB"
    assert stats.success
    assert stats.rel_res <= 1e-10


def test_bicgstab_nonconvergence_returns_iterate_and_failure_stats():
    b = torch.tensor([1.0, -2.0], dtype=torch.float64)

    x, stats = _bicgstab(lambda v: torch.zeros_like(v), b, tol=1e-12, maxiter=5)

    torch.testing.assert_close(x, torch.zeros_like(b))
    assert stats.method == "BiCGSTAB"
    assert not stats.success
    assert stats.iters == 0
    assert stats.rel_res == pytest.approx(1.0)

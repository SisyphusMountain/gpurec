"""Robustness contract for the E-adjoint GMRES fallback (``_gmres_e_adjoint``).

Pure linear-algebra tests -- no CUDA / triton / model needed.

``_neumann_e_adjoint`` sums the Neumann series ``sum_k J^k b`` for ``(I - J) x = b``, which needs
``J`` to shrink a vector. When it does not, the series is the wrong method and no iteration budget
saves it, so ``_neumann_e_adjoint`` hands the same matvec to this function instead: GMRES
minimizes the residual over the Krylov subspace and cannot break down on a nonsingular operator.

These tests pin:
  * it solves a non-contracting operator that the series cannot touch, from a zero start and from
    a warm start,
  * it reaches the residual it is asked for, in float64 and in float32,
  * it reports the residual it actually achieved rather than raising -- the caller decides,
  * a zero right-hand side returns zero,
  * an exhausted matvec budget comes back with the honest (large) residual,
  * the answer does not depend on the initial guess.
"""
import pytest
import torch

try:
    from gpurec.api._implicit_grad import _gmres_e_adjoint
except Exception as exc:  # pragma: no cover - import guard for triton-less envs
    pytest.skip(f"gpurec.api._implicit_grad unavailable: {exc}", allow_module_level=True)


def _non_contracting_banded(n: int, dtype: torch.dtype) -> torch.Tensor:
    """``A = I - J`` with a banded J of spectral radius 1.64 -- not a contraction, so the Neumann
    series diverges on it, but with a tightly clustered spectrum that leaves ``A`` well
    conditioned. That is the shape the real E-adjoint operator has."""
    diagonal = torch.full((n,), -1.2, dtype=torch.float64)
    off = torch.full((n - 1,), 0.4, dtype=torch.float64)
    J = torch.diag(diagonal) + torch.diag(off, 1) + torch.diag(0.3 * off, -1)
    assert float(torch.linalg.eigvals(J).abs().max()) > 1.0
    return (torch.eye(n, dtype=torch.float64) - J).to(dtype)


def _matvec(A):
    return lambda v: A @ v


def _rel_res(A, x, b):
    return float(torch.linalg.vector_norm((A @ x - b).double())
                 / torch.linalg.vector_norm(b.double()))


def test_solves_a_non_contracting_operator_from_zero():
    A = _non_contracting_banded(192, torch.float64)
    b = A @ torch.randn(192, dtype=torch.float64, generator=torch.Generator().manual_seed(24))
    x, residual = _gmres_e_adjoint(
        _matvec(A), b, x0=torch.zeros_like(b), tol=1e-12, max_matvecs=400, restart=48,
    )
    assert residual <= 1e-12
    assert _rel_res(A, x, b) <= 1e-12
    direct = torch.linalg.solve(A, b)
    assert float(torch.linalg.vector_norm(x - direct)
                 / torch.linalg.vector_norm(direct)) < 1e-9


def test_warm_start_reaches_the_same_answer():
    """The Neumann iterate it is warm-started from must not steer the answer."""
    A = _non_contracting_banded(192, torch.float64)
    b = A @ torch.randn(192, dtype=torch.float64, generator=torch.Generator().manual_seed(25))
    direct = torch.linalg.solve(A, b)
    guesses = (
        torch.zeros_like(b),
        b.clone(),
        1e6 * torch.randn(192, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(26)),
    )
    for guess in guesses:
        x, residual = _gmres_e_adjoint(
            _matvec(A), b, x0=guess, tol=1e-12, max_matvecs=600, restart=48,
        )
        assert residual <= 1e-12
        assert float(torch.linalg.vector_norm(x - direct)
                     / torch.linalg.vector_norm(direct)) < 1e-9


def test_fp32_matvec_still_reaches_the_fp32_floor():
    """Arnoldi runs in float64 even when the matvec is float32, so the residual floor is the
    matvec's precision and not the orthogonalization's."""
    A = _non_contracting_banded(192, torch.float32)
    b = A @ torch.randn(192, dtype=torch.float32, generator=torch.Generator().manual_seed(27))
    x, residual = _gmres_e_adjoint(
        _matvec(A), b, x0=torch.zeros_like(b), tol=1e-6, max_matvecs=400, restart=48,
    )
    assert x.dtype == torch.float32
    assert residual <= 1e-6


def test_zero_rhs_returns_zero():
    A = _non_contracting_banded(64, torch.float64)
    b = torch.zeros(64, dtype=torch.float64)
    x, residual = _gmres_e_adjoint(
        _matvec(A), b, x0=torch.ones_like(b), tol=1e-12, max_matvecs=64, restart=16,
    )
    assert float(torch.linalg.vector_norm(x)) == 0.0
    assert residual == 0.0


def test_exhausted_budget_reports_its_residual_instead_of_raising():
    A = _non_contracting_banded(192, torch.float64)
    b = A @ torch.randn(192, dtype=torch.float64, generator=torch.Generator().manual_seed(28))
    _, residual = _gmres_e_adjoint(
        _matvec(A), b, x0=torch.zeros_like(b), tol=1e-12, max_matvecs=3, restart=48,
    )
    assert residual > 1e-12          # honest: three matvecs cannot solve it
    assert residual < float("inf")


def test_rejects_nonsense_budgets():
    A = _non_contracting_banded(32, torch.float64)
    b = torch.randn(32, dtype=torch.float64, generator=torch.Generator().manual_seed(29))
    with pytest.raises(ValueError):
        _gmres_e_adjoint(_matvec(A), b, x0=torch.zeros_like(b), tol=1e-12,
                         max_matvecs=0, restart=8)
    with pytest.raises(ValueError):
        _gmres_e_adjoint(_matvec(A), b, x0=torch.zeros_like(b), tol=1e-12,
                         max_matvecs=8, restart=0)

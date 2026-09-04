"""Validation for the E-adjoint linear solver (``_neumann_e_adjoint``).

The backward E-adjoint solves ``(I - J) wE = q`` where ``J = d(E_from_E)/dE`` is the E-step
self-map Jacobian. When the forward extinction fixed point is the one the map actually has, ``J``
is a contraction (spectral radius below 1) and the Neumann series ``(I-J)^{-1} = sum_k J^k``
converges -- with no orthogonalization step, so no float32 Arnoldi-style residual floor.

The series is not, however, unconditionally safe: a caller that hands over an operator that grows
vectors gets terms that grow too. ``_neumann_e_adjoint`` detects that (the term norm stops
improving) and falls back on a solver that needs no contraction -- a direct factorization for a
system small enough to assemble, restarted GMRES otherwise.

This file pins:
  1. the series converges on a contraction and matches a direct solve,
  2. the dtype-matched residual contract: float32 default, tight float64, sub-floor ``tol``
     warns and is raised to the floor, ``tol=None`` warns not at all,
  3. a zero right-hand side returns zero,
  4. a NON-contracting operator no longer raises: the fallback returns the true solution, on both
     the small (direct factorization) and the large (GMRES) side of the size threshold
     (``_gmres_e_adjoint`` itself is pinned in test_gmres_e_adjoint.py),
  5. an operator so far from solvable that the fallback cannot reach the floor within its matvec
     budget still raises, naming both stages.
"""
import warnings

import pytest
import torch

try:
    from gpurec.api._implicit_grad import (
        _E_ADJOINT_DENSE_SOLVE_LIMIT,
        _e_adjoint_rel_tol_default,
        _neumann_e_adjoint,
    )
except Exception as exc:  # pragma: no cover - import guard for triton-less envs
    pytest.skip(f"gpurec.api._implicit_grad unavailable: {exc}", allow_module_level=True)


def _e_adjoint_like(n: int, rho: float, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """``A = I - J`` with ``||J||_2 = rho`` and a NON-symmetric J.

    ``rho`` below 1 is the healthy E-adjoint regime (the real operator measures about 0.05 on a
    2013-species tree): eigenvalues clustered near 1, small condition number, but non-symmetric.
    ``rho`` above 1 is the regime the fallback exists for.
    """
    generator = torch.Generator().manual_seed(seed)
    J = torch.randn(n, n, generator=generator, dtype=torch.float64)
    J = J - torch.diag(torch.diagonal(J))                      # zero the diagonal for asymmetry
    J = rho * J / float(torch.linalg.matrix_norm(J, ord=2))    # scale so ||J||_2 = rho
    return (torch.eye(n, dtype=torch.float64) - J).to(dtype)


def _non_contracting_banded(n: int) -> torch.Tensor:
    """``A = I - J`` with a banded J whose spectral radius is 1.64 but whose spectrum is a tight
    cluster, so ``A`` stays well conditioned -- the shape the real E-adjoint has when it is not a
    contraction, and the shape a Krylov method is meant for."""
    diagonal = torch.full((n,), -1.2, dtype=torch.float64)
    off = torch.full((n - 1,), 0.4, dtype=torch.float64)
    J = torch.diag(diagonal) + torch.diag(off, 1) + torch.diag(0.3 * off, -1)
    assert float(torch.linalg.eigvals(J).abs().max()) > 1.0    # genuinely not a contraction
    return torch.eye(n, dtype=torch.float64) - J


def _matvec(A):
    return lambda v: A @ v


def _rel_res(A, x, b):
    return float(torch.linalg.vector_norm(A @ x - b) / torch.linalg.vector_norm(b))


# ----------------------------------------------------------------------------
# 1. converges on a contraction and matches a direct solve
# ----------------------------------------------------------------------------

def test_matches_direct_solve_on_a_contraction():
    for dtype, bar in ((torch.float32, 5e-6), (torch.float64, 1e-10)):
        A = _e_adjoint_like(71, rho=0.24, dtype=dtype, seed=7)
        assert float(torch.linalg.cond(A.double())) < 2.0   # ~1.6, like the real operator
        b = A @ torch.randn(71, dtype=dtype, generator=torch.Generator().manual_seed(8))

        x = _neumann_e_adjoint(_matvec(A), b)
        direct = torch.linalg.solve(A.double(), b.double())

        assert _rel_res(A, x, b) < bar
        relative = float(
            torch.linalg.vector_norm(x.double() - direct)
            / torch.linalg.vector_norm(direct)
        )
        assert relative < 10 * bar, f"dtype={dtype}: solver vs direct solve rel={relative:.3e}"


# ----------------------------------------------------------------------------
# 2. the dtype-matched residual contract
# ----------------------------------------------------------------------------

def test_fp32_default_residual():
    A = _e_adjoint_like(64, rho=0.3, dtype=torch.float32, seed=11)
    b = A @ torch.randn(64, dtype=torch.float32, generator=torch.Generator().manual_seed(1))
    x = _neumann_e_adjoint(_matvec(A), b)
    assert _rel_res(A, x, b) <= _e_adjoint_rel_tol_default(torch.float32)


def test_fp64_reaches_tight_residual():
    A = _e_adjoint_like(48, rho=0.2, dtype=torch.float64, seed=12)
    exact = torch.randn(48, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    b = A @ exact
    x = _neumann_e_adjoint(_matvec(A), b, tol=1e-12)
    assert _rel_res(A, x, b) <= 1e-10
    assert float(torch.linalg.vector_norm(x - exact) / torch.linalg.vector_norm(exact)) < 1e-8


def test_subfloor_tol_warns_and_is_raised_to_the_floor():
    A = _e_adjoint_like(48, rho=0.2, dtype=torch.float32, seed=13)
    b = A @ torch.randn(48, dtype=torch.float32, generator=torch.Generator().manual_seed(3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = _neumann_e_adjoint(_matvec(A), b, tol=1e-12)     # unreachable in float32
    assert any("below the" in str(entry.message) for entry in caught)
    assert _rel_res(A, x, b) < 5e-6


def test_none_tol_uses_dtype_default_no_warning():
    A = _e_adjoint_like(48, rho=0.2, dtype=torch.float32, seed=14)
    b = A @ torch.randn(48, dtype=torch.float32, generator=torch.Generator().manual_seed(5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                        # any warning fails the test
        x = _neumann_e_adjoint(_matvec(A), b, tol=None)
    assert _rel_res(A, x, b) <= _e_adjoint_rel_tol_default(torch.float32)


# ----------------------------------------------------------------------------
# 3. zero right-hand side
# ----------------------------------------------------------------------------

def test_zero_rhs_returns_zero():
    A = _e_adjoint_like(16, rho=0.1, dtype=torch.float64, seed=15)
    b = torch.zeros(16, dtype=torch.float64)
    x = _neumann_e_adjoint(_matvec(A), b)
    assert float(torch.linalg.vector_norm(x)) == 0.0


# ----------------------------------------------------------------------------
# 4. a non-contracting operator falls back instead of raising
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [1.5, 4.0])
def test_non_contraction_falls_back_to_a_direct_solve(rho):
    """Below the assemble-and-factorize size the fallback is exact, whatever the growth rate."""
    n = 32
    assert n <= _E_ADJOINT_DENSE_SOLVE_LIMIT
    A = _e_adjoint_like(n, rho=rho, dtype=torch.float64, seed=16)
    b = A @ torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    x = _neumann_e_adjoint(_matvec(A), b, max_iter=64)
    direct = torch.linalg.solve(A, b)
    assert _rel_res(A, x, b) < 1e-12
    relative = float(torch.linalg.vector_norm(x - direct) / torch.linalg.vector_norm(direct))
    assert relative < 1e-10, f"rho={rho}: fallback vs direct solve rel={relative:.3e}"


def test_non_contraction_above_the_dense_limit_uses_gmres():
    """Above the assemble-and-factorize size the fallback is GMRES on the same matvec.

    The operator is built with a sparse (tridiagonal-plus-transfer) J so that GMRES converges in
    far fewer than ``n`` Krylov vectors, which is the structure the real E-adjoint has; a dense
    random J of this size needs the whole space and no restarted method would finish.
    """
    n = _E_ADJOINT_DENSE_SOLVE_LIMIT + 128
    generator = torch.Generator().manual_seed(21)
    A = _non_contracting_banded(n)
    b = A @ torch.randn(n, dtype=torch.float64, generator=generator)

    x = _neumann_e_adjoint(_matvec(A), b, max_iter=400)
    direct = torch.linalg.solve(A, b)
    assert _rel_res(A, x, b) <= 1e-12
    relative = float(torch.linalg.vector_norm(x - direct) / torch.linalg.vector_norm(direct))
    assert relative < 1e-9, f"fallback vs direct solve rel={relative:.3e}"


# ----------------------------------------------------------------------------
# 5. fail-loud when neither stage can reach the floor
# ----------------------------------------------------------------------------

def test_still_raises_when_the_budget_cannot_solve_it():
    """A dense non-contracting operator too big to assemble, with a four-matvec budget.

    Below ``_E_ADJOINT_DENSE_SOLVE_LIMIT`` the fallback assembles and factorizes the operator, so
    it always succeeds and the budget never bites; above it the budget is GMRES's matvec count,
    and four Krylov vectors cannot solve a dense 640-wide system to 1e-12.
    """
    n = _E_ADJOINT_DENSE_SOLVE_LIMIT + 128
    A = _e_adjoint_like(n, rho=3.0, dtype=torch.float64, seed=17)
    b = A @ torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(18))
    with pytest.raises(RuntimeError, match="GMRES fallback"):
        _neumann_e_adjoint(_matvec(A), b, max_iter=4)

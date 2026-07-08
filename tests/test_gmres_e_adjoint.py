"""Robustness contract for the GMRES E-adjoint solver (``_gmres``).

Pure linear-algebra tests -- no CUDA / triton / model needed. GMRES replaced
BiCGSTAB as the E-adjoint / GGN linear solver because BiCGSTAB can break down
(lose bi-orthogonality) on a *well-conditioned* but moderately non-symmetric
operator ``A = I - J_E^T`` and then raise a spurious "singular/ill-conditioned"
error. GMRES minimizes the residual over the Krylov subspace and cannot break
down on a nonsingular operator, so these tests pin:

  * it solves the E-adjoint-shaped non-symmetric operator (``I - J`` with
    spectral radius ~0.24, cond ~1.6 -- exactly the case that trips BiCGSTAB),
  * it reaches the dtype-matched default residual (fp32 1e-6 / fp64 1e-12),
  * a sub-floor ``tol`` is clamped up with a warning,
  * a genuinely under-resourced solve (too few iterations for an
    ill-conditioned system) still raises (fail-loud),
  * ``tol=None`` converges to the dtype default without warning.
"""
import warnings

import pytest
import torch

try:
    from gpurec.api._implicit_grad import (
        _gmres,
        _bicgstab_rel_tol_default,
        _bicgstab_rel_tol_floor,
    )
except Exception as exc:  # pragma: no cover - import guard for triton-less envs
    pytest.skip(f"gpurec.api._implicit_grad unavailable: {exc}", allow_module_level=True)


def _spd(n: int, cond: float, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """Symmetric positive-definite operator with a target condition number."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    eig = torch.logspace(0, float(torch.log10(torch.tensor(float(cond)))), n, dtype=torch.float64)
    A = (Q * eig) @ Q.T
    return (0.5 * (A + A.T)).to(dtype)


def _e_adjoint_like(n: int, rho: float, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """``A = I - J`` with ``spectral_radius(J) = rho`` and a NON-symmetric J.

    This mirrors the real E-adjoint ``I - J_E^T`` (rho(J_E) ~ 0.24): eigenvalues
    clustered near 1, tiny condition number, but decidedly non-symmetric -- the
    regime where BiCGSTAB breaks down and GMRES does not.
    """
    g = torch.Generator().manual_seed(seed)
    J = torch.randn(n, n, generator=g, dtype=torch.float64)
    J = J - torch.diag(torch.diagonal(J))          # zero the diagonal for asymmetry
    J = rho * J / float(torch.linalg.matrix_norm(J, ord=2))  # scale so ||J||_2 = rho
    A = torch.eye(n, dtype=torch.float64) - J
    return A.to(dtype)


def _matvec(A):
    return lambda v: A @ v


def _rel_res(A, x, b):
    return float(torch.linalg.vector_norm(A @ x - b) / torch.linalg.vector_norm(b))


def test_e_adjoint_like_nonsymmetric_solves():
    """The BiCGSTAB-breakdown regime: well-conditioned, non-symmetric ``I - J``."""
    for dtype, bar in ((torch.float32, 5e-6), (torch.float64, 1e-10)):
        A = _e_adjoint_like(71, rho=0.24, dtype=dtype, seed=7)
        assert float(torch.linalg.cond(A.double())) < 2.0  # ~1.6, like the real operator
        b = A @ torch.randn(71, dtype=dtype, generator=torch.Generator().manual_seed(8))
        x = _gmres(_matvec(A), b)
        assert _rel_res(A, x, b) < bar


def test_fp32_default_residual():
    A = _spd(64, cond=50.0, dtype=torch.float32)
    b = A @ torch.randn(64, dtype=torch.float32, generator=torch.Generator().manual_seed(1))
    x = _gmres(_matvec(A), b)
    assert _rel_res(A, x, b) <= _bicgstab_rel_tol_default(torch.float32)


def test_fp64_reaches_tight_residual():
    A = _spd(48, cond=1e3, dtype=torch.float64)
    xtrue = torch.randn(48, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    b = A @ xtrue
    x = _gmres(_matvec(A), b, tol=1e-12)
    assert _rel_res(A, x, b) <= 1e-10
    assert float(torch.linalg.vector_norm(x - xtrue) / torch.linalg.vector_norm(xtrue)) < 1e-8


def test_subfloor_tol_warns_and_clamps():
    A = _spd(48, cond=1e3, dtype=torch.float32)
    b = A @ torch.randn(48, dtype=torch.float32, generator=torch.Generator().manual_seed(3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = _gmres(_matvec(A), b, tol=1e-12)  # unreachable in fp32
    assert any("below the" in str(wi.message) for wi in w)
    assert _rel_res(A, x, b) < 5e-6


def test_genuine_nonconvergence_still_raises():
    A = _spd(64, cond=1e8, dtype=torch.float64)
    b = A @ torch.randn(64, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    with pytest.raises(RuntimeError):
        _gmres(_matvec(A), b, tol=1e-14, max_iter=2)  # 2 Krylov vectors can't reach 1e-14


def test_none_tol_uses_dtype_default_no_warning():
    """tol=None must converge to the dtype default without warning."""
    A = _spd(48, cond=1e3, dtype=torch.float32)
    b = A @ torch.randn(48, dtype=torch.float32, generator=torch.Generator().manual_seed(5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        x = _gmres(_matvec(A), b, tol=None)
    assert _rel_res(A, x, b) <= _bicgstab_rel_tol_default(torch.float32)


def test_zero_rhs_returns_zero():
    A = _spd(16, cond=10.0, dtype=torch.float64)
    b = torch.zeros(16, dtype=torch.float64)
    x = _gmres(_matvec(A), b)
    assert float(torch.linalg.vector_norm(x)) == 0.0

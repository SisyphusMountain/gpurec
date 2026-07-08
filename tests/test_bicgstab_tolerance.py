"""Dtype-aware, fail-safe BiCGSTAB tolerance behaviour.

These are pure linear-algebra tests of ``gpurec.api._implicit_grad._bicgstab`` --
no CUDA / triton / model needed -- pinning the robustness contract introduced in
``docs/optim/solver_tolerance_robustness.md``:

  * the historical fp32 crash mode (``tol=1e-7`` is 0.84x fp32 eps) now returns a
    converged solution instead of raising,
  * a sub-floor ``tol`` is clamped up with a warning,
  * a genuinely unconverged solve still raises (fail-loud),
  * fp64 solves still reach a tight residual.
"""
import warnings

import pytest
import torch

try:
    from gpurec.api._implicit_grad import (
        _bicgstab,
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


def _matvec(A):
    return lambda v: A @ v


def _rel_res(A, x, b):
    return float(torch.linalg.vector_norm(A @ x - b) / torch.linalg.vector_norm(b))


def test_old_fp32_crash_mode_now_returns():
    """Ill-conditioned fp32 @ tol=1e-7 used to RuntimeError at ~1.3e-7; now returns.

    ``max_iter`` is pinned explicitly here: this synthetic cond=3e4 matrix needs more
    Krylov steps than the production ``SolverOptions().bicgstab_max_iter`` (128, tuned
    for the well-conditioned E-adjoint that ``_bicgstab``'s default now falls back to)
    to reach the fp32 floor -- the test is about the tolerance-clamping contract, not
    about the choice of default iteration cap.
    """
    A = _spd(64, cond=3e4, dtype=torch.float32)
    b = A @ torch.randn(64, dtype=torch.float32, generator=torch.Generator().manual_seed(1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = _bicgstab(_matvec(A), b, tol=1e-7, max_iter=500)
    assert _rel_res(A, x, b) < 5e-6


def test_fp64_reaches_tight_residual():
    A = _spd(48, cond=1e3, dtype=torch.float64)
    xtrue = torch.randn(48, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    b = A @ xtrue
    x = _bicgstab(_matvec(A), b, tol=1e-12)
    assert _rel_res(A, x, b) <= 1e-10
    assert float(torch.linalg.vector_norm(x - xtrue) / torch.linalg.vector_norm(xtrue)) < 1e-8


def test_subfloor_tol_warns_and_clamps():
    A = _spd(48, cond=1e3, dtype=torch.float32)
    b = A @ torch.randn(48, dtype=torch.float32, generator=torch.Generator().manual_seed(3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = _bicgstab(_matvec(A), b, tol=1e-12)  # unreachable in fp32
    assert any("below the" in str(wi.message) for wi in w)
    assert _rel_res(A, x, b) < 5e-6


def test_genuine_nonconvergence_still_raises():
    A = _spd(64, cond=1e6, dtype=torch.float64)
    b = A @ torch.randn(64, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    with pytest.raises(RuntimeError):
        _bicgstab(_matvec(A), b, tol=1e-14, max_iter=2)  # 2 iters can't reach 1e-14


def test_defaults_are_dtype_matched_and_above_floor():
    assert _bicgstab_rel_tol_default(torch.float32) == 1e-6
    assert _bicgstab_rel_tol_default(torch.float64) == 1e-12
    assert _bicgstab_rel_tol_default(torch.float32) > _bicgstab_rel_tol_floor(torch.float32)
    assert _bicgstab_rel_tol_default(torch.float64) > _bicgstab_rel_tol_floor(torch.float64)


def test_none_tol_uses_dtype_default():
    """tol=None must converge to the dtype default without warning."""
    A = _spd(48, cond=1e3, dtype=torch.float32)
    b = A @ torch.randn(48, dtype=torch.float32, generator=torch.Generator().manual_seed(5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        x = _bicgstab(_matvec(A), b, tol=None)
    assert _rel_res(A, x, b) <= _bicgstab_rel_tol_default(torch.float32)

"""Validation for the Neumann-series E-adjoint solver (``_neumann_e_adjoint``).

The backward E-adjoint solves ``(I - J) wE = q`` where ``J = d(E_from_E)/dE`` is the E-step
self-map Jacobian. The forward E fixed point converges, so ``J`` is a contraction (spectral
radius < 1) and the Neumann series ``(I-J)^{-1} = sum_k J^k`` converges. GMRES (the existing
solver, ``_gmres``) has a theta-dependent fp32 residual floor from Arnoldi orthogonalization
(~1e-6 to 5.5e-6) that makes it fail mid-optimization at large species counts; Neumann has no
orthogonalization step, so it has no such floor.

This file pins:
  1. solver-level correctness: ``_neumann_e_adjoint`` converges on a contraction operator
     ``A = I - J`` and matches both ``_gmres`` and a direct dense solve to tight tolerance,
  2. the key correctness check -- gradient parity: on a small model where GMRES converges, the
     Neumann-computed theta/receiver gradient matches the GMRES-computed gradient to tight
     relative tolerance (both solve the SAME linear system, so a mismatch indicates a bug),
  3. the default is unchanged: ``SolverOptions().e_adjoint_solver == "gmres"`` and invalid
     values are rejected.
"""
import warnings

import pytest
import torch

try:
    from gpurec.api._implicit_grad import (
        _gmres,
        _neumann_e_adjoint,
        _bicgstab_rel_tol_default,
        _bicgstab_rel_tol_floor,
    )
except Exception as exc:  # pragma: no cover - import guard for triton-less envs
    pytest.skip(f"gpurec.api._implicit_grad unavailable: {exc}", allow_module_level=True)

from gpurec.api.solver_options import SolverOptions


def _e_adjoint_like(n: int, rho: float, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """``A = I - J`` with ``spectral_radius(J) = rho`` and a NON-symmetric J.

    Mirrors the helper in ``test_gmres_e_adjoint.py``: this is the regime of the real E-adjoint
    ``I - J_E^T`` (rho(J_E) ~ 0.24) -- eigenvalues clustered near 1, small condition number, but
    non-symmetric.
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


# ----------------------------------------------------------------------------
# 1. Solver-level: converges on a contraction operator, matches GMRES + direct solve.
# ----------------------------------------------------------------------------

def test_neumann_matches_gmres_and_direct_solve():
    for dtype, bar in ((torch.float32, 5e-6), (torch.float64, 1e-10)):
        A = _e_adjoint_like(71, rho=0.24, dtype=dtype, seed=7)
        assert float(torch.linalg.cond(A.double())) < 2.0  # ~1.6, like the real operator
        b = A @ torch.randn(71, dtype=dtype, generator=torch.Generator().manual_seed(8))

        x_neumann = _neumann_e_adjoint(_matvec(A), b)
        x_gmres = _gmres(_matvec(A), b)
        x_direct = torch.linalg.solve(A.double(), b.double())

        assert _rel_res(A, x_neumann, b) < bar
        rel_vs_gmres = float(
            torch.linalg.vector_norm(x_neumann.double() - x_gmres.double())
            / torch.linalg.vector_norm(x_gmres.double())
        )
        rel_vs_direct = float(
            torch.linalg.vector_norm(x_neumann.double() - x_direct)
            / torch.linalg.vector_norm(x_direct)
        )
        assert rel_vs_gmres < 10 * bar, f"dtype={dtype}: neumann vs gmres rel={rel_vs_gmres:.3e}"
        assert rel_vs_direct < 10 * bar, f"dtype={dtype}: neumann vs direct rel={rel_vs_direct:.3e}"


def test_fp32_default_residual():
    A = _e_adjoint_like(64, rho=0.3, dtype=torch.float32, seed=11)
    b = A @ torch.randn(64, dtype=torch.float32, generator=torch.Generator().manual_seed(1))
    x = _neumann_e_adjoint(_matvec(A), b)
    assert _rel_res(A, x, b) <= _bicgstab_rel_tol_default(torch.float32)


def test_fp64_reaches_tight_residual():
    A = _e_adjoint_like(48, rho=0.2, dtype=torch.float64, seed=12)
    xtrue = torch.randn(48, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    b = A @ xtrue
    x = _neumann_e_adjoint(_matvec(A), b, tol=1e-12)
    assert _rel_res(A, x, b) <= 1e-10
    assert float(torch.linalg.vector_norm(x - xtrue) / torch.linalg.vector_norm(xtrue)) < 1e-8


def test_subfloor_tol_warns_and_clamps():
    A = _e_adjoint_like(48, rho=0.2, dtype=torch.float32, seed=13)
    b = A @ torch.randn(48, dtype=torch.float32, generator=torch.Generator().manual_seed(3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = _neumann_e_adjoint(_matvec(A), b, tol=1e-12)  # unreachable in fp32
    assert any("below the" in str(wi.message) for wi in w)
    assert _rel_res(A, x, b) < 5e-6


def test_none_tol_uses_dtype_default_no_warning():
    A = _e_adjoint_like(48, rho=0.2, dtype=torch.float32, seed=14)
    b = A @ torch.randn(48, dtype=torch.float32, generator=torch.Generator().manual_seed(5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        x = _neumann_e_adjoint(_matvec(A), b, tol=None)
    assert _rel_res(A, x, b) <= _bicgstab_rel_tol_default(torch.float32)


def test_zero_rhs_returns_zero():
    A = _e_adjoint_like(16, rho=0.1, dtype=torch.float64, seed=15)
    b = torch.zeros(16, dtype=torch.float64)
    x = _neumann_e_adjoint(_matvec(A), b)
    assert float(torch.linalg.vector_norm(x)) == 0.0


def test_non_contraction_raises():
    """rho >= 1 -> not a contraction -> the Neumann series must not silently converge."""
    A = _e_adjoint_like(32, rho=1.5, dtype=torch.float64, seed=16)
    b = A @ torch.randn(32, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    with pytest.raises(RuntimeError):
        _neumann_e_adjoint(_matvec(A), b, max_iter=64)


# ----------------------------------------------------------------------------
# 2. SolverOptions: default unchanged, validation.
# ----------------------------------------------------------------------------

def test_solver_options_default_is_gmres():
    assert SolverOptions().e_adjoint_solver == "gmres"


def test_solver_options_accepts_neumann():
    options = SolverOptions(e_adjoint_solver=" NEUMANN ")
    options.validate()
    assert options.e_adjoint_solver == "neumann"


def test_solver_options_rejects_invalid_e_adjoint_solver():
    options = SolverOptions(e_adjoint_solver="bad")
    with pytest.raises(ValueError):
        options.validate()


# ----------------------------------------------------------------------------
# 3. Gradient parity: the key correctness check on a real small model.
# ----------------------------------------------------------------------------

_D = "tests/data/alerax/test_trees_1"


def _build_small_static(*, n_fam=5, dtype=torch.float64):
    """A small GeneReconModel (15 species / n_fam families) where GMRES converges cleanly."""
    from gpurec import GeneReconModel

    so = SolverOptions(
        e_max_iter=2000, e_tol=1e-12, pi_iters=128,
        neumann_terms=64, self_loop_solver="neumann",
        bicgstab_max_iter=200, bicgstab_tol=1e-12, bicgstab_breakdown_tol=1e-30,
        adjoint_pruning_threshold=0.0, use_adjoint_pruning=False, pibar_side_threshold=0.0,
    )
    so.validate()
    m = GeneReconModel(
        f"{_D}/sp.nwk", [f"{_D}/g.nwk"] * n_fam,
        mode="global", device="cuda", dtype=dtype, solver_options=so,
    )
    assert len(m.batch_statics) == 1
    return m


@pytest.mark.gpu
def test_neumann_gradient_matches_gmres():
    """Both solvers solve the SAME linear system (I-J)wE=q -- their gradients must match."""
    import math
    from gpurec.api._execution import evaluate_static_loss_grad

    m = _build_small_static()
    static = m.batch_statics[0]
    S = int(m.species_helpers["S"])
    theta = torch.full((3,), math.log2(0.1), device="cuda", dtype=torch.float64)
    receiver_weights = torch.zeros(S, device="cuda", dtype=torch.float64)
    origination_weights = torch.zeros(S, device="cuda", dtype=torch.float64)

    grads = {}
    for solver_name in ("gmres", "neumann"):
        static.solver_options.e_adjoint_solver = solver_name
        loss, grad_theta, grad_receiver, _ = evaluate_static_loss_grad(
            static, theta, receiver_weights, origination_weights, need_grad=True,
        )
        grads[solver_name] = (loss.clone(), grad_theta.clone(), grad_receiver.clone())

    loss_g, gtheta_g, grecv_g = grads["gmres"]
    loss_n, gtheta_n, grecv_n = grads["neumann"]

    assert torch.isfinite(gtheta_g).all() and torch.isfinite(gtheta_n).all()
    assert float(torch.linalg.vector_norm(gtheta_g)) > 0.0, "degenerate all-zero gradient probe"

    loss_rel = abs(float(loss_g) - float(loss_n)) / max(abs(float(loss_g)), 1e-30)
    theta_rel = float(torch.linalg.vector_norm(gtheta_g - gtheta_n)) / max(
        float(torch.linalg.vector_norm(gtheta_g)), 1e-30
    )
    recv_abs = float(torch.linalg.vector_norm(grecv_g - grecv_n))
    recv_scale = max(float(torch.linalg.vector_norm(grecv_g)), 1e-30)

    assert loss_rel < 1e-10, f"loss mismatch: gmres={float(loss_g):.6e} neumann={float(loss_n):.6e}"
    assert theta_rel < 1e-6, (
        f"theta gradient mismatch: rel={theta_rel:.3e}\ngmres={gtheta_g}\nneumann={gtheta_n}"
    )
    assert recv_abs / recv_scale < 1e-6 or recv_abs < 1e-10, (
        f"receiver gradient mismatch: abs={recv_abs:.3e} scale={recv_scale:.3e}"
    )


@pytest.mark.gpu
def test_default_e_adjoint_solver_path_is_gmres():
    """When ``e_adjoint_solver`` is not set (default SolverOptions), the solve uses GMRES --
    i.e. omitting the new option reproduces the pre-existing gradient exactly."""
    import math
    from gpurec.api._execution import evaluate_static_loss_grad

    m = _build_small_static()
    static = m.batch_statics[0]
    assert static.solver_options.e_adjoint_solver == "gmres"
    S = int(m.species_helpers["S"])
    theta = torch.full((3,), math.log2(0.1), device="cuda", dtype=torch.float64)
    receiver_weights = torch.zeros(S, device="cuda", dtype=torch.float64)
    origination_weights = torch.zeros(S, device="cuda", dtype=torch.float64)

    static.warm_E = None
    loss1, gtheta1, grecv1, _ = evaluate_static_loss_grad(
        static, theta, receiver_weights, origination_weights, need_grad=True,
    )
    # explicit gmres must reproduce the default (no e_adjoint_solver passed upstream) from the
    # same cold start. Not asserted bit-identical: Triton atomic-add reductions in the E-step /
    # backward accumulate in nondeterministic order across separate kernel launches (see
    # docs/backward_atomics_profiling.md), so re-running the *unchanged* gmres path twice is
    # already not bit-exact -- allclose at a tight tolerance is the correct no-regression bar.
    static.solver_options.e_adjoint_solver = "gmres"
    static.warm_E = None
    loss2, gtheta2, grecv2, _ = evaluate_static_loss_grad(
        static, theta, receiver_weights, origination_weights, need_grad=True,
    )
    assert torch.allclose(gtheta1, gtheta2, rtol=1e-10, atol=1e-12)
    assert torch.allclose(grecv1, grecv2, rtol=1e-10, atol=1e-12)
    assert torch.allclose(loss1, loss2, rtol=1e-10, atol=1e-12)

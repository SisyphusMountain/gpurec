import torch
import pytest

from gpurec.optimization import implicit_grad as implicit_grad_module
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


def test_implicit_gradient_forwards_pi_adjoint_residual_stats(monkeypatch):
    theta = torch.zeros((1, 3), dtype=torch.float64)
    pi = torch.zeros((1, 2), dtype=torch.float64)
    solved_pi_adjoint = torch.full_like(pi, 0.5)

    def fake_pi_wave_backward(*args, **kwargs):
        assert kwargs["return_residual_stats"] is True
        return {
            "v_Pi": solved_pi_adjoint,
            "pi_adjoint_residual_absmax": 0.125,
            "pi_adjoint_residual_relmax": 0.0625,
            "pi_adjoint_residual_wave_count": 3,
        }

    def fake_e_adjoint_and_theta_vjp(*args, **kwargs):
        assert kwargs["return_aux"] is True
        return (
            torch.ones_like(theta),
            implicit_grad_module._SolveStats("fake", 2, 0.0),
            {},
        )

    monkeypatch.setattr(
        implicit_grad_module,
        "Pi_wave_backward",
        fake_pi_wave_backward,
    )
    monkeypatch.setattr(
        implicit_grad_module,
        "_e_adjoint_and_theta_vjp",
        fake_e_adjoint_and_theta_vjp,
    )

    grad, stats, aux = implicit_grad_module.implicit_grad_loglik_vjp_wave(
        {
            "root_clade_ids": torch.tensor([0]),
            "family_idx": torch.tensor([0]),
        },
        {},
        Pi_star_wave=pi,
        Pibar_star_wave=pi,
        E_star=torch.zeros(2, dtype=theta.dtype),
        Ebar=torch.zeros(2, dtype=theta.dtype),
        E_s1=torch.zeros(2, dtype=theta.dtype),
        E_s2=torch.zeros(2, dtype=theta.dtype),
        log_pS=torch.zeros(2, dtype=theta.dtype),
        log_pD=torch.zeros(2, dtype=theta.dtype),
        log_pL=torch.zeros(2, dtype=theta.dtype),
        max_transfer_mat=torch.zeros(2, dtype=theta.dtype),
        root_clade_ids_perm=torch.tensor([0]),
        theta=theta,
        unnorm_row_max=torch.zeros(2, dtype=theta.dtype),
        specieswise=False,
        device=torch.device("cpu"),
        dtype=theta.dtype,
        neumann_terms=4,
        return_aux=True,
        record_pi_adjoint_residual=True,
    )

    torch.testing.assert_close(grad, torch.ones_like(theta))
    assert stats.pi_adjoint_residual_absmax == pytest.approx(0.125)
    assert stats.pi_adjoint_residual_relmax == pytest.approx(0.0625)
    assert stats.pi_adjoint_residual_wave_count == 3
    torch.testing.assert_close(aux["pi_adjoint"], solved_pi_adjoint)

import itertools
import math

import torch
import pytest

from hierarchical_adapter import (
    blocking_faces,
    complete_information,
    face_step_model,
    jacobian,
    map_hessians,
    phi_to_theta,
    regularized_tangent_direction,
    retract_feasible_ray_cpu64,
    tangent_projector,
    theta_to_phi,
    transform_gradient,
    transform_hessian,
    working_set_tangent_direction,
)


def _spd(n: int) -> torch.Tensor:
    a = torch.randn(n, 3, 3, dtype=torch.float64)
    return a.transpose(1, 2) @ a + 0.2 * torch.eye(3, dtype=torch.float64)


def test_roundtrip_jacobian_and_full_hessian_chain_rule():
    torch.manual_seed(4)
    phi = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    theta = phi_to_theta(phi)
    torch.testing.assert_close(theta_to_phi(theta), phi, rtol=2e-14, atol=2e-14)
    for row in range(phi.shape[0]):
        j_ref = torch.autograd.functional.jacobian(phi_to_theta, phi[row])
        torch.testing.assert_close(jacobian(phi[row]), j_ref, rtol=2e-14, atol=2e-14)

    htheta = _spd(7)
    linear = torch.randn(7, 3, dtype=torch.float64)
    delta = theta - torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)
    gtheta = linear + torch.einsum("bij,bj->bi", htheta, delta)
    hphi = transform_hessian(phi, gtheta, htheta)
    for row in range(phi.shape[0]):
        def objective(z):
            dt = phi_to_theta(z) - torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)
            return linear[row] @ dt + 0.5 * dt @ htheta[row] @ dt
        href = torch.autograd.functional.hessian(objective, phi[row])
        torch.testing.assert_close(hphi[row], href, rtol=3e-13, atol=3e-13)


def test_complete_information_is_fixed_count_hessian():
    torch.manual_seed(8)
    phi = torch.randn(5, 3, dtype=torch.float64)
    counts = torch.rand(5, 4, dtype=torch.float64) * 100 + 0.1
    for row in range(5):
        ns, nd, nl, nt = counts[row]
        def q(z):
            u, v, w = z
            f = lambda x: torch.logaddexp(torch.zeros_like(x), math.log(2.0) * x) / math.log(2.0)
            return ((ns + nd + nl + nt) * f(u) - (nd + nt) * u
                    + (nd + nt) * f(v) - nt * v
                    + (ns + nl) * f(w) - nl * w)
        href = torch.autograd.functional.hessian(q, phi[row])
        torch.testing.assert_close(complete_information(phi, counts)[row], href,
                                   rtol=3e-13, atol=3e-13)


def test_all_active_masks_project_and_lagrangian_curvature_matches_retraction():
    torch.manual_seed(11)
    phi0 = torch.tensor([[0.2, -0.7, 0.5]], dtype=torch.float64)
    theta0 = phi_to_theta(phi0)[0]
    gtheta = torch.tensor([[0.8, -1.2, 0.3]], dtype=torch.float64)
    htheta = _spd(1)
    gphi = transform_gradient(phi0, gtheta)
    hphi = transform_hessian(phi0, gtheta, htheta)
    for bits in itertools.product((False, True), repeat=3):
        fixed = torch.tensor([bits])
        p, corrected, _gf = face_step_model(phi0, gtheta, gphi, hphi, fixed)
        j_active = jacobian(phi0)[0, fixed[0]]
        if j_active.numel():
            torch.testing.assert_close(j_active @ p[0], torch.zeros_like(j_active), atol=2e-14, rtol=0)
        torch.testing.assert_close(p @ p, p, atol=2e-14, rtol=0)
        d = torch.einsum("bij,bj->bi", p, torch.randn(1, 3, dtype=torch.float64))[0]
        if d.norm() < 1e-12:
            continue

        def retracted_objective(s):
            raw = phi_to_theta(phi0[0] + s * d)
            trial = torch.where(fixed[0], theta0, raw)
            dt = trial - theta0
            return gtheta[0] @ dt + 0.5 * dt @ htheta[0] @ dt

        second = torch.autograd.functional.hessian(retracted_objective,
                                                    torch.tensor(0.0, dtype=torch.float64))
        predicted = d @ corrected[0] @ d
        torch.testing.assert_close(predicted, second, rtol=2e-12, atol=2e-12)


def test_regularized_step_is_tangent_and_ray_retraction_is_feasible():
    torch.manual_seed(15)
    n = 24
    lo, hi = -4.0, 1.0
    theta = lo + 0.2 + (hi - lo - 0.4) * torch.rand(n, 3, dtype=torch.float64)
    # Put deliberately blocking coordinates on native faces.
    fixed = torch.rand(n, 3) < 0.35
    upper = torch.rand(n, 3) < 0.5
    theta = torch.where(fixed & upper, torch.full_like(theta, hi), theta)
    theta = torch.where(fixed & ~upper, torch.full_like(theta, lo), theta)
    phi = theta_to_phi(theta)
    gtheta = torch.randn(n, 3, dtype=torch.float64)
    gtheta = torch.where(fixed & upper, -gtheta.abs(), gtheta)
    gtheta = torch.where(fixed & ~upper, gtheta.abs(), gtheta)
    gphi = transform_gradient(phi, gtheta)
    bphi = _spd(n)
    radius = 0.05 + 0.5 * torch.rand(n, dtype=torch.float64)
    direction, hred, gface, _ = regularized_tangent_direction(
        phi, gtheta, gphi, bphi, fixed, radius, 1e-4,
    )
    active_products = torch.einsum("bki,bi->bk", jacobian(phi), direction)
    torch.testing.assert_close(active_products[fixed], torch.zeros_like(active_products[fixed]),
                               atol=2e-11, rtol=0)
    phi1, theta1, tangent, chord, _capped = retract_feasible_ray_cpu64(
        phi, theta, direction * 8.0, fixed, lo, hi, radius,
    )
    assert bool(((theta1 >= lo) & (theta1 <= hi)).all())
    torch.testing.assert_close(theta1[fixed], theta[fixed], atol=2e-12, rtol=0)
    assert bool(((theta1 - theta).norm(dim=1) <= radius + 2e-10).all())
    torch.testing.assert_close(phi_to_theta(phi1), theta1, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(phi1 - phi, chord, atol=2e-12, rtol=0)
    assert bool(torch.isfinite(hred).all() & torch.isfinite(gface).all() & torch.isfinite(tangent).all())


def test_mapping_hessians_match_autograd():
    phi = torch.tensor([0.1, -0.5, 0.7], dtype=torch.float64)
    for k in range(3):
        href = torch.autograd.functional.hessian(lambda z: phi_to_theta(z)[k], phi)
        torch.testing.assert_close(map_hessians(phi)[k], href, atol=2e-14, rtol=2e-14)


def test_zero_and_inward_steps_start_from_authoritative_native_bound_point():
    lo, hi = math.log2(1e-6), 1.0
    # Mimic a point actually evaluated in float32, including its representable lower bound.
    theta = torch.tensor([[lo, 0.2, hi]], dtype=torch.float32)
    phi = theta_to_phi(theta)
    fixed = torch.zeros_like(theta, dtype=torch.bool)
    radius = torch.tensor([0.25], dtype=torch.float32)
    z1, th1, tangent, chord, _ = retract_feasible_ray_cpu64(
        phi, theta, torch.zeros_like(phi), fixed, lo, hi, radius,
    )
    torch.testing.assert_close(th1, theta, rtol=0, atol=0)
    torch.testing.assert_close(phi_to_theta(z1), theta, rtol=0, atol=3e-6)
    torch.testing.assert_close(tangent, torch.zeros_like(tangent), rtol=0, atol=0)
    torch.testing.assert_close(chord, torch.zeros_like(chord), rtol=0, atol=2e-7)

    # An inward native direction at the lower D face must remain available when the gradient does
    # not mark that face as blocking.
    inward_native = torch.tensor([[0.05, 0.0, 0.0]], dtype=torch.float32)
    inward_phi = torch.linalg.solve(jacobian(phi), inward_native.unsqueeze(-1)).squeeze(-1)
    _z2, th2, _t2, _c2, _ = retract_feasible_ray_cpu64(
        phi, theta, inward_phi, fixed, lo, hi, radius,
    )
    assert float(th2[0, 0]) > float(theta[0, 0])


def test_coordinate_metric_caps_tangent_not_native_displacement():
    theta = torch.tensor([[-3.0, 0.4, -2.0]], dtype=torch.float64)
    phi = theta_to_phi(theta)
    fixed = torch.zeros_like(theta, dtype=torch.bool)
    direction = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64)
    radius = torch.tensor([0.1], dtype=torch.float64)
    _z, th1, tangent, _chord, capped = retract_feasible_ray_cpu64(
        phi, theta, direction, fixed, -20.0, 1.0, radius, metric="coordinate",
    )
    torch.testing.assert_close(tangent.norm(dim=1), radius, atol=3e-10, rtol=0)
    assert bool(capped.all())
    assert bool(((th1 >= -20.0) & (th1 <= 1.0)).all())


def test_effective_float32_bound_and_inward_tangent_survive_inverse_roundoff():
    # -0.1 rounds outward as a float32 lower face. The ray checker must use that actual model face,
    # accept alpha=0 despite inverse-map float64 dust, and leave an inward direction available.
    lo, hi = -0.1, 0.7
    lo_effective = float(torch.tensor(lo, dtype=torch.float32))
    theta = torch.tensor([[lo_effective, 0.2, 0.3]], dtype=torch.float32)
    phi = theta_to_phi(theta)
    fixed = torch.zeros_like(theta, dtype=torch.bool)
    radius = torch.tensor([0.1], dtype=torch.float32)
    _z0, th0, _t0, _c0, _ = retract_feasible_ray_cpu64(
        phi, theta, torch.zeros_like(phi), fixed, lo, hi, radius,
    )
    torch.testing.assert_close(th0, theta, rtol=0, atol=0)
    inward_native = torch.tensor([[0.02, 0.0, 0.0]], dtype=torch.float32)
    inward_phi = torch.linalg.solve(jacobian(phi), inward_native.unsqueeze(-1)).squeeze(-1)
    _z1, th1, _t1, _c1, _ = retract_feasible_ray_cpu64(
        phi, theta, inward_phi, fixed, lo, hi, radius,
    )
    assert float(th1[0, 0]) > lo_effective


@pytest.mark.parametrize("offset", [0.0, 0.5e-6])
@pytest.mark.parametrize("metric", ["native", "coordinate"])
def test_working_set_repairs_coupled_outward_step_at_or_near_lower_face(offset, metric):
    lo, hi, eps = math.log2(1e-6), 1.0, 1e-6
    theta = torch.tensor([[lo + offset, -3.0, -4.0]], dtype=torch.float64)
    phi = theta_to_phi(theta)
    # D's gradient is inward at its lower face, yet this coupled SPD model makes the unconstrained
    # Newton proposal move D outward. This is the deterministic failure mode that collapsed ray
    # bisection to alpha=0 before the secondary working set was added.
    gtheta = torch.tensor([[-1.0, 0.8227965831756592, 3.862908124923706]],
                          dtype=torch.float64)
    curvature = torch.tensor([[[3.6835919896263296, 1.365157561614173, 1.4246881413709345],
                               [1.365157561614173, 2.49126611074482, -0.9727459518324018],
                               [1.4246881413709345, -0.9727459518324018, 4.402212801109616]]],
                             dtype=torch.float64)
    gphi = transform_gradient(phi, gtheta)
    authoritative = blocking_faces(theta, gtheta, lo, hi, eps)
    assert not bool(authoritative[0, 0])
    radius = torch.tensor([2.0], dtype=torch.float64)
    raw, *_ = regularized_tangent_direction(
        phi, gtheta, gphi, curvature, authoritative, radius, 1e-4, metric,
    )
    assert float((jacobian(phi) @ raw.unsqueeze(-1))[0, 0, 0]) < 0.0

    direction, _h, _gf, working, _limited = working_set_tangent_direction(
        phi, theta, gtheta, gphi, curvature, authoritative, radius, 1e-4,
        lo, hi, eps, metric,
    )
    assert bool(working[0, 0])
    native_linear = (jacobian(phi) @ direction.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(native_linear[0, 0], torch.tensor(0.0, dtype=torch.float64),
                               atol=2e-13, rtol=0)
    assert float((gphi * direction).sum()) < 0.0
    _z1, theta1, tangent, _chord, _capped = retract_feasible_ray_cpu64(
        phi, theta, direction, working, lo, hi, radius, metric=metric,
    )
    assert float(tangent.norm()) > 1e-6
    torch.testing.assert_close(theta1[0, 0], theta[0, 0], atol=2e-13, rtol=0)
    assert bool(((theta1 >= lo) & (theta1 <= hi)).all())

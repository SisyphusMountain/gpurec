from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from gpurec.optimization import LBFGSB
from gpurec.optimization.lbfgsb import _LineSearchResult


def _quadratic_model_value(
    x0: torch.Tensor,
    grad: torch.Tensor,
    hessian: torch.Tensor,
    x: torch.Tensor,
) -> float:
    step = x - x0
    value = torch.dot(grad, step) + 0.5 * torch.dot(step, hessian @ step)
    return float(value.detach().cpu())


def _run_torch_lbfgsb(
    loss_fn,
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_steps: int = 200,
) -> tuple[np.ndarray, float, float]:
    x = torch.nn.Parameter(torch.as_tensor(x0, dtype=torch.float64).clone())
    optimizer = LBFGSB(
        [x],
        lr=1.0,
        max_iter=1,
        history_size=10,
        max_ls=30,
        tolerance_grad=0.0,
        lower_bound=torch.as_tensor(lower, dtype=torch.float64),
        upper_bound=torch.as_tensor(upper, dtype=torch.float64),
    )
    loss = torch.tensor(float("nan"), dtype=torch.float64)
    projected_grad_inf = float("inf")
    for _ in range(max_steps):
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            value = loss_fn(x)
            value.backward()
            return value

        loss = optimizer.step(closure, loss_closure=lambda: loss_fn(x.detach()))
        projected_grad = optimizer.state[x]["last_projected_grad"]
        projected_grad_inf = float(projected_grad.detach().abs().amax().cpu())
        if projected_grad_inf <= 1e-7:
            break
    return x.detach().cpu().numpy(), float(loss.detach().cpu()), projected_grad_inf


def test_lbfgsb_matches_scipy_on_boxed_quadratic_boundary_solution():
    matrix = np.diag([3.0, 1.0])
    linear = np.array([-6.0, 2.0])
    lower = np.array([-1.0, -2.0])
    upper = np.array([1.0, 2.0])
    x0 = np.array([0.0, 0.0])

    def scipy_fun(x: np.ndarray) -> float:
        return float(0.5 * x @ matrix @ x + linear @ x)

    def scipy_jac(x: np.ndarray) -> np.ndarray:
        return matrix @ x + linear

    def torch_loss(x: torch.Tensor) -> torch.Tensor:
        mat = torch.as_tensor(matrix, dtype=x.dtype, device=x.device)
        lin = torch.as_tensor(linear, dtype=x.dtype, device=x.device)
        return 0.5 * x @ (mat @ x) + lin @ x

    scipy_result = minimize(
        scipy_fun,
        x0,
        jac=scipy_jac,
        bounds=list(zip(lower, upper)),
        method="L-BFGS-B",
        options={"gtol": 1e-8},
    )
    torch_x, torch_f, projected_grad_inf = _run_torch_lbfgsb(
        torch_loss,
        x0,
        lower,
        upper,
    )

    assert torch_x == pytest.approx(scipy_result.x, abs=1e-8)
    assert torch_f == pytest.approx(float(scipy_result.fun), abs=1e-10)
    assert projected_grad_inf <= 1e-7


def test_lbfgsb_matches_scipy_on_bounded_rosenbrock():
    lower = np.array([-2.0, -1.0])
    upper = np.array([2.0, 3.0])
    x0 = np.array([-1.2, 1.0])

    def scipy_fun(x: np.ndarray) -> float:
        return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] * x[0]) ** 2)

    def scipy_jac(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ]
        )

    def torch_loss(x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] * x[0]) ** 2

    scipy_result = minimize(
        scipy_fun,
        x0,
        jac=scipy_jac,
        bounds=list(zip(lower, upper)),
        method="L-BFGS-B",
        options={"gtol": 1e-8, "maxiter": 200},
    )
    torch_x, torch_f, projected_grad_inf = _run_torch_lbfgsb(
        torch_loss,
        x0,
        lower,
        upper,
    )

    assert torch_x == pytest.approx(scipy_result.x, abs=1e-5)
    assert torch_f == pytest.approx(float(scipy_result.fun), abs=1e-10)
    assert projected_grad_inf <= 1e-6


def test_generalized_cauchy_point_stops_at_increasing_kink(monkeypatch):
    x0 = torch.tensor(
        [
            -0.2567423350833423,
            -0.15070369427943198,
            -0.055927647531247904,
            -0.09312721568862237,
            -0.2140615133101822,
        ],
        dtype=torch.float64,
    )
    grad = torch.tensor(
        [
            1.058401717537042,
            -0.08892125775513729,
            0.11628027489726733,
            1.7010556606132352,
            -2.180877699448789,
        ],
        dtype=torch.float64,
    )
    lower = torch.tensor(
        [
            -0.46429860276512536,
            -0.29808648973263463,
            -0.3881458250733688,
            -0.34059094384719,
            -0.40897703095823046,
        ],
        dtype=torch.float64,
    )
    upper = torch.tensor(
        [
            0.3881798275392654,
            0.027176461267901103,
            0.2993149918962872,
            -0.05805929916561389,
            0.18116570056686265,
        ],
        dtype=torch.float64,
    )
    hessian = torch.tensor(
        [
            [
                1.0446435108618388,
                -0.054656922544021984,
                0.7825152368894889,
                -0.5827314583898133,
                -2.2794564129086052,
            ],
            [
                -0.054656922544021984,
                4.545660980409671,
                3.989246392699256,
                0.4305281749315918,
                2.5734555931578087,
            ],
            [
                0.7825152368894889,
                3.989246392699256,
                5.216122482188922,
                -0.37430122388007797,
                -1.2043647523535714,
            ],
            [
                -0.5827314583898133,
                0.4305281749315918,
                -0.37430122388007797,
                1.438743793761829,
                1.6286784746222591,
            ],
            [
                -2.2794564129086052,
                2.5734555931578087,
                -1.2043647523535714,
                1.6286784746222591,
                10.45932124406558,
            ],
        ],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            -0.41071497746985086,
            -0.1377677354709457,
            -0.07284370207528755,
            -0.34059094384719,
            0.10320507511463073,
        ],
        dtype=torch.float64,
    )
    overshot = torch.tensor(
        [
            -0.46429860276512536,
            0.027176461267901103,
            -0.38814582507336876,
            -0.34059094384719,
            0.18116570056686265,
        ],
        dtype=torch.float64,
    )
    x = torch.nn.Parameter(x0.clone())
    optimizer = LBFGSB([x], lower_bound=lower, upper_bound=upper)
    monkeypatch.setattr(
        optimizer,
        "_b_matvec",
        lambda vec, old_dirs, old_stps, theta: hessian @ vec,
    )

    cauchy = optimizer._generalized_cauchy_point(
        x0,
        grad,
        lower,
        upper,
        [],
        [],
        1.0,
        1e-12,
    )

    torch.testing.assert_close(cauchy, expected, rtol=1e-12, atol=1e-12)
    assert _quadratic_model_value(x0, grad, hessian, cauchy) < (
        _quadratic_model_value(x0, grad, hessian, overshot) - 0.5
    )


def test_subspace_step_is_scalar_limited_to_first_bound():
    cauchy = torch.tensor(
        [
            -0.08536041056250598,
            -0.6775704509017844,
            0.5376927688487777,
            0.20198489937570502,
        ],
        dtype=torch.float64,
    )
    subspace_step = torch.tensor(
        [0.6470854224516689, 0.0, 0.0, 1.4457470790716425],
        dtype=torch.float64,
    )
    lower = torch.tensor(
        [
            -0.9171130725082771,
            -0.6775704509017844,
            0.3361160054390503,
            0.062063199745474695,
        ],
        dtype=torch.float64,
    )
    upper = torch.tensor(
        [
            0.1848291978651121,
            0.19923601484922682,
            0.5376927688487777,
            0.5380252095959047,
        ],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            0.06504403378833917,
            -0.6775704509017844,
            0.5376927688487777,
            0.5380252095959047,
        ],
        dtype=torch.float64,
    )
    componentwise_projected = torch.clamp(cauchy + subspace_step, lower, upper)
    x = torch.nn.Parameter(cauchy.clone())
    optimizer = LBFGSB([x], lower_bound=lower, upper_bound=upper)

    limited = optimizer._limit_subspace_step(
        cauchy,
        subspace_step,
        lower,
        upper,
    )

    torch.testing.assert_close(limited, expected, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(limited, componentwise_projected)


def test_projected_gradient_topk_sign_direction_moves_only_largest_residuals():
    x = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    optimizer = LBFGSB([x], lower_bound=-10.0, upper_bound=10.0)
    projected_grad = torch.tensor(
        [1.0, -5.0, 3.0, 0.0, -2.0],
        dtype=torch.float64,
    )

    direction = optimizer._projected_gradient_topk_sign_direction(
        x.detach(),
        projected_grad,
        -10.0,
        10.0,
        k=2,
    )

    expected = torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(direction, expected, rtol=0.0, atol=0.0)


def test_topk_sign_fallback_search_chooses_best_tiny_prefix(monkeypatch):
    x = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64))
    optimizer = LBFGSB([x], lr=0.5, lower_bound=-10.0, upper_bound=10.0)
    flat = x.detach().clone()
    loss = torch.tensor(100.0, dtype=torch.float64)
    grad = torch.tensor([6.0, -5.0, 4.0, -3.0, 2.0, -1.0], dtype=torch.float64)
    projected_grad = grad.clone()
    calls: list[int] = []

    def fake_backtracking_line_search(**kwargs):
        direction = kwargs["direction"]
        topk_size = int((direction != 0).sum().detach().cpu())
        calls.append(topk_size)
        decrease = {1: 1.0, 2: 10.0, 4: 5.0}.get(topk_size, 0.0)
        accepted = decrease > 0.0
        delta = direction.detach().clone()
        return _LineSearchResult(
            accepted=accepted,
            flat=flat + delta,
            loss=loss - decrease,
            alpha=1.0,
            delta=delta,
            directional_derivative=float(torch.dot(grad, delta).detach().cpu()),
            step_inf=float(delta.abs().amax().detach().cpu()),
            decrease=decrease,
            loss_evals=1,
            next_alpha=0.5,
            armijo_required_decrease=0.0,
        )

    monkeypatch.setattr(
        optimizer,
        "_backtracking_line_search",
        fake_backtracking_line_search,
    )

    search, kind, loss_evals = optimizer._topk_sign_fallback_search(
        closure=lambda: loss,
        loss_closure=lambda: loss,
        flat=flat,
        loss=loss,
        grad=grad,
        projected_grad=projected_grad,
        lower_bound=-10.0,
        upper_bound=10.0,
        initial_alpha=0.5,
        max_ls=4,
        c1=1e-4,
        shrink=0.5,
        tolerance_change=0.0,
    )

    assert search is not None
    assert search.decrease == pytest.approx(10.0)
    assert kind == "projected_gradient_top2_sign_fallback"
    assert loss_evals == 5
    assert calls == [1, 2, 3, 4, 5]


def test_coordinate_sign_fallback_search_scans_beyond_prefix(monkeypatch):
    x = torch.nn.Parameter(torch.zeros(8, dtype=torch.float64))
    optimizer = LBFGSB([x], lr=0.5, lower_bound=-10.0, upper_bound=10.0)
    flat = x.detach().clone()
    loss = torch.tensor(100.0, dtype=torch.float64)
    grad = torch.tensor([8.0, -7.0, 6.0, -5.0, 4.0, -3.0, 2.0, -1.0])
    grad = grad.to(dtype=torch.float64)
    projected_grad = grad.clone()
    calls: list[int] = []
    max_ls_values: list[int] = []

    def fake_backtracking_line_search(**kwargs):
        direction = kwargs["direction"]
        index = int(torch.nonzero(direction, as_tuple=False).flatten()[0].cpu())
        calls.append(index)
        max_ls_values.append(int(kwargs["max_ls"]))
        decrease = {2: 1.0, 4: 12.0}.get(index, 0.0)
        accepted = decrease > 0.0
        delta = direction.detach().clone()
        return _LineSearchResult(
            accepted=accepted,
            flat=flat + delta,
            loss=loss - decrease,
            alpha=1.0,
            delta=delta,
            directional_derivative=float(torch.dot(grad, delta).detach().cpu()),
            step_inf=float(delta.abs().amax().detach().cpu()),
            decrease=decrease,
            loss_evals=1,
            next_alpha=0.5,
            armijo_required_decrease=0.0,
        )

    monkeypatch.setattr(
        optimizer,
        "_backtracking_line_search",
        fake_backtracking_line_search,
    )

    search, kind, loss_evals = optimizer._coordinate_sign_fallback_search(
        closure=lambda: loss,
        loss_closure=lambda: loss,
        flat=flat,
        loss=loss,
        grad=grad,
        projected_grad=projected_grad,
        lower_bound=-10.0,
        upper_bound=10.0,
        initial_alpha=0.5,
        max_coordinates=5,
        max_ls=20,
        c1=1e-4,
        shrink=0.5,
        tolerance_change=0.0,
    )

    assert search is not None
    assert search.decrease == pytest.approx(12.0)
    assert kind == "projected_gradient_coord5_sign_fallback"
    assert loss_evals == 5
    assert calls == [0, 1, 2, 3, 4]
    assert max_ls_values == [1, 1, 1, 1, 1]


def test_coordinate_sign_fallback_search_respects_loss_eval_budget(monkeypatch):
    x = torch.nn.Parameter(torch.zeros(8, dtype=torch.float64))
    optimizer = LBFGSB([x], lr=0.5, lower_bound=-10.0, upper_bound=10.0)
    flat = x.detach().clone()
    loss = torch.tensor(100.0, dtype=torch.float64)
    grad = torch.tensor([8.0, -7.0, 6.0, -5.0, 4.0, -3.0, 2.0, -1.0])
    grad = grad.to(dtype=torch.float64)
    projected_grad = grad.clone()
    calls: list[int] = []

    def fake_backtracking_line_search(**kwargs):
        direction = kwargs["direction"]
        index = int(torch.nonzero(direction, as_tuple=False).flatten()[0].cpu())
        calls.append(index)
        delta = direction.detach().clone()
        return _LineSearchResult(
            accepted=False,
            flat=flat + delta,
            loss=loss,
            alpha=1.0,
            delta=delta,
            directional_derivative=float(torch.dot(grad, delta).detach().cpu()),
            step_inf=float(delta.abs().amax().detach().cpu()),
            decrease=0.0,
            loss_evals=1,
            next_alpha=0.5,
            armijo_required_decrease=0.0,
        )

    monkeypatch.setattr(
        optimizer,
        "_backtracking_line_search",
        fake_backtracking_line_search,
    )

    search, kind, loss_evals = optimizer._coordinate_sign_fallback_search(
        closure=lambda: loss,
        loss_closure=lambda: loss,
        flat=flat,
        loss=loss,
        grad=grad,
        projected_grad=projected_grad,
        lower_bound=-10.0,
        upper_bound=10.0,
        initial_alpha=0.5,
        max_coordinates=5,
        max_ls=20,
        c1=1e-4,
        shrink=0.5,
        tolerance_change=0.0,
        max_loss_evals=3,
    )

    assert search is None
    assert kind == "none"
    assert loss_evals == 3
    assert calls == [0, 1, 2]


def test_lbfgsb_projected_gradient_fallback_recovers_after_failed_candidate_line_search(
    monkeypatch,
):
    x = torch.nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float64))
    lower = torch.full_like(x, -10.0)
    upper = torch.full_like(x, 10.0)
    optimizer = LBFGSB(
        [x],
        lr=1.0,
        max_iter=1,
        history_size=5,
        max_ls=5,
        tolerance_grad=0.0,
        tolerance_change=1e-12,
        lower_bound=lower,
        upper_bound=upper,
    )
    optimizer.state[x]["old_dirs"] = [torch.ones(2, dtype=torch.float64)]
    optimizer.state[x]["old_stps"] = [torch.ones(2, dtype=torch.float64)]

    def loss_fn(value: torch.Tensor) -> torch.Tensor:
        stiff = 0.5 * (1000.0 * value[0].square() + value[1].square())
        ripple = 1e-3 * torch.sin(7.0 * value[0])
        return stiff + ripple

    def bad_candidate(*args, **kwargs):
        flat = args[0]
        direction = torch.tensor([-1.0e6, 0.0], dtype=flat.dtype, device=flat.device)
        return direction, "subspace"

    monkeypatch.setattr(optimizer, "_candidate_direction", bad_candidate)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        value = loss_fn(x)
        value.backward()
        return value

    initial_loss = float(loss_fn(x.detach()).cpu())
    loss = optimizer.step(closure, loss_closure=lambda: loss_fn(x.detach()))
    state = optimizer.state[x]

    assert float(loss.detach().cpu()) < initial_loss
    assert state["last_fallback_attempted"] is True
    assert state["last_fallback_used"] is True
    assert state["last_fallback_reason"] == "line_search_failed"
    assert state["last_history_cleared_for_fallback"] is True
    assert state["last_direction_kind"] == "projected_gradient_fallback"
    assert state["last_alpha"] == pytest.approx(state["last_fallback_alpha"])
    assert state["last_alpha"] == pytest.approx(1.0 / 32.0)
    assert state["last_loss_evals"] > 5
    assert state["last_line_search_decrease"] > 1.0
    assert x.detach()[0] > lower[0]


def test_lbfgsb_sign_fallback_recovers_when_projected_gradient_map_is_too_long(
    monkeypatch,
):
    x = torch.nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float64))
    lower = torch.full_like(x, -1000.0)
    upper = torch.full_like(x, 1000.0)
    optimizer = LBFGSB(
        [x],
        lr=1.0,
        max_iter=1,
        history_size=5,
        max_ls=4,
        fallback_max_ls=4,
        tolerance_grad=0.0,
        tolerance_change=1e-12,
        lower_bound=lower,
        upper_bound=upper,
    )

    def loss_fn(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * (10_000.0 * value[0].square() + value[1].square())

    def bad_candidate(*args, **kwargs):
        flat = args[0]
        direction = torch.zeros_like(flat)
        direction[0] = -1000.0
        return direction, "subspace"

    monkeypatch.setattr(optimizer, "_candidate_direction", bad_candidate)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        value = loss_fn(x)
        value.backward()
        return value

    initial_loss = float(loss_fn(x.detach()).cpu())
    loss = optimizer.step(closure, loss_closure=lambda: loss_fn(x.detach()))
    state = optimizer.state[x]

    assert float(loss.detach().cpu()) < initial_loss
    assert state["last_fallback_attempted"] is True
    assert state["last_fallback_used"] is True
    assert state["last_direction_kind"] == "projected_gradient_sign_fallback"
    assert state["last_alpha"] == pytest.approx(1.0)
    torch.testing.assert_close(x.detach(), torch.zeros_like(x), rtol=0.0, atol=0.0)


def test_lbfgsb_fallback_competes_after_tiny_projected_gradient_accept(
    monkeypatch,
):
    x = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
    optimizer = LBFGSB([x], lr=1.0, lower_bound=-10.0, upper_bound=10.0)
    flat = x.detach().clone()
    loss = torch.tensor(526_824.0, dtype=torch.float32)
    grad = torch.tensor([5.0, -1.0, 0.5], dtype=torch.float32)
    projected_grad = grad.clone()
    tiny_delta = -1e-4 * projected_grad
    current_search = _LineSearchResult(
        accepted=True,
        flat=flat + tiny_delta,
        loss=loss - 0.03125,
        alpha=1e-4,
        delta=tiny_delta,
        directional_derivative=float(torch.dot(grad, tiny_delta).detach().cpu()),
        step_inf=float(tiny_delta.abs().amax().detach().cpu()),
        decrease=0.03125,
        loss_evals=1,
        next_alpha=5e-5,
        armijo_required_decrease=0.0,
    )
    calls: list[torch.Tensor] = []

    def fake_backtracking_line_search(**kwargs):
        direction = kwargs["direction"].detach().clone()
        calls.append(direction)
        decrease = 2.0 if int((direction != 0).sum().detach().cpu()) == 3 else 0.0
        accepted = decrease > 0.0
        return _LineSearchResult(
            accepted=accepted,
            flat=flat + direction,
            loss=loss - decrease,
            alpha=1.0,
            delta=direction,
            directional_derivative=float(torch.dot(grad, direction).detach().cpu()),
            step_inf=float(direction.abs().amax().detach().cpu()),
            decrease=decrease,
            loss_evals=1,
            next_alpha=0.5,
            armijo_required_decrease=0.0,
        )

    monkeypatch.setattr(
        optimizer,
        "_backtracking_line_search",
        fake_backtracking_line_search,
    )

    search, kind, loss_evals = optimizer._compete_projected_gradient_fallbacks(
        closure=lambda: loss,
        loss_closure=lambda: loss,
        state=optimizer.state[x],
        flat=flat,
        loss=loss,
        grad=grad,
        projected_grad=projected_grad,
        lower_bound=-10.0,
        upper_bound=10.0,
        current_search=current_search,
        current_kind="projected_gradient_fallback",
        lr=1.0,
        max_ls=8,
        c1=1e-4,
        shrink=0.5,
        tolerance_change=0.0,
    )

    assert kind == "projected_gradient_sign_fallback"
    assert search.decrease == pytest.approx(2.0)
    assert loss_evals == 1
    assert len(calls) == 1
    torch.testing.assert_close(
        calls[0],
        torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_lbfgsb_fallback_competes_after_resolution_limited_accept(
    monkeypatch,
):
    x = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
    optimizer = LBFGSB(
        [x],
        lr=1.0,
        lower_bound=-10.0,
        upper_bound=10.0,
        fallback_resolution_competition_factor=16.0,
    )
    flat = x.detach().clone()
    loss = torch.tensor(1_700_000.0, dtype=torch.float32)
    grad = torch.tensor([5.0, -1.0, 0.5], dtype=torch.float32)
    projected_grad = grad.clone()
    current_delta = torch.tensor([-0.04, 0.0, 0.0], dtype=torch.float32)
    current_search = _LineSearchResult(
        accepted=True,
        flat=flat + current_delta,
        loss=loss - 0.25,
        alpha=0.04,
        delta=current_delta,
        directional_derivative=float(torch.dot(grad, current_delta).detach().cpu()),
        step_inf=float(current_delta.abs().amax().detach().cpu()),
        decrease=0.25,
        loss_evals=1,
        next_alpha=0.02,
        armijo_required_decrease=0.0,
    )
    calls: list[torch.Tensor] = []

    def fake_backtracking_line_search(**kwargs):
        direction = kwargs["direction"].detach().clone()
        calls.append(direction)
        decrease = 8.0 if int((direction != 0).sum().detach().cpu()) == 3 else 0.0
        accepted = decrease > 0.0
        return _LineSearchResult(
            accepted=accepted,
            flat=flat + direction,
            loss=loss - decrease,
            alpha=1.0,
            delta=direction,
            directional_derivative=float(torch.dot(grad, direction).detach().cpu()),
            step_inf=float(direction.abs().amax().detach().cpu()),
            decrease=decrease,
            loss_evals=1,
            next_alpha=0.5,
            armijo_required_decrease=0.0,
        )

    monkeypatch.setattr(
        optimizer,
        "_backtracking_line_search",
        fake_backtracking_line_search,
    )

    search, kind, loss_evals = optimizer._compete_projected_gradient_fallbacks(
        closure=lambda: loss,
        loss_closure=lambda: loss,
        state=optimizer.state[x],
        flat=flat,
        loss=loss,
        grad=grad,
        projected_grad=projected_grad,
        lower_bound=-10.0,
        upper_bound=10.0,
        current_search=current_search,
        current_kind="projected_gradient_fallback",
        lr=1.0,
        max_ls=8,
        c1=1e-4,
        shrink=0.5,
        tolerance_change=0.0,
        resolution_competition_factor=16.0,
    )

    assert kind == "projected_gradient_sign_fallback"
    assert search.decrease == pytest.approx(8.0)
    assert loss_evals == 1
    assert len(calls) == 1


def test_lbfgsb_repeated_tiny_high_kkt_steps_trigger_projected_gradient_fallback(
    monkeypatch,
):
    x = torch.nn.Parameter(torch.tensor([1.0, -0.75], dtype=torch.float64))
    optimizer = LBFGSB(
        [x],
        lr=0.5,
        max_iter=1,
        history_size=5,
        max_ls=8,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        lower_bound=torch.full_like(x, -5.0),
        upper_bound=torch.full_like(x, 5.0),
    )
    candidate_calls = 0

    def loss_fn(value: torch.Tensor) -> torch.Tensor:
        bowl = 0.5 * (value[0].square() + 4.0 * value[1].square())
        ripple = 1e-6 * torch.sin(13.0 * value[0])
        return bowl + ripple

    def tiny_candidate(*args, **kwargs):
        nonlocal candidate_calls
        flat, grad = args[0], args[1]
        candidate_calls += 1
        return -1e-10 * grad.to(dtype=flat.dtype), "subspace"

    monkeypatch.setattr(optimizer, "_candidate_direction", tiny_candidate)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        value = loss_fn(x)
        value.backward()
        return value

    first_loss = optimizer.step(closure, loss_closure=lambda: loss_fn(x.detach()))
    first_state = dict(optimizer.state[x])
    second_loss = optimizer.step(closure, loss_closure=lambda: loss_fn(x.detach()))
    second_state = dict(optimizer.state[x])
    third_loss = optimizer.step(closure, loss_closure=lambda: loss_fn(x.detach()))
    third_state = optimizer.state[x]

    assert first_state["last_fallback_used"] is False
    assert first_state["last_high_kkt_stall_count"] == 1
    assert second_state["last_fallback_used"] is False
    assert second_state["last_high_kkt_stall_count"] == 2
    assert third_state["last_fallback_attempted"] is True
    assert third_state["last_fallback_used"] is True
    assert third_state["last_fallback_reason"] == "high_kkt_tiny_progress"
    assert third_state["last_direction_kind"] == "projected_gradient_fallback"
    assert third_state["last_high_kkt_stall_count"] == 0
    assert third_state["last_line_search_decrease"] > 1e-3
    assert candidate_calls == 2
    assert float(third_loss.detach().cpu()) < float(second_loss.detach().cpu())
    assert float(second_loss.detach().cpu()) < float(first_loss.detach().cpu())


def test_lbfgsb_step_tolerates_legacy_state_without_fallback_coordinate_key():
    x = torch.nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.float64))
    optimizer = LBFGSB(
        [x],
        lr=0.5,
        max_iter=1,
        history_size=5,
        max_ls=2,
        fallback_max_ls=2,
        fallback_max_coordinates=0,
        tolerance_grad=0.0,
        lower_bound=torch.full_like(x, -5.0),
        upper_bound=torch.full_like(x, 5.0),
    )
    del optimizer.param_groups[0]["fallback_max_coordinates"]
    del optimizer.param_groups[0]["fallback_max_loss_evals"]
    del optimizer.param_groups[0]["fallback_resolution_competition_factor"]

    def loss_fn(value: torch.Tensor) -> torch.Tensor:
        return value.square().sum()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        value = loss_fn(x)
        value.backward()
        return value

    loss = optimizer.step(closure, loss_closure=lambda: loss_fn(x.detach()))

    assert torch.isfinite(loss)
    assert optimizer.state[x]["last_grad_evals"] >= 1

from types import SimpleNamespace

import torch
import pytest

from gpurec import SolverOptions
from gpurec.api._batch_state import (
    gmres_check_schedule_for_static,
    gmres_check_schedule_state_for_static,
    gmres_solution_cache_for_static,
)
from gpurec.api._implicit_grad import (
    _bicgstab,
    _gmres_observed_schedule_iterations,
    implicit_grad_loglik_vjp_wave,
)
from gpurec.core.kernels import wave_backward


def test_solver_options_accept_gmres_fixed():
    options = SolverOptions(
        neumann_terms=10,
        self_loop_solver=" GMRES_FIXED ",
        gmres_tol=1e-8,
        gmres_check_interval=2,
        gmres_reuse_check_schedule=True,
        gmres_trust_check_schedule=True,
        gmres_trusted_schedule_validation_interval=7,
        gmres_trusted_schedule_safety_margin=1,
        gmres_reuse_solution=True,
        gmres_solution_cache_min_iterations=3,
        gmres_preconditioner=" Diagonal ",
        gmres_diagonal_preconditioner_floor=1e-5,
    )

    options.validate()

    assert options.self_loop_solver == "gmres_fixed"
    assert options.gmres_tol == 1e-8
    assert options.gmres_check_interval == 2
    assert options.gmres_reuse_check_schedule is True
    assert options.gmres_trust_check_schedule is True
    assert options.gmres_trusted_schedule_validation_interval == 7
    assert options.gmres_trusted_schedule_safety_margin == 1
    assert options.gmres_reuse_solution is True
    assert options.gmres_solution_cache_min_iterations == 3
    assert options.gmres_preconditioner == "diagonal"
    assert options.gmres_diagonal_preconditioner_floor == 1e-5


def test_bicgstab_skips_initial_zero_operator_application():
    b = torch.tensor([1.0, -2.0, 0.5])
    calls = []

    def Av(x):
        calls.append(x.detach().clone())
        return x

    x = _bicgstab(Av, b, max_iter=4, tol=1e-12)

    assert torch.allclose(x, b)
    assert len(calls) == 1
    assert torch.count_nonzero(calls[0]).item() > 0


def test_trusted_one_step_gmres_fast_path_solves_identity():
    if not torch.cuda.is_available():
        pytest.skip("trusted one-step fast path is CUDA-only")
    rhs = torch.tensor([1.0, -2.0, 0.5], device="cuda")
    stats = {}

    result = wave_backward._gmres_solve_wave_self_loop(
        lambda value: value,
        rhs,
        max_iter=4,
        tol=1e-12,
        min_check_iter=1,
        trust_min_check_iter=True,
        stats_out=stats,
    )

    assert torch.allclose(result, rhs)
    assert stats["arnoldi_backend"] == "trusted_one_step_triton"
    assert stats["a_applications"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"self_loop_solver": "bad"},
        {"gmres_tol": 0.0},
        {"gmres_check_interval": 0},
        {"gmres_solution_cache_min_iterations": 0},
        {"gmres_trusted_schedule_validation_interval": -1},
        {"gmres_trusted_schedule_safety_margin": -1},
        {"gmres_preconditioner": "bad"},
        {"gmres_diagonal_preconditioner_floor": 0.0},
    ],
)
def test_solver_options_reject_invalid_gmres_options(kwargs):
    options = SolverOptions(**kwargs)

    with pytest.raises(ValueError):
        options.validate()


def _static_for_gmres_cache(options: SolverOptions) -> SimpleNamespace:
    return SimpleNamespace(
        genewise=True,
        specieswise=False,
        family_indices=[3, 1],
        family_index_tensor=torch.tensor([3, 1], dtype=torch.long),
        species_helpers={"sp_parent": torch.tensor([-1, 0, 0], dtype=torch.int32)},
        wave_layout={"wave_metas": [{"start": 0, "W": 2}, {"start": 2, "W": 1}]},
        solver_options=options,
        gmres_check_schedule=None,
        gmres_check_schedule_key=None,
        gmres_check_schedule_pass_count=0,
        gmres_solution_cache=None,
        gmres_solution_cache_key=None,
    )


def test_gmres_check_schedule_cache_is_opt_in_and_keyed():
    static = _static_for_gmres_cache(
        SolverOptions(
            neumann_terms=10,
            self_loop_solver="gmres",
            gmres_reuse_check_schedule=True,
            gmres_tol=1e-6,
            gmres_check_interval=1,
        )
    )

    schedule = gmres_check_schedule_for_static(static)
    assert schedule == []

    schedule.extend([2, 1])
    assert gmres_check_schedule_for_static(static) is schedule

    static.solver_options.gmres_tol = 1e-7
    assert gmres_check_schedule_for_static(static) == []
    assert static.gmres_check_schedule is not schedule

    schedule = gmres_check_schedule_for_static(static)
    schedule.extend([2, 1])
    static.solver_options.gmres_trust_check_schedule = True
    assert gmres_check_schedule_for_static(static) == []
    assert static.gmres_check_schedule is not schedule

    schedule = gmres_check_schedule_for_static(static)
    schedule.extend([2, 1])
    static.solver_options.gmres_trusted_schedule_validation_interval = 3
    assert gmres_check_schedule_for_static(static) == []
    assert static.gmres_check_schedule is not schedule

    schedule = gmres_check_schedule_for_static(static)
    schedule.extend([2, 1])
    static.solver_options.gmres_trusted_schedule_safety_margin = 1
    assert gmres_check_schedule_for_static(static) == []
    assert static.gmres_check_schedule is not schedule

    static.solver_options.gmres_reuse_check_schedule = False
    assert gmres_check_schedule_for_static(static) is None
    assert static.gmres_check_schedule is None
    assert static.gmres_check_schedule_key is None
    assert static.gmres_check_schedule_pass_count == 0


def test_gmres_check_schedule_state_tracks_validation_cadence():
    static = _static_for_gmres_cache(
        SolverOptions(
            neumann_terms=10,
            self_loop_solver="gmres",
            gmres_reuse_check_schedule=True,
            gmres_trust_check_schedule=True,
            gmres_trusted_schedule_validation_interval=2,
        )
    )

    schedule, validate = gmres_check_schedule_state_for_static(static)
    assert schedule == []
    assert validate is True
    assert static.gmres_check_schedule_pass_count == 1

    schedule[:] = [2, 1]
    assert gmres_check_schedule_state_for_static(static) == (schedule, False)
    assert static.gmres_check_schedule_pass_count == 2
    assert gmres_check_schedule_state_for_static(static) == (schedule, True)
    assert static.gmres_check_schedule_pass_count == 3
    assert gmres_check_schedule_state_for_static(static) == (schedule, False)


def test_gmres_check_schedule_state_can_disable_periodic_validation():
    static = _static_for_gmres_cache(
        SolverOptions(
            neumann_terms=10,
            self_loop_solver="gmres",
            gmres_reuse_check_schedule=True,
            gmres_trust_check_schedule=True,
            gmres_trusted_schedule_validation_interval=0,
        )
    )

    schedule, validate = gmres_check_schedule_state_for_static(static)
    assert validate is True
    schedule[:] = [2, 1]
    assert gmres_check_schedule_state_for_static(static) == (schedule, False)
    assert gmres_check_schedule_state_for_static(static) == (schedule, False)


def test_gmres_observed_schedule_iterations_preserves_only_actual_trusted_skips():
    previous_schedule = [2, 4]

    assert _gmres_observed_schedule_iterations(
        {"iterations": 3, "trusted_check_used": True},
        previous_schedule,
        0,
        neumann_terms=10,
    ) == 2
    assert _gmres_observed_schedule_iterations(
        {"iterations": 5},
        previous_schedule,
        1,
        neumann_terms=10,
    ) == 5
    assert _gmres_observed_schedule_iterations(
        {"iterations": 12, "trusted_check_used": False},
        previous_schedule,
        1,
        neumann_terms=10,
    ) == 10
    assert _gmres_observed_schedule_iterations(None, previous_schedule, 0, neumann_terms=10) == 1


def test_implicit_grad_rejects_negative_trusted_schedule_margin_direct_call():
    tensor = torch.zeros((1, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="gmres_trusted_schedule_safety_margin"):
        implicit_grad_loglik_vjp_wave(
            {"root_clade_ids": torch.tensor([0]), "wave_metas": []},
            {},
            Pi_star_wave=tensor,
            Pibar_star_wave=tensor,
            E_star=tensor,
            E_s1=tensor,
            E_s2=tensor,
            Ebar=tensor,
            log_pS=tensor,
            log_pD=tensor,
            log_pL=tensor,
            max_transfer_mat=tensor,
            receiver_log_probs=torch.zeros((1,), dtype=torch.float32),
            use_receiver_weights=False,
            theta=tensor,
            receiver_weights=torch.ones((1,), dtype=torch.float32),
            uniform_pibar_row_max=tensor,
            family_idx=torch.zeros((1,), dtype=torch.long),
            gmres_trusted_schedule_safety_margin=-1,
        )


def test_gmres_solution_cache_is_opt_in_and_keyed():
    static = _static_for_gmres_cache(
        SolverOptions(
            neumann_terms=10,
            self_loop_solver="gmres",
            gmres_reuse_solution=True,
            gmres_solution_cache_min_iterations=2,
            gmres_tol=1e-6,
            gmres_check_interval=1,
        )
    )

    cache = gmres_solution_cache_for_static(static)
    assert cache == []

    cached_tensor = torch.ones((2, 3))
    cache.extend([cached_tensor, None])
    assert gmres_solution_cache_for_static(static) is cache

    static.solver_options.gmres_solution_cache_min_iterations = 3
    assert gmres_solution_cache_for_static(static) == []
    assert static.gmres_solution_cache is not cache

    static.solver_options.gmres_reuse_solution = False
    assert gmres_solution_cache_for_static(static) is None
    assert static.gmres_solution_cache is None
    assert static.gmres_solution_cache_key is None


def test_gmres_check_schedule_disabled_for_fixed_gmres():
    static = _static_for_gmres_cache(
        SolverOptions(
            neumann_terms=10,
            self_loop_solver="gmres_fixed",
            gmres_reuse_check_schedule=True,
            gmres_reuse_solution=True,
        )
    )

    assert gmres_check_schedule_for_static(static) is None
    assert gmres_solution_cache_for_static(static) is None


def test_gmres_self_loop_matches_dense_solve_cpu():
    dtype = torch.float64
    matrix = torch.tensor(
        [
            [1.40, -0.20, 0.05, 0.00],
            [0.10, 1.15, -0.10, 0.03],
            [0.00, 0.20, 1.30, -0.07],
            [-0.04, 0.00, 0.08, 1.10],
        ],
        dtype=dtype,
    )
    rhs = torch.tensor([[0.7, -1.2], [2.0, 0.3]], dtype=dtype)

    def apply_a(vec):
        return (matrix @ vec.reshape(-1)).reshape_as(vec)

    got = wave_backward._gmres_solve_wave_self_loop(
        apply_a,
        rhs,
        max_iter=4,
        tol=1e-13,
    )
    expected = torch.linalg.solve(matrix, rhs.reshape(-1)).reshape_as(rhs)

    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-11)


def test_gmres_fixed_self_loop_matches_dense_solve_cpu():
    dtype = torch.float64
    matrix = torch.tensor(
        [
            [1.40, -0.20, 0.05, 0.00],
            [0.10, 1.15, -0.10, 0.03],
            [0.00, 0.20, 1.30, -0.07],
            [-0.04, 0.00, 0.08, 1.10],
        ],
        dtype=dtype,
    )
    rhs = torch.tensor([[0.7, -1.2], [2.0, 0.3]], dtype=dtype)

    def apply_a(vec):
        return (matrix @ vec.reshape(-1)).reshape_as(vec)

    got = wave_backward._gmres_solve_wave_self_loop(
        apply_a,
        rhs,
        max_iter=4,
        tol=1e-13,
        fixed_iterations=True,
    )
    expected = torch.linalg.solve(matrix, rhs.reshape(-1)).reshape_as(rhs)

    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-11)


def test_gmres_self_loop_can_write_into_output_alias_cpu():
    dtype = torch.float64
    matrix = torch.tensor(
        [
            [1.40, -0.20, 0.05, 0.00],
            [0.10, 1.15, -0.10, 0.03],
            [0.00, 0.20, 1.30, -0.07],
            [-0.04, 0.00, 0.08, 1.10],
        ],
        dtype=dtype,
    )
    rhs = torch.tensor([[0.7, -1.2], [2.0, 0.3]], dtype=dtype)
    rhs_original = rhs.clone()

    def apply_a(vec):
        return (matrix @ vec.reshape(-1)).reshape_as(vec)

    got = wave_backward._gmres_solve_wave_self_loop(
        apply_a,
        rhs,
        max_iter=4,
        tol=1e-13,
        out=rhs,
    )
    expected = torch.linalg.solve(matrix, rhs_original.reshape(-1)).reshape_as(rhs_original)

    assert got is rhs
    torch.testing.assert_close(rhs, expected, rtol=1e-11, atol=1e-11)


def test_gmres_self_loop_records_stats_and_stops_early_cpu():
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

        got = wave_backward._gmres_solve_wave_self_loop(
            lambda vec: vec,
            rhs,
            max_iter=5,
            tol=1e-12,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, rhs, rtol=1e-12, atol=1e-12)
    assert stats == [
        {
            "iterations": 1,
            "a_applications": 1,
            "rel_res": 0.0,
            "check_count": 1,
            "arnoldi_backend": "torch_cgs2",
            "min_check_iter": 1,
            "preconditioner": "none",
            "warm_start_used": False,
            "warm_start_accepted": False,
            "warm_start_status": "not_attempted",
            "warm_start_rel_res": None,
            "residual_probe_a_applications": 0,
            "correction_iterations": 1,
        }
    ]


def test_gmres_self_loop_handles_zero_rhs_cpu():
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        rhs = torch.zeros((2, 3), dtype=torch.float64)

        got = wave_backward._gmres_solve_wave_self_loop(
            lambda vec: vec,
            rhs,
            max_iter=4,
            tol=1e-12,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, rhs)
    assert stats == [
        {
            "iterations": 0,
            "a_applications": 0,
            "rel_res": 0.0,
            "preconditioner": "none",
            "warm_start_used": False,
            "warm_start_accepted": False,
            "warm_start_status": "not_attempted",
            "residual_probe_a_applications": 0,
            "correction_iterations": 0,
        }
    ]


def test_gmres_self_loop_handles_zero_rhs_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES solve path")

    device = torch.device("cuda")
    rhs = torch.zeros((2, 3), dtype=torch.float64, device=device)

    got = wave_backward._gmres_solve_wave_self_loop(
        lambda vec: vec,
        rhs,
        max_iter=4,
        tol=1e-12,
        check_interval=2,
    )

    torch.testing.assert_close(got, rhs)
    assert bool(torch.isfinite(got).all().detach().cpu())


def test_gmres_self_loop_check_interval_counts_checks_cpu():
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    calls = 0
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

        def apply_a(vec):
            nonlocal calls
            calls += 1
            return matrix @ vec

        got = wave_backward._gmres_solve_wave_self_loop(
            apply_a,
            rhs,
            max_iter=5,
            tol=1e-12,
            check_interval=3,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, torch.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)
    assert calls == 3
    assert stats[0]["iterations"] == 3
    assert stats[0]["check_count"] == 2
    assert stats[0]["rel_res"] < 1e-12


def test_gmres_self_loop_min_check_iter_delays_first_check_cpu():
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    calls = 0
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

        def apply_a(vec):
            nonlocal calls
            calls += 1
            return matrix @ vec

        got = wave_backward._gmres_solve_wave_self_loop(
            apply_a,
            rhs,
            max_iter=5,
            tol=1e-12,
            min_check_iter=3,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, torch.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)
    assert calls == 3
    assert stats[0]["iterations"] == 3
    assert stats[0]["check_count"] == 1
    assert stats[0]["min_check_iter"] == 3


def test_gmres_self_loop_min_check_iter_continues_after_failed_check_cpu():
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    calls = 0
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        rhs = torch.tensor([1.0, -2.0, 0.5, 0.25], dtype=torch.float64)

        def apply_a(vec):
            nonlocal calls
            calls += 1
            return matrix @ vec

        got = wave_backward._gmres_solve_wave_self_loop(
            apply_a,
            rhs,
            max_iter=4,
            tol=1e-14,
            min_check_iter=2,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, torch.linalg.solve(matrix, rhs), rtol=1e-11, atol=1e-11)
    assert calls == 4
    assert stats[0]["iterations"] == 4
    assert stats[0]["check_count"] == 3
    assert stats[0]["min_check_iter"] == 2


def test_gmres_self_loop_right_preconditioner_reduces_cpu_iterations():
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    calls = 0
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        diagonal_inverse = torch.diag(matrix).reciprocal()

        def apply_a(vec):
            nonlocal calls
            calls += 1
            return matrix @ vec

        got = wave_backward._gmres_solve_wave_self_loop(
            apply_a,
            rhs,
            max_iter=3,
            tol=1e-12,
            right_preconditioner=lambda vec: vec * diagonal_inverse,
            preconditioner_name="diagonal",
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, torch.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)
    assert calls == 1
    assert stats[0]["iterations"] == 1
    assert stats[0]["check_count"] == 1
    assert stats[0]["preconditioner"] == "diagonal"


def test_gmres_self_loop_accepts_exact_initial_guess_cpu():
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    calls = 0
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

        def apply_a(vec):
            nonlocal calls
            calls += 1
            return vec

        got = wave_backward._gmres_solve_wave_self_loop(
            apply_a,
            rhs,
            max_iter=5,
            tol=1e-12,
            initial_guess=rhs.clone(),
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, rhs, rtol=1e-12, atol=1e-12)
    assert calls == 1
    assert stats[0]["iterations"] == 0
    assert stats[0]["a_applications"] == 1
    assert stats[0]["check_count"] == 1
    assert stats[0]["warm_start_used"] is True
    assert stats[0]["warm_start_accepted"] is True
    assert stats[0]["rel_res"] == pytest.approx(0.0)


def test_gmres_self_loop_solves_correction_from_initial_guess_cpu():
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    calls = 0
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        matrix = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
        rhs = torch.tensor([3.0, 4.0], dtype=torch.float64)
        initial_guess = torch.tensor([3.0, 0.0], dtype=torch.float64)

        def apply_a(vec):
            nonlocal calls
            calls += 1
            return matrix @ vec

        got = wave_backward._gmres_solve_wave_self_loop(
            apply_a,
            rhs,
            max_iter=2,
            tol=1e-12,
            initial_guess=initial_guess,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, torch.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)
    assert calls == 2
    assert stats[0]["iterations"] == 1
    assert stats[0]["a_applications"] == 2
    assert stats[0]["check_count"] == 2
    assert stats[0]["warm_start_used"] is True
    assert stats[0]["warm_start_accepted"] is False
    assert stats[0]["warm_start_rel_res"] == pytest.approx(4.0 / 5.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gmres_hessenberg_residual_kernel_matches_lstsq_cuda(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES residual kernel")

    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(123)
    max_m = 16
    hessenberg_cpu = torch.randn((max_m + 1, max_m), generator=generator, dtype=torch.float64)
    rows = torch.arange(max_m + 1)[:, None]
    cols = torch.arange(max_m)[None, :]
    hessenberg_cpu = torch.where(rows <= cols + 1, hessenberg_cpu, torch.zeros_like(hessenberg_cpu))
    hessenberg_cpu[5, 4] = 1e-12
    hessenberg = hessenberg_cpu.to(device=device, dtype=dtype)
    beta = torch.tensor(2.5, dtype=dtype, device=device)
    residual_buf = torch.empty((), dtype=dtype, device=device)
    y_buf = torch.empty((max_m,), dtype=dtype, device=device)

    for iters in (1, 4, 10, 16):
        got = wave_backward._gmres_hessenberg_rel_res(
            hessenberg,
            beta,
            iters=iters,
            b_norm=float(beta.detach().cpu()),
            residual_buf=residual_buf,
            y_buf=y_buf,
            store_y=True,
        )
        h_sub = hessenberg[: iters + 1, :iters]
        rhs_sub = torch.zeros((iters + 1,), dtype=dtype, device=device)
        rhs_sub[0] = beta
        y = torch.linalg.lstsq(h_sub, rhs_sub).solution
        expected = float(torch.linalg.vector_norm(h_sub @ y - rhs_sub).detach().cpu()) / float(beta.cpu())

        abs_tol = 2e-5 if dtype == torch.float32 else 1e-7
        rel_tol = 5e-5 if dtype == torch.float32 else 1e-10
        assert got == pytest.approx(expected, rel=rel_tol, abs=abs_tol)
        torch.testing.assert_close(
            y_buf[:iters],
            y,
            rtol=2e-4 if dtype == torch.float32 else 1e-10,
            atol=2e-5 if dtype == torch.float32 else 1e-10,
        )


def test_gmres_fused_norm_check_kernel_matches_residual_helper_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES norm/check kernel")

    device = torch.device("cuda")
    dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(321)
    max_m = 16
    hessenberg_cpu = torch.randn((max_m + 1, max_m), generator=generator, dtype=torch.float32)
    rows = torch.arange(max_m + 1)[:, None]
    cols = torch.arange(max_m)[None, :]
    hessenberg_cpu = torch.where(rows <= cols + 1, hessenberg_cpu, torch.zeros_like(hessenberg_cpu))
    norm_partials_cpu = torch.tensor([0.04, 0.09, 0.16], dtype=torch.float32)
    norm_value = torch.sqrt(norm_partials_cpu.sum())
    beta = torch.tensor(2.5, dtype=dtype, device=device)

    for iters in (1, 4, 10, 16):
        actual_h = hessenberg_cpu.to(device=device, dtype=dtype)
        expected_h = hessenberg_cpu.to(device=device, dtype=dtype)
        expected_h[iters, iters - 1] = norm_value.to(device=device, dtype=dtype)
        norm_partials = norm_partials_cpu.to(device=device, dtype=dtype)
        actual_residual = torch.empty((), dtype=dtype, device=device)
        expected_residual = torch.empty((), dtype=dtype, device=device)
        actual_y = torch.empty((max_m,), dtype=dtype, device=device)
        expected_y = torch.empty((max_m,), dtype=dtype, device=device)

        wave_backward._gmres_arnoldi_reduce_norm_check_kernel[(1,)](
            actual_h,
            norm_partials,
            beta,
            actual_residual,
            actual_y,
            NUM_TILES=int(norm_partials.numel()),
            J=int(iters - 1),
            MAX_ITER=int(max_m),
            BLOCK_TILES=int(wave_backward.triton.next_power_of_2(norm_partials.numel())),
            BLOCK_R=int(wave_backward.triton.next_power_of_2(max_m + 1)),
            BLOCK_C=int(wave_backward.triton.next_power_of_2(max_m)),
            DTYPE=wave_backward._tl_float_dtype(dtype),
            num_warps=1,
        )
        expected = wave_backward._gmres_hessenberg_rel_res(
            expected_h,
            beta,
            iters=iters,
            b_norm=float(beta.detach().cpu()),
            residual_buf=expected_residual,
            y_buf=expected_y,
            store_y=True,
        )

        got = float(actual_residual.detach().cpu())
        assert got == pytest.approx(expected, rel=5e-5, abs=2e-5)
        torch.testing.assert_close(actual_h[iters, iters - 1], expected_h[iters, iters - 1])
        torch.testing.assert_close(actual_y[:iters], expected_y[:iters], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("num_tiles", [513, 729, 1024])
@pytest.mark.parametrize("j", [0, 1, 3])
@pytest.mark.parametrize("add_to_h", [False, True])
def test_gmres_large_direct_sum_matches_staged_reduction_cuda(num_tiles, j, add_to_h):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES direct sum kernel")

    device = torch.device("cuda")
    dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(1100 + num_tiles + 11 * j + int(add_to_h))
    max_iter = 16
    group_tiles = wave_backward._GMRES_TRITON_LARGE_ARNOLDI_GROUP_TILES
    num_groups = wave_backward.triton.cdiv(num_tiles, group_tiles)
    block_k = wave_backward.triton.next_power_of_2(max_iter)
    block_groups = wave_backward.triton.next_power_of_2(num_groups)
    block_tiles = wave_backward.triton.next_power_of_2(num_tiles)

    partials = torch.randn((num_tiles, max_iter), generator=generator, dtype=dtype).to(device)
    group_partials = torch.empty((num_groups, max_iter), dtype=dtype, device=device)
    coeff_staged = torch.empty((max_iter,), dtype=dtype, device=device)
    coeff_direct = torch.empty((max_iter,), dtype=dtype, device=device)
    h0 = torch.randn((max_iter + 1, max_iter), generator=generator, dtype=dtype).to(device)
    h_staged = h0.clone()
    h_direct = h0.clone()

    wave_backward._gmres_arnoldi_reduce_group_partials_kernel[(num_groups,)](
        partials,
        group_partials,
        NUM_TILES=int(num_tiles),
        J=int(j),
        MAX_ITER=int(max_iter),
        GROUP_TILES=int(group_tiles),
        BLOCK_K=int(block_k),
        DTYPE=wave_backward._tl_float_dtype(dtype),
        num_warps=8,
    )
    wave_backward._gmres_arnoldi_sum_group_coeff_kernel[(1,)](
        group_partials,
        coeff_staged,
        h_staged,
        NUM_GROUPS=int(num_groups),
        J=int(j),
        MAX_ITER=int(max_iter),
        BLOCK_GROUPS=int(block_groups),
        BLOCK_K=int(block_k),
        ADD_TO_H=bool(add_to_h),
        DTYPE=wave_backward._tl_float_dtype(dtype),
        num_warps=1,
    )
    wave_backward._gmres_arnoldi_sum_partials_direct_kernel[(1,)](
        partials,
        coeff_direct,
        h_direct,
        NUM_TILES=int(num_tiles),
        J=int(j),
        MAX_ITER=int(max_iter),
        BLOCK_TILES=int(block_tiles),
        BLOCK_K=int(block_k),
        ADD_TO_H=bool(add_to_h),
        DTYPE=wave_backward._tl_float_dtype(dtype),
        num_warps=8,
    )

    torch.testing.assert_close(coeff_direct[: j + 1], coeff_staged[: j + 1], rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(h_direct[: j + 1, j], h_staged[: j + 1, j], rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("num_tiles", [513, 729, 1024])
@pytest.mark.parametrize("iters", [1, 4, 10])
def test_gmres_large_direct_norm_check_matches_staged_reduction_cuda(num_tiles, iters):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES direct norm check")

    device = torch.device("cuda")
    dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(1700 + num_tiles + 13 * iters)
    max_iter = 16
    group_tiles = wave_backward._GMRES_TRITON_LARGE_ARNOLDI_GROUP_TILES
    num_groups = wave_backward.triton.cdiv(num_tiles, group_tiles)
    block_groups = wave_backward.triton.next_power_of_2(num_groups)
    block_tiles = wave_backward.triton.next_power_of_2(num_tiles)
    block_k = wave_backward.triton.next_power_of_2(max_iter)

    hessenberg_cpu = torch.randn((max_iter + 1, max_iter), generator=generator, dtype=dtype)
    rows = torch.arange(max_iter + 1)[:, None]
    cols = torch.arange(max_iter)[None, :]
    hessenberg_cpu = torch.where(rows <= cols + 1, hessenberg_cpu, torch.zeros_like(hessenberg_cpu))
    norm_partials = torch.rand((num_tiles,), generator=generator, dtype=dtype).to(device) * 0.01
    norm_group_partials = torch.empty((num_groups,), dtype=dtype, device=device)
    beta = torch.tensor(3.0, dtype=dtype, device=device)
    h_staged = hessenberg_cpu.to(device)
    h_direct = hessenberg_cpu.to(device)
    residual_staged = torch.empty((), dtype=dtype, device=device)
    residual_direct = torch.empty((), dtype=dtype, device=device)
    y_staged = torch.empty((max_iter,), dtype=dtype, device=device)
    y_direct = torch.empty((max_iter,), dtype=dtype, device=device)
    j = iters - 1

    wave_backward._gmres_arnoldi_reduce_group_norm_partials_kernel[(num_groups,)](
        norm_partials,
        norm_group_partials,
        NUM_TILES=int(num_tiles),
        GROUP_TILES=int(group_tiles),
        DTYPE=wave_backward._tl_float_dtype(dtype),
        num_warps=8,
    )
    wave_backward._gmres_arnoldi_reduce_norm_check_kernel[(1,)](
        h_staged,
        norm_group_partials,
        beta,
        residual_staged,
        y_staged,
        NUM_TILES=int(num_groups),
        J=int(j),
        MAX_ITER=int(max_iter),
        BLOCK_TILES=int(block_groups),
        BLOCK_R=int(wave_backward.triton.next_power_of_2(max_iter + 1)),
        BLOCK_C=int(block_k),
        DTYPE=wave_backward._tl_float_dtype(dtype),
        num_warps=1,
    )
    wave_backward._gmres_arnoldi_reduce_norm_check_kernel[(1,)](
        h_direct,
        norm_partials,
        beta,
        residual_direct,
        y_direct,
        NUM_TILES=int(num_tiles),
        J=int(j),
        MAX_ITER=int(max_iter),
        BLOCK_TILES=int(block_tiles),
        BLOCK_R=int(wave_backward.triton.next_power_of_2(max_iter + 1)),
        BLOCK_C=int(block_k),
        DTYPE=wave_backward._tl_float_dtype(dtype),
        num_warps=1,
    )

    torch.testing.assert_close(residual_direct, residual_staged, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(h_direct[iters, j], h_staged[iters, j], rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(y_direct[:iters], y_staged[:iters], rtol=2e-4, atol=2e-5)


def test_gmres_self_loop_matches_dense_solve_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES solve path")

    device = torch.device("cuda")
    dtype = torch.float64
    matrix = torch.tensor(
        [
            [1.40, -0.20, 0.05, 0.00],
            [0.10, 1.15, -0.10, 0.03],
            [0.00, 0.20, 1.30, -0.07],
            [-0.04, 0.00, 0.08, 1.10],
        ],
        dtype=dtype,
        device=device,
    )
    rhs = torch.tensor([[0.7, -1.2], [2.0, 0.3]], dtype=dtype, device=device)

    def apply_a(vec):
        return (matrix @ vec.reshape(-1)).reshape_as(vec)

    got = wave_backward._gmres_solve_wave_self_loop(
        apply_a,
        rhs,
        max_iter=4,
        tol=1e-13,
        check_interval=2,
    )
    expected = torch.linalg.solve(matrix, rhs.reshape(-1)).reshape_as(rhs)

    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-11)


def test_gmres_self_loop_triton_split_matches_dense_solve_cuda_float32(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES solve path")

    monkeypatch.setenv("GPUREC_GMRES_TRITON_ARNOLDI", "1")
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        device = torch.device("cuda")
        dtype = torch.float32
        matrix = torch.tensor(
            [
                [1.40, -0.20, 0.05, 0.00],
                [0.10, 1.15, -0.10, 0.03],
                [0.00, 0.20, 1.30, -0.07],
                [-0.04, 0.00, 0.08, 1.10],
            ],
            dtype=dtype,
            device=device,
        )
        rhs = torch.tensor([[0.7, -1.2], [2.0, 0.3]], dtype=dtype, device=device)

        def apply_a(vec):
            return (matrix @ vec.reshape(-1)).reshape_as(vec)

        got = wave_backward._gmres_solve_wave_self_loop(
            apply_a,
            rhs,
            max_iter=4,
            tol=1e-6,
            check_interval=2,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    expected = torch.linalg.solve(matrix, rhs.reshape(-1)).reshape_as(rhs)
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)
    assert stats and stats[0]["arnoldi_backend"] == "triton_split"


def test_gmres_self_loop_triton_split_handles_multi_tile_cuda_float32(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES solve path")

    monkeypatch.setenv("GPUREC_GMRES_TRITON_ARNOLDI", "1")
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        device = torch.device("cuda")
        rhs = torch.linspace(-1.0, 1.0, 513, dtype=torch.float32, device=device)

        got = wave_backward._gmres_solve_wave_self_loop(
            lambda vec: vec,
            rhs,
            max_iter=4,
            tol=1e-6,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, rhs, rtol=2e-5, atol=2e-5)
    assert stats and stats[0]["arnoldi_backend"] == "triton_split"
    assert stats[0]["iterations"] == 1


def test_gmres_self_loop_triton_split_can_trust_min_check_iter_cuda_float32(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES solve path")

    monkeypatch.setenv("GPUREC_GMRES_TRITON_ARNOLDI", "1")
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        device = torch.device("cuda")
        idx = torch.arange(513, device=device)
        diagonal = torch.where(
            idx % 2 == 0,
            torch.tensor(1.25, dtype=torch.float32, device=device),
            torch.tensor(0.85, dtype=torch.float32, device=device),
        )
        rhs = torch.linspace(-1.0, 1.0, 513, dtype=torch.float32, device=device)

        def apply_a(vec):
            return (diagonal * vec.reshape(-1)).reshape_as(vec)

        got = wave_backward._gmres_solve_wave_self_loop(
            apply_a,
            rhs,
            max_iter=4,
            tol=1e-12,
            min_check_iter=2,
            trust_min_check_iter=True,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, rhs / diagonal, rtol=3e-5, atol=3e-5)
    assert stats and stats[0]["arnoldi_backend"] == "triton_split"
    assert stats[0]["iterations"] == 2
    assert stats[0]["trusted_check_schedule"] is True
    assert stats[0]["trusted_check_used"] is True


def test_gmres_self_loop_triton_large_handles_above_cap_cuda_float32(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES solve path")

    monkeypatch.setenv("GPUREC_GMRES_TRITON_ARNOLDI", "1")
    monkeypatch.setenv("GPUREC_GMRES_TRITON_LARGE_ARNOLDI", "1")
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        device = torch.device("cuda")
        block_n = wave_backward._GMRES_TRITON_ARNOLDI_BLOCK_N
        max_tiles = wave_backward._GMRES_TRITON_ARNOLDI_MAX_BLOCK_TILES
        n = block_n * max_tiles + 17
        idx = torch.arange(n, device=device)
        diagonal = torch.where(
            idx % 2 == 0,
            torch.tensor(1.25, dtype=torch.float32, device=device),
            torch.tensor(0.85, dtype=torch.float32, device=device),
        )
        rhs = torch.linspace(-1.0, 1.0, n, dtype=torch.float32, device=device)

        def apply_a(vec):
            return (diagonal * vec.reshape(-1)).reshape_as(vec)

        got = wave_backward._gmres_solve_wave_self_loop(
            apply_a,
            rhs,
            max_iter=4,
            tol=1e-5,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    expected = rhs / diagonal
    torch.testing.assert_close(got, expected, rtol=3e-5, atol=3e-5)
    assert stats and stats[0]["arnoldi_backend"] == "triton_large"
    assert stats[0]["large_direct_sum"] is True
    assert stats[0]["large_direct_norm_checks"] == 2
    assert stats[0]["large_group_norm_reductions"] == 1
    assert stats[0]["iterations"] == 2


def test_gmres_self_loop_triton_large_direct_sum_can_be_disabled_cuda_float32(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES solve path")

    monkeypatch.setenv("GPUREC_GMRES_TRITON_ARNOLDI", "1")
    monkeypatch.setenv("GPUREC_GMRES_TRITON_LARGE_ARNOLDI", "1")
    monkeypatch.setenv("GPUREC_GMRES_TRITON_LARGE_DIRECT_SUM", "0")
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        device = torch.device("cuda")
        block_n = wave_backward._GMRES_TRITON_ARNOLDI_BLOCK_N
        max_tiles = wave_backward._GMRES_TRITON_ARNOLDI_MAX_BLOCK_TILES
        rhs = torch.linspace(-1.0, 1.0, block_n * max_tiles + 17, dtype=torch.float32, device=device)
        got = wave_backward._gmres_solve_wave_self_loop(
            lambda vec: vec,
            rhs,
            max_iter=2,
            tol=1e-5,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, rhs, rtol=3e-5, atol=3e-5)
    assert stats and stats[0]["arnoldi_backend"] == "triton_large"
    assert stats[0]["large_direct_sum"] is False
    assert stats[0]["large_direct_norm_checks"] == 0
    assert stats[0]["large_group_norm_reductions"] == 1


def test_gmres_self_loop_triton_large_can_trust_min_check_iter_cuda_float32(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES solve path")

    monkeypatch.setenv("GPUREC_GMRES_TRITON_ARNOLDI", "1")
    monkeypatch.setenv("GPUREC_GMRES_TRITON_LARGE_ARNOLDI", "1")
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        device = torch.device("cuda")
        block_n = wave_backward._GMRES_TRITON_ARNOLDI_BLOCK_N
        max_tiles = wave_backward._GMRES_TRITON_ARNOLDI_MAX_BLOCK_TILES
        n = block_n * max_tiles + 17
        idx = torch.arange(n, device=device)
        diagonal = torch.where(
            idx % 2 == 0,
            torch.tensor(1.25, dtype=torch.float32, device=device),
            torch.tensor(0.85, dtype=torch.float32, device=device),
        )
        rhs = torch.linspace(-1.0, 1.0, n, dtype=torch.float32, device=device)

        def apply_a(vec):
            return (diagonal * vec.reshape(-1)).reshape_as(vec)

        got = wave_backward._gmres_solve_wave_self_loop(
            apply_a,
            rhs,
            max_iter=4,
            tol=1e-12,
            min_check_iter=2,
            trust_min_check_iter=True,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, rhs / diagonal, rtol=3e-5, atol=3e-5)
    assert stats and stats[0]["arnoldi_backend"] == "triton_large"
    assert stats[0]["large_direct_sum"] is True
    assert stats[0]["large_direct_norm_checks"] == 1
    assert stats[0]["large_group_norm_reductions"] == 1
    assert stats[0]["iterations"] == 2
    assert stats[0]["trusted_check_schedule"] is True
    assert stats[0]["trusted_check_used"] is True


def test_gmres_self_loop_triton_large_dispatch_boundary_cuda_float32(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES solve path")

    monkeypatch.setenv("GPUREC_GMRES_TRITON_ARNOLDI", "1")
    monkeypatch.setenv("GPUREC_GMRES_TRITON_LARGE_ARNOLDI", "1")
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        device = torch.device("cuda")
        block_n = wave_backward._GMRES_TRITON_ARNOLDI_BLOCK_N
        max_tiles = wave_backward._GMRES_TRITON_ARNOLDI_MAX_BLOCK_TILES

        small_rhs = torch.linspace(-1.0, 1.0, block_n * max_tiles, dtype=torch.float32, device=device)
        got_small = wave_backward._gmres_solve_wave_self_loop(
            lambda vec: vec,
            small_rhs,
            max_iter=2,
            tol=1e-5,
        )

        large_rhs = torch.linspace(
            -1.0,
            1.0,
            block_n * max_tiles + 1,
            dtype=torch.float32,
            device=device,
        )
        got_large = wave_backward._gmres_solve_wave_self_loop(
            lambda vec: vec,
            large_rhs,
            max_iter=2,
            tol=1e-5,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got_small, small_rhs, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(got_large, large_rhs, rtol=3e-5, atol=3e-5)
    assert [row["arnoldi_backend"] for row in stats[-2:]] == ["triton_split", "triton_large"]


def test_gmres_self_loop_triton_large_can_be_disabled_cuda_float32(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton GMRES solve path")

    monkeypatch.setenv("GPUREC_GMRES_TRITON_ARNOLDI", "1")
    monkeypatch.setenv("GPUREC_GMRES_TRITON_LARGE_ARNOLDI", "0")
    old_stats = wave_backward._GMRES_SELF_LOOP_STATS
    stats = []
    wave_backward._GMRES_SELF_LOOP_STATS = stats
    try:
        device = torch.device("cuda")
        block_n = wave_backward._GMRES_TRITON_ARNOLDI_BLOCK_N
        max_tiles = wave_backward._GMRES_TRITON_ARNOLDI_MAX_BLOCK_TILES
        rhs = torch.linspace(-1.0, 1.0, block_n * max_tiles + 1, dtype=torch.float32, device=device)
        got = wave_backward._gmres_solve_wave_self_loop(
            lambda vec: vec,
            rhs,
            max_iter=2,
            tol=1e-5,
        )
    finally:
        wave_backward._GMRES_SELF_LOOP_STATS = old_stats

    torch.testing.assert_close(got, rhs, rtol=3e-5, atol=3e-5)
    assert stats and stats[0]["arnoldi_backend"] == "torch_cgs2"


def test_triton_apply_a_zeroes_inactive_rows_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton self-loop kernel")

    device = torch.device("cuda")
    dtype = torch.float64
    W = 2
    S = 2
    term_in = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
    term_out = torch.full_like(term_in, float("nan"))
    rhs = torch.zeros_like(term_in)
    active_mask = torch.tensor([True, False], device=device)
    diag = torch.full_like(term_in, 0.5)
    zeros = torch.zeros_like(term_in)
    pibar_corr = torch.empty_like(term_in)
    v_k = torch.empty_like(term_in)
    sp_child = torch.zeros((S,), device=device, dtype=torch.int32)
    sp_parent = torch.full((S,), -1, device=device, dtype=torch.int32)
    compact_level_ptr = torch.tensor([0], device=device, dtype=torch.long)
    compact_empty = torch.empty((0,), device=device, dtype=torch.int32)

    wave_backward._wave_backward_uniform_2d_jt_kernel[(W,)](
        term_in,
        term_out,
        rhs,
        active_mask,
        diag,
        zeros,
        zeros,
        zeros,
        zeros,
        sp_child,
        sp_child,
        sp_parent,
        compact_level_ptr,
        compact_empty,
        compact_empty,
        compact_empty,
        pibar_corr,
        v_k,
        W,
        S,
        1,
        2,
        1,
        0,
        USE_ACTIVE_MASK=True,
        SKIP_INACTIVE_SCRATCH_ZERO=False,
        FIXED_POINT_UPDATE=False,
        DTYPE=wave_backward._tl_float_dtype(dtype),
        USE_CHILD_EDGE_SELF_LOOP=True,
        OUTPUT_A=True,
        ACCUMULATE_V=False,
        num_warps=1,
    )
    torch.cuda.synchronize()

    expected = torch.tensor([[0.5, 1.0], [0.0, 0.0]], device=device, dtype=dtype)
    torch.testing.assert_close(term_out, expected)

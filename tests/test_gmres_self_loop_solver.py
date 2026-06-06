import torch
import pytest

from gpurec import SolverOptions
from gpurec.core.kernels import wave_backward


def test_solver_options_accept_gmres_fixed():
    options = SolverOptions(
        neumann_terms=10,
        self_loop_solver=" GMRES_FIXED ",
        gmres_tol=1e-8,
        gmres_check_interval=2,
    )

    options.validate()

    assert options.self_loop_solver == "gmres_fixed"
    assert options.gmres_tol == 1e-8
    assert options.gmres_check_interval == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"self_loop_solver": "bad"},
        {"gmres_tol": 0.0},
        {"gmres_check_interval": 0},
    ],
)
def test_solver_options_reject_invalid_gmres_options(kwargs):
    options = SolverOptions(**kwargs)

    with pytest.raises(ValueError):
        options.validate()


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
        {"iterations": 1, "rel_res": 0.0, "check_count": 1, "arnoldi_backend": "torch_cgs2"}
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
    assert stats == [{"iterations": 0, "rel_res": 0.0}]


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

    for iters in (1, 4, 10, 16):
        got = wave_backward._gmres_hessenberg_rel_res(
            hessenberg,
            beta,
            iters=iters,
            b_norm=float(beta.detach().cpu()),
            residual_buf=residual_buf,
        )
        h_sub = hessenberg[: iters + 1, :iters]
        rhs_sub = torch.zeros((iters + 1,), dtype=dtype, device=device)
        rhs_sub[0] = beta
        y = torch.linalg.lstsq(h_sub, rhs_sub).solution
        expected = float(torch.linalg.vector_norm(h_sub @ y - rhs_sub).detach().cpu()) / float(beta.cpu())

        abs_tol = 2e-5 if dtype == torch.float32 else 1e-7
        rel_tol = 5e-5 if dtype == torch.float32 else 1e-10
        assert got == pytest.approx(expected, rel=rel_tol, abs=abs_tol)


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

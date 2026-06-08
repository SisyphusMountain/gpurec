import torch
import pytest

from gpurec import SolverOptions
from gpurec.core.kernels import wave_backward


def test_solver_options_accept_gmres_fixed():
    options = SolverOptions(
        neumann_terms=10,
        self_loop_solver=" GMRES_FIXED ",
    )

    options.validate()

    assert options.self_loop_solver == "gmres_fixed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"self_loop_solver": "bad"},
        {"self_loop_solver": "gmres"},
    ],
)
def test_solver_options_reject_invalid_gmres_options(kwargs):
    options = SolverOptions(**kwargs)

    with pytest.raises(ValueError):
        options.validate()


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
    )
    expected = torch.linalg.solve(matrix, rhs.reshape(-1)).reshape_as(rhs)

    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-11)


def test_gmres_self_loop_handles_zero_rhs_cpu():
    rhs = torch.zeros((2, 3), dtype=torch.float64)

    got = wave_backward._gmres_solve_wave_self_loop(
        lambda vec: vec,
        rhs,
        max_iter=4,
    )
    torch.testing.assert_close(got, rhs)


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

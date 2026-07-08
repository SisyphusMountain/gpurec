import torch

from gpurec.config import dtype_rel_tol_default, dtype_rel_tol_floor
from gpurec.api.solver_options import SolverOptions


def test_dtype_tol_single_source():
    assert dtype_rel_tol_default(torch.float32) == 1e-6
    assert dtype_rel_tol_default(torch.float64) == 1e-12
    assert dtype_rel_tol_floor(torch.float32) == 4.0 * torch.finfo(torch.float32).eps


def test_forward_tangent_uses_shared_helper():
    from gpurec.solver.forward_tangent import _default_tol
    assert _default_tol(torch.float64) == dtype_rel_tol_default(torch.float64)


def test_solver_options_has_tangent_tol():
    assert SolverOptions().e_tangent_tol == 1e-9

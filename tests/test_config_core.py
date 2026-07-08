import inspect

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


def test_no_divergent_signature_defaults():
    """Previously-divergent signature defaults (neumann_terms=3, bicgstab max_iter=500)
    must now agree with ``SolverOptions()`` -- either literally (a None sentinel resolved
    at call time) or, if already-numeric, by matching value."""
    from gpurec.api import _implicit_grad as ig

    so = SolverOptions()
    sig = inspect.signature(ig.implicit_grad_loglik_vjp_wave).parameters
    # None-sentinel -> resolved to SolverOptions default at call time
    assert sig["neumann_terms"].default in (None, so.neumann_terms)
    assert sig["bicgstab_max_iter"].default in (None, so.bicgstab_max_iter)
    assert sig["adjoint_pruning_threshold"].default in (None, so.adjoint_pruning_threshold)
    assert sig["pibar_side_threshold"].default in (None, so.pibar_side_threshold)
    assert inspect.signature(ig._bicgstab).parameters["max_iter"].default in (None, so.bicgstab_max_iter)


def test_e_step_fallbacks_agree_with_solver_options():
    """``e_step.py``/``e_step_tangent.py`` fixed-point max_iter/tol fallbacks must resolve
    to the same SolverOptions fields used by the production solver (``e_max_iter``,
    ``e_tol`` for the forward fixed point, ``e_tangent_tol`` for the tangent fixed point)."""
    from gpurec.core.kernels.e_step import e_fixed_point_triton
    from gpurec.core.kernels.e_step_tangent import e_tangent_fixed_point

    so = SolverOptions()
    fwd_params = inspect.signature(e_fixed_point_triton).parameters
    assert fwd_params["max_iter"].default in (None, so.e_max_iter)
    assert fwd_params["tol"].default in (None, so.e_tol)

    tan_params = inspect.signature(e_tangent_fixed_point).parameters
    assert tan_params["max_iter"].default in (None, so.e_max_iter)
    assert tan_params["tol"].default in (None, so.e_tangent_tol)

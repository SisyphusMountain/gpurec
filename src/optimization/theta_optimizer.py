"""Backward-compatible facade — real implementations live in submodules."""
from .types import FixedPointInfo, LinearSolveStats, StepRecord
from .linear_solvers import _cg, _gmres
from .implicit_grad import (
    implicit_grad_loglik_vjp_wave,
    implicit_grad_loglik_vjp_wave_genewise,
    _e_adjoint_and_theta_vjp,
)
from .wave_optimizer import optimize_theta_wave
from .genewise_optimizer import optimize_theta_genewise, _lbfgs_two_loop

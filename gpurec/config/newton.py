"""Newton / Levenberg-Marquardt / Lanczos / CG controls for the curvature solvers.

These fields were, before this module existed, copy-pasted as identical signature defaults across
``gpurec/solver/curvature.py`` (the shared ``newton_min``/``certify_min`` core) and its three
consumers ``genewise_curvature.py``, ``origination_curvature.py``, ``receiver_curvature.py``.
``NewtonOptions`` is the single source of truth for those values; the four files now read from it
(with the old individual kwargs kept as ``None``-sentinel overrides for backward compatibility).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NewtonOptions:
    """Newton / Levenberg-Marquardt / Lanczos / CG controls for the curvature solvers."""
    sigma: float = 0.01            # initial LM damping fraction (lam0 = sigma * lam_max)
    sigma_floor: float = 1e-4      # min damping fraction
    lanczos_m: int = 10            # Lanczos steps for lam_max estimate
    nu: float = 1.5                # neg-curvature damping-bump factor
    decrease: float = 1.5          # accepted-step damping-decrease factor
    max_bumps: int = 3             # max neg-curv re-solves per Newton step
    max_cg: int = 40               # CG iters per Newton system
    c1: float = 1e-4               # Armijo sufficient-decrease constant
    ls_max: int = 25               # max line-search backtracks
    gtol: float = 1e-2             # projected-gradient-norm stop
    max_newton: int = 40           # max Newton iterations
    ftol: float = 1e-9             # relative-improvement stall floor
    seed: int = 0                  # Lanczos start-vector RNG seed
    fd_eps_blockwise: float = 1e-2 # FD step, genewise 3x3 Hessian
    fd_eps_hvp: float = 1e-5       # FD step, full-HVP solvers
    lam_ceil_factor: float = 10.0  # damping ceiling = factor * lam_max
    forcing_eta: float = 0.1       # inexact-Newton forcing-term cap
    certify_m: int = 200           # Lanczos steps for PD certificate
    cg_tol: float = 1e-7           # Fisher-info CG residual tol
    cg_max: int = 400              # Fisher-info CG iteration cap

    def validate(self) -> None:
        for name in ("lanczos_m","max_cg","ls_max","max_newton","max_bumps","certify_m","cg_max"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("sigma","sigma_floor","c1","gtol","ftol","fd_eps_blockwise","fd_eps_hvp","cg_tol"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")
